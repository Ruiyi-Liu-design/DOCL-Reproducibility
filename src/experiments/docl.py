"""DOCL as an auxiliary loss on the unchanged AnoDDAE denoiser.

Only the final linear layer is treated as the decoder.  This module adds no
trainable layers and never uses the global RNG for auxiliary view sampling.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from vendor.AnoDDAE.src.model import DiffusionDenoiser, DiffusionScheduler


class DOCLDenoiser(DiffusionDenoiser):
    """Expose the original last-hidden representation without changing forward."""

    def encode(self, x: Tensor, t: Tensor) -> Tensor:
        # Keep this identical to the input preparation in the upstream forward.
        if self.time_emb_type == "learnable":
            t = t.unsqueeze(1)
        t_embedded = self.timestep_embedding(t)
        if self.time_emb_dim > 0:
            x = torch.cat([x, t_embedded], dim=1)
        return self.network[:-1](x)

    def decode(self, z: Tensor) -> Tensor:
        return self.network[-1](z)


def sample_ordered_timesteps(
    batch_size: int,
    num_timesteps: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Uniformly sample an unordered distinct pair from indices 1..T-1.

    Sampling the second index from the first index's complement avoids rejection
    sampling.  Sorting preserves a uniform distribution over all valid pairs.
    Sampling takes place on the generator's device, then transfers if required.
    """
    if generator is None:
        raise ValueError("DOCL requires a dedicated auxiliary torch.Generator.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    valid_count = num_timesteps - 1
    if valid_count < 2:
        raise ValueError("DOCL needs T >= 3 for two distinct indices in 1..T-1.")

    first = torch.randint(
        valid_count, (batch_size,), generator=generator, device=generator.device
    )
    second = torch.randint(
        valid_count - 1,
        (batch_size,),
        generator=generator,
        device=generator.device,
    )
    second = second + (second >= first).long()
    weak = (torch.minimum(first, second) + 1).to(device)
    strong = (torch.maximum(first, second) + 1).to(device)
    return weak, strong


def cosine_order_loss(
    z_anchor: Tensor,
    z_weak: Tensor,
    z_strong: Tensor,
    alpha_bar_weak: Tensor,
    alpha_bar_strong: Tensor,
    margin_scale: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Mean relu(d_weak - d_strong + scaled alpha-bar gap).

    Distances are cosine distances to the same normalized clean anchor.  The
    margin uses cumulative signal strength, rather than the timestep index.
    This pure helper has no random sampling and can be checked numerically.
    The caller supplies weak/strong signal strengths in nonincreasing order.
    Statistics are detached scalar tensors, avoiding GPU-to-CPU synchronization.
    """
    if margin_scale < 0:
        raise ValueError("margin_scale must be nonnegative.")
    if z_anchor.ndim != 2 or z_anchor.shape != z_weak.shape or z_anchor.shape != z_strong.shape:
        raise ValueError("All three embedding tensors must have matching [batch, latent] shapes.")
    if z_anchor.shape[0] == 0:
        raise ValueError("The embedding batch cannot be empty.")
    if alpha_bar_weak.shape != (z_anchor.shape[0],) or alpha_bar_strong.shape != alpha_bar_weak.shape:
        raise ValueError("alpha_bar tensors must have shape [batch].")
    anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-8)
    weak = F.normalize(z_weak, p=2, dim=1, eps=1e-8)
    strong = F.normalize(z_strong, p=2, dim=1, eps=1e-8)
    d_weak = 1.0 - (anchor * weak).sum(dim=1)
    d_strong = 1.0 - (anchor * strong).sum(dim=1)
    margin = margin_scale * (alpha_bar_weak - alpha_bar_strong)
    hinge = F.relu(d_weak - d_strong + margin)
    loss = hinge.mean()
    stats = {
        "violation_rate": (d_weak > d_strong).float().mean().detach(),
        "active_rate": (hinge > 0).float().mean().detach(),
        "dweak": d_weak.mean().detach(),
        "dstrong": d_strong.mean().detach(),
        "marginmean": margin.mean().detach(),
    }
    return loss, stats


def ordered_contrastive_loss(
    model: DOCLDenoiser,
    x0: Tensor,
    scheduler: DiffusionScheduler,
    generator: torch.Generator,
    margin_scale: float = 0.5,
    condition: str = "actual",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build shared-noise views and compute the DOCL auxiliary objective.

    actual: clean anchor receives t=0; each noisy view receives its own t.
    fixed: every view receives t=1; corruption levels still differ.
    Reconstruction is intentionally left to the caller's unchanged DDAE loop.
    """
    if x0.ndim != 2 or not x0.is_floating_point():
        raise ValueError("x0 must be a floating-point [batch, feature] tensor.")
    if condition not in ("actual", "fixed"):
        raise ValueError("condition must be 'actual' or 'fixed'.")
    t_weak, t_strong = sample_ordered_timesteps(
        x0.shape[0], scheduler.num_timesteps, generator, x0.device
    )
    # q_sample uses the global RNG; construct the equivalent closed form here
    # so only DOCL's dedicated generator advances. Both views share this epsilon.
    noise = torch.randn(
        x0.shape, dtype=x0.dtype, device=generator.device, generator=generator
    ).to(x0.device)
    alpha_bar = scheduler.alpha_bar.to(device=x0.device, dtype=x0.dtype)
    alpha_weak = alpha_bar[t_weak]
    alpha_strong = alpha_bar[t_strong]
    x_weak = alpha_weak.sqrt().unsqueeze(1) * x0 + (1 - alpha_weak).sqrt().unsqueeze(1) * noise
    x_strong = alpha_strong.sqrt().unsqueeze(1) * x0 + (1 - alpha_strong).sqrt().unsqueeze(1) * noise

    if condition == "actual":
        anchor_condition = torch.zeros_like(t_weak)
        weak_condition, strong_condition = t_weak, t_strong
    else:
        anchor_condition = torch.ones_like(t_weak)
        weak_condition = strong_condition = anchor_condition

    z_anchor = model.encode(x0, anchor_condition.float())
    z_weak = model.encode(x_weak, weak_condition.float())
    z_strong = model.encode(x_strong, strong_condition.float())
    loss, stats = cosine_order_loss(
        z_anchor, z_weak, z_strong, alpha_weak, alpha_strong, margin_scale
    )
    stats["time_gap_min"] = (t_strong - t_weak).min().detach()
    stats["tweak_mean"] = t_weak.float().mean().detach()
    stats["tstrong_mean"] = t_strong.float().mean().detach()
    return loss, stats
