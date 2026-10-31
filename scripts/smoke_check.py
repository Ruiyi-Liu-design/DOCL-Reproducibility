"""Synthetic CPU checks for the portable wrapper; no optimizer update or scoring."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import numpy as np
from torch.nn import functional as F
from experiments.docl import DOCLDenoiser, cosine_order_loss, sample_ordered_timesteps
from vendor.AnoDDAE.src.model import DiffusionDenoiser, DiffusionScheduler
import ablation_adapter
from run_experiment import split_ids


def check():
    torch.set_num_threads(2)
    torch.manual_seed(8871)
    plain = DiffusionDenoiser(5, [9, 7], 4)
    model = DOCLDenoiser(5, [9, 7], 4)
    model.load_state_dict(plain.state_dict())
    x = torch.randn(17, 5)
    t = torch.arange(17, dtype=torch.float32)
    assert torch.equal(model(x, t), plain(x, t))
    assert torch.equal(model(x, t), model.decode(model.encode(x, t)))
    generator = torch.Generator().manual_seed(991)
    global_before = torch.get_rng_state().clone()
    w, s = sample_ordered_timesteps(len(x), 50, generator, 'cpu')
    assert torch.equal(global_before, torch.get_rng_state()) and bool(((1 <= w) & (w < s) & (s <= 49)).all())
    sch = DiffusionScheduler(50, 'cpu')
    a = sch.alpha_bar
    noise = torch.randn(x.shape, generator=generator)
    xw = a[w].sqrt()[:, None] * x + (1-a[w]).sqrt()[:, None] * noise
    xs = a[s].sqrt()[:, None] * x + (1-a[s]).sqrt()[:, None] * noise
    z = [model.encode(x, torch.zeros_like(w).float()), model.encode(xw, w.float()), model.encode(xs, s.float())]
    loss, _ = cosine_order_loss(*z, a[w], a[s], 2.0)
    q = [F.normalize(v, dim=1, eps=1e-8) for v in z]
    dw = 1-(q[0]*q[1]).sum(1)
    ds = 1-(q[0]*q[2]).sum(1)
    expected = torch.maximum(dw-ds+2*(a[w]-a[s]), torch.zeros_like(dw)).mean()
    assert torch.equal(loss, expected)
    for name in ['reverse', 'zero_margin', 'fixed_margin', 'independent_noise']:
        actual, _ = ablation_adapter.controlled_loss(*z, a[w], a[s], 2., name)
        margin = torch.zeros_like(dw) if name == 'zero_margin' else torch.full_like(dw, ablation_adapter.fixed_margin(2.)) if name == 'fixed_margin' else 2*(a[w]-a[s])
        spec = torch.maximum((ds-dw if name == 'reverse' else dw-ds)+margin, torch.zeros_like(dw)).mean()
        assert torch.equal(actual, spec)
    decoder = tuple(model.network[-1].parameters())
    auxiliary_decoder_grads = torch.autograd.grad(loss, decoder, retain_graph=True, allow_unused=True)
    assert all(v is None for v in auxiliary_decoder_grads)
    reconstruction = F.mse_loss(model(xw, w.float()), x)
    grads = torch.autograd.grad(reconstruction + 1.2*loss, tuple(model.parameters()))
    assert all(torch.isfinite(v).all() for v in grads)
    assert all(torch.count_nonzero(v) > 0 for v in grads[-2:])
    labels = np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int64)
    tr, te = split_ids(labels, 42)
    normals = np.flatnonzero(labels == 0)
    rng = np.random.RandomState(42); rng.shuffle(normals)
    assert np.array_equal(tr, normals[:2]) and np.array_equal(te, np.r_[normals[2:], [1,4]])
    from baselines import native7
    baseline = native7.make('DDAE-C', 5, dict(lr=0.001, alpha=0.5, batch_size=8), 'cpu')
    assert baseline.net.time_emb_dim == 4 and baseline.scheduler.num_timesteps == 50
    assert baseline.net(x, t).shape == x.shape and not baseline.opt.state
    return dict(status='PASS', device='cpu', synthetic_only=True, optimizer_updates=0,
                real_data_forward_calls=0, checks=['backbone equivalence', 'encode/decode equivalence',
                'private pair RNG', 'forward/control loss formulas', 'decoder gradient routing', 'legacy split',
                'portable local DDAE-C source loading'])


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
