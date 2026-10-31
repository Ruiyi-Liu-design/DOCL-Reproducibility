"""Portable entry point around the frozen scientific implementations.

The default is a plan. Training starts only with the explicit --run flag.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_data import records, load_canonical, array_sha256


def split_ids(y, seed):
    import numpy as np
    normal = np.flatnonzero(y == 0)
    np.random.RandomState(seed).shuffle(normal)
    half = len(normal) // 2
    return normal[:half], np.r_[normal[half:], np.flatnonzero(y == 1)]


def prepare_arrays(record, data_dir, seed, setting):
    import numpy as np
    from vendor.AnoDDAE.src.utils import normalize_data
    x, y = load_canonical(record, data_dir)
    # normalize_data must see raw dtype and the COMPLETE original X.
    normalized = normalize_data(x)
    if setting == 'normal-only':
        train_ids, test_ids = split_ids(y, seed)
    else:
        train_ids = test_ids = np.arange(len(y), dtype=np.int64)
    return (np.array(normalized[train_ids], dtype=np.float32),
            np.array(normalized[test_ids], dtype=np.float32), y[test_ids], train_ids, test_ids)


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='Exact ID from prepare_data.py --list')
    p.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    p.add_argument('--setting', choices=['normal-only', 'mixed'], default='normal-only')
    methods = ['DDAE', 'DOCL', 'DDAE-C', 'fixed10_35', 'mainlinked12', 'gapcurriculum',
               'reverse', 'zero_margin', 'fixed_margin', 'independent_noise']
    p.add_argument('--method', choices=methods, default='DOCL')
    p.add_argument('--lambda-value', type=float, default=1.2)
    p.add_argument('--c', type=float, default=2.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    p.add_argument('--output', type=Path, help='New directory for this one model and its two scoring banks')
    p.add_argument('--run', action='store_true', help='Actually train one 100-epoch model and score A/B')
    args = p.parse_args()
    selected = [r for r in records() if r['dataset'] == args.dataset]
    if len(selected) != 1:
        p.error('Unknown dataset ID')
    if args.lambda_value <= 0 or args.c < 0:
        p.error('DOCL lambda must be positive and c nonnegative')
    if args.method in ['reverse', 'zero_margin', 'fixed_margin', 'independent_noise'] and not 1 <= args.c <= 4:
        p.error('The frozen component adapter supports 1 <= c <= 4')
    record = selected[0]
    plan = dict(dataset=args.dataset, seed=args.seed, setting=args.setting, method=args.method,
                total_epochs=100, docl_warmup_included=20 if args.method not in ['DDAE', 'DDAE-C'] else 0,
                lambda_value=args.lambda_value if args.method not in ['DDAE', 'DDAE-C'] else None,
                c=args.c if args.method not in ['DDAE', 'DDAE-C'] else None,
                scoring_seed_A=args.seed + 2000000, scoring_seed_B=args.seed + 12000000,
                device=args.device, mode='EXECUTE' if args.run else 'PLAN_ONLY')
    print(json.dumps(plan, indent=2))
    if not args.run:
        return
    if args.output is None:
        p.error('--run requires a new --output directory')
    if args.output.exists():
        p.error('Output already exists; reuse is never silently overwritten')
    import torch
    import core
    import sampling3
    import ablation_adapter
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
    from vendor.AnoDDAE.src.utils import get_batch_size
    core.configure()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu explicitly')
    train, test, y, train_ids, test_ids = prepare_arrays(record, args.data_dir, args.seed, args.setting)
    bs = get_batch_size(record['n'])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.method == 'DDAE-C':
        from baselines import native7
        # This is the documented local implementation, not an author-confirmed configuration.
        native7.configure()
        state = native7.train('DDAE-C', train, test, dict(lr=0.001, alpha=0.5, batch_size=bs), args.seed, device)
        model, info = state.net, state.info
        scoring = lambda noise_seed: native7.score(state, test, noise_seed)[0]
    else:
        if args.method == 'DDAE':
            model, sch, opt, info = core.train100(train, test, args.seed, bs, device, 0.0, 0.5)
        elif args.method == 'DOCL':
            model, sch, opt, info = core.train100(train, test, args.seed, bs, device, args.lambda_value, args.c)
        else:
            module = sampling3 if args.method in sampling3.ARMS else ablation_adapter
            model, sch, opt, info = module.train100(train, test, args.seed, bs, device,
                                                   args.lambda_value, args.c, method=args.method)
        model.eval()
        scoring = lambda noise_seed: core.reconstruct(model, sch, test, noise_seed, device)[0]
    torch.save(model.state_dict(), args.output / 'model.pt')
    arrays = dict(labels=y, train_indices=train_ids, test_indices=test_ids)
    metrics = {}
    for bank, offset in [('A', 2000000), ('B', 12000000)]:
        scores = scoring(args.seed + offset)
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise RuntimeError('Incomplete/nonfinite scores; no retry or denominator reduction')
        ap = float(average_precision_score(y, scores))
        if abs(ap - core.independent_ap(y, scores)) > 1e-12:
            raise RuntimeError('Independent AP check failed')
        precision, recall, _ = precision_recall_curve(y, scores)
        metrics[bank] = dict(AP_percent=100 * ap, ROC_AUC_percent=100 * float(roc_auc_score(y, scores)),
                             PR_AUC_trapezoidal_percent=100 * float(auc(recall, precision)))
        arrays['R_' + bank] = scores
    np.savez_compressed(args.output / 'scores.npz', **arrays)
    receipt = dict(plan, training=info, metrics=metrics, batch_size=bs,
                   input_X_sha256=record['X_array_sha256'], input_y_sha256=record['y_array_sha256'],
                   train_indices_sha256=array_sha256(train_ids), test_indices_sha256=array_sha256(test_ids),
                   score_file_sha256=hashlib.sha256((args.output / 'scores.npz').read_bytes()).hexdigest(),
                   environment=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda))
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
