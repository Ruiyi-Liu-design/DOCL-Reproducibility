"""Recompute supplementary-table macro means and training-seed sample SD."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def summarize(rows, group_fields=('method',), expected_datasets=24):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, '') for k in group_fields)].append(row)
    output = []
    for group, values in sorted(groups.items()):
        cells = defaultdict(list)
        keys = set()
        for r in values:
            seed = int(r.get('training_seed', r.get('seed')))
            bank = r.get('scoring_noise_bank', r.get('bank'))
            key = (r['dataset'], seed, bank)
            if key in keys:
                raise ValueError(f'Duplicate seed/noise cell: {group}, {key}')
            keys.add(key)
            cells[(r['dataset'], seed)].append((bank, float(r['AP_percent'])))
        if any({b for b, _ in v} != {'A', 'B'} or len(v) != 2 for v in cells.values()):
            raise ValueError(f'Incomplete A/B pair: {group}')
        datasets = sorted({d for d, _ in cells})
        seeds = sorted({s for _, s in cells})
        if len(datasets) != expected_datasets or set(cells) != {(d, s) for d in datasets for s in seeds}:
            raise ValueError(f'Incomplete dataset/seed coverage: {group}')
        seed_means = {s: statistics.mean(statistics.mean(v for _, v in cells[(d, s)]) for d in datasets) for s in seeds}
        row = dict(zip(group_fields, group))
        row.update(datasets=len(datasets), training_seeds=seeds, score_bank_rows=len(values),
                   AP_percent=statistics.mean(seed_means.values()),
                   AP_seed_SD_pp=statistics.stdev(seed_means.values()) if len(seeds) > 1 else None,
                   seed_macro_AP_percent=seed_means)
        output.append(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', type=Path, default=ROOT / 'results/supplementary/Table_S1_normal_only_seed_noise.csv')
    p.add_argument('--group', nargs='+', default=['method'])
    args = p.parse_args()
    print(json.dumps(summarize(read_csv(args.table), args.group), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
