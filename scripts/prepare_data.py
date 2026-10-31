"""Download pinned upstream inputs and reproduce the paper's canonical arrays."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def records():
    return json.loads((ROOT / 'data/datasets24.json').read_text('utf-8'))['datasets']


def convert_source(record, path):
    """Keep source dtype and row order; only make labels one-dimensional int64."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['source_file_sha256']:
        raise ValueError(f"Source SHA256 mismatch: {record['dataset']}")
    if record['format'] == 'npz':
        with np.load(path, allow_pickle=False) as z:
            x, y = z['X'], z['y'].reshape(-1).astype(np.int64)
    elif record['format'] == 'mat_trandata':
        from scipy.io import loadmat
        data = loadmat(path)['trandata']
        x, y = data[:, :-1], data[:, -1].astype(np.int64)
    elif record['format'] == 'yeast_csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        fields = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
        if not {r['Class'] for r in rows} <= {'CYT', 'NUC', 'MIT', 'ERL'}:
            raise ValueError('Unexpected Yeast class; do not silently drop or relabel rows')
        x = np.array([[float(r[k]) for k in fields] for r in rows], dtype=np.float64)
        y = np.array([int(r['Class'] == 'ERL') for r in rows], dtype=np.int64)
    else:
        raise ValueError(record['format'])
    check_arrays(record, x, y)
    return x, y


def check_arrays(record, x, y):
    if x.shape != (record['n'], record['d']) or str(x.dtype) != record['raw_dtype']:
        raise ValueError(f"Shape/dtype mismatch: {record['dataset']}")
    if y.shape != (record['n'],) or y.dtype != np.dtype('int64'):
        raise ValueError('Expected a one-dimensional int64 label vector')
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Input must be finite and contain both binary classes')
    if array_sha256(x) != record['X_array_sha256'] or array_sha256(y) != record['y_array_sha256']:
        raise ValueError(f"Canonical array SHA256 mismatch: {record['dataset']}")


def load_canonical(record, data_dir):
    with np.load(Path(data_dir) / record['file'], allow_pickle=False) as z:
        x, y = z['X'], z['y']
    check_arrays(record, x, y)
    return x, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='all', help='Exact dataset ID or all')
    p.add_argument('--output', type=Path, default=ROOT / 'data/raw')
    p.add_argument('--cache', type=Path, default=ROOT / '.download-cache')
    p.add_argument('--download', action='store_true', help='Fetch from original providers; consult their data terms')
    p.add_argument('--list', action='store_true')
    args = p.parse_args()
    rows = records()
    if args.list:
        for r in rows:
            print(f"{r['dataset']}\t{r['n']} x {r['d']}\t{r['raw_dtype']}")
        return
    selected = [r for r in rows if args.dataset in ('all', r['dataset'])]
    if not selected:
        p.error('Dataset ID not found; use --list')
    if not args.download:
        p.error('Use --download to explicitly fetch upstream inputs, or --list to inspect metadata')
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    for r in selected:
        dest = args.output / r['file']
        if dest.exists():
            load_canonical(r, args.output)
            print(f"VERIFIED existing {r['dataset']}")
            continue
        cache = args.cache / (r['source_file_sha256'] + Path(r['source_path']).suffix)
        if not cache.exists():
            req = urllib.request.Request(r['download_url'], headers={'User-Agent': 'DOCL-Reproducibility'})
            with urllib.request.urlopen(req, timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != r['source_file_sha256']:
                raise ValueError(f"Downloaded source mismatch: {r['dataset']}")
            cache.write_bytes(content)
        x, y = convert_source(r, cache)
        temporary = dest.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, X=x, y=y)
        with np.load(temporary, allow_pickle=False) as z:
            check_arrays(r, z['X'], z['y'])
        temporary.replace(dest)
        print(f"PREPARED {r['dataset']}; source and canonical array hashes verified")


if __name__ == '__main__':
    main()
