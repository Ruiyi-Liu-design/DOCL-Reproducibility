"""Verify package checksums, frozen sources, coverage and reported table means.

Uses Python's standard library only. No dataset downloads, model execution,
training, rescoring or network requests are performed.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict
from summarize_results import read_csv, summarize

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / 'results/supplementary'
SUM = ROOT / 'results/summaries'


def close(a, b, label):
    if abs(float(a) - float(b)) > 1e-9:
        raise ValueError(f'Numerical mismatch: {label}: {a} versus {b}')


def compare_stats(result, reference, label):
    close(result['AP_percent'], reference['AP_percent'], label + ' AP')
    if result['AP_seed_SD_pp'] is None:
        if reference.get('AP_seed_SD_pp', '') not in ('', None):
            raise ValueError('Single-seed SD must be missing')
    else:
        close(result['AP_seed_SD_pp'], reference['AP_seed_SD_pp'], label + ' SD')


def verify():
    report = dict(status='PASS', network_requests=0, new_training_models=0,
                  new_real_data_scoring=0, checks=[])
    checks = report['checks']
    manifest = ROOT / 'checksums.sha256'
    if not manifest.exists():
        raise FileNotFoundError('checksums.sha256 is required')
    listed = set()
    for line in manifest.read_text('utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Checksum path escapes archive root')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'File checksum mismatch: {name}')
        listed.add(name)
    checks.append(dict(name='package_SHA256', files=len(listed)))
    copies = json.loads((ROOT / 'provenance/scientific_source_manifest.json').read_text('utf-8'))['copies']
    for row in copies:
        actual = hashlib.sha256((ROOT / row['archive_path']).read_bytes()).hexdigest()
        if not actual == row['source_sha256'] == row['archive_sha256']:
            raise ValueError('Frozen scientific source changed: ' + row['archive_path'])
    checks.append(dict(name='frozen_source_equality', files=len(copies)))
    scripts = list((ROOT / 'scripts').rglob('*.py')) + list((ROOT / 'src').rglob('*.py'))
    for path in scripts:
        ast.parse(path.read_text('utf-8'), filename=path.name)
    checks.append(dict(name='Python_syntax', files=len(scripts)))
    datasets = json.loads((ROOT / 'data/datasets24.json').read_text('utf-8'))['datasets']
    ids = {r['dataset'] for r in datasets}
    if len(datasets) != 24 or len(ids) != 24:
        raise ValueError('Dataset count is not 24 unique IDs')
    split = read_csv(ROOT / 'data/split_audit.csv')
    if len(split) != 72 or {(r['dataset'], int(r['seed'])) for r in split} != {(d, s) for d in ids for s in [42, 43, 44]}:
        raise ValueError('Split metadata coverage mismatch')
    checks.append(dict(name='dataset_and_split_metadata', datasets=24, splits=72))

    tables = {}
    expected_rows = {'Table_S1_normal_only_seed_noise.csv': 432,
                     'Table_S2_mixed_seed_noise.csv': 432,
                     'Table_S3_parameter_search_summary.csv': 29,
                     'Table_S3_parameter_search_seed_noise.csv': 2736,
                     'Table_S4_sampling_seed_noise.csv': 576,
                     'Table_S5_component_seed_noise.csv': 864,
                     'Table_S5_reverse_local_exceptions.csv': 4,
                     'Table_S6_normal_representation_diagnostics.csv': 720}
    for filename, count in expected_rows.items():
        rows = read_csv(SUPP / filename)
        if len(rows) != count:
            raise ValueError(f'{filename}: {len(rows)} != {count}')
        if 'dataset' in rows[0] and {r['dataset'] for r in rows} - ids:
            raise ValueError('Unknown dataset ID')
        tables[filename] = rows
    ref = {r['method']: r for r in read_csv(SUM / 'final_reference_summary.csv')}
    component = {r['method']: r for r in read_csv(SUM / 'component_summary.csv')}
    dependent = {r['method']: r for r in read_csv(SUM / 'matched_sampling_mixed_summary.csv')}
    norm = {r['method']: r for r in summarize(tables['Table_S1_normal_only_seed_noise.csv'])}
    mixed = {r['method']: r for r in summarize(tables['Table_S2_mixed_seed_noise.csv'])}
    docl = 'DOCL_uniform_lambda1p2_margin2p0_R49'
    compare_stats(norm['DDAE_R49'], ref['DDAE_R49'], 'normal DDAE')
    compare_stats(norm[docl], ref['uniform'], 'normal DOCL')
    compare_stats(mixed['DDAE_R49'], dependent['DDAE_mixed'], 'mixed DDAE')
    compare_stats(mixed[docl], dependent['mixed'], 'mixed DOCL')
    for r in summarize(tables['Table_S4_sampling_seed_noise.csv']):
        compare_stats(r, ref['uniform'] if r['method'] == 'uniform' else dependent[r['method']], 'pairing ' + r['method'])
    for r in summarize(tables['Table_S5_component_seed_noise.csv']):
        compare_stats(r, component[r['method']], 'component ' + r['method'])
    params = summarize(tables['Table_S3_parameter_search_seed_noise.csv'], ('lambda_value', 'margin_coefficient'))
    parref = {(float(r['lambda_value']), float(r['margin_coefficient'])): r for r in tables['Table_S3_parameter_search_summary.csv']}
    for r in params:
        key = (float(r['lambda_value']), float(r['margin_coefficient']))
        compare_stats(r, parref[key], 'parameter ' + str(key))
    full = [r for r in params if r['training_seeds'] == [42, 43, 44]]
    if len(params) != 29 or len(full) != 14:
        raise ValueError('Parameter confirmation coverage mismatch')
    best = max(full, key=lambda r: r['AP_percent'])
    if (float(best['lambda_value']), float(best['margin_coefficient'])) != (1.2, 2.0):
        raise ValueError('Final parameter selection mismatch')
    checks.append(dict(name='supplementary_coverage_and_AP_SD', tables=len(tables),
                       parameter_configurations=29, three_seed_configurations=14))

    groups = defaultdict(list)
    for r in tables['Table_S5_component_seed_noise.csv']:
        if r['method'] in ('uniform', 'reverse'):
            groups[(r['dataset'], r['method'])].append(float(r['AP_percent']))
    exceptions = {d for d in ids if statistics.mean(groups[(d, 'reverse')]) > statistics.mean(groups[(d, 'uniform')])}
    if exceptions != {r['dataset'] for r in tables['Table_S5_reverse_local_exceptions.csv']} or len(exceptions) != 4:
        raise ValueError('Reverse local exceptions mismatch')
    probes = tables['Table_S6_normal_representation_diagnostics.csv']
    probe_keys = {(r['dataset'], r['seed'], r['probe_bank'], 'uniform' if r['model'] == 'uniform' else r['method']) for r in probes}
    if len(probe_keys) != 720:
        raise ValueError('Probe rows are not deduplicated by actual model')
    checks.append(dict(name='local_exceptions_and_normal_probes', reverse_exceptions=4, unique_probe_rows=720))
    checks.append(dict(name='matched_AP_differences',
                       normal_DOCL_minus_DDAE_pp=norm[docl]['AP_percent'] - norm['DDAE_R49']['AP_percent'],
                       mixed_DOCL_minus_DDAE_pp=mixed[docl]['AP_percent'] - mixed['DDAE_R49']['AP_percent']))
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, allow_nan=False))
