"""Read the exact admitted B3 TRAIN catalog; never alter or reinterpret splits."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
COMBINED = ROOT/'matcher_v2_20260930/strict_training_admission_01/combined_admission.json'
COMBINED_SHA = '07eb6da665f88088f872e64eb647ba419d3016f5c159326b7876cec21faed379'
BASE = ROOT/'s7_balanced_20260923/train24k_layered_v14_exact'
MANIFESTS = {
    'v17_filtered': ROOT/'aggressive_data_v17_full30k_20260927/dataset_03/train/archive_manifest.json',
    'v17.5': ROOT/'curriculum_full_20260928/dataset_02/v17.5/manifest.json',
    'v18': ROOT/'curriculum_full_20260928/dataset_02/v18/manifest.json',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def bound(path, expected=None):
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Source SHA mismatch: '+str(path))
    return dict(path=str(path), sha256=actual)

def source_catalog():
    bindings = [bound(COMBINED, COMBINED_SHA)]
    combined = read(COMBINED)
    manifests = {}
    for stage, path in MANIFESTS.items():
        bindings.append(bound(path))
        manifests[stage] = {r['pair_id']: r for r in read(path)['entries']}
    straight = read(combined['straight_admission']['path'])
    bindings.append(bound(**{'path':combined['straight_admission']['path'],
                             'expected':combined['straight_admission']['sha256']}))
    gen_spec = straight['full_generation_complete']
    bindings.append(bound(gen_spec['path'], gen_spec['sha256']))
    spec = read(gen_spec['path'])['datasets']['train']
    strict_path = Path(spec['manifest_path'])
    bindings.append(bound(strict_path, spec['manifest_sha256']))
    manifests['straight_seam'] = {r['pair_id']:r for r in read(strict_path)['entries']}
    rows = []
    for a in combined['catalog']:
        e = manifests[a['stage']][a['pair_id']]
        if e['sample_path'] != a['sample_path'] or e['sample_sha256'] != a['sample_sha256']:
            raise ValueError('Admitted sample differs from source manifest')
        rows.append(dict(admission=a, record=e))
    expected={('v17_filtered',True):4679,('v17_filtered',False):7500,
              ('v17.5',True):3000,('v17.5',False):3000,
              ('v18',True):1500,('v18',False):1500,
              ('straight_seam',True):3000,('straight_seam',False):3000}
    if Counter((r['admission']['stage'],r['admission']['label']) for r in rows)!=expected:
        raise ValueError('Exact B3 population changed')
    if len({r['admission']['pair_id'] for r in rows})!=len(rows):raise ValueError('Duplicate B3 pair identity')
    return rows, bindings

def expand(row):
    a, e = row['admission'], row['record']
    if a['stage']=='v17_filtered':
        slot, ordinal = map(int, Path(e['sample_path']).stem.split('_'))
        group = MANIFESTS['v17_filtered'].parent/'groups'/f'{slot:05d}.json'
        e = read(group)['records'][ordinal]
        if e['pair_id']!=a['pair_id'] or e['sample_sha256']!=a['sample_sha256']:
            raise ValueError('v17 group identity differs')
    return dict(admission=a, record=e)

def sidecars(row):
    a, e = row['admission'], row['record']
    result = {'sample': (Path(a['sample_path']), a['sample_sha256'])}
    if not a['label']:
        return result
    result['proof'] = (Path(e['proof_path']), e['proof_sha256'])
    if a['stage'] != 'straight_seam':
        lp = Path(e['latent_seam_artifact'])
        if not lp.is_absolute(): lp = Path(e['sample_path']).parent.parent/lp
        result['latent'] = (lp, e['latent_sha256'])
        old = read(BASE/'groups'/f"{e['baseline_slot']:05d}.json")['entries'][e['baseline_ordinal']]
        result['preweather'] = (BASE/old['weather_artifact'], None)
    return result
