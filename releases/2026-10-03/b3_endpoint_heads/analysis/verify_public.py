"""Offline published-file and aggregate check; no model/data/GPU/network use."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'code_manifest.json').read_text())
for item in manifest['files']:
    path = ROOT/item['path']
    data = path.read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    if path.suffix == '.py': ast.parse(data, filename=item['path'])
result = json.loads((ROOT/'evidence/select_scan_aggregates.json').read_text())
weights = {'v17_filtered':.3, 'v17.5':.3, 'v18':.3, 'strict_straight':.1}
for row in result['candidates']:
    assert sum(c['positives'] for s in row['stages'].values() for c in s['strata']) == 796
    assert sum(c['negatives'] for s in row['stages'].values() for c in s['strata']) == 800
    for stage in row['stages'].values():
        cells = stage['strata']
        for name, numerator in [('layout','layout_hits'), ('candidate_coverage','coverage_hits')]:
            rate = math.fsum(c[numerator]/c['positives'] for c in cells)/len(cells)
            assert math.isclose(rate, stage[name], abs_tol=1e-12, rel_tol=1e-12)
        assert math.isclose(stage['loss'], math.fsum(c['loss'] for c in cells)/len(cells), abs_tol=1e-12)
    for source, target in [('layout','macro_layout'), ('loss','macro_loss'),
                           ('candidate_coverage','macro_candidate_coverage')]:
        actual = math.fsum(weights[s]*row['stages'][s][source] for s in weights)
        assert math.isclose(actual, row[target], abs_tol=1e-12, rel_tol=1e-12)
cutoff = max(c['macro_layout'] for c in result['candidates'])-.005
eligible = [c for c in result['candidates'] if c['macro_layout'] >= cutoff]
assert [c['update'] for c in eligible] == result['eligible_updates']
winner = min(eligible, key=lambda c:(c['macro_loss'], -c['macro_layout'], c['update']))
assert winner['update'] == result['selected_update'] == 29667
print('Verified %d exact files and 16 aggregate candidates; winner U29667. No raw-prediction or GPU verification performed.'%len(manifest['files']))
