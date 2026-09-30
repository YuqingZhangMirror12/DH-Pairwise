"""Finite CPU replay of candidate stage groups on an existing complete cache."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from .candidate_groups import build_candidate_groups


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(args):
    root = args.cache
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['status'] != 'complete':
        raise ValueError('requires complete cache, not partially committed rows')
    names = ('points_a', 'points_b', 'valid_a', 'valid_b', 'candidate_indices',
             'candidate_valid', 'candidate_weights', 'candidate_inliers',
             'translation_a_to_b_rc', 'layout_valid', 'ready')
    arrays = {n: np.load(root/(n+'.npy'), mmap_mode='r', allow_pickle=False) for n in names}
    pairs = json.loads((root/'pairs.json').read_text())
    if len(pairs) != protocol['pair_count'] or not arrays['ready'].all():
        raise ValueError('cache row count or readiness differs')
    start, rows = time.monotonic(), []
    for i, pair in enumerate(pairs):
        result = build_candidate_groups(arrays['points_a'][i], arrays['points_b'][i],
            arrays['candidate_indices'][i], arrays['candidate_valid'][i],
            arrays['candidate_weights'][i], arrays['candidate_inliers'][i],
            arrays['translation_a_to_b_rc'][i], arrays['layout_valid'][i],
            valid_a=arrays['valid_a'][i], valid_b=arrays['valid_b'][i])
        rows.append(dict(pair_id=pair['pair_id'], **asdict(result)))
    result = dict(status='complete', source_cache=str(root.resolve()), pair_count=len(rows),
        source_checkpoint_sha256=protocol['source_checkpoint_sha256'],
        source_pairs_sha256=protocol['pairs_sha256'],
        source_is_training_probe=not protocol['formal_training_eligible'],
        matcher_forwards=0, training_performed=False, GT_used=False,
        mode_count_distribution=dict(Counter(len(r['multi_modes']) for r in rows)),
        seed_final_same_edges=sum(r['diagnostics']['seed_final_edges_equal'] is True for r in rows),
        seed_final_identical=sum(r['diagnostics']['seed_final_identical'] is True for r in rows),
        max_first_mode_replay_l2_px=max((r['diagnostics']['first_mode_replay_l2_px'] or 0 for r in rows), default=None),
        elapsed_seconds=time.monotonic()-start, rows=rows)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, default=serial, allow_nan=False)
        output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
