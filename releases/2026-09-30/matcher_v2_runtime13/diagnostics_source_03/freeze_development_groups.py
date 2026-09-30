"""Freeze complete Dunhuang CAL/SELECT strata from masks and GT, not scores.

CPU-only. No model is instantiated; fold0 and Turufan are not measured.
The output fixes all 639 development pair IDs, including all negatives.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from diagnostic_metrics import classify_positive_geometry, require
import mask_geometry as api

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SOURCE = ROOT/'runtime_work_13'
BASE = Path('/root/autodl-tmp/curriculum_training_20260928/locked_plan_02/matcher_execution.json')
OUT = ROOT/'development_strata_01'
PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
REFERENCE = Path('/root/autodl-tmp/claudecode0929/diagnostics/d10_real/dunhuang_cv.jsonl')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bound(path):
    return dict(path=str(path), sha256=sha(path))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def development_members(meta, role_plan):
    require(role_plan['role_folds'] == {'real_cal': [1], 'real_select': [2, 3, 4], 'real_test': [0]},
            'registered source roles required')
    spec = role_plan['datasets']['dunhuang_cv']
    by_id = {r['pair_id']: r for r in meta['pairs']}
    require(len(by_id) == len(meta['pairs']), 'duplicate real pair identity')
    all_roles, seen, source_groups, fragments = {}, set(spec['excluded_gt_pair_ids']), {}, {}
    for role in ('real_cal', 'real_select', 'real_test'):
        ids = spec['roles'][role]['pair_ids']
        actual = [r['pair_id'] for r in meta['pairs'] if r['fold'] in role_plan['role_folds'][role]
                  and r['pair_id'] not in spec['excluded_gt_pair_ids']]
        require(actual == ids and len(ids) == len(set(ids)) and not seen.intersection(ids), 'real role membership changed')
        seen.update(ids)
        all_roles[role] = [dict(by_id[pid], role=role) for pid in ids]
        fragments[role] = {r[key] for r in all_roles[role] for key in ('fragment_a_id', 'fragment_b_id')}
        source_groups[role] = {meta['fragment_source_group'][f] for f in fragments[role]}
        for other in source_groups:
            require(other == role or not (source_groups[role] & source_groups[other] or fragments[role] & fragments[other]),
                    'real roles overlap by source or fragment')
    require(seen == set(by_id), 'missing real role membership')
    return all_roles['real_cal'] + all_roles['real_select']


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation required')
    require(not OUT.exists(), 'new frozen-strata directory required')
    require(sha(BASE) == 'b846706db28403a3e0a145daa64c4ceb7e00a034d8032433efa5da5c64e5134f', 'B0 execution changed')
    require(sha(SOURCE/'source_binding.json') == '9558af206eb70de7f9b62689ad4c9baa57587f875cc8fa309c0d3fa5719cde6c',
            'bound measurement source changed')
    metric_path = Path(__file__).resolve().with_name('mask_geometry.py')
    require(Path(api.__file__).resolve() == metric_path, 'wrong imported metric implementation')
    base = read(BASE)
    require(sha(base['real_split']['path']) == base['real_split']['sha256'] ==
            '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6', 'real roles changed')
    plan = read(base['real_split']['path']); spec = plan['datasets']['dunhuang_cv']
    require(sha(spec['remote_manifest']) == spec['manifest_sha256'], 'Dunhuang manifest changed')
    meta = read(spec['remote_manifest']); rows = development_members(meta, plan)
    require(len(rows) == 639 and sum(r['label'] for r in rows) == 233, 'registered development population differs')
    gt_file = Path(plan['gt_path']); gt_rows = read(gt_file)['positive_pairs']
    gt = {r['pair_id']: r for r in gt_rows}
    require(len(gt) == len(gt_rows), 'duplicate GT pair identity')
    cache = Path(spec['prepared'])/'inputs.npz'
    inputs = dict(execution=bound(BASE), real_split=base['real_split'], manifest=bound(spec['remote_manifest']),
        prepared_inputs=bound(cache), layout_gt=bound(gt_file), geometry_code=bound(metric_path),
        diagnostic_metrics_code=bound(Path(__file__).with_name('diagnostic_metrics.py')))
    with np.load(cache, allow_pickle=False) as z:
        masks = z['packed_masks']
    index = {f: i for i, f in enumerate(meta['fragment_ids'])}
    require(len(index) == len(meta['fragment_ids']) and masks.shape == (len(index), 800, 100), 'prepared mask shape differs')
    cv2.setNumThreads(1)
    output = []
    for pair in rows:
        require(type(pair['label']) is bool, 'binary real label required')
        result = {k: pair[k] for k in ('pair_id', 'role', 'fold', 'label', 'fragment_a_id', 'fragment_b_id')}
        result.update(seam_group=None, geometry=None)
        if pair['label']:
            target = gt[pair['pair_id']]
            require((target['fragment_a_token'], target['fragment_b_token']) ==
                    (pair['fragment_a_id'], pair['fragment_b_id']), 'GT endpoint order differs')
            ma, mb = [np.unpackbits(masks[index[pair['fragment_' + s + '_id']]], axis=-1).astype(bool) for s in 'ab']
            translation = np.asarray(target['translation_gt_a_to_b_rc'], float)
            require(translation.shape == (2,) and np.isfinite(translation).all(), 'invalid positive GT')
            a, b = api.rectangularity(ma), api.rectangularity(mb)
            seam = api.seam_profile(ma, mb, translation)
            bend = None if seam is None else seam['bend_range']
            result['seam_group'] = classify_positive_geometry(bend, a, b)
            result['geometry'] = dict(bend_range=bend, rectangularity_a=a, rectangularity_b=b,
                seam_pixels=None if seam is None else seam['seam_px'], extent_px=None if seam is None else seam['extent_px'])
        output.append(result)
    reference = [json.loads(line) for line in REFERENCE.read_text().splitlines() if line]
    ref_by_id = {r['pair_id']: r for r in reference}
    require(len(ref_by_id) == len(reference), 'reference has duplicate IDs')
    differences, compared = [], 0
    for row in output:
        if not row['label']:
            continue
        other = ref_by_id.get(row['pair_id'])
        require(other is not None, 'reference missing a development positive')
        seam = other.get('seam')
        group = classify_positive_geometry(None if not seam else seam.get('bend_range'),
            other['shape_a']['rectangularity'], other['shape_b']['rectangularity'])
        compared += 1
        if group != row['seam_group']:
            differences.append(dict(pair_id=row['pair_id'], actual=row['seam_group'], reference=group))
    for item in inputs.values():
        require(sha(item['path']) == item['sha256'], 'a source changed during freezing')
    OUT.mkdir()
    record = dict(schema='matcher-v2-development-strata/1', status='locked', rows=output, inputs=inputs,
        model_scores_used=False, model_inference=False, test_measured=False, turufan_measured=False,
        gt_used_only_for_mask_geometry=True, source_roles_disjoint=True,
        definition=dict(seam_max_pixel_distance=6., gaussian_sigma_px=15., straight_bend_range_max_px=12.,
                        rectangularity_min_of_two_threshold=.9, negatives_have_no_gt_seam_group=True),
        script=bound(__file__))
    save(OUT/'plan.json', record)
    counts = {role: dict(Counter(r['seam_group'] if r['label'] else 'negative' for r in output if r['role'] == role))
              for role in ('real_cal', 'real_select')}
    complete = dict(schema='matcher-v2-development-strata-complete/1', status='complete', plan=bound(OUT/'plan.json'),
        pair_count=len(output), positive_count=233, counts=counts,
        reference_comparison=dict(source=bound(REFERENCE), compared=compared, differences=differences,
                                  reference_scores_not_read_for_selection=True),
        model_inference=False, gpu_used=False, test_measured=False, completed_unix=time.time())
    save(OUT/'complete.json', complete)
    print(json.dumps(complete, indent=2))


if __name__ == '__main__':
    main()
