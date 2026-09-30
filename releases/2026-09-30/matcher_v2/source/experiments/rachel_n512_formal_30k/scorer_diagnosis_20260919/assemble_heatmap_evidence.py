"""Project completed, exact frozen probes into the existing review app.

Source arrays remain untouched; this makes a bounded visual projection, not new
predictions. Existing reviews, metric rows and original case records are preserved.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

MODELS = ('s4', 's6', 's6_depth4', 's7')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def small(value):
    if isinstance(value, dict):
        return {k: small(v) for k, v in value.items()}
    if isinstance(value, list):
        return [small(v) for v in value]
    if isinstance(value, float):
        return float('%.7g' % value)
    return value


def mask_fragment(mask, original, side):
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('empty selected mask')
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = [65, 78, 84]
    rgba[..., 3] = mask * 255
    stream = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(stream, format='PNG')
    return dict(name=original.get('name', 'Fragment ' + side.upper()),
                mask_url='data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode(),
                rgb_url=original.get('rgb_url'),
                rgb_exact_alignment=False,
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                mask_rect_xywh=[0, 0, mask.shape[1], mask.shape[0]])


def summarize(probes, selections):
    """Descriptive within-sample effects, never whole-population estimates."""
    strata = {(r['dataset'], r['pair_id']): r['stratum'] for r in selections}
    groups = []
    for model, rows in probes.items():
        for stratum in sorted(set(strata.values())):
            chosen = [r for r in rows if strata[(r['dataset'], r['pair_id'])] == stratum]
            if not chosen:
                continue
            enrichments, inlier_share, inlier_fraction, deltas, controls = [], [], [], [], []
            for r in chosen:
                # Context-input attribution is spatially tied to frozen matcher tokens;
                # later tokens mix across fragments and need separate interpretation.
                for side in ('a', 'b'):
                    a = r['layer_layout_alignment'][0][side]
                    if a['enrichment_over_uniform'] is not None:
                        enrichments.append(a['enrichment_over_uniform'])
                    inlier_share.append(a['absolute_attribution_share_on_inliers'])
                    inlier_fraction.append(a['predicted_inlier_token_fraction'])
                by = {(p['selection'], p['mode']): p for p in r.get('perturbations', [])}
                if ('layout_inliers', 'mask') in by and ('random_for_layout', 'mask') in by:
                    deltas.append(by[('layout_inliers', 'mask')]['delta_logit'])
                    controls.append(by[('random_for_layout', 'mask')]['delta_logit'])
            med = lambda values: statistics.median([v for v in values if v is not None]) if any(v is not None for v in values) else None
            groups.append(dict(model=model, stratum=stratum, n=len(chosen),
                median_score=med([r['score'] for r in chosen]),
                median_input_inlier_enrichment=med(enrichments),
                median_input_attribution_share_on_inliers=med(inlier_share),
                median_inlier_token_fraction=med(inlier_fraction),
                median_delete_layout_logit_delta=med(deltas),
                median_delete_random_logit_delta=med(controls),
                median_paired_layout_minus_random_delta=med([a-b for a,b in zip(deltas,controls)])))
    return dict(status='complete', scope='fixed illustrative 40-case selection; not representative rate estimates',
                strata_reference='REAL groups use S6-D2 SIMVAL-maxF1 and its 20px layout criterion, not the currently summarized model',
                perturbation='off-manifold token masking, frozen matcher; one matched-count random draw per case', rows=groups)


def main(args):
    raw, bundle = Path(args.inputs), Path(args.bundle)
    selected = read(args.selection)
    keys = {(r['dataset'], r['pair_id']) for r in selected}
    canonical = {(('real' if r['dataset'] == 'dunhuang' else 'ood'), r['pair_id']): r for r in read(bundle / 'cases.json')['cases']}
    probes, identities = {}, {}
    for model in MODELS:
        protocol = read(raw / model / 'protocol.json')
        rows = read(raw / model / 'cases.json')
        if protocol['status'] != 'complete' or {(r['dataset'], r['pair_id']) for r in rows} != keys:
            raise ValueError('incomplete probe population: ' + model)
        probes[model] = rows
        identities[model] = protocol['model']
    lookups = {model: {(r['dataset'], r['pair_id']): r for r in rows} for model, rows in probes.items()}
    metric_rows = read(bundle / 'metrics.json')['rows']
    thresholds = {m: next(r['threshold'] for r in metric_rows if r['model_id'] == m and r['is_primary_selection'] and r['operating_point'] == 'max_f1') for m in MODELS}
    output, differences = [], []
    for choice in selected:
        key = (choice['dataset'], choice['pair_id'])
        old = canonical.get(key, {})
        ref = lookups['s4'][key]
        row = {k: choice[k] for k in ('dataset', 'pair_id', 'stratum', 'label')}
        row.update(name=choice.get('name', choice['pair_id']), models={},
                   gt_offset_xy=([-ref['target_translation_rc'][1], -ref['target_translation_rc'][0]] if ref['target_translation_rc'] is not None else None))
        masks_root = raw / 's4_masks' if (raw / 's4_masks').exists() else raw / 's4'
        with np.load(masks_root / ref['arrays_path'], allow_pickle=False) as arrays:
            for side in ('a', 'b'):
                row['fragment_' + side] = mask_fragment(arrays['mask_' + side], old.get('fragment_' + side, {}), side)
        for model in MODELS:
            r = lookups[model][key]
            if model in old.get('predictions', {}):
                difference = abs(r['score'] - old['predictions'][model]['score'])
                differences.append(difference)
                if difference > 1e-4:
                    raise ValueError('probe differs from historical prediction: %s %s %.6g' % (model, key, difference))
            layout = r['layout']
            row['models'][model] = small(dict(score=r['score'], threshold=thresholds[model],
                decision=bool(r['decision_valid'] and r['score'] >= thresholds[model]),
                logit=r['logit'], depth=r['depth'], decision_valid=r['decision_valid'],
                layout=dict(valid=layout['valid'], offset_xy=[layout['offset_b_in_a_rc'][1], layout['offset_b_in_a_rc'][0]],
                            error_px=layout['translation_l2_px'],
                            inlier_indices_a=layout['inlier_token_indices_a'], inlier_indices_b=layout['inlier_token_indices_b']),
                points_rc_a=r['points_rc_a'], points_rc_b=r['points_rc_b'],
                layers=[dict(name=x['name'], **{side:{k:x[side][k] for k in ('token_indices','signed','absolute')} for side in ('a','b')}) for x in r['layers']],
                pool=r['pool'], alignment=r['layer_layout_alignment'],
                perturbations=[{k:p[k] for k in ('selection','mode','delta_logit','delta_probability')} for p in r.get('perturbations', [])]))
        output.append(row)
    target = bundle / 'app/src/data.json'
    snapshot = read(target)
    old_query_hashes = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k,v in snapshot['queries'].items() if k != 'scorer_heatmaps'}
    snapshot['queries']['scorer_heatmaps'] = dict(rows=output, source=dict(
        label='冻结模型逐层token归因', files=[{'label':'selected_cases.receipt.json'}, {'label':'heatmap_probe.py'}, {'label':'heatmaps_v1/*/cases.json'}],
        metricDefinitions=[{'label':'Signed attribution', 'definition':'对pre-sigmoid logit求梯度，并对每个token的feature×gradient沿通道求和；仅为局部敏感性，不是因果或可加的logit分解。'},
                           {'label':'Absolute attribution', 'definition':'先对每通道feature×gradient取绝对值，再求和。'},
                           {'label':'绿色空心环', 'definition':'最佳位移峰的预测内点；不代表GT接缝。'}],
        caveats=['示例按已有分数和摆放分层选择，不是总体随机样本；不可外推总体比例。敦煌分组依据S6双层原SIMVAL阈值及20px摆放标准，不代表当前所选模型的分类。', 'RGB缩略图与800px mask不保证逐像素配准，归因只覆盖实际输入mask的轮廓token。', 'Attention与pool权重不等于因果贡献；后层token已经混合对方碎片信息。', 'Turufan与负例没有位姿GT，不把输出位移称为拼接成功。', 'token删除干预不等同于物理损伤；随机对照每例仅一次。'],
        evidenceFlow=[{'title':'原分数复现','detail':'REAL/OOD共%d个模型-pair核对，最大绝对概率差%.8g。' % (len(differences), max(differences, default=0))}] ))
    snapshot['buildStatus'] = args.build_status
    assert old_query_hashes == {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k,v in snapshot['queries'].items() if k != 'scorer_heatmaps'}
    write(target, snapshot)
    write(raw / 'attribution_summary.json', summarize(probes, selected))
    write(raw / 'projection_receipt.json', dict(status='complete', pairs=len(output), model_pairs=len(output)*len(MODELS),
        historical_score_checks=len(differences), maximum_score_error=max(differences, default=0), preserved_query_hashes=old_query_hashes))
    print(json.dumps(dict(pairs=len(output), model_pairs=len(output)*len(MODELS), snapshot_bytes=target.stat().st_size)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--build-status', choices=['updating', 'complete', 'paused'], default='updating')
    main(p.parse_args())
