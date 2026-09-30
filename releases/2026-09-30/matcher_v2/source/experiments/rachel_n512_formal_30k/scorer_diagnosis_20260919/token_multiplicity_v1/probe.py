"""Separate token count from relative evidence multiplicity in a frozen head.

No new contour sampling, Matcher inference, Sinkhorn, optimization or threshold
selection. Uniform duplication preserves the feature distribution; selective
duplication deliberately changes it without adding unique feature information.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copies(features, member, kind):
    if kind in ('uniform1', 'uniform2', 'uniform4'):
        counts = torch.full((len(features),), int(kind[-1]), dtype=torch.long)
    elif kind == 'inliers4':
        counts = torch.where(member, 4, 1)
    elif kind == 'noninliers4':
        counts = torch.where(member, 1, 4)
    else:
        raise ValueError(kind)
    return features.repeat_interleave(counts, dim=0), counts


@torch.inference_mode()
def measure(head, a, b, member_a, member_b):
    if head.training or a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('frozen CPU head and features required')
    result = []
    variants = [('baseline', 'uniform1', 'uniform1'),
        ('both_uniform2', 'uniform2', 'uniform2'),
        ('both_uniform4', 'uniform4', 'uniform4'),
        ('a_uniform4_b_uniform1', 'uniform4', 'uniform1'),
        ('both_predicted_inliers4', 'inliers4', 'inliers4'),
        ('both_noninliers4', 'noninliers4', 'noninliers4')]
    for name, ka, kb in variants:
        aa, ca = copies(a, member_a, ka)
        bb, cb = copies(b, member_b, kb)
        va, vb = torch.ones(1, len(aa), dtype=torch.bool), torch.ones(1, len(bb), dtype=torch.bool)
        z = head(aa[None], bb[None], va, vb)[0]
        if not torch.isfinite(z):
            raise ValueError('nonfinite score')
        result.append(dict(variant=name, logit=float(z), probability=float(z.sigmoid()),
            count_a=len(aa), count_b=len(bb), unique_feature_ids_a=len(a), unique_feature_ids_b=len(b),
            predicted_inlier_copies_a=int(ca[member_a].sum()), predicted_inlier_copies_b=int(cb[member_b].sum())))
    for row in result:
        row['delta_logit'] = row['logit'] - result[0]['logit']
        row['delta_probability'] = row['probability'] - result[0]['probability']
    return result


def run(args):
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only diagnostic')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, args.source_root)
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    root, output = Path(args.probe_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(Path(args.selection_json).read_text())
    wanted = [r for r in selection if r['stratum'] == 'real_FN_good']
    for stratum in ('real_FP', 'real_TN', 'sim_positive', 'ood_cross_model_disagreement'):
        wanted.append(next(r for r in selection if r['stratum'] == stratum))
    if len(wanted) != 8:
        raise ValueError('fixed four FN-good plus four first-stratum controls required')
    source = json.loads((root / 's6/protocol.json').read_text())
    cases = {r['pair_id']:r for r in json.loads((root / 's6/cases.json').read_text())}
    model, identity = evaluation.load_frozen_model(source['model']['training_run'], source['model']['selection'])
    if identity['checkpoint_sha256'] != source['model']['checkpoint_sha256']:
        raise ValueError('source checkpoint changed')
    head = model.score_head.cpu().eval().requires_grad_(False)
    del model
    started = time.monotonic()
    rows = []
    for selected in wanted:
        case = cases[selected['pair_id']]
        arrays = root / 's6' / case['arrays_path']
        if sha(arrays) != case['arrays_sha256']:
            raise ValueError('saved context changed')
        with np.load(arrays, allow_pickle=False) as archive:
            a, b = (torch.from_numpy(archive['context_input_' + side + '_features'].copy()) for side in 'ab')
            members = [torch.from_numpy(np.isin(archive['valid_indices_' + side],
                case['layout']['inlier_token_indices_' + side])) for side in 'ab']
        measured = measure(head, a, b, *members)
        replay = abs(measured[0]['logit'] - case['raw_head_logit'])
        if replay > 2e-5 + 2e-5 * abs(case['raw_head_logit']):
            raise ValueError('original-head replay failed')
        rows.append(dict(pair_id=case['pair_id'], dataset=case['dataset'], name=selected.get('name'),
            stratum=selected['stratum'], label=case['label'], threshold=identity['operating_points']['thresholds']['max_f1'],
            layout_error_px=case['layout'].get('translation_l2_px'), arrays_sha256=case['arrays_sha256'],
            baseline_replay_logit_error=replay, interventions=measured))
    result = dict(schema_version='frozen-context-token-multiplicity/1', status='complete', rows=rows,
        source_checkpoint_sha256=identity['checkpoint_sha256'], cases_sha256=sha(root/'s6/cases.json'),
        selection_sha256=sha(args.selection_json), probe_sha256=sha(__file__),
        sampling='existing four REAL FN-good + first fixed FP/TN/SIM-positive/OOD-disagreement',
        elapsed_probe_seconds=time.monotonic()-started, device='cpu', cpu_threads=1,
        torch_version=str(torch.__version__), matcher_or_sinkhorn_recomputed=False,
        new_unique_patch_features=False, parameters_fitted=False, thresholds_fitted=False,
        gt_used_for_token_membership=False,
        caveats=['Head-input off-manifold intervention on frozen post-Matcher contextual features.',
            'Uniform repetition is NOT actual 512-to-2048 contour resampling or an S5/S8 experiment.',
            'Selective duplication changes empirical evidence weights, not unique information.',
            'Predicted inliers may be wrong; stronger local output is not proven adjacency.',
            'Eight illustrative cases cannot establish population accuracy or training effects.'])
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',cases=len(rows),seconds=result['elapsed_probe_seconds'])), flush=True)
    for row in rows:
        print(json.dumps(dict(name=row['name'], stratum=row['stratum'], scores={t['variant']:t['probability'] for t in row['interventions']}),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--probe-root',required=True)
    p.add_argument('--selection-json',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
