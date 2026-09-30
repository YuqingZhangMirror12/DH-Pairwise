"""Measure encoded-patch vs post-context drift at EXACT common coordinates."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
import torch
from probe import anchor_indices, sha, save


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, args.source_root)
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as ev
    from staging.pairwise_v0_2.pairwise_data.rachel_preprocess import extract_ordered_outer_contour
    root, out = Path(args.probe_root), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads(Path(args.count_results).read_text())
    # All cases with physically identical, unique 512 anchors on BOTH sides at1024.
    rows = [r for r in previous['rows'] if all(v['maximum_distance_px'] == 0 and v['unique_current_anchors'] == 512
        for v in next(x for x in r['variants'] if x['variant']=='resampled1024')['anchor_512']['sides'].values())]
    protocol = json.loads((root/'protocol.json').read_text())
    model, identity = ev.load_frozen_model(protocol['model']['training_run'], protocol['model']['selection'])
    if identity['checkpoint_sha256'] != previous['protocol']['source_checkpoint_sha256']:
        raise ValueError('different checkpoint')
    base = model.base_model.cpu().eval().requires_grad_(False)
    base.config = replace(base.config, contour_cap=1024)
    cases = {r['pair_id']:r for r in json.loads((root/'cases.json').read_text())}
    output, started = [], time.monotonic()
    for selected in rows:
        case = cases[selected['pair_id']]
        path = root/case['arrays_path']
        if sha(path) != case['arrays_sha256']:
            raise ValueError('saved masks changed')
        with np.load(path, allow_pickle=False) as archive:
            masks = {s:archive['mask_'+s].copy() for s in 'ab'}
        snapshots = {}
        for cap in (512,1024):
            points={s:extract_ordered_outer_contour(masks[s].astype(bool),cap=cap,smoothing_sigma=3.)[0] for s in 'ab'}
            captured={}
            def pre_hook(module, inputs):
                captured['encoded_a'],captured['encoded_b'] = (x.clone() for x in inputs[:2])
            def post_hook(module, inputs, outputs):
                captured['context_a'],captured['context_b'] = (x.clone() for x in outputs)
            handles=[base.context.register_forward_pre_hook(pre_hook),base.context.register_forward_hook(post_hook)]
            try:
                tensors=[torch.tensor(masks[s][None,None],dtype=torch.float32) for s in 'ab']
                tensors += [torch.tensor(points[s][None],dtype=torch.float32) for s in 'ab']
                tensors += [torch.ones(1,len(points[s]),dtype=torch.bool) for s in 'ab']
                with torch.inference_mode():
                    base(*tensors)
            finally:
                for h in handles:
                    h.remove()
            snapshots[cap] = (points,captured)
        side_records={}
        for s in 'ab':
            ids,detail=anchor_indices(snapshots[512][0][s],snapshots[1024][0][s])
            if detail['maximum_distance_px'] != 0 or detail['unique_current_anchors'] != 512:
                raise ValueError('expected exact physical anchors')
            side_records[s]=detail
            for stage in ('encoded','context'):
                a=snapshots[512][1][stage+'_'+s]
                b=snapshots[1024][1][stage+'_'+s][:,ids,:]
                side_records[s][stage]=dict(max_abs=float((a-b).abs().max()),
                    mean_abs=float((a-b).abs().mean()),
                    mean_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean()),
                    relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a)))
        output.append(dict(name=selected['name'],pair_id=selected['pair_id'],sides=side_records))
    save(out,dict(status='complete',rows=output,elapsed_seconds=time.monotonic()-started,
        source_checkpoint_sha256=identity['checkpoint_sha256'],script_sha256=sha(__file__),
        count_results_sha256=sha(args.count_results),cpu_threads=1,GPU_used=False,
        interpretation='Same exact physical anchor patches; measured before/after full CyclicLandmarkContext. Does not separate circular convolution, normalization and landmark cross-attention.'))
    print(json.dumps(dict(status='complete',cases=len(output),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','probe-root','count-results','output'):
        p.add_argument('--'+k,required=True)
    run(p.parse_args())
