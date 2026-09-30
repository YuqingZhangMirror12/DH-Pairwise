"""Bounded CPU Scorer-only interventions on11 registered stored cases.

No Matcher forward, proposal construction, pose fitting, model update or new
case selection. Baseline must reproduce the original GPU head outputs before
any intervention is interpreted. Omitted-layer/feature tests are sensitivity
tests, not deployable alternatives or proof that a layer caused domain shift.
"""
import dataclasses
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts/consensus_threshold_20260925/training_source_01'
sys.path.insert(0,str(SOURCE))
PKG='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1'
Head=importlib.import_module(PKG+'.consensus_head').ConsensusEvidenceHead
evidence_module=importlib.import_module(PKG+'.evidence')
SideEvidence,RecalledEvidence=evidence_module.SideEvidence,evidence_module.RecalledEvidence


def main():
    torch.set_num_threads(2)
    out=ROOT/'artifacts/threshold_diagnosis_20260927'
    checkpoint=out/'threshold_m12_best_joint.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    head=Head().eval()
    head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items() if k.startswith('head.')},strict=True)
    initial={k:v.clone() for k,v in head.state_dict().items()}
    results=[]
    for job in ('m12_dunhuang_cv','m12_turufan'):
        directory=ROOT/'artifacts/threshold_m12_evaluation_20260927'/job
        index=json.loads((directory/'diagnostic_index.json').read_text())
        for entry in index['cases']:
            file=directory/entry['evidence'];meta=json.loads(file.read_text())
            cs=[c for c in meta['clusters'] if c['selected']]
            if not cs:
                results.append(dict(pair_id=meta['pair_id'],job=job,available=False,reason='no selected cluster'))
                continue
            c=cs[0];stage=c['stages']['final']
            with np.load(file.parent/meta['sidecar']['path'],allow_pickle=False) as a:
                tensor=lambda key:torch.from_numpy(np.array(a[key],copy=True))
                sides={s:SideEvidence(**{k:tensor(v) for k,v in stage[s]['input'].items()}) for s in 'ab'}
                e=RecalledEvidence(**{k:tensor(stage[k]) for k in ('pose','weights','kernels','localization_kernels')},
                                   **sides,pair=None)
                q=tensor(meta['pair']['q'])
                assert torch.equal(e.weights,q*e.kernels), 'threshold input must retain exact union Q'
                assert torch.equal(tensor(c['stages']['initial']['weights']),e.weights)
                original=float(a[c['readout']['score']])
                variants={}
                for name in ('baseline','zero_context_features','zero_self_attention','zero_cross_attention'):
                    handles=[];ev=e
                    if name=='zero_context_features':
                        ev=dataclasses.replace(e,**{s:dataclasses.replace(getattr(e,s),
                            context=torch.zeros_like(getattr(e,s).context),
                            opposite_context=torch.zeros_like(getattr(e,s).opposite_context)) for s in 'ab'})
                    if name in ('zero_self_attention','zero_cross_attention'):
                        attr=name.removeprefix('zero_')
                        for layer in head.layers:
                            handles.append(getattr(layer,attr).register_forward_hook(lambda m,args,result:torch.zeros_like(result)))
                    with torch.no_grad():
                        encoded=head(ev)
                        rr=head.readout(encoded,c['overlap']['fraction_min_area'])
                    for h in handles:h.remove()
                    if name=='baseline':
                        max_prob=max(float((getattr(encoded,s).local_probabilities-tensor(stage[s]['output']['local_probabilities'])).abs().max()) for s in 'ab')
                        assert max_prob<5e-5 and abs(float(rr.score)-original)<2e-6
                    variants[name]=dict(score=float(rr.score),logit=float(rr.logit),
                        positive_evidence_px=float(rr.positive_evidence_px),
                        conflict_evidence_px=float(rr.conflict_evidence_px),
                        delta_score=float(rr.score)-original)
                attention=[]
                for name,item in stage['attention'].items():
                    m=np.asarray(a[item['head_mean']],dtype=np.float64)
                    if m.size:
                        entropy=-(m*np.log(np.maximum(m,1e-30))).sum(-1)
                        measure=np.asarray(a[item['key_measure']],dtype=np.float64)
                        attention.append(dict(name=name,rows=m.shape[0],columns=m.shape[1],
                            median_effective_keys=float(np.median(np.exp(entropy))),
                            median_row_max=float(np.median(m.max(-1))),
                            query_count=m.shape[0],key_measure_total=float(measure.sum())))
                results.append(dict(pair_id=meta['pair_id'],job=job,available=True,
                    selected_cluster_id=c['cluster_id'],gpu_score=original,
                    baseline_local_prob_max_abs=max_prob,variants=variants,attention=attention,
                    union_q_equal=True,initial_final_union_equal=True,
                    positive_mass_in_union=float(e.weights.sum()),union_edges=int((e.kernels>0).sum())))
    assert all(torch.equal(v,initial[k]) for k,v in head.state_dict().items())
    result=dict(schema='threshold-head-registered-cpu-replay/1',status='passed',cases=results,
        matcher_forward_performed=False,optimizer_updates=0,weights_unchanged=True,
        scope='11 predetermined real C10 cases; saved final winner/pose/union held fixed',
        caveat='zeroing inputs/layers can be off-distribution; sensitivity is not causal attribution of domain gap')
    (out/'head_replay.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='passed',cases=len(results),max_local_prob_error=max(r.get('baseline_local_prob_max_abs',0) for r in results),
        scores=[dict(pair_id=r['pair_id'],variants=r.get('variants')) for r in results]),ensure_ascii=False))


if __name__=='__main__':main()
