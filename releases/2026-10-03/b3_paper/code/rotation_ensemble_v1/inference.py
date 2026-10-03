"""Frozen-runtime adapter; only six inputs enter Matcher/T16/heads."""
import importlib
import time
from pathlib import Path
import numpy as np
from .core import INPUTS, rotate_inputs, rotate_vectors

PREFIX='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


def module(name,root):
    m=importlib.import_module(PREFIX+name)
    if Path(root).resolve() not in Path(m.__file__).resolve().parents:raise ValueError('native module outside frozen source')
    return m


class RotationEnsemble:
    """Configure views at call time; returns all views for GT-free combination.

    Native proposals and trained heads run in each view's own frame. Only their
    output translations are inverse rotated. Do not inverse rotate feature
    tensors or average features/weights/poses.
    """
    def __init__(self,matcher,geometry,heads,source_root,device='cuda:0'):
        import torch
        self.torch=torch;self.matcher=matcher;self.geometry=geometry;self.heads=heads
        self.root=Path(source_root);self.device=device
        self.evidence=module('s7_consensus_v1.evidence',self.root)
        self.policy=module('s7_consensus_v1.pose_consensus',self.root)
        self.head_api=module('binary_scorer_v1.head',self.root)
        self.hashing=module('curriculum_training_v1.checkpoint_io',self.root)
        if self.policy.REVISION!='native-hypothesis-complete-link-union/1-diameter16':raise ValueError('native T16 differs')
        self.assert_frozen()

    def assert_frozen(self):
        for m in (self.matcher,*self.heads.values()):
            if any(x.training for x in m.modules()) or any(p.requires_grad for p in m.parameters()):
                raise ValueError('all modules must already be frozen/eval')

    def state_hashes(self):
        return {k:self.hashing.tree_sha(m.state_dict()) for k,m in [('matcher',self.matcher),*self.heads.items()]}

    def predict_view(self,inputs,pair_ids,angle,verify_native=False):
        torch=self.torch;self.assert_frozen()
        if len(pair_ids)!=len(set(pair_ids)):raise ValueError('duplicate batch identities')
        start=time.perf_counter();rotated=rotate_inputs(inputs,angle)
        batch={k:torch.as_tensor(rotated[k],device=self.device) for k in INPUTS}
        if len(batch['mask_a'])!=len(pair_ids):raise ValueError('batch identity mismatch')
        if str(self.device).startswith('cuda'):torch.cuda.synchronize()
        before=time.perf_counter()
        with torch.inference_mode():
            output=self.matcher(**batch)
            if str(self.device).startswith('cuda'):torch.cuda.synchronize()
            matcher_seconds=time.perf_counter()-before;rows=[]
            for i,pid in enumerate(pair_ids):
                pair=self.evidence.PairEvidence.from_matcher(output,i,batch['mask_a'],batch['mask_b'])
                bstart=time.perf_counter();builder=self.policy.PoseConsensusBuilder(self.geometry);proposals=builder(pair)
                if str(self.device).startswith('cuda'):torch.cuda.synchronize()
                builder_seconds=time.perf_counter()-bstart
                candidates=[];head_seconds={k:0. for k in self.heads}
                for j,proposal in enumerate(proposals.clusters):
                    x=self.head_api.inputs_for_cluster(pair,proposal,include_features=False)
                    heads={}
                    for key,head in self.heads.items():
                        hstart=time.perf_counter();readout=head(pair,proposal)
                        heads[key]=dict(logit=float(readout.logit),score=float(readout.score))
                        head_seconds[key]+=time.perf_counter()-hstart
                    pose=x.pose.detach().cpu().numpy()
                    candidates.append(dict(index=j,angle=angle,translation_view=pose.tolist(),
                        translation=rotate_vectors(pose,angle,inverse=True).tolist(),
                        q_sum=float(x.q.double().sum()),edge_count=len(x.edge_ids),heads=heads))
                hashes=self.hashing.tree_sha(dict(q=output.assignment[i],affinity=output.affinity[i]))
                if verify_native:
                    if angle!=0:raise ValueError('native parity is checked in the identity frame')
                    native=module('binary_scorer_v1.model',self.root)
                    diagnostic=module('curriculum_training_v1.matcher_diagnostics',self.root)
                    qrow=diagnostic.inspect_pair(pid,pair,proposals,label=None,all_clusters=builder.all_clusters)
                    if len(candidates)!=qrow['retained_count']:raise ValueError('native T16 count differs')
                    for c,n in zip(candidates,qrow['candidates']):
                        if c['q_sum']!=n['q_sum'] or c['translation']!=n['translation_rc']:raise ValueError('native Q/pose differs')
                    for key,head in self.heads.items():
                        n=native.BinaryConsensus(self.matcher,self.geometry,head=head).score_pair(pair,proposals=proposals)
                        if bool(candidates)!=n.has_candidate:raise ValueError('native head candidate validity differs')
                        if candidates:
                            winner=max(candidates,key=lambda c:(c['heads'][key]['logit'],-c['index']))
                            if winner['index']!=n.selected_cluster_id or winner['heads'][key]['score']!=float(n.score) or winner['translation']!=n.translation_a_to_b_rc.tolist():
                                raise ValueError('single-view head decision differs from native model')
                rows.append(dict(pair_id=pid,angle=angle,numeric_valid=bool(pair.numeric_valid),
                    candidates=candidates,model_inputs_sha256=self.hashing.tree_sha({k:v[i] for k,v in batch.items()}),
                    affinity_q_sha256=hashes,contour_index_order_preserved=True,
                    timing=dict(matcher_seconds_per_pair=matcher_seconds/len(pair_ids),
                                builder_seconds=builder_seconds,head_seconds=head_seconds)))
        elapsed=time.perf_counter()-start
        for row in rows:row['timing']['batch_amortized_seconds']=elapsed/len(rows)
        return rows

    def predict(self,inputs,pair_ids,views=(0,90,180,270)):
        return {a:self.predict_view(inputs,pair_ids,a) for a in views}


def prediction_without_timing(row):return {k:v for k,v in row.items() if k!='timing'}
