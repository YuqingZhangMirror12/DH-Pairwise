"""B22-frozen Matcher ablation: fixed features vs independently adapted features.

This is NOT a change to the completed v3 model. Proposal generation and all
Matcher weights are frozen in both arms. The independent branch starts as an
exact copy of the B22 Patch Encoder, scale gate and ArcContext; it receives only
Scorer losses. Final Verifier refinement/ranking is still trainable in BOTH
arms, so final selected layouts need not stay identical after training.
"""
from copy import deepcopy
from dataclasses import fields
import torch
from torch import nn
from .config import Config
from .model import SeamContextModel
from .data import INPUTS
from .losses import candidate_losses, average


class ScorerFeatures(nn.Module):
    def __init__(self, matcher):
        super().__init__();self.cfg=matcher.cfg
        self.sampler=deepcopy(matcher.sampler)
        self.patch_encoder=deepcopy(matcher.patch_encoder)
        self.scale_gate=deepcopy(matcher.scale_gate)
        self.arc_context=deepcopy(matcher.arc_context)

    def forward(self, mask_a, mask_b, ga, gb):
        fa=SeamContextModel.encode(self,mask_a,ga)
        fb=SeamContextModel.encode(self,mask_b,gb)
        ha,hb=self.arc_context(fa,fb,ga,gb)
        return fa,fb,ha,hb


class IndependentFeatureModel(SeamContextModel):
    def __init__(self,cfg,base_state,arm):
        if arm not in ('frozen_features','independent_features'):
            raise ValueError('unknown ablation arm')
        super().__init__(cfg);self.load_state_dict(base_state,strict=True)
        self.arm=arm;self.requires_grad_(False)
        self.verifier.requires_grad_(True)
        self.scorer_features=ScorerFeatures(self) if arm=='independent_features' else None
        if self.scorer_features is not None:self.scorer_features.requires_grad_(True)
        self.train(False)

    def train(self,mode=True):
        # All Matcher dropout/norm/checkpoint behavior remains evaluation-only.
        super().train(False)
        self.verifier.train(mode)
        if self.scorer_features is not None:self.scorer_features.train(mode)
        return self

    def forward(self,*inputs,decode=False,verify=False):
        with torch.no_grad():o=super().forward(*inputs,decode=decode or verify,verify=False)
        if verify:
            features=(o.fa,o.fb,o.ha,o.hb) if self.scorer_features is None else self.scorer_features(inputs[0],inputs[1],o.ga,o.gb)
            o.scorer_fa,o.scorer_fb,o.scorer_ha,o.scorer_hb=features
            o.verified=[self.verify_candidates(o,b,candidates) for b,candidates in enumerate(o.candidates)]
        return o

    def verify_candidates(self,o,b,candidates,record=None):
        return self.verifier(candidates,o.records[b] if record is None else record,b,
            o.scorer_fa,o.scorer_fb,o.scorer_ha,o.scorer_hb,o.ot1,o.ga,o.gb,*o.masks)


def from_checkpoint(path,arm):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    cfg=Config(**{f.name:cp['config'][f.name] for f in fields(Config)})
    return IndependentFeatureModel(cfg,cp['model'],arm)


class ScorerTrainingSystem(nn.Module):
    def __init__(self,model):super().__init__();self.model=model

    def forward(self,batch):
        o=self.model(*(batch[k] for k in INPUTS),decode=True,verify=True)
        zero=o.s1.sum()*0;cls=[];ranks=[];poses=[];positive=0;total=0
        for b,v in enumerate(o.verified):
            c,r,p,pos,n=candidate_losses(v,batch['labels'][b],batch['translation_valid'][b],
                batch['translation_a_to_b_rc'][b],batch['pose_enabled'][b],zero)
            cls.append(c);ranks.append(r);poses.append(p);positive+=pos;total+=n
        parts=dict(candidate=average(cls,zero),rank=average(ranks,zero),pose=average(poses,zero))
        loss=parts['candidate']+.5*parts['rank']+.25*parts['pose']
        loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        return loss,{k:v.detach() for k,v in parts.items()},dict(positive_candidates=positive,total_candidates=total)
