"""Original S7 M12 Matcher + C16 matched-token head, staged fine-tuning only."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from ..matched_only import cache
from ..matched_only.model import make_fresh_scorer
from ..candidate_local.model import select_predicted_inliers
from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import compute_samplewise_phase_loss
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from .data import INPUTS

HEAD_SHA='2658c96453996138fd953dda24168d099cc1caef866d578aaae8e942e706eedf'


class HardModel(nn.Module):
    def __init__(self,base,head,loss_config):
        super().__init__();self.base_model=base;self.score_head=head;self.loss_config=loss_config;self.set_phase('matcher')

    def set_phase(self,phase):
        if phase not in ('matcher','scorer'):raise ValueError('unknown phase')
        self.phase=phase;self.base_model.requires_grad_(phase=='matcher');self.score_head.requires_grad_(phase=='scorer')
        for module in (self.base_model.coarse,self.base_model.local_head,self.base_model.fusion):module.requires_grad_(False)
        for p in self.parameters():p.grad=None
        return self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.phase=='scorer':self.base_model.eval()
        else:self.score_head.eval()
        for module in (self.base_model.coarse,self.base_model.local_head,self.base_model.fusion):module.eval()
        return self

    def evidence(self,batch,score=True):
        with torch.no_grad() if self.phase=='scorer' else nullcontext():
            o=self.base_model(*(batch[k] for k in INPUTS))
        selected=select_predicted_inliers(o.assignment,batch['points_rc_a'],batch['points_rc_b'],batch['contour_valid_a'],batch['contour_valid_b'])
        logits=None
        if score:
            logits=self.score_head(o.token_features_a,o.token_features_b,batch['contour_valid_a'],batch['contour_valid_b'],selected).logit
        return o,selected,logits

    def matcher_loss(self,o,batch):
        return compute_samplewise_phase_loss(o,(batch['labels'],batch['target_a'],batch['target_b'],
            batch['translation_a_to_b_rc'],batch['translation_valid']),batch['pose_enabled'],self.loss_config,'matcher')

    def forward(self,batch):
        if self.phase=='matcher':
            o=self.base_model(*(batch[k] for k in INPUTS));loss,parts=self.matcher_loss(o,batch)
        else:
            o,selected,logit=self.evidence(batch)
            loss=(F.binary_cross_entropy_with_logits(logit,batch['labels'],reduction='none')*o.training_valid).mean()
            parts={'pair_bce':loss}
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite hard fine-tuning loss')
        # All ranks retain the same reducer graph even when a batch has no
        # reliable pose or no selected endpoints. Zero terms add no supervision.
        loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        return loss,{k:v.detach() for k,v in parts.items()}


def load_origin(matcher,head):
    source=cache.source_checkpoint(matcher)
    if cache.sha(head)!=HEAD_SHA:raise ValueError('requires the historical matched-token C16 head')
    saved=torch.load(head,map_location='cpu',weights_only=False)
    if saved['head_epoch']!=16 or saved['identity']['arm']!='matched_tokens':raise ValueError('wrong historical head')
    base=cache.old.load_decoupled_checkpoint(source).base_model
    scorer=make_fresh_scorer('matched_tokens',seed=saved['identity']['head_seed'])
    scorer.load_state_dict(saved['model_state_dict'],strict=True)
    loss=RachelN512LossConfig(**source['loss_config'])
    return HardModel(base,scorer,loss)
