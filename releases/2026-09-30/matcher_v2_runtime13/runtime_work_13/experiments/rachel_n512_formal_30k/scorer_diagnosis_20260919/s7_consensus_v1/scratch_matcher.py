"""Same S7 architecture, freshly initialized; original matching-only losses."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.training.rachel_n512_loss import _validate_targets
from .matcher import S7MatcherAdapter


@dataclass(frozen=True)
class MatcherLossConfig:
    assignment_weight: float = .5
    translation_weight: float = .5
    residual_weight: float = .05
    translation_scale_px: float = 32.
    residual_target: float = 1e-3
    epsilon: float = 1e-8


def fresh_matcher(reference_config,seed=26092407):
    # No state tensor is copied. Only the original model architecture/settings.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base=RachelN512Pairwise(reference_config)
    return S7MatcherAdapter(base,frozen=False)


def matching_loss(adapter,output,batch,config=None):
    cfg=config or MatcherLossConfig()
    valid=output.numeric_valid
    checked=SimpleNamespace(assignment=output.assignment,training_valid=valid)
    _validate_targets(checked,batch['labels'],batch['target_a'],batch['target_b'],
        batch['translation_a_to_b_rc'],batch['translation_valid'],validate_values=True)
    def mean(values,mask):
        return (values*mask).sum(1)/mask.sum(1).clamp_min(1)
    ta,tb=batch['target_a'],batch['target_b']
    matches=(ta>=0)&valid[:,None]
    probability=output.assignment.gather(2,ta.clamp_min(0)[...,None]).squeeze(-1)
    nll=mean(-probability.clamp_min(cfg.epsilon).log(),matches)
    ma=(ta==-1)&valid[:,None];mb=(tb==-1)&valid[:,None]
    da=mean(-output.unmatched_a.clamp_min(cfg.epsilon).log(),ma)
    db=mean(-output.unmatched_b.clamp_min(cfg.epsilon).log(),mb)
    pa,pb=ma.any(1).float(),mb.any(1).float()
    dustbin=(da*pa+db*pb)/(pa+pb).clamp_min(1)
    # This historical whole-Q pose auxiliary is CLEAN-only pretraining. It is
    # never the final consensus decoder, nor an exact-zero target on erosion.
    estimate,_,_=adapter.base._translation(output.assignment,output.points_rc_a,output.points_rc_b)
    pose=batch['pose_enabled']&batch['translation_valid']&valid
    target=torch.where(pose[:,None],batch['translation_a_to_b_rc'],torch.zeros_like(estimate))
    translation=F.smooth_l1_loss(estimate/cfg.translation_scale_px,target/cfg.translation_scale_px,
        reduction='none').mean(1)*pose
    diag=output.transport.diagnostics
    residual=torch.relu(torch.maximum(diag.row_residual_max,diag.col_residual_max)-cfg.residual_target)*valid
    total=cfg.assignment_weight*(nll+dustbin)+cfg.translation_weight*translation+cfg.residual_weight*residual
    parts=dict(assignment_nll=(nll+dustbin).mean(),match_nll=nll.mean(),dustbin_nll=dustbin.mean(),
        translation_smooth_l1=translation.mean(),sinkhorn_residual=residual.mean())
    counts=dict(pairs=len(total),match_targets=int(matches.sum()),dustbin_targets=int(ma.sum()+mb.sum()),
                exact_pose_pairs=int(pose.sum()))
    return total.mean(),parts,counts
