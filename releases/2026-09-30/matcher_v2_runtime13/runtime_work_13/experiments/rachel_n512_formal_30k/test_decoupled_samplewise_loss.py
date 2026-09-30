"""Finite small CPU fixtures: B1 accumulation vs physical B2/B4/B8."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from experiments.rachel_n512_formal_30k.decoupled_samplewise_loss import (
    LOSS_NAMES, compute_samplewise_phase_loss, samplewise_phase_terms)
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import compute_weathering_loss


@contextmanager
def raises(error, match=".*"):
    with unittest.TestCase().assertRaisesRegex(error, match):
        yield


def fixture(batch, dtype=torch.float64):
    torch.manual_seed(516)
    shapes={"assignment":(batch,4,4),"dustbin_a":(batch,4),"dustbin_b":(batch,4),
            "translation":(batch,2),"fused":(batch,),"local":(batch,),"coarse":(batch,),
            "row_residual":(batch,),"col_residual":(batch,)}
    leaves={k:torch.randn(*shape,dtype=dtype,requires_grad=True) for k,shape in shapes.items()}
    valid=torch.tensor([1,1,1,1,0,1,1,1][:batch],dtype=torch.bool)
    out=SimpleNamespace(assignment=leaves["assignment"].sigmoid(),unmatched_a=leaves["dustbin_a"].sigmoid(),
        unmatched_b=leaves["dustbin_b"].sigmoid(),translation_hat_rc=leaves["translation"]*50,
        fused_logit=leaves["fused"],local_logit=leaves["local"],coarse_logit=leaves["coarse"],
        training_valid=valid,coarse=SimpleNamespace(valid_problem=torch.ones(batch,dtype=torch.bool)),
        transport=SimpleNamespace(diagnostics=SimpleNamespace(
            row_residual_max=leaves["row_residual"].sigmoid()*.02,
            col_residual_max=leaves["col_residual"].sigmoid()*.02)))
    labels=torch.tensor([1,1,0,0,1,1,0,1][:batch],dtype=dtype)
    ta=torch.tensor([[0,1,-1,-1],[0,-1,-2,-2],[-1,-1,-1,-1],[-2,-2,-2,-2],
                     [0,-1,-2,-2],[0,1,2,-2],[-1,-1,-2,-2],[0,1,-2,-2]][:batch])
    tb=torch.tensor([[0,1,-1,-1],[0,-2,-2,-2],[-1,-1,-1,-1],[-2,-2,-2,-2],
                     [0,-1,-2,-2],[0,1,2,-1],[-2,-2,-2,-2],[0,1,-2,-2]][:batch])
    targets=(labels,ta,tb,torch.randn(batch,2,dtype=dtype)*30,labels.bool())
    pose=torch.tensor([1,0,0,0,1,1,0,0][:batch],dtype=torch.bool)
    return out,targets,pose,leaves


def sliced(output,index):
    if isinstance(output,torch.Tensor):
        return output[index:index+1]
    return SimpleNamespace(**{k:sliced(v,index) for k,v in vars(output).items()})


def legacy_single(output,targets,pose,phase):
    if phase=="matcher":
        config=replace(RachelN512LossConfig(),fused_pair_weight=0.,coarse_pair_weight=0.,local_pair_weight=0.)
        result=compute_weathering_loss(output,*targets,pose_supervision_enabled=pose,config=config)
        return result.total,{k:getattr(result,k) for k in LOSS_NAMES}
    logit=output.fused_logit
    total=(torch.nn.functional.binary_cross_entropy_with_logits(logit,targets[0],reduction="none")
           *output.training_valid.to(logit.dtype)).sum()/output.training_valid.sum().clamp_min(1)
    parts={k:total.detach().new_zeros(()) for k in LOSS_NAMES};parts["fused_pair_bce"]=total
    return total,parts


def equivalence(batch,phase,dtype=torch.float64):
    out,targets,pose,leaves=fixture(batch,dtype)
    total,parts=compute_samplewise_phase_loss(out,targets,pose,RachelN512LossConfig(),phase)
    singles=[legacy_single(sliced(out,i),tuple(t[i:i+1] for t in targets),pose[i:i+1],phase) for i in range(batch)]
    reference=torch.stack([row[0] for row in singles]).mean()
    atol=2e-7 if dtype==torch.float32 else 2e-13
    assert torch.allclose(total,reference,atol=atol,rtol=atol)
    assert set(parts)==set(LOSS_NAMES)
    for name,value in parts.items():
        assert torch.allclose(value,torch.stack([row[1][name] for row in singles]).mean(),atol=atol,rtol=atol),name
    for left,right in zip(torch.autograd.grad(total,tuple(leaves.values()),retain_graph=True,allow_unused=True),
                          torch.autograd.grad(reference,tuple(leaves.values()),allow_unused=True)):
        if left is None or right is None:
            nonnull=left if left is not None else right
            assert nonnull is None or torch.count_nonzero(nonnull)==0
        else:
            assert torch.allclose(left,right,atol=atol,rtol=atol),(left-right).abs().max().item()


def test_physical2_4_and8_match_micro1_all_components_and_gradients():
    for batch in (2,4,8):
        for phase in ("matcher","classifier"):
            equivalence(batch,phase)


def test_float32_physical2_4_and8_gradient_tolerance():
    for batch in (2,4,8):
        for phase in ("matcher","classifier"):
            equivalence(batch,phase,torch.float32)


def test_per_sample_empty_supervision_zero_and_dustbin_side_denominators():
    out,targets,pose,_=fixture(8)
    parts=samplewise_phase_terms(out,targets,pose,RachelN512LossConfig(),"matcher")
    assert parts["match_nll"][2:5].eq(0).all()  # negative, all-ignore, invalid-positive.
    assert parts["assignment_nll"][3:5].eq(0).all()
    assert parts["total"][4].item()==0  # Invalid sample stays in batch denominator.
    assert parts["translation_smooth_l1"][[1,2,3,4,6,7]].eq(0).all()
    assert torch.equal(parts["dustbin_nll"][1],parts["dustbin_a_nll"][1])
    assert torch.equal(parts["dustbin_nll"][5],parts["dustbin_b_nll"][5])
    assert torch.equal(parts["dustbin_nll"][0],.5*(parts["dustbin_a_nll"][0]+parts["dustbin_b_nll"][0]))
    assert parts["dustbin_nll"][7].item()==0


def test_all_invalid_batch_is_zero_not_rescaled_and_C_does_not_read_targets():
    out,targets,pose,leaves=fixture(8)
    out.training_valid.zero_()
    total,_=compute_samplewise_phase_loss(out,targets,pose,RachelN512LossConfig(),"matcher")
    assert total.item()==0
    other=(targets[0],object(),object(),object(),object())
    classification,_=compute_samplewise_phase_loss(out,other,object(),RachelN512LossConfig(),"classifier")
    assert classification.item()==0
    assert torch.autograd.grad(classification,leaves["fused"])[0].eq(0).all()


def test_effective16_physical2_4_or8_matches16_micro1_gradient():
    # Repeat a varied eight-pair fixture twice to exercise the group scaling.
    for physical in (2,4,8):
        out,targets,pose,leaves=fixture(8)
        losses=[]
        for start in range(0,8,physical):
            def crop(value):
                if isinstance(value,torch.Tensor):return value[start:start+physical]
                return SimpleNamespace(**{k:crop(v) for k,v in vars(value).items()})
            value,_=compute_samplewise_phase_loss(crop(out),tuple(t[start:start+physical] for t in targets),
                pose[start:start+physical],RachelN512LossConfig(),"matcher")
            losses.append(value*(physical/16))
        physical_total=2*sum(losses)
        reference=2*sum(legacy_single(sliced(out,i),tuple(t[i:i+1] for t in targets),pose[i:i+1],"matcher")[0]/16 for i in range(8))
        a=torch.autograd.grad(physical_total,tuple(leaves.values()),retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(reference,tuple(leaves.values()),allow_unused=True)
        assert torch.allclose(physical_total,reference,atol=1e-13,rtol=1e-13)
        for x,y in zip(a,b):
            if x is None or y is None:
                nonnull=x if x is not None else y
                assert nonnull is None or nonnull.eq(0).all()
            else:assert torch.allclose(x,y,atol=1e-13,rtol=1e-13)


def test_pose_contract_rejects_negative_supervision_without_changing_original_config():
    out,targets,pose,_=fixture(4)
    pose[2]=True
    config=RachelN512LossConfig()
    with raises(ValueError,"negative"):
        compute_samplewise_phase_loss(out,targets,pose,config,"matcher")
    assert config.fused_pair_weight==1. and config.coarse_pair_weight==.25 and config.local_pair_weight==.5
