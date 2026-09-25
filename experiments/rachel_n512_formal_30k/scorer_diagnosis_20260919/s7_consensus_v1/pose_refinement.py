"""Bounded robust joint fit of full individual observations, never point means."""
from dataclasses import dataclass

import torch
from torch import Tensor

from .compatibility import CompatibilityConfig
from .consensus_head import EncodedEvidence
from .geometry import pair_frame


@dataclass(frozen=True)
class Refinement:
    translation: Tensor
    localization_weights: Tensor
    compatibility_weights: Tensor
    information: Tensor
    uncertainty_eigenvalues: Tensor
    underconstrained: bool
    steps: tuple


def refine_pose(encoded:EncodedEvidence,config:CompatibilityConfig,iterations=3,trust_sigmas=3.):
    evidence=encoded.evidence;pair=evidence.pair
    initial=evidence.pose;t=initial
    ga,gb=pair.ga,pair.gb
    na,nb=pair.q.shape
    normal,tangent,reliability=pair_frame(ga.outward_normal_rc[0,:na,None],gb.outward_normal_rc[0,None,:nb],
        ga.normal_reliability[0,:na,None],gb.normal_reliability[0,None,:nb])
    reliable=reliability>=config.normal_reliability_min
    spacing=.5*(ga.cell_px[0,:na,None]+gb.cell_px[0,None,:nb])
    sn=(config.normal_sigma_per_spacing*spacing).clamp_min(config.sigma_floor_px)
    st=(config.tangent_sigma_per_spacing*spacing).clamp_min(config.sigma_floor_px)
    se=(config.evidence_tangent_sigma_per_spacing*spacing).clamp_min(config.sigma_floor_px)
    sf=(config.fallback_sigma_per_spacing*spacing).clamp_min(config.sigma_floor_px)
    nn=normal[...,None]*normal[...,None,:]
    tt=tangent[...,None]*tangent[...,None,:]
    eye=torch.eye(2,device=t.device,dtype=t.dtype)
    location_matrix=torch.where(reliable[...,None,None],nn/sn[...,None,None].square()+tt/st[...,None,None].square(),
                                 eye/sf[...,None,None].square())
    pa,pb=encoded.a.local_probabilities[:,0],encoded.b.local_probabilities[:,0]
    ra,rb=encoded.a.localization_reliability,encoded.b.localization_reliability
    ca,cb=evidence.a.observed_arc_px,evidence.b.observed_arc_px
    # Learned values are bounded; alternate partners retain a shared endpoint
    # capacity through W. They are not averaged into a synthetic observation.
    measure=.5*(ca[:,None]+cb[None])
    support=.5*(pa[:,None]+pb[None])
    learned=.5*(ra[:,None]+rb[None])
    # Large compatible erosion gaps cannot suddenly become zero-offset anchors
    # merely because sigmoid initialization is nonzero.
    anchor=learned*evidence.localization_kernels
    base=evidence.weights*measure*support
    wl=base*anchor;wc=base*(1-anchor)
    local_scale=max(config.sigma_floor_px,float((.5*(ga.cell_px[0,:na].median()+gb.cell_px[0,:nb].median())).detach())*
                    max(config.normal_sigma_per_spacing,config.tangent_sigma_per_spacing))
    trust_radius=trust_sigmas*local_scale
    steps=[];info=torch.zeros_like(eye)
    for _ in range(iterations):
        residual,c=pair.compatibility(t,config)
        active=(c.normal_px<0)|(c.normal_px>config.damage_normal_upper_px)
        matrix=torch.where(reliable[...,None,None],nn*active[...,None,None]/sn[...,None,None].square()+tt/se[...,None,None].square(),
                           eye/sf[...,None,None].square())
        error=c.unexplained_residual_rc
        robust_l=torch.rsqrt(1+(residual/local_scale).square().sum(-1))
        robust_c=torch.rsqrt(1+(error/local_scale).square().sum(-1))
        ml=location_matrix*(wl*robust_l)[...,None,None]
        mc=matrix*(wc*robust_c)[...,None,None]
        info=(ml+mc).sum((0,1))
        rhs=((ml@residual[...,None])+(mc@error[...,None])).sum((0,1)).squeeze(-1)
        ridge=.01*info.trace().detach().clamp_min(1e-6)+1e-6
        delta=torch.linalg.solve(info+ridge*eye,rhs+ridge*(initial-t))
        proposed=t+delta
        shift=proposed-initial
        shift=shift*torch.clamp(torch.as_tensor(trust_radius,device=t.device)/shift.norm().clamp_min(1e-12),max=1.)
        t=initial+shift
        steps.append(t)
    eigenvalues=torch.linalg.eigvalsh(info)
    ratio=eigenvalues[0]/eigenvalues[-1].clamp_min(1e-12)
    underconstrained=bool((ratio.detach()<.05)|(base.detach().sum()<1e-6))
    # Curvature diagnostic, NOT a calibrated probabilistic confidence interval.
    uncertainty=1/eigenvalues.clamp_min(1/config.damage_normal_upper_px**2)
    return Refinement(t,wl,wc,info,uncertainty,underconstrained,tuple(steps))
