"""Shared A/B evidence head: two independent self/cross-attention blocks."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .evidence import RecalledEvidence, SideEvidence


class MeasureAttention(nn.Module):
    def __init__(self, dim=96, heads=4):
        super().__init__()
        if dim%heads:
            raise ValueError('attention dimension must divide heads')
        self.heads,self.width=heads,dim//heads
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim);self.v=nn.Linear(dim,dim)
        self.out=nn.Linear(dim,dim)

    def forward(self, query, key, query_valid, key_valid, key_measure, geometry_bias=None):
        single=query.ndim==2
        if single:
            query,key=query[None],key[None]
            query_valid,key_valid=query_valid[None],key_valid[None]
            key_measure=key_measure[None]
            if geometry_bias is not None:
                geometry_bias=geometry_bias[None]
        batch,n,dim=query.shape;m=key.shape[1]
        if not n or not m:
            return torch.zeros_like(query[0] if single else query)
        q=self.q(query).reshape(batch,n,self.heads,self.width).transpose(1,2)
        k=self.k(key).reshape(batch,m,self.heads,self.width).transpose(1,2)
        v=self.v(key).reshape(batch,m,self.heads,self.width).transpose(1,2)
        logits=(q@k.transpose(-1,-2))/math.sqrt(self.width)
        # Attention integrates physical observation measure. Repeating a token
        # with a partitioned measure is not an additional independent vote.
        logits=logits+key_measure.clamp_min(1e-30).log()[:,None,None]
        if geometry_bias is not None:
            logits=logits+geometry_bias[:,None]
        logits=logits.masked_fill(~key_valid[:,None,None],-torch.inf)
        valid_keys=key_valid.any(-1)
        logits=torch.where(valid_keys[:,None,None,None],logits,torch.zeros_like(logits))
        weights=logits.float().softmax(-1).to(v.dtype)
        result=(weights@v).transpose(1,2).reshape(batch,n,dim)
        result=self.out(result)
        result=torch.where((query_valid&valid_keys[:,None])[...,None],result,torch.zeros_like(result))
        return result[0] if single else result


class EvidenceBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_norm=nn.LayerNorm(dim);self.cross_norm=nn.LayerNorm(dim);self.ff_norm=nn.LayerNorm(dim)
        self.self_attention=MeasureAttention(dim,heads)
        self.cross_attention=MeasureAttention(dim,heads)
        self.ff=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))

    def forward(self, a,b,va,vb,ma,mb,bias):
        na,nb=self.self_norm(a),self.self_norm(b)
        a=a+self.self_attention(na,na,va,va,ma)
        b=b+self.self_attention(nb,nb,vb,vb,mb)
        na,nb=self.cross_norm(a),self.cross_norm(b)
        # Simultaneous updates, not A-first/B-after-A: exact A/B symmetry.
        aa=a+self.cross_attention(na,nb,va,vb,mb,bias)
        bb=b+self.cross_attention(nb,na,vb,va,ma,bias.transpose(-1,-2))
        a=aa+self.ff(self.ff_norm(aa));b=bb+self.ff(self.ff_norm(bb))
        return torch.where(va[...,None],a,0.),torch.where(vb[...,None],b,0.)


@dataclass(frozen=True)
class EncodedSide:
    state: Tensor
    local_logits: Tensor  # support, unknown/evidence-insufficient, conflict
    local_probabilities: Tensor
    localization_reliability: Tensor


@dataclass(frozen=True)
class EncodedEvidence:
    a: EncodedSide
    b: EncodedSide
    evidence: RecalledEvidence


@dataclass(frozen=True)
class Readout:
    logit: Tensor
    score: Tensor
    positive_evidence_px: Tensor
    conflict_evidence_px: Tensor
    observed_mass_length_px: Tensor
    support_weights_a: Tensor
    support_weights_b: Tensor


class ConsensusEvidenceHead(nn.Module):
    def __init__(self, feature_dim=96, dim=96, heads=4, layers=2, length_scale_px=32.):
        super().__init__()
        if (dim,heads,layers)!=(96,4,2):
            raise ValueError('first registered experiment is2layers/96dim/4heads')
        self.input=nn.Sequential(nn.Linear(4*feature_dim+17,dim),nn.LayerNorm(dim),nn.GELU())
        self.layers=nn.ModuleList([EvidenceBlock(dim,heads) for _ in range(layers)])
        self.output_norm=nn.LayerNorm(dim)
        self.local_classifier=nn.Linear(dim,3)
        self.localizer=nn.Linear(dim,1)
        nn.init.constant_(self.localizer.bias,-3.)
        self.bias=nn.Parameter(torch.tensor(-2.))
        self.positive_scale=nn.Parameter(torch.tensor(1.8545865)) # softplus ~=2
        self.conflict_scale=nn.Parameter(torch.tensor(1.8545865))
        self.overlap_scale=nn.Parameter(torch.tensor(3.9815145)) # softplus ~=4
        self.register_buffer('length_scale_px',torch.tensor(float(length_scale_px)))

    def _embed(self, s:SideEvidence):
        raw=torch.cat((s.local,s.context,s.opposite_local,s.opposite_context,s.scalar_features),-1)
        raw=torch.where(s.valid[:,None],raw,torch.zeros_like(raw))
        return torch.where(s.valid[:,None],self.input(raw),0.)

    def _output(self,state,valid):
        state=self.output_norm(state)
        logits=self.local_classifier(state)
        probabilities=logits.softmax(-1)
        # No evidence is not supervised as an invented third-class target.
        # It is only a numerical absent-query mask in the forward calculation.
        probabilities=torch.where(valid[:,None],probabilities,0.)
        reliability=torch.where(valid,self.localizer(state).squeeze(-1).sigmoid(),0.)
        return EncodedSide(torch.where(valid[:,None],state,0.),logits,probabilities,reliability)

    def dense_reference(self,evidence:RecalledEvidence):
        """Unbatched masked implementation retained for numerical gate tests."""
        a,b=evidence.a,evidence.b
        ha,hb=self._embed(a),self._embed(b)
        # Finite bias retains multi-partner uncertainty, while strongly
        # geometrically incompatible cross-side interactions are suppressed.
        bias=evidence.kernels.clamp_min(1e-30).log()
        ma,mb=a.mass*a.observed_arc_px,b.mass*b.observed_arc_px
        for layer in self.layers:
            ha,hb=layer(ha,hb,a.valid,b.valid,ma,mb,bias)
        return EncodedEvidence(self._output(ha,a.valid),self._output(hb,b.valid),evidence)

    def forward_many(self,evidences):
        """Batch candidate attention; omit ONLY exactly-zero observation mass.

        Full Q and every nonzero compatible observation remain in evidence.
        This packs the existing numerical absent-query mask, not a new TopK or
        score threshold. Scatter restores the original valid endpoint IDs.
        """
        if not evidences:
            return []
        packed={}
        for side in 'ab':
            values=[getattr(e,side) for e in evidences]
            indices=[s.valid.nonzero(as_tuple=False).flatten() for s in values]
            counts=[len(i) for i in indices];width=max(1,max(counts))
            raw=[torch.cat((s.local,s.context,s.opposite_local,s.opposite_context,s.scalar_features),-1)[i]
                 for s,i in zip(values,indices)]
            raw=torch.stack([F.pad(x,(0,0,0,width-len(x))) for x in raw])
            measure=torch.stack([F.pad((s.mass*s.observed_arc_px)[i],(0,width-len(i))) for s,i in zip(values,indices)])
            valid=torch.arange(width,device=raw.device)[None]<torch.tensor(counts,device=raw.device)[:,None]
            states=torch.where(valid[...,None],self.input(raw),0.)
            packed[side]=(indices,counts,width,measure,valid,states)
        ia,ca,wa,ma,va,ha=packed['a'];ib,cb,wb,mb,vb,hb=packed['b']
        bias=torch.stack([F.pad(e.kernels[a][:,b].clamp_min(1e-30).log(),(0,wb-len(b),0,wa-len(a)))
                          for e,a,b in zip(evidences,ia,ib)])
        for layer in self.layers:
            ha,hb=layer(ha,hb,va,vb,ma,mb,bias)
        outputs=[]
        for k,evidence in enumerate(evidences):
            sides=[]
            for side,h,indices,count in (('a',ha,ia,ca),('b',hb,ib,cb)):
                local=self._output(h[k,:count[k]],torch.ones(count[k],dtype=torch.bool,device=h.device))
                n=len(getattr(evidence,side).mass)
                def scatter(x):
                    return x.new_zeros((n,)+x.shape[1:]).index_copy(0,indices[k],x)
                sides.append(EncodedSide(*(scatter(getattr(local,name)) for name in
                    ('state','local_logits','local_probabilities','localization_reliability'))))
            outputs.append(EncodedEvidence(*sides,evidence))
        return outputs

    def forward(self,evidence:RecalledEvidence):
        return self.forward_many([evidence])[0]

    def readout(self,encoded:EncodedEvidence,overlap_fraction):
        a,b=encoded.evidence.a,encoded.evidence.b
        pa,pb=encoded.a.local_probabilities,encoded.b.local_probabilities
        wa,wb=a.observed_arc_px*a.mass,b.observed_arc_px*b.mass
        support_a,support_b=wa*pa[:,0],wb*pb[:,0]
        # A/B are two views of the same observation, averaged not doubled.
        positive=.5*(support_a.sum()+support_b.sum())
        negative=.5*((wa*pa[:,2]).sum()+(wb*pb[:,2]).sum())
        logit=(self.bias+F.softplus(self.positive_scale)*torch.log1p(positive/self.length_scale_px)
            -F.softplus(self.conflict_scale)*torch.log1p(negative/self.length_scale_px)
            -F.softplus(self.overlap_scale)*overlap_fraction)
        return Readout(logit,logit.sigmoid(),positive,negative,.5*(wa.sum()+wb.sum()),support_a,support_b)
