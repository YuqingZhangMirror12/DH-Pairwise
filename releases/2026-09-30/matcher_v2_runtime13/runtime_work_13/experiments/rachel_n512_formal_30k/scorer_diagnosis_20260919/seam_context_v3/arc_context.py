"""Padding-safe pointwise normalization and true multihead self/cross attention."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .valid_contour import clean, cyclic_delta


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.d = heads, dim//heads
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, query, source, qvalid, svalid, bias=None):
        b, nq, _ = query.shape
        ns = source.shape[1]
        def split(x): return x.reshape(b, -1, self.heads, self.d).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(source)), split(self.v(source))
        logits = (q.float()@k.float().transpose(-1, -2))/math.sqrt(self.d)
        if bias is not None:
            logits = logits+bias.float()
        # Empty sources have a computational dummy key only. Their actual
        # attention output is zero, including out-projection bias.
        safe = svalid.clone()
        empty = ~safe.any(1)
        safe[:, 0] |= empty
        logits = logits.masked_fill(~safe[:, None, None], -torch.inf)
        weights = logits.softmax(-1).to(v.dtype)
        z = (weights@v).transpose(1, 2).reshape(b, nq, -1)
        return clean(self.out(z), qvalid & ~empty[:, None])


class PairBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(dim, heads), Attention(dim, heads)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim) for _ in range(3))
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, a, b, va, vb, ba=None, bb=None, cab=None, cba=None):
        a = clean(a+self.self_attn(self.n1(a), self.n1(a), va, va, ba), va)
        b = clean(b+self.self_attn(self.n1(b), self.n1(b), vb, vb, bb), vb)
        olda, oldb = self.n2(a), self.n2(b)
        a = clean(a+self.cross_attn(olda, oldb, va, vb, cab), va)
        b = clean(b+self.cross_attn(oldb, olda, vb, va, cba), vb)
        return clean(a+self.ff(self.n3(a)), va), clean(b+self.ff(self.n3(b)), vb)


class ArcContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([PairBlock(cfg.dim, cfg.heads) for _ in range(cfg.arc_layers)])
        self.relation = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, cfg.heads))
        self.position = nn.Linear(3, cfg.dim)

    def relation_bias(self, g):
        ds = cyclic_delta(g.arc[:, :, None], g.arc[:, None, :], g.perimeter[:, None, None])
        xy = g.points[:, :, None]-g.points[:, None, :]
        spacing = torch.log((g.cell[:, :, None]+1)/(g.cell[:, None, :]+1))
        hints = torch.stack([torch.exp(-ds.abs()/s) for s in (32., 64., 128., 256.)], -1)
        phase = ds/g.perimeter[:, None, None]*2*math.pi
        features = torch.cat((phase.sin()[..., None], phase.cos()[..., None],
                              torch.asinh(xy/64.), spacing[..., None], hints), -1)
        return self.relation(features).permute(0, 3, 1, 2)

    def forward(self, a, b, ga, gb):
        def pos(g):
            center = (g.points*g.valid[..., None]).sum(1)/g.counts.clamp_min(1)[:, None]
            f = torch.cat((torch.asinh((g.points-center[:, None])/64.),
                           torch.log1p(g.cell)[..., None]), -1)
            return clean(self.position(f), g.valid)
        a, b = clean(a+pos(ga), ga.valid), clean(b+pos(gb), gb.valid)
        ba, bb = self.relation_bias(ga), self.relation_bias(gb)
        for block in self.blocks:
            if self.training and self.cfg.activation_checkpointing:
                a, b = checkpoint(block, a, b, ga.valid, gb.valid, ba, bb, use_reentrant=False)
            else:
                a, b = block(a, b, ga.valid, gb.valid, ba, bb)
        return a, b
