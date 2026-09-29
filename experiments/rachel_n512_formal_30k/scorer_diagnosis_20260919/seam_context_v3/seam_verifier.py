"""Whole directed candidate arcs, gap/context observations and final-pose score."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .arc_context import PairBlock, Attention
from .seam_proposals import open_arc_indices


def robust_translation(delta, weights, initial, iterations=3):
    estimate = initial.float()
    for _ in range(iterations):
        residual = (delta.float()-estimate).norm(dim=-1)
        robust = weights.float()/(1+(residual/16.).square())
        estimate = (robust[:, None]*delta.float()).sum(0)/robust.sum().clamp_min(1e-12)
    return estimate


@torch.no_grad()
def material_evidence(ma, mb, translations):
    """Symmetric quarter-resolution area estimate in both unbounded pair frames.

    Every original A pixel can intersect B even if B extends beyond A's canvas;
    denominators are full, untranslated areas (no canvas clipping loophole).
    A 9px interior mask separates deep penetration from raster boundary noise.
    """
    h, w = ma.shape[-2:]
    def prep(m):
        exterior = F.pad(1-m.float(), (4, 4, 4, 4), value=1.)
        interior = 1-F.max_pool2d(exterior, 9, stride=1)
        return torch.cat((F.avg_pool2d(m.float(), 4), F.avg_pool2d(interior, 4)), 1)
    a, b = prep(ma), prep(mb)
    hh, ww = a.shape[-2:]
    rr, cc = torch.meshgrid(torch.arange(hh, device=ma.device)*4+1.5,
                           torch.arange(ww, device=ma.device)*4+1.5, indexing='ij')
    def overlap(source, target, t):
        # align_corners=False uses block-center coordinates.
        grid = torch.stack(((cc[None]+t[:, None, None, 1]+.5)*2/w-1,
                            (rr[None]+t[:, None, None, 0]+.5)*2/h-1), -1)
        shifted = F.grid_sample(target.expand(len(t), -1, -1, -1), grid,
                                align_corners=False, padding_mode='zeros')
        return (source*shifted).sum((-1, -2))*16
    intersection = .5*(overlap(a, b, translations)+overlap(b, a, -translations))
    area_a, area_b = ma.sum(), mb.sum()
    smaller = torch.minimum(area_a, area_b).clamp_min(1)
    union = (area_a+area_b-intersection[:, 0]).clamp_min(1)
    return torch.stack((intersection[:, 0]/smaller, intersection[:, 0]/union,
                        intersection[:, 1]/smaller), -1)


class SeamVerifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.project = nn.Linear(2*cfg.dim+14, cfg.dim)
        self.blocks = nn.ModuleList([PairBlock(cfg.dim, cfg.heads) for _ in range(cfg.verifier_layers)])
        self.geometry_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, cfg.heads))
        self.query = nn.Parameter(torch.randn(1, 1, cfg.dim)*.02)
        self.pool = Attention(cfg.dim, cfg.heads)
        self.fuse = nn.Sequential(nn.Linear(2*cfg.dim, cfg.dim), nn.LayerNorm(cfg.dim), nn.GELU())
        self.reweight = nn.Sequential(nn.Linear(3*cfg.dim+3, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.quality = nn.Sequential(nn.Linear(cfg.dim+8, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 1))
        self.null = nn.Sequential(nn.Linear(cfg.dim+8, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, candidates, record, b, fa, fb, ha, hb, ot, ga, gb, ma, mb):
        if not candidates:
            return SimpleNamespace(candidates=[], logits=ha.new_empty(0), translations=ha.new_empty(0, 2),
                                   winner=-1, score=ha.new_zeros(()), has_candidate=False)
        device = ha.device
        count = len(candidates)
        t0 = torch.as_tensor(np.stack([c.translation for c in candidates]), device=device)
        edges = record.edges
        selected, chainids = [], []
        for c in candidates:
            ids = torch.as_tensor(c.edge_ids, device=device, dtype=torch.long)
            chainids.append(ids)
            aidx, bidx = edges[ids].unbind(-1)
            chosen = []
            for g, ix in ((ga, aidx), (gb, bidx)):
                n = int(g.counts[b])
                # Context can extend256px, but each observed point is read once.
                idx, u = open_arc_indices(g.arc[b, :n].cpu().numpy(), float(g.perimeter[b]),
                                         ix.cpu().numpy(), extension=256.)
                chosen.append((torch.as_tensor(idx, device=device), torch.as_tensor(u, device=device)))
            selected.append(chosen)
        widths = [max(len(x[side][0]) for x in selected) for side in (0, 1)]
        tensors, masks, pointsets, indexmaps = [], [], [], []
        center = torch.stack([.5*(ga.points[b, edges[ids, 0]].mean(0)+
                                      gb.points[b, edges[ids, 1]].mean(0)) for ids in chainids])
        for side, (g, raw, context) in enumerate(((ga, fa, ha), (gb, fb, hb))):
            rows, valids, coordinates, mappings = [], [], [], []
            for k, ((ids, u), chain) in enumerate(zip([s[side] for s in selected], chainids)):
                n = len(ids)
                p = g.points[b, ids]+(t0[k]/2 if side == 0 else -t0[k]/2)
                raw_q = ot.real_transport[b].sum(1 if side == 0 else 0)[ids]
                unmatched = (ot.dustbin_col if side == 0 else ot.dustbin_row)[b, ids]
                support = torch.isin(ids, edges[chain, side])
                # Explicit support / internal observation / external context role.
                internal = (u >= 0) & (u <= 1)
                support_arc = g.arc[b, edges[chain, side]].unique(sorted=True)
                arc_gaps = torch.remainder(support_arc.roll(-1)-support_arc, g.perimeter[b])
                span = (g.perimeter[b]-arc_gaps.max()) if len(support_arc)>1 else support_arc.new_zeros(())
                outside_distance = ((-u).clamp_min(0)+(u-1).clamp_min(0))*span.clamp_min(1)
                ranges = torch.stack([torch.exp(-outside_distance/s) for s in
                                      (32., self.cfg.context_extension_px, 128., 256.)], -1)
                meta = torch.cat((torch.asinh((p-center[k])/64.), torch.asinh(u[:, None]),
                    torch.log1p(g.cell[b, ids])[:, None], raw_q[:, None], unmatched[:, None],
                    support.float()[:, None], (internal & ~support).float()[:, None],
                    (~internal).float()[:, None], torch.asinh((p-center[k]).norm(dim=1)/256.)[:, None], ranges), -1)
                value = self.project(torch.cat((raw[b, ids], context[b, ids], meta), -1))
                rows.append(F.pad(value, (0, 0, 0, widths[side]-n)))
                valids.append(torch.arange(widths[side], device=device) < n)
                coordinates.append(F.pad(p, (0, 0, 0, widths[side]-n)))
                mapping = torch.full((g.points.shape[1],), -1, device=device, dtype=torch.long)
                mapping[ids] = torch.arange(n, device=device)
                mappings.append(mapping)
            tensors.append(torch.stack(rows)); masks.append(torch.stack(valids))
            pointsets.append(torch.stack(coordinates)); indexmaps.append(mappings)
        a, bb = tensors
        distance = pointsets[0][:, :, None]-pointsets[1][:, None]
        rel = torch.cat((torch.asinh(distance.abs()/32.), torch.asinh(distance.norm(dim=-1, keepdim=True)/32.)), -1)
        checkpointed = self.training and self.cfg.activation_checkpointing
        bias = (checkpoint(self.geometry_bias, rel, use_reentrant=False) if checkpointed
                else self.geometry_bias(rel)).permute(0, 3, 1, 2)
        for block in self.blocks:
            if checkpointed:
                a, bb = checkpoint(block, a, bb, *masks, None, None,
                                   bias, bias.transpose(-1, -2), use_reentrant=False)
            else:
                a, bb = block(a, bb, *masks, cab=bias, cba=bias.transpose(-1, -2))
        query = self.query.expand(count, -1, -1)
        validq = torch.ones(count, 1, device=device, dtype=torch.bool)
        za = self.pool(query, a, validq, masks[0]).squeeze(1)
        zb = self.pool(query, bb, validq, masks[1]).squeeze(1)
        z = self.fuse(torch.cat((za+zb, (za-zb).abs()), -1))
        translations, summaries = [], []
        for k, ids in enumerate(chainids):
            ia, ib = edges[ids].unbind(-1)
            ea, eb = a[k, indexmaps[0][k][ia]], bb[k, indexmaps[1][k][ib]]
            q = ot.real_transport[b, ia, ib]
            delta = gb.points[b, ib]-ga.points[b, ia]
            residual0 = (delta-t0[k]).norm(dim=1)
            ua, ub = ot.dustbin_col[b, ia], ot.dustbin_row[b, ib]
            meta = torch.stack((q, .5*(ua+ub), torch.asinh(residual0/16)), -1)
            weights = q*F.softplus(self.reweight(torch.cat((ea+eb, (ea-eb).abs(),
                z[k].expand(len(ids), -1), meta), -1)).squeeze(-1))
            t = robust_translation(delta, weights, t0[k])
            translations.append(t)
            residual = (delta-t).norm(dim=1)
            normal = (ga.normals[b, ia]*gb.normals[b, ib]).sum(1)
            w = weights/weights.sum().clamp_min(1e-12)
            summaries.append(torch.stack(((q*w).sum(), ((ua+ub)*.5*w).sum(),
                torch.asinh((residual*w).sum()/16), torch.asinh(torch.sqrt((residual.square()*w).sum()+1e-8)/16),
                (normal*w).sum())))
        t = torch.stack(translations)
        geometry = material_evidence(ma[b:b+1], mb[b:b+1], t.detach())
        evidence = torch.cat((torch.stack(summaries), geometry), -1)
        combined = torch.cat((z, evidence), -1)
        quality = self.quality(combined).squeeze(-1)
        # A learned candidate-set null; padding never enters the candidate set.
        null = self.null(combined.mean(0, keepdim=True)).squeeze()
        logits = quality-null
        winner = int(logits.detach().argmax())
        return SimpleNamespace(candidates=candidates, logits=logits, translations=t,
            quality=quality, null=null, evidence=evidence, winner=winner,
            score=logits[winner].sigmoid(), has_candidate=True,
            arc_point_counts=[tuple(len(v[0]) for v in row) for row in selected])
