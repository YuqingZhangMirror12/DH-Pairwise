"""Sparse relation-node attention, support/link prediction and affinity residual."""
from types import SimpleNamespace
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .valid_contour import cyclic_delta


@torch.no_grad()
def broad_edges(q, ga, gb, b, cfg):
    na, nb = int(ga.counts[b]), int(gb.counts[b])
    if not na or not nb:
        return torch.empty((0, 2), dtype=torch.long, device=q.device)
    s = q[:na, :nb]
    kr, kc = min(cfg.topk, nb), min(cfg.topk, na)
    row = torch.argsort(s, dim=1, descending=True, stable=True)[:, :kr]
    col = torch.argsort(s, dim=0, descending=True, stable=True)[:kc]
    pairs = [torch.stack((torch.arange(na, device=q.device)[:, None].expand(-1, kr), row), -1).reshape(-1, 2),
             torch.stack((col, torch.arange(nb, device=q.device)[None].expand(kc, -1)), -1).reshape(-1, 2)]
    # Arc origins are geometric and compact; no padded-length buckets.
    ia = (ga.arc[b, :na]/ga.perimeter[b]*cfg.arc_bins).long().clamp_max(cfg.arc_bins-1)
    ib = (gb.arc[b, :nb]/gb.perimeter[b]*cfg.arc_bins).long().clamp_max(cfg.arc_bins-1)
    bucket = (ia[:, None]*cfg.arc_bins+ib[None]).flatten()
    flat = s.flatten()
    maxima = torch.full((cfg.arc_bins**2,), -torch.inf, device=q.device)
    maxima.scatter_reduce_(0, bucket, flat, reduce='amax', include_self=True)
    # Stable geometric storage tie. The row/column union remains symmetric.
    is_best = flat == maxima[bucket]
    indices = torch.arange(na*nb, device=q.device)
    selected = torch.full_like(maxima, na*nb, dtype=torch.long)
    selected.scatter_reduce_(0, bucket, torch.where(is_best, indices, na*nb), reduce='amin', include_self=True)
    selected = selected[selected < na*nb]
    pairs.append(torch.stack((selected//nb, selected % nb), -1))
    ids = torch.unique(torch.cat(pairs)[:, 0]*nb+torch.cat(pairs)[:, 1], sorted=True)
    if len(ids) > cfg.edge_cap:
        raise RuntimeError('broad candidate bound exceeded; never silently truncate to512')
    return torch.stack((ids//nb, ids % nb), -1)


def relation_features(edges, neighbors, ga, gb, b):
    i, j = edges.unbind(-1)
    ni, nj = edges[neighbors].unbind(-1)
    da = cyclic_delta(ga.arc[b, ni], ga.arc[b, i, None], ga.perimeter[b])
    db = cyclic_delta(gb.arc[b, nj], gb.arc[b, j, None], gb.perimeter[b])
    d = gb.points[b, j]-ga.points[b, i]
    dd = d[neighbors]-d[:, None]
    lo, hi = torch.minimum(da.abs(), db.abs()), torch.maximum(da.abs(), db.abs())
    shared_a, shared_b = ni == i[:, None], nj == j[:, None]
    f = torch.stack((torch.asinh(lo/32), torch.asinh(hi/32), torch.sign(da*db),
        torch.asinh(dd[..., 0].abs()/16), torch.asinh(dd[..., 1].abs()/16),
        torch.asinh(dd.norm(dim=-1)/16), (shared_a | shared_b).float(),
        (shared_a & shared_b).float(), (hi > 32).float(), torch.log((hi+1)/(lo+1))), -1)
    return f


@torch.no_grad()
def sparse_neighbors(edges, ga, gb, b, cap):
    e = len(edges)
    if not e:
        return edges.new_zeros((0, 1)), torch.zeros((0, 1), dtype=torch.bool, device=edges.device)
    i, j = edges.unbind(-1)
    aa, ab = ga.arc[b, i], gb.arc[b, j]
    delta = gb.points[b, j]-ga.points[b, i]
    result = []
    half, quarter = max(1, cap//2), max(1, cap//4)
    for start in range(0, e, 256):
        stop = min(start+256, e)
        da = cyclic_delta(aa[None], aa[start:stop, None], ga.perimeter[b])
        db = cyclic_delta(ab[None], ab[start:stop, None], gb.perimeter[b])
        physical = da.abs()+db.abs()
        dd = torch.cdist(delta[start:stop].float(), delta.float())
        local = physical+.25*dd
        gap = physical*.15+dd+((da*db) > 0)*1000.+(physical < 32)*1000.
        shared = (i[start:stop, None] == i[None]) | (j[start:stop, None] == j[None])
        conflict = physical*.1+torch.where(shared, torch.zeros_like(dd), 1000.+dd)
        choices = [torch.topk(cost, min(k, e), largest=False, sorted=False).indices
                   for cost, k in ((local, half), (gap, quarter), (conflict, quarter))]
        # Deduplicate attention keys: repeated neighbors must not get extra votes.
        ids = torch.cat(choices, 1).sort(1).values
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, 1:] = ids[:, 1:] != ids[:, :-1]
        result.append((ids, mask))
    return torch.cat([x[0] for x in result]), torch.cat([x[1] for x in result])


class SparseBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.d = heads, dim//heads
        self.norm, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim)
        self.relation = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x, neighbors, nvalid, relation):
        if not len(x):
            return x
        q, k, v = self.qkv(self.norm(x)).reshape(len(x), 3, self.heads, self.d).unbind(1)
        chunks = []
        for start in range(0, len(x), 256):
            ids = neighbors[start:start+256]
            score = (q[start:start+256, None].float()*k[ids].float()).sum(-1)/math.sqrt(self.d)
            score = score+self.relation(relation[start:start+256]).float()
            score = score.masked_fill(~nvalid[start:start+256, :, None], -torch.inf)
            weights = score.softmax(1).to(v.dtype)
            chunks.append((weights[..., None]*v[ids]).sum(1).flatten(1))
        x = x+self.out(torch.cat(chunks))
        return x+self.ff(self.norm2(x))


class CorrespondenceContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Sequential(nn.Linear(5*cfg.dim+7, cfg.dim), nn.LayerNorm(cfg.dim), nn.GELU())
        self.blocks = nn.ModuleList([SparseBlock(cfg.dim, cfg.heads) for _ in range(cfg.correspondence_layers)])
        self.support = nn.Linear(cfg.dim, 1)
        self.stop = nn.Linear(cfg.dim, 1)
        self.link = nn.Sequential(nn.Linear(2*cfg.dim+10, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, 3))
        self.delta = nn.Linear(cfg.dim, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def encode_edges(self, edges, b, fa, fb, ha, hb, roles, s, ot, ga, gb):
        i, j = edges.unbind(-1)
        pa, da, pb, db = roles
        ua, ub = ot.dustbin_col[b, i], ot.dustbin_row[b, j]
        disp = gb.points[b, j]-ga.points[b, i]
        meta = torch.stack((s[b, i, j], ot.real_transport[b, i, j],
            .5*(ua+ub), (ua-ub).abs(), torch.asinh(disp[:, 0].abs()/64),
            torch.asinh(disp[:, 1].abs()/64), torch.asinh(disp.norm(dim=-1)/64)), -1)
        features = torch.cat((fa[b, i]+fb[b, j], (fa[b, i]-fb[b, j]).abs(),
            ha[b, i]+hb[b, j], (ha[b, i]-hb[b, j]).abs(),
            .5*(pa[b, i]*db[b, j]+da[b, i]*pb[b, j]), meta), -1)
        return self.embed(features)

    def link_logits(self, x, neighbors, relation):
        return self.link(torch.cat((x[:, None]+x[neighbors], (x[:, None]-x[neighbors]).abs(), relation), -1))

    def forward(self, fa, fb, ha, hb, roles, s0, ot0, ga, gb):
        corrected, records = [], []
        for b in range(len(s0)):
            edges = broad_edges(ot0.real_transport[b], ga, gb, b, self.cfg)
            x = self.encode_edges(edges, b, fa, fb, ha, hb, roles, s0, ot0, ga, gb)
            neighbors, nvalid = sparse_neighbors(edges, ga, gb, b, self.cfg.neighbors)
            relation = relation_features(edges, neighbors, ga, gb, b)
            for block in self.blocks:
                x = checkpoint(block, x, neighbors, nvalid, relation, use_reentrant=False) if (
                    self.training and self.cfg.activation_checkpointing) else block(x, neighbors, nvalid, relation)
            support, stop = self.support(x).squeeze(-1), self.stop(x).squeeze(-1)
            links = self.link_logits(x, neighbors, relation)
            # Differentiable unique-cell scatter; never add probabilities to Q.
            delta = self.delta(x).squeeze(-1).float()
            flat = torch.zeros_like(s0[b]).flatten().scatter(0, edges[:, 0]*s0.shape[2]+edges[:, 1], delta)
            corrected.append(s0[b]+flat.reshape_as(s0[b]))
            records.append(SimpleNamespace(edges=edges, features=x, neighbors=neighbors,
                neighbor_valid=nvalid, relation=relation, support=support, stop=stop, links=links, delta=delta))
        return torch.stack(corrected), records
