"""Two-layer local CA; labels/GT never enter the network or evidence selection.

Limits count correspondence records (one Patch on each side), not a resampling
of the frozen Matcher. A repeated endpoint can belong to more than one record.
GCN is a separate, trainable Scorer-only branch over frozen context features.
"""
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ..matched_only.model import FreshMatchedScorer, _features, validate_selection


@dataclass(frozen=True)
class Config:
    name: str = "reference512_h4"
    cap: int = 512
    heads: int = 4
    graph: str = "none"
    joint_d: bool = False
    stable: bool = False
    dim: int = 96
    depth: int = 2


CONFIGS = {c.name: c for c in (
    Config(), Config("cap128_h4", cap=128), Config("cap256_h4", cap=256),
    Config("cap512_h8", heads=8), Config("gcn_pairing_h4", graph="pairing"),
    Config("gcn_shredding_h4", graph="shredding"),
    Config("joint_D_h4", joint_d=True), Config("stable_h4", stable=True),
)}


def interact(layer, a, b, valid, support=None):
    safe = valid.clone()
    safe[:, 0] |= ~valid.any(1)
    bias = torch.zeros_like(safe, dtype=a.dtype).masked_fill(~safe, -torch.inf)
    if support is not None:
        bias = bias + support.clamp_min(1e-8).log()
    q, k = layer.norm(a), layer.norm(b)
    update, _ = layer.cross_attention(q, k, k, key_padding_mask=bias, need_weights=False)
    value = a + update
    value = value + layer.ffn(layer.output_norm(value))
    return torch.where(valid[..., None], value, torch.zeros_like(value))


def batched_readout(head, a, b, valid, support=None):
    """Same weights/math as the old per-pair head, with batch padding masked."""
    for layer in (head, *head.extra_layers):
        a, b = interact(layer, a, b, valid, support), interact(layer, b, a, valid, support)
    safe = valid.clone()
    safe[:, 0] |= ~valid.any(1)
    def pool(x):
        logits = head.pool_gate(x).squeeze(-1).masked_fill(~safe, -torch.inf)
        if support is not None:
            logits = logits + support.clamp_min(1e-8).log()
        attention = logits.softmax(1)
        mean = (attention[..., None] * x).sum(1)
        # In the stable arm weak outlier tokens must not bypass weighting via max.
        maximum_source = x if support is None else x * support[..., None]
        maximum = maximum_source.masked_fill(~safe[..., None], -torch.inf).amax(1)
        return torch.cat((mean, maximum), -1)
    aa, bb = pool(a), pool(b)
    symmetric = torch.cat((.5 * (aa + bb), (aa-bb).abs()), -1)
    logit = head.classifier(symmetric).squeeze(-1)
    present = valid.any(1)
    # Keep unused fallback parameter gradients absent, like the original head.
    result = logit if bool(present.all()) else torch.where(present, logit, head.no_evidence_logit)
    return result, symmetric


def physical_step(points, valid):
    """Mean original contour chord spacing in px; no label or GT geometry."""
    n = valid.sum(1).clamp_min(1)
    ids = torch.arange(points.shape[1], device=points.device)[None].expand(len(points), -1)
    nxt = (ids+1) % n[:, None]
    bi = torch.arange(len(points), device=points.device)[:, None]
    distances = (points[bi, nxt]-points).norm(dim=-1)
    return torch.where(valid, distances, 0.).sum(1) / n


class LocalEvidenceScorer(FreshMatchedScorer):
    def __init__(self, config):
        if config.cap not in (128, 256, 512) or config.depth != 2 or config.heads not in (4, 8):
            raise ValueError("registered two-layer/4-or-8-head/cap128-256-512 configuration required")
        super().__init__("matched_edges", config.dim, config.heads)
        self.config_v2 = config
        self.graph = None
        if config.graph == "pairing":
            from staging.pairwise_v0_2.baselines.rachel_pairingnet_benchmark import _ResGCN
            self.graph = _ResGCN(config.dim, layers=14, radius=8)
        elif config.graph == "shredding":
            from staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark import ReleasedDeepResGCN
            self.graph = ReleasedDeepResGCN(SimpleNamespace(feature_dim=config.dim, resgcn_blocks=14))
        elif config.graph != "none":
            raise ValueError("unknown graph")
        if config.joint_d:
            self.candidate_head = nn.Sequential(nn.Linear(4*config.dim, 64), nn.SiLU(), nn.Linear(64, 1))
        if config.stable:
            self.coordinate_projection = nn.Linear(2, config.dim, bias=False)

    def graph_context(self, tokens, valid):
        if self.graph is None:
            return tokens
        tokens = torch.where(valid[..., None], tokens, 0.)
        if self.config_v2.graph == "pairing":
            return self.graph(tokens, valid)
        from staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark import cycle_edge_index
        edges = cycle_edge_index(valid, 8)
        compact = self.graph(tokens[valid], edges)
        out = torch.zeros_like(tokens)
        return out.index_put((valid,), compact)

    def forward(self, tokens_a, tokens_b, valid_a, valid_b, selection=None, *,
                candidate_weights=None, points_a_rc=None, points_b_rc=None, **unused):
        if unused:
            raise TypeError("unexpected forward fields; labels and GT are not network inputs")
        _features(tokens_a, tokens_b, valid_a, valid_b, self.feature_dim)
        active = validate_selection(selection, valid_a, valid_b)
        cfg, s = self.config_v2, selection
        if candidate_weights is None or points_a_rc is None or points_b_rc is None:
            raise ValueError("original correspondence weights and points required")
        bi = torch.arange(len(tokens_a), device=tokens_a.device)[:, None]
        ij = s.candidate_indices.clamp_min(0)
        pa, pb = points_a_rc[bi, ij[..., 0]], points_b_rc[bi, ij[..., 1]]
        translation = torch.nan_to_num(s.translation_a_to_b_rc)
        residual = (pb-pa-translation[:, None]).norm(dim=-1)
        support = None
        if cfg.stable:
            active = s.candidate_valid & s.layout_valid[:, None]
            scale = torch.maximum(physical_step(points_a_rc, valid_a), physical_step(points_b_rc, valid_b))
            scale = (3*scale).clamp(min=10., max=40.)
            normalized = torch.asinh(residual / scale[:, None])
            support = candidate_weights / candidate_weights.amax(1, keepdim=True).clamp_min(1e-8)
            support = support / (1 + (residual/scale[:, None]).square())
            support = torch.where(active, support.clamp_min(1e-8), 0.)
            ranking = support
        else:
            normalized = residual / 10.
            ranking = candidate_weights
        ranking = torch.where(active, ranking, torch.full_like(ranking, -torch.inf))
        # For the uncapped reference, preserve candidate order exactly. For
        # smaller budgets, select by raw Q only, with stable original-index ties.
        n = min(cfg.cap, ij.shape[1])
        order = (torch.arange(ij.shape[1], device=ij.device)[None].expand(len(ij), -1)
                 if n == ij.shape[1] else torch.argsort(ranking, dim=1, descending=True, stable=True)[:, :n])
        selected = active[bi, order]
        picked = ij[bi, order]
        a = self.graph_context(tokens_a, valid_a)
        b = self.graph_context(tokens_b, valid_b)
        ea, eb = a[bi, picked[..., 0]], b[bi, picked[..., 1]]
        ea, eb = torch.where(selected[..., None], ea, 0.), torch.where(selected[..., None], eb, 0.)
        meta = torch.stack((candidate_weights, normalized), -1)[bi, order]
        meta = torch.where(selected[..., None], meta, 0.)
        common = self.edge_projection(meta)
        a = self.self_projection(ea) + self.mate_projection(eb) + common
        b = self.self_projection(eb) + self.mate_projection(ea) + common
        if cfg.stable:
            def coordinates(p):
                p = p[bi, order]
                w = selected[..., None].to(p.dtype)
                center = (p*w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
                return torch.asinh((p-center)/scale[:, None, None])
            a = a + self.coordinate_projection(coordinates(pa))
            b = b + self.coordinate_projection(coordinates(pb))
            support = support[bi, order]
        logit, pooled = batched_readout(self.head, a, b, selected, support)
        candidate_logit = self.candidate_head(pooled).squeeze(-1) if cfg.joint_d else None
        return SimpleNamespace(logit=logit, candidate_logit=candidate_logit,
            has_raw_candidates=s.candidate_valid.any(1), has_decoded_candidate=s.layout_valid,
            used_fallback=~selected.any(1), selected_token_count_a=s.mask_a.sum(1),
            selected_token_count_b=s.mask_b.sum(1), inlier_edge_count=active.sum(1),
            used_edge_count=selected.sum(1), reasons=s.reasons)

    def metadata(self):
        return dict(schema="local-evidence-v2/1", configuration=asdict(self.config_v2),
            parameter_count=sum(p.numel() for p in self.parameters()), initialization_seed=self.initialization_seed,
            matcher_updated=False, scorer_only_graph=True, graph_layers=14 if self.graph else 0,
            graph_radius=8 if self.graph else 0, original_feature_context_retained=True,
            evidence="matched endpoint pairs + raw Sinkhorn Q + predicted displacement residual",
            cap_unit="correspondence records: at most cap Patch endpoints per side, repetitions possible",
            selection="target-blind predicted final candidate; cap Top-Q; stable arm soft same-mode support",
            pair_loss="PairBCE", candidate_loss="auxiliary BCE weight0.5" if self.config_v2.joint_d else None,
            candidate_output_role="training auxiliary and diagnostic, not a hard pair-score veto",
            unchanged_layout=True, global_coarse_score_used=False,
            stable_note="normalizes Scorer geometry only, not frozen upstream Patch/context scale")


def make(name, seed=260914):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LocalEvidenceScorer(CONFIGS[name])
        model.initialization_seed = seed
        return model
