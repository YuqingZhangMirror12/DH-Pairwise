"""Fresh, scorer-only CA arms; no training/launch/checkpoint side effects.

Cache each pair's frozen post-Matcher features, valid masks, original contour
points and every CandidateSelection field. For matched_edges additionally cache
candidate_weights[E] = Q[i,j] in EXACT candidate_indices order (zero padding).
No full Q is necessary. training_valid/decision_valid and labels belong to the
trainer, not this selector or head. Persist decoder/config/source hashes outside
this module: structural validation cannot establish an arbitrary cache's origin.

The edge arm explicitly binds both endpoint features before set attention;
merely collecting two aligned lists would discard their pair relation at pool.
Selected features still contain global frozen-Matcher context. These are NOT
physical crop/mask ablations and predicted inliers are NOT ground-truth seams.
"""
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.model import (
    CandidateSelection, DECODER_CONFIG, select_predicted_inliers,
)

SCHEMA = "fresh-matched-only-ca/1"
ARMS = ("all_tokens", "matched_tokens", "matched_edges")


@dataclass(frozen=True)
class ScoreOutput:
    logit: Tensor
    has_raw_candidates: Tensor
    has_decoded_candidate: Tensor
    used_fallback: Tensor
    selected_token_count_a: Tensor
    selected_token_count_b: Tensor
    inlier_edge_count: Tensor
    reasons: tuple


def _features(a, b, va, vb, dim):
    if (a.ndim != 3 or b.ndim != 3 or not len(a) or len(a) != len(b)
            or a.shape[-1] != dim or b.shape[-1] != dim
            or min(a.shape[1], b.shape[1]) < 1 or not a.is_floating_point()
            or not b.is_floating_point() or a.dtype != b.dtype
            or va.dtype != torch.bool or vb.dtype != torch.bool
            or va.shape != a.shape[:2] or vb.shape != b.shape[:2]
            or len({v.device for v in (a, b, va, vb)}) != 1):
        raise ValueError("expected same-device float tokens[B,N,D] and bool valid[B,N]")
    if not torch.isfinite(a[va]).all() or not torch.isfinite(b[vb]).all():
        raise ValueError("nonfinite valid frozen feature")


def validate_selection(selection, valid_a, valid_b):
    """Indices always refer to original padded token axes, never compact axes."""
    if not isinstance(selection, CandidateSelection):
        raise TypeError("requires production CandidateSelection, not a label/GT mask")
    s, batch = selection, len(valid_a)
    tensors = (s.mask_a, s.mask_b, s.layout_valid, s.translation_a_to_b_rc,
               s.candidate_indices, s.candidate_valid, s.candidate_inliers)
    if any(not isinstance(v, Tensor) or v.device != valid_a.device for v in tensors):
        raise ValueError("selection tensors must share feature device")
    if (s.mask_a.dtype != torch.bool or s.mask_b.dtype != torch.bool
            or s.mask_a.shape != valid_a.shape or s.mask_b.shape != valid_b.shape
            or s.layout_valid.dtype != torch.bool or s.layout_valid.shape != (batch,)
            or s.translation_a_to_b_rc.shape != (batch, 2)
            or not s.translation_a_to_b_rc.is_floating_point()
            or s.candidate_indices.dtype != torch.long or s.candidate_indices.ndim != 3
            or s.candidate_indices.shape[0] != batch or s.candidate_indices.shape[-1] != 2
            or not 1 <= s.candidate_indices.shape[1] <= DECODER_CONFIG.max_candidates
            or s.candidate_valid.shape != s.candidate_indices.shape[:2]
            or s.candidate_inliers.shape != s.candidate_valid.shape
            or s.candidate_valid.dtype != torch.bool or s.candidate_inliers.dtype != torch.bool
            or not isinstance(s.reasons, tuple) or len(s.reasons) != batch
            or any(not isinstance(r, str) for r in s.reasons)):
        raise ValueError("selection schema/shape/dtype mismatch")
    if (s.candidate_inliers & ~s.candidate_valid).any():
        raise ValueError("inlier is not a present candidate")
    if (s.candidate_indices[~s.candidate_valid] != -1).any():
        raise ValueError("padded candidate indices must be -1")
    indices = s.candidate_indices[s.candidate_valid]
    if (indices < 0).any() or (indices[:, 0] >= valid_a.shape[1]).any() or (indices[:, 1] >= valid_b.shape[1]).any():
        raise ValueError("candidate index outside original token axes")
    batch_index = torch.arange(batch, device=valid_a.device)[:, None].expand_as(s.candidate_valid)
    bi = batch_index[s.candidate_valid]
    if not valid_a[bi, indices[:, 0]].all() or not valid_b[bi, indices[:, 1]].all():
        raise ValueError("candidate includes padding/invalid token")
    flat_ids = (bi * valid_a.shape[1] + indices[:, 0]) * valid_b.shape[1] + indices[:, 1]
    if torch.unique(flat_ids).numel() != flat_ids.numel():
        raise ValueError("duplicate candidate edge; production union is unique")
    if not torch.isfinite(s.translation_a_to_b_rc[s.layout_valid]).all():
        raise ValueError("valid layout requires finite predicted translation")
    active = s.candidate_inliers & s.layout_valid[:, None]
    if (s.layout_valid & (active.sum(1) < DECODER_CONFIG.min_inliers)).any():
        raise ValueError("valid production layout lacks minimum inlier count")
    expected_a, expected_b = torch.zeros_like(valid_a), torch.zeros_like(valid_b)
    endpoints = s.candidate_indices[active]
    active_batch = batch_index[active]
    expected_a[active_batch, endpoints[:, 0]] = True
    expected_b[active_batch, endpoints[:, 1]] = True
    if not torch.equal(s.mask_a, expected_a) or not torch.equal(s.mask_b, expected_b):
        raise ValueError("selected masks must equal unique final-peak inlier endpoints")
    return active


@torch.no_grad()
def edge_metadata(selection, valid_a, valid_b, *, candidate_weights=None,
                  points_a_rc=None, points_b_rc=None, assignment=None,
                  edge_residual_norm=None):
    """Return detached [B,E,2] = [raw Qij, ||pb-pa-t|| / 10px].

    Cached residuals may replace points, but must be aligned to candidate order
    and already divided by the production radius. Cache provenance is external.
    Neither labels, GT offsets nor classification/layout-success flags enter.
    """
    active = validate_selection(selection, valid_a, valid_b)
    s, present = selection, selection.candidate_valid
    safe = s.candidate_indices.clamp_min(0)
    bi = torch.arange(len(valid_a), device=valid_a.device)[:, None]
    if candidate_weights is not None and (not isinstance(candidate_weights, Tensor)
            or candidate_weights.shape != present.shape or candidate_weights.device != valid_a.device
            or not candidate_weights.is_floating_point()):
        raise ValueError("candidate_weights shape/device/dtype differ from candidate order")
    if assignment is not None:
        if (assignment.shape != (len(valid_a), valid_a.shape[1], valid_b.shape[1])
                or assignment.device != valid_a.device or not assignment.is_floating_point()):
            raise ValueError("assignment dimensions/device differ from original token axes")
        derived = assignment.detach()[bi, safe[..., 0], safe[..., 1]]
        derived = torch.where(present, derived, torch.zeros_like(derived))
        if candidate_weights is None:
            candidate_weights = derived
        elif not torch.allclose(candidate_weights[present], derived[present], atol=1e-7, rtol=1e-5):
            raise ValueError("cached candidate weights disagree with supplied Q")
    if (candidate_weights is None or candidate_weights.shape != present.shape
            or candidate_weights.device != valid_a.device or not candidate_weights.is_floating_point()):
        raise ValueError("matched_edges requires aligned candidate_weights or full assignment")
    if not torch.isfinite(candidate_weights[present]).all() or (candidate_weights[present] < 0).any():
        raise ValueError("candidate weights must be finite/nonnegative")
    if edge_residual_norm is None:
        if (points_a_rc is None or points_b_rc is None
                or points_a_rc.shape != (*valid_a.shape, 2) or points_b_rc.shape != (*valid_b.shape, 2)
                or any(not p.is_floating_point() or p.device != valid_a.device for p in (points_a_rc, points_b_rc))
                or not torch.isfinite(points_a_rc[valid_a]).all()
                or not torch.isfinite(points_b_rc[valid_b]).all()):
            raise ValueError("matched_edges requires original finite points or aligned normalized residuals")
        delta = (points_b_rc[bi, safe[..., 1]] - points_a_rc[bi, safe[..., 0]]
                 - s.translation_a_to_b_rc[:, None])
        edge_residual_norm = torch.linalg.vector_norm(delta, dim=-1) / DECODER_CONFIG.inlier_radius_px
    if (edge_residual_norm.shape != present.shape or edge_residual_norm.device != valid_a.device
            or not edge_residual_norm.is_floating_point()
            or not torch.isfinite(edge_residual_norm[active]).all()
            or (edge_residual_norm[active] < 0).any()):
        raise ValueError("invalid normalized predicted residual")
    # Invalid/no-layout rows are no evidence, never filled with invented signal.
    weights = torch.where(active, candidate_weights, torch.zeros_like(candidate_weights))
    residuals = torch.where(active, edge_residual_norm, torch.zeros_like(edge_residual_norm))
    return torch.stack((weights, residuals), dim=-1).detach()


class FreshMatchedScorer(nn.Module):
    def __init__(self, arm, feature_dim=96, num_heads=4):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown fresh matched-only arm")
        self.arm, self.feature_dim = arm, feature_dim
        self.initialization_seed = None
        # Construct first: make_fresh_scorer shares these exact initial tensors.
        self.head = CrossAttentionPairHead(feature_dim, num_heads, depth=2)
        if arm == "matched_edges":
            self.self_projection = nn.Linear(feature_dim, feature_dim, bias=False)
            self.mate_projection = nn.Linear(feature_dim, feature_dim, bias=False)
            self.edge_projection = nn.Linear(2, feature_dim)

    def forward(self, tokens_a, tokens_b, valid_a, valid_b, selection=None, *,
                candidate_weights=None, points_a_rc=None, points_b_rc=None,
                assignment=None, edge_residual_norm=None):
        _features(tokens_a, tokens_b, valid_a, valid_b, self.feature_dim)
        if selection is None:
            if assignment is None or points_a_rc is None or points_b_rc is None:
                raise ValueError("supply CandidateSelection or inputs to production selector")
            selection = select_predicted_inliers(assignment, points_a_rc, points_b_rc, valid_a, valid_b)
        active = validate_selection(selection, valid_a, valid_b)
        a, b = tokens_a, tokens_b
        if self.arm == "all_tokens":
            ma, mb = valid_a, valid_b
        elif self.arm == "matched_tokens":
            ma, mb = selection.mask_a, selection.mask_b
        else:
            meta = edge_metadata(selection, valid_a, valid_b, candidate_weights=candidate_weights,
                points_a_rc=points_a_rc, points_b_rc=points_b_rc, assignment=assignment,
                edge_residual_norm=edge_residual_norm).to(tokens_a.dtype)
            safe = selection.candidate_indices.clamp_min(0)
            bi = torch.arange(len(a), device=a.device)[:, None]
            ea, eb = a[bi, safe[..., 0]], b[bi, safe[..., 1]]
            # Avoid unused padding/invalid-layout NaNs reaching projections.
            ea = torch.where(active[..., None], ea, torch.zeros_like(ea))
            eb = torch.where(active[..., None], eb, torch.zeros_like(eb))
            common = self.edge_projection(meta)
            a = self.self_projection(ea) + self.mate_projection(eb) + common
            b = self.self_projection(eb) + self.mate_projection(ea) + common
            ma = mb = active
        logit = self.head(a, b, ma, mb)
        return ScoreOutput(logit=logit, has_raw_candidates=selection.candidate_valid.any(1),
            has_decoded_candidate=selection.layout_valid,
            used_fallback=~(ma.any(1) & mb.any(1)),
            selected_token_count_a=selection.mask_a.sum(1), selected_token_count_b=selection.mask_b.sum(1),
            inlier_edge_count=active.sum(1), reasons=selection.reasons)

    def metadata(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return dict(schema_version=SCHEMA, arm=self.arm, feature_dim=self.feature_dim,
            initialization_seed=self.initialization_seed,
            heads=self.head.cross_attention.num_heads, depth=2,
            parameters=dict(total=total, shared_ca_pool_classifier=head, additional=total-head),
            initialization="fresh head, not C8/C16 copied; same seeded common head across arms",
            selector="production final predicted layout inliers; target-blind for positives AND negatives",
            decoder=asdict(DECODER_CONFIG), global_rescue=False,
            fallback="learnable no_evidence_logit; explicit used_fallback, not forced negative/positive",
            edge_features="Wself*fAi+Wmate*fBj+Wmeta[raw Qij, norm(pb-pa-t_pred)/10px]; symmetric swapped role",
            edge_caveat="extra parameters, repeated endpoints per edge, Q and residual information; not a capacity-matched token-mask-only contrast",
            context_caveat="selected tokens already contain frozen Matcher global context; not local-only raw image input",
            cache_required=["tokens_a/b", "valid_a/b", "all CandidateSelection fields",
                            "candidate_weights plus points_a/b or edge_residual_norm for edge arm"],
            cache_external=["Matcher/checkpoint/decoder/code hashes", "training_valid", "decision_valid", "labels used only by trainer"],
            no_supervision_added="PairBCE owned by trainer; no layout/correspondence loss here")


def make_fresh_scorer(arm, *, seed=260920, feature_dim=96, num_heads=4):
    """CPU construction with identical common initialization, preserves CPU RNG.

    Caller moves model to device, creates a fresh optimizer and handles training
    RNG explicitly. No checkpoint, data, CUDA initialization or remote activity.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = FreshMatchedScorer(arm, feature_dim, num_heads)
        model.initialization_seed = seed
        return model
