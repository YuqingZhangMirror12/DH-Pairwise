"""Isolated matcher-then-classifier experiments; no live model changes.

The trainer owns M12+C8, the loss profile, source checkpoint receipts and budget.
This wrapper owns freezing and inference only. Matrix CNN is ShreddingNet-like,
not an original-ShreddingNet reproduction: its input is our real Sinkhorn Q.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from typing import Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .rachel_n512 import RachelN512Config, RachelN512Pairwise

SCHEMA = "rachel-decoupled-score-model/1"
HEAD_KINDS = ("matrix_cnn", "cross_attention")
PHASES = ("matcher", "classifier")
MATRIX_HEAD_REVISIONS = ("legacy", "bn_relu_pool_v2", "per_pair_norm_v3")
DEFAULT_MATRIX_HEAD_REVISION = "bn_relu_pool_v2"
CROSS_ATTENTION_DEPTHS = (1, 2, 4)


def _cross_attention_depth(model_options):
    """The only registered classifier option; missing means original S4."""
    if model_options is None:
        return 1
    if not isinstance(model_options, dict) or set(model_options) - {"cross_attention_depth"}:
        raise ValueError("unregistered decoupled model_options")
    depth = model_options.get("cross_attention_depth", 1)
    if type(depth) is not int or depth not in CROSS_ATTENTION_DEPTHS:
        raise ValueError("cross_attention_depth must be one of 1, 2, 4")
    return depth


def antidiagonal_opening(matrix: Tensor) -> Tensor:
    """Official soft morphology: erode two neighbors, dilate including center.

    Both 3x3 kernels use constant-zero boundaries. In particular erosion does
    NOT include the center. This is performed before the hard > threshold.
    """
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("morphology requires a nonempty [Na,Nb] matrix")
    h, w = matrix.shape
    padded = F.pad(matrix, (1, 1, 1, 1), value=0.)
    eroded = torch.minimum(padded[:h, 2:w + 2], padded[2:h + 2, :w])
    padded = F.pad(eroded, (1, 1, 1, 1), value=0.)
    return torch.maximum(eroded, torch.maximum(padded[:h, 2:w + 2], padded[2:h + 2, :w]))


def _indices(valid: Tensor) -> Tensor:
    if valid.ndim != 1 or valid.dtype != torch.bool:
        raise ValueError("contour validity must be a one-dimensional bool mask")
    return torch.nonzero(valid, as_tuple=False).flatten()


class ThresholdedMatrixCNN(nn.Module):
    """Soft-Q opening -> binary matrix -> 1/32/64/128 CNN -> average -> logit."""

    def __init__(self, threshold: float = .006, revision: str = DEFAULT_MATRIX_HEAD_REVISION):
        super().__init__()
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("matrix threshold must be finite and nonnegative")
        if revision not in MATRIX_HEAD_REVISIONS:
            raise ValueError("unregistered matrix head revision")
        self.threshold = float(threshold)
        self.revision = revision
        # With per-pair B=1, ending in BN -> spatial mean erases input
        # dependence in training. V2 changes ONLY this final block ordering.
        # Keep legacy modules/keys readable for old C diagnostics and M12 reuse.
        track_running_stats = revision != "per_pair_norm_v3"
        final_norm, final_activation = nn.BatchNorm2d(128, track_running_stats=track_running_stats), nn.ReLU()
        final_layers = ((final_activation, final_norm) if revision == "legacy" else
                        (final_norm, final_activation))
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32, track_running_stats=track_running_stats), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64, track_running_stats=track_running_stats), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), *final_layers,
        )
        self.fc = nn.Linear(128, 1)
        self.no_evidence_logit = nn.Parameter(torch.zeros(()))

    def binary_matrix(self, q: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:
        """Crop valid rows/columns in their original cyclic order; no resize."""
        ia, ib = _indices(valid_a), _indices(valid_b)
        matrix = q.index_select(0, ia).index_select(1, ib)
        if matrix.numel() == 0:
            return matrix
        if not torch.isfinite(matrix).all():
            raise ValueError("non-finite Q in valid matrix cells")
        return (antidiagonal_opening(matrix) > self.threshold).to(matrix.dtype)

    def _one_orientation(self, binary: Tensor) -> Tensor:
        # Only unusually short contours need zero padding to support two pools
        # and training BN. Never shrink/rescale a real correspondence matrix.
        h, w = binary.shape
        binary = F.pad(binary, (0, max(0, 8 - w), 0, max(0, 8 - h)))
        features = self.cnn(binary[None, None])
        # Exactly adaptive average to 1x1, avoiding its nondeterministic CUDA
        # backward implementation under our strict deterministic trainer.
        return self.fc(features.mean(dim=(-2, -1))).reshape(())

    def forward(self, assignment: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:
        if (assignment.ndim != 3 or not len(assignment)
                or valid_a.shape != assignment.shape[:2]
                or valid_b.shape != (len(assignment), assignment.shape[2])):
            raise ValueError("Q and validity batch/contour dimensions differ")
        values = []
        for q, va, vb in zip(assignment, valid_a, valid_b):
            binary = self.binary_matrix(q, va, vb)
            if not binary.numel():
                values.append(self.no_evidence_logit)
            else:
                # Same weights and morphology under A/B transpose. Legacy/v2
                # use running stats at eval; v3 uses per-pair stats in both modes.
                values.append(.5 * (self._one_orientation(binary)
                                    + self._one_orientation(binary.transpose(0, 1))))
        return torch.stack(values)


class _CrossAttentionInteraction(nn.Module):
    """One extra decoder layer; its weights are shared between A<-B and B<-A."""

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(nn.Linear(feature_dim, 2 * feature_dim), nn.GELU(),
                                 nn.Linear(2 * feature_dim, feature_dim))

    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        q, s = query[None], source[None]
        update, _ = self.cross_attention(self.norm(q), self.norm(s), self.norm(s), need_weights=False)
        value = q + update
        return (value + self.ffn(self.output_norm(value)))[0]


class CrossAttentionPairHead(nn.Module):
    """Additional classification interaction AFTER the frozen base context.

    Both directions share each layer's attention/FFN; different layers have
    independent weights. Both updates at layer L read the layer L-1 tokens.
    Pool only after the final layer, with the original attentive sum and max.
    The depth-one module keys, initialization order and forward path are kept
    exactly as in the original S4, including the shared pooling/classifier.
    """

    def __init__(self, feature_dim: int, num_heads: int, depth: int = 1):
        super().__init__()
        self.depth = _cross_attention_depth({"cross_attention_depth": depth})
        self.norm = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(nn.Linear(feature_dim, 2 * feature_dim), nn.GELU(),
                                 nn.Linear(2 * feature_dim, feature_dim))
        self.pool_gate = nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(),
                                       nn.Linear(feature_dim // 2, 1))
        self.classifier = nn.Sequential(nn.Linear(4 * feature_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.no_evidence_logit = nn.Parameter(torch.zeros(()))
        # Construct extras LAST: common first-layer/pool/classifier weights have
        # the same initial values for depths 1/2/4 under a shared random seed.
        self.extra_layers = nn.ModuleList(
            [_CrossAttentionInteraction(feature_dim, num_heads) for _ in range(self.depth - 1)])

    def _interact(self, query: Tensor, source: Tensor) -> Tensor:
        # Inputs are compact valid tokens: padding cannot enter keys or pooling.
        q, s = query[None], source[None]
        update, _ = self.cross_attention(self.norm(q), self.norm(s), self.norm(s), need_weights=False)
        value = q + update
        return (value + self.ffn(self.output_norm(value)))[0]

    def _pool(self, value: Tensor) -> Tensor:
        weights = torch.softmax(self.pool_gate(value).squeeze(-1), dim=0)
        attentive = (weights[:, None] * value).sum(0)
        return torch.cat((attentive, value.amax(0)))

    def _interact_pool(self, query: Tensor, source: Tensor) -> Tensor:
        return self._pool(self._interact(query, source))

    def _decode_pair(self, a: Tensor, b: Tensor):
        # Tuple RHS is evaluated before either assignment: neither direction
        # accidentally consumes the other direction's just-updated tokens.
        a, b = self._interact(a, b), self._interact(b, a)
        for layer in self.extra_layers:
            a, b = layer(a, b), layer(b, a)
        return a, b

    def forward(self, features_a: Tensor, features_b: Tensor, valid_a: Tensor, valid_b: Tensor) -> Tensor:
        if (features_a.ndim != 3 or features_b.ndim != 3 or not len(features_a)
                or len(features_a) != len(features_b) or features_a.shape[-1] != features_b.shape[-1]
                or valid_a.shape != features_a.shape[:2] or valid_b.shape != features_b.shape[:2]):
            raise ValueError("descriptors and validity batch/contour dimensions differ")
        scores = []
        for a, b, va, vb in zip(features_a, features_b, valid_a, valid_b):
            a, b = a.index_select(0, _indices(va)), b.index_select(0, _indices(vb))
            if not len(a) or not len(b):
                scores.append(self.no_evidence_logit)
                continue
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("non-finite descriptor in a valid contour token")
            if self.depth == 1:
                aa, bb = self._interact_pool(a, b), self._interact_pool(b, a)
            else:
                a, b = self._decode_pair(a, b)
                aa, bb = self._pool(a), self._pool(b)
            symmetric = torch.cat((.5 * (aa + bb), torch.abs(aa - bb)))
            scores.append(self.classifier(symmetric).reshape(()))
        return torch.stack(scores)


class DecoupledScoreModel(nn.Module):
    def __init__(self, base: RachelN512Pairwise, head_kind: str = "matrix_cnn", matrix_threshold: float = .006,
                 matrix_head_revision=None, *, model_options=None):
        super().__init__()
        if not isinstance(base, RachelN512Pairwise) or head_kind not in HEAD_KINDS:
            raise ValueError("requires RachelN512Pairwise and a registered head kind")
        if isinstance(matrix_threshold, bool) or not math.isfinite(matrix_threshold) or matrix_threshold < 0:
            raise ValueError("matrix threshold must be finite and nonnegative")
        self.base_model, self.config = base, base.config
        self.head_kind, self.matrix_threshold = head_kind, float(matrix_threshold)
        depth = _cross_attention_depth(model_options)
        if head_kind != "cross_attention" and model_options:
            raise ValueError("model_options are only registered for cross_attention")
        self.model_options = ({"cross_attention_depth": depth} if depth != 1 else {})
        if head_kind == "matrix_cnn":
            self.matrix_head_revision = (DEFAULT_MATRIX_HEAD_REVISION if matrix_head_revision is None else
                                         matrix_head_revision)
            if self.matrix_head_revision not in MATRIX_HEAD_REVISIONS:
                raise ValueError("unregistered matrix head revision")
        else:
            if matrix_head_revision is not None:
                raise ValueError("cross-attention has no matrix head revision")
            self.matrix_head_revision = None
        self.score_head = (ThresholdedMatrixCNN(matrix_threshold, self.matrix_head_revision) if head_kind == "matrix_cnn" else
                           CrossAttentionPairHead(base.config.feature_dim, base.config.num_heads, depth=depth))
        self.phase = "matcher"
        self.set_phase("matcher")

    def set_phase(self, phase: str):
        if phase not in PHASES:
            raise ValueError("phase must be matcher or classifier")
        self.phase = phase
        for parameter in self.parameters():
            parameter.grad = None
        self.base_model.requires_grad_(phase == "matcher")
        # These branches have no optimization objective in the decoupled plan.
        for module in (self.base_model.coarse, self.base_model.local_head, self.base_model.fusion):
            module.requires_grad_(False)
        self.score_head.requires_grad_(phase == "classifier")
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.phase == "classifier":
            self.base_model.eval()
        for module in (self.base_model.coarse, self.base_model.local_head, self.base_model.fusion):
            module.eval()
        if self.phase == "matcher":
            self.score_head.eval()
        return self

    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        args = (mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b)
        if self.phase == "matcher":
            return self.base_model(*args)
        with torch.no_grad():
            original = self.base_model(*args)
        if self.head_kind == "matrix_cnn":
            score = self.score_head(original.assignment, contour_valid_a, contour_valid_b)
        else:
            score = self.score_head(original.token_features_a, original.token_features_b,
                                    contour_valid_a, contour_valid_b)
        finite = original.training_valid & torch.isfinite(score)
        score = torch.where(finite, score, torch.zeros_like(score))
        probability = score.sigmoid()
        return replace(original, fused_logit=score, local_logit=score,
                       fused_probability=probability, local_probability=probability,
                       training_valid=finite, decision_valid=finite & original.transport.diagnostics.converged)

    def metadata(self):
        metadata = dict(schema_version=SCHEMA, base_model_config=asdict(self.config),
                    head_kind=self.head_kind, matrix_threshold=self.matrix_threshold, phase=self.phase,
                    matrix_head_revision=self.matrix_head_revision,
                    coarse_role="diagnostic_only; never optimized by this protocol; expected random initialization",
                    final_score="single new Pair logit; no coarse fusion; no R; no layout reranking",
                    matrix_protocol=dict(input="real Sinkhorn Q, no dustbin; crop valid indices without resizing",
                        erosion="3x3 anti-diagonal endpoints, center excluded, constant zero boundary",
                        dilation="3x3 anti-diagonal including center, constant zero boundary",
                        threshold="strict > after soft morphology; project adaptation, not original ShreddingNet threshold calibration",
                        cnn=("1/32/64/128 Conv3+ReLU+BN; MaxPool2 after first two; spatial average; Linear1"
                             if self.matrix_head_revision not in ("bn_relu_pool_v2", "per_pair_norm_v3") else
                             "1/32/64/128; first two Conv3+ReLU+BN+MaxPool2; final Conv3+BN+ReLU; spatial average; Linear1"),
                        short_contour="zero-pad each dimension to minimum8 only to support CNN/BN",
                        symmetry="average shared CNN logits for matrix and transpose"),
                    descriptor_protocol="frozen base token_features AFTER existing context/cross-attention; additional shared bidirectional classifier attention; learned attentive pool+max",
                    loss_owner="trainer: M disables all Pair BCE; C only one new Pair BCE",
                    phases=list(PHASES))
        if self.matrix_head_revision == "per_pair_norm_v3":
            metadata["matrix_protocol"]["normalization"] = (
                "all three BatchNorm2d affine=True, track_running_stats=False; "
                "internal per-pair B=1 spatial statistics in both train and eval; "
                "v2 block order/eps/momentum unchanged; no cross-pair statistics")
        if self.model_options:
            metadata["model_options"] = dict(self.model_options)
            metadata["descriptor_protocol"] += (
                "; depth=%d independently parameterized layers, each with shared A/B direction weights; "
                "synchronous bidirectional updates from previous-layer tokens; pooling only after final layer"
                % self.model_options["cross_attention_depth"])
        return metadata


def build_decoupled_score_model(base_config: Union[dict, RachelN512Config], head_kind="matrix_cnn",
                                matrix_threshold=.006, phase="matcher", matrix_head_revision=None, *, model_options=None):
    if isinstance(base_config, dict):
        base_config = RachelN512Config(**base_config)
    return DecoupledScoreModel(RachelN512Pairwise(base_config), head_kind, matrix_threshold,
                               matrix_head_revision, model_options=model_options).set_phase(phase)


def matrix_revision_from_metadata(metadata):
    """Missing revision means ONLY the registered pre-fix model/1 semantics."""
    if metadata.get("head_kind") == "matrix_cnn":
        revision = metadata.get("matrix_head_revision", "legacy")
        if revision not in MATRIX_HEAD_REVISIONS:
            raise ValueError("unregistered matrix head revision")
        return revision
    if metadata.get("head_kind") != "cross_attention" or metadata.get("matrix_head_revision") is not None:
        raise ValueError("invalid head kind or non-matrix revision")
    return None


def convert_matrix_head_revision_state(state_dict, *, source_revision, target_revision):
    """Explicit head-only v2 -> v3 conversion, never invoked by a loader.

    Strictly validates every v2 head state key/shape/type and removes ONLY the
    nine BN running-statistic buffers. Every retained tensor is cloned with its
    value and dtype unchanged. Caller CPU RNG and the input state are preserved.
    The caller must bind new metadata/provenance and reselect on SIM VAL; old
    v2 eval scores/thresholds do not become v3 results through this conversion.
    """
    if (source_revision, target_revision) != ("bn_relu_pool_v2", "per_pair_norm_v3"):
        raise ValueError("only explicit bn_relu_pool_v2 -> per_pair_norm_v3 head conversion is registered")
    with torch.random.fork_rng(devices=[]):
        expected = ThresholdedMatrixCNN(revision="bn_relu_pool_v2").state_dict()
    if not isinstance(state_dict, dict) or set(state_dict) != set(expected):
        raise ValueError("source head state has missing or extra v2 keys")
    for name, reference in expected.items():
        value = state_dict[name]
        if (not isinstance(value, Tensor) or value.shape != reference.shape or
                value.is_floating_point() != reference.is_floating_point() or
                (not reference.is_floating_point() and value.dtype != reference.dtype)):
            raise ValueError("source head tensor shape/type differs: " + name)
    discarded = {"cnn.%d.%s" % (index, suffix) for index in (2, 6, 9)
                 for suffix in ("running_mean", "running_var", "num_batches_tracked")}
    return {name: value.detach().clone() for name, value in state_dict.items() if name not in discarded}


def load_decoupled_score_checkpoint(payload):
    """Strict independent loader: never interpret an old Full/S0-S2 payload."""
    if payload.get("decoupled_score_schema") != SCHEMA:
        raise ValueError("not a decoupled score checkpoint")
    metadata = payload["decoupled_score"]
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("decoupled metadata schema mismatch")
    revision = matrix_revision_from_metadata(metadata)
    model = build_decoupled_score_model(metadata["base_model_config"], metadata["head_kind"],
                                         metadata["matrix_threshold"], metadata["phase"], revision,
                                         model_options=metadata.get("model_options"))
    expected = model.metadata()
    if "matrix_head_revision" not in metadata:
        # No other missing or altered field is accepted as a legacy alias.
        expected.pop("matrix_head_revision")
    if json.dumps(expected, sort_keys=True) != json.dumps(metadata, sort_keys=True):
        raise ValueError("decoupled metadata differs from registered semantics")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


__all__ = ["SCHEMA", "HEAD_KINDS", "PHASES", "MATRIX_HEAD_REVISIONS", "DEFAULT_MATRIX_HEAD_REVISION",
           "CROSS_ATTENTION_DEPTHS",
           "matrix_revision_from_metadata", "convert_matrix_head_revision_state", "antidiagonal_opening", "ThresholdedMatrixCNN",
           "CrossAttentionPairHead", "DecoupledScoreModel", "build_decoupled_score_model",
           "load_decoupled_score_checkpoint"]
