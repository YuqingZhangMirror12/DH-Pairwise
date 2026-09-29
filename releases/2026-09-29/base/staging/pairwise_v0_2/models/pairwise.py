"""End-to-end, mask-only Pairwise v0.2 model skeleton.

Relative manuscript orientation is known upstream, but the facing side need
not be known at inference time.  ``forward_candidates`` therefore scores all
geometry-proposed side candidates and aggregates them as a multiple-instance
pair; it never consumes a ground-truth direction as an input feature.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .coarse import CoarseOutput, SymmetricCoarseSiamese
from .local_matcher import (
    DEFAULT_MATCHER_TEMPERATURE,
    DEFAULT_SINKHORN_ITERATIONS,
    LocalMatcherOutput,
    MatcherMode,
    OrderedLocalMatcher,
    SinkhornTrainingPolicy,
)


DEFAULT_DIRECTIONS = (
    "b_left_of_a",
    "b_right_of_a",
    "b_above_a",
    "b_below_a",
)


class ArcPoolingMode(str, Enum):
    """Frozen arc-within-direction pooling ablations.

    No mode is declared the winner before source-disjoint validation.  The
    direction level remains a separate four-slot MIL aggregation.
    """

    LOG_MEAN_EXP = "log_mean_exp"
    MAX = "max"
    TOP_K_MEAN = "top_k_mean"


class PairwiseScoreSource(str, Enum):
    """Evidence used for candidate, direction, and pair-level aggregation.

    ``FUSED`` preserves the original v0.2 behavior.  ``LOCAL`` is an explicit
    ablation boundary: the coarse network and fusion head are not executed,
    and candidate validity comes directly from the local matcher.
    """

    FUSED = "fused"
    LOCAL = "local"


def _score_source(value: object) -> PairwiseScoreSource:
    try:
        return PairwiseScoreSource(getattr(value, "value", value))
    except (TypeError, ValueError) as exc:
        raise ValueError("score_source must be 'fused' or 'local'") from exc


@dataclass(frozen=True)
class ArcPoolingConfig:
    """Configuration for pooling multiple contour arcs within one direction."""

    mode: ArcPoolingMode = ArcPoolingMode.LOG_MEAN_EXP
    temperature: float = 0.25
    top_k: int = 3

    def __post_init__(self) -> None:
        try:
            mode = ArcPoolingMode(getattr(self.mode, "value", self.mode))
        except ValueError as exc:
            raise ValueError("unsupported arc pooling mode") from exc
        object.__setattr__(self, "mode", mode)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("arc pooling temperature must be finite and positive")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("arc pooling top_k must be an integer")
        if self.top_k <= 0:
            raise ValueError("arc pooling top_k must be positive")


@dataclass(frozen=True)
class CoarseModelConfig:
    input_channels: int = 1
    widths: Tuple[int, ...] = (16, 32, 64)
    embedding_dim: int = 96
    hidden_dim: int = 96


@dataclass(frozen=True)
class LocalModelConfig:
    input_channels: int = 3
    feature_dim: int = 96
    num_heads: int = 4
    ff_dim: int = 192
    matcher_mode: str = MatcherMode.DUSTBIN_SINKHORN.value
    matcher_temperature: float = DEFAULT_MATCHER_TEMPERATURE
    sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS
    sinkhorn_tolerance: float = 1e-3
    require_sinkhorn_convergence: bool = True
    sinkhorn_training_policy: str = SinkhornTrainingPolicy.FINITE_WITH_RESIDUAL.value
    dropout: float = 0.0


@dataclass(frozen=True)
class PairwiseModelConfig:
    """Serializable architecture configuration used for checkpoint hashing."""

    coarse: CoarseModelConfig = field(default_factory=CoarseModelConfig)
    local: LocalModelConfig = field(default_factory=LocalModelConfig)
    fusion_hidden_dim: int = 32
    direction_names: Tuple[str, ...] = DEFAULT_DIRECTIONS
    arc_pooling: ArcPoolingConfig = field(default_factory=ArcPoolingConfig)
    direction_aggregation_temperature: float = 0.25
    schema_version: str = "dunhuang-pairwise-model/0.2"

    def __post_init__(self) -> None:
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be positive")
        if len(self.direction_names) != 4 or len(set(self.direction_names)) != 4:
            raise ValueError("direction_names must contain four unique candidates")
        if not isinstance(self.arc_pooling, ArcPoolingConfig):
            raise TypeError("arc_pooling must be ArcPoolingConfig")
        if (
            not math.isfinite(self.direction_aggregation_temperature)
            or self.direction_aggregation_temperature <= 0.0
        ):
            raise ValueError("direction aggregation temperature must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseOutput:
    """Coarse, local and fused evidence for one candidate side."""

    coarse_logit: Tensor
    coarse_probability: Tensor
    local_logit: Tensor
    local_probability: Tensor
    fused_logit: Tensor
    fused_probability: Tensor
    coarse_valid: Tensor
    local_training_valid: Tensor
    local_valid: Tensor
    training_valid: Tensor
    decision_valid: Tensor
    requires_review: Tensor
    coarse: CoarseOutput
    local: LocalMatcherOutput
    score_source: str = PairwiseScoreSource.FUSED.value


@dataclass(frozen=True)
class DirectionalPairwiseOutput:
    """Four side-candidate scores plus a pair-level MIL aggregation."""

    candidate_logits: Tensor
    candidate_probabilities: Tensor
    candidate_valid: Tensor
    pair_logit: Tensor
    pair_probability: Tensor
    pair_valid: Tensor
    direction_probabilities: Tensor
    best_direction_index: Tensor
    best_direction_name: Tuple[Optional[str], ...]
    direction_names: Tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalDirectionalOutput:
    """Ragged arc candidates aggregated within direction, then across sides."""

    arc_logits: Tensor
    arc_probabilities: Tensor
    arc_valid: Tensor
    arc_training_valid: Tensor
    sample_index: Tensor
    direction_index: Tensor
    direction_output: DirectionalPairwiseOutput
    training_direction_output: DirectionalPairwiseOutput
    arc_pooling: ArcPoolingConfig
    coarse_output: Optional[CoarseOutput] = None
    arc_pairwise_output: Optional[PairwiseOutput] = None
    score_source: Optional[str] = None


def aggregate_direction_candidates(
    candidate_logits: Tensor,
    candidate_valid: Tensor,
    *,
    direction_names: Sequence[str] = DEFAULT_DIRECTIONS,
    temperature: float = 0.25,
) -> DirectionalPairwiseOutput:
    """Aggregate four candidate-side logits without a direction oracle.

    A temperature-controlled log-mean-exp is a smooth MIL approximation to
    max.  Invalid candidates do not participate; a sample with none returns a
    neutral score and ``pair_valid=False`` so decision code must abstain.
    """

    if not isinstance(candidate_logits, Tensor) or candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B, K]")
    if not candidate_logits.is_floating_point():
        raise TypeError("candidate_logits must use a floating-point dtype")
    if not isinstance(candidate_valid, Tensor) or candidate_valid.dtype != torch.bool:
        raise TypeError("candidate_valid must be a bool torch.Tensor")
    if tuple(candidate_valid.shape) != tuple(candidate_logits.shape):
        raise ValueError("candidate_valid shape differs from candidate_logits")
    names = tuple(str(name) for name in direction_names)
    if len(names) != candidate_logits.shape[1] or len(set(names)) != len(names):
        raise ValueError("direction_names must uniquely name all candidates")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    finite = torch.isfinite(candidate_logits)
    valid = candidate_valid & finite
    pair_valid = valid.any(dim=1)
    safe_logits = torch.where(
        valid, candidate_logits, torch.full_like(candidate_logits, -1e4)
    )
    valid_count = valid.sum(dim=1).to(candidate_logits.dtype).clamp_min(1.0)
    raw_pair = temperature * (
        torch.logsumexp(safe_logits / temperature, dim=1) - torch.log(valid_count)
    )
    pair_logit = torch.where(pair_valid, raw_pair, torch.zeros_like(raw_pair))
    direction_logits = torch.where(
        valid, candidate_logits, torch.full_like(candidate_logits, -1e4)
    )
    direction_probabilities = torch.softmax(direction_logits, dim=1) * valid
    direction_probabilities = torch.where(
        pair_valid[:, None],
        direction_probabilities,
        torch.zeros_like(direction_probabilities),
    )
    best = direction_logits.argmax(dim=1)
    best = torch.where(pair_valid, best, torch.full_like(best, -1))
    best_names = tuple(
        names[index] if index >= 0 else None for index in best.detach().cpu().tolist()
    )
    safe_prob_logits = torch.where(
        valid, candidate_logits, torch.zeros_like(candidate_logits)
    )
    return DirectionalPairwiseOutput(
        candidate_logits=candidate_logits,
        candidate_probabilities=torch.sigmoid(safe_prob_logits),
        candidate_valid=valid,
        pair_logit=pair_logit,
        pair_probability=torch.sigmoid(pair_logit),
        pair_valid=pair_valid,
        direction_probabilities=direction_probabilities,
        best_direction_index=best,
        best_direction_name=best_names,
        direction_names=names,
    )


def _pool_arc_values(values: Tensor, config: ArcPoolingConfig) -> Tensor:
    """Pool one non-empty direction cell under a frozen ablation config."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("arc pooling requires a non-empty vector")
    if config.mode is ArcPoolingMode.LOG_MEAN_EXP:
        return config.temperature * (
            torch.logsumexp(values / config.temperature, dim=0)
            - math.log(values.numel())
        )
    if config.mode is ArcPoolingMode.MAX:
        return values.max()
    top_count = min(config.top_k, int(values.numel()))
    return torch.topk(values, top_count, sorted=False).values.mean()


def _aggregate_arc_grid(
    arc_logits: Tensor,
    finite_valid: Tensor,
    sample_index: Tensor,
    direction_index: Tensor,
    *,
    batch_size: int,
    direction_names: Tuple[str, ...],
    arc_pooling: ArcPoolingConfig,
    direction_temperature: float,
) -> DirectionalPairwiseOutput:
    """Pool ragged arcs within each direction, then aggregate four directions."""

    rows = []
    row_valid = []
    for sample in range(batch_size):
        cells = []
        valid_cells = []
        for direction in range(len(direction_names)):
            selected = (
                finite_valid & (sample_index == sample) & (direction_index == direction)
            )
            if selected.any().item():
                cells.append(_pool_arc_values(arc_logits[selected], arc_pooling))
                valid_cells.append(True)
            else:
                cells.append(arc_logits.sum() * 0.0)
                valid_cells.append(False)
        rows.append(torch.stack(cells))
        row_valid.append(valid_cells)
    direction_logits = torch.stack(rows)
    direction_valid = torch.tensor(
        row_valid, dtype=torch.bool, device=arc_logits.device
    )
    return aggregate_direction_candidates(
        direction_logits,
        direction_valid,
        direction_names=direction_names,
        temperature=direction_temperature,
    )


def aggregate_flat_arc_candidates(
    arc_logits: Tensor,
    arc_valid: Tensor,
    sample_index: Tensor,
    direction_index: Tensor,
    *,
    batch_size: int,
    direction_names: Sequence[str] = DEFAULT_DIRECTIONS,
    arc_pooling: Optional[ArcPoolingConfig] = None,
    direction_temperature: float = 0.25,
    training_arc_valid: Optional[Tensor] = None,
) -> HierarchicalDirectionalOutput:
    """Hierarchical MIL for an arbitrary number of geometry arc candidates.

    The geometry adapter supplies explicit integer indices; this function does
    not guess whether an external string such as ``left`` describes A or B.
    Multiple run-pairs/scales/arcs may map to the same sample and direction.
    """

    if not isinstance(arc_logits, Tensor) or arc_logits.ndim != 1:
        raise ValueError("arc_logits must have shape [N]")
    if not arc_logits.is_floating_point():
        raise TypeError("arc_logits must use a floating-point dtype")
    if not isinstance(arc_valid, Tensor) or arc_valid.dtype != torch.bool:
        raise TypeError("arc_valid must be a bool torch.Tensor")
    if tuple(arc_valid.shape) != tuple(arc_logits.shape):
        raise ValueError("arc_valid shape differs from arc_logits")
    for name, value in (
        ("sample_index", sample_index),
        ("direction_index", direction_index),
    ):
        if not isinstance(value, Tensor) or value.dtype != torch.long:
            raise TypeError("{} must be an int64 torch.Tensor".format(name))
        if tuple(value.shape) != tuple(arc_logits.shape):
            raise ValueError("{} shape differs from arc_logits".format(name))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    names = tuple(str(name) for name in direction_names)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("direction_names must contain four unique names")
    if ((sample_index < 0) | (sample_index >= batch_size)).any().item():
        raise ValueError("sample_index is out of range")
    if ((direction_index < 0) | (direction_index >= len(names))).any().item():
        raise ValueError("direction_index is out of range")
    pooling = arc_pooling or ArcPoolingConfig()
    if not isinstance(pooling, ArcPoolingConfig):
        raise TypeError("arc_pooling must be ArcPoolingConfig")
    if not math.isfinite(direction_temperature) or direction_temperature <= 0.0:
        raise ValueError("direction_temperature must be finite and positive")
    if training_arc_valid is None:
        training_arc_valid = arc_valid
    if (
        not isinstance(training_arc_valid, Tensor)
        or training_arc_valid.dtype != torch.bool
    ):
        raise TypeError("training_arc_valid must be a bool torch.Tensor")
    if tuple(training_arc_valid.shape) != tuple(arc_logits.shape):
        raise ValueError("training_arc_valid shape differs from arc_logits")
    finite_valid = arc_valid & torch.isfinite(arc_logits)
    finite_training_valid = training_arc_valid & torch.isfinite(arc_logits)
    if (finite_valid & ~finite_training_valid).any().item():
        raise ValueError("decision-valid arcs must also be training-valid")
    direction_output = _aggregate_arc_grid(
        arc_logits,
        finite_valid,
        sample_index,
        direction_index,
        batch_size=batch_size,
        direction_names=names,
        arc_pooling=pooling,
        direction_temperature=direction_temperature,
    )
    training_direction_output = _aggregate_arc_grid(
        arc_logits,
        finite_training_valid,
        sample_index,
        direction_index,
        batch_size=batch_size,
        direction_names=names,
        arc_pooling=pooling,
        direction_temperature=direction_temperature,
    )
    return HierarchicalDirectionalOutput(
        arc_logits=arc_logits,
        arc_probabilities=torch.sigmoid(
            torch.where(finite_valid, arc_logits, torch.zeros_like(arc_logits))
        ),
        arc_valid=finite_valid,
        arc_training_valid=finite_training_valid,
        sample_index=sample_index,
        direction_index=direction_index,
        direction_output=direction_output,
        training_direction_output=training_direction_output,
        arc_pooling=pooling,
    )


class DunhuangPairwiseV02(nn.Module):
    """Coarse Siamese plus ordered local assignment and learned fusion."""

    def __init__(self, config: Optional[PairwiseModelConfig] = None) -> None:
        super().__init__()
        self.config = config or PairwiseModelConfig()
        coarse = self.config.coarse
        local = self.config.local
        self.coarse_model = SymmetricCoarseSiamese(
            input_channels=coarse.input_channels,
            widths=coarse.widths,
            embedding_dim=coarse.embedding_dim,
            hidden_dim=coarse.hidden_dim,
        )
        self.local_model = OrderedLocalMatcher(
            input_channels=local.input_channels,
            feature_dim=local.feature_dim,
            num_heads=local.num_heads,
            ff_dim=local.ff_dim,
            matcher_mode=MatcherMode(local.matcher_mode),
            matcher_temperature=local.matcher_temperature,
            sinkhorn_iterations=local.sinkhorn_iterations,
            sinkhorn_tolerance=local.sinkhorn_tolerance,
            require_sinkhorn_convergence=local.require_sinkhorn_convergence,
            sinkhorn_training_policy=SinkhornTrainingPolicy(
                local.sinkhorn_training_policy
            ),
            dropout=local.dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(3, self.config.fusion_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.config.fusion_hidden_dim, 1),
        )

    def _fuse(self, coarse: CoarseOutput, local: LocalMatcherOutput) -> PairwiseOutput:
        features = torch.stack(
            [coarse.logit, local.logit, coarse.probability * local.probability],
            dim=1,
        )
        learned = self.fusion(features).squeeze(1)
        both_training = coarse.valid_problem & local.training_valid
        both_decision = coarse.valid_problem & local.decision_valid
        # A finite non-converged local plan stays gradient-bearing, but the
        # separate decision validity remains fail-closed for production.
        fallback = torch.where(
            coarse.valid_problem,
            coarse.logit,
            torch.where(local.training_valid, local.logit, torch.zeros_like(learned)),
        )
        fused = torch.where(both_training, learned, fallback)
        return PairwiseOutput(
            coarse_logit=coarse.logit,
            coarse_probability=coarse.probability,
            local_logit=local.logit,
            local_probability=local.probability,
            fused_logit=fused,
            fused_probability=torch.sigmoid(fused),
            coarse_valid=coarse.valid_problem,
            local_training_valid=local.training_valid,
            local_valid=local.decision_valid,
            training_valid=both_training,
            decision_valid=both_decision,
            requires_review=~both_decision,
            coarse=coarse,
            local=local,
            score_source=PairwiseScoreSource.FUSED.value,
        )

    @staticmethod
    def _local_only(local: LocalMatcherOutput) -> PairwiseOutput:
        """Adapt local evidence to the legacy pairwise-output loss contract.

        The neutral coarse fields are placeholders created from constants;
        they are not outputs of ``coarse_model``.  The legacy ``fused_*``
        aliases intentionally point at the selected local score so existing
        swap/residual helpers can operate without introducing fusion.
        """

        neutral_logit = torch.zeros_like(local.logit)
        neutral_probability = torch.full_like(local.probability, 0.5)
        coarse_valid = torch.zeros_like(local.training_valid)
        empty_embedding = local.logit.new_zeros((local.logit.shape[0], 0))
        neutral_coarse = CoarseOutput(
            logit=neutral_logit,
            probability=neutral_probability,
            embedding_a=empty_embedding,
            embedding_b=empty_embedding.clone(),
            valid_problem=coarse_valid,
        )
        return PairwiseOutput(
            coarse_logit=neutral_logit,
            coarse_probability=neutral_probability,
            local_logit=local.logit,
            local_probability=local.probability,
            fused_logit=local.logit,
            fused_probability=local.probability,
            coarse_valid=coarse_valid,
            local_training_valid=local.training_valid,
            local_valid=local.decision_valid,
            training_valid=local.training_valid,
            decision_valid=local.decision_valid,
            requires_review=~local.decision_valid,
            coarse=neutral_coarse,
            local=local,
            score_source=PairwiseScoreSource.LOCAL.value,
        )

    @staticmethod
    def _select_coarse(coarse: CoarseOutput, index: Tensor) -> CoarseOutput:
        return CoarseOutput(
            logit=coarse.logit.index_select(0, index),
            probability=coarse.probability.index_select(0, index),
            embedding_a=coarse.embedding_a.index_select(0, index),
            embedding_b=coarse.embedding_b.index_select(0, index),
            valid_problem=coarse.valid_problem.index_select(0, index),
        )

    def forward(
        self,
        coarse_a: Tensor,
        coarse_b: Tensor,
        local_a: Tensor,
        local_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        correspondence_mask: Optional[Tensor] = None,
    ) -> PairwiseOutput:
        coarse = self.coarse_model(coarse_a, coarse_b)
        local = self.local_model(
            local_a,
            local_b,
            token_mask_a,
            token_mask_b,
            correspondence_mask=correspondence_mask,
        )
        return self._fuse(coarse, local)

    def forward_candidates(
        self,
        coarse_a: Tensor,
        coarse_b: Tensor,
        local_a: Tensor,
        local_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        correspondence_mask: Optional[Tensor] = None,
        *,
        score_source: PairwiseScoreSource = PairwiseScoreSource.FUSED,
    ) -> DirectionalPairwiseOutput:
        """Score four side candidates supplied by the geometry layer.

        Local tensors have shape ``[B, K, L, C, H, W]`` and token masks have
        shape ``[B, K, L]``.  Ground-truth direction is deliberately absent.
        """

        source = _score_source(score_source)
        if local_a.ndim != 6 or local_b.ndim != 6:
            raise ValueError("candidate patches must have shape [B, K, L, C, H, W]")
        if local_a.shape[:2] != local_b.shape[:2]:
            raise ValueError("candidate A/B batch and candidate counts differ")
        batch, candidates = local_a.shape[:2]
        if candidates != len(self.config.direction_names):
            raise ValueError("candidate count differs from configured directions")
        if tuple(token_mask_a.shape[:2]) != (batch, candidates) or tuple(
            token_mask_b.shape[:2]
        ) != (batch, candidates):
            raise ValueError("candidate token-mask prefix shape is invalid")
        flat_local_a = local_a.reshape(batch * candidates, *local_a.shape[2:])
        flat_local_b = local_b.reshape(batch * candidates, *local_b.shape[2:])
        flat_mask_a = token_mask_a.reshape(batch * candidates, token_mask_a.shape[2])
        flat_mask_b = token_mask_b.reshape(batch * candidates, token_mask_b.shape[2])
        flat_correspondence = None
        if correspondence_mask is not None:
            expected = (
                batch,
                candidates,
                local_a.shape[2],
                local_b.shape[2],
            )
            if (
                not isinstance(correspondence_mask, Tensor)
                or correspondence_mask.dtype != torch.bool
            ):
                raise TypeError("correspondence_mask must be a bool torch.Tensor")
            if tuple(correspondence_mask.shape) != expected:
                raise ValueError(
                    "candidate correspondence_mask must have shape {}".format(expected)
                )
            flat_correspondence = correspondence_mask.reshape(
                batch * candidates, local_a.shape[2], local_b.shape[2]
            )
        if source is PairwiseScoreSource.LOCAL:
            local = self.local_model(
                flat_local_a,
                flat_local_b,
                flat_mask_a,
                flat_mask_b,
                correspondence_mask=flat_correspondence,
            )
            candidate_logits = local.logit
            candidate_valid = local.decision_valid
        else:
            repeat_shape = (batch * candidates,) + tuple(coarse_a.shape[1:])
            flat_coarse_a = (
                coarse_a[:, None]
                .expand(batch, candidates, *coarse_a.shape[1:])
                .reshape(repeat_shape)
            )
            flat_coarse_b = (
                coarse_b[:, None]
                .expand(batch, candidates, *coarse_b.shape[1:])
                .reshape(repeat_shape)
            )
            result = self.forward(
                flat_coarse_a,
                flat_coarse_b,
                flat_local_a,
                flat_local_b,
                flat_mask_a,
                flat_mask_b,
                flat_correspondence,
            )
            candidate_logits = result.fused_logit
            candidate_valid = result.decision_valid
        return aggregate_direction_candidates(
            candidate_logits.reshape(batch, candidates),
            candidate_valid.reshape(batch, candidates),
            direction_names=self.config.direction_names,
            temperature=self.config.direction_aggregation_temperature,
        )

    def forward_flat_candidates(
        self,
        coarse_a: Tensor,
        coarse_b: Tensor,
        local_a: Tensor,
        local_b: Tensor,
        token_mask_a: Tensor,
        token_mask_b: Tensor,
        sample_index: Tensor,
        direction_index: Tensor,
        candidate_valid: Optional[Tensor] = None,
        correspondence_mask: Optional[Tensor] = None,
        *,
        score_source: PairwiseScoreSource = PairwiseScoreSource.FUSED,
    ) -> HierarchicalDirectionalOutput:
        """Score a flattened ragged set of run/scale/arc-pair candidates."""

        source = _score_source(score_source)
        if sample_index.dtype != torch.long or direction_index.dtype != torch.long:
            raise TypeError("sample_index and direction_index must be int64")
        if (
            local_a.shape[0] != sample_index.numel()
            or local_b.shape[0] != sample_index.numel()
        ):
            raise ValueError("flattened local candidate count differs from indices")
        if candidate_valid is None:
            candidate_valid = torch.ones_like(sample_index, dtype=torch.bool)
        if candidate_valid.dtype != torch.bool or tuple(candidate_valid.shape) != tuple(
            sample_index.shape
        ):
            raise TypeError("candidate_valid must be bool [N]")
        if correspondence_mask is not None:
            expected = (local_a.shape[0], local_a.shape[1], local_b.shape[1])
            if (
                not isinstance(correspondence_mask, Tensor)
                or correspondence_mask.dtype != torch.bool
            ):
                raise TypeError("correspondence_mask must be a bool torch.Tensor")
            if tuple(correspondence_mask.shape) != expected:
                raise ValueError(
                    "flat correspondence_mask must have shape {}".format(expected)
                )
        batch = coarse_a.shape[0]
        if coarse_b.shape[0] != batch:
            raise ValueError("coarse A/B batch sizes differ")
        if ((sample_index < 0) | (sample_index >= batch)).any().item():
            raise ValueError("sample_index is out of range")
        if sample_index.numel() == 0:
            if source is PairwiseScoreSource.FUSED:
                coarse_result = self.coarse_model(coarse_a, coarse_b)
                arc_logits = coarse_result.logit.new_empty((0,))
            else:
                coarse_result = None
                arc_logits = local_a.new_empty((0,))
            arc_valid = torch.empty((0,), dtype=torch.bool, device=arc_logits.device)
            aggregated = aggregate_flat_arc_candidates(
                arc_logits,
                arc_valid,
                sample_index,
                direction_index,
                batch_size=batch,
                direction_names=self.config.direction_names,
                arc_pooling=self.config.arc_pooling,
                direction_temperature=self.config.direction_aggregation_temperature,
            )
            return replace(
                aggregated,
                coarse_output=coarse_result,
                score_source=source.value,
            )
        local_result = self.local_model(
            local_a,
            local_b,
            token_mask_a,
            token_mask_b,
            correspondence_mask=correspondence_mask,
        )
        if source is PairwiseScoreSource.LOCAL:
            coarse_result = None
            candidate_result = self._local_only(local_result)
        else:
            coarse_result = self.coarse_model(coarse_a, coarse_b)
            candidate_result = self._fuse(
                self._select_coarse(coarse_result, sample_index), local_result
            )
        aggregated = aggregate_flat_arc_candidates(
            (
                candidate_result.local_logit
                if source is PairwiseScoreSource.LOCAL
                else candidate_result.fused_logit
            ),
            candidate_result.decision_valid & candidate_valid,
            sample_index,
            direction_index,
            batch_size=batch,
            direction_names=self.config.direction_names,
            arc_pooling=self.config.arc_pooling,
            direction_temperature=self.config.direction_aggregation_temperature,
            training_arc_valid=candidate_result.training_valid & candidate_valid,
        )
        return replace(
            aggregated,
            coarse_output=coarse_result,
            arc_pairwise_output=candidate_result,
            score_source=source.value,
        )


__all__ = [
    "ArcPoolingConfig",
    "ArcPoolingMode",
    "CoarseModelConfig",
    "DEFAULT_DIRECTIONS",
    "DirectionalPairwiseOutput",
    "DunhuangPairwiseV02",
    "HierarchicalDirectionalOutput",
    "LocalModelConfig",
    "PairwiseModelConfig",
    "PairwiseOutput",
    "PairwiseScoreSource",
    "aggregate_direction_candidates",
    "aggregate_flat_arc_candidates",
]
