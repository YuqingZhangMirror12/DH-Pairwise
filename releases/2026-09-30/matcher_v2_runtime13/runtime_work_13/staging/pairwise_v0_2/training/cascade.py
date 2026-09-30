"""Operational coarse-first gate for deferred geometry/local computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from staging.pairwise_v0_2.models.coarse import CoarseOutput
from staging.pairwise_v0_2.training.thresholds import (
    CoarseGateArtifact,
    CoarseGateDecision,
    apply_high_recall_gate,
)


CASCADE_SCHEMA_VERSION = "dunhuang-pairwise-coarse-local-cascade/0.2"


class CascadeMode(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


@dataclass(frozen=True)
class LocalCascadeScores:
    """Local/fused scores aligned exactly with the selected coarse indices."""

    probability: Tensor
    valid: Tensor
    requires_review: Tensor

    def __post_init__(self) -> None:
        if self.probability.ndim != 1 or not self.probability.is_floating_point():
            raise TypeError("local probability must be floating-point [N]")
        expected = tuple(self.probability.shape)
        if self.valid.dtype != torch.bool or tuple(self.valid.shape) != expected:
            raise TypeError("local valid must be bool [N]")
        if self.requires_review.dtype != torch.bool or tuple(
            self.requires_review.shape
        ) != expected:
            raise TypeError("local requires_review must be bool [N]")
        usable = self.valid & ~self.requires_review
        if (
            (~torch.isfinite(self.probability[usable])).any().item()
            or (self.probability[usable] < 0.0).any().item()
            or (self.probability[usable] > 1.0).any().item()
        ):
            raise ValueError("valid local probabilities must be finite in [0, 1]")


@dataclass(frozen=True)
class CascadeReceipt:
    mode: str
    sample_count: int
    coarse_reject_count: int
    pass_to_local_count: int
    invalid_coarse_count: int
    geometry_callback_count: int
    local_callback_count: int
    training_gate_bypassed: bool
    gate_policy: str
    schema_version: str = CASCADE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CascadeOutput:
    coarse: CoarseOutput
    gate: CoarseGateDecision
    local_probability: Tensor
    local_valid: Tensor
    local_evaluated: Tensor
    final_probability: Tensor
    decision_valid: Tensor
    requires_review: Tensor
    receipt: CascadeReceipt


GeometryBuilder = Callable[[Tuple[int, ...]], Any]
LocalScorer = Callable[[Any, Tuple[int, ...], CoarseOutput], LocalCascadeScores]


def _validate_coarse(output: CoarseOutput) -> int:
    if not isinstance(output, CoarseOutput):
        raise TypeError("coarse_output must be CoarseOutput")
    batch = int(output.probability.numel())
    if output.probability.ndim != 1 or batch < 1:
        raise ValueError("coarse probability must be non-empty [B]")
    if output.valid_problem.dtype != torch.bool or tuple(
        output.valid_problem.shape
    ) != (batch,):
        raise TypeError("coarse valid_problem must be bool [B]")
    if tuple(output.logit.shape) != (batch,):
        raise ValueError("coarse logit must have shape [B]")
    return batch


def _training_gate(output: CoarseOutput) -> CoarseGateDecision:
    batch = output.probability.shape[0]
    reject = torch.zeros(batch, dtype=torch.bool, device=output.probability.device)
    score_valid = output.valid_problem & torch.isfinite(output.probability)
    return CoarseGateDecision(
        reject=reject,
        pass_to_local=~reject,
        requires_review=~score_valid,
        score_valid=score_valid,
    )


def execute_coarse_local_cascade(
    coarse_output: CoarseOutput,
    geometry_builder: GeometryBuilder,
    local_scorer: LocalScorer,
    *,
    mode: CascadeMode = CascadeMode.INFERENCE,
    gate_artifact: Optional[CoarseGateArtifact] = None,
    checkpoint_id: Optional[str] = None,
    config_hash: Optional[str] = None,
    bypass_gate_in_training: bool = True,
) -> CascadeOutput:
    """Gate a batch before *any* geometry or local-model callback is invoked.

    Invalid coarse evidence can never reject a sample: it passes to local and
    remains marked for review.  Training may intentionally bypass the frozen
    gate; that choice is explicit in the returned portable receipt.
    """

    batch = _validate_coarse(coarse_output)
    if not callable(geometry_builder) or not callable(local_scorer):
        raise TypeError("geometry_builder and local_scorer must be callable")
    parsed_mode = CascadeMode(mode)
    training_bypass = parsed_mode is CascadeMode.TRAINING and bypass_gate_in_training
    if training_bypass:
        gate = _training_gate(coarse_output)
        gate_policy = "bypassed_for_training_all_samples_to_local"
    else:
        if gate_artifact is None or checkpoint_id is None or config_hash is None:
            raise ValueError("a matching frozen gate artifact is required")
        gate = apply_high_recall_gate(
            coarse_output.probability,
            coarse_output.valid_problem,
            gate_artifact,
            checkpoint_id=checkpoint_id,
            config_hash=config_hash,
        )
        gate_policy = "frozen_high_recall_gate"

    selected_tensor = torch.nonzero(gate.pass_to_local, as_tuple=False).flatten()
    selected = tuple(int(index) for index in selected_tensor.detach().cpu().tolist())
    local_probability = torch.full_like(coarse_output.probability, 0.5)
    local_valid = torch.zeros(
        batch, dtype=torch.bool, device=coarse_output.probability.device
    )
    local_evaluated = torch.zeros_like(local_valid)
    geometry_calls = 0
    local_calls = 0
    local_review = torch.zeros_like(local_valid)

    if selected:
        geometry_payload = geometry_builder(selected)
        geometry_calls = 1
        scores = local_scorer(geometry_payload, selected, coarse_output)
        local_calls = 1
        if not isinstance(scores, LocalCascadeScores):
            raise TypeError("local_scorer must return LocalCascadeScores")
        if scores.probability.shape[0] != len(selected):
            raise ValueError("local scores are not aligned with selected samples")
        if scores.probability.device != coarse_output.probability.device:
            raise ValueError("local and coarse scores must share a device")
        safe_local = torch.where(
            torch.isfinite(scores.probability),
            scores.probability,
            torch.full_like(scores.probability, 0.5),
        )
        local_probability = local_probability.index_copy(0, selected_tensor, safe_local)
        local_valid = local_valid.index_copy(0, selected_tensor, scores.valid)
        local_review = local_review.index_copy(
            0, selected_tensor, scores.requires_review
        )
        local_evaluated[selected_tensor] = True

    final_probability = torch.where(
        gate.pass_to_local, local_probability, coarse_output.probability
    )
    # Invalid coarse evidence remains review-only even when local succeeds.
    decision_valid = gate.reject | (
        gate.pass_to_local
        & gate.score_valid
        & local_valid
        & ~local_review
    )
    requires_review = gate.requires_review | (
        gate.pass_to_local & (~local_valid | local_review)
    )
    receipt = CascadeReceipt(
        mode=parsed_mode.value,
        sample_count=batch,
        coarse_reject_count=int(gate.reject.sum().item()),
        pass_to_local_count=len(selected),
        invalid_coarse_count=int((~gate.score_valid).sum().item()),
        geometry_callback_count=geometry_calls,
        local_callback_count=local_calls,
        training_gate_bypassed=training_bypass,
        gate_policy=gate_policy,
    )
    return CascadeOutput(
        coarse=coarse_output,
        gate=gate,
        local_probability=local_probability,
        local_valid=local_valid,
        local_evaluated=local_evaluated,
        final_probability=final_probability,
        decision_valid=decision_valid,
        requires_review=requires_review,
        receipt=receipt,
    )


__all__ = [
    "CASCADE_SCHEMA_VERSION",
    "CascadeMode",
    "CascadeOutput",
    "CascadeReceipt",
    "LocalCascadeScores",
    "execute_coarse_local_cascade",
]
