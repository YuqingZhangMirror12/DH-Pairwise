"""Dependency-light value contract for LOCAL-Q1 pair qualification outcomes.

This module deliberately contains no geometry builder, tensor framework, SciPy,
cache, archive, model, or GPU dependency.  Geometry qualification code imports
and re-exports this exact value type; metadata-only eligibility consumers may
therefore parse frozen decisions without importing the geometry execution stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


LOCAL_Q1_PAIR_QUALIFICATION_VERSION = (
    "dunhuang-local-q1-pair-geometry-qualification/0.1"
)


@dataclass(frozen=True)
class LocalQ1PairQualification:
    """Identity-free outcome of one pair's frozen geometry/resource gates."""

    eligible: bool
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    fragment_statuses: Tuple[str, str]
    pair_status: str
    emitted_directions: Tuple[str, ...]
    candidate_count: int
    max_sequence_a: int
    max_sequence_b: int
    local_tensor_elements: int
    attention_elements_per_head: int
    affinity_elements: int
    sinkhorn_elements: int
    candidate_semantic_sha256: str

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:  # noqa: E721
            raise TypeError("eligible must be a built-in bool")
        if self.eligible != (self.failure_stage is None and self.failure_reason is None):
            raise ValueError("qualification status and failure fields disagree")
        if self.failure_stage not in {
            None,
            "fragment",
            "pair",
            "direction",
            "resource",
        }:
            raise ValueError("unsupported qualification failure stage")
        if not self.eligible and (
            not isinstance(self.failure_reason, str) or not self.failure_reason
        ):
            raise ValueError("ineligible qualification requires a failure reason")
        if len(self.fragment_statuses) != 2 or any(
            not isinstance(value, str) or not value for value in self.fragment_statuses
        ):
            raise ValueError("exactly two fragment statuses are required")
        if not isinstance(self.pair_status, str) or not self.pair_status:
            raise ValueError("pair_status is required")
        if any(not isinstance(value, str) or not value for value in self.emitted_directions):
            raise ValueError("emitted direction names must be non-empty strings")
        for name in (
            "candidate_count",
            "max_sequence_a",
            "max_sequence_b",
            "local_tensor_elements",
            "attention_elements_per_head",
            "affinity_elements",
            "sinkhorn_elements",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("{} must be a non-negative integer".format(name))
        if not re.fullmatch(r"[0-9a-f]{64}", self.candidate_semantic_sha256):
            raise ValueError("candidate_semantic_sha256 must be lowercase SHA-256")

    def portable_dict(self) -> Dict[str, Any]:
        return {
            "qualification_version": LOCAL_Q1_PAIR_QUALIFICATION_VERSION,
            "eligible": self.eligible,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "fragment_statuses": list(self.fragment_statuses),
            "pair_status": self.pair_status,
            "emitted_directions": list(self.emitted_directions),
            "candidate_count": self.candidate_count,
            "max_sequence_a": self.max_sequence_a,
            "max_sequence_b": self.max_sequence_b,
            "local_tensor_elements": self.local_tensor_elements,
            "attention_elements_per_head": self.attention_elements_per_head,
            "affinity_elements": self.affinity_elements,
            "sinkhorn_elements": self.sinkhorn_elements,
            "candidate_semantic_sha256": self.candidate_semantic_sha256,
        }


__all__ = [
    "LOCAL_Q1_PAIR_QUALIFICATION_VERSION",
    "LocalQ1PairQualification",
]
