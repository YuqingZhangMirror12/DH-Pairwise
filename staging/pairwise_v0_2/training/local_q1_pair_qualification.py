"""Pure, supervision-blind LOCAL-Q1 pair-geometry qualification.

This module is the single authority for the per-record gates shared by the
Route-A refreeze preflight and the formal read-only provider.  Its public
function deliberately accepts fragment geometry plus an opaque, label-blind
pair identifier; it has no access to a :class:`TrainingPairRecord`, labels,
historical directions, hard-negative metadata, or provenance.

An eligible pair must have two successful fragment artifacts, combine under
``direction_b_wrt_a=None``, emit non-empty candidates for every frozen upright
direction, and fit every individual GeometryBatchConfig allocation guard.
Batch-to-batch packing is intentionally outside this gate: after every record
fits alone, the provider's deterministic packer is still authoritative for
multi-record padding.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from staging.pairwise_v0_2.geometry import (
    CHANNEL_ORDER,
    DEFAULT_DIRECTION_ORDER,
    FragmentGeometryResult,
    GeometryStatus,
    combine_fragment_results,
)
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.local_q1_pair_qualification_contract import (
    LOCAL_Q1_PAIR_QUALIFICATION_VERSION,
    LocalQ1PairQualification,
)


_PAIR_ID_RE = re.compile(r"^pair/sha256/[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _outcome(
    *,
    eligible: bool,
    failure_stage: Optional[str],
    failure_reason: Optional[str],
    fragment_statuses: Tuple[str, str],
    pair_status: str = "not_evaluated",
    emitted_directions: Sequence[str] = (),
    candidate_rows: Sequence[Tuple[str, str, int, int]] = (),
    geometry_config: GeometryBatchConfig,
) -> LocalQ1PairQualification:
    rows = tuple(candidate_rows)
    candidate_count = len(rows)
    max_a = max((row[2] for row in rows), default=0)
    max_b = max((row[3] for row in rows), default=0)
    patch_height, patch_width = geometry_config.geometry.output_size
    local_elements = (
        candidate_count
        * len(CHANNEL_ORDER)
        * patch_height
        * patch_width
        * (max_a + max_b)
    )
    attention = candidate_count * (max_a + max_b) ** 2
    affinity = candidate_count * max_a * max_b
    sinkhorn = candidate_count * (max_a + 1) * (max_b + 1)
    candidate_payload = [
        {
            "candidate": candidate_id,
            "direction": direction,
            "sequence_a": length_a,
            "sequence_b": length_b,
        }
        for candidate_id, direction, length_a, length_b in rows
    ]
    return LocalQ1PairQualification(
        eligible=eligible,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        fragment_statuses=fragment_statuses,
        pair_status=pair_status,
        emitted_directions=tuple(emitted_directions),
        candidate_count=candidate_count,
        max_sequence_a=max_a,
        max_sequence_b=max_b,
        local_tensor_elements=local_elements,
        attention_elements_per_head=attention,
        affinity_elements=affinity,
        sinkhorn_elements=sinkhorn,
        candidate_semantic_sha256=_sha256(_canonical_json(candidate_payload)),
    )


def qualify_local_q1_pair_geometry(
    first: FragmentGeometryResult,
    second: FragmentGeometryResult,
    *,
    pair_id: str,
    geometry_config: GeometryBatchConfig,
    require_candidates: bool = True,
) -> LocalQ1PairQualification:
    """Apply all frozen individual-record gates without supervision access.

    ``pair_id`` is the existing deterministic dataset/endpoints hash.  It is
    accepted only to reproduce the provider's exact candidate-ID byte guard;
    the function cannot inspect a label or direction because neither is in its
    signature.
    """

    if not isinstance(first, FragmentGeometryResult) or not isinstance(
        second, FragmentGeometryResult
    ):
        raise TypeError("fragment geometry results are required")
    if not isinstance(pair_id, str) or not _PAIR_ID_RE.fullmatch(pair_id):
        raise ValueError("pair_id must be the label-blind pair SHA-256 identifier")
    if not isinstance(geometry_config, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    if type(require_candidates) is not bool:  # noqa: E721
        raise TypeError("require_candidates must be bool")

    fragment_statuses = tuple(
        sorted((first.status.value, second.status.value))
    )  # endpoint-order neutral in the receipt/index
    if (
        first.status is not GeometryStatus.OK
        or second.status is not GeometryStatus.OK
        or first.artifact is None
        or second.artifact is None
    ):
        reasons = sorted(
            {
                value
                for value in (first.failure_reason, second.failure_reason)
                if isinstance(value, str) and value
            }
        )
        return _outcome(
            eligible=False,
            failure_stage="fragment",
            failure_reason=(
                "fragment_geometry_not_ok"
                if not reasons
                else "fragment_geometry_not_ok:" + "+".join(reasons)
            ),
            fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
            geometry_config=geometry_config,
        )

    combined = combine_fragment_results(
        first,
        second,
        direction_b_wrt_a=None,
        config=geometry_config.geometry,
    )
    if combined.status is not GeometryStatus.OK:
        return _outcome(
            eligible=False,
            failure_stage="pair",
            failure_reason=combined.failure_reason or "pair_geometry_not_ok",
            fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
            pair_status=combined.status.value,
            geometry_config=geometry_config,
        )
    if require_candidates and not combined.candidates:
        return _outcome(
            eligible=False,
            failure_stage="pair",
            failure_reason="pair_emitted_no_candidates",
            fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
            pair_status=combined.status.value,
            geometry_config=geometry_config,
        )

    emitted = tuple(group.direction for group in combined.direction_groups)
    emitted_names = tuple(direction.value for direction in emitted)
    if (
        len(emitted) != len(DEFAULT_DIRECTION_ORDER)
        or len(set(emitted)) != len(emitted)
        or set(emitted) != set(DEFAULT_DIRECTION_ORDER)
    ):
        return _outcome(
            eligible=False,
            failure_stage="direction",
            failure_reason="pair_does_not_cover_all_four_upright_directions",
            fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
            pair_status=combined.status.value,
            emitted_directions=emitted_names,
            geometry_config=geometry_config,
        )

    rows = tuple(
        (
            candidate.candidate_id,
            candidate.direction.value,
            candidate.sequence_a.length,
            candidate.sequence_b.length,
        )
        for candidate in combined.candidates
    )

    def reject(reason: str) -> LocalQ1PairQualification:
        return _outcome(
            eligible=False,
            failure_stage="resource",
            failure_reason=reason,
            fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
            pair_status=combined.status.value,
            emitted_directions=emitted_names,
            candidate_rows=rows,
            geometry_config=geometry_config,
        )

    if len(rows) > geometry_config.max_candidates_per_sample:
        return reject("max_candidates_per_sample_exceeded")
    max_a = max((row[2] for row in rows), default=0)
    max_b = max((row[3] for row in rows), default=0)
    if (
        max_a > geometry_config.max_sequence_length
        or max_b > geometry_config.max_sequence_length
    ):
        return reject("max_sequence_length_exceeded")
    for candidate_id, _direction, length_a, length_b in rows:
        attention = (length_a + length_b) ** 2
        affinity = length_a * length_b
        sinkhorn = (length_a + 1) * (length_b + 1)
        if attention > geometry_config.max_attention_score_elements_per_candidate:
            return reject("candidate_attention_bound_exceeded")
        if affinity > geometry_config.max_affinity_elements_per_candidate:
            return reject("candidate_affinity_bound_exceeded")
        if sinkhorn > geometry_config.max_sinkhorn_elements_per_candidate:
            return reject("candidate_sinkhorn_bound_exceeded")
        global_id = "sample/{:06d}:{}:{}".format(0, pair_id, candidate_id)
        if len(global_id.encode("utf-8")) > geometry_config.max_candidate_id_bytes:
            return reject("candidate_id_byte_bound_exceeded")

    candidate_count = len(rows)
    patch_height, patch_width = geometry_config.geometry.output_size
    local_elements = (
        candidate_count
        * len(CHANNEL_ORDER)
        * patch_height
        * patch_width
        * (max_a + max_b)
    )
    attention = candidate_count * (max_a + max_b) ** 2
    affinity = candidate_count * max_a * max_b
    sinkhorn = candidate_count * (max_a + 1) * (max_b + 1)
    if candidate_count > geometry_config.max_candidates_per_batch:
        return reject("single_record_candidate_batch_bound_exceeded")
    if local_elements > geometry_config.max_local_tensor_elements:
        return reject("single_record_local_tensor_bound_exceeded")
    if attention > geometry_config.max_attention_score_elements_per_batch:
        return reject("single_record_attention_batch_bound_exceeded")
    if affinity > geometry_config.max_affinity_elements_per_batch:
        return reject("single_record_affinity_batch_bound_exceeded")
    if sinkhorn > geometry_config.max_sinkhorn_elements_per_batch:
        return reject("single_record_sinkhorn_batch_bound_exceeded")

    return _outcome(
        eligible=True,
        failure_stage=None,
        failure_reason=None,
        fragment_statuses=fragment_statuses,  # type: ignore[arg-type]
        pair_status=combined.status.value,
        emitted_directions=emitted_names,
        candidate_rows=rows,
        geometry_config=geometry_config,
    )


def qualification_contract(geometry_config: GeometryBatchConfig) -> Mapping[str, Any]:
    """Return the exact portable policy committed by eligibility receipts."""

    if not isinstance(geometry_config, GeometryBatchConfig):
        raise TypeError("geometry_config must be GeometryBatchConfig")
    return {
        "qualification_version": LOCAL_Q1_PAIR_QUALIFICATION_VERSION,
        "candidate_generation_direction_argument": None,
        "candidate_generation_directions": [
            direction.value for direction in DEFAULT_DIRECTION_ORDER
        ],
        "supervision_fields_read": [],
        "fragment_gate": "both_fragment_geometry_results_are_ok",
        "pair_gate": "combine_direction_none_ok_and_nonempty_candidates",
        "direction_gate": "all_four_frozen_upright_directions_exactly_once",
        "individual_resource_gate": (
            "all_candidate_sequence_attention_affinity_sinkhorn_id_and_single_"
            "record_batch_bounds"
        ),
        "geometry_batch_config_sha256": _sha256(
            _canonical_json(geometry_config.provenance_dict())
        ),
    }


__all__ = [
    "LOCAL_Q1_PAIR_QUALIFICATION_VERSION",
    "LocalQ1PairQualification",
    "qualification_contract",
    "qualify_local_q1_pair_geometry",
]
