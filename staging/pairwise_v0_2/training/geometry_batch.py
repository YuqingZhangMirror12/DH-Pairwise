"""Leakage-safe data/geometry/tensor bridge for Pairwise v0.2.

The bridge deliberately calls geometry with ``direction_b_wrt_a=None`` for
*every* record.  Thus train and inference both enumerate the same four
upright facing-side hypotheses.  A historical direction label is converted
only to an output target index; it can never select an input arc candidate.

This module is model-agnostic.  It exposes the tensor names consumed by the
current model, but imports neither a model nor a loss.  Callers may supply a
content-addressed fragment cache; the uncached path remains API-compatible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import ndimage
from torch import Tensor

from staging.pairwise_v0_2.geometry import (
    CHANNEL_ORDER,
    DEFAULT_DIRECTION_ORDER,
    GEOMETRY_VERSION,
    CandidateBuilderConfig,
    ContourKeypointConfig,
    GeometryStatus,
    KeypointPairCandidate,
    PairDirection,
    build_fragment_keypoints,
    build_keypoint_pair_candidates,
    build_pair_candidates,
    combine_fragment_results,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.fragment_geometry_cache import (
    FragmentGeometryCacheLookup,
    load_or_build_fragment_geometry,
)
from staging.pairwise_v0_2.training.geometry_cache import (
    CACHE_RECIPE_VERSION,
    CACHE_SCHEMA_VERSION,
    GeometryArtifactCache,
    fragment_cache_identity,
)


GEOMETRY_BATCH_VERSION = "dunhuang-pairwise-geometry-batch/0.4"
MULTIRUN_REPRESENTATION = "multirun_sliding_window"
KEYPOINT_REPRESENTATION = "contour_keypoint"
LOCAL_CANDIDATE_REPRESENTATIONS = (
    MULTIRUN_REPRESENTATION,
    KEYPOINT_REPRESENTATION,
)
COARSE_NUMERIC_CONTRACT = (
    "finite_float32_contiguous_reject_beyond_2eps_then_clamp_unit_interval/0.1"
)
_COARSE_PRECLAMP_RANGE_TOLERANCE = 2.0 * float(torch.finfo(torch.float32).eps)
SEQUENCE_LENGTH_BUCKET_EDGES = (16, 32, 64, 128, 256, 512)
DIRECTION_NAMES = tuple(direction.value for direction in DEFAULT_DIRECTION_ORDER)
DATA_DIRECTION_TO_PAIR_DIRECTION = MappingProxyType(
    {
        "left": PairDirection.B_LEFT_OF_A,
        "right": PairDirection.B_RIGHT_OF_A,
        "above": PairDirection.B_ABOVE_A,
        "below": PairDirection.B_BELOW_A,
    }
)
DATA_DIRECTION_TO_INDEX = MappingProxyType(
    {
        name: DEFAULT_DIRECTION_ORDER.index(direction)
        for name, direction in DATA_DIRECTION_TO_PAIR_DIRECTION.items()
    }
)
INVERSE_DATA_DIRECTION = MappingProxyType(
    {
        name: direction.inverse.short_name
        for name, direction in DATA_DIRECTION_TO_PAIR_DIRECTION.items()
    }
)


class GeometryBatchError(ValueError):
    """Raised before a partial or unbounded tensor batch can be returned."""


def _deep_freeze_receipt(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise GeometryBatchError("receipt metadata must be finite")
        return value
    if isinstance(value, Mapping):
        copied = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise GeometryBatchError("receipt metadata keys must be strings")
            copied[name] = _deep_freeze_receipt(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_receipt(item) for item in value)
    raise GeometryBatchError("receipt metadata is not JSON-compatible")


def _deep_thaw_receipt(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _deep_thaw_receipt(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw_receipt(item) for item in value]
    return value


def data_direction_to_pair_direction(value: Optional[str]) -> Optional[PairDirection]:
    """Parse the historical B-w.r.t.-A vocabulary without guessing semantics."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("data direction must be a string or None")
    try:
        return DATA_DIRECTION_TO_PAIR_DIRECTION[value]
    except KeyError as exc:
        raise GeometryBatchError(
            "unsupported B-with-respect-to-A data direction"
        ) from exc


def data_direction_to_target_index(value: Optional[str]) -> int:
    """Return the stable four-direction target index, or -1 when unavailable."""

    direction = data_direction_to_pair_direction(value)
    if direction is None:
        return -1
    return DEFAULT_DIRECTION_ORDER.index(direction)


def inverse_data_direction(value: str) -> str:
    """Invert a B-w.r.t.-A label after swapping fragments A and B."""

    direction = data_direction_to_pair_direction(value)
    if direction is None:  # pragma: no cover - ``value`` is statically non-optional
        raise GeometryBatchError("a concrete direction is required")
    return direction.inverse.short_name


def inverse_direction_index(index: int) -> int:
    """Invert a stable direction slot after swapping fragments A and B."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("direction index must be an integer")
    if index < 0 or index >= len(DEFAULT_DIRECTION_ORDER):
        raise GeometryBatchError("direction index is out of range")
    return DEFAULT_DIRECTION_ORDER.index(DEFAULT_DIRECTION_ORDER[index].inverse)


@dataclass(frozen=True)
class GeometryBatchConfig:
    """Frozen output geometry and fail-closed allocation bounds."""

    geometry: CandidateBuilderConfig = field(default_factory=CandidateBuilderConfig)
    coarse_output_size: Tuple[int, int] = (128, 128)
    coarse_resize_mode: str = "bilinear"
    coarse_preprocess_mode: str = "tight_crop_letterbox"
    coarse_content_fraction: float = 0.875
    coarse_component_connectivity: int = 4
    max_batch_size: int = 256
    max_input_pixels_per_mask: int = 100_000_000
    max_candidates_per_sample: int = 512
    max_candidates_per_batch: int = 2048
    max_sequence_length: int = 512
    max_local_tensor_elements: int = 64_000_000
    # Logical score elements.  Attention is measured per head as
    # L_a^2 + L_b^2 + 2*L_a*L_b == (L_a+L_b)^2; model head count therefore
    # remains an explicit multiplicative capacity-planning factor.
    max_attention_score_elements_per_candidate: int = 1_048_576
    max_attention_score_elements_per_batch: int = 16_777_216
    max_affinity_elements_per_candidate: int = 262_144
    max_affinity_elements_per_batch: int = 8_388_608
    max_sinkhorn_elements_per_candidate: int = 263_169
    max_sinkhorn_elements_per_batch: int = 8_421_408
    max_candidate_id_bytes: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, CandidateBuilderConfig):
            raise TypeError("geometry must be CandidateBuilderConfig")
        if (
            len(self.coarse_output_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.coarse_output_size
            )
            or min(self.coarse_output_size) < 8
        ):
            raise ValueError("coarse_output_size must contain two integers >= 8")
        if self.coarse_resize_mode not in {"nearest", "bilinear", "area"}:
            raise ValueError("unsupported coarse_resize_mode")
        if self.coarse_preprocess_mode not in {
            "tight_crop_letterbox",
            "full_canvas_stretch_legacy",
        }:
            raise ValueError("unsupported coarse_preprocess_mode")
        if (
            not np.isfinite(self.coarse_content_fraction)
            or not 0.5 <= self.coarse_content_fraction <= 1.0
        ):
            raise ValueError("coarse_content_fraction must be in [0.5, 1.0]")
        if self.coarse_component_connectivity not in {4, 8}:
            raise ValueError("coarse_component_connectivity must be 4 or 8")
        for name in (
            "max_batch_size",
            "max_input_pixels_per_mask",
            "max_candidates_per_sample",
            "max_candidates_per_batch",
            "max_sequence_length",
            "max_local_tensor_elements",
            "max_attention_score_elements_per_candidate",
            "max_attention_score_elements_per_batch",
            "max_affinity_elements_per_candidate",
            "max_affinity_elements_per_batch",
            "max_sinkhorn_elements_per_candidate",
            "max_sinkhorn_elements_per_batch",
            "max_candidate_id_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))

    def provenance_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe, complete receipt of output-affecting settings."""

        return {
            "bridge_version": GEOMETRY_BATCH_VERSION,
            "geometry_version": GEOMETRY_VERSION,
            "direction_names": list(DIRECTION_NAMES),
            "candidate_generation_direction_argument": None,
            "direction_label_usage": "output_target_only_never_geometry_input",
            "coarse_output_size": list(self.coarse_output_size),
            "coarse_resize_mode": self.coarse_resize_mode,
            "coarse_preprocess_mode": self.coarse_preprocess_mode,
            "coarse_content_fraction": self.coarse_content_fraction,
            "coarse_component_connectivity": self.coarse_component_connectivity,
            "coarse_numeric_contract": COARSE_NUMERIC_CONTRACT,
            "coarse_spatial_contract": (
                "single_fragment_largest_connected_component_foreground_bbox_"
                "then_aspect_preserving_centered_letterbox_no_common_canvas_"
                "position"
                if self.coarse_preprocess_mode == "tight_crop_letterbox"
                else "legacy_full_canvas_stretch_spatial_shortcut_risk"
            ),
            "geometry": asdict(self.geometry),
            "bounds": {
                "max_batch_size": self.max_batch_size,
                "max_input_pixels_per_mask": self.max_input_pixels_per_mask,
                "max_candidates_per_sample": self.max_candidates_per_sample,
                "max_candidates_per_batch": self.max_candidates_per_batch,
                "max_sequence_length": self.max_sequence_length,
                "max_local_tensor_elements": self.max_local_tensor_elements,
                "max_attention_score_elements_per_candidate": (
                    self.max_attention_score_elements_per_candidate
                ),
                "max_attention_score_elements_per_batch": (
                    self.max_attention_score_elements_per_batch
                ),
                "max_affinity_elements_per_candidate": (
                    self.max_affinity_elements_per_candidate
                ),
                "max_affinity_elements_per_batch": (
                    self.max_affinity_elements_per_batch
                ),
                "max_sinkhorn_elements_per_candidate": (
                    self.max_sinkhorn_elements_per_candidate
                ),
                "max_sinkhorn_elements_per_batch": (
                    self.max_sinkhorn_elements_per_batch
                ),
                "max_candidate_id_bytes": self.max_candidate_id_bytes,
            },
            "complexity_accounting": {
                "attention": "per_head_(padded_La_plus_padded_Lb)_squared",
                "affinity": "padded_La_times_padded_Lb",
                "sinkhorn": "(padded_La_plus_1)_times_(padded_Lb_plus_1)",
                "padding_scope": "flattened_candidate_batch",
                "sequence_length_bucket_edges": list(SEQUENCE_LENGTH_BUCKET_EDGES),
            },
            "cache_policy": (
                "optional_content_addressed_role_neutral_fragment_npz_"
                "same_tensor_contract"
            ),
        }

    @property
    def fingerprint(self) -> str:
        payload = _canonical_json(self.provenance_dict())
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GeometrySampleReceipt:
    """Portable per-pair preprocessing receipt; contains no local paths."""

    pair_id: str
    geometry_cache_key: str
    dataset_id: str
    canonical_group_id: str
    component_id: str
    geometry_status: str
    geometry_failure_reason: Optional[str]
    candidate_count: int
    direction_slot_valid: Tuple[bool, bool, bool, bool]
    emitted_directions: Tuple[str, ...]
    geometry_quality: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.direction_slot_valid) != 4:
            raise ValueError("direction_slot_valid must contain four entries")
        if self.candidate_count < 0:
            raise ValueError("candidate_count cannot be negative")
        object.__setattr__(
            self,
            "geometry_quality",
            _deep_freeze_receipt(dict(self.geometry_quality)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "geometry_cache_key": self.geometry_cache_key,
            "dataset_id": self.dataset_id,
            "canonical_group_id": self.canonical_group_id,
            "component_id": self.component_id,
            "geometry_status": self.geometry_status,
            "geometry_failure_reason": self.geometry_failure_reason,
            "candidate_count": self.candidate_count,
            "direction_slot_valid": list(self.direction_slot_valid),
            "emitted_directions": list(self.emitted_directions),
            "geometry_quality": _deep_thaw_receipt(self.geometry_quality),
        }


@dataclass(frozen=True)
class BatchComplexityReceipt:
    """Exact padded tensor/score costs checked before torch allocation."""

    candidate_count: int
    padded_sequence_a: int
    padded_sequence_b: int
    local_tensor_elements: int
    attention_score_elements_per_head: int
    affinity_elements: int
    sinkhorn_elements: int
    sequence_bucket_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "padded_sequence_a",
            "padded_sequence_b",
            "local_tensor_elements",
            "attention_score_elements_per_head",
            "affinity_elements",
            "sinkhorn_elements",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("{} must be a non-negative integer".format(name))
        object.__setattr__(
            self,
            "sequence_bucket_counts",
            _deep_freeze_receipt(dict(self.sequence_bucket_counts)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "padded_sequence_a": self.padded_sequence_a,
            "padded_sequence_b": self.padded_sequence_b,
            "local_tensor_elements": self.local_tensor_elements,
            "attention_score_elements_per_head": (
                self.attention_score_elements_per_head
            ),
            "affinity_elements": self.affinity_elements,
            "sinkhorn_elements": self.sinkhorn_elements,
            "sequence_bucket_counts": _deep_thaw_receipt(self.sequence_bucket_counts),
        }


@dataclass(frozen=True)
class RaggedGeometryBatch:
    """Coarse sample tensors plus flattened padded local arc candidates.

    ``local_*`` are padded across arc candidates, not across pair samples.
    ``sample_index`` maps each arc candidate back to a coarse sample and
    ``direction_index`` maps it to one of the four stable direction slots.
    Missing/invalid directions are represented in ``direction_slot_valid``.
    """

    coarse_a: Tensor
    coarse_b: Tensor
    local_a: Tensor
    local_b: Tensor
    token_mask_a: Tensor
    token_mask_b: Tensor
    sample_index: Tensor
    direction_index: Tensor
    candidate_valid: Tensor
    labels: Tensor
    direction_target: Tensor
    direction_target_valid: Tensor
    direction_slot_valid: Tensor
    geometry_valid: Tensor
    candidate_ids: Tuple[str, ...]
    sample_ids: Tuple[str, ...]
    geometry_cache_keys: Tuple[str, ...]
    sample_receipts: Tuple[GeometrySampleReceipt, ...]
    config_fingerprint: str
    config_provenance: Mapping[str, Any]
    complexity_receipt: BatchComplexityReceipt
    sequence_length_buckets: Tuple[str, ...]
    direction_names: Tuple[str, ...] = DIRECTION_NAMES
    bridge_version: str = GEOMETRY_BATCH_VERSION
    candidate_representation: str = MULTIRUN_REPRESENTATION
    correspondence_mask: Optional[Tensor] = None
    exact_assignment_target_a: Optional[Tensor] = None
    exact_assignment_target_b: Optional[Tensor] = None
    exact_supervision_config: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.coarse_a.ndim != 4 or self.coarse_b.ndim != 4:
            raise ValueError("coarse tensors must have shape [B, C, H, W]")
        if tuple(self.coarse_a.shape) != tuple(self.coarse_b.shape):
            raise ValueError("coarse A/B shapes differ")
        batch_size = int(self.coarse_a.shape[0])
        if batch_size < 1 or self.coarse_a.shape[1] != 1:
            raise ValueError("coarse tensors require a non-empty one-channel batch")
        if (
            not self.coarse_a.is_floating_point()
            or not self.coarse_b.is_floating_point()
        ):
            raise TypeError("coarse tensors must be floating point")
        if self.local_a.ndim != 5 or self.local_b.ndim != 5:
            raise ValueError("local tensors must have shape [N, L, C, H, W]")
        candidate_count = int(self.local_a.shape[0])
        if self.local_b.shape[0] != candidate_count:
            raise ValueError("local A/B candidate counts differ")
        if self.local_a.shape[2:] != self.local_b.shape[2:]:
            raise ValueError("local A/B channel and patch shapes differ")
        if not self.local_a.is_floating_point() or not self.local_b.is_floating_point():
            raise TypeError("local tensors must be floating point")
        expected_a = (candidate_count, int(self.local_a.shape[1]))
        expected_b = (candidate_count, int(self.local_b.shape[1]))
        if (
            self.token_mask_a.dtype != torch.bool
            or tuple(self.token_mask_a.shape) != expected_a
        ):
            raise TypeError("token_mask_a must be bool [N, L_a]")
        if (
            self.token_mask_b.dtype != torch.bool
            or tuple(self.token_mask_b.shape) != expected_b
        ):
            raise TypeError("token_mask_b must be bool [N, L_b]")
        if self.candidate_representation not in LOCAL_CANDIDATE_REPRESENTATIONS:
            raise ValueError("unsupported local candidate representation")
        correspondence = self.correspondence_mask
        if correspondence is None:
            correspondence = (
                self.token_mask_a[:, :, None] & self.token_mask_b[:, None, :]
            )
            object.__setattr__(self, "correspondence_mask", correspondence)
        if (
            not isinstance(correspondence, Tensor)
            or correspondence.dtype != torch.bool
            or tuple(correspondence.shape)
            != (candidate_count, expected_a[1], expected_b[1])
        ):
            raise TypeError("correspondence_mask must be bool [N, L_a, L_b]")
        if (
            (
                correspondence
                & ~(self.token_mask_a[:, :, None] & self.token_mask_b[:, None, :])
            )
            .any()
            .item()
        ):
            raise ValueError("correspondence mask enables a padded token")
        exact_a = self.exact_assignment_target_a
        exact_b = self.exact_assignment_target_b
        exact_config = self.exact_supervision_config
        if (exact_a is None) != (exact_b is None):
            raise ValueError("exact A/B assignment targets must be supplied together")
        if exact_a is None:
            if exact_config is not None:
                raise ValueError("exact supervision config requires exact targets")
        else:
            if self.candidate_representation != KEYPOINT_REPRESENTATION:
                raise ValueError("exact seam targets require keypoint candidates")
            if (
                exact_a.dtype != torch.long
                or tuple(exact_a.shape) != expected_a
                or exact_b is None
                or exact_b.dtype != torch.long
                or tuple(exact_b.shape) != expected_b
            ):
                raise TypeError("exact assignment targets must be int64 [N,L]")
            if exact_config is None or not isinstance(exact_config, Mapping):
                raise TypeError("exact supervision config must be a mapping")
            frozen_exact_config = _deep_freeze_receipt(dict(exact_config))
            object.__setattr__(
                self, "exact_supervision_config", frozen_exact_config
            )
            from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
                ExactSeamTensorBatch,
            )

            ExactSeamTensorBatch(
                assignment_target_a=exact_a,
                assignment_target_b=exact_b,
                sample_index=self.sample_index,
                direction_index=self.direction_index,
                candidate_ids=self.candidate_ids,
            )
            if (exact_a[~self.token_mask_a] != -2).any().item() or (
                exact_b[~self.token_mask_b] != -2
            ).any().item():
                raise ValueError("padded exact assignment targets must be ignore")
            # Supervision may score an inference-time edge, but it must never
            # create one.  Checking this at the typed batch boundary prevents
            # a hand-built fixture/provider from bypassing the label-blind
            # ``correspondence_mask`` contract enforced by the real builder.
            matched_a = exact_a >= 0
            safe_b = exact_a.clamp(min=0, max=max(0, expected_b[1] - 1))
            allowed_a = correspondence.gather(2, safe_b[:, :, None]).squeeze(2)
            if (matched_a & ~allowed_a).any().item():
                raise ValueError(
                    "exact assignment target uses a disabled correspondence edge"
                )
        for name, value in (
            ("sample_index", self.sample_index),
            ("direction_index", self.direction_index),
        ):
            if value.dtype != torch.long or tuple(value.shape) != (candidate_count,):
                raise TypeError("{} must be int64 [N]".format(name))
        if self.candidate_valid.dtype != torch.bool or tuple(
            self.candidate_valid.shape
        ) != (candidate_count,):
            raise TypeError("candidate_valid must be bool [N]")
        if self.labels.dtype != torch.bool or tuple(self.labels.shape) != (batch_size,):
            raise TypeError("labels must be strict bool [B]")
        if self.direction_target.dtype != torch.long or tuple(
            self.direction_target.shape
        ) != (batch_size,):
            raise TypeError("direction_target must be int64 [B]")
        if self.direction_target_valid.dtype != torch.bool or tuple(
            self.direction_target_valid.shape
        ) != (batch_size,):
            raise TypeError("direction_target_valid must be bool [B]")
        if self.direction_slot_valid.dtype != torch.bool or tuple(
            self.direction_slot_valid.shape
        ) != (batch_size, 4):
            raise TypeError("direction_slot_valid must be bool [B, 4]")
        if self.geometry_valid.dtype != torch.bool or tuple(
            self.geometry_valid.shape
        ) != (batch_size,):
            raise TypeError("geometry_valid must be bool [B]")
        if len(self.candidate_ids) != candidate_count:
            raise ValueError("candidate_ids must contain one value per candidate")
        if len(self.sequence_length_buckets) != candidate_count:
            raise ValueError(
                "sequence_length_buckets must contain one value per candidate"
            )
        if not isinstance(self.complexity_receipt, BatchComplexityReceipt):
            raise TypeError("complexity_receipt must be BatchComplexityReceipt")
        if self.complexity_receipt.candidate_count != candidate_count:
            raise ValueError("complexity receipt candidate count differs from tensors")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be globally unique within a batch")
        for value in (
            self.sample_ids,
            self.geometry_cache_keys,
            self.sample_receipts,
        ):
            if len(value) != batch_size:
                raise ValueError("sample metadata length differs from batch size")
        if self.direction_names != DIRECTION_NAMES:
            raise ValueError("direction_names differ from the frozen geometry order")
        if candidate_count:
            if (
                ((self.sample_index < 0) | (self.sample_index >= batch_size))
                .any()
                .item()
            ):
                raise ValueError("sample_index is out of range")
            if ((self.direction_index < 0) | (self.direction_index >= 4)).any().item():
                raise ValueError("direction_index is out of range")
            expected_slots = self.direction_slot_valid[
                self.sample_index, self.direction_index
            ]
            if not expected_slots.all().item():
                raise ValueError("candidate refers to an invalid direction slot")
            if (
                not self.token_mask_a.any(dim=1).all().item()
                or not self.token_mask_b.any(dim=1).all().item()
            ):
                raise ValueError("every flattened candidate needs A and B tokens")
        if (self.direction_target[~self.direction_target_valid] != -1).any().item():
            raise ValueError("unknown direction targets must equal -1")
        known = self.direction_target[self.direction_target_valid]
        if known.numel() and ((known < 0) | (known >= 4)).any().item():
            raise ValueError("known direction target is out of range")
        if (self.direction_target_valid & ~self.labels).any().item():
            raise ValueError("negative pairs cannot carry direction supervision")
        if exact_a is not None:
            negative_candidate = ~self.labels.index_select(0, self.sample_index)
            if (
                exact_a[negative_candidate][self.token_mask_a[negative_candidate]]
                != -1
            ).any().item() or (
                exact_b[negative_candidate][self.token_mask_b[negative_candidate]]
                != -1
            ).any().item():
                raise ValueError(
                    "negative exact-seam samples must supervise every real token "
                    "as dustbin"
                )
        if not torch.equal(self.geometry_valid, self.direction_slot_valid.any(dim=1)):
            raise ValueError("geometry_valid must equal any valid direction slot")
        frozen_config = _deep_freeze_receipt(dict(self.config_provenance))
        observed_fingerprint = hashlib.sha256(
            _canonical_json(_deep_thaw_receipt(frozen_config))
        ).hexdigest()
        if observed_fingerprint != self.config_fingerprint:
            raise ValueError("config fingerprint differs from frozen provenance")
        object.__setattr__(self, "config_provenance", frozen_config)

    @property
    def batch_size(self) -> int:
        return int(self.coarse_a.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.local_a.shape[0])

    def model_inputs(self) -> Dict[str, Tensor]:
        """Return only the model-facing tensors; targets remain separate."""

        value = {
            "coarse_a": self.coarse_a,
            "coarse_b": self.coarse_b,
            "local_a": self.local_a,
            "local_b": self.local_b,
            "token_mask_a": self.token_mask_a,
            "token_mask_b": self.token_mask_b,
            "sample_index": self.sample_index,
            "direction_index": self.direction_index,
            "candidate_valid": self.candidate_valid,
        }
        if self.candidate_representation == KEYPOINT_REPRESENTATION:
            value["correspondence_mask"] = self.correspondence_mask
        return value

    def exact_loss_targets(self):
        """Return supervision-only tensors, never model-facing inputs."""

        if self.exact_assignment_target_a is None:
            return None
        from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
            ExactSeamTensorBatch,
        )

        return ExactSeamTensorBatch(
            assignment_target_a=self.exact_assignment_target_a,
            assignment_target_b=self.exact_assignment_target_b,
            sample_index=self.sample_index,
            direction_index=self.direction_index,
            candidate_ids=self.candidate_ids,
        )

    def to(self, device: torch.device) -> "RaggedGeometryBatch":
        tensor_fields = {
            "coarse_a": self.coarse_a.to(device),
            "coarse_b": self.coarse_b.to(device),
            "local_a": self.local_a.to(device),
            "local_b": self.local_b.to(device),
            "token_mask_a": self.token_mask_a.to(device),
            "token_mask_b": self.token_mask_b.to(device),
            "sample_index": self.sample_index.to(device),
            "direction_index": self.direction_index.to(device),
            "candidate_valid": self.candidate_valid.to(device),
            "correspondence_mask": self.correspondence_mask.to(device),
            "labels": self.labels.to(device),
            "direction_target": self.direction_target.to(device),
            "direction_target_valid": self.direction_target_valid.to(device),
            "direction_slot_valid": self.direction_slot_valid.to(device),
            "geometry_valid": self.geometry_valid.to(device),
        }
        if self.exact_assignment_target_a is not None:
            tensor_fields["exact_assignment_target_a"] = (
                self.exact_assignment_target_a.to(device)
            )
            tensor_fields["exact_assignment_target_b"] = (
                self.exact_assignment_target_b.to(device)
            )
        return RaggedGeometryBatch(
            **tensor_fields,
            candidate_ids=self.candidate_ids,
            sample_ids=self.sample_ids,
            geometry_cache_keys=self.geometry_cache_keys,
            sample_receipts=self.sample_receipts,
            config_fingerprint=self.config_fingerprint,
            config_provenance=self.config_provenance,
            complexity_receipt=self.complexity_receipt,
            sequence_length_buckets=self.sequence_length_buckets,
            direction_names=self.direction_names,
            bridge_version=self.bridge_version,
            candidate_representation=self.candidate_representation,
            exact_supervision_config=self.exact_supervision_config,
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reference_fragment_receipt_identity(
    reference: MaskMemberRef,
) -> Dict[str, Any]:
    """Path-free fallback identity when decoded canonical masks are unavailable."""

    if reference.content_sha256 is None:
        raise GeometryBatchError(
            "reference-only geometry key requires fragment content_sha256"
        )
    return {
        "content_sha256": reference.content_sha256,
        "threshold_rule": reference.threshold_rule,
    }


def geometry_cache_key(
    record: TrainingPairRecord,
    config: GeometryBatchConfig,
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
) -> str:
    """Hash ordered, path-free fragment identities, never pair supervision.

    Decoded canonical masks produce the same content/config/threshold/recipe
    identities used by the real disk cache.  The reference-only fallback is a
    portable receipt helper and requires member content hashes; it never uses
    archive paths, fragment IDs, pair IDs, labels, or split membership.
    """

    if not isinstance(record, TrainingPairRecord):
        raise TypeError("record must be TrainingPairRecord")
    if not isinstance(config, GeometryBatchConfig):
        raise TypeError("config must be GeometryBatchConfig")
    if (mask_a is None) != (mask_b is None):
        raise ValueError("mask_a and mask_b must be supplied together")
    if mask_a is None:
        fragment_a: Mapping[str, Any] = _reference_fragment_receipt_identity(
            record.fragment_a
        )
        fragment_b: Mapping[str, Any] = _reference_fragment_receipt_identity(
            record.fragment_b
        )
        identity_kind = "member_content_receipt"
    else:
        fragment_a = fragment_cache_identity(
            mask_a,
            record.fragment_a.threshold_rule,
            config.geometry,
        ).to_dict()
        fragment_b = fragment_cache_identity(
            mask_b,
            record.fragment_b.threshold_rule,
            config.geometry,
        ).to_dict()
        identity_kind = "canonical_mask_fragment_cache"
    payload = {
        "bridge_version": GEOMETRY_BATCH_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_recipe_version": CACHE_RECIPE_VERSION,
        "identity_kind": identity_kind,
        "geometry_config": asdict(config.geometry),
        "fragment_a": fragment_a,
        "fragment_b": fragment_b,
    }
    return "geometry/sha256/" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validated_mask(value: Any, *, name: str, max_pixels: int) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise GeometryBatchError(
            "{} cannot be converted to an array".format(name)
        ) from exc
    if mask.ndim != 2 or mask.size == 0:
        raise GeometryBatchError("{} must be a non-empty 2D mask".format(name))
    if int(mask.size) > max_pixels:
        raise GeometryBatchError("{} exceeds the input pixel bound".format(name))
    if mask.dtype != np.bool_:
        raise GeometryBatchError(
            "{} must be the loader's canonical bool mask".format(name)
        )
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _interpolate_coarse(
    source: Tensor,
    size: Tuple[int, int],
    mode: str,
) -> Tensor:
    if mode == "bilinear":
        return functional.interpolate(
            source,
            size=size,
            mode=mode,
            align_corners=False,
        )
    return functional.interpolate(source, size=size, mode=mode)


def _largest_coarse_component(mask: np.ndarray, connectivity: int) -> np.ndarray:
    """Select one component by area, then lexicographic bbox tie-break."""

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(mask, structure=structure)
    if count < 1:
        return np.zeros_like(mask, dtype=np.bool_)
    areas = np.bincount(labels.reshape(-1), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)
    candidates = []
    for label in range(1, count + 1):
        component_slice = objects[label - 1]
        if component_slice is None:
            continue
        bbox = (
            int(component_slice[0].start),
            int(component_slice[1].start),
            int(component_slice[0].stop),
            int(component_slice[1].stop),
        )
        candidates.append((-int(areas[label]), bbox, label))
    if not candidates:  # pragma: no cover - guarded by ndimage.label
        raise GeometryBatchError("coarse component labeling lost foreground")
    _, _, selected = min(candidates)
    return np.ascontiguousarray(labels == selected, dtype=np.bool_)


def preprocess_coarse_mask(mask: np.ndarray, config: GeometryBatchConfig) -> Tensor:
    """Create a shortcut-safe whole-mask tensor for the coarse Siamese.

    The new synthetic masks share an 800x800 source canvas.  Resizing that
    canvas directly exposes original fragment placement, allowing a model to
    learn proximity instead of complementary shape.  The default contract
    crops each fragment independently to its foreground bounding box,
    preserves aspect ratio, and centres it in a fixed letterbox.  The previous
    full-canvas stretch remains available only as an explicitly named legacy
    ablation.
    """

    if not isinstance(config, GeometryBatchConfig):
        raise TypeError("config must be GeometryBatchConfig")
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_ or value.size == 0:
        raise GeometryBatchError("coarse input must be a non-empty 2D bool mask")
    foreground = np.argwhere(value)
    if foreground.size == 0:
        # Preserve the bridge's sample-aligned fail-closed contract: geometry
        # will mark this row invalid and expose no local candidates.  Returning
        # a neutral canvas avoids aborting unrelated records in the same batch.
        output = torch.zeros(
            (1, *config.coarse_output_size),
            dtype=torch.float32,
        )
    elif config.coarse_preprocess_mode == "full_canvas_stretch_legacy":
        source = torch.from_numpy(value.astype(np.float32, copy=True))[None, None]
        output = _interpolate_coarse(
            source,
            config.coarse_output_size,
            config.coarse_resize_mode,
        )[0]
    else:
        component = _largest_coarse_component(
            value, config.coarse_component_connectivity
        )
        foreground = np.argwhere(component)
        if foreground.size == 0:  # pragma: no cover - guarded above
            raise GeometryBatchError("coarse largest-component selection failed")
        row_min, column_min = foreground.min(axis=0)
        row_max, column_max = foreground.max(axis=0) + 1
        crop = component[row_min:row_max, column_min:column_max]
        target_rows, target_columns = config.coarse_output_size
        content_rows = max(
            1,
            min(
                target_rows,
                int(round(target_rows * config.coarse_content_fraction)),
            ),
        )
        content_columns = max(
            1,
            min(
                target_columns,
                int(round(target_columns * config.coarse_content_fraction)),
            ),
        )
        scale = min(content_rows / crop.shape[0], content_columns / crop.shape[1])
        resized_rows = max(1, min(content_rows, int(round(crop.shape[0] * scale))))
        resized_columns = max(
            1,
            min(content_columns, int(round(crop.shape[1] * scale))),
        )
        source = torch.from_numpy(crop.astype(np.float32, copy=True))[None, None]
        resized = _interpolate_coarse(
            source,
            (resized_rows, resized_columns),
            config.coarse_resize_mode,
        )[0]
        output = torch.zeros(
            (1, target_rows, target_columns),
            dtype=torch.float32,
        )
        top = (target_rows - resized_rows) // 2
        left = (target_columns - resized_columns) // 2
        output[:, top : top + resized_rows, left : left + resized_columns] = resized
    if output.dtype != torch.float32 or output.layout != torch.strided:
        raise GeometryBatchError("coarse resize changed the numeric tensor contract")
    if not torch.isfinite(output).all().item():
        raise GeometryBatchError("coarse resize produced non-finite values")
    if (output < -_COARSE_PRECLAMP_RANGE_TOLERANCE).any().item() or (
        output > 1.0 + _COARSE_PRECLAMP_RANGE_TOLERANCE
    ).any().item():
        raise GeometryBatchError("coarse resize produced out-of-range values")
    # Bilinear interpolation of binary float32 input can overshoot either
    # endpoint by one ULP.  Canonicalize that bounded round-off here so every
    # consumer, including the stricter model boundary, sees exact [0, 1].
    return output.clamp(min=0.0, max=1.0).contiguous()


def _resize_coarse(mask: np.ndarray, config: GeometryBatchConfig) -> Tensor:
    """Backward-compatible private alias for the frozen preprocessing API."""

    return preprocess_coarse_mask(mask, config)


def _direction_target(record: TrainingPairRecord) -> Tuple[int, bool]:
    target = data_direction_to_target_index(record.direction_b_wrt_a)
    if not record.label and target != -1:
        raise GeometryBatchError("negative record carries a direction target")
    return target, bool(record.label and target >= 0)


def _sequence_length_bucket(length: int) -> str:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise GeometryBatchError("candidate sequence length must be positive")
    for edge in SEQUENCE_LENGTH_BUCKET_EDGES:
        if length <= edge:
            return "le_{:04d}".format(edge)
    return "gt_{:04d}".format(SEQUENCE_LENGTH_BUCKET_EDGES[-1])


def _candidate_quadratic_cost(length_a: int, length_b: int) -> Tuple[int, int, int]:
    attention = (length_a + length_b) ** 2
    affinity = length_a * length_b
    sinkhorn = (length_a + 1) * (length_b + 1)
    return attention, affinity, sinkhorn


def _enforce_candidate_quadratic_bounds(
    length_a: int, length_b: int, settings: GeometryBatchConfig
) -> None:
    attention, affinity, sinkhorn = _candidate_quadratic_cost(length_a, length_b)
    checks = (
        (
            attention,
            settings.max_attention_score_elements_per_candidate,
            "candidate attention score elements exceed bound",
        ),
        (
            affinity,
            settings.max_affinity_elements_per_candidate,
            "candidate affinity elements exceed bound",
        ),
        (
            sinkhorn,
            settings.max_sinkhorn_elements_per_candidate,
            "candidate Sinkhorn elements exceed bound",
        ),
    )
    for observed, limit, message in checks:
        if observed > limit:
            raise GeometryBatchError(message)


def build_geometry_batch(
    records: Sequence[TrainingPairRecord],
    mask_loader: Callable[[MaskMemberRef], np.ndarray],
    config: Optional[GeometryBatchConfig] = None,
    *,
    geometry_artifact_cache: Optional[GeometryArtifactCache] = None,
    candidate_representation: str = MULTIRUN_REPRESENTATION,
    keypoint_config: Optional[ContourKeypointConfig] = None,
    exact_seam_supervision: bool = False,
    exact_seam_config: Optional[Any] = None,
    keypoint_candidate_observer: Optional[
        Callable[[Tuple[KeypointPairCandidate, ...]], None]
    ] = None,
) -> RaggedGeometryBatch:
    """Load masks and build the four-direction flattened candidate contract.

    Geometry failures invalidate their sample's four slots without leaking a
    partial candidate set.  Archive/decode/cache-integrity exceptions are not
    swallowed: the caller receives no batch and can quarantine the source or
    cache entry explicitly.  The cache receives only a canonical mask,
    threshold rule, and geometry config -- never labels, pair IDs, or paths.
    """

    settings = config or GeometryBatchConfig()
    if not isinstance(settings, GeometryBatchConfig):
        raise TypeError("config must be GeometryBatchConfig")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a finite sequence")
    if not records:
        raise GeometryBatchError("records cannot be empty")
    if len(records) > settings.max_batch_size:
        raise GeometryBatchError("record batch exceeds max_batch_size")
    if not callable(mask_loader):
        raise TypeError("mask_loader must be callable")
    if geometry_artifact_cache is not None and not isinstance(
        geometry_artifact_cache, GeometryArtifactCache
    ):
        raise TypeError("geometry_artifact_cache must be GeometryArtifactCache")
    if candidate_representation not in LOCAL_CANDIDATE_REPRESENTATIONS:
        raise ValueError("unsupported local candidate representation")
    if type(exact_seam_supervision) is not bool:
        raise TypeError("exact_seam_supervision must be bool")
    if exact_seam_supervision and candidate_representation != KEYPOINT_REPRESENTATION:
        raise GeometryBatchError("exact seam supervision requires keypoint candidates")
    if not exact_seam_supervision and exact_seam_config is not None:
        raise GeometryBatchError("exact seam config requires exact supervision")
    if keypoint_candidate_observer is not None:
        if candidate_representation != KEYPOINT_REPRESENTATION:
            raise GeometryBatchError(
                "keypoint candidate observer requires keypoint candidates"
            )
        if not callable(keypoint_candidate_observer):
            raise TypeError("keypoint_candidate_observer must be callable")
    exact_settings = None
    if exact_seam_supervision:
        from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
            ExactSeamTargetConfig,
        )

        exact_settings = exact_seam_config or ExactSeamTargetConfig()
        if not isinstance(exact_settings, ExactSeamTargetConfig):
            raise TypeError("exact_seam_config must be ExactSeamTargetConfig")
    selector = keypoint_config or ContourKeypointConfig()
    if not isinstance(selector, ContourKeypointConfig):
        raise TypeError("keypoint_config must be ContourKeypointConfig")
    if (
        candidate_representation == KEYPOINT_REPRESENTATION
        and geometry_artifact_cache is None
    ):
        raise GeometryBatchError(
            "contour-keypoint representation requires fragment geometry cache"
        )
    for record in records:
        if not isinstance(record, TrainingPairRecord):
            raise TypeError("every record must be TrainingPairRecord")

    coarse_a_values = []
    coarse_b_values = []
    flattened = []
    candidate_ids = []
    sample_indices = []
    direction_indices = []
    labels = []
    direction_targets = []
    direction_target_valid = []
    slot_rows = []
    sample_ids = []
    cache_keys = []
    receipts = []
    flattened_exact_targets = []
    # A batch may contain repeated samples, reverse pairs, or a fragment shared
    # by several pairs.  Decode each physical archive member and rehydrate each
    # content-addressed geometry artifact once per batch rather than once per
    # endpoint occurrence.
    mask_memo: Dict[MaskMemberRef, np.ndarray] = {}
    fragment_memo: Dict[str, FragmentGeometryCacheLookup] = {}
    keypoint_memo = {}

    def load_mask(reference: MaskMemberRef, name: str) -> np.ndarray:
        cached = mask_memo.get(reference)
        if cached is not None:
            return cached
        loaded = _validated_mask(
            mask_loader(reference),
            name=name,
            max_pixels=settings.max_input_pixels_per_mask,
        )
        mask_memo[reference] = loaded
        return loaded

    def load_fragment(
        mask: np.ndarray, reference: MaskMemberRef
    ) -> FragmentGeometryCacheLookup:
        assert geometry_artifact_cache is not None
        identity = fragment_cache_identity(
            mask, reference.threshold_rule, settings.geometry
        )
        cached = fragment_memo.get(identity.key)
        if cached is not None:
            return cached
        loaded = load_or_build_fragment_geometry(
            mask,
            reference.threshold_rule,
            settings.geometry,
            geometry_artifact_cache,
        )
        if loaded.identity != identity:
            raise GeometryBatchError("fragment cache identity changed during lookup")
        fragment_memo[identity.key] = loaded
        return loaded

    def load_keypoints(lookup: FragmentGeometryCacheLookup):
        cached = keypoint_memo.get(lookup.identity.key)
        if cached is not None:
            return cached
        if not lookup.result.ok or lookup.result.artifact is None:
            raise GeometryBatchError("keypoint source fragment geometry is invalid")
        selected = build_fragment_keypoints(lookup.result.artifact, selector)
        keypoint_memo[lookup.identity.key] = selected
        return selected

    for sample_index, record in enumerate(records):
        mask_a = load_mask(record.fragment_a, "fragment_a")
        mask_b = load_mask(record.fragment_b, "fragment_b")
        pair_seam = None
        if exact_seam_supervision:
            from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
                ExactSeamSupervisionError,
                extract_aligned_mask_seam,
            )

            pair_seam = extract_aligned_mask_seam(mask_a, mask_b)
            if record.label and pair_seam is None:
                raise ExactSeamSupervisionError(
                    "positive training record has no exact aligned-mask seam"
                )
            if not record.label and pair_seam is not None:
                raise ExactSeamSupervisionError(
                    "negative training record exposes an exact aligned-mask seam"
                )
        coarse_a_values.append(_resize_coarse(mask_a, settings))
        coarse_b_values.append(_resize_coarse(mask_b, settings))
        labels.append(record.label)
        target, target_valid = _direction_target(record)
        direction_targets.append(target)
        direction_target_valid.append(target_valid)
        sample_ids.append(record.pair_id)
        cache_key = geometry_cache_key(
            record,
            settings,
            mask_a=mask_a,
            mask_b=mask_b,
        )
        cache_keys.append(cache_key)

        # Critical anti-leakage boundary: neither ``record.label`` nor
        # ``record.direction_b_wrt_a`` is passed into candidate generation.
        if geometry_artifact_cache is None:
            result = build_pair_candidates(
                mask_a,
                mask_b,
                direction_b_wrt_a=None,
                config=settings.geometry,
            )
        else:
            fragment_a = load_fragment(mask_a, record.fragment_a)
            fragment_b = load_fragment(mask_b, record.fragment_b)
            result = combine_fragment_results(
                fragment_a.result,
                fragment_b.result,
                direction_b_wrt_a=None,
                config=settings.geometry,
            )
        if candidate_representation == KEYPOINT_REPRESENTATION:
            if not result.ok:
                keypoint_result = None
                candidates = ()
                emitted_directions = ()
            else:
                assert geometry_artifact_cache is not None
                keypoint_result = build_keypoint_pair_candidates(
                    load_keypoints(fragment_a),
                    load_keypoints(fragment_b),
                    selector,
                )
                candidates = keypoint_result.candidates
                emitted_directions = tuple(
                    candidate.direction.value for candidate in candidates
                )
                if emitted_directions != DIRECTION_NAMES:
                    raise GeometryBatchError(
                        "keypoint representation requires all four upright directions"
                    )
            geometry_quality = {
                "candidate_representation": KEYPOINT_REPRESENTATION,
                "keypoint_config": asdict(selector),
                "base_multirun_geometry": result.quality.to_dict(),
                "keypoint_complexity": (
                    None
                    if keypoint_result is None
                    else asdict(keypoint_result.complexity)
                ),
            }
        else:
            candidates = result.candidates
            emitted_directions = result.quality.emitted_directions
            geometry_quality = result.quality.to_dict()
        slot_valid = [False, False, False, False]
        if result.status is GeometryStatus.OK:
            if len(candidates) > settings.max_candidates_per_sample:
                raise GeometryBatchError("sample exceeds max_candidates_per_sample")
            for candidate in candidates:
                slot = DEFAULT_DIRECTION_ORDER.index(candidate.direction)
                slot_valid[slot] = True
                if candidate.direction not in DEFAULT_DIRECTION_ORDER:
                    raise GeometryBatchError("geometry emitted an unknown direction")
                length_a = (
                    len(candidate.tokens_a)
                    if isinstance(candidate, KeypointPairCandidate)
                    else candidate.sequence_a.length
                )
                length_b = (
                    len(candidate.tokens_b)
                    if isinstance(candidate, KeypointPairCandidate)
                    else candidate.sequence_b.length
                )
                if (
                    length_a > settings.max_sequence_length
                    or length_b > settings.max_sequence_length
                ):
                    raise GeometryBatchError("candidate exceeds max_sequence_length")
                _enforce_candidate_quadratic_bounds(length_a, length_b, settings)
                if candidate.patches_a.shape[1:] != (
                    len(CHANNEL_ORDER),
                    *settings.geometry.output_size,
                ) or candidate.patches_b.shape[1:] != (
                    len(CHANNEL_ORDER),
                    *settings.geometry.output_size,
                ):
                    raise GeometryBatchError(
                        "candidate patch shape disagrees with config"
                    )
                # Repeated sampling with replacement may place the same pair
                # in a batch more than once.  The stable batch position keeps
                # flattened candidate IDs unique without polluting cache keys.
                global_id = "sample/{:06d}:{}:{}".format(
                    sample_index, record.pair_id, candidate.candidate_id
                )
                if len(global_id.encode("utf-8")) > settings.max_candidate_id_bytes:
                    raise GeometryBatchError("candidate id exceeds byte bound")
                flattened.append(candidate)
                if exact_seam_supervision:
                    from staging.pairwise_v0_2.pairwise_data.exact_seam_supervision import (
                        build_keypoint_assignment_target,
                    )

                    assert isinstance(candidate, KeypointPairCandidate)
                    flattened_exact_targets.append(
                        build_keypoint_assignment_target(
                            candidate,
                            pair_seam,
                            pair_is_adjacent=record.label,
                            config=exact_settings,
                        )
                    )
                candidate_ids.append(global_id)
                sample_indices.append(sample_index)
                direction_indices.append(
                    DEFAULT_DIRECTION_ORDER.index(candidate.direction)
                )
                if len(flattened) > settings.max_candidates_per_batch:
                    raise GeometryBatchError("batch exceeds max_candidates_per_batch")
        elif result.candidates or result.direction_groups:
            raise GeometryBatchError(
                "failed geometry result exposed partial candidates"
            )

        slot_tuple = tuple(slot_valid)
        slot_rows.append(slot_tuple)
        receipts.append(
            GeometrySampleReceipt(
                pair_id=record.pair_id,
                geometry_cache_key=cache_key,
                dataset_id=record.dataset_id,
                canonical_group_id=record.canonical_group_id,
                component_id=record.component_id,
                geometry_status=result.status.value,
                geometry_failure_reason=result.failure_reason,
                candidate_count=len(candidates),
                direction_slot_valid=slot_tuple,  # type: ignore[arg-type]
                emitted_directions=emitted_directions,
                geometry_quality=geometry_quality,
            )
        )

    coarse_a = torch.stack(coarse_a_values, dim=0)
    coarse_b = torch.stack(coarse_b_values, dim=0)
    patch_height, patch_width = settings.geometry.output_size
    candidate_count = len(flattened)
    if keypoint_candidate_observer is not None:
        # Additive readout-only seam: the observer sees the immutable,
        # label-blind candidates in exactly the order used to build tensors.
        # Nothing returned by it can enter model_inputs, prepared digests, or
        # the persisted RaggedGeometryBatch contract.
        keypoint_candidate_observer(tuple(flattened))

    def candidate_lengths(candidate: Any) -> Tuple[int, int]:
        if isinstance(candidate, KeypointPairCandidate):
            return len(candidate.tokens_a), len(candidate.tokens_b)
        return candidate.sequence_a.length, candidate.sequence_b.length

    max_a = max((candidate_lengths(candidate)[0] for candidate in flattened), default=1)
    max_b = max((candidate_lengths(candidate)[1] for candidate in flattened), default=1)
    local_elements = (
        candidate_count
        * len(CHANNEL_ORDER)
        * patch_height
        * patch_width
        * (max_a + max_b)
    )
    if local_elements > settings.max_local_tensor_elements:
        raise GeometryBatchError("padded local tensors exceed element bound")
    sequence_length_buckets = tuple(
        _sequence_length_bucket(max(candidate_lengths(candidate)))
        for candidate in flattened
    )
    bucket_counts: Dict[str, int] = {}
    for bucket in sequence_length_buckets:
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    if candidate_count:
        attention_per_candidate, affinity_per_candidate, sinkhorn_per_candidate = (
            _candidate_quadratic_cost(max_a, max_b)
        )
        attention_elements = candidate_count * attention_per_candidate
        affinity_elements = candidate_count * affinity_per_candidate
        sinkhorn_elements = candidate_count * sinkhorn_per_candidate
    else:
        attention_elements = 0
        affinity_elements = 0
        sinkhorn_elements = 0
    batch_checks = (
        (
            attention_elements,
            settings.max_attention_score_elements_per_batch,
            "padded attention score elements exceed batch bound",
        ),
        (
            affinity_elements,
            settings.max_affinity_elements_per_batch,
            "padded affinity elements exceed batch bound",
        ),
        (
            sinkhorn_elements,
            settings.max_sinkhorn_elements_per_batch,
            "padded Sinkhorn elements exceed batch bound",
        ),
    )
    for observed, limit, message in batch_checks:
        if observed > limit:
            raise GeometryBatchError(message)
    complexity_receipt = BatchComplexityReceipt(
        candidate_count=candidate_count,
        padded_sequence_a=max_a if candidate_count else 0,
        padded_sequence_b=max_b if candidate_count else 0,
        local_tensor_elements=local_elements,
        attention_score_elements_per_head=attention_elements,
        affinity_elements=affinity_elements,
        sinkhorn_elements=sinkhorn_elements,
        sequence_bucket_counts=bucket_counts,
    )

    local_a = torch.zeros(
        (candidate_count, max_a, len(CHANNEL_ORDER), patch_height, patch_width),
        dtype=torch.float32,
    )
    local_b = torch.zeros(
        (candidate_count, max_b, len(CHANNEL_ORDER), patch_height, patch_width),
        dtype=torch.float32,
    )
    token_mask_a = torch.zeros((candidate_count, max_a), dtype=torch.bool)
    token_mask_b = torch.zeros((candidate_count, max_b), dtype=torch.bool)
    correspondence_mask = torch.zeros((candidate_count, max_a, max_b), dtype=torch.bool)
    exact_target_a = (
        torch.full((candidate_count, max_a), -2, dtype=torch.long)
        if exact_seam_supervision
        else None
    )
    exact_target_b = (
        torch.full((candidate_count, max_b), -2, dtype=torch.long)
        if exact_seam_supervision
        else None
    )
    for index, candidate in enumerate(flattened):
        length_a, length_b = candidate_lengths(candidate)
        local_a[index, :length_a].copy_(
            torch.from_numpy(np.asarray(candidate.patches_a, dtype=np.float32).copy())
        )
        local_b[index, :length_b].copy_(
            torch.from_numpy(np.asarray(candidate.patches_b, dtype=np.float32).copy())
        )
        if isinstance(candidate, KeypointPairCandidate):
            token_mask_a[index, :length_a] = True
            token_mask_b[index, :length_b] = True
            correspondence_mask[index, :length_a, :length_b].copy_(
                torch.from_numpy(
                    np.asarray(candidate.correspondence_mask, dtype=np.bool_).copy()
                )
            )
            if exact_seam_supervision:
                target = flattened_exact_targets[index]
                assert exact_target_a is not None and exact_target_b is not None
                exact_target_a[index, :length_a].copy_(
                    torch.from_numpy(target.assignment_target_a.copy())
                )
                exact_target_b[index, :length_b].copy_(
                    torch.from_numpy(target.assignment_target_b.copy())
                )
        else:
            token_mask_a[index, :length_a] = torch.from_numpy(
                np.asarray(candidate.valid_a, dtype=np.bool_).copy()
            )
            token_mask_b[index, :length_b] = torch.from_numpy(
                np.asarray(candidate.valid_b, dtype=np.bool_).copy()
            )
            correspondence_mask[index, :length_a, :length_b] = (
                token_mask_a[index, :length_a, None]
                & token_mask_b[index, None, :length_b]
            )

    provenance = settings.provenance_dict()
    if candidate_representation == KEYPOINT_REPRESENTATION:
        # Keep the established multi-run fingerprint byte-for-byte stable,
        # while committing every output-affecting selector setting for the
        # additive keypoint representation.
        provenance = {
            **provenance,
            "base_geometry_config_sha256": settings.fingerprint,
            "local_candidate_representation": candidate_representation,
            "contour_keypoint_config": asdict(selector),
        }
        fingerprint = hashlib.sha256(_canonical_json(provenance)).hexdigest()
    else:
        fingerprint = settings.fingerprint
    return RaggedGeometryBatch(
        coarse_a=coarse_a,
        coarse_b=coarse_b,
        local_a=local_a,
        local_b=local_b,
        token_mask_a=token_mask_a,
        token_mask_b=token_mask_b,
        sample_index=torch.tensor(sample_indices, dtype=torch.long),
        direction_index=torch.tensor(direction_indices, dtype=torch.long),
        candidate_valid=torch.ones(candidate_count, dtype=torch.bool),
        labels=torch.tensor(labels, dtype=torch.bool),
        direction_target=torch.tensor(direction_targets, dtype=torch.long),
        direction_target_valid=torch.tensor(direction_target_valid, dtype=torch.bool),
        direction_slot_valid=torch.tensor(slot_rows, dtype=torch.bool),
        geometry_valid=torch.tensor(
            [any(slots) for slots in slot_rows], dtype=torch.bool
        ),
        candidate_ids=tuple(candidate_ids),
        sample_ids=tuple(sample_ids),
        geometry_cache_keys=tuple(cache_keys),
        sample_receipts=tuple(receipts),
        config_fingerprint=fingerprint,
        config_provenance=provenance,
        complexity_receipt=complexity_receipt,
        sequence_length_buckets=sequence_length_buckets,
        candidate_representation=candidate_representation,
        correspondence_mask=correspondence_mask,
        exact_assignment_target_a=exact_target_a,
        exact_assignment_target_b=exact_target_b,
        exact_supervision_config=(
            {
                "schema_version": "exact-seam-ragged-supervision/v0.1",
                "target_encoding": {
                    "ignore": -2,
                    "dustbin": -1,
                    "matched": "non_negative_opposite_token_index",
                },
                "target_config": asdict(exact_settings),
                "model_inputs_include_targets": False,
            }
            if exact_seam_supervision
            else None
        ),
    )


__all__ = [
    "BatchComplexityReceipt",
    "COARSE_NUMERIC_CONTRACT",
    "DATA_DIRECTION_TO_INDEX",
    "DATA_DIRECTION_TO_PAIR_DIRECTION",
    "DIRECTION_NAMES",
    "GEOMETRY_BATCH_VERSION",
    "KEYPOINT_REPRESENTATION",
    "LOCAL_CANDIDATE_REPRESENTATIONS",
    "MULTIRUN_REPRESENTATION",
    "GeometryBatchConfig",
    "GeometryBatchError",
    "GeometrySampleReceipt",
    "RaggedGeometryBatch",
    "SEQUENCE_LENGTH_BUCKET_EDGES",
    "build_geometry_batch",
    "data_direction_to_pair_direction",
    "data_direction_to_target_index",
    "geometry_cache_key",
    "inverse_data_direction",
    "inverse_direction_index",
    "preprocess_coarse_mask",
]
