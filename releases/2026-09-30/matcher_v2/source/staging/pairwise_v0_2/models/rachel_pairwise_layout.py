"""Frozen coarse/sliding pair scores with independently decoded pair layout.

The Rachel classifier still consumes its original transport dispersion.  New
layout estimates never replace that evidence or feed back into its scores.
Every pair is decoded by default, including low-scoring pairs; callers can
explicitly disable layout or provide a mask unrelated to the classifier.

``t_a_to_b_rc`` maps A-local coordinates to B-local coordinates.  The offset
used to draw B on A's canvas is its negative: ``offset_b_in_a_rc``.  Rotation
is fixed to zero throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .rachel_n512 import RachelN512Output, RachelN512Pairwise
from .translation_layout import (
    TranslationLayoutConfig,
    TranslationLayoutResult,
    estimate_translation_layout,
)


@dataclass(frozen=True)
class BatchedTranslationLayout:
    t_a_to_b_rc: Tensor
    valid: Tensor
    computed: Tensor
    results: Tuple[Optional[TranslationLayoutResult], ...]

    @property
    def offset_b_in_a_rc(self) -> Tensor:
        """Translation for drawing B in A's coordinate frame (row, column)."""
        return -self.t_a_to_b_rc

    @property
    def diagnostics(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        for result in self.results:
            if result is None:
                rows.append({"computed": False, "reason": "not_requested"})
                continue
            rows.append({
                "computed": True,
                "valid": result.valid,
                "reason": result.reason,
                "candidate_count": result.candidate_count,
                "inlier_count": result.inlier_count,
                "inlier_fraction": result.inlier_fraction,
                "weighted_inlier_fraction": result.weighted_inlier_fraction,
                "residual_px": result.residual_px,
                "runner_up_support_ratio": result.runner_up_support_ratio,
                "support_weight": result.support_weight,
                "runner_up_support_weight": result.runner_up_support_weight,
            })
        return tuple(rows)


@dataclass(frozen=True)
class RachelPairwiseLayoutOutput:
    pair_output: RachelN512Output
    layouts: Dict[str, BatchedTranslationLayout]

    # These properties return the original tensors, without recalibration,
    # thresholding, or replacing the classifier's old geometry evidence.
    @property
    def coarse_logit(self) -> Tensor:
        return self.pair_output.coarse_logit

    @property
    def coarse_probability(self) -> Tensor:
        return self.pair_output.coarse_probability

    @property
    def local_logit(self) -> Tensor:
        return self.pair_output.local_logit

    @property
    def local_probability(self) -> Tensor:
        return self.pair_output.local_probability

    @property
    def fused_logit(self) -> Tensor:
        return self.pair_output.fused_logit

    @property
    def fused_probability(self) -> Tensor:
        return self.pair_output.fused_probability


def decode_pair_layouts(
    pair_output: RachelN512Output,
    points_a_rc: Tensor,
    points_b_rc: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    *,
    layout_configs: Optional[Mapping[str, TranslationLayoutConfig]] = None,
    layout_mask: Optional[Tensor] = None,
) -> RachelPairwiseLayoutOutput:
    """Decode one or several layout ablations from the same frozen evidence.

    ``layout_mask`` is an explicit caller request, never a score threshold.
    An invalid or unrequested layout has NaN coordinates and ``valid=False``;
    it does not change pair validity or the original pair scores.  The CPU
    geometric decoder has no gradients.  Batched coordinates are returned on
    the same device and with the same dtype as ``points_a_rc``.
    """
    configs = (
        {"translation_consensus": TranslationLayoutConfig()}
        if layout_configs is None else dict(layout_configs)
    )
    for name, config in configs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("layout names must be non-empty strings")
        if not isinstance(config, TranslationLayoutConfig):
            raise TypeError("layout configurations must be TranslationLayoutConfig")
    if points_a_rc.ndim != 3 or points_a_rc.shape[-1] != 2:
        raise ValueError("points_a_rc must have shape [B,N,2]")
    if points_b_rc.ndim != 3 or points_b_rc.shape[-1] != 2:
        raise ValueError("points_b_rc must have shape [B,M,2]")
    if not points_a_rc.is_floating_point() or not points_b_rc.is_floating_point():
        raise TypeError("point coordinates must be floating-point tensors")
    batch = points_a_rc.shape[0]
    if points_b_rc.shape[0] != batch:
        raise ValueError("A and B batch sizes differ")
    for mask, points, name in ((valid_a, points_a_rc, "valid_a"),
                               (valid_b, points_b_rc, "valid_b")):
        if mask.dtype != torch.bool or mask.shape != points.shape[:2]:
            raise ValueError(name + " must be Boolean [B,N] aligned with points")
    if layout_mask is None:
        requested = torch.ones(batch, dtype=torch.bool)
    else:
        if layout_mask.dtype != torch.bool or layout_mask.shape != (batch,):
            raise ValueError("layout_mask must be Boolean [B]")
        requested = layout_mask.detach().cpu()

    evidence_shape = (batch, points_a_rc.shape[1], points_b_rc.shape[1])
    arrays = {}
    for source in {"affinity" if config.score_mode == "dual_softmax"
                   else "assignment" for config in configs.values()}:
        value = getattr(pair_output, source)
        if value.shape != evidence_shape:
            raise ValueError(source + " must have shape [B,N,M]")
        arrays[source] = value.detach().cpu().double().numpy()
    pa = points_a_rc.detach().cpu().double().numpy()
    pb = points_b_rc.detach().cpu().double().numpy()
    va = valid_a.detach().cpu().numpy()
    vb = valid_b.detach().cpu().numpy()
    layouts = {}
    for name, config in configs.items():
        translation = points_a_rc.new_full((batch, 2), float("nan"))
        valid = torch.zeros(batch, dtype=torch.bool, device=points_a_rc.device)
        confidence = arrays["affinity" if config.score_mode == "dual_softmax"
                            else "assignment"]
        results = []
        for index in range(batch):
            if not bool(requested[index]):
                results.append(None)
                continue
            result = estimate_translation_layout(
                pa[index], pb[index], confidence[index], va[index], vb[index],
                config=config,
            )
            results.append(result)
            if result.valid:
                translation[index] = torch.as_tensor(
                    result.t_a_to_b_rc,
                    dtype=translation.dtype,
                    device=translation.device,
                )
                valid[index] = True
        layouts[name] = BatchedTranslationLayout(
            t_a_to_b_rc=translation,
            valid=valid,
            computed=requested.to(device=points_a_rc.device).clone(),
            results=tuple(results),
        )
    return RachelPairwiseLayoutOutput(pair_output=pair_output, layouts=layouts)


class RachelPairwiseLayout(nn.Module):
    """Run a frozen Rachel classifier and independent translation decoders."""

    def __init__(
        self,
        pair_model: RachelN512Pairwise,
        layout_configs: Optional[Mapping[str, TranslationLayoutConfig]] = None,
        *,
        precision: Optional[str] = None,
    ) -> None:
        super().__init__()
        if precision not in (None, "fp32", "bf16"):
            raise ValueError("precision must be None, 'fp32', or 'bf16'")
        self.pair_model = pair_model
        self.layout_configs = None if layout_configs is None else dict(layout_configs)
        # None preserves the caller's context; the factory freezes this field
        # to the completed training receipt unless explicitly overridden.
        self.inference_precision = precision
        self.pair_model.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "RachelPairwiseLayout":
        super().train(mode)
        self.pair_model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        mask_a: Tensor,
        mask_b: Tensor,
        points_rc_a: Tensor,
        points_rc_b: Tensor,
        contour_valid_a: Tensor,
        contour_valid_b: Tensor,
        *,
        compute_layout: bool = True,
        layout_mask: Optional[Tensor] = None,
    ) -> RachelPairwiseLayoutOutput:
        if type(compute_layout) is not bool:
            raise TypeError("compute_layout must be bool")
        context = (
            nullcontext() if self.inference_precision is None else torch.autocast(
                device_type=mask_a.device.type,
                dtype=torch.bfloat16,
                enabled=self.inference_precision == "bf16",
            )
        )
        with context:
            pair_output = self.pair_model(
                mask_a, mask_b, points_rc_a, points_rc_b,
                contour_valid_a, contour_valid_b,
            )
        if not compute_layout:
            return RachelPairwiseLayoutOutput(pair_output, {})
        return decode_pair_layouts(
            pair_output, points_rc_a, points_rc_b,
            contour_valid_a, contour_valid_b,
            layout_configs=self.layout_configs, layout_mask=layout_mask,
        )


def load_frozen_pairwise_layout(
    run_directory,
    validation_freeze_path,
    device: str = "cpu",
    *,
    precision: Optional[str] = None,
) -> RachelPairwiseLayout:
    """Load the frozen Full classifier with its validation-selected decoder.

    The existing completed-run loader restores and checks checkpoint evidence
    on CPU.  Only the Full model is retained and moved to ``device``.  This
    factory rejects exploratory probes and any freeze fitted on test or real
    data.  It does not fit a threshold or select a new decoder.  Inference uses
    receipt precision by default; ``precision='fp32'`` or ``'bf16'`` explicitly
    overrides it for an independently documented precision comparison.
    """
    freeze_path = Path(validation_freeze_path)
    with freeze_path.open("r", encoding="utf-8") as stream:
        authority = json.load(stream)
    if not isinstance(authority, dict):
        raise ValueError("validation freeze must be a JSON object")
    if authority.get("source_split") != "validation":
        raise ValueError("layout selection must come from validation")
    if authority.get("probe_only") is not False:
        raise ValueError("layout selection must be a formal validation freeze, not a probe")
    if authority.get("test_or_real_used_for_fit") is not False:
        raise ValueError("layout selection cannot be fitted on test or real data")
    count = authority.get("sample_count")
    if type(count) is not int or count <= 0:
        raise ValueError("validation freeze must record a positive sample_count")
    selected = authority.get("selected_full_decoder")
    decoders = authority.get("decoders")
    if (
        not isinstance(selected, str) or not selected
        or not isinstance(decoders, dict) or selected not in decoders
        or not isinstance(decoders[selected], dict)
    ):
        raise ValueError("selected_full_decoder must name a saved decoder configuration")
    config = TranslationLayoutConfig(**decoders[selected])

    from staging.pairwise_v0_2.training import rachel_n512_sealed_test as sealed

    receipt, receipt_sha, winners = sealed._freeze_completed_winners(Path(run_directory))
    full_winners = [winner for winner in winners if winner.arm == "full_n512"]
    if len(full_winners) != 1:
        raise ValueError("completed run must contain exactly one Full winner")
    full = full_winners[0]
    if authority.get("checkpoint_sha256") != full.checkpoint_sha256:
        raise ValueError("validation freeze references another Full checkpoint")
    threshold = float(full.threshold.threshold)
    if authority.get("original_fused_threshold") != threshold:
        raise ValueError("validation freeze references another pair classification threshold")
    checkpoint_precision = receipt["config"]["precision"]
    if checkpoint_precision not in ("fp32", "bf16"):
        raise ValueError("completed training receipt has unsupported precision")
    inference_precision = checkpoint_precision if precision is None else precision
    model = RachelPairwiseLayout(
        full.model.to(torch.device(device)), {selected: config},
        precision=inference_precision,
    ).eval()
    model.deployment_metadata = {
        "checkpoint_sha256": full.checkpoint_sha256,
        "checkpoint_epoch": full.epoch,
        "run_receipt_sha256": receipt_sha,
        "validation_freeze_path": str(freeze_path.resolve()),
        "selected_full_decoder": selected,
        "original_fused_threshold": threshold,
        "validation_sample_count": count,
        "checkpoint_precision": checkpoint_precision,
        "inference_precision": inference_precision,
        "routing_used": False,
        "rotation_estimated": False,
    }
    return model


__all__ = [
    "BatchedTranslationLayout", "RachelPairwiseLayoutOutput",
    "RachelPairwiseLayout", "decode_pair_layouts", "load_frozen_pairwise_layout",
]
