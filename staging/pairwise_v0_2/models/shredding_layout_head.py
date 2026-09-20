"""Frozen ShreddingNet correspondence module for translation-only layout.

This module contains no coarse scorer or pair classifier.  Its default weights
are the fine matcher's parameters and BatchNorm buffers from the classifier-stage
winner, matching the historical benchmark's effective geometry model.  The
matching-stage winner is an explicit alternative for a mechanism comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.baselines.rachel_shreddingnet_benchmark import (
    FREEZE_SCHEMA_VERSION,
    MAX_POSE_CORRESPONDENCES,
    SCHEMA_VERSION,
    ReleaseRecipe,
    ReleasedFineMatcher,
    _decode_safe_checkpoint_value,
    cauchy_translation_consensus,
    released_antidiagonal_morphology,
)


@dataclass(frozen=True)
class SparseLayoutCorrespondences:
    source_index: np.ndarray
    target_index: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class ShreddingLayoutOutput:
    """A-to-B translation in row/column pixels; invalid coordinates are NaN."""

    translation_hat_rc: np.ndarray
    translation_valid: np.ndarray
    correspondence_count: np.ndarray
    inlier_count: np.ndarray
    correspondences: Optional[Tuple[SparseLayoutCorrespondences, ...]] = None

    @property
    def translation_inlier_count(self) -> np.ndarray:
        return self.inlier_count


class ShreddingLayoutHead(nn.Module):
    """Inference-only local matcher followed by robust no-rotation geometry."""

    def __init__(
        self,
        matcher: nn.Module,
        recipe: ReleaseRecipe,
        *,
        amp: bool = True,
        microbatch_size: int = 1,
        checkpoint_stage: str = "classify",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        self.matcher = matcher.requires_grad_(False).eval()
        self.recipe = recipe
        self.amp = bool(amp)
        self.microbatch_size = int(microbatch_size)
        self.checkpoint_stage = checkpoint_stage
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.eval()

    def train(self, mode: bool = True) -> "ShreddingLayoutHead":
        # This head is a frozen geometry component, including its BN buffers.
        super().train(False)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.matcher.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        mask_a: Union[np.ndarray, Tensor],
        mask_b: Union[np.ndarray, Tensor],
        points_rc_a: Union[np.ndarray, Tensor],
        points_rc_b: Union[np.ndarray, Tensor],
        contour_valid_a: Union[np.ndarray, Tensor],
        contour_valid_b: Union[np.ndarray, Tensor],
        *,
        return_correspondence: bool = False,
    ) -> ShreddingLayoutOutput:
        """The input contract matches Full; no supervision fields are accepted."""

        raw_inputs = (
            mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b
        )
        batch_size = len(mask_a)
        if any(len(value) != batch_size for value in raw_inputs):
            raise ValueError("layout input batch sizes differ")
        translations = np.full((batch_size, 2), np.nan, dtype=np.float64)
        valid = np.zeros(batch_size, dtype=np.bool_)
        counts = np.zeros(batch_size, dtype=np.int64)
        inliers = np.zeros(batch_size, dtype=np.int64)
        sparse = []
        for start in range(0, batch_size, self.microbatch_size):
            stop = min(start + self.microbatch_size, batch_size)
            values = tuple(
                torch.as_tensor(
                    value[start:stop], device=self.device,
                    dtype=torch.bool if index >= 4 else torch.float32,
                )
                for index, value in enumerate(raw_inputs)
            )
            masks_a, masks_b, points_a, points_b, valid_a, valid_b = values
            # The public Rachel loader stores masks as BxHxW; Full callers may
            # already provide Bx1xHxW.  Both carry the same input information.
            if masks_a.ndim == 3:
                masks_a = masks_a[:, None]
            if masks_b.ndim == 3:
                masks_b = masks_b[:, None]
            with torch.cuda.amp.autocast(enabled=self.amp and self.device.type == "cuda"):
                probability = self.matcher(
                    masks_a, masks_b, points_a, points_b, valid_a, valid_b
                )
            probability = probability.float()
            binary = released_antidiagonal_morphology(
                probability, self.recipe.correspondence_threshold
            )
            binary &= valid_a[:, :, None] & valid_b[:, None, :]
            probability_np = probability.cpu().numpy()
            binary_np = binary.cpu().numpy()
            points_a_np = points_a.double().cpu().numpy()
            points_b_np = points_b.double().cpu().numpy()
            for offset in range(stop - start):
                rows, columns = np.nonzero(binary_np[offset])
                weights = probability_np[offset, rows, columns]
                if len(rows) > MAX_POSE_CORRESPONDENCES:
                    keep = np.lexsort((columns, rows, -weights))[:MAX_POSE_CORRESPONDENCES]
                    rows, columns, weights = rows[keep], columns[keep], weights[keep]
                estimate = cauchy_translation_consensus(
                    points_a_np[offset, rows], points_b_np[offset, columns],
                    weights, min_correspondences=16,
                )
                index = start + offset
                counts[index] = len(rows)
                inliers[index] = estimate.inlier_count
                valid[index] = estimate.valid
                if estimate.valid:
                    translations[index] = estimate.translation_rc
                if return_correspondence:
                    sparse.append(SparseLayoutCorrespondences(rows, columns, weights))
        return ShreddingLayoutOutput(
            translations, valid, counts, inliers,
            tuple(sparse) if return_correspondence else None,
        )

    def predict_batch(
        self, batch: Any, *, return_correspondence: bool = False
    ) -> ShreddingLayoutOutput:
        """Read only the six model-input attributes from a RachelBatch."""

        return self(
            batch.mask_a, batch.mask_b, batch.points_rc_a, batch.points_rc_b,
            batch.contour_valid_a, batch.contour_valid_b,
            return_correspondence=return_correspondence,
        )


def load_shredding_layout_head(
    freeze_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    checkpoint_stage: str = "classify",
    amp: bool = True,
    microbatch_size: int = 1,
) -> ShreddingLayoutHead:
    """Load only fine-matcher tensors; never instantiate the pair classifier."""

    if checkpoint_stage not in {"classify", "matching"}:
        raise ValueError("checkpoint_stage must be classify or matching")
    path = Path(freeze_path)
    with path.open(encoding="utf-8") as stream:
        freeze = json.load(stream)
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported ShreddingNet freeze schema")
    recipe = ReleaseRecipe(**freeze["recipe"])
    checkpoint_path = Path(freeze["checkpoints"][checkpoint_stage]["path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = path.parent / checkpoint_path
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = _decode_safe_checkpoint_value(torch.load(checkpoint_path, **load_kwargs))
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("stage") != checkpoint_stage
        or checkpoint.get("checkpoint_kind") != "rachel_shreddingnet_{}_winner".format(checkpoint_stage)
        or checkpoint.get("recipe") != asdict(recipe)
    ):
        raise ValueError("ShreddingNet winner stage/schema/recipe differs")
    state = checkpoint["model_state_dict"]
    if checkpoint_stage == "classify":
        state = {
            name[len("matcher."):]: value
            for name, value in state.items() if name.startswith("matcher.")
        }
    if not state:
        raise ValueError("winner contains no fine-matcher tensors")
    matcher = ReleasedFineMatcher(recipe)
    matcher.load_state_dict(state, strict=True)
    matcher.to(device)
    return ShreddingLayoutHead(
        matcher, recipe, amp=amp, microbatch_size=microbatch_size,
        checkpoint_stage=checkpoint_stage, checkpoint_path=checkpoint_path,
    )
