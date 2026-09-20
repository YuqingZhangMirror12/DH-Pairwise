"""Full coarse/sliding classification with ShreddingNet fine-matcher layout.

All requested pairs use the same frozen ShreddingNet fine matcher for primary
geometry.  The ShreddingNet coarse scorer and pair classifier are not part of
this model.  Full's validation-selected geometric decoder can be returned as
a separate alternative; no classification threshold chooses between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .rachel_pairwise_layout import (
    BatchedTranslationLayout,
    RachelPairwiseLayout,
    RachelPairwiseLayoutOutput,
    load_frozen_pairwise_layout,
)
from .shredding_layout_head import (
    ShreddingLayoutHead,
    ShreddingLayoutOutput,
    load_shredding_layout_head,
)


SHREDDING_LAYOUT_NAME = "full_with_shred_matching_layout"


@dataclass(frozen=True)
class BatchedShreddingLayout:
    t_a_to_b_rc: Tensor
    valid: Tensor
    computed: Tensor
    head_output: ShreddingLayoutOutput

    @property
    def offset_b_in_a_rc(self) -> Tensor:
        return -self.t_a_to_b_rc

    @property
    def diagnostics(self) -> Tuple[Dict[str, object], ...]:
        return tuple({
            "computed": True,
            "valid": bool(self.head_output.translation_valid[index]),
            "candidate_count": int(self.head_output.correspondence_count[index]),
            "inlier_count": int(self.head_output.inlier_count[index]),
            "geometry_source": "frozen_shredding_fine_matcher",
        } for index in range(len(self.head_output.translation_valid)))


@dataclass(frozen=True)
class RachelFullShreddingLayoutOutput(RachelPairwiseLayoutOutput):
    layouts: Dict[str, Union[BatchedTranslationLayout, BatchedShreddingLayout]]
    primary_layout_name: str = SHREDDING_LAYOUT_NAME

    @property
    def primary_layout(self) -> BatchedShreddingLayout:
        return self.layouts[self.primary_layout_name]


class RachelFullShreddingLayout(nn.Module):
    """One Full forward supplies all pair scores; fine matching supplies layout."""

    def __init__(
        self,
        full_layout: RachelPairwiseLayout,
        shredding_head: ShreddingLayoutHead,
        *,
        include_full_layout: bool = True,
    ) -> None:
        super().__init__()
        if type(include_full_layout) is not bool:
            raise TypeError("include_full_layout must be bool")
        self.full_layout = full_layout
        self.shredding_head = shredding_head
        self.include_full_layout = include_full_layout
        self.requires_grad_(False)
        self.eval()

    @property
    def pair_model(self) -> nn.Module:
        return self.full_layout.pair_model

    @property
    def layout_configs(self):
        return self.full_layout.layout_configs

    def train(self, mode: bool = True) -> "RachelFullShreddingLayout":
        super().train(False)
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
        return_correspondence: bool = False,
    ) -> RachelFullShreddingLayoutOutput:
        if type(compute_layout) is not bool:
            raise TypeError("compute_layout must be bool")
        inputs = (mask_a, mask_b, points_rc_a, points_rc_b,
                  contour_valid_a, contour_valid_b)
        full = self.full_layout(*inputs,
                                compute_layout=compute_layout and self.include_full_layout)
        layouts = dict(full.layouts)
        if compute_layout:
            fine = self.shredding_head(*inputs, return_correspondence=return_correspondence)
            # Keep the fine head's float64 geometry precision in this public
            # output.  Its A-to-B translation is independent of Full's scores.
            layouts[SHREDDING_LAYOUT_NAME] = BatchedShreddingLayout(
                t_a_to_b_rc=torch.as_tensor(fine.translation_hat_rc,
                                           dtype=torch.float64, device=points_rc_a.device),
                valid=torch.as_tensor(fine.translation_valid,
                                      dtype=torch.bool, device=points_rc_a.device),
                computed=torch.ones(len(mask_a), dtype=torch.bool, device=points_rc_a.device),
                head_output=fine,
            )
        return RachelFullShreddingLayoutOutput(full.pair_output, layouts)

    def predict_batch(self, batch, *, compute_layout: bool = True,
                      return_correspondence: bool = False) -> RachelFullShreddingLayoutOutput:
        """Read only model inputs from a RachelBatch; no labels or GT are used."""
        device = next(self.pair_model.parameters()).device
        names = ("mask_a", "mask_b", "points_rc_a", "points_rc_b",
                 "contour_valid_a", "contour_valid_b")
        values = [torch.as_tensor(getattr(batch, name), device=device,
                                  dtype=torch.bool if index >= 4 else torch.float32)
                  for index, name in enumerate(names)]
        for index in (0, 1):
            if values[index].ndim == 3:
                values[index] = values[index][:, None]
        return self(*values, compute_layout=compute_layout,
                    return_correspondence=return_correspondence)


def load_frozen_full_shredding_layout(
    run_directory,
    validation_freeze_path,
    shredding_freeze_path,
    device: str = "cpu",
    *,
    full_precision: Optional[str] = None,
    shred_amp: bool = True,
    shred_microbatch_size: int = 1,
    include_full_layout: bool = True,
    shred_checkpoint_stage: str = "classify",
) -> RachelFullShreddingLayout:
    """Load Full pairability plus a fixed ShreddingNet fine geometry module.

    ``classify`` specifies which winner supplies fine-matcher weights and BN
    buffers; that stage's pair classifier itself is never instantiated.  The
    primary output is always ShreddingNet geometry, independent of pair score.
    """
    full = load_frozen_pairwise_layout(run_directory, validation_freeze_path,
                                      device=device, precision=full_precision)
    head = load_shredding_layout_head(shredding_freeze_path, device=device,
                                      checkpoint_stage=shred_checkpoint_stage,
                                      amp=shred_amp, microbatch_size=shred_microbatch_size)
    model = RachelFullShreddingLayout(full, head, include_full_layout=include_full_layout)
    model.deployment_metadata = dict(full.deployment_metadata)
    model.deployment_metadata.update(
        primary_layout_name=SHREDDING_LAYOUT_NAME,
        shredding_fine_checkpoint_path=head.checkpoint_path,
        shredding_fine_checkpoint_stage=head.checkpoint_stage,
        shredding_amp=head.amp,
        shredding_microbatch_size=head.microbatch_size,
        include_full_layout=include_full_layout,
        shredding_coarse_scorer_executed=False,
        shredding_pair_classifier_executed=False,
        geometry_choice_uses_pair_score=False,
    )
    return model


__all__ = [
    "SHREDDING_LAYOUT_NAME", "BatchedShreddingLayout", "RachelFullShreddingLayoutOutput",
    "RachelFullShreddingLayout", "load_frozen_full_shredding_layout",
]
