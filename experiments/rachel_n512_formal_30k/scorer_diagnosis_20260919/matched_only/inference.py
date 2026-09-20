"""Six-input held-out inference for fresh matched-only and candidate-stage arms.

No loading, data access, thresholds, GT, labels, training or launch side effects.
The caller supplies the original S7 M12 base instance and its separately trained
FreshMatchedScorer. Checkpoint/source/hash validation belongs to the caller:
structural validation here does not certify checkpoint provenance.

The adapter adopts and freezes supplied instances in-place. The base still
computes its historical coarse fields for output compatibility, but neither
coarse scores/features nor old classifier scores are passed into the new head.
All Matcher, transport, validity and layout fields remain the original objects;
only the four fused/local classifier output fields are replaced.
"""
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Output, RachelN512Pairwise
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local.model import (
    DECODER_CONFIG, select_predicted_inliers,
)
from .model import ARMS, FreshMatchedScorer
from . import candidate_groups, stage_cache


SCHEMA = "s7-fresh-matched-scorer-six-input-inference/2"
REPLACED_FIELDS = frozenset(("fused_logit", "fused_probability", "local_logit", "local_probability"))


@dataclass(frozen=True)
class MatchedInferenceOutput(RachelN512Output):
    score_details: dict


@torch.no_grad()
def online_stage_groups(stage, selection, candidate_weights, points_a_rc,
                        points_b_rc, valid_a, valid_b):
    """Exact side-cache geometry/packing, without disk IO or target arguments.

    CPU NumPy calculation matches stage_cache.prepare/derive_row. The returned
    fixed1/fixed5 tensors live on the original feature device. Diagnostics are
    CPU dictionaries, separate from trained-head inputs. No group overrides the
    original production Layout, even if its Scorer logit is highest.
    """
    if stage not in stage_cache.STAGES:
        raise ValueError("unknown candidate-stage inference arm")
    batch, slots = len(points_a_rc), 1 if stage == "edge_seed" else 5
    arrays = {name: value.detach().cpu().numpy() for name, value in dict(
        points_a_rc=points_a_rc, points_b_rc=points_b_rc, valid_a=valid_a, valid_b=valid_b,
        candidate_indices=selection.candidate_indices, candidate_valid=selection.candidate_valid,
        candidate_weights=candidate_weights, final_inliers=selection.candidate_inliers,
        final_translation_rc=selection.translation_a_to_b_rc, layout_valid=selection.layout_valid).items()}
    names = ("candidate_inliers", "translation_rc", "present", "eligible", "ranks",
             "seed_candidate_id", "inlier_count", "status_code")
    packed = {name: np.zeros((batch, slots, *stage_cache.ARRAYS[name][1][1:]),
                            dtype=stage_cache.ARRAYS[name][0]) for name in names}
    packed["translation_rc"][:] = np.nan
    packed["seed_candidate_id"][:] = -1
    diagnostics = []
    for row in range(batch):
        result = candidate_groups.build_candidate_groups(
            **{name: value[row] for name, value in arrays.items()}, **stage_cache.CONFIG)
        views = [result.single_seed] if stage == "edge_seed" else list(result.multi_modes)
        for slot, group in enumerate(views):
            if group is None:
                continue
            packed["candidate_inliers"][row, slot, group.candidate_ids] = True
            packed["translation_rc"][row, slot] = group.translation_rc
            packed["present"][row, slot] = True
            packed["eligible"][row, slot] = group.inlier_count >= stage_cache.CONFIG["min_inliers"]
            packed["ranks"][row, slot] = group.rank
            packed["seed_candidate_id"][row, slot] = group.seed_candidate_id
            packed["inlier_count"][row, slot] = group.inlier_count
            packed["status_code"][row, slot] = stage_cache.STATUSES[group.status]
        diagnostics.append(dict(production_status=result.production_status, **result.diagnostics))
    return stage_cache.StageGroups(stage, **{name: torch.as_tensor(value, device=points_a_rc.device)
                                           for name, value in packed.items()}), tuple(diagnostics)


class FrozenMatchedInference(nn.Module):
    """Inference-only composition; `train(True)` deliberately raises.

    Accept all_tokens/matched_tokens/matched_edges plus edge_seed/edge_multi.
    Stage geometry comes from the exact same target-blind helper as side-cache.
    Smaller model configs are supported for CPU unit tests; formal evaluators
    must bind the exact original S7 M12 checkpoint/config externally.
    """
    def __init__(self, base_model, score_head):
        super().__init__()
        if not isinstance(base_model, RachelN512Pairwise):
            raise TypeError("supply the original RachelN512Pairwise base, not an old Scorer wrapper")
        if isinstance(score_head, FreshMatchedScorer) and score_head.arm in ARMS:
            feature_dim = score_head.feature_dim
            self.candidate_stage = False
        else:
            # This module is inert on import; training.py's only import-time
            # environment change is setdefault(CUBLAS_WORKSPACE_CONFIG).
            # Import lazily so original three-arm inference needs no trainer.
            from .train import CandidateStageScorer
            if not isinstance(score_head, CandidateStageScorer) or score_head.arm not in stage_cache.STAGES:
                raise TypeError("requires an original FreshMatchedScorer or registered CandidateStageScorer")
            feature_dim = score_head.edge_head.feature_dim
            self.candidate_stage = True
        if base_model.config.feature_dim != feature_dim:
            raise ValueError("base contextual-token dimension differs from trained Scorer")
        if base_model.config.contour_cap > 512:
            raise ValueError("this adapter is for original512 S7, not S5/S8 cap2048")
        self.base_model = base_model
        self.score_head = score_head
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        if mode:
            raise ValueError("FrozenMatchedInference is inference-only; train the isolated Scorer")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, mask_a, mask_b, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b):
        original = self.base_model(mask_a, mask_b, points_rc_a, points_rc_b,
                                   contour_valid_a, contour_valid_b)
        selected = select_predicted_inliers(original.assignment, points_rc_a, points_rc_b,
                                            contour_valid_a, contour_valid_b)
        # Match cache.compute EXACTLY: gather raw Q in the original production
        # candidate order and zero-fill padding. No score/label-based filtering.
        safe = selected.candidate_indices.clamp_min(0)
        batch = torch.arange(len(mask_a), device=original.assignment.device)[:, None]
        weights = original.assignment[batch, safe[..., 0], safe[..., 1]]
        weights = torch.where(selected.candidate_valid, weights, torch.zeros_like(weights))
        group_arguments, group_diagnostics = {}, ()
        if self.candidate_stage:
            groups, group_diagnostics = online_stage_groups(self.score_head.arm, selected,
                weights, points_rc_a, points_rc_b, contour_valid_a, contour_valid_b)
            group_arguments["groups"] = groups
        scores = self.score_head(original.token_features_a, original.token_features_b,
            contour_valid_a, contour_valid_b, selected, candidate_weights=weights,
            points_a_rc=points_rc_a, points_b_rc=points_rc_b, **group_arguments)
        if not torch.isfinite(scores.logit).all():
            # Keep original validity semantics. Do not hide a broken new head by
            # downgrading rows or reporting an invented finite prediction.
            raise ValueError("new Scorer emitted a nonfinite logit")
        deployed = torch.where(original.training_valid, scores.logit, torch.zeros_like(scores.logit))
        probability = deployed.sigmoid()
        values = {field.name: getattr(original, field.name) for field in fields(RachelN512Output)}
        values.update(fused_logit=deployed, fused_probability=probability,
                      local_logit=deployed, local_probability=probability)
        details = dict(
            schema=SCHEMA, arm=self.score_head.arm, raw_head_logit=scores.logit,
            deployed_logit=deployed, used_fallback=scores.used_fallback,
            has_raw_candidates=scores.has_raw_candidates,
            has_decoded_candidate=scores.has_decoded_candidate,
            reasons=selected.reasons, selection=selected, candidate_weights=weights,
            # The existing endpoint serializer records top-level tensors only.
            # Persist edge identity/membership without dumping full dense Q.
            candidate_indices=selected.candidate_indices,
            candidate_valid=selected.candidate_valid,
            candidate_inliers=selected.candidate_inliers)
        if self.candidate_stage:
            details.update({name: getattr(scores, name) for name in (
                "has_selected_candidate_group", "group_logits", "group_ranks", "group_eligible",
                "group_present", "group_status_code", "group_inlier_count", "group_seed_candidate_id",
                "group_translation_rc", "selected_group_rank")})
            rank_match = groups.eligible & (groups.ranks == scores.selected_group_rank[:, None])
            chosen = rank_match.to(torch.int64).argmax(1)
            selected_pose = groups.translation_rc[torch.arange(len(mask_a), device=chosen.device), chosen]
            details.update(groups=groups, group_diagnostics=group_diagnostics,
                group_candidate_inliers=groups.candidate_inliers,
                selected_group_translation_rc=torch.where(scores.has_selected_candidate_group[:, None],
                    selected_pose, torch.full_like(selected_pose, float("nan"))),
                selected_pose_role="diagnostic only; original production Layout NOT replaced")
        else:
            details.update(selected_token_count_a=scores.selected_token_count_a,
                selected_token_count_b=scores.selected_token_count_b, inlier_edge_count=scores.inlier_edge_count)
        return MatchedInferenceOutput(**values, score_details=details)

    def metadata(self):
        return dict(schema=SCHEMA, base_config=asdict(self.base_model.config),
            scorer=self.score_head.metadata(), decoder_config=asdict(DECODER_CONFIG),
            replaced_fields=sorted(REPLACED_FIELDS), source_binding="caller must verify original S7 M12 and trained-head provenance",
            score_policy="cached training evaluation: !original.training_valid gives logit0/probability0.5; raw head retained",
            coarse_used_by_new_head=False, matcher_updated=False,
            matcher_transport_layout_validity_unchanged=True,
            candidate_stage=self.candidate_stage,
            candidate_stage_pose_policy="highest-score group is metadata only; no layout reranking/deployment change",
            selected_features="same frozen post-context tokens as cache.compute; original raw Qij and target-blind predicted geometry only",
            no_candidate="trained fallback for matched arms; all_tokens still uses all valid tokens",
            thresholds_fitted=False, inference_only=True)
