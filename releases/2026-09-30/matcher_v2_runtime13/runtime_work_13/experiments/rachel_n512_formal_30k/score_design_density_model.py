"""Typed paired-density model: change only contour cap, never physical scale.

This does not make old512 labels usable at1024. Training must consume the
complete paired source-density manifests, and clean VAL must have independently
derived cap-specific targets. Both new512 and new1024 re-extract the contour.
Those data identities belong to the trainer, not to this pure model factory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json

import torch
from torch import nn

from experiments.rachel_n512_formal_30k.score_design_input_variants import (
    DEFAULT_SEED, full24_reference_config, _tensor_digest)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig

SCHEMA = "rachel-score-density-model/1"
ARCHITECTURES = ("original", "candidate_pair", "candidate_dual")
CAPS = (512, 1024)


@dataclass(frozen=True)
class ScoreDensityBuild:
    model: nn.Module
    metadata: dict


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def build_score_density_model(contour_cap, architecture="candidate_pair", seed=DEFAULT_SEED):
    """CPU random model; preserve caller RNG and share all initial tensors.

    Cap only changes the accepted token limit. There are no cap-sized learned
    embeddings or sampling-grid changes, so every tensor must start identical
    across the two caps. The training entry advances its RNG to the canonical
    post-base/head-construction state before loading training batches.
    """
    if type(contour_cap) is not int or contour_cap not in CAPS:
        raise ValueError("paired density requires contour cap512 or1024")
    if architecture not in ARCHITECTURES:
        raise ValueError("unregistered density scoring architecture")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative integer below2**63")
    config = replace(full24_reference_config(), contour_cap=contour_cap)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        base = RachelN512Pairwise(config)
        base_digest = _tensor_digest(base.state_dict())
        model = base if architecture == "original" else RachelCandidateScore(
            base, CandidateScoreConfig(), architecture)
    metadata = canonical(dict(schema_version=SCHEMA, architecture=architecture, seed=seed,
        contour_cap=contour_cap, resample_contour_cap=contour_cap,
        base_model_metadata=dict(model_kind="full", model_config=asdict(config), model_options={}),
        candidate_config=asdict(CandidateScoreConfig()) if architecture != "original" else None,
        shared_base_initial_weights_sha256=base_digest,
        initial_weights_sha256=_tensor_digest(model.state_dict()),
        training_mode="joint", source_weights_loaded=False,
        initialization="random; identical complete initial tensors across paired caps",
        density_contract=dict(physical_masks_and_coordinate_scale_changed=False,
            physical_window_sizes_px=[7., 16., 32., 64.], patch_tensor_px=[16, 16],
            coarse_input_px=[128, 128], descriptor_fusion="early", sinkhorn_count=1,
            contour_reextracted_at_both_caps=True, dense_train_source_ancestry_required=True,
            old_prepared512_is_not_the_paired_control=True,
            layout_decoder_changed=False, primal_dual_changed=False,
            other_input_axes_changed=False),
        legacy_score_checkpoint_compatible=False))
    return ScoreDensityBuild(model, metadata)


def restore_score_density_model(metadata, state_dict):
    """Restore only the typed density model; never infer a cap from weights."""
    metadata = canonical(metadata)
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("requires the independent paired-density model schema")
    built = build_score_density_model(metadata["contour_cap"], metadata["architecture"], metadata["seed"])
    if metadata != built.metadata:
        raise ValueError("density model metadata differs from registered single-cap design")
    prefix = "" if metadata["architecture"] == "original" else "base_model."
    grid_name = prefix + "patch_sampler.offsets_rc"
    expected = built.model.state_dict()[grid_name]
    stored = state_dict.get(grid_name)
    if stored is None or not torch.equal(stored.detach().cpu(), expected):
        raise ValueError("density checkpoint changed physical patch sampling offsets")
    built.model.load_state_dict(state_dict, strict=True)
    return built


__all__ = ["SCHEMA", "CAPS", "ARCHITECTURES", "ScoreDensityBuild",
           "build_score_density_model", "restore_score_density_model"]
