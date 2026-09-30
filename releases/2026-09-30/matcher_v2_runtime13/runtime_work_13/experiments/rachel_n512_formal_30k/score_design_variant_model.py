"""Independent typed checkpoint factory for candidate-score input ablations.

Unlike the live S0-S2 loader, this factory explicitly preserves the input
variant's base_model kind/config/options, including post-Sinkhorn transport.
It does not mutate the deployed model classes or accept a legacy checkpoint
under a new identity. It is not yet a training entry point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch
from torch import nn

from experiments.rachel_n512_formal_30k.score_design_input_variants import (
    DEFAULT_SEED, InputVariantSpec, build_input_variant, full24_reference_config,
    restore_input_variant,
)
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Pairwise

SCHEMA = "rachel-score-input-model/1"
ARCHITECTURES = ("original", "candidate_pair", "candidate_dual")


@dataclass(frozen=True)
class ScoreInputBuild:
    model: nn.Module
    metadata: dict


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def build_score_input_model(spec=InputVariantSpec(), *, architecture="candidate_pair",
                            seed=DEFAULT_SEED, candidate_config=CandidateScoreConfig()):
    """Share both the Full24 base and new head initialization across input axes.

    The live candidate trainer creates Full24, then its new P/R head. The
    input builder preserves caller RNG, so explicitly advance the outer RNG
    through that exact reference construction before creating the new head.
    Architecture-only extra constructors cannot change the head seed.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown score architecture")
    if not isinstance(candidate_config, CandidateScoreConfig):
        candidate_config = CandidateScoreConfig(**dict(candidate_config))
    if candidate_config != CandidateScoreConfig():
        raise ValueError("input-only experiment keeps the registered candidate head config")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative integer below2**63")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        # This construction is used only to fix the identical new-head RNG.
        reference = RachelN512Pairwise(full24_reference_config())
        del reference
        base = build_input_variant(spec, seed=seed)
        model = base.model if architecture == "original" else RachelCandidateScore(
            base.model, candidate_config, architecture)
    metadata = _json(dict(schema_version=SCHEMA, architecture=architecture, seed=seed,
        input_variant=base.metadata,
        base_model_metadata=base.metadata["model_metadata"],
        candidate_config=asdict(candidate_config) if architecture != "original" else None,
        training_mode="joint", source_weights_loaded=False,
        initialization="shared random Full24 base and identical new P/R head initialization",
        layout_decoder_changed=False, legacy_score_checkpoint_compatible=False))
    return ScoreInputBuild(model, metadata)


def restore_score_input_model(metadata, state_dict):
    """Strictly reconstruct early/post BASE kind before loading the wrapper."""
    metadata = _json(metadata)
    if metadata.get("schema_version") != SCHEMA:
        raise ValueError("requires the independent score-input-model schema")
    architecture = metadata.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown score architecture")
    input_metadata = metadata["input_variant"]
    if metadata.get("base_model_metadata") != input_metadata.get("model_metadata"):
        raise ValueError("base model kind/config/options differ from input variant")
    rebuilt = build_score_input_model(input_metadata["spec"], architecture=architecture,
        seed=metadata["seed"], candidate_config=metadata.get("candidate_config") or CandidateScoreConfig())
    if metadata != rebuilt.metadata:
        raise ValueError("score-input-model metadata differs from registered model")
    # Validate learned state against the physical sampling grid in the actual
    # base model, never silently substituting a newly generated grid.
    prefix = "" if architecture == "original" else "base_model."
    base_state = {name[len(prefix):]: tensor for name, tensor in state_dict.items()
                  if not prefix or name.startswith(prefix)}
    restore_input_variant(input_metadata, base_state)
    rebuilt.model.load_state_dict(state_dict, strict=True)
    return rebuilt


__all__ = ["SCHEMA", "ARCHITECTURES", "ScoreInputBuild", "build_score_input_model", "restore_score_input_model"]
