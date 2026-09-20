"""Explicit architecture metadata for new ablations and legacy Full weights."""
from dataclasses import asdict

from .rachel_n512 import RachelN512Config, RachelN512Pairwise


def build_rachel_model(model_kind, model_config, model_options=None):
    options = dict(model_options or {})
    if model_kind == "full":
        if options:
            raise ValueError("Full has no separate architecture options")
        return RachelN512Pairwise(RachelN512Config(**model_config))
    if model_kind == "hierarchical":
        from .rachel_hierarchical import RachelHierarchicalConfig, RachelHierarchicalPairwise
        return RachelHierarchicalPairwise(RachelN512Config(**model_config),
                                           RachelHierarchicalConfig(**options))
    if model_kind == "multiscale_transport":
        from .rachel_multiscale_transport import (
            RachelMultiscaleTransportConfig, RachelMultiscaleTransportPairwise,
        )
        if options:
            raise ValueError("Multiscale transport has no separate architecture options")
        return RachelMultiscaleTransportPairwise(RachelMultiscaleTransportConfig(**model_config))
    raise ValueError("Unknown model_kind: " + str(model_kind))


def model_metadata(model):
    if hasattr(model, "hierarchical_config"):
        return dict(model_kind="hierarchical", model_config=asdict(model.config),
                    model_options=asdict(model.hierarchical_config))
    if type(model).__name__ == "RachelMultiscaleTransportPairwise":
        return dict(model_kind="multiscale_transport", model_config=asdict(model.config),
                    model_options={})
    if type(model) is RachelN512Pairwise:
        return dict(model_kind="full", model_config=asdict(model.config), model_options={})
    raise TypeError("Unregistered Rachel model class")


def load_rachel_checkpoint(checkpoint):
    """Restore new model metadata, or an old Full checkpoint with no kind."""
    model = build_rachel_model(checkpoint.get("model_kind", "full"),
                              checkpoint["model_config"], checkpoint.get("model_options"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model
