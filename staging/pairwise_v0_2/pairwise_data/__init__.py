"""Data contracts for the Pairwise v0.2 experiments.

The package initializer is dependency-light.  Public attributes retain their
historical identities through PEP 562 lazy resolution, while metadata-only
consumers do not import mask decoders or unrelated data helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ArchiveBinding": (".training_stream", "ArchiveBinding"),
    "ArchiveSourceSpec": (".lazy_mask_loader", "ArchiveSourceSpec"),
    "ArchiveVerificationReceipt": (".lazy_mask_loader", "ArchiveVerificationReceipt"),
    "BalancedPairSampler": (".sampling", "BalancedPairSampler"),
    "BalancedSamplingConfig": (".sampling", "BalancedSamplingConfig"),
    "build_cross_parent_scale_matched_negatives": (
        ".shredding_pipeline_pairs",
        "build_cross_parent_scale_matched_negatives",
    ),
    "build_shredding_pipeline_candidate_pool": (
        ".shredding_pipeline_pairs",
        "build_shredding_pipeline_candidate_pool",
    ),
    "ECCVSplitExposureAudit": (".training_stream", "ECCVSplitExposureAudit"),
    "GEOMETRY_DIRECTION_MAP": (".training_stream", "GEOMETRY_DIRECTION_MAP"),
    "GeneratorSpec": (".shredding_pipeline_pairs", "GeneratorSpec"),
    "LazyMaskArchiveLoader": (".lazy_mask_loader", "LazyMaskArchiveLoader"),
    "LazyMaskLoaderConfig": (".lazy_mask_loader", "LazyMaskLoaderConfig"),
    "LazyMaskLoaderError": (".lazy_mask_loader", "LazyMaskLoaderError"),
    "MaskMemberRef": (".training_stream", "MaskMemberRef"),
    "ParentPairInventory": (
        ".shredding_pipeline_pairs",
        "ParentPairInventory",
    ),
    "SamplingError": (".sampling", "SamplingError"),
    "TrainingDataError": (".training_stream", "TrainingDataError"),
    "TrainingPairRecord": (".training_stream", "TrainingPairRecord"),
    "direction_to_geometry_relation": (".training_stream", "direction_to_geometry_relation"),
    "iter_frozen_validation": (".sampling", "iter_frozen_validation"),
    "iter_historical_pair_records": (".training_stream", "iter_historical_pair_records"),
    "iter_synthetic_pair_records": (".training_stream", "iter_synthetic_pair_records"),
    "iter_shredding_pipeline_pair_inventories": (
        ".shredding_pipeline_pairs",
        "iter_shredding_pipeline_pair_inventories",
    ),
    "iter_training_pair_records": (".training_stream", "iter_training_pair_records"),
    "validation_stream_fingerprint": (".sampling", "validation_stream_fingerprint"),
    "materialize_shredding_pipeline_representations": (
        ".shredding_pipeline_representations",
        "materialize_shredding_pipeline_representations",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        ) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ArchiveBinding",
    "ECCVSplitExposureAudit",
    "ArchiveSourceSpec",
    "ArchiveVerificationReceipt",
    "BalancedPairSampler",
    "BalancedSamplingConfig",
    "build_cross_parent_scale_matched_negatives",
    "build_shredding_pipeline_candidate_pool",
    "GEOMETRY_DIRECTION_MAP",
    "GeneratorSpec",
    "LazyMaskArchiveLoader",
    "LazyMaskLoaderConfig",
    "LazyMaskLoaderError",
    "MaskMemberRef",
    "ParentPairInventory",
    "SamplingError",
    "TrainingDataError",
    "TrainingPairRecord",
    "direction_to_geometry_relation",
    "iter_frozen_validation",
    "iter_historical_pair_records",
    "iter_synthetic_pair_records",
    "iter_shredding_pipeline_pair_inventories",
    "iter_training_pair_records",
    "validation_stream_fingerprint",
    "materialize_shredding_pipeline_representations",
]
