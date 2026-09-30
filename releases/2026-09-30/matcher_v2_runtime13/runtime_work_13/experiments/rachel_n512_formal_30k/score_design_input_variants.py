"""Isolated single-input-axis builders; not wired into the live score trainer.

Every variant starts with the SAME randomly initialized Full24 model for the
given seed. A new architecture receives its compatible base tensors verbatim;
only physical sampling offsets and post-OT scale weights are regenerated.
This module changes no dataset, supervision, loss, decoder, or OT mathematics.

The reference is coarse128 / four physical windows7,16,32,64 / early descriptor
fusion / N512. Change at most ONE of coarse size, window set, or fusion stage.
N1024 is intentionally unavailable until ancestry-derived supervision exists.

Metadata has its own schema. In particular it must NOT be passed as an old
score-design candidate checkpoint: that loader currently reconstructs a Full
base and cannot faithfully restore a post-OT candidate wrapper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from staging.pairwise_v0_2.models.rachel_multiscale_transport import (
    RachelMultiscaleTransportConfig, initialize_multiscale_transport_from_base,
)
from staging.pairwise_v0_2.models.rachel_model_factory import model_metadata


SCHEMA = "rachel-score-input-variant/1"
DEFAULT_SEED = 260913
FOUR_WINDOWS = (7.0, 16.0, 32.0, 64.0)


@dataclass(frozen=True)
class InputVariantSpec:
    coarse_size: int = 128
    window_sizes_px: Tuple[float, ...] = FOUR_WINDOWS
    transport_fusion: str = "early"
    contour_cap: int = 512

    def __post_init__(self):
        if type(self.coarse_size) is not int or self.coarse_size not in (128, 256, 512):
            raise ValueError("coarse_size must be 128, 256, or 512")
        if type(self.contour_cap) is not int or self.contour_cap != 512:
            raise ValueError("only N512 is registered; N1024 needs ancestry supervision first")
        if self.transport_fusion not in ("early", "post"):
            raise ValueError("transport_fusion must be early or post")
        if not isinstance(self.window_sizes_px, (tuple, list)):
            raise TypeError("window_sizes_px must be a sequence of physical sizes")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in self.window_sizes_px):
            raise TypeError("physical window sizes must be numbers")
        windows = tuple(float(value) for value in self.window_sizes_px)
        if windows not in (FOUR_WINDOWS,) + tuple((w,) for w in FOUR_WINDOWS):
            raise ValueError("windows must be one of 7/16/32/64 or the ordered four-scale set")
        object.__setattr__(self, "window_sizes_px", windows)
        if len(self.changed_axes()) > 1:
            raise ValueError("single-variable ablation: cannot change two input axes together")

    def changed_axes(self):
        return tuple(name for name, changed in (
            ("coarse_size", self.coarse_size != 128),
            ("window_sizes_px", tuple(self.window_sizes_px) != FOUR_WINDOWS),
            ("transport_fusion", self.transport_fusion != "early"),
        ) if changed)


@dataclass(frozen=True)
class InputVariantBuild:
    model: nn.Module
    metadata: dict


def full24_reference_config():
    """Exact architecture metadata of Full-E1-24K, not Rachel's 2-scale default.

    Direct source: reports/local_score_probe_20260912_001/full24/protocol.json,
    model.model_metadata.model_config. No source checkpoint weight is loaded.
    """
    return RachelN512Config(window_sizes_px=FOUR_WINDOWS, validate_runtime_inputs=False)


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tensor_digest(state: Mapping[str, Tensor]):
    # Same state-tensor digest convention as train_joint_damage.state_digest.
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sampling_and_matching_contract(spec: InputVariantSpec):
    post = spec.transport_fusion == "post"
    return {
        "prepared_canvas_px": [800, 800],
        "contour_cap": 512,
        "contour_coordinates": "unchanged prepared-canvas row,column pixels; no contour resampling",
        "coarse_input_px": [spec.coarse_size, spec.coarse_size],
        "coarse_resize": "nearest from the same prepared800 binary mask",
        "physical_window_sizes_px": list(spec.window_sizes_px),
        "patch_tensor_px": [16, 16],
        "patch_sampling": {
            "centers": "same existing ordered contour tokens for every scale",
            "offset_formula": "16 linearly spaced samples per axis from -(w-1)/2 to +(w-1)/2",
            "interpolation": "nearest", "padding": "zeros", "align_corners": True,
            "coordinate_unit": "prepared800 pixel, not independently resized fragment pixel",
        },
        "primal_dual": {
            "mode": "shared complementary primal/dual projections; unchanged",
            "affinity_formula": "0.5*(dot(primal_A,dual_B)+dot(dual_A,primal_B))",
            "per_scale_application": post,
            "parameters_shared_across_scales": True,
        },
        "fusion_stage": spec.transport_fusion,
        "early_descriptor_gate": not post,
        "post_transport": ({
            "scale_weights": "global learned softmax, uniform initialization",
            "initial_normalized_weights": [1. / len(spec.window_sizes_px)] * len(spec.window_sizes_px),
            "weights_shared_across": ["pairs", "rows", "columns", "all augmented plan blocks"],
            "mixed_blocks": ["real_transport", "dustbin_row", "dustbin_col", "dustbin_corner"],
            "mixed_marginals": "recomputed diagnostics; preserves existing partial-OT marginals up to solver residual",
            "extra_sinkhorn_after_mixing": False,
            "affinity_readout": "mixed_product",
        } if post else None),
        "sinkhorn": {"iterations": 100, "temperature": .25, "tolerance": .001,
                     "mathematics_changed": False},
        "pair_readout": "existing local head and unchanged coarse/local four-input fusion",
        "layout": "unchanged existing translation output and external canonical decoder",
        "dataset_or_target_mutation": False,
    }


def build_input_variant(spec: Union[InputVariantSpec, Mapping] = InputVariantSpec(), *,
                        seed: int = DEFAULT_SEED) -> InputVariantBuild:
    """Create a CPU random model with a strictly shared common initialization.

    Caller RNG is preserved. The default returns the actual reference object,
    making its complete state exactly equal to the current Full24 random build
    for that seed. Non-default models copy all compatible reference tensors.
    """
    if not isinstance(spec, InputVariantSpec):
        spec = InputVariantSpec(**dict(spec))
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0,2**63)")
    reference_config = full24_reference_config()
    # Do not seed CUDA or change the caller's Python/NumPy/CPU RNG trajectory.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = RachelN512Pairwise(reference_config)
        reference_state = reference.state_dict()
        configured = replace(reference_config, coarse_size=spec.coarse_size,
                             window_sizes_px=spec.window_sizes_px)
        if not spec.changed_axes():
            model = reference
        elif spec.transport_fusion == "early":
            model = RachelN512Pairwise(configured)
            state = dict(reference_state)
            # A single physical window has a different grid, not different CNN weights.
            state["patch_sampler.offsets_rc"] = model.patch_sampler.offsets_rc
            model.load_state_dict(state, strict=True)
        else:
            post_config = RachelMultiscaleTransportConfig(
                **asdict(configured), initial_scale_weights=(),
                learn_scale_weights=True, affinity_readout="mixed_product")
            model = initialize_multiscale_transport_from_base(
                reference_config, reference_state, config=post_config)
    variant_state = model.state_dict()
    shared_names = sorted(name for name, tensor in reference_state.items()
                          if name != "patch_sampler.offsets_rc" and name in variant_state
                          and tensor.shape == variant_state[name].shape)
    # Fail at construction if a future model refactor accidentally loses sharing.
    if any(not torch.equal(reference_state[name], variant_state[name]) for name in shared_names):
        raise RuntimeError("compatible tensors differ from common random Full24 base")
    metadata = _json_value({
        "schema_version": SCHEMA, "seed": seed, "spec": asdict(spec),
        "changed_axis": spec.changed_axes()[0] if spec.changed_axes() else "none",
        "reference_model_metadata": model_metadata(reference),
        "model_metadata": model_metadata(model),
        "initialization": {
            "kind": "shared random Full24 base; no pretrained/source weights loaded",
            "reference_state_sha256": _tensor_digest(reference_state),
            "variant_initial_state_sha256": _tensor_digest(variant_state),
            "shared_tensor_names": shared_names,
            "shared_tensors_sha256": _tensor_digest({name: reference_state[name] for name in shared_names}),
            "variant_only_tensors": sorted(set(variant_state) - set(reference_state)),
            "removed_reference_tensors": sorted(set(reference_state) - set(variant_state)),
            "regenerated_buffer": "patch_sampler.offsets_rc" if spec.changed_axes() else None,
        },
        "contract": _sampling_and_matching_contract(spec),
        "live_score_checkpoint_schema_changed": False,
    })
    return InputVariantBuild(model, metadata)


def restore_input_variant(metadata: Mapping, model_state_dict: Optional[Mapping[str, Tensor]] = None):
    """Strict JSON metadata roundtrip; optionally load this variant's trained state.

    The physical sampling grid is checked, never silently corrected on restore.
    Early/post model kind, primal/dual mode and complete-plan contract cannot be
    altered through edited metadata. Learned scale_logits may of course differ
    from their uniform initialization in a trained state.
    """
    document = _json_value(dict(metadata))
    if document.get("schema_version") != SCHEMA:
        raise ValueError("not an input-variant checkpoint schema")
    built = build_input_variant(document["spec"], seed=document["seed"])
    if document != built.metadata:
        raise ValueError("input-variant metadata differs from the registered spec/initialization")
    if model_state_dict is not None:
        offsets = model_state_dict.get("patch_sampler.offsets_rc")
        if offsets is None or not torch.equal(offsets.detach().cpu(), built.model.patch_sampler.offsets_rc):
            raise ValueError("checkpoint physical sampling offsets differ from metadata")
        built.model.load_state_dict(dict(model_state_dict), strict=True)
    return built


__all__ = ["SCHEMA", "DEFAULT_SEED", "FOUR_WINDOWS", "InputVariantSpec", "InputVariantBuild",
           "full24_reference_config", "build_input_variant", "restore_input_variant"]
