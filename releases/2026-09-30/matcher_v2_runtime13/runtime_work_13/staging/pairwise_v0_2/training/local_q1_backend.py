"""Typed, provider-free model backend for LOCAL-Q1 research arms.

The backend consumes only :class:`PreparedAblationBatch` values whose opaque
payload is a :class:`RaggedGeometryBatch`.  It has no archive, mask-loader,
cache, split, or sealed-real-test capability.  Provider-owned tensor digests
are recomputed before any model or optimizer mutation.

Formal local-only arms share the same seeded initialization and train only the
ordered local matcher.  Their sole model/configuration fork is
``matcher_mode``.  A formal fused arm must load independently hash-pinned C0
and local checkpoints, freezes both evidence models, and trains only fusion.
Random fused evidence is available solely through the explicit
``fixture_non_result`` execution mode.
"""

from __future__ import annotations

import enum
import hmac
import inspect
import math
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.coarse import SymmetricCoarseSiamese
from staging.pairwise_v0_2.models.local_matcher import (
    MatcherMode,
    OrderedLocalMatcher,
    SinkhornTrainingPolicy,
)
from staging.pairwise_v0_2.models.pairwise import (
    DunhuangPairwiseV02,
    HierarchicalDirectionalOutput,
    PairwiseModelConfig,
    PairwiseScoreSource,
)
from staging.pairwise_v0_2.training.checkpoint import (
    canonical_config_hash,
    canonical_tensor_tree_sha256,
    load_trusted_checkpoint,
)
from staging.pairwise_v0_2.training.engine import (
    DirectionalStepResult,
    eval_directional_step,
    swap_directional_model_inputs,
    train_directional_step,
)
from staging.pairwise_v0_2.training.geometry_batch import (
    KEYPOINT_REPRESENTATION,
    MULTIRUN_REPRESENTATION,
    RaggedGeometryBatch,
)
from staging.pairwise_v0_2.training.local_q1_provider import (
    LOCAL_Q1_BATCH_PROVIDER_VERSION,
    local_q1_prepared_digests,
)
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    BackendContract,
    EvidenceMode,
    ExecutionKind,
    PredictionBatch,
    PreparedAblationBatch,
    ShortAblationError,
    TrainBatchResult,
)


LOCAL_Q1_BACKEND_VERSION = "dunhuang-local-q1-backend/0.2"
LOCAL_Q1_MODEL_CONFIG_VERSION = "dunhuang-local-q1-model-contract/0.2"
LOCAL_Q1_CHECKPOINT_BINDING_VERSION = "dunhuang-local-q1-checkpoint-binding/0.2"
LOCAL_Q1_SWAP_TOLERANCE = 1e-5
LOCAL_Q1_AUTHORITY_KIND = "external_checkpoint_authority_receipt"
LOCAL_Q1_AUTHORITY_STATUS = "upstream_validated_before_backend_construction"
LOCAL_Q1_AUTHORITY_BOUNDARY = (
    "identity_only_backend_does_not_open_authority_receipt_"
    "future_runner_must_validate_before_construction"
)
LOCAL_Q1_COARSE_INPUT_SHAPE = (1, 128, 128)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LocalQ1BackendError(ShortAblationError):
    """A LOCAL-Q1 model/session boundary failed closed."""


class LocalQ1BackendMode(str, enum.Enum):
    """Whether outputs may be treated as a synthetic train/validation result."""

    FORMAL = "formal_synthetic_train_validation"
    FIXTURE_NON_RESULT = "fixture_non_result"


class CheckpointModelScope(str, enum.Enum):
    """Model state stored in a trusted external checkpoint."""

    COARSE_MODEL = "coarse_model"
    LOCAL_MODEL = "local_model"
    PAIRWISE_MODEL = "pairwise_model"


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LocalQ1BackendError("{} must be a lowercase SHA-256".format(name))
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return _freeze(value.value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LocalQ1BackendError(
                "portable config cannot contain non-finite values"
            )
        return value
    raise TypeError("unsupported portable config type: {}".format(type(value).__name__))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _same_config(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return hmac.compare_digest(
        canonical_config_hash(first), canonical_config_hash(second)
    )


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalQ1BackendError("{} must be a mapping".format(name))
    return value


def _required_sequence(value: Any, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise LocalQ1BackendError("{} must be a sequence".format(name))
    return tuple(value)


def checkpoint_semantic_projection(
    expected_config: Mapping[str, Any], model_scope: CheckpointModelScope
) -> Mapping[str, Any]:
    """Project one fixed, scope-specific semantic checkpoint contract.

    C0 runner checkpoints expose their exact model declaration at ``model``.
    LOCAL-Q1 checkpoints expose the complete :class:`LocalModelConfig` at
    ``model_config.architecture.local`` and the local-only training boundary at
    ``model_config.training``.  No recursive key search or inferred fallback is
    permitted; a future checkpoint schema needs an explicit binding-schema
    revision.
    """

    root = _required_mapping(expected_config, "checkpoint expected_config")
    try:
        scope = CheckpointModelScope(getattr(model_scope, "value", model_scope))
    except ValueError as exc:
        raise LocalQ1BackendError("checkpoint model scope is unsupported") from exc
    if scope is CheckpointModelScope.COARSE_MODEL:
        model = _required_mapping(root.get("model"), "C0 expected_config.model")
        required = {
            "name",
            "input_channels",
            "widths",
            "embedding_dim",
            "hidden_dim",
            "input_shape",
        }
        if set(model) != required:
            raise LocalQ1BackendError(
                "C0 expected_config.model has missing or unexpected fields"
            )
        return _freeze({key: model[key] for key in sorted(required)})

    model_config = _required_mapping(
        root.get("model_config"), "local expected_config.model_config"
    )
    architecture = _required_mapping(
        model_config.get("architecture"),
        "local expected_config.model_config.architecture",
    )
    local = _required_mapping(
        architecture.get("local"),
        "local expected_config.model_config.architecture.local",
    )
    required_local = set(asdict(PairwiseModelConfig().local))
    if set(local) != required_local:
        raise LocalQ1BackendError(
            "checkpoint local model projection has missing or unexpected fields"
        )
    training = _required_mapping(
        model_config.get("training"),
        "local expected_config.model_config.training",
    )
    score_source = training.get("score_source")
    trainable = _required_sequence(
        training.get("trainable_components"),
        "local checkpoint trainable_components",
    )
    if score_source != PairwiseScoreSource.LOCAL.value:
        raise LocalQ1BackendError("local checkpoint must bind score_source='local'")
    if trainable != ("local_model",):
        raise LocalQ1BackendError(
            "local checkpoint must bind only local_model as trainable"
        )
    return _freeze(
        {
            "checkpoint_model_scope": scope.value,
            "local_model_config": dict(local),
            "score_source": score_source,
            "trainable_components": trainable,
        }
    )


def _target_checkpoint_semantic_projection(
    model_config: PairwiseModelConfig, model_scope: CheckpointModelScope
) -> Mapping[str, Any]:
    if model_scope is CheckpointModelScope.COARSE_MODEL:
        coarse = model_config.coarse
        return _freeze(
            {
                "name": "SymmetricCoarseSiamese",
                "input_channels": coarse.input_channels,
                "widths": coarse.widths,
                "embedding_dim": coarse.embedding_dim,
                "hidden_dim": coarse.hidden_dim,
                "input_shape": LOCAL_Q1_COARSE_INPUT_SHAPE,
            }
        )
    return _freeze(
        {
            "checkpoint_model_scope": model_scope.value,
            "local_model_config": asdict(model_config.local),
            "score_source": PairwiseScoreSource.LOCAL.value,
            "trainable_components": ("local_model",),
        }
    )


def checkpoint_authority_identity_sha256(
    *,
    model_scope: CheckpointModelScope,
    checkpoint_file_sha256: str,
    checkpoint_canonical_content_sha256: str,
    checkpoint_config_sha256: str,
    semantic_projection_sha256: str,
    authority_kind: str,
    authority_status: str,
    authority_receipt_file_sha256: str,
    authority_receipt_content_sha256: str,
) -> str:
    """Commit an opaque upstream authority identity to one checkpoint role.

    This function does not read or validate the authority receipt.  The future
    runner/validator must do that before it constructs a formal backend; this
    backend only prevents the already-validated identity from being changed or
    transplanted between checkpoint roles.
    """

    try:
        scope = CheckpointModelScope(getattr(model_scope, "value", model_scope))
    except ValueError as exc:
        raise LocalQ1BackendError("checkpoint model scope is unsupported") from exc
    for name, value in (
        ("checkpoint file SHA-256", checkpoint_file_sha256),
        ("checkpoint canonical content SHA-256", checkpoint_canonical_content_sha256),
        ("checkpoint config SHA-256", checkpoint_config_sha256),
        ("semantic projection SHA-256", semantic_projection_sha256),
        ("authority receipt file SHA-256", authority_receipt_file_sha256),
        ("authority receipt content SHA-256", authority_receipt_content_sha256),
    ):
        _require_sha256(value, name)
    if authority_kind != LOCAL_Q1_AUTHORITY_KIND:
        raise LocalQ1BackendError("checkpoint authority kind is unsupported")
    if authority_status != LOCAL_Q1_AUTHORITY_STATUS:
        raise LocalQ1BackendError("checkpoint authority status is not validated")
    return canonical_config_hash(
        {
            "schema_version": LOCAL_Q1_CHECKPOINT_BINDING_VERSION,
            "model_scope": scope.value,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "checkpoint_canonical_content_sha256": (
                checkpoint_canonical_content_sha256
            ),
            "checkpoint_config_sha256": checkpoint_config_sha256,
            "semantic_projection_sha256": semantic_projection_sha256,
            "external_authority": {
                "kind": authority_kind,
                "status": authority_status,
                "receipt_file_sha256": authority_receipt_file_sha256,
                "receipt_content_sha256": authority_receipt_content_sha256,
            },
        }
    )


@dataclass(frozen=True)
class TrustedCheckpointBinding:
    """Checkpoint locks plus an opaque, upstream-validated authority identity.

    The backend deliberately does not open ``authority_receipt_*``.  A future
    runner/validator is responsible for validating that receipt before this
    binding is constructed; the backend binds its identity and semantic
    projection so neither can change afterward.
    """

    path: Path
    file_sha256: str
    canonical_content_sha256: str
    config_sha256: str
    expected_config: Mapping[str, Any]
    model_scope: CheckpointModelScope
    semantic_projection_sha256: str
    authority_kind: str
    authority_status: str
    authority_receipt_file_sha256: str
    authority_receipt_content_sha256: str
    authority_identity_sha256: str
    schema_version: str = LOCAL_Q1_CHECKPOINT_BINDING_VERSION
    semantic_projection: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != LOCAL_Q1_CHECKPOINT_BINDING_VERSION:
            raise LocalQ1BackendError("checkpoint binding schema is unsupported")
        object.__setattr__(self, "path", Path(self.path))
        for name in (
            "file_sha256",
            "canonical_content_sha256",
            "config_sha256",
            "semantic_projection_sha256",
            "authority_receipt_file_sha256",
            "authority_receipt_content_sha256",
            "authority_identity_sha256",
        ):
            _require_sha256(getattr(self, name), "checkpoint " + name)
        if not isinstance(self.expected_config, Mapping):
            raise TypeError("expected_config must be a mapping")
        frozen = _freeze(self.expected_config)
        object.__setattr__(self, "expected_config", frozen)
        if not hmac.compare_digest(canonical_config_hash(frozen), self.config_sha256):
            raise LocalQ1BackendError(
                "checkpoint config hash differs from expected_config"
            )
        try:
            scope = CheckpointModelScope(
                getattr(self.model_scope, "value", self.model_scope)
            )
        except ValueError as exc:
            raise LocalQ1BackendError("checkpoint model scope is unsupported") from exc
        object.__setattr__(self, "model_scope", scope)
        projection = checkpoint_semantic_projection(frozen, scope)
        observed_projection_sha256 = canonical_config_hash(projection)
        if not hmac.compare_digest(
            observed_projection_sha256, self.semantic_projection_sha256
        ):
            raise LocalQ1BackendError(
                "checkpoint semantic projection hash is inconsistent"
            )
        object.__setattr__(self, "semantic_projection", projection)
        observed_authority = checkpoint_authority_identity_sha256(
            model_scope=scope,
            checkpoint_file_sha256=self.file_sha256,
            checkpoint_canonical_content_sha256=self.canonical_content_sha256,
            checkpoint_config_sha256=self.config_sha256,
            semantic_projection_sha256=self.semantic_projection_sha256,
            authority_kind=self.authority_kind,
            authority_status=self.authority_status,
            authority_receipt_file_sha256=self.authority_receipt_file_sha256,
            authority_receipt_content_sha256=self.authority_receipt_content_sha256,
        )
        if not hmac.compare_digest(observed_authority, self.authority_identity_sha256):
            raise LocalQ1BackendError(
                "checkpoint external authority identity is inconsistent"
            )

    @property
    def local_matcher_mode(self) -> Optional[str]:
        if self.model_scope is CheckpointModelScope.COARSE_MODEL:
            return None
        local = _required_mapping(
            self.semantic_projection.get("local_model_config"),
            "bound local model projection",
        )
        return str(local["matcher_mode"])

    def require_target(self, model_config: PairwiseModelConfig) -> None:
        target = _target_checkpoint_semantic_projection(model_config, self.model_scope)
        if not _same_config(self.semantic_projection, target):
            raise LocalQ1BackendError(
                "checkpoint semantic projection differs from target model"
            )

    def portable_identity(self) -> Mapping[str, Any]:
        """Return the lock identity without its machine-local path."""

        return {
            "schema_version": self.schema_version,
            "model_scope": self.model_scope.value,
            "checkpoint_file_sha256": self.file_sha256,
            "checkpoint_canonical_content_sha256": (self.canonical_content_sha256),
            "checkpoint_config_sha256": self.config_sha256,
            "semantic_projection": _thaw(self.semantic_projection),
            "semantic_projection_sha256": self.semantic_projection_sha256,
            "external_authority": {
                "kind": self.authority_kind,
                "status": self.authority_status,
                "receipt_file_sha256": self.authority_receipt_file_sha256,
                "receipt_content_sha256": self.authority_receipt_content_sha256,
                "identity_sha256": self.authority_identity_sha256,
                "validation_boundary": LOCAL_Q1_AUTHORITY_BOUNDARY,
                "receipt_opened_by_backend": False,
            },
        }


@dataclass(frozen=True)
class FusedCheckpointBindings:
    """The two independently locked evidence sources required by fused."""

    coarse: TrustedCheckpointBinding
    local: TrustedCheckpointBinding

    def __post_init__(self) -> None:
        if not isinstance(self.coarse, TrustedCheckpointBinding) or not isinstance(
            self.local, TrustedCheckpointBinding
        ):
            raise TypeError("fused checkpoint bindings must be typed")
        if self.coarse.model_scope is not CheckpointModelScope.COARSE_MODEL:
            raise LocalQ1BackendError("C0 binding must contain a coarse model")
        if self.local.model_scope is CheckpointModelScope.COARSE_MODEL:
            raise LocalQ1BackendError("local binding cannot contain a coarse model")
        if self.local.local_matcher_mode != MatcherMode.DUSTBIN_SINKHORN.value:
            raise LocalQ1BackendError(
                "formal fused requires a dustbin-Sinkhorn local checkpoint"
            )
        if hmac.compare_digest(self.coarse.file_sha256, self.local.file_sha256):
            raise LocalQ1BackendError("C0 and local checkpoint files must be distinct")

    def portable_identity(self) -> Mapping[str, Any]:
        return {
            "coarse": self.coarse.portable_identity(),
            "local": self.local.portable_identity(),
        }

    def require_target(self, model_config: PairwiseModelConfig) -> None:
        self.coarse.require_target(model_config)
        self.local.require_target(model_config)


@dataclass(frozen=True)
class LocalQ1OptimizerConfig:
    """One optimizer definition shared by all LOCAL-Q1 arms."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    foreach: bool = False
    fused: Optional[bool] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("optimizer lr must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("optimizer weight_decay must be finite and non-negative")
        if len(self.betas) != 2 or any(
            not math.isfinite(value) or not 0.0 <= value < 1.0 for value in self.betas
        ):
            raise ValueError("optimizer betas are invalid")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("optimizer eps must be finite and positive")
        for name in ("amsgrad", "foreach"):
            if type(getattr(self, name)) is not bool:
                raise TypeError("optimizer {} must be bool".format(name))
        if self.fused is not None and type(self.fused) is not bool:
            raise TypeError("optimizer fused must be bool or None")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "name": "AdamW",
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "amsgrad": self.amsgrad,
            "foreach": self.foreach,
            "fused": self.fused,
        }


@dataclass(frozen=True)
class LocalQ1StepConfig:
    """Loss, swap, clipping, and numerical diagnostics shared by all arms."""

    coarse_loss_weight: float = 0.25
    direction_loss_weight: float = 0.25
    sinkhorn_residual_weight: float = 0.05
    sinkhorn_residual_target: float = 1e-3
    swap_loss_weight: float = 0.1
    swap_transport_weight: float = 0.1
    max_gradient_norm: float = 5.0
    swap_tolerance: float = LOCAL_Q1_SWAP_TOLERANCE

    def __post_init__(self) -> None:
        for name in (
            "coarse_loss_weight",
            "direction_loss_weight",
            "sinkhorn_residual_weight",
            "sinkhorn_residual_target",
            "swap_loss_weight",
            "swap_transport_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be finite and positive")
        if not math.isfinite(self.swap_tolerance) or self.swap_tolerance < 0.0:
            raise ValueError("swap_tolerance must be finite and non-negative")

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactSeamStepConfig:
    """Loss weights used only by the independent exact-seam training arm."""

    loss_weight: float = 0.25
    match_weight: float = 1.0
    dustbin_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("loss_weight", "match_weight", "dustbin_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError("{} must be finite and non-negative".format(name))
        if self.loss_weight <= 0.0:
            raise ValueError("exact-seam loss_weight must be positive")

    def to_dict(self) -> Mapping[str, float]:
        return asdict(self)


def _formal_model_template() -> PairwiseModelConfig:
    """Return the full-capacity stable Sinkhorn definition for formal work."""

    return PairwiseModelConfig()


def _model_for_matcher(
    template: PairwiseModelConfig, matcher_mode: str
) -> PairwiseModelConfig:
    matcher = MatcherMode(matcher_mode)
    return replace(
        template,
        local=replace(template.local, matcher_mode=matcher.value),
    )


def _validate_formal_numerics(template: PairwiseModelConfig) -> None:
    local = template.local
    expected = (
        local.matcher_temperature == 0.25
        and local.sinkhorn_iterations == 100
        and local.sinkhorn_tolerance == 1e-3
        and local.require_sinkhorn_convergence is True
        and local.sinkhorn_training_policy
        == SinkhornTrainingPolicy.FINITE_WITH_RESIDUAL.value
        and local.dropout == 0.0
    )
    if not expected:
        raise LocalQ1BackendError(
            "formal LOCAL-Q1 requires temperature=0.25, 100 Sinkhorn iterations, "
            "tolerance=1e-3, finite-with-residual training, convergence-gated "
            "decisions, and zero dropout"
        )


def _validate_formal_step_config(config: LocalQ1StepConfig) -> None:
    expected = LocalQ1StepConfig()
    if not _same_config(config.to_dict(), expected.to_dict()):
        raise LocalQ1BackendError(
            "formal LOCAL-Q1 step config must equal the frozen residual/swap/"
            "gradient definition"
        )


def _configure_determinism(seed: int) -> Mapping[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LocalQ1BackendError("LOCAL-Q1 seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return MappingProxyType(
        {
            "seed": seed,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
    )


def _make_optimizer(
    parameters: Tuple[nn.Parameter, ...], config: LocalQ1OptimizerConfig
) -> torch.optim.AdamW:
    if not parameters:
        raise LocalQ1BackendError("LOCAL-Q1 has no trainable parameters")
    arguments: Dict[str, Any] = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "betas": config.betas,
        "eps": config.eps,
        "amsgrad": config.amsgrad,
    }
    supported = inspect.signature(torch.optim.AdamW).parameters
    if "foreach" in supported:
        arguments["foreach"] = config.foreach
    if "fused" in supported:
        arguments["fused"] = config.fused
    return torch.optim.AdamW(parameters, **arguments)


def _ordered_local_matcher(
    config: PairwiseModelConfig, matcher: str
) -> OrderedLocalMatcher:
    local = replace(config.local, matcher_mode=MatcherMode(matcher).value)
    return OrderedLocalMatcher(
        input_channels=local.input_channels,
        feature_dim=local.feature_dim,
        num_heads=local.num_heads,
        ff_dim=local.ff_dim,
        matcher_mode=MatcherMode(local.matcher_mode),
        matcher_temperature=local.matcher_temperature,
        sinkhorn_iterations=local.sinkhorn_iterations,
        sinkhorn_tolerance=local.sinkhorn_tolerance,
        require_sinkhorn_convergence=local.require_sinkhorn_convergence,
        sinkhorn_training_policy=SinkhornTrainingPolicy(local.sinkhorn_training_policy),
        dropout=local.dropout,
    )


def _coarse_model(config: PairwiseModelConfig) -> SymmetricCoarseSiamese:
    coarse = config.coarse
    return SymmetricCoarseSiamese(
        input_channels=coarse.input_channels,
        widths=coarse.widths,
        embedding_dim=coarse.embedding_dim,
        hidden_dim=coarse.hidden_dim,
    )


def _load_binding(
    binding: TrustedCheckpointBinding, model: nn.Module
) -> Mapping[str, Any]:
    payload = load_trusted_checkpoint(
        binding.path,
        model,
        expected_config=_thaw(binding.expected_config),
        expected_file_sha256=binding.file_sha256,
        expected_canonical_content_sha256=binding.canonical_content_sha256,
        map_location="cpu",
        trusted=True,
    )
    if not hmac.compare_digest(payload["config_hash"], binding.config_sha256):
        raise LocalQ1BackendError("loaded checkpoint config hash changed")
    if not hmac.compare_digest(
        payload["canonical_content_sha256"], binding.canonical_content_sha256
    ):
        raise LocalQ1BackendError("loaded checkpoint canonical content changed")
    return payload


class _ControlledPairwiseModel(DunhuangPairwiseV02):
    """Keep frozen fused evidence modules in eval mode during fusion training."""

    def __init__(self, config: PairwiseModelConfig, *, fused_freeze: bool) -> None:
        super().__init__(config)
        self._fused_freeze = fused_freeze

    def train(self, mode: bool = True) -> "_ControlledPairwiseModel":
        super().train(mode)
        if self._fused_freeze:
            self.coarse_model.eval()
            self.local_model.eval()
        return self


class _FairnessLedger:
    """Cross-arm identity commitments retained only in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batches: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
        self._local_initial_state_sha256: Optional[str] = None

    def require_batch(self, batch: PreparedAblationBatch) -> None:
        value = (
            batch.sample_count,
            batch.prepared_input_sha256,
            batch.local_candidate_sha256,
            batch.geometry_config_sha256,
        )
        with self._lock:
            key = (
                batch.record_sequence_sha256,
                batch.candidate_representation,
            )
            previous = self._batches.setdefault(key, value)
            if previous != value:
                raise LocalQ1BackendError(
                    "prepared tensors differ across LOCAL-Q1 arms within candidate representation"
                )

    def require_common_local_initialization(self, state_sha256: str) -> None:
        _require_sha256(state_sha256, "initial model state")
        with self._lock:
            if self._local_initial_state_sha256 is None:
                self._local_initial_state_sha256 = state_sha256
            elif not hmac.compare_digest(
                self._local_initial_state_sha256, state_sha256
            ):
                raise LocalQ1BackendError(
                    "LOCAL-Q1 local-arm initialization states differ"
                )


def _inverse_direction_target(target: Tensor, valid: Tensor) -> Tensor:
    inverse = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=target.device)
    safe = target.clamp(min=0)
    swapped = inverse.index_select(0, safe)
    return torch.where(valid, swapped, torch.full_like(swapped, -1))


def _quantile_999(value: Tensor, valid: Tensor) -> Optional[float]:
    selected = value[valid & torch.isfinite(value)].detach().cpu().sort().values
    if selected.numel() == 0:
        return None
    index = max(0, int(math.ceil(0.999 * int(selected.numel()))) - 1)
    return float(selected[index].item())


class LocalQ1Session:
    """One deterministic LOCAL-Q1 arm session with a typed tensor boundary."""

    def __init__(
        self,
        *,
        arm: AblationArm,
        seed: int,
        device: torch.device,
        mode: LocalQ1BackendMode,
        model_config: PairwiseModelConfig,
        portable_model_config: Mapping[str, Any],
        optimizer_config: LocalQ1OptimizerConfig,
        step_config: LocalQ1StepConfig,
        exact_seam_step_config: ExactSeamStepConfig,
        fairness: _FairnessLedger,
        checkpoint_bindings: Optional[FusedCheckpointBindings],
    ) -> None:
        self.arm = arm
        self._uses_exact_assignment = (
            arm.name is AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM
        )
        self.device = device
        self.mode = mode
        self.score_source = (
            PairwiseScoreSource.FUSED
            if arm.name is AblationArmName.FUSED
            else PairwiseScoreSource.LOCAL
        )
        fused = arm.name is AblationArmName.FUSED
        if fused and mode is LocalQ1BackendMode.FORMAL:
            if checkpoint_bindings is None:
                raise LocalQ1BackendError(
                    "formal fused session lacks trusted checkpoint bindings"
                )
            # Scope-specific semantic commitments are verified against the
            # actual target before determinism state or a model is created.
            checkpoint_bindings.require_target(model_config)
        self.determinism_config = _configure_determinism(seed)
        self.model_config = _thaw(portable_model_config)
        self.optimizer_config = _thaw(optimizer_config.to_dict())
        self._step_config = step_config
        self._step_config_sha256 = canonical_config_hash(step_config.to_dict())
        self._exact_seam_step_config = exact_seam_step_config
        self._exact_assignment_loss_weight = (
            float(exact_seam_step_config.loss_weight)
            if self._uses_exact_assignment
            else 0.0
        )
        self._fairness = fairness
        self._forward_counts = {"coarse": 0, "local": 0, "fusion": 0}
        self._execution_lock = threading.Lock()
        self._tainted = False
        self._taint_reason: Optional[str] = None
        self.model = _ControlledPairwiseModel(model_config, fused_freeze=fused)

        checkpoint_receipts: Dict[str, Mapping[str, Any]] = {}
        if fused and mode is LocalQ1BackendMode.FORMAL:
            if checkpoint_bindings is None:  # defensive; checked before model creation
                raise LocalQ1BackendError(
                    "formal fused session lost checkpoint bindings"
                )
            coarse_source = _coarse_model(model_config)
            coarse_receipt = _load_binding(checkpoint_bindings.coarse, coarse_source)
            local_binding = checkpoint_bindings.local
            if local_binding.model_scope is CheckpointModelScope.LOCAL_MODEL:
                local_source: Union[OrderedLocalMatcher, DunhuangPairwiseV02]
                local_source = _ordered_local_matcher(
                    model_config, str(local_binding.local_matcher_mode)
                )
                local_receipt = _load_binding(local_binding, local_source)
                local_state = local_source.state_dict()
            else:
                source_config = _model_for_matcher(
                    model_config, str(local_binding.local_matcher_mode)
                )
                local_source = DunhuangPairwiseV02(source_config)
                local_receipt = _load_binding(local_binding, local_source)
                local_state = local_source.local_model.state_dict()
            # Both sources are fully verified before the production target is
            # changed.  A failed second lock can never yield a usable session.
            self.model.coarse_model.load_state_dict(
                coarse_source.state_dict(), strict=True
            )
            self.model.local_model.load_state_dict(local_state, strict=True)
            checkpoint_receipts = {
                "coarse": coarse_receipt,
                "local": local_receipt,
            }

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        trainable_module = self.model.fusion if fused else self.model.local_model
        for parameter in trainable_module.parameters():
            parameter.requires_grad_(True)
        self.trainable_parameter_names = tuple(
            name for name, value in self.model.named_parameters() if value.requires_grad
        )
        self.frozen_parameter_names = tuple(
            name
            for name, value in self.model.named_parameters()
            if not value.requires_grad
        )
        self.model.to(device)
        trainable_parameters = tuple(
            value for value in self.model.parameters() if value.requires_grad
        )
        self.optimizer = _make_optimizer(trainable_parameters, optimizer_config)

        self.initial_model_state_sha256 = canonical_tensor_tree_sha256(
            self.model.state_dict()
        )
        if not fused:
            fairness.require_common_local_initialization(
                self.initial_model_state_sha256
            )
        self.initialization_authority = (
            "trusted_external_c0_and_local_checkpoints"
            if fused and mode is LocalQ1BackendMode.FORMAL
            else (
                "fixture_non_result_random_evidence"
                if fused
                else "runner_common_seed_random_local_initialization"
            )
        )
        self.checkpoint_receipts = MappingProxyType(checkpoint_receipts)
        forward_counts = self._forward_counts

        def count_forward(name: str):
            def hook(*_arguments: Any) -> None:
                forward_counts[name] += 1

            return hook

        self._hooks = (
            self.model.coarse_model.register_forward_hook(count_forward("coarse")),
            self.model.local_model.register_forward_hook(count_forward("local")),
            self.model.fusion.register_forward_hook(count_forward("fusion")),
        )

    @property
    def step_config(self) -> LocalQ1StepConfig:
        return self._step_config

    @property
    def step_config_sha256(self) -> str:
        return self._step_config_sha256

    @property
    def exact_assignment_loss_weight(self) -> float:
        """Return the active auxiliary weight without changing optimizer state."""

        return self._exact_assignment_loss_weight

    def set_exact_assignment_loss_weight(self, value: float) -> None:
        """Change only the exact arm's runtime loss coefficient.

        The model, optimizer object, and optimizer moments remain untouched so
        a multi-epoch schedule is one continuous training trajectory.
        """

        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("exact assignment loss weight must be finite")
        parsed = float(value)
        if parsed < 0.0:
            raise ValueError("exact assignment loss weight must be non-negative")
        if not self._uses_exact_assignment and parsed != 0.0:
            raise LocalQ1BackendError(
                "weak arm exact assignment loss weight must remain zero"
            )
        self._exact_assignment_loss_weight = parsed

    @property
    def is_tainted(self) -> bool:
        return self._tainted

    @property
    def taint_reason(self) -> Optional[str]:
        return self._taint_reason

    def _require_clean(self) -> None:
        if self._tainted:
            raise LocalQ1BackendError(
                "LOCAL-Q1 session is permanently tainted after {}".format(
                    self._taint_reason or "a late execution failure"
                )
            )

    def _require_step_config_integrity(self) -> None:
        if not isinstance(self._step_config, LocalQ1StepConfig):
            raise LocalQ1BackendError("session step config type changed")
        observed = canonical_config_hash(self._step_config.to_dict())
        if not hmac.compare_digest(observed, self._step_config_sha256):
            raise LocalQ1BackendError("session step config changed after construction")
        if self.mode is LocalQ1BackendMode.FORMAL:
            _validate_formal_step_config(self._step_config)

    @contextmanager
    def _taint_on_late_failure(self, operation: str) -> Iterator[None]:
        """Permanently reject reuse after execution/mutation has begun."""

        try:
            yield
        except BaseException as exc:
            self._tainted = True
            self._taint_reason = "{}:{}".format(operation, type(exc).__name__)
            raise

    def _checked_payload(self, batch: PreparedAblationBatch) -> RaggedGeometryBatch:
        """Validate exact CPU payload commitments before any model mutation."""

        if not isinstance(batch, PreparedAblationBatch):
            raise TypeError("batch must be PreparedAblationBatch")
        payload = batch.payload
        if not isinstance(payload, RaggedGeometryBatch):
            raise LocalQ1BackendError("LOCAL-Q1 payload must be RaggedGeometryBatch")
        if batch.sample_count != payload.batch_size:
            raise LocalQ1BackendError("prepared batch cardinality changed")
        if batch.local_candidate_sha256 is None:
            raise LocalQ1BackendError("LOCAL-Q1 batch lacks a candidate digest")
        if (
            batch.candidate_representation != self.arm.candidate_representation
            or payload.candidate_representation != self.arm.candidate_representation
        ):
            raise LocalQ1BackendError(
                "candidate representation differs from LOCAL-Q1 arm"
            )
        if batch.geometry_config_sha256 is None:
            raise LocalQ1BackendError("geometry config digest is missing")
        if (
            payload.candidate_representation == MULTIRUN_REPRESENTATION
            and not hmac.compare_digest(
                batch.geometry_config_sha256, payload.config_fingerprint
            )
        ):
            raise LocalQ1BackendError("geometry config digest differs from payload")
        if payload.candidate_representation == KEYPOINT_REPRESENTATION:
            base_geometry = payload.config_provenance.get("base_geometry_config_sha256")
            if not isinstance(base_geometry, str) or not hmac.compare_digest(
                batch.geometry_config_sha256, base_geometry
            ):
                raise LocalQ1BackendError(
                    "base geometry config digest differs from keypoint payload"
                )

        # This public provider API is the sole authority for the serialization
        # recipe.  Both commitments are checked before `.to(device)`, forward,
        # zero_grad, backward, or optimizer.step can mutate runtime state.
        prepared_sha, local_sha = local_q1_prepared_digests(payload)
        if not hmac.compare_digest(prepared_sha, batch.prepared_input_sha256):
            raise LocalQ1BackendError(
                "prepared input digest does not match payload tensors"
            )
        if not hmac.compare_digest(local_sha, batch.local_candidate_sha256):
            raise LocalQ1BackendError(
                "local candidate digest does not match payload tensors"
            )
        if (
            payload.coarse_a.dtype != torch.float32
            or payload.coarse_b.dtype != torch.float32
        ):
            raise LocalQ1BackendError("coarse tensors must use float32")
        if (
            payload.local_a.dtype != torch.float32
            or payload.local_b.dtype != torch.float32
        ):
            raise LocalQ1BackendError("local tensors must use float32")
        for name in ("coarse_a", "coarse_b", "local_a", "local_b"):
            if not torch.isfinite(getattr(payload, name)).all().item():
                raise LocalQ1BackendError("{} contains non-finite values".format(name))
        if payload.candidate_count <= 0:
            raise LocalQ1BackendError("LOCAL-Q1 batch has no local candidates")
        if any(name.startswith("exact_") for name in payload.model_inputs()):
            raise LocalQ1BackendError(
                "exact supervision escaped into model-facing tensors"
            )
        if (
            batch.processing_counts.get("local_candidate_count")
            != payload.candidate_count
        ):
            raise LocalQ1BackendError("provider candidate counter differs from payload")
        self._fairness.require_batch(batch)
        return payload

    def _memory_start(self) -> Tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        torch.cuda.synchronize(self.device)
        start = int(torch.cuda.memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        return start, start

    def _memory_end(self, start: int) -> Tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        torch.cuda.synchronize(self.device)
        end = int(torch.cuda.memory_allocated(self.device))
        peak = int(torch.cuda.max_memory_allocated(self.device))
        return max(0, end - start), max(0, peak - start)

    @staticmethod
    def _validate_output(
        output: HierarchicalDirectionalOutput, batch_size: int
    ) -> None:
        pair = output.direction_output
        training = output.training_direction_output
        for name, value in (
            ("pair_logit", pair.pair_logit),
            ("pair_probability", pair.pair_probability),
            ("training_pair_logit", training.pair_logit),
            ("training_pair_probability", training.pair_probability),
        ):
            if tuple(value.shape) != (batch_size,):
                raise LocalQ1BackendError("{} cardinality changed".format(name))
            if not torch.isfinite(value).all().item():
                raise LocalQ1BackendError("{} is non-finite".format(name))
        if (
            pair.pair_valid.dtype != torch.bool
            or training.pair_valid.dtype != torch.bool
        ):
            raise LocalQ1BackendError("pair validity must be bool")
        if not torch.isfinite(output.arc_logits).all().item():
            raise LocalQ1BackendError("arc logits are non-finite")
        valid_probability = pair.pair_probability[pair.pair_valid]
        if valid_probability.numel() and (
            (valid_probability < 0.0).any().item()
            or (valid_probability > 1.0).any().item()
        ):
            raise LocalQ1BackendError("valid probabilities escaped [0, 1]")

    def _swap_errors(
        self,
        first: HierarchicalDirectionalOutput,
        second: HierarchicalDirectionalOutput,
    ) -> Tuple[float, float]:
        first_pair = first.direction_output
        second_pair = second.direction_output
        if not torch.equal(first_pair.pair_valid, second_pair.pair_valid):
            raise LocalQ1BackendError("A/B swap changed pair validity")
        logit = float(
            (first_pair.pair_logit - second_pair.pair_logit).abs().max().detach().cpu()
        )
        probability = float(
            (first_pair.pair_probability - second_pair.pair_probability)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        if not math.isfinite(logit) or not math.isfinite(probability):
            raise LocalQ1BackendError("A/B swap diagnostics are non-finite")
        if max(logit, probability) > self.step_config.swap_tolerance:
            raise LocalQ1BackendError("A/B swap error exceeds frozen tolerance")
        return logit, probability

    def _diagnostics(
        self,
        step: DirectionalStepResult,
        *,
        payload: RaggedGeometryBatch,
        count_before: Mapping[str, int],
        elapsed_seconds: float,
        memory_delta: int,
        memory_peak_delta: int,
        swap_errors: Tuple[float, float],
    ) -> Dict[str, Any]:
        transport = step.transport_diagnostics
        if transport is None:
            raise LocalQ1BackendError("LOCAL-Q1 transport diagnostics are missing")
        if transport.matcher_mode != self.arm.matcher_mode:
            raise LocalQ1BackendError("matcher diagnostics differ from arm")
        deltas = {
            name: self._forward_counts[name] - int(count_before[name])
            for name in self._forward_counts
        }
        if self.score_source is PairwiseScoreSource.LOCAL:
            if deltas["coarse"] != 0 or deltas["fusion"] != 0:
                raise LocalQ1BackendError(
                    "local-only arm executed coarse or fusion computation"
                )
        else:
            if deltas["coarse"] <= 0 or deltas["fusion"] <= 0:
                raise LocalQ1BackendError("fused arm skipped coarse/fusion evidence")
        local_output = step.output.arc_pairwise_output
        if local_output is None:
            raise LocalQ1BackendError("LOCAL-Q1 output lacks arc evidence")
        finite = local_output.local.finite_problem
        row_p999 = _quantile_999(local_output.local.row_residual_max, finite)
        col_p999 = _quantile_999(local_output.local.col_residual_max, finite)
        dustbin_forwards = (
            deltas["local"]
            if self.arm.matcher_mode == MatcherMode.DUSTBIN_SINKHORN.value
            else 0
        )
        provider_valid = int(payload.candidate_valid.sum().item())
        diagnostics: Dict[str, Any] = {
            "matcher_mode": transport.matcher_mode,
            "candidate_representation": payload.candidate_representation,
            "score_source": self.score_source.value,
            "candidate_count": transport.candidate_count,
            "finite_problem_count": transport.finite_problem_count,
            "converged_count": transport.converged_count,
            "decision_valid_count": transport.decision_valid_count,
            "nonconverged_finite_count": transport.nonconverged_finite_count,
            "row_residual_max": transport.row_residual_max,
            "col_residual_max": transport.col_residual_max,
            "row_residual_p99_9": row_p999,
            "col_residual_p99_9": col_p999,
            "sinkhorn_iteration_min": transport.iteration_min,
            "sinkhorn_iteration_max": transport.iteration_max,
            "coarse_forward_count": deltas["coarse"],
            "local_forward_count": deltas["local"],
            "fusion_forward_count": deltas["fusion"],
            "dustbin_local_forward_count": dustbin_forwards,
            "sinkhorn_problem_count": dustbin_forwards * payload.candidate_count,
            "sinkhorn_provider_valid_problem_count": (
                dustbin_forwards * provider_valid
            ),
            "provider_valid_local_problem_count": (deltas["local"] * provider_valid),
            "provider_candidate_count_per_forward": payload.candidate_count,
            "provider_candidate_valid_count_per_forward": provider_valid,
            "provider_geometry_valid_sample_count": int(
                payload.geometry_valid.sum().item()
            ),
            "effective_training_valid_candidate_count_per_forward": int(
                step.output.arc_training_valid.sum().item()
            ),
            "effective_decision_valid_candidate_count_per_forward": int(
                step.output.arc_valid.sum().item()
            ),
            "pair_training_valid_count": int(
                step.output.training_direction_output.pair_valid.sum().item()
            ),
            "pair_decision_valid_count": int(
                step.output.direction_output.pair_valid.sum().item()
            ),
            "ab_swap_logit_max_abs_error": swap_errors[0],
            "ab_swap_probability_max_abs_error": swap_errors[1],
            "elapsed_seconds": elapsed_seconds,
            "device_memory_end_minus_start_bytes": memory_delta,
            "device_memory_peak_above_start_bytes": memory_peak_delta,
            "device_type": self.device.type,
            "initialization_authority": self.initialization_authority,
            "result_eligible": self.mode is LocalQ1BackendMode.FORMAL,
            "result_eligibility_authority_boundary": (
                LOCAL_Q1_AUTHORITY_BOUNDARY
                if self.score_source is PairwiseScoreSource.FUSED
                and self.mode is LocalQ1BackendMode.FORMAL
                else "not_applicable"
            ),
            "authority_receipt_opened_by_backend": False,
            "trainable_parameter_count": len(self.trainable_parameter_names),
            "frozen_parameter_count": len(self.frozen_parameter_names),
            "prepared_digest_verified_before_execution": True,
            "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
            "exact_assignment_supervision_enabled": self._uses_exact_assignment,
            "exact_assignment_loss_weight": self._exact_assignment_loss_weight,
        }
        exact = step.exact_assignment_breakdown
        diagnostics.update(
            {
                "exact_assignment_loss": float(
                    step.exact_assignment_loss.detach().cpu()
                ),
                "exact_supervised_match_count": (
                    0 if exact is None else exact.supervised_match_count
                ),
                "exact_supervised_dustbin_a_count": (
                    0 if exact is None else exact.supervised_dustbin_a_count
                ),
                "exact_supervised_dustbin_b_count": (
                    0 if exact is None else exact.supervised_dustbin_b_count
                ),
            }
        )
        return diagnostics

    def _step_arguments(self) -> Mapping[str, Any]:
        return {
            "score_source": self.score_source,
            "coarse_loss_weight": self.step_config.coarse_loss_weight,
            "direction_loss_weight": self.step_config.direction_loss_weight,
            "sinkhorn_residual_weight": self.step_config.sinkhorn_residual_weight,
            "sinkhorn_residual_target": self.step_config.sinkhorn_residual_target,
        }

    def _train_executed(
        self, payload: RaggedGeometryBatch, moved: RaggedGeometryBatch
    ) -> TrainBatchResult:
        counts = dict(self._forward_counts)
        memory_start, _ = self._memory_start()
        started = time.perf_counter()
        step = train_directional_step(
            self.model,
            moved.model_inputs(),
            moved.labels,
            moved.direction_target,
            moved.direction_target_valid,
            self.optimizer,
            **self._step_arguments(),
            compute_swapped=True,
            swap_loss_weight=self.step_config.swap_loss_weight,
            swap_transport_weight=self.step_config.swap_transport_weight,
            exact_assignment_targets=(
                moved.exact_loss_targets() if self._uses_exact_assignment else None
            ),
            exact_assignment_loss_weight=(
                self._exact_assignment_loss_weight
                if self._uses_exact_assignment
                else 0.0
            ),
            exact_match_weight=self._exact_seam_step_config.match_weight,
            exact_dustbin_weight=self._exact_seam_step_config.dustbin_weight,
            max_gradient_norm=self.step_config.max_gradient_norm,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        memory_delta, memory_peak = self._memory_end(memory_start)
        self._validate_output(step.output, moved.batch_size)
        if step.swapped_output is None:
            raise LocalQ1BackendError("training omitted required A/B swap pass")
        self._validate_output(step.swapped_output, moved.batch_size)
        swap_errors = self._swap_errors(step.output, step.swapped_output)
        if step.gradient_norm is None or not math.isfinite(step.gradient_norm):
            raise LocalQ1BackendError("training gradient norm is non-finite")
        saw_gradient = False
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if parameter.requires_grad:
                if gradient is not None:
                    if not torch.isfinite(gradient).all().item():
                        raise LocalQ1BackendError(
                            "trainable gradient is non-finite: {}".format(name)
                        )
                    saw_gradient = saw_gradient or bool(
                        gradient.abs().sum().item() > 0.0
                    )
            elif gradient is not None:
                raise LocalQ1BackendError("frozen parameter received a gradient")
        if not saw_gradient:
            raise LocalQ1BackendError("no trainable parameter received a gradient")
        if any(
            not torch.isfinite(parameter).all().item()
            for parameter in self.model.parameters()
        ):
            raise LocalQ1BackendError("optimizer produced non-finite parameters")
        diagnostics = self._diagnostics(
            step,
            payload=payload,
            count_before=counts,
            elapsed_seconds=elapsed,
            memory_delta=memory_delta,
            memory_peak_delta=memory_peak,
            swap_errors=swap_errors,
        )
        diagnostics["gradient_l2_norm_before_clip"] = step.gradient_norm
        valid_count = int(step.output.training_direction_output.pair_valid.sum().item())
        return TrainBatchResult(
            loss=float(step.total_loss.detach().cpu()),
            valid_count=valid_count,
            diagnostics=diagnostics,
        )

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        with self._execution_lock:
            self._require_clean()
            self._require_step_config_integrity()
            payload = self._checked_payload(batch)
            if self._uses_exact_assignment and payload.exact_loss_targets() is None:
                raise LocalQ1BackendError(
                    "exact-seam training arm requires assignment targets"
                )
            moved = payload.to(self.device)
            with self._taint_on_late_failure("train_batch"):
                return self._train_executed(payload, moved)

    def _predict_executed(
        self, payload: RaggedGeometryBatch, moved: RaggedGeometryBatch
    ) -> PredictionBatch:
        counts = dict(self._forward_counts)
        memory_start, _ = self._memory_start()
        started = time.perf_counter()
        primary = eval_directional_step(
            self.model,
            moved.model_inputs(),
            moved.labels,
            moved.direction_target,
            moved.direction_target_valid,
            **self._step_arguments(),
        )
        swapped = eval_directional_step(
            self.model,
            swap_directional_model_inputs(moved.model_inputs()),
            moved.labels,
            _inverse_direction_target(
                moved.direction_target, moved.direction_target_valid
            ),
            moved.direction_target_valid,
            **self._step_arguments(),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        memory_delta, memory_peak = self._memory_end(memory_start)
        self._validate_output(primary.output, moved.batch_size)
        self._validate_output(swapped.output, moved.batch_size)
        swap_errors = self._swap_errors(primary.output, swapped.output)
        diagnostics = self._diagnostics(
            primary,
            payload=payload,
            count_before=counts,
            elapsed_seconds=elapsed,
            memory_delta=memory_delta,
            memory_peak_delta=memory_peak,
            swap_errors=swap_errors,
        )
        probability = primary.output.direction_output.pair_probability.detach()
        valid = (
            primary.output.direction_output.pair_valid & moved.geometry_valid
        ).detach()
        return PredictionBatch(
            probability=probability,
            valid=valid,
            diagnostics=diagnostics,
        )

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch:
        with self._execution_lock:
            self._require_clean()
            self._require_step_config_integrity()
            expected = (
                EvidenceMode.FUSED
                if self.score_source is PairwiseScoreSource.FUSED
                else EvidenceMode.LOCAL
            )
            if EvidenceMode(getattr(evidence, "value", evidence)) is not expected:
                raise LocalQ1BackendError("prediction evidence differs from arm")
            payload = self._checked_payload(batch)
            moved = payload.to(self.device)
            with self._taint_on_late_failure("predict_batch"):
                return self._predict_executed(payload, moved)


class LocalQ1Backend:
    """Factory and frozen configuration authority for LOCAL-Q1 research arms."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        *,
        mode: LocalQ1BackendMode = LocalQ1BackendMode.FORMAL,
        model_template: Optional[PairwiseModelConfig] = None,
        optimizer_config: Optional[LocalQ1OptimizerConfig] = None,
        step_config: Optional[LocalQ1StepConfig] = None,
        exact_seam_step_config: Optional[ExactSeamStepConfig] = None,
        fused_checkpoint_bindings: Optional[FusedCheckpointBindings] = None,
    ) -> None:
        parsed_device = torch.device(device)
        if parsed_device.type not in {"cpu", "cuda", "mps"}:
            raise LocalQ1BackendError("LOCAL-Q1 device is unsupported")
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise LocalQ1BackendError("CUDA was requested but is unavailable")
        try:
            parsed_mode = LocalQ1BackendMode(getattr(mode, "value", mode))
        except ValueError as exc:
            raise LocalQ1BackendError("LOCAL-Q1 backend mode is unsupported") from exc
        template = model_template or _formal_model_template()
        if not isinstance(template, PairwiseModelConfig):
            raise TypeError("model_template must be PairwiseModelConfig")
        if parsed_mode is LocalQ1BackendMode.FORMAL:
            _validate_formal_numerics(template)
        selected_step = step_config or LocalQ1StepConfig()
        if not isinstance(selected_step, LocalQ1StepConfig):
            raise TypeError("step_config must be LocalQ1StepConfig")
        if parsed_mode is LocalQ1BackendMode.FORMAL:
            _validate_formal_step_config(selected_step)
        selected_exact_step = exact_seam_step_config or ExactSeamStepConfig()
        if not isinstance(selected_exact_step, ExactSeamStepConfig):
            raise TypeError("exact_seam_step_config must be ExactSeamStepConfig")
        if fused_checkpoint_bindings is not None and not isinstance(
            fused_checkpoint_bindings, FusedCheckpointBindings
        ):
            raise TypeError("fused checkpoint bindings must be typed")
        self.device = parsed_device
        self.mode = parsed_mode
        self.model_template = template
        self._optimizer_spec = optimizer_config or LocalQ1OptimizerConfig()
        self._step_config = selected_step
        self._exact_seam_step_config = selected_exact_step
        self.fused_checkpoint_bindings = fused_checkpoint_bindings
        self._fairness = _FairnessLedger()
        self.optimizer_config = _thaw(self._optimizer_spec.to_dict())
        pooling = template.arc_pooling
        self.aggregation_config = {
            "arc_pooling": {
                "mode": pooling.mode.value,
                "temperature": pooling.temperature,
                "top_k": pooling.top_k,
            },
            "direction_pooling": "log_mean_exp",
            "direction_temperature": template.direction_aggregation_temperature,
            "candidate_population": (
                "same_frozen_record_plan_and_representation_specific_provider_payload"
            ),
        }
        execution_kind = (
            ExecutionKind.SYNTHETIC_TRAIN_VALIDATION
            if parsed_mode is LocalQ1BackendMode.FORMAL
            else ExecutionKind.PROVIDER_FREE_DRY_RUN
        )
        self.contract = BackendContract(
            execution_kind=execution_kind,
            backend_version=LOCAL_Q1_BACKEND_VERSION + "/" + parsed_mode.value,
            model_family="DunhuangPairwiseV02-LOCAL-Q1-five-arm",
            device_type=parsed_device.type,
            sealed_real_test_capability=False,
        )

    @property
    def step_config(self) -> LocalQ1StepConfig:
        return self._step_config

    @staticmethod
    def _arm_contract(name: AblationArmName) -> Tuple[EvidenceMode, str]:
        contracts = {
            AblationArmName.LOCAL_DUAL_SOFTMAX: (
                EvidenceMode.LOCAL,
                MatcherMode.DUAL_SOFTMAX.value,
            ),
            AblationArmName.LOCAL_DUSTBIN_SINKHORN: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.KEYPOINT_DUAL_SOFTMAX: (
                EvidenceMode.LOCAL,
                MatcherMode.DUAL_SOFTMAX.value,
            ),
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM: (
                EvidenceMode.LOCAL,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
            AblationArmName.FUSED: (
                EvidenceMode.FUSED,
                MatcherMode.DUSTBIN_SINKHORN.value,
            ),
        }
        try:
            return contracts[name]
        except KeyError as exc:
            raise LocalQ1BackendError(
                "LOCAL-Q1 backend serves multi-run/keypoint local arms and fused"
            ) from exc

    def model_config_for(self, name: AblationArmName) -> Mapping[str, Any]:
        """Return the exact arm config that a preregistration must freeze."""

        parsed = AblationArmName(getattr(name, "value", name))
        _evidence, matcher = self._arm_contract(parsed)
        architecture = _model_for_matcher(self.model_template, matcher)
        fused = parsed is AblationArmName.FUSED
        candidate_representation = (
            KEYPOINT_REPRESENTATION
            if parsed
            in {
                AblationArmName.KEYPOINT_DUAL_SOFTMAX,
                AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
                AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM,
            }
            else MULTIRUN_REPRESENTATION
        )
        if fused and self.mode is LocalQ1BackendMode.FORMAL:
            if self.fused_checkpoint_bindings is None:
                raise LocalQ1BackendError(
                    "formal fused config requires trusted C0/local checkpoints"
                )
            initialization: Mapping[str, Any] = {
                "kind": "trusted_external_checkpoints",
                "bindings": self.fused_checkpoint_bindings.portable_identity(),
                "random_evidence_allowed": False,
                "result_eligible": True,
                "result_eligibility_condition": (
                    "future_runner_validated_both_authority_receipts_before_"
                    "backend_construction"
                ),
                "authority_receipt_opened_by_backend": False,
            }
        elif fused:
            initialization = {
                "kind": "fixture_non_result_random_evidence",
                "bindings": None,
                "random_evidence_allowed": True,
                "result_eligible": False,
            }
        else:
            initialization = {
                "kind": "runner_common_seed_random_local_initialization",
                "bindings": None,
                "random_evidence_allowed": False,
                "result_eligible": self.mode is LocalQ1BackendMode.FORMAL,
            }
        value = {
            "schema_version": LOCAL_Q1_MODEL_CONFIG_VERSION,
            "architecture": architecture.to_dict(),
            "candidate_representation": candidate_representation,
            "contour_keypoint_config": (
                asdict(ContourKeypointConfig())
                if candidate_representation == KEYPOINT_REPRESENTATION
                else None
            ),
            "initialization": initialization,
            "training": {
                "trainable_components": ["fusion" if fused else "local_model"],
                "frozen_components": (
                    ["coarse_model", "local_model"]
                    if fused
                    else ["coarse_model", "fusion"]
                ),
                "score_source": "fused" if fused else "local",
                "step_config": self.step_config.to_dict(),
                "step_config_sha256": canonical_config_hash(self.step_config.to_dict()),
                "exact_assignment_supervision": (
                    {
                        "enabled": True,
                        "loss": "sample_balanced_partial_assignment_nll",
                        **dict(self._exact_seam_step_config.to_dict()),
                        "target_source": "aligned_mask_exact_4_neighbor_seam",
                        "targets_are_model_inputs": False,
                    }
                    if parsed is AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM
                    else {"enabled": False}
                ),
            },
            "provider_tensor_contract": {
                "provider_version": LOCAL_Q1_BATCH_PROVIDER_VERSION,
                "payload_type": "RaggedGeometryBatch",
                "candidate_representation": candidate_representation,
                "correspondence_mask_preserved": True,
                "prepared_and_local_digests_recomputed_before_execution": True,
                "exact_targets_required": (
                    parsed is AblationArmName.KEYPOINT_DUSTBIN_SINKHORN_EXACT_SEAM
                ),
            },
        }
        return _freeze(value)

    def _require_arm(self, arm: AblationArm) -> PairwiseModelConfig:
        if not isinstance(arm, AblationArm):
            raise TypeError("arm must be AblationArm")
        expected_evidence, matcher = self._arm_contract(arm.name)
        if arm.evidence is not expected_evidence or arm.matcher_mode != matcher:
            raise LocalQ1BackendError("LOCAL-Q1 arm evidence/matcher is inconsistent")
        if arm.arc_pooling != self.model_template.arc_pooling:
            raise LocalQ1BackendError("LOCAL-Q1 arc pooling differs across arms")
        expected_model = self.model_config_for(arm.name)
        if not _same_config(arm.model_config, expected_model):
            raise LocalQ1BackendError("arm model config differs from backend freeze")
        if not _same_config(arm.optimizer_config, self.optimizer_config):
            raise LocalQ1BackendError(
                "arm optimizer config differs from backend freeze"
            )
        if not _same_config(arm.aggregation_config, self.aggregation_config):
            raise LocalQ1BackendError("arm aggregation config differs across arms")
        return _model_for_matcher(self.model_template, matcher)

    def create_session(self, arm: AblationArm, *, seed: int) -> LocalQ1Session:
        config = self._require_arm(arm)
        if (
            arm.name is AblationArmName.FUSED
            and self.mode is LocalQ1BackendMode.FORMAL
            and self.fused_checkpoint_bindings is None
        ):
            raise LocalQ1BackendError(
                "formal fused session requires trusted checkpoint bindings"
            )
        return LocalQ1Session(
            arm=arm,
            seed=seed,
            device=self.device,
            mode=self.mode,
            model_config=config,
            portable_model_config=self.model_config_for(arm.name),
            optimizer_config=self._optimizer_spec,
            step_config=self.step_config,
            exact_seam_step_config=self._exact_seam_step_config,
            fairness=self._fairness,
            checkpoint_bindings=self.fused_checkpoint_bindings,
        )


__all__ = [
    "CheckpointModelScope",
    "FusedCheckpointBindings",
    "ExactSeamStepConfig",
    "LOCAL_Q1_AUTHORITY_BOUNDARY",
    "LOCAL_Q1_AUTHORITY_KIND",
    "LOCAL_Q1_AUTHORITY_STATUS",
    "LOCAL_Q1_BACKEND_VERSION",
    "LOCAL_Q1_CHECKPOINT_BINDING_VERSION",
    "LOCAL_Q1_MODEL_CONFIG_VERSION",
    "LOCAL_Q1_COARSE_INPUT_SHAPE",
    "LOCAL_Q1_SWAP_TOLERANCE",
    "LocalQ1Backend",
    "LocalQ1BackendError",
    "LocalQ1BackendMode",
    "LocalQ1OptimizerConfig",
    "LocalQ1Session",
    "LocalQ1StepConfig",
    "TrustedCheckpointBinding",
    "checkpoint_authority_identity_sha256",
    "checkpoint_semantic_projection",
]
