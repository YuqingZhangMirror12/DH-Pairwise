"""Frozen coarse-only backend for the C0-N-Q1 qualification run.

This module owns only model/session computation.  It does not open archives,
prepare masks, select records, save checkpoints, or access the sealed real
Dunhuang set.  The payload hash is recomputed at every session boundary so a
provider cannot substitute tensors while retaining an earlier digest.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import math
import random
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from staging.pairwise_v0_2.models.coarse import SymmetricCoarseSiamese
from staging.pairwise_v0_2.training.checkpoint import canonical_config_hash
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


C0_COARSE_BACKEND_VERSION = "c0-n-q1-coarse-backend/0.2"
C0_COARSE_PAYLOAD_VERSION = "c0-n-q1-coarse-payload/0.2"
C0_COARSE_SEED = 260828
C0_COARSE_SWAP_TOLERANCE = 1e-6

C0_COARSE_MODEL_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "name": "SymmetricCoarseSiamese",
        "input_channels": 1,
        "widths": (16, 32, 64),
        "embedding_dim": 96,
        "hidden_dim": 96,
        "input_shape": (1, 128, 128),
    }
)
C0_COARSE_OPTIMIZER_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "name": "AdamW",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        # Explicitly frozen.  PyTorch 1.13 supports ``foreach`` but not
        # ``fused``; session construction conditionally omits unsupported
        # keyword arguments while retaining the same portable config.
        "foreach": False,
        "fused": None,
    }
)

_ZERO_LOCAL_PROCESSING = {
    "geometry_build_count": 0,
    "geometry_cache_read_count": 0,
    "geometry_cache_write_count": 0,
    "local_candidate_count": 0,
}


class C0CoarseBackendError(ShortAblationError):
    """Raised when a C0 session boundary or numerical invariant is violated."""


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C0CoarseBackendError("{} must be a lowercase SHA-256".format(name))


def _tensor_bytes(value: Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    # The payload contract freezes float32, and little-endian bytes make the
    # digest portable across otherwise compatible hosts.
    return array.astype("<f4", copy=False).tobytes(order="C")


def c0_coarse_payload_content_sha256(
    coarse_a: Tensor, coarse_b: Tensor, labels: Tensor
) -> str:
    """Hash exact tensor content, shapes, dtypes and field boundaries."""

    digest = hashlib.sha256()
    digest.update(C0_COARSE_PAYLOAD_VERSION.encode("utf-8"))
    for name, value in (
        ("coarse_a", coarse_a),
        ("coarse_b", coarse_b),
        ("labels", labels),
    ):
        if not isinstance(value, Tensor):
            raise C0CoarseBackendError("{} must be a torch.Tensor".format(name))
        if value.layout != torch.strided or value.device.type == "meta":
            raise C0CoarseBackendError(
                "{} must be a materialized strided tensor".format(name)
            )
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = _tensor_bytes(value)
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class C0CoarsePayload:
    """Exact provider-to-backend tensor payload for C0-N-Q1."""

    coarse_a: Tensor
    coarse_b: Tensor
    labels: Tensor
    payload_content_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.payload_content_sha256, "payload content digest")

    @classmethod
    def from_tensors(
        cls, coarse_a: Tensor, coarse_b: Tensor, labels: Tensor
    ) -> "C0CoarsePayload":
        return cls(
            coarse_a=coarse_a,
            coarse_b=coarse_b,
            labels=labels,
            payload_content_sha256=c0_coarse_payload_content_sha256(
                coarse_a, coarse_b, labels
            ),
        )


def _same_config(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return hmac.compare_digest(
        canonical_config_hash(observed), canonical_config_hash(expected)
    )


def _require_c0_arm(arm: AblationArm) -> None:
    if not isinstance(arm, AblationArm):
        raise TypeError("arm must be AblationArm")
    if (
        arm.name is not AblationArmName.COARSE_ONLY
        or arm.evidence is not EvidenceMode.COARSE
        or arm.matcher_mode is not None
        or arm.arc_pooling is not None
    ):
        raise C0CoarseBackendError("C0 backend accepts only coarse-only evidence")
    if not _same_config(arm.model_config, C0_COARSE_MODEL_CONFIG):
        raise C0CoarseBackendError("C0 model config differs from the frozen config")
    if not _same_config(arm.optimizer_config, C0_COARSE_OPTIMIZER_CONFIG):
        raise C0CoarseBackendError("C0 optimizer config differs from the frozen config")


def _configure_determinism(seed: int) -> Mapping[str, Any]:
    if type(seed) is not int or seed != C0_COARSE_SEED:
        raise C0CoarseBackendError("C0-N-Q1 requires seed 260828")
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


def _optimizer(model: nn.Module) -> torch.optim.AdamW:
    config = C0_COARSE_OPTIMIZER_CONFIG
    arguments: Dict[str, Any] = {
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "betas": tuple(config["betas"]),
        "eps": float(config["eps"]),
        "amsgrad": bool(config["amsgrad"]),
    }
    parameters = inspect.signature(torch.optim.AdamW).parameters
    if "foreach" in parameters:
        arguments["foreach"] = config["foreach"]
    if "fused" in parameters:
        arguments["fused"] = config["fused"]
    return torch.optim.AdamW(model.parameters(), **arguments)


class C0CoarseSession:
    """One deterministic, coarse-only C0 model and optimizer session."""

    def __init__(self, arm: AblationArm, *, seed: int, device: torch.device) -> None:
        _require_c0_arm(arm)
        self.determinism_config = _configure_determinism(seed)
        self.device = device
        self.model_config = dict(C0_COARSE_MODEL_CONFIG)
        self.optimizer_config = dict(C0_COARSE_OPTIMIZER_CONFIG)
        self.model = SymmetricCoarseSiamese(
            input_channels=int(self.model_config["input_channels"]),
            widths=tuple(self.model_config["widths"]),
            embedding_dim=int(self.model_config["embedding_dim"]),
            hidden_dim=int(self.model_config["hidden_dim"]),
        ).to(device)
        self.optimizer = _optimizer(self.model)
        self.loss_function = nn.BCEWithLogitsLoss()

    def _checked_payload(
        self, batch: PreparedAblationBatch
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        if not isinstance(batch, PreparedAblationBatch):
            raise TypeError("batch must be PreparedAblationBatch")
        payload = batch.payload
        if not isinstance(payload, C0CoarsePayload):
            raise C0CoarseBackendError("C0 batch payload type is invalid")
        if batch.local_candidate_sha256 is not None:
            raise C0CoarseBackendError("coarse-only batch exposed local candidates")
        if batch.geometry_config_sha256 is not None:
            raise C0CoarseBackendError("coarse-only batch exposed geometry")
        for name, expected in _ZERO_LOCAL_PROCESSING.items():
            if batch.processing_counts.get(name) != expected:
                raise C0CoarseBackendError(
                    "coarse-only batch invoked geometry/cache/local processing"
                )

        a, b, labels = payload.coarse_a, payload.coarse_b, payload.labels
        for name, value in (("coarse_a", a), ("coarse_b", b), ("labels", labels)):
            if not isinstance(value, Tensor):
                raise C0CoarseBackendError("{} must be a tensor".format(name))
            if value.dtype != torch.float32:
                raise C0CoarseBackendError("{} must use float32".format(name))
        if a.ndim != 4 or tuple(a.shape[1:]) != (1, 128, 128):
            raise C0CoarseBackendError("coarse_a must have shape [B,1,128,128]")
        if tuple(b.shape) != tuple(a.shape):
            raise C0CoarseBackendError("coarse_b shape differs from coarse_a")
        batch_size = int(a.shape[0])
        if batch_size <= 0 or tuple(labels.shape) != (batch_size,):
            raise C0CoarseBackendError("labels must have shape [B]")
        if batch.sample_count != batch_size:
            raise C0CoarseBackendError("prepared batch cardinality changed")
        if not torch.isfinite(labels).all().item():
            raise C0CoarseBackendError("labels contain non-finite values")
        if not ((labels == 0.0) | (labels == 1.0)).all().item():
            raise C0CoarseBackendError("labels must be binary float32 values")

        observed = c0_coarse_payload_content_sha256(a, b, labels)
        if not hmac.compare_digest(observed, payload.payload_content_sha256):
            raise C0CoarseBackendError("payload content digest does not match tensors")
        if not hmac.compare_digest(observed, batch.prepared_input_sha256):
            raise C0CoarseBackendError("prepared input digest does not match payload")
        return (
            a.to(self.device, non_blocking=False),
            b.to(self.device, non_blocking=False),
            labels.to(self.device, non_blocking=False),
            batch_size,
        )

    @staticmethod
    def _checked_output(output: Any, batch_size: int) -> None:
        if tuple(output.logit.shape) != (batch_size,):
            raise C0CoarseBackendError("coarse model changed output cardinality")
        if tuple(output.probability.shape) != (batch_size,):
            raise C0CoarseBackendError("coarse probability cardinality changed")
        if tuple(output.valid_problem.shape) != (batch_size,):
            raise C0CoarseBackendError("coarse validity cardinality changed")
        if output.valid_problem.dtype != torch.bool:
            raise C0CoarseBackendError("coarse validity must be bool")
        if not output.valid_problem.all().item():
            raise C0CoarseBackendError("C0 contains an invalid coarse sample")
        if not torch.isfinite(output.logit).all().item():
            raise C0CoarseBackendError("coarse logits are non-finite")
        if not torch.isfinite(output.probability).all().item():
            raise C0CoarseBackendError("coarse probabilities are non-finite")

    @staticmethod
    def _zero_local_diagnostics() -> Dict[str, int]:
        return {
            "local_forward_count": 0,
            "sinkhorn_call_count": 0,
            "geometry_build_count": 0,
            "geometry_cache_read_count": 0,
            "geometry_cache_write_count": 0,
            "local_candidate_count": 0,
        }

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        a, b, labels, batch_size = self._checked_payload(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(a, b)
        self._checked_output(output, batch_size)
        loss = self.loss_function(output.logit, labels)
        if not torch.isfinite(loss).item():
            raise C0CoarseBackendError("C0 training loss is non-finite")
        loss.backward()

        squared_norm = 0.0
        for parameter in self.model.parameters():
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all().item():
                raise C0CoarseBackendError("C0 gradient is missing or non-finite")
            squared_norm += float(gradient.detach().double().square().sum().cpu())
        gradient_norm = math.sqrt(squared_norm)
        if not math.isfinite(gradient_norm):
            raise C0CoarseBackendError("C0 gradient norm is non-finite")
        self.optimizer.step()
        if any(
            not torch.isfinite(parameter).all().item()
            for parameter in self.model.parameters()
        ):
            raise C0CoarseBackendError("C0 optimizer produced non-finite parameters")

        diagnostics: Dict[str, Any] = {
            "coarse_forward_count": 1,
            "gradient_l2_norm": gradient_norm,
            "deterministic_algorithms": True,
            "tf32_enabled": False,
        }
        diagnostics.update(self._zero_local_diagnostics())
        return TrainBatchResult(
            loss=float(loss.detach().cpu()),
            valid_count=batch_size,
            diagnostics=diagnostics,
        )

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch:
        if evidence is not EvidenceMode.COARSE:
            raise C0CoarseBackendError("C0 prediction requires coarse evidence")
        a, b, _labels, batch_size = self._checked_payload(batch)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(a, b)
            swapped = self.model(b, a)
            self._checked_output(output, batch_size)
            self._checked_output(swapped, batch_size)
            swap_logit_error = float((output.logit - swapped.logit).abs().max().cpu())
            swap_probability_error = float(
                (output.probability - swapped.probability).abs().max().cpu()
            )
        if not math.isfinite(swap_logit_error) or not math.isfinite(
            swap_probability_error
        ):
            raise C0CoarseBackendError("A/B swap diagnostic is non-finite")
        if max(swap_logit_error, swap_probability_error) > C0_COARSE_SWAP_TOLERANCE:
            raise C0CoarseBackendError("A/B swap error exceeds 1e-6")

        diagnostics: Dict[str, Any] = {
            "coarse_forward_count": 2,
            "ab_swap_logit_max_abs_error": swap_logit_error,
            "ab_swap_probability_max_abs_error": swap_probability_error,
            "deterministic_algorithms": True,
            "tf32_enabled": False,
        }
        diagnostics.update(self._zero_local_diagnostics())
        return PredictionBatch(
            probability=output.probability,
            valid=torch.ones(batch_size, dtype=torch.bool, device=output.logit.device),
            diagnostics=diagnostics,
        )


class C0CoarseBackend:
    """Create frozen C0-N-Q1 sessions without any data-access capability."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        parsed = torch.device(device)
        if parsed.type not in {"cpu", "cuda", "mps"}:
            raise C0CoarseBackendError("C0 backend device is unsupported")
        if parsed.type == "cuda" and not torch.cuda.is_available():
            raise C0CoarseBackendError("CUDA was requested but is unavailable")
        self.device = parsed
        self.contract = BackendContract(
            execution_kind=ExecutionKind.SYNTHETIC_TRAIN_VALIDATION,
            backend_version=C0_COARSE_BACKEND_VERSION,
            model_family="SymmetricCoarseSiamese-C0-N-Q1",
            device_type=parsed.type,
            sealed_real_test_capability=False,
        )

    def create_session(self, arm: AblationArm, *, seed: int) -> C0CoarseSession:
        return C0CoarseSession(arm, seed=seed, device=self.device)


__all__ = [
    "C0_COARSE_BACKEND_VERSION",
    "C0_COARSE_MODEL_CONFIG",
    "C0_COARSE_OPTIMIZER_CONFIG",
    "C0_COARSE_PAYLOAD_VERSION",
    "C0_COARSE_SEED",
    "C0_COARSE_SWAP_TOLERANCE",
    "C0CoarseBackend",
    "C0CoarseBackendError",
    "C0CoarsePayload",
    "C0CoarseSession",
    "c0_coarse_payload_content_sha256",
]
