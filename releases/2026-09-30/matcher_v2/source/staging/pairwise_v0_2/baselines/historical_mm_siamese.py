"""Exact inference adapter for the historical MM MobileNetV2 Siamese model.

The original MM baseline consumes two whole binary masks independently, resizes
each full canvas to 64 x 64 with PIL bilinear interpolation, concatenates the
two MobileNetV2 embeddings in input order and predicts a join probability.

This module intentionally does not build or select a validation population.  It
scores the exact ``TrainingPairRecord`` sequence supplied by the caller, in the
same order, so a new local matcher and this historical checkpoint can be
evaluated on the same MM validation rows.  It has no real-Dunhuang data API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.models import mobilenet_v2

from staging.pairwise_v0_2.pairwise_data.training_stream import (
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.evaluation import evaluate_pairwise


HISTORICAL_MM_BASELINE_ID = "mm-mobilenetv2-30000-30000-focal/2023"
HISTORICAL_MM_INPUT_SIZE = (64, 64)
HISTORICAL_MM_DATASET_ID = "mm_augmented"
HISTORICAL_MM_SPLIT = "val"
MaskLoader = Callable[[MaskMemberRef], np.ndarray]
PathLike = Union[str, Path]


class HistoricalMMBaselineError(ValueError):
    """Raised when the historical-MM comparison contract is violated."""


class HistoricalMMSiamese(nn.Module):
    """Architecture used by ``mobilenet_30000_30000_focal.pt``.

    The ordered feature concatenation is deliberate: it reproduces the old
    checkpoint rather than silently replacing it with a symmetric model.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=None, dropout=0)
        backbone.features[0][0] = nn.Conv2d(
            1,
            32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.fc_in_features = backbone.last_channel
        # This exact nesting yields the historical ``mobilenet.0.*`` keys.
        self.mobilenet = nn.Sequential(*(list(backbone.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, value: Tensor) -> Tensor:
        output = self.mobilenet(value)
        output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
        return output.reshape(output.shape[0], -1)

    def forward(self, input_a: Tensor, input_b: Tensor) -> Tensor:
        embedding_a = self.forward_once(input_a)
        embedding_b = self.forward_once(input_b)
        ordered = torch.cat((embedding_a, embedding_b), dim=1)
        return self.sigmoid(self.fc(ordered))


def _checkpoint_state_dict(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping) or not payload:
        raise HistoricalMMBaselineError(
            "historical checkpoint must be a non-empty state_dict"
        )
    if not all(isinstance(name, str) and name for name in payload):
        raise HistoricalMMBaselineError("checkpoint parameter names are invalid")
    if not all(isinstance(value, Tensor) for value in payload.values()):
        raise HistoricalMMBaselineError(
            "historical checkpoint may contain tensors only"
        )
    return payload


def load_historical_mm_checkpoint(
    checkpoint_path: PathLike,
    *,
    device: Union[str, torch.device] = "cpu",
) -> HistoricalMMSiamese:
    """Load the historical tensor-only checkpoint strictly and enter eval mode."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed_device = torch.device(device)
    if parsed_device.type == "cuda" and not torch.cuda.is_available():
        raise HistoricalMMBaselineError("CUDA was requested but is unavailable")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - server uses modern PyTorch
        raise HistoricalMMBaselineError(
            "safe checkpoint loading requires PyTorch with weights_only support"
        ) from exc
    model = HistoricalMMSiamese()
    try:
        model.load_state_dict(_checkpoint_state_dict(payload), strict=True)
    except RuntimeError as exc:
        raise HistoricalMMBaselineError(
            "checkpoint does not match the historical MM MobileNetV2 architecture"
        ) from exc
    return model.to(parsed_device).eval()


def preprocess_historical_mm_mask(mask: np.ndarray) -> Tensor:
    """Reproduce ``Grayscale(1) -> Resize(64,64) -> ToTensor`` for a bool mask."""

    value = np.asarray(mask)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.bool_:
        raise HistoricalMMBaselineError(
            "historical MM preprocessing requires a non-empty 2D bool mask"
        )
    source = Image.fromarray(value.astype(np.uint8) * 255, mode="L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = source.resize(
        (HISTORICAL_MM_INPUT_SIZE[1], HISTORICAL_MM_INPUT_SIZE[0]),
        resample=resampling,
    )
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


@dataclass(frozen=True)
class HistoricalMMBaselineResult:
    """Ordered scores and metrics for exactly the supplied validation records."""

    probability: Tensor
    label: Tensor
    pair_ids: Tuple[str, ...]
    cluster_ids: Tuple[str, ...]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.pair_ids)
        if count == 0 or len(self.cluster_ids) != count:
            raise HistoricalMMBaselineError("baseline result identity count is invalid")
        if self.probability.dtype != torch.float64 or tuple(
            self.probability.shape
        ) != (count,):
            raise HistoricalMMBaselineError(
                "baseline probabilities must be float64 [N]"
            )
        if self.label.dtype != torch.bool or tuple(self.label.shape) != (count,):
            raise HistoricalMMBaselineError("baseline labels must be bool [N]")
        if not torch.isfinite(self.probability).all().item() or (
            (self.probability < 0.0) | (self.probability > 1.0)
        ).any().item():
            raise HistoricalMMBaselineError("baseline probabilities are invalid")
        if len(set(self.pair_ids)) != count:
            raise HistoricalMMBaselineError("validation pair IDs must be unique")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    def summary(self) -> Dict[str, Any]:
        """Small JSON-compatible result; raw scores remain available separately."""

        return {
            "baseline_id": HISTORICAL_MM_BASELINE_ID,
            "dataset_id": HISTORICAL_MM_DATASET_ID,
            "split": HISTORICAL_MM_SPLIT,
            "score_semantics": "ordered_A_then_B_historical_checkpoint_probability",
            "preprocessing": "full_canvas_grayscale_PIL_bilinear_64x64_to_tensor",
            "pair_count": self.pair_count,
            "positive_count": int(self.label.sum().item()),
            "negative_count": int((~self.label).sum().item()),
            "metrics": dict(self.metrics),
        }


def _validate_record(record: Any) -> TrainingPairRecord:
    if not isinstance(record, TrainingPairRecord):
        raise TypeError("records must contain TrainingPairRecord values")
    if record.dataset_id != HISTORICAL_MM_DATASET_ID:
        raise HistoricalMMBaselineError(
            "historical MM baseline accepts mm_augmented records only"
        )
    if record.split != HISTORICAL_MM_SPLIT:
        raise HistoricalMMBaselineError(
            "historical MM baseline accepts validation records only"
        )
    return record


def _score_batch(
    model: nn.Module,
    masks_a: Sequence[Tensor],
    masks_b: Sequence[Tensor],
    device: torch.device,
) -> Tensor:
    batch_a = torch.stack(tuple(masks_a), dim=0).to(device, non_blocking=False)
    batch_b = torch.stack(tuple(masks_b), dim=0).to(device, non_blocking=False)
    output = model(batch_a, batch_b)
    if not isinstance(output, Tensor) or tuple(output.shape) != (len(masks_a), 1):
        raise HistoricalMMBaselineError(
            "historical MM model must return probability with shape [B,1]"
        )
    probability = output[:, 0]
    if not torch.isfinite(probability).all().item() or (
        (probability < 0.0) | (probability > 1.0)
    ).any().item():
        raise HistoricalMMBaselineError("model returned invalid probabilities")
    return probability.detach().to(dtype=torch.float64, device="cpu")


def score_historical_mm_mask_pairs(
    model: nn.Module,
    mask_pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    *,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> Tensor:
    """Score an ordered stream of binary-mask pairs with the legacy model.

    This is the dataset-neutral inference seam for external evaluation.  It
    deliberately accepts masks rather than records, so the historical MM
    validation population checks below remain unchanged and callers cannot
    accidentally make RGB, text, canvas origins, or other metadata model
    inputs through this API.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    parsed_device = torch.device(device)
    if parsed_device.type == "cuda" and not torch.cuda.is_available():
        raise HistoricalMMBaselineError("CUDA was requested but is unavailable")
    model = model.to(parsed_device).eval()

    masks_a = []
    masks_b = []
    probabilities = []

    def flush() -> None:
        if not masks_a:
            return
        probabilities.append(_score_batch(model, masks_a, masks_b, parsed_device))
        masks_a.clear()
        masks_b.clear()

    with torch.inference_mode():
        for pair in mask_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise HistoricalMMBaselineError(
                    "historical MM mask stream must contain mask pairs"
                )
            masks_a.append(preprocess_historical_mm_mask(pair[0]))
            masks_b.append(preprocess_historical_mm_mask(pair[1]))
            if len(masks_a) == batch_size:
                flush()
        flush()
    if not probabilities:
        raise HistoricalMMBaselineError("historical MM mask-pair stream is empty")
    return torch.cat(probabilities, dim=0)


def score_historical_mm_validation(
    model: nn.Module,
    records: Iterable[TrainingPairRecord],
    mask_loader: MaskLoader,
    *,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> HistoricalMMBaselineResult:
    """Score an MM-val record stream without resampling or reordering it."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(mask_loader):
        raise TypeError("mask_loader must be callable")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    parsed_device = torch.device(device)
    if parsed_device.type == "cuda" and not torch.cuda.is_available():
        raise HistoricalMMBaselineError("CUDA was requested but is unavailable")
    model = model.to(parsed_device).eval()

    masks_a = []
    masks_b = []
    probabilities = []
    labels = []
    pair_ids = []
    cluster_ids = []

    def flush() -> None:
        if not masks_a:
            return
        probabilities.append(_score_batch(model, masks_a, masks_b, parsed_device))
        masks_a.clear()
        masks_b.clear()

    with torch.inference_mode():
        for raw_record in records:
            record = _validate_record(raw_record)
            pair_ids.append(record.pair_id)
            cluster_ids.append(record.component_id)
            labels.append(record.label)
            masks_a.append(preprocess_historical_mm_mask(mask_loader(record.fragment_a)))
            masks_b.append(preprocess_historical_mm_mask(mask_loader(record.fragment_b)))
            if len(masks_a) == batch_size:
                flush()
        flush()

    if not pair_ids:
        raise HistoricalMMBaselineError("MM validation record stream is empty")
    probability = torch.cat(probabilities, dim=0)
    label = torch.tensor(labels, dtype=torch.bool)
    metrics = evaluate_pairwise(
        probability=probability.tolist(),
        label=label.tolist(),
        valid=[True] * len(pair_ids),
        cluster_id=cluster_ids,
        threshold=0.5,
    )
    return HistoricalMMBaselineResult(
        probability=probability,
        label=label,
        pair_ids=tuple(pair_ids),
        cluster_ids=tuple(cluster_ids),
        metrics=metrics,
    )


def evaluate_historical_mm_checkpoint(
    checkpoint_path: PathLike,
    records: Iterable[TrainingPairRecord],
    mask_loader: MaskLoader,
    *,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 256,
) -> HistoricalMMBaselineResult:
    """One-call checkpoint loading and like-for-like MM validation evaluation."""

    model = load_historical_mm_checkpoint(checkpoint_path, device=device)
    return score_historical_mm_validation(
        model,
        records,
        mask_loader,
        device=device,
        batch_size=batch_size,
    )


__all__ = [
    "HISTORICAL_MM_BASELINE_ID",
    "HISTORICAL_MM_DATASET_ID",
    "HISTORICAL_MM_INPUT_SIZE",
    "HistoricalMMBaselineError",
    "HistoricalMMBaselineResult",
    "HistoricalMMSiamese",
    "evaluate_historical_mm_checkpoint",
    "load_historical_mm_checkpoint",
    "preprocess_historical_mm_mask",
    "score_historical_mm_mask_pairs",
    "score_historical_mm_validation",
]
