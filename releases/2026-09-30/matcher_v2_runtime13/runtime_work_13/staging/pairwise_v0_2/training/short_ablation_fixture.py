"""Provider-free, mask-free dry-run fixture for the short-ablation harness.

This fixture exercises the complete orchestration and checkpoint/receipt path
with a one-parameter CPU model.  It never opens a mask archive and cannot
access the sealed real test.  It is a contract smoke, not an accuracy result.

Minimal CLI::

    python -m staging.pairwise_v0_2.training.short_ablation_fixture \
        --output-root /tmp/pairwise-v02-short-ablation-dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from staging.pairwise_v0_2.geometry import ContourKeypointConfig
from staging.pairwise_v0_2.models.local_matcher import MatcherMode
from staging.pairwise_v0_2.models.pairwise import ArcPoolingConfig
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    ArchiveBinding,
    MaskMemberRef,
    TrainingPairRecord,
)
from staging.pairwise_v0_2.training.data_lock import build_experiment_data_lock
from staging.pairwise_v0_2.training.geometry_batch import GeometryBatchConfig
from staging.pairwise_v0_2.training.short_ablation import (
    AblationArm,
    AblationArmName,
    ApprovedCoarsePreprocessing,
    BackendContract,
    BatchProviderContract,
    DatasetLabelQuota,
    EvidenceMode,
    ExecutionKind,
    PilotBudget,
    PopulationSelectionConfig,
    PredictionBatch,
    PreparedAblationBatch,
    RuntimeDataArtifacts,
    ShortAblationArtifacts,
    ShortAblationConfig,
    TrainBatchResult,
    build_source_code_lock,
    freeze_population_selection,
    record_sequence_fingerprint,
    run_short_ablation,
)


FIXTURE_BINDING = ArchiveBinding(
    "canonical://short_ablation/provider_free_fixture",
    "zip",
    "f" * 64,
)
TRAIN_DATASETS = (
    "dunhuang_voronoi_masks_no_erode_v0_2",
    "eccv_1113data",
    "mm_augmented",
)
VALIDATION_DATASETS = ("eccv_1113data", "mm_augmented")


def recommended_first_pilot_template() -> Dict[str, Any]:
    """Return the frozen C0-N-Q1 plan; this function never executes it."""

    return {
        "status": "preregistered_template_not_executed",
        "run_name": "C0-N-Q1",
        "purpose": "coarse_only_spatial_shortcut_free_baseline",
        "arm": "coarse_only",
        "train_quota_per_dataset": {"positive": 4096, "negative": 4096},
        "validation_population": "full_frozen_source_order",
        "validation_counts": {
            "mm_augmented": {"positive": 53488, "negative": 208502},
            "eccv_1113data": {"positive": 9768, "negative": 1288},
        },
        "train_datasets": ["eccv_1113data", "mm_augmented"],
        "validation_datasets": list(VALIDATION_DATASETS),
        "new5k_status": "not_in_C0_N_Q1; reserved_for_later_training_ablation",
        "epochs": 5,
        "batch_size": 256,
        "population": {
            "train_rows_per_epoch": 16384,
            "validation_rows": 273046,
        },
        "raw_metadata_scan_caps": {
            "train_rows": 1275000,
            "validation_rows": 274000,
        },
        "optimizer_steps": 320,
        "arm_count": 1,
        "gpu_time_estimate_minutes": {
            "lower": 3.0,
            "upper": 10.0,
            "basis": (
                "engineering estimate for GPU compute only; archive decode and "
                "validation input I/O may dominate wall time; not a measurement"
            ),
        },
        "selection_policy": (
            "explicit_dataset_by_label_quotas_component_bounded_hash_deterministic"
        ),
        "max_records_per_component_per_label": 32,
        "preprocessing": {
            "mode": "tight_crop_letterbox",
            "output_size": [128, 128],
            "coarse_content_fraction": 0.875,
            "maximum_foreground_extent": [112, 112],
            "largest_component_connectivity": 4,
        },
        "checkpoint_selection": {
            "validation_frequency": "every_epoch_full_273046_rows",
            "checkpoint_count": 5,
            "primary_metrics": [
                "equal_domain_macro_cluster_auroc",
                "equal_domain_macro_cluster_auprc",
            ],
            "implemented_by": "concrete_C0_runner_not_generic_contract_runner",
        },
        "geometry_cache_local_sinkhorn_calls": 0,
        "winner_selected": False,
        "real_test_accessed": False,
    }


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    *, dataset: str, split: str, label: bool, component_index: int
) -> TrainingPairRecord:
    stem = "{}/{}/{}/{:03d}".format(
        dataset, split, "positive" if label else "negative", component_index
    )
    component = "fixture-component/" + _sha_text(stem)
    group = "fixture-group/" + _sha_text(stem + "/group")
    common = {
        "binding": FIXTURE_BINDING,
        "dataset_id": dataset,
        "canonical_group_id": group,
        "component_id": component,
        "split": split,
        "threshold_rule": "grayscale_uint8_gt_127",
    }
    first_id = stem + "/a"
    second_id = stem + "/b"
    first = MaskMemberRef(
        archive_member="fixture/{}/a.png".format(_sha_text(stem)),
        fragment_id=first_id,
        content_sha256=_sha_text(stem + "/mask/a"),
        **common,
    )
    second = MaskMemberRef(
        archive_member="fixture/{}/b.png".format(_sha_text(stem)),
        fragment_id=second_id,
        content_sha256=_sha_text(stem + "/mask/b"),
        **common,
    )
    return TrainingPairRecord(
        fragment_a=first,
        fragment_b=second,
        label=label,
        direction_b_wrt_a="right" if label else None,
        dataset_id=dataset,
        canonical_group_id=group,
        component_id=component,
        split=split,
        canonical_pair_key=tuple(sorted((first_id, second_id))),
        label_origin="provider_free_fixture",
        provenance={"real_dunhuang_sealed_test": False},
    )


def _records(datasets: Sequence[str], split: str) -> Tuple[TrainingPairRecord, ...]:
    return tuple(
        _record(
            dataset=dataset,
            split=split,
            label=label,
            component_index=index,
        )
        for dataset in datasets
        for label in (False, True)
        for index in range(4)
    )


@dataclass(frozen=True)
class _FixturePayload:
    feature: Tensor
    label: Tensor


class ProviderFreeBatchProvider:
    """Create deterministic scalar features without loading any image pixels."""

    def __init__(self, preprocessing: ApprovedCoarsePreprocessing) -> None:
        self.preprocessing = preprocessing
        self.contract = BatchProviderContract(
            coarse_preprocess_mode=preprocessing.mode,
            coarse_preprocessing_sha256=preprocessing.preprocessing_sha256,
            geometry_config_sha256=preprocessing.geometry_config_sha256,
            cache_interface="prepared_batch_callback_fragment_cache_ready",
            provider_version="provider-free-fixture/0.2",
        )

    def prepare(
        self,
        records: Sequence[TrainingPairRecord],
        *,
        arm: AblationArm,
        phase: str,
    ) -> PreparedAblationBatch:
        if phase not in {"train", "validation"}:
            raise ValueError("unsupported fixture phase")
        values = []
        for record in records:
            digest = hashlib.sha256(record.pair_id.encode("utf-8")).digest()
            values.append((int.from_bytes(digest[:4], "big") / (2**32 - 1)) * 2 - 1)
        payload = _FixturePayload(
            feature=torch.tensor(values, dtype=torch.float32)[:, None],
            label=torch.tensor([record.label for record in records], dtype=torch.bool),
        )
        prepared_input_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "feature_float32": [
                        float(value) for value in payload.feature[:, 0].tolist()
                    ],
                    "label_bool": [bool(value) for value in payload.label.tolist()],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return PreparedAblationBatch(
            payload=payload,
            sample_count=len(records),
            record_sequence_sha256=record_sequence_fingerprint(records),
            prepared_input_sha256=prepared_input_sha256,
            local_candidate_sha256=(
                _sha_text("provider-free-empty-local-candidates-v0.2")
                if arm.uses_local_geometry
                else None
            ),
            coarse_preprocessing_sha256=(self.preprocessing.preprocessing_sha256),
            geometry_config_sha256=(
                self.preprocessing.geometry_config_sha256
                if arm.uses_local_geometry
                else None
            ),
            processing_counts={
                "mask_load_count": 0,
                "coarse_preprocess_count": 0,
                "geometry_build_count": 0,
                "geometry_cache_read_count": 0,
                "geometry_cache_write_count": 0,
                "local_candidate_count": 0,
            },
        )


class _FixtureSession:
    def __init__(self, arm: AblationArm, seed: int) -> None:
        torch.manual_seed(seed)
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(arm.optimizer_config["learning_rate"]),
        )
        self.model_config = dict(arm.model_config)
        self.optimizer_config = dict(arm.optimizer_config)
        self.arm = arm

    def train_batch(self, batch: PreparedAblationBatch) -> TrainBatchResult:
        payload = batch.payload
        if not isinstance(payload, _FixturePayload):
            raise TypeError("fixture payload changed")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logit = self.model(payload.feature).squeeze(1)
        loss = functional.binary_cross_entropy_with_logits(
            logit, payload.label.to(logit.dtype)
        )
        loss.backward()
        self.optimizer.step()
        return TrainBatchResult(
            loss=float(loss.detach().cpu()),
            valid_count=len(payload.label),
            diagnostics={
                "coarse_forward_count": 1,
                "local_forward_count": 0,
                "sinkhorn_call_count": 0,
                "fixture_contract_smoke": True,
            },
        )

    def predict_batch(
        self, batch: PreparedAblationBatch, *, evidence: EvidenceMode
    ) -> PredictionBatch:
        del evidence
        payload = batch.payload
        if not isinstance(payload, _FixturePayload):
            raise TypeError("fixture payload changed")
        self.model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(payload.feature).squeeze(1))
        return PredictionBatch(
            probability=probability,
            valid=torch.ones_like(probability, dtype=torch.bool),
            diagnostics={
                "coarse_forward_count": 1,
                "local_forward_count": 0,
                "sinkhorn_call_count": 0,
                "fixture_contract_smoke": True,
            },
        )


class ProviderFreeBackend:
    contract = BackendContract(
        execution_kind=ExecutionKind.PROVIDER_FREE_DRY_RUN,
        backend_version="provider-free-linear-backend/0.2",
        model_family="one_parameter_linear_contract_smoke_not_pairwise_model",
        device_type="cpu",
        sealed_real_test_capability=False,
    )

    def create_session(self, arm: AblationArm, *, seed: int) -> _FixtureSession:
        return _FixtureSession(arm, seed)


def _arm(
    name: AblationArmName,
    evidence: EvidenceMode,
    matcher: Any,
    pooling: Any,
) -> AblationArm:
    model_config = {
        "fixture_model": "one_parameter_linear_contract_smoke",
        "matcher_mode": matcher,
        "evidence": evidence.value,
        "arc_pooling": (
            None
            if pooling is None
            else {
                "mode": pooling.mode.value,
                "temperature": pooling.temperature,
                "top_k": pooling.top_k,
            }
        ),
    }
    if name in {
        AblationArmName.KEYPOINT_DUAL_SOFTMAX,
        AblationArmName.KEYPOINT_DUSTBIN_SINKHORN,
    }:
        model_config.update(
            {
                "candidate_representation": "contour_keypoint",
                "contour_keypoint_config": asdict(ContourKeypointConfig()),
            }
        )
    return AblationArm(
        name=name,
        evidence=evidence,
        matcher_mode=matcher,
        model_config=model_config,
        optimizer_config={"name": "sgd", "learning_rate": 0.2},
        aggregation_config={
            "direction_pooling": "log_mean_exp",
            "direction_temperature": 0.25,
        },
        arc_pooling=pooling,
    )


def _source_paths() -> Mapping[str, Path]:
    directory = Path(__file__).resolve().parent
    package = directory.parent
    return {
        "backend": Path(__file__).resolve(),
        "batch_provider": Path(__file__).resolve(),
        "candidate_builder": package / "geometry" / "candidate_builder.py",
        "checkpoint": directory / "checkpoint.py",
        "coarse_model": package / "models" / "coarse.py",
        "data_lock": directory / "data_lock.py",
        "data_stream": package / "pairwise_data" / "training_stream.py",
        "evaluation": directory / "evaluation.py",
        "geometry": directory / "geometry_batch.py",
        "geometry_cache": directory / "geometry_cache.py",
        "geometry_schema": package / "geometry" / "schema.py",
        "harness": directory / "short_ablation.py",
        "lazy_mask_loader": package / "pairwise_data" / "lazy_mask_loader.py",
        "local_matcher": package / "models" / "local_matcher.py",
        "model": package / "models" / "pairwise.py",
        "optimal_transport": package / "models" / "optimal_transport.py",
        "sampler": directory / "short_ablation.py",
        "fragment_geometry_cache": directory / "fragment_geometry_cache.py",
        "training_engine": directory / "engine.py",
    }


def _data_artifacts(root: Path) -> RuntimeDataArtifacts:
    manifest = root / "fixture_synthetic_manifest.jsonl"
    manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    archive = {
        "logical_id": FIXTURE_BINDING.logical_id,
        "format": FIXTURE_BINDING.archive_format,
        "sha256": FIXTURE_BINDING.sha256,
    }
    split_receipt = root / "fixture_split_receipt.json"
    split_receipt.write_text(
        json.dumps(
            {
                "upstream_artifacts": {
                    "synthetic_manifest": {"sha256": manifest_sha},
                    "archives": [archive],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stream_audit = root / "fixture_stream_audit.json"
    stream_audit.write_text(
        json.dumps(
            {"inputs": {"synthetic_manifest": {"sha256": manifest_sha}}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lock = build_experiment_data_lock(
        split_receipt_path=split_receipt,
        stream_audit_path=stream_audit,
        synthetic_manifest_path=manifest,
        archive_bindings=(FIXTURE_BINDING,),
    )
    return RuntimeDataArtifacts(
        required_lock=lock,
        split_receipt_path=split_receipt,
        stream_audit_path=stream_audit,
        synthetic_manifest_path=manifest,
        archive_bindings=(FIXTURE_BINDING,),
    )


def build_provider_free_run_arguments(output_root: Path) -> Dict[str, Any]:
    """Build independently inspectable arguments for the provider-free run."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    train = _records(TRAIN_DATASETS, "train")
    validation = _records(VALIDATION_DATASETS, "val")

    def train_factory():
        return iter(train)

    def validation_factory():
        return iter(validation)

    train_selection = PopulationSelectionConfig(
        split="train",
        quotas=tuple(
            sorted(DatasetLabelQuota(dataset, 2, 2) for dataset in TRAIN_DATASETS)
        ),
        seed="provider-free-train-selection-v0.2",
        max_records_per_component_per_label=1,
    )
    validation_selection = PopulationSelectionConfig(
        split="val",
        quotas=tuple(
            sorted(DatasetLabelQuota(dataset, 2, 2) for dataset in VALIDATION_DATASETS)
        ),
        seed="provider-free-validation-selection-v0.2",
        max_records_per_component_per_label=1,
    )
    train_contract, train_selection_receipt = freeze_population_selection(
        train_factory,
        train_selection,
        maximum_rows=100,
        allowed_archive_bindings=(FIXTURE_BINDING,),
    )
    validation_contract, validation_selection_receipt = freeze_population_selection(
        validation_factory,
        validation_selection,
        maximum_rows=100,
        allowed_archive_bindings=(FIXTURE_BINDING,),
    )
    preprocessing = ApprovedCoarsePreprocessing.from_geometry_config(
        GeometryBatchConfig()
    )
    pooling = ArcPoolingConfig()
    arms = (
        _arm(AblationArmName.COARSE_ONLY, EvidenceMode.COARSE, None, None),
        _arm(
            AblationArmName.LOCAL_DUAL_SOFTMAX,
            EvidenceMode.LOCAL,
            MatcherMode.DUAL_SOFTMAX.value,
            pooling,
        ),
        _arm(
            AblationArmName.LOCAL_DUSTBIN_SINKHORN,
            EvidenceMode.LOCAL,
            MatcherMode.DUSTBIN_SINKHORN.value,
            pooling,
        ),
        _arm(
            AblationArmName.FUSED,
            EvidenceMode.FUSED,
            MatcherMode.DUSTBIN_SINKHORN.value,
            pooling,
        ),
    )
    config = ShortAblationConfig(
        preprocessing=preprocessing,
        train_selection=train_selection,
        validation_selection=validation_selection,
        arms=arms,
        budget=PilotBudget(
            epochs=1,
            batch_size=4,
            max_optimizer_steps_per_arm=4,
            max_train_stream_rows=100,
            max_validation_stream_rows=100,
            bootstrap_repetitions=100,
            gpu_minutes_lower_estimate=0.0,
            gpu_minutes_upper_estimate=0.0,
            estimate_basis="provider_free_CPU_contract_smoke_no_GPU",
        ),
        initialization_seed="provider-free-initialization-v0.2",
        expected_train_selection_sha256=(train_selection_receipt.selection_sha256),
        expected_validation_selection_sha256=(
            validation_selection_receipt.selection_sha256
        ),
    )
    source_paths = _source_paths()
    source_lock = build_source_code_lock(source_paths)
    return {
        "config": config,
        "data_artifacts": _data_artifacts(root),
        "required_train_stream": train_contract,
        "required_validation_stream": validation_contract,
        "required_source_lock": source_lock,
        "source_paths": source_paths,
        "train_records": train_factory,
        "validation_records": validation_factory,
        "batch_provider": ProviderFreeBatchProvider(preprocessing),
        "backend": ProviderFreeBackend(),
        "output_root": root / "runs",
    }


def run_provider_free_dry_run(output_root: Path) -> ShortAblationArtifacts:
    """Exercise all four arms on synthetic metadata and a CPU scalar model."""

    return run_short_ablation(**build_provider_free_run_arguments(output_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="directory for the provider-free CPU dry-run artifacts",
    )
    parser.add_argument(
        "--print-pilot-template",
        action="store_true",
        help="print the preregistered first-GPU-pilot budget without running it",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.print_pilot_template:
        print(
            json.dumps(
                recommended_first_pilot_template(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return
    if args.output_root is None:
        raise SystemExit(
            "--output-root is required unless --print-pilot-template is used"
        )
    artifacts = run_provider_free_dry_run(args.output_root)
    print(
        json.dumps(
            {
                "status": artifacts.receipt["status"],
                "run_fingerprint_sha256": artifacts.receipt["run_fingerprint_sha256"],
                "receipt_content_sha256": artifacts.receipt["content_sha256"],
                "scope": artifacts.receipt["scope"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ProviderFreeBackend",
    "ProviderFreeBatchProvider",
    "build_provider_free_run_arguments",
    "recommended_first_pilot_template",
    "run_provider_free_dry_run",
]
