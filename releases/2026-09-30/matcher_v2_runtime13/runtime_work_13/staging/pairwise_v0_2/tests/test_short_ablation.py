import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from staging.pairwise_v0_2.pairwise_data.training_stream import ArchiveBinding
from staging.pairwise_v0_2.training.geometry_batch import (
    COARSE_NUMERIC_CONTRACT,
    GeometryBatchConfig,
)
from staging.pairwise_v0_2.training.short_ablation import (
    ApprovedCoarsePreprocessing,
    BatchProviderContract,
    DatasetLabelQuota,
    PopulationSelectionConfig,
    PopulationSelectionMode,
    ShortAblationError,
    build_source_code_lock,
    freeze_population_selection,
    freeze_source_disjoint_populations,
    run_short_ablation,
)
from staging.pairwise_v0_2.training.short_ablation_fixture import (
    FIXTURE_BINDING,
    TRAIN_DATASETS,
    ProviderFreeBackend,
    ProviderFreeBatchProvider,
    _records,
    build_provider_free_run_arguments,
    recommended_first_pilot_template,
    run_provider_free_dry_run,
)


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _all_keys(value):
    keys = set()
    if isinstance(value, dict):
        keys.update(value)
        for item in value.values():
            keys.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_keys(item))
    return keys


def test_legacy_full_canvas_preprocessing_is_rejected_at_contract_creation():
    legacy = replace(
        GeometryBatchConfig(),
        coarse_preprocess_mode="full_canvas_stretch_legacy",
    )

    with pytest.raises(ShortAblationError, match="full-canvas"):
        ApprovedCoarsePreprocessing.from_geometry_config(legacy)
    with pytest.raises(ShortAblationError, match="full-canvas"):
        BatchProviderContract(
            coarse_preprocess_mode="full_canvas_stretch_legacy",
            coarse_preprocessing_sha256="1" * 64,
            geometry_config_sha256="2" * 64,
            cache_interface="fixture",
            provider_version="fixture/0.2",
        )


def test_approved_coarse_preprocessing_binds_exact_numeric_contract():
    preprocessing = ApprovedCoarsePreprocessing.from_geometry_config(
        GeometryBatchConfig()
    )

    assert preprocessing.numeric_contract == COARSE_NUMERIC_CONTRACT
    assert preprocessing.to_dict()["numeric_contract"] == COARSE_NUMERIC_CONTRACT
    with pytest.raises(ShortAblationError, match="numeric contract"):
        replace(preprocessing, numeric_contract="unbounded_float32")


def test_dataset_label_component_selection_is_exact_and_order_deterministic():
    records = _records(TRAIN_DATASETS, "train")
    selection = PopulationSelectionConfig(
        split="train",
        quotas=tuple(
            sorted(DatasetLabelQuota(dataset, 2, 2) for dataset in TRAIN_DATASETS)
        ),
        seed="selection-test",
        max_records_per_component_per_label=1,
    )

    first_contract, first = freeze_population_selection(
        lambda: iter(records),
        selection,
        maximum_rows=100,
        allowed_archive_bindings=(FIXTURE_BINDING,),
    )
    reversed_contract, second = freeze_population_selection(
        lambda: iter(tuple(reversed(records))),
        selection,
        maximum_rows=100,
        allowed_archive_bindings=(FIXTURE_BINDING,),
    )

    assert first_contract.sha256 != reversed_contract.sha256
    assert first.selection_sha256 == second.selection_sha256
    assert first.selected_count == 12
    assert first.positive_count == first.negative_count == 6
    assert first.max_selected_records_per_component_label == 1
    assert set(first.by_dataset) == set(TRAIN_DATASETS)
    for counts in first.by_dataset.values():
        assert counts == {
            "positive": 2,
            "negative": 2,
            "positive_component_count": 2,
            "negative_component_count": 2,
        }


def test_full_validation_selection_preserves_every_row_and_source_order():
    records = _records(("eccv_1113data", "mm_augmented"), "val")
    selection = PopulationSelectionConfig(
        split="val",
        quotas=(
            DatasetLabelQuota("eccv_1113data", 4, 4),
            DatasetLabelQuota("mm_augmented", 4, 4),
        ),
        seed="full-validation-contract",
        mode=PopulationSelectionMode.FULL_FROZEN_STREAM,
    )

    contract, receipt = freeze_population_selection(
        lambda: iter(records),
        selection,
        maximum_rows=20,
        allowed_archive_bindings=(FIXTURE_BINDING,),
    )

    assert receipt.selected_count == len(records) == 16
    assert receipt.selection_sha256 == contract.sha256
    assert receipt.selection_mode == "full_frozen_stream"
    assert receipt.order == "frozen_source_order"


def test_policy_or_stream_mismatch_stops_before_model_or_record_factory(tmp_path):
    arguments = build_provider_free_run_arguments(tmp_path / "fixture")

    class CountingBackend(ProviderFreeBackend):
        def __init__(self):
            self.calls = 0

        def create_session(self, arm, *, seed):
            self.calls += 1
            return super().create_session(arm, seed=seed)

    backend = CountingBackend()
    train_calls = []
    original_train = arguments["train_records"]

    def counted_train():
        train_calls.append(True)
        return original_train()

    arguments["backend"] = backend
    arguments["train_records"] = counted_train
    provider = arguments["batch_provider"]
    provider.contract = replace(
        provider.contract,
        coarse_preprocessing_sha256="0" * 64,
    )
    with pytest.raises(ShortAblationError, match="preprocessing hash"):
        run_short_ablation(**arguments)
    assert backend.calls == 0
    assert train_calls == []

    arguments = build_provider_free_run_arguments(tmp_path / "fixture-two")
    backend = CountingBackend()
    arguments["backend"] = backend
    arguments["source_paths"] = dict(arguments["source_paths"])
    arguments["source_paths"]["backend"] = Path(__file__).resolve()
    arguments["required_source_lock"] = build_source_code_lock(
        arguments["source_paths"]
    )
    arguments["required_train_stream"] = replace(
        arguments["required_train_stream"], sha256="0" * 64
    )
    with pytest.raises(ShortAblationError, match="frozen contract"):
        run_short_ablation(**arguments)
    assert backend.calls == 0


def test_provider_free_end_to_end_receipts_are_hash_bound_and_portable(tmp_path):
    artifacts = run_provider_free_dry_run(tmp_path / "dry-run")
    receipt = artifacts.receipt
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)

    assert receipt["status"] == "complete_provider_free_contract_dry_run"
    assert receipt["backend_contract"]["execution_kind"] == (
        "provider_free_contract_dry_run"
    )
    assert receipt["backend_contract"]["model_family"] == (
        "one_parameter_linear_contract_smoke_not_pairwise_model"
    )
    assert receipt["real_dunhuang_sealed_test"] == {
        "record_count": 0,
        "accessed": False,
        "uploaded": False,
        "used_for_threshold": False,
    }
    assert receipt["comparison_policy"]["winner_selected"] is False
    assert receipt["streams"]["train_validation_component_overlap_count"] == 0
    assert receipt["preprocessing"]["mode"] == "tight_crop_letterbox"
    assert [item["arm"]["name"] for item in receipt["arm_results"]] == [
        "coarse_only",
        "local_dual_softmax",
        "local_dustbin_sinkhorn",
        "fused",
    ]
    assert receipt["content_sha256"] == _canonical_hash(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    assert str(tmp_path) not in serialized
    assert "/Users/" not in serialized
    assert "file://" not in serialized
    forbidden_keys = {
        "path",
        "pair_id",
        "component_id",
        "group_id",
        "fragment_id",
        "sample_id",
        "hostname",
        "password",
    }
    assert not (_all_keys(receipt) & forbidden_keys)

    coarse = receipt["arm_results"][0]
    for phase in ("training", "validation"):
        execution = coarse[phase + "_diagnostics"]["numeric"]
        assert execution["local_forward_count"]["sum"] == 0.0
        assert execution["sinkhorn_call_count"]["sum"] == 0.0
        assert execution["coarse_forward_count"]["minimum"] == 1.0
        processing = coarse[phase + "_processing_counts"]["numeric"]
        for key in (
            "geometry_build_count",
            "geometry_cache_read_count",
            "geometry_cache_write_count",
            "local_candidate_count",
        ):
            assert processing[key]["sum"] == 0.0

    for arm_result in receipt["arm_results"]:
        name = arm_result["arm"]["name"]
        external_path = artifacts.checkpoint_receipt_paths[name]
        external = json.loads(external_path.read_text(encoding="utf-8"))
        checkpoint_bytes = artifacts.checkpoint_paths[name].read_bytes()
        assert hashlib.sha256(checkpoint_bytes).hexdigest() == external[
            "checkpoint_file_sha256"
        ]
        assert external["content_sha256"] == _canonical_hash(
            {
                key: value
                for key, value in external.items()
                if key != "content_sha256"
            }
        )
        assert arm_result["checkpoint"]["file_sha256"] == external[
            "checkpoint_file_sha256"
        ]
        assert arm_result["evaluation"]["dataset_macro"]["dataset_count"] == 2
        assert set(arm_result["evaluation"]["by_dataset"]) == {
            "mm_augmented",
            "eccv_1113data",
        }
        assert "row" in arm_result["evaluation"]["overall"]
        assert "cluster_balanced" in arm_result["evaluation"]["overall"]
        assert (
            arm_result["selection_status"]
            == "not_selected_no_winner_in_short_pilot"
        )


def test_recommended_pilot_template_is_bounded_c0_n_q1_only():
    template = recommended_first_pilot_template()

    assert template["status"] == "preregistered_template_not_executed"
    assert template["run_name"] == "C0-N-Q1"
    assert template["arm"] == "coarse_only"
    assert template["train_datasets"] == ["eccv_1113data", "mm_augmented"]
    assert template["validation_datasets"] == ["eccv_1113data", "mm_augmented"]
    assert template["population"]["train_rows_per_epoch"] == 16384
    assert template["population"]["validation_rows"] == 273046
    assert template["optimizer_steps"] == 320
    assert template["arm_count"] == 1
    assert template["preprocessing"]["coarse_content_fraction"] == 0.875
    assert template["preprocessing"]["maximum_foreground_extent"] == [112, 112]
    assert template["preprocessing"]["largest_component_connectivity"] == 4
    assert template["max_records_per_component_per_label"] == 32
    assert template["checkpoint_selection"]["checkpoint_count"] == 5
    assert template["geometry_cache_local_sinkhorn_calls"] == 0
    assert template["winner_selected"] is False
    assert template["real_test_accessed"] is False


def test_arm_configuration_is_recursively_immutable(tmp_path):
    arm = build_provider_free_run_arguments(tmp_path / "fixture")["config"].arms[1]
    before = arm.model_config_sha256

    with pytest.raises(TypeError):
        arm.model_config["runtime_mutation"] = "forbidden"
    with pytest.raises(TypeError):
        arm.model_config["arc_pooling"]["mode"] = "max"

    assert arm.model_config_sha256 == before


def test_record_archive_binding_must_belong_to_data_lock(tmp_path):
    arguments = build_provider_free_run_arguments(tmp_path / "fixture")
    records = tuple(arguments["train_records"]())
    fake = ArchiveBinding("canonical://fixture/unlocked", "zip", "e" * 64)
    changed_fragment = replace(records[0].fragment_a, binding=fake)
    changed_record = replace(records[0], fragment_a=changed_fragment)
    changed = (changed_record,) + records[1:]
    arguments["train_records"] = lambda: iter(changed)

    with pytest.raises(ShortAblationError, match="absent from.*data lock"):
        run_short_ablation(**arguments)


def test_source_disjoint_freeze_rejects_reused_fragment_with_new_component():
    train = _records(("eccv_1113data",), "train")
    validation = []
    for index, record in enumerate(train):
        component = "validation-component-{:03d}".format(index)
        validation.append(
            replace(
                record,
                fragment_a=replace(
                    record.fragment_a, split="val", component_id=component
                ),
                fragment_b=replace(
                    record.fragment_b, split="val", component_id=component
                ),
                split="val",
                component_id=component,
            )
        )
    train_selection = PopulationSelectionConfig(
        split="train",
        quotas=(DatasetLabelQuota("eccv_1113data", 2, 2),),
        seed="train-source-disjoint-test",
        max_records_per_component_per_label=1,
    )
    validation_selection = PopulationSelectionConfig(
        split="val",
        quotas=(DatasetLabelQuota("eccv_1113data", 2, 2),),
        seed="validation-source-disjoint-test",
        max_records_per_component_per_label=1,
    )

    with pytest.raises(ShortAblationError, match="physical source identity"):
        freeze_source_disjoint_populations(
            train_records=lambda: iter(train),
            validation_records=lambda: iter(validation),
            train_selection=train_selection,
            validation_selection=validation_selection,
            max_train_stream_rows=20,
            max_validation_stream_rows=20,
            allowed_archive_bindings=(FIXTURE_BINDING,),
        )


def test_prepared_input_digest_must_match_across_arms(tmp_path):
    arguments = build_provider_free_run_arguments(tmp_path / "fixture")

    class DivergentProvider(ProviderFreeBatchProvider):
        def prepare(self, records, *, arm, phase):
            prepared = super().prepare(records, arm=arm, phase=phase)
            if arm.name.value == "local_dual_softmax":
                return replace(prepared, prepared_input_sha256="d" * 64)
            return prepared

    arguments["batch_provider"] = DivergentProvider(
        arguments["config"].preprocessing
    )
    arguments["source_paths"] = dict(arguments["source_paths"])
    arguments["source_paths"]["batch_provider"] = Path(__file__).resolve()
    arguments["required_source_lock"] = build_source_code_lock(
        arguments["source_paths"]
    )

    with pytest.raises(ShortAblationError, match="tensors differ across arms"):
        run_short_ablation(**arguments)


def test_zero_training_valid_coverage_fails_closed(tmp_path):
    arguments = build_provider_free_run_arguments(tmp_path / "fixture")

    class ZeroValidSession:
        def __init__(self, delegate):
            self._delegate = delegate
            self.model = delegate.model
            self.optimizer = delegate.optimizer
            self.model_config = delegate.model_config
            self.optimizer_config = delegate.optimizer_config

        def train_batch(self, batch):
            return replace(self._delegate.train_batch(batch), valid_count=0)

        def predict_batch(self, batch, *, evidence):
            return self._delegate.predict_batch(batch, evidence=evidence)

    class ZeroValidBackend(ProviderFreeBackend):
        def create_session(self, arm, *, seed):
            return ZeroValidSession(super().create_session(arm, seed=seed))

    arguments["backend"] = ZeroValidBackend()
    arguments["source_paths"] = dict(arguments["source_paths"])
    arguments["source_paths"]["backend"] = Path(__file__).resolve()
    arguments["required_source_lock"] = build_source_code_lock(
        arguments["source_paths"]
    )

    with pytest.raises(ShortAblationError, match="training valid coverage"):
        run_short_ablation(**arguments)


def test_checkpoints_and_receipts_repeat_byte_for_byte(tmp_path):
    first = run_provider_free_dry_run(tmp_path / "first")
    second = run_provider_free_dry_run(tmp_path / "second")

    assert first.receipt == second.receipt
    for name in first.checkpoint_paths:
        assert first.checkpoint_paths[name].read_bytes() == second.checkpoint_paths[
            name
        ].read_bytes()
        assert first.checkpoint_receipt_paths[name].read_bytes() == (
            second.checkpoint_receipt_paths[name].read_bytes()
        )
