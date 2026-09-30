"""CPU tests for cached-VAL freeze and target-last benchmark evaluation."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.rachel_n512_formal_30k import evaluate_recall_benchmarks as run
from staging.pairwise_v0_2.pairwise_data import rachel_training_dataset as runtime


def source_rows(split, count):
    return [dict(pair_id="%s-%04d" % (split, i), split=split, label=i % 2 == 0,
        fragment_a=dict(fragment_token="a-%04d" % i, split_unit_id=split + "-a",
            model_mask_path="model/masks_800/a.png", contour_path="model/contours_n512/a.npz"),
        fragment_b=dict(fragment_token="b-%04d" % i, split_unit_id=split + "-b",
            model_mask_path="model/masks_800/b.png", contour_path="model/contours_n512/b.npz"),
        correspondence_path="targets/pairs/p.npz" if i % 2 == 0 else None) for i in range(count)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FakeBenchmark:
    method = "pairingnet"
    validation_score_field = "pair_probability"

    def __init__(self, root):
        self.validation_scores_path = root / "winner_validation_predictions.jsonl"
        self.identity = dict(method="pairingnet", seed=260911,
            validation_scores_sha256=run.adapter._sha256_file(self.validation_scores_path),
            validation_manifest_sha256=run.adapter._sha256_file(root / "pairs" / "val.jsonl"))
        self.calls = 0

    def predict_batch(self, batch):
        self.calls += 1
        n = len(batch.pair_ids)
        valid = np.arange(n) % 4 != 0
        translation = np.zeros((n, 2), np.float32)
        translation[~valid] = np.nan
        return run.adapter.CommonBatchPrediction(schema_version=run.adapter.SCHEMA_VERSION,
            method_key=run.adapter.PAIRINGNET_METHOD_KEY, method_id=run.pairing.METHOD_ID,
            pair_ids=batch.pair_ids, pair_probability=np.linspace(.05, .95, n, dtype=np.float32),
            decision_valid=np.ones(n, bool), translation_hat_rc=translation, translation_valid=valid,
            correspondence_indices=None, correspondence_scores=None, correspondence_semantics=None, auxiliary_scores={})


@pytest.fixture
def cached_val(tmp_path):
    rows = source_rows("val", run.VAL_COUNT)
    write_jsonl(tmp_path / "pairs" / "val.jsonl", rows)
    write_jsonl(tmp_path / "winner_validation_predictions.jsonl",
        [dict(pair_id=row["pair_id"], label=row["label"], pair_probability=.8 if row["label"] else .2) for row in rows])
    return tmp_path, FakeBenchmark(tmp_path)


def test_freeze_uses_all_saved_val_rows_without_any_forward_and_keeps_every_threshold(cached_val):
    root, benchmark = cached_val
    freeze = run.freeze_validation(benchmark, root)
    assert benchmark.calls == 0
    assert freeze["sample_count"] == 3000 and freeze["positive_count"] == 1500
    assert freeze["used_saved_winner_scores"] and not freeze["validation_forward_repeated"]
    assert set(freeze["thresholds"]) == {"max_f1", "recall_90", "recall_95", "recall_98", "recall_99", "recall_first"}
    assert freeze["thresholds"]["recall_first"] == min(freeze["thresholds"]["max_f1"], freeze["thresholds"]["recall_99"])
    assert freeze == run.freeze_validation(benchmark, root)


@pytest.mark.parametrize("corruption", ["hash", "duplicate", "missing", "label", "nonfinite"])
def test_saved_val_hash_population_order_and_labels_are_checked(cached_val, corruption):
    root, benchmark = cached_val
    rows = run.read_jsonl(benchmark.validation_scores_path)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[1] = rows[0]
    elif corruption == "label":
        rows[0]["label"] = False
    elif corruption == "nonfinite":
        rows[0]["pair_probability"] = float("nan")
    else:
        rows[0]["pair_probability"] = .79
    write_jsonl(benchmark.validation_scores_path, rows)
    if corruption != "hash":
        benchmark.identity["validation_scores_sha256"] = run.adapter._sha256_file(benchmark.validation_scores_path)
    with pytest.raises((ValueError, RuntimeError)):
        run.freeze_validation(benchmark, root)
    assert benchmark.calls == 0


def test_test_inputs_never_resolve_or_open_positive_target(tmp_path, monkeypatch):
    rows = source_rows("test", 2)
    mask = np.zeros((800, 800), np.uint8)
    mask[10:30, 10:30] = 255
    contour = np.asarray([[10, 10], [10, 30], [30, 30], [30, 10]], np.float32)
    for side in "ab":
        mask_path = tmp_path / ("model/masks_800/" + side + ".png")
        contour_path = tmp_path / ("model/contours_n512/" + side + ".npz")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        contour_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        np.savez(contour_path, points_rc=contour, valid=np.ones(4, bool))
    # The positive target deliberately does not exist: input-only loading must succeed.
    dataset = run.TargetBlindTestDataset(tmp_path, rows)
    sample = dataset[0]
    assert sample.label == 0 and not sample.translation_valid
    assert np.all(sample.target_a == -1) and np.all(sample.translation_a_to_b_rc == 0)
    batch = runtime.collate_rachel_pairs([sample])
    run.adapter._validate_model_input_batch(batch)


def fake_batch(rows, real=False):
    return SimpleNamespace(pair_ids=tuple(row["pair_id"] for row in rows),
        fragment_a_tokens=tuple(row["fragment_a_id"] if real else row["fragment_a"]["fragment_token"] for row in rows),
        fragment_b_tokens=tuple(row["fragment_b_id"] if real else row["fragment_b"]["fragment_token"] for row in rows))


@pytest.mark.parametrize("split", ["test", "real"])
def test_full_population_predictions_are_frozen_before_gt_is_opened(cached_val, monkeypatch, split):
    root, benchmark = cached_val
    output = root / ("result-" + split)
    monkeypatch.setattr(run, "load_experimental_benchmark", lambda *args: benchmark)
    if split == "test":
        rows = source_rows("test", run.TEST_COUNT)
        write_jsonl(root / "pairs" / "test.jsonl", rows)
        gt_path = root / "targets" / "pairs" / "p.npz"
        gt_path.parent.mkdir(parents=True)
        np.savez(gt_path, translation_a_to_b_rc=np.zeros(2))
        monkeypatch.setattr(run, "DataLoader", lambda *args, **kwargs: [fake_batch(rows)])
    else:
        rows = [dict(pair_id="real-%04d" % i, label=i < 508, strict=i < 547,
            fragment_a_id="a-%04d" % i, fragment_b_id="b-%04d" % i,
            case_cluster="case-%04d" % i) for i in range(run.REAL_COUNT)]
        metadata = dict(pairs=rows, manifest_sha256="a" * 64)
        (root / "manifest.json").write_text(json.dumps(metadata))
        gt_path = root / "real-gt.json"
        gt_path.write_text(json.dumps(dict(positive_pairs=[dict(pair_id=row["pair_id"],
            fragment_a_token=row["fragment_a_id"], fragment_b_token=row["fragment_b_id"],
            translation_gt_a_to_b_rc=[0., 0.]) for row in rows if row["label"]])))
        monkeypatch.setattr(run, "load_prepared_cache", lambda *args: (metadata, {}))
        monkeypatch.setattr(run.real, "input_batches", lambda *args: [fake_batch(rows, real=True)])
    old_open = Path.open
    old_np_load = np.load
    target_reads = []

    def check_target_read(path):
        if isinstance(path, (str, Path)) and Path(path) == gt_path:
            assert (output / "prediction_complete.json").is_file()
            if not target_reads:
                frozen = json.loads((output / "prediction_complete.json").read_text())
                assert frozen["sample_count"] == len(rows) and frozen["translation_gt_opened"] is False
                assert run.adapter._sha256_file(output / "pair_predictions.jsonl") == frozen["pair_predictions_sha256"]
            target_reads.append(path)

    def guarded_open(path, *args, **kwargs):
        check_target_read(path)
        return old_open(path, *args, **kwargs)

    def guarded_np_load(path, *args, **kwargs):
        check_target_read(path)
        return old_np_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(run.np, "load", guarded_np_load)
    args = run.parser().parse_args(["--method", "pairingnet", "--training-run", str(root),
        "--dataset", str(root), "--split", split, "--output", str(output), "--device", "cpu",
        "--workers", "0", "--prepared-cache", str(root), "--translation-gt-json", str(gt_path)])
    result = run.run(args)
    assert result["status"] == "complete" and result["sample_count"] == len(rows)
    assert target_reads and benchmark.calls == 1
    assert result["raw_layout"]["all_pairs_decoded"] and result["primary_operating_point"] == "recall_first"
    predictions = run.read_jsonl(output / "pair_predictions.jsonl")
    assert all("label" not in row and "target_translation_rc" not in row for row in predictions)
    assert len(run.read_jsonl(output / "pair_results.jsonl")) == len(rows)
    assert (output / "receipt.json").is_file()
    if split == "real":
        assert result["strict_summary"]["sample_count"] == 547
        assert result["strict_summary"]["positive_count"] == 508
    with pytest.raises(FileExistsError):
        run.run(args)


def test_summary_raw_layout_is_threshold_independent_and_invalid_positive_fails():
    decoder = run.DECODERS["pairingnet"]
    rows = [dict(label=label, pair_probability=score,
        layouts={decoder: dict(valid=valid, translation_l2_px=error)})
        for label, score, valid, error in ((True, .9, True, 3.), (True, .1, False, None),
                                         (False, .8, True, None), (False, .05, False, None))]
    result = run.summarize(rows, {"strict": .85, "recall_first": .05}, decoder)
    assert result["raw_layout"]["good_le10_count"] == 1
    assert result["raw_layout"]["good_le10_rate"] == .5
    assert result["operating_points"]["strict"]["classification"]["recall"] == .5
    assert result["primary"]["classification"]["recall"] == 1.
    assert result["primary"]["accepted_good_recall"] == .5
    assert result["primary"]["accepted_good_precision"] == .25


def test_shredding_wrapper_uses_existing_all_pairs_decoder_helper(monkeypatch):
    seen = []
    monkeypatch.setattr(run.adapter, "_validate_model_input_batch", lambda batch: None)
    monkeypatch.setattr(run.adapter, "_predict_shreddingnet", lambda frozen, batch, **kwargs: seen.append(kwargs) or "prediction")
    model = run.ExperimentalBenchmark("shreddingnet", run.adapter.SHREDDINGNET_METHOD_KEY,
        run.shredding.METHOD_ID, object(), torch.device("cpu"), {}, Path("scores"), "pair_score")
    assert model.predict_batch(object()) == "prediction"
    assert seen == [{"return_correspondence": False}]


def test_shredding_loader_is_not_replaced_by_an_unchecked_restore(tmp_path, monkeypatch):
    (tmp_path / "train_val_freeze.json").write_text("{}")
    calls = []

    def reject(path, **kwargs):
        calls.append(path)
        raise run.shredding.BenchmarkContractError("dependency SHA mismatch")

    monkeypatch.setattr(run.shredding, "load_frozen_inference", reject)
    with pytest.raises(run.shredding.BenchmarkContractError, match="dependency SHA"):
        run.load_experimental_benchmark("shreddingnet", tmp_path, "cpu")
    assert calls == [tmp_path / "train_val_freeze.json"]


def test_pairing_loader_retains_strict_checkpoint_validation(tmp_path, monkeypatch):
    config = run.pairing.PairingNetRachelRunConfig(dataset_root=tmp_path, output_root=tmp_path,
        official_source_root=tmp_path, train_materialized_manifest=tmp_path / "train-e1.json",
        experimental_data=True, selection_metric="recall95_precision", seed=260911,
        min_epochs=24, max_epochs=24, device="cuda:0")
    manifests = dict(train_sha256="a" * 64, val_sha256="b" * 64,
        materialized_training_hook=run.training_hook_identity())
    population = dict(formal_population_required=False, parent_lineage_disjoint=True,
        manifests=manifests, population=dict(train=dict(rows=24000), val=dict(rows=3000)))
    receipt = dict(schema_version=run.pairing.SCHEMA_VERSION, status="train_validation_complete",
        method_id=run.pairing.METHOD_ID, official_commit=run.pairing.OFFICIAL_COMMIT,
        winner_checkpoint_kind=run.pairing.WINNER_CHECKPOINT_KIND,
        sealed_synthetic_accessed=False, real_data_accessed=False,
        selection_metric="recall95_precision", pose_used_for_checkpoint_selection=False,
        population_audit=population, winner_epoch=2, selection_key=[.9, .95, -2.])
    torch.save(dict(run_config=asdict(config), model_config=asdict(run.pairing.PairingNetRachelModelConfig()),
        manifest_sha256=manifests, epoch=2, selection_key=receipt["selection_key"]), tmp_path / "winner.pt")
    for name, key in (("winner.pt", "winner_checkpoint_sha256"), ("last.pt", "last_checkpoint_sha256"),
            ("validation_threshold.json", "validation_threshold_file_sha256"),
            ("inference_contract.json", "inference_contract_sha256"),
            ("winner_validation_report.json", "winner_validation_report_sha256"),
            ("winner_validation_predictions.jsonl", "winner_validation_predictions_sha256")):
        if name != "winner.pt":
            (tmp_path / name).write_text("{}")
        receipt[key] = run.adapter._sha256_file(tmp_path / name)
    (tmp_path / "completion_receipt.json").write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(run.pairing, "load_frozen_validation_threshold",
        lambda threshold, winner: calls.append((threshold, winner)))

    def reject(path, device):
        calls.append((path, device))
        raise run.pairing.PairingNetBenchmarkError("dependency SHA mismatch")

    monkeypatch.setattr(run.pairing, "load_frozen_pairingnet_checkpoint", reject)
    with pytest.raises(run.pairing.PairingNetBenchmarkError, match="dependency SHA"):
        run.load_experimental_benchmark("pairingnet", tmp_path, "cpu")
    assert calls == [(tmp_path / "validation_threshold.json", tmp_path / "winner.pt"),
                     (tmp_path / "winner.pt", torch.device("cpu"))]


def test_incomplete_predictions_fail_without_target_reads_or_completion(cached_val, monkeypatch):
    root, benchmark = cached_val
    rows = source_rows("test", run.TEST_COUNT)
    write_jsonl(root / "pairs" / "test.jsonl", rows)
    monkeypatch.setattr(run, "load_experimental_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(run, "DataLoader", lambda *args, **kwargs: [fake_batch(rows[:-1])])

    def reject_targets(*args):
        pytest.fail("incomplete prediction run must not open targets")

    monkeypatch.setattr(run, "attach_test_targets", reject_targets)
    output = root / "incomplete"
    args = run.parser().parse_args(["--method", "pairingnet", "--training-run", str(root),
        "--dataset", str(root), "--split", "test", "--output", str(output), "--device", "cpu"])
    with pytest.raises(ValueError, match="incomplete"):
        run.run(args)
    assert json.loads((output / "protocol.json").read_text())["status"] == "failed"
    assert not (output / "prediction_complete.json").exists()
    assert not (output / "receipt.json").exists()
