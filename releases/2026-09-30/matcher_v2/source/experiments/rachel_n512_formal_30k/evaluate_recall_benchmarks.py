"""Experimental benchmark recall evaluation with saved-clean-VAL calibration.

The old formal/convergence evaluator is untouched. This narrow entry point
accepts only explicit materialized-TRAIN experiments, verifies the original
checkpoint/dependency hashes, and calibrates from the winner's already saved
VAL3000 scores. TEST/REAL targets are opened only after all score/pose outputs
are closed and fsynced. Every pair retains the method's original decoder,
independent of pair probability and every operating threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.rachel_n512_formal_30k import run_layout_decoder_experiment as common
from experiments.rachel_n512_formal_30k import run_real_layout_decoder_experiment as real
from experiments.rachel_n512_formal_30k.recall_operating_points import fit_operating_points
from experiments.rachel_n512_formal_30k.run_real_contiguous_seam_ablation import (
    DEFAULT_PREPARED_CACHE, load_prepared_cache,
)
from experiments.rachel_n512_formal_30k.train_realism_data_ablation import save_json
from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairing
from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as shredding
from staging.pairwise_v0_2.baselines import rachel_same_data_benchmark_eval_adapter as adapter
from staging.pairwise_v0_2.baselines.rachel_materialized_training import training_hook_identity
from staging.pairwise_v0_2.pairwise_data import rachel_training_dataset as runtime

SCHEMA = "rachel-recall-benchmark-evaluation/1"
VAL_COUNT, TEST_COUNT, REAL_COUNT, BATCH_SIZE = 3000, 3000, 1016, 8
DECODERS = {"pairingnet": "pairingnet_native_translation_consensus",
            "shreddingnet": "shreddingnet_native_translation_consensus"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checked_hash(path, expected):
    return adapter._require_hash(Path(path), expected, str(path))


@dataclass
class ExperimentalBenchmark:
    method: str
    method_key: str
    method_id: str
    _predictor: object
    _device: torch.device
    identity: dict
    validation_scores_path: Path
    validation_score_field: str

    def predict_batch(self, batch):
        adapter._validate_model_input_batch(batch)
        if self.method == "pairingnet":
            with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16,
                                enabled=self.identity["precision"] == "bf16"):
                return adapter._predict_pairingnet(self, batch, return_correspondence=False)
        # The existing helper explicitly sets compute_pose_for_rejected=True.
        return adapter._predict_shreddingnet(self, batch, return_correspondence=False)


def load_experimental_benchmark(method, training_run, device):
    """Keep strict checkpoint/source checks; require explicit new-data identity."""
    if method not in DECODERS:
        raise ValueError("unknown benchmark method")
    root = adapter._require_no_symlink_components(Path(training_run), "experimental training run")
    device = torch.device(device)
    if method == "pairingnet":
        authority_path = adapter._regular_file(root, "completion_receipt.json", "PairingNet completion")
        receipt = adapter._read_json(authority_path, "PairingNet completion")
        if (receipt.get("schema_version") != pairing.SCHEMA_VERSION
                or receipt.get("status") != "train_validation_complete"
                or receipt.get("method_id") != pairing.METHOD_ID
                or receipt.get("official_commit") != pairing.OFFICIAL_COMMIT
                or receipt.get("winner_checkpoint_kind") != pairing.WINNER_CHECKPOINT_KIND
                or receipt.get("sealed_synthetic_accessed") is not False
                or receipt.get("real_data_accessed") is not False
                or receipt.get("selection_metric") != "recall95_precision"
                or receipt.get("pose_used_for_checkpoint_selection") is not False):
            raise ValueError("PairingNet requires a completed explicit recall experiment")
        files = {}
        for name, digest_key in (("winner.pt", "winner_checkpoint_sha256"),
                ("last.pt", "last_checkpoint_sha256"),
                ("validation_threshold.json", "validation_threshold_file_sha256"),
                ("inference_contract.json", "inference_contract_sha256"),
                ("winner_validation_report.json", "winner_validation_report_sha256"),
                ("winner_validation_predictions.jsonl", "winner_validation_predictions_sha256")):
            files[name] = adapter._regular_file(root, name, "PairingNet " + name)
            checked_hash(files[name], receipt.get(digest_key))
        checkpoint = torch.load(files["winner.pt"], map_location="cpu")
        config = pairing.PairingNetRachelRunConfig(**checkpoint["run_config"])
        pairing._validate_formal_protocol(config, pairing.PairingNetRachelModelConfig(**checkpoint["model_config"]))
        population = receipt["population_audit"]
        if (not config.experimental_data or config.selection_metric != "recall95_precision"
                or population.get("formal_population_required") is not False
                or population.get("parent_lineage_disjoint") is not True
                or checkpoint.get("manifest_sha256") != population.get("manifests")
                or checkpoint.get("epoch") != receipt.get("winner_epoch")
                or checkpoint.get("selection_key") != receipt.get("selection_key")
                or population["manifests"].get("materialized_training_hook") != training_hook_identity()
                or population.get("population", {}).get("val", {}).get("rows") != VAL_COUNT):
            raise ValueError("PairingNet experimental data/winner identity differs")
        pairing.load_frozen_validation_threshold(files["validation_threshold.json"], files["winner.pt"])
        predictor = pairing.load_frozen_pairingnet_checkpoint(files["winner.pt"], device)
        source_sha, document_sha = pairing._adapter_identity_hashes()
        if (receipt.get("adapter_source_sha256") != source_sha
                or receipt.get("adaptation_contract_sha256") != document_sha):
            raise ValueError("PairingNet completion dependency hashes differ")
        scores_path, score_field = files["winner_validation_predictions.jsonl"], "pair_probability"
        manifests = population["manifests"]
        identity = dict(seed=config.seed, precision=config.precision,
            training_manifest_sha256=manifests["train_sha256"],
            validation_manifest_sha256=manifests["val_sha256"],
            checkpoint_sha256_by_stage={"winner": receipt["winner_checkpoint_sha256"]},
            selection_metric=config.selection_metric, train_unique_count=population["population"]["train"]["rows"],
            epochs_completed=receipt["epochs_completed"], winner_epoch=receipt["winner_epoch"],
            convergence_demonstrated=receipt.get("convergence_demonstrated", False))
        method_key, method_id = adapter.PAIRINGNET_METHOD_KEY, pairing.METHOD_ID
    else:
        authority_path = adapter._regular_file(root, "train_val_freeze.json", "ShreddingNet completion")
        # This verifies all three winners, stage receipts, threshold score bytes,
        # recipe, dependency hashes, provenance and validation score transaction.
        predictor = shredding.load_frozen_inference(authority_path, device=device)
        receipt = shredding._read_verified_json(authority_path, "ShreddingNet freeze")
        recipe = predictor.recipe
        population = receipt["dataset_audit"]
        if (not recipe.experimental_data or recipe.selection_metric != "recall95_precision"
                or population.get("formal_counts_required") is not False
                or not population.get("train_materialized_manifest")
                or population.get("lineage_disjoint") is not True
                or population.get("counts", {}).get("val", {}).get("rows") != VAL_COUNT):
            raise ValueError("ShreddingNet requires a completed explicit recall experiment")
        scores_path = adapter._regular_file(root, "validation_threshold_scores.jsonl", "ShreddingNet VAL scores")
        checked_hash(scores_path, receipt["threshold"]["score_file_sha256"])
        score_field = "pair_score"
        identity = dict(seed=recipe.seed, precision="amp" if predictor.runtime.amp else "fp32",
            training_manifest_sha256=population["manifest_sha256"]["train"],
            validation_manifest_sha256=population["manifest_sha256"]["val"],
            checkpoint_sha256_by_stage={key: value["sha256"] for key, value in receipt["checkpoints"].items()},
            selection_metric=recipe.selection_metric, train_unique_count=population["counts"]["train"]["rows"],
            planned_stage_epochs={key: getattr(recipe, key + "_epochs") for key in ("coarse", "matching", "classify")},
            convergence_demonstrated=False)
        method_key, method_id = adapter.SHREDDINGNET_METHOD_KEY, shredding.METHOD_ID
    identity.update(method=method, method_id=method_id, training_run=str(root),
        training_authority_path=str(authority_path), training_authority_sha256=adapter._sha256_file(authority_path),
        validation_scores_path=str(scores_path), validation_scores_sha256=adapter._sha256_file(scores_path),
        evaluation_source_sha256=adapter._sha256_file(Path(__file__)),
        recall_helper_sha256=training_hook_identity()["recall_operating_points.py"],
        original_decoder=DECODERS[method], all_pairs_decoded=True, test_or_real_used_for_fit=False)
    return ExperimentalBenchmark(method, method_key, method_id, predictor, device, identity, scores_path, score_field)


def population_manifest(dataset_root, split, expected_count):
    if split not in {"val", "test"}:
        raise ValueError("only original VAL/TEST manifests are allowed")
    root = Path(dataset_root).resolve(strict=True)
    path = runtime._safe_release_path(root, "pairs/" + split + ".jsonl", prefix=("pairs",), suffix=".jsonl")
    rows = read_jsonl(path)
    if (len(rows) != expected_count or len({row["pair_id"] for row in rows}) != expected_count
            or any(row.get("split") != split or type(row.get("label")) is not bool for row in rows)
            or sum(row["label"] for row in rows) != expected_count // 2):
        raise ValueError("original balanced " + split + " population differs")
    return rows, path


def freeze_validation(benchmark, dataset_root):
    """Calibrate only complete, hash-bound, saved winner VAL scores; no forward."""
    authority, val_path = population_manifest(dataset_root, "val", VAL_COUNT)
    checked_hash(val_path, benchmark.identity["validation_manifest_sha256"])
    checked_hash(benchmark.validation_scores_path, benchmark.identity["validation_scores_sha256"])
    rows = read_jsonl(benchmark.validation_scores_path)
    if (len(rows) != VAL_COUNT
            or [r.get("pair_id") for r in rows] != [r["pair_id"] for r in authority]
            or any(type(row.get("label")) is not bool or row["label"] != source["label"]
                   for row, source in zip(rows, authority))):
        raise ValueError("saved winner scores differ from original complete VAL order/labels")
    scores = np.asarray([row[benchmark.validation_score_field] for row in rows], float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("saved winner validation probabilities are invalid")
    operating = fit_operating_points([row["label"] for row in rows], scores)
    return dict(schema_version=SCHEMA, status="complete_validation_frozen", source_split="val",
        sample_count=VAL_COUNT, positive_count=VAL_COUNT // 2, negative_count=VAL_COUNT // 2,
        original_validation_unchanged=True, used_saved_winner_scores=True, validation_forward_repeated=False,
        test_or_real_used_for_fit=False, model_identity=benchmark.identity, **operating)


class TargetBlindTestDataset:
    """Load TEST masks/contours only; do not resolve or open target archives."""
    def __init__(self, root, rows):
        self.root, self.rows = Path(root).resolve(strict=True), rows
        self.config = runtime.RachelDatasetConfig()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        fragments, arrays = [], []
        for side in "ab":
            fragment = runtime._fragment_input(self.root, row["fragment_" + side], "fragment_" + side)
            mask, coarse = runtime._load_mask(fragment.mask_path, self.config)
            points, valid = runtime._load_contour(fragment.contour_path, self.config)
            fragments.append(fragment)
            arrays.append((mask, coarse, points, valid))
        a, b = arrays
        return runtime.RachelPairSample(row["pair_id"], fragments[0].token, fragments[1].token,
            a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3],
            np.where(a[3], -1, -2).astype(np.int64), np.where(b[3], -1, -2).astype(np.int64),
            np.float32(0), np.zeros(2, np.float32), np.zeros(2, np.float32), np.bool_(False))


def prediction_rows(benchmark, batch):
    output = benchmark.predict_batch(batch)
    if tuple(output.pair_ids) != tuple(batch.pair_ids):
        raise ValueError("benchmark prediction order differs")
    rows = []
    for index, pair_id in enumerate(output.pair_ids):
        valid = bool(output.translation_valid[index])
        translation = np.asarray(output.translation_hat_rc[index], float)
        rows.append(common.clean(dict(pair_id=pair_id, fragment_a=batch.fragment_a_tokens[index],
            fragment_b=batch.fragment_b_tokens[index], pair_probability=float(output.pair_probability[index]),
            decision_valid=bool(output.decision_valid[index]),
            auxiliary_scores={key: float(value[index]) for key, value in output.auxiliary_scores.items()},
            layouts={DECODERS[benchmark.method]: dict(valid=valid, translation_rc=translation if valid else None,
                offset_b_in_a_rc=-translation if valid else None, decoded_independently_of_pair_probability=True)})))
    return rows


def attach_test_targets(predictions, source_rows, dataset_root):
    root = Path(dataset_root).resolve(strict=True)
    if [row["pair_id"] for row in predictions] != [row["pair_id"] for row in source_rows]:
        raise ValueError("TEST target order differs")
    for prediction, source in zip(predictions, source_rows):
        if (prediction["fragment_a"] != source["fragment_a"]["fragment_token"]
                or prediction["fragment_b"] != source["fragment_b"]["fragment_token"]):
            raise ValueError("TEST ordered fragment identity differs")
        gt = None
        if source["label"]:
            path = runtime._safe_release_path(root, source["correspondence_path"],
                prefix=("targets", "pairs"), suffix=".npz")
            with np.load(path, allow_pickle=False) as archive:
                gt = np.asarray(archive["translation_a_to_b_rc"], float)
            if gt.shape != (2,) or not np.isfinite(gt).all():
                raise ValueError("TEST positive translation target must be finite [2]")
        prediction.update(label=source["label"], target_translation_rc=common.clean(gt),
            source_unit_ids=sorted({source["fragment_a"]["split_unit_id"], source["fragment_b"]["split_unit_id"]}))
        for layout in prediction["layouts"].values():
            layout["translation_l2_px"] = (float(np.linalg.norm(np.asarray(layout["translation_rc"]) - gt))
                if gt is not None and layout["valid"] else None)
    return predictions


def summarize(rows, thresholds, decoder):
    labels = np.asarray([row["label"] for row in rows], bool)
    scores = np.asarray([row["pair_probability"] for row in rows], float)
    valid = np.asarray([row["layouts"][decoder]["valid"] for row in rows], bool)
    errors = np.asarray([row["layouts"][decoder]["translation_l2_px"]
        if row["layouts"][decoder]["translation_l2_px"] is not None else np.inf for row in rows])
    good = labels & valid & (errors <= 10.)
    positive_count = int(labels.sum())
    result = dict(sample_count=len(rows), positive_count=positive_count, negative_count=int((~labels).sum()),
        raw_layout=dict(decoder=decoder, all_pairs_decoded=True, valid_count=int(valid.sum()),
            positive_valid_count=int((labels & valid).sum()), good_le10_count=int(good.sum()),
            good_le10_rate=float(good.sum() / max(1, positive_count)),
            denominator="all true pairs; invalid layouts count as failures"), operating_points={})
    for name, threshold in thresholds.items():
        accepted = scores >= threshold
        accepted_good = int((accepted & good).sum())
        accepted_positive = int((accepted & labels).sum())
        accepted_count = int(accepted.sum())
        result["operating_points"][name] = dict(classification=common.classification(labels, scores, threshold),
            accepted_count=accepted_count, accepted_positive_count=accepted_positive,
            accepted_good_le10_count=accepted_good,
            accepted_good_precision=accepted_good / max(1, accepted_count),
            accepted_good_recall=accepted_good / max(1, positive_count),
            accepted_good_f1=2 * accepted_good / max(1, accepted_count + positive_count),
            conditional_pose_success=accepted_good / accepted_positive if accepted_positive else None)
    if "recall_first" in result["operating_points"]:
        result["primary_operating_point"] = "recall_first"
        result["primary"] = result["operating_points"]["recall_first"]
    return result


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(common.clean(row), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = dict(schema_version=SCHEMA, status="running", method=args.method, split=args.split,
        all_pairs_decoded=True, test_or_real_used_for_fit=False, target_gt_evaluation_after_prediction_freeze=True)
    save_json(output / "protocol.json", protocol)
    predictions = []
    try:
        benchmark = load_experimental_benchmark(args.method, args.training_run, args.device)
        freeze = freeze_validation(benchmark, args.dataset)
        common.write_json(output / "validation_freeze.json", freeze)
        freeze_sha = adapter._sha256_file(output / "validation_freeze.json")
        if args.split == "test":
            source_rows, source_path = population_manifest(args.dataset, "test", TEST_COUNT)
            batches = DataLoader(TargetBlindTestDataset(args.dataset, source_rows), batch_size=BATCH_SIZE,
                shuffle=False, drop_last=False, num_workers=args.workers, collate_fn=runtime.collate_rachel_pairs)
            source = dict(test_manifest_sha256=adapter._sha256_file(source_path), dataset=str(Path(args.dataset).resolve()))
            expected_count = TEST_COUNT
        else:
            metadata, arrays = load_prepared_cache(args.prepared_cache)
            source_rows = metadata["pairs"]
            batches = real.input_batches(metadata, arrays, BATCH_SIZE)
            source = dict(prepared_cache=str(Path(args.prepared_cache).resolve()),
                prepared_cache_manifest_sha256=adapter._sha256_file(Path(args.prepared_cache) / "manifest.json"),
                original_real_manifest_sha256=metadata["manifest_sha256"])
            expected_count = REAL_COUNT
        protocol.update(model_identity=benchmark.identity, validation_freeze_sha256=freeze_sha,
            sample_count=expected_count, original_decoder=DECODERS[args.method], source=source)
        save_json(output / "protocol.json", protocol)
        with (output / "pair_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for batch in batches:
                rows = prediction_rows(benchmark, batch)
                predictions.extend(rows)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if len(predictions) % 64 < BATCH_SIZE or len(predictions) == expected_count:
                    print(json.dumps(dict(status="predicting", split=args.split, processed=len(predictions),
                        total=expected_count, elapsed_s=round(time.perf_counter() - started, 2))), flush=True)
            os.fsync(stream.fileno())
        ids = [row["pair_id"] for row in predictions]
        if (len(ids) != expected_count or len(set(ids)) != expected_count
                or ids != [row["pair_id"] for row in source_rows]):
            raise ValueError("held-out predictions are incomplete, duplicated or reordered")
        prediction_sha = adapter._sha256_file(output / "pair_predictions.jsonl")
        common.write_json(output / "prediction_complete.json", dict(status="all_predictions_frozen",
            sample_count=expected_count, pair_predictions_sha256=prediction_sha,
            translation_gt_opened=False, validation_freeze_sha256=freeze_sha))
        if args.split == "test":
            rows = attach_test_targets(predictions, source_rows, args.dataset)
        else:
            rows = real.attach_ground_truth(predictions, source_rows, args.translation_gt_json)
            protocol["translation_gt_json_sha256"] = adapter._sha256_file(Path(args.translation_gt_json))
        if sum(row["label"] for row in rows) != expected_count // 2:
            raise ValueError("held-out target balance differs")
        write_rows(output / "pair_results.jsonl", rows)
        summary = summarize(rows, freeze["thresholds"], DECODERS[args.method])
        if args.split == "real":
            strict = [row for row in rows if row["strict_member"]]
            if len(strict) != 547 or sum(row["label"] for row in strict) != 508:
                raise ValueError("strict REAL must retain 508 positive and 39 negative pairs")
            summary["strict_summary"] = summarize(strict, freeze["thresholds"], DECODERS[args.method])
            common.write_json(output / "strict547_summary.json", summary["strict_summary"])
        protocol.update(status="complete", elapsed_s=time.perf_counter() - started)
        summary.update(protocol)
        common.write_json(output / "summary.json", summary)
        save_json(output / "protocol.json", protocol)
        common.write_json(output / "receipt.json", dict(protocol, pair_predictions_sha256=prediction_sha,
            pair_results_sha256=adapter._sha256_file(output / "pair_results.jsonl"),
            summary_sha256=adapter._sha256_file(output / "summary.json")))
        print(json.dumps(dict(status="complete", method=args.method, split=args.split, output=str(output))), flush=True)
        return summary
    except Exception as error:
        protocol.update(status="failed", error=repr(error), completed_predictions=len(predictions))
        save_json(output / "protocol.json", protocol)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=tuple(DECODERS))
    p.add_argument("--training-run", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("test", "real"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepared-cache", type=Path, default=DEFAULT_PREPARED_CACHE)
    p.add_argument("--translation-gt-json", type=Path, default=real.DEFAULT_TRANSLATION_GT)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
