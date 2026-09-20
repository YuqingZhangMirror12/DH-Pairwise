"""Bounded, prediction-only reuse between one arm's frozen selections.

No model, training, threshold fitting, GT file or pair_results is read here.
An unusable cache is a miss, never grounds to stop a formal evaluation. Input
hashing belongs in the evaluator's existing post-resampling batch traversal.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "rachel-decoupled-prediction-reuse/1"
INPUT_DIGEST_SCHEMA = "rachel-decoupled-model-input-batch/1"
EVALUATION_SCHEMA = "rachel-score-decoupled-evaluation/1"
SELECTIONS = ("fixed_epoch", "max_f1", "recall95")
FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")
PREDICTIONS_FILE = "pair_predictions.jsonl"
BATCHES_FILE = "prediction_batches.jsonl"
MODEL_FIELDS = ("training_run", "budget", "freeze_path", "freeze_sha256", "checkpoint_path",
    "checkpoint_sha256", "epoch", "seed", "model_config", "architecture", "sampling",
    "training_identity", "model_design")
PROTOCOL_FIELDS = ("schema_version", "split", "sample_count", "batch_size", "precision", "decoder",
    "decoder_config", "resampling", "model_input_fields", "script_sha256", "manifest_sha256",
    "decoder_design_unchanged", "ground_truth_used_to_select_candidates",
    "gt_attached_after_complete_prediction_freeze", "thresholds_fitted", "test_or_real_used_for_fit",
    "ood_used_for_fit", "inference_runtime", "prediction_reuse_schema", "input_digest_schema")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_header(digest, value):
    data = _canonical(value).encode("utf-8")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def batch_input_record(batch):
    """Hash exactly the six normalized CPU inputs and ordered identifiers.

    Canonical float32/bool conversion mirrors the evaluator's tensor input
    casts. This does not alter original arrays or inspect any targets/labels.
    """
    ids = list(batch.pair_ids)
    a, b = list(batch.fragment_a_tokens), list(batch.fragment_b_tokens)
    if len(a) != len(ids) or len(b) != len(ids) or not all(isinstance(x, str) for x in ids + a + b):
        raise ValueError("prediction input identifiers must be aligned strings")
    digest = hashlib.sha256()
    _hash_header(digest, dict(schema_version=INPUT_DIGEST_SCHEMA, pair_ids=ids,
                             fragment_a_tokens=a, fragment_b_tokens=b))
    for field in FIELDS:
        dtype = np.dtype(np.bool_) if field.startswith("contour_valid") else np.dtype("<f4")
        array = np.asarray(getattr(batch, field), dtype=dtype, order="C")
        if not array.ndim or array.shape[0] != len(ids):
            raise ValueError("model input batch dimensions do not align")
        _hash_header(digest, dict(field=field, dtype=dtype.str, shape=list(array.shape)))
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return dict(schema_version=INPUT_DIGEST_SCHEMA, input_sha256=digest.hexdigest(),
                pair_ids=ids, sample_count=len(ids))


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_json(path):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(path.read_text(), parse_constant=bad_constant)


def _verified_lines(path, expected_sha):
    if path.is_symlink() or not _valid_sha(expected_sha):
        raise ValueError("missing prediction file seal")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("prediction file seal differs")
    def bad_constant(value):
        raise ValueError("nonfinite prediction JSON")
    return [json.loads(line, parse_constant=bad_constant) for line in raw.splitlines() if line.strip()]


def _compatible(source, target):
    if (source.get("schema_version") != EVALUATION_SCHEMA or source.get("status") != "complete" or
            target.get("schema_version") != EVALUATION_SCHEMA or
            source.get("prediction_reuse_schema") != SCHEMA or source.get("input_digest_schema") != INPUT_DIGEST_SCHEMA or
            not source.get("inference_runtime") or not target.get("inference_runtime")):
        return False
    for field in PROTOCOL_FIELDS:
        if field not in source or field not in target or _canonical(source[field]) != _canonical(target[field]):
            return False
    for field in MODEL_FIELDS:
        if (field not in source.get("model", {}) or field not in target.get("model", {}) or
                _canonical(source["model"][field]) != _canonical(target["model"][field])):
            return False
    for field in ("dataset_root", "prepared_cache"):
        if (field in source) != (field in target) or _canonical(source.get(field)) != _canonical(target.get(field)):
            return False
    return ("dataset_root" in source) != ("prepared_cache" in source)


def _validate_record(record):
    if (not isinstance(record, dict) or record.get("schema_version") != INPUT_DIGEST_SCHEMA or
            not _valid_sha(record.get("input_sha256")) or not isinstance(record.get("pair_ids"), list) or
            not record["pair_ids"] or not all(isinstance(x, str) for x in record["pair_ids"]) or
            type(record.get("sample_count")) is not int or record["sample_count"] != len(record["pair_ids"])):
        raise ValueError("invalid input digest record")


class PredictionReuse:
    """Only two fixed selection siblings, same arm/split; never a broad search."""

    def __init__(self, destination, protocol, expected_ids):
        self.candidates, self.skipped = [], []
        self.reused_batches = self.reused_pairs = self.inferred_batches = self.inferred_pairs = 0
        self.sources = {}
        self.expected_ids = list(expected_ids)
        destination = Path(destination).resolve()
        selection, split = destination.parent.name, destination.name
        if (selection not in SELECTIONS or split not in ("test", "real", "ood") or
                protocol.get("split") != split or protocol.get("model", {}).get("selection") != selection or
                protocol.get("sample_count") != len(self.expected_ids) or
                len(set(self.expected_ids)) != len(self.expected_ids)):
            self.skipped.append(dict(reason="unrecognized_selection_destination_or_population"))
            return
        evaluation_root = destination.parent.parent
        for other in SELECTIONS:
            if other == selection:
                continue
            source = evaluation_root / other / split
            if not source.exists():
                self.skipped.append(dict(selection=other, reason="not_available"))
                continue
            try:
                if source.resolve() != source or (source / "protocol.json").is_symlink():
                    raise ValueError("selection cache must remain inside its owned sibling directory")
                saved = _read_json(source / "protocol.json")
                if saved.get("model", {}).get("selection") != other or not _compatible(saved, protocol):
                    self.skipped.append(dict(selection=other, reason="different_or_legacy_prediction_contract"))
                    continue
                complete = _read_json(source / "prediction_complete.json")
                if (complete.get("status") != "all_predictions_frozen" or
                        complete.get("sample_count") != len(self.expected_ids) or
                        complete.get("checkpoint_sha256") != protocol["model"]["checkpoint_sha256"] or
                        complete.get("real_gt_opened") is not False or complete.get("review_labels_opened") is not False):
                    raise ValueError("prediction completion receipt differs")
                seals = saved["prediction_files_sha256"]
                rows = _verified_lines(source / PREDICTIONS_FILE, seals[PREDICTIONS_FILE])
                records = _verified_lines(source / BATCHES_FILE, seals[BATCHES_FILE])
                if [row["pair_id"] for row in rows] != self.expected_ids:
                    raise ValueError("prediction population/order differs")
                flattened, batches, offset = [], [], 0
                for record in records:
                    _validate_record(record)
                    flattened.extend(record["pair_ids"])
                    count = record["sample_count"]
                    batches.append(rows[offset:offset + count])
                    offset += count
                if flattened != self.expected_ids:
                    raise ValueError("batch digest population/order differs")
                self.candidates.append(dict(path=str(source), selection=other, records=records, batches=batches,
                                            checkpoint_sha256=saved["model"]["checkpoint_sha256"]))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError) as error:
                self.skipped.append(dict(selection=other, reason="unusable_prediction_cache", detail=str(error)[:160]))

    def lookup(self, batch_index, record):
        """Return a deepcopy of untouched source predictions or safe miss."""
        count = len(record.get("pair_ids", [])) if isinstance(record, dict) else 0
        try:
            _validate_record(record)
            for candidate in self.candidates:
                if type(batch_index) is not int or not 0 <= batch_index < len(candidate["records"]):
                    continue
                saved = candidate["records"][batch_index]
                if saved != record:
                    continue
                rows = candidate["batches"][batch_index]
                if [row["pair_id"] for row in rows] != record["pair_ids"]:
                    continue
                self.reused_batches += 1
                self.reused_pairs += count
                source = self.sources.setdefault(candidate["path"], dict(selection=candidate["selection"],
                    checkpoint_sha256=candidate["checkpoint_sha256"], reused_batches=0, reused_pairs=0))
                source["reused_batches"] += 1
                source["reused_pairs"] += count
                return deepcopy(rows)
        except (ValueError, TypeError, KeyError):
            pass
        self.inferred_batches += 1
        self.inferred_pairs += count
        return None

    def summary(self):
        return deepcopy(dict(schema_version=SCHEMA, candidate_count=len(self.candidates), skipped=self.skipped,
            reused_batches=self.reused_batches, reused_pairs=self.reused_pairs,
            inferred_batches=self.inferred_batches, inferred_pairs=self.inferred_pairs,
            sources=self.sources, thresholds_fitted=False, ground_truth_used=False))


__all__ = ["SCHEMA", "INPUT_DIGEST_SCHEMA", "FIELDS", "PREDICTIONS_FILE", "BATCHES_FILE",
           "batch_input_record", "PredictionReuse"]
