"""CPU precomputation/cache of permutation-invariant, non-dustbin summaries.

This module never opens a model or a REAL dataset. Expensive full rectangular
SVD belongs only to precomputation, not the training forward. No gradient is
defined through the SVD, and no singular value is interpreted as a probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

FEATURE_SCHEMA = "rachel-global-real-transport-summary10/1"
CACHE_SCHEMA = "rachel-frozen-transport-summary-cache/1"
NORMALIZER_SCHEMA = "rachel-train-only-summary-standardizer/1"
FEATURE_NAMES = (
    "log1p_total_mass", "mean_log1p_mass_per_axis", "abs_diff_log1p_mass_per_axis",
    "sigma1_over_fro", "energy_top2", "energy_top4", "energy_top8", "energy_top16",
    "entropy_effective_rank_over_min_axis", "energy_participation_ratio_over_min_axis",
)
SOURCE_HASH_FIELDS = (
    "matcher_state_sha256", "matcher_code_sha256", "input_manifest_sha256",
    "prepared_inputs_sha256", "sampling_protocol_sha256",
)
INPUT_FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(name + " must be an explicit lowercase SHA256 digest")


def digest_arrays(arrays):
    """Bind ordered array names, shapes, dtypes and exact contiguous bytes.

    Caller must pass CPU NumPy arrays from the ACTUAL frozen physical input;
    pair IDs or filenames alone are not a sufficient augmentation/cache key.
    """
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        if value.dtype.hasobject:
            raise ValueError("object arrays cannot bind a numeric input")
        value = np.ascontiguousarray(value)
        header = canonical_json(dict(name=key, dtype=value.dtype.str, shape=list(value.shape))).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.nbytes.to_bytes(8, "big"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_input_sha256(arrays):
    if set(arrays) != set(INPUT_FIELDS):
        raise ValueError("input binding requires both exact masks, points and validity arrays")
    return digest_arrays(arrays)


def validate_identity(identity):
    if not isinstance(identity, dict) or identity.get("split") not in ("train", "val", "test", "real", "ood"):
        raise ValueError("cache identity must declare its exact dataset split")
    if identity.get("fixed_physical_inputs") is not True:
        raise ValueError("one cached row per pair requires fixed physical inputs; online augmentation needs new versioned keys/caches")
    for key in SOURCE_HASH_FIELDS:
        _require_hash(identity.get(key), key)
    canonical_json(identity)


@dataclass(frozen=True)
class Summary:
    values: tuple
    n_a: int
    n_b: int
    valid: bool


def compute_summary(real_transport, valid_a, valid_b, *, matrix_kind):
    """Return fixed 3D mass + 7D normalized spectral shape (float64 CPU).

    Empty effective dimensions return valid=False and all zeros. A nonempty
    zero-mass matrix is valid and returns all zeros, including rank/PR. Padding
    is removed with boolean validity, not by deleting a guessed final row.
    """
    if matrix_kind != "real_transport":
        raise ValueError("explicit non-dustbin real_transport is required")
    if np.iscomplexobj(real_transport):
        raise ValueError("transport must be real-valued, not complex")
    matrix = np.asarray(real_transport, dtype=np.float64)
    va, vb = np.asarray(valid_a), np.asarray(valid_b)
    if (matrix.ndim != 2 or va.dtype != np.bool_ or vb.dtype != np.bool_
            or va.shape != (matrix.shape[0],) or vb.shape != (matrix.shape[1],)):
        raise ValueError("rectangular assignment/boolean validity dimensions differ; dustbin layout cannot be guessed")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("real transport must be finite and nonnegative, including padding")
    matrix = matrix[np.ix_(va, vb)]
    na, nb = matrix.shape
    result = np.zeros(10, dtype=np.float64)
    if not na or not nb:
        return Summary(tuple(result), na, nb, False)
    mass = float(matrix.sum())
    if not np.isfinite(mass):
        raise ValueError("transport mass overflow")
    if mass == 0:
        return Summary(tuple(result), na, nb, True)
    per_axis = np.log1p([mass / na, mass / nb])
    result[:3] = (np.log1p(mass), per_axis.mean(), abs(per_axis[0] - per_axis[1]))
    # P / M improves scale conditioning. These seven statistics are scalar-
    # scale invariant and equal those from raw P in spectral_diagnostics.py.
    singular = np.linalg.svd(matrix / mass, compute_uv=False)
    fro = np.linalg.norm(singular)
    sigma_sum = singular.sum()
    if not np.isfinite(singular).all() or fro <= 0 or sigma_sum <= 0:
        raise ValueError("nonzero transport has invalid singular spectrum")
    energy = (singular / fro) ** 2
    probability = singular / sigma_sum
    positive = probability > 0
    rank = np.exp(-np.sum(probability[positive] * np.log(probability[positive])))
    pr = 1. / np.sum(energy ** 2)
    result[3:] = (singular[0] / fro, *(energy[:k].sum() for k in (2, 4, 8, 16)),
                  rank / min(na, nb), pr / min(na, nb))
    if not np.isfinite(result).all() or np.any(result[3:] < -1e-10) or np.any(result[3:] > 1. + 1e-10):
        raise ValueError("invalid normalized spectral summary")
    # Only absorb float64 roundoff at the known [0,1] mathematical bounds.
    result[3:] = np.clip(result[3:], 0., 1.)
    return Summary(tuple(float(v) for v in result), na, nb, True)


@dataclass(frozen=True)
class SummaryRecord:
    pair_id: str
    input_sha256: str
    assignment_sha256: str
    values: tuple
    n_a: int
    n_b: int
    valid: bool


def make_record(pair_id, input_sha256, real_transport, valid_a, valid_b):
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("nonempty ordered pair_id is required")
    _require_hash(input_sha256, "input_sha256")
    summary = compute_summary(real_transport, valid_a, valid_b, matrix_kind="real_transport")
    assignment_digest = digest_arrays(dict(real_transport=real_transport, valid_a=valid_a, valid_b=valid_b))
    return SummaryRecord(pair_id, input_sha256, assignment_digest, summary.values,
                         summary.n_a, summary.n_b, summary.valid)


def _read_record(row):
    if set(row) != set(SummaryRecord.__dataclass_fields__):
        raise ValueError("unknown summary record schema")
    if not isinstance(row["pair_id"], str) or not row["pair_id"]:
        raise ValueError("summary pair_id is empty")
    for key in ("input_sha256", "assignment_sha256"):
        _require_hash(row[key], key)
    values = np.asarray(row["values"], dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all() or (values < 0).any() or (values[3:] > 1).any():
        raise ValueError("summary values must match finite fixed 10D schema")
    if (type(row["n_a"]) is not int or type(row["n_b"]) is not int or min(row["n_a"], row["n_b"]) < 0
            or type(row["valid"]) is not bool or row["valid"] != bool(row["n_a"] and row["n_b"])):
        raise ValueError("summary validity and effective axis counts differ")
    if not row["valid"] and np.any(values):
        raise ValueError("empty summary must remain zero")
    return SummaryRecord(**{**row, "values": tuple(float(v) for v in values)})


def write_cache(path, identity, records):
    """Write a new, no-pickle cache and return its trusted file digest.

    Store this digest in the experiment registration/checkpoint. Loading from
    an expected digest is required; an adjacent editable manifest is not trust.
    """
    validate_identity(identity)
    rows = [_read_record(asdict(record)) for record in records]
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("cache must have a nonempty unique pair population")
    payload = dict(schema_version=CACHE_SCHEMA, feature_schema=FEATURE_SCHEMA,
        feature_names=list(FEATURE_NAMES), identity=identity,
        records=[asdict(row) for row in rows])
    data = canonical_json(payload).encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


class FrozenSummaryCache:
    """Validated source-bound rows; trainer batches require input digests."""

    def __init__(self, identity, records, cache_sha256):
        validate_identity(identity)
        _require_hash(cache_sha256, "cache_sha256")
        self._identity = json.loads(canonical_json(identity))
        self.records = tuple(_read_record(asdict(row)) for row in records)
        if not self.records or len({row.pair_id for row in self.records}) != len(self.records):
            raise ValueError("cache population must be nonempty and unique")
        self.cache_sha256 = cache_sha256
        self._lookup = {row.pair_id: row for row in self.records}

    @property
    def identity(self):
        # A caller cannot turn a VAL cache into TRAIN by mutating this view.
        return json.loads(canonical_json(self._identity))

    def batch(self, pair_ids, input_sha256s):
        if not pair_ids or len(pair_ids) != len(input_sha256s):
            raise ValueError("batch IDs and physical input digests must align")
        rows = []
        for pair_id, input_digest in zip(pair_ids, input_sha256s):
            row = self._lookup.get(pair_id)
            if row is None or row.input_sha256 != input_digest:
                raise ValueError("cache missing pair or physical input changed: " + str(pair_id))
            rows.append(row)
        return dict(values=np.asarray([row.values for row in rows], dtype=np.float32),
            valid=np.asarray([row.valid for row in rows], dtype=bool),
            n_a=np.asarray([row.n_a for row in rows], dtype=np.int64),
            n_b=np.asarray([row.n_b for row in rows], dtype=np.int64))


def load_cache(path, expected_identity, expected_cache_sha256):
    validate_identity(expected_identity)
    _require_hash(expected_cache_sha256, "expected_cache_sha256")
    if file_sha256(path) != expected_cache_sha256:
        raise ValueError("summary cache file hash differs from registered digest")
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema_version") != CACHE_SCHEMA or payload.get("feature_schema") != FEATURE_SCHEMA
            or payload.get("feature_names") != list(FEATURE_NAMES)
            or canonical_json(payload.get("identity")) != canonical_json(expected_identity)):
        raise ValueError("summary cache identity/feature schema differs from experiment registration")
    rows = [_read_record(row) for row in payload["records"]]
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("cached pair population is empty or duplicated")
    return FrozenSummaryCache(expected_identity, rows, expected_cache_sha256)


def fit_train_statistics(cache, *, minimum_std=1e-6):
    """TRAIN only, valid summaries only, equal fixed pair weighting, ddof=0.

    Zero-mass nonempty summaries are INCLUDED. Empty effective dimensions are
    excluded. Near-constant coordinates use scale=1 rather than exploding them.
    This fixed default is a numerical rule, not tuned from held-out data.
    """
    if not isinstance(cache, FrozenSummaryCache) or cache.identity.get("split") != "train":
        raise ValueError("standardization can be fitted only on the registered TRAIN cache")
    if not np.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError("minimum_std must be finite and positive")
    values = np.asarray([row.values for row in cache.records if row.valid], dtype=np.float64)
    if not len(values):
        raise ValueError("no valid TRAIN summaries for standardization")
    mean, raw_std = values.mean(0), values.std(0, ddof=0)
    scale = np.where(raw_std < minimum_std, 1., raw_std)
    return dict(schema_version=NORMALIZER_SCHEMA, feature_schema=FEATURE_SCHEMA,
        feature_names=list(FEATURE_NAMES), mean=mean.tolist(), scale=scale.tolist(),
        raw_std=raw_std.tolist(), minimum_std=float(minimum_std), ddof=0,
        training_cache_sha256=cache.cache_sha256, training_identity=cache.identity,
        fitted_count=int(len(values)), excluded_empty_count=len(cache.records) - len(values),
        fit_split="train", held_out_used_for_fit=False)
