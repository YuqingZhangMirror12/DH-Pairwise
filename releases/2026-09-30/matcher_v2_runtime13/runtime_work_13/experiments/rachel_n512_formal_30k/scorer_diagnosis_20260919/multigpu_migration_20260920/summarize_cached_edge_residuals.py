"""Read only geometry from complete S7 M12 TRAIN/VAL caches, on one CPU thread.

No model, feature arrays, Q weights, threshold fitting or layout solving. Each
pair contributes its unweighted mean final-inlier residual, in canvas pixels.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SOURCE_SHA = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
COUNTS = {"train": 24000, "val": 3000}
CHUNK = 256
ARRAYS = {
    "points_a": ("float32", (512, 2)), "points_b": ("float32", (512, 2)),
    "candidate_indices": ("int64", (512, 2)),
    "candidate_valid": ("bool", (512,)), "candidate_inliers": ("bool", (512,)),
    "layout_valid": ("bool", ()), "translation_a_to_b_rc": ("float32", (2,)),
    "mask_a": ("bool", (512,)), "mask_b": ("bool", (512,)),
    "label": ("float32", ()), "ready": ("bool", ()),
}
BINS = ("1_32", "33_64", "65_128", "129_plus")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def load_cache(root, split):
    p = json.loads((root / "protocol.json").read_text())
    count = COUNTS[split]
    if (p.get("schema") != "s7-m12-frozen-token-edge-cache/1"
            or p.get("status") != "complete" or p.get("split") != split
            or p.get("source_checkpoint_sha256") != SOURCE_SHA
            or p.get("precompute_device") != "cpu"
            or any(p.get(k) != count for k in ("pair_count", "expected_full_count", "completed_pairs"))):
        raise ValueError("requires complete fixed-source full TRAIN24K/VAL3K CPU cache")
    records = json.loads((root / "pairs.json").read_text())
    if (len(records) != count or sha(root / "pairs.json") != p.get("pairs_sha256")
            or any(r.get("ordinal") != i for i, r in enumerate(records))
            or len({r["pair_id"] for r in records}) != count):
        raise ValueError("pair IDs/count/order differ from complete cache")
    arrays = {}
    for name, (dtype, tail) in ARRAYS.items():
        a = np.load(root / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        if a.shape != (count, *tail) or a.dtype != np.dtype(dtype):
            raise ValueError("cache shape/dtype mismatch: " + name)
        arrays[name] = a
    if (not arrays["ready"].all() or not np.isin(arrays["label"], [0., 1.]).all()
            or not np.array_equal(arrays["label"], [r["label"] for r in records])
            or int(arrays["label"].sum()) != p.get("positive_count")):
        raise ValueError("cache readiness/labels inconsistent")
    return arrays, records, dict(root=str(root), count=count,
        positive_count=int(arrays["label"].sum()), source_checkpoint_sha256=SOURCE_SHA,
        protocol_sha256=sha(root / "protocol.json"), pairs_sha256=p["pairs_sha256"],
        loaded_arrays=list(ARRAYS), model_loaded=False, features_loaded=False)


def bin_name(n):
    if n < 1:
        raise ValueError("valid final-inlier set has no unique endpoints")
    return "1_32" if n <= 32 else "33_64" if n <= 64 else "65_128" if n <= 128 else "129_plus"


def chunk_rows(arrays, records, split, start, stop):
    a = {k: np.asarray(v[start:stop]) for k, v in arrays.items()}
    active = a["candidate_valid"] & a["candidate_inliers"] & a["layout_valid"][:, None]
    if (a["candidate_inliers"] & ~a["candidate_valid"]).any():
        raise ValueError("inlier outside candidate set")
    for local, ordinal in enumerate(range(start, stop)):
        row = dict(split=split, ordinal=ordinal, pair_id=records[ordinal]["pair_id"],
            label=int(a["label"][local]), layout_valid=bool(a["layout_valid"][local]),
            unique_a=int(a["mask_a"][local].sum()), unique_b=int(a["mask_b"][local].sum()),
            inlier_edges=int(active[local].sum()))
        row["min_unique_endpoints"] = min(row["unique_a"], row["unique_b"])
        if not row["layout_valid"]:
            yield dict(row, endpoint_bin="invalid", mean_residual_px=None, mean_residual_norm=None)
            continue
        ij = a["candidate_indices"][local, active[local]]
        if len(ij) < 3 or (ij < 0).any() or (ij >= 512).any():
            raise ValueError("valid layout requires at least three indexed final inliers")
        if (len(np.unique(ij, axis=0)) != len(ij)
                or len(np.unique(ij[:, 0])) != row["unique_a"]
                or len(np.unique(ij[:, 1])) != row["unique_b"]):
            raise ValueError("edge identity/unique endpoint masks differ")
        delta = (a["points_b"][local, ij[:, 1]] - a["points_a"][local, ij[:, 0]]
                 - a["translation_a_to_b_rc"][local])
        # Same FP32 pb-pa-t and Euclidean norm as edge_metadata, BEFORE /10.
        residual = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float32))
        if not np.isfinite(residual).all():
            raise ValueError("nonfinite active final-inlier residual")
        mean = float(residual.mean(dtype=np.float64))
        yield dict(row, endpoint_bin=bin_name(row["min_unique_endpoints"]),
            mean_residual_px=mean, mean_residual_norm=mean / 10.)


def describe(rows):
    x = np.asarray([r["mean_residual_px"] for r in rows], dtype=np.float64)
    if not len(x):
        return dict(pair_count=0)
    return dict(pair_count=len(x), mean=float(x.mean()), median=float(np.median(x)),
        p10=float(np.quantile(x, .1)), p25=float(np.quantile(x, .25)),
        p75=float(np.quantile(x, .75)), p90=float(np.quantile(x, .9)),
        p95=float(np.quantile(x, .95)), min=float(x.min()), max=float(x.max()))


def summarize(rows):
    groups = {}
    for label in (0, 1):
        selected = [r for r in rows if r["label"] == label]
        valid = [r for r in selected if r["layout_valid"]]
        groups[str(label)] = dict(total_pairs=len(selected), valid_pairs=len(valid),
            invalid_pairs=len(selected) - len(valid), all_valid=describe(valid),
            endpoint_bins={b: describe([r for r in valid if r["endpoint_bin"] == b]) for b in BINS})
    return dict(total_pairs=len(rows), label_groups=groups)


def run(cache_root, output):
    cache_root, output = Path(cache_root).resolve(), Path(output).resolve()
    if cache_root == output or cache_root in output.parents:
        raise ValueError("output must be outside the read-only cache tree")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = dict(schema="full-cache-final-edge-residuals/1", status="running",
        source_checkpoint_sha256=SOURCE_SHA, script_sha256=sha(__file__), chunk_size=CHUNK,
        device="cpu", threads=1, arrays=list(ARRAYS), model_loaded=False, features_loaded=False,
        geometry="mean over final inlier EDGES of norm(points_b[j]-points_a[i]-predicted_t), in canvas px",
        weighting="unweighted edges within each pair; unweighted pairs in distribution summaries; NOT Q-weighted decoder residual",
        normalization="model channel equals per-edge px residual divided by10; no clipping",
        invalid_policy="separate rows/counts; excluded from residual distribution, never imputed as zero",
        source_sha_check="cache protocol fixed Matcher SHA plus small protocol/pair metadata hashes; no checkpoint/features read")
    save(output / "protocol.json", protocol)
    try:
        summary, bindings = {}, {}
        with (output / "rows.jsonl").open("x") as stream:
            for split in COUNTS:
                arrays, records, bindings[split] = load_cache(cache_root / split, split)
                rows = []
                for start in range(0, len(records), CHUNK):
                    for row in chunk_rows(arrays, records, split, start, min(start + CHUNK, len(records))):
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        rows.append(row)
                stream.flush()
                summary[split] = summarize(rows)
                del arrays
        protocol.update(status="complete", bindings=bindings, elapsed_s=time.monotonic() - started)
        save(output / "summary.json", dict(status="complete", splits=summary, elapsed_s=protocol["elapsed_s"]))
        save(output / "protocol.json", protocol)
        print(json.dumps(dict(status="complete", counts=COUNTS, elapsed_s=protocol["elapsed_s"])))
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), elapsed_s=time.monotonic() - started)
        save(output / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    run(cli.cache_root, cli.output)
