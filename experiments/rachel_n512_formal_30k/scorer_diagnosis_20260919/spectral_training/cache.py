"""Offline CPU-only fixed-Matcher summary cache; no training or GPU use.

Store ten scalars and full six-input hashes, never context tokens or full Q.
Single-pair fixed-cap512 collate matches training padding exactly. Multiprocess
preparation is bounded to 2*workers pending pairs; default one worker.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train as shared
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_head import features as f

SCHEMA = "rachel-spectral-training-cache-bundle/1"
SPLIT_COUNTS = {"train": 24000, "val": 3000, "test": 3000, "real": 1016, "ood": 301}
FIELDS = f.INPUT_FIELDS
_BASE = None
_THREAD_LIMIT = None


def cpu_inputs(batch):
    """Only input tensors; deliberately never accesses labels or GT fields."""
    arrays = {}
    for name in FIELDS:
        value = getattr(batch, name)
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError("input binding must happen before GPU transfer")
            value = value.detach().numpy()
        arrays[name] = np.asarray(value, dtype=np.bool_ if name.startswith("contour_valid") else np.float32)
    return arrays


def input_hashes(batch):
    arrays = cpu_inputs(batch)
    return [f.model_input_sha256({name: value[i] for name, value in arrays.items()})
            for i in range(len(batch.pair_ids))]


def matcher_code_identity():
    root = Path(shared.old.__file__).resolve().parents[2]
    paths = sorted((root / "staging/pairwise_v0_2/models").glob("*.py"))
    return {str(p.relative_to(root)): f.file_sha256(p) for p in paths}


def sampling_identity(config):
    return dict(model_config=asdict(config), sampling="original512", fixed_cap=512,
        input_fields=list(FIELDS), mask_and_point_dtype="float32", validity_dtype="bool",
        padding="exact original fixed-cap collate bytes", online_augmentation=False)


def initialize_worker(source_path):
    global _BASE, _THREAD_LIMIT
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Optional installed runtime helper; no dependency download or GPU access.
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMIT = threadpool_limits(limits=1)
    except ImportError:
        # If NumPy is already imported these env changes alone cannot reliably
        # reconfigure its BLAS. Refuse silent oversubscription instead.
        if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
            raise RuntimeError("threadpoolctl unavailable: start CPU cache command with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1")
        _THREAD_LIMIT = None
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    _BASE = shared.old.load_decoupled_checkpoint(checkpoint).base_model.eval().requires_grad_(False)


def compute_pair(item):
    """Initializer owns a frozen CPU base. Return small result, never matrix."""
    ordinal, pair_id, arrays = item
    if _BASE is None:
        raise RuntimeError("CPU worker not initialized")
    started = time.perf_counter()
    tensors = [torch.as_tensor(arrays[name])[None] for name in FIELDS]
    with torch.inference_mode():
        output = _BASE(*tensors)
    matcher_s = time.perf_counter() - started
    assignment = output.assignment[0].cpu().numpy()
    started_svd = time.perf_counter()
    row = f.make_record(pair_id, f.model_input_sha256(arrays), assignment,
        arrays["contour_valid_a"], arrays["contour_valid_b"])
    return ordinal, row, dict(matcher_s=matcher_s, summary_s=time.perf_counter() - started_svd,
        worker_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_unit="bytes" if platform.system() == "Darwin" else "KiB")


def pair_items(batches):
    ordinal = 0
    for wrapped in batches:
        batch = getattr(wrapped, "batch", wrapped)
        arrays = cpu_inputs(batch)
        for i, pair_id in enumerate(batch.pair_ids):
            yield ordinal, str(pair_id), {name: value[i] for name, value in arrays.items()}
            ordinal += 1


def bounded_compute(items, workers, source_path):
    if workers == 1:
        initialize_worker(source_path)
        for item in items:
            yield compute_pair(item)
        return
    # Bounded queue prevents eager map/pickle of 24K ~5MB image pairs.
    iterator = iter(items)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker, initargs=(source_path,)) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 2 * workers and not exhausted:
                try:
                    pending.add(pool.submit(compute_pair, next(iterator)))
                except StopIteration:
                    exhausted = True
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()


def populations(source):
    origin = source["resume_identity"]
    args = SimpleNamespace(sampling="original512",
        train_materialized_manifest=origin["populations"]["train"]["manifest"],
        dataset=str(Path(origin["populations"]["val"]["manifest"]).parents[1]))
    return shared.old.make_populations(args)


def heldout_batches(split, source):
    """Same original full held-out inputs; feature worker sees six inputs only."""
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as ev
    core = ev.core
    if split == "test":
        root = Path(source["resume_identity"]["populations"]["val"]["manifest"]).parents[1]
        dataset = core.RachelPairDataset(root, "test")
        path = root / "pairs/test.jsonl"
        batches = core.make_ablation_loader(dataset, list(range(3000)), batch_size=1,
            num_workers=0, seed=shared.old.SEED, contour_cap=512)
    else:
        root = Path("/root/autodl-tmp/rachel_layout_v2_20260906_001/real_preparation/prepared" if split == "real"
                    else "/root/autodl-tmp/turufan_ood_pairwise_20260912_001/prepared")
        path = root / "manifest.json"
        if split == "real":
            metadata, arrays = core.load_prepared_cache(root)
        else:
            metadata = json.loads(path.read_text())
            with np.load(root / "inputs.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ("packed_masks", "points", "valid")}
        if len(metadata["pairs"]) != SPLIT_COUNTS[split]:
            raise ValueError("held-out cache must retain unchanged complete population")
        batches = core.real.input_batches(metadata, arrays, 1)
    return batches, dict(manifest=str(path), manifest_sha256=f.file_sha256(path))


def prepare_split(root, split, batches, population, source, *, workers=1, source_path=shared.SOURCE,
                  expected_count=None):
    started = time.perf_counter()
    values = list(bounded_compute(pair_items(batches), workers, str(source_path)))
    values.sort(key=lambda x: x[0])
    if not values or [v[0] for v in values] != list(range(len(values))):
        raise ValueError("cache input order incomplete or repeated")
    rows = [v[1] for v in values]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("cache population incomplete")
    config = shared.old.load_decoupled_checkpoint(source).config
    code = matcher_code_identity()
    sampling = sampling_identity(config)
    identity = dict(split=split, fixed_physical_inputs=True,
        source_checkpoint_sha256=shared.SOURCE_SHA,
        matcher_state_sha256=source["matcher_pretraining_receipt"]["base_state_sha256"],
        matcher_code_sha256=f.digest_json(code), input_manifest_sha256=population["manifest_sha256"],
        prepared_inputs_sha256=f.digest_json([{ "pair_id": r.pair_id, "input_sha256": r.input_sha256} for r in rows]),
        sampling_protocol_sha256=f.digest_json(sampling), precompute_device="cpu",
        full_svd_dtype="float64", raw_assignment_dtype="float32", online_augmentation=False)
    path = root / (split + ".json")
    digest = f.write_cache(path, identity, rows)
    receipt = dict(path=str(path), sha256=digest, identity=identity, pair_count=len(rows),
        ordered_pair_ids_sha256=f.digest_json([r.pair_id for r in rows]),
        population=population, matcher_code=code, sampling_protocol=sampling,
        elapsed_wall_s=time.perf_counter() - started, workers=workers,
        sum_worker_matcher_s=sum(v[2]["matcher_s"] for v in values),
        sum_worker_summary_s=sum(v[2]["summary_s"] for v in values),
        maximum_worker_peak_rss=max(v[2]["worker_peak_rss"] for v in values), rss_unit=values[0][2]["rss_unit"])
    return receipt


def load_bundle(path, digest, *, require_training=True):
    path = Path(path).resolve(strict=True)
    if f.file_sha256(path) != digest:
        raise ValueError("bundle differs from externally registered SHA")
    bundle = json.loads(path.read_text())
    if (bundle.get("schema_version") != SCHEMA or bundle.get("status") != "complete"
            or bundle.get("source_checkpoint_sha256") != shared.SOURCE_SHA
            or bundle.get("matcher_state_sha256") != shared.MATCHER_SHA):
        raise ValueError("bundle incomplete or wrong source")
    required = {"train", "val"} if require_training else set(bundle["splits"])
    if not required <= set(bundle["splits"]):
        raise ValueError("both TRAIN and separate clean VAL caches required")
    caches = {}
    for split, record in bundle["splits"].items():
        identity = record["identity"]
        if (identity["split"] != split or identity.get("source_checkpoint_sha256") != shared.SOURCE_SHA
                or identity["matcher_state_sha256"] != shared.MATCHER_SHA
                or identity["matcher_code_sha256"] != f.digest_json(matcher_code_identity())):
            raise ValueError("cache source/model code/split mismatch")
        cache = f.load_cache(record["path"], identity, record["sha256"])
        if len(cache.records) != record["pair_count"]:
            raise ValueError("cache count mismatch")
        if identity["prepared_inputs_sha256"] != f.digest_json([
                dict(pair_id=r.pair_id, input_sha256=r.input_sha256) for r in cache.records]):
            raise ValueError("cache complete physical-input aggregate mismatch")
        caches[split] = cache
    if require_training:
        if caches["train"].identity["input_manifest_sha256"] == caches["val"].identity["input_manifest_sha256"]:
            raise ValueError("TRAIN and VAL must be separate source manifests")
        stats = f.fit_train_statistics(caches["train"])
        if f.digest_json(stats) != f.digest_json(bundle.get("normalizer")):
            raise ValueError("normalizer not the common full TRAIN-only 10D fit")
    return bundle, caches


def main(args):
    if not 1 <= args.workers <= 8:
        raise ValueError("CPU workers must be1..8; default1, benchmark before increasing")
    torch.set_num_threads(1)
    root = Path(args.output).resolve()
    source_path = Path(args.source).resolve(strict=True)
    source = shared.read_source(source_path)
    if root == source_path.parent or source_path.parent in root.parents:
        raise ValueError("cache may not write inside original source")
    root.mkdir(parents=True, exist_ok=False)
    if args.probe and args.split != "train_val":
        raise ValueError("timing probe is TRAIN-only")
    training, validation, cap, records = populations(source)
    selected = ("train",) if args.probe else ("train", "val") if args.split == "train_val" else (args.split,)
    result = dict(schema_version=SCHEMA, status="preparing", source_checkpoint_sha256=shared.SOURCE_SHA,
        matcher_state_sha256=shared.MATCHER_SHA, splits={}, cpu_only=True, probe_count=args.probe)
    for split in selected:
        count = args.probe or SPLIT_COUNTS[split]
        if split in ("train", "val"):
            dataset = training if split == "train" else validation
            factory = shared.old.make_weathering_loader if split == "train" else shared.old.make_ablation_loader
            batches = factory(dataset, list(range(count)), batch_size=1, num_workers=0,
                seed=shared.old.SEED, contour_cap=cap)
            record = records[split]
        else:
            batches, record = heldout_batches(split, source)
        result["splits"][split] = prepare_split(root, split, batches, record, source,
            workers=args.workers, source_path=source_path, expected_count=count)
    if args.probe:
        result.update(status="probe_only", trainable=False, normalizer=None)
    elif args.split == "train_val":
        record = result["splits"]["train"]
        cache = f.load_cache(record["path"], record["identity"], record["sha256"])
        result.update(status="complete", trainable=True, normalizer=f.fit_train_statistics(cache))
    else:
        result.update(status="complete", trainable=False, normalizer=None,
            held_out_summary_only=True, normalization_fitted=False)
    path = root / "bundle.json"
    shared.old.save_json(path, result)
    print(json.dumps(dict(path=str(path), sha256=f.file_sha256(path), status=result["status"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=str(shared.SOURCE))
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--split", choices=("train_val", "test", "real", "ood"), default="train_val")
    p.add_argument("--probe", type=int, choices=(24, 32), help="TRAIN prefix timing only, never trainable")
    main(p.parse_args())
