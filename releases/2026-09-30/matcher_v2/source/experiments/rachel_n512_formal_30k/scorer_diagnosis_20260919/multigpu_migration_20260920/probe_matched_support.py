"""CPU-only, frozen C16 support interventions on preregistered SIMVAL rows.

No training, layout re-decoding, threshold fitting, or checkpoint writes.
Uniform duplication changes multiplicity, not frozen token content. Edge
subsampling changes support content AND cardinality; it is not count-only.
"""
import os

# Set before importing Torch/project modules, irrespective of caller environment.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    # Remote sealed snapshots expose matched_only directly on PYTHONPATH.
    from matched_only.data import FormalCache, sha
    from matched_only.model import make_fresh_scorer
except ModuleNotFoundError as error:
    if error.name != "matched_only":
        raise
    # Local checkout exposes the original repository namespace instead.
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.data import FormalCache, sha
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.matched_only.model import make_fresh_scorer

SCHEMA = "matched-support-cpu-interventions/1"
SEED = 260920
TOLERANCE = 1e-4
GROUPS = (("positive_65_128", 1, 65, 128), ("positive_129_plus", 1, 129, None),
          ("negative_le32", 0, 0, 32), ("negative_33_64", 0, 33, 64))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def select_rows(cache):
    # Only small Boolean masks/metadata scanned; feature arrays remain memmapped.
    minimum = np.minimum(cache.arrays["mask_a"].sum(axis=1), cache.arrays["mask_b"].sum(axis=1))
    selected, availability = [], {}
    for name, label, lower, upper in GROUPS:
        ids = [i for i, row in enumerate(cache.records) if row["label"] == label
               and minimum[i] >= lower and (upper is None or minimum[i] <= upper)]
        ids.sort(key=lambda i: cache.records[i]["pair_id"])
        availability[name] = dict(eligible=len(ids), selected=min(8, len(ids)), requested=8)
        selected.extend(dict(ordinal=i, pair_id=cache.records[i]["pair_id"], label=label,
                             group=name, unique_endpoints_min=int(minimum[i])) for i in ids[:8])
    return selected, availability


def checkpoint_model(saved):
    identity = saved.get("identity", {})
    if (saved.get("schema") != "s7-m12-fresh-matched-scorer-training/1"
            or identity.get("arm") != "matched_tokens" or identity.get("matcher_frozen") is not True
            or saved.get("head_epoch") != 16 or saved.get("completed_segments") != 64
            or saved.get("absolute_epoch") != 28 or saved.get("classifier_pair_exposures") != 384000
            or saved.get("optimizer_updates") != 24000 or saved.get("phase") != "classifier"
            or saved.get("formal_training_counted") is not True or saved.get("matcher_updated") is not False):
        raise ValueError("requires the completed formal frozen-Matcher matched_tokens C16 checkpoint")
    model = make_fresh_scorer("matched_tokens", seed=identity["head_seed"])
    if model.metadata() != saved.get("model_metadata") or model.metadata() != identity.get("model"):
        raise ValueError("checkpoint scorer architecture/metadata differ from loaded implementation")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


def validate_binding(saved, cache):
    expected = saved["identity"]["cache_bindings"]["val"]
    # FormalCache already checks readiness, fixed population, array headers and
    # pairs.json SHA. Bind the source + committed protocol to the trained head.
    keys = ("split", "pair_count", "protocol_sha256", "pairs_sha256", "source_checkpoint_sha256",
            "population", "cache_implementation_sha256", "precompute_device", "features_dtype")
    for key in keys:
        if key not in expected or expected[key] != cache.binding.get(key):
            raise ValueError("checkpoint VAL cache binding mismatch: " + key)
    if saved["identity"].get("source_checkpoint_sha256") != cache.binding["source_checkpoint_sha256"]:
        raise ValueError("checkpoint Matcher source differs from cache source")
    return dict(validated_keys=list(keys), checkpoint_val_root=expected["root"],
                actual_val_root=str(cache.root), protocol_sha256=cache.binding["protocol_sha256"],
                source_checkpoint_sha256=cache.binding["source_checkpoint_sha256"])


def compact_score(head, a, b):
    logit = head(a, b, torch.ones(a.shape[:2], dtype=torch.bool),
                 torch.ones(b.shape[:2], dtype=torch.bool)).reshape(())
    if not torch.isfinite(logit):
        raise ValueError("nonfinite intervention logit")
    return dict(logit=float(logit), score=float(logit.sigmoid()),
                endpoints_a=a.shape[1], endpoints_b=b.shape[1],
                unique_endpoints_min=min(a.shape[1], b.shape[1]))


def seed_for(pair_id, cap, repeat):
    text = "%d|%s|%d|%d" % (SEED, pair_id, cap, repeat)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


@torch.no_grad()
def probe_case(model, cache, selected):
    batch = cache.batch([selected["ordinal"]], "cpu")
    full = model(*batch.model_args, **batch.model_kwargs)
    a, b, _, _, selection = batch.model_args
    ia = torch.nonzero(selection.mask_a[0], as_tuple=False).flatten()
    ib = torch.nonzero(selection.mask_b[0], as_tuple=False).flatten()
    aa, bb = a[:, ia], b[:, ib]
    baseline = compact_score(model.head, aa, bb)
    baseline.update(full_model_logit=float(full.logit[0]),
                    replay_error=abs(float(full.logit[0]) - baseline["logit"]))
    if not np.isfinite(baseline["full_model_logit"]) or baseline["replay_error"] > TOLERANCE:
        raise ValueError("compact/full scorer replay failed for " + selected["pair_id"])
    active = selection.candidate_valid[0] & selection.candidate_inliers[0] & selection.layout_valid[0]
    slots = torch.nonzero(active, as_tuple=False).flatten().numpy()
    edges = selection.candidate_indices[0]
    interventions = []

    def record(kind, value, **extra):
        value.update(kind=kind, delta_logit=value["logit"]-baseline["logit"],
                     delta_score=value["score"]-baseline["score"], **extra)
        interventions.append(value)

    duplicate = compact_score(model.head, aa.repeat(1, 2, 1), bb.repeat(1, 2, 1))
    # Endpoint counts in duplicate output are token multiplicities, not new
    # unique contour points. Preserve the actual unique counts explicitly.
    duplicate.update(unique_endpoints_a=len(ia), unique_endpoints_b=len(ib),
                     unique_endpoints_min=min(len(ia), len(ib)))
    record("uniform_repeat2", duplicate, factor=2,
           invariant_within_tolerance=abs(duplicate["logit"]-baseline["logit"]) <= TOLERANCE)
    for cap in (16, 32, 64):
        for repeat in range(4):
            seed = seed_for(selected["pair_id"], cap, repeat)
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(slots, size=min(cap, len(slots)), replace=False))
            kept_edges = edges[torch.as_tensor(keep, dtype=torch.long)]
            ja, jb = torch.unique(kept_edges[:, 0], sorted=True), torch.unique(kept_edges[:, 1], sorted=True)
            score = compact_score(model.head, a[:, ja], b[:, jb])
            record("inlier_edge_subsample", score, edge_cap=cap, repeat=repeat, seed=seed,
                   retained_edges=len(keep), retained_candidate_slots=keep.tolist(),
                   endpoint_indices_a=ja.tolist(), endpoint_indices_b=jb.tolist(),
                   subset_is_full=len(keep) == len(slots))
    return dict(**selected, training_valid=bool(batch.training_valid[0]),
                decision_valid=bool(batch.decision_valid[0]), layout_valid=bool(selection.layout_valid[0]),
                used_fallback=bool(full.used_fallback[0]), baseline=baseline,
                endpoint_indices_a=ia.tolist(), endpoint_indices_b=ib.tolist(),
                original_inlier_candidate_slots=slots.tolist(),
                original_inlier_edges=edges[torch.as_tensor(slots)].tolist(),
                fixed_translation_a_to_b_rc=selection.translation_a_to_b_rc[0].tolist(),
                interventions=interventions)


def distribution(values):
    if not values:
        return dict(count=0, mean=None, median=None, minimum=None, maximum=None)
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()))


def summarize(cases):
    groups = {}
    for name, _, _, _ in GROUPS:
        rows = [row for row in cases if row["group"] == name]
        controls = {}
        for control, cap in (("uniform_repeat2", None), ("inlier_edge_subsample", 16),
                             ("inlier_edge_subsample", 32), ("inlier_edge_subsample", 64)):
            values = [v for row in rows for v in row["interventions"]
                      if v["kind"] == control and (cap is None or v["edge_cap"] == cap)]
            controls[control if cap is None else "edge_cap_%d" % cap] = dict(
                delta_logit=distribution([v["delta_logit"] for v in values]),
                delta_score=distribution([v["delta_score"] for v in values]),
                unique_endpoints_min=distribution([v["unique_endpoints_min"] for v in values]))
        groups[name] = dict(cases=len(rows), baseline_logit=distribution([r["baseline"]["logit"] for r in rows]),
                            baseline_score=distribution([r["baseline"]["score"] for r in rows]), controls=controls)
    duplicate = [v for row in cases for v in row["interventions"] if v["kind"] == "uniform_repeat2"]
    return dict(groups=groups, total_cases=len(cases),
                maximum_replay_error=max((r["baseline"]["replay_error"] for r in cases), default=None),
                repeat2_maximum_absolute_logit_delta=max((abs(v["delta_logit"]) for v in duplicate), default=None),
                repeat2_invariance_all_pass=bool(duplicate) and all(v["invariant_within_tolerance"] for v in duplicate))


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema=SCHEMA, status="running", device="cpu", threads=1, cuda_visible_devices="",
        checkpoint=str(Path(args.checkpoint).resolve()), cache=str(Path(args.cache).resolve()),
        seed=SEED, selection="pair_id lexical first8 per predefined label/support bin; no scores inspected",
        groups=[dict(name=n, label=y, minimum=lo, maximum=hi) for n, y, lo, hi in GROUPS],
        controls=dict(uniform_repeat=2, edge_caps=[16, 32, 64], repeats_per_cap=4, tolerance=TOLERANCE,
                      random_seed="SHA256(seed|pair_id|cap|repeat) first8 bytes little-endian"),
        prediction="raw Scorer head logits/sigmoid; no training_valid masking, fitting or classification threshold",
        semantics="labels used only for grouping; no GT input or layout re-decoding; edge subsets retain original (i,j)",
        caveats=["Frozen features retain original full-contour context.",
                 "Uniform repeat changes token multiplicity, not unique endpoints or content.",
                 "Edge subsampling changes content, coverage and diversity as well as cardinality.",
                 "This selected SIMVAL diagnostic is not an unbiased evaluation or model selection."],
        training=False, gradients=False, checkpoint_writes=False, formal_performance_claim=False)
    write_json(output / "protocol.json", protocol)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = checkpoint_model(saved)
        cache = FormalCache(args.cache, "val")
        protocol.update(checkpoint_sha256=sha(args.checkpoint), binding=validate_binding(saved, cache),
                        scorer_metadata=model.metadata())
        selected, availability = select_rows(cache)
        protocol.update(selected=selected, availability=availability, selection_committed_before_scoring=True)
        write_json(output / "protocol.json", protocol)
        cases = []
        with (output / "cases.jsonl").open("x") as stream:
            for row in selected:
                result = probe_case(model, cache, row)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                cases.append(result)
        elapsed = time.perf_counter()-started
        summary = dict(schema=SCHEMA, status="complete", elapsed_s=elapsed,
                       availability=availability, **summarize(cases))
        write_json(output / "summary.json", summary)
        protocol.update(status="complete", completed_cases=len(cases), elapsed_s=elapsed)
        write_json(output / "protocol.json", protocol)
        print(json.dumps(dict(status="complete", output=str(output), cases=len(cases), elapsed_s=elapsed)))
    except BaseException as error:
        protocol.update(status="failed", error=type(error).__name__ + ": " + str(error),
                        elapsed_s=time.perf_counter()-started)
        write_json(output / "protocol.json", protocol)
        write_json(output / "summary.json", dict(schema=SCHEMA, status="failed", error=protocol["error"],
                                                 elapsed_s=protocol["elapsed_s"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
