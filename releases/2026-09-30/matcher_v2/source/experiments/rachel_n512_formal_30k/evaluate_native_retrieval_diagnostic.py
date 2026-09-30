"""REAL known-positive occurrence-edge retrieval, not complete-qrels accuracy.

All 938 original prepared occurrences remain in the gallery (no alias merging).
Weights/VAL operating points freeze before REAL inputs. The complete score
matrix is persisted before pair labels are accessed for reporting. Full fused
scores are directed: 938*937 pair evaluations, without symmetry assumptions or layout
decoding. Pairing stage2 and Shredding coarse use full-gallery cosine matrices.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from .train_pairing_native_retrieval import (
    NativeRetrievalHead, load_official_components, load_torch, save_json, sha256,
    verify_dependencies, SCHEMA as STAGE2_SCHEMA)

SCHEMA = "rachel-real-known-positive-occurrence-retrieval/1"
KS = (1, 5, 10, 20, 50)
PREPARED_SHA = "5e9fc2455ac8af5f15ff810b90722e6ac0e119cd3dab3c82a5b228a14931b2af"


def save_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze_input_population(prepared_root, expected_count=938):
    """Select only input metadata; do not access the manifest's pairs/labels."""
    root = Path(prepared_root).resolve(strict=True)
    manifest_path, array_path = root / "manifest.json", root / "inputs.npz"
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("schema") != "real-layout-prepared-v1":
        raise ValueError("requires original real-layout prepared inputs")
    ids = metadata["fragment_ids"]
    if (len(ids) != expected_count or len(set(ids)) != expected_count
            or any(not isinstance(x, str) or not x for x in ids)):
        raise ValueError("fragment occurrence count/identity differs")
    inputs = dict(fragment_ids=list(ids), manifest_sha256=metadata["manifest_sha256"])
    del metadata  # Pair labels are neither inspected nor retained here.
    manifest_sha = sha256(manifest_path)
    if expected_count == 938 and manifest_sha != PREPARED_SHA:
        raise ValueError("requires unchanged original938 prepared manifest")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("packed_masks", "points", "valid")}
    if (arrays["packed_masks"].shape != (expected_count, 800, 100)
            or arrays["points"].shape != (expected_count, 512, 2)
            or arrays["valid"].shape != (expected_count, 512)
            or not np.isfinite(arrays["points"]).all()
            or not arrays["valid"].any(axis=1).all()):
        raise ValueError("prepared model input shapes/values differ")
    evidence = dict(prepared_root=str(root), prepared_manifest_sha256=manifest_sha,
                    prepared_inputs_sha256=sha256(array_path), occurrence_count=expected_count,
                    fragment_order_sha256=_json_hash(ids),
                    gallery_policy="original manifest order; retain every occurrence; exclude self only",
                    pair_labels_used_to_construct_gallery=False)
    return inputs, arrays, evidence


def _json_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def directed_indices(start, stop, n):
    if n < 2 or not 0 <= start <= stop <= n * (n - 1):
        raise ValueError("invalid directed nonself interval")
    flat = np.arange(start, stop, dtype=np.int64)
    first, offset = flat // (n - 1), flat % (n - 1)
    return first, offset + (offset >= first)


def side_tensors(arrays, indices, device):
    mask = np.unpackbits(arrays["packed_masks"][indices], axis=2).astype(np.float32)
    return (torch.from_numpy(mask[:, None]).to(device),
            torch.as_tensor(np.array(arrays["points"][indices], copy=True), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(arrays["valid"][indices], copy=True), dtype=torch.bool, device=device))


def pair_tensors(arrays, first, second, device):
    a, b = side_tensors(arrays, first, device), side_tensors(arrays, second, device)
    return a[0], b[0], a[1], b[1], a[2], b[2]


def score_full_chunks(model, arrays, device, output, identity, batch_size=8, chunk_pairs=1024, resume=False,
                      progress_callback=None):
    """Atomic immutable score chunks; completed chunks are hash-checked on resume."""
    if batch_size < 1 or chunk_pairs < 1:
        raise ValueError("batch/chunk sizes must be positive")
    device, root = torch.device(device), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    n = len(arrays["points"])
    if n < 2:
        raise ValueError("full scoring requires at least two occurrences")
    total = n * (n - 1)
    binding = dict(identity=identity, n=n, batch_size=batch_size, chunk_pairs=chunk_pairs,
                   directed_order="row-major nonself", device=str(device))
    state_path = root / "chunks.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("binding") != binding:
            raise ValueError("full-score resume identity differs")
    else:
        if state_path.exists():
            raise ValueError("full score chunks exist; use resume")
        state = dict(binding=binding, chunks=[], completed_pairs=0)
        save_json(state_path, state)
    matrix = np.full((n, n), -np.inf, np.float32)
    cursor = 0
    for row in state["chunks"]:
        if row["start"] != cursor or not cursor < row["stop"] <= total:
            raise ValueError("score chunk sequence differs")
        path = root / row["path"]
        if path.parent != root or sha256(path) != row["sha256"]:
            raise ValueError("score chunk content changed")
        values = np.load(path, allow_pickle=False)
        if values.shape != (row["stop"] - cursor,) or not np.isfinite(values).all():
            raise ValueError("invalid frozen score chunk")
        ia, ib = directed_indices(cursor, row["stop"], n)
        matrix[ia, ib] = values
        cursor = row["stop"]
    if cursor != state["completed_pairs"]:
        raise ValueError("score chunk cursor differs")
    model.eval().requires_grad_(False)
    started = time.monotonic()
    with torch.inference_mode():
        while cursor < total:
            stop = min(total, cursor + chunk_pairs)
            ia, ib = directed_indices(cursor, stop, n)
            parts = []
            for offset in range(0, len(ia), batch_size):
                tensors = pair_tensors(arrays, ia[offset:offset + batch_size], ib[offset:offset + batch_size], device)
                scores = model(*tensors).fused_probability.detach().float().cpu().numpy()
                if (scores.shape != (min(batch_size, len(ia) - offset),) or not np.isfinite(scores).all()
                        or np.any((scores < 0) | (scores > 1))):
                    raise ValueError("invalid full fused score output")
                parts.append(scores)
            values = np.concatenate(parts)
            name = "%09d_%09d.npy" % (cursor, stop)
            save_array(root / name, values)
            state["chunks"].append(dict(start=cursor, stop=stop, path=name, sha256=sha256(root / name)))
            state["completed_pairs"] = stop
            save_json(state_path, state)
            matrix[ia, ib] = values
            cursor = stop
            progress = dict(status="running", phase="full_directed_scores", completed_pairs=cursor,
                            total_pairs=total, elapsed_s=time.monotonic() - started)
            save_json(root / "status.json", progress)
            if progress_callback is not None:
                progress_callback(progress)
            print(json.dumps(progress), flush=True)
    return matrix


def score_matrix_metrics(scores, positive_pairs, ks=KS):
    scores = np.asarray(scores, dtype=np.float32)
    pairs = np.asarray(positive_pairs, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("requires square gallery score matrix")
    n = len(scores)
    if not np.isfinite(scores[~np.eye(n, dtype=bool)]).all():
        raise ValueError("nonself scores must all be finite")
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0
            or np.any(pairs < 0) or np.any(pairs >= n) or np.any(pairs[:, 0] == pairs[:, 1])):
        raise ValueError("invalid positive edge endpoints")
    if not ks or any(int(k) != k or k <= 0 for k in ks):
        raise ValueError("K must be a positive integer")
    edges = sorted({tuple(sorted(row)) for row in pairs.tolist()})
    values = scores.copy()
    np.fill_diagonal(values, -np.inf)
    ranked = np.argsort(-values, axis=1, kind="stable")[:, :min(max(ks), n - 1)]
    hits, union_hits = {}, {}
    for k in ks:
        chosen = [set(row[:min(k, n - 1)].tolist()) for row in ranked]
        count, union = 0, 0
        for first, second in edges:
            a, b = second in chosen[first], first in chosen[second]
            count += int(a) + int(b)
            union += int(a or b)
        hits[str(k)], union_hits[str(k)] = count, union
    denominator = 2 * len(edges)
    return dict(**{"recall_at_%d" % k: hits[str(k)] / denominator for k in ks},
                **{"undirected_union_recall_at_%d" % k: union_hits[str(k)] / len(edges) for k in ks},
                hits=hits, undirected_union_hits=union_hits, positive_directed_edges=denominator,
                unique_undirected_positive_edges=len(edges), gallery_size=n,
                denominator="known positive undirected edges expanded in both directions",
                title="REAL known-positive occurrence-edge Recall@K diagnostic",
                all_gallery_qrels_complete=False, unknown_cross_case_pairs="unjudged",
                self_excluded=True, tie_break="frozen occurrence order", alias_merging=False)


def _load_models(args):
    """All checkpoint/threshold loading happens before REAL input loading."""
    device = torch.device(args.device)
    engine = dict(method=args.method, device=device)
    thresholds = {}
    if args.method == "full_fused":
        from .evaluate_recall_data_volume import load_winner
        if not args.checkpoint or not args.training_freeze:
            raise ValueError("full_fused requires checkpoint and training-freeze")
        path, freeze_path = Path(args.checkpoint).resolve(), Path(args.training_freeze).resolve()
        if path.name != "winner.pt" or freeze_path != path.parent / "train_val_freeze.json":
            raise ValueError("requires the completed Full winner's own training freeze")
        model, identity, _ = load_winner(path.parent)
        freeze = json.loads(freeze_path.read_text())
        thresholds = dict(freeze["operating_points"]["thresholds"])
        engine["model"] = model.to(device).eval().requires_grad_(False)
        identity.update(training_freeze_sha256=sha256(freeze_path), score_family="directed full fused probability")
    elif args.method == "pairing_stage2":
        from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairing
        if not args.stage2_checkpoint or not args.stage1_checkpoint or not args.official_source:
            raise ValueError("Pairing stage2 requires both checkpoints and official-source")
        checkpoint_path = Path(args.stage2_checkpoint).resolve()
        stage1_path = Path(args.stage1_checkpoint).resolve()
        checkpoint = load_torch(checkpoint_path)
        completion = json.loads((checkpoint_path.parent / "completion.json").read_text())
        if (checkpoint.get("schema_version") != STAGE2_SCHEMA or completion.get("status") != "complete"
                or completion.get("best_checkpoint_sha256") != sha256(checkpoint_path)
                or completion.get("identity") != checkpoint.get("identity")
                or completion.get("best_epoch") != checkpoint.get("epoch")
                or completion.get("test_or_real_used") is not False
                or checkpoint["identity"].get("driver_sha256")
                != sha256(load_official_components.__code__.co_filename)
                or checkpoint["identity"]["stage1_checkpoint_sha256"] != sha256(stage1_path)):
            raise ValueError("Pairing stage2/stage1 completion binding differs")
        pairing.audit_official_source(Path(args.official_source))
        verify_dependencies()
        for key, relative in (("official_pipeline_sha256", "PairingNet Code/utils/pipeline.py"),
                               ("official_loss_sha256", "PairingNet Code/utils/infornce_loss.py")):
            if checkpoint["identity"][key] != sha256(Path(args.official_source) / relative):
                raise ValueError("stage2 official source differs from training")
        Native, _ = load_official_components(args.official_source)
        head = NativeRetrievalHead(Native(SimpleNamespace(**checkpoint["model_config"])))
        head.load_state_dict(checkpoint["model_state_dict"], strict=True)
        stage1 = pairing.load_frozen_pairingnet_checkpoint(stage1_path, device)
        engine.update(model=stage1.eval().requires_grad_(False), head=head.to(device).eval().requires_grad_(False))
        engine["binary_precision"] = load_torch(stage1_path)["run_config"]["precision"]
        thresholds["stage1_val_cluster_f1"] = pairing.load_frozen_validation_threshold(
            stage1_path.parent / "validation_threshold.json", stage1_path)
        identity = dict(stage2_checkpoint_sha256=sha256(checkpoint_path),
                        checkpoint_sha256_by_stage={"winner": sha256(stage1_path)},
                        stage2_training_identity=checkpoint["identity"],
                        score_family="native-stage2-adapted 128D cosine", feature_precision="fp32",
                        binary_precision=engine["binary_precision"],
                        binary_score_family="separate stage1 nonofficial MLP probability")
    else:
        from .evaluate_recall_benchmarks import load_experimental_benchmark
        if not args.freeze or Path(args.freeze).name != "train_val_freeze.json":
            raise ValueError("Shredding coarse requires train_val_freeze.json")
        benchmark = load_experimental_benchmark("shreddingnet", Path(args.freeze).parent, device)
        engine["predictor"] = benchmark._predictor
        identity = dict(benchmark.identity)
        identity.update(all_pairs_decoded=False, score_family="released coarse flattened-feature cosine")
        thresholds["val_cluster_f1"] = float(benchmark._predictor.pair_threshold)
    if args.binary_freeze:
        frozen = json.loads(Path(args.binary_freeze).read_text())
        if (args.method == "full_fused" or frozen.get("status") != "complete_validation_frozen"
                or frozen.get("source_split") != "val" or frozen.get("sample_count") != 3000
                or frozen.get("test_or_real_used_for_fit") is not False
                or frozen.get("model_identity", {}).get("checkpoint_sha256_by_stage")
                != identity["checkpoint_sha256_by_stage"]):
            raise ValueError("binary operating-point freeze differs from these weights/VAL")
        thresholds = dict(frozen["thresholds"])
        identity["binary_freeze_sha256"] = sha256(args.binary_freeze)
    if not thresholds or not all(np.isfinite(v) and 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("invalid frozen binary thresholds")
    identity.update(method=args.method, thresholds=thresholds, device=str(device),
                    test_or_real_used_for_fit=False, geometry_decode_performed=False)
    return engine, identity


def encode_occurrences(engine, arrays, batch_size):
    values = []
    with torch.inference_mode():
        for start in range(0, len(arrays["points"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays["points"])))
            mask, points, valid = side_tensors(arrays, indices, engine["device"])
            prefix = torch.arange(valid.shape[1], device=valid.device)[None] < valid.sum(1)[:, None]
            if not torch.equal(valid, prefix):
                raise ValueError("original prepared REAL must retain prefix ordered contours")
            if engine["method"] == "pairing_stage2":
                _, contour, context = engine["model"]._encode(mask, points, valid)
                output = engine["head"](torch.cat((contour, context), dim=-1), valid)
            else:
                predictor = engine["predictor"]
                with torch.autocast(device_type=engine["device"].type, dtype=torch.float16,
                                    enabled=engine["device"].type == "cuda" and predictor.runtime.amp):
                    output = predictor.coarse(mask, points, valid)
            values.append(output.float().cpu())
    embeddings = torch.cat(values)
    if not torch.isfinite(embeddings).all():
        raise ValueError("non-finite occurrence embeddings")
    return embeddings


def score_binary_pairs(engine, arrays, pairs, batch_size):
    result = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            part = np.asarray(pairs[start:start + batch_size], np.int64)
            inputs = pair_tensors(arrays, part[:, 0], part[:, 1], engine["device"])
            if engine["method"] == "pairing_stage2":
                with torch.autocast(device_type=engine["device"].type, dtype=torch.bfloat16,
                                    enabled=engine["binary_precision"] == "bf16"):
                    scores = engine["model"](*inputs).pair_probability
            else:
                predictor = engine["predictor"]
                with torch.autocast(device_type=engine["device"].type, dtype=torch.float16,
                                    enabled=engine["device"].type == "cuda" and predictor.runtime.amp):
                    scores, _, _ = predictor.classify(*inputs, predictor.recipe.correspondence_threshold)
            result.extend(scores.detach().float().cpu().tolist())
    result = np.asarray(result, np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite frozen binary scores")
    return result


def attach_known_targets(prepared_root, ids):
    """Call only after the all-score prediction completion receipt is persisted."""
    metadata = json.loads((Path(prepared_root) / "manifest.json").read_text())
    rows = metadata["pairs"]
    if (len(rows) != 1016 or len({x["pair_id"] for x in rows}) != 1016
            or sum(x["label"] is True for x in rows) != 508
            or sum(x["strict"] is True for x in rows) != 547):
        raise ValueError("REAL known target population differs")
    lookup = {key: index for index, key in enumerate(ids)}
    pairs = np.asarray([[lookup[x["fragment_a_id"]], lookup[x["fragment_b_id"]]] for x in rows], np.int64)
    labels = np.asarray([x["label"] for x in rows], bool)
    strict = np.asarray([x["strict"] for x in rows], bool)
    positives = pairs[labels]
    if (len({tuple(sorted(row)) for row in positives.tolist()}) != 508
            or set(positives.reshape(-1).tolist()) != set(range(938))
            or int((~labels & strict).sum()) != 39):
        raise ValueError("known positive edge/strict denominator differs")
    return metadata, pairs, labels, strict


def duplicate_report(metadata, ids, source_manifest=None):
    path = Path(source_manifest or metadata.get("source", {}).get("real_manifest", ""))
    if not path.is_file():
        return dict(available=False, reason="original source-manifest metadata unavailable; no aliases inferred")
    source = json.loads(path.read_text())
    if source.get("manifest_sha256") != metadata["manifest_sha256"]:
        raise ValueError("duplicate provenance manifest differs")
    by_id = {}
    for case in source["cases"]:
        for fragment in case["fragments"]:
            by_id[case["case_uid"] + "/fragment/" + str(fragment["fragment_id"])] = fragment
    rows = [by_id[key] for key in ids]
    result = dict(available=True, source_manifest_sha256=sha256(path), occurrence_count=len(ids),
                  physical_entities_determined=False, aliases_merged=False,
                  limitation="same content/alpha hashes are asset duplicates, not complete physical identity or qrels")
    for field, label in (("content_sha256", "content"), ("alpha_mask_sha256", "alpha")):
        counts = Counter(row[field] for row in rows)
        result[label + "_unique_hashes"] = len(counts)
        result[label + "_duplicate_groups"] = sum(value > 1 for value in counts.values())
    return result


def run(args):
    if args.batch_size < 1 or args.chunk_pairs < 1:
        raise ValueError("batch/chunk sizes must be positive")
    if args.positive_cnn_gate and args.method != "shredding_coarse":
        raise ValueError("positive-cnn-gate only applies to Shredding coarse")
    torch.set_num_threads(1)
    engine, model_identity = _load_models(args)
    inputs, arrays, evidence = freeze_input_population(args.prepared_root)
    identity = dict(schema_version=SCHEMA, driver_sha256=sha256(__file__), model=model_identity,
                    input_evidence=evidence, batch_size=args.batch_size, chunk_pairs=args.chunk_pairs,
                    positive_cnn_gate=args.positive_cnn_gate, ks=list(KS))
    root = Path(args.output).resolve()
    if args.resume:
        if json.loads((root / "protocol.json").read_text())["identity"] != identity:
            raise ValueError("all-gallery resume identity differs")
    else:
        root.mkdir(parents=True, exist_ok=False)
        save_json(root / "protocol.json", dict(identity=identity,
                  planned_directed_nonself_pairs=938 * 937,
                  ranking_scope="all938 occurrences; no preselected-pair-list TopK",
                  original_pairs_labels_accessed_for_scoring=False,
                  metadata_contains_labels_but_only_fragment_ids_used_until_matrix_complete=True))
    started = time.monotonic()
    matrix_path, completion_path = root / "scores.npy", root / "prediction_complete.json"
    try:
        if args.resume and completion_path.exists():
            complete = json.loads(completion_path.read_text())
            if complete["identity"] != identity or complete["score_matrix_sha256"] != sha256(matrix_path):
                raise ValueError("frozen all-gallery matrix changed")
            scores = np.load(matrix_path, allow_pickle=False)
        else:
            save_json(root / "status.json", dict(status="running", phase="all_gallery_prediction", pid=os.getpid()))
            if args.method == "full_fused":
                scores = score_full_chunks(engine["model"], arrays, engine["device"], root / "score_chunks",
                                           identity, args.batch_size, args.chunk_pairs, args.resume,
                                           progress_callback=lambda row: save_json(root / "status.json", row))
            else:
                embeddings = encode_occurrences(engine, arrays, args.batch_size)
                save_array(root / "embeddings.npy", embeddings.numpy())
                normalized = F.normalize(embeddings, dim=1)
                scores = (normalized @ normalized.T).numpy()
                np.fill_diagonal(scores, -np.inf)
            save_array(matrix_path, scores)
            save_json(completion_path, dict(status="all938_by938_scores_frozen", identity=identity,
                      score_matrix_sha256=sha256(matrix_path), pair_labels_attached=False,
                      nonself_directed_scores=938 * 937, elapsed_s=time.monotonic() - started))
        # This is the first access to metadata['pairs'] or its labels.
        if sha256(Path(args.prepared_root) / "manifest.json") != evidence["prepared_manifest_sha256"]:
            raise ValueError("prepared manifest changed before target attachment")
        metadata, pairs, labels, strict = attach_known_targets(args.prepared_root, inputs["fragment_ids"])
        summary = score_matrix_metrics(scores, pairs[labels])
        summary.update(method=args.method, status="complete", identity=identity,
                       duplicates=duplicate_report(metadata, inputs["fragment_ids"], args.source_manifest),
                       full_gallery_precision_f1_ap_ndcg_reported=False,
                       target_policy="508 known positive edges; cross-case unknowns not GT negatives")
        binary_scores = (scores[pairs[:, 0], pairs[:, 1]] if args.method == "full_fused" else
                         score_binary_pairs(engine, arrays, pairs, args.batch_size))
        save_array(root / "selected_pair_binary_scores.npy", binary_scores)
        from .run_layout_decoder_experiment import classification
        summary["fixed_val_threshold_pairlist_comparison"] = dict(
            scope="original balanced1016, not full gallery;469 negatives are constructed distractors",
            score_family=model_identity.get("binary_score_family", "CNN pair probability" if args.method == "shredding_coarse" else "fused probability"),
            balanced1016={name: classification(labels, binary_scores, threshold)
                          for name, threshold in model_identity["thresholds"].items()},
            strict547={name: classification(labels[strict], binary_scores[strict], threshold)
                       for name, threshold in model_identity["thresholds"].items()})
        if args.positive_cnn_gate:
            if args.method != "shredding_coarse":
                raise ValueError("positive-cnn-gate only applies to Shredding coarse")
            positive = pairs[labels]
            directed = np.concatenate((positive, positive[:, ::-1]))
            gate_scores = score_binary_pairs(engine, arrays, directed, args.batch_size)
            ranked = np.argsort(-scores, axis=1, kind="stable")[:, :max(KS)]
            gated = {}
            gates = dict(model_identity["thresholds"], native_strict_gt_05=.5)
            for name, threshold in gates.items():
                accepted = gate_scores > threshold if name == "native_strict_gt_05" else gate_scores >= threshold
                gated[name] = {"recall_at_%d" % k: float(np.mean([
                    bool(ok) and int(b) in ranked[int(a), :k] for (a, b), ok in zip(directed, accepted)])) for k in KS}
            save_array(root / "known_positive_directed_cnn_scores.npy", gate_scores)
            summary["known_positive_directed_topk_and_cnn_gate"] = dict(
                denominator=1016, thresholds=gates, metrics=gated,
                limitation="conditional known-positive survival only; not full-pool FM precision or GA")
        save_json(root / "summary.json", summary)
        save_json(root / "receipt.json", dict(status="complete", identity=identity,
                  score_matrix_sha256=sha256(matrix_path), summary_sha256=sha256(root / "summary.json"),
                  elapsed_s=time.monotonic() - started, trained_or_calibrated_on_real=False,
                  pair_labels_attached_after_score_matrix_freeze=True))
        save_json(root / "status.json", dict(status="complete", method=args.method,
                  recall_at_10=summary["recall_at_10"], positive_directed_edges=1016))
    except Exception as error:
        save_json(root / "status.json", dict(status="failed", error=repr(error), pid=os.getpid()))
        raise
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", choices=("pairing_stage2", "shredding_coarse", "full_fused"), required=True)
    value.add_argument("--prepared-root", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=8)
    value.add_argument("--chunk-pairs", type=int, default=1024)
    value.add_argument("--stage2-checkpoint")
    value.add_argument("--stage1-checkpoint")
    value.add_argument("--official-source")
    value.add_argument("--freeze", help="ShreddingNet train_val_freeze.json")
    value.add_argument("--checkpoint", help="Full winner.pt")
    value.add_argument("--training-freeze", help="Full train_val_freeze.json")
    value.add_argument("--binary-freeze", help="optional already-frozen benchmark VAL operating points")
    value.add_argument("--source-manifest", help="optional original REAL metadata for asset-duplicate audit")
    value.add_argument("--positive-cnn-gate", action="store_true")
    value.add_argument("--resume", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
