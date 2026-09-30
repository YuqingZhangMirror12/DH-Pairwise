"""Read-only, finite REAL inputs for frozen matched_edges metadata diagnosis.

No model loading/forward, training, file writes or new threshold fitting.
select_real_cases uses labels/GT ONLY for the predeclared diagnostic groups.
iter_real_inputs supplies ONLY the original six target-blind model inputs.
"""
import hashlib
import json
import math
from pathlib import Path

FIELDS = ("mask_a", "mask_b", "points_rc_a", "points_rc_b", "contour_valid_a", "contour_valid_b")
LAYOUT = "full_top2_mode"


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _endpoint(root, arm):
    path = root/arm/"evaluation"/"c16"/"real"
    protocol = _json(path/"protocol.json")
    if (protocol.get("status") != "complete" or protocol.get("split") != "real"
            or protocol.get("sample_count") != 1016 or protocol.get("resampling") != "original512"
            or protocol.get("model_input_fields") != list(FIELDS)
            or protocol["model"]["head_epoch"] != 16
            or protocol["model"]["training_identity"]["arm"] != arm):
        raise ValueError("requires completed original512 REAL1016 C16 endpoint: " + str(path))
    rows = {}
    with (path/"pair_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = row["pair_id"]
            if pid in rows:
                raise ValueError("duplicate evaluated pair_id")
            # Do not retain the large correspondence matrices; only existing
            # scalar/pose output evidence is needed for replay validation.
            rows[pid] = {key:row[key] for key in ("pair_id", "fragment_a", "fragment_b", "label",
                "review_status", "strict_member", "decision_valid", "classification", "layouts")}
            rows[pid]["candidate_details"] = {key:row["candidate_details"][key]
                for key in ("raw_head_logit", "deployed_logit")}
    if len(rows) != 1016 or sum(r["label"] for r in rows.values()) != 508:
        raise ValueError("incomplete REAL pair results")
    return protocol, rows


def _accepted(row, threshold):
    return row["decision_valid"] and row["classification"]["fused"] >= threshold


def _correct(row):
    pose = row["layouts"][LAYOUT]
    return bool(row["label"] and row["review_status"] == "keep" and pose["valid"]
                and pose["translation_l2_px"] is not None and pose["translation_l2_px"] <= 20)


def select_real_cases(results_root):
    """Return42 deterministic metadata dicts; no tensors or model execution.

    results_root may be MIG/results or MIG/results/s7_direct. All14 specified
    newly-rejected cases are included (sorted IDs), then first14 retained
    correct-layout positives, then first7 strict and first7 constructed
    negatives. The latter are not selected by either model's score.
    """
    root = Path(results_root)
    if (root/"s7_direct").is_dir():
        root = root/"s7_direct"
    old_protocol, old = _endpoint(root, "matched_tokens")
    protocol, rows = _endpoint(root, "matched_edges")
    if rows.keys() != old.keys():
        raise ValueError("endpoint pair populations differ")
    for key in ("manifest_sha256", "keep_ids_sha256", "prepared_cache"):
        if protocol[key] != old_protocol[key]:
            raise ValueError("input/review source differs: " + key)
    if protocol["model"]["source_matcher_sha256"] != old_protocol["model"]["source_matcher_sha256"]:
        raise ValueError("frozen Matcher differs")
    for pid, row in rows.items():
        for key in ("fragment_a", "fragment_b", "label", "review_status", "strict_member", "decision_valid", "layouts"):
            if row[key] != old[pid][key]:
                raise ValueError("paired evidence differs: " + pid + ":" + key)
    old_threshold = old_protocol["model"]["operating_points"]["thresholds"]["max_f1"]
    threshold = protocol["model"]["operating_points"]["thresholds"]["max_f1"]
    expected_path = root.parent/"s7_final_edges_c16_comparison"/"newly_rejected_examples.json"
    specified = _json(expected_path)
    loss_ids = sorted(r["pair_id"] for r in specified["examples"])
    actual_loss = sorted(pid for pid,r in rows.items() if _correct(r)
        and _accepted(old[pid], old_threshold) and not _accepted(r, threshold))
    if specified.get("status") != "complete" or len(set(loss_ids)) != 14 or loss_ids != actual_loss:
        raise ValueError("all14 predeclared newly rejected correct-layout cases required")
    retained = sorted(pid for pid,r in rows.items() if _correct(r)
        and _accepted(old[pid], old_threshold) and _accepted(r, threshold))[:14]
    strict = sorted(pid for pid,r in rows.items() if not r["label"] and r["strict_member"])[:7]
    constructed = sorted(pid for pid,r in rows.items() if not r["label"] and not r["strict_member"])[:7]
    groups = (("newly_rejected_correct_layout",loss_ids), ("retained_correct_layout_both_accepted",retained),
              ("negative_strict",strict), ("negative_constructed",constructed))
    if tuple(len(ids) for _,ids in groups) != (14,14,7,7):
        raise ValueError("registered42-case composition unavailable")
    selected = []
    for group, ids in groups:
        for pid in ids:
            row, pose = rows[pid], rows[pid]["layouts"][LAYOUT]
            details = row["candidate_details"]
            if not all(math.isfinite(details[k]) for k in ("raw_head_logit", "deployed_logit")):
                raise ValueError("nonfinite saved head outputs")
            if not pose["valid"] or not all(math.isfinite(v) for v in pose["translation_rc"]):
                raise ValueError("all selected diagnostic cases need a valid finite production Layout")
            selected.append(dict(pair_id=pid, group=group, split="real", label=bool(row["label"]),
                fragment_a=row["fragment_a"], fragment_b=row["fragment_b"],
                prepared_cache=protocol["prepared_cache"], prepared_manifest_sha256=protocol["manifest_sha256"],
                model_input_fields=list(FIELDS), sampling="original512",
                checkpoint_sha256=protocol["model"]["checkpoint_sha256"],
                source_matcher_sha256=protocol["model"]["source_matcher_sha256"],
                expected_logit=details["deployed_logit"],
                expected_raw_head_logit=details["raw_head_logit"], expected_deployed_logit=details["deployed_logit"],
                expected_score=row["classification"]["fused"], expected_decision_valid=row["decision_valid"],
                expected_translation_rc=pose["translation_rc"], expected_layout_valid=pose["valid"],
                saved_prediction_source=str((root/"matched_edges"/"evaluation"/"c16"/"real").resolve()),
                selection_source=str(expected_path.resolve()), selection_source_sha256=_sha(expected_path)))
    assert len(selected) == len({r["pair_id"] for r in selected}) == 42
    return selected


def iter_real_inputs(selected, device="cpu", *, prepared_cache=None):
    """Yield (metadata, six_input_tensor_tuple), one original pair at a time.

    Optional prepared_cache only relocates an identical manifest-bound cache.
    Reuses evaluate_score_decoupled.core.load_prepared_cache, real.input_batches
    and sealed._tensor exactly; no resizing, re-extraction or alternate density.
    Labels and GT are absent from each tensor tuple and stripped from the pair
    metadata handed to input_batches. CPU-only finite diagnostic helper.
    """
    import torch
    from experiments.rachel_n512_formal_30k import evaluate_score_decoupled as evaluation
    core = evaluation.core
    selected = list(selected)
    if not 1 <= len(selected) <= 42 or len({m["pair_id"] for m in selected}) != len(selected):
        raise ValueError("finite unique selected REAL cases (maximum42) required")
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("this helper is scoped to the CPU diagnostic")
    bindings = {(m["prepared_cache"],m["prepared_manifest_sha256"]) for m in selected}
    if len(bindings) != 1 or tuple(core.FIELDS) != FIELDS:
        raise ValueError("input-source/schema mismatch")
    source_path, manifest_sha = next(iter(bindings))
    path = Path(prepared_cache) if prepared_cache is not None else Path(source_path)
    if _sha(path/"manifest.json") != manifest_sha:
        raise ValueError("prepared manifest differs from saved inference")
    metadata, arrays = core.load_prepared_cache(path)
    by_id = {r["pair_id"]:r for r in metadata["pairs"]}
    chosen_pairs = []
    for meta in selected:
        if meta["model_input_fields"] != list(FIELDS) or meta["sampling"] != "original512":
            raise ValueError("only exact original512 inputs supported")
        pair = by_id[meta["pair_id"]]
        if (pair["fragment_a_id"],pair["fragment_b_id"]) != (meta["fragment_a"],meta["fragment_b"]):
            raise ValueError("ordered fragments differ from saved inference")
        chosen_pairs.append({k:pair[k] for k in ("pair_id","fragment_a_id","fragment_b_id")})
    subset = dict(fragment_ids=metadata["fragment_ids"], pairs=chosen_pairs)
    produced = 0
    for meta, batch in zip(selected, core.real.input_batches(subset, arrays, 1)):
        if tuple(batch.pair_ids) != (meta["pair_id"],):
            raise ValueError("unexpected input-batch order")
        tensors = tuple(core.sealed._tensor(getattr(batch,field), device,
            torch.bool if field.startswith("contour_valid") else torch.float32) for field in FIELDS)
        produced += 1
        yield meta, tensors
    if produced != len(selected):
        raise ValueError("incomplete selected input iterator")
