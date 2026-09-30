"""Read completed S7 TRAIN24K Matcher evidence and final materialized targets.

No network forward, optimization, threshold fitting, candidate re-selection or
queue mutation. All GT comparisons happen after the recorded prediction. This
is in-sample Matcher diagnosis, never a held-out classification benchmark.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .metadata import join_metadata

SCHEMA = "s7-m12-training-match-support-diagnosis/1"
SOURCE_SHA = "d8a93af1eb5f3b02baaf7d42b9d8675242a446a1b11e43cde0561ba89e670e07"
TRAIN_SHA = "79a9e959f32ef9899116e299425d6350b17f6a04bd5070b5aa9a730319447c36"
FEATURES = ("candidate_count", "inlier_count", "unique_endpoints_min",
            "endpoint_coverage_min", "inlier_q_mass", "inlier_q_mean",
            "supported_chord_length_min_px", "longest_run_min_tokens")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_targets(archive, cached):
    """Match collate_rachel_pairs: prefix points, zero padding, -2 targets."""
    result = []
    for side in ("a", "b"):
        points = archive["points_rc_"+side]
        valid = archive["contour_valid_"+side]
        target = archive["target_"+side]
        n, cap = len(points), len(cached["valid_"+side])
        if n > cap or len(valid) != n or len(target) != n:
            raise ValueError("final artifact contour cap differs")
        padded_points = np.zeros_like(cached["points_"+side]); padded_points[:n] = points
        padded_valid = np.zeros(cap, bool); padded_valid[:n] = valid
        if not np.array_equal(padded_points, cached["points_"+side]) or not np.array_equal(padded_valid, cached["valid_"+side]):
            raise ValueError("target coordinates do not match frozen inference after standard batch padding")
        padded_target = np.full(cap,-2,np.int64); padded_target[:n] = target
        result.append(padded_target)
    return tuple(result)


def cyclic_support(points, valid, selected):
    """Approximate supported contour chords, not GT seam length/order."""
    p = np.asarray(points)[valid]
    mask = np.asarray(selected, bool)[valid]
    n = len(p)
    if not n:
        return dict(count=0, coverage=0., chord_length_px=0., longest_run_tokens=0)
    if mask.all():
        longest = n
    elif not mask.any():
        longest = 0
    else:
        # Start immediately after a false value so a wraparound run is whole.
        shifted = np.roll(mask, -int(np.flatnonzero(~mask)[0])-1)
        longest = run = 0
        for flag in shifted:
            run = run + 1 if flag else 0
            longest = max(longest, run)
    distance = np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1)
    supported = mask & np.roll(mask, -1)
    return dict(count=int(mask.sum()), coverage=float(mask.mean()),
        chord_length_px=float(distance[supported].sum()), longest_run_tokens=int(longest))


def row_evidence(a, target_a, target_b, gt_translation, gt_valid, report):
    """One cached row + its matching final NPZ target, never source counts."""
    present = np.asarray(a["candidate_valid"], bool)
    inliers = np.asarray(a["candidate_inliers"], bool)
    if (inliers & ~present).any():
        raise ValueError("inlier references padding")
    edges = np.asarray(a["candidate_indices"], np.int64)
    va, vb = np.asarray(a["valid_a"], bool), np.asarray(a["valid_b"], bool)
    pa, pb = np.asarray(a["points_a"]), np.asarray(a["points_b"])
    actual = edges[present]
    if len(actual) and ((actual < 0).any() or (actual[:, 0] >= len(va)).any()
                       or (actual[:, 1] >= len(vb)).any()):
        raise ValueError("candidate outside endpoint arrays")
    if len(actual) and (not va[actual[:, 0]].all() or not vb[actual[:, 1]].all()):
        raise ValueError("candidate at invalid contour point")
    ta, tb = np.asarray(target_a), np.asarray(target_b)
    matched = edges[inliers]
    masks = [np.zeros(len(va), bool), np.zeros(len(vb), bool)]
    if len(matched):
        for side in (0, 1):
            masks[side][matched[:, side]] = True
    lengths = [cyclic_support(p, v, m) for p, v, m in zip((pa, pb), (va, vb), masks)]
    q = np.asarray(a["candidate_weights"], float)[inliers]
    layout_valid = bool(a["layout_valid"])
    prediction = np.asarray(a["translation_a_to_b_rc"], float)
    if layout_valid and not np.isfinite(prediction).all():
        raise ValueError("valid layout has nonfinite translation")
    residual = None
    if layout_valid and len(matched):
        delta = pb[matched[:, 1]] - pa[matched[:, 0]]
        residual = float(np.sqrt(np.sum(q * np.square(delta-prediction).sum(1))/q.sum()))
    def matches(e):
        if not len(e):
            return np.zeros(0, bool), np.zeros(0, bool)
        supervised = (ta[e[:, 0]] >= -1) & (tb[e[:, 1]] >= -1)
        correct = (ta[e[:, 0]] == e[:, 1]) & (tb[e[:, 1]] == e[:, 0])
        return correct, supervised
    correct_all, _ = matches(actual)
    correct_final, supervised_final = matches(matched)
    # Targets are reciprocal inherited source-arc labels. Ignored (-2) edges
    # must NOT count as false correspondences, unlike supervised dustbins (-1).
    gt_ids = np.flatnonzero(va & (ta >= 0))
    if len(gt_ids) and (not vb[ta[gt_ids]].all() or not np.array_equal(tb[ta[gt_ids]], gt_ids)):
        raise ValueError("final target is not reciprocal")
    target_count = len(gt_ids)
    label = bool(a["label"])
    if not label and target_count:
        raise ValueError("negative pair has positive correspondences")
    if bool(gt_valid) != label or bool(report["pose_supervision_enabled"]) != (label and not report["changed_pair"]):
        raise ValueError("materialized target/pose eligibility contract differs")
    error = float(np.linalg.norm(prediction-np.asarray(gt_translation))) if label and gt_valid and layout_valid else None
    final_correct_count = len(np.unique(matched[correct_final, 0]))
    candidate_correct_count = len(np.unique(actual[correct_all, 0]))
    return dict(label=int(label), training_valid=bool(a["training_valid"]),
        decision_valid=bool(a["decision_valid"]), layout_valid=layout_valid,
        candidate_count=int(present.sum()), inlier_count=int(inliers.sum()),
        unique_endpoints_min=min(x["count"] for x in lengths),
        endpoint_coverage_min=min(x["coverage"] for x in lengths),
        supported_chord_length_min_px=min(x["chord_length_px"] for x in lengths),
        longest_run_min_tokens=min(x["longest_run_tokens"] for x in lengths),
        inlier_q_mass=float(q.sum()), inlier_q_mean=float(q.mean()) if len(q) else 0.,
        inlier_weighted_rms_px=residual, final_target_matches=target_count,
        final_ignored_tokens=int(((ta == -2) & va).sum()+((tb == -2) & vb).sum()),
        correct_candidate_edges=candidate_correct_count, correct_inlier_edges=final_correct_count,
        supervised_inlier_edges=int(supervised_final.sum()),
        exact_target_recall_candidates=candidate_correct_count/target_count if target_count else None,
        exact_target_recall_inliers=final_correct_count/target_count if target_count else None,
        exact_inlier_precision=float(correct_final.sum()/supervised_final.sum()) if supervised_final.any() else None,
        gt_translation_valid=bool(gt_valid), raw_layout_error_px=error,
        raw_layout20_success=bool(label and error is not None and error <= 20.),
        pose_supervision_enabled=bool(report["pose_supervision_enabled"]),
        actual_changed_pair=bool(report["changed_pair"]), fallback_reason=report.get("fallback_reason"))


def auc(labels, scores):
    """Mann-Whitney AUROC with exact average ranks for ties; high=positive."""
    y, x = np.asarray(labels, int), np.asarray(scores, float)
    if not len(y) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid diagnostic scores")
    positives = int(y.sum()); negatives = len(y)-positives
    if not positives or not negatives:
        return None
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = x[order], y[order]
    rank_sum, first = 0., 0
    while first < len(x):
        last = first+1
        while last < len(x) and sorted_x[last] == sorted_x[first]:
            last += 1
        rank_sum += ((first+1+last)/2.) * int(sorted_y[first:last].sum())
        first = last
    return float((rank_sum-positives*(positives+1)/2.)/(positives*negatives))


def summarize(rows):
    result = dict(count=len(rows), positive_count=sum(r["label"] for r in rows))
    for name in ("training_valid", "decision_valid", "layout_valid", "actual_changed_pair", "pose_supervision_enabled"):
        result[name+"_count"] = sum(bool(r[name]) for r in rows)
    result["metrics"] = {}
    for name in FEATURES + ("final_target_matches", "raw_layout_error_px", "inlier_weighted_rms_px",
                             "exact_target_recall_candidates", "exact_target_recall_inliers", "exact_inlier_precision"):
        values = np.array([r[name] for r in rows if r[name] is not None], float)
        result["metrics"][name] = dict(available_count=len(values),
            mean=float(values.mean()) if len(values) else None,
            quantiles=dict(zip(("p10", "p50", "p90"), map(float, np.quantile(values, [.1,.5,.9])))) if len(values) else {})
    positive = [r for r in rows if r["label"]]
    result["raw_layout20"] = dict(success=sum(r["raw_layout20_success"] for r in positive),
        denominator=len(positive), missing_gt=sum(not r["gt_translation_valid"] for r in positive),
        rate=sum(r["raw_layout20_success"] for r in positive)/len(positive) if positive else None)
    result["support_auroc_in_sample"] = {name: auc([r["label"] for r in rows], [r[name] for r in rows]) for name in FEATURES} if rows else {}
    return result


def run(args):
    cache, manifest, out = Path(args.cache), Path(args.manifest), Path(args.output)
    protocol = json.loads((cache/"protocol.json").read_text())
    if (protocol.get("status") != "complete" or protocol.get("pair_count") != 24000
        or protocol.get("completed_pairs") != 24000 or protocol.get("split") != "train"
        or protocol.get("source_checkpoint_sha256") != SOURCE_SHA or sha(manifest) != TRAIN_SHA):
        raise ValueError("requires complete registered S7 M12 TRAIN24K")
    payload = json.loads(manifest.read_text())
    records = json.loads((cache/"pairs.json").read_text())
    if sha(cache/"pairs.json") != protocol["pairs_sha256"] or len(records) != 24000:
        raise ValueError("cache population mismatch")
    metadata = join_metadata(payload["entries"], records)
    entries = {r["pair_id"]: r for r in payload["entries"]}
    names = ("candidate_valid", "candidate_inliers", "candidate_indices", "candidate_weights",
        "points_a", "points_b", "valid_a", "valid_b", "layout_valid", "translation_a_to_b_rc",
        "label", "training_valid", "decision_valid", "ready")
    arrays = {name: np.load(cache/(name+".npy"), mmap_mode="r", allow_pickle=False) for name in names}
    if not arrays["ready"].all() or any(len(a) != 24000 for a in arrays.values()):
        raise ValueError("cache incomplete")
    out.mkdir(parents=True, exist_ok=False)
    started, rows = time.monotonic(), []
    with (out/"rows.jsonl").open("x") as stream:
        for index, (record, meta) in enumerate(zip(records, metadata)):
            entry = entries[record["pair_id"]]
            a = {name: value[index] for name, value in arrays.items()}
            with np.load(Path(payload["artifact_root"])/entry["artifact_path"], allow_pickle=False) as z:
                if str(z["pair_id"].item()) != record["pair_id"] or float(z["label"]) != record["label"] or float(a["label"]) != record["label"]:
                    raise ValueError("final artifact identity mismatch")
                targets = align_targets(z, a)
                report = json.loads(str(z["report_json"].item()))
                metrics = row_evidence(a,*targets,z["translation_a_to_b_rc"],bool(z["translation_valid"]),report)
            row = {**meta, **metrics}
            if row["changed_pair"] != row["actual_changed_pair"]:
                raise ValueError("manifest applied-state differs from actual artifact")
            rows.append(row); stream.write(json.dumps(row, allow_nan=False)+"\n")
    groups = {}
    for key in ("label", "s7_recipe", "source_stratum", "negative_source_kind", "actual_changed_pair", "area_ratio_band"):
        parts = defaultdict(list)
        for r in rows:
            parts[str(r[key])+"|label="+str(r["label"])].append(r)
        groups[key] = {name:summarize(subset) for name,subset in sorted(parts.items())}
    result = dict(schema=SCHEMA,status="complete",elapsed_s=time.monotonic()-started,
        sources=dict(cache=str(cache),cache_protocol_sha256=sha(cache/"protocol.json"),manifest=str(manifest),
            manifest_sha256=TRAIN_SHA,source_checkpoint_sha256=SOURCE_SHA,implementation_sha256=sha(__file__)),
        scope="in-sample S7 TRAIN24K M12 Matcher diagnosis; not classifier/test performance",
        overall=summarize(rows), groups=groups, rows_sha256=sha(out/"rows.jsonl"),
        limitations=["No paired clean/damaged rerun: recipe comparisons are descriptive, not augmentation causal effects.",
            "Only exact inherited source-arc labels assessed; ignored new/missing edges excluded from precision.",
            "Raw Layout20 denominator includes every positive; failed solver is not dropped.",
            "Inlier count/chord support does not establish a true ordered seam or non-overlap.",
            "TRAIN is already seen by M12; these AUROCs are descriptive feature separation, never held-out accuracy.",
            "Manifest match counts and area ratio are source/preaugmentation metadata, not final targets."])
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="complete",count=len(rows),elapsed_s=result["elapsed_s"])))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True); p.add_argument("--manifest",required=True); p.add_argument("--output",required=True)
    run(p.parse_args())
