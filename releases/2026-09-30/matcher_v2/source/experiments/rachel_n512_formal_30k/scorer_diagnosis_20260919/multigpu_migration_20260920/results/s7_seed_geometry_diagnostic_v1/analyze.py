"""Saved REAL raw-seed versus final-support comparison; no model inference."""
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "s7_direct"
ARMS = ("matched_tokens", "edge_seed")


def read_arm(arm):
    folder = SOURCE / arm / "evaluation/c16/real"
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "complete"
    model = protocol["model"]
    assert model["head_epoch"] == model["head_budget"] == 16
    path = folder / "pair_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({r["pair_id"] for r in rows}) == 1016
    return {r["pair_id"]: r for r in rows}, model["operating_points"]["thresholds"]["max_f1"], dict(
        pair_results=str(path), protocol=str(protocol_path),
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        checkpoint_sha256=model["checkpoint_sha256"],
        matcher_source_sha256=model.get("source_matcher_sha256", model.get("source_matcher_checkpoint_sha256")),
        sample_count=len(rows))


def endpoints(edges):
    return ({a for a, _ in edges}, {b for _, b in edges})


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else None


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values)) if values else dict(n=0, mean=None, median=None, minimum=None, maximum=None)


def summarize(rows):
    return dict(n=len(rows), seed_final_edges_identical=sum(r["edges_identical"] for r in rows),
        seed_final_edges_changed=sum(not r["edges_identical"] for r in rows),
        seed_final_endpoint_sets_identical=sum(r["endpoint_sets_identical"] for r in rows),
        raw_seed_layout20_correct=sum(r["raw_seed_error_px"] <= 20 for r in rows),
        raw_seed_layout20_incorrect=sum(r["raw_seed_error_px"] > 20 for r in rows),
        final_endpoints_min_le32=sum(r["final_unique_endpoints_min"] <= 32 for r in rows),
        raw_seed_endpoints_min_le32=sum(r["raw_seed_unique_endpoints_min"] <= 32 for r in rows),
        edge_jaccard=describe([r["edge_jaccard"] for r in rows]),
        raw_seed_to_final_shift_px=describe([r["seed_to_final_shift_px"] for r in rows]),
        raw_seed_error_px=describe([r["raw_seed_error_px"] for r in rows]),
        final_error_px=describe([r["final_error_px"] for r in rows]),
        final_unique_endpoints_min=describe([r["final_unique_endpoints_min"] for r in rows]),
        raw_seed_unique_endpoints_min=describe([r["raw_seed_unique_endpoints_min"] for r in rows]))


def main():
    loaded = {arm: read_arm(arm) for arm in ARMS}
    original, seed_rows = [loaded[arm][0] for arm in ARMS]
    thresholds = {arm: loaded[arm][1] for arm in ARMS}
    assert original.keys() == seed_rows.keys()
    cases, same_layout_count, keep_positives = [], 0, 0
    for pair_id, original_row in original.items():
        seed_row = seed_rows[pair_id]
        for key in ("label", "review_status", "target_translation_rc"):
            assert original_row[key] == seed_row[key], (pair_id, key)
        final_layout = original_row["layouts"]["full_top2_mode"]
        assert final_layout == seed_row["layouts"]["full_top2_mode"], pair_id
        same_layout_count += 1
        if not (original_row["label"] == 1 and original_row["review_status"] == "keep"):
            continue
        keep_positives += 1
        if not (final_layout["valid"] and final_layout["translation_l2_px"] <= 20):
            continue
        a, b = original_row["candidate_details"], seed_row["candidate_details"]
        for key in ("candidate_indices", "candidate_inliers", "candidate_valid", "candidate_weights"):
            assert a[key] == b[key], (pair_id, key)
        assert b["group_present"] == b["group_eligible"] == [True]
        assert b["selected_group_rank"] == 1 and len(b["group_candidate_inliers"]) == 1
        edges = [tuple(e) for e in b["candidate_indices"]]
        present = b["candidate_valid"]
        raw_mask, final_mask = b["group_candidate_inliers"][0], b["candidate_inliers"]
        assert len(edges) == len(present) == len(raw_mask) == len(final_mask)
        assert all(not flag or valid for flag, valid in zip(raw_mask, present))
        raw_edges = {e for e, valid, flag in zip(edges, present, raw_mask) if valid and flag}
        final_edges = {e for e, valid, flag in zip(edges, present, final_mask) if valid and flag}
        assert len(raw_edges) == b["group_inlier_count"][0]
        assert len(final_edges) == a["inlier_edge_count"]
        sa, sb = endpoints(raw_edges)
        fa, fb = endpoints(final_edges)
        seed_pose = b["group_translation_rc"][0]
        assert seed_pose == b["selected_group_translation_rc"]
        final_pose, target = final_layout["translation_rc"], original_row["target_translation_rc"]
        # Stored target and all reported poses use A-to-B row/column convention.
        assert abs(math.dist(final_pose, target) - final_layout["translation_l2_px"]) < 1e-6
        accepted_original = original_row["classification"]["fused"] >= thresholds["matched_tokens"]
        accepted_seed = seed_row["classification"]["fused"] >= thresholds["edge_seed"]
        group = ("loss" if accepted_original and not accepted_seed else
                 "gain" if accepted_seed and not accepted_original else
                 "both_accept" if accepted_seed else "both_reject")
        cases.append(dict(pair_id=pair_id, case_id=original_row.get("case_id"), group=group,
            matched_tokens_score=original_row["classification"]["fused"],
            edge_seed_score=seed_row["classification"]["fused"],
            edges_identical=raw_edges == final_edges, endpoint_sets_identical=sa == fa and sb == fb,
            raw_seed_edge_count=len(raw_edges), final_edge_count=len(final_edges),
            common_edges=len(raw_edges & final_edges), seed_only_edges=len(raw_edges-final_edges),
            final_only_edges=len(final_edges-raw_edges), edge_jaccard=jaccard(raw_edges, final_edges),
            endpoint_jaccard_a=jaccard(sa, fa), endpoint_jaccard_b=jaccard(sb, fb),
            raw_seed_unique_endpoints_min=min(len(sa),len(sb)), final_unique_endpoints_min=min(len(fa),len(fb)),
            raw_seed_translation_rc=seed_pose, final_translation_rc=final_pose,
            seed_to_final_shift_px=math.dist(seed_pose, final_pose),
            raw_seed_error_px=math.dist(seed_pose, target), final_error_px=final_layout["translation_l2_px"],
            raw_seed_candidate_slot=b["group_seed_candidate_id"][0]))
    assert keep_positives == 295 and len(cases) == 216
    summary = dict(schema="saved-seed-final-geometry-diagnostic/1", status="complete", sources={arm:loaded[arm][2] for arm in ARMS},
        thresholds=thresholds, population=dict(full_real_pairs=1016, kept_positive_pairs=295, final_layout20_correct=216),
        original_layouts_exactly_unchanged=same_layout_count,
        definitions=dict(raw_seed="group_candidate_inliers[0] and group_translation_rc[0]: best-support seed BEFORE refinement",
                         final="candidate_inliers and layouts.full_top2_mode: production final decoded support/pose",
                         loss="matched_tokens accepts, edge_seed rejects at each frozen SIMVAL maxF1 threshold",
                         gain="matched_tokens rejects, edge_seed accepts at each frozen SIMVAL maxF1 threshold"),
        all_correct_layouts=summarize(cases),
        groups={group:summarize([r for r in cases if r["group"]==group]) for group in ("loss","gain","both_accept","both_reject")},
        fields_sufficient=True, extra_cache_required=False, inference=False, training=False,
        limitations=["Post-hoc saved-output association, not a causal seed-only ablation.",
          "matched_tokens encodes deduplicated final endpoints; edge_seed also changes to bound edge features with raw Q and seed-residual metadata.",
          "Different trained heads and separately frozen thresholds contribute to classification changes.",
          "A <=20px raw seed pose does not prove every selected edge is a true correspondence.",
          "No contour coordinates or exact GT correspondence labels were needed/claimed."])
    for name,value in (("results.json",summary),):
        with (HERE/name).open("x") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    with (HERE/"cases.jsonl").open("x") as stream:
        for row in cases:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({key:summary[key] for key in ("thresholds","population","groups")},ensure_ascii=False))


if __name__ == "__main__":
    main()
