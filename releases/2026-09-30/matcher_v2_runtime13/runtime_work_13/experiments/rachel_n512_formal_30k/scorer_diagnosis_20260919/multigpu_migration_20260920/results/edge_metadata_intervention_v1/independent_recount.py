"""Independent aggregation of saved intervention cases; no inference or fit."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
THRESHOLDS = dict(max_f1=0.9202651381492615, recall_99=0.5426456928253174)
INTERVENTIONS = ("q_zero", "residual_half", "residual_zero")


def accepted(row, op, intervention=None):
    probability = row["baseline_probability"] if intervention is None else row["interventions"][intervention]["probability"]
    return probability >= THRESHOLDS[op]


def movement(rows, intervention, op):
    old = {r["pair_id"] for r in rows if accepted(r, op)}
    new = {r["pair_id"] for r in rows if accepted(r, op, intervention)}
    return dict(n=len(rows), baseline_accepted=len(old), intervention_accepted=len(new),
        newly_accepted=len(new-old), newly_rejected=len(old-new),
        newly_accepted_pair_ids=sorted(new-old), newly_rejected_pair_ids=sorted(old-new),
        mean_delta_logit=statistics.mean(r["interventions"][intervention]["logit"]-r["baseline_logit"] for r in rows))


def main():
    path = HERE/"cases.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({(r["split"],r["pair_id"]) for r in rows}) == 90
    assert Counter(r["split"] for r in rows) == dict(val=48,real=42)
    assert all(r["skipped_intervention"] is False for r in rows)
    for row in rows:
        for op in THRESHOLDS:
            assert accepted(row,op) == row["accepted"][op]
            for name,result in row["interventions"].items():
                assert accepted(row,op,name) == result["accepted"][op]
                assert abs((result["logit"]-row["baseline_logit"])-result["delta_logit"]) < 1e-12
        assert all(math.isfinite(row[key]) for key in ("baseline_logit", "baseline_probability"))
    groups = {group:[r for r in rows if r["group"] == group] for group in sorted({r["group"] for r in rows})}
    assert len(groups["newly_rejected_correct_layout"]) == 14
    real = [r for r in rows if r["split"] == "real"]
    assert all(r["online_training_valid"] and r["online_decision_valid"] and r["expected_layout_valid"] for r in real)
    logit_errors = [(abs(r["baseline_logit"]-r["expected_logit"]),r["pair_id"]) for r in real]
    pose_errors = [(math.dist(r["fixed_translation_rc"],r["expected_translation_rc"]),r["pair_id"]) for r in real]
    for row, (logit_error,_), (pose_error,_) in zip(real,logit_errors,pose_errors):
        assert abs(logit_error-row["saved_logit_replay_error"]) < 1e-12
        assert abs(pose_error-row["saved_translation_replay_error_px"]) < 1e-12
    noop_logit = [abs(r["interventions"]["noop"]["logit"]-r["baseline_logit"]) for r in rows]
    noop_prob = [abs(r["interventions"]["noop"]["probability"]-r["baseline_probability"]) for r in rows]
    group_results = {}
    for name, selected in groups.items():
        per_case_residual = [r["residual_norm"]["mean"]*10. for r in selected]
        group_results[name] = dict(n=len(selected), split=selected[0]["split"], label=bool(selected[0]["label"]),
            mean_per_case_residual_px=statistics.mean(per_case_residual),
            median_per_case_residual_px=statistics.median(per_case_residual),
            mean_per_case_projection_rms={key:statistics.mean(r["projection_rms"][key] for r in selected)
                for key in ("q", "residual", "bias", "self_a", "mate_b")},
            interventions={i:{op:movement(selected,i,op) for op in THRESHOLDS} for i in INTERVENTIONS})
    result = dict(schema="independent-frozen-edge-metadata-intervention-recount/1",status="complete",
        source_cases=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        count=90,split_counts=dict(Counter(r["split"] for r in rows)),thresholds=THRESHOLDS,
        no_op=dict(count=90,exact_logit_equality_count=sum(x==0 for x in noop_logit),
            exact_probability_equality_count=sum(x==0 for x in noop_prob),maximum_logit_error=max(noop_logit),
            maximum_probability_error=max(noop_prob)),
        cpu_vs_saved_real=dict(count=42,maximum_logit_error=max(logit_errors)[0],
            maximum_logit_error_pair_id=max(logit_errors)[1],maximum_translation_l2_error_px=max(pose_errors)[0],
            maximum_translation_error_pair_id=max(pose_errors)[1],
            logit_tolerance_1e_3_pass=all(x<=1e-3 for x,_ in logit_errors),
            pose_tolerance_0_01px_pass=all(x<=.01 for x,_ in pose_errors)),
        groups=group_results,
        interpretation_limits=[
            "Only saved cases.jsonl used for the recount; no new inference/training or fitted threshold.",
            "All group means weight cases equally, not edges; residual_px = saved normalized-residual mean times10.",
            "Projection RMS measures feature scale, not marginal causal importance after the nonlinear head.",
            "The14 loss cases are outcome-selected and controls lexicographically sampled; not a population-performance estimate.",
            "Zeroing or scaling geometry metadata while freezing features/layout is an artificial intervention, not evidence that new physical input has lower residual.",
            "For negatives, newly_accepted counts adverse conversions; intervention_accepted also includes baseline false positives."])
    (HERE/"independent_recount.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"noop":result["no_op"],"replay":result["cpu_vs_saved_real"],
        "groups":{name:{"n":v["n"],"residual_px":v["mean_per_case_residual_px"],
            "projection_rms":v["mean_per_case_projection_rms"],
            "interventions":{i:{op:{k:z for k,z in values.items() if not k.endswith("pair_ids")}
                for op,values in ops.items()} for i,ops in v["interventions"].items()}}
            for name,v in group_results.items()}},ensure_ascii=False))


if __name__ == "__main__":
    main()
