"""Small synthetic checks for report-only, same-seed confirmation comparisons."""
from copy import deepcopy
import json

import pytest

from experiments.rachel_n512_formal_30k import summarize_matched_confirmation as target


DECODER = "frozen_decoder"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _population(count, positives, threshold, gain, strict=False):
    tp, fp, fn = positives - 10 + gain, 2 if strict else 10, 10 - gain
    pair = dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                tn=count - positives - fp, precision=tp / (tp + fp),
                recall=tp / positives, f1=2 * tp / (2 * tp + fp + fn),
                auroc=0.8 + gain / 100, auprc=0.75 + gain / 100)
    joint_tp = positives - 30 + gain
    joint_fp, joint_fn = tp + fp - joint_tp, positives - joint_tp
    layout = dict(positive_pose_coverage=1.0, median_px_conditional=3.0 - gain / 10,
                  p90_px_conditional=20.0 - gain,
                  recall={str(c): (positives - offset + gain) / positives
                          for c, offset in ((2, 40), (5, 30), (8, 20), (10, 10))},
                  assembly={"10": dict(tp=joint_tp, fp=joint_fp, fn=joint_fn,
                                       f1=2 * joint_tp / (2 * joint_tp + joint_fp + joint_fn))})
    unselected = deepcopy(layout)
    unselected["recall"] = {str(c): 1.0 for c in (2, 5, 8, 10)}
    return dict(sample_count=count, positive_count=positives,
                classification={"fused": {"at_original_frozen_threshold": pair,
                                          "at_0_5": dict(pair, threshold=0.5, f1=1.0)}},
                layout={DECODER: layout, "better_but_unselected": unselected})


def _run(root, arm, seed, gain=0):
    directory = root / ("confirmation_controls" if arm == "control" else "confirmation_candidate") / ("seed%d" % seed)
    freeze = dict(seed=seed, source_split="validation", test_or_real_used_for_fit=False,
                  probe_only=False, sample_count=3000, checkpoint_epoch=2,
                  checkpoint_sha256="%s-%d-sha" % (arm, seed), precision="fp32",
                  original_fused_threshold=0.7 if arm == "control" else 0.8,
                  selected_full_decoder=DECODER)
    _write(directory / "val" / "validation_freeze.json", freeze)
    summaries = {}
    for split, count, positives in (("val", 3000, 1500), ("test", 3000, 1500), ("real", 1016, 508)):
        summary = _population(count, positives, freeze["original_fused_threshold"], gain)
        summary.update(status="complete", split=split, selected_full_decoder=DECODER,
                       checkpoint_sha256=freeze["checkpoint_sha256"], precision="fp32")
        if split == "real":
            summary["strict_summary"] = _population(547, 508, freeze["original_fused_threshold"], gain, strict=True)
        _write(directory / split / "summary.json", summary)
        summaries[split] = summary
    return directory, freeze, summaries


def test_absent_candidates_remain_partial_with_no_deltas(tmp_path):
    for seed in (260908, 260909):
        _run(tmp_path, "control", seed)
    result = target.summarize(tmp_path)
    assert result["status"] == "partial"
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        assert row["control"]["status"] == "complete"
        assert row["candidate"]["status"] == "pending"
        assert row["candidate"]["metrics"] is None
        assert row["delta"] is None


@pytest.mark.parametrize("field,value", [("seed", 260909), ("checkpoint_sha256", "other-seed-winner")])
def test_summary_must_match_its_own_seed_freeze(tmp_path, field, value):
    directory, freeze, _ = _run(tmp_path, "candidate", 260908)
    wrong_freeze = dict(freeze, **{field: value})
    result = target.extract_summary(directory / "test" / "summary.json", wrong_freeze, 260908, "test")
    assert result["status"] == "needs_attention"
    assert result["metrics"] is None
    assert result["issues"]


def test_extracts_frozen_decoder_and_original_threshold_not_better_alternatives(tmp_path):
    directory, freeze, summaries = _run(tmp_path, "candidate", 260908)
    result = target.extract_summary(directory / "test" / "summary.json", freeze, 260908, "test")
    assert result["status"] == "complete"
    assert result["selected_decoder"] == DECODER
    assert result["threshold"] == freeze["original_fused_threshold"]
    assert result["metrics"]["pose_n10"] == 1490
    assert result["metrics"]["pose_r10"] == pytest.approx(1490 / 1500)
    assert result["metrics"]["pair_f1"] == summaries["test"]["classification"]["fused"]["at_original_frozen_threshold"]["f1"]
    assert result["metrics"]["pair_f1"] != 1.0


def test_real_strict_uses_nested_subset_not_balanced_counts(tmp_path):
    directory, freeze, _ = _run(tmp_path, "candidate", 260908)
    path = directory / "real" / "summary.json"
    balanced = target.extract_summary(path, freeze, 260908, "real_balanced1016")
    strict = target.extract_summary(path, freeze, 260908, "real_strict547")
    assert balanced["status"] == strict["status"] == "complete"
    assert (balanced["sample_count"], strict["sample_count"]) == (1016, 547)
    assert balanced["positive_count"] == strict["positive_count"] == 508
    assert (balanced["metrics"]["pair_fp"], strict["metrics"]["pair_fp"]) == (10, 2)
    assert strict["metrics"]["pair_f1"] > balanced["metrics"]["pair_f1"]
    assert strict["metrics"]["pose_n10"] == balanced["metrics"]["pose_n10"] == 498


def test_both_complete_seeds_produce_eight_same_seed_candidate_minus_control_deltas(tmp_path):
    for seed, control_gain, candidate_gain in ((260908, 0, 1), (260909, 2, 5)):
        _run(tmp_path, "control", seed, control_gain)
        _run(tmp_path, "candidate", seed, candidate_gain)
    result = target.summarize(tmp_path)
    assert result["schema_version"] == "matched-confirmation-comparison/1"
    assert result["status"] == "complete"
    assert {(row["seed"], row["population"]) for row in result["rows"]} == {
        (seed, population) for seed in (260908, 260909)
        for population in ("val", "test", "real_balanced1016", "real_strict547")}
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        gain = 1 if row["seed"] == 260908 else 3
        assert row["control"]["status"] == row["candidate"]["status"] == "complete"
        assert row["delta"]["pair_tp"] == row["delta"]["pose_n10"] == gain
        assert row["delta"]["joint10_tp"] == gain
        assert row["delta"]["pair_fn"] == -gain
        assert row["delta"]["pose_p90_px_conditional"] == -gain
        for key, value in row["delta"].items():
            assert value == pytest.approx(row["candidate"]["metrics"][key] - row["control"]["metrics"][key])


def test_conflicting_second_seed_freeze_blocks_its_deltas_without_hiding_first_seed(tmp_path):
    for seed in (260908, 260909):
        _run(tmp_path, "control", seed)
        directory, freeze, _ = _run(tmp_path, "candidate", seed, 1)
    _write(directory / "val" / "validation_freeze.json", dict(freeze, seed=260908))
    result = target.summarize(tmp_path)
    assert result["status"] == "needs_attention"
    for row in result["rows"]:
        if row["seed"] == 260909:
            assert row["candidate"]["status"] == "needs_attention"
            assert row["delta"] is None
        else:
            assert row["candidate"]["status"] == "complete"
            assert row["delta"]["pair_tp"] == 1
