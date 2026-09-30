from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.rachel_n512_formal_30k import summarize_completed_ablation_arms as target


DECODER = "val_frozen_decoder"
THRESHOLD = 0.73


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(count, positives, recall):
    classification = dict(threshold=THRESHOLD, accuracy=0.72, precision=0.7,
                          recall=0.6, f1=0.646, tp=12, fp=5, fn=8, tn=15,
                          auroc=0.81, auprc=0.79)
    selected = dict(positive_pose_coverage=0.9, median_px_conditional=None,
                    p90_px_conditional=24.0, recall={"2": recall, "5": recall, "8": recall, "10": recall},
                    assembly={"10": dict(precision=0.6, recall=recall, f1=0.4, tp=1, fp=2, fn=3)})
    return dict(sample_count=count, positive_count=positives,
                classification={
                    "coarse": {"at_0_5": {"threshold": 0.5, "auroc": 0.55},
                               "at_validation_row_f1_threshold": {"threshold": 0.29, "f1": 0.52}},
                    "local": {"at_0_5": {"threshold": 0.5, "auroc": 0.65}},
                    "fused": {"at_0_5": {"threshold": 0.5, "f1": 0.53},
                              "at_validation_row_f1_threshold": {"threshold": 0.61, "f1": 0.92},
                              "at_original_frozen_threshold": classification}},
                layout={DECODER: selected, "test_best_but_not_frozen": dict(selected, recall={"10": 0.99})})


@pytest.fixture
def artifacts(tmp_path):
    key = "fixture-run--control-1234567890"
    arm = dict(key=key, arm="control", training_root="/remote/train/control",
               checkpoint="/remote/train/control/winner.pt", pair_threshold=THRESHOLD,
               status="complete", stages={stage: {"status": "complete", "reason": "formal_summary"}
                                           for stage in target.STAGES})
    snapshot = dict(status="following", follow=True, arms=[deepcopy(arm)],
                    waiting_for_training=[{"arm": "unready_provisional", "status": "training"}],
                    pending_eligible=True, live_training_pids=[2178])
    freeze = dict(source_split="validation", probe_only=False, test_or_real_used_for_fit=False,
                  original_fused_threshold=THRESHOLD, selected_full_decoder=DECODER,
                  sample_count=3000, checkpoint_epoch=7, checkpoint_sha256="stored-sha-not-recomputed",
                  precision="fp32", decoders={DECODER: {}, "test_best_but_not_frozen": {}})
    summaries = {}
    for stage, count, positives, recall in (("val", 3000, 1500, 0.3), ("test", 3000, 1500, 0.2),
                                           ("real", 1016, 508, 0.4)):
        summary = _metrics(count, positives, recall)
        summary.update(status="complete", split=stage, probe_only=False, selected_full_decoder=DECODER,
                       checkpoint_sha256=freeze["checkpoint_sha256"], precision=freeze["precision"])
        if stage == "real":
            summary.update(population="real_balanced1016", strict_summary=_metrics(547, 508, 0.7),
                           seam_quality_evaluated=False,
                           seam_quality_unavailable="no_dataset_correspondence_reference")
        else:
            summary.update(seam_quality_evaluated=True,
                           seam_quality={DECODER: {"precision": 0.8, "median_distance_px": None}},
                           seam_quality_unavailable={"full_original": "no_explicit_selected_correspondences"})
        summaries[stage] = summary
    root = tmp_path / "evaluation"
    _write(root / "pipeline_snapshot.json", snapshot)
    _write(root / key / "pipeline_status.json", arm)
    _write(root / key / "val" / "validation_freeze.json", freeze)
    for stage, summary in summaries.items():
        _write(root / key / stage / "summary.json", summary)
    return dict(root=root, key=key, arm=arm, freeze=freeze, summaries=summaries, snapshot=snapshot)


def _aggregate(artifacts):
    return target.summarize_completed_arms(artifacts["root"])


def _change_summary(artifacts, stage, mutate):
    summary = artifacts["summaries"][stage]
    mutate(summary)
    _write(artifacts["root"] / artifacts["key"] / stage / "summary.json", summary)


def test_uses_validation_frozen_decoder_and_original_fused_threshold(artifacts):
    result = _aggregate(artifacts)
    arm = result["arms"][0]
    assert result["status"] == arm["status"] == "complete"
    assert "status" not in artifacts["freeze"]
    assert arm["checkpoint_epoch"] == 7
    assert arm["checkpoint_sha256"] == "stored-sha-not-recomputed"
    assert arm["precision"] == "fp32"
    population = arm["populations"]["test"]
    source = artifacts["summaries"]["test"]
    assert population["selected_decoder"] == DECODER
    assert population["layout"] == source["layout"][DECODER]
    assert population["layout"]["recall"]["10"] == 0.2  # Not test's 0.99 alternative.
    assert population["classification"] == source["classification"]["fused"]["at_original_frozen_threshold"]
    assert population["classification"]["threshold"] == THRESHOLD
    assert population["classification"]["f1"] != 0.92  # Not the row-F1 sensitivity threshold.
    assert population["branch_classification"] == source["classification"]
    assert population["layout"]["median_px_conditional"] is None


def test_balanced_and_strict_are_distinct_populations_same_frozen_decoder(artifacts):
    populations = _aggregate(artifacts)["arms"][0]["populations"]
    balanced, strict = populations["real_balanced1016"], populations["real_strict547"]
    assert (balanced["sample_count"], balanced["positive_count"]) == (1016, 508)
    assert (strict["sample_count"], strict["positive_count"]) == (547, 508)
    assert balanced["layout"]["recall"]["10"] == 0.4
    assert strict["layout"]["recall"]["10"] == 0.7
    assert strict["selected_decoder"] == balanced["selected_decoder"] == DECODER
    assert strict["source"]["path"] == balanced["source"]["path"]
    assert strict["source"]["json_pointer"] == "/strict_summary"
    assert balanced["source"]["json_pointer"] == ""
    assert populations["val"]["population"] == "synthetic_val"
    assert strict["branch_classification"] == artifacts["summaries"]["real"]["strict_summary"]["classification"]


def test_real_seam_unavailable_inherits_string_reason_without_zero_filling(artifacts):
    populations = _aggregate(artifacts)["arms"][0]["populations"]
    for name in ("real_balanced1016", "real_strict547"):
        seam = populations[name]["seam_quality"]
        assert seam == dict(status="unavailable", metrics=None,
                            reason="no_dataset_correspondence_reference", evaluated=False,
                            source_unavailable="no_dataset_correspondence_reference")
    assert populations["val"]["seam_quality"]["status"] == "available"
    assert populations["val"]["seam_quality"]["metrics"]["median_distance_px"] is None


def test_synthetic_decoder_specific_seam_unavailable_dictionary(artifacts):
    def mutate(summary):
        summary["seam_quality"][DECODER] = None
        summary["seam_quality_unavailable"][DECODER] = "no_explicit_selected_correspondences"
    _change_summary(artifacts, "test", mutate)
    seam = _aggregate(artifacts)["arms"][0]["populations"]["test"]["seam_quality"]
    assert seam["status"] == "unavailable"
    assert seam["metrics"] is None
    assert seam["reason"] == "no_explicit_selected_correspondences"


def test_missing_results_are_partial_not_complete(artifacts):
    root, key = artifacts["root"], artifacts["key"]
    for stage in ("test", "real"):
        (root / key / stage / "summary.json").unlink()
        artifacts["arm"]["stages"][stage]["status"] = "ready"
    artifacts["arm"]["status"] = "partial"
    _write(root / key / "pipeline_status.json", artifacts["arm"])
    result = _aggregate(artifacts)
    arm = result["arms"][0]
    assert result["status"] == arm["status"] == "partial"
    assert arm["populations"]["val"]["status"] == "complete"
    for name in ("test", "real_balanced1016", "real_strict547"):
        assert arm["populations"][name]["status"] == "missing"
        assert arm["populations"][name]["classification"] is None
        assert arm["populations"][name]["layout"] is None


def test_failed_pipeline_stage_cannot_be_completed_by_existing_summary(artifacts):
    artifacts["arm"]["stages"]["real"]["status"] = "needs_attention"
    artifacts["arm"]["status"] = "needs_attention"
    _write(artifacts["root"] / artifacts["key"] / "pipeline_status.json", artifacts["arm"])
    result = _aggregate(artifacts)
    assert result["status"] == result["arms"][0]["status"] == "needs_attention"
    for name in ("real_balanced1016", "real_strict547"):
        population = result["arms"][0]["populations"][name]
        assert population["status"] == "failed"
        assert population["classification"] is None


@pytest.mark.parametrize("stage", ["val", "test", "real"])
def test_declared_decoder_conflict_is_explicit(artifacts, stage):
    _change_summary(artifacts, stage, lambda summary: summary.update(selected_full_decoder="test_best_but_not_frozen"))
    arm = _aggregate(artifacts)["arms"][0]
    population = arm["populations"]["real_balanced1016" if stage == "real" else stage]
    assert arm["status"] == "needs_attention"
    assert population["status"] == "conflict"
    assert population["selected_decoder"] == DECODER
    assert population["classification"] is population["layout"] is None
    assert any(issue["type"] == "decoder_conflict" for issue in population["issues"])
    if stage == "real":
        strict = arm["populations"]["real_strict547"]
        assert strict["status"] == "conflict"
        assert strict["classification"] is strict["layout"] is None


@pytest.mark.parametrize("key,value", [("checkpoint_sha256", "old-winner-sha"), ("precision", "bf16")])
def test_stored_metadata_conflict_blocks_balanced_and_strict_without_reading_model(artifacts, key, value):
    _change_summary(artifacts, "real", lambda summary: summary.update({key: value}))
    arm = _aggregate(artifacts)["arms"][0]
    for name in ("real_balanced1016", "real_strict547"):
        population = arm["populations"][name]
        assert population["status"] == "conflict"
        assert population["classification"] is population["layout"] is None
        assert any(issue["type"] == key + "_conflict" for issue in population["issues"])


def test_original_threshold_conflict_is_not_replaced_with_another_branch_threshold(artifacts):
    def mutate(summary):
        summary["classification"]["fused"]["at_original_frozen_threshold"]["threshold"] = 0.61
    _change_summary(artifacts, "test", mutate)
    population = _aggregate(artifacts)["arms"][0]["populations"]["test"]
    assert population["status"] == "conflict"
    assert population["classification"] is None
    assert any(issue["type"] == "threshold_conflict" for issue in population["issues"])


def test_pipeline_threshold_conflict_invalidates_all_stages(artifacts):
    artifacts["arm"]["pair_threshold"] = 0.5
    _write(artifacts["root"] / artifacts["key"] / "pipeline_status.json", artifacts["arm"])
    arm = _aggregate(artifacts)["arms"][0]
    assert arm["status"] == "needs_attention"
    assert all(pop["status"] == "conflict" for pop in arm["populations"].values())


def test_missing_strict_cannot_be_substituted_with_balanced(artifacts):
    _change_summary(artifacts, "real", lambda summary: summary.pop("strict_summary"))
    arm = _aggregate(artifacts)["arms"][0]
    assert arm["status"] == "partial"
    assert arm["populations"]["real_balanced1016"]["status"] == "complete"
    strict = arm["populations"]["real_strict547"]
    assert strict["status"] == "missing"
    assert strict["sample_count"] is strict["layout"] is None


def test_probe_or_incomplete_population_is_not_formal_complete(artifacts):
    _change_summary(artifacts, "val", lambda summary: summary.update(probe_only=True, sample_count=256))
    arm = _aggregate(artifacts)["arms"][0]
    assert arm["status"] == "partial"
    assert arm["populations"]["val"]["status"] == "incomplete"
    assert arm["populations"]["val"]["classification"] is None


def test_snapshot_scope_does_not_infer_total_planned_arm_count(artifacts):
    result = _aggregate(artifacts)
    assert result["counts"] == dict(listed_arms=1, complete=1, partial=0, needs_attention=0)
    assert result["expected_total_arm_count"] is None
    assert "all planned arms" in result["scope"]
    assert result["pipeline_observation"]["waiting_for_training"] == artifacts["snapshot"]["waiting_for_training"]
    assert result["pipeline_observation"]["status"] == "following"


def test_cli_reads_only_authorized_jsons_and_writes_only_output(artifacts, monkeypatch, tmp_path):
    root, key = artifacts["root"], artifacts["key"]
    expected_inputs = {root / "pipeline_snapshot.json", root / key / "pipeline_status.json",
                       root / key / "val" / "validation_freeze.json"}
    expected_inputs.update(root / key / stage / "summary.json" for stage in target.STAGES)
    opened = []
    original_open = Path.open

    def tracked_open(path, mode="r", *args, **kwargs):
        opened.append((path.resolve(), mode))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    output = tmp_path / "aggregate" / "comparison.json"
    result = target.main(["--evaluation-root", str(root), "--output-json", str(output)])
    assert set(path for path, mode in opened if mode == "r") == expected_inputs
    assert [(path, mode) for path, mode in opened if mode != "r"] == [(output, "w")]
    with original_open(output, encoding="utf-8") as stream:
        assert json.load(stream) == result


def test_cli_refuses_to_overwrite_any_input(artifacts):
    root = artifacts["root"]
    source = root / artifacts["key"] / "test" / "summary.json"
    before = source.read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        target.main(["--evaluation-root", str(root), "--output-json", str(source)])
    assert exc.value.code == 2
    assert source.read_text(encoding="utf-8") == before
