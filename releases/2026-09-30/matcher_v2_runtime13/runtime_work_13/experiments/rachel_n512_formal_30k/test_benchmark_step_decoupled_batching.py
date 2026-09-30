"""Small CPU shape-selection tests; never starts benchmark CUDA workers."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.rachel_n512_formal_30k import benchmark_step_decoupled_batching as benchmark


@contextmanager
def raises(error, text):
    with unittest.TestCase().assertRaisesRegex(error, text):
        yield


def test_longest_axes_and_matrix_area_all_covered_without_inventing_points():
    counts = [[400+i,600+i] for i in range(30)]
    counts[1] = [2048,100]
    counts[2] = [120,2048]
    counts[3] = [1900,1900]
    entries = [dict(pair_id="fixture%d"%i,label=i%2==0) for i in range(30)]
    result = benchmark.describe_selection(SimpleNamespace(entries=entries),counts)
    assert result["indices"][:3] == [1,2,3]
    assert len(result["indices"]) == len(set(result["indices"])) == 16
    assert result["global_max_Na"] == result["global_max_Nb"] == 2048
    assert result["global_max_true_matrix_cells"] == 1900*1900
    assert result["formal_collation"]["padding_is_not_real_points"]
    assert not result["formal_collation"]["benchmark_trimming"]
    assert result["pairs"][0]["true_Nb"] == 100


def test_point_headers_without_mask_or_target_arrays(tmp_path):
    path = Path(tmp_path)/"header_only_fixture.npz"
    np.savez_compressed(path,points_rc_a=np.ones((37,2),np.float32),points_rc_b=np.ones((2048,2),np.float32))
    assert benchmark.point_header_counts(path) == [37,2048]
    invalid=Path(tmp_path)/"invalid.npz"
    np.savez_compressed(invalid,points_rc_a=np.ones((2049,2),np.float32),points_rc_b=np.ones((17,2),np.float32))
    with raises(ValueError,"shape/type"):
        benchmark.point_header_counts(invalid)


def test_default_is_cpu_plan_and_candidates_remain_bounded_fixed_effective16():
    args=benchmark.parser().parse_args(["--checkpoint","M12.pt","--train-manifest","train.json","--output","new"])
    assert not args.run_cuda and args.worker_phase is None
    assert benchmark.PHYSICAL == (1,2,4) and benchmark.EFFECTIVE == 16 and benchmark.CAP == 2048
    assert args.measured_groups == 2
    assert args.matrix_head_revision == "per_pair_norm_v3"


def test_empty_imported_M12_optimizer_is_not_accepted_for_capacity():
    fixture=dict(phase="matcher",epoch=12,completed_segments=48,
        optimizer_state_dict=dict(param_groups=[dict(phase_family="base",params=[0]),
            dict(phase_family="new_head",params=[1])],state={0:{"exp_avg": 0.}}))
    benchmark.require_capacity_checkpoint(fixture)
    fixture["optimizer_state_dict"]["state"]={}
    with raises(ValueError,"empty-optimizer"):
        benchmark.require_capacity_checkpoint(fixture)


def test_highest_areas_all_negative_still_include_largest_positive():
    counts = [[1900+i,1800+i] for i in range(20)] + [[50+i,70+i] for i in range(10)]
    entries = [dict(pair_id="fixture%d" % i, label=i >= 20) for i in range(30)]
    selection = benchmark.describe_selection(SimpleNamespace(entries=entries), counts)
    assert 19 in selection["indices"] and 29 in selection["indices"]
    assert selection["positive_pair_count"] >= 1 and selection["negative_pair_count"] >= 1
    assert selection["class_maximum_true_matrix_cells"]["positive"] == 59*79
    assert selection["train_pair_labels_used_for_stratification"]
    assert not selection["model_scores_or_layout_GT_used_for_selection"]
    with raises(ValueError, "both positive and negative"):
        benchmark.stress_indices(counts, [False]*30)


def worker_plan():
    entries = [dict(pair_id="fixture%d"%i, label=i%2 == 0) for i in range(30)]
    return dict(schema_version=benchmark.SCHEMA, matrix_head_revision="per_pair_norm_v3", effective_batch=16,
        physical_candidates=[1,2,4], phases=["matcher","classifier"], measured_groups=2,
        selection=benchmark.describe_selection(SimpleNamespace(entries=entries), [[100+i,120+i] for i in range(30)]))


def test_worker_revision_cli_passthrough_and_plan_mismatch_rejected(tmp_path):
    args = benchmark.parser().parse_args(["--checkpoint","M12.pt","--train-manifest","train.json",
        "--output","new","--matrix-head-revision","per_pair_norm_v3"])
    command = benchmark.worker_command(args, tmp_path / "worker", tmp_path / "selection.json", 2, "matcher")
    parsed = benchmark.parser().parse_args(command[4:])
    assert parsed.matrix_head_revision == "per_pair_norm_v3" and parsed.worker_physical == 2
    assert parsed.worker_phase == "matcher" and parsed.selection_file == str(tmp_path / "selection.json")
    plan = worker_plan()
    benchmark.require_worker_plan(parsed, plan)
    for key, bad in (("matrix_head_revision","bn_relu_pool_v2"), ("schema_version","old-schema"),
                     ("effective_batch",32)):
        changed = deepcopy(plan); changed[key] = bad
        with raises(ValueError, "plan differs"):
            benchmark.require_worker_plan(parsed, changed)
    changed = deepcopy(plan)
    for row in changed["selection"]["pairs"]: row["label"] = False
    with raises(ValueError, "class-covered"):
        benchmark.require_worker_plan(parsed, changed)


def test_cpu_plan_scans_headers_once_and_reuse_never_rescans(tmp_path, monkeypatch):
    manifest, checkpoint = tmp_path / "train.json", tmp_path / "M12.pt"
    manifest.write_text("CPU fixture TRAIN identity")
    checkpoint.write_bytes(b"CPU plan checksum fixture; not a training checkpoint")
    class FakeStep:
        def __init__(self, path, sampling):
            assert sampling == "step3"
            self.root, self.identity, self.split, self.contour_cap = tmp_path, "fixture-dataset", "train", 2048
            self.entries = [dict(pair_id="train-%d" % i, label=i%2 == 0, artifact_path="%d.npz" % i) for i in range(24000)]
        def __len__(self): return 24000
    monkeypatch.setattr(benchmark, "StepSourceDataset", FakeStep)
    calls = []
    def header(path):
        calls.append(path)
        return [100,200]
    monkeypatch.setattr(benchmark, "point_header_counts", header)
    monkeypatch.setattr(benchmark, "require_idle_gpu", lambda: (_ for _ in ()).throw(AssertionError("no GPU")))
    base = ["--checkpoint",str(checkpoint),"--train-manifest",str(manifest)]
    first = benchmark.run(benchmark.parser().parse_args(base + ["--output",str(tmp_path / "first")]))
    assert len(calls) == 24000 and first["CPU_header_scan_performed"]
    saved = (tmp_path / "first/selection.json").read_bytes()
    def forbidden(path): raise AssertionError("must reuse previous header scan")
    monkeypatch.setattr(benchmark, "point_header_counts", forbidden)
    second = benchmark.run(benchmark.parser().parse_args(base + ["--output",str(tmp_path / "second"),
        "--selection-file",str(tmp_path / "first/selection.json")]))
    assert not second["CPU_header_scan_performed"] and second["CPU_header_scan_elapsed_s"] == 0.
    assert second["selection"] == first["selection"]
    assert second["source_CPU_plan_sha256"] == benchmark.sha(tmp_path / "first/selection.json")
    assert (tmp_path / "first/selection.json").read_bytes() == saved
    assert not (tmp_path / "second/train_shape_index.json").exists()
    for index, (field, bad) in enumerate((("matrix_head_revision", "bn_relu_pool_v2"),
                                        ("manifest_sha256", "changed"))):
        altered = deepcopy(first); altered[field] = bad
        path = tmp_path / ("bad%d.json" % index); path.write_text(json.dumps(altered))
        with raises(ValueError, "registered TRAIN"):
            benchmark.run(benchmark.parser().parse_args(base + ["--output",str(tmp_path / ("outbad%d" % index)),
                "--selection-file",str(path)]))
