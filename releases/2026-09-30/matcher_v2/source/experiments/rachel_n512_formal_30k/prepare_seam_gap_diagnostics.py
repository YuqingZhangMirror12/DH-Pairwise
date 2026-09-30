"""CPU-only, prediction-independent GT-seam damage manifest for frozen TEST.

The existing stress evaluation's `actual_changed` includes erosion anywhere on
either fragment. This diagnostic distinguishes actual material loss along a
known GT seam. It neither fits a model nor changes any saved prediction. A
single manifest is shared by every model at the same condition; unmeasurable
positive seams remain explicitly unknown instead of being labelled undamaged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.utils.data import DataLoader, Subset

from experiments.rachel_n512_formal_30k.run_all_weathering_ablations import now, save_json
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairDataset
from staging.pairwise_v0_2.pairwise_data.rachel_gap_stress import RachelGapStressDataset


SCHEMA = "rachel-seam-gap-manifest/1"
SEED, TEST_COUNT, POSITIVE_COUNT = 260910, 3000, 1500


class CaptureClean:
    """Retain the exact clean object already read by this worker's wrapper."""
    def __init__(self, base):
        self.base, self.root, self.split = base, base.root, base.split
        self.latest = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        self.latest = self.base[index]
        return self.latest


class DiagnosticDataset:
    def __init__(self, base, depth, cache_dir=None):
        self.capture = CaptureClean(base)
        self.stress = RachelGapStressDataset(self.capture, max_depth_px=depth, seed=SEED, cache_dir=cache_dir)

    def __len__(self):
        return len(self.stress)

    def __getitem__(self, index):
        from staging.pairwise_v0_2.pairwise_data.rachel_seam_gap_diagnostics import seam_gap_diagnostics
        wrapped = self.stress[index]
        clean = self.capture.latest
        if clean.pair_id != wrapped.student.pair_id:
            raise ValueError("clean metric source and stress input are not the same pair")
        return dict(pair_id=clean.pair_id, fragment_a=clean.fragment_a_token,
            fragment_b=clean.fragment_b_token, label=bool(clean.label),
            actual_changed=bool(wrapped.report["changed_pair"]),
            diagnostics=seam_gap_diagnostics(clean, wrapped.student))


def identity(value):
    return value


def summarize_manifest_rows(rows):
    positive = [r for r in rows if r["label"]]
    damaged = [r for r in positive if r["diagnostics"].get("actual_seam_damaged") is True]
    measured = [r for r in positive if r["diagnostics"].get("valid") is True]
    unknown = [r for r in positive if r["diagnostics"].get("valid") is not True]
    unchanged = [r for r in measured if r["diagnostics"].get("actual_seam_damaged") is False]
    reasons = Counter(str(r["diagnostics"].get("invalid_reason")) for r in unknown)
    if len(damaged) + len(unchanged) != len(measured):
        raise ValueError("measured positive must have a true or false seam-damage decision")
    return dict(pair_count=len(rows), positive_count=len(positive), negative_count=len(rows) - len(positive),
        any_edge_changed_positive_count=sum(r["actual_changed"] for r in positive),
        measured_positive_count=len(measured), actual_seam_damaged_positive_count=len(damaged),
        measured_below_seam_damage_criterion_positive_count=len(unchanged),
        unmeasured_positive_count=len(unknown), unmeasured_reasons=dict(reasons),
        positive_measurement_coverage=len(measured) / len(positive) if positive else None,
        measured_damaged_coverage_of_all_positives=len(damaged) / len(positive) if positive else None,
        groups_defined_without_model_predictions=True)


def run(args):
    if args.workers < 0 or not 0 <= args.limit <= TEST_COUNT:
        raise ValueError("workers >=0, limit 0 (full) or 1..3000 (prefix probe)")
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    status = dict(schema_version=SCHEMA, status="running", started_at=now(), pid=os.getpid(),
        seed=SEED, max_depth_px=args.depth, full_test=args.limit == 0, probe_only=args.limit != 0,
        evaluated_pair_count=args.limit or TEST_COUNT, total_test_pair_count=TEST_COUNT,
        dataset_root=str(Path(args.dataset).resolve()), cpu_only=True, loads_checkpoint=False,
        changes_predictions=False, refits_thresholds=False, processed=0)
    save_json(destination / "status.json", status)
    try:
        torch.set_num_threads(1)
        base = RachelPairDataset(args.dataset, "test")
        manifest_path = Path(args.dataset) / "pairs" / "test.jsonl"
        with manifest_path.open() as stream:
            manifest = [json.loads(line) for line in stream if line.strip()]
        if len(base) != TEST_COUNT or len(manifest) != TEST_COUNT or sum(bool(r["label"]) for r in manifest) != POSITIVE_COUNT:
            raise ValueError("Original balanced TEST3000 is required")
        expected = manifest[:status["evaluated_pair_count"]]
        dataset = DiagnosticDataset(base, args.depth, args.cache_dir)
        loader = DataLoader(Subset(dataset, list(range(len(expected)))), batch_size=None, shuffle=False,
            num_workers=args.workers, collate_fn=identity)
        started = time.perf_counter()
        with (destination / "diagnostics_rows.jsonl").open("x", encoding="utf-8") as stream:
            for row in loader:
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if len(rows) % 128 == 0 or len(rows) == len(expected):
                    stream.flush()
                    status.update(processed=len(rows), elapsed_s=time.perf_counter() - started)
                    save_json(destination / "status.json", status)
                    print(json.dumps({k:status[k] for k in ("status", "max_depth_px", "processed", "evaluated_pair_count", "elapsed_s")}), flush=True)
            os.fsync(stream.fileno())
        if [(r["pair_id"], r["label"]) for r in rows] != [(r["pair_id"], bool(r["label"])) for r in expected]:
            raise ValueError("Diagnostic sidecars must retain original TEST order and labels")
        summary = summarize_manifest_rows(rows)
        status.update(status="complete", completed_at=now(), elapsed_s=time.perf_counter() - started,
            test_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        result = dict(status, summary=summary, rows=rows,
            interpretation=["GT-seam material-loss grouping, not a classifier or a pose solver.",
                "Fixed raster normal-probe approximations, not measured physical gap widths.",
                "Unmeasured positive seams remain explicit and never count as undamaged.",
                "Use the same depth manifest for all models; no prediction-dependent selection."])
        save_json(destination / "diagnostics.json", result)
        save_json(destination / "status.json", status)
        save_json(destination / "summary.json", dict(status, summary=summary))
        print(json.dumps(dict(status="complete", output=str(destination), summary=summary)), flush=True)
        return result
    except Exception as error:
        status.update(status="failed", error=repr(error), failed_at=now(), processed=len(rows))
        save_json(destination / "status.json", status)
        raise


def run_all(args):
    """One finite CPU preparation job for all three fixed conditions."""
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    if state_path.exists():
        raise FileExistsError("Refusing to duplicate an existing seam diagnostic preparation")
    state = dict(schema_version="rachel-seam-gap-preparation-pipeline/1", status="running",
        pid=os.getpid(), started_at=now(), active_depth=None, cpu_only=True,
        stages=[dict(depth=d, status="queued", output=str(root / ("depth" + str(d)))) for d in (0, 2, 4)])
    save_json(state_path, state)
    try:
        for stage in state["stages"]:
            stage.update(status="running", started_at=now())
            state["active_depth"] = stage["depth"]
            save_json(state_path, state)
            single = argparse.Namespace(**dict(vars(args), depth=stage["depth"], output=stage["output"]))
            result = run(single)
            stage.update(status="complete", completed_at=now(), summary=result["summary"])
            save_json(state_path, state)
        state.update(status="complete", active_depth=None, completed_at=now())
        save_json(state_path, state)
        return state
    except Exception as error:
        state.update(status="failed", failed_at=now(), error=repr(error))
        save_json(state_path, state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--depth", type=int, choices=(0, 2, 4))
    selection.add_argument("--all-depths", action="store_true", help="finite CPU preparation of 0/2/4 conditions")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (run_all if args.all_depths else run)(args)
