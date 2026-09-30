"""CPU-only continuation from four union shards to completed new manifests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from staging.pairwise_v0_2.pairwise_data.rachel_composite_training import build_composites


def run(data_root, release_root, shard_pids, *, timeout_seconds=28800):
    data_root = Path(data_root).resolve()
    state_path = data_root / "data_build_state.json"
    destination = data_root / "composite_v1"
    shards = [data_root / ("union_tiny_shard%d" % i) for i in range(4)]
    started = time.monotonic()
    base = dict(schema_version="rachel-overnight-data-build-state/1", controller_pid=os.getpid(),
                shard_pids=shard_pids, shard_roots=[str(p) for p in shards],
                output_root=str(destination), source_root=str(release_root), gpu_used=False)

    def write(status, **values):
        record = dict(base, status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(state_path)
        print(json.dumps(record), flush=True)

    try:
        previous_ready = None
        while True:
            ready = [p for p in shards if (p / "summary.json").is_file()]
            if len(ready) == 4:
                break
            if previous_ready != len(ready):
                write("waiting_for_union_shards", completed_shards=len(ready))
                previous_ready = len(ready)
            for root, pid in zip(shards, shard_pids):
                if root not in ready:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        raise RuntimeError("union shard process ended without summary: " + str(root))
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("union shards did not complete within the explicit CPU build deadline")
            time.sleep(20)
        write("building_composite", completed_shards=4)
        summary = build_composites(release_root, destination, shards)
        # All manifest writes and summary must finish before main can train.
        write("complete", data_quality_status=summary["status"],
              train60k_manifest=str(destination / "train_60k.json"),
              matched24k_manifest=str(destination / "train_matched24k.json"),
              original24k_manifest=str(destination / "train_original24k.json"),
              summary_path=str(destination / "summary.json"),
              selected_tiny_positive=summary["selected_tiny_positive"],
              required_tiny_positive=summary["required_tiny_positive"])
    except Exception as error:
        write("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--shard-pids", type=int, nargs=4, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    args = parser.parse_args()
    run(args.data_root, args.release_root, args.shard_pids, timeout_seconds=args.timeout_seconds)
