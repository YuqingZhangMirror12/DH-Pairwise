#!/usr/bin/env python3
"""Freeze C0-N-Q1 historical metadata without decoding masks or running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from staging.pairwise_v0_2.pairwise_data.historical_identity import (
    HistoricalIdentityIndex,
    assert_portable_receipt,
    freeze_c0_n_q1,
)
from staging.pairwise_v0_2.pairwise_data.training_stream import (
    iter_historical_pair_records,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MM_CACHE = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/mm_group_fingerprints.json"
)
DEFAULT_ECCV_CACHE = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/cache/eccv_group_fingerprints.json"
)
DEFAULT_SPLIT = (
    REPOSITORY_ROOT / "staging/pairwise_v0_1/data_gate/split_candidate_70_15_15.json"
)
DEFAULT_MM_ARCHIVE = Path(
    "/Users/yuqingzhang/Desktop/dataset/dunhuang_augmented_data.zip"
)
DEFAULT_ECCV_ARCHIVE = Path("/Users/yuqingzhang/Desktop/ECCV/code/1113data.tar.gz")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "c0_n_q1_freeze.json"
DEFAULT_LOCAL_NOTES = Path(__file__).resolve().parent / "c0_n_q1_freeze.local.md"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm-cache", type=Path, default=DEFAULT_MM_CACHE)
    parser.add_argument("--eccv-cache", type=Path, default=DEFAULT_ECCV_CACHE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--mm-archive", type=Path, default=DEFAULT_MM_ARCHIVE)
    parser.add_argument("--eccv-archive", type=Path, default=DEFAULT_ECCV_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-notes", type=Path, default=DEFAULT_LOCAL_NOTES)
    return parser


def main() -> int:
    args = _parser().parse_args()
    index = HistoricalIdentityIndex.from_files(
        mm_cache_path=args.mm_cache,
        eccv_cache_path=args.eccv_cache,
        split_path=args.split,
    )

    def records(split: str):
        return iter_historical_pair_records(
            split_manifest=args.split,
            split=split,
            mm_archive=args.mm_archive,
            eccv_archive=args.eccv_archive,
        )

    result = freeze_c0_n_q1(
        identity_index=index,
        validation_records=records("val"),
        training_records=records("train"),
        archive_paths={
            "mm_augmented": args.mm_archive,
            "eccv_1113data": args.eccv_archive,
        },
    )
    assert_portable_receipt(result.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    notes = [
        "# C0-N-Q1 local path notes",
        "",
        "This file is local-only and is not part of the portable receipt.",
        (
            "Historical-test assignment/split/cache metadata was parsed only for "
            "exclusion; no historical-test pair stream, archive member, identity "
            "index row, or mask pixel was read/entered. No sealed-real data or "
            "model was read/run."
        ),
        "",
        "- MM archive: `{}`".format(args.mm_archive.resolve()),
        "- ECCV archive: `{}`".format(args.eccv_archive.resolve()),
        "- MM fingerprint cache: `{}`".format(args.mm_cache.resolve()),
        "- ECCV fingerprint cache: `{}`".format(args.eccv_cache.resolve()),
        "- Historical split: `{}`".format(args.split.resolve()),
        "- Portable receipt: `{}`".format(args.output.resolve()),
        "",
    ]
    args.local_notes.write_text("\n".join(notes), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
