"""Prepare the single-source gen4 30k Pairwise dataset in one command.

The command creates the fixed 24k/3k/3k balanced pair manifests and matched
filled/contour10 PNG views.  It never reads RGB imagery and never starts model
training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Mapping, Optional, Sequence

from .gen4_pairwise30k import (
    DEFAULT_SEED,
    Gen4Pairwise30kConfig,
    SplitQuota,
    build_gen4_pairwise30k_manifest,
    write_gen4_pairwise30k_manifest,
)
from .gen4_representations import materialize_gen4_representations


def prepare_gen4_pairwise30k(
    *,
    mask_root: Path,
    output_root: Path,
    seed: str = DEFAULT_SEED,
    expected_parent_count: int = 8_000,
    parent_canvas_size: int = 800,
    split_quotas: Optional[Mapping[str, SplitQuota]] = None,
) -> Path:
    """Build pair manifests and both representations under one fresh root."""

    destination = Path(output_root)
    if destination.exists():
        raise ValueError(
            "output_root must be fresh and absent: " + destination.as_posix()
        )
    config_arguments = {
        "mask_root": Path(mask_root),
        "seed": seed,
        "expected_parent_count": expected_parent_count,
        "parent_canvas_size": parent_canvas_size,
    }
    if split_quotas is not None:
        config_arguments["split_quotas"] = split_quotas
    config = Gen4Pairwise30kConfig(
        **config_arguments,
    )
    manifest = build_gen4_pairwise30k_manifest(config)
    materialize_gen4_representations(
        manifest,
        mask_root=config.mask_root,
        output_root=destination,
    )
    try:
        write_gen4_pairwise30k_manifest(manifest, destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--expected-parent-count", type=int, default=8_000)
    parser.add_argument("--parent-canvas-size", type=int, default=800)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = prepare_gen4_pairwise30k(
        mask_root=arguments.mask_root,
        output_root=arguments.output_root,
        seed=arguments.seed,
        expected_parent_count=arguments.expected_parent_count,
        parent_canvas_size=arguments.parent_canvas_size,
    )
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "gen4_pairwise30k_prepared",
                "output_root": output.as_posix(),
                "split_counts": protocol["split_counts"],
                "representations": {
                    "filled": (output / "representations" / "filled").as_posix(),
                    "contour10": (output / "representations" / "contour10").as_posix(),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "prepare_gen4_pairwise30k"]
