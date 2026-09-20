"""Materialize the Rachel-only N=512 Pairwise training dataset.

The command writes only binary model masks and model-coordinate contour tensors
to the model-facing manifests. Parent-coordinate masks and offsets remain in a
separate target/audit tree and are never referenced by selected model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .rachel_30k_selection import (
    FragmentRecord,
    WithinPairCandidate,
    build_rachel_30k_selection,
    write_selection,
)
from .rachel_preprocess import (
    RachelPreprocessConfig,
    RachelPreprocessError,
    read_rachel_group,
)


SCHEMA_VERSION = "rachel-pairwise-n512-preprocessing/1.0"
DEFAULT_GENERATORS = (
    "gen2voronoi_1",
    "gen2voronoi_2",
    "gen3voronoi",
    "gen4voronoi",
    "gen4voronoi_1_3",
    "gen5voronoi_1_1_3",
)


class RachelMaterializationError(RuntimeError):
    """The Rachel dataset could not be materialized under the frozen contract."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)


def _atomic_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary,
        format="PNG",
        optimize=False,
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _token(generator: str, group_id: str, fragment_id: str) -> str:
    return "rachel/{}/{}/{}".format(generator, group_id, fragment_id)


def _pair_id(generator: str, group_id: str, first: str, second: str) -> str:
    digest = hashlib.sha256(
        "\0".join(("rachel-within", generator, group_id, first, second)).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return "rachel-within-" + digest


def _relative(*parts: str) -> str:
    return Path(*parts).as_posix()


def _process_group(task: Tuple[str, str, str, str, dict]) -> Dict[str, object]:
    source_text, work_text, generator, group_id, config_value = task
    source_root = Path(source_text)
    work_root = Path(work_text)
    marker_path = work_root / "groups" / generator / (group_id + ".json")
    if marker_path.is_file():
        return json.loads(marker_path.read_text(encoding="utf-8"))

    group_path = source_root / generator / "no_erode" / group_id
    config = RachelPreprocessConfig(**config_value)
    try:
        group = read_rachel_group(group_path, generator=generator, config=config)
    except (RachelPreprocessError, OSError, ValueError) as error:
        marker = {
            "schema_version": SCHEMA_VERSION,
            "status": "quarantined_group",
            "generator": generator,
            "group_id": group_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "fragments": [],
            "candidates": [],
        }
        _atomic_json(marker_path, marker)
        return marker

    fragment_rows = []
    for fragment in group.fragments:
        fragment_token = _token(generator, group_id, fragment.fragment_id)
        model_path = _relative(
            "model", "masks_800", generator, group_id, fragment.fragment_id + ".png"
        )
        contour_path = _relative(
            "model", "contours_n512", generator, group_id, fragment.fragment_id + ".npz"
        )
        parent_path = _relative(
            "targets", "parent_masks_800", generator, group_id, fragment.fragment_id + ".png"
        )
        _atomic_png(work_root / model_path, fragment.model.model_mask)
        _atomic_npz(
            work_root / contour_path,
            points_rc=np.asarray(fragment.model_contour, dtype=np.float32),
            valid=np.asarray(fragment.contour_valid, dtype=np.bool_),
        )
        _atomic_png(work_root / parent_path, fragment.parent_mask)
        fragment_rows.append(
            {
                "fragment_token": fragment_token,
                "parent_group_id": "rachel/{}/{}".format(generator, group_id),
                "split_unit_id": group.image_name,
                "image_name": group.image_name,
                "generator": generator,
                "group_id": group_id,
                "fragment_id": fragment.fragment_id,
                "mask_id": fragment.mask_id,
                "foreground_area": fragment.foreground_area,
                "bbox_aspect_ratio": fragment.bbox_aspect_ratio,
                "model_mask_path": model_path,
                "contour_path": contour_path,
                "target_audit": {
                    "parent_mask_path": parent_path,
                    "bbox_min_rc": list(fragment.model.bbox_min_rc),
                    "pad_start_rc": list(fragment.model.pad_start_rc),
                    "parent_to_model_offset_rc": list(
                        fragment.model.parent_to_model_offset_rc
                    ),
                },
            }
        )

    candidate_rows = []
    for pair in group.pairs:
        first_token = _token(generator, group_id, pair.fragment_a_id)
        second_token = _token(generator, group_id, pair.fragment_b_id)
        pair_id = _pair_id(
            generator,
            group_id,
            pair.fragment_a_id,
            pair.fragment_b_id,
        )
        target_path = None
        if pair.label:
            target_path = _relative(
                "targets", "pairs", generator, group_id, pair_id + ".npz"
            )
            translation_rc = np.asarray(pair.translation_a_to_b_rc, dtype=np.float32)
            translation_xy = np.asarray(pair.translation_a_to_b_xy, dtype=np.float32)
            _atomic_npz(
                work_root / target_path,
                correspondence_indices=np.asarray(
                    pair.token_correspondences, dtype=np.int64
                ),
                translation_a_to_b_rc=translation_rc,
                translation_a_to_b_xy_cartesian=translation_xy,
            )
        candidate_rows.append(
            {
                "pair_id": pair_id,
                "fragment_a_token": first_token,
                "fragment_b_token": second_token,
                "label": pair.label,
                "label_origin": (
                    "rachel_csv_neighbor_plus_independent_contour_seam"
                    if pair.label
                    else "rachel_csv_same_folder_nonneighbor_no_seam"
                ),
                "negative_origin": None if pair.label else "same_folder_hard",
                "seam_length_px": pair.seam_length_px,
                "main_training_eligible": pair.main_training_eligible,
                "selection_exclusion_reason": pair.selection_exclusion_reason,
                "translation_a_to_b_rc": (
                    list(pair.translation_a_to_b_rc)
                    if pair.translation_a_to_b_rc is not None
                    else None
                ),
                "translation_a_to_b_xy_cartesian": (
                    list(pair.translation_a_to_b_xy)
                    if pair.translation_a_to_b_xy is not None
                    else None
                ),
                "correspondence_path": target_path,
                "status": pair.status,
                "quarantine_reason": pair.quarantine_reason,
                "metadata": {
                    "generator": generator,
                    "group_id": group_id,
                    "image_name": group.image_name,
                    "seam_match_count": pair.accepted_seam_match_count,
                    "n512_correspondence_count": len(pair.token_correspondences),
                    "coordinate_conventions": {
                        "translation_a_to_b_rc": "(delta_row,delta_col)",
                        "translation_a_to_b_xy_cartesian": "(delta_col,-delta_row)",
                    },
                },
            }
        )

    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "processed_group",
        "generator": generator,
        "group_id": group_id,
        "image_name": group.image_name,
        "fragments": fragment_rows,
        "candidates": candidate_rows,
    }
    _atomic_json(marker_path, marker)
    return marker


def _inventory(source_root: Path, limit: Optional[int]) -> Tuple[Tuple[str, str], ...]:
    rows = []
    for generator in DEFAULT_GENERATORS:
        directory = source_root / generator / "no_erode"
        if not directory.is_dir():
            raise RachelMaterializationError("missing generator directory: " + str(directory))
        groups = sorted(
            (path.name for path in directory.iterdir() if path.is_dir()),
            key=lambda value: (len(value), value),
        )
        if limit is not None:
            groups = groups[:limit]
        rows.extend((generator, group_id) for group_id in groups)
    if not rows:
        raise RachelMaterializationError("Rachel source inventory is empty")
    return tuple(rows)


def _config_dict(config: RachelPreprocessConfig) -> dict:
    return asdict(config)


def _consolidate(
    work_root: Path,
    inventory: Sequence[Tuple[str, str]],
) -> Tuple[Tuple[dict, ...], Tuple[dict, ...], dict]:
    markers = []
    missing = []
    for generator, group_id in inventory:
        path = work_root / "groups" / generator / (group_id + ".json")
        if not path.is_file():
            missing.append("{}/{}".format(generator, group_id))
            continue
        markers.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise RachelMaterializationError(
            "missing group markers after processing: " + repr(missing[:10])
        )
    fragments = tuple(
        fragment
        for marker in markers
        if marker["status"] == "processed_group"
        for fragment in marker["fragments"]
    )
    candidates = tuple(
        candidate
        for marker in markers
        if marker["status"] == "processed_group"
        for candidate in marker["candidates"]
    )
    group_counts = Counter(marker["status"] for marker in markers)
    fragment_counts = Counter(row["generator"] for row in fragments)
    candidate_counts = Counter(
        (
            row["metadata"]["generator"],
            "positive" if row["label"] else "same_folder_hard",
            "eligible" if row["main_training_eligible"] else "excluded",
        )
        for row in candidates
    )
    seam_lengths = np.asarray(
        [row["seam_length_px"] for row in candidates if row["label"]],
        dtype=np.float64,
    )
    contour_counts = []
    for row in fragments:
        with np.load(work_root / row["contour_path"], allow_pickle=False) as archive:
            contour_counts.append(int(len(archive["points_rc"])))
    summary = {
        "group_counts": dict(group_counts),
        "fragment_count": len(fragments),
        "fragment_counts_by_generator": dict(fragment_counts),
        "candidate_count": len(candidates),
        "candidate_counts_by_generator_label_eligibility": {
            "|".join(key): value for key, value in sorted(candidate_counts.items())
        },
        "positive_seam_length_px_quantiles": (
            {
                str(q): float(np.quantile(seam_lengths, q))
                for q in (0.0, 0.05, 0.5, 0.95, 1.0)
            }
            if len(seam_lengths)
            else {}
        ),
        "contour_n512": {
            "count": len(contour_counts),
            "capped_fraction": (
                sum(value == 512 for value in contour_counts) / len(contour_counts)
                if contour_counts
                else 0.0
            ),
            "min": min(contour_counts) if contour_counts else None,
            "max": max(contour_counts) if contour_counts else None,
        },
    }
    return fragments, candidates, summary


def prepare_rachel_pairwise(
    *,
    source_root: Path,
    output_root: Path,
    workers: int = 12,
    resume: bool = False,
    limit_groups_per_generator: Optional[int] = None,
    seed: str = "rachel-pairwise-n512-v1",
) -> Path:
    source = Path(source_root).expanduser().resolve()
    final = Path(output_root).expanduser().resolve()
    work = final.with_name("." + final.name + ".partial")
    if final.exists():
        raise RachelMaterializationError("final output already exists: " + str(final))
    if work.exists() and not resume:
        raise RachelMaterializationError(
            "partial output exists; pass --resume after inspection: " + str(work)
        )
    if type(workers) is not int or workers <= 0:  # noqa: E721
        raise ValueError("workers must be a positive integer")
    if limit_groups_per_generator is not None and limit_groups_per_generator <= 0:
        raise ValueError("limit_groups_per_generator must be positive")

    config = RachelPreprocessConfig()
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "source_root": source.as_posix(),
        "output_root": final.as_posix(),
        "seed": seed,
        "generators": list(DEFAULT_GENERATORS),
        "preprocess_config": _config_dict(config),
        "limit_groups_per_generator": limit_groups_per_generator,
        "split_unit": "CSV image_name source-manuscript lineage",
        "model_input_contract": {
            "rgb_used": False,
            "binary_values_in_memory": [0, 1],
            "binary_values_on_disk": [0, 255],
            "canvas_shape": [800, 800],
            "parent_origin_exposed_to_model": False,
            "contour": "complete external CCW, Gaussian sigma=3, arc cap=512",
        },
    }
    if work.exists():
        existing = json.loads((work / "run_config.json").read_text(encoding="utf-8"))
        if existing != run_config:
            raise RachelMaterializationError("resume configuration does not match partial run")
    else:
        work.mkdir(parents=True)
        _atomic_json(work / "run_config.json", run_config)

    inventory = _inventory(source, limit_groups_per_generator)
    tasks = tuple(
        (
            source.as_posix(),
            work.as_posix(),
            generator,
            group_id,
            _config_dict(config),
        )
        for generator, group_id in inventory
    )
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for marker in executor.map(_process_group, tasks, chunksize=1):
            completed += 1
            if completed == 1 or completed % 250 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "status": "rachel_preprocess_progress",
                            "completed_groups": completed,
                            "total_groups": len(tasks),
                            "latest": "{}/{}".format(
                                marker["generator"], marker["group_id"]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    fragments, candidates, summary = _consolidate(work, inventory)
    _atomic_jsonl(work / "manifests" / "fragments.jsonl", fragments)
    _atomic_jsonl(work / "manifests" / "within_candidates.jsonl", candidates)
    _atomic_json(work / "qa" / "preprocess_summary.json", summary)

    selection_summary = None
    if limit_groups_per_generator is None:
        normalized_fragments = tuple(FragmentRecord.from_dict(row) for row in fragments)
        normalized_candidates = tuple(
            WithinPairCandidate.from_dict(row) for row in candidates
        )
        selection = build_rachel_30k_selection(
            normalized_fragments,
            normalized_candidates,
            seed=seed,
        )
        temporary_pairs = work / ".pairs.partial"
        if temporary_pairs.exists():
            shutil.rmtree(temporary_pairs)
        write_selection(selection, temporary_pairs)
        os.replace(temporary_pairs, work / "pairs")
        selection_summary = selection.summary()

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_rachel_pairwise_n512_30k"
            if selection_summary is not None
            else "complete_rachel_pairwise_n512_pilot"
        ),
        "source_authority": "Rachel RGB JPEG plus colocated label.csv only",
        "shredding_data_read": False,
        "preprocess": summary,
        "selection": selection_summary,
        "paths": {
            "model_masks": "model/masks_800",
            "model_contours": "model/contours_n512",
            "target_parent_masks": "targets/parent_masks_800",
            "target_pair_geometry": "targets/pairs",
            "fragments_manifest": "manifests/fragments.jsonl",
            "within_candidates_manifest": "manifests/within_candidates.jsonl",
            "selected_pairs": "pairs" if selection_summary is not None else None,
        },
    }
    _atomic_json(work / "preprocess_receipt.json", receipt)
    os.replace(work, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-groups-per-generator", type=int)
    parser.add_argument("--seed", default="rachel-pairwise-n512-v1")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    output = prepare_rachel_pairwise(
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        workers=arguments.workers,
        resume=arguments.resume,
        limit_groups_per_generator=arguments.limit_groups_per_generator,
        seed=arguments.seed,
    )
    receipt = json.loads(
        (output / "preprocess_receipt.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_root": output.as_posix(),
                "preprocess": receipt["preprocess"],
                "selection": receipt["selection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GENERATORS",
    "RachelMaterializationError",
    "SCHEMA_VERSION",
    "main",
    "prepare_rachel_pairwise",
]
