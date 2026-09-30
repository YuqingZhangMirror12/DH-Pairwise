"""Deterministic, lineage-disjoint Rachel 30k pair selection.

The preprocessing stage owns RGB-to-mask conversion and within-folder labels.
This module only assigns source-image lineages, selects eligible positive and
hard-negative rows, and constructs scale-matched cross-lineage negatives.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "rachel-pairwise-30k-selection/1.0"
DEFAULT_SEED = "rachel-pairwise-n512-v1"
SPLITS = ("train", "val", "test")


class RachelSelectionError(ValueError):
    """The normalized Rachel population cannot satisfy the protocol."""


class RachelSelectionShortfall(RachelSelectionError):
    """One or more requested cells cannot be filled without leakage."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value


def _json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    copied = dict(value)
    json.dumps(copied, sort_keys=True, allow_nan=False)
    return MappingProxyType(copied)


def _ratio(a: float, b: float) -> float:
    return max(a, b) / min(a, b)


def _hash(seed: str, *parts: object) -> int:
    payload = "\0".join((seed,) + tuple(str(part) for part in parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


@dataclass(frozen=True)
class FragmentRecord:
    """One processed fragment. ``split_unit_id`` is Rachel ``image_name``."""

    fragment_token: str
    parent_group_id: str
    split_unit_id: str
    generator: str
    fragment_id: str
    foreground_area: float
    bbox_aspect_ratio: float
    model_mask_path: str
    contour_path: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "fragment_token",
            "parent_group_id",
            "split_unit_id",
            "generator",
            "fragment_id",
            "model_mask_path",
            "contour_path",
        ):
            _text(getattr(self, name), name)
        for name in ("foreground_area", "bbox_aspect_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        object.__setattr__(self, "foreground_area", float(self.foreground_area))
        object.__setattr__(self, "bbox_aspect_ratio", float(self.bbox_aspect_ratio))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "FragmentRecord":
        known = {
            "fragment_token", "parent_group_id", "group_id", "split_unit_id",
            "lineage_id", "image_name", "generator", "fragment_id",
            "foreground_area", "area", "bbox_aspect_ratio", "bbox_aspect",
            "model_mask_path", "contour_path", "metadata", "split",
            "target_audit",
        }
        return cls(
            fragment_token=str(row["fragment_token"]),
            parent_group_id=str(row.get("parent_group_id", row.get("group_id"))),
            split_unit_id=str(row.get("split_unit_id", row.get("lineage_id", row.get("image_name")))),
            generator=str(row["generator"]),
            fragment_id=str(row.get("fragment_id", row["fragment_token"])),
            foreground_area=float(row.get("foreground_area", row.get("area"))),
            bbox_aspect_ratio=float(row.get("bbox_aspect_ratio", row.get("bbox_aspect"))),
            model_mask_path=str(row["model_mask_path"]),
            contour_path=str(row["contour_path"]),
            metadata={**dict(row.get("metadata", {})), **{key: row[key] for key in row if key not in known}},
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "fragment_token": self.fragment_token,
            "parent_group_id": self.parent_group_id,
            "split_unit_id": self.split_unit_id,
            "image_name": self.split_unit_id,
            "generator": self.generator,
            "fragment_id": self.fragment_id,
            "foreground_area": self.foreground_area,
            "bbox_aspect_ratio": self.bbox_aspect_ratio,
            "model_mask_path": self.model_mask_path,
            "contour_path": self.contour_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WithinPairCandidate:
    """Rachel-native semantic positive or same-folder hard negative."""

    pair_id: str
    fragment_a_token: str
    fragment_b_token: str
    label: bool
    label_origin: str
    negative_origin: Optional[str] = None
    seam_length_px: Optional[float] = None
    main_training_eligible: bool = True
    selection_exclusion_reason: Optional[str] = None
    translation_a_to_b_rc: Optional[Tuple[float, float]] = None
    correspondence_path: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("pair_id", "fragment_a_token", "fragment_b_token", "label_origin"):
            _text(getattr(self, name), name)
        if self.fragment_a_token == self.fragment_b_token:
            raise RachelSelectionError("self pair is not allowed")
        if type(self.label) is not bool or type(self.main_training_eligible) is not bool:  # noqa: E721
            raise TypeError("label and main_training_eligible must be bool")
        if self.label and self.negative_origin is not None:
            raise RachelSelectionError("positive candidate cannot have negative_origin")
        if not self.label and self.negative_origin not in {
            "same_folder_hard", "cross_folder_scale_matched"
        }:
            raise RachelSelectionError("negative_origin is not recognized")
        if self.main_training_eligible and self.selection_exclusion_reason is not None:
            raise RachelSelectionError("eligible candidate cannot have exclusion reason")
        if not self.main_training_eligible:
            _text(self.selection_exclusion_reason, "selection_exclusion_reason")
        if self.seam_length_px is not None:
            if not math.isfinite(float(self.seam_length_px)) or float(self.seam_length_px) < 0:
                raise ValueError("seam_length_px must be finite and non-negative")
            object.__setattr__(self, "seam_length_px", float(self.seam_length_px))
        if self.translation_a_to_b_rc is not None:
            if len(self.translation_a_to_b_rc) != 2:
                raise ValueError("translation_a_to_b_rc must contain dr,dc")
            object.__setattr__(self, "translation_a_to_b_rc", tuple(float(v) for v in self.translation_a_to_b_rc))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata))

    @property
    def canonical_pair_key(self) -> Tuple[str, str]:
        return tuple(sorted((self.fragment_a_token, self.fragment_b_token)))

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> "WithinPairCandidate":
        known = {
            "candidate_id", "pair_id", "fragment_a_token", "fragment_b_token",
            "fragment_a_id", "fragment_b_id", "label", "label_origin",
            "negative_origin", "seam_length_px", "seam_length",
            "main_training_eligible", "selection_exclusion_reason",
            "translation_a_to_b_rc", "correspondence_path", "metadata", "split",
            "lineage_id", "image_name", "generator", "group_id",
        }
        return cls(
            pair_id=str(row.get("pair_id", row.get("candidate_id"))),
            fragment_a_token=str(row.get("fragment_a_token", row.get("fragment_a_id"))),
            fragment_b_token=str(row.get("fragment_b_token", row.get("fragment_b_id"))),
            label=bool(row["label"]),
            label_origin=str(row["label_origin"]),
            negative_origin=row.get("negative_origin"),
            seam_length_px=row.get("seam_length_px", row.get("seam_length")),
            main_training_eligible=bool(row.get("main_training_eligible", True)),
            selection_exclusion_reason=row.get("selection_exclusion_reason"),
            translation_a_to_b_rc=row.get("translation_a_to_b_rc"),
            correspondence_path=row.get("correspondence_path"),
            metadata={**dict(row.get("metadata", {})), **{key: row[key] for key in row if key not in known}},
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "fragment_a_token": self.fragment_a_token,
            "fragment_b_token": self.fragment_b_token,
            "label": self.label,
            "label_origin": self.label_origin,
            "negative_origin": self.negative_origin,
            "seam_length_px": self.seam_length_px,
            "main_training_eligible": self.main_training_eligible,
            "selection_exclusion_reason": self.selection_exclusion_reason,
            "translation_a_to_b_rc": list(self.translation_a_to_b_rc) if self.translation_a_to_b_rc is not None else None,
            "correspondence_path": self.correspondence_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RachelSplitQuota:
    positive: int
    same_folder_hard: int
    cross_folder_scale_matched: int

    def __post_init__(self) -> None:
        values = (self.positive, self.same_folder_hard, self.cross_folder_scale_matched)
        if any(type(value) is not int or value < 0 for value in values):  # noqa: E721
            raise ValueError("quota values must be non-negative ints")
        if self.same_folder_hard != self.cross_folder_scale_matched:
            raise ValueError("negative origins must be exactly 50:50")
        if self.positive != self.same_folder_hard + self.cross_folder_scale_matched:
            raise ValueError("positive and negative quotas must be 1:1")

    @property
    def total(self) -> int:
        return 2 * self.positive

    def to_dict(self) -> Dict[str, int]:
        return {
            "positive": self.positive,
            "same_folder_hard": self.same_folder_hard,
            "cross_folder_scale_matched": self.cross_folder_scale_matched,
            "negative": self.positive,
            "total": self.total,
        }


DEFAULT_SPLIT_QUOTAS = MappingProxyType({
    "train": RachelSplitQuota(12_000, 6_000, 6_000),
    "val": RachelSplitQuota(1_500, 750, 750),
    "test": RachelSplitQuota(1_500, 750, 750),
})


@dataclass(frozen=True)
class SelectedRachelPair:
    split: str
    candidate: WithinPairCandidate
    fragment_a: FragmentRecord
    fragment_b: FragmentRecord

    def to_dict(self) -> Dict[str, object]:
        row = self.candidate.to_dict()
        row["split"] = self.split
        row["fragment_a"] = self.fragment_a.to_dict()
        row["fragment_b"] = self.fragment_b.to_dict()
        return row


@dataclass(frozen=True)
class Rachel30kSelection:
    seed: str
    split_quotas: Mapping[str, RachelSplitQuota]
    lineage_assignments: Mapping[str, str]
    rows: Tuple[SelectedRachelPair, ...]

    def iter_jsonl_rows(self, split: Optional[str] = None) -> Iterator[Dict[str, object]]:
        for row in self.rows:
            if split is None or row.split == split:
                yield row.to_dict()

    def summary(self) -> Dict[str, object]:
        counts = Counter((row.split, "positive" if row.candidate.label else row.candidate.negative_origin) for row in self.rows)
        lineage_counts = Counter(self.lineage_assignments.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "total_pairs": len(self.rows),
            "split_quotas": {split: self.split_quotas[split].to_dict() for split in SPLITS},
            "lineage_counts": {split: lineage_counts[split] for split in SPLITS},
            "selected_counts": {
                split: {
                    "positive": counts[(split, "positive")],
                    "same_folder_hard": counts[(split, "same_folder_hard")],
                    "cross_folder_scale_matched": counts[(split, "cross_folder_scale_matched")],
                }
                for split in SPLITS
            },
        }


def _normalize_quotas(value: Optional[Mapping[str, RachelSplitQuota]]) -> Mapping[str, RachelSplitQuota]:
    quotas = DEFAULT_SPLIT_QUOTAS if value is None else value
    if set(quotas) != set(SPLITS) or any(not isinstance(quotas[split], RachelSplitQuota) for split in SPLITS):
        raise ValueError("split_quotas must define train/val/test RachelSplitQuota values")
    totals = [quotas[split].total for split in SPLITS]
    if totals[0] != 8 * totals[1] or totals[1] != totals[2]:
        raise ValueError("split quotas must have exact 8:1:1 total ratio")
    return MappingProxyType(dict(quotas))


def _validate_population(fragments: Sequence[FragmentRecord], candidates: Sequence[WithinPairCandidate]):
    if not fragments or not candidates:
        raise RachelSelectionError("fragment and candidate populations must be non-empty")
    by_token: Dict[str, FragmentRecord] = {}
    for fragment in fragments:
        if fragment.fragment_token in by_token:
            raise RachelSelectionError("duplicate fragment_token: " + fragment.fragment_token)
        by_token[fragment.fragment_token] = fragment
    ids = set()
    pairs = set()
    for candidate in candidates:
        if candidate.pair_id in ids:
            raise RachelSelectionError("duplicate pair_id: " + candidate.pair_id)
        ids.add(candidate.pair_id)
        if candidate.canonical_pair_key in pairs:
            raise RachelSelectionError("duplicate/reversed unordered pair: " + repr(candidate.canonical_pair_key))
        pairs.add(candidate.canonical_pair_key)
        try:
            a = by_token[candidate.fragment_a_token]
            b = by_token[candidate.fragment_b_token]
        except KeyError as error:
            raise RachelSelectionError("candidate references unknown fragment: " + str(error)) from error
        if a.split_unit_id != b.split_unit_id:
            raise RachelSelectionError("within-folder candidate crosses image_name lineage: " + candidate.pair_id)
        if a.parent_group_id != b.parent_group_id:
            raise RachelSelectionError("within-folder candidate crosses group: " + candidate.pair_id)
        if candidate.main_training_eligible and candidate.label:
            if candidate.seam_length_px is None or candidate.seam_length_px < 64:
                raise RachelSelectionError("eligible positive must have seam_length_px >= 64: " + candidate.pair_id)
            if candidate.translation_a_to_b_rc is None or not candidate.correspondence_path:
                raise RachelSelectionError("eligible positive lacks geometry supervision: " + candidate.pair_id)
        if candidate.main_training_eligible and not candidate.label:
            if "seam_match_count" not in candidate.metadata:
                raise RachelSelectionError("eligible hard negative lacks seam_match_count: " + candidate.pair_id)
            if int(candidate.metadata["seam_match_count"]) != 0:
                raise RachelSelectionError("hard negative has recovered seam matches: " + candidate.pair_id)
    return by_token, pairs


def _lineage_counts(n: int) -> Mapping[str, int]:
    if n < 3:
        raise RachelSelectionError("at least three image_name lineages are required")
    holdout = max(1, int(round(n / 10.0)))
    while 2 * holdout >= n:
        holdout -= 1
    return {"train": n - 2 * holdout, "val": holdout, "test": holdout}


def _assign_lineages(
    fragments: Sequence[FragmentRecord],
    candidates: Sequence[WithinPairCandidate],
    quotas: Mapping[str, RachelSplitQuota],
    seed: str,
) -> Mapping[str, str]:
    lineages = sorted({fragment.split_unit_id for fragment in fragments})
    required_counts = _lineage_counts(len(lineages))
    stats = {lineage: [0, 0, 0] for lineage in lineages}
    token_lineage = {fragment.fragment_token: fragment.split_unit_id for fragment in fragments}
    for fragment in fragments:
        stats[fragment.split_unit_id][2] += 1
    for candidate in candidates:
        if not candidate.main_training_eligible:
            continue
        lineage = token_lineage[candidate.fragment_a_token]
        stats[lineage][0 if candidate.label else 1] += 1
    totals = [sum(stats[lineage][index] for lineage in lineages) for index in range(3)]
    best = None

    def missing_score(loads):
        return sum(
            (max(0, quotas[split].positive - loads[split][0]) / max(1, quotas[split].positive)) ** 2
            + (max(0, quotas[split].same_folder_hard - loads[split][1]) / max(1, quotas[split].same_folder_hard)) ** 2
            for split in SPLITS
        )

    for trial in range(24):
        order = sorted(
            lineages,
            key=lambda lineage: (
                -max(stats[lineage][i] / max(1.0, totals[i]) for i in range(3)),
                -sum(stats[lineage]),
                _hash(seed, "lineage-order", trial, lineage),
            ),
        )
        loads = {split: [0, 0, 0] for split in SPLITS}
        counts = Counter()
        assignment = {}
        for lineage in order:
            choices = []
            for split in SPLITS:
                if counts[split] >= required_counts[split]:
                    continue
                projected = [loads[split][i] + stats[lineage][i] for i in range(3)]
                fraction = required_counts[split] / float(len(lineages))
                target = [max(1.0, totals[i] * fraction) for i in range(3)]
                cost = sum((projected[i] / target[i]) ** 2 - (loads[split][i] / target[i]) ** 2 for i in range(3))
                choices.append((cost, _hash(seed, "lineage-choice", trial, lineage, split), split))
            split = min(choices)[2]
            assignment[lineage] = split
            counts[split] += 1
            loads[split] = [loads[split][i] + stats[lineage][i] for i in range(3)]

        # Whole-lineage swaps preserve the integer 8:1:1 lineage counts while
        # repairing rare quota misses caused by a heavy source manuscript.
        for _ in range(32):
            current_missing = missing_score(loads)
            if current_missing == 0:
                break
            chosen = None
            for index, left in enumerate(lineages):
                left_split = assignment[left]
                for right in lineages[index + 1:]:
                    right_split = assignment[right]
                    if left_split == right_split:
                        continue
                    trial_loads = {split: list(loads[split]) for split in SPLITS}
                    trial_loads[left_split] = [
                        trial_loads[left_split][i] - stats[left][i] + stats[right][i]
                        for i in range(3)
                    ]
                    trial_loads[right_split] = [
                        trial_loads[right_split][i] - stats[right][i] + stats[left][i]
                        for i in range(3)
                    ]
                    score = missing_score(trial_loads)
                    if score >= current_missing:
                        continue
                    key = (score, _hash(seed, "lineage-repair", trial, left, right))
                    if chosen is None or key < chosen[0]:
                        chosen = (key, left, right, trial_loads)
            if chosen is None:
                break
            _, left, right, loads = chosen
            assignment[left], assignment[right] = assignment[right], assignment[left]
        deficits = sum(max(0, quotas[split].positive - loads[split][0]) + max(0, quotas[split].same_folder_hard - loads[split][1]) for split in SPLITS)
        balance = sum(
            abs(loads[split][i] / max(1.0, totals[i]) - required_counts[split] / float(len(lineages)))
            for split in SPLITS for i in range(3)
        )
        key = (deficits, balance, tuple(sorted(assignment.items())))
        if best is None or key < best[0]:
            best = (key, assignment, loads)
    assert best is not None
    if best[0][0]:
        raise RachelSelectionShortfall("no deterministic lineage 8:1:1 assignment satisfies positive/hard quotas")
    return MappingProxyType(dict(best[1]))


def _candidate_rank(seed: str, split: str, origin: str, candidate: WithinPairCandidate) -> Tuple[int, str]:
    return (_hash(seed, "select", split, origin, candidate.pair_id, *candidate.canonical_pair_key), candidate.pair_id)


def _cross_candidates(
    fragments: Sequence[FragmentRecord],
    split: str,
    requested: int,
    seed: str,
    existing_pairs: set,
) -> Tuple[WithinPairCandidate, ...]:
    ordered_area = sorted(fragments, key=lambda fragment: (fragment.foreground_area, fragment.fragment_token))
    areas = [fragment.foreground_area for fragment in ordered_area]
    anchors = sorted(fragments, key=lambda fragment: (_hash(seed, "cross-anchor", split, fragment.fragment_token), fragment.fragment_token))
    made = []
    made_keys = set()
    for anchor in anchors:
        lo = bisect.bisect_left(areas, anchor.foreground_area / 2.0)
        hi = bisect.bisect_right(areas, anchor.foreground_area * 2.0)
        width = hi - lo
        if width <= 1:
            continue
        start = _hash(seed, "cross-start", split, anchor.fragment_token) % width
        for delta in range(width):
            other = ordered_area[lo + ((start + delta) % width)]
            if other.fragment_token == anchor.fragment_token or other.split_unit_id == anchor.split_unit_id:
                continue
            if _ratio(anchor.bbox_aspect_ratio, other.bbox_aspect_ratio) > 2.0:
                continue
            key = tuple(sorted((anchor.fragment_token, other.fragment_token)))
            if key in existing_pairs or key in made_keys:
                continue
            a, b = (anchor, other) if anchor.fragment_token < other.fragment_token else (other, anchor)
            digest = hashlib.sha256((seed + "\0cross\0" + key[0] + "\0" + key[1]).encode("utf-8")).hexdigest()[:24]
            made.append(WithinPairCandidate(
                pair_id="rachel-cross-" + digest,
                fragment_a_token=a.fragment_token,
                fragment_b_token=b.fragment_token,
                label=False,
                label_origin="different_image_name_nonmatch",
                negative_origin="cross_folder_scale_matched",
                metadata={
                    "generated_negative_origin": "cross_folder_scale_matched",
                    "area_ratio": _ratio(a.foreground_area, b.foreground_area),
                    "bbox_aspect_ratio_ratio": _ratio(a.bbox_aspect_ratio, b.bbox_aspect_ratio),
                    "seam_match_count": 0,
                },
            ))
            made_keys.add(key)
            if len(made) == requested:
                return tuple(made)
            break
    raise RachelSelectionShortfall("{} cross-folder scale-matched negatives short in {} (found {})".format(requested, split, len(made)))


def build_rachel_30k_selection(
    fragments: Iterable[FragmentRecord],
    candidates: Iterable[WithinPairCandidate],
    *,
    seed: str = DEFAULT_SEED,
    split_quotas: Optional[Mapping[str, RachelSplitQuota]] = None,
    lineage_assignments: Optional[Mapping[str, str]] = None,
) -> Rachel30kSelection:
    """Build exact 24k/3k/3k rows with source-image-disjoint splits."""

    _text(seed, "seed")
    fragment_rows = tuple(fragments)
    candidate_rows = tuple(candidates)
    by_token, existing_pairs = _validate_population(fragment_rows, candidate_rows)
    quotas = _normalize_quotas(split_quotas)
    lineages = {fragment.split_unit_id for fragment in fragment_rows}
    if lineage_assignments is None:
        assignments = _assign_lineages(fragment_rows, candidate_rows, quotas, seed)
    else:
        if set(lineage_assignments) != lineages or any(value not in SPLITS for value in lineage_assignments.values()):
            raise RachelSelectionError("lineage_assignments must cover every image_name exactly")
        assignments = MappingProxyType(dict(lineage_assignments))

    cells = defaultdict(list)
    for candidate in candidate_rows:
        if not candidate.main_training_eligible:
            continue
        split = assignments[by_token[candidate.fragment_a_token].split_unit_id]
        origin = "positive" if candidate.label else "same_folder_hard"
        cells[(split, origin)].append(candidate)
    shortages = []
    for split in SPLITS:
        for origin, need in (("positive", quotas[split].positive), ("same_folder_hard", quotas[split].same_folder_hard)):
            if len(cells[(split, origin)]) < need:
                shortages.append("{} {} requested={} available={}".format(split, origin, need, len(cells[(split, origin)])))
    if shortages:
        raise RachelSelectionShortfall("; ".join(shortages))

    rows = []
    for split in SPLITS:
        for origin, need in (("positive", quotas[split].positive), ("same_folder_hard", quotas[split].same_folder_hard)):
            for candidate in sorted(cells[(split, origin)], key=lambda value: _candidate_rank(seed, split, origin, value))[:need]:
                rows.append(SelectedRachelPair(split, candidate, by_token[candidate.fragment_a_token], by_token[candidate.fragment_b_token]))
        split_fragments = [fragment for fragment in fragment_rows if assignments[fragment.split_unit_id] == split]
        generated = _cross_candidates(split_fragments, split, quotas[split].cross_folder_scale_matched, seed, existing_pairs)
        for candidate in generated:
            rows.append(SelectedRachelPair(split, candidate, by_token[candidate.fragment_a_token], by_token[candidate.fragment_b_token]))
    rows.sort(key=lambda row: (SPLITS.index(row.split), _hash(seed, "row", row.split, row.candidate.pair_id), row.candidate.pair_id))
    return Rachel30kSelection(seed, quotas, assignments, tuple(rows))


def load_fragment_jsonl(path: Path) -> Tuple[FragmentRecord, ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return tuple(FragmentRecord.from_dict(json.loads(line)) for line in handle if line.strip())


def load_candidate_jsonl(path: Path) -> Tuple[WithinPairCandidate, ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return tuple(WithinPairCandidate.from_dict(json.loads(line)) for line in handle if line.strip())


def write_selection(selection: Rachel30kSelection, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / (split + ".jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for row in selection.iter_jsonl_rows(split):
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "lineage_splits.json").write_text(json.dumps(dict(sorted(selection.lineage_assignments.items())), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(selection.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_SEED", "DEFAULT_SPLIT_QUOTAS", "FragmentRecord", "Rachel30kSelection",
    "RachelSelectionError", "RachelSelectionShortfall", "RachelSplitQuota", "SCHEMA_VERSION",
    "SelectedRachelPair", "WithinPairCandidate", "build_rachel_30k_selection",
    "load_candidate_jsonl", "load_fragment_jsonl", "write_selection",
]
