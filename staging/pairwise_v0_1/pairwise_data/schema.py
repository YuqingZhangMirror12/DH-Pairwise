"""Typed records for the Dunhuang Pairwise Benchmark v0.1 data layer.

The schema deliberately separates a directed observation (``A -> B``) from its
canonical unordered key.  Historical MM labels are directional, while split
and duplicate checks must treat ``(A, B)`` and ``(B, A)`` as the same physical
pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Union


SCHEMA_VERSION = "pairwise-benchmark/0.1"


class DirectedRelation(str, Enum):
    """Position of fragment B relative to fragment A."""

    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"

    def inverse(self) -> "DirectedRelation":
        return {
            DirectedRelation.LEFT: DirectedRelation.RIGHT,
            DirectedRelation.RIGHT: DirectedRelation.LEFT,
            DirectedRelation.ABOVE: DirectedRelation.BELOW,
            DirectedRelation.BELOW: DirectedRelation.ABOVE,
        }[self]


class ConditionParseError(ValueError):
    """Raised when a historical MM condition has no declared interpretation."""


@dataclass(frozen=True)
class ConditionLabel:
    is_adjacent: bool
    direction_b_wrt_a: Optional[DirectedRelation]

    def __post_init__(self) -> None:
        if self.is_adjacent != (self.direction_b_wrt_a is not None):
            raise ValueError("adjacent labels require a direction; negatives require null")


# This is intentionally an explicit, closed mapping.  In particular, a truthy
# Python interpretation of the CSV string "False" is never used.
_MM_CONDITION_MAP = {
    "false": ConditionLabel(False, None),
    "left-right": ConditionLabel(True, DirectedRelation.RIGHT),
    "up-down": ConditionLabel(True, DirectedRelation.BELOW),
}


def parse_mm_condition(value: object) -> ConditionLabel:
    """Map a canonical MM CSV condition to benchmark fields.

    ``left-right`` means that pair_2 is right of pair_1.  ``up-down`` means
    pair_2 is below pair_1.  Unknown values fail closed rather than silently
    becoming positive labels.
    """

    key = str(value).strip().casefold()
    try:
        return _MM_CONDITION_MAP[key]
    except KeyError as exc:
        raise ConditionParseError("unsupported MM condition: {!r}".format(value)) from exc


def _join_id(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


@dataclass(frozen=True, order=True)
class SourceKey:
    dataset_root: str
    source_id: str

    @property
    def id(self) -> str:
        return _join_id(self.dataset_root, self.source_id)


@dataclass(frozen=True, order=True)
class GroupKey:
    source: SourceKey
    group_id: str

    @property
    def id(self) -> str:
        return _join_id(self.source.id, self.group_id)


@dataclass(frozen=True, order=True)
class VariantKey:
    group: GroupKey
    variant_id: str

    @property
    def id(self) -> str:
        return _join_id(self.group.id, self.variant_id)


@dataclass(frozen=True, order=True)
class FragmentRef:
    variant: VariantKey
    fragment_name: str
    member_path: str

    @property
    def fragment_id(self) -> str:
        return _join_id(self.variant.id, self.fragment_name)


PairKey = Tuple[str, str]


def canonical_unordered_pair_key(
    fragment_a: Union[FragmentRef, str], fragment_b: Union[FragmentRef, str]
) -> PairKey:
    """Return a stable key shared by both directed orders of a pair."""

    a_id = fragment_a.fragment_id if isinstance(fragment_a, FragmentRef) else str(fragment_a)
    b_id = fragment_b.fragment_id if isinstance(fragment_b, FragmentRef) else str(fragment_b)
    return tuple(sorted((a_id, b_id)))  # type: ignore[return-value]


@dataclass(frozen=True)
class PairRecord:
    fragment_a: FragmentRef
    fragment_b: FragmentRef
    is_adjacent: bool
    direction_b_wrt_a: Optional[DirectedRelation]
    canonical_pair_key: PairKey
    condition_raw: str
    csv_member_path: str
    csv_row_number: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.fragment_a.fragment_id == self.fragment_b.fragment_id:
            raise ValueError("self-pairs are not valid pair records")
        expected_key = canonical_unordered_pair_key(self.fragment_a, self.fragment_b)
        if self.canonical_pair_key != expected_key:
            raise ValueError("canonical_pair_key does not match the fragments")
        if self.is_adjacent != (self.direction_b_wrt_a is not None):
            raise ValueError("adjacent records require a direction; negatives require null")

    @property
    def variant(self) -> VariantKey:
        return self.fragment_a.variant

    @property
    def directed_pair_key(self) -> Tuple[str, str]:
        return (self.fragment_a.fragment_id, self.fragment_b.fragment_id)

    def reversed(self) -> "PairRecord":
        """Return the reverse directed view without changing physical identity."""

        return PairRecord(
            fragment_a=self.fragment_b,
            fragment_b=self.fragment_a,
            is_adjacent=self.is_adjacent,
            direction_b_wrt_a=(
                self.direction_b_wrt_a.inverse()
                if self.direction_b_wrt_a is not None
                else None
            ),
            canonical_pair_key=self.canonical_pair_key,
            condition_raw=self.condition_raw,
            csv_member_path=self.csv_member_path,
            csv_row_number=self.csv_row_number,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class GroupRef:
    key: GroupKey
    csv_member_path: Optional[str]
    variant_ids: Tuple[str, ...]


@dataclass(frozen=True)
class VariantRef:
    key: VariantKey
    image_members: Tuple[str, ...]


@dataclass(frozen=True)
class ParserIssue:
    code: str
    message: str
    member_path: str
    row_number: Optional[int] = None
    variant_id: Optional[str] = None


@dataclass
class MMProfile:
    source_count: int = 0
    group_count: int = 0
    variant_count: int = 0
    csv_count: int = 0
    png_count: int = 0
    csv_row_count: int = 0
    emitted_pair_count: int = 0
    adjacent_pair_count: int = 0
    negative_pair_count: int = 0
    direction_counts: Dict[str, int] = field(default_factory=dict)
    self_pair_row_count: int = 0
    invalid_condition_row_count: int = 0
    malformed_csv_row_count: int = 0
    missing_image_reference_count: int = 0
    unreferenced_png_count: int = 0
    duplicate_csv_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
