"""Read-only parser for the canonical MM augmented-data ZIP.

The archive is accessed through ``zipfile.ZipFile.open``.  No member is ever
extracted to disk, and labels are emitted only for rows explicitly present in
the group's ``pair.csv`` and whose two PNGs exist in the selected variant.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import (
    BinaryIO,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .schema import (
    ConditionParseError,
    FragmentRef,
    GroupKey,
    GroupRef,
    MMProfile,
    PairRecord,
    ParserIssue,
    SourceKey,
    VariantKey,
    VariantRef,
    canonical_unordered_pair_key,
    parse_mm_condition,
)


ZipSource = Union[str, os.PathLike, BinaryIO]


@dataclass(frozen=True)
class _ExtractedZipInfo:
    """The tiny ``ZipInfo`` surface used by :class:`MMCanonicalZip`."""

    filename: str

    def is_dir(self) -> bool:
        return False


class _ExtractedZipView:
    """Read an archive-shaped extracted tree without rebuilding a ZIP.

    The tree root is the directory corresponding to the archive root, so a
    relative path below it is exactly the original ZIP member name.  Only the
    ``infolist/open/close`` surface consumed by the existing parser is exposed;
    label parsing and member indexing therefore remain unchanged.
    """

    def __init__(self, root: Path):
        self._root = Path(root)
        if not self._root.is_dir():
            raise FileNotFoundError(self._root)
        members = []
        for directory, child_directories, filenames in os.walk(self._root):
            child_directories.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                member = path.relative_to(self._root).as_posix()
                members.append(_ExtractedZipInfo(member))
        self._members = tuple(members)
        self._closed = False

    def infolist(self) -> Tuple[_ExtractedZipInfo, ...]:
        if self._closed:
            raise ValueError("extracted ZIP view is closed")
        return self._members

    def open(self, member: str, mode: str = "r") -> BinaryIO:
        if self._closed:
            raise ValueError("extracted ZIP view is closed")
        if mode != "r":
            raise ValueError("extracted ZIP view is read-only")
        relative = PurePosixPath(str(member))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe extracted ZIP member")
        path = self._root.joinpath(*relative.parts)
        if not path.is_file():
            raise KeyError(str(member))
        return path.open("rb")

    def close(self) -> None:
        self._closed = True


@dataclass(frozen=True)
class _CsvRow:
    pair_1: str
    pair_2: str
    condition: str
    row_number: int


@dataclass
class _GroupIndex:
    key: GroupKey
    csv_members: List[str] = field(default_factory=list)
    variants: DefaultDict[str, Dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )


def _fragment_token(value: object) -> str:
    token = str(value).strip()
    path = PurePosixPath(token)
    if path.suffix.casefold() == ".png":
        return path.stem
    return path.name


class MMCanonicalZip:
    """Index and iterate the canonical MM ZIP without extracting it."""

    def __init__(self, source: ZipSource):
        if isinstance(source, (str, os.PathLike)) and Path(source).is_dir():
            self._zip = _ExtractedZipView(Path(source))
        else:
            self._zip = zipfile.ZipFile(source, mode="r")
        self._closed = False
        self._groups: Dict[GroupKey, _GroupIndex] = {}
        self._png_count = 0
        self._build_member_index()

    def __enter__(self) -> "MMCanonicalZip":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._zip.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("MMCanonicalZip is closed")

    @staticmethod
    def _group_from_csv(member_path: str) -> Optional[GroupKey]:
        parts = PurePosixPath(member_path).parts
        if len(parts) < 4 or parts[-1].casefold() != "pair.csv":
            return None
        return GroupKey(
            source=SourceKey(dataset_root="/".join(parts[:-3]), source_id=parts[-3]),
            group_id=parts[-2],
        )

    @staticmethod
    def _image_location(
        member_path: str,
    ) -> Optional[Tuple[GroupKey, str, str]]:
        path = PurePosixPath(member_path)
        parts = path.parts
        if len(parts) < 5 or path.suffix.casefold() != ".png":
            return None
        group = GroupKey(
            source=SourceKey(dataset_root="/".join(parts[:-4]), source_id=parts[-4]),
            group_id=parts[-3],
        )
        return group, parts[-2], path.stem

    def _build_member_index(self) -> None:
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            member_path = info.filename.strip("/")
            csv_group = self._group_from_csv(member_path)
            if csv_group is not None:
                group = self._groups.setdefault(csv_group, _GroupIndex(csv_group))
                group.csv_members.append(member_path)
                continue
            image_location = self._image_location(member_path)
            if image_location is None:
                continue
            group_key, variant_id, fragment_name = image_location
            group = self._groups.setdefault(group_key, _GroupIndex(group_key))
            # A duplicate member name cannot create an extra training example.
            group.variants[variant_id][fragment_name] = member_path
            self._png_count += 1

    @staticmethod
    def _matches(
        group_key: GroupKey,
        source: Optional[Union[str, SourceKey]],
        group: Optional[Union[str, GroupKey]],
    ) -> bool:
        if isinstance(source, SourceKey):
            if group_key.source != source:
                return False
        elif source is not None and group_key.source.source_id != str(source):
            return False
        if isinstance(group, GroupKey):
            if group_key != group:
                return False
        elif group is not None and group_key.group_id != str(group):
            return False
        return True

    def _selected_groups(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
    ) -> Iterator[_GroupIndex]:
        self._ensure_open()
        for key in sorted(self._groups):
            if self._matches(key, source, group):
                yield self._groups[key]

    def iter_sources(self) -> Iterator[SourceKey]:
        self._ensure_open()
        yield from sorted({key.source for key in self._groups})

    def iter_groups(
        self, source: Optional[Union[str, SourceKey]] = None
    ) -> Iterator[GroupRef]:
        for group in self._selected_groups(source=source):
            csv_member = sorted(group.csv_members)[0] if group.csv_members else None
            yield GroupRef(
                key=group.key,
                csv_member_path=csv_member,
                variant_ids=tuple(sorted(group.variants)),
            )

    def iter_variants(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
    ) -> Iterator[VariantRef]:
        for group_index in self._selected_groups(source=source, group=group):
            for variant_id in sorted(group_index.variants):
                images = group_index.variants[variant_id]
                yield VariantRef(
                    key=VariantKey(group_index.key, variant_id),
                    image_members=tuple(images[name] for name in sorted(images)),
                )

    def _iter_csv_rows(
        self, group: _GroupIndex, issues: Optional[List[ParserIssue]] = None
    ) -> Iterator[_CsvRow]:
        if not group.csv_members:
            return
        csv_members = sorted(group.csv_members)
        if len(csv_members) > 1 and issues is not None:
            for duplicate in csv_members[1:]:
                issues.append(
                    ParserIssue(
                        code="duplicate_pair_csv",
                        message="group has more than one pair.csv; only the first is used",
                        member_path=duplicate,
                    )
                )
        csv_member = csv_members[0]
        with self._zip.open(csv_member, mode="r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            fieldnames = {
                str(name).strip().casefold(): name for name in (reader.fieldnames or [])
            }
            required = ("pair_1", "pair_2", "condition")
            if not all(name in fieldnames for name in required):
                if issues is not None:
                    issues.append(
                        ParserIssue(
                            code="invalid_csv_schema",
                            message="pair.csv requires pair_1,pair_2,condition headers",
                            member_path=csv_member,
                        )
                    )
                return
            for row_number, row in enumerate(reader, start=2):
                pair_1 = _fragment_token(row.get(fieldnames["pair_1"], ""))
                pair_2 = _fragment_token(row.get(fieldnames["pair_2"], ""))
                condition = str(row.get(fieldnames["condition"], "")).strip()
                if not pair_1 or not pair_2 or not condition:
                    if issues is not None:
                        issues.append(
                            ParserIssue(
                                code="malformed_csv_row",
                                message="pair row has an empty required value",
                                member_path=csv_member,
                                row_number=row_number,
                            )
                        )
                    continue
                yield _CsvRow(pair_1, pair_2, condition, row_number)

    def _iter_group_pairs(
        self,
        group: _GroupIndex,
        variant_filter: Optional[str],
        include_reverse: bool,
        issues: Optional[List[ParserIssue]] = None,
    ) -> Iterator[PairRecord]:
        rows = tuple(self._iter_csv_rows(group, issues=issues))
        if not rows:
            return
        variants: Iterable[str] = sorted(group.variants)
        if variant_filter is not None:
            variants = (variant_filter,) if variant_filter in group.variants else ()
        csv_member = sorted(group.csv_members)[0]
        for variant_id in variants:
            images = group.variants[variant_id]
            variant_key = VariantKey(group.key, variant_id)
            for row in rows:
                if row.pair_1 == row.pair_2:
                    if issues is not None:
                        issues.append(
                            ParserIssue(
                                code="self_pair",
                                message="self-pair row was not emitted",
                                member_path=csv_member,
                                row_number=row.row_number,
                                variant_id=variant_id,
                            )
                        )
                    continue
                try:
                    label = parse_mm_condition(row.condition)
                except ConditionParseError as exc:
                    if issues is not None:
                        issues.append(
                            ParserIssue(
                                code="invalid_condition",
                                message=str(exc),
                                member_path=csv_member,
                                row_number=row.row_number,
                                variant_id=variant_id,
                            )
                        )
                    continue
                missing = [
                    token
                    for token in (row.pair_1, row.pair_2)
                    if token not in images
                ]
                if missing:
                    if issues is not None:
                        issues.append(
                            ParserIssue(
                                code="missing_image_reference",
                                message="CSV references missing PNG(s): {}".format(
                                    ", ".join(missing)
                                ),
                                member_path=csv_member,
                                row_number=row.row_number,
                                variant_id=variant_id,
                            )
                        )
                    continue
                fragment_a = FragmentRef(
                    variant=variant_key,
                    fragment_name=row.pair_1,
                    member_path=images[row.pair_1],
                )
                fragment_b = FragmentRef(
                    variant=variant_key,
                    fragment_name=row.pair_2,
                    member_path=images[row.pair_2],
                )
                record = PairRecord(
                    fragment_a=fragment_a,
                    fragment_b=fragment_b,
                    is_adjacent=label.is_adjacent,
                    direction_b_wrt_a=label.direction_b_wrt_a,
                    canonical_pair_key=canonical_unordered_pair_key(
                        fragment_a, fragment_b
                    ),
                    condition_raw=row.condition,
                    csv_member_path=csv_member,
                    csv_row_number=row.row_number,
                )
                yield record
                if include_reverse:
                    yield record.reversed()

    def iter_pairs(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
        variant: Optional[str] = None,
        include_reverse: bool = False,
    ) -> Iterator[PairRecord]:
        """Iterate valid, explicitly labelled pairs.

        Rows referring to a missing image, invalid conditions and self-pairs are
        excluded.  ``include_reverse`` adds a derived reverse directed view; it
        never changes the canonical unordered key.
        """

        for group_index in self._selected_groups(source=source, group=group):
            yield from self._iter_group_pairs(
                group_index,
                variant_filter=variant,
                include_reverse=include_reverse,
            )

    def _referenced_tokens(self, group: _GroupIndex) -> Set[str]:
        tokens: Set[str] = set()
        for row in self._iter_csv_rows(group):
            tokens.add(row.pair_1)
            tokens.add(row.pair_2)
        return tokens

    def iter_unreferenced_images(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
        variant: Optional[str] = None,
    ) -> Iterator[FragmentRef]:
        """Yield PNGs absent from pair.csv without inventing labels for them."""

        for group_index in self._selected_groups(source=source, group=group):
            referenced = self._referenced_tokens(group_index)
            for variant_id in sorted(group_index.variants):
                if variant is not None and variant_id != variant:
                    continue
                variant_key = VariantKey(group_index.key, variant_id)
                for fragment_name, member_path in sorted(
                    group_index.variants[variant_id].items()
                ):
                    if fragment_name not in referenced:
                        yield FragmentRef(variant_key, fragment_name, member_path)

    def iter_issues(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
        variant: Optional[str] = None,
    ) -> Iterator[ParserIssue]:
        """Validate selected groups and yield non-fatal parser issues."""

        for group_index in self._selected_groups(source=source, group=group):
            issues: List[ParserIssue] = []
            # Exhaustion performs validation; pair objects themselves are not kept.
            for _ in self._iter_group_pairs(
                group_index,
                variant_filter=variant,
                include_reverse=False,
                issues=issues,
            ):
                pass
            yield from issues

    def profile(
        self,
        source: Optional[Union[str, SourceKey]] = None,
        group: Optional[Union[str, GroupKey]] = None,
        variant: Optional[str] = None,
    ) -> MMProfile:
        """Return deterministic counts for a selected archive slice."""

        selected_groups = list(self._selected_groups(source=source, group=group))
        selected_sources = {entry.key.source for entry in selected_groups}
        profile = MMProfile(
            source_count=len(selected_sources),
            group_count=len(selected_groups),
            csv_count=sum(bool(entry.csv_members) for entry in selected_groups),
            duplicate_csv_count=sum(
                max(0, len(entry.csv_members) - 1) for entry in selected_groups
            ),
        )
        direction_counts: Counter[str] = Counter()
        for entry in selected_groups:
            selected_variant_ids = [
                name
                for name in sorted(entry.variants)
                if variant is None or name == variant
            ]
            profile.variant_count += len(selected_variant_ids)
            profile.png_count += sum(
                len(entry.variants[name]) for name in selected_variant_ids
            )
            issues: List[ParserIssue] = []
            rows = tuple(self._iter_csv_rows(entry, issues=issues))
            profile.csv_row_count += len(rows)
            referenced_tokens: Set[str] = set()
            for row in rows:
                referenced_tokens.update((row.pair_1, row.pair_2))
                if row.pair_1 == row.pair_2:
                    profile.self_pair_row_count += len(selected_variant_ids)
                    continue
                try:
                    label = parse_mm_condition(row.condition)
                except ConditionParseError:
                    profile.invalid_condition_row_count += len(selected_variant_ids)
                    continue
                for variant_id in selected_variant_ids:
                    images = entry.variants[variant_id]
                    if row.pair_1 not in images or row.pair_2 not in images:
                        profile.missing_image_reference_count += 1
                        continue
                    profile.emitted_pair_count += 1
                    if label.is_adjacent:
                        profile.adjacent_pair_count += 1
                        assert label.direction_b_wrt_a is not None
                        direction_counts[label.direction_b_wrt_a.value] += 1
                    else:
                        profile.negative_pair_count += 1
            profile.unreferenced_png_count += sum(
                fragment_name not in referenced_tokens
                for variant_id in selected_variant_ids
                for fragment_name in entry.variants[variant_id]
            )
            issue_counts = Counter(issue.code for issue in issues)
            profile.malformed_csv_row_count += issue_counts["malformed_csv_row"]
        profile.direction_counts = dict(sorted(direction_counts.items()))
        return profile
