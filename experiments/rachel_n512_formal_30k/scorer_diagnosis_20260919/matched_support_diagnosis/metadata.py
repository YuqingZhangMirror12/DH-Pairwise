"""Target-free metadata join for completed S7 TRAIN support diagnosis.

The caller reads the manifest['entries'] and completed cache pairs.json and
enforces the formal24000 population. This helper never reads images or models,
changes labels, filters rows, or assumes a population size.

S7's materializer copies surviving_correspondences/inherited_match_count from
its source entry without refreshing them after S7 damage. Keep these as inherited
metadata, NOT actual post-S7 correspondence target counts. Source foreground
areas likewise describe the preaugmentation source, not the final cached masks.
"""
from collections.abc import Mapping
import math
from numbers import Real

FIELDS = ("s7_recipe", "changed_pair", "source_stratum", "area_ratio_band",
          "surviving_correspondences", "inherited_match_count")
COUNT_PROVENANCE = "inherited source-entry metadata; not recomputed post-S7 correspondence targets"


def _index(rows, name):
    values, lookup = list(rows), {}
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError(name + " contains a non-object row")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(name + " has a missing pair_id")
        if pair_id in lookup:
            raise ValueError(name + " has duplicate pair_id: " + pair_id)
        lookup[pair_id] = row
    return values, lookup


def _label(value, context):
    if not isinstance(value, Real) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError("nonbinary/missing label: " + context)
    return bool(value)


def _source_area_ratio(source):
    areas = [source.get("fragment_"+side, {}).get("foreground_area") for side in "ab"]
    if any(area is None for area in areas):
        return None
    if any(isinstance(area, bool) or not isinstance(area, Real) or not math.isfinite(area)
           or area <= 0 for area in areas):
        raise ValueError("present source foreground areas must be finite and positive")
    return min(areas)/max(areas)


def join_metadata(entries, records):
    """Return a one-to-one metadata list in EXACT cache-record order.

    Both populations must be unique and identical; labels are checked rather
    than inferred. Missing metadata stays None, never a synthetic zero. Negative
    source kinds require two nonempty split_unit_id strings. Positive source
    kinds do not infer a relationship from potentially absent source identities.
    """
    _, entry_by_id = _index(entries, "TRAIN manifest")
    ordered, record_by_id = _index(records, "completed cache")
    if set(entry_by_id) != set(record_by_id):
        raise ValueError("manifest/cache full membership differs: %d absent from cache; %d absent from manifest" %
            (len(set(entry_by_id)-set(record_by_id)), len(set(record_by_id)-set(entry_by_id))))
    output = []
    for ordinal, record in enumerate(ordered):
        pair_id = record["pair_id"]
        entry = entry_by_id[pair_id]
        label = _label(entry.get("label"), "manifest "+pair_id)
        if _label(record.get("label"), "cache "+pair_id) != label:
            raise ValueError("manifest/cache label mismatch: " + pair_id)
        source = entry.get("source_row")
        if not isinstance(source, Mapping) or source.get("split") != "train":
            raise ValueError("S7 TRAIN source_row required: " + pair_id)
        if "label" in source and _label(source["label"], "source "+pair_id) != label:
            raise ValueError("source/manifest label mismatch: " + pair_id)
        if "pair_id" in source and source["pair_id"] != pair_id:
            raise ValueError("source/manifest pair_id mismatch: " + pair_id)
        if any(not isinstance(source.get("fragment_"+side, {}), Mapping) for side in "ab"):
            raise ValueError("source fragment metadata must be objects: " + pair_id)
        if label:
            kind = "positive"
        else:
            identities = [source.get("fragment_"+side, {}).get("split_unit_id") for side in "ab"]
            if any(not isinstance(value, str) or not value.strip() for value in identities):
                raise ValueError("negative pair missing split_unit_id: " + pair_id)
            kind = "same_source" if identities[0] == identities[1] else "cross_source"
        row = {name: entry.get(name) for name in FIELDS}
        row.update(pair_id=pair_id, cache_ordinal=ordinal, label=label,
            source_area_ratio=_source_area_ratio(source), source_area_ratio_stage="preaugmentation",
            correspondence_count_provenance=COUNT_PROVENANCE, negative_source_kind=kind)
        output.append(row)
    return output
