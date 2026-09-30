"""TRAIN-only, separate same-anchor auxiliary population; no training/mining.

Ordinary materialized binary pairs are not modified or replaced. An original
token match is necessary but not sufficient: a manifest-bound, caller-verified
scale ledger is required. A cohort proof may document a no-resize implementation
path; manual per-row proofs are not required. Missing/unequal scale is excluded.
The 1470 positives in the availability audit are candidates, NOT a claim that
1470 physical-scale-compatible auxiliary groups have been verified here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import (
    SCHEMA as MATERIALIZED_SCHEMA, MaterializedRachelDataset,
)

SCHEMA = "scorer-same-anchor-triplets/1"
SCALE_SCHEMA = "scorer-same-anchor-scale-ledger/1"
KNOWN_NONJOIN_ORIGINS = frozenset(("rachel_csv_same_folder_nonneighbor_no_seam",
                                  "gen5_partition_original_csv"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _identity(fragment):
    values = fragment.get("split_unit_id"), fragment.get("fragment_token")
    if not all(isinstance(x, str) and x for x in values):
        raise ValueError("explicit source identity and original fragment token required")
    return values


def _scale_proof(entry, ledger):
    """Return verified evidence or exclusion reason; never infer scale from 800."""
    if ledger is None:
        return None, "missing_verified_scale_proof"
    proof = ledger.get("pair_proofs", {}).get(entry["pair_id"])
    if proof is None:
        matches = [c["proof"] for c in ledger.get("cohorts", [])
                   if entry.get("source_root") in c.get("source_roots", [])
                   and entry.get("s7_recipe") in c.get("recipes", [])]
        if len(matches) > 1:
            raise ValueError("ambiguous scale-proof cohorts")
        proof = matches[0] if matches else None
    if proof is None:
        return None, "missing_verified_scale_proof"
    if (proof.get("verified") is not True or not proof.get("evidence")
            or not isinstance(proof.get("frame_namespace"), str) or not proof["frame_namespace"]
            or proof.get("mask_points_same_transform") is not True
            or proof.get("both_endpoints_same_scale") is not True
            or proof.get("pair_dependent_rescaling") is not False
            or proof.get("rotation_degrees") != 0
            or proof.get("canvas_hw") != [800, 800]):
        return None, "incomplete_or_unsupported_scale_proof"
    scale = proof.get("pixel_scale_rc")
    if (not isinstance(scale, (list, tuple)) or len(scale) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x <= 0 for x in scale)):
        return None, "incomplete_or_unsupported_scale_proof"
    return deepcopy(proof), None


def build_triplet_index(manifest_bytes, *, scale_ledger=None, seed=260928, max_negatives=3):
    """Pure deterministic index from exact manifest bytes; no artifact reads.

    Ledger: {schema: SCALE_SCHEMA, manifest_sha256: ..., pair_proofs: {pair_id:
    proof}, cohorts: [{source_roots: [...], recipes: [...], proof: ...}]}.
    A proof documents verified=True, evidence, frame_namespace, pixel_scale_rc,
    canvas_hw=[800,800], rotation_degrees=0, mask_points_same_transform=True,
    both_endpoints_same_scale=True, pair_dependent_rescaling=False. The evidence
    is a caller-audited preprocessing contract, not a model feature or new GT.
    No rescaling, crop, contour extraction or augmentation is performed here.
    """
    if type(seed) is not int or type(max_negatives) is not int or not 1 <= max_negatives <= 3:
        raise ValueError("integer seed and 1..3 known negative partners required")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != MATERIALIZED_SCHEMA or manifest.get("split") != "train":
        raise ValueError("only materialized TRAIN manifests are supported")
    if scale_ledger is not None and (scale_ledger.get("schema") != SCALE_SCHEMA
            or scale_ledger.get("manifest_sha256") != digest):
        raise ValueError("scale ledger is not bound to this exact TRAIN manifest")
    entries = manifest["entries"]
    ids, labels, token_sources = set(), {}, {}
    hard = defaultdict(list)
    for e in entries:
        source = e["source_row"]
        if e["pair_id"] in ids or source.get("pair_id") != e["pair_id"]:
            raise ValueError("duplicate or differing pair ID")
        ids.add(e["pair_id"])
        if (source.get("split") != "train" or e.get("label") not in (False, True)
                or source.get("label") != e["label"]):
            raise ValueError("held-out source or differing binary label")
        a, b = (_identity(source["fragment_" + side]) for side in "ab")
        if a == b:
            raise ValueError("self pair cannot supply nonjoin evidence")
        for unit, token in (a, b):
            if token in token_sources and token_sources[token] != unit:
                raise ValueError("original token has conflicting source identities")
            token_sources[token] = unit
        key = tuple(sorted((a, b)))
        if key in labels and labels[key] != bool(e["label"]):
            raise ValueError("conflicting original-token-pair labels")
        labels[key] = bool(e["label"])
        if (not e["label"] and a[0] == b[0]
                and source.get("label_origin") in KNOWN_NONJOIN_ORIGINS):
            hard[a].append((b, e))
            hard[b].append((a, e))
    groups, excluded = [], []
    candidate_positive_count = 0
    for positive in sorted((e for e in entries if e["label"]), key=lambda e: e["pair_id"]):
        source = positive["source_row"]
        a, b = (_identity(source["fragment_" + side]) for side in "ab")
        if not hard[a] and not hard[b]:
            continue
        candidate_positive_count += 1
        positive_proof, positive_reason = _scale_proof(positive, scale_ledger)
        by_anchor = defaultdict(dict)
        for anchor, partner in ((a, b), (b, a)):
            for negative_identity, negative in hard[anchor]:
                negative_proof, reason = _scale_proof(negative, scale_ledger)
                reason = positive_reason or reason
                if a[0] != b[0]:
                    reason = "positive_endpoints_not_same_source"
                if not reason and any(positive_proof[k] != negative_proof[k]
                        for k in ("frame_namespace", "pixel_scale_rc", "canvas_hw")):
                    reason = "incompatible_physical_scale_or_frame"
                if negative_identity in (anchor, partner):
                    reason = "negative_is_anchor_or_positive_partner"
                if reason:
                    excluded.append(dict(positive_pair_id=positive["pair_id"],
                        anchor_token=anchor[1], negative_pair_id=negative["pair_id"],
                        negative_token=negative_identity[1], reason=reason))
                    continue
                candidate = dict(token=negative_identity[1], entry=deepcopy(negative),
                                 scale_proof=negative_proof)
                old = by_anchor[anchor].get(negative_identity)
                # Multiple augmented rows for one C do not count as multiple partners.
                if old is None or negative["pair_id"] < old["entry"]["pair_id"]:
                    by_anchor[anchor][negative_identity] = candidate
        if not by_anchor:
            continue
        anchor = min(by_anchor, key=lambda x: _digest([seed, positive["pair_id"], "anchor", x]))
        others = sorted(by_anchor[anchor].values(),
            key=lambda x: _digest([seed, positive["pair_id"], anchor, x["token"]]))[:max_negatives]
        groups.append(dict(group_id=_digest([SCHEMA, positive["pair_id"], anchor])[:24],
            manifest_sha256=digest,
            anchor_token=anchor[1], split_unit_id=anchor[0],
            positive=dict(entry=deepcopy(positive), scale_proof=positive_proof),
            negatives=others))
    return dict(schema=SCHEMA, manifest_sha256=digest, seed=seed, max_negatives=max_negatives,
        scale_ledger_sha256=_digest(scale_ledger) if scale_ledger is not None else None,
        ordinary_population_unchanged=True, ordinary_pair_count=len(entries),
        candidate_positive_count=candidate_positive_count, usable_group_count=len(groups),
        usable_negative_pair_count=sum(len(g["negatives"]) for g in groups),
        actual_eligibility_fully_verified=not excluded,
        excluded_reason_counts=dict(Counter(e["reason"] for e in excluded)), excluded=excluded,
        groups=groups, policy="one deterministic anchor per eligible positive; up to3 distinct confirmed nonjoin partners; no group-index pairing")


def _endpoint(sample, token, entry):
    tokens = (sample.fragment_a_token, sample.fragment_b_token)
    expected = tuple(entry["source_row"]["fragment_" + side]["fragment_token"] for side in "ab")
    if sample.pair_id != entry["pair_id"] or bool(sample.label) != bool(entry["label"]) or sorted(tokens) != sorted(expected):
        raise ValueError("materialized pair identity/label/endpoints differ from proof")
    if tokens.count(token) != 1:
        raise ValueError("requested original token must identify exactly one artifact endpoint")
    side = "ab"[tokens.index(token)]
    values = tuple(np.asarray(getattr(sample, name + side))
                   for name in ("mask_", "points_rc_", "contour_valid_"))
    mask, points, valid = values
    if (mask.shape != (1, 800, 800) or mask.dtype != np.float32
            or not np.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()
            or points.ndim != 2 or points.shape[1] != 2 or points.dtype != np.float32
            or valid.shape != points.shape[:1] or valid.dtype != np.bool_
            or valid.sum() < 4 or not np.isfinite(points[valid]).all()
            or ((points[valid] < 0) | (points[valid] > 799)).any()):
        raise ValueError("mask/points/valid violate canonical800 coordinate contract")
    return values, side


@dataclass(frozen=True)
class AuxiliaryBatch:
    inputs: tuple  # exactly mask_a, mask_b, points_a, points_b, valid_a, valid_b
    labels: torch.Tensor
    known_negative_pairs: torch.Tensor
    anchor_ids: tuple
    same_anchor_confirmed: bool
    provenance: dict


def assemble_triplet_group(group, samples, *, swap=False, contour_cap=512, artifact_receipts=None):
    """Assemble one positive + 1..3 negatives; never reuse pair-context features.

    swap can be one bool or one bool per output pair. Even mixed A/B orientations
    retain identical padded anchor bytes. Only masks, points and validity enter
    the six model inputs; labels/proof/IDs remain separate loss/audit fields.
    """
    if type(contour_cap) is not int or not 4 <= contour_cap <= 512:
        raise ValueError("fixed original512 contour cap must be in4..512")
    positive = group["positive"]["entry"]
    anchor_token = group["anchor_token"]
    positive_proof = group["positive"].get("scale_proof")
    for evidence in [group["positive"]] + group["negatives"]:
        entry, proof = evidence["entry"], evidence.get("scale_proof")
        _, reason = _scale_proof(entry, {"pair_proofs": {entry["pair_id"]: proof}})
        if reason:
            raise ValueError("cannot assemble unverified scale: " + reason)
        if (entry["source_row"].get("split") != "train"
                or any(_identity(entry["source_row"]["fragment_"+s])[0] != group["split_unit_id"] for s in "ab")
                or any(proof[k] != positive_proof[k] for k in ("frame_namespace", "pixel_scale_rc", "canvas_hw"))):
            raise ValueError("incompatible source/physical-scale evidence")
    if not positive["label"] or any(n["entry"]["label"] or
            n["entry"]["source_row"].get("label_origin") not in KNOWN_NONJOIN_ORIGINS for n in group["negatives"]):
        raise ValueError("positive and confirmed source-labelled nonjoin partners required")
    other_token = next(positive["source_row"]["fragment_" + s]["fragment_token"]
        for s in "ab" if positive["source_row"]["fragment_" + s]["fragment_token"] != anchor_token)
    anchor, anchor_source_side = _endpoint(samples[positive["pair_id"]], anchor_token, positive)
    partner, _ = _endpoint(samples[positive["pair_id"]], other_token, positive)
    pairs, negative_sides = [(anchor, partner)], []
    for negative in group["negatives"]:
        entry = negative["entry"]
        if negative["token"] in (anchor_token, other_token):
            raise ValueError("negative cannot be the anchor or positive partner")
        # Verify anchor identity in AC evidence, but DISCARD its augmented pixels.
        _, side = _endpoint(samples[entry["pair_id"]], anchor_token, entry)
        c, cside = _endpoint(samples[entry["pair_id"]], negative["token"], entry)
        pairs.append((anchor, c))
        negative_sides.append(dict(pair_id=entry["pair_id"], discarded_anchor_side=side, partner_side=cside))
    if not 2 <= len(pairs) <= 4:
        raise ValueError("one positive and 1..3 negatives required")
    swaps = [swap] * len(pairs) if isinstance(swap, bool) else list(swap)
    if len(swaps) != len(pairs) or any(type(s) is not bool for s in swaps):
        raise ValueError("swap must be bool or one bool per output pair")
    padded = []
    for pair, reverse in zip(pairs, swaps):
        row = []
        for mask, points, valid in (pair[::-1] if reverse else pair):
            if len(points) > contour_cap:
                raise ValueError("contour exceeds fixed cap; never truncate/resample")
            p = np.zeros((contour_cap, 2), np.float32)
            v = np.zeros(contour_cap, np.bool_)
            p[:len(points)], v[:len(valid)] = points, valid
            row.append((mask, p, v))
        padded.append(row)
    inputs = tuple(torch.from_numpy(np.stack([row[side][kind] for row in padded]))
                   for kind, side in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)))
    canonical_anchor = json.dumps([group["split_unit_id"], anchor_token], separators=(",", ":"))
    provenance = dict(group=deepcopy(group), anchor_source_side=anchor_source_side,
        negative_artifact_sides=negative_sides, swaps=swaps,
        artifact_receipts=deepcopy(artifact_receipts or {}),
        anchor_reused_from_positive=True, negative_artifact_anchor_discarded=True,
        partner_conditioned_features_used=False, transforms_applied=False)
    return AuxiliaryBatch(inputs, torch.tensor([1.] + [0.] * (len(pairs)-1)),
        torch.tensor([[0, i] for i in range(1, len(pairs))], dtype=torch.long),
        (canonical_anchor,) * len(pairs), True, provenance)


class SameAnchorTripletDataset:
    """Finite lazy auxiliary loader; original MaterializedRachelDataset unchanged."""
    def __init__(self, manifest_path, *, scale_ledger=None, seed=260928, max_negatives=3):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.index = build_triplet_index(self.manifest_path.read_bytes(), scale_ledger=scale_ledger,
                                        seed=seed, max_negatives=max_negatives)
        self.ordinary = MaterializedRachelDataset(self.manifest_path)
        self.lookup = {e["pair_id"]: i for i, e in enumerate(self.ordinary.entries)}
        self.split = "train"

    def __len__(self):
        return len(self.index["groups"])

    def get(self, index, *, swap=False):
        group = self.index["groups"][index]
        samples, receipts = {}, {}
        for evidence in [group["positive"]] + group["negatives"]:
            entry = evidence["entry"]
            pair_id = entry["pair_id"]
            samples[pair_id] = self.ordinary[self.lookup[pair_id]]
            path = self.ordinary.root / entry["artifact_path"]
            receipts[pair_id] = dict(artifact_path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return assemble_triplet_group(group, samples, swap=swap, artifact_receipts=receipts)

    def __getitem__(self, index):
        return self.get(index)
