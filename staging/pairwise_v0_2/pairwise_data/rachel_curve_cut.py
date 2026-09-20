"""TRAIN-source manuscript outline profiles and oblique material deletion.

Only alpha silhouettes of existing TRAIN source manuscripts provide donor arcs.
The extracted arc is represented as a chord-normal height profile. This is a
transformed contour-derived profile, not pixel-identical real damage or a new
ground-truth join. No fragment rotation, translation, or RGB input is introduced.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from .rachel_preprocess import _dense_external_contour

SCHEMA = "rachel-train-outline-bank/1"


def manuscript_family(name):
    # Conservative exclusion of obvious recto/verso/detail aliases; not a claim
    # of a complete cross-corpus manuscript identity map.
    stem = Path(name).stem.casefold()
    return re.split(r"(?:[_-]?(?:recto|verso|detail|total))", stem, maxsplit=1)[0]


def profile_from_arc(points, count=129):
    points = np.asarray(points, dtype=float)
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < 64:
        return None
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    shifted = points - points[0]
    x, y = shifted @ tangent / length, shifted @ normal / length
    # Reject pronounced overhangs instead of turning them into invented edges.
    if x.min() < -.015 or x.max() > 1.015 or np.mean(np.diff(x) < -.002) > .08:
        return None
    order = np.argsort(x, kind="stable")
    unique, first, sizes = np.unique(x[order], return_index=True, return_counts=True)
    heights = np.add.reduceat(y[order], first) / sizes
    if len(unique) < 16:
        return None
    grid = np.linspace(0., 1., count)
    output = np.interp(grid, unique, heights)
    output -= (1-grid)*output[0] + grid*output[-1]
    rms = float(np.sqrt(np.mean(output**2)))
    if not .0075 <= rms <= .12 or np.ptp(output) > .30:
        return None
    return output.astype(np.float32), dict(chord_length_px=length,
        normalized_rms=rms, normalized_peak_to_peak=float(np.ptp(output)))


class OutlineBank:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "bank.json").read_text())
        if self.metadata.get("schema_version") != SCHEMA or self.metadata.get("split") != "train":
            raise ValueError("outline bank must be TRAIN-only")
        with np.load(directory / "profiles.npz", allow_pickle=False) as arrays:
            self.profiles = arrays["profiles"].astype(np.float32)
        if self.profiles.ndim != 2 or self.profiles.shape[1] != 129 or not np.isfinite(self.profiles).all():
            raise ValueError("invalid donor profiles")
        if len(self.profiles) != len(self.metadata["arcs"]) or not len(self.profiles):
            raise ValueError("donor profile/provenance count mismatch")
        if any(a.get("split") != "train" for a in self.metadata["arcs"]):
            raise ValueError("non-TRAIN donor provenance")
        self.profiles.setflags(write=False)


def curve_mask(mask, profile, *, angle_deg, fraction, keep_low, flip=False, reverse=False):
    """Crop with a real-outline profile; the fragment coordinate frame is fixed."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("nonempty 2D mask required")
    angle = float(angle_deg) % 180.
    if min(angle % 90., 90.-angle % 90.) < 15.-1e-6 or not .15 <= fraction <= .85:
        raise ValueError("cut must be oblique and within bbox-normalized fraction")
    profile = np.asarray(profile, float)
    if profile.shape != (129,) or not np.isfinite(profile).all():
        raise ValueError("129 finite contour-profile samples required")
    if reverse:
        profile = profile[::-1]
    if flip:
        profile = -profile
    rc = np.argwhere(mask)
    lo, hi = rc.min(0), rc.max(0)
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]], float)
    theta = np.deg2rad(angle)
    tangent, normal = np.array([np.cos(theta), np.sin(theta)]), np.array([-np.sin(theta), np.cos(theta)])
    t_range, n_range = corners @ tangent, corners @ normal
    span = max(1., float(t_range.max()-t_range.min()))
    rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
    t = rr*tangent[0] + cc*tangent[1]
    n = rr*normal[0] + cc*normal[1]
    u = (t-t_range.min())/span
    boundary = n_range.min() + fraction*np.ptp(n_range) + np.interp(u, np.linspace(0.,1.,129), profile)*span
    return mask & ((n <= boundary) if keep_low else (n >= boundary))


def build_bank(release_root, image_root, output, *, max_lineages=128, arcs_per_lineage=6, seed=260910):
    release_root, image_root, output = map(Path, (release_root, image_root, output))
    splits = json.loads((release_root / "pairs/lineage_splits.json").read_text())
    excluded = {manuscript_family(n) for n,s in splits.items() if s != "train"}
    eligible = [n for n,s in splits.items() if s == "train" and manuscript_family(n) not in excluded]
    eligible.sort(key=lambda n: hashlib.sha256((str(seed)+n).encode()).digest())
    output.mkdir(parents=True, exist_ok=False)
    profiles, records, reasons = [], [], Counter()
    used_families = set()
    previews = []
    unreadable_sources = []
    for name in eligible:
        if len(used_families) >= max_lineages:
            break
        family = manuscript_family(name)
        if family in used_families:
            continue
        path = image_root / name
        if not path.is_file():
            reasons["missing_source"] += 1
            continue
        try:
            with Image.open(path) as image:
                if "A" not in image.getbands():
                    reasons["no_alpha"] += 1
                    continue
                alpha = np.asarray(image.getchannel("A")) > 127
        except OSError as error:
            reasons["unreadable_source_image"] += 1
            unreadable_sources.append(dict(lineage=name, error=str(error)))
            continue
        if not alpha.any() or alpha.all():
            reasons["nonselective_alpha"] += 1
            continue
        scale = min(1., 796./max(alpha.shape))
        mask = cv2.resize(alpha.astype(np.uint8),
            (max(1,round(alpha.shape[1]*scale)), max(1,round(alpha.shape[0]*scale))), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = np.pad(mask, 2)
        dense = _dense_external_contour(mask)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256((str(seed)+name).encode()).digest()[:8], "big"))
        accepted = 0
        for attempt in range(128):
            start = int(rng.integers(len(dense)))
            count = int(rng.integers(96, min(321, len(dense)//2+1))) if len(dense) >= 194 else 0
            if not count:
                break
            arc = dense[(start+np.arange(count)) % len(dense)]
            candidate = profile_from_arc(arc)
            if candidate is None:
                continue
            profile, geometry = candidate
            records.append(dict(donor_index=len(profiles), lineage=name, family=family, split="train",
                source_path=str(path.resolve()), alpha_threshold=127, longest_side_limit=796,
                contour_start=start, contour_point_count=count, attempt=attempt, **geometry))
            profiles.append(profile)
            if len(previews) < 4:
                np.savez_compressed(output / ("donor_%02d.npz" % (len(previews)+1)),
                    packed_mask=np.packbits(mask, axis=1), mask_shape=np.array(mask.shape), arc=arc, profile=profile)
                previews.append(len(profiles)-1)
            accepted += 1
            if accepted == arcs_per_lineage:
                break
        if accepted:
            used_families.add(family)
        else:
            reasons["no_eligible_curved_arc"] += 1
    if not profiles:
        raise RuntimeError("no real TRAIN outline profiles available")
    np.savez_compressed(output / "profiles.npz", profiles=np.stack(profiles))
    metadata = dict(schema_version=SCHEMA, split="train", status="complete", seed=seed,
        release_root=str(release_root.resolve()), image_root=str(image_root.resolve()),
        split_authority=str((release_root/"pairs/lineage_splits.json").resolve()),
        profile_count=len(profiles), donor_family_count=len(used_families), arcs=records,
        excluded_heldout_family_count=len(excluded), eligible_train_image_count=len(eligible),
        skipped=dict(reasons), unreadable_sources=unreadable_sources, previews=previews,
        silhouette="alpha support of existing TRAIN source manuscripts; no REAL mask read",
        identity_caveat="obvious recto/verso/detail families additionally excluded across frozen splits; no complete cross-corpus alias map claimed",
        representation="ordered source arc projected to chord-normal profile, repeated x averaged; interpolated to129 points; scaled and obliquely oriented during crop")
    (output / "bank.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-lineages", type=int, default=128)
    args = p.parse_args()
    result = build_bank(args.release_root, args.image_root, args.output, max_lineages=args.max_lineages)
    print(json.dumps({k:v for k,v in result.items() if k != "arcs"}, ensure_ascii=False))
