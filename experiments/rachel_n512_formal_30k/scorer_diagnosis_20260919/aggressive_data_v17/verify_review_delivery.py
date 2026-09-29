"""Verify a completed CPU review bundle before publishing its existing images.

This checks delivery integrity and the recorded pixel-audit results. It does not
rerun generation, replace the pixel audit, approve the data, or start training.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root,previous):
    complete = read(root / "pipeline_complete.json")
    assert complete["status"] == "cpu_review_artifacts_complete"
    assert [(r["stage"], r["returncode"]) for r in complete["results"]] == [
        ("probe", 0), ("probe_audit", 0),
        ("generation", 0), ("pixel_audit", 0), ("render", 0)]
    assert complete["training_started"] is False
    assert complete["full_generation_authorized"] is False
    assert not list(root.glob("*failure*.json"))
    bundle, render = root / "review_bundle_01", root / "rendered_01"
    generated = read(bundle / "generation_complete.json")
    records = read(bundle / "manifest.json")["entries"]
    audit = read(bundle / "pixel_audit.json")
    rendered = read(render / "rendered.json")
    assert generated["missing"] == {} and generated["training_started"] is False
    assert audit["status"] == "passed" and audit["errors"] == []
    assert audit["full_training_generation_authorized"] is False
    assert audit["training_started"] is False
    assert audit["pilot_manifest_sha256"] == sha(root / "pilot_01/manifest.json")
    ids = [r["id"] for r in records]
    receipts = {r["id"]: r for r in audit["receipts"]}
    assert len(records) == len(set(ids)) == len(receipts) == audit["pairs"] == generated["pairs"]
    assert set(ids) == set(receipts) and all(r["status"] == "passed" for r in receipts.values())
    positives = [r for r in records if r["label"]]
    sides = Counter(r["detail"]["trim"]["size_class"] for r in positives)
    assert sides == audit["positive_size_counts"]
    assert sides["smaller"] * 10 == len(positives) * 7
    assert len(positives) * 2 == len(records)
    prior={r['id']:r for r in read(previous/'review_bundle_01/manifest.json')['entries']}
    unchanged_controls=0
    for r in records:
        name=Path(r['sample_path']).name
        assert sha(bundle/'samples'/name)==r['sample_sha256']
        assert sha(bundle/'proof'/name)==r['proof_sha256']
        older=prior[r['previous_review_id']]
        assert sha(previous/'review_bundle_01/proof'/name)==older['proof_sha256']
        with np.load(bundle/'proof'/name,allow_pickle=False) as newz,np.load(previous/'review_bundle_01/proof'/name,allow_pickle=False) as oldz:
            for side in 'ab':
                assert np.array_equal(newz['packed_trim_'+side],oldz['packed_trim_'+side])
        if r.get('unchanged_non_depth_recipe'):
            assert r['recipe'] in ('clean','partial')
            assert r['sample_sha256']==older['sample_sha256'] and r['proof_sha256']==older['proof_sha256']
            unchanged_controls+=1
        t = r["detail"]["trim"]
        assert 0 < t["material_removed_fraction"] <= .2
        assert t["donor"]["split"] == "train"
        assert t["trim_applied_before_primary"] is True
        if r["label"]:
            assert .19 - 1e-6 <= 1 - t["retained_fraction"] <= .21 + 1e-6
            for e in t["endpoint_audit"].values():
                assert e["original_order_preserved"] is True and e["interior_removed_px"] == 0
            peak = receipts[r["id"]]["paired_gap"]["new_primary_gap_peak_px"]
            assert peak is None or 5 - 1e-6 <= peak <= 25 + 1e-6
        else:
            assert t["retained_fraction"] is None
            assert receipts[r["id"]]["paired_gap"] is None
    groups, rows = rendered["groups"], rendered["rows"]
    assert len(groups) == 27
    row_ids = [r["id"] for r in rows]
    selected = set()
    group_counts = {}
    for key, g in groups.items():
        assert len(g["ids"]) == len(set(g["ids"])) == 10
        pool = [r for r in records if key in r["groups"] or (key == "light70" and r["recipe"] != "clean")]
        if key in ("native", "gen5", "union_tiny", "unequal", "mirror_h", "mirror_v") or key.startswith("negative_"):
            chosen = pool[:10]
        else:
            chosen = [r for r in pool if r["label"]][:5] + [r for r in pool if not r["label"]][:5]
        assert [r["id"] for r in chosen] == g["ids"]
        selected.update(g["ids"])
        group_counts[key] = {"label": g["label"], "examples": 10,
                             "positive": sum(bool(r["label"]) for r in chosen)}
    assert selected == set(row_ids) and len(row_ids) == len(set(row_ids))
    by_id={r['id']:r for r in records}
    for identity in groups['notch_k4']['ids']:
        r=by_id[identity];assert r['requested_gap_count']==4
        for damage in r['detail']['primary_damage'].values():
            if damage.get('applied'):
                assert len(damage['ignore_source_regions'])==4
                assert min(damage['notch_independently_removed_pixels'])>=8
    images = {}
    for r in rows:
        prefix, encoded = r["image"].split(",", 1)
        assert prefix == "data:image/png;base64"
        path = render / "images" / (r["id"] + ".png")
        assert base64.b64decode(encoded, validate=True) == path.read_bytes()
        assert r["paired_gap"] == receipts[r["id"]]["paired_gap"]
        images[r["id"]] = sha(path)
    return {"schema": "aggressive-v17-review-delivery-check/1", "status": "passed",
            "checked_at": datetime.now(timezone.utc).isoformat(), "pairs": len(records),
            "positive_pairs": len(positives), "positive_crop_sides": dict(sides),
            "display_groups": group_counts, "display_slots": sum(len(g["ids"]) for g in groups.values()),
            "unique_display_pairs": len(rows), "image_sha256": images,
            "approved_cut_pixels_unchanged_pairs":len(records),"byte_identical_clean_partial_pairs":unchanged_controls,
            "source_sha256": {str(p.relative_to(root)): sha(p) for p in [
                root / "pipeline_complete.json", bundle / "generation_complete.json",
                bundle / "manifest.json", bundle / "pixel_audit.json", render / "rendered.json"]},
            "pixel_audit_reexecuted": False, "human_approval": False,
            "full_30k_generated": False, "training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous",type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.root,args.previous)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("display_groups", "source_sha256", "image_sha256")}, ensure_ascii=False))
