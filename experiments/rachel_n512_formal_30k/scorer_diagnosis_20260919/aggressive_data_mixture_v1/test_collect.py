"""Synthetic metadata checks, not new generation or a real30K result."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .collect import GAP_BINS, collect, distribution, project, sha, summarize


def record(positive=True, fallback=False, recipe="wave", name="example"):
    trim = dict(size_class="smaller", mode="one", material_removed_fraction=.1,
                common_length_before_px=100., common_length_after_px=80., retained_fraction=.8)
    gap = dict(ray_resolved_fraction=.9, gap_bins=list(GAP_BINS),
               gap_arc_length_counts=[10.] + [0.] * 9, gap_mean_px=1.)
    paired = dict(new_primary_gap_peak_px=12., old_primary_gap_peak_px=6.,
                  resolved_arc_gap_p10_p50_p90=[.5, 1., 2.], unresolved_fraction=.1,
                  arc_fraction_gap5to35=.2)
    r = dict(id=name, pair_id=name, source_pair_id="original-" + name, label=positive,
             v14_fallback=fallback, recipe=recipe, source_stratum="test_fixture",
             source_families=["synthetic"], planned_size_class="smaller",
             sample_sha256="s", proof_sha256="p", requested_gap_count=0,
             inherited_correspondences=10 if positive else 0,
             detail=dict(trim=None if fallback else trim, gap=gap if positive and not fallback else None))
    a = dict(id=name, status="passed", sample_sha256="s", proof_sha256="p", model_input_sha256="m",
             paired_gap=paired if positive and not fallback else None,
             baseline_numerical_identity=fallback)
    if recipe in ("clean", "partial") and positive and not fallback:
        a["paired_gap"].update(new_primary_gap_peak_px=None, old_primary_gap_peak_px=None)
    return r, a


def row(**kwargs):
    return project(*record(**kwargs), "train", 0, "g", "a")


def fixture(root):
    """One successful and one fallback group, both balanced."""
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return sha(path)
    d = root / "train"
    manifest_sha = put(d / "train.json", {"synthetic": True})
    group_refs = []
    for slot, fallback in enumerate((False, True)):
        pairs = [record(positive=p, fallback=fallback, name=f"{slot}-{p}") for p in (True, False)]
        name = f"{slot:05d}.json"
        group_sha = put(d / "groups" / name, dict(status="committed", split="train", slot=slot,
            positive_replaced=False, v14_fallback=fallback, records=[p[0] for p in pairs]))
        audit_sha = put(d / "audits" / name, dict(status="passed", group_sha256=group_sha, rows=[p[1] for p in pairs]))
        group_refs.append(dict(path=str(d / "audits" / name), sha256=audit_sha))
    pixel_sha = put(d / "full_pixel_audit.json", dict(status="passed", checked_pairs=4,
        manifest_sha256=manifest_sha, group_receipts=group_refs))
    audit_sha = put(root / "full_audit.json", dict(status="passed", failures=0, checked_pairs=4,
        split_audits={"train": {"sha256": pixel_sha}}, manifest_sha256={"train": manifest_sha},
        summaries={"train": dict(v14_fallback_pairs=2, recipe_counts={"wave": 2}, cut_side_counts={"smaller": 1})}))
    contract_sha = put(root / "data_contract.json", {"aggressive_full_audit": {"sha256": audit_sha}})
    put(root / "pipeline_complete.json", dict(status="complete", data_contract_sha256=contract_sha, pairs=4))
    return contract_sha


class DistributionTests(unittest.TestCase):
    def test_quantiles_and_missing(self):
        d = distribution([0., 10., None])
        self.assertEqual((d["n"], d["missing"], d["mean"], d["quantiles"]["0.5"]), (2, 1, 5., 5.))

    def test_all_missing_not_zero(self):
        self.assertEqual(distribution([None, None]), dict(n=0, missing=2, mean=None, quantiles=None))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            distribution([float("nan")])

    def test_fallback_zero_crop_but_unknown_gap(self):
        r = row(fallback=True)
        self.assertEqual(r["additional_crop_fraction"], 0.)
        self.assertIsNone(r["primary_new_peak_px"])
        self.assertIsNone(r["common_support_after_crop_px"])
        self.assertIsNone(r["requested_v17_notches"])
        self.assertIsNone(r["gap_arc_length_counts"])

    def test_clean_has_no_primary_peak_but_has_seam_gap(self):
        r = row(recipe="clean")
        self.assertIsNone(r["primary_new_peak_px"])
        self.assertEqual(r["resolved_point_gap_p50_px"], 1.)

    def test_negative_has_no_seam_gt(self):
        r = row(positive=False)
        self.assertIsNone(r["additional_crop_fraction"])
        self.assertIsNone(r["primary_new_peak_px"])
        self.assertIsNone(r["endpoint_mode"])
        self.assertEqual(r["additional_crop_area_fraction"], .1)

    def test_corrupt_row_binding(self):
        r, a = record()
        a["proof_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "artifact binding"):
            project(r, a, "train", 0, "g", "a")

    def test_invalid_crop_fraction_rejected(self):
        r, a = record()
        r["detail"]["trim"]["retained_fraction"] = .7
        with self.assertRaisesRegex(ValueError, "ratio mismatch"):
            project(r, a, "train", 0, "g", "a")

    def test_mixture_mean_does_not_drop_zero_fallback_crops(self):
        summary = summarize([row(), row(fallback=True), row(positive=False)])
        cell = next(c for c in summary["cells"] if c["version"] == "all" and c["recipe"] is None)
        self.assertAlmostEqual(cell["metrics"]["additional_crop_fraction"]["mean"], .1)
        self.assertEqual(cell["metrics"]["primary_new_peak_px"]["missing"], 1)
        self.assertEqual(cell["positive_pairs"], 2)

    def test_pooled_hist_uses_arc_weights_not_mean_of_pair_means(self):
        a, b = row(), deepcopy(row(name="second"))
        b["gap_arc_length_counts"] = [0., 90.] + [0.] * 8
        b["resolved_arc_weighted_mean_gap_px"] = 3.
        summary = summarize([a, b])
        h = next(c for c in summary["cells"] if c["version"] == "all" and c["recipe"] is None)["v17_gap_histogram"]
        self.assertEqual(h["share"][:2], [.1, .9])
        self.assertAlmostEqual(h["pooled_arc_weighted_mean_gap_px"], 2.8)

    def test_changed_bins_rejected(self):
        r, a = record()
        r["detail"]["gap"]["gap_bins"][1] = 1.
        with self.assertRaisesRegex(ValueError, "histogram bins"):
            project(r, a, "train", 0, "g", "a")


class BindingTests(unittest.TestCase):
    def test_complete_collection_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = fixture(root)
            before = {str(p): sha(p) for p in root.rglob("*.json")}
            rows, result = collect(root, expected, {"train": 4})
            self.assertEqual(len(rows), 4)
            self.assertEqual(result["positive_pairs"], 2)
            self.assertEqual(result["group_receipts"], 2)
            self.assertEqual(before, {str(p): sha(p) for p in root.rglob("*.json")})

    def test_wrong_contract_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                collect(root, "wrong", {"train": 4})

    def test_changed_group_fails_even_when_status_is_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = fixture(root)
            p = root / "train/groups/00000.json"
            data = json.loads(p.read_text())
            data["records"][0]["detail"]["trim"]["common_length_after_px"] = 70.
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                collect(root, expected, {"train": 4})

    def test_manifest_change_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = fixture(root)
            (root / "train/train.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                collect(root, expected, {"train": 4})


if __name__ == "__main__":
    unittest.main()
