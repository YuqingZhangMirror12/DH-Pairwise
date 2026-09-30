"""Synthetic TRAIN-only index and exact-anchor assembly tests; no remote data."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import SCHEMA, save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from .data import (SCALE_SCHEMA, SameAnchorTripletDataset, assemble_triplet_group,
                   build_triplet_index)
from .model import pair_bce_with_same_anchor_ranking


def entry(pair_id, a, b, label, *, group=9, unit="page", origin=None):
    return dict(pair_id=pair_id, label=label, artifact_path=pair_id+".npz", source_root="/fixture",
        s7_group_index=group, s7_recipe="partial_curve", source_stratum="fixture",
        source_row=dict(pair_id=pair_id, label=label, split="train",
            label_origin=origin or ("rachel_csv_neighbor_plus_independent_contour_seam" if label
                                   else "rachel_csv_same_folder_nonneighbor_no_seam"),
            fragment_a=dict(fragment_token=a, split_unit_id=unit),
            fragment_b=dict(fragment_token=b, split_unit_id=unit)))


def manifest(n=4):
    # Eligible anchor is SOURCE-side b in positive; negatives alternate sides.
    rows = [entry("p", "B", "A", True, group=17), entry("q", "D", "E", True, group=17)]
    rows += [entry("n%d" % i, "A" if i % 2 else "C%d" % i,
                   "C%d" % i if i % 2 else "A", False, group=100+i) for i in range(n)]
    rows += [entry("unrelated", "U", "V", False, group=17)]
    return dict(schema_version=SCHEMA, split="train", artifact_root="/unused", entries=rows)


def payload_and_ledger(m):
    blob = json.dumps(m).encode()
    proof = dict(verified=True, evidence=["fixture only: no resize; mask and points share transform"],
        frame_namespace="original-source-pixel-grid", pixel_scale_rc=[1., 1.], canvas_hw=[800, 800],
        mask_points_same_transform=True, both_endpoints_same_scale=True,
        pair_dependent_rescaling=False, rotation_degrees=0)
    ledger = dict(schema=SCALE_SCHEMA, manifest_sha256=hashlib.sha256(blob).hexdigest(),
        cohorts=[dict(source_roots=["/fixture"], recipes=["partial_curve"], proof=proof)])
    return blob, ledger


def sample(e, *, reverse=False):
    values = {}
    sides = "ba" if reverse else "ab"
    for output, source in zip("ab", sides):
        token = e["source_row"]["fragment_"+source]["fragment_token"]
        # Deliberately different AUGMENTED A in negative vs positive artifacts.
        start = 10 if e["label"] else 30 + ord(e["pair_id"][-1]) % 10
        mask = np.zeros((1, 800, 800), np.float32)
        mask[:, start:start+8, start:start+8] = 1
        points = np.array([[start, start], [start, start+7], [start+7, start+7], [start+7, start]], np.float32)
        values.update({"fragment_"+output+"_token":token, "mask_"+output:mask,
            "coarse_mask_"+output:np.zeros((1, 128, 128), np.float32),
            "points_rc_"+output:points, "contour_valid_"+output:np.ones(4, np.bool_),
            "target_"+output:np.full(4, -1, np.int64)})
    return RachelPairSample(pair_id=e["pair_id"], label=np.float32(e["label"]),
        translation_a_to_b_rc=np.zeros(2, np.float32),
        translation_a_to_b_xy_cartesian=np.zeros(2, np.float32),
        translation_valid=np.bool_(e["label"]), **values)


class TripletDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_deterministic_capped_separate_population_and_source_proof(self):
        m = manifest()
        before = deepcopy(m)
        blob, ledger = payload_and_ledger(m)
        index = build_triplet_index(blob, scale_ledger=ledger)
        self.assertEqual(index, build_triplet_index(blob, scale_ledger=ledger))
        self.assertEqual(m, before)
        self.assertEqual(index["ordinary_pair_count"], 7)
        self.assertEqual(index["candidate_positive_count"], 1)
        self.assertEqual(index["usable_group_count"], 1)
        group = index["groups"][0]
        self.assertEqual(group["anchor_token"], "A")
        self.assertEqual(len(group["negatives"]), 3)
        self.assertEqual(len({x["token"] for x in group["negatives"]}), 3)
        self.assertEqual(group["positive"]["entry"]["source_row"], m["entries"][0]["source_row"])
        self.assertTrue(all(x["entry"]["s7_group_index"] != 17 for x in group["negatives"]))

    def test_missing_and_unequal_scale_are_explicitly_excluded(self):
        blob, ledger = payload_and_ledger(manifest(1))
        missing = build_triplet_index(blob)
        self.assertEqual(missing["usable_group_count"], 0)
        self.assertEqual(missing["candidate_positive_count"], 1)
        self.assertEqual(missing["excluded_reason_counts"], {"missing_verified_scale_proof":1})
        unequal = deepcopy(ledger["cohorts"][0]["proof"])
        unequal["pixel_scale_rc"] = [2., 2.]
        ledger["pair_proofs"] = {"n0":unequal}
        result = build_triplet_index(blob, scale_ledger=ledger)
        self.assertEqual(result["excluded_reason_counts"], {"incompatible_physical_scale_or_frame":1})
        self.assertEqual(result["usable_group_count"], 0)
        ledger["manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            build_triplet_index(blob, scale_ledger=ledger)

    def test_never_invents_same_page_or_same_group_negative(self):
        m = manifest(0)
        m["entries"].append(entry("unverified", "A", "C", False, origin="unverified_same_page"))
        blob, ledger = payload_and_ledger(m)
        self.assertEqual(build_triplet_index(blob, scale_ledger=ledger)["usable_group_count"], 0)

    def test_holdout_conflicting_labels_and_missing_identity_fail(self):
        variants = []
        m = manifest(1); m["entries"][0]["source_row"]["split"] = "val"; variants.append(m)
        m = manifest(1); m["entries"].append(entry("conflict", "A", "B", False)); variants.append(m)
        m = manifest(1); del m["entries"][0]["source_row"]["fragment_a"]["split_unit_id"]; variants.append(m)
        for m in variants:
            blob, ledger = payload_and_ledger(m)
            with self.assertRaises(ValueError):
                build_triplet_index(blob, scale_ledger=ledger)

    def test_anchor_bytes_identical_even_mixed_output_and_artifact_swaps(self):
        m = manifest(3)
        blob, ledger = payload_and_ledger(m)
        group = build_triplet_index(blob, scale_ledger=ledger)["groups"][0]
        samples = {e["pair_id"]:sample(e, reverse=True) for e in m["entries"]}
        swaps = [False, True, False, True]
        batch = assemble_triplet_group(group, samples, swap=swaps, contour_cap=8)
        self.assertEqual(len(batch.inputs), 6)
        self.assertEqual(batch.inputs[0].shape, (4, 1, 800, 800))
        for kind in range(3):
            reference = batch.inputs[2*kind][0].numpy().tobytes()
            for i, swap in enumerate(swaps):
                self.assertEqual(reference, batch.inputs[2*kind+int(swap)][i].numpy().tobytes())
        # Positive artifact is swapped, therefore its A token is now side a.
        self.assertEqual(batch.provenance["anchor_source_side"], "a")
        np.testing.assert_array_equal(batch.inputs[0][0].numpy(), samples["p"].mask_a)
        negative = samples[group["negatives"][0]["entry"]["pair_id"]]
        neg_anchor = negative.mask_a if negative.fragment_a_token == "A" else negative.mask_b
        self.assertFalse(np.array_equal(batch.inputs[0][0].numpy(), neg_anchor))
        self.assertTrue(batch.same_anchor_confirmed)
        loss = pair_bce_with_same_anchor_ranking(torch.tensor([.4, .3, .2, .1]),
            torch.zeros(4), batch.labels, known_negative_pairs=batch.known_negative_pairs,
            anchor_ids=batch.anchor_ids, same_anchor_confirmed=batch.same_anchor_confirmed)
        self.assertEqual(loss.ranked_positive_count, 1)
        self.assertFalse(batch.provenance["partner_conditioned_features_used"])

    def test_assembly_rejects_unproved_scale_or_mutated_endpoint(self):
        m = manifest(1); blob, ledger = payload_and_ledger(m)
        group = build_triplet_index(blob, scale_ledger=ledger)["groups"][0]
        samples = {e["pair_id"]:sample(e) for e in m["entries"]}
        bad = deepcopy(group); bad["negatives"][0]["scale_proof"] = None
        with self.assertRaisesRegex(ValueError, "scale"):
            assemble_triplet_group(bad, samples)
        samples["n0"] = replace(samples["n0"], fragment_a_token="wrong")
        with self.assertRaisesRegex(ValueError, "endpoints"):
            assemble_triplet_group(group, samples)

    def test_existing_materialized_loader_and_original_pairs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); m = manifest(1); m["artifact_root"] = str(root)
            for e in m["entries"]:
                save_sample(root/e["artifact_path"], sample(e, reverse=True),
                    dict(changed_pair=False, pose_supervision_enabled=e["label"]))
            blob, ledger = payload_and_ledger(m)
            path = root/"train.json"; path.write_bytes(blob)
            dataset = SameAnchorTripletDataset(path, scale_ledger=ledger)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset.ordinary), len(m["entries"]))
            original = dataset.ordinary[0]
            result = dataset.get(0, swap=True)
            self.assertTrue(result.provenance["artifact_receipts"]["p"]["sha256"])
            np.testing.assert_array_equal(original.mask_a, dataset.ordinary[0].mask_a)
            self.assertEqual(path.read_bytes(), blob)


if __name__ == "__main__":
    unittest.main()
