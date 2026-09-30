"""CPU fixture checks only; no real preparation, GPU work or formal training."""
from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_density as trainer
from experiments.rachel_n512_formal_30k import train_score_design as live
from experiments.rachel_n512_formal_30k import train_score_input_variant as input_trainer
from experiments.rachel_n512_formal_30k.test_train_score_input_variant import source_metadata, tiny_loader
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal
from experiments.rachel_n512_formal_30k.test_train_score_design import fake_validation
from experiments.rachel_n512_formal_30k.test_score_design_stages import inputs
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample


@contextmanager
def raises(exception, message=".*"):
    with unittest.TestCase().assertRaisesRegex(exception, message):
        yield


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trainer.canonical(value)))


class SizedNamespace(SimpleNamespace):
    def __len__(self):
        return len(self.entries if hasattr(self, "entries") else self.rows)


def population_fixture(root, cap=512, source_density_version="v1"):
    """Full counts are mock metadata, explicitly not24000 materialized samples."""
    derivative, cache = root / "paired", root / "clean"
    original = root / "original_fixed.json"
    dump(original, {"fixture": "original fixed data identity only"})
    entries = [dict(pair_id="train_%d" % i) for i in range(24000)]
    number = int(source_density_version[1:])
    pipeline = dict(version="paired-source-cell-density-materialization/%d" % number,
        caps=[512, 1024], smoothing_sigma=3.)
    if source_density_version != "v1":
        pipeline.update(source_density_version=source_density_version,
            ownership_protocol=trainer.OWNERSHIP_PROTOCOLS[source_density_version])
    selection = dict(schema_version="paired-source-cell-density-materialization/%d" % number, source_manifest=str(original),
        source_manifest_sha256=trainer._sha256(original), canonical_root=str(root),
        selection_mode="full_source_manifest", selected_pair_ids=[e["pair_id"] for e in entries],
        source_indices=list(range(24000)), pipeline=pipeline)
    identity = trainer.canonical_digest(selection)
    dump(derivative / "source_selection.json", dict(selection, identity_sha256=identity))
    protocol = dict(selection_mode="full_source_manifest", paired512_control_required=True,
        live_dataset_modified=False, pipeline=pipeline, source_manifest=str(original),
        source_manifest_sha256=trainer._sha256(original), companion_manifest="train_n%d.json" % (1024 if cap == 512 else 512))
    for n in (512, 1024):
        dump(derivative / ("train_n%d.json" % n), dict(schema_version="rachel-paired-source-density-train/1",
            contour_cap=n, status="complete", failed_pair_count=0, completed_pair_count=24000,
            identity_sha256=identity, entries=entries, protocol=dict(pipeline=pipeline)))
    training = SizedNamespace(root=derivative, contour_cap=cap, entries=entries, identity=identity,
        manifest_path=derivative / ("train_n%d.json" % cap), protocol=protocol)
    val_rows = [dict(pair_id="val_%d" % i, label=i < 1500) for i in range(3000)]
    val_manifest = root / "release/pairs/val.jsonl"
    dump(val_manifest, val_rows)
    common = dict(schema_version="rachel-clean-source-density-eval/%d" % number, root=str(root / "release"), split="val",
        source_manifest_sha256=trainer._sha256(val_manifest), source_pair_count=3000, weathering_applied=False,
        checkpoint_threshold_selection_permitted=True, test_used_for_selection=False,
        smoothing_sigma=3., assignment_targets="fixture real source-cell targets", python="fixture")
    version_fields = {}
    if source_density_version != "v1":
        version_fields = dict(source_density_version=source_density_version,
            ownership_protocol=trainer.OWNERSHIP_PROTOCOLS[source_density_version])
        common.update(version_fields)
    protocols = {}
    for n in (512, 1024):
        p = dict(common, contour_cap=n)
        protocols[n] = dict(p, identity_sha256=trainer.canonical_digest(p))
        dump(cache / ("val_n%d" % n) / "cache_identity.json", protocols[n])
    records = [dict(row, caps={str(n): dict(positive_matches=3 if row["label"] else 0) for n in (512, 1024)}) for row in val_rows]
    dump(cache / "clean_val_preparation.json", dict(schema_version="clean-source-density-preparation/1",
        status="complete", split="val", full_split=True, failures=[], original_split_count=3000,
        selected_count=3000, cached_pair_count=3000, pair_ids=[r["pair_id"] for r in val_rows], records=records,
        model_inference=False, model_or_threshold_selected=False, **version_fields))
    validation = SizedNamespace(rows=val_rows, contour_cap=cap, split="val", cache_dir=cache / ("val_n%d" % cap),
        protocol=protocols[cap], identity=protocols[cap]["identity_sha256"], manifest_path=val_manifest,
        stats=dict(positive=1500, negative=1500))
    return training, validation, cache


def fixture_identity(root, built, config, digests, populations):
    source = root / "source.pt"
    source.write_bytes(b"fixture source checkpoint identity")
    args = SimpleNamespace(architecture=built.metadata["architecture"], contour_cap=built.metadata["contour_cap"],
        workers=4, stop_after_epoch=5, output="one", log_every=100,
        source_density_version=populations.get("source_density_version", "v1"))
    kwargs = dict(source_path=source, model_metadata=built.metadata, config=config, digests=digests, populations=populations)
    identity = trainer.experiment_identity(args, **kwargs)
    args.stop_after_epoch, args.output, args.log_every = 20, "two", 8
    assert trainer.experiment_identity(args, **kwargs) == identity
    return identity


def test_full_population_and_common_cap_pipeline_bindings(tmp_path):
    t, v, cache = population_fixture(tmp_path)
    first = trainer.validate_populations(t, v, 512, cache)
    t2, v2, _ = population_fixture(tmp_path, 1024)
    second = trainer.validate_populations(t2, v2, 1024, cache)
    for key in ("train_source_selection_sha256", "train_source_selection_file_sha256", "train_source_pipeline",
                "train_original_manifest_sha256", "validation_manifest_sha256", "validation_common_pipeline_sha256"):
        assert first[key] == second[key], key
    assert first["train_manifest_sha256"] != second["train_manifest_sha256"]
    assert first["validation_density_identity_sha256"] != second["validation_density_identity_sha256"]
    t2.entries = t2.entries[:12]
    with raises(ValueError, "TRAIN24000"): trainer.validate_populations(t2, v2, 1024, cache)


def test_reject_smoke_clean_val_missing_targets_mixed_pipeline_and_wrong_selection(tmp_path):
    t, v, cache = population_fixture(tmp_path)
    receipt_path = cache / "clean_val_preparation.json"
    receipt = json.loads(receipt_path.read_text())
    for broken in (dict(receipt, full_split=False), dict(receipt, failures=[{"pair_id": "bad"}]),
                   dict(receipt, cached_pair_count=12)):
        dump(receipt_path, broken)
        with raises(ValueError, "clean_val_preparation"): trainer.validate_populations(t, v, 512, cache)
    missing_targets = copy.deepcopy(receipt)
    missing_targets["records"][0]["caps"]["1024"]["positive_matches"] = 0
    dump(receipt_path, missing_targets)
    with raises(ValueError, "genuine positive"): trainer.validate_populations(t, v, 512, cache)
    dump(receipt_path, receipt)
    path = cache / "val_n1024/cache_identity.json"
    pipeline = json.loads(path.read_text()); pipeline["smoothing_sigma"] = 4.
    pipeline["identity_sha256"] = trainer.canonical_digest({k: x for k, x in pipeline.items() if k != "identity_sha256"})
    dump(path, pipeline)
    with raises(ValueError, "same original pipeline"): trainer.validate_populations(t, v, 512, cache)
    t.protocol["selection_mode"] = "explicit_subset"
    with raises(ValueError, "complete paired"): trainer.validate_populations(t, v, 512, cache)


def test_caps_same_initial_tensors_rng_and_actual_one_update(tmp_path):
    """16 synthetic examples per cap; cap not reached in this cheap CPU probe."""
    torch.set_num_threads(1)
    assert trainer.train_density_segment is live.train_score_segment
    states, reports, opts, digests_list, rng = [], [], [], [], []
    for cap in (512, 1024):
        built, config, digests = trainer.build_training_model(source_metadata(), cap, "candidate_dual")
        root = tmp_path / str(cap); root.mkdir()
        dump(root / "status.json", dict(global_exposure=0, optimizer_updates=0))
        args = SimpleNamespace(output=str(root), architecture="candidate_dual", log_every=100)
        optimizer = trainer.create_optimizer(built.model, "candidate_dual")
        reports.append(trainer.train_density_segment(built.model, tiny_loader(16), optimizer, config, torch.device("cpu"), args, 1))
        states.append(copy.deepcopy(built.model.state_dict())); opts.append(copy.deepcopy(optimizer.state_dict()))
        digests_list.append(digests); rng.append(torch.get_rng_state())
        assert trainer.state_digest(built.model) != digests["initial_weights_sha256"]
    assert digests_list[0] == digests_list[1]
    assert tensor_tree_equal(states[0], states[1]) and tensor_tree_equal(opts[0], opts[1])
    assert torch.equal(rng[0], rng[1])
    assert reports[0]["loss_components"] == reports[1]["loss_components"]
    assert reports[0]["samples"] == reports[1]["samples"] == 16
    assert reports[0]["optimizer_updates"] == reports[1]["optimizer_updates"] == 1
    assert reports[0]["candidate_correctness_counts"]["valid"] > 0


def test_density_checkpoint_resume_separation_and_freeze(tmp_path):
    t, v, cache = population_fixture(tmp_path, 1024)
    populations = trainer.validate_populations(t, v, 1024, cache)
    built, config, digests = trainer.build_training_model(source_metadata(), 1024, "candidate_pair")
    identity = fixture_identity(tmp_path, built, config, digests, populations)
    optimizer = trainer.create_optimizer(built.model, "candidate_pair")
    next(p for p in built.model.parameters() if p.requires_grad).sum().backward(); optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity, config=config,
        training_data={}, completed=4, winners={}, role="cpu_fixture_counts")
    path = tmp_path / "fixture.pt"; torch.save(saved, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    fresh = trainer.build_training_model(source_metadata(), 1024, "candidate_pair")[0]
    fresh_opt = trainer.create_optimizer(fresh.model, "candidate_pair")
    assert trainer.restore_training_state(fresh.model, fresh_opt, saved, identity) == (4, {})
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert tensor_tree_equal(optimizer.state_dict(), fresh_opt.state_dict())
    assert trainer.state_digest(fresh.model) == trainer.state_digest(built.model)
    assert trainer.load_score_density_checkpoint(saved).config.contour_cap == 1024
    with raises(ValueError): live.load_score_checkpoint(saved)
    with raises(ValueError): input_trainer.load_score_input_checkpoint(saved)
    with raises(ValueError, "identity"): trainer.validate_density_checkpoint(saved, dict(identity, contour_cap=512))
    broken = copy.deepcopy(saved)
    broken["model_state_dict"]["base_model.patch_sampler.offsets_rc"].add_(1)
    with raises(ValueError, "sampling"): trainer.load_score_density_checkpoint(broken)
    report, points = fake_validation(.8, .9, .7)
    winners = trainer.update_winners({}, root=tmp_path, epoch=5, report=report, points=points)
    (tmp_path / "epoch_005.pt").write_bytes(b"fixture-winner")
    (tmp_path / "validation_005_rows.json").write_text("[]")
    trainer.publish_freezes(tmp_path, epoch=5, winners=winners, identity=identity)
    freeze = json.loads((tmp_path / "budget_freezes/005/freeze.json").read_text())
    assert freeze["schema_version"] == trainer.SCHEMA and freeze["contour_cap"] == 1024
    assert freeze["selection_population"] == "source-rebuilt cleanVAL3000 only"
    assert freeze["resume_identity_sha256"] == trainer.canonical_digest(identity)
    assert freeze["winners"]["recall95"]["validation_predictions_sha256"] == trainer._sha256(tmp_path / "validation_005_rows.json")
    assert freeze["paired_source_rebuilt512_control_required"] and not freeze["legacy512_baseline_permitted"]


def test_real_cap_collation_keeps_fresh_targets_not_inference_all_ignore():
    a, b, _, _, _, _ = inputs()
    for cap, count in ((512, 510), (1024, 900)):
        xy = np.stack(np.unravel_index(np.arange(count), (30, 30)), axis=1).astype(np.float32)
        sample = RachelPairSample(pair_id="cpu_source", fragment_a_token="a", fragment_b_token="b",
            mask_a=a[0].numpy(), mask_b=b[0].numpy(), coarse_mask_a=np.zeros((1, 128, 128), np.float32),
            coarse_mask_b=np.zeros((1, 128, 128), np.float32), points_rc_a=xy, points_rc_b=xy.copy(),
            contour_valid_a=np.ones(count, bool), contour_valid_b=np.ones(count, bool),
            target_a=np.arange(count), target_b=np.arange(count), label=np.float32(1),
            translation_a_to_b_rc=np.zeros(2, np.float32), translation_a_to_b_xy_cartesian=np.zeros(2, np.float32),
            translation_valid=np.bool_(True))
        data = [(sample, dict(pose_supervision_enabled=True, changed_pair=False))]
        train = trainer.make_weathering_loader(data, [0], batch_size=1, num_workers=0, seed=trainer.SEED, contour_cap=cap)
        batch = next(iter(train)).batch
        assert batch.points_rc_a.shape == (1, cap, 2) and batch.contour_valid_a.sum() == count
        assert (batch.target_a >= 0).sum() == count and (batch.target_a == -2).sum() == cap - count
        val = trainer.make_ablation_loader([sample], [0], batch_size=1, num_workers=0, seed=trainer.SEED, contour_cap=cap)
        assert np.array_equal(next(iter(val)).target_a, batch.target_a)


def test_cuda_guard_exact_budget_schedule_and_no_input_cross_axis(tmp_path):
    args = trainer.parser().parse_args(["--checkpoint", "absent", "--dataset", "absent", "--train-density-manifest", "absent",
        "--clean-density-cache-root", "absent", "--contour-cap", "1024", "--architecture", "original",
        "--output", str(tmp_path / "never"), "--smoke", "32", "--device", "cpu"])
    assert args.stop_after_epoch == 50
    assert args.source_density_version == "v1"
    with raises(RuntimeError, "remote Linux CUDA"): trainer.run(args)
    assert not (tmp_path / "never").exists()
    assert trainer.segment_plan() == live.segment_plan()
    assert trainer.segment_plan()[-1]["global_stop"] == 1200000
    assert not hasattr(args, "coarse_size") and not hasattr(args, "single_window")


def test_explicit_v2_populations_and_v1_identity_compatibility(tmp_path):
    first_root, second_root = tmp_path / "v1", tmp_path / "v2"
    t1, v1, c1 = population_fixture(first_root)
    p1 = trainer.validate_populations(t1, v1, 512, c1)
    assert "source_density_version" not in p1 and "source_density_consumer" not in p1
    assert trainer.identity_source_density_version(p1) == "v1"
    t2, v2, c2 = population_fixture(second_root, source_density_version="v2")
    with raises(ValueError, "TRAIN producer version"):
        trainer.validate_populations(t2, v2, 512, c2)  # No automatic version inference.
    p2 = trainer.validate_populations(t2, v2, 512, c2, "v2")
    assert trainer.identity_source_density_version(p2) == "v2"
    assert p2["source_density_consumer"]["clean_reader"] == "CleanSourceDensityV2Dataset"
    with raises(ValueError, "TRAIN producer version"):
        trainer.identity_source_density_version({k: v for k, v in p2.items() if k != "source_density_version"})
    with raises(ValueError, "TRAIN producer version"):
        trainer.validate_populations(t1, v1, 512, c1, "v2")
    receipt_path = c2 / "clean_val_preparation.json"
    receipt = json.loads(receipt_path.read_text()); receipt.pop("source_density_version")
    dump(receipt_path, receipt)
    with raises(ValueError, "receipt version"):
        trainer.validate_populations(t2, v2, 512, c2, "v2")


def test_v2_rejects_mixed_cap_cache_or_companion_manifest(tmp_path):
    t, v, cache = population_fixture(tmp_path, source_density_version="v2")
    counterpart = cache / "val_n1024/cache_identity.json"
    good = json.loads(counterpart.read_text())
    wrong = dict(good, schema_version="rachel-clean-source-density-eval/1")
    wrong.pop("source_density_version"); wrong.pop("ownership_protocol")
    wrong["identity_sha256"] = trainer.canonical_digest({k: x for k, x in wrong.items() if k != "identity_sha256"})
    dump(counterpart, wrong)
    with raises(ValueError, "clean-cache producer version"):
        trainer.validate_populations(t, v, 512, cache, "v2")
    dump(counterpart, good)
    paired_path = t.root / "train_n1024.json"
    paired = json.loads(paired_path.read_text())
    paired["protocol"]["pipeline"]["version"] = "paired-source-cell-density-materialization/1"
    dump(paired_path, paired)
    with raises(ValueError, "TRAIN producer version"):
        trainer.validate_populations(t, v, 512, cache, "v2")


def test_v1_reader_is_lazy_and_v2_does_not_fall_back():
    module = "staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v2"
    with patch.dict(sys.modules, {module: None}):
        assert trainer.get_clean_density_reader("v1") is trainer.CleanSourceDensityDataset
        with raises(ImportError): trainer.get_clean_density_reader("v2")
    with raises(ValueError): trainer.get_clean_density_reader("v5")
    v2_protocol = dict(schema_version="rachel-clean-source-density-eval/2",
        source_density_version="v2", ownership_protocol=trainer.OWNERSHIP_V2)
    returned = SimpleNamespace(protocol=v2_protocol)
    with patch.object(trainer, "get_clean_density_reader", return_value=lambda *args: returned) as reader:
        assert trainer.make_clean_density_dataset("root", "val", 1024, "cache", source_density_version="v2") is returned
        reader.assert_called_once_with("v2")
    with patch.object(trainer, "get_clean_density_reader", return_value=lambda *args: returned):
        with raises(ValueError, "clean-cache producer version"):
            trainer.make_clean_density_dataset("root", "val", 1024, "cache", source_density_version="v1")


def test_v2_checkpoint_preserves_version_and_refuses_relabeling(tmp_path):
    t, v, cache = population_fixture(tmp_path, source_density_version="v2")
    populations = trainer.validate_populations(t, v, 512, cache, "v2")
    built, config, digests = trainer.build_training_model(source_metadata(), 512, "original")
    identity = fixture_identity(tmp_path, built, config, digests, populations)
    optimizer = trainer.create_optimizer(built.model, "original")
    saved = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity, config=config,
        training_data={}, completed=0, winners={}, role="cpu_v2_fixture")
    assert trainer.load_score_density_checkpoint(saved).config.contour_cap == 512
    wrong = copy.deepcopy(saved)
    wrong["resume_identity"].pop("source_density_version")
    with raises(ValueError, "TRAIN producer version"): trainer.load_score_density_checkpoint(wrong)
    wrong = copy.deepcopy(saved)
    wrong["resume_identity"]["source_density_consumer"]["clean_reader"] = "CleanSourceDensityDataset"
    with raises(ValueError, "registered consumer"): trainer.load_score_density_checkpoint(wrong)


def test_v3_keeps_v2_descriptor_unchanged_and_cannot_consume_v2(tmp_path):
    p2 = trainer.validate_populations(*population_fixture(tmp_path / "old", source_density_version="v2")[:2],
        512, tmp_path / "old/clean", "v2")
    assert p2["source_density_consumer"] == dict(
        module="staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v2",
        clean_reader="CleanSourceDensityV2Dataset", ownership_protocol="density-local-unknown-ownership/2")
    t3, v3, c3 = population_fixture(tmp_path / "new", source_density_version="v3")
    p3 = trainer.validate_populations(t3, v3, 512, c3, "v3")
    assert trainer.identity_source_density_version(p3) == "v3"
    assert p3["source_density_consumer"] == dict(
        module="staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v3",
        clean_reader="CleanSourceDensityV3Dataset", ownership_protocol="density-local-unknown-ownership/3")
    assert set(p3) == set(p2)  # No newly invented v2 trajectory field/default.
    with raises(ValueError, "TRAIN producer version"):
        trainer.validate_populations(t3, v3, 512, c3, "v2")
    p3_without_version = {k: v for k, v in p3.items() if k != "source_density_version"}
    with raises(ValueError, "TRAIN producer version"):
        trainer.identity_source_density_version(p3_without_version)
    p3_wrong_reader = dict(p3, source_density_consumer=p2["source_density_consumer"])
    with raises(ValueError, "registered consumer"):
        trainer.identity_source_density_version(p3_wrong_reader)
    receipt = json.loads((c3 / "clean_val_preparation.json").read_text())
    receipt.update(source_density_version="v2", ownership_protocol=trainer.OWNERSHIP_V2)
    dump(c3 / "clean_val_preparation.json", receipt)
    with raises(ValueError, "receipt version"):
        trainer.validate_populations(t3, v3, 512, c3, "v3")


def test_missing_v3_module_never_falls_back_to_v2():
    v3 = "staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v3"
    with patch.dict(sys.modules, {v3: None}):
        assert trainer.get_clean_density_reader("v1") is trainer.CleanSourceDensityDataset
        # v2 remains usable even if only the v3 module is absent.
        assert trainer.get_clean_density_reader("v2").__name__ == "CleanSourceDensityV2Dataset"
        with raises(ImportError): trainer.get_clean_density_reader("v3")
    v2_protocol = dict(schema_version="rachel-clean-source-density-eval/2",
        source_density_version="v2", ownership_protocol=trainer.OWNERSHIP_V2)
    with patch.object(trainer, "get_clean_density_reader", return_value=lambda *args: SimpleNamespace(protocol=v2_protocol)):
        with raises(ValueError, "clean-cache producer version"):
            trainer.make_clean_density_dataset("root", "val", 1024, "cache", source_density_version="v3")


def test_v3_checkpoint_roundtrip_and_both_caps_require_v3(tmp_path):
    t, v, cache = population_fixture(tmp_path, cap=1024, source_density_version="v3")
    populations = trainer.validate_populations(t, v, 1024, cache, "v3")
    built, config, digests = trainer.build_training_model(source_metadata(), 1024, "original")
    identity = fixture_identity(tmp_path, built, config, digests, populations)
    optimizer = trainer.create_optimizer(built.model, "original")
    saved = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity, config=config,
        training_data={}, completed=0, winners={}, role="cpu_v3_fixture")
    assert trainer.load_score_density_checkpoint(saved).config.contour_cap == 1024
    wrong = copy.deepcopy(saved); wrong["resume_identity"]["source_density_version"] = "v2"
    with raises(ValueError, "TRAIN producer version"): trainer.load_score_density_checkpoint(wrong)
    path = cache / "val_n512/cache_identity.json"
    data = json.loads(path.read_text())
    data.update(schema_version="rachel-clean-source-density-eval/2", source_density_version="v2", ownership_protocol=trainer.OWNERSHIP_V2)
    data["identity_sha256"] = trainer.canonical_digest({k: x for k, x in data.items() if k != "identity_sha256"})
    dump(path, data)
    with raises(ValueError, "clean-cache producer version"):
        trainer.validate_populations(t, v, 1024, cache, "v3")


def test_v4_explicit_populations_keep_prior_descriptors_and_reject_mixed_receipts(tmp_path):
    previous_fields = None
    for version in ("v2", "v3", "v4"):
        t, v, cache = population_fixture(tmp_path / version, source_density_version=version)
        populations = trainer.validate_populations(t, v, 512, cache, version)
        number = version[1:]
        assert populations["source_density_consumer"] == dict(
            module="staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_" + version,
            clean_reader="CleanSourceDensityV" + number + "Dataset",
            ownership_protocol="density-local-unknown-ownership/" + number)
        assert trainer.identity_source_density_version(populations) == version
        if previous_fields is not None:
            assert set(populations) == previous_fields
        previous_fields = set(populations)
    for version in ("v1", "v2", "v3"):
        with raises(ValueError, "TRAIN producer version"):
            trainer.validate_populations(t, v, 512, cache, version)
    receipt_path = cache / "clean_val_preparation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(source_density_version="v3", ownership_protocol=trainer.OWNERSHIP_V3)
    dump(receipt_path, receipt)
    with raises(ValueError, "receipt version"):
        trainer.validate_populations(t, v, 512, cache, "v4")


def test_missing_v4_module_never_falls_back_and_v4_reader_cannot_return_v3_cache():
    module = "staging.pairwise_v0_2.pairwise_data.rachel_density_ownership_v4"
    with patch.dict(sys.modules, {module: None}):
        for version in ("v1", "v2", "v3"):
            assert trainer.get_clean_density_reader(version) is not None
        with raises(ImportError):
            trainer.get_clean_density_reader("v4")
    protocol = dict(schema_version="rachel-clean-source-density-eval/3",
        source_density_version="v3", ownership_protocol=trainer.OWNERSHIP_V3)
    with patch.object(trainer, "get_clean_density_reader", return_value=lambda *args: SimpleNamespace(protocol=protocol)):
        for split in ("val", "test"):
            with raises(ValueError, "clean-cache producer version"):
                trainer.make_clean_density_dataset("root", split, 1024, "cache", source_density_version="v4")


def test_v4_checkpoint_roundtrip_preserves_weights_and_cap_version_binding(tmp_path):
    t, v, cache = population_fixture(tmp_path, cap=1024, source_density_version="v4")
    populations = trainer.validate_populations(t, v, 1024, cache, "v4")
    built, config, digests = trainer.build_training_model(source_metadata(), 1024, "original")
    identity = fixture_identity(tmp_path, built, config, digests, populations)
    optimizer = trainer.create_optimizer(built.model, "original")
    saved = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity, config=config,
        training_data={}, completed=0, winners={}, role="cpu_v4_fixture")
    restored = trainer.load_score_density_checkpoint(saved)
    assert restored.config.contour_cap == 1024
    assert tensor_tree_equal(restored.state_dict(), built.model.state_dict())
    assert saved["loss_config"] == asdict(config)
    for prior in ("v1", "v2", "v3"):
        wrong = copy.deepcopy(saved)
        wrong["resume_identity"]["source_density_version"] = prior
        with raises(ValueError, "TRAIN producer version"):
            trainer.load_score_density_checkpoint(wrong)
    path = cache / "val_n512/cache_identity.json"
    data = json.loads(path.read_text())
    data.update(schema_version="rachel-clean-source-density-eval/3", source_density_version="v3", ownership_protocol=trainer.OWNERSHIP_V3)
    data["identity_sha256"] = trainer.canonical_digest({k: x for k, x in data.items() if k != "identity_sha256"})
    dump(path, data)
    with raises(ValueError, "clean-cache producer version"):
        trainer.validate_populations(t, v, 1024, cache, "v4")


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = [test_full_population_and_common_cap_pipeline_bindings,
        test_reject_smoke_clean_val_missing_targets_mixed_pipeline_and_wrong_selection,
        test_caps_same_initial_tensors_rng_and_actual_one_update,
        test_density_checkpoint_resume_separation_and_freeze,
        test_real_cap_collation_keeps_fresh_targets_not_inference_all_ignore,
        test_cuda_guard_exact_budget_schedule_and_no_input_cross_axis,
        test_explicit_v2_populations_and_v1_identity_compatibility,
        test_v2_rejects_mixed_cap_cache_or_companion_manifest,
        test_v1_reader_is_lazy_and_v2_does_not_fall_back,
        test_v2_checkpoint_preserves_version_and_refuses_relabeling,
        test_v3_keeps_v2_descriptor_unchanged_and_cannot_consume_v2,
        test_missing_v3_module_never_falls_back_to_v2,
        test_v3_checkpoint_roundtrip_and_both_caps_require_v3,
        test_v4_explicit_populations_keep_prior_descriptors_and_reject_mixed_receipts,
        test_missing_v4_module_never_falls_back_and_v4_reader_cannot_return_v3_cache,
        test_v4_checkpoint_roundtrip_preserves_weights_and_cap_version_binding]
    for test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory).resolve()) if test.__code__.co_argcount else test()
        print(test.__name__ + ": PASS", flush=True)
