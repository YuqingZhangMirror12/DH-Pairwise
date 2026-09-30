"""CPU-only input trainer checks; tiny fixtures never count as formal training."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_input_variant as trainer
from experiments.rachel_n512_formal_30k import train_score_design as live
from experiments.rachel_n512_formal_30k.score_design_input_variants import InputVariantSpec, full24_reference_config
from experiments.rachel_n512_formal_30k.test_score_design_stages import inputs
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal
from experiments.rachel_n512_formal_30k.test_train_score_design import fake_validation
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig
from staging.pairwise_v0_2.training.rachel_weathering_training import WeatheringBatch


@contextmanager
def raises(exception, message=".*"):
    with unittest.TestCase().assertRaisesRegex(exception, message):
        yield


def source_metadata():
    return dict(model_kind="full", model_options={}, model_config=asdict(full24_reference_config()),
        loss_config=asdict(RachelN512LossConfig()),
        model_state_dict={"ignored_pretrained_weight": torch.tensor(float("nan"))})


def identity_fixture(root, built, config, digests):
    source, manifest = root / "source.pt", root / "train.json"
    source.write_bytes(b"fixture-metadata-not-real-training")
    manifest.write_text("fixture-materialized-manifest")
    (root / "pairs").mkdir(exist_ok=True)
    (root / "pairs/val.jsonl").write_text("fixture-validation-manifest")
    args = SimpleNamespace(dataset=str(root), architecture=built.metadata["architecture"],
        workers=4, stop_after_epoch=5, output="one", log_every=100)
    kw = dict(source_path=source, metadata=built.metadata, loss_config=config,
        training=SimpleNamespace(manifest_path=manifest), digests=digests)
    identity = trainer.experiment_identity(args, **kw)
    args.stop_after_epoch, args.output, args.log_every = 20, "two", 9
    assert identity == trainer.experiment_identity(args, **kw)
    return identity


def tiny_loader(count=32):
    """Eight contour positions, real800 canvas, micro4/effective16, no padding512.

    Model N512 is an upper bound. This synthetic loader intentionally has eight
    valid samples per contour to exercise the exact production segment cheaply.
    Each microbatch includes clean/corroded positive and negative supervision.
    """
    a, b, pa, pb, va, vb = inputs()
    masks = []
    for small in (a, b):
        mask = np.zeros((4, 1, 800, 800), np.float32)
        mask[:, :, :80, :80] = small.numpy()
        masks.append(mask)
    points = [np.repeat(p.numpy(), 4, axis=0) for p in (pa, pb)]
    valid = np.ones((4, 8), np.bool_)
    target = np.repeat(np.arange(8)[None], 4, axis=0)
    target[2:] = -1
    labels = np.asarray([1, 1, 0, 0], np.float32)
    pose = np.asarray([1, 0, 0, 0], np.bool_)
    batch = SimpleNamespace(pair_ids=tuple("cpu-%d" % i for i in range(4)),
        mask_a=masks[0], mask_b=masks[1], points_rc_a=points[0], points_rc_b=points[1],
        contour_valid_a=valid, contour_valid_b=valid, target_a=target, target_b=target,
        labels=labels, translation_a_to_b_rc=np.zeros((4, 2), np.float32),
        translation_valid=labels.astype(np.bool_))
    reports = []
    for i in range(4):
        changed = bool(i % 2)
        reports.append(dict(changed_pair=changed, changed_a=changed, changed_b=False,
            pose_supervision_enabled=bool(pose[i]),
            side_a=dict(tier="fixture_corroded" if changed else "clean", original_area_px=1000),
            side_b=dict(tier="clean", original_area_px=1000)))
    wrapped = WeatheringBatch(batch, tuple(reports), pose)
    class Loader:
        dataset = range(count)
        def __len__(self): return count // 4
        def __iter__(self): return iter([wrapped] * len(self))
    return Loader()


def test_cli_one_axis_cuda_guard_and_nested_plan(tmp_path):
    argv = ["--checkpoint", "absent", "--dataset", "absent", "--train-materialized-manifest", "absent",
        "--architecture", "candidate_pair", "--output", str(tmp_path / "never"), "--device", "cpu"]
    args = trainer.parser().parse_args(argv + ["--single-window", "7", "--smoke", "32"])
    assert trainer.input_spec(args).window_sizes_px == (7.,)
    with raises(RuntimeError, "remote Linux CUDA"): trainer.run(args)
    assert not (tmp_path / "never").exists()
    mixed = trainer.parser().parse_args(argv + ["--coarse-size", "256", "--transport-fusion", "post"])
    with raises(ValueError): trainer.run(mixed)
    assert trainer.segment_plan() == live.segment_plan()
    assert trainer.segment_plan()[-1]["global_stop"] == 1200000
    assert trainer.train_input_segment is live.train_score_segment


def test_identity_binds_input_and_source_loss_not_pause(tmp_path):
    built, config, digests = trainer.build_training_model(source_metadata(), InputVariantSpec(coarse_size=256), "original")
    identity = identity_fixture(tmp_path, built, config, digests)
    assert identity["max_epochs"] == 50 and identity["source_weights_loaded"] is False
    assert identity["base_model_metadata"]["model_config"]["coarse_size"] == 256
    assert identity["reference_base_model_metadata"]["model_config"]["coarse_size"] == 128
    assert identity["input_spec"]["contour_cap"] == 512
    optimizer = trainer.create_optimizer(built.model, "original")
    checkpoint = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity,
        loss_config=config, training_data={}, completed=0, winners={}, role="cpu_fixture")
    trainer.validate_input_checkpoint(checkpoint, identity)
    with raises(ValueError, "identity"):
        trainer.validate_input_checkpoint(checkpoint, dict(identity, score_design="candidate_pair"))
    with raises(ValueError, "counts"):
        trainer.validate_input_checkpoint(dict(checkpoint, optimizer_updates=1), identity)
    with raises(ValueError, "unchanged Full24"):
        trainer.build_training_model(dict(source_metadata(), model_config={}), InputVariantSpec(), "original")


def test_post_checkpoint_optimizer_rng_restore_and_legacy_rejection(tmp_path):
    built, config, digests = trainer.build_training_model(source_metadata(), InputVariantSpec(transport_fusion="post"), "candidate_dual")
    identity = identity_fixture(tmp_path, built, config, digests)
    optimizer = trainer.create_optimizer(built.model, "candidate_dual")
    # A dummy parameter update tests optimizer serialization, not model fitting.
    parameter = next(p for p in built.model.parameters() if p.requires_grad)
    parameter.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = trainer.payload(built.model, optimizer, metadata=built.metadata, identity=identity,
        loss_config=config, training_data={}, completed=4, winners={}, role="cpu_fixture_counts")
    torch.save(checkpoint, tmp_path / "fixture.pt")
    checkpoint = torch.load(tmp_path / "fixture.pt", map_location="cpu", weights_only=False)
    assert checkpoint["base_model_kind"] == "multiscale_transport"
    original_digest = state_digest(built.model)
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    fresh = trainer.build_training_model(source_metadata(), InputVariantSpec(transport_fusion="post"), "candidate_dual")[0]
    fresh_optimizer = trainer.create_optimizer(fresh.model, "candidate_dual")
    assert trainer.restore_training_state(fresh.model, fresh_optimizer, checkpoint, identity) == (4, {})
    actual_random = (random.random(), np.random.rand(), torch.rand(2))
    assert actual_random[:2] == expected_random[:2] and torch.equal(actual_random[2], expected_random[2])
    assert state_digest(fresh.model) == original_digest
    assert tensor_tree_equal(optimizer.state_dict(), fresh_optimizer.state_dict())
    assert state_digest(trainer.load_score_input_checkpoint(checkpoint)) == original_digest
    with raises(ValueError): live.load_score_checkpoint(checkpoint)
    with raises(ValueError, "optimizer"):
        trainer.load_score_input_checkpoint({k: v for k, v in checkpoint.items() if k != "optimizer_state_dict"})
    damaged = copy.deepcopy(checkpoint)
    offset_name = next(k for k in damaged["model_state_dict"] if k.endswith("offsets_rc"))
    damaged["model_state_dict"][offset_name].add_(1)
    with raises(ValueError): trainer.load_score_input_checkpoint(damaged)


def test_new_freeze_records_population_rows_hash_and_exact_input(tmp_path):
    built, config, digests = trainer.build_training_model(source_metadata(), InputVariantSpec(window_sizes_px=(16.,)), "candidate_pair")
    identity = identity_fixture(tmp_path, built, config, digests)
    report, points = fake_validation(.8, .9, .7)
    winners = trainer.update_winners({}, root=tmp_path, epoch=5, report=report, points=points)
    (tmp_path / "epoch_005.pt").write_bytes(b"fixture-checkpoint")
    (tmp_path / "validation_005_rows.json").write_text("[]")
    trainer.publish_freezes(tmp_path, epoch=5, winners=winners, identity=identity)
    freeze = json.loads((tmp_path / "budget_freezes/005/freeze.json").read_text())
    assert freeze["schema_version"] == trainer.SCHEMA != live.SCHEMA
    assert freeze["resume_identity_sha256"] == trainer.canonical_digest(identity)
    assert freeze["score_input_metadata"] == built.metadata and freeze["eligible_epoch_range"] == [5, 5]
    assert not freeze["held_out_used_for_fit"] and not freeze["staged_training"]
    assert freeze["winners"]["recall95"]["primary_pair_threshold"] == .1
    assert freeze["winners"]["max_f1"]["validation_predictions_sha256"] == trainer._sha256(tmp_path / "validation_005_rows.json")


def test_default_input_actual_two_update_trajectory_matches_live(tmp_path):
    """Actual shared segment: all three losses, accumulation, clipping and Adam.

    This is a tiny synthetic CPU fixture:32 pairs per arm, not formal training.
    It uses the production Full24 model config and eight valid contour tokens.
    No live function/global is patched; default memory telemetry returns0 on CPU.
    """
    torch.set_num_threads(1)
    for architecture in ("original", "candidate_pair", "candidate_dual"):
        new_root, old_root = tmp_path / (architecture + "_new"), tmp_path / (architecture + "_live")
        new_root.mkdir(); old_root.mkdir()
        for root in (new_root, old_root):
            (root / "status.json").write_text(json.dumps(dict(global_exposure=0, optimizer_updates=0)))
        built, config, digests = trainer.build_training_model(source_metadata(), InputVariantSpec(), architecture)
        new_rng = trainer.capture_rng_state()
        new_optimizer = trainer.create_optimizer(built.model, architecture)
        new_args = SimpleNamespace(output=str(new_root), architecture=architecture, log_every=100)
        new_report = trainer.train_input_segment(built.model, tiny_loader(), new_optimizer, config,
            torch.device("cpu"), new_args, 1)
        assert state_digest(built.model) != digests["initial_weights_sha256"]
        new_state, new_optimizer_state = copy.deepcopy(built.model.state_dict()), copy.deepcopy(new_optimizer.state_dict())
        new_rng_end = torch.get_rng_state().clone()
        del built, new_optimizer
        reference, _, reference_loss, _ = build_random_model(source_metadata(), seed=trainer.SEED)
        if architecture != "original":
            reference = RachelCandidateScore(reference, CandidateScoreConfig(), architecture)
            reference.set_training_mode("joint")
        else:
            reference.requires_grad_(True)
        assert state_digest(reference) == digests["reference_score_initial_weights_sha256"]
        assert torch.equal(new_rng["torch"], torch.get_rng_state())
        reference.train()
        old_optimizer = torch.optim.AdamW((p for p in reference.parameters() if p.requires_grad),
            lr=live.learning_rate(1), weight_decay=1e-4)
        old_args = SimpleNamespace(output=str(old_root), architecture=architecture, log_every=100)
        old_report = live.train_score_segment(reference, tiny_loader(), old_optimizer, reference_loss,
            torch.device("cpu"), old_args, 1)
        assert new_report["samples"] == old_report["samples"] == 32
        assert new_report["optimizer_updates"] == old_report["optimizer_updates"] == 2
        assert new_report["loss_components"] == old_report["loss_components"]
        assert new_report["candidate_correctness_counts"] == old_report["candidate_correctness_counts"]
        assert tensor_tree_equal(new_state, reference.state_dict())
        assert tensor_tree_equal(new_optimizer_state, old_optimizer.state_dict())
        assert torch.equal(new_rng_end, torch.get_rng_state())
        if architecture != "original":
            assert new_report["candidate_correctness_counts"]["valid"] > 0
        del reference, old_optimizer, new_state, new_optimizer_state
        print("  default trajectory %s: exact state/optimizer/loss/RNG after2 updates" % architecture, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    tests = [test_cli_one_axis_cuda_guard_and_nested_plan, test_identity_binds_input_and_source_loss_not_pause,
        test_post_checkpoint_optimizer_rng_restore_and_legacy_rejection,
        test_new_freeze_records_population_rows_hash_and_exact_input,
        test_default_input_actual_two_update_trajectory_matches_live]
    for test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
        print(test.__name__ + ": PASS", flush=True)
