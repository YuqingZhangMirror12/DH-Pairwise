"""S3 entrypoint CPU protocol/serialization/miniature-step tests, not formal runs."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_staged as trainer
from experiments.rachel_n512_formal_30k import score_design_stages as stages
from experiments.rachel_n512_formal_30k.test_score_design_stages import model as small_model, inputs
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from experiments.rachel_n512_formal_30k.train_score_design import load_score_checkpoint
from staging.pairwise_v0_2.models.rachel_model_factory import model_metadata
from staging.pairwise_v0_2.models.rachel_candidate_score import RachelCandidateScore, CandidateScoreConfig
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import clean_report
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


@contextmanager
def raises(exception, message=".*"):
    with unittest.TestCase().assertRaisesRegex(exception, message):
        yield


def source_metadata():
    return dict(model_kind="full", model_options={},
        model_config=asdict(RachelN512Config(window_sizes_px=(7., 16., 32., 64.))),
        loss_config=asdict(RachelN512LossConfig()), model_state_dict={"never_load": torch.tensor(float("nan"))})


def miniature_identity(net):
    # Synthetic unit-test receipt identity, deliberately not a formal Full24 run.
    return dict(schema_version=trainer.SCHEMA, score_design=net.architecture,
        base_model_metadata=model_metadata(net.base_model), candidate_config=asdict(net.candidate_config),
        train_manifest_sha256="a" * 64, loss_config=asdict(RachelN512LossConfig()),
        initial_weights_sha256=state_digest(net),
        shared_base_initial_weights_sha256=state_digest(net.base_model),
        initial_matcher_state_sha256=stages.matcher_state_digest(net), seed=trainer.SEED,
        schedule="staged", max_epochs=20)


def tiny_loader(count=16):
    a, b, pa, pb, va, vb = inputs()
    data = []
    for index in range(count):
        sample = RachelPairSample(pair_id="cpu-fixture-%d" % index,
            fragment_a_token="a%d" % index, fragment_b_token="b%d" % index,
            mask_a=a[0].numpy(), mask_b=b[0].numpy(),
            coarse_mask_a=np.zeros((1, 32, 32), np.float32), coarse_mask_b=np.zeros((1, 32, 32), np.float32),
            points_rc_a=pa[0].numpy(), points_rc_b=pb[0].numpy(),
            contour_valid_a=va[0].numpy(), contour_valid_b=vb[0].numpy(),
            target_a=np.arange(8, dtype=np.int64), target_b=np.arange(8, dtype=np.int64),
            label=np.float32(1.), translation_a_to_b_rc=np.zeros(2, np.float32),
            translation_a_to_b_xy_cartesian=np.zeros(2, np.float32), translation_valid=np.bool_(True))
        data.append((sample, clean_report(sample, epoch=1)))
    return trainer.make_weathering_loader(data, list(range(count)), batch_size=4, num_workers=0,
                                          seed=trainer.SEED, contour_cap=16)


def tensor_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(tensor_tree_equal(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(tensor_tree_equal(x, y) for x, y in zip(first, second))
    return first == second


def test_plan_20_only_and_stage_boundary():
    plan = trainer.segment_plan("candidate_dual")
    assert len(plan) == 80 and plan[-1]["global_stop"] == 480000
    assert all(row["count"] % 16 == 0 for row in plan)
    assert all(row["phase"]["name"] == "M" for row in plan[:48])
    assert all(row["phase"]["name"] == "C" for row in plan[48:])
    assert plan[47]["epoch_complete"] and plan[47]["epoch"] == 12
    assert plan[48]["epoch"] == 13
    assert all(row["learning_rate"] == (1e-4 if row["epoch"] <= 3 else 2e-5) for row in plan)


def test_random_init_matches_existing_joint_and_source_weights_ignored():
    source = source_metadata()
    staged, _, _, digests = trainer.build_random_candidate(source, "candidate_dual")
    base, _, _, shared = build_random_model(source, seed=trainer.SEED)
    joint = RachelCandidateScore(base, CandidateScoreConfig(), "candidate_dual")
    assert digests["shared_base_initial_weights_sha256"] == shared
    assert digests["initial_weights_sha256"] == state_digest(joint)
    assert all(torch.equal(t, joint.state_dict()[n]) for n, t in staged.state_dict().items())
    candidate_source = dict(source, score_design=joint.metadata())
    repeated, _, _, repeated_digests = trainer.build_random_candidate(candidate_source, "candidate_pair")
    assert state_digest(repeated) == state_digest(staged) and repeated_digests == digests


def test_identity_pause_excluded_and_resume_guarded(tmp_path):
    source = tmp_path / "metadata.pt"
    source.write_bytes(b"unit-metadata")
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs/val.jsonl").write_text("unit-VAL-manifest")
    manifest = tmp_path / "train.json"
    manifest.write_text("unit-TRAIN-manifest")
    net = small_model()
    digests = dict(initial_weights_sha256=state_digest(net),
        shared_base_initial_weights_sha256=state_digest(net.base_model),
        initial_matcher_state_sha256=stages.matcher_state_digest(net))
    args = SimpleNamespace(dataset=str(tmp_path), architecture=net.architecture, workers=4,
                           stop_after_epoch=12, output="one", log_every=100)
    kwargs = (source, net, RachelN512LossConfig(), SimpleNamespace(manifest_path=manifest), digests)
    identity = trainer.experiment_identity(args, *kwargs)
    args.stop_after_epoch, args.output, args.log_every = 20, "two", 5
    assert trainer.experiment_identity(args, *kwargs) == identity
    assert identity["max_epochs"] == 20 and identity["phase_protocol"]["total_pair_exposures"] == 480000
    stages.configure_phase(net, stages.phase(1, architecture=net.architecture))
    optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(net))
    checkpoint = trainer.payload(net, optimizer, identity=identity, loss_config=RachelN512LossConfig(),
        data_record={}, completed=0, receipt=None, role="unit_initial")
    trainer.validate_checkpoint_progress(checkpoint, identity)
    with raises(ValueError, "identity"):
        trainer.validate_checkpoint_progress(checkpoint, dict(identity, seed=1))
    with raises(ValueError, "counts"):
        trainer.validate_checkpoint_progress(dict(checkpoint, global_exposure=1), identity)
    with raises(ValueError, "S3 checkpoint"):
        trainer.validate_checkpoint_progress(dict(checkpoint, score_design_schema="rachel-score-design-training/1"), identity)
    with raises(ValueError, "optimizer"):
        trainer.restore_training_state(net, optimizer, {k: v for k, v in checkpoint.items() if k != "optimizer_state_dict"}, identity)


def test_actual_phase_loop_m12_receipt_resume_c_and_dedicated_loader(tmp_path):
    # The model and16-sample steps are tiny fixtures. Saved12/13 counters only
    # exercise checkpoint-boundary code; they do not represent formal training.
    net = small_model()
    identity = miniature_identity(net)
    args = SimpleNamespace(output=str(tmp_path), log_every=1)
    config = RachelN512LossConfig()
    m = stages.phase(12, architecture=net.architecture)
    stages.configure_phase(net, m)
    optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(net), lr=2e-5, weight_decay=1e-4)
    old_score = {k: v.clone() for k, v in net.score_head.state_dict().items()}
    report = trainer.train_segment(net, tiny_loader(), optimizer, config, torch.device("cpu"), args, m)
    assert report["samples"] == 16 and report["optimizer_updates"] == 1
    assert report["supervision_counts"]["r_valid"] == 0
    assert all(torch.equal(v, net.score_head.state_dict()[k]) for k, v in old_score.items())
    receipt = stages.matcher_pretraining_receipt(net,
        initial_matcher_sha256=identity["initial_matcher_state_sha256"], completed_epochs=12,
        pair_exposures=288000, optimizer_updates=18000, train_manifest_sha256=identity["train_manifest_sha256"],
        run_identity_sha256=trainer.canonical_digest(identity))
    checkpoint = trainer.payload(net, optimizer, identity=identity, loss_config=config,
        data_record={}, completed=48, receipt=receipt, role="unit_M12")
    path = tmp_path / "epoch_012.pt"
    trainer.runner._atomic_torch_save(path, checkpoint)
    expected_rng = (torch.rand(3), np.random.rand(3), random.random())
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored = small_model()
    resumed_optimizer = torch.optim.AdamW(stages.optimizer_parameter_groups(restored), lr=9e-4)
    completed, restored_receipt = trainer.restore_training_state(restored, resumed_optimizer, saved, identity)
    assert completed == 48 and restored_receipt == receipt
    assert torch.equal(torch.rand(3), expected_rng[0])
    assert np.array_equal(np.random.rand(3), expected_rng[1]) and random.random() == expected_rng[2]
    assert tensor_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    assert state_digest(restored) == state_digest(net)
    # M12 restart switches only on entry into the next C13 segment.
    c = stages.phase(13, architecture=net.architecture)
    stages.configure_phase(restored, c, receipt=receipt, **trainer._receipt_context(identity))
    frozen = stages.matcher_state_digest(restored)
    result = trainer.train_segment(restored, tiny_loader(), resumed_optimizer, config,
        torch.device("cpu"), args, c, global_exposure=288000, optimizer_updates=18000)
    assert result["samples"] == 16 and result["optimizer_updates"] == 1
    assert stages.matcher_state_digest(restored) == frozen
    assert result["supervision_counts"]["assignment_supervised_matches"] == 0
    after = trainer.payload(restored, resumed_optimizer, identity=identity, loss_config=config,
        data_record={}, completed=49, receipt=receipt, role="unit_C13")
    inferred = trainer.load_staged_checkpoint(after)
    assert state_digest(inferred) == state_digest(restored)
    assert not inferred.base_model.training and not inferred.training
    assert not any(p.requires_grad for p in inferred.base_model.coarse.parameters())
    assert "score_design_schema" not in after
    with raises(ValueError, "score-design checkpoint"):
        load_score_checkpoint(after)
    with raises(ValueError, "receipt"):
        trainer.load_staged_checkpoint(dict(after, pretraining_receipt=None))
    altered = dict(after, s3_model_metadata=dict(after["s3_model_metadata"], model_config=dict(after["model_config"], coarse_size=64)))
    with raises(ValueError, "configuration"):
        trainer.load_staged_checkpoint(altered)


def test_entry_cuda_guard_and_cli_pause(tmp_path):
    args = trainer.parser().parse_args(["--checkpoint", "absent", "--dataset", "absent",
        "--train-materialized-manifest", "absent", "--architecture", "candidate_pair",
        "--output", str(tmp_path / "must_not_exist"), "--device", "cpu", "--stop-after-epoch", "12"])
    assert args.stop_after_epoch == 12
    with raises(RuntimeError, "remote Linux CUDA"):
        trainer.run(args)
    assert not (tmp_path / "must_not_exist").exists()


if __name__ == "__main__":
    torch.set_num_threads(1)
    for test in (test_plan_20_only_and_stage_boundary,
                 test_random_init_matches_existing_joint_and_source_weights_ignored,
                 test_identity_pause_excluded_and_resume_guarded,
                 test_actual_phase_loop_m12_receipt_resume_c_and_dedicated_loader,
                 test_entry_cuda_guard_and_cli_pause):
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory)) if test.__code__.co_argcount else test()
        print(test.__name__ + ": PASS")
