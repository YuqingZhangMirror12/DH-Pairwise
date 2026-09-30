"""CPU protocol/serialization tests; no data training or CUDA launch."""
from dataclasses import asdict
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.rachel_n512_formal_30k import train_score_design as trainer
from experiments.rachel_n512_formal_30k.train_joint_damage import build_random_model, state_digest
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.models.rachel_candidate_score import CandidateScoreConfig, RachelCandidateScore
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


@contextmanager
def expect_raises(exception, match=None):
    with unittest.TestCase().assertRaisesRegex(exception, match or ".*"):
        yield


def test_nested_budget_prefix_lr_and_complete_accumulation():
    plan = trainer.segment_plan()
    assert len(plan) == 200 and plan[-1]["global_stop"] == 1200000
    assert [x["number"] for x in plan] == list(range(1, 201))
    for index, row in enumerate(plan):
        assert row["global_start"] == index * 6000
        assert row["global_stop"] - row["global_start"] == 6000
        assert row["count"] % 16 == 0
        assert row["learning_rate"] == (1e-4 if row["epoch"] <= 3 else 2e-5)
    for budget in trainer.BUDGETS:
        prefix = [row for row in plan if row["epoch"] <= budget]
        assert prefix == plan[:4 * budget]
        assert prefix[-1]["epoch_complete"]
    with expect_raises(ValueError): trainer.learning_rate(0)
    with expect_raises(ValueError): trainer.learning_rate(51)


def test_identity_excludes_stop_but_binds_data_loss_seed_schedule(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"metadata")
    manifest = tmp_path / "train.json"
    manifest.write_text('{"split":"train"}')
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "val.jsonl").write_text('{"split":"val"}')
    class Training:
        manifest_path = manifest
        def __len__(self): return 24000
    args = SimpleNamespace(dataset=str(tmp_path), architecture="original", training_mode="joint", workers=4,
                           stop_after_epoch=5, output="one", log_every=100)
    kw = dict(source_path=source, initial_digest="a", architecture={"N": 512},
              loss_config=RachelN512LossConfig(), training=Training(), base_metadata={"model_kind": "full"})
    first = trainer.experiment_identity(args, **kw)
    args.stop_after_epoch, args.output, args.log_every = 20, "another", 9
    assert trainer.experiment_identity(args, **kw) == first
    assert first["seed"] == 260913 and first["max_epochs"] == 50
    assert first["min_selection_epoch"] == 5
    checkpoint = dict(resume_identity=first, completed_segments=4, global_exposure=24000,
                      optimizer_updates=1500, optimizer_state_dict={}, rng_state={})
    trainer.validate_resume(checkpoint, first)
    changed = dict(first, seed=1)
    with expect_raises(ValueError, match="seed"): trainer.validate_resume(checkpoint, changed)
    with expect_raises(ValueError, match="counts"):
        trainer.validate_resume(dict(checkpoint, global_exposure=23999), first)
    with expect_raises(ValueError, match="optimizer"):
        trainer.validate_resume({k: v for k, v in checkpoint.items() if k != "optimizer_state_dict"}, first)


def fake_validation(f1, ap, precision95):
    report = dict(selection_key=[f1, ap], decision_coverage=1., thresholds=dict(coarse=.2, local=.3, fused=.4))
    points = dict(selection_key=[precision95, ap], thresholds=dict(max_f1=.4, recall_95=.1))
    return report, points


def test_winners_only_from_five_keep_earliest_tie_and_separate_policies(tmp_path):
    report, points = fake_validation(.9, .95, .8)
    winners = trainer.update_winners({}, root=tmp_path, epoch=4, report=report, points=points)
    assert winners == {}
    winners = trainer.update_winners(winners, root=tmp_path, epoch=5, report=report, points=points)
    report, points = fake_validation(.8, .94, .9)
    winners = trainer.update_winners(winners, root=tmp_path, epoch=6, report=report, points=points)
    assert winners["max_f1"]["selected_epoch"] == 5
    assert winners["recall95"]["selected_epoch"] == 6
    tied = trainer.update_winners(winners, root=tmp_path, epoch=7, report=report, points=points)
    assert tied == winners
    report["decision_coverage"] = .99
    assert trainer.update_winners(winners, root=tmp_path, epoch=8, report=report, points=points) == winners
    assert winners["recall95"]["primary_pair_threshold"] == .1  # NOT recall_first/99%.


def test_budget_freezes_preserve_both_winner_sources(tmp_path):
    report, points = fake_validation(.9, .95, .8)
    winners = trainer.update_winners({}, root=tmp_path, epoch=5, report=report, points=points)
    (tmp_path / "epoch_005.pt").write_bytes(b"checkpoint")
    trainer.publish_freezes(tmp_path, epoch=5, winners=winners, identity={"seed": 260913})
    frozen = json.loads((tmp_path / "budget_freezes/005/freeze.json").read_text())
    assert frozen["status"] == "frozen_at_budget"
    assert frozen["eligible_epoch_range"] == [5, 5]
    assert set(frozen["winners"]) == {"max_f1", "recall95"}
    assert len(frozen["winners"]["max_f1"]["checkpoint_sha256"]) == 64
    assert not frozen["held_out_used_for_fit"]


def test_shared_random_initialization_and_explicit_candidate_roundtrip():
    torch.set_num_threads(1)
    source = dict(model_kind="full", model_options={},
        model_config=asdict(RachelN512Config(window_sizes_px=(7., 16., 32., 64.))),
        loss_config=asdict(RachelN512LossConfig()), model_state_dict={"ignored": torch.tensor(float("nan"))})
    original, _, _, shared = build_random_model(source, seed=trainer.SEED)
    states, full_digests = [], []
    for architecture in ("candidate_pair", "candidate_dual"):
        base, _, _, digest = build_random_model(source, seed=trainer.SEED)
        assert digest == shared
        assert all(torch.equal(v, original.state_dict()[k]) for k, v in base.state_dict().items())
        model = RachelCandidateScore(base, CandidateScoreConfig(), architecture)
        full_digests.append(state_digest(model))
        state = model.state_dict()
        payload = dict(**trainer.score_model_metadata(model, architecture), model_state_dict=state)
        assert payload["model_kind"].startswith("score_design_")
        restored = trainer.load_score_checkpoint(payload)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items())
        states.append(state)
    assert full_digests[0] == full_digests[1]
    assert all(torch.equal(v, states[1][k]) for k, v in states[0].items())
    payload = dict(**trainer.score_model_metadata(original), model_state_dict=original.state_dict())
    assert state_digest(trainer.load_score_checkpoint(payload)) == shared
    with expect_raises(ValueError): trainer.load_score_checkpoint(source)


def test_smoke_is_bounded_and_cuda_guard_before_any_output(tmp_path):
    args = trainer.parser().parse_args(["--checkpoint", "absent", "--dataset", "absent",
        "--train-materialized-manifest", "absent", "--output", str(tmp_path / "never"), "--smoke", "32",
        "--device", "cpu"])
    assert args.smoke == 32 and args.max_epochs == 50
    with expect_raises(RuntimeError, match="remote Linux CUDA"):
        trainer.run(args)
    assert not (tmp_path / "never").exists()


if __name__ == "__main__":
    # Also runnable on the read-only CPU inspection environment without pytest.
    tests = [test_nested_budget_prefix_lr_and_complete_accumulation,
             test_identity_excludes_stop_but_binds_data_loss_seed_schedule,
             test_winners_only_from_five_keep_earliest_tie_and_separate_policies,
             test_budget_freezes_preserve_both_winner_sources,
             test_shared_random_initialization_and_explicit_candidate_roundtrip,
             test_smoke_is_bounded_and_cuda_guard_before_any_output]
    for test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory)) if test.__code__.co_argcount else test()
        print(test.__name__ + ": PASS")
