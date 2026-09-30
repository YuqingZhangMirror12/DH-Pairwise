"""Tiny CPU wiring/serialization checks; no formal exposure claims or training."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rachel_n512_formal_30k import train_score_decoupled as trainer
from experiments.rachel_n512_formal_30k.test_score_design_stages import inputs, targets
from experiments.rachel_n512_formal_30k.test_train_score_staged import tensor_tree_equal
from experiments.rachel_n512_formal_30k.train_joint_damage import state_digest
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample
from staging.pairwise_v0_2.pairwise_data.rachel_staged_damage_dataset import clean_report
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig


@contextmanager
def raises(exception, match=".*"):
    with unittest.TestCase().assertRaisesRegex(exception, match):
        yield


def model(kind="matrix_cnn"):
    torch.manual_seed(28)
    config = RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
        feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
        sinkhorn_iterations=40, activation_checkpointing=False)
    return build_decoupled_score_model(config, kind)


def identity(net):
    # Synthetic receipt fixture only; no file from this fixture is a real run.
    return dict(schema_version=trainer.SCHEMA, seed=trainer.SEED, head_seed=trainer.HEAD_SEED,
        classifier_phase_seed=trainer.CLASSIFIER_SEED, head_kind=net.head_kind,
        matrix_head_revision=net.metadata()["matrix_head_revision"],
        matrix_threshold=net.matrix_threshold, base_model_config=asdict(net.config),
        sampling="original512", contour_cap=16, matcher_checkpoint_sha256=None,
        shared_base_initial_weights_sha256=state_digest(net.base_model),
        populations={"train": {"unit_fixture": True}, "val": {"unit_fixture": True}},
        matcher_loss_config=asdict(trainer.matcher_loss_config(RachelN512LossConfig())),
        matcher_epochs=12, train_count=24000, segment_pairs=6000, microbatch=1, effective_batch=16,
        optimizer="AdamW", weight_decay=1e-4, grad_clip_norm=5., precision="fp32", workers=0,
        lr_by_epoch=[trainer.learning_rate(e) for e in range(1, 21)])


def args(root, micro=1):
    return SimpleNamespace(output=str(root), microbatch=micro, effective_batch=16, log_every=1000)


def tiny_loader(count=16, micro=1, balanced=False):
    a, b, pa, pb, va, vb = inputs()
    data = []
    for i in range(count):
        positive = not balanced or i < count // 2
        sample = RachelPairSample(pair_id="decoupled-cpu-fixture-%d" % i,
            fragment_a_token="a", fragment_b_token="b", mask_a=a[0].numpy(), mask_b=b[0].numpy(),
            coarse_mask_a=np.zeros((1, 32, 32), np.float32), coarse_mask_b=np.zeros((1, 32, 32), np.float32),
            points_rc_a=pa[0].numpy(), points_rc_b=pb[0].numpy(),
            contour_valid_a=va[0].numpy(), contour_valid_b=vb[0].numpy(),
            target_a=np.arange(8, dtype=np.int64) if positive else np.full(8,-1,np.int64),
            target_b=np.arange(8, dtype=np.int64) if positive else np.full(8,-1,np.int64),
            label=np.float32(positive), translation_a_to_b_rc=np.zeros(2, np.float32),
            translation_a_to_b_xy_cartesian=np.zeros(2, np.float32), translation_valid=np.bool_(positive))
        data.append((sample, clean_report(sample, epoch=1)))
    return trainer.make_weathering_loader(data, list(range(count)), batch_size=micro, num_workers=0,
                                          seed=trainer.SEED, contour_cap=16)


def test_schedule_budget_phase_lr_and_cli():
    rows = trainer.segment_plan()
    assert len(rows) == 80
    assert all(r["phase"] == "matcher" for r in rows[:48])
    assert all(r["phase"] == "classifier" for r in rows[48:])
    assert rows[47]["global_stop"] == 288000 and rows[-1]["global_stop"] == 480000
    assert [trainer.learning_rate(e) for e in (1, 3, 4, 12, 13, 15, 16, 20)] == [1e-4,1e-4,2e-5,2e-5,1e-4,1e-4,2e-5,2e-5]
    command = "--checkpoint c --dataset d --train-materialized-manifest m --head-kind matrix_cnn --output o".split()
    parsed = trainer.parser().parse_args(command)
    trainer.validate_arguments(parsed)
    assert parsed.microbatch == 1 and parsed.effective_batch == 16 and parsed.smoke_phase == "both"
    parsed.effective_batch = 32
    with raises(ValueError, match="effective batch"):
        trainer.validate_arguments(parsed)


def test_matcher_has_no_pair_bce_gradient_and_keeps_weathering_pose_rule():
    net = model()
    net.set_phase("matcher").train()
    out = net(*inputs())
    loss, components = trainer.phase_loss(out, targets(), torch.tensor([True]), RachelN512LossConfig(), "matcher")
    expected = .5 * components["assignment_nll"] + .5 * components["translation_smooth_l1"] + .05 * components["sinkhorn_residual"]
    assert torch.equal(loss, expected)
    assert torch.autograd.grad(loss, out.fused_logit, allow_unused=True, retain_graph=True)[0].item() == 0.
    changed, altered = trainer.phase_loss(out, targets(), torch.tensor([False]), RachelN512LossConfig(), "matcher")
    assert altered["translation_smooth_l1"].item() == 0.
    assert torch.equal(changed, .5 * altered["assignment_nll"] + .05 * altered["sinkhorn_residual"])


def check_classifier_is_exactly_one_bce(kind):
    net = model(kind).set_phase("classifier").train()
    out = net(*inputs())
    labels = targets()[0]
    total, components = trainer.phase_loss(out, (labels, object(), object(), object(), object()),
                                          object(), RachelN512LossConfig(), "classifier")
    expected = torch.nn.functional.binary_cross_entropy_with_logits(out.fused_logit, labels)
    assert torch.equal(total, expected)
    assert all(components[k].item() == 0. for k in trainer.LOSS_NAMES if k != "fused_pair_bce")
    total.backward()
    assert all(p.grad is None for p in net.base_model.parameters())
    assert any(p.grad is not None for p in net.score_head.parameters())


def test_classifier_is_exactly_one_bce_and_never_reads_correspondence_or_pose():
    for kind in ("matrix_cnn", "cross_attention"):
        check_classifier_is_exactly_one_bce(kind)


def test_micro1_accumulates16_and_strictly_freezes_base_and_bn(tmp_path):
    torch.set_num_threads(1)
    net = model()
    ident = identity(net)
    optimizer = trainer.create_optimizer(net)
    net.set_phase("matcher").train()
    before_head = state_digest(net.score_head)
    report = trainer.train_segment(net, tiny_loader(), optimizer, RachelN512LossConfig(),
                                   torch.device("cpu"), args(tmp_path), 1)
    assert report["samples"] == 16 and report["optimizer_updates"] == 1
    assert state_digest(net.score_head) == before_head
    # The following counts are protocol fixtures, not claims of12 CPU epochs.
    receipt = trainer.matcher_receipt(net, ident)
    trainer.configure_phase(net, 13, ident, receipt, entering_classifier=True)
    before_base = state_digest(net.base_model)
    report = trainer.train_segment(net, tiny_loader(), optimizer, RachelN512LossConfig(),
                                   torch.device("cpu"), args(tmp_path), 13)
    assert report["samples"] == 16 and report["optimizer_updates"] == 1
    assert report["loss_components"]["total"] == report["loss_components"]["fused_pair_bce"]
    assert state_digest(net.base_model) == before_base
    assert state_digest(net.score_head) != before_head
    trainer.verify_receipt(net, receipt, ident)


def test_checkpoint_resume_continues_exact_optimizer_rng_and_frozen_buffers(tmp_path):
    torch.set_num_threads(1)
    net = model("cross_attention")
    ident = identity(net)
    receipt = trainer.matcher_receipt(net, ident)
    optimizer = trainer.create_optimizer(net)
    trainer.configure_phase(net, 13, ident, receipt, entering_classifier=True)
    trainer.train_segment(net, tiny_loader(), optimizer, RachelN512LossConfig(), torch.device("cpu"), args(tmp_path), 13)
    checkpoint = trainer.checkpoint_payload(net, optimizer, identity=ident, loss_config=RachelN512LossConfig(),
        completed=49, receipt=receipt, winners={}, role="synthetic_unit_fixture")
    checkpoint = deepcopy(checkpoint)
    trainer.train_segment(net, tiny_loader(), optimizer, RachelN512LossConfig(), torch.device("cpu"), args(tmp_path), 13)
    expected = state_digest(net)
    resumed = model("cross_attention")
    resumed_optimizer = trainer.create_optimizer(resumed)
    completed, restored, winners = trainer.restore_training_state(resumed, resumed_optimizer, checkpoint, ident)
    assert completed == 49 and restored == receipt and not winners
    trainer.configure_phase(resumed, 13, ident, restored, entering_classifier=False)
    trainer.train_segment(resumed, tiny_loader(), resumed_optimizer, RachelN512LossConfig(), torch.device("cpu"), args(tmp_path), 13)
    assert state_digest(resumed) == expected
    assert tensor_tree_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    loaded = trainer.load_decoupled_checkpoint(checkpoint)
    assert state_digest(loaded) == state_digest(trainer.load_decoupled_checkpoint(checkpoint))


def test_only_exact_matching_new_M12_can_be_reused_and_head_is_not_loaded():
    matrix = model()
    ident = identity(matrix)
    receipt = trainer.matcher_receipt(matrix, ident)
    payload = trainer.checkpoint_payload(matrix, trainer.create_optimizer(matrix), identity=ident,
        loss_config=RachelN512LossConfig(), completed=48, receipt=receipt, winners={}, role="synthetic_unit_fixture")
    cross = model("cross_attention")
    other_identity = identity(cross)
    initial_head = state_digest(cross.score_head)
    trainer.import_matcher(cross, payload, other_identity)
    assert state_digest(cross.base_model) == state_digest(matrix.base_model)
    assert state_digest(cross.score_head) == initial_head
    wrong = deepcopy(payload)
    wrong["decoupled_training_schema"] = "old-S2-joint"
    with raises(ValueError, match="not a new decoupled"):
        trainer.import_matcher(cross, wrong, other_identity)
    wrong_identity = deepcopy(other_identity)
    wrong_identity["populations"]["train"]["different"] = True
    with raises(ValueError, match="different sampling/data"):
        trainer.import_matcher(cross, payload, wrong_identity)
    corrupted = deepcopy(payload)
    corrupted["global_exposure"] += 16
    with raises(ValueError, match="exposure"):
        trainer.load_decoupled_checkpoint(corrupted)


def test_selection_C13_20_only_fixed20_and_no_threshold_direction_assumption(tmp_path):
    report = dict(decision_coverage=1., thresholds={"coarse":.5,"local":.4,"fused":.4}, selection_key=[.8,.9])
    points = dict(thresholds={"max_f1":.4,"recall_95":.9}, selection_key=[.95,.9])
    assert trainer.update_winners({}, root=tmp_path, epoch=12, report=report, points=points) == {}
    winners = trainer.update_winners({}, root=tmp_path, epoch=13, report=report, points=points)
    assert winners["recall95"]["primary_pair_threshold"] > winners["max_f1"]["primary_pair_threshold"]
    tied = trainer.update_winners(winners, root=tmp_path, epoch=14, report=report, points=points)
    assert tied == winners
    final = trainer.update_winners(winners, root=tmp_path, epoch=20, report=report, points=points)
    assert final["fixed_epoch"]["selected_epoch"] == 20 and final["max_f1"]["selected_epoch"] == 13
    for epoch in (13,20):
        (tmp_path / ("epoch_%03d.pt" % epoch)).write_bytes(b"fixture only")
    trainer.publish_freezes(tmp_path,20,final,{"fixture": True})
    import json
    freeze = json.loads((tmp_path / "classifier_freezes/freeze.json").read_text())
    assert freeze["status"] == "complete" and freeze["eligible_epoch_range"] == [13,20]
    assert set(freeze["selections"]) == {"fixed_epoch","max_f1","recall95"}
    assert freeze["held_out_used_for_fit"] is False


def test_cap2048_and_both_head_kinds_share_exact_random_base_without_loading_source_weights():
    source = dict(model_kind="full", model_options={},
        model_config=asdict(RachelN512Config(window_sizes_px=(7.,16.,32.,64.))),
        loss_config=asdict(RachelN512LossConfig()), model_state_dict={"never_read": torch.tensor(float("nan"))})
    a, _, _, da = trainer.build_random_decoupled(source,"matrix_cnn",.006,512)
    rng_a = torch.get_rng_state().clone()
    b, _, _, db = trainer.build_random_decoupled(source,"cross_attention",.006,2048)
    assert a.config.contour_cap == 512 and b.config.contour_cap == 2048
    assert da["shared_base_initial_weights_sha256"] == db["shared_base_initial_weights_sha256"]
    assert torch.equal(torch.get_rng_state(), rng_a)
    assert all(torch.equal(v, b.base_model.state_dict()[k]) for k,v in a.base_model.state_dict().items())


def test_step_and_paired_reader_hooks_same_manifest_and_operational_stop_outside_identity(tmp_path):
    class FakeStep:
        def __init__(self, path, sampling):
            self.split = "train" if str(path).endswith("train.json") else "val"
            self.contour_cap = 2048 if sampling == "step3" else 512
            self.identity = "fixture-" + sampling
            self.protocol = {"same_source": True, "sampling": sampling}
        def __len__(self):
            return 24000 if self.split == "train" else 3000
        def weathered(self, i):
            return i, {"fixture": True}
    for name in ("train.json", "val.json", "source.pt"):
        (tmp_path / name).write_bytes(b"isolated fixture")
    cli = SimpleNamespace(sampling="step3", density_train_manifest=str(tmp_path / "train.json"),
        clean_val_manifest=str(tmp_path / "val.json"))
    module = "staging.pairwise_v0_2.pairwise_data.rachel_step_dataset.StepSourceDataset"
    with patch(module, FakeStep):
        train, val, cap, records = trainer.make_populations(cli)
        assert cap == 2048 and train[2] == (2, {"fixture": True})
        cli.sampling = "paired512"
        control, control_val, control_cap, control_records = trainer.make_populations(cli)
        assert control_cap == 512 and records["train"]["manifest_sha256"] == control_records["train"]["manifest_sha256"]
        assert records["val"]["manifest"] == control_records["val"]["manifest"]
    net = model()
    operational = SimpleNamespace(head_kind="matrix_cnn", matrix_threshold=.006, sampling="original512",
        matcher_checkpoint=None, microbatch=1, effective_batch=16, workers=0,
        stop_after_epoch=12, output="one", resume=False)
    before = trainer.experiment_identity(operational,tmp_path / "source.pt",net,RachelN512LossConfig(),records,{})
    operational.stop_after_epoch=20;operational.output="two";operational.resume=True
    after = trainer.experiment_identity(operational,tmp_path / "source.pt",net,RachelN512LossConfig(),records,{})
    assert before == after


def test_input_probe_is_fixed_train64_no_grad_rng_preserving_and_cached(tmp_path):
    net = model()
    ident = identity(net)
    receipt = trainer.matcher_receipt(net, ident)
    population = SimpleNamespace(split="train",rows=[{"label":int(i<32)} for i in range(64)])
    assert trainer.classifier_probe_indices(population) == list(range(64))
    loader = tiny_loader(64, balanced=True)
    forward_calls = []
    def fake_forward(*unused):
        assert not torch.is_grad_enabled() and not net.base_model.training
        forward_calls.append(True)
        return SimpleNamespace(assignment=torch.full((1,16,16),.1),
            token_features_a=torch.ones(1,16,16),token_features_b=torch.ones(1,16,16))
    operational = args(tmp_path);operational.workers=0
    rng = torch.get_rng_state().clone()
    with patch.object(trainer,"make_weathering_loader",return_value=loader), patch.object(net.base_model,"forward",side_effect=fake_forward):
        report = trainer.ensure_classifier_input_probe(net,population,ident,receipt,torch.device("cpu"),operational,tmp_path)
        cached = trainer.ensure_classifier_input_probe(net,population,ident,receipt,torch.device("cpu"),operational,tmp_path)
    assert report == cached and len(forward_calls) == 64
    assert report["status"] == "complete" and report["groups"]["positive"]["nonempty_binary_pair_count"] == 32
    assert report["groups"]["negative"]["raw_positive_cell_count"] == 32*64
    assert net.training and torch.equal(torch.get_rng_state(),rng)
    trainer.verify_receipt(net,receipt,ident)


def test_input_probe_all_empty_positive_matrices_explicitly_fails_without_changing_tau(tmp_path):
    net = model()
    ident = identity(net);receipt=trainer.matcher_receipt(net,ident)
    population=SimpleNamespace(split="train",rows=[{"label":int(i<32)} for i in range(64)])
    output=SimpleNamespace(assignment=torch.zeros(1,16,16),
        token_features_a=torch.ones(1,16,16),token_features_b=torch.ones(1,16,16))
    loader=tiny_loader(64,balanced=True)
    operational=args(tmp_path);operational.workers=0
    with patch.object(trainer,"make_weathering_loader",return_value=loader), patch.object(net.base_model,"forward",return_value=output):
        with raises(trainer.FailedClassifierInputProbe,match="all_32_positive"):
            trainer.ensure_classifier_input_probe(net,population,ident,receipt,torch.device("cpu"),operational,tmp_path)
    import json
    record=json.loads((tmp_path/"classifier_input_probe.json").read_text())
    assert record["status"]=="failed_input_probe" and record["matrix_threshold"]==.006
    assert record["groups"]["positive"]["empty_binary_pair_count"]==32 and net.matrix_threshold==.006
    with raises(trainer.FailedClassifierInputProbe,match="registered TRAIN input probe failed"):
        trainer.ensure_classifier_input_probe(net,population,ident,receipt,torch.device("cpu"),operational,tmp_path)
