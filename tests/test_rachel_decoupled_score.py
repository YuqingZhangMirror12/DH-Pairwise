"""Bounded CPU tests for the independent M->C model, not a training benchmark."""
from dataclasses import asdict, replace
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from staging.pairwise_v0_2.models.rachel_decoupled_score import (
    SCHEMA, antidiagonal_opening, ThresholdedMatrixCNN, CrossAttentionPairHead,
    DecoupledScoreModel, build_decoupled_score_model, load_decoupled_score_checkpoint,
    DEFAULT_MATRIX_HEAD_REVISION, matrix_revision_from_metadata,
)
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config, RachelN512Pairwise
from staging.pairwise_v0_2.training.rachel_n512_loss import RachelN512LossConfig, compute_rachel_n512_loss


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return RachelN512Config(canvas_size=80, coarse_size=32, contour_cap=16, patch_size=8,
        feature_dim=16, num_heads=4, landmark_count=4, context_layers=1, evidence_dim=8,
        sinkhorn_iterations=40, activation_checkpointing=False)


def inputs():
    a, b = torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 80, 80)
    a[:, :, 15:57, 12:43] = 1; b[:, :, 10:61, 28:63] = 1
    pa = torch.tensor([[[15., 12.], [15., 27.], [15., 42.], [35., 42.],
                        [56., 42.], [56., 27.], [56., 12.], [35., 12.]]])
    pb = torch.tensor([[[10., 28.], [10., 45.], [10., 62.], [35., 62.],
                        [60., 62.], [60., 45.], [60., 28.], [35., 28.]]])
    return a, b, pa, pb, torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8, dtype=torch.bool)


def state(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def unchanged(before, module):
    for k, v in module.state_dict().items():
        assert torch.equal(before[k], v), k


def test_soft_morphology_matches_official_cv2_and_strict_threshold():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(5)
    q = rng.random((11, 13), dtype=np.float32)
    kernel = np.rot90(np.eye(3, dtype=np.uint8)).copy(); kernel[1, 1] = 0
    expected = cv2.erode(q, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel[1, 1] = 1
    expected = cv2.dilate(expected, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    np.testing.assert_array_equal(antidiagonal_opening(torch.tensor(q)).numpy(), expected)
    x = torch.full((8, 8), .006)
    assert not ThresholdedMatrixCNN(.006).binary_matrix(x, torch.ones(8, dtype=torch.bool),
                                                       torch.ones(8, dtype=torch.bool)).any()
    torch.testing.assert_close(antidiagonal_opening(torch.tensor(q).T),
                               antidiagonal_opening(torch.tensor(q)).T, rtol=0, atol=0)


def test_matrix_cnn_swap_padding_and_threshold_detaches_q():
    torch.manual_seed(8)
    head = ThresholdedMatrixCNN(.006).eval()
    q = torch.rand(1, 9, 12, requires_grad=True)
    va, vb = torch.ones(1, 9, dtype=torch.bool), torch.ones(1, 12, dtype=torch.bool)
    result = head(q, va, vb)
    torch.testing.assert_close(result, head(q.transpose(1, 2), vb, va), rtol=0, atol=0)
    padded = F.pad(q.detach(), (0, 4, 0, 7), value=float("nan"))
    torch.testing.assert_close(result, head(padded, F.pad(va, (0, 7)), F.pad(vb, (0, 4))), rtol=0, atol=0)
    F.binary_cross_entropy_with_logits(result, torch.ones_like(result)).backward()
    assert q.grad is None
    assert head.fc.weight.grad is not None and head.fc.weight.grad.abs().sum() > 0


def test_cross_attention_swap_padding_and_classifier_gradients():
    torch.manual_seed(9)
    head = CrossAttentionPairHead(16, 4).eval()
    a, b = torch.randn(2, 9, 16), torch.randn(2, 11, 16)
    va, vb = torch.ones(2, 9, dtype=torch.bool), torch.ones(2, 11, dtype=torch.bool)
    score = head(a, b, va, vb)
    torch.testing.assert_close(score, head(b, a, vb, va), rtol=0, atol=0)
    aa, bb = F.pad(a, (0, 0, 0, 3), value=float("nan")), F.pad(b, (0, 0, 0, 2), value=float("nan"))
    torch.testing.assert_close(score, head(aa, bb, F.pad(va, (0, 3)), F.pad(vb, (0, 2))), rtol=0, atol=0)
    F.binary_cross_entropy_with_logits(score, torch.tensor([1., 0.])).backward()
    for parameter in (head.cross_attention.in_proj_weight, head.pool_gate[0].weight,
                      head.classifier[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["matrix_cnn", "cross_attention"])
def test_classifier_freezes_entire_base_including_bn_and_roundtrips(kind):
    torch.manual_seed(10)
    model = DecoupledScoreModel(RachelN512Pairwise(config()), kind).set_phase("classifier").train()
    assert not model.base_model.training and model.score_head.training
    assert not any(p.requires_grad for p in model.base_model.parameters())
    before = state(model.base_model)
    optimizer = torch.optim.AdamW(model.score_head.parameters(), lr=1e-3)
    out = model(*inputs())
    assert torch.equal(out.fused_logit, out.local_logit)
    F.binary_cross_entropy_with_logits(out.fused_logit, torch.ones(1)).backward()
    optimizer.step()
    unchanged(before, model.base_model)
    assert all(p.grad is None for p in model.base_model.parameters())
    model.eval()
    expected = model(*inputs())
    frozen = model.base_model(*inputs())
    for name in ("assignment", "affinity", "translation_hat_rc", "coarse_logit"):
        torch.testing.assert_close(getattr(expected, name), getattr(frozen, name), rtol=0, atol=0)
    payload = dict(decoupled_score_schema=SCHEMA, decoupled_score=model.metadata(),
                   model_state_dict=model.state_dict())
    restored = load_decoupled_score_checkpoint(payload).eval()
    torch.testing.assert_close(restored(*inputs()).fused_logit, expected.fused_logit, rtol=0, atol=0)
    bad = copy.deepcopy(payload); bad["decoupled_score"]["final_score"] = "coarse fusion"
    with pytest.raises(ValueError, match="registered semantics"):
        load_decoupled_score_checkpoint(bad)


def test_matcher_phase_only_trains_matcher_with_original_matching_losses():
    torch.manual_seed(11)
    model = build_decoupled_score_model(asdict(config())).train()
    fixed = [model.base_model.coarse, model.base_model.local_head, model.base_model.fusion, model.score_head]
    before = [state(x) for x in fixed]
    assert all(not x.training for x in fixed)
    assert model.base_model.patch_encoder.training
    output = model(*inputs())
    targets = torch.arange(8)[None]
    cfg = replace(RachelN512LossConfig(), fused_pair_weight=0., coarse_pair_weight=0., local_pair_weight=0.)
    loss = compute_rachel_n512_loss(output, torch.ones(1), targets, targets,
                                   torch.tensor([[2., 3.]]), torch.ones(1, dtype=torch.bool), cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss.total.backward()
    assert model.base_model.primal.weight.grad is not None
    assert model.base_model.primal.weight.grad.abs().sum() > 0
    optimizer.step()
    for saved, module in zip(before, fixed):
        unchanged(saved, module)
        assert all(p.grad is None for p in module.parameters())
    model.set_phase("classifier").train()
    assert all(p.grad is None for p in model.parameters())
    assert not model.base_model.training and model.score_head.training


def test_invalid_protocols_and_empty_head_evidence_are_explicit():
    with pytest.raises(ValueError):
        ThresholdedMatrixCNN(float("nan"))
    with pytest.raises(ValueError):
        load_decoupled_score_checkpoint({"model_kind": "full"})
    model = build_decoupled_score_model(config())
    with pytest.raises(ValueError):
        model.set_phase("joint")
    q = torch.zeros(1, 8, 8); valid = torch.zeros(1, 8, dtype=torch.bool)
    assert torch.isfinite(model.score_head(q, valid, valid)).all()
    with pytest.raises(ValueError, match="dimensions differ"):
        model.score_head(q, valid.expand(2, -1), valid)
    with pytest.raises(ValueError, match="dimensions differ"):
        CrossAttentionPairHead(16, 4)(torch.zeros(1, 8, 16), torch.zeros(2, 8, 16), valid, valid)


def test_nonprefix_padding_is_cropped_in_original_order_for_both_heads():
    torch.manual_seed(12)
    ia, ib = torch.tensor([0, 2, 3, 5, 6, 7, 9, 10]), torch.tensor([1, 2, 4, 5, 6, 8, 10, 11])
    va, vb = torch.zeros(1, 12, dtype=torch.bool), torch.zeros(1, 12, dtype=torch.bool)
    va[:, ia] = True; vb[:, ib] = True
    valid = torch.ones(1, 8, dtype=torch.bool)
    small_q = torch.rand(1, 8, 8)
    large_q = torch.full((1, 12, 12), float("nan"))
    large_q[:, ia[:, None], ib[None, :]] = small_q
    matrix = ThresholdedMatrixCNN().eval()
    torch.testing.assert_close(matrix(small_q, valid, valid), matrix(large_q, va, vb), rtol=0, atol=0)
    a, b = torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    aa, bb = torch.full((1, 12, 16), float("nan")), torch.full((1, 12, 16), float("nan"))
    aa[:, ia], bb[:, ib] = a, b
    head = CrossAttentionPairHead(16, 4).eval()
    torch.testing.assert_close(head(a, b, valid, valid), head(aa, bb, va, vb), rtol=0, atol=0)


def test_shared_matcher_initialization_and_explicit_m_weight_reuse():
    torch.manual_seed(13)
    matrix = build_decoupled_score_model(config(), "matrix_cnn")
    torch.manual_seed(13)
    cross = build_decoupled_score_model(config(), "cross_attention")
    unchanged(state(matrix.base_model), cross.base_model)
    with torch.no_grad():
        matrix.base_model.primal.weight.add_(.01)  # Stand-in for a committed M update.
    source = state(matrix.base_model)
    cross.base_model.load_state_dict(source, strict=True)
    cross.set_phase("classifier").train()
    unchanged(source, cross.base_model)
    assert not cross.base_model.training
    assert cross.metadata()["head_kind"] == "cross_attention"
    assert "AFTER existing context" in cross.metadata()["descriptor_protocol"]


def contrasting_matrices(size=32):
    # Synthetic CPU diagnostic only: connected thin seam versus broad region.
    q = torch.zeros(2, size, size)
    index = torch.arange(size)
    q[0, index, size - 1 - index] = .02
    q[1, size // 4:3 * size // 4, size // 4:3 * size // 4] = .02
    valid = torch.ones(2, size, dtype=torch.bool)
    return q, valid, valid


def test_v2_changes_only_final_bn_relu_order_and_rejects_unknown_revision():
    torch.manual_seed(73)
    legacy = ThresholdedMatrixCNN(revision="legacy")
    torch.manual_seed(73)
    fixed = ThresholdedMatrixCNN()
    assert fixed.revision == DEFAULT_MATRIX_HEAD_REVISION == "bn_relu_pool_v2"
    assert isinstance(legacy.cnn[9], torch.nn.ReLU) and isinstance(legacy.cnn[10], torch.nn.BatchNorm2d)
    assert isinstance(fixed.cnn[9], torch.nn.BatchNorm2d) and isinstance(fixed.cnn[10], torch.nn.ReLU)
    for index in range(9):
        assert type(legacy.cnn[index]) is type(fixed.cnn[index])
        unchanged(state(legacy.cnn[index]), fixed.cnn[index])
    unchanged(state(legacy.cnn[10]), fixed.cnn[9])
    unchanged(state(legacy.fc), fixed.fc)
    with pytest.raises(ValueError, match="revision"):
        ThresholdedMatrixCNN(revision="guess")


def test_train_mode_v2_retains_input_signal_and_convolution_gradients():
    q, va, vb = contrasting_matrices()
    torch.manual_seed(73)
    fixed = ThresholdedMatrixCNN().train()
    calls = []
    hook = fixed.cnn.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    logits = fixed(q, va, vb)
    hook.remove()
    assert calls == [1, 1, 1, 1]  # No hidden cross-pair batch-statistics change.
    assert abs(float(logits[0] - logits[1])) > 1e-3
    F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).backward()
    for index in (0, 4, 8):
        assert fixed.cnn[index].weight.grad.norm() > 1e-3
    torch.manual_seed(73)
    legacy = ThresholdedMatrixCNN(revision="legacy").train()
    old_logits = legacy(q, va, vb)
    assert abs(float(old_logits[0] - old_logits[1])) < 1e-6


def test_v2_tiny_cpu_pair_task_learns_in_training_mode():
    torch.manual_seed(73)
    fixed = ThresholdedMatrixCNN().train()
    data = contrasting_matrices()
    labels = torch.tensor([1., 0.])
    optimizer = torch.optim.AdamW(fixed.parameters(), lr=.003)
    with torch.no_grad():
        initial = F.binary_cross_entropy_with_logits(fixed(*data), labels).item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(fixed(*data), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = fixed(*data)
        final = F.binary_cross_entropy_with_logits(logits, labels).item()
    assert final < .5 * initial
    assert torch.equal(logits > 0, labels.bool())


@pytest.mark.parametrize("revision", ["legacy", "bn_relu_pool_v2"])
def test_matrix_revision_roundtrip_is_strict_and_legacy_without_field_reads_original_semantics(revision):
    torch.manual_seed(74)
    model = build_decoupled_score_model(config(), phase="classifier", matrix_head_revision=revision).eval()
    last_bn = model.score_head.cnn[10 if revision == "legacy" else 9]
    with torch.no_grad():
        last_bn.bias.add_(.13)  # The order is genuinely semantically relevant.
    data = contrasting_matrices()
    expected = model.score_head(*data)
    payload = dict(decoupled_score_schema=SCHEMA, decoupled_score=model.metadata(), model_state_dict=state(model))
    loaded = load_decoupled_score_checkpoint(payload).eval()
    assert loaded.matrix_head_revision == revision
    torch.testing.assert_close(loaded.score_head(*data), expected, rtol=0, atol=0)
    old = copy.deepcopy(payload)
    del old["decoupled_score"]["matrix_head_revision"]
    if revision == "legacy":
        loaded_old = load_decoupled_score_checkpoint(old).eval()
        assert loaded_old.matrix_head_revision == "legacy"
        torch.testing.assert_close(loaded_old.score_head(*data), expected, rtol=0, atol=0)
        # Old M12 base tensors remain reusable; old classifier weights are not.
        new = build_decoupled_score_model(config())
        new.base_model.load_state_dict(loaded_old.base_model.state_dict(), strict=True)
        assert new.matrix_head_revision == "bn_relu_pool_v2"
        unchanged(state(loaded_old.base_model), new.base_model)
    else:
        with pytest.raises(ValueError, match="registered semantics"):
            load_decoupled_score_checkpoint(old)
    for bad_revision in (None, "unregistered", "legacy" if revision != "legacy" else "bn_relu_pool_v2"):
        bad = copy.deepcopy(payload)
        bad["decoupled_score"]["matrix_head_revision"] = bad_revision
        with pytest.raises(ValueError):
            load_decoupled_score_checkpoint(bad)


def test_cross_attention_has_no_matrix_revision_and_old_metadata_roundtrips():
    model = build_decoupled_score_model(config(), "cross_attention").eval()
    assert model.metadata()["matrix_head_revision"] is None
    payload = dict(decoupled_score_schema=SCHEMA, decoupled_score=model.metadata(), model_state_dict=state(model))
    del payload["decoupled_score"]["matrix_head_revision"]
    restored = load_decoupled_score_checkpoint(payload)
    unchanged(state(model), restored)
    with pytest.raises(ValueError, match="revision"):
        matrix_revision_from_metadata(dict(head_kind="cross_attention", matrix_head_revision="legacy"))
