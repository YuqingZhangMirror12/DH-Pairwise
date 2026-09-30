from __future__ import annotations

import copy
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as subject
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelBatch
from staging.pairwise_v0_2.training.rachel_n512_runner import epoch_indices


def _fragment(token: str, lineage: str) -> dict:
    return {
        "fragment_token": token,
        "split_unit_id": lineage,
        "model_mask_path": "model/masks_800/" + token + ".png",
        "contour_path": "model/contours_n512/" + token + ".npz",
    }


def _row(
    split: str,
    pair_id: str,
    label: bool,
    token_a: str,
    token_b: str,
    lineage_a: str,
    lineage_b: str,
) -> dict:
    return {
        "split": split,
        "pair_id": pair_id,
        "label": label,
        "fragment_a": _fragment(token_a, lineage_a),
        "fragment_b": _fragment(token_b, lineage_b),
        "correspondence_path": (
            "targets/pairs/" + pair_id + ".npz" if label else None
        ),
    }


def _tiny_manifests(root: Path, *, overlap: bool = False) -> None:
    pairs = root / "pairs"
    pairs.mkdir(parents=True)
    train = [
        _row("train", "t-pos", True, "ta", "tb", "train-1", "train-1"),
        _row("train", "t-neg", False, "tc", "td", "train-1", "train-2"),
    ]
    val_lineage = "train-1" if overlap else "val-1"
    val = [
        _row("val", "v-pos", True, "va", "vb", val_lineage, val_lineage),
        _row("val", "v-neg", False, "vc", "vd", val_lineage, "val-2"),
    ]
    for split, rows in (("train", train), ("val", val)):
        (pairs / (split + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    # If the audit ever attempts to open this path, the monkeypatched test
    # below fails before malformed contents could be mistaken for evidence.
    (pairs / "test.jsonl").write_text("SEALED-SENTINEL\n", encoding="utf-8")


def _metadata(
    pair_id: str,
    label: bool,
    token_a: str,
    token_b: str,
    lineage_a: str = "l1",
    lineage_b: str = "l1",
) -> subject.PairMetadata:
    return subject.PairMetadata(
        split="val",
        pair_id=pair_id,
        label=label,
        fragment_a_token=token_a,
        fragment_b_token=token_b,
        lineage_a=lineage_a,
        lineage_b=lineage_b,
    )


def _adapter_identity() -> dict:
    return {
        "adapter_source_filename": "adapter.py",
        "adapter_source_sha256": "a" * 64,
        "adaptation_document_filename": "ADAPTATION.md",
        "adaptation_document_sha256": "b" * 64,
    }


def _dataset_binding() -> dict:
    return {
        "manifest_content_sha256": {"train": "c" * 64, "val": "d" * 64},
        "ordered_pair_ids_sha256": {"train": "e" * 64, "val": "f" * 64},
    }


def _dataset_audit(rows: tuple[subject.PairMetadata, ...]) -> dict:
    return {
        "manifest_content_sha256": {"train": "1" * 64, "val": "2" * 64},
        "ordered_pair_ids_sha256": {
            "train": subject._ordered_pair_ids_sha256("train", rows),
            "val": subject._ordered_pair_ids_sha256("val", rows),
        },
    }


def _validation_receipt(
    stage: str, epoch_zero_based: int, recipe: subject.ReleaseRecipe
) -> dict:
    metadata = tuple(
        _metadata("receipt-{}".format(index), True, "a", "b")
        for index in range(4)
    )
    return subject._stage_validation_order(
        stage,
        tuple(range(len(metadata))),
        metadata,
        recipe.seed,
        epoch_zero_based + 1,
    )[1]


def _batch(*, label: float = 1.0, target_value: int = 0) -> RachelBatch:
    cap = 20
    points_a = np.zeros((1, cap, 2), dtype=np.float32)
    points_b = np.zeros((1, cap, 2), dtype=np.float32)
    points_a[0, :, 0] = np.arange(cap)
    points_a[0, :, 1] = 2 * np.arange(cap)
    points_b[0] = points_a[0] + np.asarray((3.0, -4.0), dtype=np.float32)
    valid = np.ones((1, cap), dtype=np.bool_)
    target = np.full((1, cap), target_value, dtype=np.int64)
    mask = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask[:, :, 2:20, 2:20] = 1.0
    return RachelBatch(
        pair_ids=("pair-1",),
        fragment_a_tokens=("a",),
        fragment_b_tokens=("b",),
        mask_a=mask,
        mask_b=mask.copy(),
        coarse_mask_a=np.zeros((1, 1, 8, 8), dtype=np.float32),
        coarse_mask_b=np.zeros((1, 1, 8, 8), dtype=np.float32),
        points_rc_a=points_a,
        points_rc_b=points_b,
        contour_valid_a=valid,
        contour_valid_b=valid.copy(),
        target_a=target,
        target_b=target.copy(),
        labels=np.asarray((label,), dtype=np.float32),
        translation_a_to_b_rc=np.asarray(((3.0, -4.0),), dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray(((-4.0, -3.0),), dtype=np.float32),
        translation_valid=np.asarray((True,), dtype=np.bool_),
    )


class _FakeCoarse(nn.Module):
    def forward(self, mask, points, valid):
        del mask
        weight = valid[:, :, None].to(points.dtype)
        return (points * weight).reshape(points.shape[0], -1)


class _FakeClassify(nn.Module):
    def forward(
        self,
        mask_a,
        mask_b,
        points_a,
        points_b,
        valid_a,
        valid_b,
        correspondence_threshold,
    ):
        del mask_a, mask_b, points_b, correspondence_threshold
        batch, cap = points_a.shape[:2]
        probability = torch.zeros(
            (batch, cap, cap), dtype=points_a.dtype, device=points_a.device
        )
        diagonal = torch.arange(min(16, cap), device=points_a.device)
        probability[:, diagonal, diagonal] = 1.0
        probability *= (valid_a[:, :, None] & valid_b[:, None, :]).to(
            probability.dtype
        )
        return (
            torch.full((batch,), 0.9, dtype=points_a.dtype, device=points_a.device),
            probability,
            probability > 0.5,
        )


class _FakeGAT(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        del kwargs
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, value, edge_index):
        del edge_index
        return self.linear(value)


class _FakeDeepGCNLayer(nn.Module):
    def __init__(self, convolution, normalizer, activation, **kwargs):
        super().__init__()
        del kwargs
        self.convolution = convolution
        self.normalizer = normalizer
        self.activation = activation

    def forward(self, value, edge_index):
        return value + self.convolution(
            self.activation(self.normalizer(value)), edge_index
        )


def _repeat_batch(batch: RachelBatch, count: int) -> RachelBatch:
    values = {}
    for name in batch.__dataclass_fields__:
        value = getattr(batch, name)
        if isinstance(value, tuple):
            values[name] = tuple(
                "{}-{}".format(item, index)
                for index in range(count)
                for item in value
            )
        else:
            values[name] = np.repeat(value, count, axis=0)
    output = RachelBatch(**values)
    cap = output.target_a.shape[1]
    diagonal = np.repeat(np.arange(cap, dtype=np.int64)[None, :], count, axis=0)
    return replace(
        output,
        labels=np.asarray((1.0, 0.0), dtype=np.float32),
        target_a=diagonal,
        target_b=diagonal.copy(),
    )


def test_release_recipe_and_formal_seed_are_locked():
    recipe = subject.ReleaseRecipe()
    assert recipe.seed == 260831
    assert recipe.official_release_seed == 1024
    assert (recipe.coarse_epochs, recipe.matching_epochs, recipe.classify_epochs) == (
        128,
        128,
        128,
    )
    assert (
        recipe.coarse_batch_size,
        recipe.matching_batch_size,
        recipe.classify_batch_size,
    ) == (54, 36, 20)
    assert subject._stage_recipe("coarse", recipe)["selection_metric"] == "infor_loss"
    assert (
        subject._stage_recipe("matching", recipe)["selection_metric"]
        == "positive_loss"
    )
    assert subject._stage_recipe("classify", recipe)["selection_metric"] == "accuracy"
    kinds = {
        subject._stage_checkpoint_kind(stage, artifact)
        for stage in ("coarse", "matching", "classify")
        for artifact in ("progress", "winner")
    }
    kinds.add(subject.FREEZE_CHECKPOINT_KIND)
    assert len(kinds) == 7


def test_official_checkout_hash_audit_and_disclosed_conflict():
    root = Path(__file__).resolve().parents[3]
    receipt = subject.audit_official_source(root / "tmp" / "benchmarks" / "shreddingnet")
    assert receipt["commit"] == subject.OFFICIAL_COMMIT
    assert receipt["clean_checkout"] is True
    assert receipt["release_yaml_executable_recipe"]["epochs"]["classify"] == 128
    assert receipt["release_yaml_executable_recipe"][
        "validation_loader_shuffle"
    ] == {"coarse": True, "matching": False, "classify": False}
    assert receipt["paper_supplement_conflict_disclosed"]["supplement_epochs"]["classify"] == 10
    assert receipt["paper_metric_definitions"]["adapter_reports_native_CM_FM_SE"] is False


def test_manifest_audit_opens_train_val_only_and_checks_lineages(
    tmp_path: Path, monkeypatch
):
    _tiny_manifests(tmp_path)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == "test.jsonl":
            raise AssertionError("sealed test manifest was opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    receipt, train, val = subject.audit_rachel_train_val(
        tmp_path, require_formal_counts=False
    )
    assert receipt["opened_manifests"] == ["pairs/train.jsonl", "pairs/val.jsonl"]
    assert receipt["sealed_test_manifest_opened"] is False
    assert receipt["manifest_content_sha256"] == receipt["manifest_sha256"]
    assert receipt["ordered_pair_ids_sha256"] == {
        "train": subject._ordered_pair_ids_sha256("train", train),
        "val": subject._ordered_pair_ids_sha256("val", val),
    }
    assert len(train) == len(val) == 2
    assert val[1].cluster_id.startswith("unit-pair:")


def test_manifest_audit_rejects_train_val_lineage_overlap(tmp_path: Path):
    _tiny_manifests(tmp_path, overlap=True)
    with pytest.raises(subject.BenchmarkContractError, match="lineage overlap"):
        subject.audit_rachel_train_val(tmp_path, require_formal_counts=False)


def test_epoch_order_matches_formal_rachel_algorithm():
    population = (11, 13, 17, 19, 23)
    observed = subject._epoch_order_indices(population, 260831, 7)
    positions = epoch_indices(len(population), 260831, 7, None)
    assert observed == tuple(population[position] for position in positions)
    assert observed == subject._epoch_order_indices(population, 260831, 7)


def test_coarse_validation_order_matches_isolated_torch_random_sampler():
    population = (5, 7, 11, 13, 17, 19)
    metadata = tuple(
        _metadata("pair-{}".format(index), True, "a", "b")
        for index in range(20)
    )
    epoch = 9
    seed = 260831
    observed, receipt = subject._stage_validation_order(
        "coarse", population, metadata, seed, epoch
    )
    derived_seed = (
        seed + subject.COARSE_VALIDATION_EPOCH_SEED_MULTIPLIER * epoch
    )
    generator = torch.Generator().manual_seed(derived_seed)
    expected_positions = list(
        torch.utils.data.RandomSampler(
            range(len(population)), generator=generator
        )
    )
    expected = tuple(population[position] for position in expected_positions)
    assert observed == expected
    assert receipt["validation_loader_shuffle"] is True
    assert receipt["validation_order_seed"] == derived_seed
    assert subject._stage_validation_order(
        "coarse", population, metadata, seed, epoch
    ) == (observed, receipt)
    fixed, fixed_receipt = subject._stage_validation_order(
        "matching", population, metadata, seed, epoch
    )
    assert fixed == population
    assert fixed_receipt["validation_loader_shuffle"] is False


def test_coarse_validation_infonce_changes_with_batch_grouping():
    feature_a = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-0.9, 0.1]]
    )
    feature_b = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-0.8, 0.2]]
    )
    tokens = ("a", "b", "c", "d")

    def grouped_validation_loss(order):
        values = []
        for start in range(0, len(order), 2):
            indices = order[start : start + 2]
            _, cross = subject.released_coarse_loss(
                feature_a[indices],
                feature_b[indices],
                [tokens[index] for index in indices],
                [tokens[index] for index in indices],
                0.12,
            )
            values.append(float(cross))
        return sum(values) / len(values)

    adjacent_hard_negatives = grouped_validation_loss([0, 1, 2, 3])
    separated_hard_negatives = grouped_validation_loss([0, 2, 1, 3])
    assert adjacent_hard_negatives > separated_hard_negatives + 0.1


def test_validation_history_order_replay_is_resume_strict():
    recipe = subject.ReleaseRecipe()
    metadata = tuple(
        _metadata("order-{}".format(index), True, "a", "b")
        for index in range(8)
    )
    population = tuple(range(len(metadata)))
    history = []
    for epoch in range(3):
        train_order = subject._epoch_order_indices(
            population, recipe.seed, epoch + 1
        )
        _, val_receipt = subject._stage_validation_order(
            "coarse", population, metadata, recipe.seed, epoch + 1
        )
        history.append(
            {
                "epoch": epoch,
                "epoch_one_based": epoch + 1,
                "training_pair_order_sha256": subject._order_sha256(
                    metadata, train_order
                ),
                **val_receipt,
            }
        )
    subject._validate_stage_history_orders(
        stage="coarse",
        history=history,
        train_indices=population,
        val_indices=population,
        train_metadata=metadata,
        val_metadata=metadata,
        seed=recipe.seed,
    )
    tampered = copy.deepcopy(history)
    tampered[1]["validation_permutation_sha256"] = "0" * 64
    with pytest.raises(subject.BenchmarkContractError, match="validation order"):
        subject._validate_stage_history_orders(
            stage="coarse",
            history=tampered,
            train_indices=population,
            val_indices=population,
            train_metadata=metadata,
            val_metadata=metadata,
            seed=recipe.seed,
        )


def test_binary_metrics_topk_and_cluster_selection_views():
    assert subject.binary_auroc([False, True, False, True], [0.1, 0.8, 0.4, 0.8]) == 1.0
    rows = (
        _metadata("p1", True, "a", "b"),
        _metadata("p2", False, "a", "c"),
        _metadata("p3", True, "b", "c"),
    )
    selected = subject.topk_candidate_mask(rows, [0.9, 0.1, 0.8], 1)
    assert selected.tolist() == [True, False, True]
    metrics = subject._selection_metric_views(
        np.asarray([True, False, True]), selected, [row.cluster_id for row in rows]
    )
    assert metrics["row"]["precision"] == 1.0
    assert metrics["row"]["recall"] == 1.0


def test_assembly_wrong_pose_is_both_false_positive_and_false_negative():
    metrics = subject._assembly_metric_views(
        labels=np.asarray([True, True, False]),
        selected=np.asarray([True, True, True]),
        pose_correct=np.asarray([True, False, False]),
        pose_valid=np.asarray([True, True, True]),
        clusters=["c1", "c1", "c2"],
    )
    assert metrics["row"]["true_positive"] == 1
    assert metrics["row"]["false_positive"] == 2
    assert metrics["row"]["false_negative"] == 1


def test_assembly_invalid_pose_emits_no_predicted_transformed_edge():
    metrics = subject._assembly_metric_views(
        labels=np.asarray([True, False]),
        selected=np.asarray([True, True]),
        pose_correct=np.asarray([False, False]),
        pose_valid=np.asarray([False, False]),
        clusters=["c1", "c2"],
    )
    assert metrics["row"]["true_positive"] == 0
    assert metrics["row"]["false_positive"] == 0
    assert metrics["row"]["false_negative"] == 1
    assert metrics["row"]["predicted_transformed_edge_count"] == 0


def test_translation_consensus_rejects_outliers():
    source = np.stack((np.arange(25), 2 * np.arange(25)), axis=1).astype(np.float64)
    target = source + np.asarray((3.0, -4.0))
    target[-5:] += np.asarray((200.0, -100.0))
    result = subject.cauchy_translation_consensus(source, target)
    assert result.valid is True
    assert result.inlier_count == 20
    np.testing.assert_allclose(result.translation_rc, (3.0, -4.0), atol=0.05)


def test_deterministic_se2_ransac_recovers_transform():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(24, 2)) * 30.0
    angle = math.radians(12.0)
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )
    target = source @ rotation.T + np.asarray((8.0, -5.0))
    first = subject.deterministic_se2_ransac(source, target, seed=9)
    second = subject.deterministic_se2_ransac(source, target, seed=9)
    assert first.valid is True
    assert first.inlier_count == len(source)
    np.testing.assert_allclose(first.rotation, rotation, atol=1e-8)
    np.testing.assert_allclose(first.translation_rc, (8.0, -5.0), atol=1e-8)
    np.testing.assert_allclose(first.translation_rc, second.translation_rc)


def test_cycle_graph_and_patch_padding_contract():
    valid = torch.tensor([[True, True, True, True, False, False]])
    edges = subject.cycle_edge_index(valid, radius=1)
    assert edges.shape == (2, 12)
    assert int(edges.max()) == 3
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask[:, :, 1:6, 1:6] = 1.0
    points = torch.tensor([[[1.0, 1.0], [6.0, 6.0], [0.0, 0.0]]])
    point_valid = torch.tensor([[True, True, False]])
    contour, support = subject.mask_only_contour_support_patches(
        mask, points, point_valid, 7
    )
    assert contour.shape == (1, 3, 7, 7)
    assert support.shape == (1, 3, 3, 7, 7)
    assert torch.count_nonzero(contour[:, 2]).item() == 0
    assert torch.count_nonzero(support[:, 2]).item() == 0

    two_fragments = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    compact_edges = subject.cycle_edge_index(two_fragments, radius=1)
    assert compact_edges.shape == (2, 15)
    assert int(compact_edges.max()) == 4
    assert torch.all(
        ((compact_edges[0] < 3) & (compact_edges[1] < 3))
        | ((compact_edges[0] >= 3) & (compact_edges[1] >= 3))
    )
    with pytest.raises(subject.BenchmarkContractError, match="contiguous prefix"):
        subject.cycle_edge_index(
            torch.tensor([[True, False, True, False]]), radius=1
        )


def test_padding_slots_do_not_change_valid_features_or_batchnorm(monkeypatch):
    monkeypatch.setattr(
        subject,
        "_require_torch_geometric",
        lambda: (_FakeDeepGCNLayer, _FakeGAT),
    )
    recipe = subject.ReleaseRecipe(resgcn_blocks=2, cycle_radius=2)

    def inputs(cap: int):
        mask = torch.zeros((2, 1, 40, 40), dtype=torch.float32)
        mask[:, :, 3:35, 3:35] = 1.0
        points = torch.full((2, cap, 2), 999.0, dtype=torch.float32)
        valid = torch.zeros((2, cap), dtype=torch.bool)
        for row, length in enumerate((12, 7)):
            index = torch.arange(length, dtype=torch.float32)
            points[row, :length, 0] = 5.0 + index
            points[row, :length, 1] = 7.0 + 0.5 * index
            valid[row, :length] = True
        return mask, points, valid

    torch.manual_seed(17)
    first = subject.ReleasedFragmentEncoder(recipe)
    second = subject.ReleasedFragmentEncoder(recipe)
    second.load_state_dict(first.state_dict(), strict=True)
    first.train()
    second.train()
    output_short = first.local_features(*inputs(16))
    output_long = second.local_features(*inputs(31))
    torch.testing.assert_close(output_short[0, :12], output_long[0, :12])
    torch.testing.assert_close(output_short[1, :7], output_long[1, :7])
    first_buffers = dict(first.named_buffers())
    second_buffers = dict(second.named_buffers())
    assert set(first_buffers) == set(second_buffers)
    for name in first_buffers:
        torch.testing.assert_close(first_buffers[name], second_buffers[name])


def test_float_morphology_matches_release_opencv_order():
    rng = np.random.default_rng(31)
    matrix = rng.random((2, 9, 11), dtype=np.float32)
    threshold = 0.6
    observed = subject.released_antidiagonal_morphology(
        torch.from_numpy(matrix), threshold
    ).numpy()
    expected = []
    for value in matrix:
        kernel = np.eye(3, dtype=np.uint8)
        kernel[1, 1] = 0
        kernel = np.rot90(kernel)
        eroded = cv2.erode(
            value, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        kernel[1, 1] = 1
        dilated = cv2.dilate(
            eroded, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        expected.append(dilated > threshold)
    np.testing.assert_array_equal(observed, np.stack(expected))


def test_dual_softmax_masks_padding_and_matches_formula():
    logits = torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]])
    valid = torch.tensor([[[True, True, False], [True, True, False]]])
    observed = subject.released_dual_softmax(logits, valid)
    masked = logits.masked_fill(~valid, -1e9)
    expected = torch.softmax(masked, dim=1) * torch.softmax(masked, dim=-1)
    expected *= valid
    torch.testing.assert_close(observed, expected)
    assert torch.count_nonzero(observed[:, :, 2]).item() == 0


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_dual_softmax_amp_logits_use_finite_fp32_probability(dtype):
    logits = torch.tensor(
        [[[1.0, 2.0, 200.0], [3.0, 4.0, -200.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    valid = torch.tensor([[[True, True, False], [True, True, False]]])
    observed = subject.released_dual_softmax(logits, valid)
    reference_logits = logits.detach().float().masked_fill(~valid, -1e9)
    reference = (
        torch.softmax(reference_logits, dim=1)
        * torch.softmax(reference_logits, dim=-1)
        * valid
    )
    assert observed.dtype == torch.float32
    assert bool(torch.isfinite(observed).all())
    assert torch.count_nonzero(observed[:, :, 2]).item() == 0
    torch.testing.assert_close(observed, reference)
    observed.sum().backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert torch.count_nonzero(logits.grad[:, :, 2]).item() == 0


def test_focal_loss_exact_zero_one_positive_negative_cells_stay_finite():
    probability = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    ground_truth = torch.tensor([[[True, False], [False, True]]])
    valid_pairs = torch.ones_like(ground_truth)
    total, positive = subject.released_focal_correspondence_loss(
        probability,
        ground_truth,
        valid_pairs,
        alpha=0.55,
        gamma=8.0,
    )
    assert bool(torch.isfinite(total))
    assert bool(torch.isfinite(positive))
    total.backward()
    assert probability.grad is not None
    assert bool(torch.isfinite(probability.grad).all())
    assert probability.grad[0, 0, 0] < 0.0  # hard positive at p=0
    assert probability.grad[0, 0, 1] > 0.0  # hard negative at p=1


def test_label_free_inference_ignores_every_supervision_field():
    runtime = subject.RuntimeBatchConfig(amp=False)
    inference = subject.RachelShreddingNetInference(
        coarse=_FakeCoarse(),
        classify=_FakeClassify(),
        recipe=subject.ReleaseRecipe(),
        runtime=runtime,
        pair_threshold=0.5,
        device=torch.device("cpu"),
    )
    first_batch = _batch(label=1.0, target_value=0)
    second_batch = replace(
        first_batch,
        labels=np.asarray((0.0,), dtype=np.float32),
        target_a=np.full_like(first_batch.target_a, -2),
        target_b=np.full_like(first_batch.target_b, -2),
        translation_a_to_b_rc=np.asarray(((999.0, 999.0),), dtype=np.float32),
        translation_a_to_b_xy_cartesian=np.asarray(
            ((999.0, 999.0),), dtype=np.float32
        ),
        translation_valid=np.asarray((False,), dtype=np.bool_),
    )
    first = inference.predict_batch(first_batch, return_correspondence=True)
    second = inference.predict_batch(second_batch, return_correspondence=True)
    assert first.method_id == subject.METHOD_ID
    assert first.translation_valid.tolist() == [True]
    np.testing.assert_allclose(first.translation_hat_rc, ((3.0, -4.0),))
    for name in (
        "coarse_score",
        "pair_score",
        "translation_hat_rc",
        "translation_valid",
        "correspondence_count",
        "se2_translation_hat_rc",
        "se2_rotation_degrees",
        "se2_valid",
        "correspondence_probability",
        "correspondence_binary",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_all_effective_batch_training_paths_run_with_microbatch_gradient_cache(
    monkeypatch,
):
    monkeypatch.setattr(
        subject,
        "_require_torch_geometric",
        lambda: (_FakeDeepGCNLayer, _FakeGAT),
    )
    recipe = subject.ReleaseRecipe(
        resgcn_blocks=1,
        decoder_blocks=1,
        coarse_epochs=1,
        matching_epochs=1,
        classify_epochs=1,
        coarse_batch_size=2,
        matching_batch_size=2,
        classify_batch_size=2,
    )
    runtime = subject.RuntimeBatchConfig(amp=False)
    batch = _repeat_batch(_batch(), 2)
    for stage in ("coarse", "matching", "classify"):
        torch.manual_seed(7)
        model = subject._stage_model(stage, recipe)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        metrics = subject._train_effective_batch(
            stage=stage,
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            recipe=recipe,
            runtime=runtime,
            device=torch.device("cpu"),
        )
        assert all(math.isfinite(float(value)) for value in metrics.values())
        if stage == "classify":
            assert model.matcher.training is True
            assert not any(
                parameter.requires_grad for parameter in model.matcher.parameters()
            )


def test_runtime_resource_and_cli_contracts():
    runtime = subject.RuntimeBatchConfig(
        coarse_microbatch=3,
        matching_microbatch=2,
        classify_microbatch=2,
        amp=True,
    )
    envelope = subject.resource_envelope(runtime=runtime)
    assert envelope["release_effective_batch"] == {
        "coarse": 54,
        "matching": 36,
        "classify": 20,
    }
    assert envelope["nominal_accumulation_steps"] == {
        "coarse": 18,
        "matching": 18,
        "classify": 10,
    }
    parsed = subject._build_parser().parse_args(
        [
            "train",
            "--dataset-root",
            "/dataset",
            "--official-repo",
            "/official",
            "--output-root",
            "/output",
            "--resume",
            "--no-amp",
        ]
    )
    assert parsed.command == "train"
    assert parsed.resume is True
    assert subject._runtime_from_arguments(parsed).amp is False


def test_resume_repairs_only_current_best_winner_commit_gap(tmp_path: Path):
    recipe = subject.ReleaseRecipe()
    runtime = subject.RuntimeBatchConfig(amp=False)
    settings = subject._stage_recipe("coarse", recipe)
    state = {"weight": torch.arange(3, dtype=torch.float32)}
    progress = {
        "epoch_completed": 4,
        "best_epoch": 4,
        "best_value": 0.125,
        "model_state_dict": state,
    }
    winner = tmp_path / "winner.pt"
    repaired = subject._repair_or_verify_progress_winner(
        stage="coarse",
        progress=progress,
        winner_path=winner,
        settings=settings,
        recipe=recipe,
        runtime=runtime,
        device=torch.device("cpu"),
        matching_winner_sha256=None,
        adapter_identity=_adapter_identity(),
        dataset_binding=_dataset_binding(),
        validation_order_receipt=_validation_receipt("coarse", 4, recipe),
    )
    assert repaired is True
    checkpoint = subject._torch_load(winner, torch.device("cpu"))
    assert checkpoint["checkpoint_kind"] == subject._stage_checkpoint_kind(
        "coarse", "winner"
    )
    assert checkpoint["train_val_provenance"] == subject._train_val_provenance(
        _adapter_identity(), _dataset_binding()
    )
    assert subject._winner_identity_matches(
        checkpoint,
        stage="coarse",
        best_epoch=4,
        best_value=0.125,
        settings=settings,
        recipe=recipe,
        runtime=runtime,
        matching_winner_sha256=None,
        adapter_identity=_adapter_identity(),
        dataset_binding=_dataset_binding(),
        validation_order_receipt=_validation_receipt("coarse", 4, recipe),
    )
    torch.testing.assert_close(checkpoint["model_state_dict"]["weight"], state["weight"])
    assert (
        subject._repair_or_verify_progress_winner(
            stage="coarse",
            progress=progress,
            winner_path=winner,
            settings=settings,
            recipe=recipe,
            runtime=runtime,
            device=torch.device("cpu"),
            matching_winner_sha256=None,
            adapter_identity=_adapter_identity(),
            dataset_binding=_dataset_binding(),
            validation_order_receipt=_validation_receipt("coarse", 4, recipe),
        )
        is False
    )


def test_resume_rejects_missing_older_best_winner(tmp_path: Path):
    recipe = subject.ReleaseRecipe()
    with pytest.raises(subject.BenchmarkContractError, match="cannot be repaired"):
        subject._repair_or_verify_progress_winner(
            stage="matching",
            progress={
                "epoch_completed": 4,
                "best_epoch": 2,
                "best_value": 0.25,
                "model_state_dict": {"weight": torch.ones(1)},
            },
            winner_path=tmp_path / "winner.pt",
            settings=subject._stage_recipe("matching", recipe),
            recipe=recipe,
            runtime=subject.RuntimeBatchConfig(amp=False),
            device=torch.device("cpu"),
            matching_winner_sha256=None,
            adapter_identity=_adapter_identity(),
            dataset_binding=_dataset_binding(),
            validation_order_receipt=_validation_receipt("matching", 2, recipe),
        )


def test_coarse_stage_persists_and_resumes_exact_validation_permutations(
    tmp_path: Path, monkeypatch
):
    recipe = subject.ReleaseRecipe(coarse_epochs=2, coarse_batch_size=2)
    runtime = subject.RuntimeBatchConfig(amp=False)
    metadata = tuple(
        _metadata("stage-{}".format(index), True, "a", "b")
        for index in range(5)
    )
    observed_orders = []

    def fake_loader(dataset, indices, **kwargs):
        del dataset
        order = tuple(indices)
        observed_orders.append(order)
        batch_size = int(kwargs["batch_size"])
        return [
            SimpleNamespace(pair_ids=tuple(str(value) for value in order[start:stop]))
            for start in range(0, len(order), batch_size)
            for stop in (min(start + batch_size, len(order)),)
        ]

    monkeypatch.setattr(subject, "_loader", fake_loader)
    monkeypatch.setattr(subject, "_stage_model", lambda *args: nn.Linear(1, 1))

    def fake_train(*, optimizer, model, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.sum() * 0.0 for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        return {"loss": 1.0, "infor_loss": 1.0, "top1_recall": 0.0, "top5_recall": 0.0}

    monkeypatch.setattr(subject, "_train_effective_batch", fake_train)
    monkeypatch.setattr(
        subject,
        "_validate_effective_batch",
        lambda **kwargs: {
            "loss": 1.0,
            "infor_loss": 1.0,
            "top1_recall": 0.0,
            "top5_recall": 0.0,
        },
    )
    arguments = {
        "stage": "coarse",
        "output_root": tmp_path,
        "train_dataset": object(),
        "val_dataset": object(),
        "train_metadata": metadata,
        "val_metadata": metadata,
        "recipe": recipe,
        "runtime": runtime,
        "device": torch.device("cpu"),
        "num_workers": 0,
        "adapter_identity": _adapter_identity(),
        "dataset_binding": _dataset_binding(),
    }
    completion = subject._run_stage(**arguments, resume=False)
    assert completion["validation_order_contract"] == subject._validation_order_contract(
        "coarse", recipe.seed
    )
    assert len(completion["history"]) == 2
    expected_validation_orders = [
        subject._stage_validation_order(
            "coarse", tuple(range(5)), metadata, recipe.seed, epoch
        )[0]
        for epoch in (1, 2)
    ]
    # Each epoch creates train then validation loader.
    assert observed_orders[1::2] == expected_validation_orders
    winner = subject._torch_load(
        tmp_path / "stages" / "coarse" / "winner.pt", torch.device("cpu")
    )
    progress = subject._torch_load(
        tmp_path / "stages" / "coarse" / "progress.pt", torch.device("cpu")
    )
    assert winner["winner_validation_order"] == completion[
        "winner_validation_order"
    ]
    assert progress["best_validation_order"] == completion[
        "winner_validation_order"
    ]
    assert subject._run_stage(**arguments, resume=True) == completion


@pytest.mark.parametrize(
    "field_path",
    (
        ("checkpoint_kind",),
        ("adapter_identity", "adapter_source_sha256"),
        ("adapter_identity", "adaptation_document_sha256"),
        ("dataset_binding", "manifest_content_sha256", "train"),
        ("dataset_binding", "manifest_content_sha256", "val"),
        ("dataset_binding", "ordered_pair_ids_sha256", "train"),
        ("dataset_binding", "ordered_pair_ids_sha256", "val"),
        ("train_val_provenance", "adapter_source_sha256"),
        ("validation_order_contract", "formal_seed"),
        ("winner_validation_order", "validation_permutation_sha256"),
    ),
)
def test_winner_identity_rejects_each_provenance_tamper(field_path):
    recipe = subject.ReleaseRecipe()
    runtime = subject.RuntimeBatchConfig(amp=False)
    adapter = _adapter_identity()
    dataset = _dataset_binding()
    settings = subject._stage_recipe("coarse", recipe)
    payload = subject._winner_checkpoint_payload(
        stage="coarse",
        epoch=3,
        settings=settings,
        selection_value=0.25,
        recipe=recipe,
        runtime=runtime,
        model_state_dict={"weight": torch.ones(1)},
        matching_winner_sha256=None,
        adapter_identity=adapter,
        dataset_binding=dataset,
        validation_order_receipt=_validation_receipt("coarse", 3, recipe),
    )
    tampered = copy.deepcopy(payload)
    cursor = tampered
    for field in field_path[:-1]:
        cursor = cursor[field]
    cursor[field_path[-1]] = "0" * 64
    assert not subject._winner_identity_matches(
        tampered,
        stage="coarse",
        best_epoch=3,
        best_value=0.25,
        settings=settings,
        recipe=recipe,
        runtime=runtime,
        matching_winner_sha256=None,
        adapter_identity=adapter,
        dataset_binding=dataset,
        validation_order_receipt=_validation_receipt("coarse", 3, recipe),
    )


def test_progress_checkpoint_cannot_load_as_winner(tmp_path: Path):
    recipe = subject.ReleaseRecipe()
    runtime = subject.RuntimeBatchConfig(amp=False)
    adapter = _adapter_identity()
    dataset = _dataset_binding()
    payload = subject._winner_checkpoint_payload(
        stage="coarse",
        epoch=0,
        settings=subject._stage_recipe("coarse", recipe),
        selection_value=1.0,
        recipe=recipe,
        runtime=runtime,
        model_state_dict={"weight": torch.ones(1)},
        matching_winner_sha256=None,
        adapter_identity=adapter,
        dataset_binding=dataset,
        validation_order_receipt=_validation_receipt("coarse", 0, recipe),
    )
    payload["checkpoint_kind"] = subject._stage_checkpoint_kind(
        "coarse", "progress"
    )
    path = tmp_path / "progress-as-winner.pt"
    subject._torch_save_atomic(path, payload)
    with pytest.raises(subject.BenchmarkContractError, match="winner contract"):
        subject._load_winner_state(
            path,
            stage="coarse",
            recipe=recipe,
            runtime=runtime,
            adapter_identity=adapter,
            dataset_binding=dataset,
            device=torch.device("cpu"),
        )


def test_path_guards_reject_live_and_dangling_symlinks(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.BenchmarkContractError, match="symlink"):
        subject._prepare_fresh_file_path(alias / "output.json", name="output")
    with pytest.raises(subject.BenchmarkContractError, match="symlink"):
        subject.load_frozen_inference(alias / "freeze.json", device="cpu")
    with pytest.raises(subject.BenchmarkContractError, match="symlink"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=alias,
            device="cpu",
            resume=True,
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(subject.BenchmarkContractError, match="symlink"):
        subject._prepare_fresh_file_path(dangling, name="output")


def test_create_only_publish_never_overwrites_existing_or_racing_file(
    tmp_path: Path, monkeypatch
):
    existing = tmp_path / "existing.json"
    existing.write_text("owner\n", encoding="utf-8")
    with pytest.raises(subject.BenchmarkContractError, match="already exists"):
        subject._write_json_new_atomic(existing, {"value": 1}, name="receipt")
    assert existing.read_text(encoding="utf-8") == "owner\n"

    racing = tmp_path / "racing.json"
    original_link = subject.os.link

    def publish_race(source, destination, **kwargs):
        Path(destination).write_text("racer\n", encoding="utf-8")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(subject.os, "link", publish_race)
    with pytest.raises(subject.BenchmarkContractError, match="appeared during publish"):
        subject._write_json_new_atomic(racing, {"value": 2}, name="receipt")
    assert racing.read_text(encoding="utf-8") == "racer\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_precreated_temporary_symlink_cannot_touch_sentinel(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "receipt.json"
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    monkeypatch.setattr(subject.time, "time_ns", lambda: 12345)
    temporary = destination.with_name(
        destination.name + ".{}.12345.tmp".format(subject.os.getpid())
    )
    temporary.symlink_to(sentinel)
    with pytest.raises(subject.BenchmarkContractError, match="temporary path"):
        subject._write_json_new_atomic(destination, {"value": 1}, name="receipt")
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"
    assert not destination.exists()


def test_post_score_pre_freeze_resume_reuses_only_exact_recomputation(
    tmp_path: Path,
):
    path = tmp_path / "validation_threshold_scores.jsonl"
    rows = (
        {"ordinal": 0, "pair_id": "p0", "pair_score": 0.25},
        {"ordinal": 1, "pair_id": "p1", "pair_score": 0.75},
    )
    first = subject._publish_or_reuse_exact_jsonl(
        path,
        rows,
        resume=False,
        name="validation threshold scores",
    )
    assert first["reused_after_verified_resume"] is False
    # Simulate a process failure after score publication but before freeze.
    assert not (tmp_path / "train_val_freeze.json").exists()
    resumed = subject._publish_or_reuse_exact_jsonl(
        path,
        rows,
        resume=True,
        name="validation threshold scores",
    )
    assert resumed["reused_after_verified_resume"] is True
    assert resumed["sha256"] == first["sha256"]
    assert resumed["row_count"] == 2

    path.write_text(
        path.read_text(encoding="utf-8").replace("0.75", "0.76"),
        encoding="utf-8",
    )
    with pytest.raises(subject.BenchmarkContractError, match="recomputation"):
        subject._publish_or_reuse_exact_jsonl(
            path,
            rows,
            resume=True,
            name="validation threshold scores",
        )


def test_threshold_freeze_crash_after_score_resumes_transaction(
    tmp_path: Path, monkeypatch
):
    metadata = (
        _metadata("p0", True, "a", "b"),
        _metadata("p1", False, "c", "d", "l1", "l2"),
    )
    score_rows = (
        {
            "schema_version": "rachel-shreddingnet-val-score/1.0",
            "method_id": subject.METHOD_ID,
            "ordinal": 0,
            "pair_id": "p0",
            "label": True,
            "cluster_id": metadata[0].cluster_id,
            "coarse_score": 0.8,
            "pair_score": 0.9,
        },
        {
            "schema_version": "rachel-shreddingnet-val-score/1.0",
            "method_id": subject.METHOD_ID,
            "ordinal": 1,
            "pair_id": "p1",
            "label": False,
            "cluster_id": metadata[1].cluster_id,
            "coarse_score": 0.2,
            "pair_score": 0.1,
        },
    )
    dataset_audit = {
        **_dataset_audit(metadata),
        "manifest_sha256": {"train": "1" * 64, "val": "2" * 64},
    }
    stage_receipts = {}
    for stage in ("coarse", "matching", "classify"):
        settings = subject._stage_recipe(stage, subject.ReleaseRecipe())
        stage_receipts[stage] = {
            "winner_path": "stages/{}/winner.pt".format(stage),
            "winner_sha256": stage[0] * 64,
            "winner_epoch": 3,
            "settings": settings,
            "content_sha256": stage[-1] * 64,
        }
    monkeypatch.setattr(
        subject, "_restore_three_stage_models", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        subject, "_score_validation_for_threshold", lambda **kwargs: score_rows
    )
    original_publish = subject._write_json_new_atomic

    def crash_at_freeze(path, value, *, name):
        if name == "train/val freeze":
            raise subject.BenchmarkContractError("injected post-score crash")
        return original_publish(path, value, name=name)

    monkeypatch.setattr(subject, "_write_json_new_atomic", crash_at_freeze)
    call = {
        "output_root": tmp_path,
        "dataset_audit": dataset_audit,
        "official_audit": {},
        "val_dataset": object(),
        "val_metadata": metadata,
        "stage_receipts": stage_receipts,
        "recipe": subject.ReleaseRecipe(),
        "runtime": subject.RuntimeBatchConfig(amp=False),
        "adapter_identity": _adapter_identity(),
        "dataset_binding": subject._dataset_binding(dataset_audit),
        "device": torch.device("cpu"),
        "num_workers": 0,
    }
    with pytest.raises(subject.BenchmarkContractError, match="post-score"):
        subject._threshold_and_freeze(**call, resume=False)
    assert (tmp_path / "validation_threshold_scores.jsonl").is_file()
    assert not (tmp_path / "train_val_freeze.json").exists()

    monkeypatch.setattr(subject, "_write_json_new_atomic", original_publish)
    freeze = subject._threshold_and_freeze(**call, resume=True)
    assert freeze["threshold"]["score_file_reused_after_verified_resume"] is True
    assert freeze["threshold"]["score_file_row_count"] == len(score_rows)
    assert (tmp_path / "train_val_freeze.json").is_file()


def test_training_root_state_accepts_empty_or_contract_and_rejects_unknown(
    tmp_path: Path,
):
    root = tmp_path / "run"
    assert subject._inspect_training_output_root(root) == "absent"
    root.mkdir()
    assert subject._inspect_training_output_root(root) == "safe_empty_orphan"
    (root / "run_contract.json").write_text("{}\n", encoding="utf-8")
    assert subject._inspect_training_output_root(root) == "contract_only"
    (root / "unknown.txt").write_text("no\n", encoding="utf-8")
    with pytest.raises(subject.BenchmarkContractError, match="unknown entries"):
        subject._inspect_training_output_root(root)


def test_run_root_mkdir_contract_crash_is_recoverable(
    tmp_path: Path, monkeypatch
):
    rows = (
        _metadata("p", True, "a", "b"),
        _metadata("n", False, "c", "d", "l1", "l2"),
    )
    audit = {
        **_dataset_audit(rows),
        "manifest_sha256": {"train": "1" * 64, "val": "2" * 64},
    }
    monkeypatch.setattr(
        subject,
        "audit_rachel_train_val",
        lambda *args, **kwargs: (audit, rows, rows),
    )
    monkeypatch.setattr(subject, "audit_official_source", lambda *args: {})
    monkeypatch.setattr(subject, "RachelPairDataset", lambda *args: object())
    monkeypatch.setattr(
        subject,
        "_formal_runtime_preflight",
        lambda **kwargs: {"preflight": "complete"},
    )
    output = tmp_path / "formal"
    original_publish = subject._write_json_new_atomic

    def crash_before_contract(path, value, *, name):
        if name == "run contract":
            raise subject.BenchmarkContractError("injected contract publish crash")
        return original_publish(path, value, name=name)

    monkeypatch.setattr(subject, "_write_json_new_atomic", crash_before_contract)
    with pytest.raises(subject.BenchmarkContractError, match="injected"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=output,
            device="cpu",
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )
    assert subject._inspect_training_output_root(output) == "safe_empty_orphan"

    monkeypatch.setattr(subject, "_write_json_new_atomic", original_publish)

    with pytest.raises(subject.BenchmarkContractError, match="must be absent"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=output,
            device="cpu",
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )

    def stop_after_contract(**kwargs):
        raise subject.BenchmarkContractError("stop after recovered contract")

    monkeypatch.setattr(subject, "_run_stage", stop_after_contract)
    with pytest.raises(subject.BenchmarkContractError, match="recovered contract"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=output,
            device="cpu",
            resume=True,
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )
    contract = subject._read_verified_json(output / "run_contract.json", "run contract")
    assert contract["scope"]["recovered_safe_empty_orphan"] is True

    with pytest.raises(subject.BenchmarkContractError, match="stop after recovered"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=output,
            device="cpu",
            resume=True,
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )


def test_adapter_identity_hashes_source_and_adaptation_document():
    identity = subject._adapter_identity()
    source = Path(subject.__file__).resolve()
    document = source.with_name(subject.ADAPTATION_DOC_FILENAME)
    assert identity["adapter_source_sha256"] == subject._sha256_file(source)
    assert identity["adaptation_document_sha256"] == subject._sha256_file(document)


def test_train_preflight_failure_leaves_no_output(tmp_path: Path, monkeypatch):
    rows = (
        _metadata("p", True, "a", "b"),
        _metadata("n", False, "c", "d", "l1", "l2"),
    )
    audit = _dataset_audit(rows)
    monkeypatch.setattr(
        subject,
        "audit_rachel_train_val",
        lambda *args, **kwargs: (audit, rows, rows),
    )
    monkeypatch.setattr(subject, "audit_official_source", lambda *args: {})
    monkeypatch.setattr(subject, "RachelPairDataset", lambda *args: object())

    def fail_preflight(**kwargs):
        raise subject.BenchmarkContractError("injected preflight failure")

    monkeypatch.setattr(subject, "_formal_runtime_preflight", fail_preflight)
    output = tmp_path / "not-created" / "formal"
    with pytest.raises(subject.BenchmarkContractError, match="injected"):
        subject.train_rachel_shreddingnet(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_root=output,
            device="cpu",
            write_validation_report=False,
            runtime=subject.RuntimeBatchConfig(amp=False),
        )
    assert not output.exists()
    assert not output.parent.exists()


def test_gpu_smoke_preflight_failure_publishes_no_success_artifact(
    tmp_path: Path, monkeypatch
):
    rows = (
        _metadata("p", True, "a", "b"),
        _metadata("n", False, "c", "d", "l1", "l2"),
    )
    audit = _dataset_audit(rows)
    monkeypatch.setattr(subject, "_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(
        subject,
        "audit_rachel_train_val",
        lambda *args, **kwargs: (audit, rows, rows),
    )
    monkeypatch.setattr(subject, "audit_official_source", lambda *args: {})
    monkeypatch.setattr(subject, "RachelPairDataset", lambda *args: object())

    def fail_preflight(**kwargs):
        raise subject.BenchmarkContractError("injected smoke preflight failure")

    monkeypatch.setattr(subject, "_formal_runtime_preflight", fail_preflight)
    output = tmp_path / "not-created" / "smoke.json"
    with pytest.raises(subject.BenchmarkContractError, match="injected"):
        subject.run_gpu_smoke(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_path=output,
        )
    assert not output.exists()
    assert not output.parent.exists()


def test_gpu_smoke_existing_output_rejected_before_device_or_model(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "smoke.json"
    output.write_text("owner\n", encoding="utf-8")

    def unexpected_device(value):
        raise AssertionError("device/model preflight must not start")

    monkeypatch.setattr(subject, "_device", unexpected_device)
    with pytest.raises(subject.BenchmarkContractError, match="already exists"):
        subject.run_gpu_smoke(
            dataset_root=tmp_path,
            official_repo=tmp_path,
            output_path=output,
        )
    assert output.read_text(encoding="utf-8") == "owner\n"


def test_real_three_model_preflight_contract_runs_with_fake_pyg(monkeypatch):
    monkeypatch.setattr(
        subject,
        "_require_torch_geometric",
        lambda: (_FakeDeepGCNLayer, _FakeGAT),
    )
    monkeypatch.setattr(subject.importlib_metadata, "version", lambda name: "test")
    rows = (
        _metadata("p", True, "a", "b"),
        _metadata("n", False, "c", "d", "l1", "l2"),
    )
    audit = _dataset_audit(rows)

    def parse(root, split):
        del root
        return rows, audit["manifest_content_sha256"][split]

    monkeypatch.setattr(subject, "_parse_pair_manifest", parse)
    monkeypatch.setattr(
        subject,
        "collate_rachel_pairs",
        lambda samples: _batch() if len(samples) == 1 else _repeat_batch(_batch(), 2),
    )

    class Dataset:
        def __getitem__(self, index):
            return index

    receipt = subject._formal_runtime_preflight(
        dataset_root="/dataset",
        dataset_audit=audit,
        train_dataset=Dataset(),
        train_metadata=rows,
        recipe=subject.ReleaseRecipe(resgcn_blocks=1, decoder_blocks=1),
        device=torch.device("cpu"),
    )
    assert receipt["real_stage_model_constructed_and_forwarded"] == {
        "coarse": True,
        "matching": True,
        "classify": True,
    }
    assert receipt["output_created_or_written"] is False


def test_balanced_pair_list_diagnostics_are_not_named_native_module_metrics():
    source = inspect.getsource(subject.evaluate_validation)
    assert "balanced_selected_pair_list_module_like_diagnostics" in source
    assert '"shreddingnet_module_metrics"' not in source
    assert '"cm"' not in source
    document = Path(subject.__file__).with_name(subject.ADAPTATION_DOC_FILENAME)
    text = document.read_text(encoding="utf-8")
    assert "No exhaustive" in text and "parent-local graph is claimed" in text
    assert "headline is pairwise assembly-edge" in text
