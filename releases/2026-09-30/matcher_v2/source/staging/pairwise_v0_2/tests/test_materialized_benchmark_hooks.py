"""Shared damaged TRAIN injection, inherited-ignore and unchanged-loss tests."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from staging.pairwise_v0_2.baselines import rachel_materialized_training as hook
from staging.pairwise_v0_2.baselines import rachel_pairingnet_benchmark as pairing
from staging.pairwise_v0_2.baselines import rachel_shreddingnet_benchmark as shredding
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import SCHEMA, save_sample
from staging.pairwise_v0_2.pairwise_data.rachel_training_dataset import RachelPairSample, collate_rachel_pairs


def sample(pair_id="train-p", label=True, gaps=True):
    a = np.asarray([[5, 5], [5, 6], [5, 7], [7, 7], [7, 6], [7, 5]], np.float32)
    va = np.asarray([True, False, True, True, False, True]) if gaps else np.ones(6, bool)
    vb = np.asarray([True, True, False, True, False, True]) if gaps else np.ones(6, bool)
    ta = np.asarray([1, -2, 3, -2, -2, -1]) if gaps else np.asarray([0, -1, 2, -1, 4, -1])
    tb = np.asarray([-1, 0, -2, 2, -2, -2]) if gaps else ta.copy()
    if not label:
        ta, tb = np.where(va, -1, -2), np.where(vb, -1, -2)
    mask = np.zeros((1, 32, 32), np.float32)
    mask[:, 4:10, 4:10] = 1.
    translation = np.asarray([3., -4.], np.float32) if label else np.zeros(2, np.float32)
    return RachelPairSample(pair_id, pair_id + "a", pair_id + "b", mask, mask.copy(),
        mask[:, ::4, ::4].copy(), mask[:, ::4, ::4].copy(), a, a + [3., -4.],
        va, vb, ta, tb, np.float32(label), translation,
        np.asarray([translation[1], -translation[0]], np.float32), np.bool_(label))


def source_row(pair_id, label, split):
    return dict(pair_id=pair_id, label=label, split=split,
        fragment_a=dict(fragment_token=pair_id + "a", split_unit_id=split + "-a",
            model_mask_path="model/masks_800/a.png", contour_path="model/contours_n512/a.npz"),
        fragment_b=dict(fragment_token=pair_id + "b", split_unit_id=split + "-b",
            model_mask_path="model/masks_800/b.png", contour_path="model/contours_n512/b.npz"),
        correspondence_path="targets/pairs/p.npz" if label else None)


@pytest.fixture
def materialized(tmp_path):
    entries = []
    for label, name in ((True, "train-p"), (False, "train-n")):
        value = sample(name, label)
        path = tmp_path / (name + ".npz")
        save_sample(path, value, dict(changed_pair=True, pose_supervision_enabled=False))
        entries.append(dict(pair_id=name, label=label, artifact_path=path.name,
            source_row=source_row(name, label, "train")))
    manifest = tmp_path / "materialized.json"
    manifest.write_text(json.dumps(dict(schema_version=SCHEMA, split="train",
        artifact_root=str(tmp_path), entries=entries)))
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "val.jsonl").write_text("".join(json.dumps(source_row(name, label, "val")) + "\n"
        for label, name in ((True, "val-p"), (False, "val-n"))))
    # Override paths must not read original TRAIN, TEST or discover REAL.
    (tmp_path / "pairs" / "train.jsonl").write_text("not json\n")
    (tmp_path / "pairs" / "test.jsonl").write_text("not json\n")
    return tmp_path, manifest


def test_both_audits_use_exact_materialized_manifest_and_original_clean_val(materialized):
    root, manifest = materialized
    expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    p = pairing.audit_train_val_population(root, require_formal_counts=False,
        train_materialized_manifest=manifest)
    s, train, val = shredding.audit_rachel_train_val(root, require_formal_counts=False,
        train_materialized_manifest=manifest)
    assert p["manifests"]["train_sha256"] == s["manifest_sha256"]["train"] == expected_hash
    assert p["population"]["train"]["rows"] == s["counts"]["train"]["rows"] == 2
    assert [row.pair_id for row in train] == ["train-p", "train-n"]
    assert [row.pair_id for row in val] == ["val-p", "val-n"]
    assert p["parent_lineage_disjoint"] and s["lineage_disjoint"]
    dataset = hook.materialized_train_dataset(manifest)
    actual = dataset[0]
    expected = sample()
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("problem", ["unbalanced", "duplicate", "label", "heldout", "lineage"])
def test_materialized_override_rejects_invalid_population(materialized, problem):
    root, manifest = materialized
    record = json.loads(manifest.read_text())
    if problem == "unbalanced":
        record["entries"].pop()
    elif problem == "duplicate":
        record["entries"][1] = record["entries"][0]
    elif problem == "label":
        record["entries"][0]["label"] = False
    elif problem == "heldout":
        record["entries"][0]["source_row"]["split"] = "test"
    else:
        record["entries"][0]["source_row"]["fragment_a"]["split_unit_id"] = "val-a"
    manifest.write_text(json.dumps(record))
    for audit in (pairing.audit_train_val_population, shredding.audit_rachel_train_val):
        with pytest.raises((ValueError, RuntimeError)):
            audit(root, require_formal_counts=False, train_materialized_manifest=manifest)


def test_compaction_preserves_surviving_order_original_translation_and_ignore_targets():
    original = sample()
    before = {key: value.copy() for key, value in vars(original).items() if isinstance(value, np.ndarray)}
    for value in before:
        getattr(original, value).flags.writeable = False
    compact = hook.compact_benchmark_sample(original)
    np.testing.assert_array_equal(compact.points_rc_a, original.points_rc_a[[0, 2, 3, 5]])
    np.testing.assert_array_equal(compact.points_rc_b, original.points_rc_b[[0, 1, 3, 5]])
    np.testing.assert_array_equal(compact.target_a, [1, 2, -2, -1])
    np.testing.assert_array_equal(compact.target_b, [-1, 0, 1, -2])
    assert compact.mask_a is original.mask_a
    assert compact.translation_a_to_b_rc is original.translation_a_to_b_rc
    assert compact.label == original.label and compact.pair_id == original.pair_id
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(original, key), value)
    for collate in (pairing.collate_rachel_pairs, shredding.collate_rachel_pairs):
        batch = collate([original])
        lengths = batch.contour_valid_a.sum(1)
        np.testing.assert_array_equal(batch.contour_valid_a, np.arange(512)[None, :] < lengths[:, None])
        pairing._batch_tensors(batch, torch.device("cpu"))
        shredding._valid_prefix_lengths(torch.as_tensor(batch.contour_valid_a))


def test_clean_samples_keep_exact_old_collation():
    original = sample(gaps=False)
    assert hook.compact_benchmark_sample(original) is original
    actual = hook.collate_benchmark_pairs([original], contour_cap=8)
    expected = collate_rachel_pairs([original], contour_cap=8)
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_pairingnet_clean_focal_equals_original_formula_and_ignores_have_zero_gradient():
    p = torch.linspace(.1, .9, 36).reshape(1, 6, 6).requires_grad_()
    target = torch.tensor([[0, -1, 2, -1, 4, -1]])
    valid = torch.ones((1, 6), dtype=torch.bool)
    truth = torch.zeros_like(p, dtype=torch.bool)
    truth[0, [0, 2, 4], [0, 2, 4]] = True
    positive, negative = p[truth], p[~truth]
    old = 400 * ((-.55 * (1 - positive) ** 8 * torch.log(positive)).sum()
        + (-.45 * negative ** 8 * torch.log(1 - negative)).sum()) / p.numel()
    actual = pairing.pairingnet_focal_loss(p, target, valid, valid, target_b=target)
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    ignored = target.clone()
    ignored[0, [1, 3, 5]] = -2
    loss = pairing.pairingnet_focal_loss(p, ignored, valid, valid, target_b=ignored)
    loss.backward()
    assert torch.count_nonzero(p.grad[:, [1, 3, 5]]) == 0
    assert torch.count_nonzero(p.grad[:, :, [1, 3, 5]]) == 0
    assert torch.count_nonzero(p.grad[:, [0, 2, 4]]) > 0


def test_shreddingnet_matching_ignores_zero_gradient_and_correct_effective_counts(monkeypatch):
    row = sample(gaps=False)
    target = np.asarray([0, -2, 2, -2, 4, -2])
    batch = collate_rachel_pairs([replace(row, target_a=target, target_b=target)], contour_cap=6)
    p = torch.linspace(.1, .9, 36).reshape(1, 6, 6).requires_grad_()
    monkeypatch.setattr(shredding, "_matching_forward", lambda *args: p)
    loss, metrics = shredding._matching_batch(None, batch, torch.device("cpu"), shredding.ReleaseRecipe())
    loss.backward()
    assert torch.count_nonzero(p.grad[:, [1, 3, 5]]) == 0
    assert torch.count_nonzero(p.grad[:, :, [1, 3, 5]]) == 0
    assert shredding._matching_cell_counts(batch) == (9, 3)
    assert metrics["positive_loss"] > 0


def test_shreddingnet_clean_focal_remains_exact_original_formula():
    p = torch.linspace(.1, .9, 36).reshape(1, 6, 6)
    batch = collate_rachel_pairs([sample(gaps=False)], contour_cap=6)
    truth = shredding.correspondence_target_matrix(batch, torch.device("cpu"))
    valid = torch.ones_like(truth)
    actual, positive_mean = shredding.released_focal_correspondence_loss(p, truth, valid, alpha=.55, gamma=8.)
    positive = -.55 * (1 - p[truth]) ** 8 * torch.log(p[truth] + 1e-9)
    negative = -.45 * p[~truth] ** 8 * torch.log(1 - p[~truth] + 1e-9)
    torch.testing.assert_close(actual, 400 * torch.cat((positive, negative)).mean(), rtol=0, atol=0)
    torch.testing.assert_close(positive_mean, positive.mean(), rtol=0, atol=0)


def test_experimental_budget_and_selection_require_explicit_opt_in(tmp_path):
    base = dict(dataset_root=tmp_path, output_root=tmp_path / "new", official_source_root=tmp_path)
    with pytest.raises(ValueError, match="seed"):
        pairing.PairingNetRachelRunConfig(**base, seed=17)
    with pytest.raises(ValueError, match="experimental"):
        pairing.PairingNetRachelRunConfig(**base, train_materialized_manifest=tmp_path / "train.json")
    config = pairing.PairingNetRachelRunConfig(**base, seed=17, min_epochs=24, max_epochs=24,
        experimental_data=True, train_materialized_manifest=tmp_path / "train.json",
        selection_metric="recall95_precision")
    pairing._validate_formal_protocol(config, pairing.PairingNetRachelModelConfig())
    with pytest.raises(pairing.PairingNetBenchmarkError, match="hyperparameters"):
        pairing._validate_formal_protocol(replace(config, learning_rate=.02), pairing.PairingNetRachelModelConfig())
    recipe = shredding.ReleaseRecipe(experimental_data=True, seed=17,
        coarse_epochs=24, matching_epochs=24, classify_epochs=24, selection_metric="recall95_precision")
    shredding._require_executable_release_recipe(recipe)
    with pytest.raises(shredding.BenchmarkContractError, match="recipe"):
        shredding._require_executable_release_recipe(replace(recipe, focal_gamma=3.))
    assert shredding._stage_recipe("matching", recipe)["selection_metric"] == "positive_loss"
    assert shredding._stage_recipe("coarse", recipe)["selection_metric"] == "infor_loss"
    assert shredding._stage_recipe("classify", recipe)["selection_metric"] == "recall95_precision"


def test_recall_pairing_selection_does_not_read_pose_and_uses_ap_tie_break():
    first = {"recall_operating_points": {"selection_key": [.8, .9]}}
    second = {"recall_operating_points": {"selection_key": [.8, .91]}}
    assert pairing.selection_key(first, 3, "recall95_precision") == (.8, .9, -3.)
    assert pairing.selection_key(second, 4, "recall95_precision") > pairing.selection_key(first, 3, "recall95_precision")


def test_recall_pairing_checkpoint_preserves_selection_and_loads_for_inference(tmp_path):
    config = pairing.PairingNetRachelRunConfig(dataset_root=tmp_path, output_root=tmp_path / "new",
        official_source_root=tmp_path, train_materialized_manifest=tmp_path / "materialized.json",
        experimental_data=True, min_epochs=24, max_epochs=24, selection_metric="recall95_precision")
    model = pairing.PairingNetRachelAdapted()
    source_sha, document_sha = pairing._adapter_identity_hashes()
    payload = pairing._checkpoint_payload(model=model, epoch=3,
        report={"recall_operating_points": {"selection_key": [.8, .91]}}, run_config=config,
        population_audit={"manifests": {"train_sha256": "a" * 64, "val_sha256": "b" * 64}},
        official_source_audit={}, adapter_source_sha256=source_sha,
        adaptation_contract_sha256=document_sha)
    assert payload["selection_key"] == [.8, .91, -3.]
    assert payload["run_config"]["selection_metric"] == "recall95_precision"
    path = tmp_path / "winner.pt"
    torch.save(payload, path)
    loaded = pairing.load_frozen_pairingnet_checkpoint(path, torch.device("cpu"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_shredding_recall_operating_point_and_ap_tie_break():
    metrics = shredding._recall_validation_metrics([True, True, False, False], [.9, .7, .8, .1])
    assert metrics["recall95_precision"] == pytest.approx(2 / 3)
    assert metrics["recall95_recall"] == 1.
    assert metrics["recall95_threshold"] == .7
    assert metrics["max_f1"] == pytest.approx(.8)
    assert shredding._recall_selection_improved(metrics, -float("inf"), -1, [])
    history = [{"val": {**metrics, "recall95_auprc": metrics["recall95_auprc"] - .01}}]
    assert shredding._recall_selection_improved(metrics, metrics["recall95_precision"], 0, history)
    assert not shredding._recall_selection_improved(metrics, metrics["recall95_precision"], 0, [{"val": metrics}])


def test_cli_passes_materialized_manifest_seed_budgets_and_recall_selection(tmp_path):
    p = pairing._parse_arguments(["--dataset-root", str(tmp_path), "--official-source-root", str(tmp_path),
        "--train-materialized-manifest", str(tmp_path / "train.json"), "--experimental-data",
        "--seed", "17", "--min-epochs", "24", "--max-epochs", "24", "--selection-metric", "recall95_precision"])
    assert p.experimental_data and p.seed == 17 and p.min_epochs == p.max_epochs == 24
    s = shredding._build_parser().parse_args(["train", "--dataset-root", str(tmp_path),
        "--official-repo", str(tmp_path), "--output-root", str(tmp_path / "new"),
        "--train-materialized-manifest", str(tmp_path / "train.json"), "--experimental-data",
        "--seed", "17", "--coarse-epochs", "24", "--matching-epochs", "24", "--classify-epochs", "24",
        "--selection-metric", "recall95_precision"])
    assert s.experimental_data and s.seed == 17 and s.coarse_epochs == s.matching_epochs == s.classify_epochs == 24
    assert p.selection_metric == s.selection_metric == "recall95_precision"
