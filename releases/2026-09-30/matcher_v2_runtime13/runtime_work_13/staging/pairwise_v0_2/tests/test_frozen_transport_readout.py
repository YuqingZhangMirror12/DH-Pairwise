from __future__ import annotations

import json

import pytest
import torch

from staging.pairwise_v0_2.baselines import frozen_transport_readout as runner
from staging.pairwise_v0_2.models.frozen_transport_readout import (
    FrozenArcDataset,
    READOUT_BASELINE,
    READOUT_ENTROPY_AWARE,
    READOUT_MASS_ONLY,
    aggregate_frozen_readout,
    fit_frozen_readout,
    transport_quality_features,
)
from staging.pairwise_v0_2.models.pairwise import (
    ArcPoolingConfig,
    aggregate_flat_arc_candidates,
)


def test_entropy_quality_separates_concentrated_real_diffuse_and_dustbin():
    assignment = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    unmatched = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    mask = torch.ones((3, 2), dtype=torch.bool)
    quality = transport_quality_features(
        assignment,
        unmatched,
        unmatched,
        mask,
        mask,
        top_k=2,
    )
    assert quality.mass_only.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert quality.entropy_aware[0].item() == pytest.approx(1.0)
    assert 0.0 < quality.entropy_aware[1].item() < 0.5
    assert quality.entropy_aware[2].item() == pytest.approx(0.0)


def test_frozen_baseline_exactly_reuses_hierarchical_lme_aggregation():
    arc_logits = torch.tensor(
        [0.2, 0.8, -0.3, 0.1, 0.4, -0.7, 0.3], dtype=torch.float64
    )
    arc_valid = torch.tensor([True, True, True, False, True, True, True])
    sample_index = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    direction_index = torch.tensor([0, 0, 1, 3, 0, 2, 2], dtype=torch.long)
    geometry_valid = torch.tensor([True, True])
    dataset = FrozenArcDataset(
        arc_logit=arc_logits,
        mass_only=torch.linspace(0.1, 0.7, 7, dtype=torch.float64),
        entropy_aware=torch.linspace(0.0, 0.6, 7, dtype=torch.float64),
        arc_valid=arc_valid,
        sample_index=sample_index,
        direction_index=direction_index,
        geometry_valid=geometry_valid,
    )
    pooling = ArcPoolingConfig()
    expected = aggregate_flat_arc_candidates(
        arc_logits,
        arc_valid,
        sample_index,
        direction_index,
        batch_size=2,
        arc_pooling=pooling,
        direction_temperature=0.25,
    ).direction_output
    observed = aggregate_frozen_readout(
        dataset,
        readout_name=READOUT_BASELINE,
        arc_pooling=pooling,
        direction_temperature=0.25,
    )
    torch.testing.assert_close(observed.direction_logits, expected.candidate_logits)
    torch.testing.assert_close(observed.direction_valid, expected.candidate_valid)
    torch.testing.assert_close(observed.pair_logit, expected.pair_logit)
    torch.testing.assert_close(observed.pair_probability, expected.pair_probability)
    torch.testing.assert_close(
        observed.best_direction_index, expected.best_direction_index
    )


def test_only_beta_and_intercept_fit_on_synthetic_labels():
    # One valid arc in one direction per pair makes hierarchical aggregation an
    # identity.  Existing logits are neutral; q alone separates the labels.
    label = torch.tensor([False, False, True, True], dtype=torch.bool)
    quality = torch.tensor([0.0, 0.1, 0.9, 1.0], dtype=torch.float64)
    dataset = FrozenArcDataset(
        arc_logit=torch.zeros(4, dtype=torch.float64),
        mass_only=quality,
        entropy_aware=quality,
        arc_valid=torch.ones(4, dtype=torch.bool),
        sample_index=torch.arange(4, dtype=torch.long),
        direction_index=torch.zeros(4, dtype=torch.long),
        geometry_valid=torch.ones(4, dtype=torch.bool),
        label=label,
    )
    baseline = fit_frozen_readout(dataset, readout_name=READOUT_BASELINE)
    fitted = fit_frozen_readout(
        dataset, readout_name=READOUT_ENTROPY_AWARE, max_iterations=32
    )
    assert baseline.beta == 0.0
    assert baseline.intercept == 0.0
    assert fitted.beta > 0.0
    assert fitted.train_loss < baseline.train_loss
    output = aggregate_frozen_readout(
        dataset,
        readout_name=READOUT_ENTROPY_AWARE,
        beta=fitted.beta,
        intercept=fitted.intercept,
    )
    assert torch.all(output.pair_probability[label] > 0.5)
    assert torch.all(output.pair_probability[~label] < 0.5)


def test_mass_and_entropy_readouts_are_distinct_but_shape_matched():
    assignment = torch.tensor([[[0.45, 0.45]]], dtype=torch.float32)
    quality = transport_quality_features(
        assignment,
        torch.tensor([[0.10]]),
        torch.tensor([[0.55, 0.55]]),
        torch.tensor([[True]]),
        torch.tensor([[True, True]]),
    )
    assert quality.mass_only.shape == quality.entropy_aware.shape == (1,)
    assert quality.mass_only.item() > quality.entropy_aware.item()
    dataset = FrozenArcDataset(
        arc_logit=torch.tensor([0.0]),
        mass_only=quality.mass_only,
        entropy_aware=quality.entropy_aware,
        arc_valid=torch.tensor([True]),
        sample_index=torch.tensor([0]),
        direction_index=torch.tensor([0]),
        geometry_valid=torch.tensor([True]),
        label=torch.tensor([True]),
    )
    mass = aggregate_frozen_readout(
        dataset, readout_name=READOUT_MASS_ONLY, beta=1.0
    )
    entropy = aggregate_frozen_readout(
        dataset, readout_name=READOUT_ENTROPY_AWARE, beta=1.0
    )
    assert mass.pair_logit.item() > entropy.pair_logit.item()


def test_transport_quality_is_exactly_ab_swap_invariant():
    assignment = torch.tensor(
        [[[0.6, 0.1, 0.0], [0.0, 0.2, 0.5]]], dtype=torch.float64
    )
    unmatched_a = torch.tensor([[0.3, 0.3]], dtype=torch.float64)
    unmatched_b = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64)
    mask_a = torch.tensor([[True, True]])
    mask_b = torch.tensor([[True, True, True]])
    edges = torch.tensor([[[True, True, False], [False, True, True]]])
    forward = transport_quality_features(
        assignment,
        unmatched_a,
        unmatched_b,
        mask_a,
        mask_b,
        edges,
        top_k=2,
    )
    swapped = transport_quality_features(
        assignment.transpose(1, 2),
        unmatched_b,
        unmatched_a,
        mask_b,
        mask_a,
        edges.transpose(1, 2),
        top_k=2,
    )
    torch.testing.assert_close(forward.mass_only, swapped.mass_only)
    torch.testing.assert_close(forward.entropy_aware, swapped.entropy_aware)


def test_real_evaluator_rejects_fit_that_accessed_real_labels(tmp_path):
    artifact = {
        "schema_version": runner.FROZEN_TRANSPORT_READOUT_VERSION,
        "status": runner.FIT_STATUS,
        "scope": {
            "real_data_opened": False,
            "real_labels_used_for_fit_or_selection": True,
        },
        "selection": {
            "source": "synthetic_validation_only",
            "real_evaluation_accessed": False,
        },
    }
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(runner.FrozenTransportReadoutError, match="synthetic-only"):
        runner._load_fit(path)
