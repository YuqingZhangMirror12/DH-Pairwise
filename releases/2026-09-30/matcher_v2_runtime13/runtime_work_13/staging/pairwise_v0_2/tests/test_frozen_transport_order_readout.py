from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from staging.pairwise_v0_2.baselines import frozen_transport_order_readout as runner
from staging.pairwise_v0_1.baselines.b1_contours import CardinalSide
from staging.pairwise_v0_2.geometry.keypoint_candidates import (
    KeypointPairCandidate,
    KeypointScaleToken,
)
from staging.pairwise_v0_2.geometry.schema import PairDirection
from staging.pairwise_v0_2.models.frozen_transport_order_readout import (
    OrderCoordinateSidecar,
    build_order_coordinate_sidecar,
    exact_target_order_concordance,
    transport_pairwise_anti_order_mass,
)


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    output = np.asarray(value, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def _token(
    name: str,
    side: CardinalSide,
    scale: int,
    fraction: float,
    run_index: int = 0,
) -> KeypointScaleToken:
    channels = _readonly(np.zeros((3, 4, 4)), np.float32)
    return KeypointScaleToken(
        token_id=name,
        anchor_id=name,
        side=side,
        run_index=run_index,
        scale_index=scale,
        scale_fraction=(0.08, 0.16)[scale],
        source_patch_index=0,
        source_arc_fraction=fraction,
        source_path_fraction=0.0,
        center_row_col=(0.0, 0.0),
        channels=channels,
    )


def _candidate(*, multi_run: bool = False) -> KeypointPairCandidate:
    # Each tuple is scale-major.  A crosses the canonical contour start while
    # B does not; the geometry-only largest-gap unwrap must recover a common
    # clockwise coordinate system separately at both scales.
    tokens_a = tuple(
        _token(
            "a{}-{}".format(scale, index),
            CardinalSide.RIGHT,
            scale,
            fraction,
            run_index=(index // 2 if multi_run else 0),
        )
        for scale in range(2)
        for index, fraction in enumerate((0.00, 0.05, 0.95))
    )
    tokens_b = tuple(
        _token(
            "b{}-{}".format(scale, index),
            CardinalSide.LEFT,
            scale,
            fraction,
            run_index=((2 - index) // 2 if multi_run else 0),
        )
        for scale in range(2)
        for index, fraction in enumerate((0.30, 0.40, 0.50))
    )
    patches_a = _readonly(np.stack([token.channels for token in tokens_a]), np.float32)
    patches_b = _readonly(np.stack([token.channels for token in tokens_b]), np.float32)
    allowed = _readonly(
        np.asarray(
            [
                [first.scale_index == second.scale_index for second in tokens_b]
                for first in tokens_a
            ],
            dtype=np.bool_,
        ),
        np.bool_,
    )
    return KeypointPairCandidate(
        candidate_id="b_right_of_a:order-fixture",
        direction=PairDirection.B_RIGHT_OF_A,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        patches_a=patches_a,
        patches_b=patches_b,
        correspondence_mask=allowed,
    )


def _tensors(candidate: KeypointPairCandidate):
    sidecar = build_order_coordinate_sidecar(
        (candidate,), padded_length_a=6, padded_length_b=6
    )
    mask = torch.ones((1, 6), dtype=torch.bool)
    edges = torch.from_numpy(candidate.correspondence_mask.copy())[None]
    return sidecar, mask, edges


def test_real_arc_coordinates_unwrap_wrap_and_reset_each_scale() -> None:
    sidecar, _, _ = _tensors(_candidate())
    expected_a = torch.tensor(
        [[0.5, 1.0, 0.0, 0.5, 1.0, 0.0]], dtype=torch.float64
    )
    expected_b = torch.tensor(
        [[0.0, 0.5, 1.0, 0.0, 0.5, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(sidecar.coordinate_a, expected_a)
    torch.testing.assert_close(sidecar.coordinate_b, expected_b)
    assert sidecar.valid_a.all().item()
    assert sidecar.valid_b.all().item()


def test_order_coherent_mass_prefers_fixed_anti_mapping_and_is_swap_invariant() -> None:
    candidate = _candidate()
    sidecar, mask, edges = _tensors(candidate)
    anti = torch.zeros((1, 6, 6), dtype=torch.float64, requires_grad=True)
    anti_mapping = (1, 0, 2, 4, 3, 5)
    with torch.no_grad():
        for index_a, index_b in enumerate(anti_mapping):
            anti[0, index_a, index_b] = 1.0
    coherent = transport_pairwise_anti_order_mass(
        anti, mask, mask, edges, sidecar
    )
    assert coherent.item() == pytest.approx(1.0)

    monotone = torch.zeros_like(anti.detach())
    monotone_mapping = (1, 2, 0, 4, 5, 3)
    for index_a, index_b in enumerate(monotone_mapping):
        monotone[0, index_a, index_b] = 1.0
    corrupted = transport_pairwise_anti_order_mass(
        monotone, mask, mask, edges, sidecar
    )
    assert coherent.item() - corrupted.item() > 0.5

    swapped_sidecar = OrderCoordinateSidecar(
        coordinate_a=sidecar.coordinate_b,
        coordinate_b=sidecar.coordinate_a,
        scale_index_a=sidecar.scale_index_b,
        scale_index_b=sidecar.scale_index_a,
        valid_a=sidecar.valid_b,
        valid_b=sidecar.valid_a,
    )
    swapped = transport_pairwise_anti_order_mass(
        anti.transpose(1, 2),
        mask,
        mask,
        edges.transpose(1, 2),
        swapped_sidecar,
    )
    torch.testing.assert_close(coherent, swapped)
    coherent.sum().backward()
    assert anti.grad is not None
    assert torch.isfinite(anti.grad).all().item()
    assert (anti.grad > 0.0).any().item()


def test_synthetic_exact_target_diagnostic_favors_anti_without_fitting_sign() -> None:
    candidate = _candidate()
    sidecar, mask, edges = _tensors(candidate)
    target = torch.full((1, 6), -1, dtype=torch.long)
    for index_a, index_b in enumerate((1, 0, 2, 4, 3, 5)):
        target[0, index_a] = index_b
    values = exact_target_order_concordance(target, mask, mask, edges, sidecar)
    assert values.candidate_index.tolist() == [0, 0]
    assert values.scale_index.tolist() == [0, 1]
    assert values.matched_edge_count.tolist() == [3, 3]
    assert values.anti_pair_count.tolist() == [3, 3]
    assert values.monotone_pair_count.tolist() == [0, 0]


def test_singleton_scale_blocks_fail_closed_to_zero() -> None:
    sidecar = OrderCoordinateSidecar(
        coordinate_a=torch.zeros((1, 1)),
        coordinate_b=torch.zeros((1, 1)),
        scale_index_a=torch.full((1, 1), -1, dtype=torch.long),
        scale_index_b=torch.full((1, 1), -1, dtype=torch.long),
        valid_a=torch.zeros((1, 1), dtype=torch.bool),
        valid_b=torch.zeros((1, 1), dtype=torch.bool),
    )
    value = transport_pairwise_anti_order_mass(
        torch.ones((1, 1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        sidecar,
    )
    assert value.tolist() == [0.0]


def test_pairwise_order_is_subsequence_and_multirun_invariant() -> None:
    candidate = _candidate(multi_run=True)
    sidecar, mask, edge = _tensors(candidate)
    plan = torch.zeros((1, 6, 6), dtype=torch.float64)
    for index_a, index_b in enumerate((1, 0, 2, 4, 3, 5)):
        plan[0, index_a, index_b] = 1.0
    value = transport_pairwise_anti_order_mass(plan, mask, mask, edge, sidecar)
    assert value.item() == pytest.approx(1.0)


def test_pairwise_order_mass_has_no_factor_two_and_retains_dustbin_coverage() -> None:
    sidecar = OrderCoordinateSidecar(
        coordinate_a=torch.tensor([[0.0, 1.0]]),
        coordinate_b=torch.tensor([[1.0, 0.0]]),
        scale_index_a=torch.zeros((1, 2), dtype=torch.long),
        scale_index_b=torch.zeros((1, 2), dtype=torch.long),
        valid_a=torch.ones((1, 2), dtype=torch.bool),
        valid_b=torch.ones((1, 2), dtype=torch.bool),
    )
    plan = 0.5 * torch.eye(2)[None]
    mask = torch.ones((1, 2), dtype=torch.bool)
    edge = torch.ones((1, 2, 2), dtype=torch.bool)
    value = transport_pairwise_anti_order_mass(plan, mask, mask, edge, sidecar)
    # One unordered anti pair contributes 0.5 * 0.5 once; missing real mass
    # (the remainder routes to dustbin) is intentionally not renormalized out.
    assert value.item() == pytest.approx(0.25)


def test_paired_component_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.asarray([False, True, False, True, False, True], dtype=np.bool_)
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    baseline = np.asarray([0.2, 0.7, 0.4, 0.6, 0.3, 0.8])
    treatment = np.asarray([0.1, 0.9, 0.2, 0.8, 0.4, 0.7])
    first = runner._paired_bootstrap(
        labels=labels,
        clusters=clusters,
        baseline=baseline,
        treatment=treatment,
        replicates=100,
        rng=np.random.default_rng(17),
    )
    second = runner._paired_bootstrap(
        labels=labels,
        clusters=clusters,
        baseline=baseline,
        treatment=treatment,
        replicates=100,
        rng=np.random.default_rng(17),
    )
    assert first == second
    assert first["sampling_cluster_count"] == 3
    assert first["comparison"].startswith("order_coherent_minus")


def test_sign_gate_uses_true_direction_and_reports_missing_direction_unobserved() -> None:
    candidate = _candidate()
    sidecar, mask, edges = _tensors(candidate)
    target = torch.full((1, 6), -1, dtype=torch.long)
    for index_a, index_b in enumerate((1, 0, 2, 4, 3, 5)):
        target[0, index_a] = index_b
    payload = SimpleNamespace(
        exact_assignment_target_a=target,
        token_mask_a=mask,
        token_mask_b=mask,
        correspondence_mask=edges,
        labels=torch.tensor([True]),
        direction_target_valid=torch.tensor([True]),
        direction_target=torch.tensor([1], dtype=torch.long),
        direction_index=torch.tensor([1], dtype=torch.long),
        sample_index=torch.tensor([0], dtype=torch.long),
    )
    accumulator = runner._TargetAccumulator()
    accumulator.add(payload, sidecar)
    result = accumulator.result()
    assert result["by_direction"]["right"]["observed_in_frozen_population"] is True
    assert result["by_direction"]["right"]["anti_pairs"] == 6
    assert result["by_direction"]["above"]["observed_in_frozen_population"] is False
    assert result["by_direction"]["above"]["anti_pairs"] == 0
