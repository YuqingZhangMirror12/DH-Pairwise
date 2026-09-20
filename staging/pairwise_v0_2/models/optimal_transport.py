"""Log-space partial assignment for local contour-patch matching.

The augmented marginals and dustbin construction follow the differentiable
optimal-transport formulation used by SuperGlue (Sarlin et al., CVPR 2020):
for ``m`` valid A tokens and ``n`` valid B tokens, normalized row marginals are
``(1, ..., 1, n) / (m + n)`` and normalized column marginals are
``(1, ..., 1, m) / (m + n)``.  The last row and column are dustbins.  Results
are rescaled by ``m + n`` before being returned, so every real token has unit
target mass and can distribute that mass between a real match and its dustbin.

This is a Pairwise v0.2 primitive for matching two already-oriented manuscript
contour-patch sequences.  It is not the ECCV CO/Gumbel Top-K selector, a
rotation estimator, or a global fragment solver.  It does not make an
adjacency decision by itself.

References:
    - Sarlin et al., "SuperGlue", CVPR 2020, Appendix/Supplement Sec. 3.
      https://arxiv.org/abs/1911.11763
    - Lu et al., "Jigsaw", NeurIPS 2023 (differentiable correspondence
      inspiration; its balanced point-cloud matching is not copied verbatim).
      https://proceedings.neurips.cc/paper_files/paper/2023/file/
      30ae2af8612ac74357363e8ae877d80c-Paper-Conference.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint


Number = Union[int, float]


@dataclass(frozen=True)
class SinkhornDiagnostics:
    """Per-sample health and mass-conservation diagnostics.

    All tensor fields have shape ``[B]``.  ``valid_problem`` is false when
    either sequence has no valid token, an affinity contains NaN/+inf, or the
    scalar dustbin logit is non-finite.  Such samples fail closed: all returned
    transport components are zero and ``iteration_count`` is zero.
    """

    valid_problem: Tensor
    input_is_usable: Tensor
    finite_output: Tensor
    converged: Tensor
    valid_a_count: Tensor
    valid_b_count: Tensor
    iteration_count: Tensor
    row_residual_max: Tensor
    col_residual_max: Tensor
    matched_mass: Tensor
    unmatched_a_mass: Tensor
    unmatched_b_mass: Tensor


@dataclass(frozen=True)
class PartialTransportOutput:
    """Dustbin-augmented transport split into semantically named tensors.

    Attributes:
        real_transport: Real A-to-real B mass, shape ``[B, M, N]``.
        dustbin_row: Dustbin-to-real B mass, shape ``[B, N]``.  This is the
            unmatched mass of B tokens.
        dustbin_col: Real A-to-dustbin mass, shape ``[B, M]``.  This is the
            unmatched mass of A tokens.
        dustbin_corner: Dustbin-to-dustbin mass, shape ``[B]``.
        diagnostics: Per-sample validity, convergence, and mass summaries.
    """

    real_transport: Tensor
    dustbin_row: Tensor
    dustbin_col: Tensor
    dustbin_corner: Tensor
    diagnostics: SinkhornDiagnostics


def _validated_mask(
    mask: Optional[Tensor],
    *,
    batch_size: int,
    token_count: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if mask is None:
        return torch.ones((batch_size, token_count), dtype=torch.bool, device=device)
    if not isinstance(mask, Tensor):
        raise TypeError("{} must be a torch.Tensor or None".format(name))
    if mask.dtype != torch.bool:
        raise TypeError("{} must have dtype torch.bool".format(name))
    if tuple(mask.shape) != (batch_size, token_count):
        raise ValueError(
            "{} must have shape ({}, {}), got {}".format(
                name, batch_size, token_count, tuple(mask.shape)
            )
        )
    return mask.to(device=device)


def _log_sinkhorn_iterations(
    log_scores: Tensor,
    log_row_marginals: Tensor,
    log_col_marginals: Tensor,
    num_iterations: int,
) -> Tensor:
    """Balance an augmented score matrix without leaving log space."""

    row_is_active = torch.isfinite(log_row_marginals)
    col_is_active = torch.isfinite(log_col_marginals)
    u = torch.where(
        row_is_active,
        torch.zeros_like(log_row_marginals),
        torch.full_like(log_row_marginals, -torch.inf),
    )
    v = torch.where(
        col_is_active,
        torch.zeros_like(log_col_marginals),
        torch.full_like(log_col_marginals, -torch.inf),
    )

    for _ in range(num_iterations):
        row_lse = torch.logsumexp(log_scores + v.unsqueeze(1), dim=2)
        row_lse = torch.where(row_is_active, row_lse, torch.zeros_like(row_lse))
        u = torch.where(
            row_is_active,
            log_row_marginals - row_lse,
            torch.full_like(row_lse, -torch.inf),
        )

        col_lse = torch.logsumexp(log_scores + u.unsqueeze(2), dim=1)
        col_lse = torch.where(col_is_active, col_lse, torch.zeros_like(col_lse))
        v = torch.where(
            col_is_active,
            log_col_marginals - col_lse,
            torch.full_like(col_lse, -torch.inf),
        )

    return log_scores + u.unsqueeze(2) + v.unsqueeze(1)


def dustbin_sinkhorn(
    affinity: Tensor,
    mask_a: Optional[Tensor] = None,
    mask_b: Optional[Tensor] = None,
    *,
    dustbin_score: Union[Number, Tensor] = 0.0,
    temperature: float = 1.0,
    num_iterations: int = 100,
    tolerance: float = 1e-3,
    checkpoint_iterations: bool = False,
) -> PartialTransportOutput:
    """Compute a balanced-with-dustbin partial assignment in log space.

    Args:
        affinity: Pair logits with shape ``[B, M, N]``.  NaN and positive
            infinity fail the affected sample closed; negative infinity is a
            supported forbidden correspondence because dustbins retain a
            feasible route.
        mask_a: Optional valid-token mask of shape ``[B, M]``.
        mask_b: Optional valid-token mask of shape ``[B, N]``.
        dustbin_score: Shared scalar logit for every real/dustbin edge and the
            dustbin corner.  It may be a scalar Tensor requiring gradients.
        temperature: Positive finite temperature applied to both real and
            dustbin logits.
        num_iterations: Fixed number of alternating log-space normalizations.
            A fixed loop is deterministic and keeps the training graph intact.
        tolerance: Absolute marginal-residual threshold used only for the
            ``converged`` diagnostic; it does not alter the returned plan.
        checkpoint_iterations: Recompute the fixed Sinkhorn loop during
            backward instead of retaining every iteration's activations.  The
            local matcher enables this only for gradient-bearing training.

    Returns:
        A :class:`PartialTransportOutput`.  For every valid A token,
        ``real_transport.sum(-1) + dustbin_col`` approaches one.  For every
        valid B token, ``real_transport.sum(-2) + dustbin_row`` approaches one.

    Notes:
        Float16/bfloat16 inputs are promoted to float32 for the log-domain
        iterations.  Float32/float64 inputs retain their dtype.  No hard
        threshold or Pairwise adjacency probability is produced here.
    """

    if not isinstance(affinity, Tensor):
        raise TypeError("affinity must be a torch.Tensor")
    if affinity.ndim != 3:
        raise ValueError("affinity must have shape [B, M, N]")
    if not affinity.is_floating_point():
        raise TypeError("affinity must have a floating-point dtype")
    batch_size, count_a, count_b = affinity.shape
    if batch_size < 1 or count_a < 1 or count_b < 1:
        raise ValueError("affinity dimensions B, M, and N must all be positive")

    if isinstance(temperature, bool):
        raise TypeError("temperature must be a positive finite number")
    temperature_value = float(temperature)
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    if isinstance(num_iterations, bool) or not isinstance(num_iterations, int):
        raise TypeError("num_iterations must be a positive integer")
    if num_iterations < 1:
        raise ValueError("num_iterations must be a positive integer")
    if isinstance(tolerance, bool):
        raise TypeError("tolerance must be a non-negative finite number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be a non-negative finite number")
    if type(checkpoint_iterations) is not bool:
        raise TypeError("checkpoint_iterations must be bool")

    valid_a = _validated_mask(
        mask_a,
        batch_size=batch_size,
        token_count=count_a,
        device=affinity.device,
        name="mask_a",
    )
    valid_b = _validated_mask(
        mask_b,
        batch_size=batch_size,
        token_count=count_b,
        device=affinity.device,
        name="mask_b",
    )

    work_dtype = (
        torch.float32
        if affinity.dtype in (torch.float16, torch.bfloat16)
        else affinity.dtype
    )
    scores = affinity.to(dtype=work_dtype)
    bin_score = torch.as_tensor(dustbin_score, dtype=work_dtype, device=affinity.device)
    if bin_score.numel() != 1:
        raise ValueError("dustbin_score must be scalar")
    bin_score = bin_score.reshape(())

    # -inf is an intentional forbidden real correspondence.  NaN and +inf in
    # a valid-real cell invalidate that sample.  Values in padded cells are
    # ignored because their token masks make them outside the stated problem.
    candidate_real = valid_a[:, :, None] & valid_b[:, None, :]
    unusable_real = candidate_real & (torch.isnan(scores) | torch.isposinf(scores))
    input_is_usable = ~(unusable_real.flatten(1).any(dim=1))
    bin_is_usable = torch.isfinite(bin_score)
    count_a_tensor = valid_a.sum(dim=1)
    count_b_tensor = valid_b.sum(dim=1)
    valid_problem = (
        (count_a_tensor > 0) & (count_b_tensor > 0) & input_is_usable & bin_is_usable
    )

    negative_infinity = torch.tensor(
        -torch.inf, dtype=work_dtype, device=affinity.device
    )
    active_real = (
        valid_problem[:, None, None] & valid_a[:, :, None] & valid_b[:, None, :]
    )
    scaled_scores = scores / temperature_value
    real_scores = torch.where(active_real, scaled_scores, negative_infinity)
    scaled_bin = bin_score / temperature_value

    active_a = valid_problem[:, None] & valid_a
    active_b = valid_problem[:, None] & valid_b
    # A zero-marginal padded row still needs one finite *computational* edge;
    # otherwise logsumexp(-inf, ..., -inf) has a NaN derivative even when a
    # later torch.where discards it.  Its dual potential remains -inf, so this
    # dummy edge carries exactly zero transport and cannot affect active
    # marginals.  The same construction is used for padded columns below.
    real_to_bin = torch.where(
        active_a[:, :, None],
        scaled_bin.expand(batch_size, count_a, 1),
        torch.zeros(
            (batch_size, count_a, 1),
            dtype=work_dtype,
            device=affinity.device,
        ),
    )
    bin_to_real = torch.where(
        active_b[:, None, :],
        scaled_bin.expand(batch_size, 1, count_b),
        torch.zeros(
            (batch_size, 1, count_b),
            dtype=work_dtype,
            device=affinity.device,
        ),
    )
    # Invalid samples use a corner-only 1x1 problem internally, then their
    # complete output is zeroed.  This avoids -inf - -inf and NaN propagation.
    corner_scores = torch.where(
        valid_problem,
        scaled_bin.expand(batch_size),
        torch.zeros(batch_size, dtype=work_dtype, device=affinity.device),
    ).reshape(batch_size, 1, 1)
    log_scores = torch.cat(
        [
            torch.cat([real_scores, real_to_bin], dim=2),
            torch.cat([bin_to_real, corner_scores], dim=2),
        ],
        dim=1,
    )

    one = torch.ones(batch_size, dtype=work_dtype, device=affinity.device)
    safe_count_a = torch.where(valid_problem, count_a_tensor.to(work_dtype), one)
    safe_count_b = torch.where(valid_problem, count_b_tensor.to(work_dtype), one)
    log_normalizer = torch.log(safe_count_a + safe_count_b)

    real_row_log_mass = torch.where(
        active_a,
        -log_normalizer[:, None],
        negative_infinity,
    )
    dustbin_row_log_mass = torch.where(
        valid_problem,
        torch.log(safe_count_b) - log_normalizer,
        torch.zeros_like(log_normalizer),
    )[:, None]
    log_row_marginals = torch.cat([real_row_log_mass, dustbin_row_log_mass], dim=1)

    real_col_log_mass = torch.where(
        active_b,
        -log_normalizer[:, None],
        negative_infinity,
    )
    dustbin_col_log_mass = torch.where(
        valid_problem,
        torch.log(safe_count_a) - log_normalizer,
        torch.zeros_like(log_normalizer),
    )[:, None]
    log_col_marginals = torch.cat([real_col_log_mass, dustbin_col_log_mass], dim=1)

    if checkpoint_iterations and torch.is_grad_enabled():
        log_transport = activation_checkpoint(
            lambda scores, rows, columns: _log_sinkhorn_iterations(
                scores,
                rows,
                columns,
                num_iterations,
            ),
            log_scores,
            log_row_marginals,
            log_col_marginals,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        log_transport = _log_sinkhorn_iterations(
            log_scores,
            log_row_marginals,
            log_col_marginals,
            num_iterations,
        )
    # Undo the probability normalization so each real token has target mass 1.
    augmented_transport = torch.exp(log_transport + log_normalizer[:, None, None])
    augmented_transport = torch.where(
        valid_problem[:, None, None],
        augmented_transport,
        torch.zeros_like(augmented_transport),
    )

    real_transport = augmented_transport[:, :count_a, :count_b]
    dustbin_col = augmented_transport[:, :count_a, count_b]
    dustbin_row = augmented_transport[:, count_a, :count_b]
    dustbin_corner = augmented_transport[:, count_a, count_b]

    target_rows = torch.cat(
        [
            active_a.to(work_dtype),
            torch.where(
                valid_problem,
                count_b_tensor.to(work_dtype),
                torch.zeros_like(safe_count_b),
            )[:, None],
        ],
        dim=1,
    )
    target_cols = torch.cat(
        [
            active_b.to(work_dtype),
            torch.where(
                valid_problem,
                count_a_tensor.to(work_dtype),
                torch.zeros_like(safe_count_a),
            )[:, None],
        ],
        dim=1,
    )
    row_residual = (augmented_transport.sum(dim=2) - target_rows).abs().amax(dim=1)
    col_residual = (augmented_transport.sum(dim=1) - target_cols).abs().amax(dim=1)
    finite_output = torch.isfinite(augmented_transport).flatten(1).all(dim=1)
    converged = (
        valid_problem
        & finite_output
        & (row_residual <= tolerance_value)
        & (col_residual <= tolerance_value)
    )
    iteration_count = torch.where(
        valid_problem,
        torch.full_like(count_a_tensor, num_iterations),
        torch.zeros_like(count_a_tensor),
    )

    diagnostics = SinkhornDiagnostics(
        valid_problem=valid_problem,
        input_is_usable=input_is_usable & bin_is_usable,
        finite_output=finite_output,
        converged=converged,
        valid_a_count=count_a_tensor,
        valid_b_count=count_b_tensor,
        iteration_count=iteration_count,
        row_residual_max=row_residual,
        col_residual_max=col_residual,
        matched_mass=real_transport.sum(dim=(1, 2)),
        unmatched_a_mass=dustbin_col.sum(dim=1),
        unmatched_b_mass=dustbin_row.sum(dim=1),
    )
    return PartialTransportOutput(
        real_transport=real_transport,
        dustbin_row=dustbin_row,
        dustbin_col=dustbin_col,
        dustbin_corner=dustbin_corner,
        diagnostics=diagnostics,
    )
