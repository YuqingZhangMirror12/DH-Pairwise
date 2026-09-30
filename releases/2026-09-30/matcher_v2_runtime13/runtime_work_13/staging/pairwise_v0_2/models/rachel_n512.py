"""Full-contour N=512 Pairwise matcher for the Rachel mask release.

The model consumes only centred binary masks and ordered contour coordinates.
It never receives adjacency labels, correspondence targets, parent-canvas
origins, or translation targets.  Two mask-only sliding windows are sampled at
every contour token, fused into one spatial token, and matched through a
dustbin-augmented partial Sinkhorn transport.  There is no cardinal-side gate
and no rotation head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .coarse import CoarseOutput, SymmetricCoarseSiamese
from .local_matcher import SharedPatchEncoder
from .optimal_transport import PartialTransportOutput, dustbin_sinkhorn


@dataclass(frozen=True)
class RachelN512Config:
    """Architecture constants for the first full-contour Rachel experiment."""

    canvas_size: int = 800
    coarse_size: int = 128
    contour_cap: int = 512
    window_sizes_px: Tuple[float, ...] = (32.0, 64.0)
    patch_size: int = 16
    feature_dim: int = 96
    num_heads: int = 4
    landmark_count: int = 32
    context_layers: int = 2
    evidence_dim: int = 24
    matcher_temperature: float = 0.25
    sinkhorn_iterations: int = 100
    sinkhorn_tolerance: float = 1e-3
    translation_consensus_scale_px: float = 8.0
    translation_consensus_iterations: int = 2
    activation_checkpointing: bool = True
    # The release loader and CUDA preflight already perform exhaustive value
    # checks.  Formal training can disable repeated GPU reductions while
    # retaining all static shape/device/dtype checks.
    validate_runtime_inputs: bool = True

    def __post_init__(self) -> None:
        integer_names = (
            "canvas_size",
            "coarse_size",
            "contour_cap",
            "patch_size",
            "feature_dim",
            "num_heads",
            "landmark_count",
            "context_layers",
            "evidence_dim",
            "sinkhorn_iterations",
            "translation_consensus_iterations",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:  # noqa: E721
                raise ValueError(name + " must be a positive integer")
        if self.feature_dim % self.num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        if self.patch_size < 8:
            raise ValueError("patch_size must support the shared CNN downsampling")
        if not self.window_sizes_px or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.window_sizes_px
        ):
            raise ValueError("window_sizes_px must be finite and positive")
        if (
            not math.isfinite(self.matcher_temperature)
            or self.matcher_temperature <= 0.0
        ):
            raise ValueError("matcher_temperature must be finite and positive")
        if not math.isfinite(self.sinkhorn_tolerance) or self.sinkhorn_tolerance < 0.0:
            raise ValueError("sinkhorn_tolerance must be finite and non-negative")
        if (
            not math.isfinite(self.translation_consensus_scale_px)
            or self.translation_consensus_scale_px <= 0.0
        ):
            raise ValueError(
                "translation_consensus_scale_px must be finite and positive"
            )
        if type(self.activation_checkpointing) is not bool:  # noqa: E721
            raise TypeError("activation_checkpointing must be bool")
        if type(self.validate_runtime_inputs) is not bool:  # noqa: E721
            raise TypeError("validate_runtime_inputs must be bool")


@dataclass(frozen=True)
class RachelN512Output:
    """Prediction and transport evidence; all fields are target-independent."""

    fused_logit: Tensor
    fused_probability: Tensor
    coarse_logit: Tensor
    coarse_probability: Tensor
    local_logit: Tensor
    local_probability: Tensor
    affinity: Tensor
    assignment: Tensor
    unmatched_a: Tensor
    unmatched_b: Tensor
    translation_hat_rc: Tensor
    translation_hat_xy_cartesian: Tensor
    translation_dispersion_px: Tensor
    matched_mass: Tensor
    token_features_a: Tensor
    token_features_b: Tensor
    coarse: CoarseOutput
    transport: PartialTransportOutput
    training_valid: Tensor
    decision_valid: Tensor


def _validate_binary_masks(
    first: Tensor,
    second: Tensor,
    canvas_size: int,
    *,
    validate_values: bool,
) -> Tuple[Tensor, Tensor]:
    if not isinstance(first, Tensor) or not isinstance(second, Tensor):
        raise TypeError("Rachel masks must be torch tensors")
    expected_tail = (1, canvas_size, canvas_size)
    if first.ndim != 4 or second.ndim != 4:
        raise ValueError("Rachel masks must have shape [B,1,H,W]")
    if tuple(first.shape[1:]) != expected_tail or tuple(second.shape) != tuple(
        first.shape
    ):
        raise ValueError("Rachel mask shape differs from the frozen canvas")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("Rachel masks must share device and dtype")
    if not first.is_floating_point():
        raise TypeError("Rachel masks must be floating-point tensors")
    if validate_values:
        for value in (first, second):
            if not torch.isfinite(value).all().item():
                raise ValueError("Rachel masks must be finite")
            if not ((value == 0.0) | (value == 1.0)).all().item():
                raise ValueError("Rachel model input must remain exactly binary")
    return first, second


def _validate_contours(
    points: Tensor,
    valid: Tensor,
    *,
    batch_size: int,
    canvas_size: int,
    contour_cap: int,
    name: str,
    validate_values: bool,
) -> Tuple[Tensor, Tensor]:
    if not isinstance(points, Tensor) or not points.is_floating_point():
        raise TypeError(name + " points must be a floating-point tensor")
    if points.ndim != 3 or points.shape[0] != batch_size or points.shape[2] != 2:
        raise ValueError(name + " points must have shape [B,N,2]")
    if points.shape[1] < 4 or points.shape[1] > contour_cap:
        raise ValueError(name + " token count is outside the frozen cap")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        raise TypeError(name + " valid mask must be bool")
    if tuple(valid.shape) != tuple(points.shape[:2]):
        raise ValueError(name + " valid mask shape differs from points")
    if points.device != valid.device:
        raise ValueError(name + " points and valid mask must share device")
    if validate_values:
        if (valid.sum(dim=1) < 4).any().item():
            raise ValueError(name + " requires at least four valid contour tokens")
        selected = points[valid]
        if not torch.isfinite(selected).all().item():
            raise ValueError(name + " valid contour coordinates must be finite")
        if ((selected < 0.0) | (selected > float(canvas_size - 1))).any().item():
            raise ValueError(name + " valid contour point is outside model canvas")
    return points, valid


class ContourPatchSampler(nn.Module):
    """Vectorized mask-only sliding windows around all ordered contour tokens."""

    def __init__(self, config: RachelN512Config) -> None:
        super().__init__()
        self.canvas_size = config.canvas_size
        self.patch_size = config.patch_size
        self.window_sizes_px = tuple(float(value) for value in config.window_sizes_px)
        axes = []
        for size in self.window_sizes_px:
            half = 0.5 * (size - 1.0)
            axis = torch.linspace(-half, half, config.patch_size)
            rows, columns = torch.meshgrid(axis, axis, indexing="ij")
            axes.append(torch.stack((rows, columns), dim=-1))
        self.register_buffer(
            "offsets_rc",
            torch.stack(axes, dim=0),
            persistent=True,
        )

    @property
    def scale_count(self) -> int:
        return len(self.window_sizes_px)

    def forward(self, masks: Tensor, points_rc: Tensor, valid: Tensor) -> Tensor:
        batch, token_count = points_rc.shape[:2]
        # Mask pixels and contour coordinates are inputs, not learnable state;
        # omitting their sampling graph materially reduces N=512 memory.
        with torch.no_grad():
            centers = points_rc.to(dtype=torch.float32)[:, :, None, None, None, :]
            offsets = self.offsets_rc.to(device=points_rc.device, dtype=torch.float32)[
                None, None
            ]
            sample_rc = centers + offsets
            denominator = float(self.canvas_size - 1)
            grid_x = sample_rc[..., 1] * (2.0 / denominator) - 1.0
            grid_y = sample_rc[..., 0] * (2.0 / denominator) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
                batch,
                token_count * self.scale_count * self.patch_size,
                self.patch_size,
                2,
            )
            sampled = F.grid_sample(
                masks.to(dtype=torch.float32),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.reshape(
                batch,
                1,
                token_count,
                self.scale_count,
                self.patch_size,
                self.patch_size,
            ).permute(0, 2, 3, 1, 4, 5)
            sampled = sampled * valid[:, :, None, None, None, None]
        return sampled


class _DeterministicSpatialMean(nn.Module):
    """AdaptiveAvgPool2d(1) equivalent with deterministic CUDA backward."""

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("spatial mean expects [B,C,H,W]")
        return value.mean(dim=(-2, -1), keepdim=True)


class CyclicLandmarkContext(nn.Module):
    """Ordered circular context plus symmetric O(NK) cross interaction."""

    def __init__(self, config: RachelN512Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        config.feature_dim,
                        config.feature_dim,
                        kernel_size=5,
                        padding=2,
                        padding_mode="circular",
                        bias=False,
                    ),
                    nn.GroupNorm(1, config.feature_dim),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(
                        config.feature_dim,
                        config.feature_dim,
                        kernel_size=3,
                        padding=1,
                        padding_mode="circular",
                        bias=False,
                    ),
                    nn.GroupNorm(1, config.feature_dim),
                )
                for _ in range(config.context_layers)
            ]
        )
        self.position = nn.Linear(2, config.feature_dim, bias=False)
        self.landmark_count = config.landmark_count
        self.cross_norm = nn.LayerNorm(config.feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.feature_dim,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim * 2),
            nn.GELU(),
            nn.Linear(config.feature_dim * 2, config.feature_dim),
        )

    def _within(self, value: Tensor, valid: Tensor, normalized_rc: Tensor) -> Tensor:
        result = value + self.position(normalized_rc)
        result = torch.where(valid[:, :, None], result, torch.zeros_like(result))
        for block in self.blocks:
            update = block(result.transpose(1, 2)).transpose(1, 2)
            result = result + update
            result = torch.where(valid[:, :, None], result, torch.zeros_like(result))
        return result

    def _landmarks(self, value: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        count = min(self.landmark_count, value.shape[1])
        # Explicit deterministic contiguous bins.  CUDA implements adaptive
        # average pooling through a backward kernel that strict deterministic
        # mode rejects; this matrix reduction has the same ordered-landmark
        # semantics and a deterministic CuBLAS path.
        token_count = value.shape[1]
        bins = torch.div(
            torch.arange(token_count, device=value.device) * count,
            token_count,
            rounding_mode="floor",
        )
        membership = F.one_hot(bins, num_classes=count).transpose(0, 1)
        membership = membership.to(dtype=value.dtype)
        weighted_membership = membership[None] * valid[:, None].to(value.dtype)
        mass = weighted_membership.sum(dim=2, keepdim=True)
        pooled = torch.matmul(weighted_membership, value) / mass.clamp_min(1e-6)
        landmark_valid = mass.squeeze(2) > 0.0
        return pooled, landmark_valid

    def _cross(
        self,
        query: Tensor,
        query_valid: Tensor,
        source: Tensor,
        source_valid: Tensor,
    ) -> Tensor:
        landmarks, landmark_valid = self._landmarks(source, source_valid)
        update, _ = self.cross_attention(
            self.cross_norm(query),
            self.cross_norm(landmarks),
            self.cross_norm(landmarks),
            key_padding_mask=~landmark_valid,
            need_weights=False,
        )
        result = query + update
        result = result + self.feed_forward(self.output_norm(result))
        return torch.where(query_valid[:, :, None], result, torch.zeros_like(result))

    def forward(
        self,
        first: Tensor,
        second: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        points_a_rc: Tensor,
        points_b_rc: Tensor,
        canvas_size: int,
    ) -> Tuple[Tensor, Tensor]:
        scale = 2.0 / float(canvas_size - 1)
        normalized_a = points_a_rc * scale - 1.0
        normalized_b = points_b_rc * scale - 1.0
        within_a = self._within(first, valid_a, normalized_a)
        within_b = self._within(second, valid_b, normalized_b)
        return (
            self._cross(within_a, valid_a, within_b, valid_b),
            self._cross(within_b, valid_b, within_a, valid_a),
        )


class TransportSequenceHead(nn.Module):
    """Retain ordered local support instead of collapsing transport to 5 scalars."""

    def __init__(self, config: RachelN512Config) -> None:
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(
                3,
                config.evidence_dim,
                kernel_size=7,
                padding=3,
                padding_mode="circular",
            ),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                config.evidence_dim,
                config.evidence_dim,
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(inplace=True),
        )
        symmetric_dim = config.evidence_dim * 4
        self.head = nn.Sequential(
            nn.Linear(symmetric_dim + 8, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _pool(sequence: Tensor, valid: Tensor) -> Tensor:
        mask = valid[:, None].to(sequence.dtype)
        masked = sequence * mask
        mean = masked.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        maximum = torch.where(
            valid[:, None], sequence, torch.full_like(sequence, -1e4)
        ).amax(dim=2)
        return 0.5 * (mean + maximum)

    @staticmethod
    def _continuity(value: Tensor, valid: Tensor) -> Tensor:
        pair_valid = valid & torch.roll(valid, shifts=-1, dims=1)
        numerator = (value * torch.roll(value, shifts=-1, dims=1) * pair_valid).sum(
            dim=1
        )
        denominator = (value.square() * valid).sum(dim=1).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        affinity: Tensor,
        assignment: Tensor,
        unmatched_a: Tensor,
        unmatched_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        dispersion_px: Tensor,
        canvas_size: int,
        *,
        weighted_affinity_numerator: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Read transport evidence, optionally overriding only Q-times-affinity.

        The override is an unnormalized [B,Na,Nb] product numerator, not an
        alternative assignment.  Mass denominators and all other evidence stay
        unchanged.  Omitting it retains the original mixed-product operations.
        """
        if weighted_affinity_numerator is not None:
            if not isinstance(weighted_affinity_numerator, Tensor):
                raise TypeError("weighted_affinity_numerator must be a Tensor")
            if weighted_affinity_numerator.shape != assignment.shape:
                raise ValueError("weighted_affinity_numerator must match assignment shape [B,Na,Nb]")
            if weighted_affinity_numerator.device != assignment.device:
                raise ValueError("weighted_affinity_numerator must be on the assignment device")
        row_mass = assignment.sum(dim=2)
        col_mass = assignment.sum(dim=1)
        if weighted_affinity_numerator is None:
            row_affinity = (assignment * affinity).sum(dim=2) / row_mass.clamp_min(1e-6)
            col_affinity = (assignment * affinity).sum(dim=1) / col_mass.clamp_min(1e-6)
        else:
            row_affinity = weighted_affinity_numerator.sum(dim=2) / row_mass.clamp_min(1e-6)
            col_affinity = weighted_affinity_numerator.sum(dim=1) / col_mass.clamp_min(1e-6)
        seq_a = torch.stack((row_mass, unmatched_a, row_affinity), dim=1)
        seq_b = torch.stack((col_mass, unmatched_b, col_affinity), dim=1)
        encoded_a = self.sequence(seq_a) * valid_a[:, None]
        encoded_b = self.sequence(seq_b) * valid_b[:, None]
        pooled_a = self._pool(encoded_a, valid_a)
        pooled_b = self._pool(encoded_b, valid_b)
        symmetric = torch.cat(
            (
                torch.abs(pooled_a - pooled_b),
                pooled_a * pooled_b,
                0.5 * (pooled_a + pooled_b),
                torch.maximum(pooled_a, pooled_b),
            ),
            dim=1,
        )
        count_a = valid_a.sum(dim=1).to(affinity.dtype).clamp_min(1.0)
        count_b = valid_b.sum(dim=1).to(affinity.dtype).clamp_min(1.0)
        mass = assignment.sum(dim=(1, 2))
        coverage_a = mass / count_a
        coverage_b = mass / count_b
        weighted_affinity = (
            assignment * affinity if weighted_affinity_numerator is None else weighted_affinity_numerator
        ).sum(dim=(1, 2)) / mass.clamp_min(
            1e-6
        )
        best_affinity = affinity.flatten(1).amax(dim=1)
        continuity_a = self._continuity(row_mass, valid_a)
        continuity_b = self._continuity(col_mass, valid_b)
        global_evidence = torch.stack(
            (
                mass / torch.minimum(count_a, count_b),
                0.5 * (coverage_a + coverage_b),
                torch.abs(coverage_a - coverage_b),
                weighted_affinity,
                best_affinity,
                0.5 * (continuity_a + continuity_b),
                torch.abs(continuity_a - continuity_b),
                dispersion_px / float(canvas_size),
            ),
            dim=1,
        )
        return self.head(torch.cat((symmetric, global_evidence), dim=1)).squeeze(
            1
        ), global_evidence


class RachelN512Pairwise(nn.Module):
    """Coarse Siamese + full-contour mask-patch partial-OT matcher."""

    def __init__(self, config: Optional[RachelN512Config] = None) -> None:
        super().__init__()
        self.config = config or RachelN512Config()
        self.coarse = SymmetricCoarseSiamese(input_channels=1)
        # PyTorch 2.5 intentionally rejects AdaptiveAvgPool2d backward under
        # strict deterministic CUDA mode.  A 1x1 adaptive average is exactly a
        # spatial mean, whose reduction path is deterministic on the target
        # runtime.  Replace it only inside this new model instance.
        self.coarse.projection[0] = _DeterministicSpatialMean()
        self.patch_sampler = ContourPatchSampler(self.config)
        self.patch_encoder = SharedPatchEncoder(1, self.config.feature_dim)
        self.patch_encoder.projection[0] = _DeterministicSpatialMean()
        self.scale_gate = nn.Linear(self.config.feature_dim, 1)
        self.context = CyclicLandmarkContext(self.config)
        self.primal = nn.Linear(
            self.config.feature_dim, self.config.feature_dim, bias=False
        )
        self.dual = nn.Linear(
            self.config.feature_dim, self.config.feature_dim, bias=False
        )
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))
        self.local_head = TransportSequenceHead(self.config)
        self.fusion = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
        )

    def _encode_patches(self, patches: Tensor, valid: Tensor) -> Tensor:
        batch, tokens, scales = patches.shape[:3]
        flat = patches.reshape(batch, tokens * scales, *patches.shape[3:])
        flat_valid = valid[:, :, None].expand(-1, -1, scales).reshape(batch, -1)
        use_checkpoint = (
            self.config.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if use_checkpoint:
            encoded = activation_checkpoint(
                self.patch_encoder,
                flat,
                flat_valid,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            encoded = self.patch_encoder(flat, flat_valid)
        encoded = encoded.reshape(batch, tokens, scales, -1)
        gates = self.scale_gate(encoded).squeeze(3)
        gates = torch.softmax(gates, dim=2)
        fused = (encoded * gates[:, :, :, None]).sum(dim=2)
        return torch.where(valid[:, :, None], fused, torch.zeros_like(fused))

    def _translation(
        self,
        assignment: Tensor,
        points_a_rc: Tensor,
        points_b_rc: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # PairingNet/ShreddingNet use RANSAC after correspondence prediction.
        # Orientation is frozen here, so the analogous transform family is a
        # 2-D translation.  This differentiable Cauchy consensus suppresses
        # edges that do not support one common displacement; final inference
        # may still use a hard weighted-median/mode estimator.
        displacement = points_b_rc[:, None, :, :] - points_a_rc[:, :, None, :]
        original_mass = assignment.sum(dim=(1, 2))
        weights = assignment
        consensus_mass = weights.sum(dim=(1, 2)).clamp_min(1e-6)
        translation = (weights[:, :, :, None] * displacement).sum(dim=(1, 2))
        translation = translation / consensus_mass[:, None]
        for _ in range(self.config.translation_consensus_iterations):
            residual = torch.linalg.vector_norm(
                displacement - translation[:, None, None, :], dim=3
            )
            robust = 1.0 / (
                1.0 + (residual / self.config.translation_consensus_scale_px).square()
            )
            weights = assignment * robust
            consensus_mass = weights.sum(dim=(1, 2)).clamp_min(1e-6)
            translation = (weights[:, :, :, None] * displacement).sum(dim=(1, 2))
            translation = translation / consensus_mass[:, None]
        final_residual = torch.linalg.vector_norm(
            displacement - translation[:, None, None, :], dim=3
        )
        # Re-evaluate the robust weights at the final estimate.  Otherwise the
        # reported dispersion lags one IRLS iteration behind the translation.
        final_robust = 1.0 / (
            1.0 + (final_residual / self.config.translation_consensus_scale_px).square()
        )
        final_weights = assignment * final_robust
        final_mass = final_weights.sum(dim=(1, 2)).clamp_min(1e-6)
        dispersion = torch.sqrt(
            (final_weights * final_residual.square()).sum(dim=(1, 2)) / final_mass
            + 1e-8
        )
        valid = (original_mass > 1e-6) & torch.isfinite(translation).all(dim=1)
        translation = torch.where(
            valid[:, None], translation, torch.zeros_like(translation)
        )
        dispersion = torch.where(valid, dispersion, torch.zeros_like(dispersion))
        return translation, dispersion, original_mass

    def forward(
        self,
        mask_a: Tensor,
        mask_b: Tensor,
        points_rc_a: Tensor,
        points_rc_b: Tensor,
        contour_valid_a: Tensor,
        contour_valid_b: Tensor,
    ) -> RachelN512Output:
        mask_a, mask_b = _validate_binary_masks(
            mask_a,
            mask_b,
            self.config.canvas_size,
            validate_values=self.config.validate_runtime_inputs,
        )
        points_rc_a, contour_valid_a = _validate_contours(
            points_rc_a,
            contour_valid_a,
            batch_size=mask_a.shape[0],
            canvas_size=self.config.canvas_size,
            contour_cap=self.config.contour_cap,
            name="A",
            validate_values=self.config.validate_runtime_inputs,
        )
        points_rc_b, contour_valid_b = _validate_contours(
            points_rc_b,
            contour_valid_b,
            batch_size=mask_a.shape[0],
            canvas_size=self.config.canvas_size,
            contour_cap=self.config.contour_cap,
            name="B",
            validate_values=self.config.validate_runtime_inputs,
        )
        coarse_a = F.interpolate(
            mask_a,
            size=(self.config.coarse_size, self.config.coarse_size),
            mode="nearest",
        )
        coarse_b = F.interpolate(
            mask_b,
            size=(self.config.coarse_size, self.config.coarse_size),
            mode="nearest",
        )
        coarse = self.coarse(coarse_a, coarse_b)
        patches_a = self.patch_sampler(mask_a, points_rc_a, contour_valid_a)
        patches_b = self.patch_sampler(mask_b, points_rc_b, contour_valid_b)
        encoded_a = self._encode_patches(patches_a, contour_valid_a)
        encoded_b = self._encode_patches(patches_b, contour_valid_b)
        context_a, context_b = self.context(
            encoded_a,
            encoded_b,
            contour_valid_a,
            contour_valid_b,
            points_rc_a,
            points_rc_b,
            self.config.canvas_size,
        )
        primal_a = F.normalize(self.primal(context_a), dim=2, eps=1e-6)
        primal_b = F.normalize(self.primal(context_b), dim=2, eps=1e-6)
        dual_a = F.normalize(self.dual(context_a), dim=2, eps=1e-6)
        dual_b = F.normalize(self.dual(context_b), dim=2, eps=1e-6)
        affinity = 0.5 * (
            torch.matmul(primal_a, dual_b.transpose(1, 2))
            + torch.matmul(dual_a, primal_b.transpose(1, 2))
        )
        transport = dustbin_sinkhorn(
            affinity,
            contour_valid_a,
            contour_valid_b,
            dustbin_score=self.dustbin_score,
            temperature=self.config.matcher_temperature,
            num_iterations=self.config.sinkhorn_iterations,
            tolerance=self.config.sinkhorn_tolerance,
            checkpoint_iterations=(
                self.config.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ),
        )
        translation, dispersion, mass = self._translation(
            transport.real_transport, points_rc_a, points_rc_b
        )
        local_logit, _ = self.local_head(
            affinity,
            transport.real_transport,
            transport.dustbin_col,
            transport.dustbin_row,
            contour_valid_a,
            contour_valid_b,
            dispersion,
            self.config.canvas_size,
        )
        fusion_input = torch.stack(
            (
                coarse.logit,
                local_logit,
                coarse.logit * local_logit,
                torch.abs(coarse.logit - local_logit),
            ),
            dim=1,
        )
        fused_logit = self.fusion(fusion_input).squeeze(1)
        finite = (
            coarse.valid_problem
            & transport.diagnostics.valid_problem
            & transport.diagnostics.finite_output
            & torch.isfinite(local_logit)
            & torch.isfinite(fused_logit)
        )
        decision_valid = finite & transport.diagnostics.converged
        fused_logit = torch.where(finite, fused_logit, torch.zeros_like(fused_logit))
        local_logit = torch.where(finite, local_logit, torch.zeros_like(local_logit))
        translation_xy = torch.stack((translation[:, 1], -translation[:, 0]), dim=1)
        return RachelN512Output(
            fused_logit=fused_logit,
            fused_probability=torch.sigmoid(fused_logit),
            coarse_logit=coarse.logit,
            coarse_probability=coarse.probability,
            local_logit=local_logit,
            local_probability=torch.sigmoid(local_logit),
            affinity=affinity,
            assignment=transport.real_transport,
            unmatched_a=transport.dustbin_col,
            unmatched_b=transport.dustbin_row,
            translation_hat_rc=translation,
            translation_hat_xy_cartesian=translation_xy,
            translation_dispersion_px=dispersion,
            matched_mass=mass,
            token_features_a=context_a,
            token_features_b=context_b,
            coarse=coarse,
            transport=transport,
            training_valid=finite,
            decision_valid=decision_valid,
        )


__all__ = [
    "ContourPatchSampler",
    "CyclicLandmarkContext",
    "RachelN512Config",
    "RachelN512Output",
    "RachelN512Pairwise",
    "TransportSequenceHead",
]
