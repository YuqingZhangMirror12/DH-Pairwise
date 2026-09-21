"""Same-capacity zero/mass/mass+spectral residuals around unchanged S6-D2 CA.

No Matcher is constructed or trained here. Caller supplies frozen contextual
tokens and a source-validated cached summary batch. The scalar head is a logit
residual, never a direct conversion of singular values into probabilities.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from staging.pairwise_v0_2.models.rachel_decoupled_score import CrossAttentionPairHead
from .features import (FEATURE_NAMES, FEATURE_SCHEMA, NORMALIZER_SCHEMA, _require_hash,
                       digest_arrays, digest_json, validate_identity)

MODEL_SCHEMA = "rachel-ca-global-spectral-residual/1"
VARIANTS = ("zero", "mass", "mass_spectral")


def module_state_sha256(module):
    return digest_arrays({name: value.detach().cpu().numpy()
                          for name, value in module.state_dict().items()})


class FrozenTrainStandardizer(nn.Module):
    def __init__(self, statistics):
        super().__init__()
        if (statistics.get("schema_version") != NORMALIZER_SCHEMA
                or statistics.get("feature_schema") != FEATURE_SCHEMA
                or statistics.get("feature_names") != list(FEATURE_NAMES)
                or statistics.get("fit_split") != "train"
                or statistics.get("held_out_used_for_fit") is not False
                or type(statistics.get("fitted_count")) is not int or statistics["fitted_count"] < 1):
            raise ValueError("requires registered TRAIN-only 10D normalization statistics")
        validate_identity(statistics["training_identity"])
        if statistics["training_identity"]["split"] != "train":
            raise ValueError("normalization source identity is not TRAIN")
        _require_hash(statistics["training_cache_sha256"], "training_cache_sha256")
        mean, scale = np.asarray(statistics["mean"]), np.asarray(statistics["scale"])
        if (mean.shape != (10,) or scale.shape != (10,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("normalizer mean/scale must be finite 10D with positive scales")
        self.statistics = deepcopy(statistics)
        self.statistics_sha256 = digest_json(statistics)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, values):
        if values.ndim != 2 or values.shape[1] != 10 or not torch.isfinite(values).all():
            raise ValueError("summary batch must have finite fixed [B,10] dimensions")
        if values.requires_grad:
            raise ValueError("summaries must come from frozen precomputation, not a differentiable SVD")
        return (values - self.mean) / self.scale

    def get_extra_state(self):
        return dict(statistics=self.statistics, statistics_sha256=self.statistics_sha256)

    def set_extra_state(self, state):
        if (state.get("statistics_sha256") != self.statistics_sha256
                or digest_json(state.get("statistics")) != self.statistics_sha256):
            raise ValueError("checkpoint TRAIN normalization differs from registered statistics")


def tensor_summary_batch(batch, device, dtype=torch.float32):
    if set(batch) != {"values", "valid", "n_a", "n_b"}:
        raise ValueError("summary batch schema differs from FrozenSummaryCache.batch")
    return {key: torch.as_tensor(value, device=device,
        dtype=dtype if key == "values" else torch.bool if key == "valid" else torch.long)
        for key, value in batch.items()}


@dataclass
class ResidualScore:
    logit: torch.Tensor
    ca_logit: torch.Tensor
    residual_logit: torch.Tensor
    branch_input: torch.Tensor


class CASpectralResidualScorer(nn.Module):
    """Clone the same completed C8 S6-D2 head; add an initial-zero 193-param MLP.

    Default train_ca=True means equally budgeted CA continuation in every arm.
    The source head itself is neither mutated nor placed in a new train mode.
    train_ca=False is supported but is a DIFFERENT explicitly registered study.
    """

    def __init__(self, ca_head, statistics, *, variant, residual_seed,
                 source_checkpoint_sha256, source_classifier_epochs=8, train_ca=True):
        super().__init__()
        if type(ca_head) is not CrossAttentionPairHead or ca_head.depth != 2:
            raise ValueError("this registered branch retains exactly the S6 depth-two global CA")
        if variant not in VARIANTS or type(residual_seed) is not int or residual_seed < 0:
            raise ValueError("registered variant and nonnegative integer residual seed required")
        if type(source_classifier_epochs) is not int or source_classifier_epochs != 8 or type(train_ca) is not bool:
            raise ValueError("source must be the completed C8 classifier; train_ca must be explicit bool")
        _require_hash(source_checkpoint_sha256, "source_checkpoint_sha256")
        self.variant, self.train_ca = variant, train_ca
        self.initialization = dict(source_checkpoint_sha256=source_checkpoint_sha256,
            source_ca_state_sha256=module_state_sha256(ca_head), source_classifier_epochs=8,
            residual_seed=residual_seed, train_ca=train_ca, depth=2)
        self.ca = deepcopy(ca_head).requires_grad_(train_ca)
        self.standardizer = FrozenTrainStandardizer(statistics)
        # Only the CPU RNG is used/restored here. No re-seeding of running CUDA
        # jobs or side effect on the caller's next data-loader random draw.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(residual_seed)
            self.residual = nn.Sequential(nn.Linear(10, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        active = torch.zeros(10)
        active[:3 if variant == "mass" else 10 if variant == "mass_spectral" else 0] = 1.
        self.register_buffer("active_features", active)
        reference = next(self.ca.parameters())
        self.to(device=reference.device, dtype=reference.dtype)
        self.train(ca_head.training)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_ca:
            self.ca.eval()
        return self

    def forward(self, features_a, features_b, valid_a, valid_b, summary):
        if features_a.requires_grad or features_b.requires_grad:
            raise ValueError("Matcher contextual tokens must be detached/frozen for this study")
        if set(summary) != {"values", "valid", "n_a", "n_b"}:
            raise ValueError("summary batch schema differs from registered cache interface")
        batch = len(features_a)
        values, sv = summary["values"], summary["valid"]
        tensors = (features_a, features_b, valid_a, valid_b, *summary.values())
        if any(value.device != features_a.device for value in tensors):
            raise ValueError("CA tokens, masks and cached summary must share a device")
        if (values.shape != (batch, 10) or sv.shape != (batch,) or sv.dtype != torch.bool
                or summary["n_a"].shape != (batch,) or summary["n_b"].shape != (batch,)
                or summary["n_a"].dtype != torch.long or summary["n_b"].dtype != torch.long
                or valid_a.dtype != torch.bool or valid_b.dtype != torch.bool
                or values.dtype != features_a.dtype):
            raise ValueError("summary batch shapes/dtypes differ from frozen token inputs")
        counts_a, counts_b = valid_a.sum(1), valid_b.sum(1)
        if (not torch.equal(summary["n_a"], counts_a) or not torch.equal(summary["n_b"], counts_b)
                or not torch.equal(sv, (counts_a > 0) & (counts_b > 0))):
            raise ValueError("cached effective dimensions differ from current valid contour tokens")
        if (values < 0).any() or (values[:, 3:] > 1).any():
            raise ValueError("raw summary outside registered mass/spectral bounds")
        if torch.any(values[~sv] != 0):
            raise ValueError("empty-axis summary must remain zero")
        ca_logit = self.ca(features_a, features_b, valid_a, valid_b)
        # Mask AFTER TRAIN normalization: raw zeros before normalization would
        # leak nonzero spectral constants into the mass-only/zero controls.
        branch_input = self.standardizer(values) * self.active_features
        residual = self.residual(branch_input).squeeze(1)
        residual = torch.where(sv, residual, torch.zeros_like(residual))
        return ResidualScore(ca_logit + residual, ca_logit, residual, branch_input)

    def apply_to_frozen_output(self, original, valid_a, valid_b, summary):
        """Preserve the existing training/decision gate and all layout fields.

        Original must be the frozen Rachel base output, from no_grad/eval.
        PairBCE uses this returned gated output exactly as the existing trainer;
        the detail record is diagnostic and must not introduce extra losses.
        """
        detail = self(original.token_features_a, original.token_features_b, valid_a, valid_b, summary)
        finite = original.training_valid & torch.isfinite(detail.logit)
        logit = torch.where(finite, detail.logit, torch.zeros_like(detail.logit))
        return replace(original, fused_logit=logit, local_logit=logit,
            fused_probability=logit.sigmoid(), local_probability=logit.sigmoid(),
            training_valid=finite, decision_valid=finite & original.transport.diagnostics.converged), detail

    def metadata(self):
        return dict(schema_version=MODEL_SCHEMA, feature_schema=FEATURE_SCHEMA,
            feature_names=list(FEATURE_NAMES), variant=self.variant,
            residual_layers=[10, 16, 1], activation="SiLU", residual_parameter_count=193,
            initialization=self.initialization,
            standardizer_sha256=self.standardizer.statistics_sha256,
            matcher_frozen_required=True, loss="existing gated PairBCE only",
            spectral_order_information=False, spectral_probability_interpretation=False,
            zero_control="same parameter count; learns at most a case-independent residual when valid, not an identical function to no residual after training")

    def get_extra_state(self):
        return self.metadata()

    def set_extra_state(self, state):
        if digest_json(state) != digest_json(self.metadata()):
            raise ValueError("checkpoint spectral arm/initialization/statistics differ from registered configuration")
