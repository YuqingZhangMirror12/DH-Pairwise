"""Frozen Matcher + existing CA spectral scorer, with CPU-bound cached inputs."""
from copy import deepcopy
from dataclasses import dataclass, fields
import time
from types import SimpleNamespace

import torch
from torch import nn
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_head import model as head
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.spectral_training import cache as binding
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.candidate_local import train as shared
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Output

RESIDUAL_SEED = 260919


@dataclass(frozen=True)
class SpectralOutput(RachelN512Output):
    score_details: dict


class FrozenSpectralModel(nn.Module):
    def __init__(self, source, statistics, variant):
        super().__init__()
        self.base_model = deepcopy(source.base_model).eval().requires_grad_(False)
        self.config = source.config
        self.source_metadata = deepcopy(source.metadata())
        self.score_head = head.CASpectralResidualScorer(source.score_head, statistics,
            variant=variant, residual_seed=RESIDUAL_SEED, source_checkpoint_sha256=shared.SOURCE_SHA,
            source_classifier_epochs=8, train_ca=True)
        self.phase = "classifier"
        self._pending_summary = None
        self.reset_binding_cost()

    def reset_binding_cost(self):
        self.binding_cost = dict(calls=0, pairs=0, cpu_hash_and_lookup_s=0.,
            svd_calls=0, gpu_to_cpu_input_copies=0)

    def bind_batch(self, batch, cache):
        if self._pending_summary is not None:
            raise ValueError("previous bound summary was not consumed")
        started = time.perf_counter()
        hashes = binding.input_hashes(batch)
        self._pending_summary = cache.batch(list(batch.pair_ids), hashes)
        self.binding_cost["calls"] += 1
        self.binding_cost["pairs"] += len(hashes)
        self.binding_cost["cpu_hash_and_lookup_s"] += time.perf_counter() - started

    def forward(self, *inputs):
        if len(inputs) != 6 or self._pending_summary is None:
            raise ValueError("six inputs require a fresh CPU source/input-verified cache binding")
        summary = self._pending_summary
        self._pending_summary = None  # one use only, including failed forward
        with torch.no_grad():
            frozen = self.base_model(*inputs)
        summary = head.tensor_summary_batch(summary, inputs[0].device, frozen.token_features_a.dtype)
        output, detail = self.score_head.apply_to_frozen_output(frozen, inputs[4], inputs[5], summary)
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        return SpectralOutput(**values, score_details=dict(ca_logit=detail.ca_logit,
            spectral_residual_logit=detail.residual_logit, spectral_branch_input=detail.branch_input))

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def set_phase(self, phase):
        if phase != "classifier":
            raise ValueError("spectral study cannot train Matcher")
        self.base_model.requires_grad_(False)
        self.score_head.ca.requires_grad_(True)
        self.score_head.residual.requires_grad_(True)
        for p in self.parameters():
            p.grad = None
        return self.train(self.training)

    def metadata(self):
        return dict(schema_version="rachel-frozen-spectral-training-model/1",
            source_model=self.source_metadata, scorer=self.score_head.metadata(),
            input_cache_binding="CPU loader exact six-input SHA before GPU transfer; single-use summary",
            matcher_unchanged=True, svd_in_forward=False)


class BoundLoader:
    """Proxy preserves original batching, labels, target/loss/statistics path."""
    def __init__(self, loader, model, cache):
        self.loader, self.model, self.cache = loader, model, cache

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        for wrapped in self.loader:
            self.model.bind_batch(getattr(wrapped, "batch", wrapped), self.cache)
            yield wrapped


def build(source_checkpoint, statistics, variant):
    rng = shared.old.capture_rng_state()
    try:
        source = shared.old.load_decoupled_checkpoint(source_checkpoint)
        return FrozenSpectralModel(source, statistics, variant)
    finally:
        shared.old.restore_rng_state(rng)


def restore_optimizer(source, model):
    facade = SimpleNamespace(base_model=model.base_model, score_head=model.score_head.ca)
    shared.cont.check_optimizer(source, facade, expected_head_step=12000)
    if head.module_state_sha256(model.score_head.ca) != model.score_head.initialization["source_ca_state_sha256"]:
        raise ValueError("CA differs from source at initial restore")
    if torch.count_nonzero(model.score_head.residual[-1].weight) or torch.count_nonzero(model.score_head.residual[-1].bias):
        raise ValueError("residual must initialize exactly zero")
    optimizer = shared.old.create_optimizer(facade)
    optimizer.load_state_dict(deepcopy(source["optimizer_state_dict"]))
    group = {k: deepcopy(v) for k, v in optimizer.param_groups[1].items() if k not in ("params", "phase_family")}
    group.update(params=list(model.score_head.residual.parameters()), phase_family="spectral_residual")
    optimizer.add_param_group(group)
    return optimizer
