"""Isolated equal-budget all_tokens D2 Scorer bridge for S7 M12/M16/M20.

Commands: cache (CPU1..4), train (GPU, fresh C16), evaluate (frozen endpoint).
No command executes on import. Existing matched_only modules are never patched;
their numerical functions run in private module dictionaries. Matcher epochs,
head epochs, and the common head-data shuffle clock are recorded separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import functools
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch

from . import continuation_core as core, evaluate_continuation as endpoint_loader
from ..matched_only import cache, data, train, evaluate, inference

SCHEMA = "s7-fixed-matcher-equal-budget-scorer-bridge/1"
CACHE_SCHEMA = "s7-fixed-matcher-token-edge-cache/1"
TRAIN_SCHEMA = "s7-fixed-matcher-fresh-all-tokens-training/1"
_WORKER_COMPUTE = None
_THREADPOOL = None


def private_function(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
        function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def private_module(module):
    """A private shared globals dict; never register aliases in sys.modules."""
    result = ModuleType(module.__name__ + ".matcher_bridge_private")
    result.__dict__.update(module.__dict__)
    for name, value in module.__dict__.items():
        if isinstance(value, FunctionType) and value.__globals__ is module.__dict__:
            result.__dict__[name] = private_function(value, result.__dict__)
    return result


def _worker_init(checkpoint, matcher_epoch, expected_sha):
    global _WORKER_COMPUTE, _THREADPOOL
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("cache workers require explicitly hidden CUDA")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    _THREADPOOL = threadpool_limits(limits=1)
    base, _, info = endpoint_loader.load_endpoint(checkpoint, matcher_epoch)
    if info["checkpoint_sha256"] != expected_sha:
        raise ValueError("Matcher checkpoint changed before worker initialization")
    _WORKER_COMPUTE = private_function(cache.compute, dict(cache.compute.__globals__, _BASE=base))


def _worker_compute(item):
    if _WORKER_COMPUTE is None:
        raise RuntimeError("cache worker not initialized")
    return _WORKER_COMPUTE(item)


def _bounded(items, workers, checkpoint, matcher_epoch, digest):
    # Same bounded 2*workers queue as cache.bounded; top-level worker functions
    # remain spawn-pickleable, unlike closures/private copied module functions.
    arguments = (str(checkpoint), matcher_epoch, digest)
    if workers == 1:
        _worker_init(*arguments)
        for item in items:
            yield _worker_compute(item)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init, initargs=arguments) as pool:
        iterator, pending, exhausted = iter(items), set(), False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
                try:
                    pending.add(pool.submit(_worker_compute, next(iterator)))
                except StopIteration:
                    exhausted = True
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()


def _clocks(value, matcher_epoch):
    """Serialization metadata only; never touches tensors or numerical inputs."""
    if isinstance(value, list):
        return [_clocks(v, matcher_epoch) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: _clocks(v, matcher_epoch) for k, v in value.items()}
    head_epoch = result.get("head_epoch", result.get("completed_head_epochs"))
    if "absolute_epoch" in result and type(head_epoch) is int:
        result.update(absolute_epoch=matcher_epoch + head_epoch, matcher_epoch=matcher_epoch,
            head_data_shuffle_epoch=12 + head_epoch)
    return result


class EndpointBridge:
    def __init__(self, checkpoint, matcher_epoch):
        self.path = Path(checkpoint).resolve(strict=True)
        self.epoch = matcher_epoch
        _, _, info = endpoint_loader.load_endpoint(self.path, matcher_epoch)
        self.source = torch.load(self.path, map_location="cpu", weights_only=False)
        origin = self.source["resume_identity"]
        for split, count, sha in (("train",24000,core.TRAIN_SHA256),("val",3000,core.VAL_SHA256)):
            record = origin["populations"][split]
            if record["manifest_sha256"] != sha or record["count"] != count:
                raise ValueError("requires original unchanged S7 TRAIN24K/clean SIMVAL3K")
        self.endpoint = dict(schema=SCHEMA, matcher_epoch=matcher_epoch,
            matcher_checkpoint=str(self.path), matcher_checkpoint_sha256=info["checkpoint_sha256"],
            matcher_state_sha256=info["matcher_state_sha256"],
            origin_m12_checkpoint_sha256=core.SOURCE_SHA256, formal_source=True)
        self.cache = self._cache_module()
        self.data = self._data_module()
        self.train = self._train_module()
        self.evaluate = self._evaluate_module()

    def source_checkpoint(self, path=None):
        if path is not None and Path(path).resolve() != self.path:
            raise ValueError("source path differs from selected Matcher endpoint")
        if data.sha(self.path) != self.endpoint["matcher_checkpoint_sha256"]:
            raise ValueError("source checkpoint changed after bridge construction")
        return self.source

    def _fresh_base(self, payload):
        if payload is not self.source:
            raise ValueError("only the bridge's verified checkpoint may be loaded")
        base, _, info = endpoint_loader.load_endpoint(self.path, self.epoch)
        if info["checkpoint_sha256"] != self.endpoint["matcher_checkpoint_sha256"]:
            raise ValueError("Matcher checkpoint changed")
        return SimpleNamespace(base_model=base)

    def _cache_module(self):
        module = private_module(cache)
        module.SCHEMA, module.SOURCE = CACHE_SCHEMA, self.path
        module.SOURCE_SHA = self.endpoint["matcher_checkpoint_sha256"]
        module.source_checkpoint = self.source_checkpoint
        module.old = SimpleNamespace(**vars(cache.old))
        module.old.load_decoupled_checkpoint = self._fresh_base
        module.bounded = lambda items, workers, source_path: _bounded(items, workers, source_path,
            self.epoch, self.endpoint["matcher_checkpoint_sha256"])
        def save(path, value):
            if Path(path).name == "protocol.json":
                value = dict(value, matcher_endpoint=deepcopy(self.endpoint),
                    bridge_implementation_sha256=data.sha(__file__),
                    numeric_cache_implementation_sha256=data.sha(cache.__file__))
            cache.save(path, value)
        module.save = save
        return module

    def _data_module(self):
        module = private_module(data)
        owner = self
        class BoundFormalCache(data.FormalCache):
            def __init__(self, root, split):
                protocol = json.loads((Path(root) / "protocol.json").read_text())
                legacy = owner.epoch == 12 and protocol.get("schema") == cache.SCHEMA
                validator_cache = SimpleNamespace(**vars(owner.cache))
                if legacy:
                    validator_cache.SCHEMA = cache.SCHEMA
                    if protocol.get("implementation_sha256") != data.sha(cache.__file__):
                        raise ValueError("legacy M12 cache numerical implementation differs")
                elif (protocol.get("matcher_endpoint") != owner.endpoint or
                        protocol.get("bridge_implementation_sha256") != data.sha(__file__) or
                        protocol.get("numeric_cache_implementation_sha256") != data.sha(cache.__file__)):
                    raise ValueError("cache belongs to a different Matcher endpoint/bridge implementation")
                initializer = private_function(data.FormalCache.__init__,
                    dict(data.FormalCache.__init__.__globals__, cache=validator_cache))
                initializer(self, root, split)
                self.binding.update(matcher_endpoint=deepcopy(owner.endpoint),
                    legacy_m12_cache_reused=legacy)
        module.FormalCache = BoundFormalCache
        module.cache = self.cache
        return module

    def _train_module(self):
        module = private_module(train)
        module.SCHEMA, module.ARMS = TRAIN_SCHEMA, ("all_tokens",)
        module.cache, module.data = self.cache, self.data
        inherited_binding = module.implementation_binding
        def binding():
            return dict(inherited_binding(), scorer_bridge=data.sha(__file__),
                matcher_endpoint_loader=data.sha(endpoint_loader.__file__),
                matcher_continuation_core=data.sha(core.__file__), cache_producer=data.sha(cache.__file__))
        module.implementation_binding = binding
        inherited_identity = module.identity
        def identity(arm, model, training, validation):
            if arm != "all_tokens":
                raise ValueError("convergence control only registers fresh all_tokens D2")
            result = inherited_identity(arm, model, training, validation)
            result.update(matcher_endpoint=deepcopy(self.endpoint), source_matcher_epochs=self.epoch,
                source_matcher_pair_exposures=self.epoch * 24000,
                absolute_epochs=[self.epoch+1,self.epoch+16], head_data_shuffle_epochs=[13,28],
                equal_budget_control="same fresh D2/4head seed260914, batch16, S7 pairs, C16 schedule/loss",
                numerical_contract="same CPU-FP32 cache algorithm; each fixed Matcher has its own features/validity")
            return result
        module.identity = identity
        inherited_plan = module.plan
        module.plan = lambda: _clocks(inherited_plan(), self.epoch)
        # Original execute uses absolute_epoch for data order. Map this explicit
        # total-clock back to the SAME historical head-data clock for all M.
        module.runner = SimpleNamespace(**vars(train.runner))
        def epoch_indices(length, seed, epoch, limit):
            head_epoch = epoch - self.epoch
            if not 1 <= head_epoch <= 16:
                raise ValueError("head data-order epoch outside fixed C16")
            return train.runner.epoch_indices(length, seed=seed, epoch=12+head_epoch, limit=limit)
        module.runner.epoch_indices = epoch_indices
        inherited_winners, inherited_payload, inherited_restore = module.update_winners, module.checkpoint_payload, module.restore
        module.update_winners = lambda *a, **k: _clocks(inherited_winners(*a, **k), self.epoch)
        module.checkpoint_payload = lambda *a, **k: _clocks(inherited_payload(*a, **k), self.epoch)
        def restore(model, optimizer, saved, ident):
            h = saved.get("head_epoch")
            if (type(h) is not int or saved.get("matcher_epoch") != self.epoch or
                    saved.get("absolute_epoch") != self.epoch+h or saved.get("head_data_shuffle_epoch") != 12+h):
                raise ValueError("head checkpoint Matcher/head/data-clock metadata differs")
            # The unchanged numerical restore checks its historical clock alias.
            # Validate true clocks above; adapt ONLY this temporary validator view.
            legacy_clock_view = dict(saved, absolute_epoch=12+h)
            return inherited_restore(model, optimizer, legacy_clock_view, ident)
        module.restore = restore
        module.save = lambda path, value: train.save(path, _clocks(value, self.epoch))
        inherited_execute = module.execute
        def execute(args, model, optimizer, training, validation, ident, device):
            if getattr(args, "smoke", None):
                if args.resume or args.smoke != 32:
                    raise ValueError("only new discard32 smoke is allowed")
                ids = train.runner.epoch_indices(len(training), seed=train.DATA_SEED, epoch=13, limit=32)
                report = module.train_segment(model, training, ids, optimizer, device)
                result = dict(schema=TRAIN_SCHEMA, status="smoke_complete", matcher_endpoint=self.endpoint,
                    identity=ident, training=report, weights_discarded=True, checkpoint_written=False,
                    formal_training_counted=False, discarded_pairs=32)
                module.save(Path(args.output)/"smoke.json", result)
                module.save(Path(args.output)/"protocol.json", result)
                return result
            return _clocks(inherited_execute(args,model,optimizer,training,validation,ident,device), self.epoch)
        module.execute = execute
        return module

    def _evaluate_module(self):
        module = private_module(evaluate)
        module.cache, module.data, module.train = self.cache, self.data, self.train
        owner = self
        class BoundInference(inference.FrozenMatchedInference):
            def metadata(self):
                return dict(super().metadata(), matcher_endpoint=deepcopy(owner.endpoint),
                    source_binding="typed fixed S7 Matcher M12/M16/M20 plus separate fresh C16 head")
        module.inference = SimpleNamespace(**vars(inference))
        module.inference.FrozenMatchedInference = BoundInference
        inherited_load = module.load_frozen_model
        def load(root, selection, *, budget=16):
            frozen = json.loads((Path(root)/"freezes"/('c%d.json'%budget)).read_text())
            legacy = self.epoch == 12 and frozen.get("schema") == train.SCHEMA
            ident = frozen.get("identity", {})
            initial = train.make_scorer("all_tokens", seed=train.HEAD_SEED)
            if (ident.get("arm") != "all_tokens" or ident.get("head_seed") != train.HEAD_SEED or
                    ident.get("data_seed") != train.DATA_SEED or
                    ident.get("initial_state_sha256") != train.state_digest(initial)):
                raise ValueError("requires identical fresh all_tokens D2 initialization and data seed")
            if not legacy and ident.get("matcher_endpoint") != self.endpoint:
                raise ValueError("trained head belongs to another fixed Matcher")
            # Existing priority M12 C16 is a valid baseline: original loader
            # enforces full completion, frozen thresholds and original code seal.
            adapter, receipt = (evaluate.load_frozen_model(root,selection,budget=budget) if legacy
                else inherited_load(root,selection,budget=budget))
            receipt.update(matcher_endpoint=deepcopy(self.endpoint), matcher_epoch=self.epoch,
                budget=self.epoch+budget, epoch=self.epoch+receipt["head_epoch"],
                head_data_shuffle_epoch=12+receipt["head_epoch"], legacy_m12_head_reused=legacy,
                source_matcher_checkpoint=str(self.path), source_matcher_sha256=self.endpoint["matcher_checkpoint_sha256"],
                evaluation_adapter_sha256=data.sha(__file__))
            return adapter, receipt
        module.load_frozen_model = load
        inherited_runtime = module.inference_runtime
        def runtime(device):
            result = inherited_runtime(device)
            result["source_sha256"].update(scorer_bridge=data.sha(__file__),
                matcher_endpoint_loader=data.sha(endpoint_loader.__file__),
                matcher_continuation_core=data.sha(core.__file__),
                matcher_checkpoint=self.endpoint["matcher_checkpoint_sha256"],
                matcher_state=self.endpoint["matcher_state_sha256"])
            return result
        module.inference_runtime = runtime
        return module


def main(argv=None):
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("command", choices=("cache","train","evaluate"))
    selector.add_argument("--matcher-checkpoint", required=True)
    selector.add_argument("--matcher-epoch", type=int, choices=(12,16,20), required=True)
    selected, remaining = selector.parse_known_args(argv)
    if selected.command == "cache":
        parser = argparse.ArgumentParser(description="CPU cache for the selected typed Matcher endpoint")
        parser.add_argument("--split",choices=("train","val"),required=True)
        parser.add_argument("--output",type=Path,required=True)
        parser.add_argument("--workers",type=int,choices=(1,2,3,4),default=1)
        parser.add_argument("--limit",type=int,help="probe only; never formal eligible")
        args = parser.parse_args(remaining)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("cache requires explicit CUDA_VISIBLE_DEVICES empty")
        args.source = Path(selected.matcher_checkpoint)
    elif selected.command == "train":
        parser = train.parser()
        parser.set_defaults(arm="all_tokens")
        next(a for a in parser._actions if a.dest=="arm").required = False
        parser.add_argument("--smoke",type=int,choices=(32,))
        args = parser.parse_args(remaining)
        if args.arm != "all_tokens" or args.train_stage_cache or args.val_stage_cache or args.smoke and args.resume:
            raise ValueError("only all_tokens without stage caches; smoke cannot resume")
    else:
        parser = evaluate.original.parser()
        parser.add_argument("--head-budget",type=int,choices=(8,16),default=16)
        args = parser.parse_args(remaining)
    bridge = EndpointBridge(selected.matcher_checkpoint, selected.matcher_epoch)
    if selected.command == "cache":
        return bridge.cache.run(args)
    if selected.command == "train":
        return bridge.train.run(args)
    return bridge.evaluate.run(args)


if __name__ == "__main__":
    result = main()
    if result is not None:
        print(json.dumps(result, sort_keys=True))
