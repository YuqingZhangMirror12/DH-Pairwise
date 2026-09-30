"""Discard-only CPU32 wiring check; never formal training or CUDA smoke.

Uses the sealed matched_only.smoke.two_steps unchanged. Every arm starts
fresh, sees the same first32 epoch13 TRAIN rows, and discards all state.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

ARMS = ("all_tokens", "matched_tokens", "matched_edges", "edge_seed", "edge_multi")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def small_binding(root):
    """Metadata hashes and array stat bindings, never hash feature contents."""
    root = Path(root)
    return dict(protocol_sha256=sha(root / "protocol.json"),
                arrays={p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                        for p in sorted(root.glob("*.npy"))})


class ObservedDataset:
    """Delegate exact base/stage inputs while recording selection eligibility."""
    def __init__(self, source, groups, arm):
        self.source, self.groups, self.arm = source, groups, arm
        self.observations = []

    def __len__(self):
        return len(self.source)

    def batch(self, indices, device):
        original = self.source.batch(indices, device)
        selection = original.model_args[4]
        row = dict(pair_ids=list(original.pair_ids),
            labels=original.labels.tolist(), training_valid=original.training_valid.tolist(),
            decision_valid=original.decision_valid.tolist(),
            candidate_count=selection.candidate_valid.sum(1).tolist(),
            decoded_layout_valid=selection.layout_valid.tolist(),
            final_inlier_count=selection.candidate_inliers.sum(1).tolist(),
            selected_token_count_a=selection.mask_a.sum(1).tolist(),
            selected_token_count_b=selection.mask_b.sum(1).tolist())
        if self.arm in ("edge_seed", "edge_multi"):
            groups = self.groups.batch(indices, self.arm, device)
            original = replace(original, model_kwargs=dict(original.model_kwargs, groups=groups))
            row.update(group_present=groups.present.tolist(), group_eligible=groups.eligible.tolist(),
                group_inlier_count=groups.inlier_count.tolist(), group_ranks=groups.ranks.tolist(),
                scorer_eligible=groups.eligible.any(1).tolist())
        elif self.arm == "all_tokens":
            va, vb = original.model_args[2:4]
            row["scorer_eligible"] = (va.any(1) & vb.any(1)).tolist()
        else:
            row["scorer_eligible"] = (selection.mask_a.any(1) & selection.mask_b.any(1)).tolist()
        self.observations.append(row)
        return original


def run(args):
    sealed = Path(args.source_root).resolve(strict=True)
    sys.path.insert(0, str(sealed))
    import numpy as np
    import torch
    from matched_only import data, smoke, stage_cache, train

    if any(sealed not in Path(m.__file__).resolve().parents for m in (data, smoke, stage_cache, train)):
        raise ValueError("must import existing sealed matched_only source, not a local overlay")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.__dict__.get("_initialized", False):
        raise RuntimeError("CPU-only process required; CUDA must not be initialized")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output).resolve()
    roots = [Path(args.train_cache).resolve(strict=True), Path(args.stage_cache).resolve(strict=True)]
    if any(output == p or output in p.parents or p in output.parents for p in roots):
        raise ValueError("new JSON output must be outside caches")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="matched-only-cpu-real32-discard/1", status="running", pid=os.getpid(),
        execution_device="cpu", threads=1, niceness=os.getpriority(os.PRIO_PROCESS, 0),
        CUDA_VISIBLE_DEVICES="", GPU_used=False, gpu_lock_acquired=False,
        CUDA_runtime_initialized=False, Matcher_loaded=False, checkpoint_loaded=False,
        formal_training_counted=False, formal_pair_exposures=0, formal_optimizer_updates=0,
        checkpoint_written=False, weights_persisted=False, checkpoint_usable_for_formal=False,
        formal_reuse_forbidden=True, scheduled_CUDA_smoke_replaced=False,
        limited_claim="CPU numerical wiring on 32 actual TRAIN pairs, not convergence, GPU speed or model quality",
        arms=list(ARMS), physical_microbatch=16, effective_batch=16, accumulation_steps=1,
        precision="fp32", sampling="same first32 of seed260913 absolute epoch13 permutation",
        script_sha256=sha(__file__), sealed_source_root=str(sealed),
        implementation_sha256=dict(smoke=sha(smoke.__file__), **train.implementation_binding()),
        torch_version=str(torch.__version__), started_unix=time.time(), completed_arms=[])
    save(output / "protocol.json", protocol)
    try:
        source = data.FormalCache(args.train_cache, "train")
        if len(source) != 24000:
            raise ValueError("requires complete formal TRAIN24K cache")
        # Existing reader validates its modest side-cache hashes once; reuse it
        # for both stages. No repeated full-array audit or feature-file hashing.
        groups = stage_cache.StageCache(args.stage_cache, source)
        before = {str(p): small_binding(p) for p in roots}
        protocol.update(source_binding=source.binding, stage_cache_binding=groups.binding,
                        source_policy="read-only mmap; existing reader bindings; no model or cache writes")
        indices = train.runner.epoch_indices(len(source), seed=train.DATA_SEED, epoch=13, limit=None)[:32]
        protocol.update(indices=[int(i) for i in indices],
                        pair_ids=[source.records[int(i)]["pair_id"] for i in indices])
        save(output / "protocol.json", protocol)
        results = {}
        for arm in ARMS:
            random.seed(train.HEAD_SEED)
            np.random.seed(train.HEAD_SEED)
            torch.random.default_generator.manual_seed(train.HEAD_SEED)
            model = train.make_scorer(arm, seed=train.HEAD_SEED)
            initial = train.state_digest(model)
            dataset = ObservedDataset(source, groups, arm)
            receipts = []
            try:
                result = smoke.two_steps(model, dataset, indices, torch.device("cpu"), receipts=receipts)
                if result["optimizer_updates"] != 2 or not result["finite_gradient_check"]:
                    raise RuntimeError("disposable optimizer did not complete two finite updates")
                expected_fallback = [sum(not v for v in b["scorer_eligible"]) for b in dataset.observations]
                if expected_fallback != [int(s["fallback_count"]) for s in result["steps"]]:
                    raise RuntimeError("recorded selection eligibility differs from scorer fallback")
                final = train.state_digest(model)
                if final == initial:
                    raise RuntimeError("fresh head did not update")
                result.update(status="complete", arm=arm, execution_device="cpu",
                    model_metadata=model.metadata(), initial_model_sha256=initial,
                    disposable_final_model_sha256=final, fresh_head_updated=True,
                    selection_batches=dataset.observations, eligibility_matches_fallback=True,
                    weights_and_optimizer_discarded=True, scheduled_CUDA_smoke_replaced=False)
                save(output / (arm + ".json"), result)
                results[arm] = result
            finally:
                del model
            protocol["completed_arms"].append(arm)
            save(output / "protocol.json", protocol)
        after = {str(p): small_binding(p) for p in roots}
        if before != after or sha(source.root / "pairs.json") != source.binding["pairs_sha256"]:
            raise RuntimeError("cache metadata/stat binding changed")
        if torch.cuda.__dict__.get("_initialized", False):
            raise RuntimeError("CPU probe unexpectedly initialized CUDA")
        protocol.update(status="complete", source_binding_unchanged=True,
                        binding_check="protocol/pairs hashes and all array size/mtime; not full feature content hashing",
                        weights_and_optimizer_discarded=True, completed_unix=time.time())
        save(output / "protocol.json", protocol)
        return dict(status="complete", output=str(output), samples_per_arm=32,
            optimizer_updates_per_arm=2, formal_optimizer_updates=0,
            source_binding_unchanged=True, CUDA_runtime_initialized=False,
            arms={arm: dict(elapsed_s=r["elapsed_s"],
                losses=[s["loss"] for s in r["steps"]],
                active_gradient_tensors=[s["active_gradient_tensors"] for s in r["steps"]],
                fallback_counts=[s["fallback_count"] for s in r["steps"]],
                initial_model_sha256=r["initial_model_sha256"],
                disposable_final_model_sha256=r["disposable_final_model_sha256"])
                for arm, r in results.items()})
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), formal_optimizer_updates=0)
        save(output / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--stage-cache", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
