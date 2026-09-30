"""Two disposable CPU updates from actual S7 M12; never writes a checkpoint.

Validates the new M-only kernel and original Adam moments on TRAIN32. It does
not establish CUDA resume equivalence, GPU throughput, or any performance gain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def run(args):
    sys.path.insert(0,str(Path(args.source_root).resolve(strict=True)))
    import torch
    import continuation_core as core
    old = core.old
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("disposable probe requires no visible CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_path = Path(args.checkpoint).resolve(strict=True)
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    protocol = dict(schema="s7-matcher-continuation-discard-cpu32/1",status="running",
        pid=os.getpid(),source_checkpoint=str(source_path),source_checkpoint_sha256=sha(source_path),
        script_sha256=sha(__file__),core_sha256=sha(core.__file__),source_root=str(args.source_root),
        GPU_used=False,CUDA_RNG_restored=False,rng_mode="cpu_only_probe",workers=0,
        physical_microbatch=16,effective_batch=16,logical_microbatch=1,
        formal_training_counted=False,formal_pair_exposures=0,checkpoint_written=False,
        limited_claim="CPU M-only loss/optimizer wiring; not CUDA trajectory or completed M13")
    save(out/"protocol.json",protocol)
    try:
        source = torch.load(source_path,map_location="cpu",weights_only=False)
        source_sha = protocol["source_checkpoint_sha256"]
        net = core.validate_source(source,source_sha).cpu()
        identity = core.build_identity(source,source_sha,rng_mode="cpu_only_probe")
        origin = source["resume_identity"]
        train,validation,cap,records = old.make_populations(SimpleNamespace(sampling="original512",
            train_materialized_manifest=origin["populations"]["train"]["manifest"],
            dataset=str(Path(origin["populations"]["val"]["manifest"]).parents[1])))
        if old.canonical_digest(records) != old.canonical_digest(origin["populations"]):
            raise ValueError("original populations changed")
        del validation
        optimizer = core.create_optimizer(net)
        context = core.restore_initial(net,optimizer,source,identity,rng_mode="cpu_only_probe")
        if net.phase != "matcher" or not net.training:
            raise ValueError("did not restore Matcher training phase")
        active = {name:p for name,p in net.named_parameters() if p.requires_grad}
        if len(active) != 43:
            raise ValueError("actual S7 active Matcher count differs")
        if {int(optimizer.state[p]["step"]) for p in active.values()} != {18000}:
            raise ValueError("M12 Adam counters not restored")
        frozen_modules = dict(head=net.score_head,coarse=net.base_model.coarse,
                              local=net.base_model.local_head,fusion=net.base_model.fusion)
        before_frozen = {k:old.state_digest(m) for k,m in frozen_modules.items()}
        before_base = old.state_digest(net.base_model)
        original_cuda_rng = [t.clone() for t in source["rng_state"]["cuda"]]
        order = old.runner.epoch_indices(24000,seed=old.SEED,epoch=13,limit=None)
        indices = order[:32]
        loader = old.make_weathering_loader(train,indices,batch_size=16,num_workers=0,
            seed=old.SEED+49,contour_cap=cap)
        runtime = SimpleNamespace(output=str(out),microbatch=1,physical_microbatch=16,
            effective_batch=16,log_every=1000)
        report = core.private_train_segment(net,loader,optimizer,
            old.RachelN512LossConfig(**source["loss_config"]),torch.device("cpu"),runtime,13)
        if (report["samples"],report["optimizer_updates"],report["phase"],report["pair_bce_weight"]) != (32,2,"matcher",0.):
            raise ValueError("CPU probe optimized wrong phase/budget")
        after_steps = {int(optimizer.state[p]["step"]) for p in active.values()}
        if after_steps != {18002}:
            raise ValueError("did not continue original active Adam counters")
        if before_frozen != {k:old.state_digest(m) for k,m in frozen_modules.items()}:
            raise ValueError("Scorer/coarse/legacy classification state changed")
        after_base = old.state_digest(net.base_model)
        if after_base == before_base:
            raise ValueError("Matcher failed to update")
        if any(not torch.equal(a,b) for a,b in zip(original_cuda_rng,source["rng_state"]["cuda"])):
            raise ValueError("source CUDA RNG bytes changed")
        if torch.cuda.is_initialized():
            raise ValueError("CPU probe unexpectedly initialized CUDA")
        result = dict(status="complete",training=report,source_epoch=12,probe_epoch=13,
            actual_active_matcher_parameters=len(active),source_Adam_step=18000,final_Adam_step=18002,
            frozen_branches_unchanged=True,Matcher_updated=True,
            base_state_before=before_base,base_state_after=after_base,
            original_cuda_rng_bytes_retained=True,CUDA_RNG_restored=False,
            discarded_pair_exposures=32,formal_pair_exposures=0,checkpoint_written=False,
            weights_and_optimizer_discarded=True,epoch13_sample_indices=[int(i) for i in indices],
            queue_modified=False,elapsed_seconds=time.monotonic()-started)
        save(out/"results.json",result)
        protocol.update(status="complete",elapsed_seconds=time.monotonic()-started)
        save(out/"protocol.json",protocol)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        protocol.update(status="failed",error=f"{type(exc).__name__}: {exc}",elapsed_seconds=time.monotonic()-started)
        save(out/"protocol.json",protocol)
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    run(p.parse_args())
