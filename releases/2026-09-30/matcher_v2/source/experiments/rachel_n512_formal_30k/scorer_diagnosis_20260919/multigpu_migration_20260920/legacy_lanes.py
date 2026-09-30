"""Read-only extraction of retained leaf jobs for the new seven-GPU server.

No launcher, old PID dependency, queue receipt mutation, or GPU allocation.
Commands come from their original immutable source, not this migration module.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

MODULE = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919."
C2_PAUSED_SHA = "08fc931e2354b28884174bea74c3fea5612e9424436c686932aa29490f6b1ef4"


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_subset(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError("completion differs: " + key)


def snapshot_stages(plan, phase, inputs=None):
    """Invoke only inert stage builders with the original interpreter/imports."""
    code = '''
import json, sys
from pathlib import Path
from importlib import import_module
p,phase,inputs=json.load(sys.stdin)
prefix="experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919."
source=Path(p["source_root"]).resolve()
if phase=="followup":
    m=import_module(prefix+"continuation_depth_controls.followup_supervisor")
    if source not in Path(m.__file__).resolve().parents: raise ValueError("wrong frozen source")
    stages=m.stage_plan(Path(p["output_root"]),p["python"])
    if stages!=p["stages"] or len(stages)!=25: raise ValueError("old depth plan changed")
    t=m.candidate.train
    last=Path(p["output_root"])/"candidate_local/c2_c16/last.pt"
    saved=t.torch.load(last,map_location="cpu",weights_only=False)
    original=t.read_source(t.SOURCE)
    identity=t.make_identity("c2",original,t.SOURCE,saved["candidate_identity"]["populations"])
    if t.validate_payload(saved,identity)!=81: raise ValueError("C2 must resume committed segment81")
    result={"candidate":m.candidate.stage_plan(Path(p["output_root"])/"candidate_local",p["python"]),"depth":stages[3:]}
else:
    m=import_module(prefix+"spectral_training.after_dependencies")
    if source not in Path(m.__file__).resolve().parents: raise ValueError("wrong frozen source")
    root=Path(p["output_root"])
    result={"spectral":m.stage_plan(p,inputs,"unused-no-supervisor")[:3]+m.queue.stage_plan(root,inputs["cache_bundle"],inputs["cache_bundle_sha"],inputs["registry"],p["python"])}
print(json.dumps(result))
'''
    env = dict(os.environ, PYTHONPATH=plan["source_root"], CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    value = subprocess.check_output([plan["python"], "-c", code],
                                   input=json.dumps([plan, phase, inputs]),
                                   cwd=plan["source_root"], env=env, text=True)
    return json.loads(value)


def spectral_inputs(cache_receipt):
    receipt = read(cache_receipt)
    require_subset(receipt, dict(status="complete", completed_stages=4))
    if len(receipt.get("stages", [])) != 4 or any(
            row.get("status") != "complete" for row in receipt["stages"]):
        raise ValueError("all four actual CPU cache stages must be complete")
    registrations = receipt["registrations"]
    if set(registrations) != {"train_val", "test", "real", "ood"}:
        raise ValueError("CPU cache registration is incomplete")
    registry_path = receipt["endpoint_registry"]
    if sha256(registry_path) != receipt["endpoint_registry_sha256"]:
        raise ValueError("endpoint cache registry changed")
    registry = read(registry_path)
    expected = {s: {k: registrations[s][k] for k in ("bundle", "sha256")}
                for s in ("test", "real", "ood")}
    if registry != expected:
        raise ValueError("registry differs from actual CPU completion")
    for record in registrations.values():
        if sha256(record["bundle"]) != record["sha256"]:
            raise ValueError("registered bundle changed")
    return dict(cache_bundle=registrations["train_val"]["bundle"],
                cache_bundle_sha=registrations["train_val"]["sha256"], registry=registry)


def completion_expect(stage):
    kind = stage["kind"]
    if kind.endswith("smoke") or kind == "smoke":
        return dict(status="smoke_complete", pair_exposures=32, optimizer_updates=2,
                    formal_training_counted=False, frozen_base_unchanged=True,
                    weights_discarded=True, no_checkpoint_written=True)
    if kind in ("training", "depth_training"):
        return dict(status="complete", completed_segments=112,
                    additional_pair_exposures=192000, additional_optimizer_updates=12000)
    if kind == "endpoint_evaluation":
        return dict(status="complete")
    raise ValueError("not a supported leaf job: " + kind)


def decorate(stages, plan, lane, *, resume_c2=False):
    result = []
    smokes = []
    training = None
    for original in stages:
        stage = deepcopy(original)
        command = stage["command"]
        if len(command) < 3 or command[1] != "-m" or any(
                token.endswith(".queue") or "supervisor" in token or "after_dependencies" in token
                for token in command):
            raise ValueError("migration accepts leaf modules only")
        expected = completion_expect(stage)
        output = Path(command[command.index("--output") + 1])
        completion = Path(stage["completion"])
        resumed = resume_c2 and stage["name"] == "c2_C9_C16"
        requirement = dict(path=str(completion), expect=expected, producer_lane=lane)
        if stage["kind"].endswith("smoke") or stage["kind"] == "smoke":
            smokes.append(requirement)
        elif stage["kind"] in ("training", "depth_training"):
            stage["prerequisites"] = deepcopy(smokes)
            training = requirement
        elif training:
            stage["prerequisites"] = [deepcopy(training)]
        if completion.exists() and not resumed:
            require_subset(read(completion), expected)
            continue  # Actual completed leaf remains untouched, not rerun.
        if output.exists() and not resumed:
            raise ValueError("incomplete existing output will not be overwritten: " + str(output))
        if resumed:
            if "--resume" in command:
                raise ValueError("original C2 command unexpectedly already resumes")
            command.append("--resume")
            stage.update(resume_checkpoint=str(output / "last.pt"),
                         resume_checkpoint_sha256=C2_PAUSED_SHA, resume_completed_segments=81)
        stage.update(cwd=plan["source_root"], output=str(output),
                     env=dict(PYTHONPATH=plan["source_root"], PYTHONUNBUFFERED="1"),
                     completion_expect=expected, gpu=True)
        result.append(stage)
    return result


def derive_legacy_lanes(followup_plan, spectral_plan, cache_receipt, *,
                        c2_checkpoint_sha256=C2_PAUSED_SHA, stage_reader=snapshot_stages):
    """Return lane0/4/5 leaf lists; call on new host where original paths exist."""
    followup, spectral = read(followup_plan), read(spectral_plan)
    last = Path(followup["output_root"]) / "candidate_local/c2_c16/last.pt"
    if c2_checkpoint_sha256 != C2_PAUSED_SHA or sha256(last) != c2_checkpoint_sha256:
        raise ValueError("C2 checkpoint differs from explicitly paused segment81")
    old = stage_reader(followup, "followup")
    candidate = old["candidate"]
    if len(candidate) != 20 or any(not s["name"].startswith("c2_") for s in candidate[10:]):
        raise ValueError("expected exactly C1 then C2 original ten-stage arms")
    if len(old["depth"]) != 22:
        raise ValueError("two retained depth arms require22 leaves")
    inputs = spectral_inputs(cache_receipt)
    spec = stage_reader(spectral, "spectral", inputs)["spectral"]
    if len(spec) != 33:
        raise ValueError("three spectral arms require33 leaves")
    return dict(lane0=decorate(candidate[10:], followup, "lane0", resume_c2=True),
                lane4=decorate(old["depth"], followup, "lane4"),
                lane5=decorate(spec, spectral, "lane5"))
