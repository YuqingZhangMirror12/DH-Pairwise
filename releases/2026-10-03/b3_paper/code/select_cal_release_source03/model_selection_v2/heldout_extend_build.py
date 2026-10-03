"""One-shot append-only reserve revision, preserving original pilot evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .heldout_build import sha,read,save
    from .heldout_adopt import admit_old_pilot,check_admission
    from .heldout_extend_plan import extend,validate_extension
except ImportError:
    from heldout_build import sha,read,save
    from heldout_adopt import admit_old_pilot,check_admission
    from heldout_extend_plan import extend,validate_extension


def catalog_payload(source_plan):
    catalog=read(source_plan["catalog_path"])
    if sha(source_plan["catalog_path"])!=source_plan["catalog_sha256"]:
        raise ValueError("catalog identity changed")
    files={str(Path(catalog["release_root"])/relative):digest for relative,digest in catalog["copied_files_sha256"].items()}
    files.update(catalog["original_files_sha256"])
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError("catalog payload changed: "+path)
    return dict(schema="heldout-catalog-payload-admission/1",status="byte_exact",files=files,
                catalog=dict(path=source_plan["catalog_path"],sha256=source_plan["catalog_sha256"]),
                copied_count=len(catalog["copied_files_sha256"]),original_count=len(catalog["original_files_sha256"]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--old-build-root",required=True,type=Path)
    args=p.parse_args()
    root,old=args.root.resolve(),args.old_build_root.resolve(strict=True)
    base=Path("/root/autodl-tmp/model_selection_v2_20261002")
    if root.exists() or not root.is_relative_to(base) or not old.is_relative_to(base) or old==root:
        raise ValueError("isolated fresh evaluation-only build root required")
    old_launch=read(old/"controller_launch.json")
    runtime=Path(old_launch["frozen_runtime"]).resolve(strict=True)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["PYTHONPATH"]=str(runtime)
    sys.path.insert(0,str(runtime))
    root.mkdir(parents=True)
    admission=admit_old_pilot(old,root/"extension_admission.json")
    plan=extend(old,root/"plan")
    _,sources=validate_extension(root/"plan"/"generation_plan.json")
    payload=catalog_payload(sources)
    save(root/"catalog_payload_admission.json",payload)
    inventory=dict(root=str(runtime),files={str(path.relative_to(runtime)):sha(path) for path in sorted(runtime.rglob("*.py"))})
    if inventory!=read(old/"frozen_source_inventory.json"):
        raise ValueError("historical frozen runtime inventory changed")
    save(root/"frozen_source_inventory.json",inventory)
    source_names=("heldout_extend_build.py","heldout_extend_run.py","heldout_extend_plan.py","heldout_adopt.py",
                  "heldout_build.py","heldout_run.py","heldout_plan.py","heldout_augment.py")
    files=[Path(__file__).with_name(name) for name in source_names]
    files += [root/name for name in ("extension_admission.json","catalog_payload_admission.json","frozen_source_inventory.json",
                                    "plan/generation_plan.json","plan/sources.json")]
    files += [Path(sources[key+"_path"]) for key in ("catalog","parent_plan","profile")]
    bindings={str(path.resolve()):sha(path) for path in files}
    save(root/"controller_launch.json",dict(schema="mixed-heldout-extension-controller/1",pid=os.getpid(),
         started_unix=time.time(),old_build_root=str(old),frozen_runtime=str(runtime),bindings=bindings,
         extension_admission_path=str(root/"extension_admission.json"),extension_admission_sha256=sha(root/"extension_admission.json"),
         cuda_visible_devices="",workers=2,threads_per_worker=1,models_loaded=False,original_outputs_changed=False,
         total_registered_reserves=12,old_pixel_reconstruction_not_repeated=True))
    runner=str(Path(__file__).with_name("heldout_extend_run.py"))
    for phase in ("pilot","full"):
        command=[sys.executable,runner,"--root",str(root)]+(["--pilot"] if phase=="pilot" else [])
        started=time.time()
        result=subprocess.run(command,cwd=root,env=os.environ.copy())
        save(root/(phase+"_actual_return.json"),dict(command=command,returncode=result.returncode,started_unix=started,
             finished_unix=time.time(),bindings=bindings))
        if result.returncode:raise SystemExit(result.returncode)
        folder="pilot_receipts" if phase=="pilot" else "build_receipts"
        want="pilot_passed" if phase=="pilot" else "complete"
        completed={role:read(root/folder/role/"complete.json") for role in ("cal","select")}
        if any(row["status"]!=want for row in completed.values()):
            save(root/(phase+"_shortfall.json"),dict(status="shortfall",stage=phase,no_next_stage_launched=True,
                 roles={role:{key:r[key] for key in ("status","planned_quotas","admitted_pairs")} for role,r in completed.items()}))
            raise SystemExit(2)
        if any(sha(path)!=digest for path,digest in bindings.items()):raise ValueError("bound source/input changed")
        check_admission(admission)
        if catalog_payload(sources)!=payload:raise ValueError("native catalog payload changed during run")
        if any(sha(runtime/rel)!=digest for rel,digest in inventory["files"].items()):raise ValueError("runtime source changed")
        print(json.dumps(dict(stage=phase,status="passed",actual_return=0)),flush=True)
    save(root/"controller_complete.json",dict(status="curriculum_complete_pending_combined_release_audit",bindings=bindings,
         finished_unix=time.time(),no_model_selection=True,no_training=True,no_gpu=True,desired_curriculum_pairs_per_role=2880,
         catalog_payload_after=dict(status="unchanged",admission=dict(path=str(root/"catalog_payload_admission.json"),
                 sha256=sha(root/"catalog_payload_admission.json"))),reserve_count=12))


if __name__=="__main__":main()
