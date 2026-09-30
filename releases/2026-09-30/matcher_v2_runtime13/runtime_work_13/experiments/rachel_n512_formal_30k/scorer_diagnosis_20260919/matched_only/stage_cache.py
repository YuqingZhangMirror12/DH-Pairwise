"""One-time CPU candidate groups: small side-cache, never per-epoch O(C²).

Stores original candidate-slot masks, not copied features/labels/GT. One file
set contains the unrefined best seed (slot0) and up to five separate refined
modes (slots1..5). Only >=3-edge groups are scorer-eligible. Geometry validity
is NOT a positive label. Probes and unfinished caches cannot enter training.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import candidate_groups, data

SCHEMA = "matched-only-stage-side-cache/1"
STAGES = {"edge_seed": (0, 1), "edge_multi": (1, 6)}
STATUSES = {"absent": 0, "ok": 1, "proposal": 2, "ambiguous_equal_modes": 3,
            "insufficient_inliers": 4, "no_candidates": 5}
STATUS_NAMES = {value:key for key,value in STATUSES.items()}
CONFIG = dict(radius=10., max_modes=5, min_inliers=3, replay_atol=1e-4)
ARRAYS = dict(candidate_inliers=("bool",(6,512)), translation_rc=("float32",(6,2)),
    present=("bool",(6,)), eligible=("bool",(6,)), ranks=("int16",(6,)),
    seed_candidate_id=("int16",(6,)), inlier_count=("int16",(6,)),
    status_code=("int8",(6,)), production_status_code=("int8",()), ready=("bool",()))


@dataclass(frozen=True)
class StageGroups:
    stage: str
    candidate_inliers: torch.Tensor  # B,G,C, original candidate slot IDs
    translation_rc: torch.Tensor
    present: torch.Tensor
    eligible: torch.Tensor
    ranks: torch.Tensor
    seed_candidate_id: torch.Tensor
    inlier_count: torch.Tensor
    status_code: torch.Tensor


def save(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def implementation():
    return dict(stage_cache=data.sha(__file__), candidate_groups=data.sha(candidate_groups.__file__))


def derive_row(arrays, row):
    """No target/label lookup: only the frozen cache's predicted geometry."""
    return candidate_groups.build_candidate_groups(
        points_a_rc=arrays["points_a"][row], points_b_rc=arrays["points_b"][row],
        valid_a=arrays["valid_a"][row], valid_b=arrays["valid_b"][row],
        candidate_indices=arrays["candidate_indices"][row],
        candidate_valid=arrays["candidate_valid"][row], candidate_weights=arrays["candidate_weights"][row],
        final_inliers=arrays["candidate_inliers"][row],
        final_translation_rc=arrays["translation_a_to_b_rc"][row],
        layout_valid=arrays["layout_valid"][row], **CONFIG)


def prepare(source, root, *, limit=None):
    """Finite one-time CPU generation; does not load or run a neural network."""
    root=Path(root).resolve()
    if root==source.root or root in source.root.parents or source.root in root.parents:
        raise ValueError("side-cache must be outside original cache tree")
    count=len(source) if limit is None else limit
    if type(count) is not int or not 1 <= count <= len(source):
        raise ValueError("invalid stage side-cache limit")
    root.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema=SCHEMA,status="running",split=source.split,pair_count=count,
        formal_training_eligible=limit is None, source_binding=source.binding,
        implementation_sha256=implementation(), candidate_groups_schema=candidate_groups.SCHEMA,
        config=CONFIG, stages={k:list(v) for k,v in STAGES.items()}, status_codes=STATUSES,
        arrays={k:dict(dtype=d,shape=[count,*tail]) for k,(d,tail) in ARRAYS.items()},
        completed_pairs=0, gt_used=False, labels_used=False, neural_inference=False,
        seed_definition="raw best-support seed; radius10 support BEFORE mean refinement",
        multi_definition="up to5 separate refined modes; original first refined mode replays valid final",
        eligibility="present and at least3 inlier edges; ambiguity/proposal statuses remain observable",
        output_use="side-cache only; never a pair label or a layout-correctness label")
    save(root/"protocol.json",protocol)
    outputs={k:np.lib.format.open_memmap(root/(k+".npy"),mode="w+",dtype=d,shape=(count,*tail))
             for k,(d,tail) in ARRAYS.items()}
    for array in outputs.values(): array[:]=0
    outputs["translation_rc"][:]=np.nan
    outputs["seed_candidate_id"][:]=-1
    started=time.perf_counter()
    try:
        for row in range(count):
            groups=derive_row(source.arrays,row)
            outputs["production_status_code"][row]=STATUSES[groups.production_status]
            views=[groups.single_seed]+list(groups.multi_modes)+[None]*(5-len(groups.multi_modes))
            for slot,group in enumerate(views):
                if group is None: continue
                outputs["candidate_inliers"][row,slot,group.candidate_ids]=True
                outputs["translation_rc"][row,slot]=group.translation_rc
                outputs["present"][row,slot]=True
                outputs["eligible"][row,slot]=group.inlier_count>=CONFIG["min_inliers"]
                outputs["ranks"][row,slot]=group.rank
                outputs["seed_candidate_id"][row,slot]=group.seed_candidate_id
                outputs["inlier_count"][row,slot]=group.inlier_count
                outputs["status_code"][row,slot]=STATUSES[group.status]
            outputs["ready"][row]=True
            protocol["completed_pairs"]=row+1
            if (row+1)%512==0:
                protocol["elapsed_s"]=time.perf_counter()-started
                save(root/"protocol.json",protocol)
        for array in outputs.values(): array.flush()
        protocol.update(status="complete",elapsed_s=time.perf_counter()-started,
            array_sha256={k:data.sha(root/(k+".npy")) for k in ARRAYS},
            eligible_pairs={stage:int(outputs["eligible"][:,start:end].any(1).sum())
                            for stage,(start,end) in STAGES.items()},
            total_groups={stage:int(outputs["eligible"][:,start:end].sum())
                          for stage,(start,end) in STAGES.items()},
            output_bytes=sum((root/(k+".npy")).stat().st_size for k in ARRAYS))
    except BaseException as error:
        protocol.update(status="failed",error=repr(error),elapsed_s=time.perf_counter()-started)
        raise
    finally:
        save(root/"protocol.json",protocol)
    return protocol


class StageCache:
    def __init__(self,root,source):
        self.root=Path(root).resolve(strict=True)
        p=json.loads((self.root/"protocol.json").read_text())
        if (p.get("schema")!=SCHEMA or p.get("status")!="complete"
                or p.get("formal_training_eligible") is not True
                or p.get("split")!=source.split or p.get("pair_count")!=len(source)
                or p.get("completed_pairs")!=len(source) or p.get("source_binding")!=source.binding
                or p.get("implementation_sha256")!=implementation()
                or p.get("config")!=CONFIG or p.get("stages")!={k:list(v) for k,v in STAGES.items()}
                or p.get("status_codes")!=STATUSES or p.get("gt_used") is not False
                or p.get("labels_used") is not False or p.get("neural_inference") is not False):
            raise ValueError("stage cache must be complete formal, exact-source-bound and target-blind")
        self.arrays={}
        for name,(dtype,tail) in ARRAYS.items():
            path=self.root/(name+".npy")
            if (p.get("arrays",{}).get(name)!=dict(dtype=dtype,shape=[len(source),*tail])
                    or p.get("array_sha256",{}).get(name)!=data.sha(path)):
                raise ValueError("stage cache array schema/hash changed: "+name)
            value=np.load(path,mmap_mode="r",allow_pickle=False)
            if value.shape!=(len(source),*tail) or value.dtype!=np.dtype(dtype):
                raise ValueError("stage cache array header changed: "+name)
            self.arrays[name]=value
        a=self.arrays
        if (not a["ready"].all() or not np.array_equal(a["inlier_count"],a["candidate_inliers"].sum(-1))
                or not np.array_equal(a["eligible"],a["present"] & (a["inlier_count"]>=3))
                or not np.isfinite(a["translation_rc"][a["present"]]).all()
                or (a["candidate_inliers"] & ~source.arrays["candidate_valid"][:,None,:]).any()):
            raise ValueError("stage cache readiness/membership/eligibility inconsistent")
        self.binding=dict(root=str(self.root),protocol_sha256=data.sha(self.root/"protocol.json"),
            array_sha256=p["array_sha256"],implementation_sha256=p["implementation_sha256"],
            schema=SCHEMA,config=CONFIG,source_binding=source.binding)

    def batch(self,indices,stage,device="cpu"):
        if stage not in STAGES: raise ValueError("unknown candidate stage")
        ids=np.asarray(indices,dtype=np.int64)
        if ids.ndim!=1 or not len(ids) or (ids<0).any() or (ids>=len(self.arrays["ready"])).any():
            raise ValueError("stage row indices out of bounds")
        start,end=STAGES[stage]
        def get(key):
            return torch.from_numpy(np.array(self.arrays[key][ids,start:end],copy=True)).to(device)
        return StageGroups(stage,**{key:get(key) for key in (
            "candidate_inliers","translation_rc","present","eligible","ranks",
            "seed_candidate_id","inlier_count","status_code")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache",required=True)
    parser.add_argument("--split",choices=("train","val"),required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--limit",type=int,help="CPU timing probe only; ineligible for formal training")
    args=parser.parse_args()
    source=data.FormalCache(args.base_cache,args.split)
    print(json.dumps(prepare(source,args.output,limit=args.limit),sort_keys=True))
