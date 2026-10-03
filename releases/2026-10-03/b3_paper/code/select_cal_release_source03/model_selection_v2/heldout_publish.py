"""Seal reviewed data components, without selecting or running any model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from .heldout_augment import tensor_identity
    from .protocol import ROW_FIELDS, MANIFEST_SCHEMA, _validate_manifest
except ImportError:
    from heldout_augment import tensor_identity
    from protocol import ROW_FIELDS, MANIFEST_SCHEMA, _validate_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(path=str(path.resolve()), sha256=sha(path))


def verify_receipt_tree(value):
    """Follow explicit receipt references only, not arbitrary source strings."""
    count=0
    if isinstance(value,dict):
        if isinstance(value.get("path"),str) and isinstance(value.get("sha256"),str):
            path=Path(value["path"])
            if not path.is_absolute() or not path.is_file() or sha(path)!=value["sha256"]:
                raise ValueError("component bound receipt changed: "+str(path))
            count+=1
        for child in value.values():count+=verify_receipt_tree(child)
    elif isinstance(value,list):
        for child in value:count+=verify_receipt_tree(child)
    return count


def verify_normalization_seals(seals, paths):
    if not isinstance(seals,dict) or set(seals)!={"curriculum","strict"}:
        raise ValueError("both actual normalization launch/return/completion seals required")
    for kind, bundle in seals.items():
        if set(bundle)!={"completion","launch","actual_return"}:
            raise ValueError("fixed normalization receipt slots required")
        if verify_receipt_tree(bundle)!=3:
            raise ValueError("three explicit immutable normalization receipts required")
        done,launch,actual=(read(bundle[key]["path"]) for key in ("completion","launch","actual_return"))
        expected=5760 if kind=="curriculum" else 640
        if (done.get("schema")!=kind+"-normalization-completion/1" or
            done.get("status")!="verified_"+kind+"_component_only" or done.get("rows")!=expected):
            raise ValueError("actual normalizer did not finish the expected component")
        if (actual.get("returncode")!=0 or actual.get("command")!=launch.get("command") or
            actual.get("source")!=launch.get("source")):
            raise ValueError("normalizer actual return/launch identity mismatch")
        executed=launch.get("source",{})
        if set(executed)!={"path","sha256"} or verify_receipt_tree(executed)!=1:
            raise ValueError("executed normalizer source needs an exact file receipt")
        output_root=Path(bundle["completion"]["path"]).resolve().parent
        if any(Path(bundle[k]["path"]).resolve().parent!=output_root for k in ("launch","actual_return")):
            raise ValueError("normalizer receipts belong to different invocation roots")
        command=launch.get("command")
        if not isinstance(command,list) or len(command)<2:
            raise ValueError("normalizer worker command missing")
        if kind=="strict":
            input_ref=done.get("original_completion",{})
            expected_arguments=["--root",str(output_root),"--completion-sha",input_ref.get("sha256"),"--worker"]
        else:
            input_ref=done.get("build_completion",{})
            build_root=done.get("build_root")
            if not isinstance(build_root,str) or Path(input_ref.get("path","")).resolve()!=Path(build_root).resolve()/"controller_complete.json":
                raise ValueError("normalizer build-root/input identity missing")
            expected_arguments=["--root",str(output_root),"--build-root",build_root,
                                "--build-completion-sha",input_ref.get("sha256"),"--worker"]
        if (set(input_ref)!={"path","sha256"} or verify_receipt_tree(input_ref)!=1 or
            command[1]!=executed["path"] or command[2:]!=expected_arguments):
            raise ValueError("normalizer worker argv does not bind its actual input and output")
        sources=done.get("bound_source_sha256",{})
        if not sources or sources.get(actual["source"]["path"])!=actual["source"]["sha256"]:
            raise ValueError("normalizer completion not bound to executed source")
        if any(sha(path)!=digest for path,digest in sources.items()):
            raise ValueError("bound normalizer implementation changed")
        roles=done.get("roles",[])
        if len(roles)!=2 or {r.get("role") for r in roles}!={"cal","select"}:
            raise ValueError("complete normalizer roles missing")
        for row in roles:
            reference=row["expanded"]
            role=row["role"]
            if (not isinstance(reference,dict) or set(reference)!={"path","sha256"} or
                not isinstance(reference["path"],str) or not reference["path"] or
                not isinstance(reference["sha256"],str) or len(reference["sha256"])!=64):
                raise ValueError("complete normalized output requires a path and SHA256")
            if row["rows"]!=expected//2 or Path(reference["path"]).resolve()!=Path(paths[role][kind]).resolve():
                raise ValueError("normalized output differs from admitted population/path")
            if Path(reference["path"]).resolve().parent!=output_root:
                raise ValueError("normalizer output does not belong to executed output root")
            if verify_receipt_tree(reference)!=1:
                raise ValueError("normalized output has no exact completion hash binding")


def population(rows):
    return dict(rows=len(rows), positives=sum(r["label"] for r in rows),
         negatives=sum(not r["label"] for r in rows),
         primary_parent_count=len({s for r in rows for s in r["parent_ids"]}),
         primary_fragment_count=len({s for r in rows for s in r["fragment_ids"]}),
         independent_base_pair_count=len({s for r in rows for s in r["base_pair_ids"]}),
         donor_parent_count=len({s for r in rows for s in r["donor_parent_ids"]}),
         model_input_content_count=len({r["model_tensors_sha256"] for r in rows}))


def validate_population(rows):
    expected = {(stage,gen,label):120 for stage in ("v17_filtered","v17.5","v18")
                for gen in ("Gen2","Gen3","Gen4","Gen5") for label in (False,True)}
    expected.update({("strict_straight","straight_strip",label):160 for label in (False,True)})
    if len(rows) != 3200 or Counter((r["stage"],r["generator"],r["label"]) for r in rows) != Counter(expected):
        raise ValueError("actual final fold differs from preregistered 3200 population")
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate final Pair ID")
    if len({r["model_tensors_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate model input, despite distinct IDs/archive metadata")


def publish(root, parent_plan_path, paths, load_sample, seals):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("preserve existing release; new release directory required")
    verify_normalization_seals(seals,paths)
    parent_plan = read(parent_plan_path)
    allowed = {role:{r["family"] for r in parent_plan["folds"][role]} for role in ("cal","select")}
    forbidden = set().union(*map(set,parent_plan["exclusion_families"].values()),
                             set(parent_plan.get("hard_exclusions",{})))
    if allowed["cal"] & allowed["select"] or (allowed["cal"]|allowed["select"]) & forbidden:
        raise ValueError("new role source families overlap forbidden originals")
    bindings = {"parent_plan":dict(path=str(Path(parent_plan_path).resolve()),sha256=sha(parent_plan_path))}
    bindings["normalization_seals"]=seals
    folds, summaries = {}, {}
    for role in ("cal","select"):
        components=[]
        for kind in ("curriculum","strict"):
            path=Path(paths[role][kind]).resolve()
            value=read(path)
            if value.get("status") != "verified_component_only" or value.get("role") != role:
                raise ValueError("not an admitted native component: "+str(path))
            if verify_receipt_tree(value.get("bound_receipts",{}))==0:
                raise ValueError("component has no verified source/completion bindings")
            wanted=2880 if kind=="curriculum" else 320
            if len(value["entries"]) != wanted:
                raise ValueError("component population differs")
            bindings[role+"_"+kind]=dict(path=str(path),sha256=sha(path))
            components.extend(value["entries"])
        for entry in components:
            sample_path=Path(entry["sample_path"])
            if not sample_path.is_absolute() or sha(sample_path)!=entry["sample_sha256"]:
                raise ValueError("actual sample differs from normalized component")
            sample,report=load_sample(sample_path)
            if sample.pair_id!=entry["pair_id"] or bool(sample.label)!=entry["label"]:
                raise ValueError("actual final ID/label mismatch")
            if report.get("data_role",report.get("split"))!=role:
                raise ValueError("actual archive belongs to a different decision fold")
            content=tensor_identity(sample)
            if entry.get("model_tensors_sha256",content)!=content:
                raise ValueError("normalized actual tensor content changed")
            entry["model_tensors_sha256"]=content
            entry["supervised_tensors_sha256"]=tensor_identity(sample,include_supervision=True)
            families=set(entry["parent_ids"])|set(entry["donor_parent_ids"])
            families={f for f in families if not f.startswith("procedural-strip/")}
            if not families<=allowed[role] or families&forbidden:
                raise ValueError("primary/attempted donor parent outside role isolation")
            if not isinstance(entry.get("recipe"),str) or not entry["recipe"]:
                raise ValueError("native collator requires explicit recipe")
        components.sort(key=lambda r:(r["stage"],r["generator"],r["pair_id"]))
        validate_population(components)
        lineage=dict(schema=MANIFEST_SCHEMA,role=role,
                     entries=[{key:r[key] for key in ROW_FIELDS} for r in components])
        _validate_manifest(lineage)
        folds[role]=dict(entries=components,lineage=lineage)
        summaries[role]=dict(**population(components),
            strata={stage:{gen:population([r for r in components if r["stage"]==stage and r["generator"]==gen])
                           for gen in sorted({r["generator"] for r in components if r["stage"]==stage})}
                    for stage in ("v17_filtered","v17.5","v18","strict_straight")},
            independent_bases_are_not_augmentation_views=True)
    overlaps={}
    for identity in ("pair_id","sample_sha256","model_tensors_sha256","supervised_tensors_sha256"):
        intersection={r[identity] for r in folds["cal"]["entries"]}&{r[identity] for r in folds["select"]["entries"]}
        overlaps[identity]=len(intersection)
    for identity in ("parent_ids","base_pair_ids","fragment_ids"):
        identities=[{s for r in folds[role]["entries"] for field in (identity,"donor_"+identity) for s in r[field]}
                    for role in ("cal","select")]
        overlaps[identity]=len(identities[0]&identities[1])
    if any(overlaps.values()):
        raise ValueError("CAL/SELECT overlap: "+json.dumps(overlaps))
    verify_receipt_tree(bindings)
    verify_normalization_seals(seals,paths)
    outputs={}
    for role in ("cal","select"):
        manifest=dict(schema_version="mixed-simulation-heldout/1",split=role,real_used=False,test_used=False,
                      train_used=False,entries=folds[role]["entries"],summary=summaries[role],
                      original_training_or_test_modified=False,checkpoint_selection_performed=False)
        outputs[role]=dict(manifest=save(root/role/(role+".json"),manifest),
                           lineage=save(root/role/"lineage.json",folds[role]["lineage"]))
    verification=dict(schema="mixed-heldout-data-release/1",status="complete_for_data_review",
         rows=6400,folds=summaries,cross_role_overlaps=overlaps,outputs=outputs,bindings=bindings,
         model_inference=False,checkpoint_selected=False,head_training_started=False,
         historical_gen23_base_edge_donor_provenance_complete=False,
         frozen_label_semantics_not_repaired=True,
         note="Dataset complete is not approval to run checkpoint selection. Await independent data review.")
    return save(root/"verification.json",verification)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--parents",required=True)
    p.add_argument("--normalization-seals",required=True)
    for role in ("cal","select"):
        for kind in ("curriculum","strict"):p.add_argument("--"+role+"-"+kind,required=True)
    a=p.parse_args()
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    paths={role:{kind:getattr(a,role+"_"+kind) for kind in ("curriculum","strict")} for role in ("cal","select")}
    print(json.dumps(publish(a.root,a.parents,paths,load_sample,read(a.normalization_seals))),flush=True)


if __name__=="__main__":main()
