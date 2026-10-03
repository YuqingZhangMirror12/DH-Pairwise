import unittest
import json
from pathlib import Path
import tempfile
from .heldout_publish import validate_population, population, verify_normalization_seals, sha


def examples():
    out=[]
    for stage in ("v17_filtered","v17.5","v18","strict_straight"):
        for gen in (("straight_strip",) if stage=="strict_straight" else ("Gen2","Gen3","Gen4","Gen5")):
            for label in (False,True):
                for number in range(160 if stage=="strict_straight" else 120):
                    key=f"{stage}/{gen}/{label}/{number}"
                    out.append(dict(stage=stage,generator=gen,label=label,pair_id=key,model_tensors_sha256=key,
                                    parent_ids=["mother"],base_pair_ids=["base"],fragment_ids=["a","b"],donor_parent_ids=[]))
    return out


class PublishTests(unittest.TestCase):
    def test_exact_population_is_required(self):
        rows=examples()
        validate_population(rows)
        self.assertEqual(population(rows)["rows"],3200)
        self.assertEqual(population(rows)["independent_base_pair_count"],1)
        with self.assertRaises(ValueError):validate_population(rows[:-1])

    def test_duplicate_actual_input_or_cell_relabel_cannot_hide_behind_new_id(self):
        for key,value in (("model_tensors_sha256",examples()[1]["model_tensors_sha256"]),
                          ("pair_id",examples()[1]["pair_id"]),("generator","Gen5")):
            rows=examples();rows[0][key]=value
            with self.assertRaises(ValueError):validate_population(rows)

    def test_actual_normalization_seals_bind_execution_and_both_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            def write(name,value):
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
                return dict(path=str(path),sha256=sha(path))
            paths={role:{} for role in ("cal","select")}
            seals={}
            for kind,total in (("curriculum",5760),("strict",640)):
                source=write(kind+"/source.py",{"immutable":"normalizer"})
                input_ref=write("inputs/"+kind+"/controller_complete.json",{"input":kind})
                metadata={"original_completion":input_ref} if kind=="strict" else dict(
                    build_completion=input_ref,build_root=str(Path(input_ref["path"]).parent))
                argv=(["--completion-sha",input_ref["sha256"]] if kind=="strict" else
                      ["--build-root",metadata["build_root"],"--build-completion-sha",input_ref["sha256"]])
                launch=dict(command=["python",source["path"],"--root",str(root/kind),*argv,"--worker"],source=source)
                roles=[]
                for role in paths:
                    ref=write(kind+"/"+role+".json",{"output":kind+role})
                    paths[role][kind]=ref["path"]
                    roles.append(dict(role=role,rows=total//2,expanded=ref))
                seals[kind]=dict(
                    completion=write(kind+"/complete.json",dict(
                        schema=kind+"-normalization-completion/1",
                        status="verified_"+kind+"_component_only",rows=total,roles=roles,
                        bound_source_sha256={source["path"]:source["sha256"]},**metadata)),
                    launch=write(kind+"/launch.json",launch),
                    actual_return=write(kind+"/actual.json",dict(**launch,returncode=0)))
            verify_normalization_seals(seals,paths)
            with self.assertRaises(ValueError):
                verify_normalization_seals({"strict":seals["strict"]},paths)
            altered={role:dict(v) for role,v in paths.items()}
            altered["select"]["strict"]=paths["cal"]["strict"]
            with self.assertRaises(ValueError):verify_normalization_seals(seals,altered)
            path=Path(seals["strict"]["actual_return"]["path"])
            original=path.read_text()
            actual=json.loads(original)
            for key,value in (("returncode",1),("command",["different-command"])):
                path.write_text(json.dumps(dict(actual,**{key:value})))
                seals["strict"]["actual_return"]["sha256"]=sha(path)
                with self.assertRaises(ValueError):verify_normalization_seals(seals,paths)
            path.write_text(original)
            seals["strict"]["actual_return"]["sha256"]=sha(path)
            launch_path=Path(seals["strict"]["launch"]["path"])
            original_launch=launch_path.read_text()
            wrong=json.loads(original_launch)
            wrong["command"][1]="unrelated_worker.py"
            launch_path.write_text(json.dumps(wrong))
            path.write_text(json.dumps(dict(wrong,returncode=0)))
            seals["strict"]["launch"]["sha256"]=sha(launch_path)
            seals["strict"]["actual_return"]["sha256"]=sha(path)
            with self.assertRaises(ValueError):verify_normalization_seals(seals,paths)
            launch_path.write_text(original_launch)
            path.write_text(original)
            seals["strict"]["launch"]["sha256"]=sha(launch_path)
            seals["strict"]["actual_return"]["sha256"]=sha(path)
            complete=Path(seals["strict"]["completion"]["path"])
            original_complete=complete.read_text()
            malformed=json.loads(original_complete)
            del malformed["roles"][0]["expanded"]["sha256"]
            complete.write_text(json.dumps(malformed))
            seals["strict"]["completion"]["sha256"]=sha(complete)
            with self.assertRaises(ValueError):verify_normalization_seals(seals,paths)
            complete.write_text(original_complete)
            seals["strict"]["completion"]["sha256"]=sha(complete)
            Path(paths["select"]["strict"]).write_text("changed output")
            with self.assertRaises(ValueError):verify_normalization_seals(seals,paths)


if __name__=="__main__":unittest.main()
