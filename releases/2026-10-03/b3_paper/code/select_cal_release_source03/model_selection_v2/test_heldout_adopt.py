"""Synthetic receipt tests only; no frozen pixels, generation, SSH or GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from . import heldout_adopt as adopt
from . import heldout_augment as augmentation


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def blob(path, value="synthetic immutable bytes"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def fixture(directory):
    outer = Path(directory).resolve()
    root = outer / "old_build"
    root.mkdir()
    scripts = outer / "old_source"
    for name in ("heldout_build.py", "heldout_plan.py", "heldout_run.py"):
        blob(scripts / name)
    blob(scripts / "heldout_augment.py", Path(augmentation.__file__).read_text())
    runtime = outer / "runtime"
    module = blob(runtime / "native.py")
    inventory = save(root / "frozen_source_inventory.json", dict(root=str(runtime), files={"native.py": adopt.sha(module)}))
    external = {key: save(outer / (key + ".json"), {"immutable": key}) for key in ("catalog", "parents", "profile")}
    sources = dict(splits={})
    for prefix, name in (("catalog", "catalog"), ("parent_plan", "parents"), ("profile", "profile")):
        sources[prefix + "_path"] = str(external[name])
        sources[prefix + "_sha256"] = adopt.sha(external[name])
    tasks = []
    for role in adopt.ROLES:
        bank = root / "plan" / "banks" / role
        metadata = save(bank / "bank.json", {"role": role})
        profiles = blob(bank / "profiles.npz")
        spec = dict(positive=[], negative=[], schedule={"recipes": []}, slot_generators=[],
                    donor_bank=dict(path=str(bank), metadata_sha256=adopt.sha(metadata), profiles_sha256=adopt.sha(profiles)))
        for gen_index, gen in enumerate(("Gen2", "Gen3", "Gen4", "Gen5")):
            for reserve in range(3):
                for quota in range(120):
                    slot = gen_index * 360 + reserve * 120 + quota
                    recipe = ("clean", "clean", "gaps_weak", "gaps_weak")[quota] if quota < 4 else "partial"
                    pair_ids = [f"{role}/{gen}/p{quota}", f"{role}/{gen}/n{quota}"]
                    spec["positive"].append(dict(pair_id=pair_ids[0]))
                    spec["negative"].append(dict(pair_id=pair_ids[1]))
                    spec["schedule"]["recipes"].append(recipe)
                    spec["slot_generators"].append(gen)
                    for stage in adopt.STAGES:
                        tasks.append(dict(role=role, stage=stage, generator=gen, slot=slot,
                            master_seed=1, reserve_index=reserve, quota_slot=quota, recipe=recipe,
                            base_pair_ids=pair_ids, size_class="smaller" if quota % 2 == 0 else "larger",
                            mode="one", k=1, trim_target=.3))
        sources["splits"][role] = spec
    source_path = save(root / "plan" / "sources.json", sources)
    plan_path = save(root / "plan" / "generation_plan.json", dict(tasks=tasks,
                       source_plan_path=str(source_path), source_plan_sha256=adopt.sha(source_path)))
    plan_sha = adopt.sha(plan_path)
    indexed = {(t["role"], t["stage"], t["slot"]): t for t in tasks}
    bindings = {str(path): adopt.sha(path) for path in [*sorted(scripts.glob("*.py")), *external.values(), inventory]}
    save(root / "controller_launch.json", dict(bindings=bindings, frozen_runtime=str(runtime),
                                               cuda_visible_devices="", models_loaded=False))
    plan_command = [sys.executable, str(scripts / "heldout_build.py"), "--root", str(root),
                    "--catalog", str(external["catalog"]), "--parents", str(external["parents"]),
                    "--profile", str(external["profile"]), "--frozen-runtime", str(runtime), "--phase", "plan"]
    pilot_command = [sys.executable, str(scripts / "heldout_run.py"), "--root", str(root), "--pilot"]
    save(root / "plan_actual_return.json", dict(command=plan_command, bindings=bindings,
                                               returncode=0, started_unix=1, finished_unix=2))
    save(root / "pilot_actual_return.json", dict(command=pilot_command, bindings=bindings,
                                                returncode=0, started_unix=3, finished_unix=4))
    short_roles = {}
    for role in adopt.ROLES:
        base_audit, records, failures = {}, [], []
        def baseline(slot):
            base_root = root / "baseline" / role
            if str(slot) in base_audit:
                return json.loads((base_root / "groups" / f"{slot:05d}.json").read_text())
            task = indexed[(role, adopt.STAGES[0], slot)]
            entries = []
            for ordinal in (0, 1):
                paths = dict(artifact_path=f"samples/{slot:05d}_{ordinal}.npz",
                             weather_artifact=f"weather/{slot:05d}_{ordinal}.npz",
                             background_artifact=f"background/{slot:05d}_{ordinal}.npz",
                             latent_seam_artifact=f"latent/{slot:05d}_{ordinal}.npz" if ordinal == 0 else None,
                             target_metadata=str(base_root / "targets" / f"{slot:05d}_{ordinal}.npz"))
                for value in paths.values():
                    if value:
                        blob(Path(value) if Path(value).is_absolute() else base_root / value)
                entries.append(dict(pair_id=f"base-{role}-{slot}-{ordinal}", label=ordinal == 0,
                                    corrosion_recipe=task["recipe"], **paths))
            group = dict(slot=slot, entries=entries)
            path = save(base_root / "groups" / f"{slot:05d}.json", group)
            base_audit[str(slot)] = dict(group_sha256=adopt.sha(path), audit_rows=[dict(
                pair_id=e["pair_id"], recipe=e["corrosion_recipe"], raster_depth_lossless=True) for e in entries])
            return group
        for gen_index, gen in enumerate(("Gen2", "Gen3", "Gen4", "Gen5")):
            for quota in range(4):
                slot = gen_index * 360 + quota
                for stage in adopt.STAGES:
                    if gen == "Gen2" and quota == 0 and stage == "v17_filtered":
                        for reserve in range(3):
                            failed_slot = reserve * 120
                            baseline(failed_slot)
                            task = indexed[(role, stage, failed_slot)]
                            reasons = {"unchanged frozen geometry rejection": 24}
                            save(root / "augmented" / role / stage / "rejections" / f"{failed_slot:05d}.json",
                                 dict(status="rejected", task=task, reasons=reasons, counts_toward_final=False))
                            failures.append(dict(task=task, phase="augmentation", error=reasons))
                        failures.append(dict(quota=[gen, quota, stage], phase="quota_exhausted", error="all candidates rejected"))
                        continue
                    task = indexed[(role, stage, slot)]
                    base = baseline(slot)
                    entries, audits = [], []
                    for ordinal in (0, 1):
                        identity = f"{role}-{stage}-{slot}-{ordinal}"
                        stage_root = root / "augmented" / role / stage
                        row = dict(pair_id=identity, id=identity, label=ordinal == 0,
                                   data_role=role, stage=stage, generator=gen, baseline_slot=slot,
                                   baseline_ordinal=ordinal, source_pair_id=task["base_pair_ids"][ordinal])
                        for field, directory_name, digest_key in (("sample_path", "samples", "sample_sha256"),
                                    ("proof_path", "proof", "proof_sha256"),
                                    ("target_metadata", "targets", "target_metadata_sha256"),
                                    ("unaugmented_sample_path", "unaugmented", "unaugmented_sample_sha256")):
                            path = blob(stage_root / directory_name / f"{slot:05d}_{ordinal}.npz", identity + field)
                            row[field], row[digest_key] = str(path), adopt.sha(path)
                        base_root = root / "baseline" / role
                        row["baseline_sample_path"] = str(base_root / base["entries"][ordinal]["artifact_path"])
                        row["baseline_sample_sha256"] = adopt.sha(row["baseline_sample_path"])
                        row["baseline_group_path"] = str(base_root / "groups" / f"{slot:05d}.json")
                        row["baseline_group_sha256"] = adopt.sha(row["baseline_group_path"])
                        row["latent_seam_artifact"] = None
                        entries.append(row)
                        audits.append(dict(id=identity, status="passed", sample_sha256=row["sample_sha256"], proof_sha256=row["proof_sha256"]))
                    save(root / "augmented" / role / stage / "groups" / f"{slot:05d}.json",
                         dict(status="committed", task=task, plan_sha256=plan_sha, records=entries, audit_rows=audits))
                    records.extend(entries)
        complete = dict(schema="mixed-sim-heldout-role-build/1", role=role, pilot_only=True, status="shortfall",
                        planned_quotas=48, admitted_pairs=len(records), records=records, failures=failures,
                        plan_sha256=plan_sha, gpu_used=False, model_outputs_used=False,
                        base_audit=base_audit, actual_base_groups=len(base_audit),
                        frozen_source_sha256={str(module): adopt.sha(module)})
        save(root / "pilot_receipts" / role / "complete.json", complete)
        short_roles[role] = {key: complete[key] for key in ("status", "planned_quotas", "admitted_pairs")}
    save(root / "pilot_shortfall.json", dict(status="shortfall", stage="pilot", roles=short_roles, no_next_stage_launched=True))
    return root, outer / "admission" / "admitted.json"


class AdoptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.output = fixture(self.temp.name)

    def mutate(self, relative, change):
        path = self.root / relative
        value = json.loads(path.read_text())
        change(value)
        save(path, value)

    def test_success_binds_every_old_file_without_writing_old_root(self):
        before = {str(p): adopt.sha(p) for p in self.root.rglob("*") if p.is_file()}
        value = adopt.admit_old_pilot(self.root, self.output)
        self.assertTrue(adopt.check_admission(value))
        self.assertEqual(value["schema"], adopt.SCHEMA)
        self.assertEqual(len(value["roles"]["cal"]["commits"]), 47)
        self.assertEqual(len(value["roles"]["cal"]["rejections"]), 3)
        self.assertEqual(before, {str(p): adopt.sha(p) for p in self.root.rglob("*") if p.is_file()})
        base = value["roles"]["cal"]["baselines"]["0"]
        self.assertTrue(any("weather" in path for path in base["files"]))
        self.assertTrue(any("targets" in path for path in base["files"]))
        with self.assertRaises(ValueError):
            adopt.admit_old_pilot(self.root, self.output)

    def test_changed_bound_baseline_bytes_fail_recheck(self):
        value = adopt.admit_old_pilot(self.root, self.output)
        path = self.root / "baseline" / "cal" / "weather" / "00000_0.npz"
        path.write_text("changed after admission")
        with self.assertRaises(ValueError):
            adopt.check_admission(value)

    def test_missing_unaudited_optional_artifact_does_not_silently_pass(self):
        (self.root / "baseline" / "cal" / "background" / "00000_0.npz").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            adopt.admit_old_pilot(self.root, self.output)

    def test_wrong_pilot_command_or_false_return_rejected(self):
        path = self.root / "pilot_actual_return.json"
        original = json.loads(path.read_text())
        for changed in (dict(original, returncode=False), dict(original, returncode=1),
                        dict(original, command=original["command"][:-1])):
            save(path, changed)
            with self.assertRaises(ValueError):
                adopt.admit_old_pilot(self.root, self.output)

    def test_unrecorded_runtime_source_rejected(self):
        blob(Path(self.temp.name) / "runtime" / "unbound.py")
        with self.assertRaises(ValueError):
            adopt.admit_old_pilot(self.root, self.output)

    def test_actual_full_build_is_never_adopted_as_pilot(self):
        save(self.root / "full_actual_return.json", dict(returncode=0))
        with self.assertRaises(ValueError):
            adopt.admit_old_pilot(self.root, self.output)

    def test_missing_commit_audit_or_wrong_receipt_record_rejected(self):
        path = self.root / "augmented" / "cal" / "v17.5" / "groups" / "00000.json"
        value = json.loads(path.read_text())
        value["audit_rows"][0]["status"] = "failed"
        save(path, value)
        with self.assertRaises(ValueError):
            adopt.admit_old_pilot(self.root, self.output)

    def test_duplicate_input_rejection_not_imported_as_exclusion(self):
        def change(value):
            value["failures"][0] = dict(value["failures"][0], phase="duplicate_model_input", error="duplicate")
        self.mutate("pilot_receipts/cal/complete.json", change)
        value = adopt.admit_old_pilot(self.root, self.output)
        self.assertEqual(len(value["roles"]["cal"]["rejections"]), 2)
        self.assertTrue(value["pilot_acceptance_does_not_reserve_final_duplicate_priority"])

    def test_partial_population_and_fabricated_baseline_audit_rejected(self):
        self.mutate("pilot_receipts/cal/complete.json", lambda value: value["records"].pop())
        with self.assertRaises(ValueError):
            adopt.admit_old_pilot(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
