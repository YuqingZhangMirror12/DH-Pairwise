"""Read-only admission of one frozen, shortfall-only curriculum pilot.

This does not replay pixel generation or retrofit old receipts.  Hashes absent
from an old receipt are explicitly established now, in a separate admission.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

try:
    from . import heldout_augment as augmentation
except ImportError:
    import heldout_augment as augmentation

SCHEMA = "mixed-heldout-old-pilot-admission/1"
ROLES = ("cal", "select")
STAGES = ("v17_filtered", "v17.5", "v18")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_task(task):
    return json.dumps(task, sort_keys=True, separators=(",", ":"))


def _path(value):
    path = Path(value)
    need(path.is_absolute(), "absolute artifact identity required")
    return path.resolve(strict=True)


class _Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = _path(path)
        need(path.is_file(), "bound artifact is not a file")
        value = sha(path)
        need(expected is None or value == expected, "bound artifact changed: " + str(path))
        need(str(path) not in self.files or self.files[str(path)] == value,
             "artifact changed during admission: " + str(path))
        self.files[str(path)] = value
        return dict(path=str(path), sha256=value)

    def read(self, path, expected=None):
        reference = self.bind(path, expected)
        return json.loads(Path(reference["path"]).read_text()), reference


def _full_absent(root):
    for relative in ("full_actual_return.json", "full_shortfall.json", "controller_complete.json", "build_receipts"):
        need(not (root / relative).exists(), "full build already occurred: " + relative)


def _source_bindings(root, ev):
    launch, launch_ref = ev.read(root / "controller_launch.json")
    need(launch.get("cuda_visible_devices") == "" and launch.get("models_loaded") is False,
         "old controller was not an evaluation-data-only CPU launch")
    bindings = launch.get("bindings")
    need(isinstance(bindings, dict) and bool(bindings), "controller bindings missing")
    for path, digest in bindings.items():
        ev.bind(path, digest)
    modules = {}
    for name in ("heldout_build.py", "heldout_plan.py", "heldout_run.py", "heldout_augment.py"):
        found = [path for path in bindings if Path(path).name == name]
        need(len(found) == 1, "unique frozen controller module missing: " + name)
        modules[name] = found[0]
    need(len({str(Path(path).parent) for path in modules.values()}) == 1, "controller modules do not share frozen source")
    need(sha(augmentation.__file__) == bindings[modules["heldout_augment.py"]],
         "admission verify_commit implementation differs from old frozen implementation")
    ev.bind(Path(augmentation.__file__).resolve())
    inventory_path = str(root / "frozen_source_inventory.json")
    need(inventory_path in bindings, "frozen inventory not bound before launch")
    inventory, inventory_ref = ev.read(inventory_path, bindings[inventory_path])
    runtime = _path(inventory["root"])
    need(str(runtime) == launch.get("frozen_runtime"), "controller/runtime inventory mismatch")
    inventory_files = inventory.get("files")
    need(isinstance(inventory_files, dict) and bool(inventory_files), "empty frozen runtime inventory")
    actual_names = {str(path.relative_to(runtime)) for path in runtime.rglob("*.py") if path.is_file()}
    need(set(inventory_files) == actual_names, "frozen runtime Python inventory is incomplete or changed")
    for relative, digest in inventory_files.items():
        path = runtime / relative
        need(not Path(relative).is_absolute() and _path(path).is_relative_to(runtime), "unsafe frozen inventory path")
        ev.bind(path, digest)
    plan, plan_ref = ev.read(root / "plan" / "generation_plan.json")
    sources, source_ref = ev.read(plan["source_plan_path"], plan["source_plan_sha256"])
    need(_path(plan["source_plan_path"]) == root / "plan" / "sources.json", "source plan outside old build")
    for prefix in ("catalog", "parent_plan", "profile"):
        path, digest = sources[prefix + "_path"], sources[prefix + "_sha256"]
        need(bindings.get(str(_path(path))) == digest, "registered input not in controller launch: " + prefix)
        ev.bind(path, digest)
    for role in ROLES:
        bank = sources["splits"][role]["donor_bank"]
        bank_root = _path(bank["path"])
        ev.bind(bank_root / "bank.json", bank["metadata_sha256"])
        ev.bind(bank_root / "profiles.npz", bank["profiles_sha256"])
    plan_return, plan_return_ref = ev.read(root / "plan_actual_return.json")
    pilot_return, pilot_return_ref = ev.read(root / "pilot_actual_return.json")
    command = pilot_return.get("command")
    need(isinstance(command, list) and len(command) == 5 and
         isinstance(command[0], str) and Path(command[0]).is_absolute() and
         command[1:] == [modules["heldout_run.py"], "--root", str(root), "--pilot"],
         "actual pilot command is not exact frozen heldout_run --root OLD --pilot")
    expected_plan = [command[0], modules["heldout_build.py"], "--root", str(root),
                     "--catalog", sources["catalog_path"], "--parents", sources["parent_plan_path"],
                     "--profile", sources["profile_path"], "--frozen-runtime", str(runtime), "--phase", "plan"]
    for value, expected in ((plan_return, expected_plan), (pilot_return, command)):
        need(type(value.get("returncode")) is int and value["returncode"] == 0 and
             value.get("command") == expected and value.get("bindings") == bindings,
             "actual subprocess return/command/bindings mismatch")
        need(isinstance(value.get("started_unix"), (int, float)) and
             isinstance(value.get("finished_unix"), (int, float)) and
             value["finished_unix"] >= value["started_unix"], "invalid actual subprocess chronology")
    need(pilot_return["started_unix"] >= plan_return["finished_unix"], "pilot predates actual plan completion")
    return plan, sources, dict(controller_launch=launch_ref, frozen_source_inventory=inventory_ref,
                              old_plan=plan_ref, old_sources=source_ref,
                              plan_actual_return=plan_return_ref, pilot_actual_return=pilot_return_ref)


def _tasks(plan):
    buckets, indexed = defaultdict(list), {}
    for task in plan["tasks"]:
        augmentation.validate_task(task)
        key = (task["role"], task["stage"], task["slot"])
        need(key not in indexed, "duplicate registered old task")
        indexed[key] = task
        buckets[(task["role"], task["generator"], task["quota_slot"], task["stage"])].append(task)
    expected = {(role, gen, j, stage) for role in ROLES for gen in ("Gen2", "Gen3", "Gen4", "Gen5")
                for j in range(120) for stage in STAGES}
    need(set(buckets) == expected and all(sorted(t["reserve_index"] for t in values) == [0, 1, 2]
                                         for values in buckets.values()), "old three-reserve task population changed")
    pilots = {}
    for role in ROLES:
        chosen = {}
        for _, gen, quota, stage in sorted(key for key in buckets if key[0] == role):
            task = min(buckets[(role, gen, quota, stage)], key=lambda t:t["reserve_index"])
            if task["recipe"] in ("clean", "gaps_weak"):
                chosen.setdefault((gen, stage, task["recipe"], task["size_class"]), (gen, quota, stage))
        pilots[role] = set(chosen.values())
        need(len(pilots[role]) == 48, "old pilot is not the registered 48 representative quotas")
    return indexed, pilots


def _source_tasks(tasks, sources):
    for (role, stage, slot), task in tasks.items():
        source = sources["splits"][role]
        need([source["positive"][slot]["pair_id"], source["negative"][slot]["pair_id"]] == task["base_pair_ids"] and
             source["schedule"]["recipes"][slot] == task["recipe"] and
             source["slot_generators"][slot] == task["generator"], "task differs from frozen source-plan slot")


def _baseline(root, role, slot, audit, ev):
    base_root = root / "baseline" / role
    path = base_root / "groups" / f"{slot:05d}.json"
    group, group_ref = ev.read(path, audit["group_sha256"])
    entries, rows = group["entries"], audit["audit_rows"]
    need(group["slot"] == slot and len(entries) == 2 and len(rows) == 2, "invalid old baseline pair group")
    need([e["label"] for e in entries] == [True, False], "old baseline labels changed")
    files = {group_ref["path"]: group_ref["sha256"]}
    for entry, row in zip(entries, rows):
        need(row.get("pair_id") == entry["pair_id"] and row.get("recipe") == entry["corrosion_recipe"] and
             row.get("raster_depth_lossless") is True and row.get("status", "passed") == "passed",
             "baseline audit identity/final assertion mismatch")
        for key in ("artifact_path", "weather_artifact", "background_artifact", "latent_seam_artifact", "target_metadata"):
            value = entry.get(key)
            need(key not in ("artifact_path", "target_metadata") or bool(value), "required baseline artifact missing")
            if value:
                original = Path(value)
                path = original if original.is_absolute() else base_root / original
                need(original.is_absolute() or _path(path).is_relative_to(base_root), "baseline artifact escapes original root")
                ref = ev.bind(path)
                files[ref["path"]] = ref["sha256"]
    return dict(root=str(base_root), group=group_ref, audit_rows=rows, files=files,
                artifact_hashes_established_by_this_admission=True,
                pixel_audit_reexecuted=False), group


def _bind_record(record, ev):
    for path_key, hash_key in (("sample_path", "sample_sha256"), ("proof_path", "proof_sha256"),
                              ("target_metadata", "target_metadata_sha256"),
                              ("unaugmented_sample_path", "unaugmented_sample_sha256"),
                              ("baseline_sample_path", "baseline_sample_sha256"),
                              ("baseline_group_path", "baseline_group_sha256")):
        ev.bind(record[path_key], record[hash_key])
    if record.get("latent_seam_artifact"):
        ev.bind(record["latent_seam_artifact"], record["latent_sha256"])


def _baseline_exhaustion(error, task):
    try:
        value = json.loads(error)
    except (TypeError, ValueError):
        return False
    keys = {"slot", "recipe", "partial", "negative", "failures", "damage_attempts", "source_draws",
            "static_length_sources_excluded", "damage_attempt_budget", "source_pool_exhausted"}
    return (isinstance(value, dict) and set(value) == keys and value["slot"] == task["slot"] and
            value["recipe"] == task["recipe"] and value["negative"] == task["base_pair_ids"][1] and
            isinstance(value["failures"], dict) and value["damage_attempt_budget"] == 1024 and
            0 <= value["damage_attempts"] <= 1024)


def check_admission(admission):
    """Recheck all original files; this is not a pixel-audit rerun."""
    need(isinstance(admission, dict) and admission.get("schema") == SCHEMA and
         admission.get("status") == "admitted_old_pilot_evidence_only", "invalid old-pilot admission")
    root = _path(admission["old_build_root"])
    _full_absent(root)
    bound = admission.get("bound_files")
    need(isinstance(bound, dict) and bool(bound), "empty admitted evidence")
    for path, digest in bound.items():
        need(sha(_path(path)) == digest, "admitted file changed: " + path)
    need(set(admission.get("roles", {})) == set(ROLES), "both admitted roles required")
    def references(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                need(bound.get(str(_path(value["path"]))) == value["sha256"], "receipt not covered by bound_files")
            for child in value.values():
                references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)
    references({key:value for key,value in admission.items() if key != "bound_files"})
    for role, value in admission["roles"].items():
        for slot, baseline in value["baselines"].items():
            need(baseline["root"] == str(root / "baseline" / role), "admitted baseline root changed")
            need(baseline["group"]["path"] == str(root / "baseline" / role / "groups" / f"{int(slot):05d}.json"),
                 "admitted baseline slot changed")
            for path, digest in baseline["files"].items():
                need(bound.get(path) == digest, "baseline file missing from full admission binding")
    return True


def admit_old_pilot(old_build_root, output_path):
    root, output = _path(old_build_root), Path(output_path).resolve()
    need(not output.exists() and not output.is_relative_to(root), "new admission outside immutable old build required")
    _full_absent(root)
    ev = _Evidence()
    ev.bind(Path(__file__).resolve())
    plan, sources, refs = _source_bindings(root, ev)
    tasks, pilot_quotas = _tasks(plan)
    _source_tasks(tasks, sources)
    shortfall, shortfall_ref = ev.read(root / "pilot_shortfall.json")
    need(shortfall.get("status") == "shortfall" and shortfall.get("stage") == "pilot" and
         shortfall.get("no_next_stage_launched") is True and set(shortfall.get("roles", {})) == set(ROLES),
         "actual shortfall-only controller termination missing")
    role_results = {}
    for role in ROLES:
        need(not (root / "pilot_receipts" / role / "failure.json").exists(), "pilot has an actual failure")
        complete, complete_ref = ev.read(root / "pilot_receipts" / role / "complete.json")
        need(complete.get("schema") == "mixed-sim-heldout-role-build/1" and complete.get("role") == role and
             complete.get("pilot_only") is True and complete.get("status") in ("shortfall", "pilot_passed") and
             complete.get("planned_quotas") == 48 and complete.get("plan_sha256") == refs["old_plan"]["sha256"] and
             complete.get("gpu_used") is False and complete.get("model_outputs_used") is False,
             "old role receipt is not the exact frozen 48-quota pilot")
        need(shortfall["roles"][role] == {key:complete[key] for key in ("status", "planned_quotas", "admitted_pairs")},
             "controller/role population disagrees")
        loaded = complete.get("frozen_source_sha256")
        need(isinstance(loaded, dict) and bool(loaded), "old role loaded-source receipt missing")
        for path, digest in loaded.items():
            need(ev.files.get(str(_path(path))) == digest, "loaded module not in complete frozen inventory")
        records = complete["records"]
        need(len(records) == complete["admitted_pairs"] and len(records) % 2 == 0,
             "admitted record count is not actual record population")
        baselines, baseline_groups = {}, {}
        for slot, audit in complete["base_audit"].items():
            need(str(int(slot)) == slot and any((role, stage, int(slot)) in tasks for stage in STAGES),
                 "unregistered old baseline slot")
            baselines[slot], baseline_groups[slot] = _baseline(root, role, int(slot), audit, ev)
        need(complete["actual_base_groups"] == len(baselines), "old baseline audit population differs")
        grouped = defaultdict(list)
        for record in records:
            grouped[(record["stage"], record["baseline_slot"])].append(record)
        commits, accepted = {}, set()
        for (stage, slot), entries in grouped.items():
            task = tasks.get((role, stage, slot))
            need(task is not None and len(entries) == 2, "unregistered or partial committed old group")
            quota = (task["generator"], task["quota_slot"], stage)
            need(quota in pilot_quotas[role] and quota not in accepted, "duplicate/non-pilot admitted quota")
            accepted.add(quota)
            path = root / "augmented" / role / stage / "groups" / f"{slot:05d}.json"
            committed = augmentation.verify_commit(path, task, refs["old_plan"]["sha256"])
            commit_ref = ev.bind(path)
            need(committed["records"] == entries and len(committed["audit_rows"]) == 2,
                 "old pilot accepted records differ from actual old commit")
            need(str(slot) in baselines, "accepted commit lacks old baseline audit")
            for ordinal, (entry, audit) in enumerate(zip(entries, committed["audit_rows"])):
                need(entry["data_role"] == role and entry["stage"] == stage and
                     entry["generator"] == task["generator"] and entry["label"] is (ordinal == 0) and
                     entry["baseline_ordinal"] == ordinal and entry["source_pair_id"] == task["base_pair_ids"][ordinal],
                     "committed source/role/pair ordinal differs from registered task")
                need(audit.get("status") == "passed" and audit.get("id") == entry["id"] and
                     audit.get("sample_sha256") == entry["sample_sha256"] and
                     audit.get("proof_sha256") == entry["proof_sha256"], "old augmentation audit does not prove this record")
                need(entry["baseline_group_path"] == baselines[str(slot)]["group"]["path"] and
                     entry["baseline_group_sha256"] == baselines[str(slot)]["group"]["sha256"],
                     "committed baseline differs from audited old group")
                base_entry = baseline_groups[str(slot)]["entries"][ordinal]
                need(_path(entry["baseline_sample_path"]) == _path(Path(baselines[str(slot)]["root"]) / base_entry["artifact_path"]),
                     "committed baseline sample differs from audited ordinal")
                _bind_record(entry, ev)
            commits[f"{stage}:{slot}"] = dict(commit=commit_ref, old_plan=refs["old_plan"], old_pilot_complete=complete_ref)
        rejections, exhausted = {}, set()
        for failure in complete["failures"]:
            phase = failure["phase"]
            if phase == "quota_exhausted":
                quota = tuple(failure["quota"])
                need(quota in pilot_quotas[role] and quota not in exhausted, "invalid exhausted pilot quota")
                exhausted.add(quota)
                continue
            need(phase in ("baseline_geometry_exhausted", "augmentation", "duplicate_model_input"),
                 "unexpected failure cannot be admitted as a recoverable rejection")
            task = failure["task"]
            need(tasks.get((role, task["stage"], task["slot"])) == task and
                 (task["generator"], task["quota_slot"], task["stage"]) in pilot_quotas[role],
                 "failure belongs to an unregistered task")
            if phase == "duplicate_model_input":
                continue  # New whole-population ordering must reconsider this candidate.
            if phase == "baseline_geometry_exhausted":
                need(_baseline_exhaustion(failure["error"], task), "not exact old bounded baseline exhaustion")
            else:
                rejection, _ = ev.read(root / "augmented" / role / task["stage"] / "rejections" / f'{task["slot"]:05d}.json')
                need(rejection.get("status") == "rejected" and rejection.get("task") == task and
                     rejection.get("reasons") == failure["error"] and rejection.get("counts_toward_final") is False,
                     "old augmentation rejection does not match actual artifact")
            key = canonical_task(task)
            need(key not in rejections, "duplicate old rejection task")
            rejections[key] = dict(phase=phase, error=failure["error"], old_pilot_complete=complete_ref)
        need(not (accepted & exhausted) and accepted | exhausted == pilot_quotas[role], "pilot quota accounting incomplete")
        need((complete["status"] == "shortfall") == bool(exhausted), "old shortfall status differs from actual quotas")
        role_results[role] = dict(baselines=baselines, commits=commits, rejections=rejections,
                                  old_pilot_complete=complete_ref, admitted_pairs=len(records),
                                  exhausted_quotas=[list(key) for key in sorted(exhausted)])
    need(any(role_results[role]["exhausted_quotas"] for role in ROLES), "not a shortfall pilot")
    admission = dict(schema=SCHEMA, status="admitted_old_pilot_evidence_only", old_build_root=str(root),
                     old_plan_sha256=refs["old_plan"]["sha256"], old_plan=refs["old_plan"],
                     original_receipts=refs, old_pilot_shortfall=shortfall_ref, roles=role_results,
                     bound_files=ev.files, old_artifacts_modified=False, old_pixel_audits_reexecuted=False,
                     historical_hashes_not_retroactively_claimed=True, full_build_not_started=True,
                     pilot_acceptance_does_not_reserve_final_duplicate_priority=True)
    check_admission(admission)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(admission, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return admission
