"""Preregister new heldout source/recipe quotas without model outcomes.

The normalized native catalog is assembled from existing Gen2/3 and newly
generated Gen4/5 in a new release with the original loader's safe relative
path schema. Old release manifests and files are never edited in place.
"""
from collections import Counter
import hashlib
import importlib
import itertools
import json
from pathlib import Path

PREFIX = "experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919"
MASTER_SEED = 261002910
GENS = ("Gen2", "Gen3", "Gen4", "Gen5")
ROLES = ("cal", "select")
STAGES = ("v17_filtered", "v17.5", "v18")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("immutable new registration already exists: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def canonical_generator(name):
    for number in range(2, 6):
        if name.lower().startswith("gen"+str(number)):
            return "Gen"+str(number)
    raise ValueError("unrecognized generator: " + name)


def native_negative_pool(fragments, role, generator, seed, limit=360):
    """Different known parent families, same generator, original scale gate."""
    import numpy as np
    pool = sorted((f for f in fragments if f["role"] == role and
                   canonical_generator(f["row"]["generator"]) == generator),
                  key=lambda f:f["row"]["fragment_token"])
    candidates = []
    for a, b in itertools.combinations(pool, 2):
        if a["family"] == b["family"]:
            continue
        ar, br = a["row"], b["row"]
        ratios = {key:max(float(ar[key]),float(br[key]))/min(float(ar[key]),float(br[key]))
                  for key in ("foreground_area", "bbox_aspect_ratio")}
        if max(ratios.values()) > 2.0:
            continue
        token = "|".join((ar["fragment_token"],br["fragment_token"]))
        candidates.append(dict(pair_id="mixsim2-cross-"+hashlib.sha256(token.encode()).hexdigest()[:24],
                               label=False, split="val", fragment_a=ar, fragment_b=br,
                               correspondence_path=None, negative_origin="cross_folder_scale_matched",
                               label_origin="distinct_canonical_manuscript_families", scale_match=ratios))
    if not candidates:
        raise ValueError(f"no isolated same-generator negative pool: {role}/{generator}")
    rng = np.random.default_rng(seed)
    return [candidates[int(i)] for i in rng.permutation(len(candidates))[:limit]]


def build(catalog_path, parents_path, out, profile_path, master_seed=MASTER_SEED, reserves=3):
    """120 final paired groups/gen/stage, bounded 3x source candidates.

    Every reserve is assigned before geometry is evaluated. It has its own
    recorded baseline; no rejected example changes identity, labels or recipe.
    Only geometry/loader gates may exclude candidates, never model scores.
    """
    import numpy as np
    from scipy import ndimage
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    heldout = importlib.import_module(PREFIX+".s7_consensus_v1.heldout_v14")
    prep = importlib.import_module(PREFIX+".seam_context_v3.prepare")
    old_sides = importlib.import_module(PREFIX+".aggressive_data_full_v17.plan")
    out = Path(out).resolve()
    if (out/"sources.json").exists():
        raise ValueError("new source registration exists")
    catalog = json.loads(Path(catalog_path).read_text())
    release = Path(catalog["release_root"]).resolve(strict=True)
    parents = json.loads(Path(parents_path).read_text())
    profile = json.loads(Path(profile_path).read_text())
    if not profile.get("preserve_original_scale_for_heldout"):
        raise ValueError("only frozen native-scale heldout profile is allowed")
    families = {role:{r["family"] for r in parents["folds"][role]} for role in ROLES}
    forbidden = set().union(*map(set, parents["exclusion_families"].values()))
    if families["cal"] & families["select"] or set.union(*families.values()) & forbidden:
        raise ValueError("parent roles overlap TRAIN or TEST")
    fragments = catalog["fragments"]
    for f in fragments:
        if f["role"] not in ROLES or f["family"] not in families[f["role"]]:
            raise ValueError("unregistered fragment parent")
        if Path(f["row"]["model_mask_path"]).is_absolute():
            raise ValueError("original native loader requires safe relative paths")
    plan = dict(schema="s7-v14-heldout-source-plan/1", dataset_root=str(release), seed=master_seed,
                forbidden_sources=sorted(forbidden), splits={},
                profile_path=str(Path(profile_path).resolve()),profile_sha256=sha(profile_path),
                authorization="new SELECT/CAL only; original TRAIN/TEST/B3 remain untouched",
                catalog_path=str(Path(catalog_path).resolve()), catalog_sha256=sha(catalog_path),
                parent_plan_path=str(Path(parents_path).resolve()),parent_plan_sha256=sha(parents_path))
    tasks = []
    for role_i, role in enumerate(ROLES):
        schedule = {k:[] for k in ("recipes","partial","partial_modes","bins","mirrors")}
        positives, negatives, pool, all_negative, slot_generators = [], [], [], [], []
        counts = {}
        for gen_i, gen in enumerate(GENS):
            seed = master_seed + 10000*role_i + 100*gen_i
            rng = np.random.default_rng(seed)
            pp = [r for r in catalog["positive_rows"][role]
                  if canonical_generator(r["fragment_a"]["generator"]) == gen]
            pp = [r for r in pp if heldout.eligible_sample(release,r)]
            if not pp:
                raise ValueError(f"empty native positive source cell {role}/{gen}")
            npool = native_negative_pool(fragments,role,gen,seed+1,120*reserves)
            npool = [r for r in npool if heldout.eligible_sample(release,r)]
            if not npool:
                raise ValueError("negative native loader gate exhausted")
            pp = [pp[int(i)] for i in rng.permutation(len(pp))]
            pool.extend(pp)
            all_negative.extend(npool)
            local = heldout.schedule(120,profile,seed+2)
            sides = old_sides.side_schedule(120,seed+3)
            counts[gen] = dict(positive_base_pairs=len(pp),negative_base_pairs=len(npool),
                              candidate_groups=120*reserves, final_groups_per_stage=120,
                              recipe_counts=local["integer_macro_counts"],
                              geometry_exclusion_only=True)
            for reserve in range(reserves):
                for j in range(120):
                    slot = len(positives)
                    p = pp[(reserve*120+j)%len(pp)]
                    n = npool[(reserve*120+j)%len(npool)]
                    positives.append(dict(pair_id=p["pair_id"],source_stratum="native_positive"))
                    negatives.append(dict(mode="native", pair_id=n["pair_id"], row=n,
                                          source_stratum="native_negative",anchor_group_id=None,kind="cross_manuscript"))
                    slot_generators.append(gen)
                    for key in schedule:
                        schedule[key].append(local[key][j])
                    for stage in STAGES:
                        target_seed=int(hashlib.sha256(f'{master_seed}:{role}:{stage}:{p["pair_id"]}:trim'.encode()).hexdigest()[:16],16)
                        tasks.append(dict(role=role,stage=stage,generator=gen,slot=slot, master_seed=master_seed,
                                          reserve_index=reserve,quota_slot=j, recipe=local["recipes"][j],
                                          base_pair_ids=[p["pair_id"],n["pair_id"]], size_class=sides[j],
                                          mode="one" if j%2 else "both",k=1+j%4,
                                          trim_target=float(np.random.default_rng(target_seed).uniform(.25,.40))))
        all_fragments={f["row"]["fragment_token"]:f["row"] for f in fragments if f["role"]==role}
        bank = heldout.make_bank(all_fragments,families[role],role,release,out/"banks"/role,master_seed+70000+role_i)
        schedule["rounding"]="frozen largest remainder per120 groups, repeated reserve schedule"
        plan["splits"][role]=dict(source_families=sorted(families[role]), release_split="val",
                                  pairs=len(positives)*2, positive=positives, positive_pool=pool,
                                  negative=negatives,schedule=schedule,donor_bank=bank,
                                  generator_counts=counts, slot_generators=slot_generators,
                                  unique_negative_pool=len({r["pair_id"] for r in all_negative}))
    write(out/"sources.json",plan)
    generation=dict(schema="mixed-sim-heldout-generation-plan/1",master_seed=master_seed,
                    source_plan_path=str(out/"sources.json"),source_plan_sha256=sha(out/"sources.json"),
                    roles=list(ROLES), stages=list(STAGES), tasks=tasks,
                    desired_pairs_per_role=3200,desired_pairs_per_curriculum_stage=960,
                    desired_strict_pairs_per_role=320,
                    reserve_policy="first pixel-admitted candidate per quota_slot, in registered reserve order",
                    quotas_fixed_before_generation=True,model_outputs_used=False,
                    label_policy="frozen native/augmentation target inheritance; no target repair",
                    historical_gen23_edge_donors_complete=False,
                    no_train_or_test_generation=True)
    write(out/"generation_plan.json",generation)
    return generation
