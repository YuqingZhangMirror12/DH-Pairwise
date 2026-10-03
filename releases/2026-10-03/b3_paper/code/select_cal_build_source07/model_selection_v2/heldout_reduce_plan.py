"""User-requested 1600-pair projection: Gen AND augmentation-balanced.

Consumes only frozen registration metadata, never generated success/failure,
pixels, model predictions, or real scores. Task and source identities stay
unchanged, so completed matching outputs can be reused without regeneration.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROLES = ('cal', 'select')
GENS = ('Gen2', 'Gen3', 'Gen4', 'Gen5')
STAGES = ('v17_filtered', 'v17.5', 'v18')
OLD_RECIPES = dict(clean=18, partial=30, mild=18, wave=9, wave_weak=9,
                   local_abrupt=5, local_abrupt_weak=5, local_gradual=4,
                   local_gradual_weak=4, gaps=9, gaps_weak=9)
SCHEMA = 'mixed-heldout-user-reduction-plan/1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def priority(role, gen, quota):
    text = f'user-reduction-1600/1:261002910:{role}:{gen}:{quota}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16**12


def targets(gen):
    out = {k:v//2 for k,v in OLD_RECIPES.items()}
    # Complementary rounding across four Gen strata exactly preserves the
    # aggregate recipe proportions; weak/non-weak variants all remain present.
    up = ('wave', 'local_abrupt', 'gaps') if GENS.index(gen)%2 == 0 else (
          'wave_weak', 'local_abrupt_weak', 'gaps_weak')
    for key in up: out[key] += 1
    assert sum(out.values()) == 60 and min(out.values()) >= 2
    return out


def descriptions(tasks, sources, role, gen):
    rows = sorted((t for t in tasks if t['role'] == role and t['generator'] == gen
                   and t['stage'] == STAGES[0] and t['reserve_index'] == 0), key=lambda t:t['quota_slot'])
    if len(rows) != 120 or [r['quota_slot'] for r in rows] != list(range(120)):
        raise ValueError('original full 120-quota cell required')
    if Counter(r['recipe'] for r in rows) != Counter(OLD_RECIPES):
        raise ValueError('frozen recipe distribution changed')
    schedule = sources['splits'][role]['schedule']
    return [dict(task=t, partial_mode=schedule['partial_modes'][t['slot']],
                 mirror=schedule['mirrors'][t['slot']]) for t in rows]


def measures(rows):
    return dict(paired_groups=len(rows), image_pairs=2*len(rows),
                recipe=dict(Counter(r['task']['recipe'] for r in rows)),
                size_class=dict(Counter(r['task']['size_class'] for r in rows)),
                partial_mode=dict(Counter(r['partial_mode'] for r in rows if r['task']['recipe']=='partial')),
                mirror={str(k):v for k,v in Counter(r['mirror'] for r in rows).items()},
                endpoint_mode=dict(Counter(r['task']['mode'] for r in rows)),
                notch_count={str(k):v for k,v in Counter(r['task']['k'] for r in rows).items()})


def validate_cell(rows, gen):
    m = measures(rows)
    parity = GENS.index(gen)%2
    wanted = dict(paired_groups=60, image_pairs=120, recipe=targets(gen),
                  size_class={'smaller':42,'larger':18},
                  partial_mode={'end':8-parity,'middle':7+parity},
                  mirror={'0':51,'1':5-parity,'2':4+parity},
                  endpoint_mode={'one':30,'both':30},
                  notch_count={str(k):15 for k in range(1,5)})
    if m != wanted:
        raise ValueError('reduced Gen/recipe/secondary quotas differ: '+repr((m,wanted)))
    return m


def select_cell(rows, role, gen):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    parity = GENS.index(gen)%2
    masks, values = [], []
    def add(predicate, value):
        masks.append([int(predicate(r)) for r in rows]); values.append(value)
    for recipe,n in targets(gen).items(): add(lambda r,recipe=recipe:r['task']['recipe']==recipe,n)
    add(lambda r:r['task']['size_class']=='smaller',42)
    add(lambda r:r['partial_mode']=='end' and r['task']['recipe']=='partial',8-parity)
    for k,n in ((0,51),(1,5-parity),(2,4+parity)): add(lambda r,k=k:r['mirror']==k,n)
    for k in range(1,5): add(lambda r,k=k:r['task']['k']==k,15)
    add(lambda r:r['task']['mode']=='one',30)
    cost=np.asarray([priority(role,gen,r['task']['quota_slot']) for r in rows])
    result=milp(cost,integrality=np.ones(len(rows)),bounds=Bounds(0,1),
                constraints=LinearConstraint(np.asarray(masks),values,values),
                options={'time_limit':15,'mip_rel_gap':0.0})
    if not result.success or result.x is None:
        raise ValueError('balanced 60-quota projection infeasible; do not relax silently: '+str(result.message))
    chosen=[r for r,x in zip(rows,result.x) if x>.5]
    validate_cell(chosen,gen)
    return chosen


def construct(generation,sources,generation_ref):
    cells={}
    for role in ROLES:
        cells[role]={}
        for gen in GENS:
            rows=descriptions(generation['tasks'],sources,role,gen)
            selected=select_cell(rows,role,gen)
            cells[role][gen]=dict(quota_slots=[r['task']['quota_slot'] for r in selected],
                                  planned_per_stage=measures(selected))
    return dict(schema=SCHEMA, original_generation_plan=generation_ref,
                desired_pairs_per_role=1600, desired_curriculum_pairs_per_role=1440,
                desired_strict_pairs_per_role=160, positive_per_role=800,negative_per_role=800,
                cells=cells, stages=list(STAGES), selected_after_user_size_change=True,
                selection_inputs='original recipe/Gen/size/mirror/partial/mode/k metadata only',
                generated_success_failure_read=False, model_outputs_used=False,
                all_stages_share_same_quota_ids=True, original_tasks_and_seeds_unchanged=True,
                source_pools_unchanged=True, pixel_gates_and_targets_unchanged=True,
                strict_subset_rule='label-balanced half within frozen M/J/R and J subtype; stable sample-ID SHA order')


def validate(projection,generation,sources):
    if projection.get('schema')!=SCHEMA or tuple(projection.get(k) for k in (
        'desired_pairs_per_role','desired_curriculum_pairs_per_role','desired_strict_pairs_per_role',
        'positive_per_role','negative_per_role'))!=(1600,1440,160,800,800):
        raise ValueError('reduced budget mismatch')
    if projection.get('model_outputs_used') is not False or projection.get('generated_success_failure_read') is not False:
        raise ValueError('outcome-conditioned reduction forbidden')
    for role in ROLES:
        for gen in GENS:
            rows=descriptions(generation['tasks'],sources,role,gen)
            cell=projection['cells'][role][gen]; ids=cell['quota_slots']
            if len(ids)!=60 or len(set(ids))!=60 or not set(ids)<=set(range(120)):
                raise ValueError('invalid selected quota IDs')
            actual=validate_cell([r for r in rows if r['task']['quota_slot'] in ids],gen)
            if actual!=cell['planned_per_stage']:raise ValueError('planned statistics mismatch')
    return True


def chosen_tasks(projection,generation,role):
    return [t for t in generation['tasks'] if t['role']==role and
            t['quota_slot'] in projection['cells'][role][t['generator']]['quota_slots']]
