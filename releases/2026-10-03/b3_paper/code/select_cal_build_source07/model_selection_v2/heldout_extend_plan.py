"""Append nine preregistered heldout reserves without replacing old three.

No geometry/model outcomes are read. Pools are only the already eligible rows
in the immutable old source plan; no new eligibility scan, donor bank, or data
generation is performed. Old task objects and source-array prefixes are exact.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

try:
    from .heldout_plan import canonical_generator, ROLES, GENS, STAGES
except ImportError:
    from heldout_plan import canonical_generator, ROLES, GENS, STAGES

SCHEMA = 'mixed-sim-heldout-generation-plan/2'
OLD_SCHEMA = 'mixed-sim-heldout-generation-plan/1'
OLD_RESERVES = 3
RESERVE_COUNT = 12
EXTRA_RESERVES = RESERVE_COUNT - OLD_RESERVES
QUOTAS = 120
SCHEDULE_FIELDS = ('recipes', 'partial', 'partial_modes', 'bins', 'mirrors')
TASK_FIELDS = {'role', 'stage', 'generator', 'slot', 'master_seed', 'reserve_index',
               'quota_slot', 'recipe', 'base_pair_ids', 'size_class', 'mode', 'k', 'trim_target'}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path))


def bound_read(ref):
    require(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute(), 'exact absolute receipt required')
    require(sha(ref['path']) == ref['sha256'], 'registered input bytes changed: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def trim_target(master_seed, role, stage, positive_pair_id):
    import numpy as np
    seed = int(hashlib.sha256(f'{master_seed}:{role}:{stage}:{positive_pair_id}:trim'.encode()).hexdigest()[:16], 16)
    return float(np.random.default_rng(seed).uniform(.25, .40))


def _pool(rows, generator, label):
    by_id = {}
    for row in rows:
        gen = canonical_generator(row['fragment_a']['generator'])
        require(gen == canonical_generator(row['fragment_b']['generator']), 'cross-generator source row')
        require(type(row['label']) is bool and row['label'] == label and row['split'] == 'val',
                'only original native heldout label/val rows allowed')
        pair = row['pair_id']
        require(isinstance(pair, str) and pair, 'source pair identity missing')
        if pair in by_id: require(row == by_id[pair], 'same source ID has conflicting original row bytes')
        by_id[pair] = row
    return {pair: row for pair, row in by_id.items()
            if canonical_generator(row['fragment_a']['generator']) == generator}


def validate_old(generation, sources):
    """Bind every old task to its unchanged native source slot and quota."""
    require(generation.get('schema') == OLD_SCHEMA and sources.get('schema') == 's7-v14-heldout-source-plan/1',
            'only original three-reserve source03 plans may be extended')
    require(generation['master_seed'] == sources['seed'] and type(sources['seed']) is int,
            'original master seed identity mismatch')
    require(generation['roles'] == list(ROLES) and generation['stages'] == list(STAGES)
            and set(sources['splits']) == set(ROLES), 'full CAL/SELECT three-stage plan required')
    require(generation.get('quotas_fixed_before_generation') is True
            and generation.get('model_outputs_used') is False
            and generation.get('no_train_or_test_generation') is True,
            'original registration is not outcome-free heldout-only')
    require((generation['desired_pairs_per_role'], generation['desired_pairs_per_curriculum_stage'],
             generation['desired_strict_pairs_per_role']) == (3200, 960, 320), 'final population changed')
    cells, slots, tasks = defaultdict(dict), defaultdict(set), generation['tasks']
    for task in tasks:
        require(set(task) == TASK_FIELDS, 'original task schema differs')
        role, gen, stage = task['role'], task['generator'], task['stage']
        j, reserve, slot = task['quota_slot'], task['reserve_index'], task['slot']
        require(role in ROLES and gen in GENS and stage in STAGES and type(j) is int and 0 <= j < QUOTAS
                and type(reserve) is int and 0 <= reserve < OLD_RESERVES and type(slot) is int and slot >= 0,
                'old role/generator/stage/quota/reserve/slot invalid')
        require(task['master_seed'] == generation['master_seed'], 'old task master seed differs')
        require(task['size_class'] in ('smaller', 'larger') and task['mode'] in ('one', 'both')
                and type(task['k']) is int and 1 <= task['k'] <= 4, 'old geometry quota differs')
        spec = sources['splits'][role]
        require(slot < len(spec['positive']) and task['base_pair_ids'] ==
                [spec['positive'][slot]['pair_id'], spec['negative'][slot]['pair_id']], 'old task source binding differs')
        require(spec['slot_generators'][slot] == gen and task['recipe'] == spec['schedule']['recipes'][slot],
                'old task generator/recipe differs from source arrays')
        require(task['trim_target'] == trim_target(generation['master_seed'], role, stage, task['base_pair_ids'][0]),
                'old task trim target differs from original formula')
        key = (role, gen, j)
        require((reserve, stage) not in cells[key], 'duplicate old task')
        cells[key][reserve, stage] = task
        slots[role].add(slot)
    expected = {(role, gen, j) for role in ROLES for gen in GENS for j in range(QUOTAS)}
    require(set(cells) == expected and len(tasks) == 2*4*120*3*3, 'incomplete old task registration')
    pools = {}
    forbidden = set(sources['forbidden_sources'])
    family_sets = {role: set(sources['splits'][role]['source_families']) for role in ROLES}
    require(not family_sets['cal'] & family_sets['select'] and not set.union(*family_sets.values()) & forbidden,
            'old source families intersect another role/TRAIN/TEST')
    for role in ROLES:
        spec = sources['splits'][role]; n = 4*QUOTAS*OLD_RESERVES
        require(spec['release_split'] == 'val' and spec['pairs'] == 2*n and slots[role] == set(range(n)),
                'complete contiguous old source slots required')
        require(len(spec['positive']) == len(spec['negative']) == len(spec['slot_generators']) == n,
                'old source arrays have different populations')
        require(all(isinstance(spec['schedule'][key], list) and len(spec['schedule'][key]) == n for key in SCHEDULE_FIELDS),
                'old schedule population differs')
        neg_entries = {}
        for item in spec['negative']:
            require(item['mode'] == 'native' and item['pair_id'] == item['row']['pair_id'], 'old native negative identity differs')
            if item['pair_id'] in neg_entries:
                require(item == neg_entries[item['pair_id']], 'negative source entry conflicts')
            neg_entries[item['pair_id']] = item
        for gen in GENS:
            pp = _pool(spec['positive_pool'], gen, True)
            nn = _pool([e['row'] for e in spec['negative']], gen, False)
            require(pp and nn, 'old eligible source pool is empty')
            count = spec['generator_counts'][gen]
            require(count['positive_base_pairs'] == len(pp) and count['negative_base_pairs'] == len(nn)
                    and count['candidate_groups'] == QUOTAS*OLD_RESERVES
                    and count['final_groups_per_stage'] == QUOTAS and count['geometry_exclusion_only'] is True,
                    'old eligible pool inventory differs from declared counts')
            pools[role, gen] = (pp, {pair: neg_entries[pair] for pair in nn})
            for j in range(QUOTAS):
                group = cells[role, gen, j]
                require(set(group) == {(r, s) for r in range(3) for s in STAGES}, 'old reserve/stage set incomplete')
                template = group[0, STAGES[0]]
                immutable_keys = ('role', 'generator', 'master_seed', 'quota_slot', 'recipe', 'size_class', 'mode', 'k')
                for task in group.values():
                    require(all(task[k] == template[k] for k in immutable_keys), 'old quota attributes vary by reserve/stage')
                    p, q = task['base_pair_ids']; require(p in pp and q in nn, 'old task uses source outside eligible pools')
                    require(spec['positive'][task['slot']] == {'pair_id': p, 'source_stratum': 'native_positive'},
                            'old positive source entry schema differs')
                    ref_slot = template['slot']
                    require(all(spec['schedule'][key][task['slot']] == spec['schedule'][key][ref_slot]
                                for key in SCHEDULE_FIELDS), 'old schedule quota varies by reserve/stage')
                for reserve in range(3):
                    group_r = [group[reserve, stage] for stage in STAGES]
                    require(len({t['slot'] for t in group_r}) == 1 and
                            len({tuple(t['base_pair_ids']) for t in group_r}) == 1, 'three stages lack a shared baseline')
    return cells, pools


def independent_order(ids, seed, role, gen, quota, polarity, already_used):
    """Separate fixed domains; no stage or old stride120 enters source draws."""
    return sorted(ids, key=lambda pair: (pair in already_used,
        content_sha(['heldout-reserve-extension/1', seed, role, gen, quota, polarity, pair]), pair))


def append_candidates(positive_ids, negative_ids, old_pairs, *, seed, role, gen, quota, count=EXTRA_RESERVES):
    """Negative-cycle coverage first, combination uniqueness until exhausted.

    Positive coverage is preferred, not falsely promised when pair uniqueness
    and the negative cycle make it impossible. Such positive view reuse is
    explicit. If the negative-cycle and unique-combination constraints are
    incompatible before the Cartesian pool is exhausted, fail closed.
    """
    pp, nn = set(positive_ids), set(negative_ids)
    require(pp and nn and all(p in pp and n in nn for p, n in old_pairs), 'candidate sources outside old pools')
    old_p, old_n = {p for p, _ in old_pairs}, {n for _, n in old_pairs}
    p_order = independent_order(pp, seed, role, gen, quota, 'positive', old_p)
    n_order = independent_order(nn, seed, role, gen, quota, 'negative', old_n)
    p_rank, n_rank = {p:i for i,p in enumerate(p_order)}, {n:i for i,n in enumerate(n_order)}
    p_total, n_total, combinations = Counter(p for p,_ in old_pairs), Counter(n for _,n in old_pairs), Counter(old_pairs)
    p_cycle, n_cycle, result = set(), set(), []
    for offset in range(count):
        if len(p_cycle) == len(pp): p_cycle.clear()
        if len(n_cycle) == len(nn): n_cycle.clear()
        product_exhausted = len(combinations) == len(pp)*len(nn)
        n_candidates = sorted(nn-n_cycle, key=lambda n: (n_total[n] > 0, n_total[n], n_rank[n]))
        p_candidates = sorted(pp, key=lambda p: (p in p_cycle, p_total[p] > 0, p_total[p], p_rank[p]))
        chosen = next(((p,n) for n in n_candidates for p in p_candidates
                       if product_exhausted or (p,n) not in combinations), None)
        require(chosen is not None, 'negative-cycle coverage and unused combinations cannot both be satisfied; register a reviewed policy instead')
        p,n = chosen
        result.append(dict(positive_pair_id=p, negative_pair_id=n,
            reserve_index=OLD_RESERVES+offset,
            positive_cycle_reuse=p in p_cycle, negative_cycle_reuse=False,
            quota_combination_prior_views=combinations[p,n],
            quota_positive_prior_views=p_total[p], quota_negative_prior_views=n_total[n],
            combination_is_new_for_quota=(p,n) not in combinations,
            cartesian_pool_exhausted_before_draw=product_exhausted,
            view_reuse_note='Reused bases/pair combinations are augmentation views, never new independent sources.'))
        p_cycle.add(p); n_cycle.add(n); p_total[p]+=1; n_total[n]+=1; combinations[p,n]+=1
    return result


def _prefix_receipts(generation, sources):
    return dict(old_task_count=len(generation['tasks']), old_tasks_sha256=content_sha(generation['tasks']),
        role_sources={role: dict(old_slots=len(sources['splits'][role]['positive']),
            positive_sha256=content_sha(sources['splits'][role]['positive']),
            negative_sha256=content_sha(sources['splits'][role]['negative']),
            positive_pool_sha256=content_sha(sources['splits'][role]['positive_pool']),
            slot_generators_sha256=content_sha(sources['splits'][role]['slot_generators']),
            donor_bank_sha256=content_sha(sources['splits'][role]['donor_bank']),
            schedule_sha256={key:content_sha(sources['splits'][role]['schedule'][key]) for key in SCHEDULE_FIELDS})
            for role in ROLES})


def construct(generation, sources, extension_of, new_source_path):
    cells, pools = validate_old(generation, sources)
    new_sources, new_generation = copy.deepcopy(sources), copy.deepcopy(generation)
    audits, extra_tasks = [], []
    for role in ROLES:
        old_spec, spec = sources['splits'][role], new_sources['splits'][role]
        for gen in GENS:
            positives, negatives = pools[role, gen]
            assignments = {}
            for j in range(QUOTAS):
                old_pairs = [tuple(cells[role,gen,j][r,STAGES[0]]['base_pair_ids']) for r in range(3)]
                extra = append_candidates(positives, negatives, old_pairs, seed=generation['master_seed'],
                                          role=role, gen=gen, quota=j)
                assignments[j] = extra
                audits.append(dict(role=role, generator=gen, quota_slot=j,
                    available_positive_bases=len(positives), available_negative_bases=len(negatives),
                    old_positive_unique=len({p for p,n in old_pairs}), old_negative_unique=len({n for p,n in old_pairs}),
                    old_combination_unique=len(set(old_pairs)), old_prefix_duplicate_views=len(old_pairs)-len(set(old_pairs)),
                    appended_views=extra,
                    all12_positive_unique=len({p for p,n in old_pairs} | {e['positive_pair_id'] for e in extra}),
                    all12_negative_unique=len({n for p,n in old_pairs} | {e['negative_pair_id'] for e in extra}),
                    old_prefix_not_rewritten=True))
            for r in range(3, RESERVE_COUNT):
                for j in range(QUOTAS):
                    pair = assignments[j][r-3]; p,n = pair['positive_pair_id'], pair['negative_pair_id']
                    slot = len(spec['positive']); pair['slot'] = slot
                    spec['positive'].append(dict(pair_id=p, source_stratum='native_positive'))
                    spec['negative'].append(copy.deepcopy(negatives[n]))
                    spec['slot_generators'].append(gen)
                    template_slot = cells[role,gen,j][0,STAGES[0]]['slot']
                    for key in SCHEDULE_FIELDS:
                        spec['schedule'][key].append(copy.deepcopy(old_spec['schedule'][key][template_slot]))
                    for stage in STAGES:
                        task = copy.deepcopy(cells[role,gen,j][0,stage])
                        task.update(slot=slot, reserve_index=r, base_pair_ids=[p,n],
                                    trim_target=trim_target(generation['master_seed'],role,stage,p))
                        extra_tasks.append(task)
            spec['generator_counts'][gen]['candidate_groups'] = QUOTAS*RESERVE_COUNT
        spec['pairs'] = 2*len(spec['positive'])
    new_generation['tasks'].extend(extra_tasks)
    new_generation.update(schema=SCHEMA, source_plan_path=str(Path(new_source_path).resolve()),
        source_plan_sha256=hashlib.sha256(encoded(new_sources)).hexdigest(), reserve_count=RESERVE_COUNT,
        extension_of=copy.deepcopy(extension_of), prefix_receipts=_prefix_receipts(generation,sources),
        candidate_extension=dict(schema='outcome-free-heldout-reserve-extension/1', old_reserve_count=3,
            appended_reserve_indices=list(range(3,12)), appended_candidates_per_quota=9,
            source_pool_policy='only original eligible positive_pool and original negative row union; no new source admission',
            draw_policy='independent role/generator/quota/polarity SHA permutations; negative unused-source cycle first, positive coverage preferred; unique combinations until Cartesian exhaustion',
            removed_stride='(reserve*120+quota)%pool_size is not used for appended candidate identity',
            view_policy='old duplicate prefixes remain disclosed; repeated base or exhausted pair combinations are views, not independent data',
            stages_share_registered_baseline=True, old_task_objects_modified=False, donor_banks_rebuilt=False,
            geometry_outcomes_read=False, model_outputs_used=False, training_started=False,
            registration_completed_before_candidate_execution=True, quota_source_audit=audits))
    return new_generation, new_sources


def _input_refs(old_build_root):
    root = Path(old_build_root).resolve(strict=True)
    generation_ref = receipt(root/'plan'/'generation_plan.json')
    generation = bound_read(generation_ref)
    source_ref = receipt(generation['source_plan_path'])
    require(source_ref['path'] == str((root/'plan'/'sources.json').resolve(strict=True)), 'old source plan outside named build/plan')
    require(source_ref['sha256'] == generation['source_plan_sha256'], 'old source SHA mismatch')
    sources = bound_read(source_ref)
    # Preserve and bind the old profile/catalog/parent bytes and donor-bank
    # files, not merely their path strings. No bank reconstruction or native
    # source eligibility scan occurs here.
    for stem in ('profile', 'catalog', 'parent_plan'):
        bound_read(dict(path=sources[stem+'_path'], sha256=sources[stem+'_sha256']))
    require(Path(sources['dataset_root']).is_absolute() and Path(sources['dataset_root']).is_dir(),
            'old native release root missing')
    for role, spec in sources['splits'].items():
        bank = spec['donor_bank']; directory = Path(bank['path'])
        metadata = bound_read(dict(path=str(directory/'bank.json'),sha256=bank['metadata_sha256']))
        require(sha(directory/'profiles.npz') == bank['profiles_sha256'], 'old donor profile bytes changed')
        require(metadata['split'] == role and all(a['split'] == role for a in metadata['arcs'])
                and set(metadata['source_families']) <= set(spec['source_families']), 'old bank role/family differs')
    return dict(generation_plan=generation_ref, source_plan=source_ref, build_root=str(root)), generation, sources


def extend(old_build_root, out_plan_dir):
    """Write sources.json + generation_plan.json in a fresh local plan folder."""
    out = Path(out_plan_dir).resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), 'new empty output plan directory required')
    refs, old_generation, old_sources = _input_refs(old_build_root)
    require(not out.is_relative_to(Path(refs['build_root'])), 'extension must not write into old build root')
    generation, sources = construct(old_generation, old_sources, refs, out/'sources.json')
    # Re-read immutable parents after planning before either new output is made.
    require(bound_read(refs['generation_plan']) == old_generation and bound_read(refs['source_plan']) == old_sources,
            'old registration changed during extension')
    out.mkdir(parents=True, exist_ok=True)
    for name,value in [('sources.json',sources), ('generation_plan.json',generation)]:
        with (out/name).open('xb') as stream: stream.write(encoded(value))
    return generation


def validate_extension(generation_path):
    """Return (generation, sources) only after exact deterministic reconstruction."""
    path = Path(generation_path).resolve(strict=True)
    generation = json.loads(path.read_text())
    require(generation.get('schema') == SCHEMA and generation.get('reserve_count') == RESERVE_COUNT,
            'registered total12 extension required')
    refs = generation.get('extension_of', {})
    require(set(refs) == {'generation_plan','source_plan','build_root'}, 'exact parent receipts required')
    expected_refs, old_generation, old_sources = _input_refs(refs['build_root'])
    require(refs == expected_refs, 'extension parent identity mismatch')
    source_path = path.parent/'sources.json'
    require(generation['source_plan_path'] == str(source_path) and not path.is_relative_to(Path(refs['build_root'])),
            'new source/build output identity mismatch')
    sources = bound_read(dict(path=str(source_path),sha256=generation['source_plan_sha256']))
    expected_generation, expected_sources = construct(old_generation,old_sources,refs,source_path)
    require(sources == expected_sources and generation == expected_generation,
            'extension source/task prefix, pool, bank, seed, quota or registration differs')
    require(sha(source_path) == hashlib.sha256(encoded(expected_sources)).hexdigest(), 'registered source serialization differs')
    # Validate parent bytes again after reconstruction; never inspect scores.
    bound_read(refs['generation_plan']); bound_read(refs['source_plan'])
    return generation,sources


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-build-root',required=True)
    parser.add_argument('--out',required=True,help='new plan directory, not the old build root')
    args=parser.parse_args(); generation=extend(args.old_build_root,args.out)
    print(json.dumps(dict(status='extension_registered_not_executed', reserve_count=RESERVE_COUNT,
        tasks=len(generation['tasks']), generation_plan=str(Path(args.out).resolve()/'generation_plan.json'),
        generation_plan_sha256=sha(Path(args.out)/'generation_plan.json'), model_outputs_used=False)))


if __name__=='__main__': main()
