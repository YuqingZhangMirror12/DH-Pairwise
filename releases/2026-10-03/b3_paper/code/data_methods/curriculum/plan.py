"""Disjoint underlying pair IDs and fixed review strata before rejection."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..aggressive_data_v16.run import tags,GROUP_NAMES as OLD_NAMES
from ..s7_compound_v1.materialize import read,digest
from .spec import SPECS
from . import SEED

def base_key(entry):
    return str(entry['source_root'])+'::'+str(entry['source_pair_id'])

def bucket(key):
    return 'v17.5' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2==0 else 'v18'

def group_names(version):
    spec=SPECS[version];names={k:v for k,v in OLD_NAMES.items() if k in (
        'clean','mild','wave','wave_weak','local_abrupt','local_abrupt_weak',
        'local_gradual','local_gradual_weak','gaps','gaps_weak','partial_end','partial_middle')}
    names.update(mild='连续弱腐蚀',gaps='多处独立缺口',gaps_weak='多处缺口＋连续弱腐蚀',
                 clean='无腐蚀对照',partial_end='Partial端部裁切对照',partial_middle='Partial中段裁切对照')
    return names

def plan(root,maximum):
    root=Path(root);entries=read(root/'train_s7b_24k.json')['entries']
    metrics={v['pair_id']:v for v in map(json.loads,(root/'pair_metrics.jsonl').read_text().splitlines())}
    byslot=defaultdict(dict)
    for e in entries:
        slot,ordinal=map(int,Path(e['artifact_path']).stem.split('_'));byslot[slot][ordinal]=e
    outputs={}
    for vi,version in enumerate(SPECS):
        pools=defaultdict(list);lookup={}
        for slot,pair in byslot.items():
            keys=[base_key(pair[j]) for j in (0,1)]
            if any(bucket(key)!=version for key in keys):continue
            labels=set()
            for e in pair.values():labels.update(tags(e,metrics[e['pair_id']]))
            task=dict(slot=slot,version=version,recipe=pair[0]['corrosion_recipe'],
                base_keys=keys,positive_candidates=[slot],mode='one' if slot%2 else 'both',
                allowed_modes=['one','both'],k=0,anticipated_groups=sorted(labels))
            target_seed=int(hashlib.sha256((version+'::'+keys[0]+'::trim25to40').encode()).hexdigest()[:16],16)
            task['trim_target']=float(np.random.default_rng(target_seed).uniform(*SPECS[version]['trim_range']))
            lookup[slot]=task
            for label in labels:pools[label].append(slot)
        rng=np.random.default_rng(SEED+vi)
        for values in pools.values():rng.shuffle(values)
        candidates=[];seen=set();used_keys=set()
        for rank in range(max(map(len,pools.values()))):
            for key in sorted(pools):
                values=pools[key]
                if rank>=len(values):continue
                slot=values[rank];task=lookup[slot]
                if slot in seen or any(k in used_keys for k in task['base_keys']):continue
                candidates.append(task);seen.add(slot);used_keys.update(task['base_keys'])
        # Keep all valid planned alternatives; run-time dispatch prefers deficient strata.
        gap_index=0
        for task in candidates:
            if task['recipe'].startswith('gaps'):
                counts=SPECS[version]['notch_counts'];task['k']=counts[gap_index%len(counts)];gap_index+=1
                task['anticipated_groups'].append('notches_'+str(task['k']))
        outputs[version]=candidates
    keys=[{k for t in tasks for k in t['base_keys']} for tasks in outputs.values()]
    assert keys[0].isdisjoint(keys[1])
    return dict(revision='curriculum-review-plan/2-trim25to40',baseline=str(root),
        baseline_sha256=digest(root/'train_s7b_24k.json'),tasks=outputs,
        maximum_candidates_per_version=maximum,version_assignment='SHA256 unaugmented source pair identity; both positive and negative must belong to same partition',
        within_version_policy='no repeated unaugmented source pair IDs',full_generation_authorized=False)

def choose(tasks,counts,used):
    available=[t for t in tasks if t['slot'] not in used]
    if not available:return None
    def score(task):
        wanted=group_names(task['version'])
        return sum(max(0,10-counts.get(k,0)) for k in task['anticipated_groups'] if k in wanted)
    return max(available,key=score)
