"""Fixed full-data quotas and source-identity partitions, never GT-outcome selection."""
from collections import Counter,defaultdict,deque
import hashlib
from pathlib import Path
import numpy as np
from . import REVISION,SEED
from .plan import base_key
from .spec import SPECS
from ..s7_compound_v1.materialize import read,digest


def allocation(total,weights):
    if total<1 or not weights or any(v<=0 for v in weights.values()):raise ValueError('positive quotas required')
    denominator=sum(weights.values())
    values={k:total*v//denominator for k,v in weights.items()}
    order=sorted(weights,key=lambda k:(-(total*weights[k]%denominator),k))
    for k in order[:total-sum(values.values())]:values[k]+=1
    return values


def assignment(key,targets):
    names=sorted(targets);total=sum(targets.values())
    draw=int(hashlib.sha256(('curriculum-full-source::'+key).encode()).hexdigest(),16)/2**256
    cursor=0.
    for v in names:
        cursor+=targets[v]/total
        if draw<cursor:return v
    return names[-1]


def validate_targets(targets):
    if set(targets)!=set(SPECS):raise ValueError('exactly v17.5 and v18 are active')
    if any(type(n) is not int or n<=0 or n%20 for n in targets.values()):
        raise ValueError('pair counts must be positive multiples of20 for balanced labels and70/30 side quota')


def build(root,targets):
    validate_targets(targets);root=Path(root)
    entries=read(root/'train_s7b_24k.json')['entries'];groups=defaultdict(dict)
    for e in entries:
        slot,ordinal=map(int,Path(e['artifact_path']).stem.split('_'));groups[slot][ordinal]=e
    if any(set(g)!={0,1} for g in groups.values()):raise ValueError('baseline paired labels incomplete')
    pos={base_key(g[0]) for g in groups.values()};neg=[base_key(g[1]) for g in groups.values()]
    if len(set(neg))!=len(neg) or not pos.isdisjoint(neg):
        raise ValueError('paired negative identity requires a new explicit source assignment')
    recipes=Counter(g[0]['corrosion_recipe'] for g in groups.values())
    quotas={v:allocation(targets[v]//2,recipes) for v in SPECS}
    tasks={v:[] for v in SPECS};gap_counts=Counter()
    ordered=sorted(groups.items(),key=lambda kv:hashlib.sha256((str(SEED)+'::'+str(kv[0])).encode()).hexdigest())
    for slot,pair in ordered:
        key=base_key(pair[0]);version=assignment(key,targets);recipe=pair[0]['corrosion_recipe']
        target_seed=int(hashlib.sha256((version+'::'+key+'::trim25to40').encode()).hexdigest()[:16],16)
        k=0
        if recipe.startswith('gaps'):
            k=SPECS[version]['notch_counts'][gap_counts[version]%4];gap_counts[version]+=1
        tasks[version].append(dict(slot=slot,version=version,recipe=recipe,
            base_keys=[key,base_key(pair[1])],positive_candidates=[slot],
            mode='one' if slot%2 else 'both',allowed_modes=['one','both'],k=k,
            trim_target=float(np.random.default_rng(target_seed).uniform(*SPECS[version]['trim_range']))))
    capacity={}
    for v,rows in tasks.items():
        capacity[v]={k:len({t['base_keys'][0] for t in rows if t['recipe']==k}) for k in quotas[v]}
        if len({t['base_keys'][0] for t in rows})<targets[v]//2:
            raise ValueError('insufficient distinct positive bases before geometry: '+v)
        for recipe,n in quotas[v].items():
            if capacity[v][recipe]<n:raise ValueError('insufficient distinct recipe bases: '+v+' '+recipe)
    return dict(revision=REVISION,purpose='full TRAIN curriculum supplement',targets=targets,
        baseline=str(root),baseline_manifest_sha256=digest(root/'train_s7b_24k.json'),
        recipe_group_quotas=quotas,baseline_recipe_group_counts=dict(recipes),tasks=tasks,
        distinct_positive_capacity_by_recipe=capacity,
        assignment='stable hash of original positive pair; fixed paired negative follows; accepted identities unique across both versions',
        full_generation_authorized=True,test_or_real_used=False)


def tickets(quotas):
    result={}
    for index,(version,counts) in enumerate(quotas.items()):
        total=sum(counts.values())
        if total%10:raise ValueError('side quota cannot close')
        rng=np.random.default_rng(SEED+701+index)
        recipes=[k for k,n in sorted(counts.items()) for _ in range(n)];rng.shuffle(recipes)
        sides=['smaller']*(7*total//10)+['larger']*(3*total//10);rng.shuffle(sides)
        output=defaultdict(deque)
        for recipe,side in zip(recipes,sides):output[recipe].append(side)
        result[version]=dict(output)
    return result


def choose(tasks,requests,attempted,accepted_keys,reserved):
    available=defaultdict(list)
    for t in tasks:
        if (t['slot'] in attempted or not requests.get(t['recipe'])
                or not accepted_keys.isdisjoint(t['base_keys']) or not reserved.isdisjoint(t['base_keys'])):continue
        available[t['recipe']].append(t)
    if not available:return None
    # Scarce strata first; the target quotas never follow acceptance rate.
    recipe=min(available,key=lambda k:(len({t['base_keys'][0] for t in available[k]})/len(requests[k]),k))
    task=dict(available[recipe][0],size_class=requests[recipe].popleft())
    attempted.add(task['slot']);reserved.update(task['base_keys'])
    return task


def accept_unique(records,audits,keys,hashes):
    if len(records)!=2 or sorted(bool(r['label']) for r in records)!=[False,True]:
        raise ValueError('one positive and one negative per group')
    incoming=[r['source_base_key'] for r in records];inputs=[r['model_input_sha256'] for r in audits]
    if len(set(incoming))!=2 or not keys.isdisjoint(incoming):raise ValueError('duplicate original base')
    if len(inputs)!=2 or len(set(inputs))!=2 or not hashes.isdisjoint(inputs):raise ValueError('duplicate model input')
    keys.update(incoming);hashes.update(inputs)
