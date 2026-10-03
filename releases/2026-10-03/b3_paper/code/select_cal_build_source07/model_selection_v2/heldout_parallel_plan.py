"""Scheduling only: each quota's reserves and three stages have one owner."""
from collections import Counter
try:
    from . import heldout_reduce_plan as reduced
except ImportError:
    import heldout_reduce_plan as reduced


def make_shards(projection):
    result=[]
    for role in reduced.ROLES:
        for gen in reduced.GENS:
            ids=sorted(projection['cells'][role][gen]['quota_slots'])
            if len(ids)!=60 or len(set(ids))!=60:raise ValueError('fixed reduced quota set required')
            for part in range(4):
                result.append(dict(name=f'{role}_{gen}_{part:02d}',role=role,generator=gen,
                                   quota_slots=ids[part::4]))
    return result


def validate_shards(shards,projection,generation):
    if shards!=make_shards(projection):raise ValueError('parallel schedule changed')
    owners={}
    for shard in shards:
        if len(shard['quota_slots'])!=15:raise ValueError('shard size differs')
        for q in shard['quota_slots']:
            key=(shard['role'],shard['generator'],q)
            if key in owners:raise ValueError('quota has two writers')
            owners[key]=shard['name']
    paths={};seen=Counter()
    for role in reduced.ROLES:
        for t in reduced.chosen_tasks(projection,generation,role):
            owner=owners[t['role'],t['generator'],t['quota_slot']]
            path=(role,t['slot'])
            if paths.setdefault(path,owner)!=owner:raise ValueError('shared baseline has two workers')
            seen[(role,t['generator'],t['quota_slot'],t['stage'])]+=1
    expected={(r,g,q,s) for r,g,q in owners for s in reduced.STAGES}
    if set(seen)!=expected or len(set(seen.values()))!=1:raise ValueError('missing stage or reserve schedule')
    return owners


def validate_merged(records,tasks):
    if len(records)!=1440:raise ValueError('role does not have 1440 actual curriculum pairs')
    expected=Counter({(s,g,label):60 for s in reduced.STAGES for g in reduced.GENS for label in (False,True)})
    if Counter((r['stage'],r['generator'],r['label']) for r in records)!=expected:
        raise ValueError('merged Gen/stage/label quotas differ')
    for key in ('pair_id','model_tensors_sha256'):
        if len({r[key] for r in records})!=1440:raise ValueError('cross-shard duplicate '+key)
    groups={}
    for r in records:
        key=(r['stage'],r['baseline_slot'])
        if key not in tasks:raise ValueError('unregistered merged record')
        t=tasks[key];q=(t['stage'],t['generator'],t['quota_slot'])
        if q in groups and groups[q]!=key:raise ValueError('multiple reserves admitted for quota')
        groups[q]=key
        if (r['generator'],r['recipe'])!=(t['generator'],t['recipe']):raise ValueError('merged actual recipe differs')
    if len(groups)!=720:raise ValueError('missing merged quota')
    for stage in reduced.STAGES:
        for gen in reduced.GENS:
            actual=Counter(tasks[key]['recipe'] for q,key in groups.items() if q[:2]==(stage,gen))
            if actual!=Counter(reduced.targets(gen)):raise ValueError('merged augmentation coverage differs')
    return True
