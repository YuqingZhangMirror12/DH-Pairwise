"""Generate only the user's reduced, multidimensionally balanced projection.

Old completed groups are referenced and hash-checked, not regenerated. Only
missing selected groups are generated in this new build. GPU work is excluded.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import os
from pathlib import Path
import time
import traceback

try:
    from . import heldout_run as original, heldout_augment as aug, heldout_reduce_plan as reduced
    from .heldout_adopt import check_admission
except ImportError:
    import heldout_run as original, heldout_augment as aug, heldout_reduce_plan as reduced
    from heldout_adopt import check_admission


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    if aug.digest(ref['path'])!=ref['sha256']:raise ValueError('registered bytes changed: '+ref['path'])
    return read(ref['path'])


def reuse_commit(path,task,plan_ref,output):
    result=aug.verify_commit(path,task,plan_ref['sha256'])
    wrapper=dict(status='committed',task=task,plan_sha256=plan_ref['sha256'],
                 records=result['records'],audit_rows=result['audit_rows'],source_replaced=False,v14_fallback=False,
                 reused_from=dict(commit=reduced.ref(path),original_generation_plan=plan_ref),
                 old_pixels_regenerated=False)
    if output.exists():
        if read(output)!=wrapper:raise ValueError('existing reduced reuse wrapper differs')
    else:aug.save_json(output,wrapper)
    return wrapper


def baseline_identity(baseline, task, source_role):
    """A recipe/Gen quota cannot inherit the old generator's source fallback."""
    slot=task['slot']; entries=baseline['entries']
    if len(entries)!=2:raise ValueError('baseline must contain exactly two entries')
    spec=source_role
    positive=spec['positive'][slot]['pair_id']; negative=spec['negative'][slot]['pair_id']
    if task['base_pair_ids']!=[positive,negative]:raise ValueError('task differs from registered source arrays')
    if baseline['slot']!=slot or baseline['original_positive_pair_id']!=positive or baseline['source_negative_pair_id']!=negative:
        raise ValueError('baseline slot/original registration differs')
    if baseline['recipe']!=task['recipe']:raise ValueError('baseline recipe differs')
    actual=[e['source_pair_id'] for e in entries]
    if baseline['source_positive_pair_id']!=actual[0] or actual[1]!=negative:
        raise ValueError('inconsistent baseline source metadata')
    if actual!=task['base_pair_ids']:return False
    pool={r['pair_id']:r for r in spec['positive_pool']}
    for ordinal,(entry,row) in enumerate(zip(entries,(pool[positive],spec['negative'][slot]['row']))):
        if entry['label'] is not (ordinal==0):raise ValueError('baseline labels differ')
        if {k:v for k,v in entry['source_row'].items() if k!='pair_id'}!={k:v for k,v in row.items() if k!='pair_id'}:
            raise ValueError('baseline underlying fragment row differs')
        for side in ('a','b'):
            gen=row['fragment_'+side]['generator'].lower()
            if not gen.startswith(task['generator'].lower()):raise ValueError('registered source generator differs')
    return True


def fixed_source_slot(heldout, material, slot):
    """Only constrain the in-memory source bucket, never frozen source files.

    SourceSearch's original seeds, 1024 damage attempts and pixel predicates
    stay unchanged. All retries refer to the already registered positive pair.
    A candidate can change only at the outer, preregistered reserve boundary.
    """
    old=material.STATE['buckets']
    original=material.STATE['positive'][slot]
    material.STATE['buckets']={original['source_stratum']:[original]}
    try:return heldout.slot(slot)
    finally:material.STATE['buckets']=old


def run_role(args):
    root,role=args[:2]; shard=args[2] if len(args)==3 else None; root=Path(root).resolve()
    launch=read(root/'controller_launch.json')
    projection=checked(launch['reduction_plan'])
    plan=checked(projection['original_generation_plan'])
    sources=checked(dict(path=plan['source_plan_path'],sha256=plan['source_plan_sha256']))
    reduced.validate(projection,plan,sources)
    old=Path(launch['reused_build_root'])
    previous=Path(launch['previous_reduced_build']) if launch.get('previous_reduced_build') else None
    admission=checked(launch['original_pilot_admission'])
    # Parallel controller performs this full, immutable admission once before
    # dispatch; each worker still checks every artifact it actually reuses.
    if shard is None:check_admission(admission)
    elif launch.get('parallel_admission_checked') is not True:
        raise ValueError('parallel admission missing')
    inherited=admission['roles'][role]
    old_pilot=checked(launch['old_pilot_completions'][role])
    if old_pilot['status']!='pilot_passed' or old_pilot['plan_sha256']!=projection['original_generation_plan']['sha256']:
        raise ValueError('old source04 pilot is not the passed registered plan')
    inventory=checked(launch['frozen_source_inventory'])
    receipts=root/'shard_receipts'/shard['name'] if shard else root/'build_receipts'/role
    if (receipts/'complete.json').exists():raise ValueError('completed reduced build must not be repeated')
    receipts.mkdir(parents=True,exist_ok=True)
    start=time.time()
    if shard:aug.save_json(receipts/'status.json',dict(status='initializing',role=role,shard=shard['name'],
             pid=os.getpid(),started_unix=start,planned_quotas=45,admitted_pairs=0))
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        os.nice(10)
        import torch
        import cv2
        torch.set_num_threads(1); cv2.setNumThreads(1)
        heldout=importlib.import_module(original.PREFIX+'.s7_consensus_v1.heldout_v14')
        material=importlib.import_module(original.PREFIX+'.s7_balanced_v2.materialize')
        audit=importlib.import_module(original.PREFIX+'.s7_balanced_v2.audit_layered')
        source=importlib.import_module(original.PREFIX+'.aggressive_data_full_v17.source')
        baseline_root=root/'baseline'/role; baseline_root.mkdir(parents=True,exist_ok=True)
        heldout.initialize(dict(sources=plan['source_plan_path'],split=role,out=str(baseline_root),
            seed=plan['master_seed']+(10001 if role=='cal' else 20002),attempts=1024))
        config=dict(out=str(root/'augmented'),generation_plan_sha256=projection['original_generation_plan']['sha256'],
                    geometry_attempts=24)
        source.STATE.clear(); source.baseline.cache_clear()
        source.STATE.update(config=config,split=role,root=baseline_root,groups={},bank=material.STATE['bank'],
                           negative={e['pair_id']:e for e in material.STATE['negative_plan']})
        aug.STATE.clear(); aug.STATE.update(config=config,role=role,source=source)
        bound=original.bind_loaded_sources(); original.validate_loaded_inventory(bound,inventory)
        completed,failures,base_audit,base_roots=[],[],{},{}
        seen=set(); base_rejected={}
        quotas=original.quota_groups(reduced.chosen_tasks(projection,plan,role),role,False)
        if len(quotas)!=720:raise ValueError('reduced quota count is not 720 paired groups')
        if shard:
            try:from .heldout_parallel_plan import validate_shards
            except ImportError:from heldout_parallel_plan import validate_shards
            validate_shards(launch['shards'],projection,plan)
            if shard not in launch['shards'] or shard['role']!=role:raise ValueError('unregistered shard')
            owned={(shard['generator'],q) for q in shard['quota_slots']}
            quotas=[(key,tasks) for key,tasks in quotas if key[:2] in owned]
            if len(quotas)!=45:raise ValueError('shard must own fifteen bases and all three stages')
        planned=len(quotas)
        prior_roots=[Path(p) for p in launch.get('reuse_roots',[]) ]
        if not prior_roots:prior_roots=([previous] if previous else [])+[old]
        # The current output tree may only be shared by disjoint quota owners.
        admission_roots=([root] if shard else [])+prior_roots
        for key,tasks in quotas:
            accepted=None
            for task in tasks:
                slot=task['slot']; s=str(slot)
                if shard:aug.save_json(receipts/'current_task.json',dict(status='working',pid=os.getpid(),
                         task=task,started_unix=time.time(),committed_pairs=2*len(completed)))
                prior_baseline_rejection=next((p/'baseline_rejections'/role/f'{slot:05d}.json'
                    for p in admission_roots if (p/'baseline_rejections'/role/f'{slot:05d}.json').exists()),None)
                if prior_baseline_rejection is not None:
                    rejected=read(prior_baseline_rejection)
                    if not original.expected_baseline_exhaustion(RuntimeError(rejected['error']),task):
                        raise ValueError('unrelated baseline failure cannot be reused')
                    base_rejected[slot]=rejected['error']
                if slot in base_rejected:
                    failures.append(dict(task=task,phase='baseline_geometry_exhausted',error=base_rejected[slot],
                                         reused_same_baseline_rejection=True)); continue
                old_failure=inherited['rejections'].get(json.dumps(task,sort_keys=True,separators=(',',':')))
                if old_failure:
                    failures.append(dict(task=task,phase=old_failure['phase'],error=old_failure['error'],
                                         imported_rejection_from=old_failure['old_pilot_complete'],replayed=False)); continue
                old_rejection=next((p/'augmented'/role/task['stage']/'rejections'/f'{slot:05d}.json'
                    for p in admission_roots if (p/'augmented'/role/task['stage']/'rejections'/f'{slot:05d}.json').exists()),None)
                if old_rejection is not None:
                    rejection=read(old_rejection)
                    if rejection.get('task')!=task or rejection.get('status')!='rejected' or rejection.get('v14_fallback') is not False:
                        raise ValueError('old augmentation rejection identity differs')
                    failures.append(dict(task=task,phase='augmentation',error=rejection['reasons'],
                                         reused_from=reduced.ref(old_rejection),replayed=False)); continue
                commit=root/'augmented'/role/task['stage']/'groups'/f'{slot:05d}.json'
                prior=next((p/'augmented'/role/task['stage']/'groups'/f'{slot:05d}.json'
                    for p in prior_roots if (p/'augmented'/role/task['stage']/'groups'/f'{slot:05d}.json').exists()),None)
                if slot not in source.STATE['groups']:
                    recorded=old_pilot['base_audit'].get(s)
                    legacy=inherited['baselines'].get(s)
                    earlier_admission=next((candidate for p in admission_roots for candidate in (
                        p/'baseline_admissions'/role/f'{slot:05d}.json',
                        p/'build_receipts'/role/'baseline_admissions'/f'{slot:05d}.json') if candidate.exists()),None)
                    if earlier_admission is not None and earlier_admission.exists():
                        registered=read(earlier_admission);base=Path(registered['root']);group_path=base/'groups'/f'{slot:05d}.json'
                        if aug.digest(group_path)!=registered['group_sha256']:raise ValueError('previous reduced baseline changed')
                        baseline=read(group_path)
                        item=dict(registered,reused_audit_from=reduced.ref(earlier_admission))
                    elif recorded:
                        base=Path(recorded['root']); group_path=base/'groups'/f'{slot:05d}.json'
                        if aug.digest(group_path)!=recorded['group_sha256']:raise ValueError('old pilot baseline changed')
                        baseline=read(group_path)
                        item=dict(recorded,reused_audit_from=launch['old_pilot_completions'][role])
                    elif legacy:
                        base=Path(legacy['root']); group_path=Path(legacy['group']['path'])
                        baseline=checked(legacy['group'])
                        item=dict(root=str(base),group_sha256=legacy['group']['sha256'],audit_rows=legacy['audit_rows'],
                                  reused_audit_from=launch['original_pilot_admission'])
                    else:
                        existing=next((p/'baseline'/role/'groups'/f'{slot:05d}.json' for p in admission_roots
                            if (p/'baseline'/role/'groups'/f'{slot:05d}.json').exists()),None)
                        if existing is not None and baseline_identity(read(existing),task,sources['splits'][role]):
                            base=existing.parent.parent; baseline=read(existing)
                            audits=[audit.one((str(base),e)) for e in baseline['entries']]
                            item=dict(root=str(base),group_sha256=aug.digest(existing),audit_rows=audits,
                                      partial_build_baseline_admitted_now=True)
                        else:
                            if existing is not None:
                                aug.save_json(receipts/'excluded_old_baselines'/f'{slot:05d}.json',dict(
                                    reason='old_generator_replaced_registered_positive',task=task,
                                    old_group=reduced.ref(existing),counted=False,old_output_changed=False))
                            base=baseline_root
                            try:baseline=fixed_source_slot(heldout,material,slot)
                            except RuntimeError as error:
                                if not original.expected_baseline_exhaustion(error,task):raise
                                base_rejected[slot]=str(error)
                                failures.append(dict(task=task,phase='baseline_geometry_exhausted',error=str(error)))
                                aug.save_json(root/'baseline_rejections'/role/f'{slot:05d}.json',failures[-1])
                                continue
                            audits=[audit.one((str(base),e)) for e in baseline['entries']]
                            item=dict(root=str(base),group_sha256=aug.digest(base/'groups'/f'{slot:05d}.json'),audit_rows=audits)
                    if not baseline_identity(baseline,task,sources['splits'][role]):
                        raise ValueError('baseline source identities differ from unchanged task')
                    base_audit[s]=item; base_roots[slot]=base
                    source.STATE['groups'][slot]={i:e for i,e in enumerate(baseline['entries'])}
                    aug.save_json(receipts/'baseline_admissions'/f'{slot:05d}.json',item)
                    if shard:aug.save_json(root/'baseline_admissions'/role/f'{slot:05d}.json',item)
                source.STATE['root']=base_roots[slot]
                result=reuse_commit(prior,task,projection['original_generation_plan'],commit) if prior is not None else aug.process(task)
                if result['status']=='committed':
                    hashes=[r['model_tensors_sha256'] for r in result['records']]
                    if len(set(hashes))!=2 or set(hashes)&seen:
                        failures.append(dict(task=task,phase='duplicate_model_input',error='exact input duplicate; old output retained'))
                        continue
                    seen.update(hashes); accepted=result; break
                failures.append(dict(task=task,phase='augmentation',error=result.get('reasons')))
            if accepted is None:failures.append(dict(quota=list(key),phase='quota_exhausted',error='all 12 fixed candidates rejected'))
            else:completed.append(accepted)
            progress=dict(status='running',role=role,pid=os.getpid(),planned_quotas=planned,
                finished_quotas=len(completed)+sum(f['phase']=='quota_exhausted' for f in failures),
                admitted_pairs=2*len(completed),reused_groups=sum('reused_from' in c for c in completed),
                elapsed_seconds=time.time()-start)
            aug.save_json(receipts/'status.json',progress)
            if shard:progress['shard']=shard['name']
            if progress['finished_quotas']%20==0 or progress['finished_quotas']==planned:
                print(json.dumps(progress),flush=True)
                aug.save_json(receipts/'checkpoint.json',dict(progress=progress,commits=[c['task'] for c in completed],
                                                            failures=failures,base_audit=base_audit))
        original.check_sources(bound); later=original.bind_loaded_sources()
        original.validate_loaded_inventory(later,inventory); bound.update(later)
        records=[r for c in completed for r in c['records']]
        exhausted=[f for f in failures if f['phase']=='quota_exhausted']
        if not exhausted:
            expected=Counter((key[2],key[0],label) for key,_ in quotas for label in (False,True))
            if len(records)!=planned*2 or Counter((r['stage'],r['generator'],r['label']) for r in records)!=expected:
                raise ValueError('actual reduced population differs')
            for stage in reduced.STAGES:
                for gen in reduced.GENS:
                    actual=Counter(c['task']['recipe'] for c in completed if c['task']['stage']==stage and c['task']['generator']==gen)
                    wanted=Counter(tasks[0]['recipe'] for key,tasks in quotas if key[0]==gen and key[2]==stage)
                    if actual!=wanted:raise ValueError('actual augmentation coverage differs')
        done=dict(schema='mixed-sim-heldout-reduced-role-build/1',role=role,status='shortfall' if exhausted else 'complete',
            pilot_only=False,planned_quotas=planned,admitted_pairs=len(records),records=records,base_audit=base_audit,failures=failures,
            reduction_plan=launch['reduction_plan'],original_generation_plan=projection['original_generation_plan'],
            frozen_source_sha256=bound,gpu_used=False,model_outputs_used=False,reserve_count=12,
            reused_groups=sum('reused_from' in c for c in completed),elapsed_seconds=time.time()-start,
            actual_by_stage_gen_recipe={stage:{gen:dict(Counter(c['task']['recipe'] for c in completed
                 if c['task']['stage']==stage and c['task']['generator']==gen)) for gen in reduced.GENS} for stage in reduced.STAGES})
        if shard:done.update(schema='mixed-sim-heldout-parallel-shard/1',shard=shard)
        aug.save_json(receipts/'complete.json',done)
        brief={k:done[k] for k in ('status','role','planned_quotas','admitted_pairs','reused_groups','elapsed_seconds')}
        aug.save_json(receipts/'status.json',brief)
        return brief
    except BaseException as error:
        aug.save_json(receipts/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),pid=os.getpid()))
        raise


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(run_role,(args.root,role)) for role in reduced.ROLES]):
            print(json.dumps(result.result()),flush=True)


if __name__=='__main__':main()
