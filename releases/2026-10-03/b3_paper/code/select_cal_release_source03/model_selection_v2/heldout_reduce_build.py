"""One-shot smaller CPU dataset build; old jobs and outputs stay retired."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from . import heldout_reduce_plan as p
    from .heldout_extend_plan import validate_extension
except ImportError:
    import heldout_reduce_plan as p
    from heldout_extend_plan import validate_extension

BASE=Path('/root/autodl-tmp/model_selection_v2_20261002')


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')


def strict_projection(root):
    old=BASE/'strict_normalized_02'
    complete=read(old/'complete.json'); launch=read(old/'launch.json'); actual=read(old/'actual_process_return.json')
    if (complete['status']!='verified_strict_component_only' or complete['rows']!=640 or
        type(actual['returncode']) is not int or actual['returncode']!=0 or actual['command']!=launch['command'] or
        actual['source']!=launch['source']):raise ValueError('strict normalized source lacks actual successful seal')
    sealed={str(old/name):p.sha(old/name) for name in ('complete.json','launch.json','actual_process_return.json')}
    sealed.update(complete['bound_source_sha256'])
    for path,digest in sealed.items():
        if p.sha(path)!=digest:raise ValueError('strict normalization source changed')
    outputs={}
    for role in p.ROLES:
        row=next(x for x in complete['roles'] if x['role']==role)
        source=row['expanded']; original=read(source['path'])
        if p.sha(source['path'])!=source['sha256'] or len(original['entries'])!=320:
            raise ValueError('strict normalized data changed')
        groups=defaultdict(list)
        for e in original['entries']:
            key=(e['label'],e['recipe'],e['native_entry']['meta']['base'] if e['recipe']=='straight_J' else '')
            groups[key].append(e)
        selected=[]
        for label in (False,True):
            round_up=(int(label)+p.ROLES.index(role))%2
            wanted={('straight_M',''):20,('straight_R',''):23-round_up,
                    ('straight_J','torn_rachel'):30,('straight_J','margin_fragment'):4+round_up,
                    ('straight_J','torn_strip'):3}
            for (recipe,subtype),count in wanted.items():
                cell=groups[label,recipe,subtype]
                ordered=sorted(cell,key=lambda e:hashlib.sha256(
                    ('strict-reduce160/1:'+role+':'+e['pair_id']).encode()).hexdigest())
                if len(ordered)<count:raise ValueError('strict original subtype shortfall')
                selected.extend(ordered[:count])
        if len(selected)!=160 or Counter(e['label'] for e in selected)!=Counter({False:80,True:80}):
            raise ValueError('strict projected labels incorrect')
        data=dict(schema='mixed-heldout-strict-user-subset/1',status='subset_of_verified_normalized_component',
                  role=role,rows=160,entries=selected,original_expanded=source,
                  original_normalization_seals={name:p.ref(old/name) for name in
                     ('complete.json','launch.json','actual_process_return.json')},
                  source_files=sealed,subset_uses_model_outputs=False,pixel_generation_repeated=False,
                  label_recipe_counts={str(label):dict(Counter(e['recipe'] for e in selected if e['label']==label))
                                       for label in (False,True)})
        output=root/'strict_subset'/f'{role}.json';save(output,data);outputs[role]=p.ref(output)
    return outputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);args=ap.parse_args()
    root=args.root.resolve();old=BASE/'curriculum_build_02';previous=BASE/'curriculum_build_03'
    if root!=BASE/'curriculum_build_04' or root.exists():raise ValueError('fresh named fixed-source reduced build required')
    retired=BASE/'reduction_transition_01/retired.json'
    if read(retired)['status']!='retired_for_user_requested_reduced_build':raise ValueError('old full build not retired')
    pause=read(retired.with_name('paused.json'))
    for item in pause['processes']:
        stat=Path('/proc')/str(item['pid'])/'stat'
        if stat.exists():
            parts=stat.read_text().rsplit(')',1)[1].split()
            if int(parts[19])==item['starttime'] and parts[0]!='Z':raise ValueError('old build process still alive')
    generation,sources=validate_extension(old/'plan/generation_plan.json')
    prior_launch=read(previous/'controller_launch.json')
    actual=read(previous/'full_actual_return.json')
    if (actual['returncode']!=1 or actual['bindings']!=prior_launch['bindings'] or
        actual['command'][-2:]!=['--root',str(previous)]):raise ValueError('previous reduced build not the registered stopped failure')
    for role in p.ROLES:
        failure=read(previous/'build_receipts'/role/'failure.json')
        if failure['error']!="ValueError('baseline source identities differ from unchanged task')":
            raise ValueError('unrelated previous build error')
        if (Path('/proc')/str(failure['pid'])/'cmdline').exists():
            command=(Path('/proc')/str(failure['pid'])/'cmdline').read_bytes()
            if b'heldout_reduce_run.py' in command:raise ValueError('previous reduced worker still alive')
    plan=read(previous/'reduction_plan.json')
    if p.ref(previous/'reduction_plan.json')!=prior_launch['reduction_plan']:
        raise ValueError('prior selected reduction plan changed')
    if plan['original_generation_plan']!=p.ref(old/'plan/generation_plan.json'):
        raise ValueError('original generation plan changed')
    p.validate(plan,generation,sources)
    root.mkdir()
    save(root/'reduction_plan.json',plan)
    if p.sha(root/'reduction_plan.json')!=p.sha(previous/'reduction_plan.json'):
        raise ValueError('reduction plan projection changed')
    strict=prior_launch['strict_subsets']
    for ref in strict.values():
        if p.sha(ref['path'])!=ref['sha256']:raise ValueError('strict subset changed')
    old_launch=read(old/'controller_launch.json')
    runtime=Path(old_launch['frozen_runtime'])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONPATH']=str(runtime)
    names=('heldout_reduce_plan.py','heldout_reduce_run.py','heldout_reduce_build.py',
           'heldout_run.py','heldout_augment.py','heldout_plan.py','heldout_extend_plan.py','heldout_adopt.py')
    bindings={str(Path(__file__).with_name(n).resolve()):p.sha(Path(__file__).with_name(n)) for n in names}
    bindings.update(old_launch['bindings'])
    bindings.update(prior_launch['bindings'])
    for path in (retired,retired.with_name('paused.json'),old/'pilot_actual_return.json',root/'reduction_plan.json',
                 old/'pilot_receipts/cal/complete.json',old/'pilot_receipts/select/complete.json',
                 previous/'controller_launch.json',previous/'full_actual_return.json',
                 previous/'build_receipts/cal/failure.json',previous/'build_receipts/select/failure.json'):
        bindings[str(path)]=p.sha(path)
    for ref in strict.values():bindings[ref['path']]=ref['sha256']
    for path,digest in bindings.items():
        if p.sha(path)!=digest:raise ValueError('registered source or admission changed: '+path)
    original_admission=p.ref(old/'extension_admission.json')
    if read(old/'pilot_actual_return.json')['returncode']!=0:raise ValueError('old source04 pilot actual failure')
    save(root/'controller_launch.json',dict(schema='mixed-heldout-reduced-controller/1',pid=os.getpid(),
        started_unix=time.time(),bindings=bindings,reduction_plan=p.ref(root/'reduction_plan.json'),
        reused_build_root=str(old),previous_reduced_build=str(previous),original_pilot_admission=original_admission,
        old_pilot_completions={r:p.ref(old/'pilot_receipts'/r/'complete.json') for r in p.ROLES},
        frozen_source_inventory=p.ref(old/'frozen_source_inventory.json'),frozen_runtime=str(runtime),
        strict_subsets=strict,user_requested_limit=dict(positive=800,negative=800,per_role=1600),
        cuda_visible_devices='',workers=2,threads_per_worker=1,models_loaded=False,
        old_sources_modified=False,original_pixels_regenerated=False,prototype_not_rerun=True,
        source_search_policy='registered_positive_only; outer twelve registered candidates unchanged',
        faulty_baselines_retained_but_not_admitted=True,reduction_projection_repeated=False))
    command=[sys.executable,str(Path(__file__).with_name('heldout_reduce_run.py')),'--root',str(root)]
    began=time.time();result=subprocess.run(command,cwd=root,env=os.environ.copy())
    save(root/'full_actual_return.json',dict(command=command,returncode=result.returncode,bindings=bindings,
                                          started_unix=began,finished_unix=time.time()))
    if result.returncode:raise SystemExit(result.returncode)
    roles={r:read(root/'build_receipts'/r/'complete.json') for r in p.ROLES}
    if any(r['status']!='complete' or r['admitted_pairs']!=1440 for r in roles.values()):
        save(root/'full_shortfall.json',dict(status='shortfall',no_release_claim=True,
                    roles={name:{k:r[k] for k in ('status','admitted_pairs')} for name,r in roles.items()}))
        raise SystemExit(2)
    for path,digest in bindings.items():
        if p.sha(path)!=digest:raise ValueError('source/input changed during reduced build')
    save(root/'controller_complete.json',dict(status='reduced_curriculum_complete_pending_combined_release_audit',
        finished_unix=time.time(),bindings=bindings,desired_curriculum_pairs_per_role=1440,
        desired_total_pairs_per_role=1600,roles={r:p.ref(root/'build_receipts'/r/'complete.json') for r in p.ROLES},
        strict_subsets=strict,no_training=True,no_model_selection=True,no_gpu=True,old_outputs_changed=False))
    print(json.dumps(dict(status='reduced_curriculum_complete_pending_combined_release_audit',root=str(root))),flush=True)


if __name__=='__main__':main()
