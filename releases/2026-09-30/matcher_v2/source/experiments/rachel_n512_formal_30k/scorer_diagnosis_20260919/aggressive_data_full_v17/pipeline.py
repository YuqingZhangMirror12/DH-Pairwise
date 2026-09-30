"""CPU-only full30K: gate four splits, commit exact quotas, audit all, calibrate TRAIN."""
import argparse,datetime,fcntl,json,os,subprocess,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from . import REVISION,COUNTS
from .resources import low_priority,snapshot,worker_cap,dispatch_allowed
from .plan import make_tasks,pilot_slots
from .source import initialize
from .generate import process
from .audit import audit_record
from .finish import finish
from ..s7_compound_v1.materialize import read,save_json,digest

USER_APPROVAL='我已经人工审核了你现在这个版本的展示样本，我放宽“主损伤区域的双侧间隙峰值到”5-35px“\n\n这批数据效果令人满意，可以通过了，批准生成30K全量数据，请你在ssh服务器远端生成。该服务器CPU性能较好，你可以在不影响GPU训练的前提下充分利用。'
FALLBACK_APPROVAL='如果找不到，那就不抢行施加v17的裁切，保留v14的数据类型。不要抢行施加新的约束'


def audit_task(task):
    from .source import STATE
    config=STATE['config'];split=STATE['split'];out=Path(config['out'])/split
    group_path=out/'groups'/f'{task["slot"]:05d}.json';result=out/'audits'/group_path.name
    if result.exists():
        r=read(result)
        if r['group_sha256']!=digest(group_path) or r['status']!='passed':raise ValueError('audit resume mismatch')
        return dict(slot=task['slot'],status='passed',cached=True)
    group=read(group_path)
    rows=[audit_record(r,STATE['root']) for r in group['records']]
    save_json(result,dict(status='passed',group_sha256=digest(group_path),rows=rows))
    return dict(slot=task['slot'],status='passed')


def run_pool(config,split,tasks,phase):
    out=Path(config['out']);n=len(tasks);done=0;start=time.time();running={};todo=iter(tasks)
    workers=worker_cap(snapshot(out),config['workers']);job=process if phase.endswith('generate') else audit_task
    exhausted=False
    with ProcessPoolExecutor(workers,initializer=initialize,initargs=(config,split)) as pool:
        try:
            while not exhausted or running:
                resource=snapshot(out)
                if resource['disk_free_bytes']<30*1024**3:raise ValueError('disk reserve below30GiB; stop without removing data')
                if dispatch_allowed(resource):
                    while len(running)<workers and not exhausted:
                        try:t=next(todo)
                        except StopIteration:exhausted=True;break
                        running[pool.submit(job,t)]=t['slot']
                if running:
                    ready,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                    for future in ready:
                        future.result();running.pop(future);done+=1
                else:
                    time.sleep(30)
                save_json(out/'pipeline_status.json',dict(status='running' if dispatch_allowed(resource) else 'resource_throttled',
                    phase=phase,split=split,completed_groups=done,planned_groups=n,workers=workers,
                    outstanding=len(running),phase_elapsed_seconds=time.time()-start,resource=resource,
                    pid=os.getpid(),updated_unix=time.time(),gpu_used=False))
        except BaseException:
            for f in running:f.cancel()
            raise
    receipt=dict(status='passed',phase=phase,split=split,groups=n,elapsed_seconds=time.time()-start,workers=workers)
    save_json(out/'phase_receipts'/f'{phase}_{split}.json',receipt)
    return receipt


def prepare(out,review,source,workers):
    if out.exists():raise ValueError('new full dataset root required')
    out.mkdir(parents=True)
    binding=read(source/'source_binding.json')
    for name,sha in binding.items():
        if digest(source/name)!=sha:raise ValueError('source binding changed:'+name)
    specs={}
    for key,relative in [('review_generation','review_bundle_01/generation_complete.json'),
                         ('review_pixel_audit','review_bundle_01/pixel_audit.json'),
                         ('review_rendered','rendered_01/rendered.json'),
                         ('review_manifest','review_bundle_01/manifest.json')]:
        p=review/relative;specs[key]=dict(path=str(p),sha256=digest(p))
    generation=read(specs['review_generation']['path']);audit=read(specs['review_pixel_audit']['path'])
    rendered=read(specs['review_rendered']['path'])
    if generation['pairs']!=240 or generation['missing'] or audit['status']!='passed' or audit['errors']:
        raise ValueError('full v17 review/audit required')
    if (audit['pairs']!=240 or len(audit['receipts'])!=240 or
            any(r['status']!='passed' for r in audit['receipts']) or
            audit['pilot_manifest_sha256']!=specs['review_manifest']['sha256']):
        raise ValueError('review audit does not bind the entire approved population')
    if len(rendered['groups'])!=27 or any(len(g['ids'])!=10 for g in rendered['groups'].values()):
        raise ValueError('approved complete per-type review required')
    audited={r['id'] for r in audit['receipts']}
    if not {r['id'] for r in rendered['rows']}<=audited:
        raise ValueError('review contains unaudited images')
    import hashlib
    approval=dict(schema='aggressive-data-human-approval/2',status='approved',scope='full30k_after_full_per_type10_review',
        approved_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        recorded_time_note='receipt creation time; not an invented exact user-message timestamp',
        evidence_kind='explicit_user_message_in_current_thread',user_approval_text=USER_APPROVAL,
        user_approval_text_sha256=hashlib.sha256(USER_APPROVAL.encode()).hexdigest(),
        fallback_authorization_text=FALLBACK_APPROVAL,
        fallback_authorization_sha256=hashlib.sha256(FALLBACK_APPROVAL.encode()).hexdigest(),
        augmentation_revision=REVISION,approved_review_revision='aggressive-v17-depth-review/1',
        authorized_delta=dict(primary_final_bilateral_gap_peak_px_before=[5,25],after=[5,35],
            applied_v17_pixel_rules_changed=False,retain_original_v14_on_failed_new_augmentation=True),
        **specs)
    approval_path=out/'human_approval.json';save_json(approval_path,approval)
    base=Path('/root/autodl-tmp/s7_balanced_20260923')
    config=dict(out=str(out),source=str(source),source_binding_sha256=digest(source/'source_binding.json'),
        approval=str(approval_path),approval_sha256=digest(approval_path),workers=workers,revision=REVISION,
        baselines=dict(train=str(base/'train24k_layered_v14_exact'),
            **{s:str(base/'heldout6k_v14/full'/s) for s in ('cal','select','test')}),
        heldout_sources=str(base/'heldout6k_v14/sources.json'),
        baseline_contract='/root/autodl-tmp/s7_consensus_threshold_v1_20260925/data_contract.json',
        no_automatic_restart=True,gpu_used=False,priority=15,
        resource_policy='at most32 single-thread workers and half CPU entitlement; throttle at80%load/75%memory; reserve50GiB disk',
        split_donor_policy='TRAIN bank for TRAIN; original fold-specific heldout banks for CAL/SELECT/TEST, no cross-fold shapes')
    hashes={}
    for split in COUNTS:
        plan=make_tasks(config['baselines'][split],split);p=out/'plans'/(split+'.json');save_json(p,plan);hashes[split]=digest(p)
    config['plan_hashes']=hashes;save_json(out/'config.json',config)
    return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--prepare',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--pilot-only',action='store_true');a=p.parse_args();low_priority()
    if a.prepare:
        result=prepare(a.out.resolve(),a.review.resolve(),a.source.resolve(),a.workers)
        print(json.dumps(dict(status='prepared',out=result['out'],plans=result['plan_hashes'])));return
    out=a.out.resolve();config=read(out/'config.json')
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'pipeline_complete.json').exists():raise ValueError('already complete; never rerun')
    if (out/'pipeline_launch.json').exists() and not a.resume:raise ValueError('explicit same-task resume required')
    for name,sha in read(Path(config['source'])/'source_binding.json').items():
        if digest(Path(config['source'])/name)!=sha:raise ValueError('immutable source changed')
    if digest(config['approval'])!=config['approval_sha256']:raise ValueError('approval binding changed')
    for split,sha in config['plan_hashes'].items():
        if digest(out/'plans'/(split+'.json'))!=sha:raise ValueError('plan changed')
    start=time.time();launch=dict(pid=os.getpid(),starttime=int(Path('/proc/self/stat').read_text().split(') ')[-1].split()[19]),
        cmdline=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
        started_unix=start,config_sha256=digest(out/'config.json'),gpu_used=False,automatic_retries=0)
    save_json(out/('pipeline_resume_'+str(os.getpid())+'.json' if a.resume else 'pipeline_launch.json'),launch)
    try:
        plans={s:read(out/'plans'/(s+'.json')) for s in COUNTS}
        if not (out/'pilot_complete.json').exists():
            for split,plan in plans.items():
                ids=set(pilot_slots(plan));tasks=[t for t in plan['tasks'] if t['slot'] in ids]
                run_pool(config,split,tasks,'pilot_generate');run_pool(config,split,tasks,'pilot_audit')
            save_json(out/'pilot_complete.json',dict(status='passed',all_four_splits=True,
                pairs=sum(2*len(pilot_slots(p)) for p in plans.values()),elapsed_seconds=time.time()-start))
        if a.pilot_only:return
        for split,plan in plans.items():
            for phase in ('full_generate','full_audit'):
                if not (out/'phase_receipts'/f'{phase}_{split}.json').exists():run_pool(config,split,plan['tasks'],phase)
        summaries=finish(config)
        save_json(out/'pipeline_status.json',dict(status='running',phase='train_geometry_calibration',gpu_used=False))
        calibration=out/'geometry_calibration_v2'
        if not (calibration/'geometry_calibration.json').exists():
            command=[sys.executable,'-m','experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.calibrate_geometry',
                '--contract',str(out/'data_contract.json'),'--out',str(calibration),'--workers',str(min(16,config['workers']))]
            with (out/'geometry_calibration.log').open('a') as log:
                subprocess.run(command,cwd=config['source'],stdout=log,stderr=subprocess.STDOUT,check=True)
        complete=dict(status='complete',pairs=30000,summaries=summaries,elapsed_seconds=time.time()-start,
            data_contract_sha256=digest(out/'data_contract.json'),geometry_sha256=digest(calibration/'geometry_calibration.json'),
            gpu_used=False,training_started=False)
        save_json(out/'pipeline_complete.json',complete);save_json(out/'pipeline_status.json',complete)
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start,
            output_preserved=True,automatic_retries=0,gpu_used=False)
        save_json(out/'pipeline_failure.json',failure);save_json(out/'pipeline_status.json',failure);raise

if __name__=='__main__':main()
