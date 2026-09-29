"""Bounded CPU review driver with immutable provenance, no expansion/training."""
import argparse,json,os,time,traceback,fcntl,subprocess,sys
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import SEED,REVISION
from .spec import SPECS
from .plan import plan,choose,group_names
from .generate import process
from ..aggressive_data_full_v17.source import initialize
from ..aggressive_data_full_v17.resources import low_priority,snapshot,dispatch_allowed,worker_cap
from ..s7_compound_v1.materialize import save_json,read,digest

def init(config):
    low_priority();initialize(config,'train')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--probe',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--maximum',type=int,default=600)
    args=p.parse_args()
    if not 1<=args.workers<=16 or args.maximum>600 or args.probe>12:raise ValueError('bounded CPU review only')
    root=Path(args.out)
    if root.exists():raise ValueError('fresh output required; automatic restart disabled')
    root.mkdir(parents=True)
    args.workers=worker_cap(snapshot(root),args.workers)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    low_priority();start=time.time()
    config=read('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03/config.json')
    config.update(out=str(root),workers=args.workers)
    prepared=plan(config['baselines']['train'],args.maximum)
    save_json(root/'plan.json',prepared);config['plan_sha256']=digest(root/'plan.json')
    save_json(root/'protocol.json',dict(revision=REVISION,contracts=SPECS,
        baseline=config['baselines']['train'],baseline_manifest_sha256=prepared['baseline_sha256'],
        source_binding_sha256=digest(Path.cwd()/'source_binding.json'),
        plan_sha256=config['plan_sha256'],workers=args.workers,nice=15,gpu_used=False,
        full_generation_authorized=False,training_started=False,
        purpose='review examples, stratified; not natural training prevalence',
        unchanged_archives=['v14','v17'],probe=args.probe,
        per_type=10,review_labels='positive_only',
        side_quota='planned blocks of7 smaller3 larger; skipped crops reported separately, no forced cuts'))
    records={v:[] for v in SPECS};counts={v:Counter() for v in SPECS};attempts={v:[] for v in SPECS}
    queues={v:deque() for v in SPECS};used={v:set() for v in SPECS}
    completed=set();rng=np.random.default_rng(SEED+111)
    audits={v:[] for v in SPECS}
    def status(state):
        value=dict(status=state,updated_unix=time.time(),elapsed_seconds=time.time()-start,
            gpu_used=False,training_started=False,full_generation_authorized=False,versions={})
        for version in SPECS:
            value['versions'][version]=dict(attempted_groups=len(attempts[version]),
                accepted_groups=len(records[version])//2,pairs=len(records[version]),
                counts=dict(counts[version]),pending_side_requests=list(queues[version]),
                missing={k:10-counts[version][k] for k in group_names(version) if counts[version][k]<10})
        save_json(root/'pipeline_status.json',value);return value
    try:
        with ProcessPoolExecutor(args.workers,initializer=init,initargs=(config,)) as pool:
            while len(completed)<len(SPECS):
                resource=snapshot(root);save_json(root/'resources.json',resource)
                if not dispatch_allowed(resource):
                    status('resource_wait');time.sleep(30);continue
                batch=[]
                for version in SPECS:
                    if version in completed:continue
                    if not queues[version]:
                        block=['smaller']*7+['larger']*3;rng.shuffle(block);queues[version].extend(block)
                    for _ in range(min(max(1,args.workers//2),len(queues[version]))):
                        if args.probe and len(used[version])>=args.probe:break
                        if len(used[version])>=args.maximum:break
                        task=choose(prepared['tasks'][version],counts[version],used[version])
                        if task is None:break
                        task=dict(task,size_class=queues[version].popleft())
                        used[version].add(task['slot']);batch.append(task)
                if not batch:break
                for task,result in zip(batch,pool.map(process,batch)):
                    v=task['version']
                    if result['status'] in ('passed','committed'):
                        records[v].extend(result['records']);audits[v].extend(result['audit_rows'])
                        for row in result['records']:
                            if row['label']:counts[v].update(row['groups'])
                    else:queues[v].append(task['size_class'])
                    receipt={k:x for k,x in result.items() if k not in ('records','audit_rows')}
                    attempts[v].append(receipt)
                    save_json(root/v/'attempts'/f"{task['slot']:05d}.json",receipt)
                    save_json(root/v/'manifest.json',dict(entries=records[v]))
                    save_json(root/v/'pixel_audit.json',dict(status='passed',pairs=len(audits[v]),receipts=audits[v],
                        independent_pixels_and_actual_supervision=True))
                    status('generating')
                for version in SPECS:
                    ready=all(counts[version][k]>=10 for k in group_names(version))
                    if (ready and not queues[version]) or (args.probe and len(used[version])>=args.probe):
                        completed.add(version)
                    if len(records[version])//2>=240 and not ready:
                        raise ValueError('review ceiling240 accepted groups reached; need inspect missing strata')
        final=status('probe_complete' if args.probe else 'generated_pending_render')
        for v in SPECS:
            if not args.probe and (final['versions'][v]['missing'] or queues[v]):
                raise ValueError('bounded review incomplete: '+v+' '+repr(final['versions'][v]['missing']))
            if not args.probe:
                sizes=Counter(r['detail']['trim']['size_class'] for r in records[v] if r['label'])
                if sizes['smaller']*10 != 7*sum(sizes.values()):raise AssertionError('accepted70/30 quota')
            input_hashes=[r['model_input_sha256'] for r in audits[v]]
            if len(input_hashes)!=len(set(input_hashes)):raise AssertionError('duplicate model inputs within version')
            save_json(root/v/'generation_complete.json',final['versions'][v])
        keys=[{r['source_base_key'] for r in records[v]} for v in SPECS]
        if not keys[0].isdisjoint(keys[1]):raise AssertionError('cross-version base identity overlap')
        hashes=[{r['model_input_sha256'] for r in audits[v]} for v in SPECS]
        if not hashes[0].isdisjoint(hashes[1]):raise AssertionError('duplicate model input across versions')
        exclusions={v:sorted(keys[i]) for i,v in enumerate(SPECS)}
        save_json(root/'curriculum_base_exclusion.json',dict(status='review_only_proposed_exclusion',
            unchanged_v14_v17=True,source_pair_keys=exclusions,review_approval_required_before_expansion=True,
            policy='exclude these original base views from a FUTURE curriculum base pool, not archived datasets'))
        save_json(root/('probe_complete.json' if args.probe else 'generation_complete.json'),final)
        if not args.probe:
            status('rendering_audited_examples')
            command=[sys.executable,'-m','experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.curriculum_data_v18_v19.render_review',
                '--root',str(root),'--out',str(root/'rendered_01')]
            subprocess.run(command,check=True)
            rendered=root/'rendered_01'/'rendered.json'
            final=status('complete')
            final.update(rendered_sha256=digest(rendered),rendered_path=str(rendered),
                human_review_required=True,full_generation_authorized=False)
            save_json(root/'pipeline_complete.json',final)
        print(json.dumps(final,ensure_ascii=False))
    except BaseException as error:
        save_json(root/'pipeline_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            no_automatic_retry=True,gpu_used=False,training_unchanged=True))
        status('failed');raise

if __name__=='__main__':main()
