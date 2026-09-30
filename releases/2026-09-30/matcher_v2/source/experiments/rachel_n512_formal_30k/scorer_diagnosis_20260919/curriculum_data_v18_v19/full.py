"""Approved CPU-only full release; fixed quotas, fail closed, immutable inputs."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import REVISION
from .spec import SPECS
from .full_plan import build,tickets,choose,accept_unique,validate_targets
from .admission import load as load_admission
from .seam_contract import CONTRACT
from .generate import process
from .audit import audit_record
from ..aggressive_data_full_v17.source import initialize,STATE
from ..aggressive_data_full_v17.resources import low_priority,snapshot,dispatch_allowed,worker_cap
from ..s7_compound_v1.materialize import read,save_json,digest
from ..seam_context_v3.prepare import family


def init(config):
    low_priority();initialize(config,'train')


def recheck(record):
    return audit_record(record,STATE['root'])


def quantiles(values):
    values=[float(x) for x in values if x is not None]
    if not values:return dict(count=0,quantiles=None)
    return dict(count=len(values),probabilities=[0,.1,.25,.5,.75,.9,.95,1],
        quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,.95,1]).tolist())


def family_set(entries):
    return {family(e['source_row']['fragment_'+s]['split_unit_id']) for e in entries for s in 'ab'}


def finish(root,config,prepared,records,audits,accepted_keys,accepted_hashes):
    summaries={};all_keys=set();all_hashes=set()
    for v in SPECS:
        rows=records[v];checks=audits[v];target=prepared['targets'][v]
        if len(rows)!=target or len(checks)!=target:raise ValueError('full target count not reached: '+v)
        if any(r['status']!='passed' for r in checks):raise ValueError('independent audit failed')
        by_id={r['id']:r for r in checks}
        if len(by_id)!=target or set(by_id)!={r['id'] for r in rows}:raise ValueError('audit identities mismatch')
        labels=Counter(bool(r['label']) for r in rows)
        if labels!={False:target//2,True:target//2}:raise ValueError('balanced label quota not met')
        pos=[r for r in rows if r['label']]
        seam_checks=[by_id[r['id']]['connectable_seam'] for r in pos]
        if any(x['contract']!=CONTRACT or not x['eligible'] or x['primary_damage_included'] for x in seam_checks):
            raise ValueError('independent original20/final30 contract not passed')
        sides=Counter(r['detail']['trim']['size_class'] for r in pos)
        if sides!={'smaller':7*len(pos)//10,'larger':3*len(pos)//10}:raise ValueError('70/30 accepted side quota')
        recipes=Counter(r['recipe'] for r in pos)
        if dict(recipes)!=prepared['recipe_group_quotas'][v]:raise ValueError('fixed recipe quotas not met')
        for i in range(0,len(rows),2):accept_unique(rows[i:i+2],[by_id[r['id']] for r in rows[i:i+2]],all_keys,all_hashes)
        summary=dict(status='passed',pairs=len(rows),positives=len(pos),negatives=len(rows)-len(pos),
            accepted_side_counts=dict(sides),recipe_group_counts=dict(recipes),
            seam_contract=CONTRACT,all_original20_and_final30_passed=True,
            original_common_over_smaller_perimeter=quantiles(x['original_over_smaller_perimeter'] for x in seam_checks),
            final_connectable_over_original_common=quantiles(x['final_over_original'] for x in seam_checks),
            primary_damage_counted_as_connectable=False,final_light_limit_per_side_px=4.,
            notch_counts=dict(Counter(r['requested_gap_count'] for r in pos if r['recipe'].startswith('gaps'))),
            source_base_keys=len({r['source_base_key'] for r in rows}),
            pristine_fraction=quantiles(r['detail']['pristine_seam']['conservative_fraction'] for r in pos),
            primary_gap_peak_px=quantiles(r['detail']['gap']['primary_gap_peak_px'] for r in pos),
            trim_fraction=quantiles(1-r['detail']['trim']['retained_fraction'] for r in pos),
            cut_area_loss=quantiles(r['detail']['trim']['material_removed_fraction'] for r in rows),
            inherited_gt_matches=quantiles(r['inherited_correspondences'] for r in pos),
            light_actual_depth_px=quantiles(x['applied_max_depth_px'] for r in rows for x in r['detail']['background'].values()),
            light_requested_peaks_px=quantiles(p for r in rows for x in r['detail']['background'].values() for p in x['requested_peak_depths_px']),
            pristine_quota_enforced=False,contact_protection_enabled=False,v14_fallback_count=0)
        save_json(root/v/'summary.json',summary)
        save_json(root/v/'pixel_audit.json',dict(status='passed',pairs=len(checks),receipts=checks,
            independent_pixels_and_actual_supervision=True,all_rows_rechecked_after_generation=True))
        save_json(root/v/'generation_complete.json',summary);summaries[v]=summary
    if all_keys!=accepted_keys or all_hashes!=accepted_hashes:raise ValueError('online/final uniqueness reconciliation failed')
    heldout={}
    for split in ('cal','select','test'):
        p=Path(config['baselines'][split])/'archive_manifest.json'
        heldout[split]=dict(manifest=str(p),sha256=digest(p),families=sorted(family_set(read(p)['entries'])))
    evaluation_families=set().union(*(set(x['families']) for x in heldout.values()))
    generated_sources=set().union(*(set(r['source_families'])|set(r['donor_families']) for rows in records.values() for r in rows))
    if generated_sources&evaluation_families:raise ValueError('TRAIN or donor overlaps heldout family')
    source_audit=dict(status='passed',within_and_cross_version_original_pair_unique=True,
        within_and_cross_version_model_input_unique=True,unique_original_pairs=len(all_keys),
        unique_model_inputs=len(all_hashes),heldout=heldout,generated_families=sorted(generated_sources),
        heldout_family_overlap=[],real_data_used=False,test_used=False)
    save_json(root/'source_audit.json',source_audit)
    save_json(root/'curriculum_base_exclusion.json',dict(status='full_generation_passed',
        source_pair_keys=sorted(all_keys),unchanged_v14_v17=True,
        policy='exclude every archived view sharing one of these original pair identities from a future curriculum basic pool'))
    # A new reference manifest only; leave all original archives and ongoing runs alone.
    base=Path('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03/train')
    manifest=base/'manifest.json'
    if not manifest.is_file():
        alternatives=sorted(base.glob('*manifest*.json'))
        if len(alternatives)!=1:raise ValueError('cannot identify v17 TRAIN manifest for curriculum exclusion')
        manifest=alternatives[0]
    archive=read(manifest);entries=archive['entries']
    from .plan import base_key
    kept=[dict(e,artifact_path=str((base/e['artifact_path']).resolve())) for e in entries if base_key(e) not in all_keys]
    save_json(root/'v17_curriculum_base_reference_manifest.json',dict(entries=kept,
        archive_manifest=str(manifest),archive_sha256=digest(manifest),
        excluded_rows=len(entries)-len(kept),original_rows=len(entries),training_started=False,
        note='reference-only candidate basic pool; original v17 unchanged; curriculum stage sampling not yet selected'))
    return dict(status='complete',versions=summaries,source_audit_sha256=digest(root/'source_audit.json'),
        base_exclusion_sha256=digest(root/'curriculum_base_exclusion.json'),
        v17_basic_pool_rows=len(kept),gpu_used=False,training_started=False,full_generation_authorized=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--v175-pairs',type=int,required=True);p.add_argument('--v18-pairs',type=int,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--probe-root',required=True)
    p.add_argument('--admission-root',required=True)
    a=p.parse_args();targets={'v17.5':a.v175_pairs,'v18':a.v18_pairs};validate_targets(targets)
    root=Path(a.out).resolve();source=Path.cwd();probe=Path(a.probe_root)
    if root.exists():raise ValueError('fresh output only; no automatic restart')
    binding=read(source/'source_binding.json')
    for name,sha in binding.items():
        if digest(source/name)!=sha:raise ValueError('source binding changed:'+name)
    if (probe/'pipeline_failure.json').exists() or not (probe/'probe_complete.json').is_file():
        raise ValueError('actual-pixel source-matched probe must finish first')
    protocol=read(probe/'protocol.json')
    if protocol['revision']!=REVISION or protocol['source_binding_sha256']!=digest(source/'source_binding.json'):
        raise ValueError('probe source or revision differs')
    for v in SPECS:
        audit=read(probe/v/'pixel_audit.json')
        if audit['status']!='passed' or audit['pairs']<2:raise ValueError('each version requires real passed probe pairs')
        rows=read(probe/v/'manifest.json')['entries'];checks={r['id']:r for r in audit['receipts']}
        if len(rows)!=audit['pairs'] or set(checks)!={r['id'] for r in rows}:
            raise ValueError('probe manifest/audit identity mismatch')
        for row in rows:
            check=checks[row['id']]
            if check['status']!='passed':raise ValueError('probe pixel audit did not pass')
            if row['label'] and (check['connectable_seam']['contract']!=CONTRACT or not check['connectable_seam']['eligible']):
                raise ValueError('probe did not pass current original20/final30 contract')
            for field,sha_field in [('sample_path','sample_sha256'),('proof_path','proof_sha256'),
                                    ('target_metadata','target_metadata_sha256'),('latent_seam_artifact','latent_sha256')]:
                if row.get(field):
                    path=Path(row[field]);path=path if path.is_absolute() else probe/v/path
                    if digest(path)!=row[sha_field]:raise ValueError('probe actual artifact hash mismatch:'+field)
    root.mkdir(parents=True);lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    low_priority();started=time.time()
    config=read('/root/autodl-tmp/aggressive_data_v17_full30k_20260927/dataset_03/config.json')
    resource=snapshot(root);workers=worker_cap(resource,a.workers)
    config.update(out=str(root),workers=workers,full_generation_authorized=True,geometry_attempts=20)
    prepared=load_admission(a.admission_root,build(config['baselines']['train'],targets),source)
    save_json(root/'plan.json',prepared)
    config['plan_sha256']=digest(root/'plan.json');save_json(root/'config.json',config)
    save_json(root/'protocol.json',dict(revision=REVISION,contracts=SPECS,targets=targets,
        recipe_group_quotas=prepared['recipe_group_quotas'],source_binding_sha256=digest(source/'source_binding.json'),
        baseline_manifest_sha256=prepared['baseline_manifest_sha256'],plan_sha256=config['plan_sha256'],
        workers=workers,nice=15,gpu_used=False,training_started=False,full_generation_authorized=True,
        seam_contract=CONTRACT,admission=prepared['admission'],
        probe_root=str(probe),probe_receipt_sha256=digest(probe/'probe_complete.json'),
        pristine_minimum_removed=True,unchanged_archives=['v14','v17'],source_assignment=prepared['assignment']))
    records={v:[] for v in SPECS};audits={v:[] for v in SPECS};attempts={v:[] for v in SPECS}
    requests=tickets(prepared['recipe_group_quotas']);attempted={v:set() for v in SPECS}
    accepted_keys=set();accepted_hashes=set();cursor=0
    def status(state):
        value=dict(status=state,updated_unix=time.time(),elapsed_seconds=time.time()-started,
            gpu_used=False,training_started=False,full_generation_authorized=True,workers=workers,versions={})
        for v in SPECS:
            value['versions'][v]=dict(target_pairs=targets[v],pairs=len(records[v]),accepted_groups=len(records[v])//2,
                attempted_groups=len(attempts[v]),pending_groups=sum(map(len,requests[v].values())),
                missing_by_recipe={k:len(q) for k,q in requests[v].items() if q},
                successful_side_counts=dict(Counter(r['detail']['trim']['size_class'] for r in records[v] if r['label'])))
        save_json(root/'pipeline_status.json',value);return value
    try:
        with ProcessPoolExecutor(workers,initializer=init,initargs=(config,)) as pool:
            while any(q for by_recipe in requests.values() for q in by_recipe.values()):
                resource=snapshot(root);save_json(root/'resources.json',resource)
                if not dispatch_allowed(resource):status('resource_wait');time.sleep(30);continue
                batch=[];reserved=set();names=list(SPECS)
                for i in range(workers):
                    task=None
                    for j in range(len(names)):
                        v=names[(cursor+j)%len(names)]
                        task=choose(prepared['tasks'][v],requests[v],attempted[v],accepted_keys,reserved)
                        if task is not None:cursor=(cursor+j+1)%len(names);break
                    if task is None:break
                    batch.append(task)
                if not batch:raise ValueError('source pool exhausted under approved geometry/identity/quota rules; no forced fallback')
                for task,result in zip(batch,pool.map(process,batch)):
                    v=task['version'];receipt={k:x for k,x in result.items() if k not in ('records','audit_rows')}
                    attempts[v].append(receipt);save_json(root/v/'attempts'/f"{task['slot']:05d}.json",receipt)
                    if result['status'] in ('passed','committed'):
                        accept_unique(result['records'],result['audit_rows'],accepted_keys,accepted_hashes)
                        records[v].extend(result['records']);audits[v].extend(result['audit_rows'])
                    else:requests[v][task['recipe']].append(task['size_class'])
                for v in SPECS:
                    save_json(root/v/'manifest.json',dict(entries=records[v],split='train',revision=REVISION))
                    save_json(root/v/'pixel_audit.json',dict(status='passed',pairs=len(audits[v]),receipts=audits[v],
                        independent_pixels_and_actual_supervision=True,all_rows_rechecked_after_generation=False))
                status('generating')
            status('independent_full_reaudit')
            for v in SPECS:
                checks=[]
                # Bounded batches allow resource gating during the independent pass too.
                for start in range(0,len(records[v]),workers):
                    while not dispatch_allowed(snapshot(root)):status('audit_resource_wait');time.sleep(30)
                    checks.extend(pool.map(recheck,records[v][start:start+workers]))
                audits[v]=checks
        final=finish(root,config,prepared,records,audits,accepted_keys,accepted_hashes)
        final.update(elapsed_seconds=time.time()-started,completed_unix=time.time(),
            source_binding_sha256=digest(source/'source_binding.json'))
        save_json(root/'generation_complete.json',final)
        save_json(root/'pipeline_complete.json',final);status('complete');print(json.dumps(final))
    except BaseException as error:
        save_json(root/'pipeline_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            automatic_retry=False,partial_outputs_preserved=True,gpu_used=False,training_unchanged=True))
        status('failed');raise


if __name__=='__main__':main()
