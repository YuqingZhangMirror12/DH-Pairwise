"""New release, resampling ONLY positive geometry violations in the parent.

Unchanged samples/proofs remain at their original immutable paths and hashes.
Replacement uses the same type and J base; negative pairs are never changed.
TEST uses the fixed generator predicate, never an inferred score or new tuning.
"""
import argparse
from collections import Counter
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

if __package__:
    from . import generate as g, full_generate as f
    from .geometry_contract import CONTRACT, violations
    from .pair_identity import base_pair_sha256
else:
    import generate as g
    import full_generate as f
    from geometry_contract import CONTRACT, violations
    from pair_identity import base_pair_sha256

REPAIR_SEEDS = {'train':26093094, 'select':26093095, 'test':26093096}


def metric_module(path):
    spec=importlib.util.spec_from_file_location('fixed_repair_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def initialize(reference_dir, rows, counts, metrics_code):
    f.initialize(reference_dir,rows,counts)
    metrics=metric_module(metrics_code)
    ref=g.CONTEXT['reference']; previous=ref.finalize

    def finalize(a,b,meta,label=True):
        result,reason=previous(a,b,meta,label)
        if result is None or not label:
            return result,reason
        ia,ib,t,*_=result
        measured=metrics.measure(SimpleNamespace(mask_a=ia[None],mask_b=ib[None],
            translation_a_to_b_rc=t,label=True))
        reasons=violations(meta['type'],measured)
        g.CONTEXT['geometry_attempts'].append(dict(reasons=reasons,passed=not reasons))
        if reasons:
            reason='geometry:'+','.join(reasons)
            g.CONTEXT['rejections'][reason]+=1
            # The wrapped finalizer has just logged this same accepted attempt.
            g.CONTEXT['finalization_attempts'][-1]['reason']=reason
            return None,reason
        return result,None

    ref.finalize=finalize


def replace_one(job):
    old,out,seed,reasons,parent_sha=job
    if not old['label'] or not reasons:
        raise ValueError('Only an identified positive outlier may be regenerated')
    kind=old['recipe'][-1]; idx=int(old['id'].rsplit('_',1)[1])
    ref=g.CONTEXT['reference']; original=ref.base_of
    ref.base_of=lambda *args:old['meta']['base']
    g.CONTEXT['geometry_attempts']=[]
    try:
        row=f.generate_one((kind,1,idx,old['split'],out,seed))
    finally:
        ref.base_of=original
    if row.get('failed'):
        return dict(omitted=True,id=old['id'],pair_id=old['pair_id'],label=True,
            recipe=old['recipe'],base=old['meta']['base'],reasons=reasons,
            candidate_attempt_budget=300,exhaustion=row,parent_manifest_sha256=parent_sha,
            note='User permits a small shortfall; no fallback to an invalid pair or another type.')
    if (row['id']!=old['id'] or row['meta']['base']!=old['meta']['base']
            or row['pair_id']==old['pair_id'] or not g.CONTEXT['geometry_attempts'][-1]['passed']):
        raise ValueError('Replacement identity/base/geometry contract differs')
    row.update(generation_seed=seed,geometry_contract=CONTRACT,
        geometry_attempts=g.CONTEXT['geometry_attempts'],
        replacement_of=dict(parent_manifest_sha256=parent_sha,pair_id=old['pair_id'],
            sample_sha256=old['sample_sha256'],reasons=reasons))
    return row


def replacement_plan(entries, records, seed, parent_sha, measure_test):
    """Retain every passing row verbatim except its explicit replay-seed field."""
    lookup={r['pair_id']:r for r in records}
    if len(lookup)!=len(entries):raise ValueError('Parent audit membership differs')
    kept=[]; jobs=[]
    for entry in entries:
        row=lookup[entry['pair_id']]
        if row['sample_sha256']!=entry['sample_sha256'] or row['positive']!=entry['label']:
            raise ValueError('Parent audited row differs')
        reasons=[]
        if entry['label']:
            measured=measure_test(entry) if entry['split']=='test' else row['metrics']
            reasons=violations(entry['recipe'][-1],measured)
        if reasons:
            jobs.append((entry,reasons))
        else:
            kept.append(dict(entry,generation_seed=entry.get('generation_seed',seed)))
    return kept,jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-root','parent-audit','reference-dir','source-admission','real-audit','metrics-code','official-code','output-new'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    old_manifest=a.parent_root/'manifest.json';old_plan=a.parent_root/'plan.json'
    manifest=json.loads(old_manifest.read_text());plan=json.loads(old_plan.read_text());audit=json.loads(a.parent_audit.read_text())
    split=manifest['split']; counts=plan['counts_per_label'];n=sum(counts.values())*2
    assert counts==f.COUNTS[split] or (plan.get('preflight') is True and split=='train' and counts==dict(M=5,J=25,R=5))
    assert not manifest['failed'] and manifest['plan_sha256']==g.sha(old_plan)
    assert audit['status']=='passed_integrity_and_supervision' and audit['source_manifest_sha256']==g.sha(old_manifest)
    assert len(manifest['entries'])==len(audit['records'])==audit['rows']==n
    assert g.sha(a.source_admission)==plan['source_admission_sha256'] and g.sha(a.real_audit)==plan['real_audit_sha256']
    _,hashes=g.load_reference(a.reference_dir);assert hashes==plan['reference_sha256']
    _,pool,_=g.source_pool(json.loads(a.source_admission.read_text()),json.loads(a.real_audit.read_text()),split)
    sys.path.insert(0,str(a.official_code))
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    metrics=metric_module(a.metrics_code)
    frozen={str(p):g.sha(p) for p in (old_manifest,old_plan,a.parent_audit,a.source_admission,a.real_audit,a.metrics_code)}
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'samples').mkdir();(out/'proof').mkdir()
    started=time.time()
    g.save(out/'launch.json',dict(pid=os.getpid(),started_unix=started,split=split,
        parent_manifest_sha256=g.sha(old_manifest),file_hashes=frozen,contract=CONTRACT,gpu=False))
    try:
        for entry in manifest['entries']:
            assert g.sha(entry['sample_path'])==entry['sample_sha256']
            if entry['label']:assert g.sha(entry['proof_path'])==entry['proof_sha256']
        kept,jobs=replacement_plan(manifest['entries'],audit['records'],manifest['seed'],g.sha(old_manifest),
            lambda e:metrics.measure(load_sample(e['sample_path'])[0]))
        seed=26093097 if plan.get('preflight') else REPAIR_SEEDS[split]
        new_plan=dict(plan,seed=seed,parent_manifest=dict(path=str(old_manifest),sha256=g.sha(old_manifest)),
            parent_audit=dict(path=str(a.parent_audit),sha256=g.sha(a.parent_audit)),
            repair_code_sha256=g.sha(__file__),metrics_code_sha256=g.sha(a.metrics_code),
            geometry_contract=CONTRACT,repair_candidates=[dict(id=e['id'],pair_id=e['pair_id'],reasons=r) for e,r in jobs],
            rows_retained=len(kept),rows_to_replace=len(jobs),gpu=False,training_admitted=False)
        g.save(out/'plan.json',new_plan)
        print(json.dumps(dict(stage='resampling',split=split,unchanged=len(kept),replace=len(jobs))),flush=True)
        completed=[];omitted=[]
        with (out/'replacement_records.jsonl').open('x') as stream:
            with multiprocessing.get_context('spawn').Pool(a.workers,initializer=initialize,
                initargs=(a.reference_dir,pool,counts,a.metrics_code)) as workers:
                tasks=[(e,str(out),seed,r,g.sha(old_manifest)) for e,r in jobs]
                for row in workers.imap_unordered(replace_one,tasks,chunksize=1):
                    (omitted if row.get('omitted') else completed).append(row)
                    stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
                    if len(omitted)>20:raise ValueError('More than twenty omissions require a new decision')
                    if (len(completed)+len(omitted))%25==0 or len(completed)+len(omitted)==len(jobs):
                        print(json.dumps(dict(replaced=len(completed),omitted=len(omitted),total=len(jobs),split=split)),flush=True)
        g.save(out/'omissions.json',dict(rows=omitted,count=len(omitted),requested_population=n,
            actual_population=n-len(omitted),strict_geometry_not_relaxed=True,training_admitted=False))
        entries=sorted(kept+completed,key=lambda r:r['id'])
        seen_base={}
        for entry in entries:
            if not entry['label']:continue
            with np.load(entry['proof_path']) as fz:proof={k:fz[k] for k in fz.files}
            identity=base_pair_sha256(f.s.unpack(proof,'cut_a'),f.s.unpack(proof,'cut_b'))
            if identity in seen_base:
                raise ValueError('Repeated PRE-DAMAGE pair; retain output for resampling, never train: '+entry['id'])
            seen_base[identity]=entry['pair_id'];entry['pre_damage_pair_sha256']=identity
        omitted_ids={r['id'] for r in omitted}
        expected_entries=[r for r in manifest['entries'] if r['id'] not in omitted_ids]
        assert len(entries)==n-len(omitted) and {r['id'] for r in entries}=={r['id'] for r in expected_entries}
        assert Counter((r['recipe'],r['label'],r['meta']['base']) for r in entries)==Counter(
            (r['recipe'],r['label'],r['meta']['base']) for r in expected_entries)
        g.save(out/'manifest.json',dict(schema='codex-v42-strict-repair/1',split=split,seed=seed,
            entries=entries,failed=[],omissions_sha256=g.sha(out/'omissions.json'),plan_sha256=g.sha(out/'plan.json'),training_admitted=False))
        assert all(g.sha(path)==digest for path,digest in frozen.items())
        g.save(out/'generation_complete.json',dict(status='generated',generated=len(entries),expected=len(entries),requested=n,failed=0,
            manifest_sha256=g.sha(out/'manifest.json'),rows_retained=len(kept),rows_replaced=len(completed),
            rows_omitted=len(omitted),omissions_sha256=g.sha(out/'omissions.json'),
            parent_files_unchanged=True,seconds=time.time()-started,training_admitted=False,gpu_training_started=False))
    except Exception as exc:
        g.save(out/'failure.json',dict(status='failed',error=repr(exc),training_admitted=False))
        raise


if __name__=='__main__':main()
