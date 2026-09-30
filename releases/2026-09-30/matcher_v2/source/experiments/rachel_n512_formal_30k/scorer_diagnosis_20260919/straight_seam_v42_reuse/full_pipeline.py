"""Fail-closed CPU preparation: 70-pair gate then TRAIN6000/SELECT900/TEST900.

No training or model evaluation. TEST integrity cannot tune the generator.
Each step is immutable, single-attempt; a failure keeps all previous outputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import generate as g
    from .pipeline import write
else:
    import generate as g
    from pipeline import write


def commands(args,out):
    source=Path(__file__).resolve().parent
    common=['--reference-dir',str(args.reference_dir),'--source-admission',str(args.source_admission),
            '--real-audit',str(args.real_audit)]
    gen=[sys.executable,str(source/'full_generate.py')]+common+['--workers',str(args.workers)]
    audit=[sys.executable,str(source/'audit_full.py')]+common+[
        '--target-code',str(args.target_code),'--metrics-code',str(args.metrics_code),'--official-code',str(args.official_code)]
    steps=[]
    for name,split,extra in [('preflight70','train',['--preflight']),('train6000','train',[]),
                             ('select900','select',[]),('test900','test',[])]:
        steps.append((name+'_generate',gen+['--split',split,'--output-new',str(out/name)]+extra))
        steps.append((name+'_audit',audit+['--root',str(out/name),'--output-new',str(out/(name+'_audit'))]))
    return steps


def final_receipt(out):
    manifests={};all_ids=set();all_inputs=set();splits={}
    for name,split,expected in [('train6000','train',6000),('select900','select',900),('test900','test',900)]:
        manifest=out/name/'manifest.json';audit_path=out/(name+'_audit')/'audit.json'
        complete=json.loads((out/(name+'_audit')/'complete.json').read_text())
        assert complete['status']=='complete' and complete['rows']==expected
        assert complete['audit_sha256']==g.sha(audit_path) and complete['source_manifest_sha256']==g.sha(manifest)
        data=json.loads(manifest.read_text());audit=json.loads(audit_path.read_text())
        assert data['split']==split and audit['status']=='passed_integrity_and_supervision'
        assert audit['rows']==len(data['entries'])==expected and not data['failed']
        ids={r['pair_id'] for r in audit['records']};inputs={r['model_input_sha256'] for r in audit['records']}
        assert not (ids&all_ids or inputs&all_inputs),'cross-split pair/input collision'
        all_ids|=ids;all_inputs|=inputs
        families=set();source_paths=set()
        for entry in data['entries']:
            for donor in entry['attempted_donor_references']:
                families.add(donor['source_family']);source_paths.add(donor['path'])
        for previous in splits.values():
            assert not (families&previous['families'] or source_paths&previous['paths']),'cross-split donor collision'
        splits[split]=dict(families=families,paths=source_paths)
        manifests[split]=dict(rows=expected,manifest_path=str(manifest),manifest_sha256=g.sha(manifest),
            audit_path=str(audit_path),audit_sha256=g.sha(audit_path),seed=data['seed'],
            donor_families=len(families),donor_paths=len(source_paths))
    assert len({value['seed'] for value in manifests.values()})==3
    return dict(status='complete_generation_integrity_supervision',datasets=manifests,
        total_rows=7800,cross_split_model_input_overlap=0,cross_split_source_family_overlap=0,
        training_admitted=False,model_inference=False,gpu=False,
        pending=['Population acceptance and frozen E32 SELECT-only calibration.',
                 'Final source-lineage admission, execution-ledger binding and training GPU gates.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir','source-admission','real-audit','target-code','metrics-code','official-code','output-new'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False)
    steps=commands(a,out);frozen={}
    for directory in (Path(__file__).resolve().parent,a.reference_dir):
        frozen.update({str(path):g.sha(path) for path in directory.glob('*.py')})
    for path in (a.source_admission,a.real_audit,a.target_code,a.metrics_code):frozen[str(path)]=g.sha(path)
    launch=dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(),commands=steps,file_hashes=frozen,automatic_retries=0,
        workers=a.workers,gpu=False,training_admitted=False,output_root=str(out))
    write(out/'launch.json',launch);completed=[]
    try:
        for name,command in steps:
            assert all(g.sha(path)==digest for path,digest in frozen.items()),'frozen source changed'
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            started=time.time()
            with (out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                write(out/'status.json',dict(stage=name,child_pid=child.pid,completed=completed,
                    started_unix=started,training_admitted=False))
                rc=child.wait()
            receipt=dict(stage=name,returncode=rc,seconds=time.time()-started,finished_unix=time.time())
            write(out/(name+'_return.json'),receipt)
            if rc:
                write(out/'failure.json',dict(**receipt,completed=completed));return rc
            completed.append(receipt)
        assert all(g.sha(path)==digest for path,digest in frozen.items())
        result=final_receipt(out);result.update(completed=completed,finished_unix=time.time(),files_unchanged=True)
        write(out/'complete.json',result)
        write(out/'status.json',dict(stage='complete',completed=completed,training_admitted=False))
        return 0
    except Exception as exc:
        write(out/'failure.json',dict(status='failed',error=repr(exc),completed=completed));return 2


if __name__=='__main__':raise SystemExit(main())
