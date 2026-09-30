"""Isolated data-only revision: export, actual-loader validation, distributions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from ..s7_compound_v1.materialize import read,save_json,digest
from .reuse_completed import compatible_profile,reuse


def validate_human_approval(profile, pilot, approval):
    """An explicit receipt binds approval to the exact completed review pilot."""
    if approval is None:
        raise ValueError('800-pair human review required. Supply explicit approved-pilot receipt.')
    record=read(approval)
    if (record.get('schema')!='s7-reviewed-pilot-approval/1'
            or record.get('user_approved') is not True
            or record.get('approved_train_pairs')!=24000
            or record.get('approved_validation_pairs')!=6000
            or record.get('profile_sha256')!=digest(profile)
            or Path(record.get('pilot_root','')).resolve()!=pilot.resolve()
            or record.get('pilot_protocol_sha256')!=digest(pilot/'protocol.json')
            or record.get('pilot_validation_sha256')!=digest(pilot/'validation.json')
            or record.get('pilot_pixel_audit_sha256')!=digest(pilot/'independent_pixel_audit.json')):
        raise ValueError('approval is missing or differs from the reviewed pilot/profile')
    if (read(pilot/'status.json').get('sample_count')!=800
            or read(pilot/'validation.json').get('status')!='passed'
            or read(pilot/'independent_pixel_audit.json').get('status')!='passed'
            or read(pilot/'pipeline_status.json').get('stage')!='data_ready'
            or read(pilot/'pipeline_status.json').get('status')!='complete'):
        raise ValueError('review pilot is not complete and verified')
    return record


def run(a):
    root=Path(a.root).resolve();out=root/a.output
    profile=Path(a.profile).resolve()
    if out.exists():raise ValueError('output already exists; no overwrite or automatic restart')
    approval=None
    if a.groups>400 and read(profile).get('human_pilot_approval_required_before_full'):
        approval=validate_human_approval(profile,root/a.pilot,a.approval)
    if a.groups==12000:
        pilot=root/a.pilot
        if read(pilot/'validation.json')['status']!='passed':raise ValueError('passed pilot required')
        pilot_profile=read(pilot/'protocol.json')['distribution_revision']
        if not compatible_profile(pilot_profile,read(profile)):
            raise ValueError('pilot and full profiles differ')
    out.mkdir()
    if approval is not None:
        save_json(out/'human_approval.json',approval)
    for name in ('s7_balanced_v2','s7_compound_v1','gap_distribution_v2'):
        shutil.copytree(Path(__file__).parent.parent/name,out/'generator_snapshot'/name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    if a.reuse_completed_from:
        reuse(a.reuse_completed_from,out,read(profile),groups=a.groups,seed=a.seed,source_plan=root/'sources_v2.json')
    base='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
    commands=[('generate',[sys.executable,'-u','-m',base+'s7_balanced_v2.materialize',
        '--sources',str(root/'sources_v2.json'),'--out',str(out),'--groups',str(a.groups),
        '--workers',str(a.workers),'--seed',str(a.seed),'--attempts','1024','--profile',str(profile)]),
        ('validate',[sys.executable,'-u','-m',base+'s7_balanced_v2.validate','--root',str(out)]),
        ('describe',[sys.executable,'-u','-m',base+'s7_balanced_v2.describe','--root',str(out)]),
        ('measure',[sys.executable,'-u','-m',base+'gap_distribution_v2.measure','sim',
            '--manifest',str(out/'train_s7b_24k.json'),'--out',str(root/'distributions'),
            '--name',a.output,'--workers',str(min(a.workers,16))])]
    if a.groups!=12000:commands=[x for x in commands if x[0]!='describe']
    if read(profile).get('layered_damage_exclusive'):
        commands.insert(2,('pixel_audit',[sys.executable,'-u','-m',base+'s7_balanced_v2.audit_layered',
            '--root',str(out),'--workers',str(min(a.workers,12))]))
    if read(profile).get('conservative_v12'):
        commands=[x for x in commands if x[0]!='describe']
        commands.append(('review',[sys.executable,'-u','-m',base+'s7_balanced_v2.conservative_review',
            '--root',str(out),'--out',str(out/'conservative_review'),'--per-type',
            str(read(profile).get('review_examples_per_type',10 if a.groups==12000 else 2))]))
    start=time.time();os.nice(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for stage,command in commands:
            with (out/(stage+'.log')).open('a') as stream:
                child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                save_json(out/'pipeline_status.json',dict(status='running',stage=stage,parent_pid=os.getpid(),
                    child_pid=child.pid,command=command,start=start,profile_sha256=digest(profile),training_started=False))
                if child.wait():raise RuntimeError(stage+' failed; no fallback or training started')
        save_json(out/'pipeline_status.json',dict(status='complete',stage='data_ready',parent_pid=os.getpid(),
            elapsed_seconds=time.time()-start,training_started=False))
    except BaseException as exc:
        save_json(out/'pipeline_status.json',dict(status='failed',stage=stage,parent_pid=os.getpid(),
            error=repr(exc),elapsed_seconds=time.time()-start,training_started=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--output',required=True);p.add_argument('--groups',type=int,default=400)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=26092343)
    p.add_argument('--pilot',default='pilot_distribution_003')
    p.add_argument('--approval',help='Explicit user approval bound to the completed800-pair pilot')
    p.add_argument('--reuse-completed-from',help='Explicit failed generation with unchanged schedule/recipe; original files preserved')
    a=p.parse_args();run(a)


if __name__=='__main__':main()
