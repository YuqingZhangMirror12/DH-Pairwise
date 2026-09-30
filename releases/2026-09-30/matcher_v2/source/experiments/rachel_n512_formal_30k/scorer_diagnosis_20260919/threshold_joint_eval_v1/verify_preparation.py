"""CPU fixtures only; never open a real model checkpoint or dataset."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

from entry import bootstrap, inventory, digest, RELATIVE


def worker(args):
    training, helper = bootstrap(args.source, args.common_source, args.joint_source)
    configuration = training.TrainingConfig().record()
    if configuration.get('threshold_policy', {}).get('pose_diameter_px') != 16.:
        raise ValueError('not registered fixed16 threshold implementation')
    joint = importlib.import_module(training.__package__+'.config').TrainingConfig
    expected_joint = args.kind == 'joint'
    if hasattr(joint(), 'matcher_learning_rate') != expected_joint:
        raise ValueError('joint and frozen configurations were not isolated')
    names = ['consensus_joint_eval_adapter.'+n for n in ('test_contracts','test_evaluate','test_entry')]
    # Preserve the original frozen-source guards and numeric export tests.
    names += ['consensus_joint_eval_common.'+n for n in ('test_frozen','test_evaluate',
        'test_snapshot','test_audit','test_attention_trace','test_attention_gate','test_view_data')]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    record=dict(kind=args.kind,status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        real_inference_performed=False,real_checkpoints_opened=False,gpu_inference_performed=False,
        training_import=str(Path(training.__file__).resolve()),
        helper_sha256=digest(helper.__file__),implementation=inventory(Path(args.source)/RELATIVE))
    Path(args.out).write_text(json.dumps(record,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('common-source','joint-source','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--frozen-source');p.add_argument('--real-plan')
    p.add_argument('--worker',action='store_true');p.add_argument('--source');p.add_argument('--kind',choices=('joint','frozen'))
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation only')
    if args.worker:return worker(args)
    if not args.frozen_source or not args.real_plan:raise ValueError('both implementations and source split required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    before=dict(adapter=inventory(Path(__file__).parent),common=inventory(args.common_source),
        joint=inventory(Path(args.joint_source)/RELATIVE),frozen=inventory(Path(args.frozen_source)/RELATIVE))
    results=[]
    for kind,source in (('joint',args.joint_source),('frozen',args.frozen_source)):
        result_path=out/(kind+'_cpu.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--kind',kind,'--source',source,
                 '--common-source',args.common_source,'--joint-source',args.joint_source,'--out',str(result_path)]
        with (out/(kind+'_cpu.log')).open('xb') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        result_record=json.loads(result_path.read_text()) if result_path.exists() else dict(kind=kind,status='failed',tests=0,errors=1,failures=0)
        result_record['returncode']=result.returncode;results.append(result_record)
    after=dict(adapter=inventory(Path(__file__).parent),common=inventory(args.common_source),
        joint=inventory(Path(args.joint_source)/RELATIVE),frozen=inventory(Path(args.frozen_source)/RELATIVE))
    ok=before==after and all(r['status']=='passed' and r['returncode']==0 for r in results)
    record=dict(schema='threshold-joint-evaluation-preparation/1',status='cpu_preparation_passed' if ok else 'failed',
        adapter_python_sha256=before['adapter'],common_python_sha256=before['common'],
        implementations={k:before[k] for k in ('joint','frozen')},
        real_development_helper_sha256=digest(Path(args.joint_source)/RELATIVE/'real_development.py'),
        real_plan_sha256=digest(args.real_plan),tests=sum(r['tests'] for r in results),
        fixed_case_plan_sha256=digest(Path(args.common_source)/'case_plan.json'),
        errors=sum(r['errors'] for r in results),failures=sum(r['failures'] for r in results),
        skipped=sum(r.get('skipped',0) for r in results),both_implementations_import_verified=ok,
        source_files_unchanged=before==after,real_inference_performed=False,real_checkpoints_opened=False,
        gpu_preflight=False,formal_training_started=False,results=results,recorded_unix=time.time())
    (out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','tests','errors','failures','skipped','both_implementations_import_verified')}))
    if not ok:raise SystemExit(1)


if __name__=='__main__':main()
