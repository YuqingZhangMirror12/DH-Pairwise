"""Immutable inputs for CPU-only post-terminal B1--B3 companions."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/matcher_v2_20260930')
SOURCE = ROOT/'runtime_work_13'
PACKAGE = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919'
PYTHON = Path('/root/autodl-tmp/dunhuang_pairwise_v02/envs/rachel-paper-benchmarks-v1/bin/python')
PREPARATION = ROOT/'runtime_work_13_cpu_remote.json'
CANONICAL = ROOT/'strict_admission_checks_02/canonical.json'
STRATA = ROOT/'development_strata_01/plan.json'
ADMISSION = ROOT/'reference_select_pipeline_01/admission'
ADMISSION_SHA = 'f50ca7bd4b27d502f7eaea9fb7058660007856ddee92591e6a02170bc17571df'
SOURCE_SHA = '9558af206eb70de7f9b62689ad4c9baa57587f875cc8fa309c0d3fa5719cde6c'
PREPARATION_SHA = 'f412dcc20e26f7a1316dfdd425e177780a695e3b574bab8835420ff4141b8a4b'
MATCHER_SPEC_SHAS = {
    'B1':'8f6dcc6a134753c7609ad96cf00a88d9d71e657c5ac00752bcc603302958a69a',
    'B2':'5f1f277ff7d2eb3944ff25b8085619a88b9d975862776e2408f516050e85dc33',
    'B3':'4c21cb624e2b9b20e34c03f9535ecd40405437d49e1db384ecc05454eac4d37f',
}
DEV_STRATA_SHA = '5f0d7a7be2f39857a92935a262c078f540890d36794dbea5ee91cebc17ebc357'
MODULES = ('matcher','scorer_patch','scorer_stats')
ARMS = ('B3','B1','B2')
GPU_ASSIGNMENTS = {'B3':(0,5),'B1':(1,2),'B2':(6,7)}  # Identity only: never allocated or queried here.
DEP_DIRS = ('reference_select_source_01','reference_scorer_source_01','diagnostics_source_03','cal_budget_source_01')


def require(value, message):
    if not value:raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def save(path, value, *, replace=False):
    path=Path(path); target=path.with_name(path.name+'.tmp') if replace else path
    with target.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if replace:os.replace(target,path)


def own_code():return {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}


def dependency_code(root):
    root=Path(root); result={}
    for name in DEP_DIRS:
        files=sorted((root/name).glob('*.py'))
        require(files,'missing bound companion '+name)
        result.update({str(p.relative_to(root)):sha(p) for p in files})
    return result


def check_static():
    for path,signature in ((SOURCE/'source_binding.json',SOURCE_SHA),(PREPARATION,PREPARATION_SHA),
                           (ADMISSION/'complete.json',ADMISSION_SHA),(STRATA,DEV_STRATA_SHA)):
        require(sha(path)==signature,'registered static input changed: '+str(path))
    for arm,signature in MATCHER_SPEC_SHAS.items():require(sha(paths(arm,'matcher')[0])==signature,'Matcher execution changed')


def paths(arm,module):
    require(arm in ARMS and module in MODULES,'unregistered B-arm/module')
    stem=arm.lower()+'_'+module
    return ROOT/(stem+'_locked_01/execution.json'),ROOT/(stem+'_pipeline_01')


def upstream_failure(arm,module):
    _,root=paths(arm,module); _,matcher=paths(arm,'matcher')
    candidates=[root/'failure.json',root/'training/controller_failure.json',root/'training/formal/failure.json',
        root/'evaluation/controller_failure.json',matcher/'failure.json']
    candidates+=sorted((root/'training/formal').glob('failure_attempt_*.json'))
    if arm in ('B1','B2'):
        candidates += [ROOT/'b12_control_queue_01'/arm/'failure.json',ROOT/'b12_control_queue_01/failure.json']
    elif module!='matcher':candidates += [ROOT/'b3_head_queue_01/failure.json',ROOT/'b3_head_queue_01'/module/'failure.json']
    return next((dict(file=str(p),sha256=sha(p),detail=read(p)) for p in candidates if p.is_file()),None)


def load_runtime():
    sys.path.insert(0,str(SOURCE))
    result={name:importlib.import_module(PACKAGE+'.matcher_v2_v1.'+name)
            for name in ('pipeline','runtime_inputs','terminal')}
    entry=importlib.import_module(PACKAGE+'.curriculum_scorer_eval_v1.entry')
    package=SOURCE/PACKAGE.replace('.','/')
    entry.bind_evaluation(package/'s7_consensus_eval_v14',package/'binary_eval_v1')
    for value in (*result.values(),entry):require(SOURCE in Path(value.__file__).resolve().parents,'wrong runtime import')
    result['pipeline'].check_preparation(PREPARATION,SOURCE)
    return result


def terminal_ready(arm,module,runtime):
    """Return no readiness until the full native/head pipeline really completed."""
    failure=upstream_failure(arm,module)
    require(failure is None,'upstream failure; no evaluation/retry: '+repr(failure))
    spec_path,root=paths(arm,module)
    if not (root/'complete.json').is_file():return None
    spec=read(spec_path)
    require(spec['arm']==arm and spec['module']==module,'completed task identity differs')
    inputs=runtime['runtime_inputs'].load_inputs(spec)
    budget=24000 if arm=='B2' else 31667
    require(inputs['plan'].record['total_updates']==budget,'locked full budget differs')
    gpu=GPU_ASSIGNMENTS[arm]
    assigned=list(gpu if module=='matcher' else (gpu[0 if module=='scorer_patch' else 1],))
    args=SimpleNamespace(spec=spec_path,preparation=PREPARATION,canonical_straight=CANONICAL,
        case_plan=SOURCE/PACKAGE.replace('.','/')/'s7_consensus_eval_v14/case_plan.json',
        python=PYTHON,out=root,gpus=assigned)
    sequence=runtime['pipeline'].commands(args,root,len(assigned))
    actual=read(root/'complete.json'); expected=runtime['pipeline'].verify_all(root,args,inputs,sequence)
    require(all(actual.get(key)==value for key,value in expected.items() if key!='completed_unix'),
            'complete training, selected exports, successful returns and12 evaluations required')
    _,origin=runtime['terminal'].verified_export(root/'training',spec_path,inputs['plan'],'sim_best')
    require(origin['arm']==arm and origin['module']==module and origin['selection_kind']=='sim_best'
            and origin['total_completed_updates']==budget and origin['real_used_for_selection'] is False,
            'own completed SIM-selected model required')
    return dict(arm=arm,module=module,execution=bound(spec_path),pipeline_complete=bound(root/'complete.json'),
        origin=origin,original_training_not_modified=True,ordinary_terminal_evaluations_verified=True)


def check_origin(origin,ready):
    fields=('arm','module','selection_kind','checkpoint_sha256','model_state_sha256',
            'matcher_state_sha256','selected_updates','total_completed_updates','common_plan_sha256')
    require(all(origin.get(key)==ready['origin'][key] for key in fields),'supplemental model differs from completed model')
    require(sha(origin['checkpoint'])==origin['checkpoint_sha256'],'selected checkpoint file changed')


def cpu_environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
