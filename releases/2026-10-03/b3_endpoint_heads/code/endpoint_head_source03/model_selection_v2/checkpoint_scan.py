"""Independent B3 checkpoint scan. No training, retries, or waiting GPU queue.

Prepare is CPU-only. Controller starts only on the explicitly assigned idle
GPUs, consumes actual child returns, and exports only after the complete fixed
inventory succeeds. The old runtime, pixels, labels and exports stay immutable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from . import posthoc_export as ex
from .native_eval import IMPLEMENTATION, bind_materialized_select, evaluate_protocol_select
from .protocol import digest, require, validate_protocol
from .released_protocol import freeze_release_protocol, checked
from .selection import REPORT_SCHEMA, summarize_report

PREFIX = 'experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'
UPDATES = list(range(1667, 31668, 2000))


def save(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ValueError('preserve earlier output: ' + str(path))
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return ex.receipt(path)


def process():
    st = Path('/proc/self/stat').read_text().split()
    return dict(pid=os.getpid(), start_ticks=int(st[21]), command=sys.argv,
                captured_unix=time.time())


def module(name, runtime):
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    mod = importlib.import_module(PREFIX+name)
    require(Path(runtime).resolve() in Path(mod.__file__).resolve().parents, 'wrong frozen native module')
    return mod


def verify_source(request):
    here = Path(__file__).parent
    require({p.name: ex.file_sha(p) for p in here.glob('*.py')} == request['external_python'],
            'independent scan source changed')
    inventory = checked(request['native_inventory'])
    actual = {str(p.relative_to(request['runtime'])): ex.file_sha(p)
              for p in Path(request['runtime']).rglob('*.py')}
    require(actual == inventory, 'frozen native runtime changed')
    return inventory


class FrozenBackend:
    def __init__(self, runtime):
        self.runtime = runtime
        self.io = module('curriculum_training_v1.runtime_io', runtime)
        self.hashing = module('curriculum_training_v1.checkpoint_io', runtime)

    def verify_original(self, controller_root, spec_path, arm):
        require(arm == 'B3', 'this rollout is B3 only')
        binding = ex.read(Path(controller_root)/'formal/training_complete.json')['binding']
        Plan = module('curriculum_training_v1.runtime_plan', self.runtime).RuntimePlan
        plan = Plan(binding['common_plan'], binding['common_plan_sha256'])
        return module('matcher_v2_v1.terminal', self.runtime).verified_export(
            controller_root, spec_path, plan, 'equal_budget_endpoint')

    def checkpoint_at(self, *args): return self.io.checkpoint_at(*args)
    def model_only(self, *args): return self.io.model_only(*args)
    def tree_sha(self, *args): return self.hashing.tree_sha(*args)
    def save_model(self, *args): return self.io.save_model(*args)


def check_candidate(state, committed, candidate, binding, backend):
    require(committed == candidate['committed_checkpoint'] and state['binding'] == binding
            and state['sampling']['completed_updates'] == candidate['update'], 'committed checkpoint differs')
    require(ex.file_sha(candidate['checkpoint_file']['path']) == candidate['checkpoint_file']['sha256'],
            'checkpoint file changed')
    tensors = backend.model_only(state, binding)
    require(backend.tree_sha(tensors) == candidate['model_state_sha256'], 'checkpoint tensors changed')
    return tensors


def load_model(tensors, binding, runtime, backend):
    import torch
    spec = binding['model_spec']
    matcher_state = {k[len('matcher.'):]: v for k, v in tensors.items() if k.startswith('matcher.')}
    matcher = module('matcher_v2_v1.spec', runtime).from_model_spec(
        spec['matcher_implementation'], frozen=True, state=matcher_state)
    cls = module('matcher_v2_v1.adapter', runtime).MatcherV2Adapter
    require(isinstance(matcher, cls) and not any(p.requires_grad for p in matcher.parameters())
            and not any(m.training for n, m in matcher.named_modules() if n),
            'root eval correction requires exact frozen adapter and eval children')
    before = backend.tree_sha(matcher.state_dict()); matcher.eval()
    require(backend.tree_sha(matcher.state_dict()) == before, 'root mode correction changed weights')
    geometry = module('s7_consensus_v1.compatibility', runtime).CompatibilityConfig(**spec['geometry'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(binding['common_plan']['head_seed'])
        head = module('binary_scorer_v1.head', runtime).BinaryClusterHead(spec['scorer_variant'])
    model = module('binary_scorer_v1.model', runtime).BinaryConsensus(matcher, geometry, head=head)
    model.load_state_dict(tensors, strict=True)
    model.requires_grad_(False); model.eval()
    require(backend.tree_sha(model.state_dict()) == backend.tree_sha(tensors), 'exact checkpoint import failed')
    return model


def prepare(root, request_path):
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(not root.exists(), 'new scan root required; no implicit resume')
    request = ex.read(request_path)
    inventory = verify_source(request)
    publication_complete = checked(request['publication_complete'])
    publication_return = checked(publication_complete['actual_return'])
    require(publication_complete['verification'] == request['publication']
            and publication_return['returncode'] == 0, 'successful published data return required')
    require(request['arm'] == 'B3' and request['updates'] == UPDATES,
            'pre-registered 16 trained validation checkpoints required')
    backend = FrozenBackend(request['runtime'])
    original = ex.original_training_evidence(request['training_root'], request['execution_spec'], 'B3', backend)
    binding = original['source_binding']
    require(binding['common_plan']['validation_updates'] == [0]+UPDATES,
            'candidate points differ from original registered validation schedule')
    plan = freeze_release_protocol(request['publication'], UPDATES)
    files = bind_materialized_select(plan, request['select']['path'], request['select']['sha256'])
    config = module('s7_consensus_v1.config', request['runtime']).TrainingConfig(microbatch=1,
                world_size=1, accumulate=1, workers_per_rank=0)
    candidates = []
    for update in UPDATES:
        state, committed = backend.checkpoint_at(Path(request['training_root'])/'formal/checkpoints', update, binding)
        model = backend.model_only(state, binding)
        candidates.append(dict(update=update, committed_checkpoint=committed,
            checkpoint_file=ex.receipt(Path(request['training_root'])/'formal/checkpoints'/('update_%06d'%update)/'rank_00.pt'),
            model_state_sha256=backend.tree_sha(model)))
        # CPU import gate is limited to first/old-selected/endpoint; every
        # checkpoint's committed state and tensor hash is already verified.
        if update in (1667, 9667, 31667):
            loaded = load_model(model, binding, request['runtime'], backend)
            del loaded
        del state, model
    require(inventory == verify_source(request), 'runtime changed during preparation')
    root.mkdir(parents=True, exist_ok=False)
    req_ref = save(root/'request.json', request)
    refs = dict(request=req_ref, protocol=save(root/'protocol.json', plan),
        candidates=save(root/'candidates.json', dict(schema=ex.CANDIDATE_SCHEMA, arm='B3',
            source_controller_root=request['training_root'], source_binding_sha256=digest(binding), candidates=candidates)),
        original=save(root/'original.json', original), config=save(root/'evaluation_config.json', asdict(config)))
    save(root/'prepared.json', dict(status='prepared_no_gpu_inference', refs=refs, process=process(),
         materialized_pairs=len(files), candidate_count=len(candidates), gpu_inference=False,
         no_training=True, no_real_test_inference=True, real_test_metrics_used_for_selection=False,
         task3_overlay_applied=False,
         head_training_started=False, finished_unix=time.time()))
    return refs


def inputs(root):
    prepared = ex.read(root/'prepared.json')
    refs = prepared['refs']
    return prepared, {name: checked(ref) for name, ref in refs.items()}


def worker(root, update):
    import torch
    _, data = inputs(root); request = data['request']; plan = data['protocol']
    inventory = verify_source(request)
    candidate = next(c for c in data['candidates']['candidates'] if c['update'] == update)
    out = root/'evaluations'/('update_%06d'%update)
    require(out.exists() and not (out/'report.json').exists(), 'one-shot pre-launched output required')
    launch = ex.read(out/'launch.json')
    require(launch['update'] == update and launch['protocol_sha256'] == plan['sha256']
            and launch['checkpoint_sha256'] == candidate['checkpoint_file']['sha256'], 'worker launch differs')
    save(out/'process.json', process())
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    backend = FrozenBackend(request['runtime']); binding = data['original']['source_binding']
    state, committed = backend.checkpoint_at(Path(request['training_root'])/'formal/checkpoints', update, binding)
    tensors = check_candidate(state, committed, candidate, binding, backend); del state
    model = load_model(tensors, binding, request['runtime'], backend).to('cuda:0'); del tensors
    before = backend.tree_sha(model.state_dict())
    config = module('s7_consensus_v1.config', request['runtime']).TrainingConfig(**data['config'])
    save(out/'status.json', dict(state='evaluating_new_select', update=update, pairs=1596,
         last_update_unix=time.time(), native_microbatch=1, model_state_sha256=before))
    rows = evaluate_protocol_select(plan, model, request['select']['path'], request['select']['sha256'],
                                   torch.device('cuda:0'), config, request['runtime'], inventory)
    after = backend.tree_sha(model.state_dict())
    require(before == after == candidate['model_state_sha256'], 'model changed during inference')
    verify_source(request)
    report = dict(schema=REPORT_SCHEMA, protocol_sha256=plan['sha256'], update=update, role='select',
         manifest_sha256=plan['manifest_bindings']['select']['sha256'], status='completed', returncode=0,
         checkpoint_sha256=candidate['checkpoint_file']['sha256'], model_state_sha256=before, rows=rows)
    summary = summarize_report(plan, report)
    report_ref = save(out/'report.json', report)
    save(out/'summary.json', summary)
    save(out/'complete.json', dict(report=report_ref, models_before=before, models_after=after,
         no_training=True, processed=len(rows), finished_unix=time.time()))
    save(out/'status.json', dict(state='inference_complete_parent_return_pending', update=update,
        processed=len(rows), last_update_unix=time.time()), replace=True)


def resource_admission(gpus, gpu_rows, compute_rows):
    require(len(gpus) == len(set(gpus)) and all(type(g) is int and 0 <= g < 4 for g in gpus), 'explicit GPU indices required')
    available = {}
    for line in gpu_rows.strip().splitlines():
        idx, uuid, memory = [s.strip() for s in line.split(',')]
        available[int(idx)] = dict(uuid=uuid, memory_mib=int(memory))
    busy = {line.split(',')[0].strip() for line in compute_rows.strip().splitlines() if line.strip()}
    require(all(g in available for g in gpus), 'assigned GPU absent')
    return dict(admitted=all(available[g]['uuid'] not in busy and available[g]['memory_mib'] < 100 for g in gpus),
                devices={str(g): available[g] for g in gpus}, compute_processes=compute_rows,
                captured_unix=time.time())


def evaluation_launch(root, data, candidate, gpu):
    update = candidate['update']; out = root/'evaluations'/('update_%06d'%update)
    require(not out.exists(), 'candidate already attempted; no retry')
    out.mkdir(parents=True)
    command = [sys.executable, '-m', 'model_selection_v2.checkpoint_scan', 'worker', '--root', str(root), '--update', str(update)]
    request = data['request']
    launch = dict(schema='model-selection-evaluation-launch/2', phase='matcher_select', role='select',
        protocol_sha256=data['protocol']['sha256'], candidate_manifest_sha256=ex.file_sha(root/'candidates.json'),
        update=update, checkpoint_sha256=candidate['checkpoint_file']['sha256'],
        model_state_sha256=candidate['model_state_sha256'], report_path=str(out/'report.json'),
        command=command, native_evaluator_revision=IMPLEMENTATION, materialized_select=request['select'],
        source_inventory=request['native_inventory'], evaluation_config=ex.receipt(root/'evaluation_config.json'),
        cuda_visible_devices=str(gpu), started_unix=time.time())
    launch_ref = save(out/'launch.json', launch)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(Path(__file__).parent.parent), PYTHONDONTWRITEBYTECODE='1')
    with (out/'stdout.log').open('xb') as log:
        result = subprocess.run(command, env=env, cwd=Path(__file__).parent.parent, stdout=log, stderr=subprocess.STDOUT)
    report_ref = ex.receipt(out/'report.json') if (out/'report.json').exists() else None
    ret = dict(schema='model-selection-evaluation-return/2', phase='matcher_select', returncode=result.returncode,
        launch_sha256=launch_ref['sha256'], report_path=str(out/'report.json'),
        report_sha256=None if report_ref is None else report_ref['sha256'], finished_unix=time.time())
    return_ref = save(out/'process_return.json', ret)
    require(result.returncode == 0 and report_ref is not None and (out/'complete.json').exists(),
            'candidate failed; retain evidence and do not retry')
    return dict(report=report_ref, launch=launch_ref, process_return=return_ref)


def controller(root, gpus):
    _, data = inputs(root); verify_source(data['request'])
    require(not (root/'controller_launch.json').exists(), 'one-shot controller already launched')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    compute_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    admission = resource_admission(gpus, gpu_rows, compute_rows)
    if not admission['admitted']:
        print(json.dumps(dict(status='resources_busy_no_process_started', admission=admission)), flush=True)
        return 3
    save(root/'controller_launch.json', dict(process=process(), gpus=gpus, admission=admission,
         prepared=ex.receipt(root/'prepared.json'), automatic_retry=False))
    candidates = data['candidates']['candidates']
    stopped = threading.Event()
    def lane(index, gpu):
        result = {}
        for candidate in candidates[index::len(gpus)]:
            if stopped.is_set(): break
            try:
                result[candidate['update']] = evaluation_launch(root, data, candidate, gpu)
            except Exception as exc:
                stopped.set()
                save(root/('failure_gpu_%s.json'%gpu), dict(error=str(exc), traceback=traceback.format_exc(), time_unix=time.time()))
                raise
        return result
    evaluations = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, i, gpu) for i, gpu in enumerate(gpus)]
        for future in futures:
            evaluations.update(future.result())
    require(sorted(evaluations) == UPDATES, 'incomplete scan')
    save(root/'evaluations.json', {str(k): v for k, v in evaluations.items()})
    request = data['request']
    result = ex.export_reselected_matcher(root/'reselected_export', request['training_root'],
        request['execution_spec'], 'B3', ex.receipt(root/'protocol.json'), ex.receipt(root/'candidates.json'),
        evaluations, backend=FrozenBackend(request['runtime']))
    save(root/'complete.json', dict(status='new_select_checkpoint_scan_and_export_complete',
        export_complete=ex.receipt(root/'reselected_export/complete.json'), selected_update=result['selected_update'],
        actual_completed_candidates=len(evaluations), new_select_pairs=1596, no_training=True,
        matched_head_training_completed=False, cal_recomputed_for_new_choice=False,
        finished_unix=time.time()))
    return 0


def driver(root, gpus):
    """Record the controller's OS return, never infer success from a flag."""
    require(not (root/'actual_return.json').exists() and not (root/'controller_launch.json').exists(),
            'controller already attempted; do not create a second runner')
    command=[sys.executable,'-m','model_selection_v2.checkpoint_scan','controller','--root',str(root),'--gpus',gpus]
    token=str(time.time_ns())
    ref=save(root/('driver_launch_'+token+'.json'),dict(process=process(),command=command,started_unix=time.time()))
    with (root/('controller_'+token+'.log')).open('xb') as log:
        child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record=dict(returncode=child.returncode,launch=ref,finished_unix=time.time(),
                complete=ex.receipt(root/'complete.json') if (root/'complete.json').exists() else None)
    if child.returncode==3 and not (root/'controller_launch.json').exists():
        save(root/('resource_deferred_'+token+'.json'),dict(record,status='busy_no_gpu_work_started_no_waiter'))
        return
    require(child.returncode!=0 or record['complete'] is not None,'zero return without completion')
    save(root/'actual_return.json',record)
    require(child.returncode==0,'controller failed; inspect actual_return and failure, no automatic retry')


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'controller', 'worker', 'driver'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--request', type=Path)
    p.add_argument('--update', type=int); p.add_argument('--gpus', default='0,1')
    a=p.parse_args(); root=a.root.resolve()
    try:
        if a.mode == 'prepare': prepare(root, a.request); return
        if a.mode == 'worker': worker(root, a.update); return
        if a.mode == 'driver': driver(root, a.gpus); return
        rc = controller(root, [int(s) for s in a.gpus.split(',')])
        if rc: raise SystemExit(rc)
    except Exception as exc:
        directory = root/'evaluations'/('update_%06d'%a.update) if a.mode == 'worker' else root
        save(directory/('failure.json' if a.mode != 'prepare' else 'prepare_failure.json'),
             dict(error=str(exc), traceback=traceback.format_exc(), time_unix=time.time()))
        raise


if __name__ == '__main__': main()
