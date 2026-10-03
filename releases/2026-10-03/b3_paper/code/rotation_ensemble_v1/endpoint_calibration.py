"""Finite endpoint-only new mixed CAL -> sealed thresholds -> retained TEST.

New orchestration only; unchanged four-view geometry/engine/metrics are reused.
No old CAL, no development rerun, no training, no automatic retry.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from .io import read, sha, checked, ref, save, loaded_sources
from .controller import admit, GPU_UUIDS
from .core import combine
from .runtime import load, Population
from .report import annotate_compact
from .metrics import summarize, fit_threshold, fpr_threshold
from .worker import verify_sources
from .inference import PREFIX

BASE = Path('/root/autodl-tmp/rotation_ensemble_20261002')
SETTING_SHA = '7aebd66bff58802a201c8213e78a0b05749c7e5768d8d9f07abde8f203a242a9'
OLD_PROTOCOL_SHA = 'd161d9fc7a22a3bcc1ca87762bd538deed202a8c0e9b24e1eb8178e5482b067e'
MODEL_SHA = '01c7070242c943861c1d44a6c5df4a5b463bb369a26439f9fbb053ea1faf6ac1'
HEAD_SHA = '099b4ae61c364268a8409689620eb5aa02cbd6c13fe936f56fa9d7545971b548'
CAL_COUNTS = (1587, 793, 794)
DEDUP_AUDIT_SHA = '6c30e269cdead51913a38b78053869c51fb98c15712ae80eccd5de506bf5e473'


def require(ok, message):
    if not ok: raise ValueError(message)


def check_release(done, actual, verification, manifest):
    require(done['status'] == 'released_with_authorized_shortfalls_and_exact_deduplication' and done['rows'] == 3183 and
            verification['rows'] == 3183 and verification['raw_generated_rows'] == 3196 and
            actual['returncode'] == 0 and verification['status'] == 'verified_data_release', 'data not formally released')
    require(verification['original_builder_actual_return'] == 2 and
            {tuple(x) for x in verification['authorized_missing_quotas']} ==
            {('cal','Gen3',19,'v17_filtered'),('cal','Gen5',77,'v17_filtered')}, 'unapproved missing CAL data')
    require(not any(verification['cross_role_overlaps'].values()) and verification['train_test_source_overlap'] == 0,
            'data release isolation failed')
    require(manifest['split'] == 'cal' and len(manifest['entries']) == CAL_COUNTS[0] and
            sum(e['label'] for e in manifest['entries']) == CAL_COUNTS[1] and
            all(type(e['label']) is bool for e in manifest['entries']) and
            len({e['pair_id'] for e in manifest['entries']}) == CAL_COUNTS[0] and
            len({e['model_tensors_sha256'] for e in manifest['entries']}) == CAL_COUNTS[0], 'wrong actual new CAL population')


def check_deduplication(value):
    require(value['audit']['sha256'] == DEDUP_AUDIT_SHA and value['original_files_unchanged'] is True and
            value['model_outputs_used'] is False and value['folds']['cal']['actual_pairs'] == CAL_COUNTS[0] and
            value['folds']['select']['actual_pairs'] == 1596 and
            len(value['folds']['cal']['excluded_rows']) == 9 and len(value['folds']['select']['excluded_rows']) == 4,
            'release deduplication not the authorized exact-input policy')


def validate_protocol(p):
    require(p['schema'] == 'b3-endpoint-new-mixed-cal-test/1' and p['angles'] == [0,90,180,270] and
            p['models']['matcher_endpoint']['sha256'] == MODEL_SHA and p['models']['matcher_endpoint']['update'] == 31667 and
            p['models']['patch_sim']['sha256'] == HEAD_SHA and p['models']['patch_sim']['update'] == 29667,
            'endpoint/angle identity changed')
    lock = checked(p['setting_lock'])
    require(p['setting_lock']['sha256'] == SETTING_SHA and lock['setting']['matcher'] == 'matcher_endpoint' and
            lock['setting']['head'] == 'patch_sim' and lock['setting']['method'] == 'B' and
            lock['setting']['angles'] == p['angles'], 'locked main setting changed')
    done = checked(p['data_release']); v = checked(done['verification']); actual = checked(done['actual_return'])
    require(v['outputs']['cal']['manifest'] == p['populations']['sim_cal']['manifest'], 'CAL is not the published manifest')
    check_release(done, actual, v, checked(p['populations']['sim_cal']['manifest']))
    check_deduplication(checked(v['authorized_deduplication']))
    verify_sources(p)


def prepare(root, release):
    require(not root.exists(), 'new run directory required; no retry')
    original = checked(dict(path=str(BASE/'run_01'/'protocol.json'), sha256=OLD_PROTOCOL_SHA))
    setting = dict(path=str(BASE/'endpoint_readout_01'/'setting_lock.json'), sha256=SETTING_SHA)
    lock = checked(setting); checked(lock['development_report'])
    complete_ref = ref(release/'complete.json'); done = checked(complete_ref)
    v = checked(done['verification']); actual = checked(done['actual_return'])
    cal = v['outputs']['cal']['manifest']; manifest = checked(cal)
    check_release(done, actual, v, manifest)
    p = copy.deepcopy(original)
    p.update(schema='b3-endpoint-new-mixed-cal-test/1', created_unix=time.time(), setting_lock=setting,
             data_release=complete_ref, old_protocol=ref(BASE/'run_01'/'protocol.json'),
             source_python={x.name:sha(x) for x in Path(__file__).parent.glob('*.py')},
             main_head='patch_sim', shared_head_diagnostics=False, cal_shards=2,
             old_cal_read=False, development_repeated=False, select_inference=False,
             no_training=True, automatic_retry=False,
             calibration_rule='joint_F1_grid_0.20_0.80_step0.01_tie_nearest0.30; CAL-only FPR1/2/5%',
             baseline='same endpoint and Patch head; reuse zero-degree view; independently CAL-fit threshold')
    # The original input/GT/model refs remain unchanged. Only CAL is replaced.
    p['populations']['sim_cal'] = dict(manifest=cal, role='cal', actual_rows=CAL_COUNTS[0], positive=CAL_COUNTS[1], negative=CAL_COUNTS[2])
    require(p['models']['patch_sim']['update'] == 29667, 'wrong locked Patch head')
    for filename, digest in original['source_python'].items():
        require(sha(Path(__file__).parent/filename) == digest, 'existing engine modified: '+filename)
    validate_protocol(p)
    root.mkdir(parents=True)
    save(root/'protocol.json',p)
    return p


def phase_indices(n, shard):
    require(shard in (0,1), 'only two registered CAL shards')
    return list(range(shard,n,2))


def assert_test_admission(p, seal, protocol_ref):
    require(seal['protocol'] == protocol_ref and seal['setting_lock'] == p['setting_lock'] and
            seal['cal_manifest'] == p['populations']['sim_cal']['manifest'] and
            seal['cal_pairs'] == CAL_COUNTS[0] and seal['cal_positive'] == CAL_COUNTS[1] and seal['test_used'] is False and
            set(seal['thresholds']) == {'four_view','identity_baseline'}, 'TEST requires frozen new CAL thresholds')


def infer(root, job):
    import numpy as np
    import torch
    from .inference import module
    from .core import INPUTS
    p=read(root/'protocol.json'); validate_protocol(p)
    is_cal=job.startswith('cal_'); shard=int(job[-1]) if is_cal else None
    dataset='sim_cal' if is_cal else job
    require(is_cal or dataset in ('dunhuang_cv','turufan'), 'unknown job')
    seal_ref=None
    if not is_cal:
        seal_ref=ref(root/'calibration.json'); seal=checked(seal_ref)
        assert_test_admission(p,seal,ref(root/'protocol.json'))
    out=root/job; out.mkdir(parents=True,exist_ok=False)
    save(out/'launch.json',dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(), argv=sys.argv, job=job, protocol=ref(root/'protocol.json'), calibration=seal_ref,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    # load() installs the immutable runtime import root before Population uses it.
    engine=load(p,'matcher_endpoint',['patch_sim']); before=engine.state_hashes()
    population=Population(p,dataset,'development' if is_cal else 'test')
    indices=phase_indices(len(population.pairs),shard) if is_cal else list(range(len(population.pairs)))
    expected=[population.pairs[i]['pair_id'] for i in indices]
    if not is_cal: require(len(expected)=={'dunhuang_cv':161,'turufan':122}[job], 'wrong retained TEST count')
    path=out/'predictions.jsonl'; seen=[]; started=time.time(); micro=p['physical_microbatch']
    with path.open('x') as f:
        for start in range(0,len(indices),micro):
            selected=indices[start:start+micro]
            if is_cal:
                items=[population.dataset[i] for i in selected]
                for i,(_,_,e) in zip(selected,items):
                    require(sha(e['sample_path'])==e['sample_sha256'], 'published CAL sample bytes changed')
                b=population.collate(items); inputs={k:b[k].numpy() for k in INPUTS}
                ids=[population.pairs[i]['pair_id'] for i in selected]
            else: ids,inputs=population.batch(start,micro)
            rows={a:engine.predict_view(inputs,ids,a) for a in p['angles']}
            for i,pid in enumerate(ids):
                f.write(json.dumps(dict(pair_id=pid,views={a:rows[a][i] for a in p['angles']}),allow_nan=False)+'\n')
            f.flush(); os.fsync(f.fileno()); seen.extend(ids)
            save(out/'status.json',dict(state='running_inference',job=job,processed=len(seen),total=len(expected),
                pair_views=4*len(seen),last_update_unix=time.time()),replace=True)
            if start%80==0: print(job,len(seen),'/',len(expected),flush=True)
    require(seen==expected, 'prediction IDs/order changed')
    # Ground truth is collected only after all phase predictions are durable.
    targets=population.targets(p,only_ids=set(expected)); require(set(targets)==set(expected),'target identity mismatch')
    save(out/'targets.json',targets)
    after=engine.state_hashes(); require(after==before,'inference changed model state')
    validate_protocol(p)
    for k in ('matcher_endpoint','matcher_sim','patch_sim'):
        require(sha(p['models'][k]['path'])==p['models'][k]['sha256'],'export changed during inference')
    if not is_cal: require(ref(root/'calibration.json')==seal_ref,'thresholds changed during TEST')
    save(out/'complete.json',dict(status='complete',job=job,pairs=len(seen),pair_ids=seen,angles=p['angles'],
        predictions=ref(path),targets=ref(out/'targets.json'),protocol=ref(root/'protocol.json'),calibration=seal_ref,
        models_before=before,models_after=after,elapsed_seconds=time.time()-started,
        native_loaded_sources=loaded_sources(p['runtime'],PREFIX)))


def accepted_job(root,job):
    p=read(root/'protocol.json'); out=root/job
    done=read(out/'complete.json'); actual=read(out/'actual_return.json'); launch=checked(actual['launch'])
    require(actual['returncode']==0 and done['status']=='complete' and done['models_before']==done['models_after'] and
            done['protocol']==ref(root/'protocol.json') and launch['job']==job and
            launch['command'][-2:]==['--job',job] and actual['pid']==launch['pid'], 'job lacks successful actual return/model integrity')
    require(sha(done['predictions']['path'])==done['predictions']['sha256'],'prediction file changed')
    raw=[json.loads(line) for line in Path(done['predictions']['path']).open()]; targets=checked(done['targets'])
    require(len(raw)==done['pairs'] and [r['pair_id'] for r in raw]==done['pair_ids'] and set(targets)==set(done['pair_ids']),
            'prediction rows/targets differ')
    for r in raw:
        r['views']={int(a):v for a,v in r['views'].items()}
        require(set(r['views'])=={0,90,180,270} and all(v['pair_id']==r['pair_id'] for v in r['views'].values()), 'view identity mismatch')
    for path,digest in done['native_loaded_sources'].items(): require(sha(path)==digest,'loaded native source changed')
    return raw, targets


def annotated(data, angles):
    raw,targets=data
    return [annotate_compact(combine(r['views'],tuple(angles),'B','patch_sim'),targets[r['pair_id']],r['pair_id']) for r in raw]


def calibrate(root):
    p=read(root/'protocol.json'); validate_protocol(p)
    raw=[]; targets={}
    for job in ('cal_0','cal_1'):
        rs,ts=accepted_job(root,job); require(not set(ts)&set(targets),'CAL shard overlap');raw+=rs;targets.update(ts)
    manifest=checked(p['populations']['sim_cal']['manifest']); ordered=[e['pair_id'] for e in manifest['entries']]
    index={r['pair_id']:r for r in raw}; require(len(raw)==CAL_COUNTS[0] and set(index)==set(ordered),'CAL shard union incomplete')
    data=([index[k] for k in ordered],targets); thresholds={}; summaries={}
    for name,angles in [('four_view',[0,90,180,270]),('identity_baseline',[0])]:
        rows=annotated(data,angles)
        thresholds[name]=dict(primary=fit_threshold(rows,'patch_sim'),
            frozen_cal_fpr={str(rate):fpr_threshold(rows,rate) for rate in (.01,.02,.05)})
        summaries[name]=dict(primary=summarize(rows,thresholds[name]['primary']),
            frozen_cal_fpr={rate:summarize(rows,t) for rate,t in thresholds[name]['frozen_cal_fpr'].items()})
    save(root/'calibration.json',dict(schema='endpoint-new-cal-threshold-seal/1',locked_unix=time.time(),
        protocol=ref(root/'protocol.json'),setting_lock=p['setting_lock'],cal_manifest=p['populations']['sim_cal']['manifest'],
        cal_pairs=len(raw),cal_positive=sum(t['label'] for t in targets.values()),thresholds=thresholds,
        fitting_population_metrics_not_test=summaries,cal_jobs={j:ref(root/j/'complete.json') for j in ('cal_0','cal_1')},
        test_used=False,old_cal_used=False,setting_reselected=False))


def report(root):
    p=read(root/'protocol.json');seal=read(root/'calibration.json');assert_test_admission(p,seal,ref(root/'protocol.json'))
    outputs={}; cases=[]
    for job in ('dunhuang_cv','turufan'):
        data=accepted_job(root,job); l=read(root/job/'launch.json')
        require(l['calibration']==ref(root/'calibration.json') and l['started_unix']>=seal['locked_unix'], 'TEST predates threshold seal')
        require(all(t['fold']==0 for t in data[1].values()),'non-TEST row present')
        outputs[job]={}; all_rows={}
        for name,angles in [('four_view',[0,90,180,270]),('identity_baseline',[0])]:
            rows=annotated(data,angles); all_rows[name]=rows; ts=seal['thresholds'][name]
            outputs[job][name]=dict(primary=summarize(rows,ts['primary']),
                frozen_cal_fpr={rate:summarize(rows,t) for rate,t in ts['frozen_cal_fpr'].items()})
            if job=='turufan':require(outputs[job][name]['primary']['layout_correct'] is None,'Turu has no layout GT')
        cases.extend(dict(dataset=job,four_view=a,identity_baseline=b) for a,b in zip(all_rows['four_view'],all_rows['identity_baseline']))
    save(root/'test_cases.json',cases)
    save(root/'test_report.json',dict(protocol=ref(root/'protocol.json'),calibration=ref(root/'calibration.json'),
        populations=outputs,cases=ref(root/'test_cases.json'),no_test_threshold_fit=True,no_test_setting_selection=True,
        endpoint_substituted_with_existing_9667_trained_head=True,development_repeated=False,
        historical_real_development_exposure=True))


def launch_job(root,job,gpu):
    command=[sys.executable,'-m','rotation_ensemble_v1.endpoint_calibration','--action','infer','--root',str(root),'--job',job]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1')
    f=(root/(job+'.log')).open('x'); process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/(job+'_launch.json'),dict(job=job,pid=process.pid,start_ticks=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21],
        command=command,gpu=gpu,uuid=GPU_UUIDS[gpu],started_unix=time.time()))
    return process,f


def jobs(root, names):
    admit([0,1]); running={name:launch_job(root,name,gpu) for gpu,name in enumerate(names)}
    failures=[]
    for name,(process,f) in running.items():
        code=process.wait();f.close()
        save(root/name/'actual_return.json',dict(pid=process.pid,returncode=code,returned_unix=time.time(),launch=ref(root/(name+'_launch.json'))))
        if code!=0:failures.append(name)
    require(not failures, 'new endpoint inference failed: '+repr(failures))


def controller(root):
    p=read(root/'protocol.json'); validate_protocol(p)
    save(root/'controller_launch.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        started_unix=time.time(),protocol=ref(root/'protocol.json'),command=sys.argv,admission=admit([0,1])))
    jobs(root,('cal_0','cal_1'))
    for action in ('calibrate','report'):
        if action=='report':jobs(root,('dunhuang_cv','turufan'))
        command=[sys.executable,'-m','rotation_ensemble_v1.endpoint_calibration','--action',action,'--root',str(root)]
        result=subprocess.run(command)
        save(root/(action+'_actual_return.json'),dict(command=command,returncode=result.returncode,finished_unix=time.time()))
        require(result.returncode==0,'CPU '+action+' failed')
    validate_protocol(p)
    save(root/'complete.json',dict(status='complete',protocol=ref(root/'protocol.json'),calibration=ref(root/'calibration.json'),
        test=ref(root/'test_report.json'),jobs={j:ref(root/j/'complete.json') for j in ('cal_0','cal_1','dunhuang_cv','turufan')},
        finished_unix=time.time(),model_inference_only=True,original_models_and_data_unchanged=True))


def supervise(root):
    """One finite parent records the controller's real return; no retry or polling."""
    command=[sys.executable,'-m','rotation_ensemble_v1.endpoint_calibration','--action','controller','--root',str(root)]
    started=time.time(); process=subprocess.Popen(command)
    save(root/'supervisor_launch.json',dict(pid=os.getpid(), child_pid=process.pid, command=command,
        started_unix=started, start_ticks=Path('/proc/self/stat').read_text().split()[21]))
    code=process.wait()
    save(root/'actual_return.json',dict(command=command,pid=process.pid,returncode=code,
        started_unix=started,finished_unix=time.time(),supervisor_launch=ref(root/'supervisor_launch.json')))
    require(code==0,'finite endpoint controller failed; no automatic retry')
    save(root/'completed_return_bound.json',dict(complete=ref(root/'complete.json'),actual_return=ref(root/'actual_return.json')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--action',choices=['launch','supervise','controller','infer','calibrate','report'],required=True)
    ap.add_argument('--release',type=Path);ap.add_argument('--job');a=ap.parse_args()
    try:
        if a.action=='launch':
            prepare(a.root,a.release)
            command=[sys.executable,'-m','rotation_ensemble_v1.endpoint_calibration','--action','supervise','--root',str(a.root)]
            with (a.root/'controller.log').open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(a.root/'dispatch.json',dict(pid=process.pid,command=command,dispatched_unix=time.time()))
            print(json.dumps(dict(pid=process.pid,root=str(a.root))),flush=True)
        elif a.action=='infer':infer(a.root,a.job)
        elif a.action=='calibrate':calibrate(a.root)
        elif a.action=='report':report(a.root)
        elif a.action=='supervise':supervise(a.root)
        else:controller(a.root)
    except BaseException as error:
        out=a.root/a.job if a.action=='infer' else a.root
        save(out/('failure.json' if a.action=='infer' else a.action+'_failure.json'),
             dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':main()
