"""Prepare a new, explicit repair run without modifying the stopped original."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(old,new):
    old,new=Path(old),Path(new)
    if any((new/name).exists() for name in ('migration_plan.json','formal_launch.json','formal_m12','formal_scratch')):
        raise ValueError('new migration root already registered; no implicit restart')
    # Parent and GPU child identities are checked by the caller before this
    # preparation. No stop or restart signals are sent by this utility.
    if not (new/'source').is_dir():raise ValueError('deploy the isolated source first')
    old_config=old/'formal_scratch/CONFIG.json'
    origin=json.loads(old_config.read_text())
    if origin['arm']!='scratch' or not origin['formal_training']:raise ValueError('wrong original arm')
    def snapshot(folder,name):
        path=old/'diagnostics'/folder/name
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['binding']!=origin or value['stage']!='matcher':raise ValueError('snapshot identity mismatch')
        return dict(path=str(path),sha256=sha(path),epoch=value['epoch'])
    rows=[snapshot('matcher_preserved_for_merge_repair_01','best_layout.pt'),
          snapshot('matcher_preserved_for_merge_repair_01','best_joint.pt'),
          snapshot('matcher_preserved_for_merge_repair_02','last.pt'),
          snapshot('matcher_preserved_for_merge_repair_03','best_joint.pt'),
          snapshot('matcher_preserved_for_merge_repair_03','last.pt')]
    if [r['epoch'] for r in rows]!=[2,4,6,22,26]:raise ValueError('fresh snapshot epochs require explicit review')
    plan=dict(schema='s7-consensus-c04-authorized-migration/1',authorization='repair_now_and_retrain',
        authorized_requirement='union multiple compatible correspondence groups supporting one nearby pose for both final layout and scorer',
        prepared_unix=time.time(),origin_root=str(old),new_root=str(new),
        origin_config=dict(path=str(old_config),sha256=sha(old_config)),
        origin_binding_digest=hashlib.sha256(json.dumps(origin,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        resume=rows[-1],reevaluate=rows,head_policy='fresh_same_original_seed',
        selection_policy='reevaluate_available_weights_CAL_SELECT_only',maximum_epochs=48,
        plateau_migration='keep optimizer/LR/consumed reductions; rebase metric patience at current repaired baseline; three new nonimproving validation observations required',
        retained_matcher_updates=19500,retained_matcher_exposures=624000,
        source_and_data_changed_in_place=False,missing_historical_weights_not_reconstructed=True,
        old_head_compute_retained_as_history_not_reused=True,TEST_or_real_used=False)
    new.mkdir(parents=True,exist_ok=True)
    for relative in ('data_contract.json','geometry_calibration_v2/geometry_calibration.json'):
        dest=new/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('preserve existing destination: '+str(dest))
        shutil.copyfile(old/relative,dest)
        if sha(old/relative)!=sha(dest):raise AssertionError('contract/calibration copy differs')
    (new/'migration_plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    sys.path.insert(0,str(new/'source'))
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.config import TrainingConfig
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.train import make_binding
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.migration import load_migration
    from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.launch import REFERENCE
    from types import SimpleNamespace
    args=SimpleNamespace(arm='scratch',checkpoint=REFERENCE,contract=str(new/'data_contract.json'),
        calibration=str(new/'geometry_calibration_v2/geometry_calibration.json'),
        migration_plan=str(new/'migration_plan.json'),preflight_steps=12)
    binding=make_binding(args,TrainingConfig(),json.loads((new/'data_contract.json').read_text()))
    load_migration(args.migration_plan,binding,TrainingConfig())
    print(json.dumps(dict(status='prepared_and_validated_not_training',new_root=str(new),
        resume=rows[-1],reevaluation_epochs=[r['epoch'] for r in rows],plan_sha256=sha(args.migration_plan))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True)
    a=p.parse_args();torch.set_num_threads(2);prepare(a.old,a.new)
