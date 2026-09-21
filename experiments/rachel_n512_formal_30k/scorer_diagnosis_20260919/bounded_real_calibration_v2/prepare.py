"""Reuse the prior fixed folds/negatives; register yesterday's fixed C16 heads."""
import argparse
from copy import deepcopy
from pathlib import Path
from common import read,save,sha,GRID

R=Path('/root/autodl-tmp/rachel_score_design_20260913_001')
D=R/'scorer_diagnosis_20260919'
P='experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.'


def flag(command,name,default=None):
    return command[command.index(name)+1] if name in command else default


def build(root):
    root=Path(root);root.mkdir(exist_ok=False)
    prior=R/'real_domain_calibration_v1_20260921'
    registry={}
    for split in ('real','ood'):
        (root/split).mkdir()
        (root/split/'manifest.json').symlink_to(prior/split/'manifest.json')
    plan=read(D/'multigpu_20260920/plan.json')
    for lane in plan['lanes']:
        for stage in lane['stages']:
            c=stage['command']
            if flag(c,'--split')!='real' or flag(c,'--selection','fixed_epoch')!='fixed_epoch' or flag(c,'--head-budget','16')!='16':
                continue
            training=flag(c,'--training-run')
            if not training:continue
            key='old_'+Path(training).name
            epoch=flag(c,'--matcher-epoch')
            if epoch:key='old_m'+epoch+'_all_tokens'
            entry=dict(key=key,cohort='2026-09-20',module=c[2],training_run=training,
                pythonpath=stage['env'].get('PYTHONPATH',stage['cwd']),cwd=stage['cwd'],
                evaluation_root=str(Path(flag(c,'--output')).parent),
                head_budget=int(flag(c,'--head-budget')) if '--head-budget' in c else None,
                matcher_checkpoint=flag(c,'--matcher-checkpoint'),matcher_epoch=int(epoch) if epoch else None,
                batch_size=1,selection='fixed_epoch')
            if key in registry:raise ValueError('duplicate model '+key)
            registry[key]=entry
    # These endpoints had completed before the old seven-GPU plan was built.
    c1=deepcopy(registry['old_c2_c16'])
    c1.update(key='old_c1_c16',training_run=c1['training_run'].replace('/c2_c16','/c1_c16'),
        evaluation_root=c1['evaluation_root'].replace('/c2_c16','/c1_c16'))
    registry[c1['key']]=c1
    zero=deepcopy(registry['old_mass_c16'])
    zero.update(key='old_zero_c16',training_run=zero['training_run'].replace('/mass_c16','/zero_c16'),
        evaluation_root=zero['evaluation_root'].replace('/mass_c16','/zero_c16'))
    registry[zero['key']]=zero
    for name in ('s6_d2_c16','s7_c16'):
        training=D/'continuation_v1'/name
        entry=dict(key='old_'+name,cohort='2026-09-20',module=P+'continuation.evaluate_continuation',
            training_run=str(training),pythonpath=str(D/'followup_source_v1'),cwd=str(D/'followup_source_v1'),
            evaluation_root=str(training/'evaluation/fixed_epoch'),head_budget=None,
            matcher_checkpoint=None,matcher_epoch=None,batch_size=1,selection='fixed_epoch')
        registry[entry['key']]=entry
    for entry in registry.values():
        provenance={}
        for split in ('real','ood'):
            path=Path(entry['evaluation_root'])/split/'protocol.json'
            protocol=read(path)
            if protocol['status']!='complete':raise ValueError('incomplete endpoint '+str(path))
            receipt=protocol['model']
            if receipt['selection']!='fixed_epoch':raise ValueError('not a fixed endpoint')
            provenance[split]=dict(protocol=str(path),protocol_sha256=sha(path),
                pair_results=str(path.parent/'pair_results.jsonl'),model=receipt)
        if provenance['real']['model']['checkpoint_sha256'] != provenance['ood']['model']['checkpoint_sha256']:
            raise ValueError('different weights across datasets')
        entry.update(provenance=provenance,checkpoint_sha256=provenance['real']['model']['checkpoint_sha256'])
    for arm,receipt in read(prior/'model_freezes.json').items():
        entry=dict(key=arm,cohort='2026-09-21',reuse_predictions=str(prior/'predictions'/arm),
            checkpoint_sha256=receipt['checkpoint_sha256'],model_receipt=receipt)
        registry[arm]=entry
    save(root/'registry.json',registry)
    save(root/'protocol.json',dict(status='prepared',prior_root=str(prior),models=list(registry),
        threshold_grid=list(GRID),primary_policy='bounded_max_f1',fixed_reference_threshold=.3,
        secondary_policy='bounded_recall95 with explicit unattainable flag',
        tie_break='maximum F1; closest to .3; precision; higher threshold',
        model_weights_updated=False,raw_scores_rescaled=False,new_negatives_sampled=False,
        prior_real_design_exposure=True,one_cycle_five_source_group_folds=True,
        interpretation='Constrained operating-point sensitivity, not pristine blind testing after repeated real-domain analysis.'))
    (root/'logs').mkdir()
    print({k:v.get('module','reuse') for k,v in registry.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);build(p.parse_args().root)
