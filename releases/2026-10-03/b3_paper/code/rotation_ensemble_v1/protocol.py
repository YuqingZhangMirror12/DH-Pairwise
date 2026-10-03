"""One-time protocol registration, before new model scores are computed."""
import argparse
import time
from pathlib import Path
from .io import read,sha,ref,checked,save
from .core import REVISION,VIEW_SETS,METHODS

BASE=Path('/root/autodl-tmp/matcher_v2_20260930')
ROOT=Path('/root/autodl-tmp/rotation_ensemble_20261002')
MODEL_PINS={
 'matcher_sim':('b3_matcher_pipeline_01','sim_best','cdf5f03a9317b52926fbd757c85c481551edac333aa49df0e833383923ba0fbd'),
 'matcher_endpoint':('b3_matcher_pipeline_01','equal_budget_endpoint','01c7070242c943861c1d44a6c5df4a5b463bb369a26439f9fbb053ea1faf6ac1'),
 'patch_sim':('b3_scorer_patch_pipeline_02','sim_best','099b4ae61c364268a8409689620eb5aa02cbd6c13fe936f56fa9d7545971b548'),
 'stats_sim':('b3_scorer_stats_pipeline_02','sim_best','feff10a3a97c5607045040864487959afa07cec5b5f00062106234b6d1eb3c0d'),
 'patch_real':('b3_scorer_patch_pipeline_02','real_best','23c412ec23862a319ac645e7c3ffa5696ed2d6d9f343fcdd6dfd05289682c009'),
 'stats_real':('b3_scorer_stats_pipeline_02','real_best','672d2c377514600e9b84432c42e71973d504b1fea127e7054b6b754077d5d42d')}


def prepare(out):
    if out.exists():raise ValueError('already registered; inspect, do not replace')
    spec=read(BASE/'b3_matcher_locked_01/execution.json');base=checked(spec['base_execution'])
    inventory=checked(spec['source_binding']);runtime=Path(spec['source_binding']['path']).parent
    for name,value in inventory.items():
        if sha(runtime/name)!=value:raise ValueError('frozen runtime changed: '+name)
    contract=checked(base['simulation_contract']);roles=checked(base['real_split'])
    models={}
    for key,(run,choice,pin) in MODEL_PINS.items():
        complete_path=BASE/run/'training/formal/exports/training_complete.json';complete=read(complete_path)
        value=complete['exports'][choice]
        if value['sha256']!=pin or sha(value['path'])!=pin:raise ValueError('registered B3 export changed')
        models[key]=dict(value,choice=choice,training_complete=ref(complete_path),
                        trained_matcher='matcher_sim',cross_matcher_diagnostic_allowed=key.startswith(('patch','stats')))
    populations={}
    for name,specification in roles['datasets'].items():
        manifest=Path(specification['remote_manifest']);meta=read(manifest)
        if sha(manifest)!=specification['manifest_sha256']:raise ValueError('real manifest changed')
        excluded=set(specification['excluded_gt_pair_ids']);selected=[p for p in meta['pairs'] if p['pair_id'] not in excluded]
        r=specification['roles'];allowed=set().union(*(set(r[k]['pair_ids']) for k in r))
        if {p['pair_id'] for p in selected}!=allowed:raise ValueError('real role population differs')
        for role,value in r.items():
            if {p['pair_id'] for p in selected if p['fold'] in value['folds']}!=set(value['pair_ids']):raise ValueError('fold/ID split differs')
        populations[name]=dict(manifest=ref(manifest),inputs=ref(Path(meta['prepared'])/'inputs.npz'),
            exclusions=sorted(excluded),roles={k:dict(pair_ids=v['pair_ids'],folds=v['folds']) for k,v in r.items()},
            counts=dict(development=sum(p['fold']!=0 for p in selected),test=sum(p['fold']==0 for p in selected)))
    cal=contract['validation']['cal_mixed'];metadata=checked(cal)
    if metadata['split']!='cal' or len(metadata['entries'])!=1500:raise ValueError('original SIM-CAL differs')
    populations['sim_cal']=dict(manifest=ref(cal['path']),pairs=len(metadata['entries']),
        caveat='Original v17 full30k CAL, not the new mixed CAL under construction; Task1 later re-evaluates separately.')
    traits=Path('/root/autodl-tmp/claudecode0929/analysis_v3/q1_traits.json')
    # Inherit only type labels, never previous model outcome fields.
    seam_types={k:v['type'] for k,v in read(traits).items()}
    if set(seam_types.values())-{'J','R','C'}:raise ValueError('unknown inherited seam type')
    source_dir=Path(__file__).parent
    protocol=dict(schema=REVISION,created_unix=time.time(),runtime=str(runtime),source_binding=spec['source_binding'],
        source_inventory_sha256=sha(spec['source_binding']['path']),models=models,populations=populations,
        real_split=base['real_split'],gt=ref(roles['gt_path']),view_sets=[list(v) for v in VIEW_SETS],methods=list(METHODS),
        angles=[0,90,180,270],dedup_radius_px=16,layout_tolerance_px=20,physical_microbatch=8,
        no_training=True,no_dataset_edits=True,no_test_before_default_locked=True,
        default_selection=dict(population='dunhuang_cv folds2,3,4 only',
            eligible_matchers=['matcher_sim','matcher_endpoint'],eligible_heads=['patch_sim','stats_sim','patch_real','stats_real'],
            key=['empirical_dev_joint_tp_at_fpr2pct','pair_auc','empirical_dev_joint_tp_at_fpr1pct',
                 'empirical_dev_joint_tp_at_fpr5pct','fewer_views','method_A_then_B_then_C','head_name'],
            pairing='Report original matched pipeline separately; endpoint + existing head is an inference-only Matcher swap and must be explicitly identified if selected.',
            other_heads='q-only reported as Matcher diagnostic, not a trained classifier',
            turufan_used=False,test_used=False),
        calibration=dict(sim_heads='original SIM-CAL; threshold refit for every setting',
            real_heads_dunhuang='Dunhuang fold1; select on folds2-4',
            real_heads_turufan='SIM-CAL, preserving existing no-Turufan-selection rule',
            primary_head_grid=[.2,.8,.01],tie_preference=.3,q_only='CAL negative FPR2%',
            fixed_fpr=[.01,.02,.05],negative_ties='never split ties; nextafter excluded boundary',
            development_ROC='same-population negative threshold is diagnostic, NOT test operating threshold'),
        seam_type_source=ref(traits),seam_types=seam_types,
        seam_type_caveat='Inherited other-analysis J/R/C tags, not newly human-adjudicated; missing tags remain unknown. Turufan lacks layout GT.',
        source_python={p.name:sha(p) for p in source_dir.glob('*.py')},
        retained_test_is_not_historically_blind=True,flip_views=False,evidence_fusion=False)
    out.mkdir(parents=True);save(out/'protocol.json',protocol)
    print('registered',out/'protocol.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);prepare(ap.parse_args().out)
