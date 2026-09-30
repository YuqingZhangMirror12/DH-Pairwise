"""Fixed simulation/real populations for terminal native Matcher evaluation.

No new sampling, augmentation, threshold fitting or checkpoint selection.
SIM manifests come from the admitted common contract, not a hardcoded v14
path: v17 heldout data must not accidentally be reported as v14 data.
"""
from pathlib import Path

import numpy as np
import torch

from .checkpoint_io import file_sha
from .execution import read_bound
from .exposure import digest
from .model_adapter import BASE, bound_module, require
from .runtime_io import read

SPLITS = ('sim_select', 'sim_test', 'dunhuang_cv', 'turufan')
REAL_PLAN_SHA = '0f8c13bbc4277052ecd6d21e335e97bf58dc8522174e2a11341c3221d50450f6'
CASE_PLAN_SHA = '93dfeec269c0120ed6585580fdc944a10ddc347bc18d54d6125df9f1168c01a9'
INPUTS = ('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b')
COUNTS = dict(sim_select=1500, sim_test=3000, dunhuang_cv=803, turufan=602)


def freeze_plan(spec_path, case_plan, source_root):
    """Run before C/M inference; the same immutable plan is used by both."""
    spec_path=Path(spec_path).resolve(); case_plan=Path(case_plan).resolve(); spec=read(spec_path)
    require(spec['schema']=='curriculum-execution/1' and spec['locked'] is True, 'locked execution required')
    require(file_sha(spec['real_split']['path'])==spec['real_split']['sha256']==REAL_PLAN_SHA,
            'registered real source roles required')
    require(file_sha(case_plan)==CASE_PLAN_SHA, 'previously fixed11 diagnostic cases required')
    role_plan=read_bound(spec['real_split']); cases=read(case_plan)
    require(set(cases['user_confirmed_gt_exclusions'])==set(role_plan['datasets']['dunhuang_cv']['excluded_gt_pair_ids']),
            'fixed-case and real exclusions differ')
    contract=read_bound(spec['simulation_contract'])
    require(contract['status']=='passed' and contract['source_disjoint'] is True, 'passed source-isolated SIM contract required')
    sims={}
    for split,view in (('sim_select',contract['validation']['select_mixed']),('sim_test',contract['test']['mixed'])):
        record=read_bound(view)
        require(view['pair_count']==len(record['entries'])==COUNTS[split] and record['split']==split[4:],
                'fixed simulation population/role differs')
        sims[split]=dict(path=view['path'],sha256=view['sha256'],pair_count=COUNTS[split],
                        pair_ids_sha256=digest([r['pair_id'] for r in record['entries']]))
    real_api=bound_module(BASE+'s7_consensus_v1.real_development',source_root)
    real_binding=real_api.bind_plan(spec['real_split']['path'])
    return dict(schema='curriculum-matcher-population-plan/1',status='locked',
        execution_manifest_sha256=file_sha(spec_path),simulation_contract=spec['simulation_contract'],
        simulation_revision=contract.get('augmentation_revision',contract.get('schema')),
        real_split=spec['real_split'],real_binding=real_binding,simulation=sims,
        case_plan=dict(path=str(case_plan),sha256=CASE_PLAN_SHA),splits=list(SPLITS),
        pair_counts=dict(COUNTS),scorer_used=False,threshold_fitting=False,
        real_used_for_matcher_selection=False,historical_real_development_exposure=True)


def validate_plan(plan, spec_path, source_root):
    require(plan['schema']=='curriculum-matcher-population-plan/1' and plan['status']=='locked', 'locked population plan required')
    # Reopen every bound manifest/cache/GT identity, but never inspect model
    # scores to decide a population or diagnostic example.
    require(plan==freeze_plan(spec_path,plan['case_plan']['path'],source_root), 'population plan/data binding changed')


def population_groups(rows, split, role_plan):
    require(split in SPLITS and rows and len({r['pair_id'] for r in rows})==len(rows), 'unique registered population required')
    if split.startswith('sim_'):
        return {'all':rows}
    spec=role_plan['datasets'][split]; by_id={r['pair_id']:r for r in rows}
    roles=('real_cal','real_select','real_test'); seen=set(spec['excluded_gt_pair_ids'])
    for role in roles:
        ids=spec['roles'][role]['pair_ids']
        require(len(ids)==len(set(ids)) and not seen.intersection(ids), 'overlapping real roles/exclusions')
        seen.update(ids)
    require(set(by_id)==seen, 'real population differs from source roles')
    groups={'all_development_context':rows}
    for role in roles: groups[role]=[by_id[p] for p in spec['roles'][role]['pair_ids']]
    if split=='dunhuang_cv':
        groups['gt_corrected_800_development_context']=[r for r in rows if r['pair_id'] not in spec['excluded_gt_pair_ids']]
    return groups


def tensor_inputs(batch, device):
    """Targets may be collated for storage; only these six enter prediction."""
    return {name:torch.as_tensor(np.array(value,copy=True) if not isinstance(value,torch.Tensor) else value,
            device=device,dtype=torch.bool if name.startswith('contour_valid') else torch.float32)
            for name in INPUTS for value in (batch[name],)}


def load_population(split, plan, source_root):
    require(split in SPLITS, 'unregistered population')
    if split.startswith('sim_'):
        view=plan['simulation'][split]; api=bound_module(BASE+'s7_consensus_v1.data',source_root)
        dataset=api.Dataset(view['path'],view['sha256']); meta=dict(pairs=dataset.entries)
        require(len(dataset)==COUNTS[split] and digest([p['pair_id'] for p in meta['pairs']])==view['pair_ids_sha256'],
                'SIM membership changed')
        def batches():
            for start in range(0,len(dataset),8):
                index=range(start,min(start+8,len(dataset)))
                yield [dataset.entries[i] for i in index],api.collate([dataset[i] for i in index])
        source=dict(manifest=view['path'],manifest_sha256=view['sha256'],
                    preprocessing='unchanged materialized archive',simulation_revision=plan['simulation_revision'])
        return meta,batches(),source,dataset
    # Only the unchanged data-loading function is called, never the v3 model.
    api=bound_module(BASE+'seam_context_v3.evaluate_external',source_root)
    meta,batches,source,dataset=api.load_inputs(split,8)
    expected=plan['real_binding']['sources'][split]
    require(len(meta['pairs'])==COUNTS[split] and source['manifest_sha256']==expected['manifest_sha256']
            and source['inputs_sha256']==expected['inputs_sha256'], 'frozen real population/cache differs')
    return meta,batches,source,dataset


def targets_after_prediction(meta, split, plan, dataset=None):
    """Caller must persist prediction_complete before invoking this function."""
    require(split in SPLITS, 'unregistered target population')
    gt={}
    if split=='dunhuang_cv':
        roles=read_bound(plan['real_split']); path=roles['gt_path']
        require(file_sha(path)==plan['real_binding']['gt_sha256'], 'real GT file changed')
        items=read(path)['positive_pairs'];gt={r['pair_id']:r for r in items}
        require(len(gt)==len(items), 'duplicate real GT')
    result=[]
    for index,item in enumerate(meta['pairs']):
        target=None
        if split.startswith('sim_'):
            sample,_,_=dataset[index]
            require(sample.pair_id==item['pair_id'] and float(sample.label) in (0.,1.), 'SIM target identity differs')
            label=bool(sample.label)
            if label and sample.translation_valid: target=np.asarray(sample.translation_a_to_b_rc).tolist()
            require(not label or target is not None, 'SIM positive layout GT missing')
        else:
            require(type(item['label']) is bool, 'binary real label required')
            label=item['label']
            if split=='dunhuang_cv' and label:
                entry=gt[item['pair_id']]
                require((entry['fragment_a_token'],entry['fragment_b_token'])==(item['fragment_a_id'],item['fragment_b_id']),
                        'Dunhuang GT endpoint order differs')
                target=entry['translation_gt_a_to_b_rc']
        result.append(dict(pair_id=item['pair_id'],label=label,gt_pose=target))
    return result
