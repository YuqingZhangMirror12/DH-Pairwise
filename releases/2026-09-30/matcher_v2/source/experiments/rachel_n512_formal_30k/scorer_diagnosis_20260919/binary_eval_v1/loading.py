"""Load a terminal binary head; never reinterpret a complex or joint head."""
from pathlib import Path
import torch
from consensus_binary_eval_common import frozen as common
from .contracts import (read,sha,inventory,E32_SHA,PLAN_SHA,DOMAINS,validate_terminal,
                        validate_budget,verify_selection_curve,metric_identity)

def verify_origin(formal,binding,state):
    spec=binding.get('fixed_matcher',{})
    if (spec.get('sha256')!=E32_SHA or spec.get('head_imported') is not False
            or spec.get('matcher_training') is not False):raise ValueError('approved frozen E32 import required')
    path=Path(spec['path']).resolve()
    if (path.name!='best_joint.pt' or path.parent.name!='matcher'
            or path.parent.parent.name!='formal_scratch' or sha(path)!=E32_SHA):
        raise ValueError('E32 origin path/content differs')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    s,c=read(path.parent/'selection.json'),read(path.parent/'complete.json');b=cp.get('binding',{})
    if (cp.get('stage')!='matcher' or cp.get('epoch')!=32 or b.get('arm')!='scratch'
            or b.get('formal_training') is not True or b.get('preflight_steps')!=0
            or s.get('status')!='selected' or c.get('status')!='stage_complete'
            or s.get('selection_on_real') is not False or s.get('best',{}).get('epoch')!=32
            or c.get('best')!=s['best'] or cp.get('metrics',{}).get('key')!=s['best']['key']
            or any(r.get('binding')!=b or r.get('best_joint_sha256')!=E32_SHA for r in (s,c))):
        raise ValueError('origin is not completed SIM-selected E32')
    validate_budget(s)
    for k in ('actual_epochs','updates','exposures','stop_reason'):
        if c.get(k)!=s.get(k):raise ValueError('origin terminal differs')
    for k in ('data_contract_sha256','geometry_calibration_sha256','reference_checkpoint_sha256'):
        if not b.get(k) or b[k]!=binding.get(k):raise ValueError('origin source/geometry/reference differs')
    rel=Path(*common.training.__package__.split('.'))
    if inventory(path.parents[2]/'source'/rel)!=b['implementation_sha256']:raise ValueError('origin source changed')
    imp=read(Path(formal)/'fixed_matcher_import.json')
    if (Path(imp.get('path','')).resolve()!=path or imp.get('sha256')!=E32_SHA or imp.get('epoch')!=32
            or imp.get('old_head_imported') is not False or imp.get('optimizer_imported') is not False
            or imp.get('matcher_frozen') is not True):raise ValueError('binary E32 import receipt differs')
    common.equal_matcher(cp['model'],state)
    return dict(checkpoint=str(path),sha256=E32_SHA,selected_epoch=32,
                historical_matcher_epochs=s['actual_epochs'],matcher_retrained_for_this_head=False)

def load_selected(root,variant,reference,choice,real_plan,helper):
    root=Path(root).resolve();formal=root/('formal_'+variant);stage=formal/'scorer'
    if any((formal/n).exists() for n in ('failure.json','scorer/failure.json')):
        raise ValueError('binary failure requires explicit recovery review')
    terminal=read(formal/'training_complete.json');selection=read(stage/'selection.json');complete=read(stage/'complete.json')
    filename={'sim':'best_joint.pt','real':'best_real.pt'}[choice];path=stage/filename
    if sha(path)!=selection[filename[:-3]+'_sha256']:raise ValueError('selected binary weights changed')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    training=common.training;config=training.TrainingConfig(scorer_variant=variant)
    binding=validate_terminal(cp,selection,complete,terminal,training.canonical_record(config.record()),variant,choice)
    common.verify_code_binding(binding)
    if inventory(Path(training.__file__).parent.parent/'binary_scorer_v1')!=binding['binary_scorer_sha256']:
        raise ValueError('binary implementation changed')
    for file,key in [(root/'data_contract.json','data_contract_sha256'),
                     (root/'geometry_calibration_v2/geometry_calibration.json','geometry_calibration_sha256'),
                     (Path(reference),'reference_checkpoint_sha256')]:
        if sha(file)!=binding[key]:raise ValueError('source artifact differs: '+key)
    contract=read(root/'data_contract.json');calibration=read(root/'geometry_calibration_v2/geometry_calibration.json')
    if (contract.get('status')!='passed' or contract.get('schema')!='s7-consensus-data-contract/3'
            or contract.get('source_disjoint') is not True
            or calibration.get('contract_sha256')!=binding['data_contract_sha256']):
        raise ValueError('v14 source-isolated TRAIN calibration required')
    if sha(real_plan)!=PLAN_SHA or helper.bind_plan(real_plan)!=binding['real_development']:
        raise ValueError('REAL development source binding differs')
    origin=verify_origin(formal,binding,cp['model'])
    rows=[read(stage/f'epoch_{e:03d}_validation.json') for e in range(0,selection['actual_epochs']+1,2)]
    verify_selection_curve(rows,selection)
    validation=rows[cp['epoch']//2]
    recorded=({k:v for k,v in validation.items() if k not in ('epoch','updates','exposures')}
              if choice=='sim' else validation['real_development'])
    if metric_identity(recorded)!=metric_identity(cp['metrics']):raise ValueError('selected metrics differ from validation')
    thresholds=({s:cp['threshold'] for s in ('sim_test_v14',)+DOMAINS} if choice=='sim'
                else dict(cp['thresholds'],sim_test_v14=validation['threshold']))
    model=training.S7Consensus(common.S7MatcherAdapter.from_s7_m12(reference),
        common.CompatibilityConfig.from_calibration(calibration),head=training.fresh_head(config.head_seed,variant))
    model.load_state_dict(cp['model'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite selected model')
    model.eval().requires_grad_(False);model.matcher.set_frozen(True)
    provenance=dict(arm=variant,variant='binary_'+variant,evidence_mode='exact_union_q_then_q_arc_pool',
        checkpoint=str(path),checkpoint_sha256=sha(path),selected_epoch=cp['epoch'],last_epoch=selection['actual_epochs'],
        selection_kind=choice,thresholds=thresholds,selection_on_real=choice=='real',selection_on_test=False,
        selected_epoch_zero=cp['epoch']==0,matcher_origin=origin,matcher_updated_during_training=False,
        stop_reason=selection['stop_reason'],real_plan_sha256=PLAN_SHA,real_development_binding=binding['real_development'],
        data_contract_sha256=binding['data_contract_sha256'],geometry_calibration_sha256=binding['geometry_calibration_sha256'],
        training_implementation_sha256=binding['implementation_sha256'],binary_implementation_sha256=binding['binary_scorer_sha256'],
        selection_sha256=sha(stage/'selection.json'),terminal_receipt_sha256=sha(formal/'training_complete.json'),
        real_used_for_stopping=False,threshold_refitted=False,development_evaluation=True,
        historical_real_development_exposure=True,attention_present=False,local_conflict_head_present=False,
        learned_refinement=False)
    return model,contract,provenance
