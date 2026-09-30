"""Create NEW immutable preparation snapshots, never edit a running source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REL=Path('experiments/rachel_n512_formal_30k/scorer_diagnosis_20260919')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('ambiguous source adapter anchor: '+old[:70])
    return text.replace(old,new)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--real-helper-source',required=True)
    a=p.parse_args();base=Path(a.baseline).resolve();out=Path(a.out).resolve()
    if out.exists():raise ValueError('new preparation root required')
    before={str(x.relative_to(base)):sha(x) for x in base.rglob('*.py')}
    shutil.copytree(base,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(Path(__file__).parent,out/REL/'binary_scorer_v1',ignore=shutil.ignore_patterns('__pycache__'))
    pkg=out/REL/'s7_consensus_v1';train=(pkg/'train.py').read_text()
    for old,new in [
        ('from .consensus_head import ConsensusEvidenceHead','from ..binary_scorer_v1.head import BinaryClusterHead'),
        ('from .losses import batch_loss','from ..binary_scorer_v1.loss import batch_loss'),
        ('from .model import S7Consensus','from ..binary_scorer_v1.model import BinaryConsensus as S7Consensus'),
        ('def fresh_head(seed):','def fresh_head(seed, variant):'),
        ('return ConsensusEvidenceHead()','return BinaryClusterHead(variant)'),
        ('config=TrainingConfig()','config=TrainingConfig(scorer_variant=args.scorer_variant)'),
        ('head=fresh_head(config.head_seed)','head=fresh_head(config.head_seed,config.scorer_variant)'),
        ("parser.add_argument('--arm',choices=('m12','scratch_fixed'),required=True)",
         "parser.add_argument('--arm',choices=('scratch_fixed',),required=True)\n    parser.add_argument('--scorer-variant',choices=('patch','stats'),required=True)"),
        ("if args.arm not in ('m12','scratch_fixed'):","if args.arm != 'scratch_fixed':"),
    ]:train=replace_once(train,old,new)
    train=replace_once(train,"binding['fixed_matcher']=dict(path=str(args.frozen_matcher_state),",
        "if args.frozen_matcher_sha256 != '80cac47d5bc5340df35a7a7c36ab4a3580a9eea8c99adf797ff5744cd2068b17':\n            raise ValueError('experiments1/2 require the confirmed E32 Matcher')\n        binding['binary_scorer_sha256']={p.name:digest(p) for p in (Path(__file__).parent.parent/'binary_scorer_v1').glob('*.py')}\n        binding['fixed_matcher']=dict(path=str(args.frozen_matcher_state),")
    helper=Path(a.real_helper_source)/REL/'s7_consensus_v1'
    for name in ('real_development.py','evaluation_checkpoint.py','test_evaluation_checkpoint.py'):
        shutil.copy2(helper/name,pkg/name)
    edits=[
        ('from .frozen_start import load_selected_matcher','from .frozen_start import load_selected_matcher\nfrom .real_development import bind_plan,RealDevelopment'),
        ("binding['fixed_matcher']=dict(path=str(args.frozen_matcher_state),","binding['real_development']=bind_plan(args.real_split)\n        binding['fixed_matcher']=dict(path=str(args.frozen_matcher_state),"),
        ("raw=TrainModule(model,stage,config,None if caches is None else caches['train'])","development=None if args.preflight_steps else RealDevelopment(args.real_split,binding['real_development'])\n    raw=TrainModule(model,stage,config,None if caches is None else caches['train'])"),
        ('best=None;best_layout=None;curve=[]','best=None;best_layout=None;best_real=None;curve=[]'),
        ("best,best_layout,curve=record['best'],record['best_layout'],record['curve']","best,best_layout,curve=record['best'],record['best_layout'],record['curve']\n        best_real=record.get('best_real')"),
        ('curve=curve,rng=rng,preflight_origin=preflight_origin,migration_origin=migration_origin))',
         'curve=curve,rng=rng,preflight_origin=preflight_origin,migration_origin=migration_origin,best_real=best_real))'),
        ('nonlocal best,best_layout','nonlocal best,best_layout,best_real'),
        ('report,predictions=validate(model,contract,stage,device,config,caches)',
         "report,predictions=validate(model,contract,stage,device,config,caches)\n        real_report,real_predictions=development.evaluate(model,device,config)\n        report['real_development']=real_report\n        predictions['real_development']=real_predictions\n        real_improve=epoch_number>0 and (best_real is None or tuple(real_report['key'])>tuple(best_real['key']))\n        if real_improve:\n            best_real=dict(epoch=epoch_number,key=real_report['key'],thresholds=real_report['thresholds'])"),
        ("save_json(root/'selection.json',dict(status='provisional',best=best,best_layout=best_layout,\n                binding=binding,selection_on_real=False))",
         "if real_improve:\n                save_torch(root/'best_real.pt',dict(binding=binding,stage=stage,epoch=epoch_number,\n                    thresholds=real_report['thresholds'],metrics=real_report,model={k:v.detach().cpu() for k,v in model.state_dict().items()}))\n            save_json(root/'selection.json',dict(status='provisional',best=best,best_layout=best_layout,\n                best_real=best_real,binding=binding,selection_on_real=True,test_used=False))"),
        ("selection=dict(status='selected',best=best,best_layout=best_layout,stop_reason=stop_reason,",
         "if best_real is None:raise AssertionError('no trained REAL-SELECT checkpoint')\n    selection=dict(status='selected',best=best,best_layout=best_layout,stop_reason=stop_reason,"),
        ("selection_on_real=False,matcher_unchanged=stage=='scorer',migration_origin=migration_origin)",
         "selection_on_real=True,test_used=False,best_real=best_real,best_real_sha256=digest(root/'best_real.pt'),\n        matcher_unchanged=stage=='scorer',migration_origin=migration_origin)"),
        ("parser.add_argument('--resume',action='store_true')","parser.add_argument('--real-split',required=True)\n    parser.add_argument('--resume',action='store_true')"),
    ]
    for old,new in edits:train=replace_once(train,old,new)
    (pkg/'train.py').write_text(train)
    config=(pkg/'config.py').read_text()
    config=replace_once(config,'from .losses import LossConfig','from ..binary_scorer_v1.loss import BinaryLossConfig as LossConfig')
    config=replace_once(config,"schema: str = 's7-consensus-training/1'","schema: str = 'binary-cluster-scorer/1'\n    scorer_variant: str = 'patch'")
    config=replace_once(config,"head=dict(layers=2,dim=96,heads=4,length_scale_px=32.)",
        "head=dict(variant=self.scorer_variant,whole_cluster_binary=True,attention=False,local_conflict=False,learned_refinement=False,edge_mlp=[392,64,32] if self.scorer_variant=='patch' else None,cluster_mlp=[80 if self.scorer_variant=='patch' else 16,64,32,1])")
    config=replace_once(config,"extension_schedule='stable hash(pair_id, epoch) modulo4; only native correct candidates/known edges'",
        "extension_schedule='disabled; candidate BCE + within-pair ranking only'")
    config=replace_once(config,'real_epoch_or_geometry_selection=False',
        "real_epoch_or_geometry_selection='epoch only: source-isolated developer REAL-CAL/SELECT; fixed16px; TEST withheld'")
    (pkg/'config.py').write_text(config)
    (pkg/'launch.py').write_text("raise RuntimeError('Inherited controller disabled. Use dedicated binary-scorer launcher after queued experiments.')\n")
    after={str(x.relative_to(base)):sha(x) for x in base.rglob('*.py')}
    if after!=before:raise AssertionError('baseline source was modified')
    inventory={str(x.relative_to(out)):sha(x) for x in out.rglob('*.py')}
    receipt=dict(status='prepared_not_validated',baseline_sha256=before,source_sha256=inventory,
        new_source_only=True,formal_training_started=False,gpu_preflight=False,
        real_selection_adapter_pending=False,dedicated_launcher_pending=True)
    (out.parent/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],source=str(out),files=len(inventory))))

if __name__=='__main__':main()
