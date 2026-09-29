"""Private browser-test fixture; never a measured experimental result."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from dataclasses import replace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only fixture required')
    here=Path(__file__).resolve().parent;base=here.parent
    args.out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('binary_ui_bootstrap',base/'binary_eval_v1/entry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.bootstrap(args.source,base/'s7_consensus_eval_v14')
    namespace=types.ModuleType('binary_report_checks');namespace.__path__=[str(here)]
    sys.modules['binary_report_checks']=namespace
    testing=importlib.import_module('binary_report_checks.test_export')
    exporter=importlib.import_module('binary_report_checks.export')
    snapshot=importlib.import_module('consensus_binary_eval_adapter.snapshot')
    trace_module=importlib.import_module('consensus_binary_eval_adapter.trace')
    geometry=importlib.import_module('experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.geometry')
    import torch
    rows=[]
    for name,variant,empty in [('patch','patch',False),('stats','stats',False),('empty','patch',True)]:
        # Two-dimensional synthetic contours make coordinate/edge display
        # errors visible. These explicitly supplied test proposals are not a
        # model-accuracy test or a new contour-data experiment.
        model,pair,proposals,_=testing.fixture(variant)
        n=pair.q.shape[0];theta=torch.arange(n)*2*torch.pi/n
        pa=torch.stack((180+150*theta.sin(),180+120*theta.cos()),dim=1)
        pb=pa+(pair.gb.points[0]-pair.ga.points[0])
        valid=torch.ones(1,n,dtype=torch.bool)
        pair=replace(pair,ga=geometry.compact_contour(pa[None],valid),gb=geometry.compact_contour(pb[None],valid))
        if empty:proposals=replace(proposals,clusters=())
        model.eval().requires_grad_(False)
        with torch.no_grad(),trace_module.MLPTrace(model.head) as trace:
            pred=model.score_pair(pair,proposals=proposals,threshold=.3)
        meta,arrays=snapshot.snapshot_prediction('synthetic-case',pair,pred,threshold=.3,
                    provenance={'variant':'binary_'+variant},trace=trace)
        meta['pair_id']='synthetic-ui-'+name
        meta['provenance'].update(synthetic_fixture=True,trained_checkpoint_opened=False,
                                  checkpoint_sha256='synthetic-untrained-'+name)
        directory=args.out/name;snapshot.write_snapshot(directory,meta,arrays)
        audit=snapshot.audit_snapshot(directory/'evidence.json')
        if audit['status']!='passed':raise ValueError('synthetic snapshot numeric audit failed')
        testing.save(directory/'audit.json',audit)
        binding=dict(pair_id=meta['pair_id'],evidence_sha256=exporter.sha(directory/'evidence.json'),
                     sidecar_sha256=exporter.sha(directory/'arrays.npz'))
        record=exporter.export_case(directory/'evidence.json',directory/'audit.json',binding)
        rows.append(dict(id=name,label={'patch':'Patch/Context 轻头','stats':'Q/几何轻头','empty':'无候选边界情况'}[name],
                         fixture=True,payload=json.dumps(record,ensure_ascii=False,allow_nan=False)))
    data=dict(title='轻量 Scorer 显示组件 · 内部合成测试',status='synthetic fixture',buildStatus='creating',
        filters=[],queries={'binary_ui_fixtures':dict(rows=rows,payloadColumns=['payload'],source=dict(
            kind='synthetic',name='Untrained CPU forward test, not experimental evidence',
            files=['binary_report_v1/make_ui_fixture.py','binary_scorer_v1/test_binary.py'],
            evidenceFlow=[dict(title='Synthetic fixture',detail='Untrained 4-dimensional fixture with two-dimensional synthetic contours and explicitly supplied proposals; actual CPU MLP forward, existing numeric audit and read-only export. Not a builder-accuracy test. No dataset or trained checkpoint opened.')],
            caveats=['Development-only visual tests. Not experiment outcomes. Not formal 96-dimensional features.']))})
    testing.save(args.out/'snapshot.json',data)
    print(json.dumps(dict(status='synthetic_fixture_ready',cases=len(rows),gpu_used=False,real_results=False)))


if __name__=='__main__':main()
