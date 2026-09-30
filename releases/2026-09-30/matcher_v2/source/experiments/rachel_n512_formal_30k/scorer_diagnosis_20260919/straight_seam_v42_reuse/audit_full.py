"""Independent replay, pixel and supervision audit for full generated datasets.

The full-generation wrappers are not used to reconstruct accepted geometry or
targets here. Held-out TEST gets integrity checks only, not calibration metrics.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

if __package__:
    from . import generate as g, supervision as s
else:
    import generate as g
    import supervision as s


def quantiles(values):
    return {str(p):float(np.percentile(values,p)) for p in (0,10,25,50,75,90,100)} if values else None


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference-dir','source-admission','real-audit','target-code','metrics-code','official-code','output-new'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    sys.path.insert(0,str(a.official_code))
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    builder=load_module('independent_target_builder',a.target_code)
    metrics=load_module('independent_metrics',a.metrics_code)
    manifest_path=a.root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    plan=json.loads((a.root/'plan.json').read_text());complete=json.loads((a.root/'generation_complete.json').read_text())
    assert complete['status']=='generated' and not manifest['failed']
    assert complete['manifest_sha256']==g.sha(manifest_path)
    assert manifest['plan_sha256']==g.sha(a.root/'plan.json')
    assert len(manifest['entries'])==complete['expected']==plan['expected']
    assert g.sha(a.source_admission)==plan['source_admission_sha256'] and g.sha(a.real_audit)==plan['real_audit_sha256']
    _,pool,excluded=g.source_pool(json.loads(a.source_admission.read_text()),json.loads(a.real_audit.read_text()),manifest['split'])
    _,hashes=g.load_reference(a.reference_dir);assert hashes==plan['reference_sha256']
    g.initialize(a.reference_dir,pool)
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time();records=[];input_hashes={}
    g.save(out/'launch.json',dict(pid=os.getpid(),started_unix=started,source_manifest_sha256=g.sha(manifest_path),
        code_sha256={Path(path).name:g.sha(path) for path in (__file__,s.__file__,a.target_code,a.metrics_code)},
        reference_hashes=hashes,training_admitted=False,split=manifest['split']))
    try:
        for i,e in enumerate(manifest['entries']):
            path=Path(e['sample_path']);assert g.sha(path)==e['sample_sha256']
            sample,report=load_sample(path)
            assert sample.pair_id==e['pair_id'] and bool(sample.label)==bool(e['label'])
            assert bool(report['pose_supervision_enabled'])==(bool(sample.label) and not report['changed_pair'])
            digest=hashlib.sha256()
            for field in ('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b'):
                value=np.ascontiguousarray(getattr(sample,field));digest.update(field.encode());digest.update(value.tobytes())
            input_hash=digest.hexdigest()
            assert input_hash not in input_hashes,'duplicate actual model inputs'
            input_hashes[input_hash]=sample.pair_id
            if e['label']:
                assert g.sha(e['proof_path'])==e['proof_sha256']
                with np.load(e['proof_path']) as f:proof={key:f[key] for key in f.files}
                g.CONTEXT.update(donors={},rejections=Counter(),proof=None)
                aa,bb,trace=s.replay_final_attempt(g.CONTEXT['reference'],e,manifest['seed'])
                np.testing.assert_array_equal(aa,s.unpack(proof,'final_parent_a'))
                np.testing.assert_array_equal(bb,s.unpack(proof,'final_parent_b'))
                ca,cb,_=g.CONTEXT['cut'];_,_,post_a,post_b=g.CONTEXT['misfit']
                for name,value in (('cut_a',ca),('cut_b',cb),('post_misfit_a',post_a),('post_misfit_b',post_b)):
                    np.testing.assert_array_equal(value,s.unpack(proof,name))
                original,reason=g.CONTEXT['reference'].finalize(aa,bb,dict(e['meta']),True)
                assert reason is None
                ia,ib,t,pa,va,pb,vb,ta,tb=original
                for name,expected in (('mask_a',ia[None].astype(np.float32)),('mask_b',ib[None].astype(np.float32)),
                    ('points_rc_a',pa.astype(np.float32)),('points_rc_b',pb.astype(np.float32)),
                    ('contour_valid_a',va),('contour_valid_b',vb),('translation_a_to_b_rc',t)):
                    np.testing.assert_array_equal(getattr(sample,name),expected,err_msg=e['id']+':'+name)
                np.testing.assert_array_equal(proof['original_target_a'],ta)
                np.testing.assert_array_equal(proof['original_target_b'],tb)
                reference_sample=SimpleNamespace(points_rc_a=pa,points_rc_b=pb,contour_valid_a=va,contour_valid_b=vb,target_a=ta,target_b=tb)
                target,details=s.build_supervision(reference_sample,proof,trace,builder.projected_interval_damage)
                np.testing.assert_array_equal(sample.target_a,target['target_a'])
                np.testing.assert_array_equal(sample.target_b,target['target_b'])
                assert details['correspondence_count']>=8
            else:
                assert np.all(sample.target_a==-1) and np.all(sample.target_b==-1) and not sample.translation_valid
                details=dict(correspondence_count=0)
                if e['recipe']=='straight_R':
                    h1,h2=e['meta']['a']['strip_height'],e['meta']['b']['strip_height']
                    assert abs(h2-h1)<=.03*h1
                left,right=e['meta']['a'].get('source_family'),e['meta']['b'].get('source_family')
                assert not left or not right or left!=right
            records.append(dict(id=e['id'],pair_id=e['pair_id'],positive=e['label'],kind=e['recipe'][-1],base=e['meta']['base'],
                model_input_sha256=input_hash,sample_sha256=e['sample_sha256'],target_audit=details,
                metrics=metrics.measure(sample) if manifest['split']!='test' else None,
                test_metrics_withheld=manifest['split']=='test',tries=e['tries'],rejections=e['rejections']))
            if (i+1)%100==0 or i+1==len(manifest['entries']):
                print(json.dumps(dict(done=i+1,total=len(manifest['entries']),seconds=time.time()-started)),flush=True)
        assert g.sha(manifest_path)==complete['manifest_sha256']
        assert len({r['pair_id'] for r in records})==len(records)
        summary={}
        for kind in 'MJR':
            for positive in (True,False):
                selected=[r for r in records if r['kind']==kind and r['positive']==positive]
                expected=plan['counts_per_label'][kind];assert len(selected)==expected
                bases=Counter(r['base'] for r in selected)
                if kind=='J':assert bases==dict(torn_rachel=expected*20//25,margin_fragment=expected*3//25,torn_strip=expected*2//25)
                summary[kind+('_pos' if positive else '_neg')]=dict(rows=len(selected),bases=dict(bases),
                    tries=quantiles([r['tries'] for r in selected]),
                    healthy_correspondences=quantiles([r['target_audit']['correspondence_count'] for r in selected]) if positive else None,
                    rejections=dict(sum((Counter(r['rejections']) for r in selected),Counter())))
        g.save(out/'audit.json',dict(status='passed_integrity_and_supervision',rows=len(records),summary=summary,records=records,
            original_geometry_replayed=True,source_manifest_sha256=complete['manifest_sha256'],training_admitted=False,
            pending=['Population geometry acceptance and frozen E32 SELECT calibration are separate.',
                     'Known real-source exclusion is enforced; complete cross-library manuscript lineage is not proved.'],
            seconds=time.time()-started))
        g.save(out/'complete.json',dict(status='complete',rows=len(records),audit_sha256=g.sha(out/'audit.json'),
            source_manifest_sha256=complete['manifest_sha256'],training_admitted=False,finished_unix=time.time()))
        print(json.dumps(dict(status='complete',rows=len(records),summary=summary)),flush=True)
    except Exception as exc:
        g.save(out/'failure.json',dict(status='failed',rows_completed=len(records),error=repr(exc),training_admitted=False))
        raise


if __name__=='__main__':main()
