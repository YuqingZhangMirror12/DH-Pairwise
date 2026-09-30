"""Shadow-only audit of accepted v4.2 samples; write new supervision sidecars.

Original NPZs, generated masks, layouts, manifests and running jobs are read-only.
Does not admit data or initiate training. Thin supervision is reported, not rerolled.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from . import generate as g, supervision as s
else:
    import generate as g
    import supervision as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference-dir','source-admission','real-audit','target-code','official-code','output-new'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only explicit CUDA exclusion required')
    sys.path.insert(0,str(a.official_code))
    from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
    spec=importlib.util.spec_from_file_location('frozen_target_builder',a.target_code)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    manifest_path=a.root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    receipt=json.loads((a.root/'generation_complete.json').read_text())
    assert receipt['status']=='generated' and not manifest['failed']
    assert receipt['manifest_sha256']==g.sha(manifest_path)
    plan=json.loads((a.root/'plan.json').read_text())
    assert g.sha(a.source_admission)==plan['source_admission_sha256']
    assert g.sha(a.real_audit)==plan['real_audit_sha256']
    _,rows,excluded=g.source_pool(json.loads(a.source_admission.read_text()),json.loads(a.real_audit.read_text()),manifest['split'])
    _,reference_hashes=g.load_reference(a.reference_dir)
    assert reference_hashes==plan['reference_sha256']
    g.initialize(a.reference_dir,rows)
    out=a.output_new.resolve();out.mkdir(parents=True,exist_ok=False);(out/'targets').mkdir()
    started=time.time();records=[];input_hashes_before={}
    g.save(out/'launch.json',dict(pid=os.getpid(),started_unix=started,manifest_sha256=g.sha(manifest_path),
        reference_hashes=reference_hashes,supervision_code_sha256=g.sha(s.__file__),target_builder_sha256=g.sha(a.target_code),
        audit_code_sha256=g.sha(__file__),source_admission_sha256=g.sha(a.source_admission),real_audit_sha256=g.sha(a.real_audit),
        training_admitted=False,original_data_read_only=True,gpu=False))
    try:
        for index,e in enumerate(manifest['entries']):
            sample_path=Path(e['sample_path']);sample_sha=g.sha(sample_path)
            assert sample_sha==e['sample_sha256'];input_hashes_before[str(sample_path)]=sample_sha
            sample,report=load_sample(sample_path)
            assert bool(report['pose_supervision_enabled'])==(bool(sample.label) and not report['changed_pair'])
            if e['label']:
                proof_path=Path(e['proof_path']);assert g.sha(proof_path)==e['proof_sha256']
                with np.load(proof_path) as f:proof={k:f[k] for k in f.files}
                g.CONTEXT.update(donors={},rejections=Counter(),proof=None)
                aa,bb,trace=s.replay_final_attempt(g.CONTEXT['reference'],e,manifest['seed'])
                np.testing.assert_array_equal(aa,s.unpack(proof,'final_parent_a'))
                np.testing.assert_array_equal(bb,s.unpack(proof,'final_parent_b'))
                ca,cb,_=g.CONTEXT['cut'];pa,pb,post_a,post_b=g.CONTEXT['misfit']
                np.testing.assert_array_equal(ca,s.unpack(proof,'cut_a'));np.testing.assert_array_equal(cb,s.unpack(proof,'cut_b'))
                np.testing.assert_array_equal(post_a,s.unpack(proof,'post_misfit_a'));np.testing.assert_array_equal(post_b,s.unpack(proof,'post_misfit_b'))
                targets,details=s.build_supervision(sample,proof,trace,builder.projected_interval_damage)
                details.update(replayed_cut_and_damaged_pixels_identical=True,recipe=e['recipe'],base=e['meta']['base'],
                               source_pair_id=e['pair_id'],source_sample_sha256=sample_sha,trace=trace)
            else:
                assert np.all(sample.target_a==-1) and np.all(sample.target_b==-1) and not sample.translation_valid
                targets=dict(target_a=sample.target_a,target_b=sample.target_b)
                details=dict(recipe=e['recipe'],base=e['meta']['base'],source_pair_id=e['pair_id'],positive=False,
                             source_sample_sha256=sample_sha,correspondence_count=0,original_matches_touching_damage=0,
                             six_model_inputs_unchanged=True,training_admitted=False)
            target_path=out/'targets'/(e['id']+'.npz')
            with target_path.open('xb') as stream:np.savez_compressed(stream,**targets)
            details.update(target_path=str(target_path),target_sha256=g.sha(target_path),positive=e['label'])
            records.append(details)
            if (index+1)%100==0 or index+1==len(manifest['entries']):
                print(json.dumps(dict(done=index+1,total=len(manifest['entries']),seconds=time.time()-started)),flush=True)
        assert all(g.sha(Path(path))==digest for path,digest in input_hashes_before.items())
        assert g.sha(manifest_path)==receipt['manifest_sha256']
        summary={}
        for kind in 'MJR':
            part=[r for r in records if r['positive'] and r['recipe']=='straight_'+kind]
            count=[r['correspondence_count'] for r in part]
            summary[kind]=dict(positive_pairs=len(part),pairs_with_original_damage_matches=sum(r['original_matches_touching_damage']>0 for r in part),
                original_damage_matches=sum(r['original_matches_touching_damage'] for r in part),
                original_matches=sum(r['original_correspondences'] for r in part),
                retained_matches=sum(count),fewer_than_8=sum(v<8 for v in count),
                retained_quantiles={str(p):float(np.percentile(count,p)) for p in (0,10,25,50,75,90,100)})
        g.save(out/'audit.json',dict(status='complete_shadow_target_audit',rows=len(records),summary=summary,records=records,
            source_manifest_sha256=receipt['manifest_sha256'],original_files_unchanged=True,training_admitted=False,
            caveats=['Only labels and target-only provenance are sidecar outputs; these are not model inputs.',
                     'Samples below eight healthy correspondences are retained and flagged, not regenerated or silently dropped.',
                     'Dataset-level source admission and geometry distribution checks are separate.'],seconds=time.time()-started))
        g.save(out/'complete.json',dict(status='complete',audit_sha256=g.sha(out/'audit.json'),rows=len(records),summary=summary,
            original_files_unchanged=True,training_admitted=False,finished_unix=time.time()))
        print(json.dumps(dict(status='complete_shadow_target_audit',rows=len(records),summary=summary)),flush=True)
    except Exception as exc:
        g.save(out/'failure.json',dict(status='failed',rows_completed=len(records),error=repr(exc),training_admitted=False))
        raise


if __name__=='__main__':main()
