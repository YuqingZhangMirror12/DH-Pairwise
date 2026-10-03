import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from . import heldout_curriculum_release as h


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return h.sha(path)


class CurriculumReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def controller(self):
        runtime = self.root/'runtime'; runtime.mkdir()
        native = runtime/'native.py'; native.write_text('x = 1\n')
        runner = self.root/'heldout_run.py'; runner.write_text('# frozen runner\n')
        inv = self.root/'frozen_source_inventory.json'
        save(inv, dict(root=str(runtime), files={'native.py':h.sha(native)}))
        bindings = {str(inv):h.sha(inv),str(runner):h.sha(runner)}
        launch = dict(bindings=bindings, frozen_runtime=str(runtime),started_unix=1,
                      cuda_visible_devices='',models_loaded=False,original_outputs_changed=False,
                      workers=2,threads_per_worker=1)
        actual = dict(command=['python',str(runner),'--root',str(self.root)],returncode=0,
                      bindings=bindings,started_unix=2,finished_unix=3)
        final = dict(status='curriculum_complete_pending_combined_release_audit',
                     bindings=bindings,finished_unix=4,no_model_selection=True,no_training=True,no_gpu=True,
                     desired_curriculum_pairs_per_role=2880)
        for name, value in [('controller_launch',launch),('full_actual_return',actual),('controller_complete',final)]:
            save(self.root/(name+'.json'),value)
        return launch,actual,final

    def source(self):
        def fragment(token, family, generator='gen2voronoi_1'):
            row = dict(fragment_token=token,parent_group_id='group-'+token,split_unit_id=family+'.png',
                       model_mask_path='m/'+token+'.png',contour_path='c/'+token+'.npz',generator=generator)
            return dict(role='cal',family=family,row=row)
        fa,fb,fc = fragment('a','family1'),fragment('b','family1'),fragment('c','family2')
        fragments = {f['row']['fragment_token']:f for f in (fa,fb,fc)}
        positive = dict(pair_id='p1',label=True,fragment_a=fa['row'],fragment_b=fb['row'],split='val')
        negative = dict(pair_id='n1',label=False,fragment_a=fa['row'],fragment_b=fc['row'],split='val',
                        correspondence_path=None,label_origin='distinct_canonical_manuscript_families')
        spec = dict(positive=[dict(pair_id='p1')],positive_pool=[positive],
                    negative=[dict(mode='native',pair_id='n1',row=negative)])
        cat = dict(positive_rows={'cal':[positive]})
        task = dict(slot=0,base_pair_ids=['p1','n1'],generator='Gen2')
        record = dict(baseline_ordinal=0,source_pair_id='p1',label=True,
                      source_row=dict(positive,pair_id='baseline-id'))
        return fragments,spec,cat,task,record

    def extension_controller(self):
        launch,actual,final=self.controller()
        runner=self.root/'heldout_extend_run.py'; runner.write_text('# extension runner\n')
        admission=self.root/'extension_admission.json'; save(admission,{'unit_fixture':True})
        plan=self.root/'plan'/'generation_plan.json'; save(plan,{'schema':'mixed-sim-heldout-generation-plan/2'})
        native=self.root/'native_sample.npz'; native.write_bytes(b'original catalog content')
        payload=self.root/'catalog_payload_admission.json'
        save(payload,dict(schema='heldout-catalog-payload-admission/1',status='byte_exact',files={str(native):h.sha(native)}))
        launch['bindings'].update({str(p):h.sha(p) for p in (runner,admission,plan,payload)})
        launch.update(schema='mixed-heldout-extension-controller/1',total_registered_reserves=12,
                      old_pixel_reconstruction_not_repeated=True,old_build_root=str(self.root/'old'),
                      extension_admission_path=str(admission),extension_admission_sha256=h.sha(admission))
        actual['command']=['python',str(runner),'--root',str(self.root)]
        final.update(reserve_count=12,catalog_payload_after=dict(status='unchanged',admission=dict(path=str(payload),sha256=h.sha(payload))))
        for name,value in [('controller_launch',launch),('full_actual_return',actual),('controller_complete',final)]:
            save(self.root/(name+'.json'),value)
        return launch,actual,final

    def imported(self):
        old=self.root/'old'; old.mkdir()
        task=dict(stage='v17_filtered',slot=1,role='cal')
        plan=old/'plan'/'generation_plan.json'; psha=save(plan,dict(tasks=[task]))
        record=dict(id='old-proof')
        audit=dict(id='old-proof',status='passed')
        prior=dict(status='committed',task=task,plan_sha256=psha,records=[record],audit_rows=[audit])
        commit=old/'augmented'/'cal'/task['stage']/'groups'/'00001.json'; csha=save(commit,prior)
        pilot=old/'pilot_receipts'/'cal'/'complete.json'
        pilotsha=save(pilot,dict(plan_sha256=psha,role='cal',pilot_only=True,status='shortfall',records=[record]))
        imported=dict(commit=dict(path=str(commit),sha256=csha),old_plan=dict(path=str(plan),sha256=psha),
                      old_pilot_complete=dict(path=str(pilot),sha256=pilotsha))
        group=dict(prior,plan_sha256='new-plan-sha',imported_from=imported,old_pixels_regenerated=False)
        ext=dict(old_root=old,admission=dict(roles={'cal':dict(commits={'v17_filtered:1':imported})}))
        return group,task,ext

    def bank(self):
        import numpy as np
        fragments,*_ = self.source()
        arcs = [dict(split='cal',family=f['family'],lineage=f['row']['split_unit_id'],fragment_token=token)
                for token,f in fragments.items()]
        root = self.root/'bank'; root.mkdir()
        bank = dict(split='cal',train_profiles_used=False,arcs=arcs)
        save(root/'bank.json',bank)
        np.savez_compressed(root/'profiles.npz',profiles=np.zeros((len(arcs),8),dtype=np.float32))
        ref = dict(path=str(root),metadata_sha256=h.sha(root/'bank.json'),
                   profiles_sha256=h.sha(root/'profiles.npz'),profile_count=len(arcs))
        bank['_profiles']=np.zeros((len(arcs),8),dtype=np.float32)
        return bank,fragments,dict(donor_bank=ref)

    def archive_fixture(self):
        import numpy as np
        fragments,spec,cat,task,record=self.source()
        task.update(role='cal',stage='v17_filtered',recipe='clean',size_class='smaller',k=1,
                    mode='one',trim_target=.3,quota_slot=0,reserve_index=0)
        spec.update(slot_generators=['Gen2'],schedule=dict(recipes=['clean'],mirrors=['none']))
        cat['release_root']=str(self.root/'catalog')
        base=dict(pair_id='baseline-id',label=True)
        for stem in ('mask_','coarse_mask_','points_rc_','contour_valid_'):
            for side in 'ab':base[stem+side]=np.array([[1.,0.]],dtype=np.float32)
        base.update(target_a=np.array([0,-1]),target_b=np.array([0,-1]),
                    translation_a_to_b_rc=np.array([0.,1.]),translation_valid=True)
        sample=SimpleNamespace(**dict(base,pair_id='final-id'))
        clean=SimpleNamespace(**dict(base,pair_id='clean-id'))
        baseline_sample=SimpleNamespace(**base)
        profile=np.zeros(8,dtype=np.float32)
        bank=dict(arcs=[dict(split='cal',family='family1',lineage='family1.png',fragment_token='a')],_profiles=profile[None])
        detail=dict(trim=dict(donor_index=0,donor=bank['arcs'][0],size_class='smaller',mode='one'),
                    requested_gap_count=0,primary_damage={})
        detail['trim'].update(profile=profile.tolist(),profile_sha256=hashlib.sha256(profile.tobytes()).hexdigest())
        record.update(pair_id='final-id',id='final-id',data_role='cal',stage='v17_filtered',version='v17_filtered',
                      generator='Gen2',recipe='clean',corrosion_recipe='clean',v14_fallback=False,
                      baseline_slot=0,source_root=cat['release_root'],negative_kind=None,
                      offline_paired_mirror='none',partial_applied=False,detail=detail,requested_gap_count=1,
                      augmentation_donor_sources=['family1.png'],model_tensors_sha256=h.tensor_identity(sample),
                      supervised_tensors_sha256=h.tensor_identity(sample,include_supervision=True),
                      unaugmented_model_tensors_sha256=h.tensor_identity(clean))
        stage=self.root/'augmented'/'cal'/'v17_filtered'
        paths=[('sample_path','sample_sha256',stage/'samples'/'00000_0.npz'),
               ('proof_path','proof_sha256',stage/'proof'/'00000_0.npz'),
               ('target_metadata','target_metadata_sha256',stage/'targets'/'00000_0.npz'),
               ('unaugmented_sample_path','unaugmented_sample_sha256',self.root/'augmented'/'cal'/'unaugmented'/'00000_0.npz'),
               ('baseline_sample_path','baseline_sample_sha256',self.root/'baseline'/'cal'/'samples'/'base.npz'),
               ('latent_seam_artifact','latent_sha256',stage/'latent'/'00000_0.npz')]
        for name,sha_name,path in paths:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(name.encode())
            record[name]=str(path); record[sha_name]=h.sha(path)
        entry={k:record[k] for k in ('source_pair_id','source_row','source_root','negative_kind','offline_paired_mirror')}
        entry.update(pair_id='baseline-id',corrosion_recipe='clean',artifact_path='samples/base.npz')
        bpath=self.root/'baseline'/'cal'/'groups'/'00000.json'
        record['baseline_group_path']=str(bpath)
        record['baseline_group_sha256']=save(bpath,dict(entries=[entry,dict(pair_id='negative')]))
        report=dict(data_role='cal',recipe='clean',paired_review=detail,source_pair_id='p1',base_v14_pair_id='baseline-id',
                    v14_fallback=False,compound=dict(partial=None))
        table={record['sample_path']:(sample,report),record['unaugmented_sample_path']:(clean,{}),
               record['baseline_sample_path']:(baseline_sample,dict(compound=dict(partial=None)))}
        group=dict(audit_rows=[dict(id='final-id',status='passed',sample_sha256=record['sample_sha256'],
              proof_sha256=record['proof_sha256'],model_input_sha256=h._audit_hash(sample)),dict(id='negative')])
        args=[record,task,group,spec,cat,fragments,{},bank,
              dict(parent_ids=['family1','family2'],fragment_ids=['a','b','c'],base_pair_ids=[]),
              self.root,'cal',h.Evidence(),lambda path:table[str(path)]]
        return args,table

    def test_actual_complete_controller_is_admitted(self):
        self.controller()
        launch,inventory = h._controller(self.root,h.Evidence())
        self.assertEqual(inventory['root'],launch['frozen_runtime'])

    def test_false_is_not_integer_actual_return_zero(self):
        _,actual,_ = self.controller(); actual['returncode']=False
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaisesRegex(ValueError,'integer zero'): h._controller(self.root,h.Evidence())

    def test_success_report_cannot_override_nonzero_return(self):
        _,actual,_ = self.controller(); actual['returncode']=2
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaises(ValueError): h._controller(self.root,h.Evidence())

    def test_pilot_command_cannot_be_full_return(self):
        _,actual,_ = self.controller(); actual['command'].append('--pilot')
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaisesRegex(ValueError,'full heldout_run'): h._controller(self.root,h.Evidence())

    def test_complete_without_bound_return_rejected(self):
        _,actual,_ = self.controller(); actual['bindings']={}
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaisesRegex(ValueError,'bindings disagree'): h._controller(self.root,h.Evidence())

    def test_native_code_change_rejected(self):
        self.controller(); (self.root/'runtime'/'native.py').write_text('x=2\n')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'): h._controller(self.root,h.Evidence())

    def test_chronology_rejected(self):
        _,actual,_ = self.controller(); actual['finished_unix']=10
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaisesRegex(ValueError,'chronology'): h._controller(self.root,h.Evidence())

    def test_incomplete_controller_never_normalizes(self):
        self.controller(); (self.root/'controller_complete.json').unlink()
        with self.assertRaises(ValueError): h.normalize_curriculum_release(self.root,'cal')

    def test_no_test_role(self):
        with self.assertRaisesRegex(ValueError,'CAL or SELECT'): h.normalize_curriculum_release(self.root,'test')

    def test_evidence_detects_mutation_after_first_hash(self):
        p=self.root/'sample.npz'; p.write_bytes(b'one')
        evidence=h.Evidence(); evidence.file(p)
        p.write_bytes(b'two')
        with self.assertRaisesRegex(ValueError,'changed during admission'): evidence.finish()

    def test_conflicting_receipt_rejected(self):
        p=self.root/'evidence.json'; p.write_text('{}')
        evidence=h.Evidence(); evidence.file(p)
        with self.assertRaisesRegex(ValueError,'SHA mismatch'): evidence.file(p,'0'*64)

    def test_relative_mapping_cannot_escape_catalog(self):
        with self.assertRaisesRegex(ValueError,'unsafe relative'): h.Evidence().mapping({'../out':'0'*64},self.root)

    def test_primary_is_exact_token_join_not_namespace_guess(self):
        fragments,spec,cat,task,record=self.source()
        row,found=h._source_row(record,task,spec,cat,fragments,'cal')
        self.assertEqual(row['pair_id'],'p1')
        self.assertEqual({f['family'] for f in found},{'family1'})

    def test_record_cannot_relabel_positive(self):
        fragments,spec,cat,task,record=self.source(); record['label']=False
        with self.assertRaisesRegex(ValueError,'label differs'): h._source_row(record,task,spec,cat,fragments,'cal')

    def test_record_cannot_change_native_fragment_bytes_identity(self):
        fragments,spec,cat,task,record=self.source()
        record=copy.deepcopy(record); record['source_row']['fragment_a']['model_mask_path']='other.png'
        with self.assertRaisesRegex(ValueError,'source row altered'): h._source_row(record,task,spec,cat,fragments,'cal')

    def test_source_cannot_relabel_generator(self):
        fragments,spec,cat,task,record=self.source(); task['generator']='Gen5'
        with self.assertRaisesRegex(ValueError,'generator relabelled'): h._source_row(record,task,spec,cat,fragments,'cal')

    def test_negative_must_have_distinct_parent_families(self):
        fragments,spec,cat,task,_=self.source()
        row=spec['negative'][0]['row']; row['fragment_b']=fragments['b']['row']
        record=dict(baseline_ordinal=1,source_pair_id='n1',label=False,source_row=dict(row,pair_id='baseline'))
        with self.assertRaisesRegex(ValueError,'shares a canonical parent'): h._source_row(record,task,spec,cat,fragments,'cal')

    def test_bank_upper_bound_includes_all_samefold_ancestors(self):
        bank,fragments,spec=self.bank()
        edges={'group-c':dict(edges=[dict(parent_id='edge-parent',fragment_ids=['raw-hash','variant-hash'])])}
        _,bounds=h._donor_bounds('cal',spec,fragments,edges,h.Evidence())
        self.assertEqual(set(bounds['parent_ids']),{'family1','family2','edge-parent'})
        self.assertEqual(set(bounds['fragment_ids']),{'a','b','c','raw-hash','variant-hash'})
        self.assertEqual(bounds['base_pair_ids'],[])

    def test_crossfold_bank_arc_rejected(self):
        bank,fragments,spec=self.bank(); fragments['c']['role']='select'
        with self.assertRaisesRegex(ValueError,'same-fold'): h._donor_bounds('cal',spec,fragments,{},h.Evidence())

    def test_accepted_trim_and_partial_have_separate_exact_indices(self):
        bank,fragments,_=self.bank()
        partial=dict(applied=True,donor_index=2,donor_lineage='family2.png',donor_family='family2',donor_split='cal')
        record=dict(partial_applied=True,detail=dict(trim=dict(donor_index=0,donor=bank['arcs'][0])),
                    augmentation_donor_sources=['family1.png','family2.png'])
        profile=bank['_profiles'][0]
        record['detail']['trim'].update(profile=profile.tolist(),profile_sha256=hashlib.sha256(profile.tobytes()).hexdigest())
        report=dict(compound=dict(partial=partial))
        accepted=h._accepted_donors(record,report,report,bank,fragments,{},'cal')
        self.assertEqual([d['bank_index'] for d in accepted],[0,2])
        self.assertEqual([d['kind'] for d in accepted],['trim','partial'])

    def test_missing_accepted_donor_index_is_not_guessed(self):
        bank,fragments,_=self.bank()
        record=dict(partial_applied=False,detail=dict(trim=dict(donor=bank['arcs'][0])),
                    augmentation_donor_sources=['family1.png'])
        with self.assertRaisesRegex(ValueError,'donor index'):
            h._accepted_donors(record,dict(compound=dict(partial=None)),{},bank,fragments,{},'cal')

    def test_accepted_arc_cannot_use_samefamily_different_fragment(self):
        bank,fragments,_=self.bank()
        record=dict(partial_applied=False,detail=dict(trim=dict(donor_index=0,donor=bank['arcs'][1])),
                    augmentation_donor_sources=['family1.png'])
        with self.assertRaisesRegex(ValueError,'trim actual donor differs'):
            h._accepted_donors(record,dict(compound=dict(partial=None)),{},bank,fragments,{},'cal')

    def test_archive_task_audit_and_lineage_join_success(self):
        args,_=self.archive_fixture(); row=h._record(*args)
        self.assertEqual(row['parent_ids'],['family1'])
        self.assertEqual(row['base_pair_ids'],['p1'])
        self.assertEqual(row['fragment_ids'],['a','b'])
        self.assertEqual(row['accepted_augmentation_donors'][0]['bank_index'],0)
        self.assertIn('NOT all actually used',row['donor_identity_semantics'])

    def test_actual_model_tensor_change_cannot_hide_in_record(self):
        args,table=self.archive_fixture()
        table[args[0]['sample_path']][0].mask_a[0,0]=8.
        with self.assertRaisesRegex(ValueError,'tensor identity'):h._record(*args)

    def test_final_audit_must_bind_actual_proof(self):
        args,_=self.archive_fixture(); args[2]['audit_rows'][0]['proof_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'pixel audit'):h._record(*args)

    def test_actual_archive_id_not_just_manifest_id(self):
        args,table=self.archive_fixture(); table[args[0]['sample_path']][0].pair_id='different'
        with self.assertRaisesRegex(ValueError,'archive ID/label'):h._record(*args)

    def test_record_cannot_change_registered_recipe(self):
        args,_=self.archive_fixture(); args[0]['recipe']='gaps_weak'
        with self.assertRaisesRegex(ValueError,'metadata differs'):h._record(*args)

    def test_record_cannot_change_registered_size(self):
        args,_=self.archive_fixture(); args[0]['detail']['trim']['size_class']='larger'
        with self.assertRaisesRegex(ValueError,'trim size'):h._record(*args)

    def test_record_cannot_change_registered_mirror(self):
        args,_=self.archive_fixture(); args[0]['offline_paired_mirror']='horizontal'
        with self.assertRaisesRegex(ValueError,'mirror'):h._record(*args)

    def test_unchanged_ids_do_not_hide_sample_file_change(self):
        args,_=self.archive_fixture(); Path(args[0]['sample_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):h._record(*args)

    def test_population_counts_base_reuse_separately(self):
        args,_=self.archive_fixture(); row=h._record(*args)
        values=[row,dict(row,pair_id='another',model_tensors_sha256='x')]
        counts=h._population(values)
        self.assertEqual(counts['rows'],2)
        self.assertEqual(counts['base_pair_count'],1)
        self.assertEqual(counts['max_views_per_base_pair'],2)

    def test_trim_profile_pixels_must_match_index_not_only_lineage(self):
        args,_=self.archive_fixture(); args[0]['detail']['trim']['profile'][0]=1.
        with self.assertRaisesRegex(ValueError,'curve profile differs'):h._record(*args)

    def test_trim_profile_sha_must_match_original_bank_dtype_bytes(self):
        args,_=self.archive_fixture(); args[0]['detail']['trim']['profile_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'curve profile differs'):h._record(*args)

    def test_extension_controller_requires_explicit_bound_configuration(self):
        self.extension_controller()
        launch,_=h._controller(self.root,h.Evidence())
        self.assertEqual(launch['schema'],'mixed-heldout-extension-controller/1')

    def test_extension_runner_not_accepted_under_legacy_schema(self):
        launch,_,_=self.extension_controller(); launch.pop('schema')
        save(self.root/'controller_launch.json',launch)
        with self.assertRaisesRegex(ValueError,'runner unbound'):h._controller(self.root,h.Evidence())

    def test_extension_requires_exact_start_end_catalog_receipt(self):
        _,_,final=self.extension_controller(); final['catalog_payload_after']['status']='unknown'
        save(self.root/'controller_complete.json',final)
        with self.assertRaisesRegex(ValueError,'before/after'):h._controller(self.root,h.Evidence())

    def test_extension_payload_bytes_are_not_a_metadata_only_claim(self):
        self.extension_controller(); (self.root/'native_sample.npz').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):h._controller(self.root,h.Evidence())

    def test_extension_still_rejects_false_actual_return(self):
        _,actual,_=self.extension_controller(); actual['returncode']=False
        save(self.root/'full_actual_return.json',actual)
        with self.assertRaisesRegex(ValueError,'integer zero'):h._controller(self.root,h.Evidence())

    def test_imported_pixels_keep_actual_old_root(self):
        group,task,ext=self.imported()
        actual=h._imported_group(group,task,self.root,'cal',ext,h.Evidence())
        self.assertEqual(actual,ext['old_root'])
        self.assertNotEqual(actual,self.root)

    def test_imported_records_cannot_be_relabelled(self):
        group,task,ext=self.imported(); group=copy.deepcopy(group); group['records'][0]['id']='changed'
        with self.assertRaisesRegex(ValueError,'old success proof changed'):
            h._imported_group(group,task,self.root,'cal',ext,h.Evidence())

    def test_imported_audit_cannot_be_rewritten(self):
        group,task,ext=self.imported(); group=copy.deepcopy(group); group['audit_rows'][0]['status']='new-proof'
        with self.assertRaisesRegex(ValueError,'old success proof changed'):
            h._imported_group(group,task,self.root,'cal',ext,h.Evidence())

    def test_imported_group_requires_exact_admission_index(self):
        group,task,ext=self.imported(); ext['admission']['roles']['cal']['commits']={}
        with self.assertRaisesRegex(ValueError,'admission index'):
            h._imported_group(group,task,self.root,'cal',ext,h.Evidence())

    def test_legacy_group_cannot_claim_foreign_pixels(self):
        group,task,_=self.imported()
        with self.assertRaisesRegex(ValueError,'legacy group'):
            h._imported_group(group,task,self.root,'cal',None,h.Evidence())

    def test_old_baseline_reuse_requires_exact_group_and_audit(self):
        old=self.root/'old'; base=old/'baseline'/'cal'
        path=base/'groups'/'00003.json'; digest=save(path,{'entries':[]})
        ref=dict(path=str(path),sha256=digest)
        adopted=dict(root=str(base),group=ref,audit_rows=[{'pair_id':'p'}])
        ext=dict(old_root=old,admission=dict(roles={'cal':dict(baselines={'3':adopted})}))
        audit=dict(root=str(base),imported_from=ref,group_sha256=digest,audit_rows=adopted['audit_rows'])
        self.assertEqual(h._baseline_root(self.root,'cal','3',audit,ext,h.Evidence()),base)
        audit['audit_rows']=[]
        with self.assertRaisesRegex(ValueError,'exact adoption index'):
            h._baseline_root(self.root,'cal','3',audit,ext,h.Evidence())

    def test_missing_adoption_cannot_point_baseline_to_old_tree(self):
        ext=dict(old_root=self.root/'old',admission=dict(roles={'cal':dict(baselines={})}))
        with self.assertRaisesRegex(ValueError,'not current build'):
            h._baseline_root(self.root,'cal','3',dict(root=str(self.root/'old'/'baseline'/'cal')),ext,h.Evidence())

    def test_private_record_reads_old_pixels_only_after_explicit_root_resolution(self):
        args,_=self.archive_fixture(); old=args[9]; new=self.root/'new'; new.mkdir(); args[9]=new
        with self.assertRaisesRegex(ValueError,'path escaped'):
            h._record(*args)
        row=h._record(*args,sample_root=old,baseline_root=old/'baseline'/'cal')
        self.assertEqual(row['sample_path'],str(old/'augmented'/'cal'/'v17_filtered'/'samples'/'00000_0.npz'))


if __name__ == '__main__': unittest.main()
