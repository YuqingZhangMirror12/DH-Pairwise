"""New local CPU-only extension tests; fixtures are not admitted real datasets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import heldout_extend_plan as e


def native_row(role, gen, i, label):
    token = f'{role}/{gen}/{"p" if label else "n"}/{i}'
    return dict(pair_id=token, label=label, split='val', correspondence_path='gt.json' if label else None,
                fragment_a=dict(fragment_token=token+'/a',generator=gen.lower()+'voronoi_1',model_mask_path='mask/a.png'),
                fragment_b=dict(fragment_token=token+'/b',generator=gen.lower()+'voronoi_1',model_mask_path='mask/b.png'))


def fixture(root):
    old = root/'old'; plan = old/'plan'; plan.mkdir(parents=True)
    release = root/'release'; release.mkdir()
    sources = dict(schema='s7-v14-heldout-source-plan/1', dataset_root=str(release), seed=12345,
                   forbidden_sources=['TRAIN-parent','TEST-parent'], splits={})
    for stem in ('profile','catalog','parent_plan'):
        path=root/(stem+'.json'); path.write_text('{}')
        sources[stem+'_path']=str(path); sources[stem+'_sha256']=e.sha(path)
    tasks=[]
    for role in e.ROLES:
        bankdir=plan/'banks'/role; bankdir.mkdir(parents=True)
        (bankdir/'profiles.npz').write_bytes(b'fixture-native-bank')
        (bankdir/'bank.json').write_text(json.dumps(dict(split=role,arcs=[{'split':role}],source_families=[role+'-parent'])))
        bank=dict(path=str(bankdir),profile_count=1,source_count=1,metadata_sha256=e.sha(bankdir/'bank.json'),
                  profiles_sha256=e.sha(bankdir/'profiles.npz'))
        spec=dict(source_families=[role+'-parent'],release_split='val',pairs=2880,positive=[],negative=[],
                  positive_pool=[],slot_generators=[],generator_counts={},unique_negative_pool=40,donor_bank=bank,
                  schedule={key:[] for key in e.SCHEDULE_FIELDS})
        spec['schedule']['rounding']='frozen rounding text'
        for gen in e.GENS:
            pp=[native_row(role,gen,i,True) for i in range(15)]
            nn=[native_row(role,gen,i,False) for i in range(10)]
            spec['positive_pool'].extend(pp)
            spec['generator_counts'][gen]=dict(positive_base_pairs=15,negative_base_pairs=10,candidate_groups=360,
                final_groups_per_stage=120,recipe_counts={'clean':60,'gaps_weak':60},geometry_exclusion_only=True)
            for reserve in range(3):
                for j in range(120):
                    slot=len(spec['positive']); p=pp[(reserve*120+j)%15]; n=nn[(reserve*120+j)%10]
                    spec['positive'].append(dict(pair_id=p['pair_id'],source_stratum='native_positive'))
                    spec['negative'].append(dict(mode='native',pair_id=n['pair_id'],row=n,source_stratum='native_negative',
                                                anchor_group_id=None,kind='cross_manuscript'))
                    spec['slot_generators'].append(gen)
                    recipe='clean' if j%2 else 'gaps_weak'
                    local=dict(recipes=recipe,partial=False,partial_modes='end',bins='fixture-bin',mirrors=j%3)
                    for key in e.SCHEDULE_FIELDS: spec['schedule'][key].append(local[key])
                    for stage in e.STAGES:
                        tasks.append(dict(role=role,generator=gen,stage=stage,slot=slot,master_seed=12345,
                            reserve_index=reserve,quota_slot=j,recipe=recipe,base_pair_ids=[p['pair_id'],n['pair_id']],
                            size_class='smaller' if j%2 else 'larger',mode='one' if j%2 else 'both',k=1+j%4,
                            trim_target=e.trim_target(12345,role,stage,p['pair_id'])))
        sources['splits'][role]=spec
    (plan/'sources.json').write_bytes(e.encoded(sources))
    generation=dict(schema=e.OLD_SCHEMA,master_seed=12345,source_plan_path=str(plan/'sources.json'),
        source_plan_sha256=e.sha(plan/'sources.json'),roles=list(e.ROLES),stages=list(e.STAGES),tasks=tasks,
        desired_pairs_per_role=3200,desired_pairs_per_curriculum_stage=960,desired_strict_pairs_per_role=320,
        reserve_policy='first pixel-admitted candidate per quota_slot, in registered reserve order',
        quotas_fixed_before_generation=True,model_outputs_used=False,
        label_policy='original targets unchanged',historical_gen23_edge_donors_complete=False,no_train_or_test_generation=True)
    (plan/'generation_plan.json').write_bytes(e.encoded(generation))
    return old,generation,sources


class CandidateOrderTests(unittest.TestCase):
    def test_ten_negative_pool_no_longer_repeats_stride120(self):
        args=dict(seed=9,role='cal',gen='Gen3',quota=17)
        result=e.append_candidates([f'p{i}' for i in range(15)],[f'n{i}' for i in range(10)], [('p0','n0')]*3,**args)
        self.assertEqual(list(range(3,12)),[r['reserve_index'] for r in result])
        self.assertEqual(9,len({r['negative_pair_id'] for r in result}))
        self.assertNotIn('n0',{r['negative_pair_id'] for r in result})
        self.assertEqual(9,len({r['positive_pair_id'] for r in result}))
        self.assertTrue(all(r['combination_is_new_for_quota'] for r in result))
        self.assertEqual(result,e.append_candidates([f'p{i}' for i in reversed(range(15))],
            [f'n{i}' for i in reversed(range(10))],[('p0','n0')]*3,**args))

    def test_each_quota_and_role_has_independent_domain_not_stage_domain(self):
        ids=[str(i) for i in range(20)]
        first=e.independent_order(ids,1,'cal','Gen2',0,'positive',set())
        for role,quota,polarity in [('select',0,'positive'),('cal',1,'positive'),('cal',0,'negative')]:
            self.assertNotEqual(first,e.independent_order(ids,1,role,'Gen2',quota,polarity,set()))

    def test_finite_small_pool_reuse_is_explicit_only_after_combo_exhaustion(self):
        result=e.append_candidates(['p'],['n0','n1'],[('p','n0')]*3,seed=3,role='cal',gen='Gen3',quota=0)
        self.assertEqual('n1',result[0]['negative_pair_id'])
        self.assertTrue(result[0]['combination_is_new_for_quota'])
        self.assertTrue(all(r['cartesian_pool_exhausted_before_draw'] and not r['combination_is_new_for_quota'] for r in result[1:]))
        self.assertTrue(all('views' in r['view_reuse_note'] for r in result))
        self.assertEqual(2,len({r['negative_pair_id'] for r in result[:2]}))

    def test_foreign_source_and_incompatible_cycle_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'outside old pools'):
            e.append_candidates(['p'],['n'],[('TRAIN','n')],seed=1,role='cal',gen='Gen3',quota=0)
        with self.assertRaisesRegex(ValueError,'cannot both be satisfied'):
            e.append_candidates(['p0','p1','p2'],['n0','n1'],[('p0','n0'),('p1','n0'),('p2','n0')],
                                seed=1,role='cal',gen='Gen3',quota=0)


class ExtensionRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name).resolve()
        cls.old,cls.old_g,cls.old_s=fixture(cls.root)
        cls.out=cls.root/'new'/'plan'; cls.new_g=e.extend(cls.old,cls.out)
        cls.new_s=json.loads((cls.out/'sources.json').read_text())

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_exact_old_prefix_and_unchanged_eligible_pool_banks_and_seed(self):
        self.assertEqual(self.old_g['tasks'],self.new_g['tasks'][:len(self.old_g['tasks'])])
        self.assertEqual(2*3*4*120*12,len(self.new_g['tasks']))
        self.assertEqual(12,self.new_g['reserve_count'])
        for role in e.ROLES:
            old,new=self.old_s['splits'][role],self.new_s['splits'][role]
            for key in ('positive','negative','slot_generators'): self.assertEqual(old[key],new[key][:len(old[key])])
            for key in ('positive_pool','donor_bank','source_families','unique_negative_pool'):
                self.assertEqual(old[key],new[key])
            for key in e.SCHEDULE_FIELDS: self.assertEqual(old['schedule'][key],new['schedule'][key][:1440])
            self.assertEqual(11520,new['pairs'])
        self.assertEqual(self.old_s['seed'],self.new_s['seed'])
        self.assertFalse(self.new_g['candidate_extension']['geometry_outcomes_read'])

    def test_new_slots_append_and_stages_share_baseline_with_original_trim_formula(self):
        slots={}
        for task in self.new_g['tasks'][len(self.old_g['tasks']):]:
            self.assertGreaterEqual(task['slot'],1440)
            key=(task['role'],task['generator'],task['quota_slot'],task['reserve_index'])
            slots.setdefault(key,[]).append(task)
            self.assertEqual(e.trim_target(12345,task['role'],task['stage'],task['base_pair_ids'][0]),task['trim_target'])
        self.assertEqual(2*4*120*9,len(slots))
        for group in slots.values():
            self.assertEqual(3,len(group)); self.assertEqual(1,len({t['slot'] for t in group}))
            self.assertEqual(1,len({tuple(t['base_pair_ids']) for t in group}))

    def test_actual_roundtrip_validator_returns_exact_documents(self):
        actual_g,actual_s=e.validate_extension(self.out/'generation_plan.json')
        self.assertEqual(self.new_g,actual_g); self.assertEqual(self.new_s,actual_s)

    def test_mutations_of_old_task_pool_bank_or_registration_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve(); old,g,s=fixture(base)
            out=base/'new'/'plan'; new=e.extend(old,out)
            before=(out/'generation_plan.json').read_bytes(); source=(out/'sources.json').read_bytes()
            for change in ('old_task','new_trim','reserve_count','new_pool','bank'):
                gp=json.loads(before); sp=json.loads(source)
                if change=='old_task': gp['tasks'][0]['size_class']='larger' if gp['tasks'][0]['size_class']=='smaller' else 'smaller'
                elif change=='new_trim': gp['tasks'][-1]['trim_target']=.33
                elif change=='reserve_count': gp['reserve_count']=15
                elif change=='new_pool': sp['splits']['cal']['positive_pool'].append(native_row('cal','Gen2',9000,True))
                elif change=='bank': sp['splits']['cal']['donor_bank']['path']='/new-bank'
                (out/'sources.json').write_bytes(e.encoded(sp)); gp['source_plan_sha256']=e.sha(out/'sources.json')
                (out/'generation_plan.json').write_bytes(e.encoded(gp))
                with self.subTest(change=change),self.assertRaises(ValueError): e.validate_extension(out/'generation_plan.json')

    def test_changed_original_input_or_bank_file_blocks_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve(); old,g,s=fixture(base)
            bank=Path(s['splits']['cal']['donor_bank']['path'])/'profiles.npz'; bank.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'donor profile bytes changed'): e.extend(old,base/'new')

    def test_no_overwrite_or_writes_into_old_build(self):
        with self.assertRaisesRegex(ValueError,'empty output'): e.extend(self.old,self.out)
        with self.assertRaisesRegex(ValueError,'old build root'): e.extend(self.old,self.old/'extra-plan')

    def test_old_incomplete_or_inconsistent_stage_registration_rejected(self):
        bad=copy.deepcopy(self.old_g); bad['tasks'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete old'): e.validate_old(bad,self.old_s)
        bad=copy.deepcopy(self.old_g); bad['tasks'][1]['base_pair_ids'][0]='other'
        with self.assertRaisesRegex(ValueError,'source binding'): e.validate_old(bad,self.old_s)


if __name__=='__main__': unittest.main()
