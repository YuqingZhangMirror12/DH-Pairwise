"""Focused new-quota tests. No old GPU/pixel tests or inference repeated."""
import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from . import heldout_reduce_plan as p
from . import heldout_reduce_run as r


class ReducedQuotaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).resolve().parents[4]/'artifacts/model_selection_v2_20261002/build_audit/curriculum_pilot_sources_01/plan'
        cls.g=json.loads((folder/'generation_plan.json').read_text())
        cls.s=json.loads((folder/'sources.json').read_text())
        cls.plan=p.construct(cls.g,cls.s,p.ref(folder/'generation_plan.json'))

    def test_all_recipes_and_secondary_dimensions_survive_each_gen(self):
        self.assertTrue(p.validate(self.plan,self.g,self.s))
        for role in p.ROLES:
            for gen in p.GENS:
                m=self.plan['cells'][role][gen]['planned_per_stage']
                self.assertEqual(11,len(m['recipe']));self.assertGreaterEqual(min(m['recipe'].values()),2)
                self.assertEqual({'smaller':42,'larger':18},m['size_class'])

    def test_complementary_rounding_preserves_aggregate_recipe_mix(self):
        total=Counter()
        for gen in p.GENS:total.update(p.targets(gen))
        self.assertEqual(Counter({k:2*v for k,v in p.OLD_RECIPES.items()}),total)

    def test_exactly_same_quota_ids_and_unchanged_objects_across_three_stages(self):
        for role in p.ROLES:
            selected=p.chosen_tasks(self.plan,self.g,role)
            self.assertEqual(4*60*3*3,len(selected))
            for gen in p.GENS:
                ids=[{t['quota_slot'] for t in selected if t['stage']==stage and t['generator']==gen} for stage in p.STAGES]
                self.assertEqual(ids[0],ids[1]);self.assertEqual(ids[0],ids[2])
            self.assertTrue(all(any(t is original for original in self.g['tasks']) for t in selected))

    def test_changed_budget_duplicate_ids_recipe_counts_or_outcome_flag_rejected(self):
        for alteration in ('budget','duplicate','counts','model'):
            modified=copy.deepcopy(self.plan)
            if alteration=='budget':modified['desired_pairs_per_role']=3200
            elif alteration=='duplicate':modified['cells']['select']['Gen2']['quota_slots'][0]=modified['cells']['select']['Gen2']['quota_slots'][1]
            elif alteration=='counts':modified['cells']['select']['Gen3']['planned_per_stage']['recipe']['clean']+=1
            else:modified['model_outputs_used']=True
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):p.validate(modified,self.g,self.s)

    def test_reuse_commit_is_by_reference_without_rewriting_old_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);artifact=base/'sample.npz';artifact.write_bytes(b'fixture-not-real-pixels')
            fields=('sample','proof','unaugmented_sample','baseline_sample','baseline_group')
            row={k:v for prefix in fields for k,v in ((prefix+'_path',str(artifact)),(prefix+'_sha256',p.sha(artifact)))}
            row.update(target_metadata=str(artifact),target_metadata_sha256=p.sha(artifact))
            task={'slot':3};plan={'path':'registered-plan.json','sha256':'p'}
            original={'status':'committed','task':task,'plan_sha256':'p','records':[row,row],'audit_rows':['frozen-audit']}
            prior=base/'old.json';prior.write_text(json.dumps(original));before=prior.read_bytes()
            output=base/'new.json';got=r.reuse_commit(prior,task,plan,output)
            self.assertEqual(before,prior.read_bytes());self.assertFalse(got['old_pixels_regenerated'])
            self.assertEqual(got['records'],original['records']);self.assertEqual(got['reused_from']['commit'],p.ref(prior))
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):r.reuse_commit(prior,task,plan,output)


if __name__=='__main__':unittest.main()
