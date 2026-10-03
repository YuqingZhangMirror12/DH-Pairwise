"""New scheduling coverage/race/dedup checks; do not regenerate any pixels."""
import copy
import json
from pathlib import Path
import unittest
from . import heldout_parallel_plan as s, heldout_reduce_plan as p
from .heldout_augment import rng_key


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).resolve().parents[4]/'artifacts/model_selection_v2_20261002/build_audit/curriculum_pilot_sources_01/plan'
        cls.g=json.loads((folder/'generation_plan.json').read_text())
        # Tests scheduling only. Projected IDs are supplied metadata, not a
        # repeated solve/audit of the production reduction plan.
        cls.projection={'cells':{r:{gen:{'quota_slots':list(range(60))} for gen in p.GENS} for r in p.ROLES}}
        cls.shards=s.make_shards(cls.projection)

    def test_32_disjoint_shards_own_every_quota_once(self):
        owners=s.validate_shards(self.shards,self.projection,self.g)
        self.assertEqual(32,len(self.shards));self.assertEqual(480,len(owners))
        self.assertTrue(all(len(x['quota_slots'])==15 for x in self.shards))

    def test_all_stages_and_reserves_of_one_baseline_share_owner(self):
        owners=s.validate_shards(self.shards,self.projection,self.g);paths={}
        for role in p.ROLES:
            for t in p.chosen_tasks(self.projection,self.g,role):
                owner=owners[role,t['generator'],t['quota_slot']]
                self.assertEqual(owner,paths.setdefault((role,t['slot']),owner))

    def test_missing_duplicate_or_changed_worker_assignment_rejected(self):
        for kind in ('missing','duplicate','rename'):
            shards=copy.deepcopy(self.shards)
            if kind=='missing':shards.pop()
            elif kind=='duplicate':shards[0]['quota_slots'][0]=shards[1]['quota_slots'][0]
            else:shards[0]['name']='other'
            with self.subTest(kind=kind),self.assertRaises(ValueError):s.validate_shards(shards,self.projection,self.g)

    def test_scheduling_preserves_original_rng_domains(self):
        owners=s.validate_shards(self.shards,self.projection,self.g)
        tasks=p.chosen_tasks(self.projection,self.g,'select')
        before={json.dumps(t,sort_keys=True):rng_key(t,7) for t in tasks}
        after={json.dumps(t,sort_keys=True):rng_key(t,7) for shard in reversed(self.shards)
               for t in tasks if owners['select',t['generator'],t['quota_slot']]==shard['name']}
        self.assertEqual(before,after)

    def merged_fixture(self):
        tasks={};records=[];slot=0
        for stage in p.STAGES:
            for gen in p.GENS:
                q=0
                for recipe,count in p.targets(gen).items():
                    for _ in range(count):
                        task=dict(stage=stage,generator=gen,quota_slot=q,slot=slot,recipe=recipe);tasks[stage,slot]=task
                        for label in (True,False):
                            identifier=f'{stage}:{slot}:{label}'
                            records.append(dict(stage=stage,generator=gen,label=label,pair_id=identifier,
                                model_tensors_sha256=identifier,baseline_slot=slot,recipe=recipe))
                        slot+=1;q+=1
        return records,tasks

    def test_merged_counts_and_cross_shard_dedup(self):
        rows,tasks=self.merged_fixture();self.assertTrue(s.validate_merged(rows,tasks))
        rows[-1]['model_tensors_sha256']=rows[0]['model_tensors_sha256']
        with self.assertRaisesRegex(ValueError,'cross-shard duplicate'):s.validate_merged(rows,tasks)

    def test_recipe_or_omitted_pair_cannot_be_counted_complete(self):
        rows,tasks=self.merged_fixture();rows[0]['recipe']='unregistered'
        with self.assertRaises(ValueError):s.validate_merged(rows,tasks)
        rows,tasks=self.merged_fixture();rows.pop()
        with self.assertRaises(ValueError):s.validate_merged(rows,tasks)


if __name__=='__main__':unittest.main()
