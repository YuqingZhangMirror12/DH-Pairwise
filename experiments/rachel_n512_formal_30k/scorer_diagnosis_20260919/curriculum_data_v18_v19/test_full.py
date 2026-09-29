import unittest
from collections import Counter
from .full_plan import allocation,assignment,validate_targets,tickets,choose,accept_unique


class FullTests(unittest.TestCase):
    def test_largest_remainder_preserves_exact_totals(self):
        result=allocation(1500,{'clean':2,'wave':3,'gaps':7})
        self.assertEqual(result,{'clean':250,'wave':375,'gaps':875})
        self.assertEqual(sum(allocation(13,{'a':7,'b':3,'c':2}).values()),13)

    def test_size_contract_requires_label_and_side_balance(self):
        validate_targets({'v17.5':6000,'v18':3000})
        for invalid in ({'v17.5':6001,'v18':3000},{'v17.5':0,'v18':3000},{'v19':3000,'v18':6000}):
            with self.assertRaises(ValueError):validate_targets(invalid)

    def test_source_assignment_is_identity_stable(self):
        targets={'v17.5':6000,'v18':3000}
        self.assertEqual(assignment('root::pair',targets),assignment('root::pair',targets))
        self.assertEqual({assignment(str(i),targets) for i in range(100)},{'v17.5','v18'})

    def test_tickets_preserve_recipe_and_global70_30(self):
        q={'v17.5':{'clean':20,'wave':30,'gaps':50},'v18':{'clean':10,'wave':15,'gaps':25}}
        output=tickets(q)
        for v,requests in output.items():
            self.assertEqual({r:len(s) for r,s in requests.items()},q[v])
            count=Counter(s for sides in requests.values() for s in sides)
            self.assertEqual(count['smaller']*10,7*sum(q[v].values()))
            self.assertEqual(count['larger']*10,3*sum(q[v].values()))

    def test_reserved_original_not_dispatched_concurrently(self):
        q=tickets({'v17.5':{'wave':10}})['v17.5'];used=set();reserved=set()
        tasks=[dict(slot=1,recipe='wave',base_keys=['p','n1']),dict(slot=2,recipe='wave',base_keys=['p','n2']),
               dict(slot=3,recipe='wave',base_keys=['p3','n3'])]
        first=choose(tasks,q,used,set(),reserved);second=choose(tasks,q,used,set(),reserved)
        self.assertEqual((first['slot'],second['slot']),(1,3))
        self.assertIsNone(choose(tasks,q,used,set(),reserved))

    def test_accepted_original_never_reused(self):
        q=tickets({'v18':{'wave':10}})['v18'];task=dict(slot=2,recipe='wave',base_keys=['p','n2'])
        self.assertIsNone(choose([task],q,set(),{'p'},set()))

    def test_duplicate_source_and_inputs_rejected(self):
        r=[dict(label=True,source_base_key='p'),dict(label=False,source_base_key='n')]
        a=[dict(model_input_sha256='a'),dict(model_input_sha256='b')]
        keys=set();hashes=set();accept_unique(r,a,keys,hashes)
        self.assertEqual(keys,{'p','n'});self.assertEqual(hashes,{'a','b'})
        with self.assertRaises(ValueError):accept_unique(r,a,keys,hashes)
        with self.assertRaises(ValueError):accept_unique(r,a,set(),{'b'})

    def test_rejection_keeps_requested_side_and_recipe(self):
        q=tickets({'v18':{'gaps':10}})['v18'];before=Counter(q['gaps'])
        task=choose([dict(slot=1,recipe='gaps',base_keys=['p','n'])],q,set(),set(),set())
        q[task['recipe']].append(task['size_class'])
        self.assertEqual(before,Counter(q['gaps']))


if __name__=='__main__':unittest.main()
