"""New supplement logic only. No pixels, old gates, models, or remote jobs."""
import copy
import unittest

import heldout_extend_plan as extension
import heldout_shortfall_supplement as supplement


def fixture():
    def row(pid, label):
        return dict(pair_id=pid, label=label, split='val',
            fragment_a=dict(generator='Gen3'), fragment_b=dict(generator='Gen3'))
    positives = [row(f'p{i}', True) for i in range(4)]
    negative = [dict(pair_id=f'n{i}', mode='native', row=row(f'n{i}', False)) for i in range(4)]
    combinations = [(i, j) for i in range(4) for j in range(4) if i != j]
    tasks = [dict(role='cal', generator='Gen3', stage='v17_filtered', quota_slot=19,
        recipe='partial', size_class='smaller', mode='one', k=4, master_seed=261002910,
        reserve_index=r, slot=r, base_pair_ids=[f'p{i}', f'n{j}'], trim_target=.3)
        for r,(i,j) in enumerate(combinations)]
    spec = dict(source_families=['CAL_ONLY'], positive_pool=positives,
        positive=[dict(pair_id=f'p{i}', source_stratum='native_positive') for i,j in combinations],
        negative=[copy.deepcopy(negative[j]) for i,j in combinations],
        slot_generators=['Gen3']*12,
        schedule=dict(recipes=['partial']*12, partial=[True]*12, partial_modes=['end']*12,
                      bins=['native']*12, mirrors=[0]*12),
        donor_bank=dict(path='/original/cal/bank'), pairs=24,
        generator_counts=dict(Gen3=dict(candidate_groups=12)))
    sources = dict(seed=261002910, forbidden_sources=['TRAIN_ONLY'],
        splits=dict(cal=spec, select=dict(source_families=['SELECT_ONLY'])))
    generation = dict(master_seed=261002910, tasks=tasks)
    shortfall = dict(status='shortfall', role='cal', shard=dict(name='cal_Gen3_02'),
        planned_quotas=45, admitted_pairs=88,
        records=[dict(label=bool(i%2)) for i in range(88)],
        failures=[dict(task=t, phase='augmentation') for t in tasks]+
            [dict(quota=['Gen3',19,'v17_filtered'], phase='quota_exhausted')])
    return generation, sources, shortfall


class SupplementTests(unittest.TestCase):
    def construct(self):
        g,s,f = fixture()
        return supplement.make_extension(g,s,f,extension)

    def test_four_candidates_are_registered_deterministically(self):
        self.assertEqual(self.construct(), self.construct())
        _,p = self.construct()
        self.assertEqual([t['reserve_index'] for t in p['tasks']], [12,13,14,15])
        self.assertEqual([t['slot'] for t in p['tasks']], [12,13,14,15])
        self.assertEqual(p['max_candidates'],4)
        self.assertEqual(p['per_candidate_seconds'],900)

    def test_original_objects_and_source_pools_are_unchanged(self):
        g,s,f = fixture()
        before = copy.deepcopy((g,s,f))
        out,p = supplement.make_extension(g,s,f,extension)
        self.assertEqual((g,s,f), before)
        self.assertEqual(out['splits']['select'],s['splits']['select'])
        for key in ('positive_pool','donor_bank','source_families'):
            self.assertEqual(out['splits']['cal'][key],s['splits']['cal'][key])
        for key in ('positive','negative','slot_generators'):
            old = s['splits']['cal'][key]
            self.assertEqual(out['splits']['cal'][key][:len(old)],old)
        for key in supplement.SCHEDULE_FIELDS:
            self.assertEqual(out['splits']['cal']['schedule'][key][:12],s['splits']['cal']['schedule'][key])

    def test_extra_candidates_stay_in_exact_gen_role_recipe(self):
        out,p = self.construct()
        self.assertEqual(len({tuple(t['base_pair_ids']) for t in p['tasks']}),4)
        for t in p['tasks']:
            self.assertEqual((t['role'],t['generator'],t['quota_slot'],t['stage']),supplement.TARGET)
            self.assertEqual((t['recipe'],t['size_class'],t['mode'],t['k']),('partial','smaller','one',4))
            self.assertEqual(t['trim_target'],extension.trim_target(t['master_seed'],'cal',t['stage'],t['base_pair_ids'][0]))

    def test_reject_other_or_multiple_shortfalls(self):
        for change in ('other','multiple','count'):
            g,s,f=fixture()
            if change=='other':f['failures'][-1]['quota'][0]='Gen4'
            elif change=='multiple':f['failures'].append(copy.deepcopy(f['failures'][-1]))
            else:f['admitted_pairs']=86
            with self.assertRaises(ValueError):supplement.make_extension(g,s,f,extension)

    def test_reject_missing_original_rejection_or_leaking_family(self):
        g,s,f=fixture();f['failures'].pop(0)
        with self.assertRaises(ValueError):supplement.make_extension(g,s,f,extension)
        g,s,f=fixture();s['splits']['cal']['source_families'].append('SELECT_ONLY')
        with self.assertRaises(ValueError):supplement.make_extension(g,s,f,extension)

    def test_acceptance_does_not_depend_on_completion_order(self):
        results=[dict(index=i,status='candidate_ready_pending_release' if i in (1,3) else 'rejected') for i in range(4)]
        a=supplement.summarize(results);b=supplement.summarize(results[::-1])
        self.assertEqual(a,b)
        self.assertEqual(a['provisional_selected_index'],1)
        self.assertTrue(a['final_cross_fold_dedup_and_release_pending'])
        self.assertEqual(a['added_pairs_limit'],2)

    def test_exhaustion_can_leave_exact_two_pair_deficit(self):
        result=supplement.summarize([dict(index=i,status='rejected' if i%2 else 'bounded_time_budget_exhausted') for i in range(4)])
        self.assertEqual(result['status'],'bounded_batch_exhausted_shortfall_authorized')
        self.assertEqual(result['unresolved_missing_pairs'],2)
        self.assertIsNone(result['provisional_selected_index'])
        self.assertFalse(result['automatic_retry'])

    def test_unexpected_failure_or_missing_actual_result_is_not_waived(self):
        values=[dict(index=i,status='rejected') for i in range(4)]
        with self.assertRaises(ValueError):supplement.summarize(values[:3])
        values[1]['status']='unexpected_failure'
        with self.assertRaises(ValueError):supplement.summarize(values)


if __name__=='__main__':
    unittest.main()
