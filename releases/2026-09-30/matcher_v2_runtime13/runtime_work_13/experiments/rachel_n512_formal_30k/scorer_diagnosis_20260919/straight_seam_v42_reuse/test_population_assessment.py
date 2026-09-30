import copy
import unittest

from .test_frozen_diagnostic import fixture
from .population_assessment import inspect, distribution, selection_bias


def full_fixture():
    manifest, audit = fixture()
    counters = dict(M=0, J=0, R=0)
    for e, r in zip(manifest['entries'], audit['records']):
        k = r['kind']
        if not e['label']:
            continue
        i = counters[k]; counters[k] += 1
        base = 'rachel' if k == 'M' else ('strip' if k == 'R' else (
            'torn_rachel' if i < 160 else ('margin_fragment' if i < 184 else 'torn_strip')))
        shape = dict(rough_std_px=1., rough_corr_px=5., bend_amp_px=3.)
        trace = dict(wear=[0., 0.], gap_coverage_draw=.3, overlap_active=False, overlap_coverage_draw=None)
        e.update(meta=dict(base=base, **shape), accepted_damage_trace=trace, tries=1, rejections={},
            piece_attempts=[dict(base=base, shape=shape, piece_rejection=None, misfit_trace=trace)])
        r.update(base=base, id=e['pair_id'], metrics=dict(smaller_rectangularity=.96 if k == 'R' else .75,
            seam=dict(extent_px=468. if k == 'R' else 223., bend_range=6.1 if k == 'R' else 9.,
                rough_std=.71 if k == 'R' else 1.42, gap_frac_over3=.43 if k == 'R' else .27,
                overlap_per_extent=0., slide_m10=.7, slide_m20=1., slide_m40=2., complementary_150=k == 'R')))
    plan=dict(split='select', preflight=False, SEAM={k:{} for k in 'MJR'}, MIS={k:{} for k in 'MJR'})
    return manifest, audit, plan


class PopulationAssessmentTest(unittest.TestCase):
    def test_full_population_passes_without_training_admission(self):
        m, a, p = full_fixture()
        before=copy.deepcopy((m,a,p))
        r=inspect(m,a,p,'b'*64)
        self.assertEqual(r['status'],'passed_geometry_checks')
        self.assertFalse(r['training_admitted'])
        self.assertEqual((m,a,p),before)

    def test_tail_does_not_hide_behind_median(self):
        m,a,p=full_fixture()
        next(r for r in a['records'] if r['positive'] and r['kind']=='J')['metrics']['seam']['bend_range']=20.
        r=inspect(m,a,p,'b'*64)
        self.assertEqual(r['status'],'not_admitted')
        self.assertEqual(r['groups']['J']['bend_over15'],1)
        self.assertTrue(next(c for c in r['checks'] if c['kind']=='J' and c['metric']=='bend_range')['passed'])

    def test_rectangularity_tail_reported(self):
        m,a,p=full_fixture()
        next(r for r in a['records'] if r['positive'] and r['kind']=='R')['metrics']['smaller_rectangularity']=.899
        self.assertEqual(inspect(m,a,p,'b'*64)['groups']['R']['rectangularity_outliers'],1)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):distribution([float('nan')])

    def test_test_and_preflight_forbidden(self):
        for change in ({'split':'test'},{'preflight':True}):
            m,a,p=full_fixture();p.update(change)
            with self.assertRaises(ValueError):inspect(m,a,p,'b'*64)

    def test_missing_damage_not_zero_filled(self):
        m,_,_=full_fixture();e=next(r for r in m['entries'] if r['label'])
        e['piece_attempts'].insert(0,dict(base=e['meta']['base'],
            shape=dict(rough_std_px=2.,rough_corr_px=5.,bend_amp_px=3.),
            misfit_trace=None,piece_rejection='seam_length'))
        r=selection_bias([e])
        self.assertEqual(r['attempts_before_damage_sampling'],1)
        self.assertEqual(r['damage']['wear_sum_px']['attempted']['n'],1)
        self.assertEqual(r['shape']['rough_std_px']['survivor_minus_attempt_mean'],-.5)

    def test_trace_tampering_rejected(self):
        m,_,_=full_fixture();e=next(r for r in m['entries'] if r['label'])
        e['meta']['bend_amp_px']=6
        with self.assertRaises(ValueError):selection_bias([e])


if __name__=='__main__':unittest.main()
