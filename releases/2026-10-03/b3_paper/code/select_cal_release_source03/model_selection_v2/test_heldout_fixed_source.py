"""Regression for actual source fallback; no geometry/inference reruns."""
import copy
from types import SimpleNamespace
import unittest
from .heldout_reduce_run import baseline_identity, fixed_source_slot
from ..s7_balanced_v2.source_search import SourceSearch


class FixedSourceTests(unittest.TestCase):
    def fixture(self):
        def row(pid,label):return dict(pair_id=pid,label=label,fragment_a={'generator':'gen2voronoi_1'},fragment_b={'generator':'gen2voronoi_1'})
        pos,neg=row('p',True),row('n',False)
        spec=dict(positive=[{'pair_id':'p'}],negative=[{'pair_id':'n','row':neg}],positive_pool=[pos])
        task=dict(slot=0,base_pair_ids=['p','n'],recipe='partial',generator='Gen2')
        baseline=dict(slot=0,original_positive_pair_id='p',source_positive_pair_id='p',source_negative_pair_id='n',recipe='partial',
                      entries=[dict(source_pair_id=r['pair_id'],label=r['label'],source_row=dict(r,pair_id='sample-id')) for r in (pos,neg)])
        return baseline,task,spec

    def test_exact_sources_admitted_but_old_implicit_replacement_not(self):
        b,t,s=self.fixture();self.assertTrue(baseline_identity(b,t,s))
        b['source_positive_pair_id']='other';b['entries'][0]['source_pair_id']='other'
        self.assertFalse(baseline_identity(b,t,s))

    def test_changed_negative_original_gen_row_or_label_raises(self):
        for key in ('negative','original','gen','row','label'):
            b,t,s=self.fixture()
            if key=='negative':b['source_negative_pair_id']='other'
            elif key=='original':b['original_positive_pair_id']='other'
            elif key=='gen':t['generator']='Gen3'
            elif key=='row':b['entries'][0]['source_row']['extra']='unexpected'
            else:b['entries'][0]['label']=False
            with self.subTest(key=key),self.assertRaises(ValueError):baseline_identity(b,t,s)

    def test_original_source_search_stays_fixed_after_four_draws(self):
        import numpy as np
        original={'pair_id':'p','source_stratum':'native_positive'}
        state={'positive':[original],'buckets':{'native_positive':[original,{'pair_id':'other','source_stratum':'native_positive'}]}}
        before=copy.deepcopy(state);material=SimpleNamespace(STATE=state)
        def slot(index):
            search=SourceSearch(original,state['buckets']['native_positive'],np.random.default_rng(1),1024,False)
            sources=[]
            while (draw:=search.draw()) is not None:
                _,source=draw;sources.append(source['pair_id']);search.rejected(source)
            self.assertEqual(1024,len(sources));self.assertEqual({'p'},set(sources));return 'generated'
        self.assertEqual('generated',fixed_source_slot(SimpleNamespace(slot=slot),material,0))
        self.assertEqual(before,state)

    def test_state_is_restored_even_on_real_error(self):
        original={'pair_id':'p','source_stratum':'native_positive'}
        material=SimpleNamespace(STATE={'positive':[original],'buckets':{'native_positive':[original]}})
        previous=material.STATE['buckets']
        def failed(_):raise MemoryError('not geometry exhaustion')
        with self.assertRaises(MemoryError):fixed_source_slot(SimpleNamespace(slot=failed),material,0)
        self.assertIs(previous,material.STATE['buckets'])


if __name__=='__main__':unittest.main()
