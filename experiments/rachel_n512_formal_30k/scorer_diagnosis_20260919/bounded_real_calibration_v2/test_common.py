import unittest
from common import select,metrics,GRID,crossfit
from copy import deepcopy


class TestThresholds(unittest.TestCase):
    def test_boundaries(self):
        chosen=select([1,1,0,0],[.01,.02,.001,.0001],[True]*4,'bounded_max_f1')
        self.assertEqual(chosen['threshold'],.3)
        self.assertEqual(chosen['calibration_metrics']['recall'],0)
    def test_tie_prefers_03(self):
        chosen=select([1,0],[.75,.02],[True,True],'bounded_max_f1')
        self.assertEqual(chosen['threshold'],.3)
    def test_unattainable_recall_never_escapes_range(self):
        chosen=select([1,1,0],[.01,.5,.3],[True]*3,'bounded_recall95')
        self.assertFalse(chosen['target_met']);self.assertEqual(chosen['threshold'],.2)
    def test_invalid_never_accepted(self):
        chosen=select([1,1,0],[.99,.7,.4],[False,True,True],'bounded_max_f1')
        self.assertEqual(chosen['calibration_metrics']['tp'],1)
    def test_tied_auc(self):
        self.assertEqual(metrics([1,0],[True,True],[.5,.5])['auroc'],.5)
    def test_grid(self):
        self.assertEqual((len(GRID),min(GRID),max(GRID)),(61,.2,.8))
    def test_heldout_labels_do_not_set_own_threshold(self):
        population=[];predicted=[];groups={}
        for k in range(5):
            for y in (0,1):
                pid=f'{k}_{y}'; a=pid+'_a'; b=pid+'_b'
                groups[a]=groups[b]=str(k)
                population.append(dict(pair_id=pid,fragment_a_id=a,fragment_b_id=b,fold=k,label=y))
                predicted.append(dict(pair_id=pid,score=.7 if y else .2,decision_valid=True))
        meta=dict(split='ood',pairs=population,fragment_source_group=groups)
        before,_=crossfit(meta,predicted,'bounded_max_f1')
        changed=deepcopy(meta)
        for r in changed['pairs']:
            if r['fold']==0:r['label']=1-r['label']
        after,_=crossfit(changed,predicted,'bounded_max_f1')
        self.assertEqual(before['folds'][0]['threshold'],after['folds'][0]['threshold'])
    def test_confusion_arithmetic(self):
        m=metrics([1,1,1,0,0],[True,True,False,True,False],[.9,.8,.1,.7,.2])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(2,1,1,1))
        self.assertAlmostEqual(m['accuracy'],.6)
        self.assertAlmostEqual(m['f1'],2/3)


if __name__=='__main__':unittest.main()
