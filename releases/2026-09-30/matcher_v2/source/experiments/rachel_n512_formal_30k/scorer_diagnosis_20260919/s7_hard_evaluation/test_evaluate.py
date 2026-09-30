from copy import deepcopy
import unittest
from .evaluate import check_selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        best=dict(epoch=0,key=[.7,.8,.9],threshold=.3)
        self.status=dict(status='training_complete',stop_reason='prespecified_M8_C8_finetune_budget_completed',
            matcher=dict(epochs=8,updates=984,exposures=62656),
            scorer=dict(epochs=8,updates=984,exposures=62656,best=best))
        self.sel=dict(best,binding={'version':1})
        self.cp=dict(phase='scorer',best=best,binding=self.sel['binding'])

    def test_epoch_zero_is_eligible(self):
        check_selection(self.status,self.sel,self.cp)

    def test_incomplete_training_rejected(self):
        s=deepcopy(self.status);s['status']='training'
        with self.assertRaises(ValueError):check_selection(s,self.sel,self.cp)

    def test_partial_budget_rejected(self):
        s=deepcopy(self.status);s['matcher']['updates']=983
        with self.assertRaises(ValueError):check_selection(s,self.sel,self.cp)

    def test_unselected_checkpoint_rejected(self):
        c=deepcopy(self.cp);c['best']['epoch']=2
        with self.assertRaises(ValueError):check_selection(self.status,self.sel,c)

    def test_changed_binding_rejected(self):
        c=deepcopy(self.cp);c['binding']={'version':2}
        with self.assertRaises(ValueError):check_selection(self.status,self.sel,c)


if __name__=='__main__':unittest.main()
