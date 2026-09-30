from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import recover as r

LOCAL=Path(__file__).resolve().parents[4]
SOURCE=LOCAL if (LOCAL/'staging').exists() else r.BASE/'source_03'
sys.path.insert(0,str(SOURCE))
f=r.finalizer()
LOADER='staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset.load_sample'


@dataclass
class NumericFixture:
    pair_id: str
    mask_a: np.ndarray
    target_a: np.ndarray
    coarse_mask_a: np.ndarray
    label: float


def sample(name='a',value=1.):
    return NumericFixture(name,np.array([value],np.float32),np.array([0],np.int64),np.array([value],np.float32),1.)


def record(name):
    return dict(v14_fallback=True,source_pair_id='same-source',label=True,pair_id=name,
                sample_path=name+'.npz',baseline_sample_path=name+'-v14.npz')


class Tests(unittest.TestCase):
    def call(self,a=None,b=None,sa='train',sb='train',values=None):
        a=a or record('a');b=b or record('b')
        values=values or {x:sample(x) for x in ['a.npz','b.npz','a-v14.npz','b-v14.npz']}
        with patch(LOADER,side_effect=lambda path:(values[path],{})),patch.object(f,'digest',side_effect=lambda p:'bound-'+p):
            return f.inherited_duplicate(a,b,sa,sb)

    def test_same_fold_original_retained(self):
        x=self.call();self.assertTrue(x['same_original_v14_numerical_content']);self.assertTrue(x['model_inputs_labels_weights_unchanged'])

    def test_cross_fold_forbidden(self):
        with self.assertRaisesRegex(ValueError,'crosses'):self.call(sb='test')

    def test_new_v17_duplicate_forbidden(self):
        b=record('b');b['v14_fallback']=False
        with self.assertRaisesRegex(ValueError,'newly-created'):self.call(b=b)

    def test_distinct_source_forbidden(self):
        b=record('b');b['source_pair_id']='different'
        with self.assertRaisesRegex(ValueError,'lineage'):self.call(b=b)

    def test_contradictory_labels_forbidden(self):
        b=record('b');b['label']=False
        with self.assertRaisesRegex(ValueError,'lineage'):self.call(b=b)

    def test_duplicate_pair_id_forbidden(self):
        with self.assertRaisesRegex(ValueError,'lineage'):self.call(b=record('a'))

    def test_not_identical_to_baseline_forbidden(self):
        values={x:sample(x) for x in ['a.npz','b.npz','a-v14.npz','b-v14.npz']};values['b.npz']=sample(value=2.)
        with self.assertRaisesRegex(ValueError,'array changed'):self.call(values=values)

    def test_numeric_fields_beyond_mask_checked(self):
        values={x:sample(x) for x in ['a.npz','b.npz','a-v14.npz','b-v14.npz']}
        values['b.npz'].coarse_mask_a[0]=2.;values['b-v14.npz'].coarse_mask_a[0]=2.
        with self.assertRaisesRegex(ValueError,'coarse_mask_a'):self.call(values=values)

    def test_original_targets_cannot_change(self):
        values={x:sample(x) for x in ['a.npz','b.npz','a-v14.npz','b-v14.npz']};values['b.npz'].target_a[0]=9
        with self.assertRaisesRegex(ValueError,'target_a'):self.call(values=values)

    def test_zero_claim_removed(self):
        code=Path(f.__file__).read_text()
        self.assertIn('exact_numerical_duplicate_count=len(inherited_duplicates)',code)
        self.assertNotIn('exact_numerical_duplicate_count=0',code)

    def test_required_original_checks_retained(self):
        code=Path(f.__file__).read_text()
        for text in ['manuscript leakage between splits','historical TRAIN/donor leakage','unverified committed group',
                     'accepted sample quotas changed','archive changed after audit','duplicate Pair ID']:
            self.assertIn(text,code)

    def test_preservation_enforced(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';data.mkdir();p=data/'one.json';r.save(p,{'a':1})
            r.save(root/'preexisting.json',{'one.json':r.sha(p)})
            with patch.object(r,'DATA',data),patch.object(r,'ROOT',root):
                r.preserve_check();r.save(p,{'a':2})
                with self.assertRaisesRegex(ValueError,'modified'):r.preserve_check()

    def test_launcher_receipt_without_real_process(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fake=SimpleNamespace(environment=lambda:{},identity=lambda pid:dict(pid=pid,starttime=1,cmdline='fixture'))
            with patch.object(r,'ROOT',root),patch.object(r,'validate_tests'),patch.object(r,'validate'),\
                 patch.object(r,'numeric',return_value=fake),patch.object(r.subprocess,'Popen',return_value=SimpleNamespace(pid=123)),\
                 patch.object(sys,'argv',['fixture']),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                r.main()
                self.assertEqual(r.read(root/'controller_launch.json')['controller']['pid'],123)
                with self.assertRaisesRegex(ValueError,'already registered'):r.main()

    def test_bad_test_binding_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);r.save(root/'cpu_tests_remote.json',dict(status='passed',tests=20,errors=0,failures=0,skipped=0,source_sha256={}))
            with patch.object(r,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'binding differs'):r.validate_tests()


if __name__=='__main__':unittest.main()
