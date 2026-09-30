from collections import Counter
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import full_generate as f, generate as g
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample

REFERENCE=Path(os.environ.get('REFERENCE_DIR','artifacts/matcher_v2_20260930/reference_v42/data_gen')).resolve()


class FullTests(unittest.TestCase):
    def test_full_counts_and_split_seed_separation(self):
        self.assertEqual([2*sum(x.values()) for x in f.COUNTS.values()],[6000,900,900])
        self.assertEqual(len(set(f.SEEDS.values())),3)
        self.assertFalse(set(f.SEEDS.values()) & {26093004,26093015,26093016,26093082,26093083})

    def test_J_quota_held_exact_for_both_labels(self):
        ref,_=g.load_reference(REFERENCE)
        for split in f.COUNTS:
            for label in (0,1):
                n=f.COUNTS[split]['J']
                bases=f.j_bases(n,f.SEEDS[split],split,label,ref.SPLIT_SEED[split])
                self.assertEqual(Counter(bases),{'torn_rachel':n*20//25,'margin_fragment':n*3//25,'torn_strip':n*2//25})
                self.assertEqual(bases,f.j_bases(n,f.SEEDS[split],split,label,ref.SPLIT_SEED[split]))
        with self.assertRaises(ValueError):f.j_bases(26,1,'train',1,1)

    def test_parameter_rng_preserves_stream(self):
        a=np.random.default_rng(19); b=np.random.default_rng(19); proxy=f.ParameterRNG(b)
        for low,high in ((1,2),(3,12),(2,4),(0,1)):
            self.assertEqual(a.uniform(low,high),proxy.uniform(low,high))
        np.testing.assert_array_equal(a.integers(100,size=10),proxy.integers(100,size=10))
        self.assertEqual(len(proxy.shape_draws),3)

    def test_interval_boundaries_and_invalid(self):
        np.testing.assert_array_equal(f.interval_damage(np.array([6,7,13,33,34]),[(10,30)]),[False,True,True,True,False])
        with self.assertRaises(ValueError):f.interval_damage(np.zeros(1),[(30,10)])

    def test_R_model_inputs_and_GT_unchanged_labels_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('old','new'):
                (root/name/'samples').mkdir(parents=True);(root/name/'proof').mkdir()
            g.initialize(REFERENCE,{})
            old=g.generate_one(('R',1,0,'train',str(root/'old'),26093082))
            f.initialize(REFERENCE,{},f.COUNTS['train'])
            new=f.generate_one(('R',1,0,'train',str(root/'new'),26093082))
            self.assertEqual(old['tries'],new['tries'])
            a,_=load_sample(old['sample_path']);b,report=load_sample(new['sample_path'])
            for key in ('mask_a','mask_b','points_rc_a','points_rc_b','contour_valid_a','contour_valid_b','translation_a_to_b_rc'):
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key),err_msg=key)
            self.assertGreater((b.target_a==-2).sum(),0)
            self.assertGreaterEqual((b.target_a>=0).sum(),8)
            self.assertLess((b.target_a>=0).sum(),(a.target_a>=0).sum())
            self.assertEqual(report['pose_supervision_enabled'],bool(b.label) and not report['changed_pair'])
            self.assertIn('ignore (-2)',report['inheritance_rule'])
            self.assertEqual(new['target_audit']['correspondence_count'],(b.target_a>=0).sum())
            with np.load(new['proof_path']) as proof:
                np.testing.assert_array_equal(proof['original_target_a'],a.target_a)

    def test_insufficient_labels_reject_without_relaxing_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'samples').mkdir();(root/'proof').mkdir()
            f.initialize(REFERENCE,{},f.COUNTS['train'])
            original=f.s.build_supervision
            calls=[]
            def reject_first(*args,**kwargs):
                targets,report=original(*args,**kwargs)
                calls.append(1)
                if len(calls)==1:report['correspondence_count']=7
                return targets,report
            with mock.patch.object(f.s,'build_supervision',side_effect=reject_first):
                entry=f.generate_one(('R',1,0,'train',str(root),26093082))
            self.assertFalse(entry.get('failed'))
            self.assertGreaterEqual(entry['rejections']['few_healthy_corr'],1)
            self.assertGreater(entry['tries'],1)
            self.assertGreaterEqual(entry['target_audit']['correspondence_count'],8)
            self.assertEqual(entry['meta']['base'],'strip')

    def test_negative_targets_and_no_GT_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'samples').mkdir();(root/'proof').mkdir()
            f.initialize(REFERENCE,{},f.COUNTS['train'])
            entry=f.generate_one(('R',0,0,'train',str(root),26093082))
            sample,report=load_sample(entry['sample_path'])
            self.assertTrue(np.all(sample.target_a==-1) and np.all(sample.target_b==-1))
            self.assertFalse(sample.translation_valid or report['pose_supervision_enabled'])
            self.assertIsNone(entry['target_audit'])


if __name__=='__main__':unittest.main()
