from collections import Counter
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import generate as g
from . import supervision as s
if os.environ.get('TARGET_CODE'):
    spec=importlib.util.spec_from_file_location('target_builder',os.environ['TARGET_CODE'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    projected_interval_damage=module.projected_interval_damage
else:
    from ..matcher_v2_v1.seam_supervision import projected_interval_damage
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample


REFERENCE = Path(os.environ.get('REFERENCE_DIR', 'artifacts/matcher_v2_20260930/reference_v42/data_gen')).resolve()


class SupervisionTests(unittest.TestCase):
    def test_rng_proxy_has_no_rng_side_effect(self):
        a = np.random.default_rng(99); b = np.random.default_rng(99)
        recorded = s.RecordingRNG(b)
        self.assertEqual(a.uniform(2,5), recorded.uniform(2,5))
        self.assertEqual(a.random(), recorded.random())
        np.testing.assert_array_equal(a.integers(100,size=20),b.integers(100,size=20))

    def test_parser_rejects_changed_reference_protocol(self):
        p=dict(wear=None,gap_cover=(.1,.2),gap_depth=(2,3),ov_p=0.)
        with self.assertRaises(ValueError):s.parse_events([('uniform',.5,.6,.55)],p,0,100)

    def test_reference_pixel_replay_and_explicit_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'samples').mkdir();(root/'proof').mkdir()
            seed=26093082
            g.initialize(REFERENCE,{})
            e=g.generate_one(('R',1,0,'train',str(root),seed))
            sample,_=load_sample(e['sample_path'])
            with np.load(e['proof_path']) as f:proof={k:f[k] for k in f.files}
            g.CONTEXT.update(donors={},rejections=Counter(),proof=None)
            a,b,trace=s.replay_final_attempt(g.CONTEXT['reference'],e,seed)
            np.testing.assert_array_equal(a,s.unpack(proof,'final_parent_a'))
            np.testing.assert_array_equal(b,s.unpack(proof,'final_parent_b'))
            targets, report=s.build_supervision(sample,proof,trace,projected_interval_damage)
            self.assertGreater(report['original_matches_touching_damage'],0)
            self.assertGreater(report['correspondence_count'],8)
            self.assertFalse(report['training_admitted'])
            for side in 'ab':
                target=targets['target_'+side]
                np.testing.assert_array_equal(target[targets['damaged_'+side]],-2)
                valid=target>=0
                np.testing.assert_array_equal(target[valid],getattr(sample,'target_'+side)[valid])
            self.assertTrue(all(0<=v<=1.4 for v in trace['wear']))
            self.assertNotIn('uniform_wear',[i['kind'] for i in trace['intervals']])

    def test_recorded_interval_shoulders_not_only_removed_pixels(self):
        along=np.array([9.,10.,11.,29.,30.,31.,34.])
        actual=projected_interval_damage(along,[(10.,30.)],margin_px=3.)
        np.testing.assert_array_equal(actual,[True,True,True,True,True,True,False])


if __name__=='__main__':unittest.main()
