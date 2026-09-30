import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from . import generate as g

REFERENCE = Path(os.environ.get('REFERENCE_DIR', 'artifacts/matcher_v2_20260930/reference_v42/data_gen')).resolve()


class ReuseTests(unittest.TestCase):
    def test_blocks_non_synthetic_and_excludes_parent_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(path=str(root/'model/masks_800/gen3voronoi/1/0.png'),
                         fragment_token='rachel/gen3voronoi/1/0', source_family='u4761_frag3.png'),
                    dict(path=str(root/'model/masks_800/gen3voronoi/2/0.png'),
                         fragment_token='rachel/gen3voronoi/2/0', source_family='safe-synthetic-parent.png')]
            admission = dict(status='passed', errors=[], collision_counts={'path': 0},
                             source_root=folder, inventory={'train':rows})
            audit = dict(synthetic_inventory={'train':dict(
                potential_turufan_parent_aliases={'u4761_frag3.png':['u4761_frag1']}, exact_prepared_mask_matches=[])})
            _, allowed, excluded = g.source_pool(admission, audit, 'train')
            self.assertEqual(len(allowed),1); self.assertEqual(len(excluded),1)
            rows[1]['path'] = str(root/'real_dunhuang/1.png')
            with self.assertRaises(ValueError):g.source_pool(admission, audit, 'train')

    def test_rejects_real_mask_collision(self):
        admission = dict(status='passed', errors=[], collision_counts={}, source_root='/tmp', inventory={'train':[]})
        audit = dict(synthetic_inventory={'train':dict(potential_turufan_parent_aliases={}, exact_prepared_mask_matches=[{}])})
        with self.assertRaises(AssertionError):g.source_pool(admission,audit,'train')

    def test_review_exactly_30_independent_ids_and_subtypes(self):
        ref,_=g.load_reference(REFERENCE)
        jobs=g.choose_review_jobs(ref,Path('/tmp/review'),26093082)
        self.assertEqual(len(jobs),30)
        self.assertEqual(len({(x[0],x[1],x[2]) for x in jobs}),30)
        for kind in 'MJR':
            self.assertEqual(sum(j[0]==kind and j[1]==1 for j in jobs),8)
            self.assertEqual(sum(j[0]==kind and j[1]==0 for j in jobs),2)
        bases={ref.base_of(j[0],j[5],j[3],j[1],j[2]) for j in jobs if j[0]=='J' and j[1]}
        self.assertEqual(bases,{'torn_rachel','margin_fragment','torn_strip'})

    def test_instrumentation_preserves_reference_pixels_and_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('plain','instrumented'):
                (root/name/'samples').mkdir(parents=True);(root/name/'proof').mkdir()
            ref,_=g.load_reference(REFERENCE);ref.SRCS=[]
            a=ref.one(('R',1,0,'train',str(root/'plain'),26093082))
            g.initialize(REFERENCE,{})
            b=g.generate_one(('R',1,0,'train',str(root/'instrumented'),26093082))
            self.assertFalse(a.get('failed'));self.assertFalse(b.get('failed'))
            with np.load(a['sample_path']) as left,np.load(b['sample_path']) as right:
                self.assertEqual(left.files,right.files)
                for key in left.files:np.testing.assert_array_equal(left[key],right[key],err_msg=key)
            self.assertTrue(Path(b['proof_path']).is_file())


if __name__=='__main__':unittest.main()
