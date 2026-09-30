import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from .geometry_contract import violations
from .pair_identity import base_pair_sha256
from .repair_geometry import replacement_plan
from .full_pipeline import commands


def row(i,label=True,kind='J',bend=9,rect=.7,split='train'):
    e=dict(id=str(i),pair_id='pair'+str(i),label=label,recipe='straight_'+kind,split=split,sample_sha256=str(i),meta={'base':'torn_rachel'})
    a=dict(pair_id=e['pair_id'],sample_sha256=str(i),positive=label,
           metrics=dict(seam={'bend_range':bend},smaller_rectangularity=rect))
    return e,a


class RepairGeometryTest(unittest.TestCase):
    def test_exact_bend_boundary(self):
        m=dict(seam={'bend_range':15.},smaller_rectangularity=.7)
        self.assertEqual(violations('J',m),[])
        m['seam']['bend_range']=15.00001
        self.assertEqual(violations('J',m),['bend_gt15'])

    def test_rect_boundaries(self):
        m=dict(seam={'bend_range':9.},smaller_rectangularity=.9)
        self.assertEqual(violations('J',m),['J_rect_ge0.9'])
        self.assertEqual(violations('R',m),[])
        m['smaller_rectangularity']=.89999
        self.assertEqual(violations('R',m),['R_rect_lt0.9'])

    def test_invalid_measurement(self):
        self.assertEqual(violations('M',{'seam':None}),['unmeasurable_seam'])
        with self.assertRaises(ValueError):violations('M',dict(seam={'bend_range':float('nan')},smaller_rectangularity=.7))

    def test_only_bad_positive_replaced(self):
        values=[row(1),row(2,bend=19),row(3,label=False,bend=20)]
        entries=[x[0] for x in values]; audits=[x[1] for x in values];before=copy.deepcopy(entries)
        kept,jobs=replacement_plan(entries,audits,14,'sha',lambda _:self.fail('No TEST measurements allowed here'))
        self.assertEqual([r['pair_id'] for r in kept],['pair1','pair3'])
        self.assertEqual(jobs,[(entries[1],['bend_gt15'])]);self.assertEqual(entries,before)
        self.assertTrue(all(r['generation_seed']==14 for r in kept))

    def test_test_fixed_rule_not_calibration(self):
        e,a=row(1,split='test');a['metrics']=None;calls=[]
        def measure(e):calls.append(e['pair_id']);return dict(seam={'bend_range':20},smaller_rectangularity=.7)
        kept,jobs=replacement_plan([e],[a],14,'sha',measure)
        self.assertEqual(calls,['pair1']);self.assertEqual(len(jobs),1);self.assertEqual(kept,[])

    def test_parent_binding_fails_closed(self):
        e,a=row(1);a['sample_sha256']='other'
        with self.assertRaises(ValueError):replacement_plan([e],[a],14,'sha',None)

    def test_pipeline_only_repairs_then_audits(self):
        a=SimpleNamespace(reference_dir=Path('/ref'),source_admission=Path('/admission'),real_audit=Path('/real'),
            workers=3,target_code=Path('/targets'),metrics_code=Path('/metrics'),official_code=Path('/official'),parent_root=Path('/old'))
        steps=commands(a,Path('/new'))
        self.assertEqual([name for name,_ in steps],['train6000_repair','train6000_audit','select900_repair','select900_audit','test900_repair','test900_audit'])
        self.assertTrue(all('full_generate.py' not in str(cmd) for _,cmd in steps))

    def test_pair_identity_ignores_swap_center_mirror(self):
        a=np.array([[1,1,0],[1,0,0]],bool);b=np.array([[0,1],[1,1],[0,1]],bool)
        h=base_pair_sha256(a,b)
        self.assertEqual(h,base_pair_sha256(np.pad(b,3),np.pad(a,((2,1),(4,0)))))
        self.assertEqual(h,base_pair_sha256(np.rot90(a)[:,::-1],np.rot90(b)[:,::-1]))
        changed=b.copy();changed[0,0]=True
        self.assertNotEqual(h,base_pair_sha256(a,changed))


if __name__=='__main__':unittest.main()
