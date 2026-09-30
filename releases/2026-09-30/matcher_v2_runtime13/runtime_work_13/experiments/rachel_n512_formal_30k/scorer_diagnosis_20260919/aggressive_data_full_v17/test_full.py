import tempfile,unittest
from dataclasses import dataclass,replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .plan import side_schedule,pilot_slots
from .geometry import valid_primary_peak,weather_pair
from ..aggressive_data_v17.geometry import weather_pair as approved_weather
from .resources import worker_cap,dispatch_allowed,GIB
from .generate import atomic_npz,verify_group,attempt_mode
from .fallback import numerical_identity
from ..s7_compound_v1.materialize import save_json,digest

class FullContractTests(unittest.TestCase):
    def test_v14_fallback_preserves_every_array_and_scalar(self):
        @dataclass
        class Example:
            pair_id:str
            mask_a:np.ndarray
            label:bool
        original=Example('old',np.arange(8),True)
        numerical_identity(replace(original,pair_id='new'),original)
        with self.assertRaises(ValueError):numerical_identity(replace(original,mask_a=np.zeros(8)),original)
        with self.assertRaises(ValueError):numerical_identity(replace(original,label=False),original)
    def test_endpoint_resampling_never_changes_side_quota(self):
        task=dict(mode='both',allowed_modes=['one','both'],size_class='larger',k=4,recipe='gaps')
        original=dict(task)
        self.assertEqual([attempt_mode(task,i) for i in range(4)],['both','one','both','one'])
        self.assertEqual(task,original)
        task['allowed_modes']=['middle']
        with self.assertRaises(ValueError):attempt_mode(task,1)
    def test_exact_side_blocks(self):
        for n in (750,1500,12000):
            result=side_schedule(n,91)
            self.assertEqual(result.count('smaller'),n*7//10)
            for offset in range(0,n,10):self.assertEqual(result[offset:offset+10].count('smaller'),7)
    def test_side_deterministic(self):self.assertEqual(side_schedule(30,18),side_schedule(30,18))
    def test_reject_partial_block(self):
        with self.assertRaises(ValueError):side_schedule(11,0)
    def test_only_peak_ceiling_changed(self):
        self.assertIs(weather_pair,approved_weather)
        for x in (5,15,25,30,35):self.assertTrue(valid_primary_peak(x))
        for x in (4.999,35.001,np.inf,np.nan):self.assertFalse(valid_primary_peak(x))
    def test_cpu_reserve(self):
        self.assertEqual(worker_cap(dict(cpu_quota_cores=108),64),32)
        self.assertEqual(worker_cap(dict(cpu_quota_cores=20),32),10)
    def resource(self):return dict(cpu_quota_cores=108,load1=27,memory_bytes=46*GIB,memory_limit_bytes=360*GIB,disk_free_bytes=122*GIB)
    def test_resource_good(self):self.assertTrue(dispatch_allowed(self.resource()))
    def test_resource_memory(self):
        x=self.resource();x['memory_bytes']=300*GIB;self.assertFalse(dispatch_allowed(x))
    def test_resource_disk(self):
        x=self.resource();x['disk_free_bytes']=49*GIB;self.assertFalse(dispatch_allowed(x))
    def test_resource_load(self):
        x=self.resource();x['load1']=90;self.assertFalse(dispatch_allowed(x))
    def test_atomic_archive(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'a.npz';atomic_npz(p,x=np.arange(7))
            with np.load(p) as z:np.testing.assert_array_equal(z['x'],np.arange(7))
            self.assertFalse(p.with_name(p.name+'.tmp').exists())
    def test_commit_binding(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'commit.json';save_json(p,dict(task={'slot':1},plan_sha256='old',status='committed',records=[]))
            with self.assertRaises(ValueError):verify_group(p,{'slot':1},'new')
            with self.assertRaises(ValueError):verify_group(p,{'slot':1},'old')
    def test_pilot_covers_k_and_side(self):
        ts=[dict(slot=i,recipe='gaps',partial_mode=None,size_class='smaller' if i%2 else 'larger',k=i//2+1) for i in range(8)]
        self.assertEqual(pilot_slots({'tasks':ts}),list(range(8)))

if __name__=='__main__':unittest.main()
