"""Local regression against the actual rejected v16 clean sample, not a toy."""
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from staging.pairwise_v0_2.pairwise_data.rachel_materialized_dataset import load_sample
from ..s7_balanced_v2.latent_seam import source_band
from ..s7_balanced_v2.partial_v14 import retained_support
from .endpoints import audit_endpoints


class ActualMaskRegression(unittest.TestCase):
    def test_previously_accepted_middle_cut_fails(self):
        root=Path('artifacts/aggressive_data_v16_20260927/probe_audit_01')
        if not root.is_dir():self.skipTest('local historical mask fixture not mounted')
        sample,_=load_sample(root/'baseline/01111_0.npz')
        with np.load(root/'proof/01111_0.npz') as z:
            cropped=replace(sample,**{'mask_'+s:np.unpackbits(z['packed_trim_'+s],axis=1)[None].astype(np.float32) for s in 'ab'})
        bands=source_band(sample,bridge=0.)
        _,before=retained_support(sample,sample,bands)
        _,after=retained_support(sample,cropped,bands)
        with self.assertRaisesRegex(ValueError,'interior'):
            audit_endpoints(bands,before,after,'one')

if __name__=='__main__':unittest.main()
