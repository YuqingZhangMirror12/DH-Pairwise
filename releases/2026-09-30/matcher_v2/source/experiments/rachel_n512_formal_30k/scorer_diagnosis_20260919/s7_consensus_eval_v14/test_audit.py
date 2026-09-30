import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .audit import audit_snapshot, validate_arrays
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.compatibility import CompatibilityConfig
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.consensus_head import ConsensusEvidenceHead
from .snapshot import snapshot_prediction
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.diagnostics import write_snapshot
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.model import S7Consensus
from experiments.rachel_n512_formal_30k.scorer_diagnosis_20260919.s7_consensus_v1.test_evidence import fixture


class EvidenceReadbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,30.),
                              head=ConsensusEvidenceHead(feature_dim=4)).eval()
        self.pair=fixture()
        with torch.no_grad():
            self.pred=self.model.score_pair(self.pair,capture_diagnostics=True)
            self.meta,self.arrays=snapshot_prediction('toy',self.pair,self.pred,threshold=.5,
                provenance={'purpose':'CPU evidence contract, not trained real performance'})

    def test_actual_model_evidence_all_accounting(self):
        result=validate_arrays(self.meta,self.arrays)
        self.assertEqual(result['status'],'passed')
        self.assertFalse(result['layout_correctness_verified'])
        self.assertEqual(len(result['clusters']),len(self.pred.clusters))

    def test_disk_content_must_match_sidecar_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            write_snapshot(Path(temp)/'new',self.meta,self.arrays)
            result=audit_snapshot(Path(temp)/'new/evidence.json')
            self.assertEqual(result['status'],'passed')
            with (Path(temp)/'new/arrays.npz').open('ab') as f:
                f.write(b'altered')
            with self.assertRaisesRegex(ValueError,'sidecar'):
                audit_snapshot(Path(temp)/'new/evidence.json')

    def test_array_mutation_without_rehash_fails(self):
        key=self.meta['pair']['q'];self.arrays[key][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'array content'):
            validate_arrays(self.meta,self.arrays)

    def test_semantic_mismatch_fails_even_if_rehashed(self):
        c=self.meta['clusters'][0]
        key=c['readout']['support_weights_a']
        self.arrays[key]=self.arrays[key]*2+.01
        self.meta['arrays'][key]['sha256']=hashlib.sha256(self.arrays[key].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'final score support'):
            validate_arrays(self.meta,self.arrays)

    def test_final_position_and_threshold_cannot_be_relabelled(self):
        meta=copy.deepcopy(self.meta);meta['translation_a_to_b_rc'][0]+=1
        with self.assertRaisesRegex(ValueError,'selected pose'):
            validate_arrays(meta,self.arrays)
        meta=copy.deepcopy(self.meta);meta['accepted']=not meta['accepted']
        with self.assertRaisesRegex(ValueError,'acceptance'):
            validate_arrays(meta,self.arrays)

    def test_localizer_cannot_use_final_pass_instead_of_actual_initial(self):
        c=self.meta['clusters'][0];key=c['refinement']['localization_weights']
        self.arrays[key]=self.arrays[key]*.5
        self.meta['arrays'][key]['sha256']=hashlib.sha256(self.arrays[key].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'initial localization weights'):
            validate_arrays(self.meta,self.arrays)

    def test_invalid_empty_no_fabricated_layout(self):
        q=torch.zeros(5,5);q[0,0]=float('nan')
        pair=replace(fixture(q),numeric_valid=False)
        with torch.no_grad():
            pred=self.model.score_pair(pair,capture_diagnostics=True)
            meta,arrays=snapshot_prediction('invalid',pair,pred,threshold=.5,
                                             provenance={'purpose':'CPU invalid handling'})
        self.assertFalse(validate_arrays(meta,arrays)['has_candidate'])


if __name__=='__main__':
    unittest.main()
