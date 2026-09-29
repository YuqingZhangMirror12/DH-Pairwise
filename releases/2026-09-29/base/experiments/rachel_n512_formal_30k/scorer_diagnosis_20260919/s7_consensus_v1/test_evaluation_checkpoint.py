import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from .config import TrainingConfig
from .evaluation_checkpoint import save_evaluation_checkpoint
from .model import S7Consensus
from .pose_consensus import PoseConsensusBuilder, REVISION


class EvaluationCheckpointTests(unittest.TestCase):
    def record(self):
        return dict(binding={'fixture':'CPU only'},stage='matcher',epoch=2,updates=1500,exposures=48000,
            threshold=.3,metrics=dict(key=[.8,.7,-1.2],elapsed_seconds=4.),model={'w':torch.tensor([1.,2.])})

    def test_immutable_copy_and_identical_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'epoch_002_weights.pt';r=self.record()
            first=save_evaluation_checkpoint(p,r)
            self.assertFalse(first['already_existed'])
            second=save_evaluation_checkpoint(p,r)
            self.assertTrue(second['already_existed'])
            self.assertEqual(first['sha256'],second['sha256'])

    def test_timing_only_change_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'e.pt';r=self.record();save_evaluation_checkpoint(p,r)
            before=p.read_bytes();r['metrics']['elapsed_seconds']=44.
            save_evaluation_checkpoint(p,r)
            self.assertEqual(before,p.read_bytes())

    def test_different_weights_or_selection_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'e.pt';r=self.record();save_evaluation_checkpoint(p,r);before=p.read_bytes()
            altered=copy.deepcopy(r);altered['model']['w'][0]+=1
            with self.assertRaises(ValueError):save_evaluation_checkpoint(p,altered)
            altered=copy.deepcopy(r);altered['threshold']=.4
            with self.assertRaises(ValueError):save_evaluation_checkpoint(p,altered)
            altered=copy.deepcopy(r);altered['metrics']['key'][0]=.9
            with self.assertRaises(ValueError):save_evaluation_checkpoint(p,altered)
            self.assertEqual(before,p.read_bytes())

    def test_repair_policy_is_part_of_training_binding(self):
        record=TrainingConfig().record()
        self.assertEqual(record['proposal_revision'],REVISION)
        self.assertEqual(record['merge_repair_policy']['maximum_lost_explained_mass_fraction'],.05)
        self.assertEqual(record['effective_batch'],32)
        self.assertEqual(record['head'],dict(layers=2,dim=96,heads=4,length_scale_px=32.))


if __name__=='__main__':
    unittest.main()
