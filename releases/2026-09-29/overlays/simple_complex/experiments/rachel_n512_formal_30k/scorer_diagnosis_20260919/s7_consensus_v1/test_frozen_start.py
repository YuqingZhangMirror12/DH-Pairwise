import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from .frozen_start import load_selected_matcher
from .config import TrainingConfig
from .model import S7Consensus
from .compatibility import CompatibilityConfig
from .simple_builder import SimplePoseBuilder


class Adapter(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Linear(3,4)
    def set_frozen(self,value):
        self.frozen=value;self.requires_grad_(not value)


class FrozenStartTests(unittest.TestCase):
    def checkpoint(self,path,stage='matcher'):
        source=Adapter()
        torch.save(dict(stage=stage,epoch=32,updates=24000,exposures=768000,
            model={**{'matcher.'+k:v for k,v in source.state_dict().items()},
                'head.unused':torch.ones(2)}),path)
        return source,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_exact_matcher_only_frozen_import(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt';source,sha=self.checkpoint(path);target=Adapter()
            result=load_selected_matcher(target,path,sha)
            for k,v in source.state_dict().items():self.assertTrue(torch.equal(v,target.state_dict()[k]))
            self.assertTrue(all(not p.requires_grad for p in target.parameters()))
            self.assertFalse(result['old_head_imported']);self.assertEqual(result['epoch'],32)

    def test_hash_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt';self.checkpoint(path)
            with self.assertRaises(ValueError):load_selected_matcher(Adapter(),path,'0'*64)

    def test_cannot_import_trained_scorer_as_matcher_selection(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt';_,sha=self.checkpoint(path,'scorer')
            with self.assertRaises(ValueError):load_selected_matcher(Adapter(),path,sha)

    def test_model_and_config_use_selected_simple_revision(self):
        c=TrainingConfig().record()
        self.assertEqual(c['effective_batch'],32)
        self.assertNotIn('merge_repair_policy',c)
        self.assertEqual(c['simple_policy']['pose_radius_px'],16)
        model=S7Consensus(None,CompatibilityConfig(.5,.5,.5,.5,1.,9.))
        self.assertIsInstance(model.builder,SimplePoseBuilder)


if __name__=='__main__':unittest.main()
