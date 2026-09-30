"""Frozen endpoint-loader tests; synthetic local heads, no real inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import evaluate as e


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, arm='matched_edges', epoch=16):
        head = e.train.make_scorer(arm, seed=e.train.HEAD_SEED)
        ident = dict(source_checkpoint_sha256=e.cache.SOURCE_SHA, matcher_frozen=True, arm=arm,
            classifier_epochs=16, training_count=24000, validation_count=3000, real_ood_used=False,
            implementation_sha256={'unit':'binding'}, head_seed=e.train.HEAD_SEED, data_seed=e.train.DATA_SEED)
        saved = dict(schema=e.train.SCHEMA, identity=ident, head_epoch=epoch, completed_segments=4*epoch,
            classifier_pair_exposures=24000*epoch, optimizer_updates=1500*epoch,
            matcher_updated=False, phase='classifier', formal_training_counted=True,
            model_metadata=head.metadata(), model_state_dict=head.state_dict())
        cp = root / ('head_epoch_%03d.pt' % epoch)
        torch.save(saved, cp)
        operating = dict(thresholds=dict(max_f1=.65,recall_95=.25))
        validation_path=root/('validation_head_%03d.json'%epoch)
        e.train.save(validation_path,dict(sample_count=3000,positive_count=1500,operating_points=operating))
        chosen = dict(head_epoch=epoch,checkpoint=cp.name,checkpoint_sha256=e.data.sha(cp),
                      primary_pair_threshold=.65,operating_points=operating,
                      validation=validation_path.name,validation_sha256=e.data.sha(validation_path))
        freeze = dict(schema=e.train.SCHEMA,status='complete_endpoint',budget_head_epochs=epoch,
            real_ood_used=False, selection_population='clean SIMVAL3000 only',
            primary_selection='fixed_endpoint', identity=ident,
            selections=dict(fixed_endpoint=chosen,max_f1=deepcopy(chosen),recall95=dict(chosen,primary_pair_threshold=.25)))
        (root/'freezes').mkdir()
        e.train.save(root/'freezes'/('c%d.json'%epoch),freeze)
        e.train.save(root/'status.json',dict(status='complete',completed_segments=64))
        return freeze

    def load(self,root,selection='fixed_epoch',budget=16):
        from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config
        base=SimpleNamespace(config=RachelN512Config())
        adapter=SimpleNamespace(metadata=lambda: {'unit':'adapter'})
        with patch.object(e.train,'implementation_binding',return_value={'unit':'binding'}), \
                patch.object(e.cache,'source_checkpoint',return_value={'unit':'originalS7M12'}) as source, \
                patch.object(e.cache.old,'load_decoupled_checkpoint',return_value=SimpleNamespace(base_model=base)), \
                patch.object(e.inference,'FrozenMatchedInference',return_value=adapter) as wrapper:
            model,receipt=e.load_frozen_model(root,selection,budget=budget)
            source.assert_called_once_with()
            self.assertIs(wrapper.call_args.args[0],base)
            return model,receipt

    def test_all_five_heads_load_same_simval_threshold_contract(self):
        for arm in e.train.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.fixture(root,arm)
                _,r=self.load(root)
                self.assertEqual(r['head_epoch'],16)
                self.assertEqual(r['head_budget'],16)
                self.assertEqual(r['classifier_thresholds']['fused'],.65)
                self.assertEqual(r['operating_points']['thresholds']['recall_95'],.25)
                self.assertFalse(r['test_or_real_used_for_fit'])

    def test_c8_is_separate_budget_only_after_c16_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,epoch=8)
            _,r=self.load(root,budget=8)
            self.assertEqual((r['epoch'],r['budget']),(20,20))
            e.train.save(root/'status.json',dict(status='running',completed_segments=32))
            with self.assertRaisesRegex(ValueError,'completed'):
                self.load(root,budget=8)

    def test_recall_selection_keeps_both_operating_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            _,r=self.load(root,selection='recall95')
            self.assertEqual(r['winner_record']['primary_pair_threshold'],.25)
            self.assertEqual(r['classifier_thresholds']['fused'],.65)

    def test_wrong_source_and_threshold_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=self.fixture(root)
            path=root/'freezes/c16.json'
            f['identity']['source_checkpoint_sha256']='wrong'
            e.train.save(path,f)
            with self.assertRaisesRegex(ValueError,'completed'):
                self.load(root)
            f['identity']['source_checkpoint_sha256']=e.cache.SOURCE_SHA
            f['selections']['fixed_endpoint']['primary_pair_threshold']=.1
            e.train.save(path,f)
            with self.assertRaisesRegex(ValueError,'operating point'):
                self.load(root)

    def test_checkpoint_after_freeze_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            torch.save({'not':'same checkpoint'},root/'head_epoch_016.pt')
            with self.assertRaisesRegex(ValueError,'hash'):
                self.load(root)

    def test_auxiliary_secondary_threshold_is_also_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            report=dict(sample_count=3000,positive_count=1500,
                operating_points=dict(thresholds=dict(max_f1=.65,recall_95=.1)))
            e.train.save(root/'validation_head_016.json',report)
            with self.assertRaisesRegex(ValueError,'report hash'):
                self.load(root,selection='max_f1')


if __name__=='__main__':
    unittest.main()
