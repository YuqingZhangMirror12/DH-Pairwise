"""CPU synthetic endpoint bindings, no checkpoint corpus or held-out inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import evaluate as e
from staging.pairwise_v0_2.models.rachel_decoupled_score import build_decoupled_score_model
from staging.pairwise_v0_2.models.rachel_n512 import RachelN512Config


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, arm='G0', epoch=16):
        torch.manual_seed(270928)
        config = RachelN512Config(canvas_size=64, coarse_size=32, contour_cap=8,
            feature_dim=16, landmark_count=4, context_layers=1, sinkhorn_iterations=30,
            window_sizes_px=(7.,16.,32.,64.), activation_checkpointing=False)
        self.source = build_decoupled_score_model(config, 'cross_attention', phase='matcher',
                                                  model_options={'cross_attention_depth':1})
        model = e.architecture.ScorerFeatureAdaptationModel(self.source, feature_trainable=arm == 'G1')
        negatives = [2]*88 + [1]*1382
        ident = dict(schema=e.train.SCHEMA, arm=arm, source_checkpoint_sha256=e.source_cache.SOURCE_SHA,
            source_matcher_epochs=12, head_epochs=16, ordinary_pairs_per_epoch=24000, ordinary_physical_batch=16,
            auxiliary_groups_per_epoch=1470, auxiliary_pairs_per_epoch=3028, auxiliary_bce=False,
            real_ood_used=False, negative_counts=negatives, implementation_sha256={'unit':'binding'},
            frozen_digests=e.train.frozen_digests(model), model=model.metadata(),
            head_seed=e.train.HEAD_SEED, data_seed=e.train.DATA_SEED)
        saved = dict(schema=e.train.SCHEMA, identity=ident, completed_segments=4*epoch,
            exposures=e.train.exposure_ledger(4*epoch,negatives), role='epoch_anchor',
            matcher_updated=False, phase='classifier', formal_training_counted=True,
            model_metadata=model.metadata(), model_state_dict=model.state_dict())
        checkpoint = root/('head_epoch_%03d.pt'%epoch)
        torch.save(saved,checkpoint)
        operating = dict(thresholds=dict(max_f1=.65,recall_95=.25))
        validation = root/('validation_head_%03d.json'%epoch)
        e.train.save(validation,dict(sample_count=3000,positive_count=1500,operating_points=operating))
        chosen = dict(head_epoch=epoch,checkpoint=checkpoint.name,checkpoint_sha256=e.cache_data.sha(checkpoint),
            primary_pair_threshold=.65,operating_points=operating,
            validation=validation.name,validation_sha256=e.cache_data.sha(validation))
        freeze = dict(schema=e.train.SCHEMA,status='complete_endpoint',identity=ident,
            budget_head_epochs=epoch,primary_selection='fixed_endpoint',selection_population='clean SIMVAL3000 only',
            real_ood_used=False,selections=dict(fixed_endpoint=chosen,max_f1=deepcopy(chosen),
                                               recall95=dict(chosen,primary_pair_threshold=.25)))
        (root/'freezes').mkdir()
        e.train.save(root/'freezes'/('c%d.json'%epoch),freeze)
        e.train.save(root/'status.json',dict(status='complete',completed_segments=64,completed_head_epochs=16,
            identity=ident,exposures=e.train.exposure_ledger(64,negatives)))
        return freeze

    def load(self, root, selection='fixed_epoch', budget=16):
        with patch.object(e.train,'implementation_binding',return_value={'unit':'binding'}), \
             patch.object(e.source_cache,'source_checkpoint',return_value={'unit':'S7M12'}) as source, \
             patch.object(e.train.source_training,'load_decoupled_checkpoint',return_value=self.source) as loader:
            model, receipt = e.load_frozen_model(root,selection,budget=budget)
            source.assert_called_once_with()
            loader.assert_called_once_with({'unit':'S7M12'})
            return model,receipt

    def test_both_arms_restore_frozen_source_and_simval_operating_points(self):
        for arm in ('G0','G1'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.fixture(root,arm)
                model,receipt=self.load(root)
                self.assertEqual(model.arm,arm)
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertEqual(e.train.state_digest(model.base_model),e.train.state_digest(self.source.base_model))
                self.assertEqual(receipt['classifier_thresholds']['fused'],.65)
                self.assertEqual(receipt['operating_points']['thresholds']['recall_95'],.25)
                self.assertEqual(receipt['exposures']['pair_forwards'],27028*16)
                self.assertFalse(receipt['classifier_only_pair_bce'])
                self.assertFalse(receipt['test_or_real_used_for_fit'])

    def test_c8_retained_endpoint_requires_full_c16_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,epoch=8)
            _,receipt=self.load(root,budget=8)
            self.assertEqual((receipt['epoch'],receipt['budget']),(20,20))
            self.assertEqual(receipt['exposures']['ordinary_pairs'],24000*8)
            e.train.save(root/'status.json',dict(status='paused',completed_segments=32))
            with self.assertRaisesRegex(ValueError,'completed'):
                self.load(root,budget=8)

    def test_recall_selection_keeps_both_frozen_operating_points(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            _,receipt=self.load(root,selection='recall95')
            self.assertEqual(receipt['winner_record']['primary_pair_threshold'],.25)
            self.assertEqual(receipt['classifier_thresholds']['fused'],.65)

    def test_secondary_threshold_report_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            e.train.save(root/'validation_head_016.json',dict(sample_count=3000,positive_count=1500,
                operating_points=dict(thresholds=dict(max_f1=.65,recall_95=.10))))
            with self.assertRaisesRegex(ValueError,'report hash'):
                self.load(root,selection='max_f1')

    def test_replaced_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            torch.save({'changed':'checkpoint'},root/'head_epoch_016.pt')
            with self.assertRaisesRegex(ValueError,'hash'):
                self.load(root)

    def test_frozen_matcher_mutation_rejected_even_if_checkpoint_hash_rebound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);freeze=self.fixture(root,arm='G1')
            checkpoint=root/'head_epoch_016.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            name=next(name for name,value in saved['model_state_dict'].items()
                      if name.startswith('base_model.') and value.is_floating_point() and value.numel())
            saved['model_state_dict'][name]=saved['model_state_dict'][name]+.1
            torch.save(saved,checkpoint)
            freeze['selections']['fixed_endpoint']['checkpoint_sha256']=e.cache_data.sha(checkpoint)
            e.train.save(root/'freezes/c16.json',freeze)
            with self.assertRaisesRegex(ValueError,'changed frozen'):
                self.load(root)

    def test_auxiliary_exposure_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            status=e.json.loads((root/'status.json').read_text())
            status['exposures']['auxiliary_pairs']=0
            e.train.save(root/'status.json',status)
            with self.assertRaisesRegex(ValueError,'exposure ledger'):
                self.load(root)

    def test_one_matcher_forward_preserves_assignment_and_exposes_raw_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,arm='G1')
            model,_=self.load(root)
            mask=torch.zeros(1,1,64,64);mask[:,:,12:40,12:40]=1
            points=torch.tensor([[[12.,12.],[12.,25.],[12.,39.],[25.,39.],
                                  [39.,39.],[39.,25.],[39.,12.],[25.,12.]]])
            valid=torch.ones(1,8,dtype=torch.bool)
            inputs=(mask,mask.flip(-1),points,points.flip(1),valid,valid)
            original=model.base_model(*inputs)
            with patch.object(model.base_model,'forward',return_value=original) as forward:
                output=model(*inputs)
            self.assertEqual(forward.call_count,1)
            self.assertIs(output.assignment,original.assignment)
            self.assertIs(output.transport,original.transport)
            self.assertIs(output.token_features_a,original.token_features_a)
            self.assertTrue(torch.equal(output.score_details['raw_similarity'],output.raw_similarity))
            self.assertTrue(torch.equal(output.fused_logit,output.score_details['final_logit']))


if __name__=='__main__':
    unittest.main()
