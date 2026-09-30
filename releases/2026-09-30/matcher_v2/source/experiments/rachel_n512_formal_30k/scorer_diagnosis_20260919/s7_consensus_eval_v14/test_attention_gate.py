import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import attention_gate


class AttentionGateTests(unittest.TestCase):
    def test_cpu_full_gate_uses_production_width_and_restores_rng(self):
        rng=torch.get_rng_state().clone()
        result=attention_gate.run('cpu')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(result['status'],'passed')
        self.assertEqual(len(result['operators']),5)
        self.assertEqual([x['attention_calls'] for x in result['synthetic_model_cases']],[16,0])
        self.assertFalse(result['trained_checkpoint_or_dataset_opened'])
        self.assertEqual(result['synthetic_model_cases'][0]['production_head_feature_dim'],96)
        self.assertFalse(result['numeric_settings_executed']['tf32_matmul'])
        self.assertFalse(result['numeric_settings_executed']['tf32_cudnn'])
        self.assertTrue(result['numeric_settings_executed']['deterministic_algorithms'])
        json.dumps(result,allow_nan=False)

    def test_numeric_protocol_restored_on_exception(self):
        before=attention_gate.numeric_settings()
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with attention_gate.numeric_protocol() as settings:
                self.assertFalse(settings['tf32_matmul'])
                self.assertTrue(settings['deterministic_algorithms'])
                raise RuntimeError('synthetic')
        self.assertEqual(before,attention_gate.numeric_settings())

    def test_device_must_be_explicit_supported_backend(self):
        for device in ('cuda','mps','meta'):
            with self.subTest(device=device),self.assertRaisesRegex(ValueError,'explicitly indexed'):
                attention_gate.run(device)

    def test_unavailable_cuda_must_not_silently_fall_back(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'unavailable'):
                attention_gate.run('cuda:0')

    def test_existing_receipt_rejected_before_any_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'receipt.json';out.touch()
            with patch.object(attention_gate,'run') as run:
                with self.assertRaisesRegex(FileExistsError,'overwrite'):
                    attention_gate.main(['--device','cpu','--out',str(out)])
                run.assert_not_called()


if __name__=='__main__': unittest.main()
