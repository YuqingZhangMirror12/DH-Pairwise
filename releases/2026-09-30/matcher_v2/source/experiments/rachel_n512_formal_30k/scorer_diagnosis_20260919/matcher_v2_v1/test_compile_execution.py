from pathlib import Path
import unittest
from unittest.mock import patch

from ..curriculum_training_v1.checkpoint_io import write_json
from ..curriculum_training_v1.runtime_io import read
from . import compile_execution as api
from . import test_runtime_inputs as fixtures


class CompilationTests(unittest.TestCase):
    def test_compile_B2_full_original_budget_then_independently_reload(self):
        fixture = fixtures.RuntimeInputsTests(); fixture.setUp()
        try:
            base = fixture.root/'base_execution.json'; write_json(base, fixture.base)
            source = Path(__file__).resolve().parents[4]
            with patch.object(api, 'verify_composed_source', return_value=(source, fixture.original.record['baseline_sources_sha256'])), \
                 patch.object(api, 'load_inputs', side_effect=fixture.load):
                spec = api.compile_execution(base, 'B2', 'matcher', fixture.root/'compiled')
            self.assertEqual(read(spec['runtime_plan']['path'])['total_updates'], 24000)
            self.assertEqual(read(spec['runtime_schedule']['path'])['added_straight_updates'], 0)
            self.assertIsNone(spec['selected_matcher'])
            with self.assertRaisesRegex(ValueError, 'new locked-plan'):
                api.compile_execution(base, 'B2', 'matcher', fixture.root/'compiled')
        finally:fixture.doCleanups()

    def test_head_needs_completed_own_arm_Matcher_before_any_files(self):
        with self.assertRaisesRegex(ValueError, 'completed Matcher'):
            api.compile_execution('/does-not-exist', 'B3', 'scorer_patch', '/unused')
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            api.compile_execution('/does-not-exist', 'B4', 'matcher', '/unused')


if __name__ == '__main__':unittest.main()
