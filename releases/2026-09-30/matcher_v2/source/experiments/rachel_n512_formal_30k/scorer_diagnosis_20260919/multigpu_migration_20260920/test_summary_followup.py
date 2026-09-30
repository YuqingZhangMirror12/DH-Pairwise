import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import summary_followup as s


class TestSummaryFollowup(unittest.TestCase):
    def test_only_cpu_summary(self):
        stage=dict(name='summarize_new_seven_arms',command=['python','-m','matched_only.summarize_priority'])
        self.assertEqual(s.source_stage(dict(stages=[stage])),stage)
        stage['command'][2]='matched_only.train'
        with self.assertRaises(ValueError):s.source_stage(dict(stages=[stage]))

    def test_wait_and_complete(self):
        with patch.object(s,'read',return_value=dict(status='running')):
            self.assertFalse(s.readiness('/does-not-exist'))
        with patch.object(s,'read',return_value=dict(status='complete')):
            self.assertTrue(s.readiness('/does-not-exist'))

    def test_failed_producer_is_not_missing_result(self):
        with patch.object(s,'read',return_value=dict(status='failed')):
            with self.assertRaises(RuntimeError):s.readiness('/does-not-exist')


if __name__=='__main__':unittest.main()
