import tempfile
import unittest
from unittest.mock import Mock

import torch

from .compatibility import CompatibilityConfig
from .config import TrainingConfig
from .metrics import average_precision,choose_threshold,summarize
from .pose_consensus import PoseConsensusBuilder
from .proposal_cache import ProposalCache
from .test_evidence import fixture


def row(i,label,score,good=False,covered=False,known=True,has=True):
    return dict(pair_id=str(i),label=label,score=score,layout20=good,candidate_coverage=covered,
                gt_known=known and label,numeric_valid=True,has_candidate=has)


class MetricsCacheTests(unittest.TestCase):
    def test_pair_success_and_layout_success_are_separate(self):
        rows=[row(0,True,.8,True,True),row(1,True,.9,False,True),row(2,True,.1,True,True),
              row(3,True,.1,False,False),row(4,False,.7),row(5,False,.1)]
        m=summarize(rows,.5)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(2,1,2))
        self.assertEqual((m['joint_tp'],m['joint_fp'],m['joint_fn']),(1,2,3))
        self.assertEqual((m['covered_but_winner_wrong'],m['winner_correct_but_rejected'],m['positive_no_correct_candidate']),(1,1,1))
        self.assertAlmostEqual(m['layout20'],.5)

    def test_unknown_real_gt_has_no_fake_layout_accuracy(self):
        m=summarize([row(0,True,.7,known=False),row(1,False,.2)],.3)
        self.assertEqual(m['f1'],1.)
        self.assertIsNone(m['layout20']);self.assertIsNone(m['joint_f1'])

    def test_no_candidate_is_rejected_even_if_saved_score_high(self):
        m=summarize([row(0,False,.9,has=False),row(1,True,.1,has=False)],.3)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(0,0,1))

    def test_threshold_tie_prefers_point3_not_an_extreme(self):
        rows=[row(0,True,.9,True,True),row(1,False,.1)]
        t=choose_threshold(dict(clean=rows,hard=rows),TrainingConfig())
        self.assertAlmostEqual(t,.3)

    def test_ap_groups_tied_scores(self):
        self.assertAlmostEqual(average_precision([True,False],[.5,.5]),.5)

    def test_cache_roundtrip_keeps_union_and_rejects_other_matcher(self):
        pair=fixture();builder=PoseConsensusBuilder(CompatibilityConfig(.5,.5,.5,.5,1.,30.))
        with tempfile.TemporaryDirectory() as directory:
            cache=ProposalCache(directory,dict(matcher_sha='a',data_sha='b'))
            a=cache.get('pair',builder,pair)
            no_call=Mock(side_effect=AssertionError('should use frozen bound proposals'))
            b=cache.get('pair',no_call,pair)
            torch.testing.assert_close(a.seeds,b.seeds)
            for ca,cb in zip(a.clusters,b.clusters):
                torch.testing.assert_close(ca.translation,cb.translation)
                torch.testing.assert_close(ca.edge_ids,cb.edge_ids)
            self.assertEqual((cache.hits,cache.misses),(1,1))
            with self.assertRaises(ValueError):
                ProposalCache(directory,dict(matcher_sha='c',data_sha='b'))


if __name__=='__main__':
    unittest.main()
