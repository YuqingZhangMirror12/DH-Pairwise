"""Threshold/union variant; isolated queued experiment, never the running code.

16px is the predeclared first control, now a cluster DIAMETER; old mean-shift
radius results do not validate this distinct interpretation. No GT admission.
"""
from .legacy_pose_consensus import (
    ConsensusProposals, EdgeCloud, PoseCluster, ProposalConfig, _cloud, _pose_key)
from .threshold_builder import ThresholdPolicy, ThresholdCluster, ThresholdPoseBuilder

PoseConsensusBuilder = ThresholdPoseBuilder
REVISION = 'native-hypothesis-complete-link-union/1-diameter16'
