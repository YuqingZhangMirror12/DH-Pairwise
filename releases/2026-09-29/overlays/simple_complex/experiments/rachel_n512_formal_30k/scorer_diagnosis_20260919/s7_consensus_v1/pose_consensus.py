"""Simplified displacement-mode aggregation; isolated future comparison.

16px selected on frozen SIM SELECT. Native-hypothesis gates passed, but
budget-loss and raw-edge mixing acceptance remain unmet: not a production
replacement claim. The running mergefix experiment does NOT use this source.
"""
from .legacy_pose_consensus import (
    ConsensusProposals, EdgeCloud, PoseCluster, ProposalConfig, _cloud, _pose_key)
from .simple_builder import SimplePolicy, SimpleCluster, SimplePoseBuilder

PoseConsensusBuilder = SimplePoseBuilder
REVISION = 'raw-displacement-modes-simple/1-sim16'
