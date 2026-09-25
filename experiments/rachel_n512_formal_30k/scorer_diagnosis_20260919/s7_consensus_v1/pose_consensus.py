"""Explicit C04 merge-repair entrypoint in an isolated source snapshot.

Both training and inference use this builder. Revision and policy are bound to
the experiment; old proposal caches/checkpoints cannot silently resume.
"""
from .legacy_pose_consensus import (
    ConsensusProposals, EdgeCloud, PoseCluster, ProposalConfig, _cloud, _pose_key)
from .pose_consensus_repair import (
    MergeRepairPolicy, MergedPoseCluster, RepairedPoseConsensusBuilder)

PoseConsensusBuilder = RepairedPoseConsensusBuilder
REVISION = 'common-pose-merge-repair/1'
