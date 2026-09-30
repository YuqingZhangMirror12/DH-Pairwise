"""Patch/Context light head with one explicit residual cluster block.

Everything before cluster_mlp uses the existing BinaryClusterHead.forward:
exact union, online Q, edge encoder, Q*arc mean, max and absolute statistics.
The fixed16 builder, Matcher, labels and loss are not changed here.
"""
from torch import nn

from ..binary_scorer_v1.head import BinaryClusterHead, SCALAR_NAMES


ARCHITECTURE = 'patch-context-cluster-residual/1-width64-bottleneck32'
MODULE = 'scorer_patch_residual'


class ResidualBlock(nn.Module):
    """One bottleneck F(h)=Linear32to64(GELU(Linear64to32(h)))."""

    def __init__(self):
        super().__init__()
        self.down = nn.Linear(64, 32)
        self.activation = nn.GELU()
        self.up = nn.Linear(32, 64)

    def forward(self, h):
        return h + self.up(self.activation(self.down(h)))


class ResidualClusterHead(BinaryClusterHead):
    """New head; never imports a trained baseline head or its optimizer.

The feature *variant* remains patch for the shared evidence path. The distinct
architecture and module IDs must be recorded by a future runtime/evaluator;
it is not a compatible replacement for a saved binary_patch state dict.
    """
    architecture = ARCHITECTURE
    experiment_module = MODULE

    def __init__(self, feature_dim=96):
        nn.Module.__init__(self)
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError('positive integer feature width required')
        self.variant = 'patch'
        self.feature_dim = feature_dim
        # Same initialization order/shapes as the baseline through the down
        # projection. The new up and output layers are freshly initialized.
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim + 8, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU())
        self.cluster_mlp = nn.Sequential(
            nn.Linear(len(SCALAR_NAMES) + 64, 64), nn.GELU(),
            ResidualBlock(), nn.Linear(64, 1))


def architecture_record(head):
    if type(head) is not ResidualClusterHead:
        raise ValueError('explicit residual head required')
    return dict(architecture=ARCHITECTURE, module=MODULE, feature_variant='patch',
        feature_dim=head.feature_dim, parameter_count=sum(p.numel() for p in head.parameters()),
        cluster_network='80 -> 64 -> (h + Linear64(GELU(Linear32(h)))) -> 1',
        activation_after_addition=False, batch_norm=False, attention=False,
        local_conflict_classifier=False, fixed_conflict_penalty=False,
        learned_pose_refinement=False, initialization='fresh head; no head/optimizer import',
        comparison_limit='residual connection, depth and parameter count all change')
