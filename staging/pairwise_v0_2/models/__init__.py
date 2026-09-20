"""Learnable-model primitives for the Pairwise v0.2 experiments."""

from .coarse import CoarseOutput, SymmetricCoarseSiamese
from .frozen_transport_matrix_readout import (
    FrozenTransportMatrixReadout,
    StructuredTransportReadoutOutput,
    TRANSPORT_MATRIX_CHANNELS,
)
from .local_matcher import (
    DEFAULT_MATCHER_TEMPERATURE,
    DEFAULT_SINKHORN_ITERATIONS,
    LocalMatcherOutput,
    MatcherMode,
    OrderedLocalMatcher,
    OrderedSelfCrossContext,
    SharedPatchEncoder,
    SinkhornTrainingPolicy,
)
from .optimal_transport import (
    PartialTransportOutput,
    SinkhornDiagnostics,
    dustbin_sinkhorn,
)
from .pairwise import (
    ArcPoolingConfig,
    ArcPoolingMode,
    CoarseModelConfig,
    DEFAULT_DIRECTIONS,
    DirectionalPairwiseOutput,
    DunhuangPairwiseV02,
    HierarchicalDirectionalOutput,
    LocalModelConfig,
    PairwiseModelConfig,
    PairwiseOutput,
    aggregate_direction_candidates,
    aggregate_flat_arc_candidates,
)
from .rachel_n512 import (
    ContourPatchSampler,
    CyclicLandmarkContext,
    RachelN512Config,
    RachelN512Output,
    RachelN512Pairwise,
    TransportSequenceHead,
)

__all__ = [
    "ArcPoolingConfig",
    "ArcPoolingMode",
    "CoarseModelConfig",
    "CoarseOutput",
    "ContourPatchSampler",
    "CyclicLandmarkContext",
    "DEFAULT_DIRECTIONS",
    "DirectionalPairwiseOutput",
    "DunhuangPairwiseV02",
    "DEFAULT_MATCHER_TEMPERATURE",
    "DEFAULT_SINKHORN_ITERATIONS",
    "HierarchicalDirectionalOutput",
    "FrozenTransportMatrixReadout",
    "LocalMatcherOutput",
    "LocalModelConfig",
    "MatcherMode",
    "OrderedLocalMatcher",
    "OrderedSelfCrossContext",
    "PartialTransportOutput",
    "PairwiseModelConfig",
    "PairwiseOutput",
    "RachelN512Config",
    "RachelN512Output",
    "RachelN512Pairwise",
    "SharedPatchEncoder",
    "SinkhornTrainingPolicy",
    "SinkhornDiagnostics",
    "StructuredTransportReadoutOutput",
    "SymmetricCoarseSiamese",
    "TRANSPORT_MATRIX_CHANNELS",
    "TransportSequenceHead",
    "aggregate_direction_candidates",
    "aggregate_flat_arc_candidates",
    "dustbin_sinkhorn",
]
