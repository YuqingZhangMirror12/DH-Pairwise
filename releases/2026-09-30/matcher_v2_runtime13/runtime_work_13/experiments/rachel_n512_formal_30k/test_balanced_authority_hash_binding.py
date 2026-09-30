from experiments.rachel_n512_formal_30k import paired_cluster_bootstrap
from staging.pairwise_v0_2.baselines import rachel_n512_real_external


FROZEN_CONSTRUCTED_SELECTION_SHA256 = (
    "ba469ad0bbc1f9f61c8e74bf6159b8846c1e37853724379e5cfaa1b42d60db6b"
)


def test_frozen_constructed_selection_authority_is_shared_by_build_and_analysis():
    assert (
        rachel_n512_real_external.EXPECTED_CONSTRUCTED_SELECTION_SHA256
        == FROZEN_CONSTRUCTED_SELECTION_SHA256
    )
    assert (
        paired_cluster_bootstrap.EXPECTED_CONSTRUCTED_SELECTION_SHA256
        == FROZEN_CONSTRUCTED_SELECTION_SHA256
    )
