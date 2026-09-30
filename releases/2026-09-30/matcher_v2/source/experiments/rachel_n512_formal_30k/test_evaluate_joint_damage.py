import copy

import pytest

from experiments.rachel_n512_formal_30k.evaluate_joint_damage import validate_freeze


def complete():
    thresholds = dict(coarse=.3, local=.4, fused=.5)
    return dict(status="complete", completed_global_exposures=144000,
        completed_optimizer_updates=9000, completed_validation_events=6,
        unique_count=24000, test_or_real_used_for_fit=False,
        original_validation_unchanged=True, initialization="random",
        initial_weights_sha256="a" * 64, classifier_thresholds=thresholds,
        validation=dict(sample_count=3000, positive_count=1500, negative_count=1500,
            pose_used_for_selection=False, decision_coverage=1.0, thresholds=copy.deepcopy(thresholds)))


def test_accept_new_complete_receipt():
    assert validate_freeze(complete()) == dict(coarse=.3, local=.4, fused=.5)


@pytest.mark.parametrize("key,value", [("status", "provisional"),
    ("completed_global_exposures", 120000), ("completed_optimizer_updates", 7500),
    ("completed_validation_events", 5), ("initialization", "warm-start"),
    ("test_or_real_used_for_fit", True)])
def test_reject_old_or_unfinished_run(key, value):
    data = complete()
    data[key] = value
    with pytest.raises(ValueError):
        validate_freeze(data)


def test_reject_changed_threshold():
    data = complete()
    data["classifier_thresholds"]["fused"] = .6
    with pytest.raises(ValueError):
        validate_freeze(data)
