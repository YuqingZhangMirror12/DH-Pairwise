import pytest

from experiments.rachel_n512_formal_30k.prepare_score_historical_references import (
    compact, identities, positive_only,
)


def test_positive_only_has_no_invented_binary_or_layout_metrics():
    rows = [dict(pair_id='a', score=.5, decision_valid=True),
            dict(pair_id='b', score=.9, decision_valid=False),
            dict(pair_id='c', score=.1, decision_valid=True)]
    out = positive_only(rows, .5)
    c = out['classification']
    assert (c['tp'], c['fn'], c['recall']) == (1, 2, 1/3)
    assert out['decision_valid_count'] == 2
    assert out['positive_only_layout'] is None
    assert all(c[key] is None for key in ('fp', 'tn', 'precision', 'f1', 'accuracy', 'ap', 'auroc'))


@pytest.mark.parametrize('rows,threshold', [([], .5),
    ([dict(score=float('nan'), decision_valid=True)], .5),
    ([dict(score=.5, decision_valid=1)], .5),
    ([dict(score=.5, decision_valid=True)], float('inf'))])
def test_positive_only_rejects_undefined_input(rows, threshold):
    with pytest.raises(ValueError):
        positive_only(rows, threshold)


def test_population_join_preserves_endpoints_and_gt_and_rejects_duplicates():
    row = dict(pair_id='a', label=True, fragment_a='A', fragment_b='B', target_translation_rc=[1,2])
    assert identities([row]) != identities([dict(row, target_translation_rc=[2,1])])
    assert identities([row]) != identities([dict(row, fragment_a='B', fragment_b='A')])
    with pytest.raises(ValueError):
        identities([row, row])
    assert identities([dict(pair_id='a')], False) == {'a': ['a']}


def test_compact_removes_id_lists_without_removing_denominators():
    assert compact({'classification': {'tp': 2, 'true_positive_pair_ids': ['a','b']},
                    'positive_count': 4}) == {'classification': {'tp': 2}, 'positive_count': 4}
