from copy import deepcopy

import pytest

from scripts.diagnose.compare_kl8_main_plays import candidate_slate, compare


def history():
    return [{'issue': str(2026000 + i), 'numbers': list(range(1, 21))} for i in range(200)]


def test_slate_does_not_modify_incumbent_or_ticket_count():
    incumbent = {'feature_weights': {'frequency': .4, 'next_transition': .3, 'pair_cooccurrence': .1},
                 'model_weights': {'rank': 1}, 'window_size': 100}
    before = deepcopy(incumbent)
    slate = candidate_slate(incumbent)
    assert incumbent == before
    assert len(slate) == 4
    assert slate['without_sparse_transition_pair']['feature_weights']['frequency'] == .4
    assert slate['without_sparse_transition_pair']['feature_weights']['next_transition'] == 0


def test_only_prior_draws_enter_prediction_and_later_results_cannot_choose_winner():
    data = history()
    before = deepcopy(data)
    candidates = {'current': {'name': 'current'}, 'challenger': {'name': 'challenger'}}
    def predictor(past, strategy):
        assert len(past) < len(data)
        assert past[-1]['issue'] == data[len(past) - 1]['issue']
        six = [1, 2, 3, 41, 42, 43]
        if strategy['name'] == 'challenger':
            six = [1, 2, 3, 4, 5, 6] if len(past) < 150 else [41, 42, 43, 44, 45, 46]
        return {'select_6': six, 'fu_shi_7': six + [47]}
    report = compare(data, candidates, predictor=predictor)
    assert report['locked_earlier_winner'] == 'challenger'
    assert report['development_later_check']['gate']['passed'] is False
    assert report['promotion_allowed'] is False
    assert data == before


def test_missing_or_repeated_target_draws_are_not_a_valid_comparison():
    data = history()
    data[-1] = data[-2]
    with pytest.raises(ValueError, match='unique'):
        compare(data, {'current': {}})
