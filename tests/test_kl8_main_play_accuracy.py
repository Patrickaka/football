from copy import deepcopy
from itertools import combinations

import pytest

from src.kl8.main_play_accuracy import evaluate_main_play_pair
from src.kl8.stats import hypergeom_p_ge


def _ticket(size, hits):
    return list(range(1, hits + 1)) + list(range(21, 21 + size - hits))


def _rows(six_hits, pool_hits):
    assert len(six_hits) == len(pool_hits)
    return [{'issue': str(2026200 + index), 'actual_numbers': list(range(1, 21)),
             'tickets': {'select_6': _ticket(6, six), 'fu_shi_7': _ticket(7, pool)}}
            for index, (six, pool) in enumerate(zip(six_hits, pool_hits))]


def test_identical_results_pass_descriptive_gate_without_promotion():
    rows = _rows([0, 3, 4, 5, 6], [0, 3, 4, 5, 7])
    result = evaluate_main_play_pair(rows, list(reversed(rows)))
    assert result['n'] == 5
    assert result['gate']['passed']
    assert not result['gate']['has_improvement']
    assert not result['promotion_allowed']
    assert result['incumbent']['select_6']['mean_hits'] == 18 / 5
    assert result['incumbent']['select_6']['at_least']['6']['rate'] == .2
    assert result['incumbent']['fu_shi_7']['fair_baseline']['mean_hits'] == 1.75


@pytest.mark.parametrize('pool_hits', range(8))
def test_fushi_combination_counts_match_enumeration(pool_hits):
    rows = _rows([2], [pool_hits])
    result = evaluate_main_play_pair(rows, rows)['candidate']['fu_shi_7']
    combos = result['combinations']
    observed = [len(set(ticket) & set(range(1, 21)))
                for ticket in combinations(rows[0]['tickets']['fu_shi_7'], 5)]
    assert combos['hit_distribution'] == {str(k): observed.count(k) for k in range(6)}
    assert sum(combos['hit_distribution'].values()) == 21
    assert combos['full_hit_draws'] == int(pool_hits >= 5)
    assert combos['full_hit_tickets'] == observed.count(5)
    assert combos['at_least']['5']['fair_baseline'] == hypergeom_p_ge(5, 5)


def test_pool_five_hits_is_one_of_twenty_one_full_hit_tickets():
    rows = _rows([4], [5])
    result = evaluate_main_play_pair(rows, rows)['candidate']['fu_shi_7']
    assert result['at_least']['5']['rate'] == 1.0
    assert result['combinations']['at_least']['5']['rate'] == 1 / 21
    assert result['combinations']['full_hit_tickets'] == 1


def test_three_hit_gain_cannot_mask_four_and_five_hit_regression():
    before = _rows([5, 0, 0], [5, 0, 0])
    after = _rows([3, 3, 3], [3, 3, 3])
    result = evaluate_main_play_pair(before, after)
    assert result['comparison']['select_6']['mean_hits']['delta'] > 0
    assert result['comparison']['fu_shi_7']['at_least_3']['delta'] > 0
    assert not result['gate']['passed']
    failures = {(row['play'], row['metric']) for row in result['gate']['failures']}
    assert {('select_6', 'at_least_4'), ('select_6', 'at_least_5'),
            ('fu_shi_7', 'at_least_4'), ('fu_shi_7', 'at_least_5')} <= failures


def test_one_play_cannot_compensate_for_the_other():
    result = evaluate_main_play_pair(_rows([3], [5]), _rows([6], [4]))
    assert not result['gate']['passed']
    assert all(row['play'] == 'fu_shi_7' for row in result['gate']['failures'])


def test_combo_full_hit_regression_cannot_hide_behind_same_pool_five_rate():
    result = evaluate_main_play_pair(_rows([3, 3], [7, 0]), _rows([3, 3], [5, 3]))
    assert result['comparison']['fu_shi_7']['at_least_5']['delta'] == 0
    assert result['comparison']['fu_shi_7']['mean_hits']['delta'] > 0
    assert not result['gate']['passed']
    assert any(row['metric'] == 'combination_at_least_5' for row in result['gate']['failures'])


def test_non_worse_gain_has_higher_ranking_but_is_not_promotion():
    before = _rows([0, 3, 4], [0, 3, 4])
    same = evaluate_main_play_pair(before, before)
    improved = evaluate_main_play_pair(before, _rows([1, 4, 5], [1, 4, 5]))
    assert improved['gate']['passed'] and improved['gate']['has_improvement']
    assert improved['ranking_key'] > same['ranking_key']
    assert not improved['promotion_allowed']


def test_ranking_prioritizes_the_weakest_high_tier_gain_across_both_plays():
    before = _rows([3] * 10, [3] * 10)
    one_play_only = evaluate_main_play_pair(before, _rows([6] * 10, [3] * 10))
    both_plays = evaluate_main_play_pair(before, _rows([5] + [3] * 9, [5] + [3] * 9))
    assert one_play_only['gate']['passed'] and both_plays['gate']['passed']
    assert both_plays['ranking_key'] > one_play_only['ranking_key']


def test_zero_rate_regression_is_reported_even_with_better_high_tiers():
    result = evaluate_main_play_pair(_rows([1, 1], [1, 1]), _rows([0, 6], [0, 7]))
    assert not result['gate']['passed']
    assert {row['metric'] for row in result['gate']['failures']} == {'zero_rate'}


@pytest.mark.parametrize('mutate', [
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(issue=2026200),
    lambda rows: rows[0].update(issue='２０２６２００'),
    lambda rows: rows[0].update(issue='20262000'),
    lambda rows: rows[0].update(actual_numbers=list(range(1, 20))),
    lambda rows: rows[0]['actual_numbers'].__setitem__(0, True),
    lambda rows: rows[0]['actual_numbers'].__setitem__(0, 2),
    lambda rows: rows[0]['tickets'].pop('fu_shi_7'),
    lambda rows: rows[0]['tickets'].update(select_5=[1, 2, 3, 4, 5]),
    lambda rows: rows[0]['tickets'].update(select_6=[1, 2, 3, 4, 5]),
    lambda rows: rows[0]['tickets'].update(fu_shi_7=list(range(1, 9))),
    lambda rows: rows[0]['tickets']['select_6'].__setitem__(0, 81),
    lambda rows: rows[0]['tickets']['fu_shi_7'].__setitem__(0, 2),
    lambda rows: rows[0]['tickets']['select_6'].__setitem__(0, 1.0),
])
def test_rejects_invalid_draws_and_tickets(mutate):
    before = _rows([3], [3])
    after = deepcopy(before)
    mutate(after)
    with pytest.raises(ValueError):
        evaluate_main_play_pair(before, after)


def test_requires_identical_issue_coverage_and_actual_results():
    before = _rows([3, 4], [3, 4])
    with pytest.raises(ValueError, match='same issues'):
        evaluate_main_play_pair(before, before[:1])
    after = deepcopy(before)
    after[0]['actual_numbers'][0] = 80
    with pytest.raises(ValueError, match='actual draw mismatch'):
        evaluate_main_play_pair(before, after)
    with pytest.raises(ValueError):
        evaluate_main_play_pair([], [])


def test_recomputes_hits_without_mutating_inputs():
    rows = _rows([3], [4])
    rows[0]['hits'] = {'select_6': 6, 'fu_shi_7': 7}
    original = deepcopy(rows)
    result = evaluate_main_play_pair(rows, rows)
    assert result['candidate']['select_6']['mean_hits'] == 3
    assert result['candidate']['fu_shi_7']['mean_hits'] == 4
    assert rows == original
