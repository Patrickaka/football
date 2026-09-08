from copy import deepcopy

import pytest

from src.domain.sports.football.score_tail import assess_score_tail
from src.domain.sports.football.scoring_model import build_score_matrix


def test_common_top3_can_coexist_with_substantial_high_total_probability():
    matrix = build_score_matrix(1.8, 1.8, rho=0)
    rows = sorted(matrix.items(), key=lambda item: (-item[1], item[0]))
    before = deepcopy(rows)
    assessment = assess_score_tail(rows)
    assert assessment['available']
    assert assessment['top3_contains_high_score'] is False
    assert .48 < assessment['tail_probabilities']['4+'] < .49
    assert [score for score, _ in rows[:3]] == [(1, 1), (1, 2), (2, 1)]
    assert rows == before
    for candidate in assessment['high_score_candidates']:
        score = tuple(map(int, candidate['score'].split('-')))
        assert candidate['probability'] == matrix[score]
        assert candidate['rank'] == [score for score, _ in rows].index(score) + 1
        assert candidate['probability'] < assessment['tail_probabilities']['4+']


def test_tail_probabilities_are_exact_nested_sums_and_include_extreme_scores():
    matrix = {(h, a): 0.0 for h in range(9) for a in range(8)}
    matrix.update({(1, 1): .10, (2, 1): .15, (2, 2): .20, (3, 2): .25, (8, 0): .30})
    assessment = assess_score_tail(list(matrix.items()))
    assert assessment['tail_probabilities'] == pytest.approx({'4+': .75, '5+': .55, '6+': .30})
    assert assessment['high_score_candidates'][0] == {'score': '8-0', 'probability': .30, 'rank': 1, 'total_goals': 8}


@pytest.mark.parametrize('rows', [[], [((2, 2), 1)], [((0, 0), float('nan'))],
                                  [((0, 0), .5), ((0, 0), .5)], [((0, 0), .9)]])
def test_partial_or_invalid_matrix_does_not_invent_tail_probability(rows):
    assessment = assess_score_tail(rows)
    assert assessment['available'] is False
    assert 'tail_probabilities' not in assessment


def test_small_nonzero_tail_is_not_rounded_down_by_scenario_selection_gate():
    matrix = build_score_matrix(.65, .65)
    result = assess_score_tail(list(matrix.items()))
    assert 0 < result['tail_probabilities']['4+'] < .20

    from src.football.pipeline import build_match_analysis
    analysis = build_match_analysis({'model': {'candidates': list(matrix.items())}})
    assert analysis['goals']['high_score_probability'] == pytest.approx(
        result['tail_probabilities']['4+'])
    assert all(pick['type'] != '大比分' for pick in analysis['score_picks'])
