"""O/U movement compares the same goal scale, never different betting events."""

from copy import deepcopy
import json
import math
from unittest.mock import patch

import pytest

from src.domain.sports.football import scoring
from src.domain.sports.football.market_anchoring import anchor_candidates_to_market
from src.domain.sports.football.markets import fair_over_probability, implied_total_goals
from src.domain.sports.football.scoring_model import build_score_matrix


def quoted_total(open_mean, close_mean, open_line=2.5, close_line=3.5):
    """Construct valid fair quotes independently of the movement calculation."""
    open_over = fair_over_probability(open_mean, open_line)
    close_over = fair_over_probability(close_mean, close_line)
    return {
        'source': 'test_market', 'source_matched': True,
        'history_available': True,
        'open_line': open_line, 'close_line': close_line,
        'open_prob': {'over': open_over, 'under': 1 - open_over},
        'close_prob': {'over': close_over, 'under': 1 - close_over},
        'implied_total': close_mean,
    }


@pytest.mark.parametrize('open_line,close_line', [
    (2.5, 3.0), (2.5, 3.5), (2.75, 3.25), (3.25, 2.75), (2.69, 3.12),
])
def test_equal_goal_mean_remains_stable_across_settlement_lines(open_line, close_line):
    total = quoted_total(3.2, 3.2, open_line, close_line)
    signal = scoring._total_market_tempo_signal(total)
    assert signal['available']
    assert signal['reason'] == 'stable_total_market'
    assert signal['signal'] == signal['implied_change'] == 0
    assert signal['open_implied_total'] == pytest.approx(3.2)
    assert signal['implied_total'] == pytest.approx(3.2)
    assert not signal['conflict']
    assert not signal['over_delta_comparable']
    scores = {'1-1': .4, '2-1': .35, '3-2': .25}
    adjusted, meta = scoring._adjust_score_probs_with_total_movement(scores, total)
    assert adjusted == scores
    assert not meta['applied']


def test_rising_total_with_lower_cross_line_over_probability_is_not_a_conflict():
    total = quoted_total(3.0, 3.6)
    signal = scoring._total_market_tempo_signal(total)
    assert signal['over_delta'] < 0  # The new event is over 3.5, not over 2.5.
    assert signal['implied_change'] == pytest.approx(.6)
    assert signal['signal'] > 0
    assert signal['available'] and not signal['conflict']


@pytest.mark.parametrize('before,after', [(2.8, 3.2), (3.2, 2.8)])
def test_same_line_water_movement_preserves_its_actual_direction(before, after):
    signal = scoring._total_market_tempo_signal(quoted_total(before, after, 2.75, 2.75))
    assert signal['over_delta_comparable']
    assert signal['line_delta'] == 0
    assert signal['signal'] * (after - before) > 0
    assert signal['implied_change'] == pytest.approx(after - before)


@pytest.mark.parametrize('line', [1.75, 2.5, 3.5, 4.0])
def test_stable_high_or_low_market_is_not_movement(line):
    total = quoted_total(3.2, 3.2, line, line)
    assert scoring._total_market_tempo_signal(total)['signal'] == 0


@pytest.mark.parametrize('missing', ['open_line', 'close_line', 'open_prob', 'close_prob'])
def test_both_observed_quotes_are_required(missing):
    total = quoted_total(3.0, 3.6)
    total.pop(missing)
    before = deepcopy(total)
    signal = scoring._total_market_tempo_signal(total)
    assert not signal['available']
    assert signal['signal'] == 0
    assert signal['implied_change'] is None
    assert total == before


@pytest.mark.parametrize('override', [
    {'history_available': False}, {'source': 'model_proxy'},
    {'source': 'unavailable'}, {'source_matched': False},
])
def test_filled_defaults_and_explicitly_missing_history_cannot_create_movement(override):
    total = {**quoted_total(3.0, 3.6), **override}
    signal = scoring._total_market_tempo_signal(total)
    assert not signal['available'] and signal['signal'] == 0
    scores = {'1-1': .5, '3-1': .5}
    adjusted, meta = scoring._adjust_score_probs_with_total_movement(scores, total)
    assert adjusted == scores and not meta['applied']
    goals = {2: .5, 4: .5}
    adjusted_goals, goals_meta = scoring._adjust_goal_dist_with_total_movement(goals, total)
    assert adjusted_goals == goals and not goals_meta['applied']


@pytest.mark.parametrize('field,value', [
    ('open_line', float('nan')), ('close_line', float('inf')),
    ('open_line', -2.5), ('open_line', 0), ('close_line', 1e308),
    ('open_over', float('nan')), ('close_over', float('-inf')),
    ('open_over', 0), ('close_over', 1), ('close_over', 1e-300),
])
def test_invalid_or_out_of_range_quotes_are_finite_no_ops(field, value):
    total = quoted_total(3.0, 3.6)
    if field.endswith('_over'):
        total[field.split('_')[0] + '_prob']['over'] = value
    else:
        total[field] = value
    signal = scoring._total_market_tempo_signal(total)
    assert not signal['available'] and signal['signal'] == 0
    json.dumps(signal, allow_nan=False)


def test_raw_quotes_determine_movement_even_when_a_caller_has_changed_implied_metadata():
    total = quoted_total(3.0, 3.6)
    total.update({'open_implied_total': 100, 'implied_total': 100, 'implied_change': -10})
    signal = scoring._total_market_tempo_signal(total)
    assert signal['open_implied_total'] == pytest.approx(3.0)
    assert signal['implied_total'] == pytest.approx(3.6)
    assert signal['implied_change'] == pytest.approx(.6)


def test_each_distribution_inverts_only_two_quotes_and_does_not_mutate_input():
    matrix = build_score_matrix(1.7, 1.3, 7, 0)
    scores = {f'{home}-{away}': p for (home, away), p in matrix.items()}
    before = deepcopy(scores)
    total = quoted_total(3.0, 3.6)
    with patch.object(scoring, 'implied_total_goals', wraps=implied_total_goals) as inverter:
        adjusted, meta = scoring._adjust_score_probs_with_total_movement(scores, total)
    assert inverter.call_count == 2
    assert scores == before
    assert meta['applied'] and meta['direction'] == 'over'
    assert sum(adjusted.values()) == pytest.approx(1)
    assert all(math.isfinite(p) and p >= 0 for p in adjusted.values())


def test_score_and_goal_adjusters_use_the_same_comparable_signal():
    total = quoted_total(3.0, 3.6)
    scores = {'1-1': .4, '2-1': .35, '3-2': .25}
    _, score_meta = scoring._adjust_score_probs_with_total_movement(scores, total)
    _, goal_meta = scoring._adjust_goal_dist_with_total_movement({2: .4, 3: .35, 5: .25}, total)
    assert score_meta['tempo'] == goal_meta['tempo']
    assert score_meta['direction'] == goal_meta['direction'] == 'over'
    assert score_meta['expected_after'] > score_meta['expected_before']
    assert goal_meta['expected_after'] > goal_meta['expected_before']


def test_final_market_anchor_preserves_1x2_and_all_totals_share_the_final_matrix():
    base_matrix = build_score_matrix(1.7, 1.3, 7, 0)
    total = quoted_total(3.0, 4.2, 2.5, 3.5)
    scores, _ = scoring._adjust_score_probs_with_total_movement(
        {f'{h}-{a}': p for (h, a), p in base_matrix.items()}, total)
    candidates = [(tuple(map(int, score.split('-'))), p) for score, p in scores.items()]
    outcomes = {'home': .46, 'draw': .28, 'away': .26}
    final, _ = anchor_candidates_to_market(candidates, total, {'close': outcomes}, {})
    assert sum(p for _, p in final) == pytest.approx(1)
    predicates = {'home': lambda h, a: h > a, 'draw': lambda h, a: h == a,
                  'away': lambda h, a: h < a}
    for outcome, predicate in predicates.items():
        assert sum(p for (h, a), p in final if predicate(h, a)) == pytest.approx(outcomes[outcome])
    goals = {}
    for (h, a), probability in final:
        goals[h + a] = goals.get(h + a, 0) + probability
    for threshold in (4, 5, 6):
        assert sum(p for g, p in goals.items() if g >= threshold) == pytest.approx(
            sum(p for (h, a), p in final if h + a >= threshold))
    assert sum(p for (h, a), p in final if h + a >= 4) > sum(
        p for (h, a), p in base_matrix.items() if h + a >= 4)
    # The existing selector still ranks unconditional exact-score probability.
    from src.domain.sports.football.policy import select_top_score_candidates
    top3 = select_top_score_candidates(final, 3)
    assert top3 == sorted(final, key=lambda item: (-item[1], item[0]))[:3]
