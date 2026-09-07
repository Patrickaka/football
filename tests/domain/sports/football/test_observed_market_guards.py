"""Absent or synthetic quotes cannot supply a second market vote."""
from copy import deepcopy

import pytest

from src.domain.sports.football import scoring
from src.domain.sports.football.policy import select_top_score_candidates
from src.domain.sports.football.scoring_model import build_score_matrix


@pytest.fixture
def candidates():
    return sorted(build_score_matrix(2.1, 1.4, 7, -0.11).items(),
                  key=lambda item: -item[1])


@pytest.mark.parametrize('total', [None, {},
    {'source': 'model_proxy', 'implied_total': 2.5},
    {'source_matched': False, 'close_line': 2.5},
    {'source': 'hkjc', 'history_available': False},
])
def test_missing_total_does_not_invent_a_2_5_goal_anchor(candidates, total):
    adjusted, meta = scoring._anchor_score_candidates_to_goal_mean(candidates, total)
    assert adjusted == candidates
    assert not meta['applied']


@pytest.mark.parametrize('target', [float('nan'), float('inf'), -1.0])
def test_invalid_total_target_does_not_corrupt_scores(candidates, target):
    adjusted, meta = scoring._anchor_score_candidates_to_goal_mean(
        candidates, {'implied_total': target})
    assert adjusted == candidates
    assert meta['reason'] == 'invalid_total_target'


def test_genuine_current_quote_works_without_movement_history(candidates):
    adjusted, meta = scoring._anchor_score_candidates_to_goal_mean(candidates, {
        'source': 'hkjc', 'source_matched': True,
        'close_line': 2.25, 'history_available': False,
    })
    assert meta['applied']
    assert sum(sum(score) * p for score, p in adjusted) < sum(
        sum(score) * p for score, p in candidates)
    assert sum(p for _, p in adjusted) == pytest.approx(1)


def test_proxy_prices_and_movements_cannot_reshape_scores(candidates):
    asian = {'source': 'model_proxy', 'source_matched': False,
             'handicap': 0, 'handicap_change': 0.5,
             'close_prob': {'home': 0.5, 'away': 0.5},
             'prob_change': {'home': 0.08}}
    total = {'source': 'model_proxy', 'source_matched': False,
             'open_line': 2.5, 'close_line': 3.5,
             'open_prob': {'over': 0.4}, 'close_prob': {'over': 0.6}}
    adjusted, meta = scoring._apply_joint_market_state(candidates, asian, {}, total)
    assert adjusted == candidates
    assert not meta['applied']
    assert meta['tempo_signal'] == meta['direction_signal'] == 0


def test_missing_asian_keeps_genuine_total_constraint_active(candidates):
    total = {'source': 'hkjc', 'source_matched': True, 'close_line': 2.5,
             'close_prob': {'over': 0.5, 'under': 0.5}}
    adjusted, meta = scoring._apply_joint_market_state(candidates, None, {}, total)
    assert meta['total_constraint']['applied']
    assert not meta['asian_constraint']['applied']
    assert adjusted != candidates
    for predicate in (lambda h, a: h > a, lambda h, a: h == a, lambda h, a: h < a):
        assert sum(p for (h, a), p in adjusted if predicate(h, a)) == pytest.approx(
            sum(p for (h, a), p in candidates if predicate(h, a)))


def test_filled_proxy_fields_do_not_count_as_complete_market_data():
    asian = {'handicap': 0, 'open_prob': {'home': .5}, 'close_prob': {'home': .5}}
    total = {'close_line': 2.5, 'open_prob': {'over': .5}, 'close_prob': {'over': .5}}
    euro = {'close': {'home': .4, 'draw': .3, 'away': .3}}
    real = scoring._assess_market_data_quality(asian, euro, total)
    proxy = scoring._assess_market_data_quality(
        {**asian, 'source': 'model_proxy'}, euro, {**total, 'source_matched': False})
    assert real['grade'] == 'high'
    assert proxy['weight_factor'] < real['weight_factor']
    assert 'missing_total_line' in proxy['reasons']


@pytest.mark.parametrize('asian', [None, {}, {'source': 'model_proxy', 'handicap': 0}])
def test_missing_asian_does_not_boost_half_time_draws(asian):
    prediction = {'probs': [{'code': code, 'raw_prob': 1 / 9}
                           for code in ('HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA')]}
    before = deepcopy(prediction)
    adjusted = scoring._adjust_half_full_with_market_context(prediction, asian, None)
    assert adjusted == before
    assert prediction == before


def test_score_top5_retains_high_probability_away_scores_and_fixed_size():
    candidates = [((1, 0), .18), ((2, 0), .09), ((0, 1), .22),
                  ((1, 1), .21), ((1, 2), .12), ((0, 0), .10), ((2, 1), .08)]
    before = list(candidates)
    selected = select_top_score_candidates(candidates)
    assert [score for score, _ in selected] == [(0, 1), (1, 1), (1, 0), (1, 2), (0, 0)]
    assert selected[0][1] == .22  # Still an unconditional score probability.
    assert len(select_top_score_candidates(candidates, 3)) == 3
    assert select_top_score_candidates(candidates, 0) == []
    assert candidates == before
