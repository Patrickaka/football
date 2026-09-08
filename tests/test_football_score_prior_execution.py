"""The learned score prior must change the real matrix, with a bounded weight."""
import math
from unittest import mock

import pytest

from src.football import market_db, prediction_policy, scoring
from src.domain.sports.football.policy import select_top_score_candidates


BASE = {(1, 0): .31, (1, 1): .29, (0, 1): .24, (2, 1): .16}
PRIOR = {'sample_count': 300, 'distance': 0.0, 'exact_match': True,
         'probabilities': {'1-0': .10, '1-1': .10, '2-1': .80}}


def test_prior_uses_real_convex_weight_and_can_change_the_displayed_top3():
    adjusted, trace = scoring.apply_static_score_prior(BASE, PRIOR)
    assert trace['applied']
    assert trace['weight'] == pytest.approx(.15)
    assert adjusted[(2, 1)] == pytest.approx(.85 * .16 + .15 * .8)
    assert adjusted[(0, 1)] == pytest.approx(.85 * .24)
    assert sum(adjusted.values()) == pytest.approx(1)
    assert [s for s, _ in select_top_score_candidates(adjusted.items(), 3)] == [(1, 0), (1, 1), (2, 1)]
    assert set(BASE).issubset(adjusted)


@pytest.mark.parametrize('n,quality,weight', [(30, 1, .015), (150, 1, .075),
                                            (300, .4, .06), (30000, 1, .15)])
def test_more_samples_and_data_quality_control_the_existing_cap(n, quality, weight):
    _, trace = scoring.apply_static_score_prior(BASE, dict(PRIOR, sample_count=n), quality_factor=quality)
    assert trace['weight'] == pytest.approx(weight)


@pytest.mark.parametrize('changes,kwargs', [
    ({'sample_count': 29}, {}), ({'distance': .51}, {}), ({'distance': -1}, {}),
    ({'sample_count': math.nan}, {}), ({'probabilities': {'1-0': 1.0}}, {}),
    ({'probabilities': {'1-0': .2, '1-1': .2, '2-1': math.nan}}, {}),
    ({'probabilities': {'1-0': .2, '1-1': .2, '2-1': .2}}, {}),
    ({}, {'max_weight': 0}), ({}, {'quality_factor': 0}),
])
def test_ineligible_prior_never_changes_probabilities(changes, kwargs):
    adjusted, trace = scoring.apply_static_score_prior(BASE, dict(PRIOR, **changes), **kwargs)
    assert adjusted == BASE
    assert trace['applied'] is False
    assert trace['weight'] == 0


def _predict_with_prior(source=None):
    asian = {'handicap': 0, 'implied_supremacy': 0, 'open_handicap': 0,
             'close_prob': {'home': .5, 'away': .5}, 'open_prob': {'home': .5, 'away': .5}}
    euro = {'close': {'home': .47, 'draw': .29, 'away': .24}, 'implied_supremacy': 0}
    euro['open'] = dict(euro['close'])
    total = {'close_line': 2.5, 'open_line': 2.5, 'implied_total': 2.7,
             'close_prob': {'over': .5, 'under': .5}, 'open_prob': {'over': .5, 'under': .5}}
    if source:
        total['source'] = source
    with mock.patch.object(scoring, 'fit_lambdas_from_markets', return_value=(1.4, 1.3, 2.7, 0)), \
         mock.patch.object(scoring._modeling_mod, 'build_score_matrix', return_value=dict(BASE)), \
         mock.patch.object(scoring, 'apply_residual_correction', side_effect=lambda matrix, features: matrix), \
         mock.patch.object(prediction_policy, 'apply_score_distribution_policy', side_effect=lambda matrix, **kw: (matrix, {})), \
         mock.patch.object(scoring, '_assess_market_data_quality', return_value={'weight_factor': 1}), \
         mock.patch.object(prediction_policy, 'get_prediction_policy', return_value={'static_market_cap': .15}), \
         mock.patch.object(market_db, 'get_market_score_prob', return_value=PRIOR), \
         mock.patch.object(market_db.MarketChangeDB, 'get_change_stats', side_effect=AssertionError('movement must run once outside ensemble')):
        return scoring.predict_scores(asian, euro, total, enable_ensemble=True)


def test_real_score_ensemble_executes_prior_and_exposes_actual_weight():
    candidates, _, _, meta = _predict_with_prior()
    expected, _ = scoring.apply_static_score_prior(BASE, PRIOR)
    assert dict(candidates) == pytest.approx(expected)
    assert meta['market_db_used'] is True
    assert meta['static_market_prior']['applied'] is True
    assert meta['static_market_prior']['weight'] == pytest.approx(.15)
    assert set(meta['static_market_prior']['members']) == {'poisson', 'negative_binomial'}


@pytest.mark.parametrize('source', ['model_proxy', 'unavailable'])
def test_proxy_total_cannot_activate_historical_betting_line_prior(source):
    candidates, _, _, meta = _predict_with_prior(source)
    assert {score: p for score, p in candidates if p > 0} == pytest.approx(BASE)
    assert meta['market_db_used'] is False
    assert meta['static_market_prior']['weight'] == 0


def test_member_failure_does_not_mislabel_or_weight_the_failed_model():
    def member(*args, model_type, **kwargs):
        if model_type == 'poisson':
            raise ValueError('unavailable')
        return list(BASE.items()), 1.4, 1.3, {}
    with mock.patch.object(scoring, 'predict_scores', side_effect=member):
        rows, lh, la, meta = scoring.ensemble_predict_scores({}, {}, {'close_line': 2.5})
    assert dict(rows) == pytest.approx(BASE)
    assert (lh, la) == (1.4, 1.3)
    assert meta['ensemble_weights'] == {'poisson': 0, 'negative_binomial': 1}


def test_observed_tail_extends_full_matrix_without_discarding_probability():
    from src.domain.sports.football.prediction_evaluation import score_matrix
    base = scoring._modeling_mod.build_score_matrix(1.3, 1.1)
    prior = dict(PRIOR, probabilities={'1-0': .5, '1-1': .49, '8-0': .01})
    result, trace = scoring.apply_static_score_prior(base, prior)
    assert trace['applied']
    assert result[(8, 0)] == pytest.approx(.0015)
    assert result[(8, 7)] == 0
    assert score_matrix({f'{h}-{a}': p for (h, a), p in result.items()}) is not None


def test_equal_probability_ranks_match_frozen_event_evaluation():
    from src.domain.sports.football.prediction_evaluation import score_metrics
    rows = [((1, 1), .25), ((1, 0), .25), ((0, 1), .25), ((0, 0), .25)]
    ranked = select_top_score_candidates(rows, 3)
    metrics = score_metrics({f'{h}-{a}': p for (h, a), p in rows}, '1-0')
    assert [score for score, _ in ranked] == [(0, 0), (0, 1), (1, 0)]
    assert metrics['actual_score_rank'] == 3
    assert metrics['hit_top3']
