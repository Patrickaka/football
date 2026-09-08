"""A conditional handicap scenario must never become the saved match forecast."""

import copy
import unittest
from contextlib import ExitStack
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.domain.sports.football.lottery import lottery_market_probabilities
from src.football import pipeline, result_sync


# Draw is the largest ordinary outcome. Given a draw, +1 always settles as
# 让胜 and -1 as 让负, while either full-match handicap distribution has 3 bins.
CANDIDATES = [((1, 1), .40), ((1, 0), .15), ((2, 0), .10),
              ((0, 1), .20), ((0, 2), .15)]


def _match(handicap):
    return {
        'match_id': f'rqspf-persistence-{handicap}',
        'home': '主队', 'away': '客队', 'league': '英超',
        'time': '2099-09-09 00:45', 'schedule_source': 'sporttery',
        'analysis_source_id_available': False,
        'lottery_handicap': handicap,
        # Unmatched offer keeps both model distributions available. The odds
        # below are equal, so they preserve the draw as the ordinary top pick.
        'lottery_spf_odds': {'胜': 3.0, '平': 3.0, '负': 3.0},
    }


def _lottery(match):
    return lottery_market_probabilities(
        CANDIDATES, match['lottery_handicap'],
        spf_odds=match['lottery_spf_odds'],
    )


class RqspfProbabilityPersistenceTests(unittest.TestCase):
    def assert_saved_marginal(self, save, result, handicap):
        save.assert_called_once()
        saved = save.call_args.kwargs
        lottery = result['lottery']
        marginal = lottery['handicap']['probabilities']
        linked = lottery['linked_recommendation']
        conditional = linked['handicap_conditional_probabilities']
        certain_result = '让胜' if handicap == 1 else '让负'

        self.assertEqual(linked['standard_prediction'], '平')
        self.assertEqual(conditional, {certain_result: 1.0})
        self.assertEqual(set(marginal), {'让胜', '让平', '让负'})
        self.assertAlmostEqual(sum(marginal.values()), 1.0)
        self.assertTrue(all(0 < p < 1 for p in marginal.values()))
        self.assertEqual(saved['lottery_handicap'], handicap)
        self.assertEqual(saved['predicted_rqspf'], marginal)
        self.assertNotEqual(saved['predicted_rqspf'], conditional)
        # The conditional explanation remains available for display/auditing.
        self.assertEqual(
            saved['odds_data']['lottery']['linked_recommendation'], linked)

    def test_cache_hit_saves_all_match_outcomes_not_draw_condition(self):
        for handicap in (1, -1):
            with self.subTest(handicap=handicap):
                match = _match(handicap)
                cached = {
                    'model': {
                        'prediction_logic_version': pipeline.FOOTBALL_PREDICTION_LOGIC_VERSION,
                        'candidates': copy.deepcopy(CANDIDATES),
                    },
                    'lottery': _lottery(match),
                    'model_status': {'prediction_saved': True},
                }
                original = copy.deepcopy(cached)
                with patch.object(pipeline, 'CACHE_AVAILABLE', True), \
                     patch.object(pipeline, 'get_cache', return_value=cached) as cache, \
                     patch.object(pipeline, 'set_cache') as write_cache, \
                     patch.object(pipeline, 'predict_scores') as calculate, \
                     patch.object(result_sync, 'save_prediction', return_value={
                         'saved': True, 'persistence_backend': 'test'}) as save:
                    result = pipeline.analyze_match(match)

                cache.assert_called_once_with(
                    'match_analysis', pipeline.analysis_cache_key(match), match['time'])
                calculate.assert_not_called()
                write_cache.assert_not_called()
                self.assertIs(result, cached)
                self.assertEqual(cached, original)
                self.assert_saved_marginal(save, result, handicap)

    def test_fresh_analysis_saves_all_match_outcomes_not_draw_condition(self):
        # A completed save, an already persisted prediction, and a failed or
        # unknown save are different outcomes of the real persistence API.
        save_outcomes = (
            ({'saved': True, 'persistence_backend': 'mysql'}, True),
            ({'saved': True, 'persistence_backend': 'fallback'}, True),
            ({'saved': False, 'persistence_backend': 'unchanged'}, True),
            ({'saved': False, 'persistence_backend': 'failed'}, False),
            ({}, False),
        )
        for handicap, (save_response, expected_saved) in product((1, -1), save_outcomes):
            with self.subTest(handicap=handicap, save_response=save_response), ExitStack() as stack:
                # Keep the real orchestration, market aggregation and save call;
                # replace persistence and model inputs with isolated fixtures.
                for flag in ('CACHE_AVAILABLE', 'BAYESIAN_CALIBRATION_AVAILABLE',
                             'DYNAMIC_WEIGHTS_AVAILABLE', 'SIMILAR_MARKET_AVAILABLE',
                             'STEAM_MOVE_AVAILABLE'):
                    stack.enter_context(patch.object(pipeline, flag, False))
                stack.enter_context(patch.object(pipeline, 'clear_fetch_cache'))
                stack.enter_context(patch.object(
                    pipeline, 'predict_scores',
                    return_value=(copy.deepcopy(CANDIDATES), 1.2, 1.2, {})))
                stack.enter_context(patch.object(
                    pipeline, 'apply_market_change_prior',
                    side_effect=lambda scores, *args, **kwargs: (scores, {'used': False})))
                stack.enter_context(patch.object(
                    pipeline, 'anchor_candidates_to_market',
                    return_value=(copy.deepcopy(CANDIDATES), {})))
                stack.enter_context(patch.object(
                    pipeline, 'calculate_half_full_time_probs', return_value={}))
                stack.enter_context(patch(
                    'src.football.history_calibration.get_runtime_history_profile',
                    return_value=None))
                stack.enter_context(patch(
                    'src.football.ml.load_trained_ml_model', return_value=False))
                stack.enter_context(patch(
                    'src.football.market_db.MarketScoreDB',
                    return_value=SimpleNamespace(sample_counts={})))
                stack.enter_context(patch(
                    'src.football.bayes_report.load_professional_validation_summary',
                    return_value={'available': False, 'prediction_ready': False}))
                stack.enter_context(patch.object(
                    result_sync, 'get_history',
                    return_value=SimpleNamespace(records=[], get_record=Mock(return_value=None))))
                stack.enter_context(patch(
                    'src.football.research_runtime.completed_intelligence', return_value=None))
                stack.enter_context(patch.object(
                    result_sync, 'get_history_stats', return_value={}))
                save = stack.enter_context(patch.object(
                    result_sync, 'save_prediction', return_value=save_response))
                fetch = stack.enter_context(patch.object(
                    pipeline._fetching_mod, 'fetch', side_effect=AssertionError('network request')))

                result = pipeline.analyze_match(_match(handicap), force_refresh=True)

                fetch.assert_not_called()
                self.assertIs(result['model_status']['prediction_saved'], expected_saved)
                self.assert_saved_marginal(save, result, handicap)


if __name__ == '__main__':
    unittest.main()
