"""Sealed outcomes must not alter KL8 parameter-search candidate selection."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.kl8 import backtest as backtest_module
from src.kl8 import config
from src.kl8.backtest import KL8RollingBacktest


class ParameterSearchHoldoutTests(unittest.TestCase):
    def search(self, validation, final, *, top_n=3):
        """Keep the real search orchestration; supply controlled phase metrics."""
        backtest = KL8RollingBacktest.__new__(KL8RollingBacktest)
        backtest.analyzer = SimpleNamespace(history_data=[{} for _ in range(800)])
        plays = list(validation)
        names = list(dict.fromkeys(name for rows in validation.values() for name in rows))
        candidates = {
            name: {'strategy_id': name, 'feature_weights': {'marker': name},
                   'model_weights': {'rank': 1.0}, 'window_size': 100}
            for name in names
        }
        calls = []

        def rolling(weights, model_weights, *, start_idx, **kwargs):
            phase = 'validation' if start_idx == 300 else 'final'
            name = weights['marker']
            calls.append((phase, name))
            if phase == 'final' and final is None:
                return {'error': 'sealed outcomes unavailable'}
            source = validation if phase == 'validation' else final
            results = {}
            for play in plays:
                score, hit_rate, lift = source[play][name]
                expected = 1.75 if play == 'fu_shi_7' else 1.5
                results[play] = {
                    'controlled_score': score, 'controlled_hit_rate': hit_rate,
                    'lift': lift, 'mean_hits': expected * (1 + lift),
                    'pool_mean_hits': expected * (1 + lift),
                    'pool_expected_random': expected,
                    'n_tests': 200, 'is_significant': True,
                }
            return results

        def score(metrics, *args):
            return metrics['controlled_score'], {
                'hit_rate_score': metrics['controlled_hit_rate'], 'hit_rate_lifts': {},
            }

        registry_before = deepcopy(config.ACTIVE_STRATEGIES)
        with patch.object(backtest, '_build_parameter_search_candidates', return_value=candidates), \
             patch.object(backtest, '_split_three_stage', return_value={
                 'val': (300, 600), 'final_test': (600, 800)}), \
             patch.object(backtest, '_rolling_backtest_parametric', side_effect=rolling), \
             patch.object(backtest_module, '_practical_validation_score', side_effect=score):
            result = backtest.run_parameter_search(play_types=plays, max_candidates=len(names), top_n=top_n)
        self.assertEqual(config.ACTIVE_STRATEGIES, registry_before)
        return result, calls

    def test_swapping_final_outcomes_cannot_change_either_play_winner_or_scores(self):
        validation = {
            'select_6': {'A': (1.0, .2, .1), 'B': (.99, .1, .08)},
            'fu_shi_7': {'A': (.8, .1, .05), 'B': (1.2, .3, .12)},
        }
        final_a = {play: {'A': (1.0, .9, .5), 'B': (0.0, 0.0, -.2)} for play in validation}
        final_b = {play: {'A': (0.0, 0.0, -.2), 'B': (1.0, .9, .5)} for play in validation}
        first, _ = self.search(validation, final_a)
        second, _ = self.search(validation, final_b)
        for play, winner in (('select_6', 'A'), ('fu_shi_7', 'B')):
            with self.subTest(play=play):
                self.assertEqual(first['best_by_play'][play]['candidate'], winner)
                self.assertEqual(second['best_by_play'][play]['candidate'], winner)
                for report in (first, second):
                    for row in report['rankings'][play]:
                        self.assertEqual(row['score'], validation[play][row['candidate']][0])
                self.assertNotEqual(first['best_by_play'][play]['final_test_lift'],
                                    second['best_by_play'][play]['final_test_lift'])

    def test_final_hit_rate_cannot_override_a_validation_lift_tiebreak(self):
        validation = {'select_6': {'A': (1.0, .2, .2), 'B': (1.0, .2, .1)}}
        final = {'select_6': {'A': (0.0, 0.0, -.5), 'B': (0.0, 1.0, 1.0)}}
        result, _ = self.search(validation, final)
        self.assertEqual([row['candidate'] for row in result['rankings']['select_6']], ['A', 'B'])

    def test_exact_validation_tie_remains_stable_when_only_final_lift_changes(self):
        validation = {'select_6': {'A': (1.0, .2, .1), 'B': (1.0, .2, .1)}}
        final = {'select_6': {'A': (0.0, 0.0, -.5), 'B': (0.0, 0.0, 1.0)}}
        result, _ = self.search(validation, final)
        self.assertEqual(result['best_by_play']['select_6']['candidate'], 'A')

    def test_missing_final_results_preserve_validation_order_and_report_missing_metrics(self):
        validation = {'select_6': {'A': (.99, .2, .1), 'B': (1.0, .1, .08)}}
        result, _ = self.search(validation, None)
        self.assertEqual(result['best_by_play']['select_6']['candidate'], 'B')
        self.assertTrue(all(row['final_test_lift'] is None for row in result['rankings']['select_6']))

    def test_validation_shortlist_is_fixed_before_any_final_outcome_is_opened(self):
        validation = {'select_6': {name: (score, .1, .1) for name, score in (
            ('A', 1.0), ('B', .9), ('C', .8), ('D', .7))}}
        final = {'select_6': {name: (score, .1, .1) for name, score in (
            ('A', -.1), ('B', -.1), ('C', -.1), ('D', 100.0))}}
        result, calls = self.search(validation, final, top_n=4)
        self.assertEqual(calls[:4], [('validation', name) for name in ('A', 'B', 'C', 'D')])
        self.assertEqual(set(calls[4:]), {('final', name) for name in ('A', 'B', 'C')})
        self.assertEqual([row['candidate'] for row in result['rankings']['select_6']], ['A', 'B', 'C', 'D'])
        self.assertIsNone(result['rankings']['select_6'][-1]['final_test_lift'])


if __name__ == '__main__':
    unittest.main()
