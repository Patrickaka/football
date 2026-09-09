"""Regression checks for live select-6 / fushi-7 composition in offline evaluation."""
import gzip
import json
import unittest
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.kl8 import analyzer as analyzer_module
from src.kl8 import backtest as backtest_module
from src.kl8 import config, strategies
from src.kl8.analyzer import KL8Analyzer
from src.kl8.backtest import (
    KL8RollingBacktest, _predict_fushi7_from_select6, _predict_select6_primary,
)
from src.kl8.candidates import _adaptive_repeat_cap


def _analyzer(history):
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = sorted(history, key=lambda row: row['issue'], reverse=True)
    analyzer.using_simulated_data = False
    analyzer.history_file = ''
    analyzer._data_mtime = 0
    analyzer.update_statistics()
    return analyzer


@contextmanager
def _live_without_storage(analyzer, strategy):
    """Exercise predict_all while keeping runtime snapshots and records untouched."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(strategies, 'resolve_play_strategy', return_value=strategy))
        stack.enter_context(patch.object(analyzer_module, '_load_last_snapshot', return_value=None))
        stack.enter_context(patch.object(analyzer_module, '_prediction_config_fingerprint', return_value='test'))
        stack.enter_context(patch.object(analyzer_module, '_build_recent_settlement_performance', return_value={}))
        stack.enter_context(patch('src.kl8.fetch.count_valid_history_periods', return_value=len(analyzer.history_data)))
        stack.enter_context(patch.object(analyzer, '_save_prediction_snapshot', return_value=None))
        stack.enter_context(patch.object(analyzer, '_save_exclude_recalculation', return_value={}))
        yield


class FushiBacktestParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent / 'fixtures/numeric/kl8_history.json.gz'
        with gzip.open(fixture, 'rt', encoding='utf-8') as stream:
            cls.history = json.load(stream)['results']
        with patch.object(config, 'ACTIVE_STRATEGIES', {}), patch.object(config, 'VERIFY_ONLY_MODE', False):
            cls.strategy = strategies.resolve_play_strategy('select_6')

    def test_primary_seventh_uses_best_available_rank_and_later_rounds_stay_disjoint(self):
        analyzer = _analyzer(self.history[:100])
        analyzer.statistics['last_numbers'] = set(range(61, 81))
        strategy = {**deepcopy(self.strategy), 'pool_max_last_numbers': 6}
        ranking = [(number, float(81 - number)) for number in range(1, 81)]
        with _live_without_storage(analyzer, strategy), patch.object(
            analyzer, 'build_pool_by_strategy', return_value={'candidates': ranking, 'selected': list(range(1, 81))},
        ):
            live = analyzer.predict_all()
            primary, offline_ranking = _predict_select6_primary(analyzer, strategy)
            compound = _predict_fushi7_from_select6(analyzer, strategy, primary, offline_ranking)
            first, _ = analyzer._calculate_select_recalculation('select_6', primary)
            compound_first = analyzer.recalculate_play_excluding(
                'fu_shi_7', compound, record_context={'select6_round': first},
            )

        self.assertEqual(primary, [1, 2, 3, 4, 5, 6])
        self.assertEqual(first['numbers'], [7, 8, 9, 10, 11, 12])
        self.assertEqual(compound, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(compound, live['fu_shi_7']['top7_numbers'])
        self.assertEqual(compound_first['top7_numbers'], [8, 9, 10, 11, 12, 13, 14])
        self.assertEqual(compound_first['replaced_numbers'], [7])
        self.assertEqual(compound_first['total_combinations'], 21)
        self.assertTrue((set(first['numbers']) - set(compound)).issubset(compound_first['top7_numbers']))
        self.assertTrue(set(compound).isdisjoint(compound_first['top7_numbers']))

    def test_real_history_primary_and_compound_match_live_for_multiple_shapes(self):
        for offset in (0, 37, 80):
            for diversified, mode in ((False, 'concentrated'), (True, 'balanced')):
                with self.subTest(offset=offset, mode=mode):
                    analyzer = _analyzer(self.history[offset:offset + 100])
                    strategy = {**deepcopy(self.strategy), 'pool_diversify': diversified, 'final_selection_mode': mode}
                    with _live_without_storage(analyzer, strategy):
                        live = analyzer.predict_all()
                    with patch.object(strategies, 'resolve_play_strategy', side_effect=AssertionError('global strategy read')):
                        primary, ranking = _predict_select6_primary(analyzer, strategy)
                        compound = _predict_fushi7_from_select6(analyzer, strategy, primary, ranking)
                    self.assertEqual(primary, live['select_6']['numbers'])
                    self.assertEqual(compound, live['fu_shi_7']['top7_numbers'])
                    self.assertEqual(len(primary), 6)
                    self.assertEqual(len(set(compound)), 7)
                    self.assertTrue(set(primary).issubset(compound))
                    expected_seventh = next(n for n, _ in ranking if n not in primary)
                    self.assertEqual(set(compound), set(primary) | {expected_seventh})
                    if mode == 'balanced':
                        self.assertLessEqual(
                            len(set(primary) & analyzer.statistics['last_numbers']),
                            min(strategy['pool_max_last_numbers'], _adaptive_repeat_cap(analyzer.history_data, 6)),
                        )

    def test_primary_uses_adaptive_cap_when_strategy_allows_more_repeats(self):
        history = [
            {'issue': str(2026000 + index), 'date': '2026-01-01',
             'numbers': list(range(1 + (index % 4) * 20, 21 + (index % 4) * 20))}
            for index in range(1, 101)
        ]
        analyzer = _analyzer(history)
        strategy = deepcopy(self.strategy)
        strategy['final_selection_mode'] = 'balanced'
        ordered = [1, 7, 13, 21, 31, 41, 51]
        ordered += [number for number in range(1, 81) if number not in ordered]
        ranking = [(number, float(81 - index)) for index, number in enumerate(ordered)]
        self.assertEqual(strategy['pool_max_last_numbers'], 3)
        self.assertEqual(_adaptive_repeat_cap(analyzer.history_data, 6), 2)
        with _live_without_storage(analyzer, strategy), patch.object(
            analyzer, 'build_pool_by_strategy', return_value={'candidates': ranking, 'selected': ordered},
        ):
            live = analyzer.predict_all()
            primary, _ = _predict_select6_primary(analyzer, strategy)
        self.assertEqual(primary, [1, 7, 21, 31, 41, 51])
        self.assertEqual(primary, live['select_6']['numbers'])

    def test_explicit_first_round_override_matches_default_live_resolution(self):
        analyzer = _analyzer(self.history[:100])
        strategy = deepcopy(self.strategy)
        strategy['first_exclusion_strategy'] = {
            'strategy_id': 'test-only-first-round', 'feature_weights': {'frequency': 1.0},
            'frequency_mode': 'hot', 'exclusion_selection_mode': 'zone_spread',
        }
        original_strategy = deepcopy(strategy)
        with _live_without_storage(analyzer, strategy):
            live = analyzer.predict_all()
        with patch.object(strategies, 'resolve_play_strategy', side_effect=AssertionError('global strategy read')):
            primary, ranking = _predict_select6_primary(analyzer, strategy)
            compound = _predict_fushi7_from_select6(analyzer, strategy, primary, ranking)
        self.assertEqual(compound, live['fu_shi_7']['top7_numbers'])
        self.assertEqual(strategy, original_strategy)

    def test_rolling_and_permutation_use_identical_live_select6_picks(self):
        history = self.history[:110]
        ascending = sorted(history, key=lambda row: row['issue'])
        strategy = deepcopy(self.strategy)
        strategy.update({'pool_diversify': True, 'final_selection_mode': 'balanced'})
        expected = {}
        expected_compounds = {}
        for index in range(100, 110):
            analyzer = _analyzer(ascending[index - 100:index])
            with _live_without_storage(analyzer, strategy):
                live = analyzer.predict_all()
            issue = analyzer.history_data[0]['issue']
            expected[issue] = live['select_6']['numbers']
            expected_compounds[issue] = live['fu_shi_7']['top7_numbers']

        primary_calls = []
        compound_calls = []

        def primary_spy(analyzer, current_strategy):
            numbers, ranking = _predict_select6_primary(analyzer, current_strategy)
            primary_calls.append((analyzer.history_data[0]['issue'], numbers))
            return numbers, ranking

        def compound_spy(analyzer, current_strategy, primary, ranking):
            numbers = _predict_fushi7_from_select6(analyzer, current_strategy, primary, ranking)
            compound_calls.append((analyzer.history_data[0]['issue'], numbers))
            return numbers

        options = {key: strategy[key] for key in (
            'window_size', 'repeat_direction', 'pool_diversify', 'pool_max_last_numbers', 'final_selection_mode',
        )}
        backtest = KL8RollingBacktest(_analyzer(history))
        with patch.object(backtest_module, '_predict_select6_primary', side_effect=primary_spy), \
                patch.object(backtest_module, '_predict_fushi7_from_select6', side_effect=compound_spy), \
                patch.object(strategies, 'resolve_play_strategy', side_effect=AssertionError('global strategy read')):
            result = backtest._rolling_backtest_parametric(
                strategy['feature_weights'], strategy['model_weights'], 100, 110, **options,
            )
            rolling_primary_calls = list(primary_calls)
            primary_calls.clear()
            with patch.object(backtest, '_rolling_backtest_parametric', return_value=result):
                permutation = backtest._permutation_test(
                    strategy['feature_weights'], strategy['model_weights'], 100, 110,
                    pick_n=6, n_permutations=3, **options,
                )

        self.assertEqual(dict(rolling_primary_calls), expected)
        self.assertEqual(primary_calls, rolling_primary_calls)
        self.assertEqual(dict(compound_calls), expected_compounds)
        self.assertEqual(result['select_6']['n_tests'], 10)
        self.assertEqual(result['fu_shi_7']['pool_size'], 7)
        self.assertEqual(result['fu_shi_7']['total_bet'], 10 * 21 * 2)
        expected_hits = [len(set(expected_compounds[ascending[index - 1]['issue']]) & set(ascending[index]['numbers'])) for index in range(100, 110)]
        self.assertEqual(result['fu_shi_7']['pool_mean_hits'], sum(expected_hits) / 10)
        self.assertNotIn('error', permutation)


if __name__ == '__main__':
    unittest.main()
