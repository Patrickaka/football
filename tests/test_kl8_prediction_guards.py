import unittest
import json
import tempfile
from pathlib import Path

import src.kl8 as kl8_module
from src.kl8 import (
    KL8RollingBacktest,
    KL8Analyzer,
    KL8_MIN_PREDICTION_PERIODS,
    _clean_pick_numbers,
    _diversify_candidate_pool,
    _compute_next_issue,
    normalize_record,
    validate_and_activate_strategy,
    _adaptive_repeat_cap,
    _hit_rate_priority_score,
    _hit_rate_priority_thresholds,
    _practical_validation_score,
    _shape_balanced_candidate_pool,
    _shape_profile,
)


def _record(issue: int):
    base = ((issue - 1) % 60) + 1
    nums = [((base + i - 1) % 80) + 1 for i in range(20)]
    return {'issue': str(2026000 + issue), 'numbers': sorted(set(nums)), 'date': '2026-01-01'}


class KL8PredictionGuardTests(unittest.TestCase):
    def test_normalize_record_strips_issue_and_rejects_bad_numbers(self):
        record = normalize_record({'issue': ' 2026001 ', 'numbers': list(range(1, 21))})
        self.assertEqual(record['issue'], '2026001')

        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': list(range(1, 20))}))
        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': [1] * 20}))
        self.assertIsNone(normalize_record({'issue': '2026001', 'numbers': list(range(62, 82))}))

    def test_clean_pick_numbers_requires_exact_unique_range(self):
        self.assertEqual(_clean_pick_numbers([1, 2, 3], 3), [1, 2, 3])
        self.assertEqual(_clean_pick_numbers([1, 1, 2], 3), [])
        self.assertEqual(_clean_pick_numbers([1, 2, 81], 3), [])
        self.assertEqual(_clean_pick_numbers([1, 2], 3), [])

    def test_compute_next_issue_uses_recent_diffs(self):
        history = [
            {'issue': '2026001', 'numbers': list(range(1, 21))},
            {'issue': '2026010', 'numbers': list(range(1, 21))},
            {'issue': '2026011', 'numbers': list(range(1, 21))},
            {'issue': '2026012', 'numbers': list(range(1, 21))},
        ]
        self.assertEqual(_compute_next_issue('2026012', history), '2026013')

    def test_diversify_candidate_pool_limits_basic_concentration(self):
        candidates = [
            (1, 100.0), (2, 99.0), (3, 98.0), (4, 97.0), (5, 96.0),
            (6, 95.0), (7, 94.0), (11, 93.0), (21, 92.0), (31, 91.0),
            (41, 90.0), (51, 89.0), (61, 88.0), (71, 87.0),
        ]
        diversified = _diversify_candidate_pool(candidates, 7, set(range(1, 21)))
        nums = [n for n, _ in diversified]

        self.assertEqual(len(nums), 7)
        self.assertLessEqual(sum(1 for n in nums if n <= 20), 3)
        self.assertLessEqual(max(nums.count(n) for n in nums), 1)

    def test_diversify_candidate_pool_accepts_repeat_cap(self):
        candidates = [
            (1, 100.0), (2, 99.0), (3, 98.0), (4, 97.0), (5, 96.0),
            (21, 95.0), (31, 94.0), (41, 93.0), (51, 92.0),
        ]
        diversified = _diversify_candidate_pool(
            candidates,
            5,
            set(range(1, 21)),
            max_last_numbers=1,
        )
        nums = [n for n, _ in diversified]

        self.assertEqual(len(nums), 5)
        self.assertLessEqual(sum(1 for n in nums if n <= 20), 1)

    def test_adaptive_repeat_cap_relaxes_when_recent_overlap_is_high(self):
        history = [_record(i) for i in range(40, 0, -1)]

        self.assertEqual(_adaptive_repeat_cap(history, 5), 3)
        self.assertEqual(_adaptive_repeat_cap(history, 7), 4)

    def test_build_pool_by_strategy_uses_adaptive_repeat_cap(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(40, 0, -1)]
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.using_simulated_data = False

        captured = {}

        def fake_voting(**kwargs):
            captured.update(kwargs)
            return {
                'selected': list(range(1, 8)),
                'candidates': [(n, float(80 - n)) for n in range(1, 21)],
                'votes': {},
            }

        original_build = KL8Analyzer._build_window_analyzer
        try:
            KL8Analyzer._build_window_analyzer = lambda self, window_size: type(
                'TempAnalyzer',
                (),
                {
                    'history_data': analyzer.history_data[:window_size],
                    'multi_model_voting': staticmethod(fake_voting),
                },
            )()
            analyzer.build_pool_by_strategy(
                {
                    'feature_weights': {'frequency': 1.0},
                    'model_weights': {'rank': 1.0},
                    'window_size': 40,
                    'repeat_direction': 'neutral',
                },
                pool_size=7,
            )
        finally:
            KL8Analyzer._build_window_analyzer = original_build

        self.assertEqual(captured['pool_max_last_numbers'], 4)

    def test_recalculate_play_excluding_removes_current_numbers(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        try:
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 40) + 1)),
                'candidates': [(n, float(100 - n)) for n in range(1, 41)],
                'votes': {},
            }
            result = analyzer.recalculate_play_excluding('select_5', [1, 2, 3, 4, 5])
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build

        self.assertNotIn('error', result)
        self.assertEqual(len(result['numbers']), 5)
        self.assertEqual(result['numbers'], sorted(result['numbers']))
        self.assertFalse(set(result['numbers']) & {1, 2, 3, 4, 5})
        self.assertEqual(result['excluded_numbers'], [1, 2, 3, 4, 5])

    def test_recalculate_play_excluding_supports_select_10(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        try:
            unsorted_candidates = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11] + list(range(21, 51))
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 50) + 1)),
                'candidates': [(n, float(100 - i)) for i, n in enumerate(unsorted_candidates)],
                'votes': {},
            }
            result = analyzer.recalculate_play_excluding('select_10', list(range(1, 11)))
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build

        self.assertNotIn('error', result)
        self.assertEqual(len(result['numbers']), 10)
        self.assertEqual(result['numbers'], sorted(result['numbers']))
        self.assertFalse(set(result['numbers']) & set(range(1, 11)))
        self.assertEqual(result['excluded_numbers'], list(range(1, 11)))
        self.assertNotEqual(result['quality']['selection_mode'], 'low_repeat')
        self.assertEqual(result['quality']['requested_selection_mode'], 'best_variant')

    def test_recalculate_play_excluding_respects_strategy_selection_mode(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original_build = KL8Analyzer.build_pool_by_strategy
        original_resolve = kl8_module.resolve_play_strategy
        try:
            KL8Analyzer.build_pool_by_strategy = lambda self, strategy, pool_size=20: {
                'selected': list(range(1, min(pool_size, 50) + 1)),
                'candidates': [(n, float(100 - n)) for n in range(1, 51)],
                'votes': {},
            }
            kl8_module.resolve_play_strategy = lambda play_type: {
                'strategy_id': f'{play_type}_forced_prize_floor',
                'feature_weights': {'frequency': 1.0},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
                'final_selection_mode': 'prize_floor',
                'prediction_mode': 'reference_unvalidated',
                'is_validated': False,
            }
            result = analyzer.recalculate_play_excluding('select_10', list(range(1, 11)))
        finally:
            KL8Analyzer.build_pool_by_strategy = original_build
            kl8_module.resolve_play_strategy = original_resolve

        self.assertNotIn('error', result)
        self.assertEqual(result['quality']['requested_selection_mode'], 'prize_floor')
        self.assertEqual(result['quality']['selection_mode'], 'prize_floor')

    def test_multi_model_voting_uses_broader_diversified_pool(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.statistics = {'last_numbers': set(range(1, 21))}

        original = KL8Analyzer._model_rank

        def fake_rank(self, top_n=20, **kwargs):
            return list(range(1, top_n + 1))

        try:
            KL8Analyzer._model_rank = fake_rank
            result = analyzer.multi_model_voting(
                pick_n=7,
                top_n=7,
                feature_weights={'frequency': 1.0},
                model_weights={'rank': 1.0},
            )
        finally:
            KL8Analyzer._model_rank = original

        self.assertTrue(result['diversified'])
        self.assertEqual(result['raw_candidate_count'], 40)
        self.assertEqual(len(result['selected']), 7)
        self.assertLessEqual(sum(1 for n in result['selected'] if n <= 20), 3)

    def test_shape_balanced_candidate_pool_controls_zone_and_odd_even(self):
        candidates = [
            (1, 100.0), (3, 99.0), (5, 98.0), (7, 97.0), (9, 96.0),
            (22, 95.0), (24, 94.0), (26, 93.0),
            (41, 92.0), (43, 91.0), (62, 90.0), (64, 89.0),
        ]

        pool = _shape_balanced_candidate_pool(
            candidates,
            8,
            last_numbers={1, 3, 5, 7, 9},
            max_last_numbers=2,
        )
        nums = [num for num, _ in pool]
        profile = _shape_profile(nums, {1, 3, 5, 7, 9})

        self.assertEqual(len(nums), 8)
        self.assertLessEqual(max(profile['zone20']), 3)
        self.assertLessEqual(abs(profile['odd_even']['odd'] - profile['odd_even']['even']), 2)
        self.assertLessEqual(profile['repeat_from_last'], 3)

    def test_predict_all_blocks_tiny_history(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(10, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.statistics = {}

        result = analyzer.predict_all()

        self.assertIn('error', result)
        self.assertEqual(result['data_quality']['min_required'], KL8_MIN_PREDICTION_PERIODS)
        self.assertEqual(result['data_quality']['reason'], 'insufficient_history')

    def test_strategy_activation_rejects_invalid_repeat_configuration(self):
        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            repeat_direction='sideways',
        )
        self.assertIn('repeat_direction', result['error'])

        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            pool_max_last_numbers=-1,
        )
        self.assertIn('pool_max_last_numbers', result['error'])

        result = validate_and_activate_strategy(
            'select_5',
            {'frequency': 1.0},
            {'rank': 1.0},
            50,
            pool_max_last_numbers=6,
        )
        self.assertIn('pick count', result['error'])

    def test_reference_select_5_6_and_10_use_best_variant_selection(self):
        original_verify_only = kl8_module.VERIFY_ONLY_MODE
        try:
            kl8_module.VERIFY_ONLY_MODE = False
            select5 = kl8_module.resolve_play_strategy('select_5')
            select6 = kl8_module.resolve_play_strategy('select_6')
            select10 = kl8_module.resolve_play_strategy('select_10')
        finally:
            kl8_module.VERIFY_ONLY_MODE = original_verify_only

        self.assertEqual(select5['final_selection_mode'], 'best_variant')
        self.assertEqual(select6['final_selection_mode'], 'best_variant')
        self.assertEqual(select10['final_selection_mode'], 'best_variant')
        self.assertIn('best_variant', select5['strategy_id'])
        self.assertIn('best_variant', select6['strategy_id'])
        self.assertIn('best_variant', select10['strategy_id'])

    def test_select6_hit_rate_priority_targets_prize_floor(self):
        self.assertEqual(_hit_rate_priority_thresholds('select_5'), ['>=2', '>=3'])
        self.assertEqual(_hit_rate_priority_thresholds('select_6'), ['>=3', '>=4'])

        select6_score, select6_detail = _hit_rate_priority_score(
            {
                'probabilities': {'>=2': 0.9, '>=3': 0.22, '>=4': 0.05},
                'theoretical_probs': {'>=2': 0.47, '>=3': 0.17, '>=4': 0.04},
            },
            'select_6',
        )

        self.assertGreater(select6_score, 0)
        self.assertNotIn('>=2', select6_detail)
        self.assertIn('>=3', select6_detail)
        self.assertIn('>=4', select6_detail)

    def test_predict_all_includes_select_8_9_10_and_fushi_10_11(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False
        analyzer.history_file = ''
        analyzer._data_mtime = 0
        analyzer.statistics = {}
        analyzer.update_statistics()

        original_save = KL8Analyzer._save_prediction_snapshot
        try:
            KL8Analyzer._save_prediction_snapshot = lambda self, prediction_result: None
            result = analyzer.predict_all()
        finally:
            KL8Analyzer._save_prediction_snapshot = original_save

        for pick in [8, 9, 10]:
            key = f'select_{pick}'
            self.assertIn(key, result)
            self.assertEqual(result[key]['pick'], pick)
            self.assertEqual(len(result[key]['numbers']), pick)
            self.assertEqual(result[key]['numbers'], sorted(result[key]['numbers']))

        for key in [f'select_{pick}' for pick in range(3, 11)]:
            expected_mode = 'best_variant' if key in {'select_5', 'select_6', 'select_10'} else 'prize_floor'
            self.assertEqual(
                result['resolved_strategies'][key]['final_selection_mode'],
                expected_mode,
            )
            self.assertIn('repeat_follow', result[key]['variants'])
            self.assertIn('zone_spread', result[key]['variants'])
            self.assertIn('prize_floor', result[key]['variants'])

        self.assertEqual(result['resolved_strategies']['select_5']['pool_max_last_numbers'], 3)
        self.assertEqual(result['resolved_strategies']['select_6']['pool_max_last_numbers'], 4)
        self.assertEqual(result['select_6']['prize_hit_thresholds'], ['>=3', '>=4'])
        self.assertEqual(result['select_6']['hit_rate_priority_thresholds'], ['>=3', '>=4'])
        self.assertEqual(
            result['resolved_strategies']['select_10']['final_selection_mode'],
            'best_variant',
        )
        self.assertEqual(result['resolved_strategies']['select_10']['pool_max_last_numbers'], 5)

        self.assertEqual(
            result['resolved_strategies']['fu_shi_7']['final_selection_mode'],
            'prize_floor',
        )
        self.assertIn('repeat_follow', result['fu_shi_7']['variants'])
        self.assertIn('zone_spread', result['fu_shi_7']['variants'])
        self.assertIn('prize_floor', result['fu_shi_7']['variants'])
        self.assertEqual(result['fu_shi_7']['prize_hit_thresholds'], ['>=3'])
        self.assertEqual(result['resolved_strategies']['fu_shi_7']['pool_max_last_numbers'], 4)

        self.assertIn('fu_shi_10_11', result)
        self.assertEqual(
            result['resolved_strategies']['fu_shi_10_11']['final_selection_mode'],
            'prize_floor',
        )
        self.assertEqual(len(result['fu_shi_10_11']['top11_numbers']), 11)
        self.assertEqual(
            result['fu_shi_10_11']['top11_numbers'],
            sorted(result['fu_shi_10_11']['top11_numbers']),
        )
        self.assertEqual(result['fu_shi_10_11']['combo_pick'], 10)
        self.assertEqual(result['fu_shi_10_11']['pool_size'], 11)
        self.assertEqual(result['fu_shi_10_11']['total_combinations'], 11)

    def test_backtest_passes_repeat_configuration_to_voting(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(80, 0, -1)]
        analyzer.using_simulated_data = False

        captured = []
        original = KL8Analyzer.multi_model_voting

        def fake_voting(self, **kwargs):
            captured.append(kwargs)
            return {
                'selected': list(range(1, 21)),
                'candidates': [(n, float(21 - n)) for n in range(1, 21)],
                'votes': {},
            }

        try:
            KL8Analyzer.multi_model_voting = fake_voting
            result = KL8RollingBacktest(analyzer)._rolling_backtest_parametric(
                {'frequency': 1.0},
                {'rank': 1.0},
                start_idx=55,
                end_idx=70,
                min_train=50,
                window_size=50,
                repeat_direction='follow',
                repeat_follow_score=0.92,
                repeat_non_follow_score=0.55,
                pool_diversify=False,
                pool_max_last_numbers=1,
            )
        finally:
            KL8Analyzer.multi_model_voting = original

        self.assertNotIn('error', result)
        self.assertTrue(captured)
        self.assertTrue(all(c['repeat_direction'] == 'follow' for c in captured))
        self.assertTrue(all(c['repeat_follow_score'] == 0.92 for c in captured))
        self.assertTrue(all(c['repeat_non_follow_score'] == 0.55 for c in captured))
        self.assertTrue(all(c['pool_diversify'] is False for c in captured))
        self.assertTrue(all(c['pool_max_last_numbers'] == 1 for c in captured))

    def test_permutation_passes_pool_configuration_to_voting(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(90, 0, -1)]
        analyzer.using_simulated_data = False

        backtest = KL8RollingBacktest(analyzer)
        captured = []

        def fake_backtest(*args, **kwargs):
            return {'select_5': {'lift': 0.1, 'mean_hits': 1.4}}

        original_voting = KL8Analyzer.multi_model_voting

        def fake_voting(self, **kwargs):
            captured.append(kwargs)
            return {
                'selected': list(range(1, 21)),
                'candidates': [(n, float(21 - n)) for n in range(1, 21)],
                'votes': {},
            }

        try:
            backtest._rolling_backtest_parametric = fake_backtest
            KL8Analyzer.multi_model_voting = fake_voting
            result = backtest._permutation_test(
                {'frequency': 1.0},
                {'rank': 1.0},
                start_idx=55,
                end_idx=70,
                pick_n=5,
                n_permutations=1,
                window_size=50,
                repeat_direction='follow',
                repeat_follow_score=0.91,
                repeat_non_follow_score=0.54,
                pool_diversify=False,
                pool_max_last_numbers=2,
            )
        finally:
            KL8Analyzer.multi_model_voting = original_voting

        self.assertNotIn('error', result)
        self.assertTrue(captured)
        self.assertTrue(all(c['repeat_direction'] == 'follow' for c in captured))
        self.assertTrue(all(c['repeat_follow_score'] == 0.91 for c in captured))
        self.assertTrue(all(c['repeat_non_follow_score'] == 0.54 for c in captured))
        self.assertTrue(all(c['pool_diversify'] is False for c in captured))
        self.assertTrue(all(c['pool_max_last_numbers'] == 2 for c in captured))

    def test_parameter_search_ranks_best_candidate_per_play(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        backtest._build_parameter_search_candidates = lambda max_candidates=80: {
            'candidate_a': {
                'strategy_id': 'candidate_a',
                'feature_weights': {'marker': 'a'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
            'candidate_b': {
                'strategy_id': 'candidate_b',
                'feature_weights': {'marker': 'b'},
                'model_weights': {'rank': 1.0},
                'window_size': 100,
            },
        }

        def fake_rolling(feature_weights, model_weights, **kwargs):
            marker = feature_weights['marker']
            is_final = kwargs['start_idx'] >= 600
            if marker == 'a':
                select_lift = 0.20 if not is_final else 0.04
                fushi_mean = 3.1 if not is_final else 2.9
            else:
                select_lift = 0.10 if not is_final else 0.30
                fushi_mean = 3.6 if not is_final else 3.4
            return {
                'select_5': {
                    'lift': select_lift,
                    'mean_hits': 1.25,
                    'expected_random': 1.25,
                    'profit_roi': -0.2,
                    'random_profit_roi': -0.3,
                    'return_multiple': 0.8,
                    'n_tests': 100,
                },
                'fu_shi_10_11': {
                    'pool_mean_hits': fushi_mean,
                    'pool_expected_random': 2.75,
                    'profit_roi': -0.1,
                    'random_profit_roi': -0.4,
                    'return_multiple': 0.9,
                    'n_tests': 100,
                },
            }

        backtest._rolling_backtest_parametric = fake_rolling

        result = backtest.run_parameter_search(
            play_types=['select_5', 'fu_shi_10_11'],
            max_candidates=2,
            top_n=2,
        )

        self.assertNotIn('error', result)
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['best_by_play']['select_5']['candidate'], 'candidate_a')
        self.assertEqual(result['best_by_play']['fu_shi_10_11']['candidate'], 'candidate_b')
        self.assertEqual(len(result['rankings']['select_5']), 2)

    def test_parameter_search_prioritizes_key_hit_rates(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        backtest._build_parameter_search_candidates = lambda max_candidates=80: {
            'mean_lift_only': {
                'strategy_id': 'mean_lift_only',
                'feature_weights': {'marker': 'mean'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
            'hit_rate_first': {
                'strategy_id': 'hit_rate_first',
                'feature_weights': {'marker': 'hit'},
                'model_weights': {'rank': 1.0},
                'window_size': 50,
            },
        }

        def fake_rolling(feature_weights, model_weights, **kwargs):
            marker = feature_weights['marker']
            if marker == 'hit':
                return {
                    'select_5': {
                        'lift': 0.05,
                        'mean_hits': 1.31,
                        'expected_random': 1.25,
                        'probabilities': {'>=2': 0.35, '>=3': 0.08},
                        'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                        'profit_roi': -0.2,
                        'random_profit_roi': -0.3,
                        'return_multiple': 0.8,
                        'n_tests': 100,
                    },
                }
            return {
                'select_5': {
                    'lift': 0.10,
                    'mean_hits': 1.38,
                    'expected_random': 1.25,
                    'probabilities': {'>=2': 0.31, '>=3': 0.07},
                    'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                    'profit_roi': -0.2,
                    'random_profit_roi': -0.3,
                    'return_multiple': 0.8,
                    'n_tests': 100,
                },
            }

        backtest._rolling_backtest_parametric = fake_rolling

        result = backtest.run_parameter_search(
            play_types=['select_5'],
            max_candidates=2,
            top_n=2,
        )

        self.assertNotIn('error', result)
        best = result['best_by_play']['select_5']
        self.assertEqual(best['candidate'], 'hit_rate_first')
        self.assertGreater(best['validation_hit_rate_score'], 0)

    def test_practical_score_prioritizes_prize_thresholds_over_mean_lift(self):
        hit_score, hit_detail = _practical_validation_score(
            {
                'lift': 0.03,
                'probabilities': {'>=2': 0.36, '>=3': 0.09},
                'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                'profit_roi': -0.2,
                'random_profit_roi': -0.3,
                'return_multiple': 0.8,
            },
            'select_5',
        )
        mean_score, mean_detail = _practical_validation_score(
            {
                'lift': 0.12,
                'probabilities': {'>=2': 0.30, '>=3': 0.07},
                'theoretical_probs': {'>=2': 0.30, '>=3': 0.07},
                'profit_roi': -0.2,
                'random_profit_roi': -0.3,
                'return_multiple': 0.8,
            },
            'select_5',
        )

        self.assertGreater(hit_score, mean_score)
        self.assertGreater(hit_detail['hit_rate_score'], mean_detail['hit_rate_score'])

    def test_seeded_random_validation_candidates_are_present(self):
        candidates = kl8_module.VALIDATION_CANDIDATES

        self.assertIn('random_shape_50', candidates)
        self.assertIn('random_prize_floor_100', candidates)
        self.assertEqual(candidates['random_shape_50']['feature_weights'], {'seeded_random': 1.0})
        self.assertEqual(candidates['random_prize_floor_100']['final_selection_mode'], 'prize_floor')

    def test_per_play_tournament_final_test_uses_strategy_pool_config(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(820, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        captured = []

        def fake_rolling(feature_weights, model_weights, **kwargs):
            captured.append(kwargs)
            return {
                'select_5': {
                    'lift': 0.1,
                    'mean_hits': 1.4,
                    'probabilities': {'>=3': 0.12, '>=4': 0.03},
                    'theoretical_probs': {'>=3': 0.1, '>=4': 0.02},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                },
                'fu_shi_7': {
                    'pool_mean_hits': 1.9,
                    'pool_expected_random': 1.75,
                    'probabilities': {'>=3': 0.25},
                    'theoretical_probs': {'>=3': 0.2},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                },
            }

        def fake_permutation(*args, **kwargs):
            return {'p_value': 0.01}

        original_activate = kl8_module.activate_verified_strategy
        original_persist = kl8_module._persist_trial_results
        original_trials = kl8_module.STRATEGY_TRIAL_RESULTS

        try:
            backtest._rolling_backtest_parametric = fake_rolling
            backtest._permutation_test = fake_permutation
            kl8_module.activate_verified_strategy = lambda *args, **kwargs: None
            kl8_module._persist_trial_results = lambda: None
            kl8_module.STRATEGY_TRIAL_RESULTS = []

            result = backtest.run_candidate_tournament_per_play_type(
                'select_5',
                candidate_strategies={
                    'pool_limited': {
                        'strategy_id': 'pool_limited',
                        'feature_weights': {'frequency': 1.0},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': False,
                        'pool_max_last_numbers': 2,
                    },
                },
                n_permutations=1,
            )
        finally:
            kl8_module.activate_verified_strategy = original_activate
            kl8_module._persist_trial_results = original_persist
            kl8_module.STRATEGY_TRIAL_RESULTS = original_trials

        self.assertTrue(result.get('activated'))
        final_calls = [c for c in captured if c['start_idx'] == 600 and c['end_idx'] == 820]
        self.assertTrue(final_calls)
        self.assertTrue(all(c['pool_diversify'] is False for c in final_calls))
        self.assertTrue(all(c['pool_max_last_numbers'] == 2 for c in final_calls))

    def test_legacy_tournament_validation_uses_each_strategy_pool_config(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(700, 0, -1)]
        analyzer.using_simulated_data = False
        backtest = KL8RollingBacktest(analyzer)

        validation_calls = []

        def fake_rolling(feature_weights, model_weights, **kwargs):
            if kwargs['start_idx'] == 200 and kwargs['end_idx'] == 500:
                validation_calls.append((feature_weights.get('marker'), kwargs))
            return {
                f'select_{pick}': {
                    'lift': 0.1,
                    'mean_hits': 1.0,
                    'probabilities': {},
                    'theoretical_probs': {},
                    'profit_roi': -0.4,
                    'random_profit_roi': -0.5,
                }
                for pick in [3, 4, 5, 6, 7]
            }

        def fake_permutation(*args, **kwargs):
            return {'p_value': 1.0}

        original_persist = kl8_module._persist_trial_results
        original_trials = kl8_module.STRATEGY_TRIAL_RESULTS

        try:
            backtest._rolling_backtest_parametric = fake_rolling
            backtest._permutation_test = fake_permutation
            kl8_module._persist_trial_results = lambda: None
            kl8_module.STRATEGY_TRIAL_RESULTS = []

            result = backtest.run_candidate_tournament(
                candidate_strategies={
                    'first': {
                        'strategy_id': 'first',
                        'feature_weights': {'frequency': 1.0, 'marker': 'first'},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': False,
                        'pool_max_last_numbers': 2,
                    },
                    'second': {
                        'strategy_id': 'second',
                        'feature_weights': {'frequency': 1.0, 'marker': 'second'},
                        'model_weights': {'rank': 1.0},
                        'window_size': 50,
                        'pool_diversify': True,
                        'pool_max_last_numbers': 7,
                    },
                },
                n_permutations=1,
            )
        finally:
            kl8_module._persist_trial_results = original_persist
            kl8_module.STRATEGY_TRIAL_RESULTS = original_trials

        self.assertNotIn('error', result)
        by_marker = {marker: kwargs for marker, kwargs in validation_calls}
        self.assertFalse(by_marker['first']['pool_diversify'])
        self.assertEqual(by_marker['first']['pool_max_last_numbers'], 2)
        self.assertTrue(by_marker['second']['pool_diversify'])
        self.assertEqual(by_marker['second']['pool_max_last_numbers'], 7)

    def test_settlement_includes_select_8_9_10_and_fushi_10_11(self):
        analyzer = KL8Analyzer.__new__(KL8Analyzer)
        analyzer.history_data = [_record(i) for i in range(20, 0, -1)]
        analyzer.using_simulated_data = False

        snapshot = {
            'snapshot_id': 'settle-new-plays',
            'based_on_issue': '2026001',
            'select_8': list(range(1, 9)),
            'select_9': list(range(1, 10)),
            'select_10': list(range(1, 11)),
            'fu_shi_10_11': list(range(1, 12)),
            'play_strategies': {},
            'prediction_modes': {},
            'resolved_strategies': {},
        }

        original_snapshot_dir = kl8_module.KL8_SNAPSHOT_DIR
        original_settlement_dir = kl8_module.KL8_SETTLEMENT_DIR

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / 'snapshots'
            settlement_dir = Path(tmp) / 'settlements'
            snapshot_dir.mkdir()
            settlement_dir.mkdir()
            snapshot_path = snapshot_dir / 'snapshot_new_plays.json'
            snapshot_path.write_text(json.dumps(snapshot), encoding='utf-8')

            try:
                kl8_module.KL8_SNAPSHOT_DIR = str(snapshot_dir)
                kl8_module.KL8_SETTLEMENT_DIR = str(settlement_dir)

                result = analyzer.settle_prediction(
                    snapshot_path.name,
                    '2026002',
                    list(range(1, 21)),
                )
            finally:
                kl8_module.KL8_SNAPSHOT_DIR = original_snapshot_dir
                kl8_module.KL8_SETTLEMENT_DIR = original_settlement_dir

        self.assertTrue(result['success'])
        settlement = result['settlement']

        self.assertEqual(settlement['hit_select_8'], 8)
        self.assertEqual(settlement['hit_select_9'], 9)
        self.assertEqual(settlement['hit_select_10'], 10)
        self.assertEqual(settlement['prize_settlement']['select_8']['prize'], 1000000)
        self.assertEqual(settlement['prize_settlement']['select_9']['prize'], 3000000)
        self.assertEqual(settlement['prize_settlement']['select_10']['prize'], 5000000)

        fushi = settlement['fushi_settlement']['fu_shi_10_11']
        self.assertTrue(fushi['placed'])
        self.assertEqual(fushi['pool_hits'], 11)
        self.assertEqual(fushi['max_combo_hits'], 10)
        self.assertEqual(fushi['total_combinations'], 11)
        self.assertEqual(fushi['total_bet'], 22)
        self.assertEqual(fushi['total_prize'], 55000000)
        self.assertEqual(fushi['hit_distribution'], {10: 11})

    def test_recent_settlement_performance_compares_random_baseline(self):
        original_settlement_dir = kl8_module.KL8_SETTLEMENT_DIR

        with tempfile.TemporaryDirectory() as tmp:
            settlement_dir = Path(tmp) / 'settlements'
            settlement_dir.mkdir()
            rows = [
                {
                    'snapshot_id': 'perf-1',
                    'settled_at': '2026-01-02T00:00:00',
                    'prize_settlement': {
                        'select_5': {'placed': True, 'hits': 3, 'bet': 2, 'prize': 10},
                    },
                    'fushi_settlement': {
                        'fu_shi_7': {'placed': True, 'pool_hits': 2, 'total_bet': 42, 'total_prize': 20},
                    },
                },
                {
                    'snapshot_id': 'perf-2',
                    'settled_at': '2026-01-01T00:00:00',
                    'prize_settlement': {
                        'select_5': {'placed': True, 'hits': 1, 'bet': 2, 'prize': 0},
                    },
                    'fushi_settlement': {
                        'fu_shi_7': {'placed': True, 'pool_hits': 4, 'total_bet': 42, 'total_prize': 100},
                    },
                },
            ]
            for row in rows:
                path = settlement_dir / f"settlement_{row['snapshot_id']}.json"
                path.write_text(json.dumps(row), encoding='utf-8')

            try:
                kl8_module.KL8_SETTLEMENT_DIR = str(settlement_dir)
                result = kl8_module._build_recent_settlement_performance(windows=(2,))
            finally:
                kl8_module.KL8_SETTLEMENT_DIR = original_settlement_dir

        self.assertEqual(result['available_count'], 2)
        window = result['windows'][0]
        self.assertEqual(window['settled_count'], 2)
        select5 = window['play_stats']['select_5']
        self.assertEqual(select5['settled_count'], 2)
        self.assertEqual(select5['avg_hits'], 2.0)
        self.assertEqual(select5['random_expected_hits'], 1.25)
        self.assertEqual(select5['hit_delta_vs_random'], 0.75)
        self.assertEqual(select5['profit_roi'], 1.5)
        fushi7 = window['play_stats']['fu_shi_7']
        self.assertEqual(fushi7['avg_hits'], 3.0)
        self.assertEqual(fushi7['random_expected_hits'], 1.75)

    def test_strategy_health_combines_validation_and_recent_settlements(self):
        original_strategies = kl8_module.ACTIVE_STRATEGIES
        try:
            kl8_module.ACTIVE_STRATEGIES = {
                key: {'strategy_id': '', 'feature_weights': {}, 'model_weights': {}, 'window_size': 0}
                for key in list(kl8_module.SELECT_PLAY_KEYS) + list(kl8_module.FUSHI_PLAY_KEYS)
            }
            kl8_module.ACTIVE_STRATEGIES['select_5'] = {
                'strategy_id': 'select_5_good',
                'is_validated': True,
                'validation_report': {'validation_lift': 0.2, 'final_test_lift': 0.1},
            }
            kl8_module.ACTIVE_STRATEGIES['select_6'] = {
                'strategy_id': 'select_6_weak',
                'is_validated': True,
                'validation_report': {'validation_lift': -0.1, 'final_test_lift': -0.05},
            }

            performance = {
                'available_count': 30,
                'windows': [
                    {
                        'window_size': 30,
                        'settled_count': 30,
                        'play_stats': {
                            'select_5': {
                                'settled_count': 30,
                                'hit_delta_vs_random': 0.35,
                                'profit_roi': 0.1,
                            },
                            'select_6': {
                                'settled_count': 30,
                                'hit_delta_vs_random': -0.35,
                                'profit_roi': -0.9,
                            },
                        },
                    }
                ],
            }

            result = kl8_module._build_strategy_health(performance)
        finally:
            kl8_module.ACTIVE_STRATEGIES = original_strategies

        health = result['health_by_play']
        self.assertEqual(health['select_5']['status'], 'healthy')
        self.assertEqual(health['select_6']['status'], 'cool_down')
        self.assertEqual(health['select_10']['status'], 'unverified')
        self.assertEqual(result['window_size'], 30)


if __name__ == '__main__':
    unittest.main()
