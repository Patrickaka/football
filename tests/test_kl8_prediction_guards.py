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

        self.assertIn('fu_shi_10_11', result)
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


if __name__ == '__main__':
    unittest.main()
