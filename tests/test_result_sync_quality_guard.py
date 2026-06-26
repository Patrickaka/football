import unittest
from datetime import datetime

import src.football.result_sync as result_sync
from src.football.bayesian_calibration import BayesianCalibrator
from src.football.backtest import (
    _objective_score,
    apply_diagnostic_tuning,
    build_diagnostic_tuning_plan,
    rolling_backtest_report,
    run_backtest_report,
)
from src.football.goal_count_calibrator import GoalCountCalibrator
from src.football.half_time_stats import HalfTimeStatsDB
from src.football.result_sync import (
    PredictionHistory,
    _assess_result_quality,
    _calibration_sample_weight,
    _is_match_settle_due,
    _parse_match_datetime,
    time_layer_weight,
)
from src.football.sample_quality import assess_record_quality


class ResultSyncQualityGuardTests(unittest.TestCase):
    def test_mmdd_future_date_stays_current_year(self):
        parsed = _parse_match_datetime('06-27 11:00')

        self.assertEqual(parsed.year, datetime.now().year)
        self.assertEqual(parsed.month, 6)
        self.assertEqual(parsed.day, 27)

    def test_future_match_is_not_settle_due(self):
        self.assertFalse(_is_match_settle_due(
            '06-27 11:00',
            now=datetime(2026, 6, 25, 10, 0),
        ))

    def test_update_result_rejects_future_match(self):
        history = PredictionHistory()
        history._save = lambda: None
        history.records = [{
            'match_id': 'future-1',
            'home': '埃及',
            'away': '伊朗',
            'match_time': '06-27 11:00',
            'settled': False,
            'sync_status': 'pending',
        }]

        self.assertFalse(history.update_result('future-1', '1-1', 'D', source='live_fid'))
        self.assertFalse(history.records[0]['settled'])
        self.assertEqual(history.records[0]['sync_status'], 'pending')
        self.assertIsNone(history.records[0].get('actual_score'))

    def test_repair_future_settlements_resets_bad_record(self):
        history = PredictionHistory()
        history._save = lambda: None
        history.records = [{
            'match_id': 'future-2',
            'home': '乌拉圭',
            'away': '西班牙',
            'match_time': '06-27 08:00',
            'actual_score': '0-0',
            'actual_result': 'D',
            'settled': True,
            'sync_status': 'synced',
        }]

        result = history.repair_future_settlements()

        self.assertEqual(result['repaired'], 1)
        self.assertFalse(history.records[0]['settled'])
        self.assertEqual(history.records[0]['sync_status'], 'pending')
        self.assertIsNone(history.records[0]['actual_score'])

    def test_prediction_records_hide_future_settlement(self):
        original_history = result_sync._global_history
        history = PredictionHistory()
        history._save = lambda: None
        history.records = [{
            'match_id': 'future-3',
            'home': '新西兰',
            'away': '比利时',
            'match_time': '06-27 11:00',
            'actual_score': '1-1',
            'actual_result': 'D',
            'settled': True,
            'sync_status': 'synced',
            'predicted_scores': {'1-1': 0.2},
        }]
        try:
            result_sync._global_history = history
            rows = result_sync.get_prediction_records(include_hidden=True)
        finally:
            result_sync._global_history = original_history

        self.assertEqual(rows[0]['sync_status'], 'pending')
        self.assertFalse(rows[0]['settled'])
        self.assertIsNone(rows[0]['actual_score'])

    def test_time_layer_stats_do_not_backfill_missing_layers(self):
        history = PredictionHistory()
        history._save = lambda: None
        history.records = [{
            'match_id': 'layer-1',
            'settled': True,
            'actual_score': '1-0',
            'actual_result': 'H',
            'predicted_scores': {'1-0': 0.6, '0-0': 0.4},
            'predicted_1x2': {'H': 0.6, 'D': 0.3, 'A': 0.1},
            'time_layers': {
                'T-1h': {'0-0': 0.7, '1-0': 0.3},
                'final': {'1-0': 0.6, '0-0': 0.4},
            },
        }]

        stats = history.get_stats()

        self.assertEqual(stats['by_time_layer']['T-24h']['total'], 0)
        self.assertEqual(stats['by_time_layer']['T-6h']['total'], 0)
        self.assertEqual(stats['by_time_layer']['T-1h']['total'], 1)
        self.assertEqual(stats['by_time_layer']['final']['total'], 1)
        self.assertEqual(stats['by_time_layer']['T-1h']['weight'], time_layer_weight('T-1h'))
        self.assertAlmostEqual(stats['by_time_layer']['T-1h']['weighted_total'], 0.75)

    def test_time_layer_weight_increases_toward_kickoff(self):
        self.assertLess(time_layer_weight('T-24h'), time_layer_weight('T-6h'))
        self.assertLess(time_layer_weight('T-6h'), time_layer_weight('T-1h'))
        self.assertLess(time_layer_weight('T-1h'), time_layer_weight('T-15min'))
        self.assertLess(time_layer_weight('T-15min'), time_layer_weight('final'))

    def test_audit_prediction_history_reports_without_repairing_by_default(self):
        history = PredictionHistory()
        history._save = lambda: None
        history.records = [{
            'match_id': 'audit-future',
            'home': 'A',
            'away': 'B',
            'match_time': '06-27 11:00',
            'actual_score': '1-1',
            'actual_result': 'D',
            'settled': True,
            'sync_status': 'synced',
            'predicted_scores': {'1-1': 0.2},
            'predicted_1x2': {'D': 0.5},
        }]

        result = history.audit_prediction_history(repair=False)

        self.assertEqual(result['issue_counts']['future_settlement'], 1)
        self.assertTrue(history.records[0]['settled'])
        self.assertEqual(history.records[0]['actual_score'], '1-1')

    def test_audit_prediction_history_repair_marks_unsafe_samples(self):
        history = PredictionHistory()
        saved = {'called': False}
        history._save = lambda: saved.__setitem__('called', True)
        history.records = [
            {
                'match_id': 'audit-future-repair',
                'home': 'A',
                'away': 'B',
                'match_time': '06-27 11:00',
                'actual_score': '1-1',
                'actual_result': 'D',
                'settled': True,
                'sync_status': 'synced',
            },
            {
                'match_id': 'audit-low',
                'home': 'C',
                'away': 'D',
                'match_time': '2026-06-20 11:00',
                'actual_score': '2-1',
                'actual_result': 'H',
                'actual_half_score': 'bad',
                'actual_half_result': 'D',
                'actual_half_full': 'DH',
                'half_time_data_quality': 'invalid',
                'settled': True,
                'sync_status': 'synced',
                'predicted_scores': {'2-1': 0.2},
                'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
                'asian': 0,
                'total_line': 2.5,
                'odds_snapshot': {'x': 1},
                'result_quality': {'grade': 'low', 'source': 'shuju'},
            },
        ]

        result = history.audit_prediction_history(repair=True)

        self.assertTrue(saved['called'])
        self.assertEqual(result['repaired_count'], 3)
        self.assertFalse(history.records[0]['settled'])
        self.assertIsNone(history.records[0]['actual_score'])
        self.assertTrue(history.records[1]['exclude_from_calibration'])
        self.assertEqual(history.records[1]['half_time_data_quality'], 'missing')
        self.assertIsNone(history.records[1]['actual_half_score'])

    def test_excluded_record_has_zero_calibration_weight(self):
        weight = _calibration_sample_weight({
            'exclude_from_calibration': True,
            'settled': True,
            'actual_score': '2-1',
            'predicted_scores': {'2-1': 0.2},
            'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
            'asian': 0,
            'total_line': 2.5,
            'odds_snapshot': {'x': 1},
            'result_quality': {'grade': 'high', 'source': 'live_fid'},
        })

        self.assertEqual(weight, 0.0)

    def test_result_quality_marks_low_information_shuju_score(self):
        quality = _assess_result_quality(
            {'match_id': 'past-1', 'match_time': '2026-06-20 11:00'},
            '1-1',
            'D',
            source='shuju',
        )

        self.assertEqual(quality['grade'], 'medium')
        self.assertIn('low_information_score_without_live_source', quality['reasons'])

    def test_sample_quality_penalizes_low_result_quality(self):
        quality = assess_record_quality({
            'match_id': 'past-2',
            'settled': True,
            'actual_score': '2-1',
            'predicted_scores': {'2-1': 0.2},
            'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
            'asian': -0.25,
            'total_line': 2.5,
            'odds_snapshot': {'x': 1},
            'result_quality': {'grade': 'low'},
        })

        self.assertIn('result_quality_low', quality['reasons'])
        self.assertFalse(quality['usable_for_calibration'])
        self.assertEqual(quality['calibration_weight'], 0.0)

    def test_sample_quality_weights_result_sources(self):
        base = {
            'match_id': 'past-source',
            'settled': True,
            'actual_score': '2-1',
            'predicted_scores': {'2-1': 0.2},
            'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
            'asian': -0.25,
            'total_line': 2.5,
            'odds_snapshot': {'x': 1},
        }
        live = assess_record_quality({
            **base,
            'result_quality': {'grade': 'high', 'source': 'live_fid'},
        })
        shuju = assess_record_quality({
            **base,
            'result_quality': {'grade': 'high', 'source': 'shuju'},
        })

        self.assertGreater(live['calibration_weight'], shuju['calibration_weight'])
        self.assertEqual(live['source_weight'], 1.0)
        self.assertEqual(shuju['source_weight'], 0.6)

    def test_result_sync_uses_sample_quality_weight(self):
        weight = _calibration_sample_weight({
            'settled': True,
            'actual_score': '2-1',
            'predicted_scores': {'2-1': 0.2},
            'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2},
            'asian': -0.25,
            'total_line': 2.5,
            'odds_snapshot': {'x': 1},
            'result_quality': {'grade': 'high', 'source': 'shuju'},
        })

        self.assertEqual(weight, 0.6)

    def test_bayesian_calibrator_tracks_weighted_samples(self):
        calibrator = BayesianCalibrator()
        calibrator.history = {}

        calibrator.add_record('1-1', 0.2, True, 'Test', 2.5, 0.0, sample_weight=1.0)
        calibrator.add_record('1-1', 0.2, False, 'Test', 2.5, 0.0, sample_weight=0.5)

        key = calibrator._get_bucket_key('1-1', 'Test', 2.5, 0.0, 1)
        bucket = calibrator.history[key]

        self.assertEqual(bucket['count'], 2)
        self.assertAlmostEqual(bucket['weighted_count'], 1.5)
        self.assertAlmostEqual(bucket['weighted_success'], 1.0)

    def test_goal_count_calibrator_uses_weighted_effective_samples(self):
        calibrator = GoalCountCalibrator()
        calibrator.db = {}

        dist = {2: 0.7, 3: 0.3}
        for _ in range(9):
            calibrator.record_result('Test', 2.5, dist, 2, 2.3, 0.0, sample_weight=1.0)
        calibrator.record_result('Test', 2.5, dist, 3, 2.3, 0.0, sample_weight=0.2)

        key = calibrator._get_bucket_key('Test', 2.5, 0.0, 2.3)
        bucket = calibrator.db[key]

        self.assertEqual(bucket['sample_count'], 10)
        self.assertAlmostEqual(bucket['weighted_sample_count'], 9.2)
        self.assertEqual(bucket['calibration_factors'], {})

    def test_half_time_stats_uses_weighted_distribution(self):
        db = HalfTimeStatsDB()
        db.db = {}

        db.record_match('Test', 2.5, 0.0, 'league', 1, 0, 2, 0, sample_weight=1.0)
        db.record_match('Test', 2.5, 0.0, 'league', 0, 0, 1, 1, sample_weight=0.5)
        stats = db.get_stats('Test', 2.5, 0.0, 'league', min_samples=1)

        self.assertAlmostEqual(stats['effective_sample_count'], 1.5)
        self.assertAlmostEqual(stats['home_lead_at_half_rate'], 0.667, places=3)
        self.assertAlmostEqual(stats['first_half_draw_rate'], 0.333, places=3)

    def test_half_time_stats_falls_back_to_nearest_bucket(self):
        db = HalfTimeStatsDB()
        db.db = {}

        for _ in range(12):
            db.record_match('Test', 2.5, 0.0, 'league', 0, 0, 1, 1, sample_weight=1.0)

        stats = db.get_nearest_stats('Test', 2.75, 0.25, 'league', min_samples=10, max_distance=0.75)

        self.assertIsNotNone(stats)
        self.assertEqual(stats['_meta']['source'], 'nearest')
        self.assertEqual(stats['_meta']['league'], 'Test')
        self.assertAlmostEqual(stats['_meta']['distance'], 0.375, places=3)
        self.assertGreater(stats['half_full_distribution']['DD'], 0)

    def test_half_time_stats_nearest_prefers_same_league(self):
        db = HalfTimeStatsDB()
        db.db = {}

        for _ in range(12):
            db.record_match('Other', 2.5, 0.0, 'league', 1, 0, 2, 0, sample_weight=1.0)
            db.record_match('Test', 2.0, 0.0, 'league', 0, 0, 1, 1, sample_weight=1.0)

        stats = db.get_nearest_stats('Test', 2.5, 0.0, 'league', min_samples=10, max_distance=0.75)

        self.assertIsNotNone(stats)
        self.assertEqual(stats['_meta']['league'], 'Test')
        self.assertEqual(stats['_meta']['bucket_key'], 'Test_league_2.00_0.00')

    def test_backtest_reports_goal_distribution_quality(self):
        report = run_backtest_report([{
            'match_id': 'goal-1',
            'league': 'Test League',
            'home': 'A',
            'away': 'B',
            'actual_score': '2-1',
            'actual_result': 'H',
            'predicted_scores': {'2-1': 0.20, '1-1': 0.15},
            'predicted_1x2': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'goal_count': {'distribution_dict': {'2': 0.25, '3': 0.50, '4': 0.25}},
            'predicted_half_full': {'HH': 0.40, 'HD': 0.20, 'DH': 0.15},
            'actual_half_full': 'HH',
            'half_time_data_quality': 'real',
            'result_quality': {'grade': 'high'},
            'asian': -0.25,
            'total_line': 2.5,
        }], verbose=False)

        summary = report['summary']
        self.assertEqual(summary['goal_count_total'], 1)
        self.assertGreater(summary['goal_logloss'], 0)
        self.assertGreater(summary['goal_brier'], 0)
        self.assertEqual(summary['htf_total'], 1)
        self.assertLess(
            _objective_score({'goal_count_total': 1, 'goal_logloss': 0.2, 'goal_brier': 0.1, 'hit_rate_total': 1.0}, 'goals'),
            _objective_score({'goal_count_total': 1, 'goal_logloss': 1.0, 'goal_brier': 0.5, 'hit_rate_total': 0.0}, 'goals'),
        )

    def test_backtest_excludes_inferred_half_full_samples(self):
        report = run_backtest_report([{
            'match_id': 'htf-inferred-1',
            'league': 'Test League',
            'home': 'A',
            'away': 'B',
            'actual_score': '1-1',
            'actual_result': 'D',
            'predicted_scores': {'1-1': 0.20},
            'predicted_1x2': {'H': 0.30, 'D': 0.45, 'A': 0.25},
            'goal_count': {'distribution_dict': {2: 1.0}},
            'predicted_half_full': {'DD': 0.60, 'HD': 0.20},
            'actual_half_full': 'DD',
            'half_time_data_quality': 'inferred',
            'result_quality': {'grade': 'high'},
            'asian': 0,
            'total_line': 2.0,
        }], verbose=False)

        self.assertEqual(report['summary']['htf_total'], 0)

    def test_backtest_report_includes_bias_diagnostics(self):
        records = []
        for idx, actual in enumerate(['2-1', '2-0', '3-1', '1-2', '2-2', '3-0']):
            home_goals, away_goals = map(int, actual.split('-'))
            actual_result = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
            records.append({
                'match_id': f'bias-{idx}',
                'league': 'Bias League',
                'home': 'A',
                'away': 'B',
                'actual_score': actual,
                'actual_result': actual_result,
                'predicted_scores': {'1-1': 0.35, '1-0': 0.20, '2-1': 0.10},
                'predicted_1x2': {'H': 0.40, 'D': 0.35, 'A': 0.25},
                'goal_count': {'distribution_dict': {2: 0.60, 3: 0.25, 4: 0.15}},
                'asian': 0,
                'total_line': 2.5,
            })

        report = run_backtest_report(records, verbose=False)
        diagnostics = report['diagnostics']

        self.assertIn('common_scores_overheated', diagnostics['notes'])
        self.assertIn('draw', diagnostics)
        self.assertIn('weak_buckets', diagnostics)
        self.assertIn('bucket_tuning_candidates', diagnostics)
        self.assertTrue(diagnostics['bucket_tuning_candidates'])
        self.assertIn('param_deltas', diagnostics['bucket_tuning_candidates'][0])
        self.assertIn('diagnostic_suggestions', report)
        self.assertTrue(report['diagnostic_suggestions']['suggestions'])
        self.assertTrue(report['diagnostic_suggestions']['bucket_tuning_candidates'])

    def test_backtest_report_includes_real_time_layer_metrics(self):
        report = run_backtest_report([{
            'match_id': 'layer-backtest-1',
            'league': 'Layer League',
            'home': 'A',
            'away': 'B',
            'actual_score': '1-0',
            'actual_result': 'H',
            'predicted_scores': {'1-0': 0.55, '0-0': 0.30, '1-1': 0.15},
            'predicted_1x2': {'H': 0.55, 'D': 0.30, 'A': 0.15},
            'time_layers': {
                'T-1h': {'0-0': 0.55, '1-0': 0.30, '1-1': 0.15},
                'final': {'1-0': 0.55, '0-0': 0.30, '1-1': 0.15},
            },
            'asian': 0,
            'total_line': 2.0,
        }], verbose=False)

        by_layer = report['by_time_layer']
        self.assertEqual(by_layer['T-24h']['total'], 0)
        self.assertEqual(by_layer['T-1h']['total'], 1)
        self.assertEqual(by_layer['final']['total'], 1)
        self.assertEqual(by_layer['T-1h']['top1'], 0.0)
        self.assertEqual(by_layer['T-1h']['top3'], 1.0)
        self.assertEqual(by_layer['final']['top1'], 1.0)
        self.assertAlmostEqual(by_layer['T-1h']['weighted_total'], 0.75)

    def test_backtest_diagnostics_recommends_late_layer_weight_when_stronger(self):
        records = []
        actual_scores = ['1-0', '2-1', '1-1', '2-0', '0-1', '3-1', '1-0', '2-1']
        for idx, actual in enumerate(actual_scores):
            home_goals, away_goals = map(int, actual.split('-'))
            actual_result = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
            early_scores = {'0-0': 0.50, '0-1': 0.30, '1-1': 0.20}
            late_scores = {actual: 0.55, '0-0': 0.25, '1-1': 0.20}
            records.append({
                'match_id': f'late-layer-{idx}',
                'league': 'Layer League',
                'home': 'A',
                'away': 'B',
                'actual_score': actual,
                'actual_result': actual_result,
                'predicted_scores': late_scores,
                'predicted_1x2': {'H': 0.45, 'D': 0.30, 'A': 0.25},
                'time_layers': {
                    'T-24h': early_scores,
                    'T-6h': early_scores,
                    'T-1h': late_scores,
                    'T-15min': late_scores,
                    'final': late_scores,
                },
                'asian': 0,
                'total_line': 2.5,
            })

        report = run_backtest_report(records, verbose=False)
        signal = report['diagnostics']['time_layer_signal']

        self.assertTrue(signal['available'])
        self.assertEqual(signal['action'], 'raise_late_market_weight')
        self.assertGreater(signal['top3_lift'], 0)
        self.assertEqual(report['diagnostic_suggestions']['time_layer_signal']['action'], 'raise_late_market_weight')

    def test_rolling_backtest_report_includes_recent_windows_and_suggestions(self):
        records = []
        actual_scores = ['2-1', '2-0', '3-1', '1-2', '2-2', '3-0', '2-1', '4-1']
        for idx, actual in enumerate(actual_scores):
            home_goals, away_goals = map(int, actual.split('-'))
            actual_result = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
            records.append({
                'match_id': f'rolling-{idx}',
                'league': 'Rolling League',
                'home': 'A',
                'away': 'B',
                'actual_score': actual,
                'actual_result': actual_result,
                'predicted_scores': {'1-1': 0.36, '0-0': 0.19, '2-1': 0.12},
                'predicted_1x2': {'H': 0.38, 'D': 0.40, 'A': 0.22},
                'goal_count': {'distribution_dict': {2: 0.65, 3: 0.20, 4: 0.15}},
                'asian': 0,
                'total_line': 2.5,
                'result_quality': {'grade': 'high'},
            })

        report = rolling_backtest_report(
            records,
            windows=(3, 6),
            verbose=False,
            quality_filter=False,
        )

        self.assertEqual(report['available_samples'], len(records))
        self.assertEqual(report['latest_window'], '6')
        self.assertIn('3', report['windows'])
        self.assertIn('6', report['windows'])
        self.assertEqual(report['windows']['3']['sample_count'], 3)
        self.assertTrue(report['diagnostic_suggestions']['suggestions'])

    def test_diagnostic_tuning_plan_requires_consistent_windows(self):
        report = {
            'windows': {
                '3': {
                    'sample_count': 3,
                    'diagnostic_suggestions': {'param_deltas': {'draw_bias': -0.03}},
                },
                '6': {
                    'sample_count': 6,
                    'diagnostic_suggestions': {'param_deltas': {'draw_bias': -0.05}},
                },
            }
        }

        plan = build_diagnostic_tuning_plan(
            report,
            min_consistent_windows=2,
            min_window_samples=3,
            max_abs_delta=0.04,
        )

        self.assertTrue(plan['ready'])
        self.assertEqual(plan['param_deltas']['draw_bias'], -0.04)

        conflicting = {
            'windows': {
                '3': {
                    'sample_count': 3,
                    'diagnostic_suggestions': {'param_deltas': {'draw_bias': -0.03}},
                },
                '6': {
                    'sample_count': 6,
                    'diagnostic_suggestions': {'param_deltas': {'draw_bias': 0.03}},
                },
            }
        }

        blocked = build_diagnostic_tuning_plan(
            conflicting,
            min_consistent_windows=2,
            min_window_samples=3,
        )

        self.assertFalse(blocked['ready'])

    def test_apply_diagnostic_tuning_dry_run_builds_new_params_without_saving(self):
        records = []
        actual_scores = ['2-1', '2-0', '3-1', '1-2', '2-2', '3-0', '2-1', '4-1']
        for idx, actual in enumerate(actual_scores):
            home_goals, away_goals = map(int, actual.split('-'))
            actual_result = 'H' if home_goals > away_goals else 'A' if home_goals < away_goals else 'D'
            records.append({
                'match_id': f'apply-diagnostic-{idx}',
                'league': 'Apply League',
                'home': 'A',
                'away': 'B',
                'actual_score': actual,
                'actual_result': actual_result,
                'predicted_scores': {'1-1': 0.36, '0-0': 0.19, '2-1': 0.12},
                'predicted_1x2': {'H': 0.38, 'D': 0.40, 'A': 0.22},
                'goal_count': {'distribution_dict': {2: 0.65, 3: 0.20, 4: 0.15}},
                'asian': 0,
                'total_line': 2.5,
                'result_quality': {'grade': 'high'},
            })

        result = apply_diagnostic_tuning(
            records,
            windows=(3, 6),
            quality_filter=False,
            min_consistent_windows=2,
            min_window_samples=3,
            dry_run=True,
        )

        self.assertFalse(result['applied'])
        self.assertTrue(result['dry_run'])
        self.assertTrue(result['plan']['ready'])
        self.assertIn('new_params', result)
        self.assertIsNone(result['save_result'])


if __name__ == '__main__':
    unittest.main()
