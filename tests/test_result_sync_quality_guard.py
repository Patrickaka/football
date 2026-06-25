import unittest
from datetime import datetime

import src.football.result_sync as result_sync
from src.football.backtest import _objective_score, run_backtest_report
from src.football.result_sync import (
    PredictionHistory,
    _assess_result_quality,
    _is_match_settle_due,
    _parse_match_datetime,
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


if __name__ == '__main__':
    unittest.main()
