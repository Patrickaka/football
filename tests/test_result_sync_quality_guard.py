import unittest
from datetime import datetime

import src.football.result_sync as result_sync
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


if __name__ == '__main__':
    unittest.main()
