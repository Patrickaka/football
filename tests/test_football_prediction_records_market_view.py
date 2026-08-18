import unittest
from unittest.mock import patch

from src.football import result_sync


class FootballPredictionRecordsMarketViewTests(unittest.TestCase):
    def test_list_exposes_market_predictions_and_only_actual_score(self):
        record = {
            'match_id': 'market-view-1',
            'home': '主队',
            'away': '客队',
            'match_time': '2020-01-01 12:00',
            'settled': True,
            'sync_status': 'synced',
            'predicted_scores': {'2-1': 0.3},
            'predicted_1x2': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'predicted_rqspf': {'让胜': 0.25, '让平': 0.45, '让负': 0.30},
            'lottery_handicap': -1,
            'actual_score': '2-1',
            'actual_result': 'H',
            'actual_rqspf': '让平',
            'hit_1x2': True,
            'hit_rqspf': True,
        }

        with patch.object(result_sync._global_history, 'records', [record]):
            row = result_sync.get_prediction_records(include_hidden=True)[0]

        self.assertEqual(row['predicted_1x2']['H'], 0.55)
        self.assertEqual(row['predicted_rqspf']['让平'], 0.45)
        self.assertEqual(row['lottery_handicap'], -1)
        self.assertEqual(row['actual_score'], '2-1')
        self.assertTrue(row['hit_1x2'])
        self.assertTrue(row['hit_rqspf'])
        self.assertNotIn('predicted_scores', row)


if __name__ == '__main__':
    unittest.main()
