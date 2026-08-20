import unittest
from unittest.mock import patch

from src.basketball import _official_pick_status
from src.basketball.records import get_prediction_stats, save_predictions


class BasketballOfficialPickTests(unittest.TestCase):
    def test_low_confidence_pick_is_not_official(self):
        status = _official_pick_status('spf', 0.58, 'low')

        self.assertFalse(status['playable'])
        self.assertFalse(status['official'])
        self.assertEqual(status['skip_reason'], 'low_confidence')

    def test_stats_skip_non_playable_predictions(self):
        records = [{
            'result': {'home_score': 92, 'away_score': 88},
            'spf': {
                'available': True,
                'playable': False,
                'recommendation': '主胜',
            },
            'rqspf': None,
            'dx': None,
        }]

        with patch('src.basketball.records.kv_store.load', return_value=records):
            stats = get_prediction_stats()

        self.assertEqual(stats['settled_count'], 1)
        self.assertEqual(stats['official_predictions'], 0)
        self.assertEqual(stats['spf']['total'], 0)
        self.assertEqual(stats['spf']['correct'], 0)

    def test_stats_count_playable_predictions(self):
        records = [{
            'result': {'home_score': 92, 'away_score': 88},
            'spf': {
                'available': True,
                'playable': True,
                'recommendation': '主胜',
            },
            'rqspf': None,
            'dx': None,
        }]

        with patch('src.basketball.records.kv_store.load', return_value=records):
            stats = get_prediction_stats()

        self.assertEqual(stats['official_predictions'], 1)
        self.assertEqual(stats['spf']['total'], 1)
        self.assertEqual(stats['spf']['correct'], 1)
        self.assertEqual(stats['spf']['accuracy'], 1.0)

    def test_save_predictions_persists_playable_metadata(self):
        saved = []
        matches = [{
            'match': {
                'id': 'm1',
                'league': 'NBA',
                'home': 'Home',
                'away': 'Away',
            },
            'spf': {
                'available': True,
                'recommendation': '主胜',
                'pick_prob': 0.57,
                'playable': False,
                'official': False,
                'skip_reason': 'low_confidence',
                'home_prob': 0.57,
                'away_prob': 0.43,
                'confidence': 'low',
            },
        }]

        with patch('src.basketball.records.kv_store.load', return_value=[]), \
                patch('src.basketball.records.kv_store.save', side_effect=lambda _key, rows: saved.extend(rows)):
            save_predictions('2026-07-14', matches)

        self.assertEqual(saved[0]['spf']['playable'], False)
        self.assertEqual(saved[0]['spf']['official'], False)
        self.assertEqual(saved[0]['spf']['skip_reason'], 'low_confidence')
        self.assertEqual(saved[0]['spf']['pick_prob'], 0.57)

    def test_save_predictions_persists_water_inference_audit_fields(self):
        saved = []
        matches = [{
            'match': {'id': 'm2', 'league': 'NBA', 'home': 'H', 'away': 'A'},
            'rqspf': {
                'available': True, 'recommendation': '让负', 'pick_prob': .58,
                'playable': True, 'official': True, 'movement_led': True,
                'water_inference': {'actionable': True, 'recommendation': '让负'},
            },
            'dx': {
                'available': True, 'recommendation': '小分', 'pick_prob': .57,
                'playable': True, 'official': True, 'movement_led': True,
                'water_inference': {'actionable': True, 'recommendation': '小分'},
            },
        }]
        with patch('src.basketball.records.kv_store.load', return_value=[]), \
                patch('src.basketball.records.kv_store.save', side_effect=lambda _key, rows: saved.extend(rows)):
            save_predictions('2026-08-20', matches)
        self.assertTrue(saved[0]['rqspf']['movement_led'])
        self.assertEqual(saved[0]['rqspf']['water_inference']['recommendation'], '让负')
        self.assertTrue(saved[0]['dx']['movement_led'])

    def test_stats_report_water_led_accuracy_separately(self):
        records = [{
            'result': {'home_score': 100, 'away_score': 95},
            'spf': None,
            'rqspf': {
                'available': True, 'playable': True, 'recommendation': '让负',
                'handicap': -8.5, 'movement_led': True,
            },
            'dx': {
                'available': True, 'playable': True, 'recommendation': '小分',
                'total_line': 210.5, 'movement_led': True,
            },
        }]
        with patch('src.basketball.records.kv_store.load', return_value=records):
            stats = get_prediction_stats()
        self.assertEqual(stats['water_inference']['rqspf']['accuracy'], 1.0)
        self.assertEqual(stats['water_inference']['dx']['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
