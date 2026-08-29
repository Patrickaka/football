import unittest
from unittest.mock import patch

from src.api.services import football as service


class ServerFootballExportTests(unittest.TestCase):
    def test_export_uses_full_prediction_snapshots(self):
        full_record = {
            'match_id': 'full-1',
            'settled': True,
            'model_version': 'football-test',
            'predicted_scores': {'2-1': 0.2},
            'predicted_1x2': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'result_quality': {'grade': 'high'},
        }
        with patch(
            'src.football.result_sync.get_prediction_export',
            return_value={'records': [full_record], 'stats': {'valid_1x2_predictions': 1}},
        ), patch(
            'src.football.result_sync.get_sync_status_summary', return_value={'settled': 1}
        ), patch.object(
            service, 'football_diagnostics_payload',
            return_value={'result': {'available_samples': 1}}
        ):
            payload = service.predictions_export_payload()['result']

        self.assertEqual(payload['records'][0]['predicted_1x2']['H'], 0.55)
        self.assertEqual(payload['stats']['valid_1x2_predictions'], 1)
        self.assertEqual(payload['model_versions'], ['football-test'])


if __name__ == '__main__':
    unittest.main()
