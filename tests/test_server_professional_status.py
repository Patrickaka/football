import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.api.services import football as service
from src.webapp import caching as webapp_caching
from src.webapp import jobs as webapp_jobs
from src.webapp import lazy_modules as webapp_lazy


class ServerProfessionalStatusTests(unittest.TestCase):
    def test_status_exposes_honest_production_gate(self):
        payload = service.football_professional_status_payload()
        self.assertNotIn('error', payload)
        result = payload['result']
        self.assertEqual(result['schema_version'], 'football-professional-status-v1')
        self.assertIn('disk_healthy', result['checks'])
        self.assertFalse(result['production_ready'])
        self.assertFalse(result['official_betting_allowed'])
        self.assertGreaterEqual(result['out_of_sample_n'], 1000)
        self.assertEqual(
            result['monitoring']['schema_version'],
            'football-professional-monitoring-v1',
        )
        self.assertIn('spf', result['monitoring'])
        self.assertIn('rqspf', result['monitoring'])
        self.assertIn('market_timing', result['monitoring'])

    def test_status_uses_bundled_baseline_without_report_file(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(webapp_jobs, 'REPORTS_DIR', Path(temp_dir)):
                payload = service.football_professional_status_payload()
        result = payload['result']
        self.assertEqual(result['out_of_sample_n'], 1804)
        self.assertEqual(result['validation_source'], 'bundled_audited_baseline')
        self.assertEqual(result['baseline_version'], 'football-oos-2026-07-23-v1')


if __name__ == '__main__':
    unittest.main()
