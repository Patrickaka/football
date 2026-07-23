import unittest

import server


class ServerProfessionalStatusTests(unittest.TestCase):
    def test_status_exposes_honest_production_gate(self):
        handler = server.Handler.__new__(server.Handler)
        handler._log = server.log
        payload = handler._football_professional_status_payload()
        self.assertNotIn('error', payload)
        result = payload['result']
        self.assertEqual(result['schema_version'], 'football-professional-status-v1')
        self.assertIn('disk_healthy', result['checks'])
        self.assertFalse(result['production_ready'])
        self.assertFalse(result['official_betting_allowed'])
        self.assertGreaterEqual(result['out_of_sample_n'], 1000)


if __name__ == '__main__':
    unittest.main()
