import unittest
from datetime import datetime, timezone

from src.football.live_context_quality import assess_live_context


class LiveContextQualityTests(unittest.TestCase):
    def test_missing_required_lineup_blocks_official_bet(self):
        result = assess_live_context({}, require_confirmed_lineup=True)
        self.assertFalse(result['official_bet_allowed'])
        self.assertIn('confirmed_lineup_missing', result['blockers'])
        self.assertLess(result['confidence_multiplier'], 1.0)

    def test_fresh_sourced_context_passes(self):
        now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
        context = {
            'injuries': [{'team': 'A', 'ts': '2026-07-23T10:00:00+00:00'}],
            'lineup': {'source': 'provider', 'ts': '2026-07-23T11:00:00+00:00'},
            'possession': {'home': .55},
        }
        result = assess_live_context(context, now=now, require_confirmed_lineup=True)
        self.assertTrue(result['official_bet_allowed'])
        self.assertEqual(result['checks']['freshness'], 'passed')


if __name__ == '__main__':
    unittest.main()
