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
            'injuries': [{'team': 'A', 'source': 'provider', 'ts': '2026-07-23T10:00:00+00:00'}],
            'lineup': {'source': 'provider', 'confirmed': True, 'ts': '2026-07-23T11:00:00+00:00'},
            'possession': {'home': .55},
        }
        result = assess_live_context(context, now=now, require_confirmed_lineup=True)
        self.assertTrue(result['official_bet_allowed'])
        self.assertEqual(result['checks']['freshness'], 'passed')

    def test_fresh_lineup_does_not_hide_stale_injury_source(self):
        now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
        result = assess_live_context({
            'injuries': [{'source': 'provider', 'ts': '2026-07-20T12:00:00Z'}],
            'lineup': {'source': 'provider', 'confirmed': True, 'ts': '2026-07-23T11:00:00Z'},
        }, now=now)
        self.assertFalse(result['official_bet_allowed'])
        self.assertEqual(result['checks']['freshness'], 'stale')
        self.assertEqual(result['age_hours'], 72.0)

    def test_missing_future_and_unavailable_sources_do_not_verify(self):
        now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
        for entry in [{'ts': '2026-07-23T11:00:00Z'},
                      {'source': 'UNAVAILABLE', 'ts': '2026-07-23T11:00:00Z'},
                      {'source': 'provider', 'ts': '2026-07-24T11:00:00Z'},
                      {'source': 'provider', 'ts': '2026-07-23T11:00:00'},
                      {'source': 'provider', 'ts': 'bad'}, 'not-a-record']:
            with self.subTest(entry=entry):
                result = assess_live_context({'injuries': [entry], 'lineup': {
                    'source': 'provider', 'confirmed': True, 'ts': '2026-07-23T11:00:00Z',
                }}, now=now)
                self.assertFalse(result['official_bet_allowed'])
                self.assertFalse(result['source_audit']['injuries'][0]['verified'])

    def test_unconfirmed_lineup_cannot_pass_even_when_recent_and_sourced(self):
        now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
        result = assess_live_context({
            'injuries': [{'source': 'provider', 'ts': '2026-07-23T11:00:00Z'}],
            'lineup': {'source': 'provider', 'confirmed': False, 'ts': '2026-07-23T11:00:00Z'},
        }, now=now)
        self.assertFalse(result['official_bet_allowed'])
        self.assertIn('confirmed_lineup_unverified', result['blockers'])

    def test_one_unverified_injury_cannot_borrow_another_sources_timestamp(self):
        now = datetime(2026, 7, 23, 12, tzinfo=timezone.utc)
        result = assess_live_context({
            'injuries': [{'source': 'provider', 'ts': '2026-07-23T11:00:00Z'}, {'source': 'other'}],
            'lineup': {'source': 'provider', 'confirmed': True, 'ts': '2026-07-23T11:00:00Z'},
        }, now=now)
        self.assertFalse(result['official_bet_allowed'])
        self.assertEqual(result['checks']['freshness'], 'unknown')


if __name__ == '__main__':
    unittest.main()
