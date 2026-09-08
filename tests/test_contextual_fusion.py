import unittest
from datetime import datetime, timezone

from src.football.contextual_fusion import apply_contextual_fusion


class ContextualFusionTests(unittest.TestCase):
    def test_qualified_h2h_and_motivation_adjust_distribution(self):
        candidates = [((1, 0), .4), ((1, 1), .35), ((0, 1), .25)]
        now = datetime(2026, 9, 8, 3, tzinfo=timezone.utc)
        adjusted, meta = apply_contextual_fusion(candidates, {
            'h2h': {'games': 8, 'home_wins': 5, 'draws': 2, 'away_wins': 1,
                    'avg_goals': 3.1, 'quality_score': .8,
                    'source': 'match-provider', 'ts': '2026-09-08T02:00:00Z'},
            'motivation': {'home': 1, 'away': 0, 'quality_score': .9,
                           'source': 'official', 'ts': '2026-09-08T02:00:00Z'},
        }, now=now)
        values = dict(adjusted)
        self.assertTrue(meta['applied'])
        self.assertGreater(values[(1, 0)], .4)
        self.assertAlmostEqual(sum(values.values()), 1.0)

    def test_free_text_or_low_quality_context_is_not_used(self):
        candidates = [((1, 0), .6), ((0, 1), .4)]
        adjusted, meta = apply_contextual_fusion(candidates, {
            'style_notes': '主队必须赢',
            'motivation': {'home': 1, 'away': 0, 'quality_score': .2},
        })
        self.assertFalse(meta['applied'])
        self.assertEqual(adjusted, candidates)

    def test_high_quality_label_cannot_replace_source_and_valid_time(self):
        candidates = [((1, 0), .6), ((0, 1), .4)]
        now = datetime(2026, 9, 8, 3, tzinfo=timezone.utc)
        for metadata in [{}, {'source': 'UNAVAILABLE', 'ts': '2026-09-08T02:00:00Z'},
                         {'source': 'provider', 'ts': '2026-09-09T02:00:00Z'},
                         {'source': 'provider', 'ts': '2026-09-01T02:00:00Z'}]:
            with self.subTest(metadata=metadata):
                adjusted, result = apply_contextual_fusion(candidates, {
                    'h2h': {'games': 10, 'home_wins': 8, 'draws': 1, 'away_wins': 1,
                            'avg_goals': 3.1, 'quality_score': .99, **metadata},
                    'motivation': {'home': 1, 'away': 0, 'quality_score': .99, **metadata},
                }, now=now)
                self.assertEqual(adjusted, candidates)
                self.assertFalse(result['applied'])
                self.assertFalse(result['source_audit']['h2h']['verified'])

    def test_sourced_h2h_still_requires_valid_result_counts(self):
        candidates = [((1, 0), .6), ((0, 1), .4)]
        now = datetime(2026, 9, 8, 3, tzinfo=timezone.utc)
        for games, wins in [(10, 20), (float('nan'), 8), (10, float('inf')), (10, True)]:
            with self.subTest(games=games, wins=wins):
                adjusted, result = apply_contextual_fusion(candidates, {'h2h': {
                    'games': games, 'home_wins': wins, 'draws': 1, 'away_wins': 1,
                    'quality_score': .99, 'source': 'provider', 'ts': '2026-09-08T02:00:00Z',
                }}, now=now)
                self.assertEqual(adjusted, candidates)
                self.assertFalse(result['applied'])


if __name__ == '__main__':
    unittest.main()
