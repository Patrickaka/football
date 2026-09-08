import unittest

from src.domain.sports.football.lottery import lottery_market_probabilities
from src.football.pipeline import build_match_analysis


class FootballDirectionIntegrationTests(unittest.TestCase):
    def test_analyst_uses_verified_common_scenario_and_explicit_probability_bases(self):
        candidates = [((1, 0), .25), ((1, 1), .21), ((0, 1), .24), ((0, 2), .30)]
        lottery = lottery_market_probabilities(candidates, 1)
        analysis = build_match_analysis({'model': {'candidates': candidates}, 'lottery': lottery})
        self.assertIsNotNone(analysis)
        self.assertTrue(lottery['direction_analysis']['available'])
        verdict = analysis['lottery_verdict']
        self.assertIn('客胜整场概率 54.0%', verdict)
        self.assertIn('若客胜成立', verdict)
        self.assertIn('让负条件概率 55.6%', verdict)
        self.assertIn('联合估计 30.0%', verdict)
        self.assertNotIn('首选让胜', verdict)

    def test_old_independent_predictions_never_fill_missing_common_scenario(self):
        candidates = [((1, 0), .25), ((1, 1), .21), ((0, 1), .24), ((0, 2), .30)]
        lottery = lottery_market_probabilities(candidates, 1)
        lottery.pop('direction_analysis')
        analysis = build_match_analysis({'model': {'candidates': candidates}, 'lottery': lottery})
        self.assertEqual(analysis['lottery_verdict'], '统一情景暂不可用，需完整比分分布重新分析')


if __name__ == '__main__':
    unittest.main()
