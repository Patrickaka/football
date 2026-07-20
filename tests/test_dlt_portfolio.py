import unittest

from src.lottery import LotteryAnalyzer, LOTTERY_PREDICTOR_VERSION


class DltPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LotteryAnalyzer()

    def test_portfolio_exposes_anchor_policy_and_unique_tickets(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        policy = result['portfolio_policy']
        recommendations = result['recommendations']

        self.assertEqual(policy['name'], 'front_anchor_back_full_coverage')
        self.assertEqual(len(policy['front_anchors']), 2)
        self.assertEqual(len(policy['back_anchors']), 0)
        tickets = {
            (tuple(item['front']), tuple(item['back']))
            for item in recommendations
        }
        self.assertEqual(len(tickets), len(recommendations))
        back_numbers = {
            number
            for item in recommendations
            for number in item['back']
        }
        self.assertEqual(len(back_numbers), 10)
        self.assertEqual(result['back_coverage_profile']['unique_number_count'], 10)
        self.assertAlmostEqual(
            result['back_coverage_profile']['at_least_one_group_ge1_probability'],
            65 / 66,
            places=6,
        )
        self.assertAlmostEqual(
            result['back_coverage_profile']['at_least_one_group_ge2_probability'],
            5 / 66,
            places=6,
        )

    def test_rank_alternative_reuses_anchors_but_is_not_primary_clone(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        by_strategy = {item['strategy']: item for item in result['recommendations']}
        primary = by_strategy['primary_rank']
        alternative = by_strategy['rank']
        anchors = set(result['portfolio_policy']['front_anchors'])

        self.assertTrue(anchors.issubset(alternative['front']))
        self.assertNotEqual(primary['front'], alternative['front'])

    def test_predictor_version_invalidates_old_cache(self):
        self.assertEqual(LOTTERY_PREDICTOR_VERSION, 'dlt-v3.9-back-coverage')


if __name__ == '__main__':
    unittest.main()
