import unittest

from src.lottery import LotteryAnalyzer, LOTTERY_PREDICTOR_VERSION


class DltPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LotteryAnalyzer()

    def test_portfolio_exposes_anchor_policy_and_unique_tickets(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        policy = result['portfolio_policy']
        recommendations = result['recommendations']

        self.assertEqual(policy['name'], 'rank_core_rotating_primary_back_coverage')
        self.assertEqual(len(policy['front_anchors']), 2)
        self.assertEqual(len(policy['back_anchors']), 0)
        portfolio = [item for item in recommendations if not item['strategy'].startswith('picked')]
        tickets = {
            (tuple(item['front']), tuple(item['back']))
            for item in portfolio
        }
        self.assertEqual(len(tickets), len(portfolio))
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

    def test_primary_keeps_true_rank_cores_and_rotates_with_issue(self):
        front_ranked, back_ranked = self.analyzer.rank_model(top_n=20)
        first = self.analyzer.generate_multi_strategy_recommendations()
        first_primary = {x['strategy']: x for x in first['recommendations']}['primary_rank']

        original_issue = self.analyzer.history_data[0]['issue']
        self.analyzer.history_data[0]['issue'] = str(int(original_issue) + 1)
        second = self.analyzer.generate_multi_strategy_recommendations()
        second_primary = {x['strategy']: x for x in second['recommendations']}['primary_rank']
        self.analyzer.history_data[0]['issue'] = original_issue

        self.assertTrue(set(n for n, _, _ in front_ranked[:2]).issubset(first_primary['front']))
        self.assertEqual(set(first_primary['core_front']), set(n for n, _, _ in front_ranked[:2]))
        self.assertEqual(set(first_primary['core_back']), {back_ranked[0][0]})
        self.assertNotEqual(
            (first_primary['front'], first_primary['back']),
            (second_primary['front'], second_primary['back']),
        )

    def test_predictor_version_invalidates_old_cache(self):
        self.assertEqual(LOTTERY_PREDICTOR_VERSION, 'dlt-v4.3-balanced-weights')

    def test_single_pick_designates_walk_forward_winner_without_fake_mix(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        by_strategy = {item['strategy']: item for item in result['recommendations']}
        picked = by_strategy['picked_v8']
        primary = by_strategy['primary_rank']

        self.assertEqual(picked['front'], primary['front'])
        self.assertEqual(picked['back'], primary['back'])
        self.assertEqual(picked['selected_from'], 'primary_rank')
        self.assertEqual(picked['validation_evidence']['method'], 'walk_forward')
        self.assertFalse(picked['validation_evidence']['statistically_validated'])


if __name__ == '__main__':
    unittest.main()
