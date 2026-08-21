import unittest

from src.lottery import LotteryAnalyzer, LOTTERY_PREDICTOR_VERSION


class DltPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LotteryAnalyzer()

    def test_portfolio_exposes_anchor_policy_and_unique_tickets(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        policy = result['portfolio_policy']
        recommendations = result['recommendations']

        self.assertEqual(policy['name'], 'portfolio_cover_v4.4')
        self.assertEqual(len(policy['front_anchors']), 2)
        self.assertEqual(len(policy['back_anchors']), 0)
        portfolio = [item for item in recommendations if not item['strategy'].startswith('picked')]
        tickets = {
            (tuple(item['front']), tuple(item['back']))
            for item in portfolio
        }
        self.assertEqual(len(tickets), len(portfolio))
        # v4.4 组合覆盖: 5注前区完全不重叠, union=25
        front_union = set()
        for item in portfolio:
            front_union.update(item['front'])
        self.assertEqual(len(front_union), 25)
        self.assertEqual(policy['front_union'], 25)
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

    def test_cover_tickets_do_not_overlap_primary_front(self):
        """v4.4: 第2-5注前区与主推完全不重叠(覆盖策略核心)"""
        result = self.analyzer.generate_multi_strategy_recommendations()
        by_strategy = {item['strategy']: item for item in result['recommendations']}
        primary = by_strategy['primary_rank']
        primary_front = set(primary['front'])
        for key in ('balanced', 'rank', 'hot', 'cold'):
            other = set(by_strategy[key]['front'])
            self.assertTrue(
                primary_front.isdisjoint(other),
                f"{key} 前区与主推重叠: {primary_front & other}",
            )
        # 且第2-5注彼此也不重叠
        fronts = [set(by_strategy[k]['front']) for k in ('balanced', 'rank', 'hot', 'cold')]
        for i in range(len(fronts)):
            for j in range(i + 1, len(fronts)):
                self.assertTrue(fronts[i].isdisjoint(fronts[j]))

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
        self.assertEqual(LOTTERY_PREDICTOR_VERSION, 'dlt-v4.5-next-issue')

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
