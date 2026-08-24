import unittest

from src.lottery import LotteryAnalyzer, LOTTERY_PREDICTOR_VERSION


class DltPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = LotteryAnalyzer()

    def test_portfolio_exposes_anchor_policy_and_unique_tickets(self):
        result = self.analyzer.generate_multi_strategy_recommendations()
        policy = result['portfolio_policy']
        recommendations = result['recommendations']

        self.assertEqual(policy['name'], 'portfolio_cover_v4.6')
        self.assertEqual(len(policy['front_anchors']), 2)
        self.assertEqual(len(policy['back_anchors']), 0)
        portfolio = [item for item in recommendations if not item['strategy'].startswith('picked')]
        tickets = {
            (tuple(item['front']), tuple(item['back']))
            for item in portfolio
        }
        self.assertEqual(len(tickets), len(portfolio))
        # v4.4 组合覆盖: 前5注(primary+4策略)前区完全不重叠, union=25;
        # v4.6 第6注 back_cover 不参与前区覆盖结构
        core_portfolio = [item for item in portfolio if item['strategy'] != 'back_cover']
        self.assertEqual(len(core_portfolio), 5)
        front_union = set()
        for item in core_portfolio:
            front_union.update(item['front'])
        self.assertEqual(len(front_union), 25)
        self.assertEqual(policy['front_union'], 25)
        # v4.6: 6注后区覆盖全部12码
        back_numbers = {
            number
            for item in portfolio
            for number in item['back']
        }
        self.assertEqual(len(back_numbers), 12)
        self.assertEqual(result['back_coverage_profile']['unique_number_count'], 12)
        self.assertAlmostEqual(
            result['back_coverage_profile']['at_least_one_group_ge1_probability'],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            result['back_coverage_profile']['at_least_one_group_ge2_probability'],
            6 / 66,
            places=6,
        )

    def test_back_cover_note_partitions_all_back_numbers(self):
        """v4.6: 第6注后区=前5注未覆盖的2码, 6注后区不相交划分全部12码"""
        result = self.analyzer.generate_multi_strategy_recommendations()
        by_strategy = {item['strategy']: item for item in result['recommendations']}
        self.assertIn('back_cover', by_strategy)

        cover = by_strategy['back_cover']
        self.assertEqual(len(cover['front']), 5)
        self.assertEqual(len(cover['back']), 2)

        portfolio = [item for item in result['recommendations']
                     if not item['strategy'].startswith('picked')]
        back_pairs = [tuple(sorted(item['back'])) for item in portfolio]
        # 6注后区对互不重叠
        all_backs = [n for pair in back_pairs for n in pair]
        self.assertEqual(len(all_backs), 12)
        self.assertEqual(len(set(all_backs)), 12)
        # 保底注的后区 = 前5注未覆盖的号码
        core_backs = {n for item in portfolio if item['strategy'] != 'back_cover'
                      for n in item['back']}
        self.assertEqual(set(cover['back']) & core_backs, set())
        self.assertEqual(set(cover['back']) | core_backs,
                         set(range(1, 13)))

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
        self.assertEqual(LOTTERY_PREDICTOR_VERSION, 'dlt-v4.6-back-cover')

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
