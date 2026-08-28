"""北单的盘口走势与派生因子：亚盘、大小球、比分三条线。

参照物是从迁移前的 `markets.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_trends.json.gz`，135 条），**逐条相同**。
语料按每个门槛的两侧构造，覆盖五种亚盘方向、四种大小球方向、
三档总进球因子与**全部五档**亚盘因子。

**迁移前这些阈值全是函数体里的裸数字**（`0.02`、`0.03`、`0.05`、`0.15`、
`1.2`、`0.85`…），既没有名字也没有出处，改一个要先读懂整段代码。
现在集中在 `src/beidan/markets.py` 的常量区。

另有两处「原地改写入参」的旧语义（`adjust_zjq_by_goals` 与
`enhance_scores_with_cs`）：领域层返回新对象，适配层写回入参保住兼容——
有调用方依赖「传进去的那份也变了」，改掉是另一件事。
"""
import gzip
import importlib
import json
import pathlib
import unittest

from src.domain.sports.beidan import trends
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_trends.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组门槛，写死不 import（判据 12）
ASIAN_MOVE, ASIAN_DIRECTION, ASIAN_FACTOR = 0.02, 0.03, 0.15
GOALS_DIRECTION = 0.05
GOALS_OVER, GOALS_UNDER, GOALS_MARGIN = 1.2, 0.85, 0.5


def _asian(start_home, step_home, start_away, step_away, count=6):
    return [{'home_odds': round(start_home + step_home * i, 4),
             'away_odds': round(start_away + step_away * i, 4)}
            for i in range(count)]


def _ou(start_over, step_over, start_under, step_under, count=6):
    return [{'over_odds': round(start_over + step_over * i, 4),
             'under_odds': round(start_under + step_under * i, 4)}
            for i in range(count)]


def golden_entries():
    from scripts.gen_beidan_trends_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class AsianDirectionTests(unittest.TestCase):
    """五种方向，每个门槛的两侧都要有样本。"""

    def test_home_odds_falling_is_backing(self):
        result = trends.analyze_asian(_asian(1.05, -0.05, 0.85, 0.03))
        self.assertEqual(result['direction'], 'home_backing')

    def test_home_odds_rising_is_laying(self):
        result = trends.analyze_asian(_asian(0.85, 0.05, 1.05, -0.03))
        self.assertEqual(result['direction'], 'home_laying')

    def test_move_below_threshold_stays_stable(self):
        """低于 0.03 就不判方向——**只测「动了」那侧的话，
        把门槛改成 0 也全绿。**"""
        result = trends.analyze_asian(_asian(0.95, -0.015, 0.95, 0.005))
        self.assertEqual(result['direction'], 'stable')

    def test_home_takes_precedence_over_away(self):
        """两边同时动时以主队为准——北单的盘口以主队为基准报价。"""
        result = trends.analyze_asian(_asian(1.05, -0.05, 1.05, -0.05))
        self.assertEqual(result['direction'], 'home_backing')

    def test_away_direction_only_when_home_is_quiet(self):
        result = trends.analyze_asian(_asian(0.95, 0.0, 1.02, -0.05))
        self.assertEqual(result['direction'], 'away_backing')

    def test_single_entry_cannot_have_a_direction(self):
        """只有一期就没有「变化」可言。"""
        self.assertEqual(trends.analyze_asian(_asian(0.95, 0, 0.95, 0, count=1)),
                         {'direction': 'stable', 'strength': 0})

    def test_empty_history_is_stable(self):
        self.assertEqual(trends.analyze_asian([]),
                         {'direction': 'stable', 'strength': 0})

    def test_strength_adds_both_sides(self):
        result = trends.analyze_asian(_asian(1.05, -0.05, 0.85, 0.03))
        self.assertAlmostEqual(
            result['strength'],
            abs(result['avg_home_change']) + abs(result['avg_away_change']), places=4)


class AsianProbabilityAdjustmentTests(unittest.TestCase):

    PROBS = (0.45, 0.28, 0.27)

    def test_backing_the_home_side_lifts_its_probability(self):
        home, _, _ = trends.adjust_probs_by_asian(*self.PROBS,
                                                  _asian(1.05, -0.05, 0.85, 0.03))
        self.assertGreater(home, self.PROBS[0])

    def test_laying_the_home_side_cuts_its_probability(self):
        home, _, _ = trends.adjust_probs_by_asian(*self.PROBS,
                                                  _asian(0.85, 0.05, 1.05, -0.03))
        self.assertLess(home, self.PROBS[0])

    def test_movement_below_threshold_changes_nothing(self):
        """低于 0.02 不动——**门槛的另一侧**。"""
        result = trends.adjust_probs_by_asian(*self.PROBS,
                                              _asian(0.95, -0.01, 0.95, 0.005))
        for got, expected in zip(result, self.PROBS):
            self.assertAlmostEqual(got, expected, places=9)

    def test_output_is_normalised(self):
        result = trends.adjust_probs_by_asian(*self.PROBS,
                                              _asian(1.05, -0.05, 0.85, 0.03))
        self.assertAlmostEqual(sum(result), 1.0, places=9)

    def test_draw_probability_is_never_adjusted_directly(self):
        """平局不参与调整——**亚盘让球盘本来就不表达平局**。
        它的值只会因为归一化而变，比例上不该被主动动过。"""
        home, draw, away = trends.adjust_probs_by_asian(
            *self.PROBS, _asian(1.05, -0.05, 0.85, 0.03))
        # 归一化前平局是原值，所以调整后它相对于总量只随分母变化
        self.assertAlmostEqual(draw / (home + draw + away), draw, places=9)

    def test_default_counter_ratio_is_half(self):
        """**不传参数时反方向就该只有一半力度。**

        下一条用例显式传了 `counter_ratio`，所以改默认值影响不到它；
        适配层也是显式传的——于是领域层这个默认值没有任何路径覆盖。
        这条走默认值，用「两侧的相对变化之比」来断言，
        归一化会同比缩放两边，比值不受影响。
        """
        home_only = _asian(1.05, -0.05, 0.95, 0.0)   # 只有主队在降赔
        home, _, away = trends.adjust_probs_by_asian(*self.PROBS, home_only)

        # **用主客比值而不是各自的涨幅**：函数末尾会归一化，
        # 那会把两边同时缩放，「相对原值涨了多少」因此不再反映调整力度。
        # 比值不受归一化影响——它正好等于 (1 + factor) / (1 - factor * ratio)。
        before = self.PROBS[0] / self.PROBS[2]
        after = home / away
        self.assertAlmostEqual(after / before,
                               (1 + ASIAN_FACTOR) / (1 - ASIAN_FACTOR * 0.5),
                               places=6)

    def test_counter_side_gets_only_half_the_force(self):
        """反方向只给一半力度——**对称调整会把一条信息用成两条**。"""
        strong = trends.adjust_probs_by_asian(
            *self.PROBS, _asian(1.05, -0.05, 0.95, 0.0),
            factor=0.2, counter_ratio=0.5)
        symmetric = trends.adjust_probs_by_asian(
            *self.PROBS, _asian(1.05, -0.05, 0.95, 0.0),
            factor=0.2, counter_ratio=1.0)
        self.assertNotEqual(strong, symmetric)
        self.assertGreater(symmetric[0], strong[0], '对称时主队被推得更高')

    def test_too_short_history_is_returned_untouched(self):
        result = trends.adjust_probs_by_asian(*self.PROBS, [{'home_odds': 0.9}])
        self.assertEqual(result, self.PROBS)


class GoalsTrendTests(unittest.TestCase):

    def test_over_odds_falling_is_backing(self):
        self.assertEqual(trends.analyze_goals(_ou(1.05, -0.08, 0.85, 0.04))['direction'],
                         'over_backing')

    def test_over_odds_rising_is_laying(self):
        self.assertEqual(trends.analyze_goals(_ou(0.85, 0.08, 1.05, -0.04))['direction'],
                         'over_laying')

    def test_threshold_is_looser_than_asian(self):
        """0.04 的幅度在亚盘算「动了」，在大小球还不算——
        **大小球的水位波动本来就更大**。"""
        move = _ou(1.0, -0.04, 0.9, 0.01)
        self.assertEqual(trends.analyze_goals(move)['direction'], 'stable')
        asian_like = _asian(1.0, -0.04, 0.9, 0.01)
        self.assertEqual(trends.analyze_asian(asian_like)['direction'], 'home_backing')

    def test_under_direction_only_when_over_is_quiet(self):
        self.assertEqual(trends.analyze_goals(_ou(0.95, 0.01, 1.05, -0.08))['direction'],
                         'under_backing')


class GoalsFactorTests(unittest.TestCase):

    def test_cheaper_over_lifts_the_factor(self):
        self.assertEqual(trends.goals_factor(_ou(0.80, 0, 1.05, 0)), GOALS_OVER)

    def test_much_cheaper_under_cuts_the_factor(self):
        self.assertEqual(trends.goals_factor(_ou(1.20, 0, 0.65, 0)), GOALS_UNDER)

    def test_the_two_sides_are_deliberately_asymmetric(self):
        """偏小球要多出 0.5 的差值才认，偏大球不用——**大球贴水天然偏低，
        对称门槛会把常态误判成偏小球**。这条正是为那个不对称而设。"""
        # 差值 0.42，够不到 0.5 → 中性
        self.assertEqual(trends.goals_factor(_ou(1.10, 0, 0.68, 0)), 1.0)
        # 反过来只要大球更便宜就认，不需要任何余量
        self.assertEqual(trends.goals_factor(_ou(0.94, 0, 0.95, 0)), GOALS_OVER)

    def test_entries_missing_a_side_are_skipped(self):
        """缺一侧的那条跳过，**不补 0**——补 0 会把「没数据」算成「没变化」。"""
        self.assertEqual(trends.goals_factor([{'over_odds': 0.9}, {'over_odds': 0.8}]),
                         1.0)


class AsianGoalFactorTests(unittest.TestCase):
    """五档分档，每档的边界两侧都要有样本。"""

    def _factor(self, total):
        return trends.asian_goal_factor([{'home_odds': total / 2,
                                          'away_odds': total / 2}] * 4)

    def test_all_five_tiers_are_reachable(self):
        self.assertEqual(self._factor(3.4), 1.3)
        self.assertEqual(self._factor(3.8), 1.15)
        self.assertEqual(self._factor(4.2), 1.0)
        self.assertEqual(self._factor(4.6), 0.9)
        self.assertEqual(self._factor(5.2), 0.75)

    def test_tier_boundaries_belong_to_the_upper_tier(self):
        """恰好等于门槛时落到**下**一档（判断是严格小于）。
        每档都测边界，把某个门槛挪一点就会红。"""
        self.assertEqual(self._factor(3.6), 1.15)
        self.assertEqual(self._factor(4.0), 1.0)
        self.assertEqual(self._factor(4.4), 0.9)
        self.assertEqual(self._factor(4.8), 0.75)

    def test_too_short_history_is_neutral(self):
        self.assertEqual(trends.asian_goal_factor([{'home_odds': 1.0,
                                                    'away_odds': 1.0}]), 1.0)


class GoalBucketAdjustmentTests(unittest.TestCase):

    BUCKETS = {'0': 0.05, '1': 0.15, '2': 0.22, '3': 0.20,
               '4': 0.15, '5': 0.10, '6': 0.08, '7+': 0.05}

    def test_money_on_over_lifts_the_high_buckets(self):
        adjusted = trends.adjust_goal_buckets(self.BUCKETS, _ou(1.05, -0.08, 0.85, 0.04))
        self.assertGreater(adjusted['3'], self.BUCKETS['3'])
        self.assertLess(adjusted['1'], self.BUCKETS['1'])

    def test_money_on_under_lifts_the_low_buckets(self):
        adjusted = trends.adjust_goal_buckets(self.BUCKETS, _ou(0.85, 0.08, 1.05, -0.04))
        self.assertGreater(adjusted['1'], self.BUCKETS['1'])
        self.assertLess(adjusted['3'], self.BUCKETS['3'])

    def test_quiet_market_leaves_the_buckets_alone(self):
        adjusted = trends.adjust_goal_buckets(self.BUCKETS, _ou(0.95, -0.01, 0.95, 0.0))
        self.assertEqual(adjusted, self.BUCKETS)

    def test_input_is_not_mutated(self):
        """**领域层返回新字典**。迁移前这里直接改写入参，
        调用方拿到的和传出去的是同一个对象，读的人分不清哪份是原始值。"""
        original = dict(self.BUCKETS)
        trends.adjust_goal_buckets(self.BUCKETS, _ou(1.05, -0.08, 0.85, 0.04))
        self.assertEqual(self.BUCKETS, original)

    def test_adapter_keeps_the_in_place_behaviour(self):
        """适配层仍然改写入参——有调用方依赖这个语义。"""
        markets = importlib.import_module('src.beidan.markets')
        buckets = dict(self.BUCKETS)
        returned = markets.adjust_zjq_by_goals(buckets, _ou(1.05, -0.08, 0.85, 0.04))
        self.assertIs(returned, buckets)
        self.assertNotEqual(buckets, self.BUCKETS)

    def test_missing_buckets_are_created_not_skipped(self):
        sparse = {'1': 0.4, '3': 0.6}
        adjusted = trends.adjust_goal_buckets(sparse, _ou(1.05, -0.08, 0.85, 0.04))
        self.assertIn('7+', adjusted)
        self.assertEqual(adjusted['7+'], 0)


class CorrectScoreTests(unittest.TestCase):

    HISTORY = [{'score': '1-0', 'odds': 8.5}, {'score': '1-1', 'odds': 7.2},
               {'score': '1-0', 'odds': 8.0}, {'score': '2-1', 'odds': 9.5},
               {'score': '1-1', 'odds': 7.5}]

    def test_hot_scores_are_sorted_by_current_odds(self):
        """赔率低的排前面——**赔率低就是被押得多**。"""
        hot = trends.analyze_correct_score(self.HISTORY)['hot_scores']
        odds = [item['current_odds'] for item in hot]
        self.assertEqual(odds, sorted(odds))

    def test_trend_direction_per_score(self):
        firming = trends.analyze_correct_score(
            [{'score': '1-0', 'odds': 8.3}, {'score': '1-0', 'odds': 8.0}])
        self.assertEqual(firming['hot_scores'][0]['trend'], 'down')
        drifting = trends.analyze_correct_score(
            [{'score': '1-0', 'odds': 8.0}, {'score': '1-0', 'odds': 8.3}])
        self.assertEqual(drifting['hot_scores'][0]['trend'], 'up')

    def test_movement_below_threshold_is_stable(self):
        hairline = trends.analyze_correct_score(
            [{'score': '1-0', 'odds': 8.0}, {'score': '1-0', 'odds': 8.05}])
        self.assertEqual(hairline['hot_scores'][0]['trend'], 'stable')

    def test_malformed_entries_are_skipped(self):
        result = trends.analyze_correct_score(
            [{'score': None, 'odds': 8.0}, {'odds': 7.0}, {'score': '1-1'}])
        self.assertEqual(result['hot_scores'], [])

    def test_blend_averages_model_and_market(self):
        """模型概率与盘口隐含概率取平均。"""
        blended = trends.blend_scores_with_market(
            [{'score': '1-0', 'probability': 0.12}], self.HISTORY)
        expected = (0.12 + 1.0 / 8.0) / 2
        self.assertAlmostEqual(blended[0]['probability'], expected, places=9)
        self.assertEqual(blended[0]['source'], 'cs_enhanced')

    def test_market_only_scores_come_in_discounted(self):
        """盘口有、模型没算到的比分打折补进来——**全额采信会让长尾比分
        挤掉模型的主推**。"""
        blended = trends.blend_scores_with_market(
            [{'score': '3-3', 'probability': 0.9}], self.HISTORY, new_score_discount=0.5)
        # **按 score 定位而不是取第一个**：结果按概率降序排，
        # 盘口里赔率最低的那个会排到前面，取 [0] 测的是排序而不是折扣
        by_score = {item['score']: item for item in blended}
        self.assertEqual(by_score['1-0']['source'], 'cs_new')
        self.assertAlmostEqual(by_score['1-0']['probability'],
                               (1.0 / 8.0) * 0.5, places=9)

    def test_goal_parts_are_none_when_unparsable(self):
        """**0 是一个真实的比分**，不能拿它表示解析失败。"""
        blended = trends.blend_scores_with_market(
            [{'score': 'x-y', 'probability': 0.5}], self.HISTORY)
        broken = [item for item in blended if item['score'] == 'x-y']
        self.assertIsNone(broken[0]['home_goals'])

    def test_tuple_form_is_accepted(self):
        blended = trends.blend_scores_with_market([('1-0', 0.12)], self.HISTORY)
        self.assertEqual(blended[0]['score'], '1-0')

    def test_input_list_is_not_mutated(self):
        scores = [{'score': '1-0', 'probability': 0.12}]
        snapshot = [dict(item) for item in scores]
        trends.blend_scores_with_market(scores, self.HISTORY)
        self.assertEqual(scores, snapshot)


if __name__ == '__main__':
    unittest.main()
