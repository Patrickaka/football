# -*- coding: utf-8 -*-
"""足球的市场口径：亚盘 / 欧赔 / 大小球 / 凯利 / 离散度 / 联合异常。

参照物是从迁移前的 `src/football/markets.py` 生成的黄金文件
（`tests/fixtures/golden/football_markets.json.gz`，2624 条），**逐条相同**。
迁移当时另跑过一轮 2590 条的新旧双跑差分（含错误路径），零差异。

语料两部分（见 `scripts/gen_football_markets_golden.py`）：
线上 114 个 `match_analysis` 缓存反推出的 56 组真实赔率，
外加专攻真实语料没碰到的分支的合成样本（判据 8）。

**真实语料的三处盲区**（判据 8 的实证）：
1. 大小球 signal_strength 56 条**全是 weak**——线上盘口变化从没到过 0.25；
2. 预期进球区间六档里有三档没走到；
3. **凯利离散度 56 条恒为 0**，见下。

**凯利那条值得单独记**：`kelly_i = o_i × p_i × 100`，而 `analyze_euro` 传的
`p_i` 是**同一组赔率**的去水概率 `(1/o_i)/Σ(1/o_j)`，于是
`o_i × p_i ≡ 100/Σ(1/o_j)`——三项数学上恒等，spread 恒为 0（实测 ≤1.4e-14）。
所以线上 `hardest`/`favored` 永远是 `neutral`、`risks`/`favors` 恒空、
summary 永远是「凯利离散度0.0，三项较为均衡；暂无明显最难项」。
`analyze_kelly` 是公开导出的函数，传别的概率就走得到，属判据 9 第三行
「当前调用方碰巧不触发」——**补语料，不删代码**，行为原样保留。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import markets
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_markets.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时 config 里生效的那组阈值。**写死不 import**（判据 4）——
# 断言引用被测常量的话，把常量改坏、期望值跟着挪，照样全绿。
HANDICAP_TREND_EPS = 0.02
WATER_TREND_EPS = 0.05
EURO_PROB_TREND_EPS = 0.02
KELLY_BIAS_EPS = 2.0
TOTAL_LEAN_THRESHOLD = 0.55
TOTAL_LINE_TREND_EPS = 0.125


def golden_entries():
    from scripts.gen_football_markets_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class DefaultsArePartOfTheContract(unittest.TestCase):
    """判据 29：领域层的默认值是公开契约的一部分。

    适配层每次都显式传阈值，所以**没有任何生产路径覆盖这些默认值**——
    不专门测一遍，把默认值改坏是零反应的。
    """

    def test_asian_defaults_match_the_production_thresholds(self):
        d = {'open': {'handicap': 0.0, 'home_odds': 0.90, 'away_odds': 0.90},
             'close': {'handicap': 0.0, 'home_odds': 0.90, 'away_odds': 0.90}}
        # 让球恰好动了 0.03（>0.02），不传参数也应当判成 up
        moved = {'open': dict(d['open']), 'close': dict(d['close'], handicap=0.03)}
        self.assertEqual(markets.analyze_asian(moved)['trend_direction'], 'up')
        # 动 0.01（<0.02）应当是 stable
        still = {'open': dict(d['open']), 'close': dict(d['close'], handicap=0.01)}
        self.assertEqual(markets.analyze_asian(still)['trend_direction'], 'stable')

    def test_water_trend_default_eps_is_five_hundredths(self):
        base = {'handicap': 0.0, 'home_odds': 0.90, 'away_odds': 0.90}
        moved = markets.analyze_asian(
            {'open': base, 'close': dict(base, home_odds=0.96)})
        self.assertEqual(moved['water_trend'], "主队水位上升 → 资金偏向客队")
        still = markets.analyze_asian(
            {'open': base, 'close': dict(base, home_odds=0.94)})
        self.assertEqual(still['water_trend'], "水位基本稳定")

    def test_total_lean_default_threshold_is_fifty_five_hundredths(self):
        # 大球去水概率 0.5625 ≥ 0.55 → over
        over = markets.analyze_total(
            {'open': {'line': 2.5, 'over_odds': 0.80, 'under_odds': 1.0286},
             'close': {'line': 2.5, 'over_odds': 0.80, 'under_odds': 1.0286}})
        self.assertEqual(over['lean'], 'over')
        # 0.5102 < 0.55 → 均衡
        flat = markets.analyze_total(
            {'open': {'line': 2.5, 'over_odds': 0.96, 'under_odds': 1.0},
             'close': {'line': 2.5, 'over_odds': 0.96, 'under_odds': 1.0}})
        self.assertIsNone(flat['lean'])

    def test_euro_to_handicap_default_k_is_one_point_eight(self):
        self.assertAlmostEqual(markets.euro_to_handicap_implied(0.6, 0.2), 0.72)

    def test_kelly_trend_default_window_is_five(self):
        """默认只看最近 5 条：更早的数据不该影响斜率。

        **`series` 是倒序的**（最新在前），内部 `reversed` 之后取前 5——
        也就是**输入的最后 5 条**。第一版用例把噪声追加在末尾，
        那恰恰是被取用的那 5 条，斜率从 -2.02 变成 -959.57。
        判据 28：先跑一遍验算，别猜它会走哪条分支。
        """
        five = [[2.0 + 0.1 * i, 3.4, 3.8, 93.0] for i in range(5)]
        older = [[99.0, 99.0, 99.0, 93.0]] * 3 + five
        self.assertEqual(markets.analyze_kelly_trend(five)['slopes'],
                         markets.analyze_kelly_trend(older)['slopes'])
        # 反过来：把噪声放进窗口内，斜率必须变——否则这条断言什么也没守住
        self.assertNotEqual(markets.analyze_kelly_trend(five)['slopes'],
                            markets.analyze_kelly_trend(five + [[99.0, 99.0, 99.0, 93.0]] * 3)['slopes'])
        # **窗口必须恰好是 5，不是「至少 3」**（判据 5：单向断言挡不住反方向）。
        # 后两条反向：窗口 5 的斜率与窗口 3 的必须不同。
        reversing = [[2.0, 3.4, 3.8, 93.0], [2.1, 3.4, 3.8, 93.0], [2.2, 3.4, 3.8, 93.0],
                     [1.5, 3.4, 3.8, 93.0], [1.4, 3.4, 3.8, 93.0]]
        self.assertEqual(markets.analyze_kelly_trend(reversing)['slopes'],
                         markets.analyze_kelly_trend(reversing, 5)['slopes'])
        self.assertNotEqual(markets.analyze_kelly_trend(reversing)['slopes'],
                            markets.analyze_kelly_trend(reversing, 3)['slopes'])


class AsianBoundaries(unittest.TestCase):
    """每一档让球都测两侧——只测「够格」那边，把门槛改低照样全绿（判据 5）。"""

    @staticmethod
    def _asian(hcap):
        side = {'handicap': hcap, 'home_odds': 0.90, 'away_odds': 0.90}
        return markets.analyze_asian({'open': side, 'close': side})

    def test_the_five_handicap_buckets_have_distinct_descriptions(self):
        self.assertEqual(self._asian(0.25)['diff_desc'], "势均力敌")
        self.assertEqual(self._asian(0.26)['diff_desc'], "预期1球差")
        self.assertEqual(self._asian(0.75)['diff_desc'], "预期1球差")
        self.assertEqual(self._asian(0.76)['diff_desc'], "预期1-2球差")
        self.assertEqual(self._asian(1.25)['diff_desc'], "预期1-2球差")
        self.assertEqual(self._asian(1.26)['diff_desc'], "预期2球差")
        self.assertEqual(self._asian(1.75)['diff_desc'], "预期2球差")
        self.assertEqual(self._asian(1.76)['diff_desc'], "预期1.8球差以上")

    def test_buckets_are_symmetric_around_zero(self):
        """负让球走的是 abs()——两侧必须同档（判据 7：对称的两条分支只测一条不够）。"""
        for h in (0.25, 0.8, 1.3, 1.8, 2.5):
            self.assertEqual(self._asian(h)['diff_desc'], self._asian(-h)['diff_desc'])
            self.assertEqual(self._asian(h)['diff_range'], self._asian(-h)['diff_range'])

    def test_favor_has_three_distinct_cases(self):
        self.assertEqual(self._asian(0.5)['favor'], 'home')
        self.assertEqual(self._asian(-0.5)['favor'], 'away')
        self.assertEqual(self._asian(0.0)['favor'], 'even')

    def test_probability_keys_differ_by_who_gives_the_handicap(self):
        """让球方与受让方的键名不同——这是给人看的标签，没有别的东西盯着（判据 12）。"""
        self.assertEqual(set(self._asian(0.5)['open_prob']), {'home_give', 'away_recv'})
        self.assertEqual(set(self._asian(-0.5)['open_prob']), {'home_recv', 'away_give'})
        self.assertEqual(set(self._asian(0.0)['open_prob']), {'home', 'away'})

    def test_signal_strength_needs_exactly_the_stated_deltas(self):
        def sig(open_h, close_h):
            return markets.analyze_asian({
                'open': {'handicap': open_h, 'home_odds': 0.9, 'away_odds': 0.9},
                'close': {'handicap': close_h, 'home_odds': 0.9, 'away_odds': 0.9},
            })['signal_strength']
        self.assertEqual(sig(0.0, 0.5), 'strong')
        self.assertEqual(sig(0.0, 0.49), 'medium')
        self.assertEqual(sig(0.0, 0.25), 'medium')
        self.assertEqual(sig(0.0, 0.24), 'weak')

    def test_missing_keys_raise_with_the_available_keys_listed(self):
        with self.assertRaises(ValueError) as ctx:
            markets.analyze_asian({'close': {}})
        self.assertIn("'open'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            markets.analyze_asian({'open': {}})
        self.assertIn("'close'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            markets.analyze_asian([])
        self.assertIn('格式错误', str(ctx.exception))


class TotalBoundaries(unittest.TestCase):

    @staticmethod
    def _total(line, over_odds=0.90, under_odds=0.90, open_line=None):
        return markets.analyze_total({
            'open': {'line': line if open_line is None else open_line,
                     'over_odds': over_odds, 'under_odds': under_odds},
            'close': {'line': line, 'over_odds': over_odds, 'under_odds': under_odds},
        })

    def test_expected_goals_has_six_buckets_and_the_lean_splits_three_of_them(self):
        over, under = (0.75, 1.05), (1.05, 0.75)
        self.assertEqual(self._total(1.0, *over)['expected_goals'], [1, 3])
        self.assertEqual(self._total(2.0, *over)['expected_goals'], [1, 4])
        self.assertEqual(self._total(2.5, *over)['expected_goals'], [2, 4])
        self.assertEqual(self._total(2.5, *under)['expected_goals'], [1, 3])
        self.assertEqual(self._total(3.0, *over)['expected_goals'], [2, 5])
        self.assertEqual(self._total(3.0, *under)['expected_goals'], [1, 3])
        self.assertEqual(self._total(3.5, *over)['expected_goals'], [3, 6])
        self.assertEqual(self._total(3.5, *under)['expected_goals'], [2, 4])
        self.assertEqual(self._total(4.5, *over)['expected_goals'], [4, 6])

    def test_line_trend_eps_is_an_order_of_magnitude_wider_than_the_asian_one(self):
        """大小球用 0.125，亚盘用 0.02——**同一个包里两个数量级**。

        判据 17「一半严格一半放任」的形状。这里只是把现状钉住，不是认可它。
        """
        # 门槛是严格大于：dl 恰好等于 0.125 仍算 stable
        self.assertEqual(self._total(2.63, open_line=2.5)['trend_direction'], 'up')
        self.assertEqual(self._total(2.625, open_line=2.5)['trend_direction'], 'stable')
        self.assertEqual(self._total(2.6, open_line=2.5)['trend_direction'], 'stable')
        # 亚盘同样的 0.1 变化早就算 up 了
        moved = markets.analyze_asian({
            'open': {'handicap': 2.5, 'home_odds': 0.9, 'away_odds': 0.9},
            'close': {'handicap': 2.6, 'home_odds': 0.9, 'away_odds': 0.9}})
        self.assertEqual(moved['trend_direction'], 'up')

    def test_implied_total_rises_with_the_over_probability(self):
        low = markets.implied_total_goals(2.5, 0.30)
        mid = markets.implied_total_goals(2.5, 0.50)
        high = markets.implied_total_goals(2.5, 0.70)
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_implied_total_clamps_extreme_probabilities_into_the_search_bounds(self):
        """0/1 概率不该让二分跑飞——夹到 [0.02, 0.98] 再搜 [0.3, 6.5]。

        **断言的是夹紧后的具体值，不是「大于下界」**：不夹的话 p=0 会一路
        收敛到 0.3 附近，而 `> 0.3` 对 0.30000001 也成立——那条断言什么也
        没守住（判据 5）。夹到 0.02 之后落在 0.567。
        """
        self.assertAlmostEqual(markets.implied_total_goals(2.5, 0.0), 0.5672, places=3)
        # 上界那侧要挑一条低盘口才咬得到：λ=6.5 时 P(进球>0.5) 已达 0.9985 > 0.98
        self.assertLess(markets.implied_total_goals(0.5, 1.0), 6.5)
        self.assertGreater(markets.implied_total_goals(0.5, 1.0), 3.0)


class KellyDegeneracy(unittest.TestCase):
    """把「线上凯利离散度恒为 0」这件事钉住，免得下一个人重新推一遍。"""

    ODDS = {'open': {'home': 2.0, 'draw': 3.4, 'away': 3.8},
            'close': {'home': 2.0, 'draw': 3.4, 'away': 3.8}}

    def test_devigged_probabilities_make_all_three_kelly_indices_identical(self):
        for odds in ((2.0, 3.4, 3.8), (1.3, 5.0, 12.0), (1.01, 50.0, 60.0)):
            with self.subTest(odds=odds):
                probs = markets.remove_vig(*odds)
                k = markets.kelly_index_triple(*odds, *probs)
                self.assertAlmostEqual(k['home'], k['draw'], places=9)
                self.assertAlmostEqual(k['draw'], k['away'], places=9)

    def test_the_production_call_shape_always_lands_on_neutral(self):
        probs = markets.remove_vig(2.0, 3.4, 3.8)
        result = markets.analyze_kelly(self.ODDS, probs, probs)
        self.assertEqual(result['hardest'], 'neutral')
        self.assertEqual(result['favored'], 'neutral')
        self.assertEqual(result['risks'], [])
        self.assertEqual(result['favors'], [])
        self.assertIn('暂无明显最难项', result['summary'])

    def test_independent_probabilities_do_reach_the_other_two_branches(self):
        """传非同源概率就走得到——所以这段代码是活的，不能删（判据 9）。"""
        diverged = markets.analyze_kelly(self.ODDS, (0.55, 0.25, 0.20), (0.55, 0.25, 0.20))
        self.assertGreaterEqual(diverged['spread'], 4.0)
        self.assertEqual(diverged['hardest'], 'home')
        self.assertIn('庄家态度分化明显', diverged['summary'])
        self.assertTrue(diverged['risks'])

        devig = markets.remove_vig(2.0, 3.4, 3.8)
        nudged = (devig[0] + 0.004, devig[1], devig[2] - 0.004)
        middle = markets.analyze_kelly(self.ODDS, nudged, nudged)
        self.assertTrue(1.0 <= middle['spread'] < 4.0)
        self.assertIn('最难项倾向', middle['summary'])

    def test_the_neutral_cutoff_is_one_not_merely_near_zero(self):
        """spread 在 (0, 1) 之间仍算中性——**这一档必须专门喂**。

        不喂的话把 `KELLY_NEUTRAL_SPREAD` 从 1.0 改成 0.1 是零反应的：
        同源去水那档 spread≈1e-14，在两个门槛下都算中性（判据 5 的反方向）。
        """
        devig = markets.remove_vig(2.0, 3.4, 3.8)
        tiny = (devig[0] + 0.0005, devig[1], devig[2] - 0.0005)
        result = markets.analyze_kelly(self.ODDS, tiny, tiny)
        self.assertTrue(0.1 < result['spread'] < 1.0)
        self.assertEqual(result['hardest'], 'neutral')
        self.assertIn('暂无明显最难项', result['summary'])


class ReturnRateFallback(unittest.TestCase):
    """返还率兜底：**契约内走不到**，只有负赔率才触发。

    变异 `DEFAULT_RETURN_RATE` 不红就是因为这个。这里把它钉住并说明原因，
    免得下一个人把它当成漏测去补一个正赔率的用例——那是补不出来的。
    """

    def test_positive_odds_can_never_reach_the_fallback(self):
        """三个正数的倒数之和恒大于 0，所以返还率永远是算出来的。"""
        for odds in ((2.0, 3.4, 3.8), (1.01, 50.0, 60.0), (1000.0, 1000.0, 1000.0)):
            with self.subTest(odds=odds):
                self.assertNotEqual(markets.return_rate_from_odds(*odds), 92.0)
                self.assertAlmostEqual(markets.return_rate_from_odds(*odds),
                                       100.0 / sum(1.0 / o for o in odds))

    def test_the_fallback_value_is_ninety_two(self):
        """只有让倒数之和恰好为 0 的负赔率才走得到——不是合法输入，只为钉值。"""
        self.assertEqual(markets.return_rate_from_odds(-1.0, 2.0, 2.0), 92.0)


class SeriesGuards(unittest.TestCase):
    """时序类函数的守卫：样本不足要走同一条兜底路（判据 7）。"""

    def test_kelly_trend_needs_two_usable_records(self):
        for series in (None, [], [[2.0, 3.0, 4.0]]):
            with self.subTest(series=series):
                self.assertEqual(markets.analyze_kelly_trend(series)['summary'], '数据不足')

    def test_records_shorter_than_three_odds_are_dropped_not_crashed(self):
        """两条都是残缺记录 → 可用的凯利不足两条 → 仍走「数据不足」。"""
        self.assertEqual(
            markets.analyze_kelly_trend([[2.0, 3.0], [2.1, 3.1]])['summary'], '数据不足')

    def test_momentum_and_dispersion_need_two_points(self):
        for series in (None, [], [[2.0, 3.0, 4.0]]):
            with self.subTest(series=series):
                self.assertEqual(
                    markets.analyze_euro_momentum(series)['shift_supremacy'], 0.0)
                self.assertEqual(markets.compute_dispersion(series), 0.0)

    def test_slope_is_zero_when_x_has_no_variance(self):
        self.assertEqual(markets.linear_regression_slope([0, 0, 0], [1.0, 2.0, 3.0]), 0.0)
        self.assertEqual(markets.linear_regression_slope([0], [1.0]), 0.0)


class EuroErrorContract(unittest.TestCase):
    """`analyze_euro` 把一切异常归一成 `ValueError('欧赔分析失败: ...')`。

    调用方（`pipeline`）按这个契约接，所以归一本身是契约的一部分。
    """

    GOOD = {'home': 2.0, 'draw': 3.4, 'away': 3.8}

    def test_every_malformed_input_becomes_the_same_error_shape(self):
        cases = [
            {},
            {'open': None, 'close': None},
            {'open': {'home': 2.0}, 'close': dict(GOOD := {'home': 2.0, 'draw': 3.4, 'away': 3.8})},
            {'open': dict(GOOD, home=0), 'close': dict(GOOD)},
            {'open': dict(GOOD, home='x'), 'close': dict(GOOD)},
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError) as ctx:
                    markets.analyze_euro(data)
                self.assertTrue(str(ctx.exception).startswith('欧赔分析失败: '))

    def test_a_valid_payload_reports_only_the_moves_beyond_the_epsilon(self):
        drifted = markets.analyze_euro({
            'open': {'home': 2.50, 'draw': 3.30, 'away': 2.80},
            'close': {'home': 1.90, 'draw': 3.50, 'away': 4.20}})
        self.assertTrue(any('主胜概率↑' in c for c in drifted['changes']))
        self.assertTrue(any('客胜概率↓' in c for c in drifted['changes']))
        unchanged = markets.analyze_euro({'open': dict(self.GOOD), 'close': dict(self.GOOD)})
        self.assertEqual(unchanged['changes'], [])


class JointAnomaly(unittest.TestCase):

    ASIAN_DOWN = {'open': {'handicap': 0, 'home_odds': 1.00, 'away_odds': 0.80},
                  'close': {'handicap': 0, 'home_odds': 0.90, 'away_odds': 0.90}}
    TOTAL_DOWN = {'open': {'line': 2.5, 'over_odds': 1.00, 'under_odds': 0.80},
                  'close': {'line': 2.5, 'over_odds': 0.90, 'under_odds': 0.90}}

    def test_both_waters_must_fall_for_the_big_win_hint(self):
        """**两个条件是与不是或**——只满足一个不该报（判据 7）。"""
        both = markets.compute_joint_anomaly(self.ASIAN_DOWN, self.TOTAL_DOWN)
        self.assertTrue(both['hint_big_win'])
        self.assertIsNotNone(both['hint_desc'])

        flat_total = {'open': {'line': 2.5, 'over_odds': 0.90, 'under_odds': 0.90},
                      'close': {'line': 2.5, 'over_odds': 0.90, 'under_odds': 0.90}}
        only_asian = markets.compute_joint_anomaly(self.ASIAN_DOWN, flat_total)
        self.assertFalse(only_asian['hint_big_win'])
        self.assertIsNone(only_asian['hint_desc'])

        flat_asian = {'open': {'handicap': 0, 'home_odds': 0.90, 'away_odds': 0.90},
                      'close': {'handicap': 0, 'home_odds': 0.90, 'away_odds': 0.90}}
        only_total = markets.compute_joint_anomaly(flat_asian, self.TOTAL_DOWN)
        self.assertFalse(only_total['hint_big_win'])


if __name__ == '__main__':
    unittest.main()
