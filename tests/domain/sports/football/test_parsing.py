# -*- coding: utf-8 -*-
"""足球的赔率页解析与竞彩玩法概率。

参照物是从迁移前的 `src/football/parsing.py` 生成的黄金文件
（`tests/fixtures/golden/football_parsing.json.gz`，752 条），**逐条相同**。
迁移当时另跑过 **803 条**新旧双跑差分（769 条纯计算 + 34 条打桩网络层），零差异。

语料三部分：
- **真实 HTML**：`tests/fixtures/football_odds_pages.json.gz`，2026-08-28 从
  500.com 抓的六页（两场 × 亚盘/大小球/数据分析）。只抓一次、存进版本控制
  （判据 20：测试依赖的东西必须在版本控制里，本地能跑 CI 也要能跑）。
- **真实竞彩赔率**：线上 114 场缓存去重出的 45 组官方赔率与让球值。
- **合成**：真实语料没碰到的分支——让球 0（平手）、各档让球文本、畸形输入。

## 两处迁移期查明、行为原样保留的事

**1. `extract_company_odds`（169 行）线上从来没成功过。**
2026-08-28 实测：真实页面的纯文本里既没有 `Bet365`/`Pinnacle`，也没有代码里
写的任何一个别名（`**t3*5` / `Pi****le` …）——只有「平均值」在。
配上 114 场缓存的 `bookmaker_consensus` **全是 None**、7 天日志零痕迹，
这块「Sharp Money 分歧指数」（约 350 行，含 `fetch_single_company_odds`
与 `bookmaker_consensus`）完全不工作。**不是代码不可达，是上游页面变了**
——所以用例用合成 HTML 把它该有的行为钉住，代码一行没删。

**2. 「让球文字→数字」有两份实现，31 个文本里 12 个答案不同**（判据 11）：
`parse_handicap` 不认 `平半`/`半一`（返回 0），且 `受让X` 会被
`.replace('受','')` 变成 `让X` 而落到 0；`handicap_text_to_num` 则完全不认
`三球`/`三球半`，还会把 `两球半/三球` 按包含匹配命中 `球半` 返回 **1.5**。
生产上只有 `parse_handicap` 走得到（经 `extract_handicap_from_segment`），
而且真实平均值行给的是数字不是文字，所以两者都没造成过实际损失。
**用例把 12 条差异逐条钉住**，合并与否是单独的决策。
"""
import ast
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import lottery, parsing
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_parsing.json.gz',
                             'rt', encoding='utf-8'))
PAGES = json.loads(gzip.decompress(
    (FIXTURES / 'football_odds_pages.json.gz').read_bytes()))

# 迁移当时 config 的真实取值。**写死不 import**（判据 4）
CLOSE_BLEND_WEIGHT = 0.72
LOTTERY_OFFICIAL_ODDS_WEIGHT = 0.80
MIN_AVG_NUMBERS = 6


def golden_entries():
    from scripts.gen_football_parsing_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class TwoHandicapParsersDisagree(unittest.TestCase):
    """把 `parse_handicap` 与 `handicap_text_to_num` 的 12 处分歧逐条钉住。

    判据 11「同一件事写两遍，迟早漂」。合并它们会改变行为，属单独决策——
    这里只保证**分歧不会在无人察觉时变化**。
    """

    # (文本, parse_handicap 的答案, handicap_text_to_num 的答案)
    DIVERGENCES = [
        ('三球', 3.0, 0),          # 后者的表里根本没有「三球」
        ('三球半', 3.5, 1.5),       # 包含匹配命中了「球半」
        ('两球半/三球', 2.75, 1.5),  # 同上
        ('三球/三球半', 3.25, 1.5),  # 同上
        ('平半', 0, 0.25),          # 前者的表里没有「平半」
        ('半一', 0, 0.75),          # 前者的表里没有「半一」
        ('受三球', -3.0, 0),
        ('受让半球', 0, -0.5),       # 前者把「受」删掉剩「让半球」，落到 0
        ('受让一球', 0, -1.0),
        ('受让球半', 0, -1.5),
        ('受让两球', 0, -2.0),
        ('受让两球半', 0, -2.5),
    ]

    AGREEMENTS = [
        ('平手', 0), ('半球', 0.5), ('一球', 1.0), ('球半', 1.5),
        ('两球', 2.0), ('两球半', 2.5),
        ('平手/半球', 0.25), ('半球/一球', 0.75), ('一球/球半', 1.25),
        ('球半/两球', 1.75), ('两球/两球半', 2.25),
        ('受平手', 0), ('受半球', -0.5), ('受一球', -1.0),
        ('受球半', -1.5), ('受两球', -2.0), ('受两球半', -2.5),
    ]

    def test_the_twelve_divergences_are_exactly_these(self):
        for text, expected_a, expected_b in self.DIVERGENCES:
            with self.subTest(text=text):
                self.assertEqual(parsing.parse_handicap(text), expected_a)
                self.assertEqual(parsing.handicap_text_to_num(text), expected_b)
                self.assertNotEqual(parsing.parse_handicap(text),
                                    parsing.handicap_text_to_num(text))

    def test_everything_else_agrees(self):
        """**反方向也要测**——只钉分歧，把两边改成一样也发现不了（判据 5）。"""
        for text, expected in self.AGREEMENTS:
            with self.subTest(text=text):
                self.assertEqual(parsing.parse_handicap(text), expected)
                self.assertEqual(parsing.handicap_text_to_num(text), expected)

    def test_the_containment_pass_is_what_makes_two_and_a_half_slash_three_wrong(self):
        """`两球半/三球` 精确匹配落空后，包含匹配在表里先撞上「球半」。

        断言的是**原因**：把「球半」从表里拿掉，答案就不再是 1.5。
        """
        without_ball_half = [(k, v) for k, v in parsing.HANDICAP_TEXT_MAP if k != '球半']
        self.assertEqual(parsing.handicap_text_to_num('两球半/三球'), 1.5)
        self.assertNotEqual(
            parsing.handicap_text_to_num('两球半/三球', without_ball_half), 1.5)


class CompanyOddsNeverMatchesTheLivePages(unittest.TestCase):
    """线上真实页面里根本没有公司名——把这件事钉住，免得下一个人重新查一遍。"""

    ODDS_PAGES = ('yazhi:1430311', 'daxiao:1430311', 'yazhi:1430017', 'daxiao:1430017')

    def test_no_company_name_or_alias_appears_on_the_odds_pages(self):
        """生产只在亚盘页与大小球页上调它——那两种页面里一个公司名都没有。"""
        for key in self.ODDS_PAGES:
            text = parsing.html_to_text(PAGES[key])
            with self.subTest(page=key):
                self.assertIn('平均值', text, '页面本身应当是有效的赔率页')
                for company, aliases in parsing.COMPANY_ALIASES.items():
                    self.assertNotIn(company, text)
                    for alias in aliases:
                        self.assertNotIn(alias, text)

    def test_extraction_returns_none_for_every_odds_page(self):
        for key in self.ODDS_PAGES:
            for company in parsing.COMPANY_ALIASES:
                for is_total in (False, True):
                    with self.subTest(page=key, company=company, is_total=is_total):
                        self.assertIsNone(
                            parsing.extract_company_odds(PAGES[key], company, is_total))

    def test_on_a_page_that_merely_mentions_the_company_it_returns_garbage(self):
        """**隐患**：数据分析页的表头里列着 `Bet365`，于是这个函数会从一张
        毫不相干的表里刮出数字来，而不是返回 None——它没有任何一处校验
        「找到的是不是那家公司的赔率行」。

        生产不在这种页面上调它，所以垃圾没漏出去；但 500.com 哪天把公司名
        放回赔率页的别的位置，拿到的就会是这种东西。钉住现状，不是认可它。
        """
        for key in ('shuju:1430311', 'shuju:1430017'):
            with self.subTest(page=key):
                text = parsing.html_to_text(PAGES[key])
                self.assertIn('Bet365', text, '数据分析页的表头里确实有公司名')
                row = parsing.extract_company_odds(PAGES[key], 'Bet365', is_total=True)
                self.assertIsNotNone(row, '它不会因为找错表而返回 None')
                self.assertEqual(len(row), 8)
                self.assertIsNone(row[6], '刮不到时间戳')

    def test_a_row_in_the_expected_shape_still_extracts(self):
        """代码本身还能工作，只是喂不到料——不测这条，把它改成 `return None` 也全绿。"""
        text = 'Bet365 0.950 半球 0.900 06-09 16:17 0.900 平手 0.930 06-08 09:36'
        row = parsing.extract_company_odds(f'<td>{text}</td>', 'Bet365', is_total=False)
        self.assertEqual(row, [0.9, 0.5, 0.93, 0.95, 0.5, 0.9,
                               '06-09 16:17', '06-08 09:36'])

    def test_a_total_row_in_the_expected_shape_still_extracts(self):
        text = 'Bet365 0.950 2.5球 0.900 06-09 16:17 0.900 2.5球 0.930 06-08 09:36'
        row = parsing.extract_company_odds(f'<td>{text}</td>', 'Bet365', is_total=True)
        self.assertEqual(row[4], 2.5)                 # 终盘线
        self.assertEqual(row[6], '06-09 16:17')
        self.assertEqual(row[7], '06-08 09:36')
        # **初盘小球水位取到的是盘口线 2.5 而不是水位**——过滤器把 2.5 同时
        # 当成水位和盘口收了进来。死路上的老毛病，原样钉住。
        self.assertEqual(row[2], 2.5)

    def test_the_second_company_name_shortcut_never_fires(self):
        """`segment.find('**', 10)` 从第 10 个字符起找第二个 `**`，而别名
        `**t3*5 **t3*5` 的第二个 `**` 在第 7 位——**这条分支连它自己文档里
        举的例子都走不到**。走的是 `segment[len(company_name):]` 那条。
        """
        text = '**t3*5 **t3*5 0.950 半球 0.900 06-09 16:17'
        segment = parsing.html_to_text(f'<td>{text}</td>')
        self.assertEqual(segment.find('**', 10), -1)

    def test_consensus_stays_unavailable_when_either_company_is_missing(self):
        """**两家都要有**才算数——只有一家不该报（判据 7）。"""
        present = {'asian': {'close': {'handicap': 0.5}}}
        for bet, pin in ((None, present), (present, None), (None, None)):
            with self.subTest(bet=bool(bet), pin=bool(pin)):
                self.assertFalse(parsing.bookmaker_consensus(bet, pin, 0.25)['available'])
        both = parsing.bookmaker_consensus(present, present, 0.25)
        self.assertTrue(both['available'])


class ConsensusThresholds(unittest.TestCase):

    @staticmethod
    def _consensus(pinnacle_handicap, avg=0.0):
        side = lambda h: {'asian': {'close': {'handicap': h}}}
        return parsing.bookmaker_consensus(side(0.0), side(pinnacle_handicap), avg)

    def test_the_direction_needs_more_than_an_eighth_of_a_goal(self):
        """门槛是严格大于 0.125——恰好等于仍算中性（判据 28）。"""
        self.assertEqual(self._consensus(0.125)['sharp_bias'], 'neutral')
        self.assertEqual(self._consensus(0.13)['sharp_bias'], 'home')
        self.assertEqual(self._consensus(-0.125)['sharp_bias'], 'neutral')
        self.assertEqual(self._consensus(-0.13)['sharp_bias'], 'away')

    def test_confidence_saturates_at_half_a_goal(self):
        self.assertAlmostEqual(self._consensus(0.25)['confidence'], 0.5)
        self.assertAlmostEqual(self._consensus(0.5)['confidence'], 1.0)
        self.assertAlmostEqual(self._consensus(2.0)['confidence'], 1.0)

    def test_the_adjustment_is_signed_and_scaled_by_confidence(self):
        self.assertAlmostEqual(self._consensus(0.5)['adjustment'], 0.15)
        self.assertAlmostEqual(self._consensus(-0.5)['adjustment'], -0.15)
        self.assertAlmostEqual(self._consensus(0.25)['adjustment'], 0.075)
        self.assertEqual(self._consensus(0.0)['adjustment'], 0.0)


class LotterySelection(unittest.TestCase):
    """胜平负选号的三个门槛，每一档都测两侧。"""

    def test_single_when_the_draw_is_neither_close_nor_likely(self):
        result = lottery.spf_selection_profile({'胜': 0.45, '平': 0.24, '负': 0.31})
        self.assertEqual(result['mode'], 'single')
        self.assertEqual(result['selections'], ['胜'])
        self.assertTrue(result['is_single'])

    def test_draw_cover_needs_both_a_likely_draw_and_a_small_gap(self):
        """**两个条件是与不是或**（判据 7）。"""
        covered = lottery.spf_selection_profile({'胜': 0.40, '平': 0.29, '负': 0.31})
        self.assertEqual(covered['mode'], 'draw_cover')
        self.assertEqual(covered['selections'], ['胜', '平'])

        # 平局够可能，但差距 0.21 > 0.12
        wide_gap = lottery.spf_selection_profile({'胜': 0.50, '平': 0.29, '负': 0.21})
        self.assertEqual(wide_gap['mode'], 'single')

        # 差距够小，但平局概率 0.23 < 0.24
        thin_draw = lottery.spf_selection_profile({'胜': 0.45, '平': 0.23, '负': 0.32})
        self.assertEqual(thin_draw['mode'], 'single')

    def test_draw_primary_cover_uses_a_tighter_gap_than_draw_cover(self):
        """首选是平时用 0.08，不是 0.12——**两个门槛不是同一个数**。"""
        covered = lottery.spf_selection_profile({'胜': 0.34, '平': 0.35, '负': 0.31})
        self.assertEqual(covered['mode'], 'draw_primary_cover')
        self.assertEqual(covered['selections'], ['平', '胜'])

        # 差距 0.10：在 draw_cover 的 0.12 之内，但超过 draw_primary 的 0.08
        wider = lottery.spf_selection_profile({'胜': 0.30, '平': 0.40, '负': 0.30})
        self.assertEqual(wider['mode'], 'single')

    def test_all_zero_probabilities_are_unavailable_not_a_crash(self):
        for probs in ({'胜': 0, '平': 0, '负': 0}, {}, None):
            with self.subTest(probs=probs):
                result = lottery.spf_selection_profile(probs)
                self.assertEqual(result['mode'], 'unavailable')
                self.assertEqual(result['selections'], [])
                self.assertIsNone(result['primary'])


class LotteryMarkets(unittest.TestCase):

    CANDIDATES = [((1, 0), 0.30), ((0, 0), 0.25), ((0, 1), 0.20),
                  ((2, 0), 0.15), ((1, 2), 0.10)]

    def test_a_zero_handicap_still_builds_the_handicap_market_but_is_not_primary(self):
        """让球 0 与「没有让球」不是一回事——**前者仍要算让球盘**。

        `primary_market` 的判定是 `handicap not in (None, 0)`，
        所以 0 落在 spf；但 `parse_lottery_handicap(0)` 返回 0 而不是 None，
        让球段照样建出来。这个落差没有别的东西盯着（判据 12）。
        """
        zero = lottery.lottery_market_probabilities(self.CANDIDATES, 0)
        self.assertEqual(zero['primary_market'], 'spf')
        self.assertIsNotNone(zero['handicap'])
        self.assertEqual(zero['handicap']['handicap'], 0)

        none = lottery.lottery_market_probabilities(self.CANDIDATES, None)
        self.assertEqual(none['primary_market'], 'spf')
        self.assertIsNone(none['handicap'])

        one = lottery.lottery_market_probabilities(self.CANDIDATES, 1)
        self.assertEqual(one['primary_market'], 'rqspf')

    def test_handicaps_beyond_five_goals_are_rejected(self):
        self.assertEqual(lottery.parse_lottery_handicap(5), 5)
        self.assertIsNone(lottery.parse_lottery_handicap(6))
        self.assertEqual(lottery.parse_lottery_handicap(-5), -5)
        self.assertIsNone(lottery.parse_lottery_handicap(-6))

    def test_non_integer_handicaps_are_rejected(self):
        """竞彩让球只接受整数——四分盘不该混进来。"""
        self.assertIsNone(lottery.parse_lottery_handicap('2.5'))
        self.assertIsNone(lottery.parse_lottery_handicap(1.5))
        self.assertEqual(lottery.parse_lottery_handicap('2.0'), 2)

    def test_market_probabilities_shift_the_model_towards_the_official_odds(self):
        """官方赔率权重 0.8：融合结果必须明显偏向市场，不只是「变了」。"""
        model_only = lottery.lottery_market_probabilities(self.CANDIDATES, 1)
        blended = lottery.lottery_market_probabilities(
            self.CANDIDATES, 1, spf_odds={'胜': 5.0, '平': 4.0, '负': 1.5})
        market = blended['standard']['market_probabilities']
        self.assertIsNotNone(market)
        self.assertEqual(blended['standard']['market_weight'], 0.80)
        for key in ('胜', '平', '负'):
            with self.subTest(key=key):
                expected = (0.2 * model_only['standard']['probabilities'][key]
                            + 0.8 * market[key])
                self.assertAlmostEqual(blended['standard']['probabilities'][key],
                                       expected, places=9)

    def test_odds_of_one_or_less_are_treated_as_unpriced(self):
        """赔率 ≤ 1 不是低赔，是无效——整组市场概率作废。"""
        self.assertIsNone(lottery.lottery_odds_probabilities(
            {'胜': 1.0, '平': 3.0, '负': 4.0}, ('胜', '平', '负')))
        self.assertIsNotNone(lottery.lottery_odds_probabilities(
            {'胜': 1.01, '平': 3.0, '负': 4.0}, ('胜', '平', '负')))

    def test_availability_gate_blanks_the_standard_market_in_place(self):
        """就地改传入的 dict——迁移前就是这个契约。"""
        closed = {'offer_matched': True, 'spf_available': False,
                  'standard': 'S', 'joint_recommendation': 'J', 'linked_recommendation': 'L'}
        self.assertFalse(lottery.apply_lottery_market_availability(closed))
        self.assertIsNone(closed['standard'])
        self.assertIsNone(closed['joint_recommendation'])
        self.assertIsNone(closed['linked_recommendation'])

        # 未匹配官方场次时不关闭——**反方向也要测**
        unmatched = {'offer_matched': False, 'spf_available': False, 'standard': 'S'}
        self.assertTrue(lottery.apply_lottery_market_availability(unmatched))
        self.assertEqual(unmatched['standard'], 'S')


class DefaultsArePartOfTheContract(unittest.TestCase):
    """判据 29：适配层每次都显式传阈值，默认值没有生产路径覆盖。"""

    def test_close_blend_weight_defaults_to_seventy_two_hundredths(self):
        self.assertAlmostEqual(parsing.blend_close_open(1.0, 0.0), 0.72)
        self.assertAlmostEqual(parsing.blend_close_open(2.0, 1.0), 1.72)

    def test_blend_close_open_returns_the_close_value_when_open_is_missing(self):
        self.assertEqual(parsing.blend_close_open(1.23, None), 1.23)

    def test_lottery_market_weight_defaults_to_eight_tenths(self):
        model = {'胜': 1.0, '平': 0.0, '负': 0.0}
        market = {'胜': 0.0, '平': 1.0, '负': 0.0}
        blended = lottery.blend_lottery_probabilities(model, market)
        self.assertAlmostEqual(blended['胜'], 0.2)
        self.assertAlmostEqual(blended['平'], 0.8)

    def test_market_weight_is_clamped_into_zero_one(self):
        model = {'胜': 1.0, '平': 0.0, '负': 0.0}
        market = {'胜': 0.0, '平': 1.0, '负': 0.0}
        self.assertAlmostEqual(
            lottery.blend_lottery_probabilities(model, market, 1.5)['平'], 1.0)
        self.assertAlmostEqual(
            lottery.blend_lottery_probabilities(model, market, -0.5)['胜'], 1.0)

    def test_total_line_falls_back_to_two_and_a_half(self):
        self.assertEqual(parsing.get_close_total_line({}), 2.5)
        self.assertEqual(parsing.parse_total_line('不是数字'), 2.5)

    def test_zero_is_treated_as_missing_by_get_close_total_line(self):
        """用的是 `or` 不是 `is None`——0 会落到下一个来源。

        大小球线不可能是 0，所以这个落差无害；钉住是因为它不明显。
        """
        self.assertEqual(parsing.get_close_total_line({'close_line': 0, 'line': 3.0}), 3.0)
        self.assertEqual(parsing.get_close_total_line({'close_line': 0}), 2.5)


class LeagueProfiles(unittest.TestCase):

    PROFILES = {
        'default': {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
        '英超': {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95},
        '英格兰超级联赛': {'avg_goal': 1.55, 'home_boost': 1.09, 'low_score': 0.87, 'draw_mult': 0.94},
    }

    def test_the_longest_matching_key_wins(self):
        """按键长降序匹配——短键先命中就永远轮不到长键。"""
        long_name = parsing.resolve_static_league_profile('英格兰超级联赛', self.PROFILES)
        self.assertAlmostEqual(long_name['avg_goal'], 1.55)
        short_name = parsing.resolve_static_league_profile('英超', self.PROFILES)
        self.assertAlmostEqual(short_name['avg_goal'], 1.52)

    def test_an_unknown_league_falls_back_to_default(self):
        for name in ('未知联赛', '', None):
            with self.subTest(name=name):
                profile = parsing.resolve_static_league_profile(name, self.PROFILES)
                self.assertAlmostEqual(profile['avg_goal'], 1.42)

    def test_live_profiles_need_fifty_samples_to_count(self):
        static = parsing.resolve_static_league_profile('英超', self.PROFILES)
        live = {'avg_goal': 2.0, 'draw_rate': 0.30, 'sample_size': 49}
        self.assertEqual(
            parsing.blend_league_profiles(static, live, '英超')['source'], 'static')
        live['sample_size'] = 50
        self.assertEqual(
            parsing.blend_league_profiles(static, live, '英超')['source'], 'blended')

    def test_only_avg_goal_and_draw_mult_are_blended(self):
        """`home_boost` / `low_score` 直接取静态值——实时画像里没有这两项。"""
        static = parsing.resolve_static_league_profile('英超', self.PROFILES)
        live = {'avg_goal': 2.0, 'draw_rate': 0.50, 'sample_size': 200}
        blended = parsing.blend_league_profiles(static, live, '英超')
        self.assertAlmostEqual(blended['avg_goal'], 0.7 * 1.52 + 0.3 * 2.0)
        self.assertAlmostEqual(blended['draw_mult'], 0.7 * 0.95 + 0.3 * (0.50 / 0.25))
        self.assertAlmostEqual(blended['home_boost'], 1.08)
        self.assertAlmostEqual(blended['low_score'], 0.88)
        self.assertEqual(blended['live_sample'], 200)

    def test_the_fallback_goal_average_only_fires_on_an_incomplete_profile(self):
        """`DEFAULT_AVG_GOAL` 是 `.get(..., default)` 的兜底。

        两个产出画像的函数都必然带 `avg_goal`，所以生产路径走不到它——
        但 `LEAGUE_PROFILES` 是手写配置，`default` 少一个键就会落到这里，
        属判据 9 第二行「配置让它不可达」→ **补用例，不要删**。
        """
        static = {'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95}
        live = {'avg_goal': 2.0, 'draw_rate': 0.25, 'sample_size': 200}
        blended = parsing.blend_league_profiles(static, live, '英超')
        self.assertAlmostEqual(blended['avg_goal'], 0.7 * 1.42 + 0.3 * 2.0)

        # 反过来：实时画像缺 avg_goal 时同样落到兜底
        no_live_goal = parsing.blend_league_profiles(
            {'avg_goal': 1.52}, {'draw_rate': 0.25, 'sample_size': 200}, '英超')
        self.assertAlmostEqual(no_live_goal['avg_goal'], 0.7 * 1.52 + 0.3 * 1.42)

    def test_both_profile_producers_always_include_the_goal_average(self):
        """把「生产路径走不到兜底」这件事也钉住——否则上一条读起来像在测主路径。"""
        self.assertIn('avg_goal', parsing.league_profile_from_matches([{'score': '1-1'}]))
        self.assertIn('avg_goal', parsing.resolve_static_league_profile(
            '任意联赛', self.PROFILES))

    def test_league_profile_from_matches_skips_unparsable_scores(self):
        profile = parsing.league_profile_from_matches(
            [{'score': '2-1'}, {'score': 'bad'}, {'score': None},
             {'score': '1-1'}, {'no_score': 1}])
        self.assertEqual(profile['sample_size'], 2)
        self.assertAlmostEqual(profile['avg_goal'], 5 / 4)
        self.assertAlmostEqual(profile['draw_rate'], 0.5)

    def test_no_usable_match_returns_none_not_a_zero_division(self):
        for matches in ([], None, [{'score': 'bad'}], [{}]):
            with self.subTest(matches=matches):
                self.assertIsNone(parsing.league_profile_from_matches(matches))


class TeamStrength(unittest.TestCase):

    HTML = ''.join(
        f'<div>{team} 近{games}场战绩 <span class="ying">{w}胜</span>'
        f'<span class="ping">{d}平</span><span class="shu">{l}负</span>'
        f'进<span class="ying">{gf}球</span>失<span class="shu">{ga}球</span></div>'
        for team, games, w, d, l, gf, ga in (
            ('阿森纳', 10, 6, 2, 2, 18, 9),
            ('阿森纳', 5, 4, 1, 0, 12, 3),
            ('切尔西', 10, 3, 3, 4, 11, 13),
            ('切尔西', 5, 1, 2, 2, 4, 7),
        ))

    def test_the_first_row_is_overall_and_the_second_is_venue(self):
        result = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西')
        self.assertEqual(result['home_recent']['games'], 10)
        self.assertEqual(result['home_venue']['games'], 5)
        self.assertEqual(result['away_recent']['games'], 10)
        self.assertEqual(result['away_venue']['games'], 5)

    def test_attack_blends_venue_over_overall_at_the_given_weight(self):
        result = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西',
                                             venue_weight=0.68)
        self.assertAlmostEqual(result['attack_home'], 0.68 * (12 / 5) + 0.32 * (18 / 10))

    def test_momentum_is_clamped_both_ways(self):
        result = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西',
                                             momentum_scale=100.0)
        self.assertAlmostEqual(result['momentum_supremacy'], 0.35)
        flipped = parsing.parse_team_strength(self.HTML, '切尔西', '阿森纳',
                                              momentum_scale=100.0)
        self.assertAlmostEqual(flipped['momentum_supremacy'], -0.35)

    def test_fewer_than_two_blocks_returns_none(self):
        single = self.HTML[:self.HTML.index('</div>') + 6]
        self.assertIsNone(parsing.parse_team_strength(single, '阿森纳', '切尔西'))

    def test_an_unmatched_team_returns_none(self):
        self.assertIsNone(parsing.parse_team_strength(self.HTML, '皇马', '巴萨'))

    def test_the_context_window_is_what_separates_the_teams(self):
        """队名只在战绩前 140 字符内找。断言的是**原因**：窗口太小谁也匹配不上。"""
        self.assertIsNotNone(parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西'))
        self.assertIsNone(parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西',
                                                      context_chars=1))

    def test_too_wide_a_window_makes_the_teams_bleed_into_each_other(self):
        """窗口的**上界**才是它存在的理由：开太大，客队块的上下文里会先撞上
        主队名（`team_in_context(ctx, home)` 排在前面），四块全归主队，
        `away_all` 落空 → 返回 None。这正是代码注释说的「避免多场数据串台」。
        """
        self.assertIsNotNone(parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西',
                                                         context_chars=140))
        self.assertIsNone(parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西',
                                                      context_chars=400))

    def test_the_exact_window_size_is_not_load_bearing_on_real_pages(self):
        """**140 这个数本身没有守住任何东西**——实测真实页面上 15 到 1180
        之间任何窗口都得到同一结果。所以把 `TEAM_CONTEXT_CHARS` 改成 20
        是**等价变异**，不是漏测（判据 30）。

        写下来免得下一个人去补一个补不出来的用例。真正承重的是两端：
        太窄谁也匹配不上，太宽会串台——上面两条各守一端。
        """
        real = PAGES['shuju:1430017']
        baseline = parsing.parse_team_strength(real, '波鸿', '奥斯纳布吕克')
        self.assertIsNotNone(baseline)
        for chars in (20, 60, 140, 400, 1000):
            with self.subTest(chars=chars):
                self.assertEqual(
                    parsing.parse_team_strength(real, '波鸿', '奥斯纳布吕克',
                                                context_chars=chars),
                    baseline)

    def test_a_venue_weight_change_actually_moves_the_numbers(self):
        """**迁移时在这里踩过一次**：适配层原本传了 `CLOSE_BLEND_WEIGHT`(0.72)，
        而迁移前这里硬编码的是 0.68——两个数长得像但不是一回事。

        新旧双跑差分没抓住，因为当时用的队名在真实页面里不存在、每条都返回
        None（判据 8）。变异验证抓住了。这条断言保证权重真的在起作用。
        """
        low = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西', venue_weight=0.68)
        high = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西', venue_weight=0.72)
        self.assertNotEqual(low['attack_home'], high['attack_home'])
        self.assertAlmostEqual(low['attack_home'], 0.68 * (12 / 5) + 0.32 * (18 / 10))
        self.assertAlmostEqual(high['attack_home'], 0.72 * (12 / 5) + 0.28 * (18 / 10))
        # 不传参数时用的是 0.68
        default = parsing.parse_team_strength(self.HTML, '阿森纳', '切尔西')
        self.assertEqual(default['attack_home'], low['attack_home'])

    def test_a_three_character_suffix_is_what_matches_a_misspelled_name(self):
        """真实语料里赛程写「不伦瑞克」而页面写「布伦瑞克」——**首字都不同**。

        救回来的是 3 字尾串 `伦瑞克`。把 `TEAM_SUFFIX_LENGTHS` 砍成 `(4,)`
        这一场就匹配不上了，而只测 4 字尾串是发现不了的（判据 5）。
        """
        ctx = '大 布伦瑞克'
        self.assertTrue(parsing.team_in_context(ctx, '不伦瑞克'))
        self.assertFalse(parsing.team_in_context(ctx, '曼彻斯特联'))
        html = self.HTML.replace('阿森纳', '布伦瑞克')
        self.assertIsNotNone(parsing.parse_team_strength(html, '不伦瑞克', '切尔西'))


class TeamNameSuffixMatching(unittest.TestCase):
    """`TEAM_SUFFIX_LENGTHS = (4, 3, 2)` 每一档都要有样本。"""

    def test_each_suffix_length_is_load_bearing(self):
        # 4 字尾串：曼彻斯特联 → 彻斯特联
        self.assertTrue(parsing.team_in_context('主队 彻斯特联 近', '曼彻斯特联'))
        # 3 字尾串：不伦瑞克 → 伦瑞克（首字都不同，真实语料里的情况）
        self.assertTrue(parsing.team_in_context('大 布伦瑞克', '不伦瑞克'))
        # 2 字尾串：北京国安 → 国安
        self.assertTrue(parsing.team_in_context('主队 国安 近', '北京国安'))

    def test_shortening_the_suffix_list_breaks_the_three_character_case(self):
        """**断言的是原因**：只留 4 字尾串，`不伦瑞克` 就匹配不上 `布伦瑞克`。"""
        import src.domain.sports.football.parsing as module
        original = module.TEAM_SUFFIX_LENGTHS
        try:
            module.TEAM_SUFFIX_LENGTHS = (4,)
            self.assertFalse(parsing.team_in_context('大 布伦瑞克', '不伦瑞克'))
            module.TEAM_SUFFIX_LENGTHS = (4, 3)
            self.assertTrue(parsing.team_in_context('大 布伦瑞克', '不伦瑞克'))
            module.TEAM_SUFFIX_LENGTHS = (4, 3)
            self.assertFalse(parsing.team_in_context('主队 国安 近', '北京国安'))
        finally:
            module.TEAM_SUFFIX_LENGTHS = original

    def test_an_empty_name_never_matches(self):
        for name in ('', None):
            with self.subTest(name=name):
                self.assertFalse(parsing.team_in_context('任何上下文', name))


class DrawCoverThresholdsAreCoupled(unittest.TestCase):
    """`DRAW_COVER_MIN_PROBABILITY = 0.24` 在归一化输入下永远不单独决定结果。

    判据 28「条件之间是耦合的」——与北单 `upset` 那三个门槛同一形状。
    证明：`平 < 0.24` 且 `首选 - 平 <= 0.12` ⇒ `首选 <= 0.36`，
    于是第三项 `= 1 - 首选 - 平 >= 0.40 > 首选`，与「首选是最大值」矛盾。

    所以把这个常量从 0.24 改到 0.10，**任何归一化输入的输出都不变**——
    变异不红不是漏测。这里把它证下来，免得下一个人去补一个补不出来的用例。
    """

    def test_no_normalised_input_can_isolate_the_draw_probability_gate(self):
        step = 0.005
        counterexamples = []
        for i in range(int(1 / step) + 1):
            win = round(i * step, 3)
            for j in range(int((1 - win) / step) + 1):
                draw = round(j * step, 3)
                lose = round(1 - win - draw, 3)
                if lose < 0:
                    continue
                probs = {'胜': win, '平': draw, '负': lose}
                primary = max(probs, key=probs.get)
                if primary == '平':
                    continue
                if probs[primary] - draw <= 0.12 and draw < 0.24:
                    counterexamples.append(probs)
        self.assertEqual(counterexamples, [],
                         '若出现反例，说明这条耦合结论不再成立，注释要跟着改')

    def test_lowering_the_gate_changes_nothing_for_normalised_inputs(self):
        """直接验一遍：把门槛降到 0.10，一组归一化样本的输出逐条不变。"""
        samples = [{'胜': w / 100, '平': d / 100, '负': (100 - w - d) / 100}
                   for w in range(0, 101, 5) for d in range(0, 101 - w, 5)]
        for probs in samples:
            with self.subTest(probs=probs):
                self.assertEqual(
                    lottery.spf_selection_profile(probs),
                    lottery.spf_selection_profile(probs, draw_cover_min=0.10))

    def test_the_gate_does_bite_on_unnormalised_input(self):
        """它不是死代码——调用方传未归一化的概率时就有作用。"""
        loose = {'胜': 0.30, '平': 0.20, '负': 0.10}
        self.assertEqual(lottery.spf_selection_profile(loose)['mode'], 'single')
        self.assertEqual(
            lottery.spf_selection_profile(loose, draw_cover_min=0.10)['mode'], 'draw_cover')


class OuzhiSeriesContract(unittest.TestCase):
    """欧赔序列**倒序**：第 0 条是终盘、最后一条是初盘。"""

    SERIES = [[2.0, 3.4, 3.8, 93.1], [2.1, 3.3, 3.7, 93.0], [2.5, 3.2, 3.0, 92.9]]

    def test_the_first_record_is_close_and_the_last_is_open(self):
        result = parsing.ouzhi_from_series(self.SERIES, 'm1')
        self.assertEqual(result['close']['home'], 2.0)
        self.assertEqual(result['open']['home'], 2.5)
        self.assertEqual(result['series'], self.SERIES)

    def test_every_malformed_series_raises_value_error(self):
        for bad in ([], {'a': 1}, [[2.0, 3.0]] + SERIES if (SERIES := self.SERIES) else [],
                    [[0, 3.0, 4.0, 93.0]] + self.SERIES,
                    [['x', 3.0, 4.0, 93.0]] + self.SERIES):
            with self.subTest(bad=str(bad)[:40]):
                with self.assertRaises(ValueError):
                    parsing.ouzhi_from_series(bad, 'm1')

    def test_a_missing_return_rate_becomes_none_not_an_error(self):
        result = parsing.ouzhi_from_series([[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]], 'm1')
        self.assertIsNone(result['close']['return_rate'])
        self.assertIsNone(result['open']['return_rate'])


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'urllib.error', 'src.common.kv_store', 'src.foundation.store',
                     'src.football.fetching', 'src.football.config'}
FORBIDDEN_CALLS = {'now', 'today', 'utcnow', 'strftime'}


class NoSideEffectTests(unittest.TestCase):
    """领域层不许碰存储、网络、时钟与全局配置（判据 16）。

    守卫本身也要能被证伪——拿适配层去试，它**应该**命中。
    """

    DOMAIN = ('src/domain/sports/football/parsing.py',
              'src/domain/sports/football/lottery.py',
              'src/domain/sports/football/markets.py')
    ADAPTER = 'src/football/parsing.py'

    def _tree(self, path):
        return ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))

    def _imports(self, path):
        found = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def _clock_calls(self, path):
        return {node.func.attr for node in ast.walk(self._tree(path))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in FORBIDDEN_CALLS}

    def test_domain_imports_nothing_stateful(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_domain_never_reads_the_clock(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._clock_calls(path), set())

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
