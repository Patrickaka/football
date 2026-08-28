# -*- coding: utf-8 -*-
"""盘口口径归一、相似盘口匹配、聚类先验与资金流突变。

参照物是从迁移前四个模块生成的黄金文件
（`tests/fixtures/golden/football_market.json.gz`，201 条），**逐条相同**。
迁移当时另跑过 **165 条**新旧双跑差分，零差异。

**`_normalize_match_time` 的时钟已注入**（判据 16）：源站的时间串常常不带年，
补的是「当前年」——不注入的话黄金跨年就红。这是这个模块里唯一的时钟依赖。

**`half_full_probs_from_records` 恒返回空字典**——它的 docstring 明写
「已废弃，不再生成伪半场数据」，半全场概率应从 `HalfTimeStatsDB` 取真实数据。
覆盖报告里它「一次都没产出有效值」是正确行为，不是漏测。
"""
import ast
import datetime
import gzip
import json
import pathlib
import unittest

from src.domain.sports.football import market_matching as mm, steam
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/football_market.json.gz',
                             'rt', encoding='utf-8'))
NOW = datetime.datetime(2026, 8, 28, 12, 0, 0)


def golden_entries():
    from scripts.gen_football_market_golden import entries
    return entries()


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class HandicapNormalisation(unittest.TestCase):
    """盘口归一到标准档——**四分盘要落到最近的标准值**。"""

    def test_it_snaps_to_the_nearest_standard_line(self):
        self.assertEqual(mm.normalize_asian(-0.27), -0.25)
        self.assertEqual(mm.normalize_asian(0.13), 0.25)
        self.assertEqual(mm.normalize_asian(0.0), 0.0)

    def test_totals_snap_on_their_own_ladder(self):
        self.assertEqual(mm.normalize_ou(2.63), 2.75)
        self.assertEqual(mm.normalize_ou(2.5), 2.5)

    def test_out_of_range_values_clamp_to_the_extremes(self):
        """超出标准档两端的值要落到端点，不能越界。"""
        low = mm.normalize_asian(-10.0)
        high = mm.normalize_asian(10.0)
        self.assertEqual(low, min(mm.STANDARD_ASIAN))
        self.assertEqual(high, max(mm.STANDARD_ASIAN))

    def test_the_snap_is_symmetric_around_zero(self):
        """**对称的两条分支只测一条不够**（判据 7）。"""
        for value in (0.13, 0.27, 0.62, 1.13):
            with self.subTest(value=value):
                self.assertEqual(mm.normalize_asian(value),
                                 -mm.normalize_asian(-value))


class RecencyWeighting(unittest.TestCase):
    """**它收的是 `MatchRecord` 不是日期**——第一版用例喂了 `date`，
    每条都落到默认 0.7，看着"通过"其实什么也没测（判据 23）。
    """

    RECORD_DATA = {'league': '英超', 'season': '2526', 'date': '2026-08-28',
                   'home_team': 'A', 'away_team': 'B', 'home_odds': 2.0,
                   'draw_odds': 3.4, 'away_odds': 3.8, 'handicap': 0.5,
                   'total': 2.5, 'home_goals': 2, 'away_goals': 1, 'ftr': 'H'}

    def _weight(self, days):
        from src.football.similar_market import MatchRecord
        stamp = (datetime.date(2026, 8, 28) - datetime.timedelta(days=days)).isoformat()
        return mm._recency_weight(MatchRecord(dict(self.RECORD_DATA, date=stamp)), NOW)

    def test_recent_matches_weigh_more_and_it_decays(self):
        weights = [self._weight(d) for d in (0, 10, 90, 400, 1200)]
        for earlier, later in zip(weights, weights[1:]):
            self.assertGreaterEqual(earlier, later)
        self.assertGreater(weights[0], weights[-1])

    def test_the_weight_never_goes_negative(self):
        self.assertGreaterEqual(self._weight(20000), 0.0)

    def test_an_unparseable_date_falls_back_to_a_constant(self):
        from src.football.similar_market import MatchRecord
        fallback = mm._recency_weight(MatchRecord(dict(self.RECORD_DATA, date='bad')), NOW)
        self.assertIsInstance(fallback, float)


class SteamDetection(unittest.TestCase):

    ASIAN = {'open_handicap': 0.5, 'handicap': 0.25,
             'open_time': '2026-08-28 10:00:00', 'close_time': '2026-08-28 19:30:00',
             'open_water': {'home': 0.9, 'away': 0.9},
             'close_water': {'home': 1.05, 'away': 0.75}}
    MATCH_TIME = '2026-08-28 20:00:00'

    def test_a_moving_market_produces_signals(self):
        result = steam._analyze_asian_steam(self.ASIAN, self.MATCH_TIME)
        self.assertTrue(result['signals'])
        self.assertNotEqual(result['handicap_speed'], 0.0)

    def test_a_still_market_produces_none(self):
        """**反方向**：盘口水位都不动就不该报信号。"""
        still = dict(self.ASIAN, handicap=0.5,
                     close_water={'home': 0.9, 'away': 0.9})
        result = steam._analyze_asian_steam(still, self.MATCH_TIME)
        self.assertEqual(result['signals'], [])
        self.assertEqual(result['handicap_speed'], 0.0)

    def test_missing_data_degrades_instead_of_crashing(self):
        for data in ({}, {'handicap': 0.5}, {'open_water': {}}):
            with self.subTest(data=data):
                result = steam._analyze_asian_steam(data, self.MATCH_TIME)
                self.assertEqual(result['signals'], [])

    def test_time_diff_needs_the_full_timestamp_format(self):
        """只认 `%Y-%m-%d %H:%M:%S`——少了秒或年就返回 None。"""
        self.assertEqual(steam._calculate_time_diff('2026-08-28 10:00:00',
                                                    '2026-08-28 18:00:00'), 480.0)
        self.assertIsNone(steam._calculate_time_diff('08-28 10:00', '08-28 18:00'))
        self.assertIsNone(steam._calculate_time_diff('', ''))

    def test_a_backwards_interval_is_clamped_not_negative(self):
        self.assertEqual(steam._calculate_time_diff('2026-08-28 18:00:00',
                                                    '2026-08-28 10:00:00'), 0.1)


class MatchTimeNeedsAnInjectedClock(unittest.TestCase):
    """判据 16：不带年的时间串补的是「当前年」——那是时钟依赖。"""

    def test_a_yearless_stamp_takes_the_injected_year(self):
        """**返回的是字符串不是 datetime**（第一版用例取 `.year` 直接炸）。"""
        self.assertEqual(
            steam._normalize_match_time('03-15 20:00', now=datetime.datetime(2020, 1, 1)),
            '2020-03-15 20:00:00')
        self.assertEqual(
            steam._normalize_match_time('03-15 20:00', now=datetime.datetime(2031, 1, 1)),
            '2031-03-15 20:00:00')

    def test_a_full_stamp_ignores_the_injected_clock(self):
        """**反方向**：带年的串不该被覆盖，否则上一条对任何输入都成立。"""
        self.assertEqual(
            steam._normalize_match_time('2026-08-28 20:00',
                                        now=datetime.datetime(2020, 1, 1)),
            '2026-08-28 20:00:00')

    def test_unparseable_input_comes_back_unchanged_not_none(self):
        """**认不出的串原样返回**，不是 None（第一版用例断言 None，实测不是）。

        空串与 None 走 falsy 早退返回 None；而 `'bad'` **原样返回 `'bad'`**——
        所以下游拿到的可能是一个没法解析的时间串，而不是一个明确的"没有"。
        这个落差没有别的东西盯着（判据 12）。
        """
        self.assertEqual(steam._normalize_match_time('bad', now=NOW), 'bad')
        # 空串与 None 走的是「falsy 直接返回 None」那条早退
        self.assertIsNone(steam._normalize_match_time('', now=NOW))
        self.assertIsNone(steam._normalize_match_time(None, now=NOW))


class DeprecatedHalfFullProbs(unittest.TestCase):
    """把「恒空是正确行为」钉住，免得下一个人当漏测去补。"""

    def test_it_always_returns_an_empty_mapping(self):
        for asian, ou in ((0.0, 2.5), (-0.5, 2.25), (1.5, 3.5)):
            with self.subTest(asian=asian, ou=ou):
                self.assertEqual(mm.half_full_probs_from_records(asian, ou), {})


FORBIDDEN_IMPORTS = {'os', 'pathlib', 'requests', 'urllib.request', 'csv',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.common.match_store', 'src.football.config'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = ('src/domain/sports/football/market_matching.py',
              'src/domain/sports/football/steam.py')
    ADAPTER = 'src/football/similar_market.py'

    def _imports(self, path):
        found = set()
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def test_domain_imports_nothing_stateful(self):
        for path in self.DOMAIN:
            with self.subTest(path=path):
                self.assertEqual(self._imports(path) & FORBIDDEN_IMPORTS, set())

    def test_the_domain_never_reads_the_wall_clock_directly(self):
        """`datetime.now()` 只允许出现在**默认参数的兜底**里，不能是硬依赖。"""
        for path in self.DOMAIN:
            source = pathlib.Path(path).read_text(encoding='utf-8')
            with self.subTest(path=path):
                for line in source.splitlines():
                    if 'datetime.now()' in line:
                        self.assertIn('or datetime.now()', line,
                                      f'{path} 里有一处不可注入的时钟: {line.strip()}')

    def test_the_guard_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())


if __name__ == '__main__':
    unittest.main()
