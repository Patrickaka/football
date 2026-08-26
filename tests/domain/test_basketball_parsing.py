"""500 源赛程解析迁入领域层。

夹具是**线上真实页面**（`tests/fixtures/basketball/jclq_500.html`，2026-08-26 抓取），
不是手搓的 HTML——按判据 4，解析器的正确性只能对真实结构验证。

差分测试仍是主体：旧解析器要到端点切换才删，对同一份 HTML 跑新旧两份、
断言输出逐字相等。
"""
import pathlib
import unittest
from datetime import datetime
from unittest import mock

import src.basketball as legacy
from src.domain.sports.basketball import parsing

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / 'fixtures' / 'basketball'
JCLQ_HTML = (FIXTURES / 'jclq_500.html').read_text(encoding='utf-8')
NOW = datetime(2026, 8, 26, 12, 0, 0)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def _legacy_schedule(date, html=JCLQ_HTML):
    with mock.patch.object(legacy, 'fetch', lambda *a, **k: html), \
         mock.patch.object(legacy, 'datetime', _FrozenDatetime):
        return legacy.fetch_basketball_schedule(date)


class RealPageParityTests(unittest.TestCase):
    DATES = ['2026-08-26', '2026-08-27', '2027-01-05']

    def test_parsed_matches_equal_legacy(self):
        for date in self.DATES:
            with self.subTest(date=date):
                self.assertEqual(
                    parsing.parse_schedule(JCLQ_HTML, date),
                    _legacy_rows(date))

    def test_full_fetch_equals_legacy(self):
        for date in self.DATES:
            with self.subTest(date=date):
                fetcher = parsing.ScheduleFetcher(
                    transport=lambda url: JCLQ_HTML, now_fn=lambda: NOW)
                self.assertEqual(fetcher.fetch(date), _legacy_schedule(date))

    def test_real_page_yields_the_expected_card(self):
        """把真实页面的解析结果钉死，正则被改动时能立刻看出差别。"""
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        self.assertEqual(len(matches), 2)
        first = matches[0]
        self.assertEqual(first['home'], '金州女武神')
        self.assertEqual(first['away'], '太阳')
        self.assertEqual(first['league'], '美职女篮')
        self.assertEqual(first['num'], '周三301')
        self.assertEqual(first['handicap'], '+13.5')
        self.assertEqual(first['total_line'], 150.5)
        self.assertEqual(first['dx_over'], 1.62)
        self.assertEqual(first['dx_under'], 1.78)
        self.assertIsNone(first['spf_home'], '这场没开胜负盘，缺赔率要留 None')
        self.assertEqual(first['id'], '2026-08-27_金州女武神_太阳')
        self.assertEqual(matches[1]['spf_home'], 2.55)
        self.assertEqual(matches[1]['spf_away'], 1.27)


def _legacy_rows(date):
    """旧实现没有独立的解析入口，用「时间冻结在很久以前」拿到未经状态过滤的行。"""
    class _Ancient(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2000, 1, 1)

    with mock.patch.object(legacy, 'fetch', lambda *a, **k: JCLQ_HTML), \
         mock.patch.object(legacy, 'datetime', _Ancient):
        return legacy.fetch_basketball_schedule(date)


class TomorrowFallbackTests(unittest.TestCase):
    """今日无赛时回退次日。旧实现把整段行解析抄了第二遍，两份必须行为一致。"""

    EMPTY = '<html><body><table></table></body></html>'

    def test_falls_back_to_tomorrow_when_today_is_empty(self):
        requested = []

        def transport(url):
            requested.append(url)
            return self.EMPTY if len(requested) == 1 else JCLQ_HTML

        fetcher = parsing.ScheduleFetcher(transport=transport, now_fn=lambda: NOW)
        matches = fetcher.fetch('2026-08-26')
        self.assertEqual(len(requested), 2)
        self.assertIn('date=2026-08-27', requested[1])
        self.assertTrue(matches)

    def test_tomorrow_rows_are_dated_by_tomorrow(self):
        """回退时基准日期换成次日——年份从它取，行内只给月日。"""
        def transport(url):
            return self.EMPTY if 'date=2026-12-31' in url else JCLQ_HTML

        fetcher = parsing.ScheduleFetcher(transport=transport, now_fn=lambda: NOW)
        with mock.patch.object(parsing, 'datetime', _FrozenDatetime):
            matches = fetcher.fetch('2026-12-31')
        self.assertTrue(matches)
        self.assertTrue(all(m['date'].startswith('2026-') for m in matches))

    def test_fallback_path_equals_legacy(self):
        """回退路径在旧实现里是**另一份抄写**，必须单独差分——主路径相等
        不能证明它也相等。"""
        def html_for(url):
            return self.EMPTY if 'date=2026-08-26' in url else JCLQ_HTML

        with mock.patch.object(
                legacy, 'fetch',
                lambda url, encoding='utf-8', referer=None: html_for(url)), \
             mock.patch.object(legacy, 'datetime', _FrozenDatetime):
            expected = legacy.fetch_basketball_schedule('2026-08-26')

        fetcher = parsing.ScheduleFetcher(transport=html_for, now_fn=lambda: NOW)
        self.assertEqual(fetcher.fetch('2026-08-26'), expected)
        self.assertTrue(expected, '回退没取到任何场次，这条差分是空跑')

    def test_no_fallback_when_today_has_matches(self):
        requested = []
        fetcher = parsing.ScheduleFetcher(
            transport=lambda url: requested.append(url) or JCLQ_HTML,
            now_fn=lambda: NOW)
        fetcher.fetch('2026-08-26')
        self.assertEqual(len(requested), 1)

    def test_empty_both_days_returns_empty(self):
        fetcher = parsing.ScheduleFetcher(transport=lambda url: self.EMPTY,
                                          now_fn=lambda: NOW)
        self.assertEqual(fetcher.fetch('2026-08-26'), [])

    def test_transport_failure_returns_empty(self):
        def boom(url):
            raise IOError('源站挂了')

        fetcher = parsing.ScheduleFetcher(transport=boom, now_fn=lambda: NOW)
        self.assertEqual(fetcher.fetch('2026-08-26'), [])


class StatusFilterTests(unittest.TestCase):
    """开赛状态判定与过滤。"""

    # 解析器出来的行一定带 status，合成输入也照这个形状
    ROWS = [
        {'date': '2026-08-26', 'time': '20:00', 'home': 'A', 'status': 'not_started'},
        {'date': '2026-08-26', 'time': '12:30', 'home': 'C', 'status': 'not_started'},
        {'date': '2026-08-26', 'time': '05:00', 'home': 'E', 'status': 'not_started'},
        {'date': '2026-08-26', 'time': '11:30', 'home': 'G', 'status': 'not_started'},
    ]

    def test_status_by_kickoff_distance(self):
        rows = [dict(r) for r in self.ROWS]
        parsing.annotate_status(rows, NOW)
        self.assertEqual([r['status'] for r in rows],
                         ['not_started', 'in_progress', 'finished', 'in_progress'])

    def test_started_matches_are_dropped(self):
        """分界是开赛时刻本身：12:30 那场在 12:00 尚未开赛，仍要留下。"""
        rows = [dict(r) for r in self.ROWS]
        parsing.annotate_status(rows, NOW)
        kept = parsing.select_upcoming(rows, NOW)
        self.assertEqual([r['home'] for r in kept], ['A', 'C'])

    def test_unparsable_time_keeps_the_row(self):
        """时间格式不认识时不猜、不丢——宁可多显示一场，也不静默吞掉。"""
        rows = [{'date': '2026-08-26', 'time': '不是时间', 'home': 'X',
                 'status': 'not_started'}]
        parsing.annotate_status(rows, NOW)
        self.assertEqual(rows[0].get('status'), 'not_started')
        self.assertEqual(parsing.select_upcoming(rows, NOW), rows)


class KnownDefectTests(unittest.TestCase):
    """把一处**已知缺陷**钉住，避免它在后续重构里被当成正常行为默认下来。

    500 源页面的时间单元格是 `08-27 07:00`（带月日），而状态判定按
    `%Y-%m-%d %H:%M` 解析 `f"{date} {time}"`，拼出来是
    `2026-08-27 08-27 07:00`，必然 ValueError 并被静默吞掉。后果是
    **500 源的开赛过滤从来没生效过**，所有场次恒为 not_started。

    本批不改它：改了就破坏差分测试这个唯一的正确性依据。修复排在端点
    切换那一批，那时行为变化能在线上直接验证。
    """

    def test_500_source_time_carries_month_day_and_defeats_the_filter(self):
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        self.assertTrue(all(' ' in m['time'] for m in matches),
                        '页面时间格式变了，这处缺陷的前提需要重新确认')
        long_past = datetime(2030, 1, 1)
        parsing.annotate_status(matches, long_past)
        self.assertTrue(all(m['status'] == 'not_started' for m in matches),
                        '状态判定开始生效了——缺陷已被修，请同步删掉本测试')
        self.assertEqual(len(parsing.select_upcoming(matches, long_past)),
                         len(matches))


class NoLegacyImportTests(unittest.TestCase):
    def test_parsing_does_not_import_legacy_package(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(parsing))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith('src.basketball'))
            elif isinstance(node, ast.ImportFrom):
                module = ('.' * (node.level or 0)) + (node.module or '')
                self.assertFalse(module.startswith('src.basketball'), module)
                self.assertFalse(module.startswith('.'), f'不该有相对导入: {module}')


if __name__ == '__main__':
    unittest.main()
