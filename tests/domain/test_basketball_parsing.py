"""500 源赛程解析迁入领域层。

夹具是**线上真实页面**（`tests/fixtures/basketball/jclq_500.html`，2026-08-26 抓取），
不是手搓的 HTML——按判据 4，解析器的正确性只能对真实结构验证。

差分测试仍是主体：旧解析器要到端点切换才删，对同一份 HTML 跑新旧两份、
断言输出逐字相等。
"""
import gzip
import pathlib
import unittest
from datetime import datetime
from unittest import mock

import src.basketball as legacy
from src.domain.sports.basketball import parsing

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / 'fixtures' / 'basketball'
# 夹具按 gzip 存放：真实页面 80KB 起步，压缩后不到四分之一，
# 而解析必须对完整页面做——截断片段会把「正则匹到了别处」这类问题掩盖掉。
JCLQ_HTML = gzip.open(FIXTURES / 'jclq_500.html.gz', 'rt', encoding='utf-8').read()
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

    def test_kickoff_moment_itself_counts_as_started(self):
        """分界取 `>`：恰好到点的那一场已经开赛，要撤下；下一分钟的留着。"""
        at_kickoff = {'date': '2026-08-26', 'time': '12:00', 'home': 'Z',
                      'status': 'not_started'}
        one_minute_later = {'date': '2026-08-26', 'time': '12:01', 'home': 'W',
                            'status': 'not_started'}
        self.assertEqual(
            parsing.select_upcoming([at_kickoff, one_minute_later], NOW),
            [one_minute_later])

    def test_unparsable_time_keeps_the_row(self):
        """时间格式不认识时不猜、不丢——宁可多显示一场，也不静默吞掉。"""
        rows = [{'date': '2026-08-26', 'time': '不是时间', 'home': 'X',
                 'status': 'not_started'}]
        parsing.annotate_status(rows, NOW)
        self.assertEqual(rows[0].get('status'), 'not_started')
        self.assertEqual(parsing.select_upcoming(rows, NOW), rows)


def _row(cells):
    return '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'


def _spans(*values):
    return ''.join(f'<span>{v}</span>' for v in values)


def _page(*rows):
    return '<html><body><table>' + ''.join(rows) + '</table></body></html>'


GOOD_CELLS = ['周三301', '美职女篮', '08-27 07:00', '[主]甲队VS乙队[客]',
              _spans('1.80', '2.00'), _spans('1.90', '-3.5', '1.90'),
              _spans('1.85', '210.5', '1.95')]


class MalformedRowTests(unittest.TestCase):
    """真实页面只覆盖了「一切正常」这一种行。表头、广告、脏数据这些
    每天都会出现的行，只能合成——它们正是每一道守卫存在的理由。"""

    def _parse(self, cells):
        return parsing.parse_schedule(_page(_row(cells)), '2026-08-27')

    def test_good_row_is_the_control(self):
        self.assertEqual(len(self._parse(GOOD_CELLS)), 1)

    def test_row_with_too_few_cells_is_skipped(self):
        self.assertEqual(self._parse(GOOD_CELLS[:6]), [])

    def test_numeric_first_cell_is_not_a_match(self):
        """表头与统计行的首格是纯数字或空。期号必然以中文或字母开头，
        这道校验去掉后它们会被当成比赛。"""
        for num in ('123', '', '   ', '2026-08-27'):
            with self.subTest(num=num):
                self.assertEqual(self._parse([num] + GOOD_CELLS[1:]), [])

    def test_all_three_versus_markers_are_recognised(self):
        for marker, expect_home in (('VS', '甲队'), ('vs', '甲队'), ('对', '甲队')):
            with self.subTest(marker=marker):
                cells = list(GOOD_CELLS)
                cells[3] = f'[主]甲队{marker}乙队[客]'
                matches = self._parse(cells)
                self.assertEqual(len(matches), 1, f'分隔符 {marker} 没被认出来')
                self.assertEqual(matches[0]['home'], expect_home)

    def test_row_without_versus_marker_is_skipped(self):
        cells = list(GOOD_CELLS)
        cells[3] = '甲队 乙队'
        self.assertEqual(self._parse(cells), [])

    def test_row_that_raises_mid_parse_is_skipped_not_fatal(self):
        """`1.2.3` 能通过 isdigit 检查却过不了 float——脏数据要吞掉这一行，
        而不是让整页解析失败。"""
        cells = list(GOOD_CELLS)
        cells[4] = _spans('1.2.3', '2.00')
        good = _row(GOOD_CELLS)
        page = _page(_row(cells), good)
        matches = parsing.parse_schedule(page, '2026-08-27')
        self.assertEqual(len(matches), 1, '脏行没被跳过，或把好行一起带走了')

    def test_missing_odds_stay_none(self):
        cells = list(GOOD_CELLS)
        cells[4] = ''
        match = self._parse(cells)[0]
        self.assertIsNone(match['spf_home'])
        self.assertIsNone(match['spf_away'])


class KickoffFromRealPageTests(unittest.TestCase):
    """开赛过滤在 500 源上曾经完全失效，这里守住修复。

    页面的时间单元格是 `08-27 07:00`（带月日），迁移前直接拿它去拼
    `%Y-%m-%d %H:%M`，拼出 `2026-08-27 08-27 07:00`，必然 ValueError 且被
    静默吞掉。后果是所有场次恒为 not_started，打完的比赛照样出现在推荐里。

    这条缺陷在解析器迁移那一批里被原样搬过来并写测试钉住（当时不能改：
    差分测试是唯一的正确性依据），端点切换时一并修掉。
    """

    def test_time_cell_still_carries_month_and_day(self):
        """修复的前提。页面格式若改回纯时分，这里会先亮。"""
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        self.assertTrue(matches)
        self.assertTrue(all(' ' in m['time'] for m in matches),
                        [m['time'] for m in matches])

    def test_status_is_annotated_on_real_rows(self):
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        parsing.annotate_status(matches, datetime(2030, 1, 1))
        self.assertEqual({m['status'] for m in matches}, {'finished'},
                         '开赛过滤仍未生效——时间解析又回到静默失败了')

    def test_finished_matches_are_dropped_from_the_card(self):
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        self.assertEqual(parsing.select_upcoming(matches, datetime(2030, 1, 1)), [])

    def test_upcoming_matches_are_kept(self):
        matches = parsing.parse_schedule(JCLQ_HTML, '2026-08-27')
        self.assertEqual(len(parsing.select_upcoming(matches, datetime(2020, 1, 1))),
                         len(matches))

    def test_divergence_from_legacy_is_explicit(self):
        """与旧实现的**唯一**分歧，写出来免得它藏在差分测试的盲区里。

        上面那几条差分测试用的时钟都落在比赛之前，两侧都判「未开赛」，
        正好走不到分歧点。把时钟推到比赛之后，差别才显现：旧实现仍然
        把打完的比赛当成未开赛返回，新实现把它们撤下。
        """
        after = datetime(2030, 1, 1)

        class _Late(datetime):
            @classmethod
            def now(cls, tz=None):
                return after

        with mock.patch.object(legacy, 'fetch', lambda *a, **k: JCLQ_HTML), \
             mock.patch.object(legacy, 'datetime', _Late):
            legacy_result = legacy.fetch_basketball_schedule('2026-08-27')

        fetcher = parsing.ScheduleFetcher(transport=lambda url: JCLQ_HTML,
                                           now_fn=lambda: after)
        self.assertTrue(legacy_result, '旧实现本该把打完的比赛也返回')
        self.assertTrue(all(m['status'] == 'not_started' for m in legacy_result))
        self.assertEqual(fetcher.fetch('2026-08-27'), [])

    def test_plain_clock_without_month_day_still_works(self):
        """澳客那边给的是纯 `07:00`，同一个函数两种输入都要认。"""
        rows = [{'date': '2026-08-27', 'time': '07:00', 'status': 'not_started'}]
        parsing.annotate_status(rows, datetime(2026, 8, 27, 12, 0))
        self.assertEqual(rows[0]['status'], 'finished')


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
