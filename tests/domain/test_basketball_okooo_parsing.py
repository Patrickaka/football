"""澳客页面的解析迁入领域层。

四份夹具都是线上真实页面（2026-08-26 抓取）：混合过关赛程页，以及一场比赛的
欧赔 / 让分 / 大小三张详情页。

**抓夹具时确认的一件事**：澳客详情页在服务器 IP 上稳定返回 WAF 拦截页，
换本机 IP 立刻通——封锁是按 IP 的。也就是说 ml/ah/ou 这三张页面的解析
在线上实际拿不到数据，spf 的走势增强是死路；而赛程页自带的
rf_trend / dx_trend（让分与大小分走势）是活的。代码照迁，事实记在这里。
"""
import gzip
import pathlib
import re
import unittest
from datetime import datetime
from unittest import mock

from src.basketball import okooo as legacy
from src.domain.sports.basketball import okooo_parsing as new

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / 'fixtures' / 'basketball'


def _fixture(name):
    return gzip.open(FIXTURES / name, 'rt', encoding='utf-8').read()


HUNHE_HTML = _fixture('okooo_hunhe.html.gz')
DETAIL_PAGES = {kind: _fixture(f'okooo_{kind}.html.gz') for kind in ('ml', 'ah', 'ou')}
SAMPLE_MATCH_ID = '5381400'
NOW = datetime(2026, 8, 26, 12, 0, 0)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


RFLISTS = [
    '',
    None,
    '这不是盘口历史',
    '07/14 11:47 1.75 (1.5) 1.65',
    '07/14 11:47 1.75 (1.5) 1.65,07/14 10:17 1.60 (-1.5) 1.81',
    '08/26 20:00 1.90 (+13.5) 1.70,08/26 12:00 1.85 (13.5) 1.75,'
    '08/25 09:30 2.05 (-2) 1.60',
    '07/14 11:47 1.75 (1.5) 1.65,格式不对的一段,07/14 10:17 1.60 (-1.5) 1.81',
]

HISTORIES = [
    [],
    [{'home_odds': 1.9, 'away_odds': 1.9, 'line': 0}],
    [{'home_odds': 1.9, 'away_odds': 1.9, 'line': 0},
     {'home_odds': 1.9, 'away_odds': 1.9, 'line': 0}],
    [{'home_odds': 2.0, 'away_odds': 1.8, 'line': -3.5},
     {'home_odds': 1.7, 'away_odds': 2.1, 'line': -5.5}],
    [{'home_odds': 1.8, 'away_odds': 2.0, 'line': 210.5},
     {'home_odds': 2.1, 'away_odds': 1.7, 'line': 214.5}],
    [{'home_odds': 1.9, 'away_odds': 1.9, 'line': 3.5},
     {'home_odds': 1.9, 'away_odds': 1.9, 'line': 5.5}],
    [{'home_odds': None, 'away_odds': None, 'line': None},
     {'home_odds': 1.9, 'away_odds': 1.9, 'line': 1.0}],
]


class RflistParityTests(unittest.TestCase):
    def test_matches_legacy(self):
        for rflist in RFLISTS:
            with self.subTest(rflist=rflist):
                self.assertEqual(new.parse_rflist(rflist),
                                 legacy.parse_rflist(rflist))

    def test_entries_are_returned_oldest_first(self):
        """页面是新→旧，走势计算要按时间升序才对。"""
        entries = new.parse_rflist(RFLISTS[5])
        self.assertEqual([e['date'] for e in entries], ['08/25', '08/26', '08/26'])
        self.assertEqual([e['time'] for e in entries], ['09:30', '12:00', '20:00'])

    def test_unparsable_segments_are_dropped_not_fatal(self):
        self.assertEqual(len(new.parse_rflist(RFLISTS[6])), 2)


class LineTrendParityTests(unittest.TestCase):
    def test_matches_legacy(self):
        for history in HISTORIES:
            for kind in ('ah', 'ou', 'ml'):
                with self.subTest(n=len(history), kind=kind):
                    self.assertEqual(new.analyze_line_trend(history, kind),
                                     legacy.analyze_line_trend(history, kind))


class ScheduleParityTests(unittest.TestCase):
    DATES = ['2026-08-26', '2026-08-27', '2026-09-01']

    def test_raw_rows_match_legacy(self):
        for date in self.DATES:
            with self.subTest(date=date):
                self.assertEqual(new.parse_schedule(HUNHE_HTML, date),
                                 _legacy_raw_rows(date))

    def test_full_fetch_matches_legacy(self):
        for date in self.DATES:
            with self.subTest(date=date):
                with mock.patch.object(legacy, 'fetch_okooo',
                                       lambda *a, **k: HUNHE_HTML), \
                     mock.patch.object(legacy, 'datetime', _FrozenDatetime):
                    expected = legacy.fetch_okooo_basketball_schedule(date)
                fetcher = new.OkoooScheduleFetcher(
                    transport=lambda url: HUNHE_HTML, now_fn=lambda: NOW)
                self.assertEqual(fetcher.fetch(date), expected)

    def test_real_page_yields_the_expected_card(self):
        matches = new.parse_schedule(HUNHE_HTML, '2026-08-27')
        by_id = {m['id']: m for m in matches}
        self.assertIn(SAMPLE_MATCH_ID, by_id)
        first = by_id[SAMPLE_MATCH_ID]
        self.assertEqual(first['home'], '金州女武神')
        self.assertEqual(first['away'], '康涅狄格太阳')
        self.assertEqual(first['league'], 'WNBA')
        self.assertEqual(first['date'], '2026-08-27')
        self.assertEqual(first['time'], '07:00')
        self.assertEqual(first['handicap'], '+13.5')
        self.assertEqual(first['total_line'], 150.5)
        self.assertEqual(first['source'], 'okooo')
        self.assertEqual(first['dx_trend']['kind'], 'ou')

    def test_finished_matches_are_dropped(self):
        matches = new.parse_schedule(HUNHE_HTML, '2026-08-27')
        live = new.select_live(matches, '2026-08-27', NOW)
        self.assertTrue(any(m['status'] == 'finished' for m in matches),
                        '真实页面里没有完场比赛，这条断言是空跑')
        self.assertTrue(all(m['status'] == 'not_started' for m in live))

    def test_empty_html_returns_empty(self):
        self.assertEqual(new.parse_schedule('', '2026-08-27'), [])
        self.assertEqual(new.parse_schedule('<html></html>', '2026-08-27'), [])

    def test_transport_failure_returns_empty(self):
        def boom(url):
            raise IOError('WAF')

        fetcher = new.OkoooScheduleFetcher(transport=boom, now_fn=lambda: NOW)
        self.assertEqual(fetcher.fetch('2026-08-27'), [])


def _legacy_raw_rows(date):
    """旧实现没有独立的解析入口，照抄它的分组扫描逻辑取未经过滤的行。"""
    import re

    tables = re.findall(r'<table[^>]*>(.*?)</table>', HUNHE_HTML, re.S)
    main = max(tables, key=len)
    rows = []
    current = date
    for attrs, body in re.findall(r'<tr([^>]*)>(.*?)</tr>', main, re.S):
        cls = re.search(r'class="([^"]*)"', attrs)
        cls_v = cls.group(1) if cls else ''
        date_m = re.search(r'(\d{4}-\d{2}-\d{2})', body)
        if date_m and 'alltrObj' not in cls_v:
            current = date_m.group(1)
            continue
        if 'alltrObj' not in cls_v:
            continue
        parsed = legacy._parse_match_row(body, current)
        if parsed:
            rows.append(parsed)
    return rows


def _real_row(match_id=SAMPLE_MATCH_ID):
    """从真实页面里取出一整行，供派生合成用例。

    合成整行 HTML 不现实——一行有 11 个单元格、每个都有自己的 class 约定。
    以真实行为模板做定点改动，既保证结构真实，又能造出真实页面里当天
    恰好没有的形态（无 title 的队名、属性反序的让分、已完场等）。
    """
    import re

    tables = re.findall(r'<table[^>]*>(.*?)</table>', HUNHE_HTML, re.S)
    main = max(tables, key=len)
    for attrs, body in re.findall(r'<tr([^>]*)>(.*?)</tr>', main, re.S):
        if 'alltrObj' in attrs and f'/basketball/match/{match_id}/' in body:
            return attrs, body
    raise AssertionError(f'夹具里找不到 {match_id} 这一行')


def _page_of(rows):
    body = ''.join(f'<tr{attrs}>{html}</tr>' for attrs, html in rows)
    return ('<html><table>短表</table>'
            f'<table class="main">{body}</table></html>')


class DerivedRowTests(unittest.TestCase):
    """真实页面当天只有一种形态，每条兜底分支都没走到。这些用例从真实行
    派生出那些形态——它们在别的日子是常态（完场、无 title 的队名等）。"""

    def setUp(self):
        self.attrs, self.row = _real_row()

    def _parse(self, row=None, attrs=None, date='2026-08-27'):
        return new.parse_schedule(
            _page_of([(attrs if attrs is not None else self.attrs,
                       row if row is not None else self.row)]), date)

    def test_template_row_is_the_control(self):
        matches = self._parse()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['home'], '金州女武神')

    def test_row_without_match_row_class_is_a_date_header(self):
        """没有 alltrObj 的行是日期分组表头，它携带的日期要被记下来，
        而不是被当成一场比赛。"""
        header = ' class="dateGroup"'
        matches = new.parse_schedule(
            _page_of([(header, '<td>2026-09-09</td>'), (self.attrs, self.row)]),
            '2026-08-27')
        self.assertEqual(len(matches), 1, '表头行被当成比赛了')

    def test_group_date_is_used_when_the_row_has_no_full_datetime(self):
        stripped = self.row.replace('2026-08-27', '')
        matches = new.parse_schedule(
            _page_of([(' class="dateGroup"', '<td>2026-09-09</td>'),
                      (self.attrs, stripped)]),
            '2026-08-27')
        self.assertEqual(matches[0]['date'], '2026-09-09')

    def test_team_names_fall_back_to_plain_text_and_drop_brackets(self):
        """队名的 title 属性偶尔缺失，此时只能从纯文本取，而纯文本里
        带着 `[西2]` 这样的排名标记，必须剥掉。"""
        without_titles = re.sub(r'title="[^"]*"', '', self.row)
        match = self._parse(row=without_titles)[0]
        self.assertNotIn('[', match['home'])
        self.assertNotIn('[', match['away'])
        self.assertIn('金州女', match['home'])

    def test_handicap_attributes_in_reverse_order(self):
        """rflist 与 class 的先后在页面上出现过两种排法，只认一种会让
        让分盘口整个丢失。"""
        expected = self._parse()[0]
        reversed_row = _move_rflist_to_front(self.row)
        self.assertNotEqual(reversed_row, self.row, '模板里没有可调换的属性对')
        self.assertLess(reversed_row.index('rflist='),
                        reversed_row.index('rfsfrfzObj'),
                        'rflist 没被挪到 class 前面，这条用例没造出反序')
        actual = self._parse(row=reversed_row)[0]
        self.assertEqual(actual['handicap'], expected['handicap'])
        self.assertEqual(actual['rf_history'], expected['rf_history'])

    def test_finished_match_on_the_requested_date_is_dropped(self):
        """当天已完场的比赛必须按比分剔除。只靠开赛时刻判断的话，
        次日清晨的完场记录会被当成未开赛留下来。"""
        finished_row = self.row.replace('<td class="scoretd">-</td>',
                                        '<td class="scoretd">88-90</td>')
        if finished_row == self.row:
            finished_row = re.sub(r'(<td[^>]*>)\s*-\s*(</td>)', r'\g<1>88-90\g<2>',
                                  self.row, count=1)
        matches = new.parse_schedule(_page_of([(self.attrs, finished_row)]),
                                     '2026-08-27')
        self.assertTrue(any(m['status'] == 'finished' for m in matches),
                        '没造出完场行，这条用例是空跑')
        live = new.select_live(matches, '2026-08-27', NOW)
        self.assertEqual(live, [], '已完场的比赛没被剔除')

    def test_kickoff_moment_counts_as_in_progress(self):
        """分界取 `<=`：恰好到点就算已开赛。"""
        matches = self._parse()
        kickoff = datetime(2026, 8, 27, 7, 0)
        self.assertEqual(new.select_live(list(matches), '2026-08-27', kickoff), [])
        one_second_before = datetime(2026, 8, 27, 6, 59, 59)
        self.assertEqual(
            len(new.select_live(list(matches), '2026-08-27', one_second_before)), 1)


def _move_rflist_to_front(row):
    """把让分 span 的 rflist 属性挪到 class 之前，造出页面上那另一种排法。

    真实行里两个属性中间还隔着 onMouseOver 等，所以不能靠相邻替换。
    """
    span = re.search(r'<span([^>]*rfsfrfzObj[^>]*)>', row)
    if not span:
        return row
    attrs = span.group(1)
    rflist = re.search(r'\s*rflist="[^"]*"', attrs)
    if not rflist:
        return row
    reordered = rflist.group(0).strip() + ' ' + attrs.replace(rflist.group(0), '')
    return row.replace(span.group(0), f'<span {reordered}>', 1)


class BookRowRangeTests(unittest.TestCase):
    """各家行靠数值区间定位真正的数据段。区间放宽后会从行首的序号里
    取到假数据——真实页面里恰好取不到，因为序号都在区间外。"""

    def _row(self, numbers):
        cells = ''.join(f'<td><span>{n}</span></td>' for n in numbers)
        return f'<tr><td><a href="/ah/change/99/">变</a></td>{cells}</tr>'

    def test_handicap_range_rejects_out_of_scale_lines(self):
        """让分盘口不会有 200 分。区间失守时这行会被当成有效数据。"""
        row = self._row([1.90, 200, 1.90, 1.85, 205, 1.95, 1.90, 13.5, 1.90,
                         1.88, 13.5, 1.92])
        books = new.parse_book_rows(row, 'ah')
        self.assertEqual([b['line_init'] for b in books], [13.5])

    def test_total_range_rejects_out_of_scale_lines(self):
        row = ('<tr><td><a href="/ou/change/99/">变</a></td>'
               + ''.join(f'<td><span>{n}</span></td>'
                         for n in [1.90, 3.5, 1.90, 1.85, 4.5, 1.95,
                                   1.90, 150.5, 1.90, 1.88, 152.5, 1.92])
               + '</tr>')
        books = new.parse_book_rows(row, 'ou')
        self.assertEqual([b['line_init'] for b in books], [150.5])


class BookRowParityTests(unittest.TestCase):
    def test_book_rows_match_legacy(self):
        for kind, html in DETAIL_PAGES.items():
            with self.subTest(kind=kind):
                self.assertEqual(new.parse_book_rows(html, kind),
                                 legacy._parse_book_rows(html, kind))

    def test_average_row_matches_legacy(self):
        for kind, html in DETAIL_PAGES.items():
            with self.subTest(kind=kind):
                self.assertEqual(new.parse_average_row(html, kind),
                                 legacy._parse_average_row(html, kind))

    def test_consensus_matches_legacy(self):
        for kind, html in DETAIL_PAGES.items():
            books = legacy._parse_book_rows(html, kind)
            with self.subTest(kind=kind):
                self.assertEqual(new.consensus_from_books(books, kind),
                                 legacy._consensus_from_books(books, kind))

    def test_consensus_of_no_books(self):
        for kind in ('ml', 'ah', 'ou'):
            with self.subTest(kind=kind):
                self.assertEqual(new.consensus_from_books([], kind),
                                 legacy._consensus_from_books([], kind))

    def test_real_pages_yield_enough_books(self):
        """各家行数是这套正则最容易悄悄退化的地方——少匹到几十家不会报错，
        只是共识变得不可靠。把真实页面的量级钉住。"""
        counts = {kind: len(new.parse_book_rows(html, kind))
                  for kind, html in DETAIL_PAGES.items()}
        self.assertGreaterEqual(counts['ml'], 50, counts)
        self.assertGreaterEqual(counts['ah'], 40, counts)
        self.assertGreaterEqual(counts['ou'], 40, counts)


class BundleParityTests(unittest.TestCase):
    def _legacy_bundle(self):
        def fake(url, referer=None, max_retries=2):
            for kind, suffix in (('ml', '/odds/'), ('ah', '/ah/'), ('ou', '/ou/')):
                if url.endswith(suffix):
                    return DETAIL_PAGES[kind]
            return None

        with mock.patch.object(legacy, 'fetch_okooo', fake):
            return legacy.fetch_match_market_bundle(SAMPLE_MATCH_ID, use_cache=False)

    def test_bundle_matches_legacy(self):
        self.assertEqual(new.build_bundle(SAMPLE_MATCH_ID, DETAIL_PAGES),
                         self._legacy_bundle())

    def test_missing_pages_leave_sections_unavailable(self):
        bundle = new.build_bundle(SAMPLE_MATCH_ID, {'ml': DETAIL_PAGES['ml']})
        self.assertTrue(bundle['ml']['available'])
        self.assertEqual(bundle['ah'], {'available': False})
        self.assertEqual(bundle['ou'], {'available': False})

    def test_no_pages_at_all(self):
        """线上的常态：WAF 把三张详情页全拦掉。必须是「都不可用」，不是报错。"""
        bundle = new.build_bundle(SAMPLE_MATCH_ID, {})
        self.assertEqual(bundle, {'match_id': SAMPLE_MATCH_ID,
                                  'ml': {'available': False},
                                  'ah': {'available': False},
                                  'ou': {'available': False}})

    def test_page_average_beats_arithmetic_mean_for_moneyline(self):
        """欧赔页脚自带平均值，比自己对各家做算术平均更权威。"""
        bundle = new.build_bundle(SAMPLE_MATCH_ID, DETAIL_PAGES)
        self.assertEqual(bundle['ml']['source'], 'page_avg')
        self.assertEqual(bundle['ml']['home'], 1.1)
        self.assertEqual(bundle['ml']['away'], 6.83)


class BundleFetcherTests(unittest.TestCase):
    def test_fetches_three_pages_per_match(self):
        seen = []

        def transport(url):
            seen.append(url)
            for kind, suffix in (('ml', '/odds/'), ('ah', '/ah/'), ('ou', '/ou/')):
                if url.endswith(suffix):
                    return DETAIL_PAGES[kind]
            return None

        fetcher = new.MarketBundleFetcher(transport=transport, max_workers=2)
        bundles = fetcher.fetch_many([SAMPLE_MATCH_ID, '999'])
        self.assertEqual(set(bundles), {SAMPLE_MATCH_ID, '999'})
        self.assertEqual(len(seen), 6)
        self.assertTrue(bundles[SAMPLE_MATCH_ID]['ml']['available'])

    def test_blocked_transport_yields_unavailable_bundles(self):
        def blocked(url):
            raise IOError('okooo WAF 拦截')

        fetcher = new.MarketBundleFetcher(transport=blocked, max_workers=2)
        bundles = fetcher.fetch_many(['1', '2'])
        self.assertEqual(len(bundles), 2)
        self.assertFalse(bundles['1']['ml']['available'])

    def test_empty_id_list_does_not_touch_the_network(self):
        fetcher = new.MarketBundleFetcher(
            transport=lambda url: self.fail('不该发请求'))
        self.assertEqual(fetcher.fetch_many([]), {})
        self.assertEqual(fetcher.fetch_many([None, '']), {})


class NoLegacyImportTests(unittest.TestCase):
    def test_does_not_import_legacy_package(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(new))
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
