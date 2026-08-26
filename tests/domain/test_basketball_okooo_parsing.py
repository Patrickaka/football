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
