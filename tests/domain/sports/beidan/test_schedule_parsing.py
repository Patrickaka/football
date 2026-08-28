"""北单赛程解析：okooo 的表格页与 500.com 的即时赔率页。

参照物是从迁移前的 `fetching.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_schedule.json.gz`，67 条），**逐条相同**。

`tests/fixtures/index_jczq.html` 是 500.com 的真实快照（4 场比赛、16 处联赛块），
正是这一层要解析的东西。okooo 那张表只能构造——线上抓不到真页面（WAF）。

**时钟钉死在四个时刻上**，不跟着今天跑：500 那条路要按开赛时刻判断比赛
是否已结束，语料里写一个「未来的日期」就是一颗定时炸弹（判据 24）。
"""
import ast
import datetime
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.beidan import fetching as adapter
from src.domain.sports.beidan import parsing
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_schedule.json.gz',
                             'rt', encoding='utf-8'))
REAL_500_PAGE = (FIXTURES / 'index_jczq.html').read_text(encoding='utf-8')

# 迁移当时生效的那组常量，写死不 import（判据 4、12）
IN_PROGRESS_BEFORE_HOURS = 1
FINISHED_AFTER_HOURS = 3
OKOOO_MINIMUM_TABLES = 2
OKOOO_SCHEDULE_CELLS = 6
OKOOO_FULL_ODDS = 6
DATE = '2026-08-28'
KICKOFF = datetime.datetime(2026, 8, 28, 19, 30)


def golden_entries():
    from scripts.gen_beidan_schedule_golden import entries
    return entries()


def _link(match_id, home, away, suffix='数据'):
    return f'<a href="shuju-{match_id}.shtml" title="{home}VS{away}{suffix}"></a>'


def _time_cell(match_id, when):
    return f'<td rowspan="2">{when}</td><a href="shuju-{match_id}.shtml"></a>'


def _parse_500(html, now=None, date=DATE):
    return parsing.parse_500_schedule(html, date,
                                      now or datetime.datetime(2020, 1, 1))


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class FiveHundredScheduleTests(unittest.TestCase):

    ONE = _link('999001', '甲队', '乙队') + _time_cell('999001', '08-28 19:30')

    def test_the_real_page_parses(self):
        """真实快照里有 4 场比赛——**构造语料测不出正则对不对**。"""
        matches = _parse_500(REAL_500_PAGE)
        self.assertEqual(len(matches), 4)
        self.assertTrue(all(m['id'] and m['home'] and m['away'] for m in matches))

    def test_time_league_and_number_are_stitched_back_by_match_id(self):
        """三样东西在页面上分散在三处，按比赛号拼回去。"""
        html = ('<a href="//liansai.500.com/zuqiu-100/">英超</a>'
                + _link('999001', '甲队', '乙队')
                + '<input value="999001" /> 周五001'
                + _time_cell('999001', '08-28 19:30'))
        match = _parse_500(html)[0]
        self.assertEqual((match['time'], match['league'], match['num']),
                         ('19:30', '英超', '周五001'))

    def test_entry_suffixes_are_stripped_from_team_names(self):
        """同一场比赛在页面上有九个入口，队名后面跟着入口的名字。"""
        for suffix in ('百家', '欧赔', '亚赔', '亚盘', '数据', '盘口',
                       '指数', '对比', '分析'):
            with self.subTest(suffix=suffix):
                html = (_link('999001', f'甲队{suffix}', f'乙队{suffix}', suffix)
                        + _time_cell('999001', '08-28 19:30'))
                match = _parse_500(html)[0]
                self.assertEqual((match['home'], match['away']), ('甲队', '乙队'))

    def test_a_kickoff_on_another_day_moves_the_record(self):
        """跨零点的比赛属于第二天——**日期跟着时间走，不跟着请求走**。"""
        html = _link('999001', '甲队', '乙队') + _time_cell('999001', '08-29 02:00')
        match = _parse_500(html)[0]
        self.assertEqual((match['date'], match['time']), ('2026-08-29', '02:00'))

    def test_a_kickoff_on_the_requested_day_keeps_the_date(self):
        self.assertEqual(_parse_500(self.ONE)[0]['date'], DATE)

    def test_the_second_time_layout_is_also_accepted(self):
        """时间跟在链接**后面**的那种排布。"""
        html = (_link('999001', '甲队', '乙队')
                + '<a href="shuju-999001.shtml"></a>08-28 19:30')
        self.assertEqual(_parse_500(html)[0]['time'], '19:30')

    def test_an_unrecognised_time_is_kept_verbatim(self):
        html = (_link('999001', '甲队', '乙队')
                + '<td rowspan="2">稍后</td><a href="shuju-999001.shtml"></a>')
        # 时间认不出来 → 后面按日期判断，而那条路会抛（见下面一组）
        with self.assertRaises(TypeError):
            _parse_500(html)

    def test_an_empty_team_name_drops_the_match(self):
        html = '<a href="shuju-999001.shtml" title="VS乙队数据"></a>'
        self.assertEqual(_parse_500(html), [])

    def test_leagues_are_scoped_to_their_block(self):
        html = ('<a href="//liansai.500.com/zuqiu-100/">英超</a>'
                + _link('999001', '甲队', '乙队')
                + '<a href="//liansai.500.com/zuqiu-200/">西甲</a>'
                + _link('999002', '丙队', '丁队')
                + _time_cell('999001', '08-28 19:30')
                + _time_cell('999002', '08-28 21:30'))
        leagues = {m['id']: m['league'] for m in _parse_500(html)}
        self.assertEqual(leagues, {'999001': '英超', '999002': '西甲'})

    def test_empty_or_linkless_pages_yield_nothing(self):
        for html in ('', '<html><body>什么也没有</body></html>'):
            with self.subTest(html=html[:20]):
                self.assertEqual(_parse_500(html), [])


class MatchStatusTests(unittest.TestCase):
    """状态由开赛时刻与当前时刻决定。两道门槛，各测两侧。"""

    ONE = _link('999001', '甲队', '乙队') + _time_cell('999001', '08-28 19:30')

    def _status_at(self, now):
        matches = _parse_500(self.ONE, now=now)
        return matches[0]['status'] if matches else None

    def test_well_before_kickoff_is_not_started(self):
        self.assertEqual(self._status_at(KICKOFF - datetime.timedelta(hours=5)),
                         'not_started')

    def test_within_an_hour_of_kickoff_is_in_progress(self):
        """开赛前一小时那道门槛的两侧。"""
        just_outside = KICKOFF - datetime.timedelta(
            hours=IN_PROGRESS_BEFORE_HOURS, seconds=1)
        self.assertEqual(self._status_at(just_outside), 'not_started')
        just_inside = KICKOFF - datetime.timedelta(
            hours=IN_PROGRESS_BEFORE_HOURS) + datetime.timedelta(seconds=1)
        self.assertEqual(self._status_at(just_inside), 'in_progress')

    def test_three_hours_after_kickoff_is_finished_and_filtered_out(self):
        """赛后三小时那道门槛的两侧。已结束的**在返回前就被滤掉**。"""
        just_inside = KICKOFF + datetime.timedelta(
            hours=FINISHED_AFTER_HOURS) - datetime.timedelta(seconds=1)
        self.assertEqual(self._status_at(just_inside), 'in_progress')
        just_outside = KICKOFF + datetime.timedelta(
            hours=FINISHED_AFTER_HOURS, seconds=1)
        self.assertIsNone(self._status_at(just_outside))

    def test_a_match_without_a_kickoff_time_raises(self):
        """**钉住一处迁移前就有的缺陷**（判据 9 第三类：路径可达，只是
        当前没人走）。

        拿不到开赛时刻时会落到「只按日期判断」那条分支，而它比较的是
        `datetime` 与 `date`——Python 直接抛 `TypeError`。适配层只把它当成
        「抓取失败」记一条日志、返回空列表，于是**一场没解析到时间的比赛
        足以让当天所有比赛都消失**（§十一·3 那类「200 加 0 场比赛」）。

        这条路只在 okooo 挂掉、回退到 500.com 时才走——线上 7 天内一次
        都没走过，所以迁移期没有动它。修它会改变回退路径的返回值。
        """
        timeless = _link('999001', '甲队', '乙队')
        with self.assertRaises(TypeError) as caught:
            _parse_500(timeless)
        self.assertIn('datetime.date', str(caught.exception))

    def test_an_unparsable_record_date_falls_back_to_the_current_status(self):
        """连日期都认不出来时不抛，保留原状态——那条 `except ValueError`
        拦得住的只有这一种。"""
        timeless = _link('999001', '甲队', '乙队')
        matches = parsing.parse_500_schedule(timeless, '不是日期',
                                             datetime.datetime(2026, 8, 28))
        self.assertEqual(matches[0]['status'], 'not_started')


class OkoooScheduleTests(unittest.TestCase):

    @staticmethod
    def _row(num='001', league='英超', match_id='1320957',
             time_cell='08-28 19:30', mtime=None, score='-',
             home='安山小绿人', away='大邱FC', handicap='(-1)',
             odds=('1.80', '3.60', '4.20', '2.20', '3.40', '3.10')):
        first = (f'<span class="xh"><i>{num}</i></span>'
                 f'<a href="//www.okooo.com/soccer/league/100/">{league}</a>')
        time_attr = f' mTime="{mtime}"' if mtime else ''
        teams = (f'<span class="homenameobj" title="{home}">{home}</span>'
                 f'<span class="awaynameobj" title="{away}">{away}</span>')
        if handicap is not None:
            teams += f'<span class="handicapobj">{handicap}</span>'
        teams += ''.join(f'<em>{value}</em>' for value in odds)
        cells = [first, f'<span{time_attr}>{time_cell}</span>', teams,
                 '', '', score]
        return ('<tr>' + f'<a href="/soccer/match/{match_id}/"></a>'
                + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')

    @staticmethod
    def _page(*rows, tables=OKOOO_MINIMUM_TABLES):
        filler = '<table><tr><td>目录</td></tr></table>'
        return ('<html><body>' + filler * (tables - 1)
                + '<table>' + ''.join(rows) + '</table></body></html>')

    def _parse(self, html):
        return parsing.parse_okooo_schedule(html, DATE)

    def test_parses_a_row(self):
        matches, tables = self._parse(self._page(self._row()))
        self.assertEqual(tables, OKOOO_MINIMUM_TABLES)
        match = matches[0]
        self.assertEqual((match['id'], match['home'], match['away']),
                         ('1320957', '安山小绿人', '大邱FC'))
        self.assertEqual((match['num'], match['league'], match['time']),
                         ('001', '英超', '19:30'))
        self.assertEqual(match['handicap'], '(-1)')
        self.assertEqual(match['source'], 'okooo')

    def test_too_few_tables_is_reported_separately_from_no_matches(self):
        """**「页面不对」与「今天没有未完结比赛」不是一回事。**

        两者都会让调用方去找备用数据源，但日志要能说清是哪一种——
        §十一·3 那类故障里最难查的正是分不清这两者。
        """
        broken, tables = self._parse('<html><table><tr><td>x</td></tr></table></html>')
        self.assertIsNone(broken)
        self.assertLess(tables, OKOOO_MINIMUM_TABLES)
        empty, tables = self._parse(self._page(self._row(score='2:1')))
        self.assertEqual(empty, [])
        self.assertEqual(tables, OKOOO_MINIMUM_TABLES)

    def test_status_comes_from_the_score_column_not_the_clock(self):
        """**页面自己说的比我们算的准**——有比分就是踢完了。"""
        finished, _ = self._parse(self._page(self._row(score='2:1')))
        self.assertEqual(finished, [])
        pending, _ = self._parse(self._page(self._row(score='-')))
        self.assertEqual(pending[0]['status'], 'not_started')

    def test_a_dash_score_means_not_started(self):
        for score in ('-', ''):
            with self.subTest(score=score):
                matches, _ = self._parse(self._page(self._row(score=score)))
                self.assertEqual(len(matches), 1)

    def test_rows_with_too_few_cells_carry_the_section_date(self):
        """短行不是比赛，是日期分隔行——**它决定后面几场归哪一天**。"""
        matches, _ = self._parse(self._page(
            '<tr><td>2026-08-29</td></tr>',
            self._row(time_cell='稍后')))
        self.assertEqual(matches[0]['date'], '2026-08-29')

    def test_mtime_wins_over_the_cell_text(self):
        matches, _ = self._parse(self._page(
            self._row(time_cell='稍后', mtime='08-29 02:00')))
        self.assertEqual((matches[0]['date'], matches[0]['time']),
                         ('2026-08-29', '02:00'))

    def test_an_unparsable_kickoff_is_kept_verbatim(self):
        """两个来源在这一点上处置相同：认不出格式就原样留着那段文本。"""
        from_mtime, _ = self._parse(self._page(
            self._row(time_cell='x', mtime='稍后')))
        from_cell, _ = self._parse(self._page(self._row(time_cell='稍后')))
        self.assertEqual(from_mtime[0]['time'], '稍后')
        self.assertEqual(from_cell[0]['time'], '稍后')

    def test_the_handicap_odds_need_all_six_prices(self):
        """**让球那三个价要六个都在才算数**：只报了前三个时后三个是别的
        东西，取来会得到一组假赔率。"""
        full, _ = self._parse(self._page(self._row()))
        self.assertEqual(full[0]['rqspf_sp'], 2.20)
        for count in range(OKOOO_FULL_ODDS):
            with self.subTest(count=count):
                partial, _ = self._parse(self._page(
                    self._row(odds=tuple('1.80' for _ in range(count)))))
                self.assertIsNone(partial[0]['rqspf_sp'])

    def test_the_win_draw_lose_prices_fill_in_one_at_a_time(self):
        """胜平负那三个与让球那三个**门槛不同**：前者有几个算几个。"""
        one, _ = self._parse(self._page(self._row(odds=('1.80',))))
        self.assertEqual(one[0]['spf_sp'], 1.80)
        self.assertIsNone(one[0]['spf_s'])

    def test_the_handicap_odds_bundle_needs_every_price_above_one(self):
        """赔率不可能不到 1——**整组都不给**，不是只丢那一个。"""
        matches, _ = self._parse(self._page(self._row(
            odds=('1.80', '3.60', '4.20', '2.20', '0.95', '3.10'))))
        self.assertIsNone(matches[0]['rqspf_odds'])
        self.assertEqual(matches[0]['rqspf_s'], 0.95)

    def test_a_missing_match_id_is_synthesised_from_the_date_and_number(self):
        html = self._page(self._row().replace(
            '<a href="/soccer/match/1320957/"></a>', ''))
        matches, _ = self._parse(html)
        self.assertEqual(matches[0]['id'], '20260828_001')

    def test_a_row_without_both_team_names_is_skipped(self):
        for broken in ('homenameobj', 'awaynameobj'):
            with self.subTest(missing=broken):
                html = self._page(self._row().replace(broken, 'xx'))
                self.assertEqual(self._parse(html)[0], [])

    def test_missing_optional_fields_become_empty_not_absent(self):
        html = self._page(self._row(handicap=None).replace(
            '<span class="xh"><i>001</i></span>', ''))
        match = self._parse(html)[0][0]
        self.assertEqual(match['num'], '')
        self.assertIsNone(match['handicap'])


class AdapterTests(unittest.TestCase):

    ONE = _link('999001', '甲队', '乙队') + _time_cell('999001', '08-28 19:30')

    def test_the_clock_is_injected_by_the_adapter(self):
        with mock.patch.object(adapter, 'fetch', return_value=self.ONE):
            with mock.patch.object(adapter, 'parsing', create=True):
                pass
        with mock.patch.object(adapter, 'fetch', return_value=self.ONE):
            with mock.patch.object(adapter._parsing, 'parse_500_schedule',
                                   return_value=[]) as parsed:
                adapter.fetch_beidan_schedule(DATE)
        self.assertIsInstance(parsed.call_args.args[2], datetime.datetime)

    def test_the_type_error_is_swallowed_into_an_empty_schedule(self):
        """**这就是「200 加 0 场比赛」的样子**：一场没解析到时间的比赛，
        整份赛程变成空列表，而日志里只有一行 `抓取北单赛程失败`。"""
        timeless = _link('999001', '甲队', '乙队')
        with mock.patch.object(adapter, 'fetch', return_value=timeless):
            with mock.patch.object(adapter.log, 'error') as logged:
                self.assertEqual(adapter.fetch_beidan_schedule(DATE), [])
        self.assertIn('抓取北单赛程失败', logged.call_args.args[0])

    def test_an_empty_page_returns_an_empty_schedule(self):
        with mock.patch.object(adapter, 'fetch', return_value=''):
            self.assertEqual(adapter.fetch_beidan_schedule(DATE), [])

    def test_the_source_picks_the_url(self):
        for source, attribute in (('dc', 'DC_SCHEDULE_URL'),
                                  ('jczq', 'SCHEDULE_URL')):
            with self.subTest(source=source):
                with mock.patch.object(adapter, 'fetch',
                                       return_value='') as fetched:
                    adapter.fetch_beidan_schedule(DATE, source=source)
                self.assertEqual(fetched.call_args.args[0],
                                 getattr(adapter, attribute))

    def test_okooo_falls_back_and_marks_the_source(self):
        with mock.patch.object(adapter, 'fetch_okooo', return_value=''):
            with mock.patch.object(adapter, 'fetch_beidan_schedule',
                                   return_value=[{'id': 'x'}]) as fallback:
                result = adapter.fetch_okooo_schedule(DATE)
        self.assertEqual(result[0]['source'], '500.com')
        self.assertEqual(fallback.call_args_list[0].kwargs.get('source'), 'dc')

    def test_the_three_fallback_reasons_are_logged_apart(self):
        """页面为空、结构不对、没有未完结比赛——三种都回退，日志要分得开。"""
        cases = {
            '': 'WAF拦截',
            '<html><table><tr><td>x</td></tr></table></html>': '未找到比赛表格',
            OkoooScheduleTests._page(
                OkoooScheduleTests._row(score='2:1')): '未找到未完结比赛',
        }
        for html, expected in cases.items():
            with self.subTest(expected=expected):
                with mock.patch.object(adapter, 'fetch_okooo', return_value=html):
                    with mock.patch.object(adapter, 'fetch_beidan_schedule',
                                           return_value=[]):
                        with mock.patch.object(adapter.log, 'warning') as warned:
                            adapter.fetch_okooo_schedule(DATE)
                self.assertIn(expected, warned.call_args.args[0])


FORBIDDEN_CALLS = {'now', 'today', 'utcnow', 'strftime'}


class NoClockInTheDomainTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/beidan/parsing.py'

    def test_the_domain_never_reads_the_clock(self):
        """`datetime` 进得来（要做日期加减与解析），`datetime.now()` 进不来。"""
        tree = ast.parse(pathlib.Path(self.DOMAIN).read_text(encoding='utf-8'))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in FORBIDDEN_CALLS}
        self.assertEqual(calls, set())

    def test_the_guard_would_catch_a_real_violation(self):
        tree = ast.parse(pathlib.Path('src/beidan/fetching.py').read_text(
            encoding='utf-8'))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in FORBIDDEN_CALLS}
        self.assertNotEqual(calls, set())


if __name__ == '__main__':
    unittest.main()
