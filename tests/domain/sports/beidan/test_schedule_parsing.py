"""北单赛程解析：中国足彩网的单场页与 500.com 的即时赔率页。

参照物是黄金文件（`tests/fixtures/golden/beidan_schedule.json.gz`，48 条），
**逐条相同**——只覆盖 500.com 那条路。

`tests/fixtures/index_jczq.html` 是 500.com 的真实快照（4 场比赛、16 处联赛块），
正是这一层要解析的东西。中国足彩网那张表只能构造——线上抓不到真页面，
所以它不进黄金文件，改由 `ZgzcwScheduleTests` 直接对着解析器的分支写。

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

    def test_only_a_trailing_suffix_is_stripped(self):
        """**后缀要在末尾才剥**。队名里带「数据」两个字的球队并不稀奇，
        按「包含」匹配会把它截掉一截——而截出来的还是个像样的名字，
        不会报错，只会让这场比赛在下游对不上号。
        """
        html = (_link('999001', '数据队', '乙队')
                + _time_cell('999001', '08-28 19:30'))
        self.assertEqual(_parse_500(html)[0]['home'], '数据队')

    def test_a_name_that_is_entirely_a_suffix_drops_the_match(self):
        """剥完之后队名空了 → 整场丢掉。**这条才真正走到那道守卫**：
        上一版用的 `title="VS乙队数据"` 连正则都匹配不上，守卫根本没参与
        （判据 23）。
        """
        html = '<a href="shuju-999001.shtml" title="数据VS乙队数据"></a>'
        self.assertEqual(_parse_500(html), [])

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

    def test_an_unrecognised_time_never_reaches_the_record(self):
        r"""**认不出格式的时间根本进不了 `times`**，所以时间栏留空。

        迁移前这里有一条 `else: match['time'] = when` 的兜底，写这条用例时
        才发现它任何输入都走不到：能进 `times` 的字符串是被
        `\d{2}-\d{2}\s+\d{2}:\d{2}` 捕获出来的，而兜底前面那次匹配
        用的是同一个形状，必然成功。已删（判据 9 第一类）。
        """
        html = (_link('999001', '甲队', '乙队')
                + '<td rowspan="2">稍后</td><a href="shuju-999001.shtml"></a>')
        matches = parsing.parse_500_schedule(html, '不是日期',
                                             datetime.datetime(2026, 8, 28))
        self.assertEqual(matches[0]['time'], '')

    def test_a_time_that_cannot_be_parsed_still_reaches_the_status_check(self):
        html = (_link('999001', '甲队', '乙队')
                + '<td rowspan="2">稍后</td><a href="shuju-999001.shtml"></a>')
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
        """开赛前一小时那道门槛的两侧，**含恰好等于的那一刻**。

        只测 ±1 秒是分不出 `<` 与 `<=` 的——门槛的严格与否只在相等时才
        显形（判据 5）。恰好差一小时时算「未开始」，因为判的是
        `开赛时刻 < 现在 + 1 小时`。
        """
        just_outside = KICKOFF - datetime.timedelta(
            hours=IN_PROGRESS_BEFORE_HOURS, seconds=1)
        self.assertEqual(self._status_at(just_outside), 'not_started')
        exactly = KICKOFF - datetime.timedelta(hours=IN_PROGRESS_BEFORE_HOURS)
        self.assertEqual(self._status_at(exactly), 'not_started')
        just_inside = exactly + datetime.timedelta(seconds=1)
        self.assertEqual(self._status_at(just_inside), 'in_progress')

    def test_three_hours_after_kickoff_is_finished_and_filtered_out(self):
        """赛后三小时那道门槛的两侧，同样含恰好等于的那一刻。
        已结束的**在返回前就被滤掉**。"""
        just_inside = KICKOFF + datetime.timedelta(
            hours=FINISHED_AFTER_HOURS) - datetime.timedelta(seconds=1)
        self.assertEqual(self._status_at(just_inside), 'in_progress')
        exactly = KICKOFF + datetime.timedelta(hours=FINISHED_AFTER_HOURS)
        self.assertEqual(self._status_at(exactly), 'in_progress')
        just_outside = exactly + datetime.timedelta(seconds=1)
        self.assertIsNone(self._status_at(just_outside))

    def test_a_match_without_a_kickoff_time_raises(self):
        """**钉住一处迁移前就有的缺陷**（判据 9 第三类：路径可达，只是
        当前没人走）。

        拿不到开赛时刻时会落到「只按日期判断」那条分支，而它比较的是
        `datetime` 与 `date`——Python 直接抛 `TypeError`。适配层只把它当成
        「抓取失败」记一条日志、返回空列表，于是**一场没解析到时间的比赛
        足以让当天所有比赛都消失**（§十一·3 那类「200 加 0 场比赛」）。

        这条路只在中国足彩网挂掉、回退到 500.com 时才走——线上 7 天内一次
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


class ZgzcwScheduleTests(unittest.TestCase):
    """中国足彩网单场页 → 未完结比赛。

    行的形状取自真实页面：`tr_<id>` 行、`wh-N` 列、队名在 `tn` 属性上、
    `newplayid` 是赔率详情的稳定 ID、比分格是 `VS` 表示还没踢。
    """

    @staticmethod
    def _row(row_id='371', league='墨西超', num='371', analysis_id='4553848',
             kickoff='2026-08-28 19:30', t_attr='2026-08-28 06:00:00',
             score='VS', home='蒙特雷', away='圣路易斯',
             prices=('1.83', '3.74', '3.82'), league_cell='墨西超'):
        title = (f'<span title="比赛时间:{kickoff}">{kickoff[-5:]}</span>'
                 if kickoff else '<span>待定</span>')
        newplay = f' newplayid="{analysis_id}"' if analysis_id else ''
        home_cell = f'<td class="wh-4 t-r" tn="{home}"><a>{home}</a></td>' if home \
            else '<td class="wh-4 t-r"></td>'
        away_cell = f'<td class="wh-6 t-l" tn="{away}"><a>{away}</a></td>' if away \
            else '<td class="wh-6 t-l"></td>'
        return (
            f'<tr id="tr_{row_id}" m="{league}" t="{t_attr}">'
            f'<td class="wh-1"><a>{num}</a></td>'
            f'<td class="wh-2">{league_cell}</td>'
            f'<td class="wh-3">{title}</td>'
            f'{home_cell}<td class="wh-5">{score}</td>{away_cell}'
            f'<td class="wh-8"{newplay}></td>'
            f'<td class="wh-9"><div>'
            + ''.join(f'<span>{value}</span>' for value in prices)
            + '</div></td></tr>')

    def _parse(self, *rows, date=DATE):
        return parsing.parse_zgzcw_schedule(f'<table>{"".join(rows)}</table>', date)

    def test_parses_a_row(self):
        match = self._parse(self._row())[0]
        self.assertEqual((match['id'], match['zgzcw_id'], match['analysis_id']),
                         ('4553848', '371', '4553848'))
        self.assertEqual((match['home'], match['away']), ('蒙特雷', '圣路易斯'))
        self.assertEqual((match['num'], match['league'], match['time']),
                         ('371', '墨西超', '19:30'))
        self.assertEqual(match['date'], DATE)
        self.assertEqual((match['spf_sp'], match['spf_s'], match['spf_f']),
                         (1.83, 3.74, 3.82))
        self.assertEqual((match['status'], match['source']), ('not_started', 'zgzcw'))

    def test_the_analysis_id_is_preferred_over_the_row_id(self):
        """`newplayid` 才是赔率详情能用的 ID，行号只是页面自己的编号。"""
        match = self._parse(self._row(row_id='999', analysis_id='4553848'))[0]
        self.assertEqual(match['id'], '4553848')
        self.assertEqual(match['zgzcw_id'], '999')

    def test_a_missing_analysis_id_is_synthesised_from_the_date_and_number(self):
        match = self._parse(self._row(analysis_id=None))[0]
        self.assertIsNone(match['analysis_id'])
        self.assertEqual(match['id'], f'zgzcw_{DATE}_371')

    def test_rows_that_are_not_match_rows_are_skipped(self):
        """页面里还有表头、分组行——只有 `tr_` 开头的才是比赛。"""
        self.assertEqual(
            self._parse('<tr id="thead"><td class="wh-4" tn="甲"><a>甲</a></td>'
                        '<td class="wh-6" tn="乙"><a>乙</a></td></tr>'),
            [])

    def test_a_row_without_both_team_names_is_skipped(self):
        for missing in ({'home': ''}, {'away': ''}):
            with self.subTest(**missing):
                self.assertEqual(self._parse(self._row(**missing)), [])

    def test_a_finished_score_is_filtered_out(self):
        """比分格有真实比分就是踢完了。北单只推还能买的场次。"""
        for score in ('2:1', '0-0', '2 : 1'):
            with self.subTest(score=score):
                self.assertEqual(self._parse(self._row(score=score)), [])

    def test_vs_in_the_score_cell_means_it_has_not_started(self):
        self.assertEqual(len(self._parse(self._row(score='VS'))), 1)

    def test_another_days_match_is_filtered_out(self):
        self.assertEqual(self._parse(self._row(kickoff='2026-08-29 19:30')), [])

    def test_no_date_filter_keeps_every_day(self):
        matches = self._parse(self._row(kickoff='2026-08-29 19:30'), date=None)
        self.assertEqual(matches[0]['date'], '2026-08-29')

    def test_the_kickoff_title_wins_over_the_row_attribute(self):
        """两个时间都在行上，标题里那个才是开赛时刻；`t` 是销售截止之类。"""
        match = self._parse(self._row(kickoff='2026-08-28 19:30',
                                      t_attr='2026-08-28 06:00:00'))[0]
        self.assertEqual(match['time'], '19:30')

    def test_the_row_attribute_fills_in_when_the_title_is_missing(self):
        match = self._parse(self._row(kickoff='', t_attr='2026-08-28 06:00:00'))[0]
        self.assertEqual((match['date'], match['time']), (DATE, '06:00'))

    def test_incomplete_prices_become_none_rather_than_a_short_list(self):
        """缺价就三个都置空，绝不能让下游按位置取到错位的赔率。"""
        match = self._parse(self._row(prices=('1.83', '3.74')))[0]
        self.assertEqual((match['spf_sp'], match['spf_s'], match['spf_f']),
                         (None, None, None))

    def test_the_league_falls_back_to_its_cell(self):
        match = self._parse(self._row(league='', league_cell='<b>西甲</b>'))[0]
        self.assertEqual(match['league'], '西甲')

    def test_the_handicap_fields_are_present_but_empty(self):
        """单场页不带让球盘。字段要在、值为空——缺字段会让下游 KeyError。"""
        match = self._parse(self._row())[0]
        for field in ('rqspf_sp', 'rqspf_s', 'rqspf_f', 'rqspf_odds', 'handicap'):
            with self.subTest(field=field):
                self.assertIsNone(match[field])

    def test_an_empty_page_yields_nothing(self):
        for html in ('', None, '<html><body>什么也没有</body></html>'):
            with self.subTest(html=html):
                self.assertEqual(parsing.parse_zgzcw_schedule(html, DATE), [])


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

    def test_zgzcw_falls_back_and_marks_the_source(self):
        with mock.patch.object(adapter, 'fetch_zgzcw', return_value=''):
            with mock.patch.object(adapter, 'fetch_beidan_schedule',
                                   return_value=[{'id': 'x'}]) as fallback:
                result = adapter.fetch_zgzcw_schedule(DATE)
        self.assertEqual(result[0]['source'], '500.com')
        self.assertEqual(fallback.call_args_list[0].kwargs.get('source'), 'dc')

    def test_the_two_fallback_reasons_are_logged_apart(self):
        """页面为空、没有未完结比赛——两种都回退，日志要分得开。

        §十一·3 那类「接口返回 200 加 0 场比赛」的故障里，最难查的正是
        分不清「压根没抓到」和「抓到了但今天没球」。
        """
        cases = {
            '': '返回为空',
            f'<table>{ZgzcwScheduleTests._row(score="2:1")}</table>':
                '未找到指定日期的未完结比赛',
        }
        for html, expected in cases.items():
            with self.subTest(expected=expected):
                with mock.patch.object(adapter, 'fetch_zgzcw', return_value=html):
                    with mock.patch.object(adapter, 'fetch_beidan_schedule',
                                           return_value=[]):
                        with mock.patch.object(adapter.log, 'warning') as warned:
                            adapter.fetch_zgzcw_schedule(DATE)
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
