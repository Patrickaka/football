"""北单页面解析：三张管道赔率表，以及走势历史的取数策略。

参照物是黄金文件（`tests/fixtures/golden/beidan_parsing.json.gz`，27 条），
**逐条相同**——只覆盖三张管道分隔的赔率表（比分/总进球/半全场）。

**走势历史那 53 条语料已经删掉**：它们编码的是旧解析器的边界（表头关键字、
列数不足、script 回退……），换到 zgzcw 的结构化历史页后一律解析为空，
留着是 53 条恒真断言。取数策略（公司明细页 → 汇总页两步）由下面的
`AdapterTests` 守着，解析本身在 `tests/domain/test_zgzcw_sources.py`。

语料按 `schedules.py` 里**实际用的正则与下标**铺出来，不是按字段名猜的。

**解码不在这一层**：`fetch` 拿到字节后按 utf-8 / gbk / gb2312 依次**严格**解码
（不是 `errors='replace'`），解析层收到的已经是 str。§十一·1 记的那个
「四个候选里第一个总是成功、gbk 页面解出整页乱码」的坑就在那里。
"""
import ast
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.beidan import fetching as fetching_adapter
from src.beidan import schedules as schedules_adapter
from src.domain.sports.beidan import parsing
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_parsing.json.gz',
                             'rt', encoding='utf-8'))

# 迁移当时生效的那组常量，写死不 import（判据 4、12）
MAX_TIME_LENGTH = 8
MAX_HANDICAP_LENGTH = 20
MIN_SCRIPT_LENGTH = 50
ZJQ_MINIMUM, BQC_MINIMUM = 8, 10
ZJQ_BUCKETS = ('0', '1', '2', '3', '4', '5', '6', '7+')
BQC_OPTIONS = ('胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负')


def golden_entries():
    from scripts.gen_beidan_parsing_golden import entries
    return entries()


def _table(header, rows):
    head = '<tr>' + ''.join(f'<th>{c}</th>' for c in header) + '</tr>'
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>'
                   for row in rows)
    return f'<html><body><table>{head}{body}</table></body></html>'


def _script_page(text):
    return f'<html><body><script>{text}</script></body></html>'


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class PairTableTests(unittest.TestCase):
    """比分盘那张变长的表：`比赛号|比分|价|比分|价…`。"""

    def test_parses_pairs(self):
        result = parsing.parse_pair_table('1320957|1-0|8.00|1-1|7.50')
        self.assertEqual(result, {'1320957': {'1-0': 8.0, '1-1': 7.5}})

    def test_comments_and_blank_lines_are_skipped(self):
        content = '# 注释\n\n   \n1320957|1-0|8.00'
        self.assertEqual(parsing.parse_pair_table(content),
                         {'1320957': {'1-0': 8.0}})

    def test_a_comment_that_would_otherwise_parse_is_still_skipped(self):
        """**注释那道检查要用一行「不注释掉就会被解析」的语料测**。

        `'# 注释'` 分不出段数，不跳过它也照样进不了结果——用它测等于
        什么也没测（判据 5 的形状）。这一行去掉 `#` 就是合法数据。
        """
        self.assertEqual(parsing.parse_pair_table('#1320957|1-0|8.00'), {})
        self.assertEqual(parsing.parse_pair_table('1320957|1-0|8.00'),
                         {'1320957': {'1-0': 8.0}})

    def test_surrounding_whitespace_is_trimmed_off_each_line(self):
        """不逐行 strip 的话，比赛号会带着缩进进结果——**下游按号取数就全落空**。

        **要用多行语料**：单行的首尾空白靠「先对整段 strip 一次」也能去掉，
        两处清理互相盖住，谁失效都测不出来（迁移前正是两处都有，
        整段那次因此一个字符也改变不了结果，已删）。
        缩进只出现在**中间那一行**时，才只有逐行 strip 能救。
        """
        content = '1320957|1-0|8.00\n   1320958|0-0|9.50   \n1320959|2-2|15.00'
        self.assertEqual(sorted(parsing.parse_pair_table(content)),
                         ['1320957', '1320958', '1320959'])
        self.assertEqual(parsing.parse_pair_table('  1320957|1-0|8.00  '),
                         {'1320957': {'1-0': 8.0}})

    def test_a_bad_price_drops_only_that_pair(self):
        """**与定长表相反**：坏价只丢那一对，整行还留着。"""
        result = parsing.parse_pair_table('1320957|1-0|8.00|1-1|x|2-1|9.00')
        self.assertEqual(result, {'1320957': {'1-0': 8.0, '2-1': 9.0}})

    def test_a_row_with_no_usable_price_is_dropped_entirely(self):
        self.assertEqual(parsing.parse_pair_table('1320957|1-0|x|1-1|y'), {})

    def test_columns_are_read_two_at_a_time(self):
        """**要用「价本身也像比分」的语料**才分得出步长是 2 还是 1。

        寻常语料里第二段是 `1-1` 这种，按步长 1 读时 `float('1-1')` 抛掉、
        结果碰巧一样。这里第三对的比分是 `9.5`，两种步长会给出不同的档位。
        """
        self.assertEqual(parsing.parse_pair_table('1320957|1-0|8.00|9.5|7.50'),
                         {'1320957': {'1-0': 8.0, '9.5': 7.5}})

    def test_a_dangling_score_is_ignored(self):
        """末尾落单的比分没有配对的价，直接忽略而不是补一个默认值。"""
        self.assertEqual(parsing.parse_pair_table('1320957|1-0|8.00|1-1'),
                         {'1320957': {'1-0': 8.0}})

    def test_an_id_alone_is_not_enough(self):
        """段数不足时那个取对的循环一对也取不出，末尾的 `if odds` 丢掉这一行
        ——**迁移前那道 `len(parts) < 2` 的守卫因此一次也没拦下过东西**，
        已删（判据 9 第一类）。"""
        self.assertEqual(parsing.parse_pair_table('1320957'), {})
        self.assertEqual(parsing.parse_pair_table('1320957|1-0'), {})

    def test_empty_input(self):
        for content in ('', None, '   ', '#只有注释'):
            with self.subTest(content=content):
                self.assertEqual(parsing.parse_pair_table(content), {})


class ColumnTableTests(unittest.TestCase):
    """总进球与半全场那两张定长表。"""

    ZJQ_ROW = '1320957|11.00|5.60|3.90|4.30|6.50|11.00|21.00|26.00'
    BQC_ROW = '1320957|3.20|15.00|41.00|7.50|6.10|12.00|51.00|17.00|4.30'

    def _zjq(self, content):
        return parsing.parse_column_table(content, parsing.ZJQ_COLUMNS,
                                          parsing.ZJQ_MINIMUM)

    def _bqc(self, content):
        return parsing.parse_column_table(content, parsing.BQC_COLUMNS,
                                          parsing.BQC_MINIMUM)

    def test_maps_columns_to_named_buckets(self):
        result, failures = self._zjq(self.ZJQ_ROW)
        self.assertEqual(sorted(result['1320957']), sorted(ZJQ_BUCKETS))
        self.assertEqual(result['1320957']['0'], 11.0)
        self.assertEqual(result['1320957']['7+'], 26.0)
        self.assertEqual(failures, [])

    def test_bqc_maps_all_nine_combinations(self):
        result, _ = self._bqc(self.BQC_ROW)
        self.assertEqual(sorted(result['1320957']), sorted(BQC_OPTIONS))
        self.assertEqual(result['1320957']['胜胜'], 3.2)
        self.assertEqual(result['1320957']['负负'], 4.3)

    def test_minimum_length_is_tested_on_both_sides(self):
        """门槛两侧各一条，而且**要连 `failures` 一起断言**。

        只看结果字典分不出「被门槛挡下」与「进去之后越界失败」——两种情况
        的结果都是 `{}`。前者不该产生任何失败记录，后者会（判据 5：
        只断言一侧，反方向照样通过）。
        """
        exact = '|'.join(['1320957'] + ['1.5'] * (ZJQ_MINIMUM - 1))
        self.assertIn('1320957', self._zjq(exact)[0])
        short = '|'.join(['1320957'] + ['1.5'] * (ZJQ_MINIMUM - 2))
        self.assertEqual(self._zjq(short), ({}, []))

    def test_bqc_needs_two_more_columns_than_zjq(self):
        nine = '|'.join(['1320957'] + ['1.5'] * (BQC_MINIMUM - 2))
        self.assertEqual(self._bqc(nine), ({}, []))
        ten = '|'.join(['1320957'] + ['1.5'] * (BQC_MINIMUM - 1))
        self.assertIn('1320957', self._bqc(ten)[0])

    def test_a_bad_price_takes_the_whole_row(self):
        """**与比分那张表相反**：定长表的每一列都是必需的。"""
        result, failures = self._zjq(
            '1320957|11.00|x|3.90|4.30|6.50|11.00|21.00|26.00')
        self.assertEqual(result, {})
        self.assertEqual(len(failures), 1)
        self.assertIn('1320957', failures[0][0])

    def test_a_bad_row_does_not_take_its_neighbours(self):
        """整行丢弃指的是**那一行**，不是整批（判据 18）。"""
        result, failures = self._zjq(
            self.ZJQ_ROW + '\n1320958|11.00|x|3.90|4.30|6.50|11.00|21.00|26.00')
        self.assertEqual(list(result), ['1320957'])
        self.assertEqual(len(failures), 1)

    def test_an_empty_field_reads_as_none_not_zero(self):
        """空水位是「没有报价」，不是「赔率为零」——读成 0 会让下游
        把它当成一个真实的赔率。**每一档都要能空**，只测一档的话
        把其中某一档改成「缺席容错」是发现不了的。"""
        for position in range(1, ZJQ_MINIMUM):
            with self.subTest(position=position):
                parts = ['1320957'] + ['1.5'] * (ZJQ_MINIMUM - 1) + ['2.5']
                parts[position] = ''
                result, failures = self._zjq('|'.join(parts))
                self.assertEqual(failures, [])
                bucket = ZJQ_BUCKETS[position - 1]
                self.assertIsNone(result['1320957'][bucket])

    def test_the_last_bucket_is_missing_tolerant_but_not_empty_tolerant(self):
        """**总进球最后一档的读法与其余七档不一样**（判据 17）。

        整段缺席（只有 8 段）→ `None`；段在但是空字符串 → `float('')` 抛，
        **整行被丢掉**。其余各档两种情况都读成 `None`。
        原样保留自迁移前，这条用例把差别钉住。
        """
        eight = '1320957|11.00|5.60|3.90|4.30|6.50|11.00|21.00'
        result, failures = self._zjq(eight)
        self.assertIsNone(result['1320957']['7+'])
        self.assertEqual(failures, [])

        empty_last = eight + '|'
        result, failures = self._zjq(empty_last)
        self.assertEqual(result, {})
        self.assertEqual(len(failures), 1)

    def test_extra_columns_are_ignored(self):
        result, _ = self._bqc(self.BQC_ROW + '|9.99')
        self.assertEqual(sorted(result['1320957']), sorted(BQC_OPTIONS))


class HistoryTableTests(unittest.TestCase):

    ASIAN_ROWS = [('09:00', '-0.5', '0.95', '0.90'),
                  ('10:00', '-0.75', '0.88', '0.98')]

    def _asian(self, html):
        return parsing.parse_history(html, parsing.ASIAN)

    def test_parses_the_asian_table(self):
        records, source = self._asian(_table(['时间', '亚盘', '主', '客'],
                                             self.ASIAN_ROWS))
        self.assertEqual(source, parsing.FROM_TABLE)
        self.assertEqual(records[0], {'time': '09:00', 'handicap': '-0.5',
                                      'home_odds': 0.95, 'away_odds': 0.90})

    def test_every_header_keyword_is_accepted(self):
        for keyword in ('亚盘', '让球', '盘口'):
            with self.subTest(keyword=keyword):
                records, _ = self._asian(
                    _table(['时间', keyword, '主', '客'], self.ASIAN_ROWS))
                self.assertEqual(len(records), 2)

    def test_only_the_header_row_decides_whether_a_table_counts(self):
        """**关键字只在表头里找**。数据格里出现「亚盘」两个字不该让一张
        欧赔表被当成亚盘表——按整张表匹配就会。
        """
        html = _table(['时间', '欧赔', '主', '客'],
                      [('09:00', '亚盘', '0.95', '0.90')])
        self.assertEqual(self._asian(html), ([], None))

    def test_a_header_row_that_also_has_data_cells_is_not_read_as_data(self):
        """表头行**只当表头**。真实页面里 `<th>` 与 `<td>` 混在同一行并不罕见，
        把它也当数据行会凭空多出一条 `time='时间'` 的记录——而那条记录
        能通过所有校验，不会报错。
        """
        header = ('<tr><th>时间</th><th>亚盘</th><th>主</th><th>客</th>'
                  '<td>时间</td><td>亚盘</td><td>0.90</td><td>0.90</td></tr>')
        body = ('<tr><td>09:00</td><td>-0.5</td><td>0.95</td><td>0.90</td></tr>')
        records, _ = self._asian(f'<html><table>{header}{body}</table></html>')
        self.assertEqual([r['time'] for r in records], ['09:00'])

    def test_a_table_without_a_keyword_is_skipped_entirely(self):
        records, source = self._asian(
            _table(['时间', '欧赔', '主', '客'], self.ASIAN_ROWS))
        self.assertEqual(records, [])
        self.assertIsNone(source)

    def test_only_the_correct_score_header_is_case_insensitive(self):
        """**三条路的表头匹配不一致**：比分那路多一次 `.lower()`，
        所以 `CS` 这样的大写表头只有它认。原样保留自迁移前（判据 17）。"""
        cs_rows = [('09:00', '1-0', '8.00')]
        records, _ = parsing.parse_history(
            _table(['time', 'CS', 'odds'], cs_rows), parsing.CORRECT_SCORE)
        self.assertEqual(len(records), 1)
        # 亚盘那路没有小写化，大写关键字命不中（拿它自己的英文关键字试）
        self.assertEqual(parsing.ASIAN.header_keywords_lower, ())

    def test_a_header_only_table_yields_nothing(self):
        self.assertEqual(self._asian(_table(['时间', '亚盘', '主', '客'], []))[0],
                         [])

    def test_minimum_cell_count_differs_between_markets(self):
        """亚盘要四列、大小球只要三列——**同一份三列的表在两处结果不同**。"""
        three = [('09:00', '-0.5', '0.95')]
        self.assertEqual(self._asian(_table(['时间', '亚盘', '主'], three))[0], [])
        goals, _ = parsing.parse_history(
            _table(['时间', '进球', '大'], [('09:00', '2.5', '0.95')]),
            parsing.GOALS)
        self.assertEqual(len(goals), 1)
        self.assertIsNone(goals[0]['under_odds'])

    def test_the_time_guard_is_tested_on_both_sides(self):
        rows = [('', '-0.5', '0.95', '0.90'),
                ('X' * (MAX_TIME_LENGTH + 1), '-0.5', '0.95', '0.90'),
                ('X' * MAX_TIME_LENGTH, '-0.5', '0.95', '0.90')]
        records, _ = self._asian(_table(['时间', '亚盘', '主', '客'], rows))
        self.assertEqual([r['time'] for r in records], ['X' * MAX_TIME_LENGTH])

    def test_the_handicap_guard_is_tested_on_both_sides(self):
        """**只有亚盘有这道门槛**，边界是「超过 20 才丢」。"""
        rows = [('09:00', '主' * (MAX_HANDICAP_LENGTH + 1), '0.95', '0.90'),
                ('10:00', '主' * MAX_HANDICAP_LENGTH, '0.95', '0.90')]
        records, _ = self._asian(_table(['时间', '亚盘', '主', '客'], rows))
        self.assertEqual([r['time'] for r in records], ['10:00'])

    def test_goals_has_no_length_guard_on_its_line(self):
        """同样长的一格在大小球那边留着——两条路的严格度不同。"""
        long_line = '大' * (MAX_HANDICAP_LENGTH + 1)
        records, _ = parsing.parse_history(
            _table(['时间', '进球', '大', '小'],
                   [('09:00', long_line, '0.95', '0.90')]), parsing.GOALS)
        self.assertEqual(records[0]['line'], long_line)

    def test_the_score_guard_requires_a_dash(self):
        rows = [('09:00', '10', '8.00'), ('10:00', '', '8.00'),
                ('11:00', '1-0', '8.00')]
        records, _ = parsing.parse_history(_table(['时间', '比分', '赔率'], rows),
                                           parsing.CORRECT_SCORE)
        self.assertEqual([r['score'] for r in records], ['1-0'])

    def test_non_numeric_prices_become_none_without_dropping_the_row(self):
        """水位读不出来时那一格给 `None`，**这一行还在**——
        时间与盘口本身仍然是有用的信息。"""
        records, _ = self._asian(_table(['时间', '亚盘', '主', '客'],
                                        [('09:00', '-0.5', '封', '-')]))
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]['home_odds'])
        self.assertIsNone(records[0]['away_odds'])

    def test_negative_prices_are_refused(self):
        """水位不会是负数，出现负值说明取错了列。"""
        records, _ = self._asian(_table(['时间', '亚盘', '主', '客'],
                                        [('09:00', '-0.5', '-0.95', '0.90')]))
        self.assertIsNone(records[0]['home_odds'])
        self.assertEqual(records[0]['away_odds'], 0.90)

    def test_nested_tags_are_stripped(self):
        records, _ = self._asian(_table(
            ['时间', '亚盘', '主', '客'],
            [('<b>09:00</b>', '<i>-0.5</i>', '<span>0.95</span>', '0.90')]))
        self.assertEqual(records[0]['time'], '09:00')
        self.assertEqual(records[0]['home_odds'], 0.95)

    def test_hidden_tables_are_parsed_too(self):
        """`display:none` 挡不住解析——**页面上藏与不藏在这一层没有区别**。"""
        hidden = _table(['时间', '亚盘', '主', '客'], self.ASIAN_ROWS).replace(
            '<table>', '<table style="display:none">')
        self.assertEqual(len(self._asian(hidden)[0]), 2)

    def test_the_hidden_style_scrub_only_touches_cell_text(self):
        """**那道清洗对表格抽取是空操作**，钉住它真正的效果。

        `<table[^>]*>` 这类正则本来就不看属性，所以上一条即使不做清洗也一样
        通过。清洗唯一能改变的是**单元格文本里恰好出现这几个字**的情况
        ——留着它是因为删掉算行为改动，但别以为「藏起来的行靠它才解析得到」。
        """
        records, _ = self._asian(_table(
            ['时间', '亚盘', '主', '客'],
            [('09:00', '-0.5display:none', '0.95', '0.90')]))
        self.assertEqual(records[0]['handicap'], '-0.5')


class HistoryScriptTests(unittest.TestCase):

    SERIES = ' 09:00, -0.5, 0.95, 0.90 ; 10:00, -0.75, 0.88, 0.98 '

    def _asian(self, html):
        return parsing.parse_history(html, parsing.ASIAN)

    def test_falls_back_to_the_script(self):
        records, source = self._asian(
            _script_page('亚盘 ' + 'x' * MIN_SCRIPT_LENGTH + self.SERIES))
        self.assertEqual(source, parsing.FROM_SCRIPT)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['home_odds'], 0.95)

    def test_a_table_with_records_wins_over_the_script(self):
        page = (_table(['时间', '亚盘', '主', '客'],
                       [('09:00', '-0.5', '0.95', '0.90')]).replace(
                           '</body></html>', '')
                + f'<script>亚盘 {"x" * MIN_SCRIPT_LENGTH} 23:59, -9.5, 1.11, 2.22'
                  '</script></body></html>')
        records, source = self._asian(page)
        self.assertEqual(source, parsing.FROM_TABLE)
        self.assertEqual([r['time'] for r in records], ['09:00'])

    ONE_ROW = ' 09:00, -0.5, 0.95, 0.90'

    def test_short_scripts_are_skipped(self):
        """门槛两侧各一条，**长度要真的数出来**。

        第一版拿 `'亚盘' + SERIES` 当「太短」的样本，而它有 53 个字符、
        本来就够长，于是那一半什么也没测（判据 28）。
        """
        keyword = '亚盘'
        short = keyword + self.ONE_ROW
        self.assertLess(len(short), MIN_SCRIPT_LENGTH)
        self.assertEqual(self._asian(_script_page(short))[0], [])

        padding = 'x' * (MIN_SCRIPT_LENGTH - len(short))
        exactly_long_enough = keyword + padding + self.ONE_ROW
        self.assertEqual(len(exactly_long_enough), MIN_SCRIPT_LENGTH)
        self.assertEqual(len(self._asian(_script_page(exactly_long_enough))[0]), 1)

    def test_a_script_without_the_keyword_is_skipped(self):
        records, _ = self._asian(
            _script_page('欧赔 ' + 'x' * MIN_SCRIPT_LENGTH + self.SERIES))
        self.assertEqual(records, [])

    def test_each_market_has_its_own_script_keywords(self):
        for keyword in ('亚盘', 'AH', 'asian'):
            with self.subTest(keyword=keyword):
                records, _ = self._asian(_script_page(
                    keyword + ' ' + 'x' * MIN_SCRIPT_LENGTH + self.SERIES))
                self.assertEqual(len(records), 2)

    def test_a_json_style_series_is_not_matched(self):
        """**钉住一处限制**：时间那一组前面没有可选引号，所以
        `["09:00", …]` 一个也匹配不上，只有裸的 `09:00, …` 才认。
        真页面若改成 JSON 数组，这条回退路会静默失效。"""
        json_style = '["09:00", -0.5, 0.95, 0.90],["10:00", -0.75, 0.88, 0.98]'
        records, source = self._asian(
            _script_page('亚盘 ' + 'x' * MIN_SCRIPT_LENGTH + json_style))
        self.assertEqual(records, [])
        self.assertIsNone(source)

    def test_a_quoted_value_is_matched(self):
        records, _ = self._asian(_script_page(
            '亚盘 ' + 'x' * MIN_SCRIPT_LENGTH + ' 09:00,"-0.5",0.95,0.90'))
        self.assertEqual(records[0]['handicap'], '-0.5')

    def test_the_script_path_skips_the_table_guards(self):
        """**从脚本刮出来的记录不走时间长度那道校验**——同样的值在表格里
        会被丢掉。这正是 `parse_history` 要一并返回来源的原因。"""
        self.assertEqual(parsing.ASIAN.guards[0][0], 'time')
        records, source = self._asian(_script_page(
            '亚盘 ' + 'x' * MIN_SCRIPT_LENGTH + ' 09:00, ' + '主' * 40
            + ', 0.95, 0.90'))
        self.assertEqual(source, parsing.FROM_SCRIPT)
        self.assertEqual(len(records[0]['handicap']), 40)

    def test_only_the_first_matching_script_contributes(self):
        """命中一段就停——**多段脚本往往是同一份数据的不同视图**，
        全收进来会重复。用两段都能匹配的脚本才测得到这一点。
        """
        page = ('<html><body>'
                f'<script>亚盘 {"x" * MIN_SCRIPT_LENGTH}{self.ONE_ROW}</script>'
                f'<script>亚盘 {"x" * MIN_SCRIPT_LENGTH} 23:59, -9.5, 1.11, 2.22'
                '</script></body></html>')
        records, source = self._asian(page)
        self.assertEqual(source, parsing.FROM_SCRIPT)
        self.assertEqual([r['time'] for r in records], ['09:00'])

    def test_empty_html_yields_nothing(self):
        for html in ('', None):
            with self.subTest(html=html):
                self.assertEqual(self._asian(html), ([], None))


class AdapterTests(unittest.TestCase):

    ASIAN_PAGE = _table(['时间', '亚盘', '主', '客'],
                        [('09:00', '-0.5', '0.95', '0.90')])

    COMPANY_PAGE = '''
<h2>韦*指数变化</h2><table>
<tr><th>序号</th><th>时间</th><th>更新</th><th>指数</th></tr>
<tr><td>1</td><td>2026-08-30 01:01:05</td><td><span>赛前0分</span></td>
 <td><span>0.81</span></td><td><span>平手</span></td><td><span>0.99</span></td>
 <td>52.37</td><td>47.63</td><td>0.92</td><td>0.98</td><td>0.95</td></tr>
</table>
'''

    SUMMARY_PAGE = '''
<table><tr firsttime="2026-08-24 22:14:55"><td>1</td><td>平均*</td>
 <td id="chupan-w-0" data="0.67">0.67</td><td id="chupan-s-0" data="0.25">平/半</td>
 <td id="chupan-l-0" data="1.11">1.11</td>
 <td cid="0" data="0.99">0.99</td><td cid="0" data="0.25">平/半</td>
 <td cid="0" data="0.83">0.83</td></tr></table>
'''

    def _urls(self, pages, fn=None, match_id='1320957'):
        fn = fn or fetching_adapter.fetch_zgzcw_asian_history
        with mock.patch.object(fetching_adapter, 'fetch_zgzcw',
                               side_effect=pages) as fetched:
            result = fn(match_id)
        return result, [call.args[0] for call in fetched.call_args_list]

    def test_asian_asks_the_company_detail_before_the_summary(self):
        """先要 Bet365 的完整序列，拿不到才退回初盘/即时盘那两点摘要。

        **只请求一个公司明细页**，不逐公司扫十几页——那是旧实现的代价。
        """
        _, urls = self._urls(['', ''])
        self.assertEqual(len(urls), 2)
        self.assertIn('ypdb/zhishu?company_id=2', urls[0])
        self.assertTrue(urls[1].endswith('/ypdb'), urls[1])

    def test_goals_uses_its_own_path(self):
        _, urls = self._urls(['', ''], fetching_adapter.fetch_zgzcw_goals_history)
        self.assertIn('dxdb/zhishu?company_id=2', urls[0])
        self.assertTrue(urls[1].endswith('/dxdb'), urls[1])

    def test_cs_history_makes_no_request_at_all(self):
        """中国足彩网不公开比分盘历史。返回空比拿欧亚盘冒充要好。"""
        with mock.patch.object(fetching_adapter, 'fetch_zgzcw') as fetched:
            self.assertEqual(fetching_adapter.fetch_zgzcw_cs_history('1320957'),
                             {'history': []})
        fetched.assert_not_called()

    def test_the_company_detail_wins_and_the_summary_is_not_asked(self):
        result, urls = self._urls([self.COMPANY_PAGE, 'unused'])
        self.assertEqual(len(urls), 1, '明细页已经给出序列，不该再问汇总页')
        self.assertEqual(result['history_source'], 'zgzcw_company_detail')
        self.assertEqual(result['company_id'], '2')
        self.assertTrue(result['history'])

    def test_an_empty_detail_falls_back_to_the_summary(self):
        """两个来源要能分辨——下游据此判断这份序列有多完整。"""
        result, urls = self._urls(['', self.SUMMARY_PAGE])
        self.assertEqual(len(urls), 2)
        self.assertEqual(result['history_source'],
                         'zgzcw_opening_current_fallback')
        self.assertEqual(result['company'], '平均*')

    def test_a_failing_detail_does_not_stop_the_summary(self):
        with mock.patch.object(fetching_adapter, 'fetch_zgzcw',
                               side_effect=[RuntimeError('boom'), self.SUMMARY_PAGE]):
            with mock.patch.object(fetching_adapter.log, 'warning') as warned:
                result = fetching_adapter.fetch_zgzcw_asian_history('1320957')
        self.assertTrue(result['history'])
        warned.assert_called_once()

    def test_all_pages_empty_yields_an_empty_history(self):
        with mock.patch.object(fetching_adapter, 'fetch_zgzcw', return_value=None):
            self.assertEqual(
                fetching_adapter.fetch_zgzcw_asian_history('1320957'),
                {'history': []})

    def test_table_adapters_return_an_empty_dict_when_nothing_was_fetched(self):
        for fn in (schedules_adapter.fetch_beidan_bifen,
                   schedules_adapter.fetch_beidan_zjq,
                   schedules_adapter.fetch_beidan_bqc):
            for content in (None, ''):
                with self.subTest(fn=fn.__name__, content=content):
                    with mock.patch.object(schedules_adapter, 'fetch',
                                           return_value=content):
                        self.assertEqual(fn('2026-08-28'), {})

    def test_table_adapters_log_the_rows_they_could_not_read(self):
        bad = '1320957|11.00|x|3.90|4.30|6.50|11.00|21.00|26.00'
        with mock.patch.object(schedules_adapter, 'fetch', return_value=bad):
            with mock.patch.object(schedules_adapter.log, 'warning') as warned:
                result = schedules_adapter.fetch_beidan_zjq('2026-08-28')
        self.assertEqual(result, {})
        warned.assert_called_once()
        self.assertIn('总进球', warned.call_args.args[0])

    def test_the_pair_table_adapter_never_reports_failures(self):
        """比分那张表逐对跳过，所以没有「读不懂的行」可报。"""
        with mock.patch.object(schedules_adapter, 'fetch',
                               return_value='1320957|1-0|x'):
            with mock.patch.object(schedules_adapter.log, 'warning') as warned:
                schedules_adapter.fetch_beidan_bifen('2026-08-28')
        warned.assert_not_called()

    def test_a_fetch_failure_is_swallowed_into_an_empty_dict(self):
        with mock.patch.object(schedules_adapter, 'fetch',
                               side_effect=RuntimeError('boom')):
            with mock.patch.object(schedules_adapter.log, 'error') as logged:
                self.assertEqual(
                    schedules_adapter.fetch_beidan_zjq('2026-08-28'), {})
        logged.assert_called_once()


class DecodingTests(unittest.TestCase):
    """解码不在解析层，但 §十一·1 那个坑值得留一条用例盯着。

    迁移前 `fetch` 写的是 `decode(enc, errors='replace')`——那个调用**永远
    不抛异常**，于是四个候选编码里第一个总是「成功」，gbk 页面被当 utf-8
    解出整页乱码，接口返回 200、列表为空、不报任何错。现在是严格解码。
    """

    def _fetch_bytes(self, raw):
        response = mock.MagicMock()
        response.read.return_value = raw
        response.__enter__.return_value = response
        with mock.patch.object(fetching_adapter.urllib.request, 'urlopen',
                               return_value=response):
            return fetching_adapter.fetch('http://example.invalid/x')

    def test_a_gbk_page_comes_back_as_readable_chinese(self):
        text = '北京单场 主胜 1.85'
        self.assertEqual(self._fetch_bytes(text.encode('gbk')), text)

    def test_a_utf8_page_comes_back_unchanged(self):
        text = '北京单场 主胜 1.85'
        self.assertEqual(self._fetch_bytes(text.encode('utf-8')), text)

    def test_the_candidate_encoding_loop_is_strict(self):
        """**按 AST 判定，而且只判循环里那一处**。

        只断言「gbk 页面能解对」是不够的：退回 `errors='replace'` 之后，
        utf-8 也能把 gbk 字节「解成功」——只是解成乱码，而乱码在别的断言里
        长得和成功一样。所以直接盯写法。

        **兜底的那两处不算**：四种编码全失败之后总要返回点什么，
        那里的 `errors='replace'` 是有意的。第一版按整个文件数，
        把它们也算了进去（判据 5 的形状：范围划错，断言就守错了东西）。
        """
        source = pathlib.Path('src/beidan/fetching.py').read_text(encoding='utf-8')
        fetch_fn = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'fetch')
        loops = [node for node in ast.walk(fetch_fn) if isinstance(node, ast.For)]
        self.assertTrue(loops, '候选编码的循环不见了')
        in_loop = [call for loop in loops for call in ast.walk(loop)
                   if isinstance(call, ast.Call)
                   and isinstance(call.func, ast.Attribute)
                   and call.func.attr == 'decode']
        self.assertTrue(in_loop, '循环里没有解码调用')
        for call in in_loop:
            self.assertEqual([kw.arg for kw in call.keywords], [],
                             f'第 {call.lineno} 行的解码不是严格的')


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'urllib.error', 'src.common.kv_store',
                     'src.foundation.store', 'src.beidan.fetching'}
FORBIDDEN_CALLS = {'now', 'today', 'utcnow', 'strftime'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/beidan/parsing.py'
    ADAPTER = 'src/beidan/fetching.py'

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
        self.assertEqual(self._imports(self.DOMAIN) & FORBIDDEN_IMPORTS, set())

    def test_domain_never_reads_the_clock(self):
        self.assertEqual(self._clock_calls(self.DOMAIN), set())

    def test_the_guards_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())
        self.assertNotEqual(self._clock_calls(self.ADAPTER), set())


if __name__ == '__main__':
    unittest.main()
