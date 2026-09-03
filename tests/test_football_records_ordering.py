# -*- coding: utf-8 -*-
"""预测记录页按竞彩期次分组、按赛事编号排序。

这些函数是纯函数，直接抽出来丢给 node 执行——字符串断言只能证明代码里写了
某个词，证明不了比较器真的把 010 排在 007 后面。
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


HTML = Path('web/index.html').read_text(encoding='utf-8')
NODE = shutil.which('node')

EXPORTED = (
    'predictionRecordDateKey',
    'predictionMatchNumParts',
    'predictionSessionKey',
    'predictionSessionLabel',
    'sortPredictionRecords',
)


def _extract_function(name):
    start = HTML.index(f'function {name}(')
    depth = 0
    for index in range(HTML.index('{', start), len(HTML)):
        if HTML[index] == '{':
            depth += 1
        elif HTML[index] == '}':
            depth -= 1
            if depth == 0:
                return HTML[start:index + 1]
    raise AssertionError(f'{name} 的函数体没有闭合')


def _extract_const(name):
    match = re.search(rf'^const {name} = .*?;$', HTML, re.M | re.S)
    if not match:
        raise AssertionError(f'找不到常量 {name}')
    return match.group(0)


def _run_js(expression):
    source = '\n'.join(
        [_extract_const('PREDICTION_WEEKDAYS')]
        + [_extract_function(name) for name in EXPORTED]
        + [f'console.log(JSON.stringify({expression}));']
    )
    completed = subprocess.run(
        [NODE, '--input-type=module', '-e', source],
        capture_output=True, text=True, timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(f'node 执行失败: {completed.stderr.strip()}')
    return json.loads(completed.stdout)


def _record(match_num, match_time, created_at='2026-09-01T12:00:00'):
    return {'match_num': match_num, 'match_time': match_time,
            'created_at': created_at, 'home': 'H', 'away': 'A'}


@unittest.skipIf(NODE is None, 'node 不可用')
class PredictionSessionGrouping(unittest.TestCase):

    def test_early_morning_match_belongs_to_the_previous_betting_day(self):
        """09-03 是周四，但编号写着周三——凌晨场算前一天的期次。"""
        key = _run_js(
            "predictionSessionKey(%s)" % json.dumps(_record('周三010', '09-03 02:45'))
        )
        self.assertEqual(key, '2026-09-02')

    def test_same_day_match_keeps_its_own_date_as_the_session(self):
        key = _run_js(
            "predictionSessionKey(%s)" % json.dumps(_record('周三005', '09-02 21:00'))
        )
        self.assertEqual(key, '2026-09-02')

    def test_session_label_shows_date_and_weekday(self):
        self.assertEqual(_run_js("predictionSessionLabel('2026-09-02')"), '09-02 周三')

    def test_record_without_a_lottery_number_falls_back_to_its_match_date(self):
        key = _run_js(
            "predictionSessionKey(%s)" % json.dumps(_record(None, '09-02 21:00'))
        )
        self.assertEqual(key, '2026-09-02')


@unittest.skipIf(NODE is None, 'node 不可用')
class PredictionRecordOrdering(unittest.TestCase):

    def _order(self, records, ascending=True):
        return _run_js(
            'sortPredictionRecords(%s, %s).map(r => r.match_num)'
            % (json.dumps(records), 'true' if ascending else 'false')
        )

    def test_numbers_sort_ascending_within_one_betting_day(self):
        records = [
            _record('周三010', '09-03 02:45'),
            _record('周三007', '09-03 02:45'),
            _record('周三008', '09-03 02:45'),
        ]
        self.assertEqual(self._order(records), ['周三007', '周三008', '周三010'])

    def test_descending_flag_reverses_only_the_numbers(self):
        records = [
            _record('周三007', '09-03 02:45'),
            _record('周三010', '09-03 02:45'),
        ]
        self.assertEqual(self._order(records, ascending=False),
                         ['周三010', '周三007'])

    def test_newer_betting_day_comes_first_regardless_of_direction(self):
        records = [
            _record('周二003', '09-02 02:45'),
            _record('周三001', '09-02 17:30'),
        ]
        for ascending in (True, False):
            self.assertEqual(self._order(records, ascending)[0], '周三001')

    def test_records_without_a_number_sink_below_numbered_ones(self):
        records = [
            _record(None, '09-02 23:00'),
            _record('周三001', '09-02 17:30'),
        ]
        self.assertEqual(self._order(records), ['周三001', None])

    def test_unnumbered_records_are_ordered_by_match_time_desc(self):
        records = [
            _record(None, '09-02 18:00'),
            _record(None, '09-02 23:00'),
        ]
        times = _run_js(
            'sortPredictionRecords(%s, true).map(r => r.match_time)'
            % json.dumps(records)
        )
        self.assertEqual(times, ['09-02 23:00', '09-02 18:00'])


class PredictionToolbarWiring(unittest.TestCase):
    """排序/分组函数必须真的被列表用上，写了不接等于没写。"""

    def _football_renderer(self):
        body = HTML.split('async function loadPredictions() {', 1)[1]
        end = re.search(r'^(?:async )?function ', body, re.M)
        return body[:end.start()] if end else body

    def test_filtering_groups_by_betting_session_not_calendar_date(self):
        renderer = self._football_renderer()
        self.assertIn('predictionSessionKey(record) === predictionDateFilter', renderer)

    def test_list_is_rendered_through_the_number_sorter(self):
        renderer = self._football_renderer()
        self.assertIn('sortPredictionRecords(', renderer)

    def test_toolbar_offers_an_order_toggle(self):
        self.assertIn('function togglePredictionSortOrder(', HTML)
        self.assertIn('onclick="togglePredictionSortOrder()"', HTML)

    def test_calendar_counts_are_keyed_by_session(self):
        prepare = HTML.split('function preparePredictionCalendar(', 1)[1].split(
            '\nfunction ', 1)[0]
        self.assertIn('predictionSessionKey', prepare)
        self.assertNotIn('predictionRecordDateKey', prepare)

    def test_session_options_are_labelled_with_the_weekday(self):
        self.assertIn('predictionSessionLabel(', HTML.split(
            'function renderPredictionDateToolbar(', 1)[1])
