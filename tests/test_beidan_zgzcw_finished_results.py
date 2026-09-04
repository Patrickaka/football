# -*- coding: utf-8 -*-
"""北单赛果优先从中国足彩网单场页取，而不是绕道 500.com 猜队名。

北单赛程本来就抓自这张页，页上完场那几行直接带终场比分和同一个 newplayid，
按 ID 对齐没有任何队名歧义。原解析器把完场行整段跳过（只要未完结的），
赛果就只能去 500.com 按队名找，跨站命名不一致时永远找不到。
"""

import unittest
from unittest import mock

from src.beidan import settling
from src.domain.sports.beidan.parsing import (
    parse_zgzcw_finished_results,
    parse_zgzcw_schedule,
)


def _row(row_id, num, home, away, score, newplayid, kickoff=''):
    title = f' title="比赛时间:{kickoff}"' if kickoff else ''
    return (
        f'<tr id="tr_{row_id}" m="智利甲" t="{kickoff}">'
        f'<td class="wh-1">{num}</td>'
        f'<td class="wh-2">智利甲</td>'
        f'<td class="wh-3"{title}>{kickoff}</td>'
        f'<td class="wh-4"><a newplayid="{newplayid}">{home}</a></td>'
        f'<td class="wh-5">{score}</td>'
        f'<td class="wh-6"><a>{away}</a></td>'
        f'</tr>'
    )


FINISHED = _row('1', '101', '利马切', '纽夫莱', '2:1', '4476410')
UNFINISHED = _row('2', '102', '萨斯菲', '博卡', 'VS', '4590966',
                  kickoff='2026-09-04 06:00')
HTML = f'<table>{FINISHED}{UNFINISHED}</table>'


class ParseFinishedResults(unittest.TestCase):

    def test_finished_row_is_indexed_by_its_analysis_id(self):
        results = parse_zgzcw_finished_results(HTML)

        self.assertEqual(results['4476410']['score'], '2-1')
        self.assertEqual(results['4476410']['home'], '利马切')
        self.assertEqual(results['4476410']['away'], '纽夫莱')

    def test_finished_row_is_also_indexed_by_its_row_id(self):
        """记录里存的可能是 zgzcw_id 而不是 analysis_id。"""
        self.assertIn('1', parse_zgzcw_finished_results(HTML))

    def test_unfinished_row_is_not_reported_as_a_result(self):
        results = parse_zgzcw_finished_results(HTML)

        self.assertNotIn('4590966', results)
        self.assertNotIn('2', results)

    def test_schedule_parser_still_only_returns_unfinished_matches(self):
        """两个解析器读同一张页，各取一半，不能互相串味。"""
        scheduled = parse_zgzcw_schedule(HTML)

        self.assertEqual([m['analysis_id'] for m in scheduled], ['4590966'])


class SettleFromZgzcwPage(unittest.TestCase):

    def _record(self):
        return {
            'match_id': '4476410', 'source': 'zgzcw', 'league': '智利甲',
            'home': '利马切', 'away': '纽夫莱',
            'date': '2020-09-03', 'time': '06:00',
            'sync_status': 'retry', 'sync_attempts': 3, 'settled': False,
            'predicted_1x2': {'H': 0.4, 'D': 0.3, 'A': 0.3},
        }

    def _run(self, record, finished_map):
        by_id = mock.Mock(return_value=None)
        by_team = mock.Mock(return_value=None)
        with mock.patch.object(settling, '_load_beidan_history',
                               return_value=[record]), \
             mock.patch.object(settling, '_save_beidan_sync_updates'), \
             mock.patch.object(settling, 'fetch_zgzcw_finished_results',
                               return_value=finished_map) as page:
            summary = settling.sync_beidan_results(
                fetch_by_id=by_id, fetch_by_team=by_team, force_retry=True,
            )
        return summary, by_id, by_team, page

    def test_record_settles_from_the_page_without_touching_500(self):
        record = self._record()
        summary, by_id, by_team, _ = self._run(
            record, {'4476410': {'score': '2-1', 'home': '利马切', 'away': '纽夫莱'}})

        self.assertEqual(summary['synced'], 1)
        self.assertTrue(record['settled'])
        self.assertEqual(record['actual_score'], '2-1')
        by_id.assert_not_called()
        by_team.assert_not_called()

    def test_missing_from_the_page_still_falls_back_to_500(self):
        record = self._record()
        summary, by_id, by_team, _ = self._run(record, {})

        by_team.assert_called_once()
        self.assertEqual(summary['failed'], 1)

    def test_page_is_fetched_once_per_round_not_once_per_record(self):
        records = [self._record(), dict(self._record(), match_id='4590966')]
        with mock.patch.object(settling, '_load_beidan_history',
                               return_value=records), \
             mock.patch.object(settling, '_save_beidan_sync_updates'), \
             mock.patch.object(settling, 'fetch_zgzcw_finished_results',
                               return_value={}) as page:
            settling.sync_beidan_results(
                fetch_by_id=mock.Mock(return_value=None),
                fetch_by_team=mock.Mock(return_value=None),
                force_retry=True,
            )

        self.assertEqual(page.call_count, 1)

    def test_page_failure_does_not_break_the_round(self):
        """源站抖动不该让整轮回填崩掉，退回 500 即可。"""
        record = self._record()
        by_team = mock.Mock(return_value={'score': '2-1', 'source': 'live_team'})
        with mock.patch.object(settling, '_load_beidan_history',
                               return_value=[record]), \
             mock.patch.object(settling, '_save_beidan_sync_updates'), \
             mock.patch.object(settling, 'fetch_zgzcw_finished_results',
                               side_effect=RuntimeError('WAF')):
            summary = settling.sync_beidan_results(
                fetch_by_id=mock.Mock(return_value=None),
                fetch_by_team=by_team, force_retry=True,
            )

        self.assertEqual(summary['synced'], 1)
