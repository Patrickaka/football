# -*- coding: utf-8 -*-
"""赛果同步应优先查询涵盖全部赛事的完场页。"""

from unittest import mock

from src.football import result_sync


FINISHED_ROW = '''
<tr id="a1320957">
  <td><span class="mainName">KTP科特卡</span></td>
  <td><div class="pk"><a class="clt1">0</a><span>-</span>
      <a class="clt3">3</a></div></td>
  <td><span class="clientName">PK35万塔</span></td>
</tr>
'''


def test_team_lookup_uses_the_all_matches_finished_page_before_jczq():
    with mock.patch.object(
            result_sync, '_finished_query_dates', return_value=['2026-08-20']), \
         mock.patch.object(
             result_sync, '_fetch_finished_html', return_value=FINISHED_ROW), \
         mock.patch.object(result_sync, '_fetch_live_html') as live_page:
        result = result_sync.fetch_result_by_team_and_date(
            '科特卡', 'Pk-35万塔', '2026-08-20 23:00',
        )

    assert result == {'score': '0-3', 'result': 'A', 'source': 'live_team'}
    live_page.assert_not_called()


def test_fid_lookup_accepts_finished_page_row_ids_prefixed_with_a():
    with mock.patch.object(
            result_sync, '_finished_query_dates', return_value=['2026-08-20']), \
         mock.patch.object(
             result_sync, '_fetch_finished_html', return_value=FINISHED_ROW), \
         mock.patch.object(result_sync, '_fetch_live_html') as live_page:
        score = result_sync._fetch_live_score_by_fid(
            '1320957', '2026-08-20 23:00',
        )

    assert score == '0-3'
    live_page.assert_not_called()
