# -*- coding: utf-8 -*-
"""竞彩官网赛程的 sporttery_* 记录也必须走队名+日期兜底完成回填。

主赛程源换成竞彩官网后 match_id 变成 `sporttery_<id>`，若同步入口直接按
「非数字 ID」整条跳过，这些记录会永远停在「准备同步」。
"""

from unittest import mock

from src.football import result_sync
from src.football.result_sync import PredictionHistory


# 写死到过去的绝对时刻：不带年的 MM-DD 会被补成当前年，那种用例会随时钟变红。
SETTLED_MATCH_TIME = '2020-09-02 21:00'


def _history_with(record):
    history = PredictionHistory()
    history._save = lambda: None
    history._save_record = lambda saved: 'memory'
    history.records = [record]
    return history


def _sporttery_record():
    return {
        'match_id': 'sporttery_2041241',
        'league': '意大利杯',
        'home': '乌迪内斯',
        'away': '威尼斯',
        'match_time': SETTLED_MATCH_TIME,
        'sync_status': 'ready',
        'settled': False,
    }


def test_sporttery_record_settles_through_team_and_date_fallback():
    history = _history_with(_sporttery_record())
    team_result = {'score': '2-1', 'result': 'H', 'source': 'live_team'}

    with mock.patch.object(result_sync, '_global_history', history), \
         mock.patch.object(
             result_sync, 'fetch_result_by_team_and_date',
             return_value=team_result) as by_team:
        summary = result_sync.auto_sync_results()

    by_team.assert_called_once_with('乌迪内斯', '威尼斯', SETTLED_MATCH_TIME)
    assert summary['synced'] == 1
    assert history.records[0]['settled'] is True
    assert history.records[0]['actual_score'] == '2-1'


def test_sporttery_record_never_hits_the_500_analysis_page():
    """`sporttery_*` 不是 500 的 fid，按它去查 500 详情页只会抓错场次。"""
    history = _history_with(_sporttery_record())

    with mock.patch.object(result_sync, '_global_history', history), \
         mock.patch.object(result_sync, '_fetch_match_html') as analysis_page, \
         mock.patch.object(result_sync, '_fetch_live_score_by_fid') as by_fid, \
         mock.patch.object(
             result_sync, 'fetch_result_by_team_and_date',
             return_value={'score': '2-1', 'result': 'H', 'source': 'live_team'}):
        result_sync.auto_sync_results()

    analysis_page.assert_not_called()
    by_fid.assert_not_called()


def test_sporttery_record_without_any_result_is_counted_as_failed():
    history = _history_with(_sporttery_record())

    with mock.patch.object(result_sync, '_global_history', history), \
         mock.patch.object(
             result_sync, 'fetch_result_by_team_and_date', return_value=None):
        summary = result_sync.auto_sync_results()

    assert summary['failed'] == 1
    assert history.records[0]['settled'] is False
    assert history.records[0]['last_sync_error'] == '未找到赛果'
