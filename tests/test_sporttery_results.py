# -*- coding: utf-8 -*-
"""竞彩官网赛程的记录，赛果也从竞彩官网按 matchId 取。

主赛程换成竞彩官网后，记录里的队名是竞彩简称（迈季宽广、赛哈海湾、胡巴卡德），
500.com 的完场页写的是另一套（迈季迈阿宽广、塞哈特海湾、卡达西亚），
按队名模糊匹配永远对不上，这些场次会一直停在「重试中」。竞彩官网的开奖接口
按 matchId 返回全场/半场比分，与记录里的 `sporttery_<id>` 一一对应，不需要
任何队名比对。
"""

from unittest import mock

import pytest

from src.domain.sports.football import settlement
from src.football import result_sync
from src.football.result_sync import PredictionHistory
from src.football.sporttery import (
    parse_sporttery_results,
    sporttery_result_url,
)


SETTLED_MATCH_TIME = '2020-09-03 23:55'


def _result_item(**overrides):
    item = {
        'matchId': 2041234, 'matchNumStr': '周四001', 'matchDate': '2026-09-03',
        'homeTeam': '迈季宽广', 'awayTeam': '拉斯永恒',
        'allHomeTeam': '迈季迈阿宽广', 'allAwayTeam': '拉斯永恒',
        'poolStatus': 'Payout', 'matchResultStatus': '2', 'resultStatus': '',
        'sectionsNo1': '1:1', 'sectionsNo999': '2:2', 'winFlag': 'D',
    }
    item.update(overrides)
    return item


def _payload(items, pages=1, page_no=1):
    return {
        'success': True, 'errorCode': '0',
        'value': {'pages': pages, 'pageNo': page_no, 'matchResult': items},
    }


class ParseSportteryResultsTests:
    def test_keys_results_by_sporttery_id_with_full_and_half_score(self):
        parsed = parse_sporttery_results(_payload([_result_item()]))

        assert parsed == {'2041234': {
            'score': '2-2', 'half_score': '1-1',
            'match_num': '周四001', 'match_date': '2026-09-03',
        }}

    def test_score_without_pool_status_is_still_a_final_score(self):
        """奥斯纳 vs 拜仁 这种单关未开售的场次 poolStatus 为空，但比分是真的。"""
        parsed = parse_sporttery_results(_payload([
            _result_item(matchId=2041242, poolStatus='', winFlag='',
                         sectionsNo1='1:2', sectionsNo999='1:4'),
        ]))

        assert parsed['2041242']['score'] == '1-4'
        assert parsed['2041242']['half_score'] == '1-2'

    def test_missing_or_malformed_full_score_is_skipped(self):
        parsed = parse_sporttery_results(_payload([
            _result_item(matchId=1, sectionsNo999=''),
            _result_item(matchId=2, sectionsNo999='-'),
            _result_item(matchId=3, sectionsNo999=None),
        ]))

        assert parsed == {}

    def test_malformed_half_score_does_not_drop_the_full_score(self):
        parsed = parse_sporttery_results(_payload([
            _result_item(sectionsNo1=''),
        ]))

        assert parsed['2041234']['score'] == '2-2'
        assert parsed['2041234']['half_score'] is None

    def test_api_error_raises(self):
        with pytest.raises(ValueError):
            parse_sporttery_results({'success': False, 'errorCode': '500',
                                     'errorMessage': 'boom'})

    def test_non_object_payload_raises(self):
        with pytest.raises(ValueError):
            parse_sporttery_results([])


class SportteryResultUrlTests:
    def test_url_carries_date_window_and_page(self):
        url = sporttery_result_url('2026-09-02', '2026-09-04', page_no=2)

        assert url.startswith(
            'https://webapi.sporttery.cn/gateway/uniform/football/'
            'getUniformMatchResultV1.qry?'
        )
        assert 'matchBeginDate=2026-09-02' in url
        assert 'matchEndDate=2026-09-04' in url
        assert 'pageNo=2' in url
        assert 'pageSize=100' in url


class FetchResultBySportteryIdTests:
    def test_non_sporttery_ids_are_not_looked_up(self):
        with mock.patch.object(result_sync, '_fetch_sporttery_results') as fetched:
            assert result_sync.fetch_result_by_sporttery_id(
                '1467708', SETTLED_MATCH_TIME) is None
            assert result_sync.fetch_result_by_sporttery_id(
                '', SETTLED_MATCH_TIME) is None

        fetched.assert_not_called()

    def test_hit_is_returned_with_result_and_official_source(self):
        with mock.patch.object(
                result_sync, '_fetch_sporttery_results',
                return_value={'2041234': {
                    'score': '2-2', 'half_score': '1-1',
                    'match_num': '周四001', 'match_date': '2020-09-03',
                }}) as fetched:
            result = result_sync.fetch_result_by_sporttery_id(
                'sporttery_2041234', SETTLED_MATCH_TIME)

        assert result == {
            'score': '2-2', 'result': 'D', 'half_score': '1-1',
            'source': 'sporttery',
        }
        fetched.assert_called_once_with('2020-09-02', '2020-09-04')

    def test_miss_returns_none(self):
        with mock.patch.object(
                result_sync, '_fetch_sporttery_results', return_value={}):
            assert result_sync.fetch_result_by_sporttery_id(
                'sporttery_2041234', SETTLED_MATCH_TIME) is None

    def test_unparseable_match_time_skips_the_request(self):
        with mock.patch.object(result_sync, '_fetch_sporttery_results') as fetched:
            assert result_sync.fetch_result_by_sporttery_id(
                'sporttery_2041234', '') is None

        fetched.assert_not_called()


class FetchSportteryResultsPagingTests:
    def test_all_pages_are_merged(self):
        pages = {
            1: _payload([_result_item(matchId=1)], pages=2, page_no=1),
            2: _payload([_result_item(matchId=2)], pages=2, page_no=2),
        }

        def fake_fetch_json(url, referer=None):
            page_no = int(url.split('pageNo=')[1].split('&')[0])
            return pages[page_no]

        with mock.patch.object(result_sync, '_sporttery_fetch_json', fake_fetch_json):
            merged = result_sync._fetch_sporttery_results('2026-09-02', '2026-09-04')

        assert set(merged) == {'1', '2'}


def _history_with(record):
    history = PredictionHistory()
    history._save = lambda: None
    history._save_record = lambda saved: 'memory'
    history.records = [record]
    return history


def _sporttery_record():
    return {
        'match_id': 'sporttery_2041234',
        'league': '沙职',
        'home': '迈季宽广',
        'away': '拉斯永恒',
        'match_time': SETTLED_MATCH_TIME,
        'sync_status': 'ready',
        'settled': False,
    }


class AutoSyncPrefersSportteryResultsTests:
    def test_sporttery_record_settles_from_official_result_without_team_lookup(self):
        history = _history_with(_sporttery_record())
        official = {'score': '2-2', 'result': 'D', 'half_score': '1-1',
                    'source': 'sporttery'}

        with mock.patch.object(result_sync, '_global_history', history), \
             mock.patch.object(
                 result_sync, 'fetch_result_by_sporttery_id',
                 return_value=official), \
             mock.patch.object(result_sync, 'fetch_result_by_team_and_date') as by_team, \
             mock.patch.object(result_sync, '_fetch_match_html') as analysis_page:
            summary = result_sync.auto_sync_results()

        by_team.assert_not_called()
        analysis_page.assert_not_called()
        assert summary['synced'] == 1
        record = history.records[0]
        assert record['settled'] is True
        assert record['actual_score'] == '2-2'
        assert record['actual_half_score'] == '1-1'
        assert record['result_quality']['grade'] == 'high'

    def test_official_miss_still_falls_back_to_team_lookup(self):
        history = _history_with(_sporttery_record())

        with mock.patch.object(result_sync, '_global_history', history), \
             mock.patch.object(
                 result_sync, 'fetch_result_by_sporttery_id', return_value=None), \
             mock.patch.object(
                 result_sync, 'fetch_result_by_team_and_date',
                 return_value={'score': '2-2', 'result': 'D', 'source': 'live_team'}) as by_team:
            summary = result_sync.auto_sync_results()

        by_team.assert_called_once_with('迈季宽广', '拉斯永恒', SETTLED_MATCH_TIME)
        assert summary['synced'] == 1


class ResultQualityTreatsSportteryAsOfficialTests:
    def _record(self):
        return {'match_id': 'sporttery_2041234', 'match_time': SETTLED_MATCH_TIME}

    def test_sporttery_source_is_high_grade(self):
        quality = settlement._assess_result_quality(
            self._record(), '3-0', 'H', source='sporttery')

        assert quality['grade'] == 'high'
        assert 'unknown_source' not in quality['reasons']

    def test_low_information_scores_are_not_penalised_for_official_source(self):
        quality = settlement._assess_result_quality(
            self._record(), '1-1', 'D', source='sporttery')

        assert quality['grade'] == 'high'
        assert 'low_information_score_without_live_source' not in quality['reasons']
