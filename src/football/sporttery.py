# -*- coding: utf-8 -*-
"""中国竞彩网公开竞彩足球数据。

该接口由 sporttery.cn 自己的混合过关页面调用，不需要 API Key。这里只读取
公开赛程、销售玩法、官方让球和固定奖金；不提交任何投注或用户数据。
"""

from __future__ import annotations

import re
from typing import Dict, List


SPORTTERY_BASE = 'https://www.sporttery.cn'
SPORTTERY_CALCULATOR_URL = (
    'https://webapi.sporttery.cn/gateway/uniform/football/'
    'getMatchCalculatorV1.qry?channel=pc'
)
SPORTTERY_REFERER = SPORTTERY_BASE + '/jc/jsq/zqhhgg/'


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _odds(market: Dict, labels) -> Dict | None:
    if not isinstance(market, dict):
        return None
    values = [_number(market.get(key)) for key in ('h', 'd', 'a')]
    if any(value is None or value <= 1.0 for value in values):
        return None
    return dict(zip(labels, values))


def _selling_pools(match: Dict) -> set[str]:
    return {
        str(pool.get('poolCode') or '').upper()
        for pool in (match.get('poolList') or [])
        if isinstance(pool, dict) and pool.get('poolStatus') == 'Selling'
    }


def parse_sporttery_calculator(payload: Dict) -> List[Dict]:
    """把竞彩网计算器 JSON 转成足球主列表结构。"""
    if not isinstance(payload, dict):
        raise ValueError('sporttery response is not an object')
    if str(payload.get('errorCode')) != '0' or not payload.get('success'):
        raise ValueError(
            'sporttery error: ' + str(payload.get('errorMessage') or 'unknown')
        )
    value = payload.get('value') or {}
    groups = value.get('matchInfoList') or []
    converted = []
    for group in groups:
        for raw in (group.get('subMatchList') or []):
            if not isinstance(raw, dict) or raw.get('isHide'):
                continue
            sporttery_id = str(raw.get('matchId') or '').strip()
            home = str(raw.get('homeTeamAbbName') or raw.get('homeTeamAllName') or '').strip()
            away = str(raw.get('awayTeamAbbName') or raw.get('awayTeamAllName') or '').strip()
            num = str(raw.get('matchNumStr') or '').replace(' ', '')
            if not (sporttery_id and home and away and num):
                continue

            pools = _selling_pools(raw)
            spf_odds = _odds(raw.get('had'), ('胜', '平', '负')) if 'HAD' in pools else None
            rqspf_odds = _odds(
                raw.get('hhad'), ('让胜', '让平', '让负')
            ) if 'HHAD' in pools else None
            handicap = _number((raw.get('hhad') or {}).get('goalLineValue'))
            if handicap is None:
                handicap = _number((raw.get('hhad') or {}).get('goalLine'))
            if handicap is not None and handicap.is_integer():
                handicap = int(handicap)
            rqspf_available = bool(rqspf_odds and handicap not in (None, 0))
            spf_available = bool(spf_odds)
            if not (spf_available or rqspf_available):
                continue

            match_date = str(raw.get('matchDate') or '').strip()
            match_time = str(raw.get('matchTime') or '').strip()[:5]
            short_date = match_date[5:] if len(match_date) >= 10 else match_date
            available = [
                name for name, enabled in (
                    ('spf', spf_available), ('rqspf', rqspf_available)
                ) if enabled
            ]
            converted.append({
                'home': home,
                'away': away,
                'home_full': str(raw.get('homeTeamAllName') or home).strip(),
                'away_full': str(raw.get('awayTeamAllName') or away).strip(),
                'home_code': str(
                    raw.get('homeTeamCode') or raw.get('homeTeamAbbEnName') or ''
                ).strip(),
                'away_code': str(
                    raw.get('awayTeamCode') or raw.get('awayTeamAbbEnName') or ''
                ).strip(),
                # 与 500 分析 ID 分域，避免把竞彩网 ID 错送给 500 详情页。
                'match_id': f'sporttery_{sporttery_id}',
                'sporttery_id': sporttery_id,
                'time': f'{short_date} {match_time}'.strip(),
                'date': match_date,
                'league': str(raw.get('leagueAbbName') or raw.get('leagueAllName') or '').strip(),
                'num': num,
                'schedule_source': 'sporttery',
                'analysis_source_id_available': False,
                'analysis_id': '',
                'lottery_source': 'sporttery',
                'lottery_offer_matched': True,
                'lottery_unavailable_reason': None,
                'lottery_handicap': handicap,
                'lottery_primary_market': (
                    'rqspf' if rqspf_available else 'spf' if spf_available else None
                ),
                'lottery_available_markets': available,
                'lottery_spf_available': spf_available,
                'lottery_rqspf_available': rqspf_available,
                'lottery_spf_odds': spf_odds,
                'lottery_rqspf_odds': rqspf_odds,
                'lottery_updated_at': max(
                    (f"{market.get('updateDate', '')} {market.get('updateTime', '')}".strip()
                     for market in (raw.get('had') or {}, raw.get('hhad') or {})
                     if market),
                    default='',
                ),
            })
    return converted


SPORTTERY_RESULT_URL = (
    'https://webapi.sporttery.cn/gateway/uniform/football/'
    'getUniformMatchResultV1.qry'
)
SPORTTERY_RESULT_REFERER = SPORTTERY_BASE + '/jc/zqsgkj/'
# 接口对 pageSize 的上限就是 100，传更大也按 100 返回。
SPORTTERY_RESULT_PAGE_SIZE = 100


def sporttery_result_url(begin_date: str, end_date: str, page_no: int = 1) -> str:
    """竞彩官网「足球赛果开奖」页调用的接口，按比赛日期窗口分页。"""
    return (
        f'{SPORTTERY_RESULT_URL}?matchBeginDate={begin_date}'
        f'&matchEndDate={end_date}&leagueId='
        f'&pageSize={SPORTTERY_RESULT_PAGE_SIZE}&pageNo={int(page_no)}'
        '&isFix=0&matchPage=1&pcOrWap=1'
    )


def _section_score(value) -> str | None:
    text = str(value or '').strip()
    if not re.fullmatch(r'\d{1,2}:\d{1,2}', text):
        return None
    return text.replace(':', '-')


def parse_sporttery_results(payload: Dict) -> Dict[str, Dict]:
    """把开奖接口 JSON 转成 {竞彩 matchId: 赛果}。

    未开赛的场次根本不会出现在这个接口里，出现了且 `sectionsNo999` 是合法
    比分就是完场。`poolStatus` 不作为门槛：单关未开售的场次它是空串，
    但比分照样是真的（奥斯纳 vs 拜仁 1:4）。
    """
    if not isinstance(payload, dict):
        raise ValueError('sporttery result response is not an object')
    if str(payload.get('errorCode')) != '0' or not payload.get('success'):
        raise ValueError(
            'sporttery result error: '
            + str(payload.get('errorMessage') or 'unknown')
        )
    value = payload.get('value') or {}
    results = {}
    for raw in (value.get('matchResult') or []):
        if not isinstance(raw, dict):
            continue
        sporttery_id = str(raw.get('matchId') or '').strip()
        score = _section_score(raw.get('sectionsNo999'))
        if not (sporttery_id and score):
            continue
        results[sporttery_id] = {
            'score': score,
            'half_score': _section_score(raw.get('sectionsNo1')),
            'match_num': str(raw.get('matchNumStr') or '').replace(' ', ''),
            'match_date': str(raw.get('matchDate') or '').strip(),
        }
    return results
