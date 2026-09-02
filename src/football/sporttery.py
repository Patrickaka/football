# -*- coding: utf-8 -*-
"""中国竞彩网公开竞彩足球数据。

该接口由 sporttery.cn 自己的混合过关页面调用，不需要 API Key。这里只读取
公开赛程、销售玩法、官方让球和固定奖金；不提交任何投注或用户数据。
"""

from __future__ import annotations

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
