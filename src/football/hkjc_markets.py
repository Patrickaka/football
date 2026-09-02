# -*- coding: utf-8 -*-
"""香港赛马会公开足球盘口，以及与中国竞彩网赛程的安全匹配。

HKJC 的足球页面通过公开 GraphQL 接口读取 HDC（亚洲让球）、HIL（大小球）
和 HAD（主客和）。接口不需要 API Key。这里只读取售前主盘口，不进行任何
投注操作。跨源匹配同时使用日期、开赛时间、球队代码/名称与 1X2 概率，低
分或歧义结果不会强行合并。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional


HKJC_GRAPHQL_URL = 'https://info.cld.hkjc.com/graphql/base/'
HKJC_REFERER = 'https://bet.hkjc.com/football/'

# HKJC 对足球查询使用服务端白名单，字段集合需与其公开网页查询保持一致。
# 只请求三个窄市场，避免响应体过大触发下游限制。
HKJC_MATCHES_QUERY = r'''
query matchList($startIndex: Int, $endIndex: Int,$startDate: String, $endDate: String, $matchIds: [String], $tournIds: [String], $fbOddsTypes: [FBOddsType]!, $fbOddsTypesM: [FBOddsType]!, $inplayOnly: Boolean, $featuredMatchesOnly: Boolean, $frontEndIds: [String], $earlySettlementOnly: Boolean, $showAllMatch: Boolean) {
  matches(startIndex: $startIndex,endIndex: $endIndex, startDate: $startDate, endDate: $endDate, matchIds: $matchIds, tournIds: $tournIds, fbOddsTypes: $fbOddsTypesM, inplayOnly: $inplayOnly, featuredMatchesOnly: $featuredMatchesOnly, frontEndIds: $frontEndIds, earlySettlementOnly: $earlySettlementOnly, showAllMatch: $showAllMatch) {
    id
    frontEndId
    matchDate
    kickOffTime
    status
    updateAt
    sequence
    esIndicatorEnabled
    homeTeam { id name_en name_ch }
    awayTeam { id name_en name_ch }
    tournament { id frontEndId nameProfileId isInteractiveServiceAvailable code name_en name_ch }
    isInteractiveServiceAvailable
    inplayDelay
    venue { code name_en name_ch }
    tvChannels { code name_en name_ch }
    liveEvents { id code }
    featureStartTime
    featureMatchSequence
    poolInfo { normalPools inplayPools sellingPools ntsInfo entInfo definedPools }
    runningResult { homeScore awayScore corner homeCorner awayCorner }
    runningResultExtra { homeScore awayScore corner homeCorner awayCorner }
    adminOperation { remark { typ } }
    foPools(fbOddsTypes: $fbOddsTypes) {
      id
      status
      oddsType
      instNo
      inplay
      name_ch
      name_en
      updateAt
      expectedSuspendDateTime
      lines {
        lineId
        status
        condition
        main
        combinations {
          combId
          str
          status
          offerEarlySettlement
          currentOdds
          selections { selId str name_ch name_en }
        }
      }
    }
  }
}
'''


def hkjc_request_payload() -> Dict:
    return {
        'query': HKJC_MATCHES_QUERY,
        'variables': {
            'fbOddsTypes': ['HAD', 'HDC', 'HIL'],
            'fbOddsTypesM': ['HAD', 'HDC', 'HIL'],
            'startDate': None,
            'endDate': None,
            'tournIds': None,
            'matchIds': None,
            'inplayOnly': False,
            'featuredMatchesOnly': False,
            'frontEndIds': None,
            'earlySettlementOnly': False,
            'showAllMatch': False,
            'startIndex': None,
            'endIndex': None,
        },
    }


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _line_value(condition) -> Optional[float]:
    """把 ``-0.5/-1.0`` 等拆盘条件转换为均值 -0.75。"""
    values = [_number(item) for item in str(condition or '').split('/')]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _main_line(pool: Dict) -> Optional[Dict]:
    available = [
        line for line in (pool.get('lines') or [])
        if isinstance(line, dict) and line.get('status') == 'AVAILABLE'
    ]
    return next((line for line in available if line.get('main')), None) or (
        available[0] if available else None
    )


def _combination_odds(line: Optional[Dict]) -> Dict[str, float]:
    result = {}
    for item in ((line or {}).get('combinations') or []):
        if not isinstance(item, dict) or item.get('status') != 'AVAILABLE':
            continue
        value = _number(item.get('currentOdds'))
        if value is not None and value > 1.0:
            result[str(item.get('str') or '').upper()] = value
    return result


def parse_hkjc_matches(payload: Dict) -> List[Dict]:
    """解析 HKJC 当前主盘口；原始盘口方向转换为本项目的主队让球为正。"""
    if not isinstance(payload, dict):
        raise ValueError('hkjc response is not an object')
    if payload.get('errors'):
        raise ValueError('hkjc graphql error: ' + str(payload['errors'][0].get('message')))
    raw_matches = ((payload.get('data') or {}).get('matches') or [])
    converted = []
    for raw in raw_matches:
        if not isinstance(raw, dict):
            continue
        pools = {
            str(pool.get('oddsType') or '').upper(): pool
            for pool in (raw.get('foPools') or []) if isinstance(pool, dict)
        }
        had_line, hdc_line, hil_line = (
            _main_line(pools.get(code) or {}) for code in ('HAD', 'HDC', 'HIL')
        )
        had_odds = _combination_odds(had_line)
        hdc_odds = _combination_odds(hdc_line)
        hil_odds = _combination_odds(hil_line)

        had = ({'胜': had_odds['H'], '平': had_odds['D'], '负': had_odds['A']}
               if all(key in had_odds for key in ('H', 'D', 'A')) else None)
        raw_handicap = _line_value((hdc_line or {}).get('condition'))
        # HKJC: 负数表示主队让；本项目：正数表示主队让。
        asian = ({
            'handicap': -raw_handicap,
            'home_odds': hdc_odds['H'],
            'away_odds': hdc_odds['A'],
        } if raw_handicap is not None and all(key in hdc_odds for key in ('H', 'A'))
            else None)
        total_line = _line_value((hil_line or {}).get('condition'))
        total = ({
            'line': total_line,
            'over_odds': hil_odds['H'],
            'under_odds': hil_odds['L'],
        } if total_line is not None and all(key in hil_odds for key in ('H', 'L'))
            else None)
        if not (had or asian or total):
            continue

        home, away = raw.get('homeTeam') or {}, raw.get('awayTeam') or {}
        match_date = str(raw.get('matchDate') or '')[:10]
        kick_off = str(raw.get('kickOffTime') or '')
        clock = kick_off[11:16] if len(kick_off) >= 16 else kick_off[:5]
        updates = [
            str(pools[code].get('updateAt') or '')
            for code in ('HAD', 'HDC', 'HIL') if code in pools
        ]
        converted.append({
            'hkjc_id': str(raw.get('id') or ''),
            'date': match_date,
            'clock': clock,
            'home_en': str(home.get('name_en') or ''),
            'away_en': str(away.get('name_en') or ''),
            'home_ch': str(home.get('name_ch') or ''),
            'away_ch': str(away.get('name_ch') or ''),
            'had_odds': had,
            'asian_current': asian,
            'total_current': total,
            'updated_at': max(updates, default=str(raw.get('updateAt') or '')),
        })
    return converted


def _clean(value) -> str:
    return re.sub(r'[^0-9a-z\u4e00-\u9fff]', '', str(value or '').lower())


def _subsequence_ratio(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    cursor = 0
    for char in haystack:
        if cursor < len(needle) and needle[cursor] == char:
            cursor += 1
    return cursor / len(needle)


def _team_similarity(code, short_name, full_name, english, chinese) -> float:
    code = _clean(code)
    english = _clean(english)
    names = [_clean(short_name), _clean(full_name)]
    chinese = _clean(chinese)
    chinese_score = max((SequenceMatcher(None, name, chinese).ratio()
                         for name in names if name and chinese), default=0.0)
    code_score = 0.0
    if code and english:
        if english.startswith(code):
            code_score = 1.0
        else:
            code_score = max(
                SequenceMatcher(None, code, english[:max(3, len(code))]).ratio(),
                _subsequence_ratio(code, english),
            )
    return max(chinese_score, code_score)


def _normalised_probabilities(odds: Dict, labels: Iterable[str]) -> Optional[List[float]]:
    try:
        inverse = [1.0 / float(odds[label]) for label in labels]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    total = sum(inverse)
    return [value / total for value in inverse] if total > 0 else None


def _match_score(sporttery: Dict, hkjc: Dict) -> float:
    home_score = _team_similarity(
        sporttery.get('home_code'), sporttery.get('home'), sporttery.get('home_full'),
        hkjc.get('home_en'), hkjc.get('home_ch'),
    )
    away_score = _team_similarity(
        sporttery.get('away_code'), sporttery.get('away'), sporttery.get('away_full'),
        hkjc.get('away_en'), hkjc.get('away_ch'),
    )
    team_score = (home_score + away_score) / 2.0
    left = _normalised_probabilities(
        sporttery.get('lottery_spf_odds') or {}, ('胜', '平', '负')
    )
    right = _normalised_probabilities(hkjc.get('had_odds') or {}, ('胜', '平', '负'))
    if left and right:
        distance = sum(abs(a - b) for a, b in zip(left, right))
        odds_score = max(0.0, 1.0 - distance / 0.45)
        return 0.72 * team_score + 0.28 * odds_score
    return team_score


def enrich_with_hkjc_markets(matches: List[Dict], hkjc_matches: List[Dict]) -> List[Dict]:
    """以一对一高置信匹配把 HKJC 主盘口挂到竞彩网比赛上。"""
    edges = []
    for sporttery in matches:
        date = str(sporttery.get('date') or '')[:10]
        clock = str(sporttery.get('time') or '')[-5:]
        for hkjc in hkjc_matches:
            if date != hkjc.get('date') or clock != hkjc.get('clock'):
                continue
            score = _match_score(sporttery, hkjc)
            if score >= 0.44:
                edges.append((score, sporttery, hkjc))

    used_sporttery, used_hkjc = set(), set()
    for score, sporttery, hkjc in sorted(edges, key=lambda item: -item[0]):
        sporttery_key = id(sporttery)
        hkjc_key = hkjc.get('hkjc_id')
        if sporttery_key in used_sporttery or hkjc_key in used_hkjc:
            continue
        used_sporttery.add(sporttery_key)
        used_hkjc.add(hkjc_key)
        asian = hkjc.get('asian_current')
        total = hkjc.get('total_current')
        sporttery.update({
            'hkjc_id': hkjc_key,
            'hkjc_match_score': round(score, 4),
            'hkjc_had_odds': hkjc.get('had_odds'),
            'hkjc_updated_at': hkjc.get('updated_at'),
            'asian_source': 'hkjc' if asian else 'unavailable',
            'asian_offer_matched': bool(asian),
            'asian_current': asian,
            'total_source': 'hkjc' if total else 'unavailable',
            'total_offer_matched': bool(total),
            'total_current': total,
        })

    for match in matches:
        if id(match) not in used_sporttery:
            match.update({
                'asian_source': 'unavailable', 'asian_offer_matched': False,
                'total_source': 'unavailable', 'total_offer_matched': False,
            })
    return matches
