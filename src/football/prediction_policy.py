#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conservative league/market policies for football score distributions.
"""

from typing import Dict, Tuple


LOW_GOAL_LEAGUE_HINTS = ('意甲', '葡超', '希腊', '阿甲', '巴乙', '日乙')
HIGH_VARIANCE_HINTS = ('荷甲', '挪超', '瑞典', '巴甲', '美职')
CUP_HINTS = ('杯', '欧冠', '欧联', '世俱', '世界杯', '欧洲杯')


def _contains_any(text: str, hints: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint.lower() in lowered for hint in hints)


def _league_text(league=None, league_profile=None) -> str:
    if league:
        return str(league)
    if isinstance(league_profile, dict):
        return str(league_profile.get('name') or league_profile.get('league') or '')
    return ''


def get_total_bucket(total_line) -> str:
    if total_line is None:
        return 'unknown'
    try:
        line = float(total_line)
    except (TypeError, ValueError):
        return 'unknown'
    if line <= 2.25:
        return 'low'
    if line >= 3.0:
        return 'high'
    return 'normal'


def get_handicap_bucket(handicap) -> str:
    if handicap is None:
        return 'unknown'
    try:
        depth = abs(float(handicap))
    except (TypeError, ValueError):
        return 'unknown'
    if depth <= 0.25:
        return 'level'
    if depth < 1.0:
        return 'mid'
    return 'deep'


def get_prediction_policy(league=None, total_line=None, handicap=None, league_profile=None) -> Dict:
    league_name = _league_text(league, league_profile)
    total_bucket = get_total_bucket(total_line)
    handicap_bucket = get_handicap_bucket(handicap)

    static_market_cap = 0.15
    change_market_cap = 0.15
    half_full_real_weight = 0.25
    half_full_market_cap = 0.10
    draw_bias = 1.0
    low_score_bias = 1.0
    high_score_bias = 1.0

    if total_bucket == 'low':
        low_score_bias *= 1.08
        high_score_bias *= 0.88
        draw_bias *= 1.04
        half_full_real_weight = 0.28
    elif total_bucket == 'high':
        high_score_bias *= 1.08
        low_score_bias *= 0.92
        draw_bias *= 0.96
        half_full_real_weight = 0.22

    if handicap_bucket == 'level':
        draw_bias *= 1.05
        change_market_cap = 0.12
    elif handicap_bucket == 'deep':
        draw_bias *= 0.92
        static_market_cap = 0.18
        change_market_cap = 0.18

    if _contains_any(league_name, LOW_GOAL_LEAGUE_HINTS):
        low_score_bias *= 1.06
        high_score_bias *= 0.90
        draw_bias *= 1.03
    elif _contains_any(league_name, HIGH_VARIANCE_HINTS):
        high_score_bias *= 1.06
        draw_bias *= 0.97

    if _contains_any(league_name, CUP_HINTS):
        static_market_cap = min(static_market_cap, 0.12)
        change_market_cap = min(change_market_cap, 0.12)
        half_full_real_weight = min(half_full_real_weight, 0.20)

    return {
        'league': league_name,
        'total_bucket': total_bucket,
        'handicap_bucket': handicap_bucket,
        'static_market_cap': static_market_cap,
        'change_market_cap': change_market_cap,
        'half_full_real_weight': half_full_real_weight,
        'half_full_market_cap': half_full_market_cap,
        'draw_bias': draw_bias,
        'low_score_bias': low_score_bias,
        'high_score_bias': high_score_bias,
    }


def normalize_score_matrix(matrix: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = sum(matrix.values())
    if total <= 0:
        return matrix
    return {score: prob / total for score, prob in matrix.items()}


def apply_score_distribution_policy(matrix: Dict[Tuple[int, int], float],
                                    asian: Dict = None,
                                    total: Dict = None,
                                    league_profile: Dict = None,
                                    league: str = None) -> Tuple[Dict[Tuple[int, int], float], Dict]:
    """Apply small, auditable bias corrections to common score-distribution skews."""
    if not matrix:
        return matrix, {'applied': False}

    asian = asian or {}
    total = total or {}
    total_line = total.get('close_line', total.get('line'))
    handicap = asian.get('handicap')
    policy = get_prediction_policy(league=league, total_line=total_line, handicap=handicap, league_profile=league_profile)
    adjusted = {}

    favor = asian.get('favor', 'home')
    try:
        handicap_depth = abs(float(handicap or 0))
    except (TypeError, ValueError):
        handicap_depth = 0.0

    for (h, a), prob in matrix.items():
        goals = h + a
        factor = 1.0

        if goals <= 2:
            factor *= policy['low_score_bias']
        elif goals >= 4:
            factor *= policy['high_score_bias']

        if h == a:
            factor *= policy['draw_bias']

        if (h, a) == (1, 1) and prob >= 0.14:
            factor *= 0.88
        elif (h, a) == (0, 0) and policy['total_bucket'] == 'high':
            factor *= 0.78

        if handicap_depth >= 1.0:
            diff = h - a
            upset = (favor == 'home' and diff < 0) or (favor == 'away' and diff > 0)
            favorite_margin = (favor == 'home' and diff > 0) or (favor == 'away' and diff < 0)
            if upset:
                factor *= 0.62
            elif h == a:
                factor *= 0.88
            elif favorite_margin and abs(diff) <= 2:
                factor *= 1.04

        adjusted[(h, a)] = prob * factor

    adjusted = normalize_score_matrix(adjusted)
    return adjusted, {
        'applied': True,
        'policy': policy,
    }
