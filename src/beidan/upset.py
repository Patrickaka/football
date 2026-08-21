# -*- coding: utf-8 -*-
"""北单爆冷识别"""

import sys
import math
import re
from collections import defaultdict
import time
import json
import urllib.request
import urllib.error
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('beidan')

from .quality import (
    UPSET_CONFIDENT_FAV_MIN, UPSET_CONFIDENT_GAP_MIN, UPSET_HIGH_FAV_MAX, UPSET_HIGH_GAP_MAX, UPSET_HIGH_MASS_MIN, UPSET_MED_FAV_MAX, UPSET_MED_GAP_MAX, UPSET_MED_MASS_MIN,
)

def _result_from_score(key):
    """从比分键（(h,a) 元组或 'h-a' 字符串）推导赛果 胜/平/负。"""
    try:
        if isinstance(key, (tuple, list)):
            h, a = int(key[0]), int(key[1])
        else:
            h, a = (int(x) for x in str(key).replace(':', '-').split('-')[:2])
    except (ValueError, TypeError, IndexError):
        return None
    return '胜' if h > a else ('负' if h < a else '平')


def _fmt_score(key):
    if isinstance(key, (tuple, list)):
        return f"{int(key[0])}-{int(key[1])}"
    return str(key)


def assess_upset_risk(probs_1x2):
    """基于 1X2 概率评估爆冷风险。

    返回 level(high/medium/low)、favorite(热门赛果)、favorite_prob、
    upset_prob(非热门总概率)、gap(热门与次热门差)、alert(是否预警)。
    """
    if not probs_1x2:
        return {'level': 'low', 'alert': False, 'favorite': None,
                'favorite_prob': 0.0, 'upset_prob': 0.0, 'gap': 0.0}
    probs = {k: float(v) for k, v in probs_1x2.items() if v is not None}
    if not probs:
        return {'level': 'low', 'alert': False, 'favorite': None,
                'favorite_prob': 0.0, 'upset_prob': 0.0, 'gap': 0.0}
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    favorite, fav_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = fav_p - second_p
    upset_mass = 1.0 - fav_p

    if fav_p < UPSET_HIGH_FAV_MAX and gap <= UPSET_HIGH_GAP_MAX and upset_mass >= UPSET_HIGH_MASS_MIN:
        level, label, alert = 'high', '爆冷高风险', True
    elif fav_p < UPSET_MED_FAV_MAX and gap <= UPSET_MED_GAP_MAX and upset_mass >= UPSET_MED_MASS_MIN:
        level, label, alert = 'medium', '爆冷预警', True
    else:
        level, label, alert = 'low', '热门稳健', False

    # 反向：稳胆档（强热门 + 差距悬殊 → 真实冷门率约 30%，两半均稳）。
    # 仅细化未预警场次的正面信号，不改变 level/alert 契约（非破坏性）。
    confident = (not alert and fav_p >= UPSET_CONFIDENT_FAV_MIN
                 and gap >= UPSET_CONFIDENT_GAP_MIN)
    if confident:
        label = '热门稳胆'

    reverse_labels = {
        '胜': [('平', '防冷平'), ('负', '客胜冷门')],
        '负': [('平', '防冷平'), ('胜', '主胜冷门')],
        '平': [('胜', '主胜反向'), ('负', '客胜反向')],
    }.get(favorite, [])
    defensive_selections = [
        {
            'result': result_label,
            'type': selection_type,
            'probability': round(probs.get(result_label, 0.0), 6),
        }
        for result_label, selection_type in reverse_labels
    ] if alert else []
    defensive_selections.sort(key=lambda item: -item['probability'])
    signals = []
    if alert:
        signals.append('热门强度不足' if fav_p >= UPSET_HIGH_FAV_MAX else '弱热门')
        if gap <= UPSET_HIGH_GAP_MAX:
            signals.append('三项概率胶着')
        if upset_mass >= UPSET_HIGH_MASS_MIN:
            signals.append('非热门合计概率偏高')

    return {
        'level': level,
        'label': label,
        'alert': alert,
        'confident': confident,
        'favorite': favorite,
        'favorite_prob': round(fav_p, 6),
        'upset_prob': round(upset_mass, 6),
        'gap': round(gap, 6),
        'signals': signals,
        'defensive_selections': defensive_selections,
        'recommended_cover': '/'.join(
            item['result'] for item in defensive_selections
        ) if defensive_selections else None,
    }


def pick_upset_scores(score_matrix, favorite_result, top_n=2):
    """从比分分布中挑出"与热门赛果相反方向"上概率最高的若干爆冷比分。

    热门主胜 → 候选取 平/负 比分；热门主负 → 取 胜/平；热门平局 → 取 胜/负。
    回测：预警且真爆冷时，方向命中率约 50%、精确比分命中率约 22%（远高于普通比分 12%）。
    """
    if not score_matrix or not favorite_result:
        return []
    if favorite_result == '胜':
        allow = {'平', '负'}
    elif favorite_result == '负':
        allow = {'胜', '平'}
    else:
        allow = {'胜', '负'}
    cands = []
    for key, prob in sorted(score_matrix.items(), key=lambda x: -x[1]):
        res = _result_from_score(key)
        if res in allow:
            cands.append({
                'score': _fmt_score(key),
                'result': res,
                'probability': round(float(prob), 6),
            })
        if len(cands) >= top_n:
            break
    return cands


def _score_result_label(score):
    if isinstance(score, dict):
        home_goals = score.get('home_goals')
        away_goals = score.get('away_goals')
        if home_goals is None or away_goals is None:
            score_text = str(score.get('score', ''))
        else:
            try:
                home_goals = int(home_goals)
                away_goals = int(away_goals)
                return '胜' if home_goals > away_goals else ('负' if home_goals < away_goals else '平')
            except (TypeError, ValueError):
                score_text = str(score.get('score', ''))
    else:
        score_text = str(score[0]) if score else ''

    if '-' not in score_text:
        return None
    left, right = score_text.split('-', 1)
    try:
        home_goals = int(left)
        away_goals = int(right)
    except ValueError:
        return None
    return '胜' if home_goals > away_goals else ('负' if home_goals < away_goals else '平')


def assess_score_consistency(scores, prediction):
    if not scores or not prediction:
        return {'available': False, 'conflict': False}

    weights = {'胜': 0.0, '平': 0.0, '负': 0.0}
    top_result = None
    total_weight = 0.0
    for idx, score in enumerate(scores[:3]):
        result = _score_result_label(score)
        if not result:
            continue
        if top_result is None:
            top_result = result
        if isinstance(score, dict):
            prob = score.get('probability')
        else:
            prob = score[1] if len(score) > 1 else None
        weight = float(prob) if prob is not None else max(0.1, 1.0 - idx * 0.25)
        weights[result] += weight
        total_weight += weight

    if total_weight <= 0:
        return {'available': False, 'conflict': False}

    agreement = weights.get(prediction, 0.0) / total_weight
    conflict = top_result != prediction and agreement < 0.45
    return {
        'available': True,
        'conflict': conflict,
        'top_score_result': top_result,
        'agreement': round(agreement, 6),
        'result_weights': {k: round(v / total_weight, 6) for k, v in weights.items()},
    }


