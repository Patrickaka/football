# -*- coding: utf-8 -*-
"""北单市场分析：亚盘/大小球/比分/进球趋势与联合市场状态"""

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

from .modeling import (
    _asian_over_profit, _parse_total_line_value, _to_euro_odds, parse_beidan_handicap, rqspf_probs_from_score_probs,
)





def build_beidan_joint_market_state(asian_data=None, goals_data=None):
    """Build one direction/tempo state from Beidan handicap and O/U histories."""
    asian_history = (asian_data or {}).get('history') or []
    goals_history = (goals_data or {}).get('history') or []
    asian_trend = analyze_asian_trend(asian_history)
    goals_trend = analyze_goals_trend(goals_history)

    direction_map = {
        'home_backing': 1.0, 'away_laying': 0.65,
        'away_backing': -1.0, 'home_laying': -0.65,
    }
    direction = direction_map.get(asian_trend.get('direction'), 0.0)
    try:
        direction *= min(1.0, max(0.25, float(asian_trend.get('strength', 0.0)) / 0.12))
    except (TypeError, ValueError):
        direction = 0.0

    handicap_signal = 0.0
    if len(asian_history) >= 2:
        try:
            handicap_signal = max(-1.0, min(1.0,
                (float(asian_history[-1].get('handicap')) -
                 float(asian_history[0].get('handicap'))) / 0.5
            ))
        except (TypeError, ValueError):
            handicap_signal = 0.0
    direction = 0.65 * direction + 0.35 * handicap_signal

    tempo_map = {
        'over_backing': 1.0, 'under_laying': 0.65,
        'under_backing': -1.0, 'over_laying': -0.65,
    }
    water_tempo = tempo_map.get(goals_trend.get('direction'), 0.0)
    line_signal = 0.0
    if len(goals_history) >= 2:
        try:
            first_line = float(re.search(r'[\d.]+', str(goals_history[0].get('line'))).group())
            last_line = float(re.search(r'[\d.]+', str(goals_history[-1].get('line'))).group())
            line_signal = max(-1.0, min(1.0, (last_line - first_line) / 0.5))
        except (AttributeError, TypeError, ValueError):
            line_signal = 0.0
    water_line_conflict = water_tempo * line_signal < -0.12
    tempo = 0.55 * water_tempo + 0.45 * line_signal

    if water_line_conflict:
        tempo *= 0.40
    return {
        'direction_signal': max(-1.0, min(1.0, direction)),
        'tempo_signal': max(-1.0, min(1.0, tempo)),
        'handicap_signal': handicap_signal,
        'line_signal': line_signal,
        'asian_trend': asian_trend,
        'goals_trend': goals_trend,
        'conflict': water_line_conflict,
        'agreement_factor': 0.40 if water_line_conflict else 1.0,
    }


def apply_beidan_joint_market_state(score_probs, asian_data=None, goals_data=None):
    """Fit the shared score matrix to fair Asian and O/U closing prices.

    Open-to-close movement controls reliability/conflict; it is not counted as
    a second independent prediction after the closing price has already
    absorbed that information.
    """
    if not score_probs:
        return score_probs, {'applied': False, 'reason': 'empty_distribution'}
    state = build_beidan_joint_market_state(asian_data, goals_data)
    matrix = {}
    for score, probability in score_probs.items():
        try:
            parsed = int(score[0]), int(score[1])
            matrix[parsed] = matrix.get(parsed, 0.0) + max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
    total = sum(matrix.values())
    if total <= 0:
        return score_probs, {**state, 'applied': False, 'reason': 'zero_raw_mass'}
    matrix = {score: probability / total for score, probability in matrix.items()}
    before = dict(matrix)

    def constrain(source, feature, strength):
        values = {score: feature(score) for score in source}
        buckets = defaultdict(float)
        for score, probability in source.items():
            buckets[round(values[score], 12)] += probability

        def expectation(theta):
            weighted = [
                (value, probability * math.exp(max(-20.0, min(20.0, theta * value))))
                for value, probability in buckets.items()
            ]
            denominator = sum(probability for _, probability in weighted)
            return sum(value * probability for value, probability in weighted) / denominator

        lo, hi = -12.0, 12.0
        if expectation(lo) > 0 or expectation(hi) < 0:
            return source, {'applied': False, 'reason': 'target_outside_support'}
        expected_before = expectation(0.0)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if expectation(mid) < 0:
                lo = mid
            else:
                hi = mid
        theta = strength * (lo + hi) / 2.0
        adjusted = {
            score: probability * math.exp(max(-20.0, min(20.0, theta * values[score])))
            for score, probability in source.items()
        }
        denominator = sum(adjusted.values())
        adjusted = {score: probability / denominator for score, probability in adjusted.items()}
        return adjusted, {
            'applied': True,
            'theta': round(theta, 5),
            'fair_profit_before': round(expected_before, 5),
            'fair_profit_after': round(
                sum(adjusted[score] * values[score] for score in adjusted), 5
            ),
        }

    asian_market = None
    for entry in reversed((asian_data or {}).get('history') or []):
        if entry.get('home_odds') and entry.get('away_odds'):
            asian_market = entry
            break
    total_line, over_odds, under_odds = _latest_ou_market(goals_data)
    reliability = 0.40 if state.get('conflict') else 1.0
    pass_strength = 0.35 * reliability / 3.0
    asian_meta = {'applied': False, 'reason': 'missing_price_or_line'}
    total_meta = {'applied': False, 'reason': 'missing_price_or_line'}

    for _ in range(3):
        if asian_market:
            home_odds = _to_euro_odds(asian_market.get('home_odds'))
            away_odds = _to_euro_odds(asian_market.get('away_odds'))
            if home_odds and away_odds:
                inverse_home, inverse_away = 1.0 / home_odds, 1.0 / away_odds
                fair_home_odds = (inverse_home + inverse_away) / inverse_home
                # Some fallback feeds omit the line while retaining two-way
                # prices. Treat those rare records as a PK market instead of
                # discarding the only directional price evidence.
                line = float(asian_market.get('handicap') or 0.0)
                matrix, asian_meta = constrain(
                    matrix,
                    lambda score, line=line, odds=fair_home_odds: _asian_over_profit(
                        score[0] - score[1], line, odds
                    ),
                    pass_strength,
                )
        over_decimal, under_decimal = _to_euro_odds(over_odds), _to_euro_odds(under_odds)
        if over_decimal and under_decimal:
            inverse_over, inverse_under = 1.0 / over_decimal, 1.0 / under_decimal
            fair_over_odds = (inverse_over + inverse_under) / inverse_over
            matrix, total_meta = constrain(
                matrix,
                lambda score, line=total_line, odds=fair_over_odds: _asian_over_profit(
                    score[0] + score[1], line, odds
                ),
                pass_strength,
            )

    if not asian_meta.get('applied') and not total_meta.get('applied'):
        return score_probs, {**state, 'applied': False, 'reason': 'missing_closing_market_prices'}

    adjusted = matrix
    state.update({
        'applied': True,
        'method': 'maximum_entropy_fair_price_constraint',
        'constraint_strength': round(0.35 * reliability, 3),
        'asian_constraint': asian_meta,
        'total_constraint': total_meta,
        'expected_goals_before': sum(sum(score) * p for score, p in before.items()),
        'expected_goals_after': sum(sum(score) * p for score, p in adjusted.items()),
        'home_win_before': sum(p for (h, a), p in before.items() if h > a),
        'home_win_after': sum(p for (h, a), p in adjusted.items() if h > a),
    })
    return adjusted, state


def build_water_market_prediction(spf_result, handicap):
    """从欧赔基准 + 亚盘水位修正后的同一比分矩阵派生各体彩市场。"""
    raw_scores = (spf_result or {}).get('score_probs') or []
    score_probs = {}
    for item in raw_scores:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                home, away, probability = int(item[0]), int(item[1]), float(item[2])
            else:
                continue
            if probability > 0:
                score_probs[(home, away)] = score_probs.get((home, away), 0.0) + probability
        except (TypeError, ValueError):
            continue
    total_mass = sum(score_probs.values())
    if total_mass <= 0:
        return {'available': False, 'reason': 'score_distribution_unavailable'}
    score_probs = {score: probability / total_mass for score, probability in score_probs.items()}

    spf = {'胜': 0.0, '平': 0.0, '负': 0.0}
    goals = {}
    for (home, away), probability in score_probs.items():
        spf['胜' if home > away else '负' if home < away else '平'] += probability
        goal_key = '7+' if home + away >= 7 else str(home + away)
        goals[goal_key] = goals.get(goal_key, 0.0) + probability

    handicap_value = parse_beidan_handicap(handicap)
    rqspf, _ = rqspf_probs_from_score_probs(score_probs, handicap_value)
    goal_top3 = sorted(goals.items(), key=lambda item: -item[1])[:3]
    joint_state = (spf_result or {}).get('joint_market_state') or {}
    euro_probs = dict((spf_result or {}).get('raw_probabilities') or {})
    euro_prediction = max(euro_probs, key=euro_probs.get) if euro_probs else None
    combined_prediction = max(spf, key=spf.get)
    direction_signal = float(joint_state.get('direction_signal') or 0.0)
    asian_direction = (
        '主队增强' if direction_signal > 0.08 else
        '客队增强' if direction_signal < -0.08 else
        '方向稳定'
    )
    return {
        'available': True,
        'source': 'euro_asian_adjusted_shared_score_matrix',
        'asian_adjusted': bool(joint_state.get('applied')),
        'joint_market_state': joint_state,
        'evidence': {
            'euro_prediction': euro_prediction,
            'euro_probabilities': euro_probs,
            'asian_direction': asian_direction,
            'direction_signal': direction_signal,
            'conflict': bool(euro_prediction and (
                (euro_prediction == '胜' and direction_signal < -0.08) or
                (euro_prediction == '负' and direction_signal > 0.08)
            )),
        },
        'spf': {
            'prediction': combined_prediction,
            'probabilities': spf,
        },
        'rqspf': ({
            'handicap': handicap_value,
            'prediction': max(rqspf, key=rqspf.get),
            'probabilities': rqspf,
        } if rqspf else None),
        'goals': {
            'prediction': goal_top3[0][0] if goal_top3 else None,
            'top3': [[key, probability] for key, probability in goal_top3],
            'probabilities': goals,
        },
    }


def build_beidan_market_admission(section, bet_type, asian_data=None, goals_data=None):
    """Accuracy-first gate using the same-match handicap/O-U time series."""
    state = build_beidan_joint_market_state(asian_data, goals_data)
    asian_samples = len((asian_data or {}).get('history') or [])
    goals_samples = len((goals_data or {}).get('history') or [])
    signal = state['direction_signal'] if bet_type == 'rqspf' else state['tempo_signal']
    prediction = str((section or {}).get('prediction') or '')
    if bet_type == 'rqspf':
        pick_signal = 1.0 if prediction == '让胜' else (-1.0 if prediction == '让负' else 0.0)
        enough = asian_samples >= 2
    else:
        try:
            goals = 7 if prediction == '7+' else int(prediction)
        except (TypeError, ValueError):
            goals = None
        pick_signal = 1.0 if goals is not None and goals >= 3 else (-1.0 if goals is not None else 0.0)
        enough = goals_samples >= 2
    aligned = enough and abs(signal) >= 0.08 and pick_signal * signal > 0
    reason = None
    if not enough:
        reason = 'market_history_insufficient'
    elif abs(signal) < 0.08:
        reason = 'market_signal_weak'
    elif not aligned:
        reason = 'market_conflicts_with_model'
    return {
        'official': bool(aligned), 'playable': bool(aligned), 'skip_reason': reason,
        'aligned': bool(aligned), 'signal': round(signal, 4),
        'asian_samples': asian_samples, 'goals_samples': goals_samples,
        'state': state,
    }


def _beidan_market_snapshot(match):
    """Compact current snapshot for repeated same-match refreshes."""
    asian_history = ((match.get('asian') or {}).get('history') or [])
    goals_history = ((match.get('goals') or {}).get('history') or [])
    asian = asian_history[-1] if asian_history else {}
    goals = goals_history[-1] if goals_history else {}
    return {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'asian': {k: asian.get(k) for k in ('handicap', 'home_odds', 'away_odds')},
        'total': {k: goals.get(k) for k in ('line', 'over_odds', 'under_odds')},
        'spf_odds': dict((match.get('spf') or {}).get('odds') or {}),
        'rqspf_odds': dict((match.get('rqspf') or {}).get('odds') or {}),
        'rqspf_admission': (match.get('rqspf') or {}).get('market_admission'),
        'zjq_admission': (match.get('zjq') or {}).get('market_admission'),
    }














def _latest_ou_market(goals_data):
    """Return the latest total line together with over/under prices."""
    if not goals_data or not goals_data.get('history'):
        return 2.5, None, None
    for entry in reversed(goals_data['history']):
        o = entry.get('over_odds')
        u = entry.get('under_odds')
        if o and u:
            return _parse_total_line_value(entry.get('line'), default=2.5), o, u
    return 2.5, None, None


def _latest_ou_odds(goals_data):
    """Backward-compatible two-value view used by older callers/tests."""
    _, over_odds, under_odds = _latest_ou_market(goals_data)
    return over_odds, under_odds


# ─── 领域层适配（走势与因子）───
#
# 算法在 `src/domain/sports/beidan/trends.py`。迁移前这些阈值全是函数体里的
# 裸数字（0.02、0.03、0.05、0.15、1.2、0.85…），既没有名字也没有出处；
# 现在集中在这里，改一个不必再读懂整段代码。
#
# 联合市场状态与准入（`build_beidan_joint_market_state` 等）仍在本文件里，
# 属下一批。`_beidan_market_snapshot` 会读时钟，按判据 16 永远留在这一层。

from src.domain.sports.beidan import trends as _trends

# 亚盘水位的观察窗口与门槛
ASIAN_TREND_WINDOW = 5
ASIAN_MOVE_THRESHOLD = 0.02      # 超过这个幅度才动 1X2
ASIAN_DIRECTION_THRESHOLD = 0.03  # 超过这个幅度才判方向
ASIAN_ADJUST_FACTOR = 0.15        # 调整幅度上限
ASIAN_COUNTER_RATIO = 0.5         # 反方向只给一半力度

# 大小球：窗口更长、门槛更松——它的水位波动本来就比亚盘大
GOALS_TREND_WINDOW = 10
GOALS_DIRECTION_THRESHOLD = 0.05
GOALS_ADJUST_WINDOW = 5
GOALS_BUCKET_LIFT = 1.2
GOALS_BUCKET_CUT = 0.85

# 总进球因子：偏大球与偏小球不对称——大球贴水天然偏低，
# 用对称门槛会把常态误判成偏小球
GOALS_FACTOR_OVER = 1.2
GOALS_FACTOR_UNDER = 0.85
GOALS_FACTOR_UNDER_MARGIN = 0.5

# 亚盘水位之和 → 总进球因子的分档
ASIAN_GOAL_FACTOR_TIERS = ((3.6, 1.3), (4.0, 1.15), (4.4, 1.0), (4.8, 0.9))
ASIAN_GOAL_FACTOR_FLOOR = 0.75

# 比分盘
CS_TREND_WINDOW = 10
CS_MOVE_THRESHOLD = 0.1
CS_HOT_KEPT = 5
CS_BLEND_WINDOW = 5
CS_NEW_SCORE_DISCOUNT = 0.5   # 盘口有、模型没算到的比分打这个折
CS_SCORES_KEPT = 3


def adjust_probs_by_asian(home_win_prob, draw_prob, away_win_prob, asian_history):
    return _trends.adjust_probs_by_asian(
        home_win_prob, draw_prob, away_win_prob, asian_history,
        window=ASIAN_TREND_WINDOW, move_threshold=ASIAN_MOVE_THRESHOLD,
        factor=ASIAN_ADJUST_FACTOR, counter_ratio=ASIAN_COUNTER_RATIO)


def analyze_asian_trend(asian_history):
    return _trends.analyze_asian(
        asian_history, window=ASIAN_TREND_WINDOW,
        direction_threshold=ASIAN_DIRECTION_THRESHOLD)


def analyze_goals_trend(goals_history):
    return _trends.analyze_goals(
        goals_history, window=GOALS_TREND_WINDOW,
        direction_threshold=GOALS_DIRECTION_THRESHOLD)


def analyze_cs_trend(cs_history):
    return _trends.analyze_correct_score(
        cs_history, window=CS_TREND_WINDOW,
        move_threshold=CS_MOVE_THRESHOLD, kept=CS_HOT_KEPT)


def calculate_goals_factor(goals_history):
    return _trends.goals_factor(
        goals_history, window=GOALS_TREND_WINDOW,
        lean_over=GOALS_FACTOR_OVER, lean_under=GOALS_FACTOR_UNDER,
        under_margin=GOALS_FACTOR_UNDER_MARGIN)


def calculate_asian_goal_factor(asian_history):
    return _trends.asian_goal_factor(
        asian_history, window=GOALS_TREND_WINDOW,
        tiers=ASIAN_GOAL_FACTOR_TIERS, floor=ASIAN_GOAL_FACTOR_FLOOR)


def adjust_zjq_by_goals(zjq_probs, goals_history):
    """按大小球走势调整总进球分桶。

    **迁移前这个函数原地改写入参**，调用方拿到的和传进去的是同一个对象。
    领域层返回新字典；这里把结果写回入参再返回，保住旧语义——
    有调用方依赖「传进去的那份也变了」，改掉是另一件事。
    """
    adjusted = _trends.adjust_goal_buckets(
        zjq_probs, goals_history, window=GOALS_ADJUST_WINDOW,
        trend_threshold=GOALS_DIRECTION_THRESHOLD,
        lift=GOALS_BUCKET_LIFT, cut=GOALS_BUCKET_CUT)
    zjq_probs.update(adjusted)
    return zjq_probs


def enhance_scores_with_cs(score_prediction, cs_history):
    """用比分盘的隐含概率修正模型的比分推荐。

    同样保住「原地改写」的旧语义：领域层算出新列表，这里写回 `top3`。
    """
    blended = _trends.blend_scores_with_market(
        score_prediction.get('top3') or [], cs_history,
        window=CS_BLEND_WINDOW, new_score_discount=CS_NEW_SCORE_DISCOUNT,
        kept=CS_SCORES_KEPT)
    if blended:
        score_prediction['top3'] = blended
    return score_prediction
