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


# ─── 领域层适配（联合市场状态）───
#
# 算法在 `src/domain/sports/beidan/market_state.py`：走势合成、指数倾斜、
# 公平赔率换算。这里只喂权重与门槛——迁移前它们同样是函数体里的裸数字。

from src.domain.sports.beidan import market_state as _state

# 走势合成：强度的归一化除数与下限
JOINT_STRENGTH_DIVISOR = 0.12
JOINT_STRENGTH_FLOOR = 0.25
# 盘口线移动归一化的除数：半球算满格
JOINT_HANDICAP_DIVISOR = 0.5
JOINT_LINE_DIVISOR = 0.5
# 水位与线的融合权重（线的权重略低——它是离散的，单次变动信息量更少）
JOINT_DIRECTION_BLEND = 0.65
JOINT_TEMPO_BLEND = 0.55
# 水位与线指向相反时判为冲突，并把节奏信号衰减到这个比例
JOINT_CONFLICT_THRESHOLD = -0.12
JOINT_CONFLICT_DAMPING = 0.40

# 约束强度：总量 0.35，分三轮施加，冲突时按 damping 打折。
# **分多轮是有意的**——亚盘与大小球两个约束会互相拉扯，
# 一次到位会让后施加的那个把前一个顶掉
CONSTRAINT_STRENGTH = 0.35
CONSTRAINT_PASSES = 3


def build_beidan_joint_market_state(asian_data=None, goals_data=None):
    """由亚盘与大小球的走势合成方向与节奏信号。"""
    asian_history = (asian_data or {}).get('history') or []
    goals_history = (goals_data or {}).get('history') or []
    return _state.joint_state(
        analyze_asian_trend(asian_history), analyze_goals_trend(goals_history),
        asian_history, goals_history,
        strength_divisor=JOINT_STRENGTH_DIVISOR,
        strength_floor=JOINT_STRENGTH_FLOOR,
        handicap_divisor=JOINT_HANDICAP_DIVISOR,
        line_divisor=JOINT_LINE_DIVISOR,
        direction_blend=JOINT_DIRECTION_BLEND,
        tempo_blend=JOINT_TEMPO_BLEND,
        conflict_threshold=JOINT_CONFLICT_THRESHOLD,
        conflict_damping=JOINT_CONFLICT_DAMPING)


def apply_beidan_joint_market_state(score_probs, asian_data=None, goals_data=None):
    """把比分矩阵拟合到亚盘与大小球的公平价上。

    开盘到临场的走势**不当作第二个独立预测**——终盘价已经吸收了那段信息，
    再叠一次等于把同一条证据用两遍；走势只用来判断可靠性与冲突。
    """
    if not score_probs:
        return score_probs, {'applied': False, 'reason': 'empty_distribution'}

    state = build_beidan_joint_market_state(asian_data, goals_data)
    matrix = _state.normalise_matrix(score_probs)
    if matrix is None:
        return score_probs, {**state, 'applied': False, 'reason': 'zero_raw_mass'}
    before = dict(matrix)

    asian_market = None
    for entry in reversed((asian_data or {}).get('history') or []):
        if entry.get('home_odds') and entry.get('away_odds'):
            asian_market = entry
            break
    total_line, over_odds, under_odds = _latest_ou_market(goals_data)

    reliability = JOINT_CONFLICT_DAMPING if state.get('conflict') else 1.0
    pass_strength = CONSTRAINT_STRENGTH * reliability / CONSTRAINT_PASSES
    missing = {'applied': False, 'reason': 'missing_price_or_line'}
    asian_meta, total_meta = dict(missing), dict(missing)

    for _ in range(CONSTRAINT_PASSES):
        if asian_market:
            fair_home = _state.fair_odds(_to_euro_odds(asian_market.get('home_odds')),
                                         _to_euro_odds(asian_market.get('away_odds')))
            if fair_home:
                # 少数备用源只给两侧报价而没有盘口线。**按平手盘处理**
                # 而不是丢弃——那是这场唯一的方向性价格证据
                line = float(asian_market.get('handicap') or 0.0)
                matrix, asian_meta = _state.tilt_to_fair_price(
                    matrix,
                    lambda score, line=line, odds=fair_home: _asian_over_profit(
                        score[0] - score[1], line, odds),
                    pass_strength)
        fair_over = _state.fair_odds(_to_euro_odds(over_odds), _to_euro_odds(under_odds))
        if fair_over:
            matrix, total_meta = _state.tilt_to_fair_price(
                matrix,
                lambda score, line=total_line, odds=fair_over: _asian_over_profit(
                    score[0] + score[1], line, odds),
                pass_strength)

    if not asian_meta.get('applied') and not total_meta.get('applied'):
        return score_probs, {**state, 'applied': False,
                             'reason': 'missing_closing_market_prices'}

    state.update({
        'applied': True,
        'method': 'maximum_entropy_fair_price_constraint',
        'constraint_strength': round(CONSTRAINT_STRENGTH * reliability, 3),
        'asian_constraint': asian_meta,
        'total_constraint': total_meta,
        **_state.summarise_shift(before, matrix),
    })
    return matrix, state
