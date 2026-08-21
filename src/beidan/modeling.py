# -*- coding: utf-8 -*-
"""北单概率建模：泊松/DC矩阵/λ推导/盘口解析/比分锚定"""

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

from .config import (
    LEAGUE_PROFILES, MAX_GOALS, SCORE_SPLIT,
)

def poisson_pmf(k, mu):
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)


def euro_implied_lambdas(p_home, p_draw, p_away, target_total):
    supremacy = (p_home - p_away) / (p_home + p_draw + p_away + 1e-9)
    lam_home = target_total * (0.5 + supremacy * SCORE_SPLIT)
    lam_away = target_total * (0.5 - supremacy * SCORE_SPLIT)
    return max(0.01, lam_home), max(0.01, lam_away)


def calibrate_draw_probability(p_home, p_draw, p_away, handicap,
                               home_draw_rate=0.25, away_draw_rate=0.25,
                               league_draw_rate=0.25):
    ref_draw_rate = (home_draw_rate + away_draw_rate + league_draw_rate) / 3
    if handicap is not None:
        try:
            if isinstance(handicap, str):
                handicap = float(handicap.replace('(', '').replace(')', ''))
            if abs(handicap) >= 1.0:
                ref_draw_rate *= 0.8
            elif abs(handicap) >= 0.5:
                ref_draw_rate *= 0.95
        except (ValueError, TypeError):
            pass
    
    total = p_home + p_draw + p_away + 1e-9
    current_draw_rate = p_draw / total
    
    if current_draw_rate < ref_draw_rate * 0.8:
        p_draw *= 1.2
    elif current_draw_rate > ref_draw_rate * 1.3:
        p_draw *= 0.9
    
    total_new = p_home + p_draw + p_away + 1e-9
    return p_home / total_new, p_draw / total_new, p_away / total_new


def predict_scores_by_poisson(home_prob, draw_prob, away_prob, league='', handicap=0,
                              total_over_odds=None, total_under_odds=None, use_dc=True,
                              total_line=2.5):
    league_profile = LEAGUE_PROFILES.get(league, {'avg_goals': 2.6, 'draw_rate': 0.27})
    avg_goals = league_profile['avg_goals']
    
    p_home, p_draw, p_away = calibrate_draw_probability(
        home_prob, draw_prob, away_prob, handicap,
        league_draw_rate=league_profile['draw_rate']
    )
    
    # 目标总进球：优先用大小球隐含总进球，否则退化为联赛均值
    target_total = match_target_total(
        league=league,
        total_over_odds=total_over_odds,
        total_under_odds=total_under_odds,
        total_line=total_line,
    )
    
    lam_home, lam_away = match_lambdas(p_home, p_draw, p_away, target_total)
    
    if use_dc:
        score_probs = build_dixon_coles_matrix(lam_home, lam_away)
    else:
        score_probs = {}
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                prob = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
                if prob > 1e-6:
                    score_probs[(h, a)] = prob
        total_prob = sum(score_probs.values()) + 1e-9
        score_probs = {k: v / total_prob for k, v in score_probs.items()}

    score_probs, outcome_anchor = anchor_score_outcomes(
        score_probs,
        {'胜': p_home, '平': p_draw, '负': p_away},
    )
    
    sorted_scores = sorted(score_probs.items(), key=lambda x: -x[1])
    
    top3 = []
    for (h, a), prob in sorted_scores[:3]:
        top3.append({
            'score': f"{h}-{a}",
            'probability': prob,
            'home_goals': h,
            'away_goals': a,
        })
    
    return {
        'top3': top3,
        'score_probs': score_probs,
        'lambda_home': lam_home,
        'lambda_away': lam_away,
        'target_total': target_total,
        '1x2_prob': {'H': p_home, 'D': p_draw, 'A': p_away},
        'outcome_anchor': outcome_anchor,
    }


DC_RHO = 0.0


OU_TOTAL_BLEND = 0.6


TARGET_TOTAL_MIN = 1.8


TARGET_TOTAL_MAX = 3.6


FACTOR_MIN = 0.85


FACTOR_MAX = 1.15


STRENGTH_SPLIT = SCORE_SPLIT  # = 0.45


BEIDAN_STRONG_MIN_PROBABILITY = 0.65


BEIDAN_STRONG_MIN_LEAD = 0.10


BEIDAN_MEDIUM_MIN_PROBABILITY = 0.60


BEIDAN_MEDIUM_MIN_LEAD = 0.10


BEIDAN_HIGH_PRECISION_MIN_PROBABILITY = 0.70


SCORE_OUTCOME_ANCHOR_STRENGTH = 0.75


def _to_euro_odds(x):
    """亚洲盘贴水(water)转欧赔。

    okooo/bet365 的大小球盘口常给「贴水」格式（如大球 0.83 / 小球 1.0），
    其值域约 0.5~1.3，欧赔 = 贴水 + 1.0。若直接当作欧赔用 1/odds 会严重高估。
    经验阈值 <=1.2 视为贴水，否则视为欧赔。
    """
    if x is None:
        return None
    try:
        x = float(x)
    except (ValueError, TypeError):
        return None
    if x <= 0:
        return None
    if x <= 1.2:
        return x + 1.0
    return x


def _parse_total_line_value(value, default=2.5):
    """Parse whole, half and split Asian total lines (for example ``2.5/3``)."""
    if isinstance(value, (int, float)):
        return float(value)
    numbers = re.findall(r'\d+(?:\.\d+)?', str(value or ''))
    if not numbers:
        return default
    parsed = [float(number) for number in numbers[:2]]
    return sum(parsed) / len(parsed)


def _asian_line_parts(line):
    """Return the one or two settlement lines represented by an Asian line."""
    line = round(float(line) * 4.0) / 4.0
    fraction = round(line - math.floor(line), 2)
    if fraction == 0.25:
        return (math.floor(line), math.floor(line) + 0.5)
    if fraction == 0.75:
        return (math.floor(line) + 0.5, math.floor(line) + 1.0)
    return (line,)


def _asian_over_profit(goals, line, decimal_odds):
    profits = []
    for settlement_line in _asian_line_parts(line):
        if goals > settlement_line:
            profits.append(decimal_odds - 1.0)
        elif goals == settlement_line:
            profits.append(0.0)
        else:
            profits.append(-1.0)
    return sum(profits) / len(profits)


def implied_total_from_ou(over_odds, under_odds, line=2.5):
    """由大小球盘口反推隐含总进球数 λ_total（Poisson 假设）。

    兼容两种格式：
      - 欧赔格式（如 1.85 / 1.95）：直接用 1/odds
      - 亚洲盘贴水格式（如 0.83 / 1.0，okooo/bet365 常见）：先 _to_euro_odds 转换
    """
    if not over_odds or not under_odds:
        return None
    oe = _to_euro_odds(over_odds)
    ue = _to_euro_odds(under_odds)
    if not oe or not ue:
        return None
    try:
        po = 1.0 / oe
        pu = 1.0 / ue
    except (ValueError, TypeError):
        return None
    tot = po + pu
    if tot <= 0:
        return None
    p_over = po / tot
    fair_over_odds = 1.0 / max(p_over, 1e-9)
    total_line = _parse_total_line_value(line)

    def expected_over_profit(mean):
        # 15+ goals has negligible mass inside the guarded search interval.
        return sum(
            poisson_pmf(goals, mean) * _asian_over_profit(goals, total_line, fair_over_odds)
            for goals in range(16)
        )

    lo, hi = 0.5, 9.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected_over_profit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_dixon_coles_matrix(lam_home, lam_away, rho=DC_RHO, max_goals=MAX_GOALS):
    """Dixon-Coles 修正的独立泊松联合比分分布 {(h,a): prob}"""
    probs = {}
    raw_sum = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
            if h == 0 and a == 0:
                tau = 1 - lam_home * lam_away * rho
            elif h == 0 and a == 1:
                tau = 1 + lam_home * rho
            elif h == 1 and a == 0:
                tau = 1 + lam_away * rho
            elif h == 1 and a == 1:
                tau = 1 - rho
            else:
                tau = 1.0
            p = base * tau
            probs[(h, a)] = p
            raw_sum += p
    if raw_sum <= 0:
        # 退化情形回退独立泊松
        for k in list(probs.keys()):
            probs[k] = poisson_pmf(k[0], lam_home) * poisson_pmf(k[1], lam_away)
        raw_sum = sum(probs.values()) + 1e-9
    return {k: v / raw_sum for k, v in probs.items()}


def aggregate_goals_from_scores(score_dist):
    """从比分分布聚合总进球分布（桶 0..6, 7+）"""
    goal = defaultdict(float)
    for (h, a), p in score_dist.items():
        t = h + a
        goal['7+' if t >= 7 else str(t)] += p
    return dict(goal)


def anchor_score_outcomes(score_dist, target_probabilities,
                          strength=SCORE_OUTCOME_ANCHOR_STRENGTH):
    """Partially anchor score-matrix 1X2 mass to the closing market."""
    if not score_dist or not target_probabilities:
        return score_dist, {'applied': False, 'reason': 'missing_distribution_or_target'}

    aliases = {
        '胜': ('胜', 'H', 'home'),
        '平': ('平', 'D', 'draw'),
        '负': ('负', 'A', 'away'),
    }
    target = {}
    for label, keys in aliases.items():
        value = next((target_probabilities.get(key) for key in keys
                      if target_probabilities.get(key) is not None), 0.0)
        try:
            target[label] = max(0.0, float(value))
        except (TypeError, ValueError):
            target[label] = 0.0
    target_total = sum(target.values())
    if target_total <= 0:
        return score_dist, {'applied': False, 'reason': 'invalid_target'}
    target = {key: value / target_total for key, value in target.items()}

    current = {'胜': 0.0, '平': 0.0, '负': 0.0}
    normalized = {}
    for score, probability in score_dist.items():
        try:
            home, away = int(score[0]), int(score[1])
            probability = max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
        label = '胜' if home > away else ('负' if home < away else '平')
        normalized[(home, away)] = normalized.get((home, away), 0.0) + probability
        current[label] += probability
    raw_total = sum(current.values())
    if raw_total <= 0 or any(current[key] <= 0 or target[key] <= 0 for key in current):
        return score_dist, {'applied': False, 'reason': 'incomplete_outcome_mass'}
    current = {key: value / raw_total for key, value in current.items()}

    weight = max(0.0, min(1.0, float(strength)))
    adjusted = {}
    for (home, away), probability in normalized.items():
        label = '胜' if home > away else ('负' if home < away else '平')
        factor = (target[label] / current[label]) ** weight
        adjusted[(home, away)] = probability * factor
    adjusted_total = sum(adjusted.values())
    if adjusted_total <= 0:
        return score_dist, {'applied': False, 'reason': 'zero_adjusted_total'}
    adjusted = {score: value / adjusted_total for score, value in adjusted.items()}
    after = {
        label: sum(value for (home, away), value in adjusted.items()
                   if ('胜' if home > away else ('负' if home < away else '平')) == label)
        for label in ('胜', '平', '负')
    }
    return adjusted, {
        'applied': True,
        'strength': weight,
        'before': {key: round(value, 6) for key, value in current.items()},
        'target': {key: round(value, 6) for key, value in target.items()},
        'after': {key: round(value, 6) for key, value in after.items()},
    }


def match_target_total(league='', total_over_odds=None, total_under_odds=None,
                       asian_factor=1.0, goals_factor=1.0, total_line=2.5):
    """计算比赛目标总进球：大小球隐含总进球与联赛均值融合，再叠加盘口趋势因子。

    关键修正：
      - 盘口隐含总进球经 _to_euro_odds 正确处理亚洲盘贴水格式
      - 盘口权重降至 OU_TOTAL_BLEND（联赛均值主导），避免噪声盘口主导
      - 趋势因子做软约束，目标总进球做硬约束到合理区间
    """
    league_profile = LEAGUE_PROFILES.get(league, {'avg_goals': 2.6, 'draw_rate': 0.27})
    avg_goals = league_profile['avg_goals']
    ou_total = implied_total_from_ou(total_over_odds, total_under_odds, total_line)
    if ou_total:
        target = OU_TOTAL_BLEND * ou_total + (1 - OU_TOTAL_BLEND) * avg_goals
    else:
        target = avg_goals
    # 盘口趋势因子软约束，防止亚洲/总进球历史极端放大
    try:
        asian_factor = max(FACTOR_MIN, min(FACTOR_MAX, float(asian_factor)))
        goals_factor = max(FACTOR_MIN, min(FACTOR_MAX, float(goals_factor)))
    except (ValueError, TypeError):
        asian_factor, goals_factor = 1.0, 1.0
    target = target * asian_factor * goals_factor
    # 目标总进球硬约束到合理区间（公平赛事峰值 2-3）
    return max(TARGET_TOTAL_MIN, min(TARGET_TOTAL_MAX, target))


def match_lambdas(home_prob, draw_prob, away_prob, target_total,
                  split=STRENGTH_SPLIT):
    """由 1X2 隐含概率与目标总进球计算主客 λ（强度感知分配）"""
    return euro_implied_lambdas(home_prob, draw_prob, away_prob, target_total)


def parse_beidan_handicap(handicap):
    if handicap is None:
        return None
    if isinstance(handicap, (int, float)):
        return float(handicap)

    text = str(handicap).strip()
    if not text:
        return None
    text = text.replace('（', '(').replace('）', ')')
    match = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def rqspf_probs_from_score_probs(score_probs, handicap):
    handicap_value = parse_beidan_handicap(handicap)
    if handicap_value is None:
        return {}, {'available': False, 'reason': 'missing_handicap'}

    probs = {'让胜': 0.0, '让平': 0.0, '让负': 0.0}
    top_scores = []
    for (home_goals, away_goals), prob in score_probs.items():
        adjusted_margin = home_goals + handicap_value - away_goals
        if adjusted_margin > 0:
            label = '让胜'
        elif adjusted_margin < 0:
            label = '让负'
        else:
            label = '让平'
        probs[label] += prob
        top_scores.append({
            'score': f"{home_goals}-{away_goals}",
            'handicap_score': f"{home_goals + handicap_value:g}-{away_goals}",
            'result': label,
            'probability': prob,
        })

    total = sum(probs.values())
    if total > 0:
        probs = {key: value / total for key, value in probs.items()}

    top_scores.sort(key=lambda item: -item['probability'])
    return probs, {
        'available': True,
        'handicap': handicap_value,
        'top_scores': top_scores[:5],
    }


