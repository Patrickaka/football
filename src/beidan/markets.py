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

def adjust_probs_by_asian(home_win_prob, draw_prob, away_win_prob, asian_history):
    if not asian_history or len(asian_history) < 2:
        return home_win_prob, draw_prob, away_win_prob
    
    recent_changes = asian_history[-5:]
    
    home_odds_changes = []
    away_odds_changes = []
    
    for i in range(1, len(recent_changes)):
        prev = recent_changes[i-1]
        curr = recent_changes[i]
        
        if prev.get('home_odds') and curr.get('home_odds'):
            home_odds_changes.append(curr['home_odds'] - prev['home_odds'])
        if prev.get('away_odds') and curr.get('away_odds'):
            away_odds_changes.append(curr['away_odds'] - prev['away_odds'])
    
    home_trend = sum(home_odds_changes) / len(home_odds_changes) if home_odds_changes else 0
    away_trend = sum(away_odds_changes) / len(away_odds_changes) if away_odds_changes else 0
    
    adjustment_factor = 0.15
    
    if home_trend > 0.02:
        home_win_prob *= (1 - adjustment_factor)
        away_win_prob *= (1 + adjustment_factor * 0.5)
    elif home_trend < -0.02:
        home_win_prob *= (1 + adjustment_factor)
        away_win_prob *= (1 - adjustment_factor * 0.5)
    
    if away_trend > 0.02:
        away_win_prob *= (1 - adjustment_factor)
        home_win_prob *= (1 + adjustment_factor * 0.5)
    elif away_trend < -0.02:
        away_win_prob *= (1 + adjustment_factor)
        home_win_prob *= (1 - adjustment_factor * 0.5)
    
    total_prob = home_win_prob + draw_prob + away_win_prob
    if total_prob > 0:
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
    
    return home_win_prob, draw_prob, away_win_prob


def analyze_asian_trend(asian_history):
    if not asian_history:
        return {'direction': 'stable', 'strength': 0}
    
    recent = asian_history[-5:]
    if len(recent) < 2:
        return {'direction': 'stable', 'strength': 0}
    
    home_changes = []
    away_changes = []
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        
        if prev.get('home_odds') and curr.get('home_odds'):
            home_changes.append(curr['home_odds'] - prev['home_odds'])
        if prev.get('away_odds') and curr.get('away_odds'):
            away_changes.append(curr['away_odds'] - prev['away_odds'])
    
    avg_home_change = sum(home_changes) / len(home_changes) if home_changes else 0
    avg_away_change = sum(away_changes) / len(away_changes) if away_changes else 0
    
    strength = abs(avg_home_change) + abs(avg_away_change)
    
    if avg_home_change < -0.03:
        direction = 'home_backing'
    elif avg_home_change > 0.03:
        direction = 'home_laying'
    elif avg_away_change < -0.03:
        direction = 'away_backing'
    elif avg_away_change > 0.03:
        direction = 'away_laying'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'strength': round(strength, 4),
        'avg_home_change': round(avg_home_change, 4),
        'avg_away_change': round(avg_away_change, 4),
    }


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


def analyze_cs_trend(cs_history):
    if not cs_history or len(cs_history) < 2:
        return {'direction': 'stable', 'strength': 0, 'hot_scores': []}
    
    recent = cs_history[-10:]
    
    score_odds_map = {}
    for entry in recent:
        score = entry.get('score')
        odds = entry.get('odds')
        if score and odds:
            if score not in score_odds_map:
                score_odds_map[score] = []
            score_odds_map[score].append(odds)
    
    hot_scores = []
    for score, odds_list in score_odds_map.items():
        avg_odds = sum(odds_list) / len(odds_list)
        trend = odds_list[-1] - odds_list[0] if len(odds_list) >= 2 else 0
        hot_scores.append({
            'score': score,
            'avg_odds': round(avg_odds, 2),
            'trend': 'down' if trend < -0.1 else ('up' if trend > 0.1 else 'stable'),
            'current_odds': odds_list[-1],
        })
    
    hot_scores.sort(key=lambda x: x['current_odds'])
    
    return {
        'direction': 'active' if len(hot_scores) > 0 else 'stable',
        'strength': len(hot_scores),
        'hot_scores': hot_scores[:5],
    }


def enhance_scores_with_cs(score_prediction, cs_history):
    if not cs_history or len(cs_history) < 2:
        return score_prediction
    
    recent = cs_history[-5:]
    
    cs_odds_map = {}
    for entry in recent:
        score = entry.get('score')
        odds = entry.get('odds')
        if score and odds:
            cs_odds_map[score] = odds
    
    if not cs_odds_map:
        return score_prediction
    
    enhanced_scores = []
    for score_item in score_prediction['top3']:
        if isinstance(score_item, dict):
            score = score_item.get('score')
            prob = score_item.get('probability', 0)
        else:
            score = score_item[0]
            prob = score_item[1]
        if not score:
            continue
        
        if score in cs_odds_map:
            cs_odds = cs_odds_map[score]
            cs_prob = 1.0 / cs_odds
            enhanced_prob = (prob + cs_prob) / 2
            enhanced_scores.append((score, enhanced_prob, 'cs_enhanced'))
        else:
            enhanced_scores.append((score, prob, 'poisson'))
    
    for score, odds in cs_odds_map.items():
        if score not in [s[0] for s in enhanced_scores]:
            cs_prob = 1.0 / odds
            enhanced_scores.append((score, cs_prob * 0.5, 'cs_new'))
    
    enhanced_scores.sort(key=lambda x: -x[1])
    
    score_prediction['top3'] = [
        {
            'score': s[0],
            'probability': s[1],
            'source': s[2],
            'home_goals': int(s[0].split('-')[0]) if '-' in s[0] and s[0].split('-')[0].isdigit() else None,
            'away_goals': int(s[0].split('-')[1]) if '-' in s[0] and s[0].split('-')[1].isdigit() else None,
        }
        for s in enhanced_scores[:3]
    ]
    
    return score_prediction


def calculate_goals_factor(goals_history):
    if not goals_history or len(goals_history) < 2:
        return 1.0
    
    recent = goals_history[-10:]
    
    total_over_odds = 0
    total_under_odds = 0
    count = 0
    
    for entry in recent:
        over_odds = entry.get('over_odds')
        under_odds = entry.get('under_odds')
        if over_odds and under_odds:
            total_over_odds += over_odds
            total_under_odds += under_odds
            count += 1
    
    if count == 0:
        return 1.0
    
    avg_over = total_over_odds / count
    avg_under = total_under_odds / count
    
    if avg_over < avg_under:
        return 1.2
    elif avg_over > avg_under + 0.5:
        return 0.85
    else:
        return 1.0


def adjust_zjq_by_goals(zjq_probs, goals_history):
    if not goals_history or len(goals_history) < 2:
        return zjq_probs
    
    recent = goals_history[-5:]
    
    over_trend = 0
    under_trend = 0
    count = 0
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        if prev.get('over_odds') and curr.get('over_odds'):
            over_trend += curr['over_odds'] - prev['over_odds']
            count += 1
        if prev.get('under_odds') and curr.get('under_odds'):
            under_trend += curr['under_odds'] - prev['under_odds']
    
    if count == 0:
        return zjq_probs
    
    over_trend_avg = over_trend / count
    
    if over_trend_avg < -0.05:
        for key in ['3', '4', '5', '6', '7+']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 1.2
        for key in ['0', '1', '2']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 0.85
    elif over_trend_avg > 0.05:
        for key in ['0', '1', '2']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 1.2
        for key in ['3', '4', '5', '6', '7+']:
            zjq_probs[key] = zjq_probs.get(key, 0) * 0.85
    
    return zjq_probs


def analyze_goals_trend(goals_history):
    if not goals_history or len(goals_history) < 2:
        return {'direction': 'stable', 'strength': 0}
    
    recent = goals_history[-10:]
    
    over_changes = []
    under_changes = []
    
    for i in range(1, len(recent)):
        prev = recent[i-1]
        curr = recent[i]
        if prev.get('over_odds') and curr.get('over_odds'):
            over_changes.append(curr['over_odds'] - prev['over_odds'])
        if prev.get('under_odds') and curr.get('under_odds'):
            under_changes.append(curr['under_odds'] - prev['under_odds'])
    
    avg_over_change = sum(over_changes) / len(over_changes) if over_changes else 0
    avg_under_change = sum(under_changes) / len(under_changes) if under_changes else 0
    
    strength = abs(avg_over_change) + abs(avg_under_change)
    
    if avg_over_change < -0.05:
        direction = 'over_backing'
    elif avg_over_change > 0.05:
        direction = 'over_laying'
    elif avg_under_change < -0.05:
        direction = 'under_backing'
    elif avg_under_change > 0.05:
        direction = 'under_laying'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'strength': round(strength, 4),
        'avg_over_change': round(avg_over_change, 4),
        'avg_under_change': round(avg_under_change, 4),
    }


def calculate_asian_goal_factor(asian_history):
    if not asian_history or len(asian_history) < 2:
        return 1.0
    
    recent = asian_history[-10:]
    
    total_odds_sum = 0
    count = 0
    
    for entry in recent:
        home_odds = entry.get('home_odds')
        away_odds = entry.get('away_odds')
        if home_odds and away_odds:
            total_odds_sum += home_odds + away_odds
            count += 1
    
    if count == 0:
        return 1.0
    
    avg_total_odds = total_odds_sum / count
    
    if avg_total_odds < 3.6:
        return 1.3
    elif avg_total_odds < 4.0:
        return 1.15
    elif avg_total_odds < 4.4:
        return 1.0
    elif avg_total_odds < 4.8:
        return 0.9
    else:
        return 0.75


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


