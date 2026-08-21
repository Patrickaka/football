# -*- coding: utf-8 -*-
"""北单赛果提取、历史校准与历史记录读写"""

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
    BEIDAN_HISTORY_KEY, BEIDAN_HISTORY_LIMIT,
)

def calculate_implied_probability(odds_dict):
    if not odds_dict:
        return {}
    
    prob_sum = sum(1 / o for o in odds_dict.values() if o and o > 0)
    if prob_sum == 0:
        return {}
    
    return {k: (1 / v) / prob_sum for k, v in odds_dict.items() if v and v > 0}


def _actual_spf_from_record(record):
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}

    direct = (
        record.get('actual_spf')
        or actual.get('spf')
        or settlement.get('spf')
        or settlement.get('actual_spf')
    )
    if direct in ('胜', '平', '负'):
        return direct

    score = (
        record.get('actual_score')
        or actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
    )
    if not score or '-' not in str(score):
        return None
    try:
        home_goals, away_goals = map(int, str(score).split('-', 1))
    except ValueError:
        return None
    if home_goals > away_goals:
        return '胜'
    if home_goals < away_goals:
        return '负'
    return '平'


def _actual_zjq_from_record(record):
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}

    direct = (
        record.get('actual_zjq')
        or actual.get('zjq')
        or settlement.get('zjq')
        or settlement.get('actual_zjq')
    )
    if direct is not None:
        direct = str(direct)
        if direct in {'0', '1', '2', '3', '4', '5', '6', '7+'}:
            return direct

    score = (
        record.get('actual_score')
        or actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
    )
    if not score or '-' not in str(score):
        return None
    try:
        home_goals, away_goals = map(int, str(score).split('-', 1))
    except ValueError:
        return None
    total_goals = home_goals + away_goals
    return '7+' if total_goals >= 7 else str(total_goals)


def _actual_bifen_from_record(record):
    """从已结算快照解析实际比分字符串 'h-a'（用于比分历史校准）"""
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}

    direct = (
        record.get('actual_score')
        or actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
    )
    if not direct or '-' not in str(direct):
        return None
    try:
        home_goals, away_goals = map(int, str(direct).split('-', 1))
    except (ValueError, TypeError):
        return None
    return f"{home_goals}-{away_goals}"


def _actual_rqspf_from_record(record):
    """从已结算快照的实际比分 + 让球值，推导让球胜平负实际结果。"""
    hc = record.get('handicap')
    try:
        hv = float(hc) if hc is not None else 0.0
    except (TypeError, ValueError):
        hv = 0.0
    actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
    settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}
    score = (
        actual.get('score')
        or actual.get('actual_score')
        or settlement.get('score')
        or settlement.get('actual_score')
        or record.get('actual_score')
    )
    if not score or '-' not in str(score):
        return None
    try:
        home_goals, away_goals = map(int, str(score).split('-', 1))
    except (ValueError, TypeError):
        return None
    margin = home_goals + hv - away_goals
    return '让胜' if margin > 0 else ('让负' if margin < 0 else '让平')


def apply_beidan_history_calibration(probabilities, bet_type, league=None, min_samples=8, limit=200):
    """Use settled Beidan snapshots as a conservative reliability correction."""
    if not probabilities:
        return probabilities, {'applied': False, 'reason': 'empty_probabilities'}

    records = _load_beidan_history()
    if not records:
        return probabilities, {'applied': False, 'reason': 'no_history'}

    expected = {str(k): 0.0 for k in probabilities}
    actuals = {str(k): 0.0 for k in probabilities}
    samples = 0

    for record in records[:limit]:
        if not record.get('settled'):
            continue
        section = record.get(bet_type) if isinstance(record.get(bet_type), dict) else {}
        past_probs = section.get('probabilities') if isinstance(section.get('probabilities'), dict) else {}
        if not past_probs:
            continue

        if bet_type == 'spf':
            actual = _actual_spf_from_record(record)
        elif bet_type == 'zjq':
            actual = _actual_zjq_from_record(record)
        elif bet_type == 'bifen':
            actual = _actual_bifen_from_record(record)
        elif bet_type == 'rqspf':
            actual = _actual_rqspf_from_record(record)
        else:
            actual = None
        if actual not in expected:
            continue

        league_weight = 1.25 if league and record.get('league') == league else 1.0
        for option in expected:
            expected[option] += float(past_probs.get(option, 0.0) or 0.0) * league_weight
        actuals[actual] += league_weight
        samples += league_weight

    if samples < min_samples:
        return probabilities, {
            'applied': False,
            'reason': 'insufficient_settled_samples',
            'sample_count': round(samples, 3),
            'min_samples': min_samples,
        }

    factors = {}
    prior = 6.0
    option_count = max(len(expected), 1)
    prior_each = prior / option_count
    for option in expected:
        ratio = (actuals[option] + prior_each) / max(expected[option] + prior_each, 1e-9)
        factors[option] = max(0.86, min(1.16, ratio))

    adjusted = {
        option: float(probabilities.get(option, 0.0) or 0.0) * factors.get(option, 1.0)
        for option in probabilities
    }
    total = sum(adjusted.values())
    if total <= 0:
        return probabilities, {'applied': False, 'reason': 'zero_adjusted_total'}

    adjusted = {option: value / total for option, value in adjusted.items()}
    return adjusted, {
        'applied': True,
        'sample_count': round(samples, 3),
        'factors': {k: round(v, 6) for k, v in factors.items()},
        'actuals': {k: round(v, 3) for k, v in actuals.items()},
        'expected': {k: round(v, 3) for k, v in expected.items()},
    }


def _beidan_record_key(match):
    return '|'.join(str(match.get(k, '')) for k in ('date', 'num', 'home', 'away'))


def _load_beidan_history():
    data = kv_store.load(BEIDAN_HISTORY_KEY, [])
    return data if isinstance(data, list) else []


def _save_beidan_history(records):
    records = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)
    return kv_store.save(BEIDAN_HISTORY_KEY, records[:BEIDAN_HISTORY_LIMIT])


