#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conservative league/market policies for football score distributions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from ..common.paths import data_path
from ..domain.sports.football.policy import (  # noqa: F401
    CUP_HINTS,
    HIGH_VARIANCE_HINTS,
    LOW_GOAL_LEAGUE_HINTS,
    PARAM_ALIASES,
    PARAM_RANGES,
    POLICY_PARAM_KEYS,
    _canonical_params,
    _contains_any,
    _empty_tuning_config,
    _league_text,
    blend_score_matrices,
    get_handicap_bucket,
    get_total_bucket,
    normalize_score_matrix,
    policy_bucket_key,
    select_diverse_score_scenarios,
)



TUNING_KEY = 'football_prediction_tuning'
TUNING_FILE = Path(data_path('football_prediction_tuning.json'))








def _load_tuning_config() -> Dict:
    config = None
    try:
        from ..common import kv_store
        config = kv_store.load(TUNING_KEY)
    except Exception:
        config = None

    if not config:
        try:
            if TUNING_FILE.exists():
                config = json.loads(TUNING_FILE.read_text(encoding='utf-8'))
        except Exception:
            config = None

    if not isinstance(config, dict):
        config = _empty_tuning_config()

    base = _empty_tuning_config()
    base.update(config)
    for key in ('global', 'leagues', 'buckets'):
        if not isinstance(base.get(key), dict):
            base[key] = {}
    if not isinstance(base.get('history'), list):
        base['history'] = []
    return base


def _save_tuning_config(config: Dict) -> bool:
    config = dict(config or {})
    config['version'] = config.get('version', 1)
    config['updated_at'] = datetime.now().isoformat()
    saved = False

    try:
        from ..common import kv_store
        kv_store.save(TUNING_KEY, config)
        saved = True
    except Exception:
        saved = False

    try:
        TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
        TUNING_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        saved = True
    except Exception:
        pass

    return saved








def save_tuning_params(params: Dict,
                       league=None,
                       total_line=None,
                       handicap=None,
                       league_profile=None,
                       scope: str = 'bucket',
                       metrics: Dict = None) -> Dict:
    """Persist optimized policy parameters for global, league, or bucket scope."""
    cleaned = _canonical_params(params)
    if not cleaned:
        return {'saved': False, 'error': 'no supported policy params', 'params': {}}

    config = _load_tuning_config()
    league_name = _league_text(league, league_profile) or '*'
    bucket_key = policy_bucket_key(league_name, total_line, handicap)

    if scope == 'global':
        config['global'].update(cleaned)
        target = 'global'
    elif scope == 'league':
        config['leagues'].setdefault(league_name, {}).update(cleaned)
        target = f"league:{league_name}"
    else:
        config['buckets'].setdefault(bucket_key, {}).update(cleaned)
        target = f"bucket:{bucket_key}"

    config.setdefault('history', []).append({
        'saved_at': datetime.now().isoformat(),
        'scope': scope,
        'target': target,
        'params': cleaned,
        'metrics': metrics or {},
    })
    config['history'] = config['history'][-50:]

    saved = _save_tuning_config(config)
    return {
        'saved': saved,
        'scope': scope,
        'target': target,
        'params': cleaned,
        'path': str(TUNING_FILE),
    }


def load_tuning_params(league=None, total_line=None, handicap=None, league_profile=None) -> Dict:
    """Load merged tuning params for one league/market bucket."""
    config = _load_tuning_config()
    league_name = _league_text(league, league_profile) or '*'
    bucket_key = policy_bucket_key(league_name, total_line, handicap)
    wildcard_bucket = policy_bucket_key('*', total_line, handicap)

    merged = {}
    merged.update(_canonical_params(config.get('global', {})))
    merged.update(_canonical_params(config.get('leagues', {}).get(league_name, {})))
    merged.update(_canonical_params(config.get('buckets', {}).get(wildcard_bucket, {})))
    merged.update(_canonical_params(config.get('buckets', {}).get(bucket_key, {})))
    return {
        'params': merged,
        'bucket_key': bucket_key,
        'source': TUNING_KEY,
        'path': str(TUNING_FILE),
    }


def get_tuning_config() -> Dict:
    """Return the full persisted tuning config for inspection."""
    config = _load_tuning_config()
    config['path'] = str(TUNING_FILE)
    return config


def clear_tuning_params(scope: str = None,
                        league=None,
                        total_line=None,
                        handicap=None,
                        league_profile=None) -> Dict:
    """Clear persisted tuning params by scope. Use scope=None to clear all."""
    config = _load_tuning_config()

    if scope is None or scope == 'all':
        config['global'] = {}
        config['leagues'] = {}
        config['buckets'] = {}
        target = 'all'
    elif scope == 'global':
        config['global'] = {}
        target = 'global'
    elif scope == 'league':
        league_name = _league_text(league, league_profile) or '*'
        config['leagues'].pop(league_name, None)
        target = f"league:{league_name}"
    else:
        league_name = _league_text(league, league_profile) or '*'
        bucket_key = policy_bucket_key(league_name, total_line, handicap)
        config['buckets'].pop(bucket_key, None)
        target = f"bucket:{bucket_key}"

    config.setdefault('history', []).append({
        'saved_at': datetime.now().isoformat(),
        'action': 'clear',
        'scope': scope or 'all',
        'target': target,
    })
    config['history'] = config['history'][-50:]
    saved = _save_tuning_config(config)
    return {'cleared': saved, 'target': target, 'path': str(TUNING_FILE)}


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
    late_market_weight_bias = 0.0

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

    policy = {
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
        'late_market_weight_bias': late_market_weight_bias,
    }
    tuning = load_tuning_params(
        league=league_name,
        total_line=total_line,
        handicap=handicap,
        league_profile=league_profile,
    )
    if tuning['params']:
        policy.update(tuning['params'])
        policy['tuning'] = {
            'applied': True,
            'bucket_key': tuning['bucket_key'],
            'params': tuning['params'],
            'path': tuning['path'],
        }
    else:
        policy['tuning'] = {'applied': False, 'bucket_key': tuning['bucket_key']}
    return policy






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
