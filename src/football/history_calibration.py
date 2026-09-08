#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative production-history calibration for football score distributions.

估算与套用已迁至 `src.domain.sports.football.calibration_history`；
这里只留读兜底档案、读生产历史和 profile 缓存——都是存储。
"""

import json
from pathlib import Path
from typing import Dict

from ..domain.sports.football.calibration_history import (  # noqa: F401
    HISTORY_HALF_LIFE,
    MAX_GOAL_BETA,
    MIN_HISTORY_SAMPLES,
    _mean_after_beta,
    _normalized_scores,
    _outcome,
    _quality_weight,
    _score_tuple,
    apply_history_calibration,
    estimate_history_calibration,
)


PROFILE_PATH = Path(__file__).resolve().parents[2] / 'data' / 'football_history_calibration.json'

_PROFILE_CACHE = {'key': None, 'profile': None}












def _load_fallback_profile() -> Dict:
    try:
        with PROFILE_PATH.open('r', encoding='utf-8') as handle:
            profile = json.load(handle)
        return profile if isinstance(profile, dict) else {}
    except Exception:
        return {}


def get_runtime_history_profile(*, as_of=None) -> Dict:
    """Read the current production history, caching until its latest record changes."""
    try:
        from .result_sync import get_history

        records = get_history().records
        if as_of is not None:
            from .research import timestamp
            cutoff = timestamp(as_of)
            if cutoff is None:
                raise ValueError('as_of must include a timezone')
            records = [r for r in records if timestamp(r.get('settled_at')) is not None
                       and timestamp(r['settled_at']) < cutoff]
        latest = max((str(item.get('updated_at') or item.get('settled_at') or '') for item in records), default='')
        cache_key = (len(records), latest, str(as_of) if as_of is not None else None)
        if _PROFILE_CACHE['key'] == cache_key and _PROFILE_CACHE['profile'] is not None:
            return _PROFILE_CACHE['profile']
        profile = estimate_history_calibration(records)
        if not profile.get('applied') and as_of is None:
            profile = _load_fallback_profile() or profile
        _PROFILE_CACHE.update({'key': cache_key, 'profile': profile})
        return profile
    except Exception:
        return _load_fallback_profile() if as_of is None else {}
