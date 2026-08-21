# -*- coding: utf-8 -*-
"""北单推荐质量评估"""

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
    BEIDAN_HIGH_PRECISION_MIN_PROBABILITY, BEIDAN_MEDIUM_MIN_LEAD, BEIDAN_MEDIUM_MIN_PROBABILITY, BEIDAN_STRONG_MIN_LEAD, BEIDAN_STRONG_MIN_PROBABILITY,
)

def assess_recommendation_quality(probabilities, prediction=None, context=None):
    """Classify narrow calls so the UI can avoid treating them as strong picks."""
    if not probabilities:
        return {
            'level': 'unknown',
            'label': '无数据',
            'confidence': 0,
            'lead': 0,
            'top2': [],
            'advice': '跳过',
            'avoid_single': True,
        }

    ranked = sorted(
        ((str(k), float(v)) for k, v in probabilities.items() if v is not None),
        key=lambda x: -x[1]
    )
    if not ranked:
        return {
            'level': 'unknown',
            'label': '无数据',
            'confidence': 0,
            'lead': 0,
            'top2': [],
            'advice': '跳过',
            'avoid_single': True,
        }

    top_key, top_prob = ranked[0]
    if prediction and prediction != top_key:
        for key, prob in ranked:
            if key == prediction:
                top_key, top_prob = key, prob
                break

    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    lead = top_prob - second_prob
    top2 = [{'option': key, 'probability': prob} for key, prob in ranked[:2]]

    context = context or {}
    asian_direction = context.get('asian_direction')
    score_consistency = context.get('score_consistency') or {}
    conflict = False
    if asian_direction in ('home_backing', 'away_laying') and top_key == '负':
        conflict = True
    elif asian_direction in ('away_backing', 'home_laying') and top_key == '胜':
        conflict = True
    if score_consistency.get('conflict'):
        conflict = True

    if (top_prob >= BEIDAN_STRONG_MIN_PROBABILITY
            and lead >= BEIDAN_STRONG_MIN_LEAD and not conflict):
        level, label, advice = 'strong', '强推荐', f'单选 {top_key}'
    elif (top_prob >= BEIDAN_MEDIUM_MIN_PROBABILITY
          and lead >= BEIDAN_MEDIUM_MIN_LEAD and not conflict):
        level, label = 'medium', '可参考'
        combo = '/'.join(item['option'] for item in top2)
        advice = f'建议双选 {combo}' if len(top2) >= 2 else f'谨慎 {top_key}'
    elif lead < 0.035 or conflict:
        level, label = 'split', '分歧较大'
        combo = '/'.join(item['option'] for item in top2)
        advice = f'建议双选 {combo}' if len(top2) >= 2 else f'谨慎 {top_key}'
    else:
        level, label, advice = 'low', '低置信', f'谨慎 {top_key}'

    return {
        'level': level,
        'label': label,
        'prediction': top_key,
        'confidence': round(top_prob, 6),
        'lead': round(lead, 6),
        'top2': top2,
        'advice': advice,
        'avoid_single': level != 'strong',
        'single_allowed': level == 'strong',
        'high_precision': (
            level == 'strong' and top_prob >= BEIDAN_HIGH_PRECISION_MIN_PROBABILITY
        ),
        'thresholds': {
            'strong_probability': BEIDAN_STRONG_MIN_PROBABILITY,
            'strong_lead': BEIDAN_STRONG_MIN_LEAD,
            'medium_probability': BEIDAN_MEDIUM_MIN_PROBABILITY,
            'medium_lead': BEIDAN_MEDIUM_MIN_LEAD,
            'high_precision_probability': BEIDAN_HIGH_PRECISION_MIN_PROBABILITY,
        },
        'conflict': conflict,
        'score_consistency': score_consistency,
    }


UPSET_MED_FAV_MAX = 0.52


UPSET_MED_GAP_MAX = 0.16


UPSET_MED_MASS_MIN = 0.52


UPSET_HIGH_FAV_MAX = 0.45


UPSET_HIGH_GAP_MAX = 0.10


UPSET_HIGH_MASS_MIN = 0.58


UPSET_CONFIDENT_FAV_MIN = 0.58


UPSET_CONFIDENT_GAP_MIN = 0.20


