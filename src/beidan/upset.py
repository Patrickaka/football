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

# ─── 领域层适配 ───
#
# 判定规则在 `src/domain/sports/beidan/upset.py`。这里把八个门槛喂进去，
# 并保住旧名字（含三个下划线开头的——`__init__.py` 把它们也导出了）。

from src.domain.sports.beidan import upset as _upset

_result_from_score = _upset.result_from_score
_fmt_score = _upset.format_score
_score_result_label = _upset._label_of


def assess_upset_risk(probs_1x2):
    """按 1X2 概率评估爆冷风险，并给出防守方向。"""
    return _upset.assess_risk(
        probs_1x2,
        high_fav_max=UPSET_HIGH_FAV_MAX, high_gap_max=UPSET_HIGH_GAP_MAX,
        high_mass_min=UPSET_HIGH_MASS_MIN,
        medium_fav_max=UPSET_MED_FAV_MAX, medium_gap_max=UPSET_MED_GAP_MAX,
        medium_mass_min=UPSET_MED_MASS_MIN,
        confident_fav_min=UPSET_CONFIDENT_FAV_MIN,
        confident_gap_min=UPSET_CONFIDENT_GAP_MIN)


def pick_upset_scores(score_matrix, favorite_result, top_n=2):
    """挑出与热门赛果相反方向上概率最高的几个比分。"""
    return _upset.pick_scores(score_matrix, favorite_result, top_n=top_n)


def assess_score_consistency(scores, prediction):
    """比分分布指向的赛果与 1X2 推荐是否冲突。"""
    return _upset.score_consistency(scores, prediction)
