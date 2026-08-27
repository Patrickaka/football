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



UPSET_MED_FAV_MAX = 0.52


UPSET_MED_GAP_MAX = 0.16


UPSET_MED_MASS_MIN = 0.52


UPSET_HIGH_FAV_MAX = 0.45


UPSET_HIGH_GAP_MAX = 0.10


UPSET_HIGH_MASS_MIN = 0.58


UPSET_CONFIDENT_FAV_MIN = 0.58


UPSET_CONFIDENT_GAP_MIN = 0.20

# ─── 领域层适配 ───
#
# 分档规则在 `src/domain/sports/beidan/quality.py`。这里只把门槛喂进去，
# 并保住旧名字——`__init__.py` 导出了它，`recommending` 按名字导入。

from src.domain.sports.beidan import quality as _quality

# 「分歧较大」那一档的门槛。**迁移前它是判断里一个写死的裸数字 0.035**，
# 没有名字也没有出处。领先不到 3.5 个百分点时谁排第一基本由噪声决定。
BEIDAN_SPLIT_MAX_LEAD = 0.035


def assess_recommendation_quality(probabilities, prediction=None, context=None):
    """分档：这一注够不够格单选。"""
    return _quality.assess(
        probabilities, prediction=prediction, context=context,
        strong_probability=BEIDAN_STRONG_MIN_PROBABILITY,
        strong_lead=BEIDAN_STRONG_MIN_LEAD,
        medium_probability=BEIDAN_MEDIUM_MIN_PROBABILITY,
        medium_lead=BEIDAN_MEDIUM_MIN_LEAD,
        high_precision_probability=BEIDAN_HIGH_PRECISION_MIN_PROBABILITY,
        split_lead=BEIDAN_SPLIT_MAX_LEAD)
