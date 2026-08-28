# -*- coding: utf-8 -*-
"""足球预测常量配置、联赛画像与可选子系统延迟导入"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')


FOOTBALL_PREDICTION_LOGIC_VERSION = '2026-08-21-jczq-market-availability-v34'
# Official 1X2 prices are the strongest observable production signal.  Raising
# their share from 40% to 80% improved the pooled high-confidence proxy from
# 61.67% to 62.55%; the model retains 20% for team/Asian/context information.
LOTTERY_OFFICIAL_ODDS_WEIGHT = 0.80
SCORE_1X2_MARKET_ANCHOR_STRENGTH = 0.75
ACTIONABLE_1X2_MIN_PROBABILITY = 0.65
ACTIONABLE_1X2_MIN_MARGIN = 0.10

# ELO 评分系统（延迟导入）
try:
    from .elo import get_elo_system, elo_to_goals_expected, elo_to_strength_factor
    ELO_AVAILABLE = True
except Exception:
    ELO_AVAILABLE = False
    log.warning("ELO 模块未导入，将使用默认球队实力计算")

# 相似盘口数据库（延迟导入）
try:
    from .similar_market import similar_market_match
    SIMILAR_MARKET_AVAILABLE = True
except Exception:
    SIMILAR_MARKET_AVAILABLE = False
    log.warning("相似盘口数据库模块未导入")

# 临场资金流检测器（延迟导入）
try:
    from .steam_move import steam_move_detector, integrate_steam_signal
    STEAM_MOVE_AVAILABLE = True
except Exception:
    STEAM_MOVE_AVAILABLE = False
    log.warning("临场资金流检测器模块未导入")

# 贝叶斯校准层（延迟导入）
try:
    from .bayesian_calibration import calibrate_predictions, get_calibrator
    BAYESIAN_CALIBRATION_AVAILABLE = True
except Exception:
    BAYESIAN_CALIBRATION_AVAILABLE = False

# 缓存管理器（延迟导入）
try:
    from .cache_manager import get_cache, set_cache, invalidate_cache, clear_all_cache
    CACHE_AVAILABLE = True
except Exception:
    CACHE_AVAILABLE = False
    log.warning("缓存管理器模块未导入")

# 动态ELO系统（延迟导入）
try:
    from .dynamic_elo import get_team_elo, get_elo_difference
    DYNAMIC_ELO_AVAILABLE = True
except Exception:
    DYNAMIC_ELO_AVAILABLE = False
    log.warning("动态ELO系统模块未导入")

# 赔率价值分析（延迟导入）
try:
    from .value_betting import adjust_by_value, identify_value_bets, calculate_value, calculate_ev
    VALUE_BETTING_AVAILABLE = True
except Exception:
    VALUE_BETTING_AVAILABLE = False
    log.warning("赔率价值分析模块未导入")

# 动态权重调整（延迟导入）
try:
    from .dynamic_weights import get_dynamic_weights, fuse_predictions
    DYNAMIC_WEIGHTS_AVAILABLE = True
except Exception:
    DYNAMIC_WEIGHTS_AVAILABLE = False
    log.warning("动态权重调整模块未导入")

# 盘口聚类（延迟导入）
try:
    from .market_clustering import fuse_poisson_with_prior, get_market_prior
    MARKET_CLUSTERING_AVAILABLE = True
except Exception:
    MARKET_CLUSTERING_AVAILABLE = False
    log.warning("盘口聚类模块未导入")

# ===================== 常量 =====================
BASE = 'https://odds.500.com'
INDEX_URL = f'{BASE}/index_jczq.shtml'
INDEX_URLS = (
    INDEX_URL,
    'http://odds.500.com/index_jczq.shtml',
    f'{BASE}/',
)
MATCH_LIST_CACHE_PATH = data_path('football_match_list_cache.json')
_MATCH_LIST_STATUS = {'source': 'not_requested', 'stale': False, 'error': None}
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 平均值行至少应包含 初/终 各 3 个数值
MIN_AVG_NUMBERS = 6

# 趋势判定阈值：低于该幅度视为"无变化/稳定"
HANDICAP_TREND_EPS = 0.02
WATER_TREND_EPS = 0.05
EURO_PROB_TREND_EPS = 0.02

# 大小球倾向阈值：单边真实概率达到该值才判定为大/小球倾向
TOTAL_LEAN_THRESHOLD = 0.55

# 凯利指数：某项高于返还率超过该值视为「打出难度大」
KELLY_BIAS_EPS = 2.0

# 泊松比分矩阵枚举的单队最大进球数
MAX_GOALS = 7

# 初/终盘融合权重（终盘为主，初盘平滑噪声）
CLOSE_BLEND_WEIGHT = 0.72

# λ 网格拟合：粗搜步长、细搜步长与半径
LAMBDA_COARSE_STEP = 0.12
LAMBDA_FINE_STEP = 0.04
LAMBDA_FINE_RADIUS = 0.18

# 拟合目标权重：1X2 / 总进球 / 净胜球(反推) / 大小球分布 / 球队攻防先验
FIT_W_1X2 = 3.0
FIT_W_TOTAL = 1.4
FIT_W_SUPREMACY = 2.0
FIT_W_OU_DIST = 0.9
FIT_W_TEAM = 1.35

# 净胜球：亚盘反推 vs 欧赔反推 的融合权重（不再使用让球盘数值本身）
SUP_ASIAN_WEIGHT = 0.48
SUP_EURO_WEIGHT = 0.52

# 联赛场均进球基准（用于球队攻防强度归一化，可被联赛配置覆盖）
AVG_LEAGUE_GOAL = 1.35
HOME_VENUE_ATTACK_BOOST = 1.06

# 净胜球亚盘/欧赔严重分歧时改等权融合
SUPREMACY_CONFLICT_GAP = 0.75

# 欧赔走势对净胜球的修正幅度
MOMENTUM_SUPREMACY_WEIGHT = 0.22

# 离散度计算窗口（最近N条记录）
DISPERSION_WINDOW = 5

# 坐标下降精调 λ 的迭代次数与步长
LAMBDA_REFINE_STEPS = 28
LAMBDA_REFINE_STEP0 = 0.07

# 预测置信度：低于该值仅推荐 1 个比分
CONFIDENCE_LOW_THRESHOLD = 0.52
CONFIDENCE_HIGH_THRESHOLD = 0.72

# 联赛画像：场均进球、主场加成、低比分倾向（乘在 0-2 球基准频率上）
LEAGUE_PROFILES = {
    'default': {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
    '英超': {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95},
    '英冠': {'avg_goal': 1.46, 'home_boost': 1.07, 'low_score': 0.90, 'draw_mult': 0.96},
    '西甲': {'avg_goal': 1.42, 'home_boost': 1.07, 'low_score': 0.95, 'draw_mult': 1.05},
    '意甲': {'avg_goal': 1.32, 'home_boost': 1.05, 'low_score': 1.05, 'draw_mult': 1.08},
    '德甲': {'avg_goal': 1.56, 'home_boost': 1.06, 'low_score': 0.86, 'draw_mult': 0.94},
    '法甲': {'avg_goal': 1.36, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.02},
    '荷甲': {'avg_goal': 1.58, 'home_boost': 1.05, 'low_score': 0.85, 'draw_mult': 0.93},
    '葡超': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '欧冠': {'avg_goal': 1.50, 'home_boost': 1.04, 'low_score': 0.92, 'draw_mult': 0.98},
    '欧联': {'avg_goal': 1.44, 'home_boost': 1.05, 'low_score': 0.94, 'draw_mult': 1.0},
    '世界杯': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '欧洲杯': {'avg_goal': 1.40, 'home_boost': 1.04, 'low_score': 0.98, 'draw_mult': 1.02},
    '友谊': {'avg_goal': 1.44, 'home_boost': 1.02, 'low_score': 0.95, 'draw_mult': 0.97},
    '国际': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '巴甲': {'avg_goal': 1.42, 'home_boost': 1.08, 'low_score': 0.92, 'draw_mult': 0.96},
    '阿甲': {'avg_goal': 1.36, 'home_boost': 1.07, 'low_score': 0.96, 'draw_mult': 0.98},
    '中超': {'avg_goal': 1.32, 'home_boost': 1.07, 'low_score': 1.00, 'draw_mult': 1.04},
    '日职': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '韩K': {'avg_goal': 1.30, 'home_boost': 1.06, 'low_score': 1.02, 'draw_mult': 1.04},
}

# 比分冷热：相对历史基准频率的比值阈值
HEAT_RATIO_HOT = 0.70
HEAT_RATIO_COLD = 1.32
HEAT_FILTER_PENALTY = 0.90   # 提高惩罚系数，减少对高比分的压制（原0.75）
COLD_FILTER_BONUS = 1.08     # 原1.18，缩小冷门奖励，防止低比分通过"冷门"机制反复被加权

# 盘口变化影响因子
HANDICAP_CHANGE_LAMBDA_BOOST = 0.15  # 让球每变化1球对lambda的影响
TOTAL_LINE_CHANGE_LAMBDA_BOOST = 0.6  # 大小球每变化1球对总进球的影响

# λ 融合权重（市场盘口为主）
LAMBDA_WEIGHT_MARKET = 0.5   # 盘口反推 λ 权重
LAMBDA_WEIGHT_TEAM = 0.3     # 球队数据 λ 权重
LAMBDA_WEIGHT_ELO = 0.2      # ELO xG 权重

# 常见比分历史基准频率（用于冷热，非市场赔率）——参考欧洲主流联赛真实分布上调
SCORE_BASELINE_FREQ = {
    (0, 0): 0.075, (1, 0): 0.085, (0, 1): 0.065, (1, 1): 0.110,
    (2, 0): 0.078, (0, 2): 0.055, (2, 1): 0.105, (1, 2): 0.068,
    (2, 2): 0.045, (3, 0): 0.042, (0, 3): 0.025, (3, 1): 0.052,
    (1, 3): 0.032, (3, 2): 0.035, (2, 3): 0.025, (4, 0): 0.020,
    (0, 4): 0.012, (4, 1): 0.025, (1, 4): 0.014, (3, 3): 0.015,
}

# 仅亚盘/大小球走 HTML 抓取（无平均值 JSON 接口）；欧赔走 JSON
ODDS_PAGES = {
    'yazhi': '亚盘',
    'daxiao': '大小球',
}

# 欧赔平均值时间序列 JSON 接口：cid=0 即"平均值"，按时间降序（首条=终盘）
OUZHI_JSON_URL = f'{BASE}/fenxi1/json/ouzhi.php'

# ===================== 工具函数 =====================

