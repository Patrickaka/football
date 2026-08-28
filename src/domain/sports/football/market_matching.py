# -*- coding: utf-8 -*-
"""盘口口径归一、相似盘口匹配与聚类先验。

纯计算——**四个模块的 JSON 存储、CSV 下载与数据库读写全部留在适配层**
（判据 16）。这里只有「给定数据怎么算」。

`similar_market` 与 `elo` 的存储走 `common.repositories`（测试 SQLite /
生产 MySQL）。判据 21：凡是依赖方言行为的地方，断言要写在**表定义**上
——SQLite 上复现不出 MySQL 的自增默认值与缺列时的 ORDER BY 次序。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import math
import re
from datetime import datetime

log = logging.getLogger('domain.football.market_matching')


# 迁移当时的真实常量（从四个源模块原样搬来）
STANDARD_ASIAN = [
    -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
    0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
]

STANDARD_OU = [
    1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0
]

LEAGUE_TIERS = {
    'tier1': ['英超', '英超联赛', 'English Premier League'],
    'tier2': ['德甲', '德甲联赛', 'Bundesliga'],
    'tier3': ['西甲', '西甲联赛', 'La Liga'],
    'tier4': ['意甲', '意甲联赛', 'Serie A'],
    'tier5': ['法甲', '法甲联赛', 'Ligue 1'],
    'tier6': ['中超', '中国超级联赛', 'Chinese Super League'],
}

MIN_ODDS = 1.01   # 最小有效赔率

MAX_ODDS = 100.0  # 最大有效赔率

RECENT_SEASONS = ['2026-27', '2025-26', '2024-25']

FEATURE_WEIGHTS = {
    'asian': 1.0,      # 亚盘让球
    'asian_odds': 0.8, # 亚盘赔率
    'total': 0.8,      # 大小球
    'total_odds': 0.6, # 大小球赔率
    'euro_home': 1.2,  # 主胜赔率
    'euro_draw': 0.5,  # 平局赔率
    'euro_away': 1.2,  # 客胜赔率
}

STANDARD_HANDICAPS = [-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, 
                      -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 
                      1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

STANDARD_TOTALS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]


def parse_handicap_value(value: str) -> Optional[float]:
    """解析让球值"""
    if not value or value.strip() == '':
        return None
    try:
        return round(float(value), 2)
    except ValueError:
        return None


def parse_odds_value(value: str) -> Optional[float]:
    """解析赔率值"""
    if not value or value.strip() == '':
        return None
    try:
        return round(float(value), 2)
    except ValueError:
        return None


def to_internal_asian(raw_asian: float, source: str = 'football_data') -> float:
    """
    将外部数据源的亚盘值转换为内部统一格式
    
    内部约定：
    - asian > 0: 主队让球
    - asian < 0: 客队让球
    
    参数：
        raw_asian: 原始让球值
        source: 数据源名称
    
    返回：
        转换后的内部格式让球值
    """
    if raw_asian is None:
        return None
    
    # Football-Data 的 AHh 字段符号与内部约定相反
    # AHh 为正数表示客队让球（负数为主队让球）
    # 转换为内部格式：正数为主队让球，负数为客队让球
    if source == 'football_data':
        return -raw_asian
    
    return raw_asian


def normalize_asian(handicap: float) -> float:
    """
    将亚盘值标准化到标准值列表
    
    参数：
        handicap: 原始让球值（已转换为内部格式）
    
    返回：
        标准化后的让球值
    """
    if handicap is None:
        return None
    
    # 找到最近的标准值
    nearest = min(STANDARD_ASIAN, key=lambda x: abs(x - handicap))
    return round(nearest, 2)


def normalize_ou(line: float) -> float:
    """
    将大小球值标准化到标准值列表
    
    参数：
        line: 原始大小球线
    
    返回：
        标准化后的大小球线
    """
    if line is None:
        return None
    
    # 找到最近的标准值
    nearest = min(STANDARD_OU, key=lambda x: abs(x - line))
    return round(nearest, 2)


def normalize_handicap_from_odds(home_odds: float, away_odds: float) -> float:
    """
    从赔率反推让球值（当没有直接让球数据时）
    
    参数：
        home_odds: 主队赔率
        away_odds: 客队赔率
    
    返回：
        反推的让球值（内部格式：正数为主队让球）
    """
    if home_odds is None or away_odds is None:
        return 0.0
    
    # 简化的让球反推公式
    odds_ratio = away_odds / home_odds
    # 赔率比 > 1 表示主队赔率低（更被看好），应该主队让球（正数）
    # 赔率比 < 1 表示客队赔率低（更被看好），应该客队让球（负数）
    handicap = (odds_ratio - 1) * 0.5
    return normalize_asian(handicap)


def market_score_key(asian: float, ou: float) -> str:
    """生成盘口组合的唯一键"""
    return f"{asian:.2f}_{ou:.2f}"


def implied_total_from_odds(over_odds: float, under_odds: float) -> float:
    """从大小球赔率反推期望总进球"""
    if over_odds is None or under_odds is None:
        return 2.5
    
    # 简化计算：赔率比对应的总进球
    try:
        p_over = 1.0 / over_odds
        p_under = 1.0 / under_odds
        total = p_over + p_under
        p_over_normalized = p_over / total
        
        # 从大球概率反推总进球（简化版）
        # P(over 2.5) = 1 - P(0) - P(1) - P(2)
        # 这里用线性近似
        return 2.5 + (p_over_normalized - 0.5) * 2.0
    except:
        return 2.5


def half_full_probs_from_records(asian: float, ou: float) -> Dict[str, float]:
    """
    获取指定盘口组合的半全场概率分布
    
    参数：
        asian: 亚盘让球
        ou: 大小球线
    
    返回：
        半全场概率字典
    """
    """
    获取指定盘口组合的半全场概率分布
    
    注意：此方法已废弃，不再生成伪半场数据。
    半全场概率应从 HalfTimeStatsDB 获取真实半场数据。
    
    参数：
        asian: 亚盘让球
        ou: 大小球线
    
    返回：
        空字典（半全场概率应从 HalfTimeStatsDB 获取）
    """
    # 返回空字典，强制使用真实半场数据
    # 伪数据（从全场比分推算半场比分）会严重影响准确率
    return {}


def _parse_record_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d-%m-%y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _recency_weight(record: 'MatchRecord', now: datetime = None) -> float:
    now = now or datetime.now()
    record_date = _parse_record_date(getattr(record, 'date', ''))
    if not record_date:
        season = str(getattr(record, 'season', '') or '')
        if any(season_key in season for season_key in RECENT_SEASONS):
            return 0.90
        return 0.70

    age_days = max(0, (now - record_date).days)
    if age_days <= 180:
        return 1.00
    if age_days <= 365:
        return 0.92
    if age_days <= 730:
        return 0.78
    if age_days <= 1095:
        return 0.62
    return 0.48


def parse_float(value: str) -> Optional[float]:
    """解析浮点数值"""
    if not value or value.strip() == '':
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def infer_handicap(home_odds: float, away_odds: float) -> float:
    """从欧赔反推亚盘让球"""
    if home_odds is None or away_odds is None:
        return 0.0
    
    # 简化公式：让球 ≈ (客胜赔率 - 主胜赔率) / (主胜赔率 + 客胜赔率) * 2
    try:
        ratio = away_odds / home_odds
        handicap = (1 - ratio) * 0.8
        # 限制范围
        return max(-2.0, min(2.0, round(handicap * 4) / 4))  # 标准化到0.25间隔
    except:
        return 0.0


def estimate_total(over_odds: float, under_odds: float) -> float:
    """从大小球赔率估算大小球线"""
    if over_odds is None or under_odds is None:
        return 2.5
    
    try:
        p_over = 1.0 / over_odds
        p_under = 1.0 / under_odds
        total_prob = p_over + p_under
        
        if total_prob <= 0:
            return 2.5
        
        p_over_norm = p_over / total_prob
        
        # 从大球概率反推总进球线
        # P(over 2.5) ≈ 0.5 对应 2.5球
        # P(over 2.5) ≈ 0.7 对应 3.0球
        # P(over 2.5) ≈ 0.3 对应 2.0球
        total_line = 2.5 + (p_over_norm - 0.5) * 3.0
        
        # 标准化到标准大小球线
        standard_lines = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
        return min(standard_lines, key=lambda x: abs(x - total_line))
    
    except:
        return 2.5


def filter_record(record, query_league: str = '', 
                  filter_friendly: bool = True, 
                  filter_recent_seasons: bool = True,
                  filter_odds_anomaly: bool = True) -> bool:
    """
    过滤记录（样本质量控制）
    
    参数：
        record: 待过滤的记录
        query_league: 查询的联赛（用于同联赛优先）
        filter_friendly: 是否过滤友谊赛
        filter_recent_seasons: 是否只保留近三赛季
        filter_odds_anomaly: 是否过滤赔率异常记录
    
    返回：
        True: 保留该记录
        False: 过滤掉该记录
    """
    # 1. 过滤友谊赛
    if filter_friendly and record.is_friendly:
        return False
    
    # 2. 过滤赔率异常记录
    if filter_odds_anomaly:
        odds_values = [
            record.euro_home, record.euro_draw, record.euro_away,
            record.asian_odds_home, record.asian_odds_away,
            record.total_over, record.total_under
        ]
        for odds in odds_values:
            if odds > 0:
                if odds < MIN_ODDS or odds > MAX_ODDS:
                    return False
    
    # 3. 过滤非近三赛季（如果指定了赛季）
    if filter_recent_seasons and record.season:
        if record.season not in RECENT_SEASONS:
            return False
    
    # 4. 检查结果是否存在
    if not record.result:
        return False
    
    return True


def extract_features(record) -> List[float]:
    """
    提取特征向量（已标准化）
    
    返回：[asian_norm, asian_odds_norm, total_norm, total_odds_norm, 
           euro_home_norm, euro_draw_norm, euro_away_norm]
    """
    features = []
    
    # 亚盘让球 (-3.0 ~ +3.0) → (-1 ~ +1)
    asian_norm = max(-1.0, min(1.0, record.asian / 3.0))
    features.append(asian_norm * FEATURE_WEIGHTS['asian'])
    
    # 亚盘赔率比值 (转换为概率后标准化)
    if record.asian_odds_home > 0 and record.asian_odds_away > 0:
        p_home = 1.0 / record.asian_odds_home
        p_away = 1.0 / record.asian_odds_away
        total = p_home + p_away
        if total > 0:
            p_home_norm = (p_home / total - 0.5) * 2  # (-1 ~ +1)
            features.append(p_home_norm * FEATURE_WEIGHTS['asian_odds'])
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 大小球 (1.0 ~ 5.0) → (-1 ~ +1)
    total_norm = max(-1.0, min(1.0, (record.total - 3.0) / 2.0))
    features.append(total_norm * FEATURE_WEIGHTS['total'])
    
    # 大小球赔率比值
    if record.total_over > 0 and record.total_under > 0:
        p_over = 1.0 / record.total_over
        p_under = 1.0 / record.total_under
        total = p_over + p_under
        if total > 0:
            p_over_norm = (p_over / total - 0.5) * 2
            features.append(p_over_norm * FEATURE_WEIGHTS['total_odds'])
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # 主胜赔率 (转换为概率的对数)
    if record.euro_home > 1.0:
        p_home = 1.0 / record.euro_home
        # 将概率压缩到 (-1 ~ +1) 范围
        home_norm = math.tanh((p_home - 0.5) * 4)
        features.append(home_norm * FEATURE_WEIGHTS['euro_home'])
    else:
        features.append(0.0)
    
    # 平局赔率
    if record.euro_draw > 1.0:
        p_draw = 1.0 / record.euro_draw
        draw_norm = math.tanh((p_draw - 0.33) * 6)
        features.append(draw_norm * FEATURE_WEIGHTS['euro_draw'])
    else:
        features.append(0.0)
    
    # 客胜赔率
    if record.euro_away > 1.0:
        p_away = 1.0 / record.euro_away
        away_norm = math.tanh((p_away - 0.5) * 4)
        features.append(away_norm * FEATURE_WEIGHTS['euro_away'])
    else:
        features.append(0.0)
    
    return features


def dynamic_k(asian: float) -> int:
    """
    根据让球大小动态调整K值
    
    深盘口样本少，用较小的K值
    平手盘样本多，用较大的K值
    """
    abs_asian = abs(asian)
    if abs_asian >= 1.5:
        return 200    # 深盘口
    elif abs_asian >= 1.0:
        return 500    # 中深盘
    elif abs_asian >= 0.5:
        return 1000   # 普通盘
    else:
        return 1500   # 平手盘


def league_tier(league_name: str) -> Optional[str]:
    """获取联赛所属层级"""
    for tier, leagues in LEAGUE_TIERS.items():
        for league in leagues:
            if league in league_name or league_name in league:
                return tier
    return None


def round_to_standard(handicap: float, total: float) -> Tuple[float, float]:
    """
    将盘口四舍五入到标准盘口
    
    参数：
        handicap: 亚盘让球
        total: 大小球线
    
    返回：
        (标准让球, 标准大小球)
    """
    # 找到最近的标准让球
    std_hcap = min(STANDARD_HANDICAPS, key=lambda x: abs(x - handicap))
    
    # 找到最近的标准大小球
    std_total = min(STANDARD_TOTALS, key=lambda x: abs(x - total))
    
    return std_hcap, std_total
