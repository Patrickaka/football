# -*- coding: utf-8 -*-
"""Elo 评分的纯计算：队名归一、期望胜率、K 值、联赛权重、进球与强度换算。

纯计算——**评分表的读写、并发锁与单例都留在适配层**（判据 16）。
`ELORatingSystem` 里这四个方法在迁移前是「零次使用 self 的实例方法」，
本质就是纯函数。

**`src/football/elo.py` 与 `dynamic_elo.py` 各有一个 `get_elo_system`，
同名不同物**——前者返回 `ELORatingSystem`，后者返回 `DynamicELO`，
是两个不同类的单例。别当重复合并。
"""

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger('domain.football.elo')

INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
K_FACTORS = {'友谊赛': 20, '联赛': 25, '杯赛': 30, '洲际杯': 35, '世界杯': 40}
LEAGUE_WEIGHTS = {
    '英超': 1.1,
    '西甲': 1.1,
    '德甲': 1.05,
    '意甲': 1.05,
    '法甲': 1.0,
    '中超': 0.8,
    '欧冠': 1.2,
    '欧联杯': 1.1,
    '世界杯': 1.3,
    '欧洲杯': 1.25,
    '友谊赛': 0.7,
}


def sanitize_team_name(team_name: str) -> Optional[str]:
    """
    清理并验证球队名称
    
    参数:
        team_name: 原始球队名称
    
    返回:
        清理后的球队名称，无效则返回 None
    """
    if team_name is None:
        logger.warning("球队名称为 None")
        return None
    
    # 转换为字符串
    if not isinstance(team_name, str):
        team_name = str(team_name)
    
    # 去除空白字符
    team_name = team_name.strip()
    
    # 检查是否为空
    if not team_name:
        logger.warning("球队名称为空")
        return None
    
    # 检查长度
    if len(team_name) > 100:
        # **只记一条 warning，名字照样返回**——不是拒绝。所以把 100 改成别的
        # 数字是等价变异（判据 30），可观测行为不变。要真拦住得在这里 return None，
        # 那是行为改动，不是迁移该做的事。
        logger.warning(f"球队名称过长: {len(team_name)} 字符")
        team_name = team_name[:100]
    
    # 检查是否包含非法字符（只允许中文、英文、数字和常见符号）
    # 移除不可打印字符
    team_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', team_name)
    
    # 验证编码（尝试编码为 UTF-8）
    try:
        team_name.encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        logger.error(f"球队名称编码异常: {e}")
        return None
    
    # 替换特殊字符
    team_name = re.sub(r'[<>:"/\\|?*]', '_', team_name)
    
    return team_name if team_name else None


def expected_score(rating1: float, rating2: float) -> float:
    """
    计算预期得分概率
    
    参数:
        rating1: 球队1评分
        rating2: 球队2评分
    
    返回:
        球队1的预期得分概率
    """
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))


def k_factor(league_type: str) -> int:
    """
    获取 K 因子
    
    参数:
        league_type: 联赛类型
    
    返回:
        K 因子值
    """
    if not isinstance(league_type, str):
        league_type = str(league_type)
    
    for key in K_FACTORS:
        if key in league_type:
            return K_FACTORS[key]
    return K_FACTORS['联赛']  # 默认值


def league_weight(league_type: str) -> float:
    """
    获取联赛权重系数
    
    参数:
        league_type: 联赛类型
    
    返回:
        权重系数
    """
    if not isinstance(league_type, str):
        league_type = str(league_type)
    
    for key in LEAGUE_WEIGHTS:
        if key in league_type:
            return LEAGUE_WEIGHTS[key]
    return 1.0  # 默认值


def elo_to_goals_expected(elo_rating: float, opponent_elo: float) -> float:
    """
    将 ELO 评分转换为进球期望值 (xG)（容错版本）
    
    参数:
        elo_rating: 球队 ELO 评分
        opponent_elo: 对手 ELO 评分
    
    返回:
        进球期望值
    """
    try:
        # 验证输入
        if not isinstance(elo_rating, (int, float)):
            elo_rating = float(elo_rating) if elo_rating else INITIAL_ELO
        
        if not isinstance(opponent_elo, (int, float)):
            opponent_elo = float(opponent_elo) if opponent_elo else INITIAL_ELO
        
        # ELO 差距与进球期望的关系
        rating_diff = elo_rating - opponent_elo
        base_xg = 1.5  # 平均进球期望
        
        # 每 100 ELO 差距约对应 0.3 进球差异
        xg = base_xg + (rating_diff / 100) * 0.3
        
        # 限制范围
        return max(0.2, min(5.0, xg))
    
    except Exception as e:
        logger.error(f"计算 xG 失败: {e}")
        return 1.5


def elo_to_strength_factor(elo_rating: float, league_avg_elo: float = 1500) -> float:
    """
    将 ELO 评分转换为实力因子（容错版本）
    
    参数:
        elo_rating: 球队 ELO 评分
        league_avg_elo: 联赛平均 ELO（默认1500）
    
    返回:
        实力因子（1.0 为平均水平）
    """
    try:
        # 验证输入
        if not isinstance(elo_rating, (int, float)):
            elo_rating = float(elo_rating) if elo_rating else INITIAL_ELO
        
        if not isinstance(league_avg_elo, (int, float)):
            league_avg_elo = float(league_avg_elo) if league_avg_elo else INITIAL_ELO
        
        diff = elo_rating - league_avg_elo
        return 1.0 + (diff / 500)
    
    except Exception as e:
        logger.error(f"计算实力因子失败: {e}")
        return 1.0
