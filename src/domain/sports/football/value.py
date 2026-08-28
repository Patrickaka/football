# -*- coding: utf-8 -*-
"""赔率价值分析与盘口聚类先验融合。

纯计算——不读全局配置、不碰存储与时钟。聚类库的读写留在
`src/football/market_clustering.py`；这里只有「给定先验怎么融合」。
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

log = logging.getLogger('domain.football.value')

def calculate_value(estimated_prob: float, odds: float) -> float:
    """
    计算赔率价值
    
    参数：
        estimated_prob: 模型估计的概率
        odds: 市场赔率
    
    返回：
        价值分数（>0表示存在价值）
    """
    if odds <= 1.0:
        return 0.0
    
    # 市场隐含概率（考虑抽水）
    implied_prob = 1.0 / odds
    
    # 价值 = 估计概率 - 隐含概率
    value = estimated_prob - implied_prob
    
    return value


def calculate_ev(estimated_prob: float, odds: float) -> float:
    """
    计算期望值（Expected Value）
    
    参数：
        estimated_prob: 模型估计的概率
        odds: 市场赔率
    
    返回：
        期望值（每单位投注的期望收益）
    """
    if odds <= 1.0:
        return 0.0
    
    # EV = P(win) * (odds - 1) - P(lose) * 1
    ev = estimated_prob * (odds - 1) - (1 - estimated_prob) * 1
    
    return ev


def adjust_by_value(predictions: Dict[str, float], 
                   market_odds: Dict[str, float],
                   value_weight: float = 0.3) -> Dict[str, float]:
    """
    根据赔率价值调整预测概率权重
    
    参数：
        predictions: {比分: 概率}
        market_odds: {比分: 赔率}
        value_weight: 价值权重（0~1）
    
    返回：
        调整后的概率字典
    """
    adjusted = {}
    total_weight = 0.0
    
    for score, prob in predictions.items():
        odds = market_odds.get(score, 1.0)
        
        if odds > 1.0:
            # 计算价值
            value = calculate_value(prob, odds)
            ev = calculate_ev(prob, odds)
            
            # 根据价值调整权重
            # 价值越高，权重越大
            value_bonus = 1.0 + value * 10 * value_weight
            
            # EV为正时增加权重
            if ev > 0:
                value_bonus *= (1 + ev * 5)
            
            adjusted[score] = prob * value_bonus
        else:
            adjusted[score] = prob
        
        total_weight += adjusted[score]
    
    # 归一化
    if total_weight > 0:
        return {k: v / total_weight for k, v in adjusted.items()}
    return predictions


def identify_value_bets(predictions: Dict[str, float], 
                       market_odds: Dict[str, float],
                       threshold: float = 0.5) -> List[Tuple[str, float, float]]:
    """
    识别存在价值的投注
    
    参数：
        predictions: {比分: 概率}
        market_odds: {比分: 赔率}
        threshold: 价值阈值
    
    返回：
        价值投注列表 [(比分, 价值, EV), ...]
    """
    value_bets = []
    
    for score, prob in predictions.items():
        odds = market_odds.get(score)
        if odds:
            value = calculate_value(prob, odds)
            ev = calculate_ev(prob, odds)
            
            if value >= threshold:
                value_bets.append((score, value, ev))
    
    # 按价值排序
    value_bets.sort(key=lambda x: -x[1])
    
    return value_bets


def fuse_with_prior(poisson_probs: Dict[str, float],
                   handicap: float, total: float,
                   prior_weight: float = 0.9, prior=None) -> Dict[str, float]:
    """
    将泊松概率与盘口先验融合
    
    参数：
        poisson_probs: 泊松模型预测概率
        handicap: 亚盘让球
        total: 大小球线
        prior_weight: 先验权重
    
    返回：
        融合后的概率
    """
    # **prior 由调用方传入**：取先验要读聚类库，那是存储（判据 16）。
    # 适配层的 `MarketCluster.fuse_with_prior` 负责先 `get_prior` 再调这里。
    if prior is None:
        return poisson_probs, {'applied': False, 'reason': 'no_prior'}
    
    if not prior:
        # 没有先验数据，直接返回泊松概率
        return poisson_probs
    
    # 获取所有比分
    all_scores = set(poisson_probs.keys()) | set(prior.keys())
    
    fused = {}
    for score in all_scores:
        p_poisson = poisson_probs.get(score, 0.0)
        p_prior = prior.get(score, 0.0)
        
        # 加权融合
        fused[score] = (1 - prior_weight) * p_poisson + prior_weight * p_prior
    
    # 归一化
    total_prob = sum(fused.values())
    if total_prob > 0:
        return {k: v / total_prob for k, v in fused.items()}
    return poisson_probs
