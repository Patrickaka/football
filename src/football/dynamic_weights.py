#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态权重模块 - 根据置信度调整融合权重
========================================

功能：
1. 根据模型置信度动态调整各预测源的权重
2. 置信度高时：更依赖市场数据
3. 置信度低时：更依赖球队实力和ELO

默认权重：MARKET=0.5, TEAM=0.3, ELO=0.2

动态调整：
- 置信度高（>0.7）：MARKET=0.7, TEAM=0.2, ELO=0.1
- 置信度低（<0.3）：MARKET=0.3, TEAM=0.4, ELO=0.3
"""

from typing import Dict, Tuple

def get_dynamic_weights(confidence: float) -> Tuple[float, float, float]:
    """
    根据置信度获取动态权重
    
    参数：
        confidence: 置信度（0~1）
    
    返回：
        (market_weight, team_weight, elo_weight)
    """
    # 默认权重
    base_market = 0.5
    base_team = 0.3
    base_elo = 0.2
    
    # 目标权重
    high_market = 0.7
    high_team = 0.2
    high_elo = 0.1
    
    low_market = 0.3
    low_team = 0.4
    low_elo = 0.3
    
    if confidence >= 0.7:
        # 高置信度：更依赖市场
        return high_market, high_team, high_elo
    elif confidence <= 0.3:
        # 低置信度：更依赖球队实力
        return low_market, low_team, low_elo
    else:
        # 中间值：线性插值
        if confidence <= 0.5:
            # 从低到中
            t = (confidence - 0.3) / 0.2
            market = low_market + t * (base_market - low_market)
            team = low_team + t * (base_team - low_team)
            elo = low_elo + t * (base_elo - low_elo)
        else:
            # 从中到高
            t = (confidence - 0.5) / 0.2
            market = base_market + t * (high_market - base_market)
            team = base_team + t * (high_team - base_team)
            elo = base_elo + t * (high_elo - base_elo)
        
        return market, team, elo

def fuse_predictions(market_pred: Dict[str, float],
                    team_pred: Dict[str, float],
                    elo_pred: Dict[str, float],
                    confidence: float = 0.5) -> Dict[str, float]:
    """
    根据动态权重融合多个预测源
    
    参数：
        market_pred: 市场数据预测
        team_pred: 球队实力预测
        elo_pred: ELO预测
        confidence: 置信度
    
    返回：
        融合后的预测
    """
    market_w, team_w, elo_w = get_dynamic_weights(confidence)
    
    # 获取所有可能的比分
    all_scores = set(market_pred.keys()) | set(team_pred.keys()) | set(elo_pred.keys())
    
    fused = {}
    for score in all_scores:
        m_prob = market_pred.get(score, 0.0)
        t_prob = team_pred.get(score, 0.0)
        e_prob = elo_pred.get(score, 0.0)
        
        # 加权融合
        fused[score] = market_w * m_prob + team_w * t_prob + elo_w * e_prob
    
    # 归一化
    total = sum(fused.values())
    if total > 0:
        return {k: v / total for k, v in fused.items()}
    return fused
