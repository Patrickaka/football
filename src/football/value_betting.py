#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赔率价值分析模块
=================

功能：
1. 计算比分的期望值和价值
2. 识别存在价值的比分
3. 根据价值调整推荐权重

职业博彩模型基本都这样干。
"""

import math
from typing import Dict, List, Tuple, Optional

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
                       threshold: float = 0.02) -> List[Tuple[str, float, float]]:
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
