#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动态权重：按置信度与 ML 资格分配市场/球队/ELO/ML 四路权重

原本这里还有一个 478 行的 `MetaWeightModel`（xgboost/lightgbm/sklearn 学一个
权重回归器），2026-08-28 删除：把它接到 `get_dynamic_weights` 上的唯一函数
`init_meta_model()` **在全仓没有任何调用者**，也不在 `src/football/__init__.py`
的导出里，所以 `get_dynamic_weights._meta_model` 永远不存在、那条分支恒为假。
判据 9「代码本身不可达」——留着一段任何测试都保护不了的代码比没有它更糟，
它看起来像道防线。依据全文见 docs/superpowers/notes/2026-08-football-活死清单.md §四。

删掉它顺带把三个 ML 库的依赖也去掉了：本模块现在只用标准库。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def get_dynamic_weights(confidence: float = 0.5, match_data: Optional[Dict] = None) -> Tuple[float, float, float, float]:
    """
    获取动态权重（兼容旧接口，新增ML权重）
    
    参数：
        confidence: 置信度（0~1）- 备用方案
        match_data: 比赛特征数据 - 用于Meta模型
        
    返回：
        (market_weight, team_weight, elo_weight, ml_weight)
    """
    # 基于置信度的固定规则
    base_market, base_team, base_elo = 0.5, 0.3, 0.2
    high_market, high_team, high_elo = 0.7, 0.2, 0.1
    low_market, low_team, low_elo = 0.3, 0.4, 0.3
    
    # 检查 ML 是否有资格参与融合
    try:
        from .result_sync import get_history, check_ml_fusion_eligibility, get_ml_fusion_weight
        import src.football.ml as ml_module
        
        # 确保模型已加载并获取测试集样本数
        ml_module.load_trained_ml_model()
        test_set_samples = ml_module._trained_ml_metadata.get('test_count', 0) if ml_module._trained_ml_metadata else 0
        
        history = get_history()
        ml_stats = history.get_ml_evaluation_stats()
        eligibility = check_ml_fusion_eligibility(ml_stats, test_set_samples)
        
        if eligibility['eligible']:
            shadow_samples = eligibility['shadow_samples']
            ml_weight = get_ml_fusion_weight(True, shadow_samples, 0.0)
        else:
            ml_weight = 0.0
    except Exception:
        ml_weight = 0.0  # 默认无ML权重
    
    # 根据置信度确定基础权重
    if confidence >= 0.7:
        market_w, team_w, elo_w = high_market, high_team, high_elo
    elif confidence <= 0.3:
        market_w, team_w, elo_w = low_market, low_team, low_elo
    else:
        if confidence <= 0.5:
            t = (confidence - 0.3) / 0.2
            market_w = low_market + t * (base_market - low_market)
            team_w = low_team + t * (base_team - low_team)
            elo_w = low_elo + t * (base_elo - low_elo)
        else:
            t = (confidence - 0.5) / 0.2
            market_w = base_market + t * (high_market - base_market)
            team_w = base_team + t * (high_team - base_team)
            elo_w = base_elo + t * (high_elo - base_elo)
    
    # 如果有ML权重，需要从其他权重中按比例扣除
    if ml_weight > 0 and ml_weight < 1.0:
        # 计算其他权重的总和
        base_total = market_w + team_w + elo_w
        if base_total > 0:
            # 按比例缩减其他权重，腾出ML权重空间
            scale_factor = (1.0 - ml_weight) / base_total
            market_w *= scale_factor
            team_w *= scale_factor
            elo_w *= scale_factor
    
    return market_w, team_w, elo_w, ml_weight


def fuse_predictions(market_pred: Dict[str, float],
                    team_pred: Dict[str, float],
                    elo_pred: Dict[str, float],
                    ml_pred: Optional[Dict[str, float]] = None,
                    confidence: float = 0.5,
                    match_data: Optional[Dict] = None) -> Dict[str, float]:
    """
    根据动态权重融合多个预测源（支持ML预测）
    
    参数：
        market_pred: 市场数据预测
        team_pred: 球队实力预测
        elo_pred: ELO预测
        ml_pred: 机器学习预测（可选）
        confidence: 置信度（备用）
        match_data: 比赛特征数据（用于Meta模型）
    
    返回：
        融合后的预测
    """
    market_w, team_w, elo_w, ml_w = get_dynamic_weights(confidence, match_data)
    
    # 获取所有可能的比分
    all_scores = set(market_pred.keys()) | set(team_pred.keys()) | set(elo_pred.keys())
    if ml_pred:
        all_scores |= set(ml_pred.keys())
    
    fused = {}
    for score in all_scores:
        m_prob = market_pred.get(score, 0.0)
        t_prob = team_pred.get(score, 0.0)
        e_prob = elo_pred.get(score, 0.0)
        ml_prob = ml_pred.get(score, 0.0) if ml_pred else 0.0
        
        # 加权融合
        fused[score] = market_w * m_prob + team_w * t_prob + elo_w * e_prob + ml_w * ml_prob
    
    # 归一化
    total = sum(fused.values())
    if total > 0:
        return {k: v / total for k, v in fused.items()}
    return fused
