#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透规则+ML融合预测"""

import math
from collections import Counter
from typing import Dict, List, Tuple

from ..common.logger import setup_logger
from .config import RANDOM_BASELINE

log = setup_logger('lottery')

def fuse_rule_ml(rule_front_ranked: List[Tuple], rule_back_ranked: List[Tuple],
                 ml_prediction: Dict, rule_weight: float = 0.55,
                 ml_weight: float = 0.45) -> Dict:
    """融合规则排名模型和ML模型的预测结果

    与 lottery3d 的融合逻辑一致: 基于回测表现的动态权重融合。

    Args:
        rule_front_ranked: 规则模型前区排名 [(num, score, features), ...]
        rule_back_ranked: 规则模型后区排名 [(num, score, features), ...]
        ml_prediction: ML模型预测结果 {front_probs, back_probs, ...}
        rule_weight: 规则模型权重
        ml_weight: ML模型权重

    Returns:
        {
            'front_ranked': 融合后前区排名 [(num, fused_score, tag), ...],
            'back_ranked': 融合后后区排名 [(num, fused_score, tag), ...],
            'front_top12': 前区Top12推荐,
            'back_top6': 后区Top6推荐,
        }
    """
    total_w = rule_weight + ml_weight
    rule_w = rule_weight / total_w
    ml_w = ml_weight / total_w

    # 归一化规则得分到 0-1
    rule_front_scores = {}
    if rule_front_ranked:
        max_s = max(s for _, s, _ in rule_front_ranked)
        min_s = min(s for _, s, _ in rule_front_ranked)
        s_range = max_s - min_s if max_s > min_s else 1.0
        for num, score, _ in rule_front_ranked:
            rule_front_scores[num] = (score - min_s) / s_range
    else:
        rule_front_scores = {}

    rule_back_scores = {}
    if rule_back_ranked:
        max_s = max(s for _, s, _ in rule_back_ranked)
        min_s = min(s for _, s, _ in rule_back_ranked)
        s_range = max_s - min_s if max_s > min_s else 1.0
        for num, score, _ in rule_back_ranked:
            rule_back_scores[num] = (score - min_s) / s_range
    else:
        rule_back_scores = {}

    # ML概率
    ml_front = ml_prediction.get('front_probs', {}) if ml_prediction else {}
    ml_back = ml_prediction.get('back_probs', {}) if ml_prediction else {}

    # 前区融合
    all_front = set(rule_front_scores.keys()) | set(ml_front.keys())
    front_fused = []
    for num in sorted(all_front):
        r_score = rule_front_scores.get(num, 0.0)
        m_score = ml_front.get(num, 0.0)
        fused = r_score * rule_w + m_score * ml_w
        in_rule = num in rule_front_scores
        in_ml = num in ml_front
        if in_rule and in_ml:
            tag = 'high_confidence'
            fused += 0.1  # 双方一致加分
        elif in_rule:
            tag = 'rule_preferred'
        elif in_ml:
            tag = 'ml_preferred'
        else:
            tag = 'other'
        front_fused.append((num, round(fused, 4), tag, in_rule, in_ml))

    front_fused.sort(key=lambda x: -x[1])

    # 后区融合
    all_back = set(rule_back_scores.keys()) | set(ml_back.keys())
    back_fused = []
    for num in sorted(all_back):
        r_score = rule_back_scores.get(num, 0.0)
        m_score = ml_back.get(num, 0.0)
        fused = r_score * rule_w + m_score * ml_w
        in_rule = num in rule_back_scores
        in_ml = num in ml_back
        if in_rule and in_ml:
            tag = 'high_confidence'
            fused += 0.1
        elif in_rule:
            tag = 'rule_preferred'
        elif in_ml:
            tag = 'ml_preferred'
        else:
            tag = 'other'
        back_fused.append((num, round(fused, 4), tag, in_rule, in_ml))

    back_fused.sort(key=lambda x: -x[1])

    return {
        'front_ranked': front_fused,
        'back_ranked': back_fused,
        'front_top12': [num for num, _, _, _, _ in front_fused[:12]],
        'back_top6': [num for num, _, _, _, _ in back_fused[:6]],
        'rule_weight': rule_w,
        'ml_weight': ml_w,
    }


def compute_fusion_weights(rule_backtest: Dict, ml_backtest: Dict) -> Tuple[float, float]:
    """基于回测表现计算融合权重

    Args:
        rule_backtest: 规则模型回测结果
        ml_backtest: ML模型回测结果

    Returns:
        (rule_weight, ml_weight)
    """
    if not ml_backtest or ml_backtest.get('error'):
        return (1.0, 0.0)

    # 使用前区≥2命中率作为核心指标（兼容顶层字段与 rates 嵌套）
    rule_front_ge2 = (
        rule_backtest.get('front_ge2_rate')
        or (rule_backtest.get('rates') or {}).get('front_ge2_rate')
        or 0
    )
    ml_front_ge2 = ml_backtest.get('front_ge2_rate') or 0
    baseline = RANDOM_BASELINE.get('front_ge2_rate', 0.1389)

    if ml_front_ge2 <= 0 and rule_front_ge2 <= 0:
        return (0.70, 0.30)

    # 相对随机基准的 lift；ML 未超过基准时压低权重，避免噪声模型拖累
    rule_lift = max(rule_front_ge2 - baseline, 0.0)
    ml_lift = max(ml_front_ge2 - baseline, 0.0)
    if ml_lift <= 0 and rule_lift <= 0:
        return (0.75, 0.25)
    if ml_lift <= 0:
        return (0.85, 0.15)

    total = rule_lift + ml_lift
    # v4.1优化: 规则底权重从0.55降至0.35。诊断(diagnose_dlt_fusion.py)显示
    # 规则0.35/ML0.65时前区ge4=7.3%(vs 0.55/0.45的6.0%)，ML前区命中≥4码能力更强。
    rule_w = max(0.35, rule_lift / total)
    ml_w = 1.0 - rule_w
    return (round(rule_w, 2), round(ml_w, 2))

