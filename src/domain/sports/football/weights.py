# -*- coding: utf-8 -*-
"""多预测源的动态融合权重。

**ML 权重由调用方注入**（判据 16）：它要读历史、要问 ML 模型的测试集样本数，
那些都是存储。领域层只负责「给定 ML 权重，其余三源怎么分」。
"""

from typing import Dict, Optional, Tuple

BASE_WEIGHTS = (0.5, 0.3, 0.2)
HIGH_CONFIDENCE_WEIGHTS = (0.7, 0.2, 0.1)
LOW_CONFIDENCE_WEIGHTS = (0.3, 0.4, 0.3)
HIGH_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.3


def _interpolate(low: Tuple[float, ...], high: Tuple[float, ...],
                 t: float) -> Tuple[float, ...]:
    return tuple(a + t * (b - a) for a, b in zip(low, high))


def confidence_weights(confidence: float) -> Tuple[float, float, float]:
    """置信度 → (市场, 球队, ELO) 三源权重。

    两端是平台期，中间在 0.3–0.5–0.7 之间分两段线性插值。
    """
    if confidence >= HIGH_CONFIDENCE:
        return HIGH_CONFIDENCE_WEIGHTS
    if confidence <= LOW_CONFIDENCE:
        return LOW_CONFIDENCE_WEIGHTS
    if confidence <= 0.5:
        return _interpolate(LOW_CONFIDENCE_WEIGHTS, BASE_WEIGHTS,
                            (confidence - LOW_CONFIDENCE) / 0.2)
    return _interpolate(BASE_WEIGHTS, HIGH_CONFIDENCE_WEIGHTS,
                        (confidence - 0.5) / 0.2)


def _make_room_for_ml(weights: Tuple[float, float, float],
                      ml_weight: float) -> Tuple[float, float, float]:
    """按比例缩减三源权重，把 ML 的份额腾出来。

    `ml_weight >= 1.0` 时**原样返回**——不是笔误，是既有行为：
    权重和会大于 1，由下游归一化兜住。
    """
    if not 0 < ml_weight < 1.0:
        return weights
    total = sum(weights)
    if total <= 0:
        return weights
    scale = (1.0 - ml_weight) / total
    return tuple(w * scale for w in weights)


def get_dynamic_weights(confidence: float = 0.5,
                        ml_weight: float = 0.0) -> Tuple[float, float, float, float]:
    """(市场, 球队, ELO, ML) 四源权重。"""
    market_w, team_w, elo_w = _make_room_for_ml(confidence_weights(confidence),
                                                ml_weight)
    return market_w, team_w, elo_w, ml_weight


def fuse_predictions(market_pred: Dict[str, float],
                     team_pred: Dict[str, float],
                     elo_pred: Dict[str, float],
                     ml_pred: Optional[Dict[str, float]] = None,
                     weights: Tuple[float, float, float, float] = None,
                     confidence: float = 0.5) -> Dict[str, float]:
    """按权重融合各源的比分分布并归一化。

    `weights` 不给就按 `confidence` 现算——**ML 权重取 0**，
    因为领域层拿不到判定 ML 资格所需的历史。
    """
    market_w, team_w, elo_w, ml_w = weights or get_dynamic_weights(confidence)

    all_scores = set(market_pred) | set(team_pred) | set(elo_pred)
    if ml_pred:
        all_scores |= set(ml_pred)

    fused = {
        score: (market_w * market_pred.get(score, 0.0)
                + team_w * team_pred.get(score, 0.0)
                + elo_w * elo_pred.get(score, 0.0)
                + ml_w * (ml_pred.get(score, 0.0) if ml_pred else 0.0))
        for score in all_scores
    }

    total = sum(fused.values())
    return {k: v / total for k, v in fused.items()} if total > 0 else fused
