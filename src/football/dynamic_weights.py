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

权重怎么分已迁至 `src.domain.sports.football.weights`；这里只剩一件事——
**去问 ML 有没有资格参与融合**，那要读历史、要问 ML 模型的测试集样本数。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..domain.sports.football.weights import (  # noqa: F401
    BASE_WEIGHTS,
    HIGH_CONFIDENCE,
    HIGH_CONFIDENCE_WEIGHTS,
    LOW_CONFIDENCE,
    LOW_CONFIDENCE_WEIGHTS,
    _interpolate,
    _make_room_for_ml,
    confidence_weights,
)
from ..domain.sports.football import weights as _w


def _ml_fusion_weight() -> float:
    """ML 当前能拿到的融合权重；拿不到资格或读不到历史时为 0。"""
    try:
        from .result_sync import (get_history, check_ml_fusion_eligibility,
                                  get_ml_fusion_weight)
        import src.football.ml as ml_module

        ml_module.load_trained_ml_model()
        metadata = ml_module._trained_ml_metadata
        test_set_samples = metadata.get('test_count', 0) if metadata else 0

        eligibility = check_ml_fusion_eligibility(
            get_history().get_ml_evaluation_stats(), test_set_samples)
        if not eligibility['eligible']:
            return 0.0
        return get_ml_fusion_weight(True, eligibility['shadow_samples'], 0.0)
    except Exception:
        return 0.0


def get_dynamic_weights(confidence: float = 0.5,
                        match_data: Optional[Dict] = None
                        ) -> Tuple[float, float, float, float]:
    """(市场, 球队, ELO, ML) 四路权重。

    `match_data` 从 `MetaWeightModel` 删除起就**没有任何读取**，
    保留只为不动既有调用方的签名。
    """
    return _w.get_dynamic_weights(confidence, _ml_fusion_weight())


def fuse_predictions(market_pred: Dict[str, float],
                     team_pred: Dict[str, float],
                     elo_pred: Dict[str, float],
                     ml_pred: Optional[Dict[str, float]] = None,
                     confidence: float = 0.5,
                     match_data: Optional[Dict] = None) -> Dict[str, float]:
    """按动态权重融合各源的比分分布。"""
    return _w.fuse_predictions(market_pred, team_pred, elo_pred, ml_pred,
                               weights=get_dynamic_weights(confidence, match_data))
