# -*- coding: utf-8 -*-
"""把候选比分依次锚定到市场信号，最后落到一个自洽的联合状态。

顺序是有讲究的，改动前先读注释：**收盘 1X2 定方向的边际、大小球盘口定
进球均值、亚盘/大小球的公平价再施加软结算约束**，下游所有
SPF / RQSPF / 比分 / 进球输出都从这个最终矩阵派生。

**生产历史的档案由调用方注入**（判据 16）：`get_runtime_history_profile`
要读生产历史、还带缓存，那是存储。不注入就退化成不校准——
和迁移前"读不到历史"时的行为一致。

每一步都各自 try：某一步失败不该让整条流水线塌掉，`meta` 里留下
`{'applied': False, 'reason': ...}` 供报告层展示。
"""

import logging
from typing import Dict, List, Optional, Tuple

from .calibration_history import apply_history_calibration
from .scoring import (
    _anchor_score_candidates_to_1x2,
    _anchor_score_candidates_to_goal_mean,
    _apply_joint_market_state,
)

log = logging.getLogger('domain.football.market_anchoring')


def anchor_candidates_to_market(candidates: List,
                                total: Dict,
                                euro: Dict,
                                asian: Dict,
                                history_profile: Optional[Dict] = None
                                ) -> Tuple[List, Dict]:
    """依次锚定候选比分，返回 `(候选, meta)`。

    `meta` 的五个键与迁移前逐字一致，报告层按名字取用：
    `score_goal_anchor` / `production_history_calibration` /
    `outcome_market_anchor` / `final_score_goal_anchor` / `joint_market_state`。
    """
    meta: Dict = {}
    try:
        candidates, score_goal_anchor = _anchor_score_candidates_to_goal_mean(
            candidates, total
        )
        meta['score_goal_anchor'] = score_goal_anchor
        if score_goal_anchor.get('applied'):
            log.debug(
                "score goal mean anchored: %.3f -> %.3f (target %.3f)",
                score_goal_anchor['expected_before'],
                score_goal_anchor['expected_after'],
                score_goal_anchor['target'],
            )
    except Exception as e:
        meta['score_goal_anchor'] = {'applied': False, 'reason': str(e)}
        log.warning(f"score goal mean anchor failed: {e}")

    # Production-history correction runs after the market anchor.  It is
    # deliberately shrunk by effective sample size and capped, so a short hot
    # streak cannot overwhelm the current match's odds and total-goal line.
    try:
        candidates, history_adjustment = apply_history_calibration(
            candidates, history_profile or {})
        meta['production_history_calibration'] = history_adjustment
        if history_adjustment.get('applied'):
            log.debug(
                "production history calibrated: n=%s beta=%.4f goals %.3f -> %.3f",
                history_adjustment.get('sample_count'),
                history_adjustment.get('goal_beta', 0.0),
                history_adjustment.get('expected_goals_before', 0.0),
                history_adjustment.get('expected_goals_after', 0.0),
            )
    except Exception as e:
        meta['production_history_calibration'] = {'applied': False, 'reason': str(e)}
        log.warning(f"production history calibration failed: {e}")

    # Multi-stage score corrections can unintentionally move aggregate H/D/A
    # mass away from the efficient closing market.  Re-anchor those marginals
    # before the final goal-mean repair; the latter preserves 1X2 by design.
    try:
        candidates, outcome_market_anchor = _anchor_score_candidates_to_1x2(
            candidates, euro
        )
        meta['outcome_market_anchor'] = outcome_market_anchor
    except Exception as e:
        meta['outcome_market_anchor'] = {'applied': False, 'reason': str(e)}
        log.warning(f"score 1X2 market anchor failed: {e}")

    # History calibration is useful for residual bias, but it must not undo the
    # current O/U market's total-goal signal.  Make the market anchor the final
    # score-distribution transform so high-total matches retain their 4+ tail.
    try:
        candidates, final_score_goal_anchor = _anchor_score_candidates_to_goal_mean(
            candidates, total
        )
        meta['final_score_goal_anchor'] = final_score_goal_anchor
        if final_score_goal_anchor.get('applied'):
            log.debug(
                "final score goal mean anchored after history: %.3f -> %.3f (target %.3f)",
                final_score_goal_anchor['expected_before'],
                final_score_goal_anchor['expected_after'],
                final_score_goal_anchor['target'],
            )
    except Exception as e:
        meta['final_score_goal_anchor'] = {'applied': False, 'reason': str(e)}
        log.warning(f"final score goal mean anchor failed: {e}")

    # The final distribution is a single market-consistent state: closing 1X2
    # sets the outcome marginal above, the O/U line sets the goal mean, and the
    # fair Asian/O-U prices now apply the validated soft settlement constraint.
    # All downstream SPF/RQSPF/score/goal outputs are derived from this matrix.
    try:
        candidates, joint_market_adjustment = _apply_joint_market_state(
            candidates, asian, euro, total
        )
        meta['joint_market_state'] = joint_market_adjustment
    except Exception as e:
        meta['joint_market_state'] = {'applied': False, 'reason': str(e)}
        log.warning(f"joint market constraint failed: {e}")

    dixon_coles_result = None

    return candidates, meta
