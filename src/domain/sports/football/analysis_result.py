# -*- coding: utf-8 -*-
"""把一场比赛的各路分析结果拼成对外的 `result`。

**这是输出契约**：字段名、嵌套形状、`accuracy_gate` 的挂载位置，
下游报告层与 HTTP 层都按这些名字取用（判据 12：展示字段没人盯着，
一旦改名只会在页面上悄悄空掉）。

**没有存储、没有抓取**：四十个入参都是上游算好的结果，这里只做整形。
`professional_evidence` 那一步调的是领域层的 `readiness`，
迁移前它是指向适配层的延迟 import。
"""

import logging
from typing import Any, Dict, List, Optional

from .accuracy_gate import build_accuracy_gate, build_total_goals_gate
from .readiness import build_match_evidence_profile

log = logging.getLogger('domain.football.analysis_result')


def _native(value):
    """把 numpy 的标量与数组换成原生 Python 类型。

    Dixon-Coles 的比分矩阵在**计算时**就该是 numpy——那是它该有的样子。
    但它一旦进入对外的 `result`，就得是原生类型：JSON 序列化器不认识
    `ndarray`，FastAPI 的 `jsonable_encoder` 撞上它直接 500。

    转换放在**输出契约的构造点**，而不是响应层：响应层兜底等于承认
    领域层可以吐任何东西，那种兜底一旦漏掉一条路由就是一个 500。
    """
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    if hasattr(value, 'item') and not isinstance(value, (str, bytes, bytearray, bool)):
        return value.item()
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def build_analysis_result(*,
                            asian,
                            calibration_effect,
                            candidates,
                            confidence,
                            dixon_coles_result,
                            euro,
                            euro_asian_dev,
                            goal_count_result,
                            goal_dist_after_calibration,
                            goal_dist_before_calibration,
                            half_full_time,
                            joint_anomaly,
                            lam_away,
                            lam_home,
                            league_profile,
                            live_context,
                            live_context_quality,
                            lottery,
                            market_change_result,
                            match,
                            meta,
                            ml_result,
                            model_status,
                            model_weights,
                            probability_rank,
                            production_spf_policy,
                            recommend,
                            recommend_rank,
                            risk,
                            settlement,
                            similar_market_detail,
                            similar_market_result,
                            single_odds,
                            steam_result,
                            team,
                            top_scores,
                            total,
                            upset,
                            value_bets) -> Dict[str, Any]:
    """组装最终的分析结果。入参一律用关键字——四十个位置参数没法读。"""
    lottery['accuracy_gate'] = build_accuracy_gate(
        lottery,
        confidence=confidence,
        anomaly={
            'joint_water': joint_anomaly,
            'euro_asian_deviation': euro_asian_dev,
        },
        upset=upset,
        league=match.get('league'),
        production_spf_policy=production_spf_policy,
    )
    total_goals_gate = build_total_goals_gate(
        total,
        league=match.get('league'),
        goal_count=goal_count_result,
    )
    lottery['accuracy_gate']['total_goals'] = total_goals_gate
    if goal_count_result is not None:
        goal_count_result['accuracy_gate'] = total_goals_gate

    result = {
        'match': {k: match.get(k) for k in (
            'home', 'away', 'league', 'time', 'match_id', 'num',
            'lottery_handicap', 'lottery_primary_market', 'lottery_source',
            'lottery_offer_matched', 'lottery_available_markets',
            'lottery_spf_available', 'lottery_rqspf_available',
            'lottery_unavailable_reason', 'okooo_id'
        )},
        'lottery': lottery,
        'league_profile': league_profile,
        'asian': asian,
        'euro': euro,
        'total': total,
        'team': team,
        'single_odds': single_odds,
        'bookmaker_consensus': asian.get('bookmaker_consensus'),
        'confidence': confidence,
        'anomaly': {
            'joint_water': joint_anomaly,
            'euro_asian_deviation': euro_asian_dev,
        },
        'similar_market': similar_market_result,
        'steam_move': steam_result,
        'market_change': market_change_result,
        'live_context': live_context,
        'live_context_quality': live_context_quality,
        
        # ========== 新增字段 ==========
        'model_status': model_status,
        'probability_rank': probability_rank,
        'recommend_rank': recommend_rank,
        'model_weights': model_weights,
        'calibration_effect': calibration_effect,
        'similar_market_detail': similar_market_detail,
        'risk_level': {
            'level': risk['level'],
            'description': risk['description'],
            'risk_score': risk['risk_score'],
            'risk_factors': risk['risk_factors'],
            'recommend_count': risk['recommend_count'],
        },
        'settlement': settlement,

        # ========== 爆冷识别结果（对齐北单，前端「爆冷预警」展示）==========
        'upset': upset,

        'model': {
            'lam_home': lam_home, 'lam_away': lam_away,
            'top_scores': top_scores, 'recommend': recommend,
            'value_bets': value_bets,
            'half_full_time': half_full_time,
            'goal_count': goal_count_result,
            'goal_calibration': {
                'before': goal_dist_before_calibration,
                'after': goal_dist_after_calibration,
                'calibrated': goal_dist_after_calibration is not None
            },
            'dixon_coles': _native(dixon_coles_result),
            'ml': ml_result,
            'risk_level': {
                'level': risk['level'],
                'description': risk['description'],
                'risk_score': risk['risk_score'],
                'risk_factors': risk['risk_factors'],
                'recommend_count': risk['recommend_count'],
            },
            'candidates': candidates,
            **meta,
        },
    }

    try:
        result['professional_evidence'] = build_match_evidence_profile(result)
    except Exception as e:
        log.warning(f"专业证据覆盖评估失败: {e}")
        result['professional_evidence'] = None

    return result
