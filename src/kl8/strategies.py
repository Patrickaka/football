# -*- coding: utf-8 -*-
"""快乐8策略可用性判定与解析"""

from copy import deepcopy
import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')
from . import config as _cfg

from .config import (
    FEATURE_CONFIG, MODEL_CONFIG, REFERENCE_STRATEGY,
)

def is_prediction_ready() -> bool:
    """预测准备就绪判断

    v6: 基于 _cfg.ACTIVE_STRATEGIES 判断
    任一玩法有非空策略（strategy_id不为空且有权重），即视为有信号
    无信号时ranking不返回[1..20]，返回空列表
    """
    for play_type, strategy in _cfg.ACTIVE_STRATEGIES.items():
        if strategy.get('strategy_id', ''):
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {})
            has_feature_weight = any(w > 0 for w in fw.values())
            has_model_weight = any(w > 0 for w in mw.values())
            if has_feature_weight or has_model_weight:
                return True
    return False


def has_active_signal() -> bool:
    """是否有任何启用的特征或模型（向后兼容，但推荐用is_prediction_ready）"""
    return is_prediction_ready()


def _strategy_is_usable(strategy: Dict) -> bool:
    """判断一个策略是否可用（有权重且非全零）

    可用条件: rank模型启用 + 至少一个特征有权重，或者其他模型(bayesian/markov)启用
    """
    fw = strategy.get('feature_weights', {})
    mw = strategy.get('model_weights', {})

    rank_ready = (
        mw.get('rank', 0) > 0
        and any(weight > 0 for weight in fw.values())
    )

    return (
        rank_ready
        or mw.get('bayesian', 0) > 0
        or mw.get('markov', 0) > 0
    )


def resolve_play_strategy(play_type: str, allow_reference: bool = False) -> Optional[Dict]:
    """解析玩法策略：优先使用已验证策略

    v9.2改动:
    - _cfg.VERIFY_ONLY_MODE=True时，找不到validated策略返回None（不再回退参考策略）
    - allow_reference=True时允许回退（仅供后台影子运行、回测等）
    - 返回None时，predict_all()输出 verification_pending 状态

    v9.2.1改动:
    - reference模式下，每个玩法使用不同的默认策略配置（小窗口+多特征）
    - 不再所有玩法共用同一个250期纯频率策略，避免号码每天固定不变
    - 选3/4用freq_50(50期窗口，变化灵敏)，选5/6/7/复式用不同组合

    返回 Dict 包含:
        strategy_id: 策略标识
        feature_weights: 特征权重
        model_weights: 模型权重
        window_size: 统计窗口
        prediction_mode: 'validated' 或 'reference_unvalidated'
        is_validated: True 或 False
    """
    # 选5复式7码只是选6主推的扩展池，不再维护另一套可能漂移的排名。
    # 所有调用方（预测、回测、后台任务）解析该玩法时都拿到选6的实际策略。
    if play_type == 'fu_shi_7':
        linked = resolve_play_strategy('select_6', allow_reference=allow_reference)
        if linked is None:
            return None
        linked['ranking_source'] = 'select_6'
        linked['linked_play_type'] = 'select_6'
        return linked

    strategy = _cfg.ACTIVE_STRATEGIES.get(play_type, {})

    # 已验证的正式策略（且不在降级观察中）
    if (
        strategy.get('strategy_id')
        and strategy.get('status') == 'validated'
        and _strategy_is_usable(strategy)
        and strategy.get('degradation_status') != 'yellow_watch'
    ):
        result = deepcopy(strategy)
        result['prediction_mode'] = 'validated'
        result['is_validated'] = True
        return result

    # v9.2: _cfg.VERIFY_ONLY_MODE=True时，不回退参考策略
    if _cfg.VERIFY_ONLY_MODE and not allow_reference:
        return None

    # v9.2.1: 每个玩法使用不同的默认策略配置（小窗口+多特征组合）
    # v9.3: 加入趋势(trend)、组合共现(pair_cooccurrence)、细粒化位置残差(position_residual_cross)特征
    _REFERENCE_STRATEGIES_BY_PLAY = {
        'fu_shi_4': {
            'strategy_id': 'fu_shi_4_ref_trend100_mix_shape_balanced',
            'feature_weights': {'frequency': 0.35, 'gap': 0.20, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.075, 'position_residual_cross': 0.075, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_max_last_numbers': 4,
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_3': {
            'strategy_id': 'select_3_ref_trend50_shape_balanced',
            'feature_weights': {'frequency': 0.40, 'gap': 0.20, 'trend': 0.25, 'pair_cooccurrence': 0.10, 'position_residual': 0.05, 'position_residual_cross': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 50,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_4': {
            'strategy_id': 'select_4_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.35, 'gap': 0.15, 'trend': 0.25, 'pair_cooccurrence': 0.15, 'position_residual': 0.10, 'position_residual_cross': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_5': {
            'strategy_id': 'select_5_ref_transition_repeat_v3',
            'feature_weights': {'frequency': 0.22, 'gap': 0.15, 'trend': 0.12, 'next_transition': 0.20, 'pair_cooccurrence': 0.04, 'position_residual': 0.12, 'position_residual_cross': 0.08, 'road_residual': 0.04, 'repeat': 0.03, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'follow',
            'pool_max_last_numbers': 2,
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'target_hits': 4,
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_6': {
            # v10.10 keeps the automatic exclusion chain, but restores the
            # v10.8 primary ranking.  The primary ticket is the accuracy
            # target; covering almost all 80 numbers is not predictive lift.
            'strategy_id': 'select_6_ref_transition_primary_v5',
            'feature_weights': {'frequency': 0.18, 'gap': 0.14, 'trend': 0.12, 'next_transition': 0.22, 'pair_cooccurrence': 0.04, 'position_residual': 0.11, 'position_residual_cross': 0.08, 'road_residual': 0.08, 'repeat': 0.03, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'follow',
            'pool_max_last_numbers': 3,
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'chain_objective': 'primary_accuracy_then_early_exclusion',
            'chain_audit_rounds': 5,
            'target_hits': 5,
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_7': {
            'strategy_id': 'select_7_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.30, 'gap': 0.15, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.10, 'position_residual_cross': 0.05, 'road_residual': 0.05, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_8': {
            'strategy_id': 'select_8_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.35, 'gap': 0.15, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.075, 'position_residual_cross': 0.075, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_9': {
            'strategy_id': 'select_9_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.35, 'gap': 0.15, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.075, 'position_residual_cross': 0.075, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'select_10': {
            'strategy_id': 'select_10_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.30, 'gap': 0.15, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.10, 'position_residual_cross': 0.05, 'road_residual': 0.05, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
        'fu_shi_10_11': {
            'strategy_id': 'fu_shi_10_11_ref_trend100_shape_balanced',
            'feature_weights': {'frequency': 0.35, 'gap': 0.20, 'trend': 0.20, 'pair_cooccurrence': 0.15, 'position_residual': 0.075, 'position_residual_cross': 0.075, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
            'window_size': 100,
            'repeat_direction': 'neutral',
            'pool_max_last_numbers': 4,
            'pool_diversify': False,
            'final_selection_mode': 'concentrated',
            'prediction_mode': 'reference_unvalidated',
            'is_validated': False,
        },
    }

    # 使用玩法专属配置，找不到则回退全局默认
    ref_config = _REFERENCE_STRATEGIES_BY_PLAY.get(play_type)
    if ref_config:
        result = deepcopy(ref_config)
    else:
        result = deepcopy(REFERENCE_STRATEGY)
        result['strategy_id'] = f'{play_type}_reference_heuristic_v1'

    # 未验证阶段使用玩法专属的动态组合，而不是按期号生成的确定性随机排名。
    # 各玩法窗口、特征权重和形态约束彼此独立，随新开奖更新；验证状态仍为
    # reference_unvalidated，只有通过严格样本外门槛才会升级为 validated。
    result['baseline_type'] = 'adaptive_pattern_reference'
    return result


def get_active_feature_weights() -> Dict[str, float]:
    """获取当前启用的特征权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in FEATURE_CONFIG.items()}


def get_active_model_weights() -> Dict[str, float]:
    """获取当前启用的模型权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in MODEL_CONFIG.items()}


FEATURE_WEIGHTS = get_active_feature_weights()


MODEL_WEIGHTS = get_active_model_weights()


