# -*- coding: utf-8 -*-
"""快乐8策略验证编排"""

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
from . import snapshots as _snapshots_mod
from . import records as _records_mod
from . import config as _cfg

from .config import (
    BACKTEST_PERMUTATION_COUNT, BACKTEST_STABILITY_THRESHOLD, BACKTEST_STABILITY_WINDOWS, KL8_PREDICTOR_VERSION,
)
from .stats import (
    _parse_play_pick_n, _play_lift, _prize_tier_thresholds, benjamini_hochberg_fdr, bonferroni_correction, hypergeom_p_ge,
)

from .analyzer import (
    get_kl8_analyzer,
)

from .backtest import (
    KL8RollingBacktest,
)

def validate_and_activate_strategy(
    play_type: str,
    feature_weights: Dict[str, float],
    model_weights: Dict[str, float],
    window_size: int,
    repeat_direction: str = 'neutral',
    repeat_avoid_score: float = 0.10,
    repeat_non_avoid_score: float = 0.85,
    repeat_follow_score: float = 0.90,
    repeat_non_follow_score: float = 0.50,
    pool_diversify: bool = True,
    pool_max_last_numbers: Optional[int] = None,
    frequency_mode: str = 'mean_reversion',
    final_selection_mode: str = 'balanced',
    auto_activate: bool = False,
    n_permutations: int = BACKTEST_PERMUTATION_COUNT,
) -> Dict:
    """策略激活流程 — 回测结果满足条件后才允许写入ACTIVE_STRATEGIES

    v7核心设计:
    1. 验证集 Lift > 0
    2. FDR 校正后 p < 0.05
    3. 稳定性窗口至少 3/4 为正（将验证段分成4个子窗口，各检查Lift）
    4. 最终封存测试集只做结果确认，不参与激活决策

    参数:
        play_type: 玩法名称，如 'select_5', 'fu_shi_7'
        feature_weights: 特征权重字典
        model_weights: 模型权重字典
        window_size: 统计窗口大小
        auto_activate: 是否验证通过后自动写入ACTIVE_STRATEGIES（默认False=需人工审核确认）
        n_permutations: 置换检验次数

    返回:
        验证报告 Dict，包含各条件是否通过、最终建议、以及（若auto_activate=True且通过）激活结果
    """
    # ── 参数校验 ──
    repeat_direction = (repeat_direction or 'neutral').strip().lower()

    valid_play_types = list(_cfg.ACTIVE_STRATEGIES.keys())
    if play_type not in valid_play_types:
        return {
            'error': f'无效玩法: {play_type}',
            'valid_play_types': valid_play_types,
        }

    # 确定选号数量（用于置换检验和Lift计算）
    pick_n = _parse_play_pick_n(play_type)
    if pick_n is None:
        return {'error': f'无法解析玩法: {play_type}'}

    # 至少有一个有效权重
    has_fw = any(w > 0 for w in feature_weights.values())
    has_mw = any(w > 0 for w in model_weights.values())
    if not (has_fw or has_mw):
        return {'error': 'feature_weights和model_weights至少有一个非零权重'}

    if window_size <= 0:
        return {'error': 'window_size必须为正整数'}

    # ── 获取数据 ──
    if repeat_direction not in ('neutral', 'avoid', 'follow'):
        return {'error': 'repeat_direction must be neutral/avoid/follow'}

    repeat_scores = (
        repeat_avoid_score,
        repeat_non_avoid_score,
        repeat_follow_score,
        repeat_non_follow_score,
    )
    if any(not math.isfinite(score) for score in repeat_scores):
        return {'error': 'repeat score parameters must be finite numbers'}

    if pool_max_last_numbers is not None:
        if pool_max_last_numbers < 0:
            return {'error': 'pool_max_last_numbers must be >= 0'}
        if pool_max_last_numbers > pick_n:
            return {'error': f'pool_max_last_numbers must be <= pick count ({pick_n})'}

    analyzer = get_kl8_analyzer()
    if not analyzer.history_data:
        return {'error': '历史数据不足，无法验证策略'}

    history = analyzer.history_data
    n = len(history)

    # ── 三段式分割 ──
    bt = KL8RollingBacktest(analyzer)
    try:
        split = bt._split_three_stage(n)
    except ValueError as e:
        return {'error': str(e)}

    val_range = split['val']
    final_test_range = split['final_test']

    # ── 条件1: 验证集 Lift > 0 ──
    val_result = bt._rolling_backtest_parametric(
        feature_weights, model_weights,
        start_idx=val_range[0],
        end_idx=val_range[1],
        min_train=50,
        window_size=window_size,
        repeat_direction=repeat_direction,
        repeat_avoid_score=repeat_avoid_score,
        repeat_non_avoid_score=repeat_non_avoid_score,
        repeat_follow_score=repeat_follow_score,
        repeat_non_follow_score=repeat_non_follow_score,
        pool_diversify=pool_diversify,
        pool_max_last_numbers=pool_max_last_numbers,
        frequency_mode=frequency_mode,
        final_selection_mode=final_selection_mode,
    )

    if 'error' in val_result:
        return {'error': f'验证集回测失败: {val_result["error"]}'}

    s_key = play_type
    val_lift = _play_lift(val_result, play_type)

    condition_1_lift_positive = val_lift > 0

    # ── 条件2: FDR 校正后 p < 0.05 ──
    # 对当前玩法的置换检验 + BH FDR校正
    # BH FDR需要多个p值做校正：当前特征在该玩法下的p值 + 同玩法下其他已测特征的p值
    # 简化做法：对该玩法做一次置换检验，然后与其他玩法做FDR校正
    perm_result = bt._permutation_test(
        feature_weights, model_weights,
        start_idx=val_range[0],
        end_idx=val_range[1],
        pick_n=pick_n,
        metric='mean_hits',
        n_permutations=n_permutations,
        window_size=window_size,  # v8: 确保与回测使用相同窗口
        repeat_direction=repeat_direction,
        repeat_avoid_score=repeat_avoid_score,
        repeat_non_avoid_score=repeat_non_avoid_score,
        repeat_follow_score=repeat_follow_score,
        repeat_non_follow_score=repeat_non_follow_score,
        pool_diversify=pool_diversify,
        pool_max_last_numbers=pool_max_last_numbers,
        frequency_mode=frequency_mode,
        final_selection_mode=final_selection_mode,
    )

    if 'error' in perm_result:
        return {'error': f'置换检验失败: {perm_result["error"]}'}

    raw_p_value = perm_result.get('p_value', 1.0)

    # v8: 记录到策略试验结果表（供后续全量FDR校正）
    trial_record = {
        'strategy_id': f'{play_type}_w{window_size}_{hashlib.sha256(json.dumps(feature_weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:6]}',
        'play_type': play_type,
        'feature_weights': feature_weights,
        'model_weights': model_weights,
        'window_size': window_size,
        'raw_p_value': raw_p_value,
        'validation_lift': round(val_lift, 4),
        'n_permutations': n_permutations,
        'tested_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    _cfg.STRATEGY_TRIAL_RESULTS.append(trial_record)
    _records_mod._persist_trial_results()  # v9: 每次新增试验后持久化

    # v8: 全量BH FDR校正 — 对同一玩法下所有候选策略的p值统一校正
    # 收集同玩法的所有试验记录
    same_play_trials = [t for t in _cfg.STRATEGY_TRIAL_RESULTS if t['play_type'] == play_type]
    same_play_p_values = [t['raw_p_value'] for t in same_play_trials]

    # 如果只有1个p值（当前刚添加的），FDR校正等于不校正
    # 但随着候选策略增多，FDR校正越来越有意义
    if len(same_play_p_values) > 1:
        adjusted_p_values = benjamini_hochberg_fdr(same_play_p_values)
        # 找到当前试验在列表中的索引
        current_idx = len(same_play_p_values) - 1  # 刚添加的是最后一个
        adjusted_p = adjusted_p_values[current_idx]
    else:
        adjusted_p = raw_p_value  # 单次检验时FDR校正等于原始p值

    # 同时做Bonferroni校正（保守版）
    bonferroni_p = bonferroni_correction(raw_p_value, len(valid_play_types))

    condition_2_fdr_significant = adjusted_p < 0.05

    # ── 条件3: 稳定性窗口至少 3/4 为正 ──
    # 将验证段分成4个子窗口，每个检查Lift是否>0
    val_start = val_range[0]
    val_end = val_range[1]
    val_len = val_end - val_start
    sub_window_size = val_len // BACKTEST_STABILITY_WINDOWS

    sub_window_lifts = []
    for i in range(BACKTEST_STABILITY_WINDOWS):
        sub_start = val_start + i * sub_window_size
        sub_end = val_start + (i + 1) * sub_window_size
        if i == BACKTEST_STABILITY_WINDOWS - 1:
            sub_end = val_end  # 最后一段用剩余全部

        sub_result = bt._rolling_backtest_parametric(
            feature_weights, model_weights,
            start_idx=sub_start,
            end_idx=sub_end,
            min_train=50,
            window_size=window_size,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
            pool_diversify=pool_diversify,
            pool_max_last_numbers=pool_max_last_numbers,
            frequency_mode=frequency_mode,
            final_selection_mode=final_selection_mode,
        )

        if 'error' in sub_result:
            sub_window_lifts.append(0)  # 出错视为0
            continue

        sub_lift = _play_lift(sub_result, play_type)

        sub_window_lifts.append(sub_lift)

    n_positive_sub_windows = sum(1 for l in sub_window_lifts if l > 0)
    condition_3_stability = n_positive_sub_windows >= BACKTEST_STABILITY_THRESHOLD

    # ── 条件4: 最终封存测试集结果确认（只报告，不参与激活决策）───
    final_test_result = bt._rolling_backtest_parametric(
        feature_weights, model_weights,
        start_idx=final_test_range[0],
        end_idx=final_test_range[1],
        min_train=50,
        window_size=window_size,
        repeat_direction=repeat_direction,
        repeat_avoid_score=repeat_avoid_score,
        repeat_non_avoid_score=repeat_non_avoid_score,
        repeat_follow_score=repeat_follow_score,
        repeat_non_follow_score=repeat_non_follow_score,
        pool_diversify=pool_diversify,
        pool_max_last_numbers=pool_max_last_numbers,
        frequency_mode=frequency_mode,
        final_selection_mode=final_selection_mode,
    )

    final_test_lift = None
    if 'error' not in final_test_result:
        final_test_lift = _play_lift(final_test_result, play_type)

    # ── v9: 条件4 — 关键奖级概率不低于随机 ──
    # 不只看平均命中Lift，还要看关键中奖档位的概率
    threshold_tiers = _prize_tier_thresholds(play_type)
    val_prize_probs = val_result.get(s_key, {}).get('probabilities', {})
    theoretical_probs = val_result.get(s_key, {}).get('theoretical_probs', {})

    prize_tier_passed = True
    prize_tier_details = {}

    for tier in threshold_tiers:
        actual_prob = val_prize_probs.get(tier, 0)
        random_prob = theoretical_probs.get(tier, hypergeom_p_ge(pick_n, int(tier.replace('>=', ''))))
        tier_passed = actual_prob >= random_prob * 0.9  # 允许略低于随机(90%即可)
        prize_tier_details[tier] = {
            'actual_prob': round(actual_prob, 4),
            'random_prob': round(random_prob, 4),
            'passed': tier_passed,
        }
        if not tier_passed:
            prize_tier_passed = False

    # ── v9: 条件5 — 收益率不能显著差于随机 ──
    val_roi = val_result.get(s_key, {}).get('profit_roi', 0)
    random_roi = val_result.get(s_key, {}).get('random_profit_roi', 0)
    roi_not_significantly_worse = val_roi >= random_roi * 0.8  # 允许80%即可

    # ── 激活判断（v9: 5个条件）───
    all_conditions_passed = (
        condition_1_lift_positive
        and condition_2_fdr_significant
        and condition_3_stability
        and prize_tier_passed
        and roi_not_significantly_worse
    )

    # 生成 strategy_id
    strategy_id = f'{play_type}_w{window_size}_v1'
    # 加入特征哈希以区分不同配置
    fw_hash = hashlib.sha256(
        json.dumps(feature_weights, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()[:6]
    strategy_id = f'{play_type}_w{window_size}_{fw_hash}'

    activated = False

    # ── 返回验证报告 ──
    report = {
        'play_type': play_type,
        'strategy_id': strategy_id,
        'window_size': window_size,
        'feature_weights': feature_weights,
        'model_weights': model_weights,
        'repeat_direction': repeat_direction,
        'repeat_avoid_score': repeat_avoid_score,
        'repeat_non_avoid_score': repeat_non_avoid_score,
        'repeat_follow_score': repeat_follow_score,
        'repeat_non_follow_score': repeat_non_follow_score,
        'pool_diversify': pool_diversify,
        'pool_max_last_numbers': pool_max_last_numbers,
        'frequency_mode': frequency_mode,
        'final_selection_mode': final_selection_mode,
        'conditions': {
            'condition_1_lift_positive': {
                'passed': condition_1_lift_positive,
                'val_lift': round(val_lift, 4),
                'detail': f'验证集 Lift = {round(val_lift, 4)}，要求 > 0',
            },
            'condition_2_fdr_significant': {
                'passed': condition_2_fdr_significant,
                'raw_p_value': raw_p_value,
                'bh_fdr_adjusted_p': round(adjusted_p, 6),
                'bonferroni_adjusted_p': round(bonferroni_p, 6),
                'detail': f'BH FDR校正后 p = {round(adjusted_p, 6)}，要求 < 0.05',
            },
            'condition_3_stability': {
                'passed': condition_3_stability,
                'n_positive_sub_windows': n_positive_sub_windows,
                'sub_window_lifts': [round(l, 4) for l in sub_window_lifts],
                'detail': f'稳定性 {n_positive_sub_windows}/{BACKTEST_STABILITY_WINDOWS} 窗口为正，要求 ≥ {BACKTEST_STABILITY_THRESHOLD}',
            },
            'condition_4_final_test_confirmation': {
                'final_test_lift': round(final_test_lift, 4) if final_test_lift is not None else None,
                'note': '最终封存测试集只做结果确认，不参与激活决策',
            },
            'condition_5_prize_tier_thresholds': {
                'passed': prize_tier_passed,
                'details': prize_tier_details,
                'detail': f'关键奖级概率不低于随机*0.9',
            },
            'condition_6_roi_not_worse': {
                'passed': roi_not_significantly_worse,
                'val_profit_roi': round(val_roi, 4),
                'random_profit_roi': round(random_roi, 4),
                'detail': f'策略ROI({round(val_roi, 4)}) >= 随机ROI*0.8({round(random_roi * 0.8, 4)})',
            },
        },
        'all_conditions_passed': all_conditions_passed,
        'val_result_summary': {
            s_key: {
                'lift': round(val_result.get(s_key, {}).get('lift', 0), 4),
                'mean_hits': round(val_result.get(s_key, {}).get('mean_hits', 0), 4),
                'n_tests': val_result.get(s_key, {}).get('n_tests', 0),
                'profit_roi': round(val_result.get(s_key, {}).get('profit_roi', 0), 4),
            }
        },
        'recommendation': 'activate' if all_conditions_passed else 'keep_disabled',
        'activated': activated,
        'auto_activate': auto_activate,
        'version': KL8_PREDICTOR_VERSION,
    }

    if final_test_lift is not None:
        report['final_test_result_summary'] = {
            s_key: {
                'lift': round(final_test_result.get(s_key, {}).get('lift', 0), 4),
                'mean_hits': round(final_test_result.get(s_key, {}).get('mean_hits', 0), 4),
                'profit_roi': round(final_test_result.get(s_key, {}).get('profit_roi', 0), 4),
            }
        }

    # ── 激活（若条件通过 + auto_activate=True）───
    if all_conditions_passed and auto_activate:
        strategy_dict = {
            'strategy_id': strategy_id,
            'feature_weights': feature_weights,
            'model_weights': model_weights,
            'window_size': window_size,
            'repeat_direction': repeat_direction,
            'repeat_avoid_score': repeat_avoid_score,
            'repeat_non_avoid_score': repeat_non_avoid_score,
            'repeat_follow_score': repeat_follow_score,
            'repeat_non_follow_score': repeat_non_follow_score,
            'pool_diversify': pool_diversify,
            'pool_max_last_numbers': pool_max_last_numbers,
            'frequency_mode': frequency_mode,
            'final_selection_mode': final_selection_mode,
        }
        _snapshots_mod.activate_verified_strategy(play_type, strategy_dict, report)
        activated = True
        report['activated'] = True
        log.info(f'快乐8: 策略已激活 {play_type} -> {strategy_id}')
    elif all_conditions_passed and not auto_activate:
        log.info(f'快乐8: 策略验证通过 {play_type}，但auto_activate=False，需人工确认')

    return report


