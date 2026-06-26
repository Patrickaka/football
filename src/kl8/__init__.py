"""
快乐8预测模块
=============

快乐8玩法：从1-80中开出20个号码，玩家可选1-10个号码进行投注。
本模块提供选3/选4/选5/选6/选7预测，以及选5复式7码预测。

v6 核心改动:
1. 真正三段式: train→val→final_test, final_test完全冻结不参与启用判断
2. 回测使用multi_model_voting管道（model_weights真正生效）
3. ACTIVE_STRATEGIES策略注册表（按玩法分别配置）
4. 置换检验按玩法分别执行（pick_n=select_type）
5. 多重检验校正（Benjamini-Hochberg FDR + Bonferroni备用）
6. ROI统一为return_multiple和profit_roi两个字段
7. 复式7码每期全部21注组合都计算投注和奖金
8. 无号码推荐时不扣投注（只有placed=true才计）
9. 快照结算添加期号校验（actual_issue > based_on_issue）
10. 第二数据源交叉校验接入抓取流程+冲突写入队列
11. 超几何分布替代二项分布(80选20不放回)
12. 数据排序+连续性检查+冲突审核队列

版本: kl8-v9.0-systematic-fix

v9 核心改动:
1. 快照结算: 只结算 target_issue==actual_issue 的快照，不再宽泛匹配
2. target_type='next_draw_after_based_on': 结算时验证actual是based_on的直接下一期
3. _compute_next_issue(): 从历史数据推导下一期期号（不再简单int+1）
4. 结算策略ID: 从快照读取 play_strategies/prediction_modes，不再读当前ACTIVE_STRATEGIES
5. resolved_strategies: 快照保存每种玩法当时的完整策略配置
6. 统一候选池: build_candidate_pool()生成Top20，所有玩法从同一份截取
7. 候选策略锦标赛: run_candidate_tournament() 训练→验证→最终测试
8. 策略持久化: STRATEGY_TRIAL_RESULTS/ACTIVE_STRATEGIES 持久化到JSON
9. 策略指纹: _strategy_fingerprint()包含完整配置（不再只含feature_weights）
10. 最终测试报告只允许写入一次
11. 置换检验: 打乱实际开奖期顺序+加一修正（不再随机抽号码）
12. 激活条件: 增加奖级概率门槛+收益率不低于随机
13. 特征消融: 使用独立ABLATION_FEATURES试验表
14. 冲突过滤: 返回完整conflict_issues列表
15. 策略降级: 不自动清空，改为黄色观察→人工确认
16. 快照唯一约束: 同期同策略只保留一份正式快照，其他标记is_experiment
17. 概率校准框架预留（Brier Score/LogLoss/Calibration curve，暂不启用）
"""

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

KL8_PREDICTOR_VERSION = "kl8-v9.0-systematic-fix"

# ─── 快乐8常量 ───
KL8_NUM_RANGE = 80       # 号码范围 1-80
KL8_DRAW_COUNT = 20      # 每期开出20个号码
KL8_DEFAULT_HISTORY = 250  # 默认使用最近250期
KL8_EXPECTED_GAP = (KL8_NUM_RANGE - KL8_DRAW_COUNT) / KL8_DRAW_COUNT  # = 3.0

# ─── 回测常量 ───
BACKTEST_MIN_OOS_PERIODS = 300   # 最小样本外期数
BACKTEST_FINAL_TEST_PERIODS = 200  # 最终封存测试期数
BACKTEST_PERMUTATION_COUNT = 1000  # 置换检验次数
BACKTEST_STABILITY_WINDOWS = 4     # 稳定性检查窗口数
BACKTEST_STABILITY_THRESHOLD = 3   # 至少3/4窗口Lift>0

# ─── 选型配置：选3~选7各选多少号码 ───
SELECT_CONFIG = {
    3: {'pick': 3, 'top_n': 10,  'desc': '选3'},
    4: {'pick': 4, 'top_n': 12,  'desc': '选4'},
    5: {'pick': 5, 'top_n': 15,  'desc': '选5'},
    6: {'pick': 6, 'top_n': 15,  'desc': '选6'},
    7: {'pick': 7, 'top_n': 18,  'desc': '选7'},
}

# ─── 特征开关配置（v5：所有特征默认停用，需回测验证才能启用）───
# 按玩法分开评估: 每个特征可以有per-play-type的enabled状态
FEATURE_CONFIG = {
    'frequency':        {'enabled': False, 'weight': 1.0,   'desc': '频率偏离度(均值回归:冷号加分,热号降分)'},
    'gap':              {'enabled': False, 'weight': 0.0,   'desc': '遗漏偏离度 -- 仅展示指标,不参与预测'},
    'position_residual': {'enabled': False, 'weight': 0.0,   'desc': '区内残差(剔除全局频率后的区位偏移)'},
    'road_residual':    {'enabled': False, 'weight': 0.0,   'desc': '路内残差(剔除全局频率后的路数偏移)'},
    'sum':              {'enabled': False, 'weight': 0.0,   'desc': '和值特征 -- 停用'},
    'zone':             {'enabled': False, 'weight': 0.0,   'desc': '区位近期开出率 -- 停用'},
    'repeat':           {'enabled': False, 'weight': 0.0,   'desc': '重号特征(3个候选方向: neutral/avoid/follow)'},
    'adjacent':         {'enabled': False, 'weight': 0.0,   'desc': '邻号特征 -- 停用'},
    'odd_even':         {'enabled': False, 'weight': 0.0,   'desc': '奇偶特征(暂停,等单特征回测)'},
    'big_small':        {'enabled': False, 'weight': 0.0,   'desc': '大小特征(暂停,等单特征回测)'},
}

# ─── 投票模型权重（v6：停用，等策略注册表接管）───
MODEL_CONFIG = {
    'bayesian': {'enabled': False, 'weight': 0.0, 'desc': '停用: 倾向热号,与排名频率冷号方向相反'},
    'rank':     {'enabled': False, 'weight': 0.0, 'desc': '排名模型 -- 停用,等回测验证'},
    'markov':   {'enabled': False, 'weight': 0.0, 'desc': '停用: 低号码偏差,未出现号码全0.25'},
}

# ─── v6 策略注册表（按玩法分别配置，取代全局FEATURE_CONFIG的预测权重）───
# 每个玩法有独立的 strategy_id、feature_weights、model_weights、window_size
# 预测、快照、结算、回测都必须记录 strategy_id
# 当前所有策略默认空（无信号），需回测验证后才能填入具体配置
ACTIVE_STRATEGIES = {
    'select_3': {
        'strategy_id': '',           # 空=无验证策略
        'feature_weights': {},       # 空=不启用任何特征
        'model_weights': {},         # 空=不启用任何模型
        'window_size': 0,            # 0=无固定窗口
    },
    'select_4': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_5': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_6': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'fu_shi_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
}

# ─── 默认参考策略（v7.1新增）───
# 回测未通过时，自动降级到此策略：基础统计排序，但明确标记为"未验证参考"
from copy import deepcopy

REFERENCE_STRATEGY = {
    'strategy_id': 'reference_heuristic_v1',
    'feature_weights': {
        'frequency': 1.0,      # 全局频率偏离（唯一核心特征）
        'position_residual': 0.0,  # 区内残差（暂停用，等单特征回测）
        'road_residual': 0.0,      # 路内残差（暂停用，等单特征回测）
        'repeat': 0.0,             # 重号特征（暂停用，等候选策略回测）
        'odd_even': 0.0,
        'big_small': 0.0,
    },
    'model_weights': {
        'rank': 1.0,
        'bayesian': 0.0,
        'markov': 0.0,
    },
    'window_size': 250,
    'prediction_mode': 'reference_unvalidated',
    'is_validated': False,
}

# ─── 候选策略试验表（v8新增）───
# 3种重号方向候选策略，都标记is_validated=False，等回测胜出后才进入正式策略
# repeat_neutral: 不处理重号（repeat权重=0）
# repeat_avoid: 避开上期号（上期号得分0.10，非上期0.85，repeat权重0.25）
# repeat_follow: 适度保留上期号（上期号得分0.90，非上期0.50，repeat权重0.15）

CANDIDATE_STRATEGIES = {
    'repeat_neutral': {
        'strategy_id': 'candidate_repeat_neutral',
        'feature_weights': {
            'frequency': 1.0,
            'position_residual': 0.0,
            'road_residual': 0.0,
            'repeat': 0.0,
            'odd_even': 0.0,
            'big_small': 0.0,
        },
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 250,
        'prediction_mode': 'candidate_unvalidated',
        'is_validated': False,
        'candidate_group': 'repeat_direction',
        'repeat_direction': 'neutral',
    },
    'repeat_avoid': {
        'strategy_id': 'candidate_repeat_avoid',
        'feature_weights': {
            'frequency': 0.60,
            'position_residual': 0.0,
            'road_residual': 0.0,
            'repeat': 0.25,
            'odd_even': 0.0,
            'big_small': 0.0,
        },
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 250,
        'prediction_mode': 'candidate_unvalidated',
        'is_validated': False,
        'candidate_group': 'repeat_direction',
        'repeat_direction': 'avoid',
        'repeat_avoid_score': 0.10,   # 上期号得分
        'repeat_non_avoid_score': 0.85,  # 非上期号得分
    },
    'repeat_follow': {
        'strategy_id': 'candidate_repeat_follow',
        'feature_weights': {
            'frequency': 0.60,
            'position_residual': 0.0,
            'road_residual': 0.0,
            'repeat': 0.15,
            'odd_even': 0.0,
            'big_small': 0.0,
        },
        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
        'window_size': 250,
        'prediction_mode': 'candidate_unvalidated',
        'is_validated': False,
        'candidate_group': 'repeat_direction',
        'repeat_direction': 'follow',
        'repeat_follow_score': 0.90,     # 上期号得分
        'repeat_non_follow_score': 0.50, # 非上期号得分
    },
}

# ─── v9: 独立消融试验表（不再依赖当前启用权重）───
# 消融回测不再只测 FEATURE_CONFIG 中 weight>0 的特征
# 而是使用这张独立试验表，确保残差、重号、路数策略都参与完整消融
ABLATION_FEATURES = {
    'frequency': 1.0,
    'position_residual': 1.0,
    'road_residual': 1.0,
    'repeat_avoid': 0.25,   # repeat方向=avoid
    'repeat_follow': 0.15,  # repeat方向=follow
}

# ─── 策略试验结果记录表（v8新增）───
# 所有候选策略的回测结果统一记录于此，最终做全量FDR校正
# 格式: [{'strategy_id': ..., 'play_type': ..., 'raw_p_value': ..., 'validation_lift': ..., ...}]
STRATEGY_TRIAL_RESULTS = []


KL8_SNAPSHOT_DIR = data_path('kl8_snapshots')
KL8_SETTLEMENT_DIR = data_path('kl8_settlements')
KL8_PRIZE_TABLE_FILE = data_path('kl8_prize_table.json')

# ─── v9: 策略试验与激活持久化 ───
KL8_STRATEGY_TRIAL_FILE = data_path('kl8_strategy_trials.json')
KL8_ACTIVE_STRATEGIES_FILE = data_path('kl8_active_strategies.json')
KL8_FINAL_TEST_REPORT_FILE = data_path('kl8_final_test_report.json')

# ─── 冲突审核队列 ───
KL8_CONFLICT_QUEUE_FILE = data_path('kl8_conflict_queue.json')


# ─── 预测就绪检查（v6: 基于ACTIVE_STRATEGIES判断）───

def is_prediction_ready() -> bool:
    """预测准备就绪判断

    v6: 基于 ACTIVE_STRATEGIES 判断
    任一玩法有非空策略（strategy_id不为空且有权重），即视为有信号
    无信号时ranking不返回[1..20]，返回空列表
    """
    for play_type, strategy in ACTIVE_STRATEGIES.items():
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


# ─── 策略解析（v7.1新增）───

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


def resolve_play_strategy(play_type: str) -> Dict:
    """解析玩法策略：优先使用已验证策略，回测未通过时自动降级到参考策略

    返回 Dict 包含:
        strategy_id: 策略标识
        feature_weights: 特征权重
        model_weights: 模型权重
        window_size: 统计窗口
        prediction_mode: 'validated' 或 'reference_unvalidated'
        is_validated: True 或 False
    """
    strategy = ACTIVE_STRATEGIES.get(play_type, {})

    # 已验证的正式策略
    if strategy.get('strategy_id') and _strategy_is_usable(strategy):
        result = deepcopy(strategy)
        result['prediction_mode'] = 'validated'
        result['is_validated'] = True
        return result

    # 没通过回测时，自动使用参考策略
    result = deepcopy(REFERENCE_STRATEGY)
    result['strategy_id'] = f'{play_type}_reference_heuristic_v1'
    return result


# ─── 策略激活流程（v7新增）───

def validate_and_activate_strategy(
    play_type: str,
    feature_weights: Dict[str, float],
    model_weights: Dict[str, float],
    window_size: int,
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
    valid_play_types = list(ACTIVE_STRATEGIES.keys())
    if play_type not in valid_play_types:
        return {
            'error': f'无效玩法: {play_type}',
            'valid_play_types': valid_play_types,
        }

    # 确定 select_type（用于置换检验和Lift计算）
    if play_type == 'fu_shi_7':
        pick_n = 7
    elif play_type.startswith('select_'):
        try:
            pick_n = int(play_type.split('_')[1])
        except (ValueError, IndexError):
            return {'error': f'无法解析玩法select_type: {play_type}'}
    else:
        return {'error': f'无法解析玩法: {play_type}'}

    # 至少有一个有效权重
    has_fw = any(w > 0 for w in feature_weights.values())
    has_mw = any(w > 0 for w in model_weights.values())
    if not (has_fw or has_mw):
        return {'error': 'feature_weights和model_weights至少有一个非零权重'}

    if window_size <= 0:
        return {'error': 'window_size必须为正整数'}

    # ── 获取数据 ──
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
    )

    if 'error' in val_result:
        return {'error': f'验证集回测失败: {val_result["error"]}'}

    s_key = f'select_{pick_n}' if play_type != 'fu_shi_7' else 'fu_shi_7'
    val_lift = val_result.get(s_key, {}).get('lift', 0)
    # fu_shi_7 用 pool_mean_hits 的 lift
    if play_type == 'fu_shi_7':
        fu7_val = val_result.get('fu_shi_7', {})
        pool_mean = fu7_val.get('pool_mean_hits', 0)
        expected_random = fu7_val.get('pool_expected_random', hypergeom_expected(7))
        val_lift = (pool_mean - expected_random) / expected_random if expected_random > 0 else 0

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
    STRATEGY_TRIAL_RESULTS.append(trial_record)
    _persist_trial_results()  # v9: 每次新增试验后持久化

    # v8: 全量BH FDR校正 — 对同一玩法下所有候选策略的p值统一校正
    # 收集同玩法的所有试验记录
    same_play_trials = [t for t in STRATEGY_TRIAL_RESULTS if t['play_type'] == play_type]
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
        )

        if 'error' in sub_result:
            sub_window_lifts.append(0)  # 出错视为0
            continue

        sub_lift = sub_result.get(s_key, {}).get('lift', 0)
        if play_type == 'fu_shi_7':
            fu7_sub = sub_result.get('fu_shi_7', {})
            sub_pool_mean = fu7_sub.get('pool_mean_hits', 0)
            sub_expected = fu7_sub.get('pool_expected_random', hypergeom_expected(7))
            sub_lift = (sub_pool_mean - sub_expected) / sub_expected if sub_expected > 0 else 0

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
    )

    final_test_lift = None
    if 'error' not in final_test_result:
        final_test_lift = final_test_result.get(s_key, {}).get('lift', 0)
        if play_type == 'fu_shi_7':
            fu7_ft = final_test_result.get('fu_shi_7', {})
            ft_pool_mean = fu7_ft.get('pool_mean_hits', 0)
            ft_expected = fu7_ft.get('pool_expected_random', hypergeom_expected(7))
            final_test_lift = (ft_pool_mean - ft_expected) / ft_expected if ft_expected > 0 else 0

    # ── v9: 条件4 — 关键奖级概率不低于随机 ──
    # 不只看平均命中Lift，还要看关键中奖档位的概率
    prize_tier_thresholds = {
        'select_3': ['>=2', '>=3'],
        'select_4': ['>=2', '>=3'],
        'select_5': ['>=3', '>=4'],
        'select_6': ['>=3', '>=4'],
        'select_7': ['>=3', '>=4'],
        'fu_shi_7': ['>=3'],  # 复式7码看池命中>=3
    }

    threshold_tiers = prize_tier_thresholds.get(play_type, ['>=3'])
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

    # ── 激活（若条件通过 + auto_activate=True）───
    activated = False
    if all_conditions_passed and auto_activate:
        ACTIVE_STRATEGIES[play_type] = {
            'strategy_id': strategy_id,
            'feature_weights': feature_weights,
            'model_weights': model_weights,
            'window_size': window_size,
            'repeat_direction': repeat_direction,
        }
        _persist_active_strategies()  # v9: 持久化
        clear_cache()
        activated = True
        log.info(f'快乐8: 策略已激活 {play_type} -> {strategy_id}')
    elif all_conditions_passed and not auto_activate:
        log.info(f'快乐8: 策略验证通过 {play_type}，但auto_activate=False，需人工确认')

    # ── 返回验证报告 ──
    report = {
        'play_type': play_type,
        'strategy_id': strategy_id,
        'window_size': window_size,
        'feature_weights': feature_weights,
        'model_weights': model_weights,
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

    return report


def get_active_feature_weights() -> Dict[str, float]:
    """获取当前启用的特征权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in FEATURE_CONFIG.items()}


def get_active_model_weights() -> Dict[str, float]:
    """获取当前启用的模型权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in MODEL_CONFIG.items()}


FEATURE_WEIGHTS = get_active_feature_weights()
MODEL_WEIGHTS = get_active_model_weights()


# ─── 超几何分布 ───

def hypergeom_pmf(pick_n: int, hits: int) -> float:
    """超几何分布PMF: 从80个号码中选pick_n个，开出20个，命中hits个的概率

    P(X=hits) = C(20,hits) * C(60,pick_n-hits) / C(80,pick_n)
    """
    from math import comb
    if hits < 0 or hits > min(pick_n, KL8_DRAW_COUNT):
        return 0.0
    if pick_n - hits > KL8_NUM_RANGE - KL8_DRAW_COUNT:
        return 0.0
    return comb(KL8_DRAW_COUNT, hits) * comb(KL8_NUM_RANGE - KL8_DRAW_COUNT, pick_n - hits) / comb(KL8_NUM_RANGE, pick_n)


def hypergeom_p_ge(pick_n: int, min_hits: int) -> float:
    """超几何分布 P(X >= min_hits)"""
    total = 0.0
    for k in range(min_hits, min(pick_n, KL8_DRAW_COUNT) + 1):
        total += hypergeom_pmf(pick_n, k)
    return total


def hypergeom_expected(pick_n: int) -> float:
    """超几何分布期望命中数 = pick_n * 20/80"""
    return pick_n * KL8_DRAW_COUNT / KL8_NUM_RANGE


# ─── 多重检验校正（v6新增）───

def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR校正

    对同一玩法下所有特征、窗口、权重的p值做FDR校正
    步骤:
    1. p值从小到大排序
    2. 每个p值校正为: p_adjusted = p * m / rank
       (m = 总检验次数, rank = 该p值在排序中的序位)
    3. 确保单调性: 从大到小遍历，取min(p_adjusted, 下一个校正值)

    参数:
        p_values: 原始p值列表

    返回:
        校正后的p值列表（与输入顺序对应）
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    prev_adjusted = 1.0

    # 从最大p值开始，确保单调性
    for i in range(m - 1, -1, -1):
        original_idx, p = indexed[i]
        rank = i + 1  # 序位从1开始
        bh_adjusted = min(1.0, p * m / rank)
        # 确保单调性: 不比后面(更小rank)的校正值更大
        adjusted[original_idx] = min(bh_adjusted, prev_adjusted)
        prev_adjusted = adjusted[original_idx]

    return adjusted


def bonferroni_correction(p_value: float, n_experiments: int) -> float:
    """Bonferroni校正（保守版多重检验校正）

    p_adjusted = min(1.0, p_value * number_of_experiments)

    参数:
        p_value: 原始p值
        n_experiments: 总检验次数

    返回:
        校正后的p值
    """
    return min(1.0, p_value * n_experiments)


def _clean_pick_numbers(numbers, expected_len: int) -> List[int]:
    """Return a validated pick list, or [] when the pick is malformed."""
    if not isinstance(numbers, (list, tuple, set)):
        return []
    try:
        nums = [int(n) for n in numbers]
    except (TypeError, ValueError):
        return []
    if len(nums) != expected_len or len(set(nums)) != expected_len:
        return []
    if any(n < 1 or n > KL8_NUM_RANGE for n in nums):
        return []
    return nums


def normalize_record(record, keep_meta: bool = False) -> Optional[Dict]:
    """校验并标准化单条记录

    v5加固:
    - 坏JSON字符串 -> 返回None(不再抛异常)
    - 坏数据类型 -> 返回None
    - 20个号码必须唯一且在1-80范围
    - 期号不为空
    - keep_meta=True时保留source/fetched_at/checksum溯源字段
    """
    if not isinstance(record, dict):
        return None

    nums = record.get('numbers') or record.get('draw_numbers')

    if isinstance(nums, str):
        try:
            nums = json.loads(nums)
        except (json.JSONDecodeError, TypeError):
            return None

    if not nums:
        return None

    if not isinstance(nums, (list, tuple, set)):
        return None

    try:
        nums = sorted(int(x) for x in nums)
    except (ValueError, TypeError):
        return None

    if len(nums) != 20:
        return None
    if len(set(nums)) != 20:
        return None
    if any(n < 1 or n > 80 for n in nums):
        return None

    issue = str(record.get('issue', '')).strip()
    if not issue:
        return None

    result = {
        'issue': issue,
        'numbers': nums,
        'date': record.get('date') or record.get('draw_date', ''),
    }

    if keep_meta:
        result.update({
            'source': record.get('source', ''),
            'fetched_at': record.get('fetched_at', ''),
            'checksum': record.get('checksum', _checksum_numbers(nums)),
        })

    return result


def _checksum_numbers(nums: List[int]) -> str:
    """号码列表的短校验码"""
    s = json.dumps(sorted(nums), separators=(',', ':'))
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _compute_next_issue(latest_issue: str, history_data: List[Dict]) -> str:
    """从历史数据推导下一期期号（不再简单int+1）

    策略:
    1. 从历史数据中找相邻期号的差值模式
    2. 使用最常见的差值推算下一期
    3. 跨年(如2026365→2027001)和停开期间能正确处理
    """
    if not history_data:
        return f'next_after_{latest_issue}'

    # 收集相邻期号差值
    history_asc = sorted(history_data, key=lambda x: x['issue'])
    diffs = []
    start_idx = max(0, len(history_asc) - 21)
    for i in range(start_idx, len(history_asc) - 1):
        try:
            curr = int(history_asc[i + 1]['issue'])
            prev = int(history_asc[i]['issue'])
            diff = curr - prev
            if diff > 0:
                diffs.append(diff)
        except (ValueError, TypeError):
            continue

    if not diffs:
        return f'next_after_{latest_issue}'

    # 使用最常见的差值
    from collections import Counter as _Counter
    most_common_diff = _Counter(diffs).most_common(1)[0][0]

    try:
        latest_int = int(latest_issue)
        return str(latest_int + most_common_diff)
    except (ValueError, TypeError):
        return f'next_after_{latest_issue}'


# ─── 奖金表 ───

def load_prize_table() -> Dict:
    """加载可配置奖金表

    格式: {
        "select_3": {"3": 25, "2": 5, "1": 0, "0": 0, "bet": 2},
        "select_4": {"4": 100, "3": 20, ...},
        ...
        "fu_shi_7": {"5": 10000, "4": 500, ...}
    }

    每个玩法包含: 命中档位奖金 + "bet"单注金额
    如果文件不存在，返回默认值
    """
    path = Path(KL8_PRIZE_TABLE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            log.warning(f'奖金表加载失败: {e}')

    # 默认奖金表（中国快乐8官方奖金，单位:元）
    return {
        'select_3': {'3': 25, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_4': {'4': 100, '3': 20, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_5': {'5': 10000, '4': 500, '3': 30, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_6': {'6': 300000, '5': 5000, '4': 100, '3': 10, '2': 0, '1': 0, '0': 0, 'bet': 2},
        'select_7': {'7': 1000000, '6': 50000, '5': 1000, '4': 50, '3': 5, '2': 0, '1': 0, '0': 0, 'bet': 2},
        'fu_shi_7': {'5': 10000, '4': 500, '3': 30, '2': 5, '1': 0, '0': 0, 'bet_per_combo': 2},
    }


# ─── v9: 策略持久化 ───

def _strategy_fingerprint(strategy: Dict) -> str:
    """策略指纹 — 包含完整配置字段（不再只包含 feature_weights）

    v9新增: 策略指纹必须包含所有影响预测结果的字段
    """
    fp_data = {
        'feature_weights': strategy.get('feature_weights', {}),
        'model_weights': strategy.get('model_weights', {}),
        'window_size': strategy.get('window_size', 0),
        'repeat_direction': strategy.get('repeat_direction', 'neutral'),
        'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
        'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
        'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
        'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
    }
    return hashlib.sha256(
        json.dumps(fp_data, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()[:12]


def _persist_trial_results():
    """持久化策略试验结果 — 追加、去重、原子写入

    v9新增: STRATEGY_TRIAL_RESULTS 不再只在内存，服务重启后能恢复
    """
    path = Path(KL8_STRATEGY_TRIAL_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 去重: strategy_id + play_type + tournament_round + tested_at 组合唯一
    unique_trials = []
    seen_keys = set()
    for trial in STRATEGY_TRIAL_RESULTS:
        key = f"{trial.get('strategy_id', '')}_{trial.get('play_type', '')}_{trial.get('tournament_round', '')}_{trial.get('tested_at', '')}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_trials.append(trial)

    # 原子写入
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps(unique_trials, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
    except Exception as e:
        log.warning(f'持久化策略试验结果失败: {e}')
        if temp_path.exists():
            temp_path.unlink()


def _load_trial_results():
    """加载持久化的策略试验结果"""
    path = Path(KL8_STRATEGY_TRIAL_FILE)
    if not path.exists():
        return []

    try:
        trials = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(trials, list):
            return trials
    except Exception as e:
        log.warning(f'加载策略试验结果失败: {e}')

    return []


def _persist_active_strategies():
    """持久化已激活策略 — 服务启动时自动加载"""
    path = Path(KL8_ACTIVE_STRATEGIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 原子写入
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps(ACTIVE_STRATEGIES, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
    except Exception as e:
        log.warning(f'持久化已激活策略失败: {e}')
        if temp_path.exists():
            temp_path.unlink()


def _load_active_strategies():
    """加载持久化的已激活策略"""
    path = Path(KL8_ACTIVE_STRATEGIES_FILE)
    if not path.exists():
        return None

    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(loaded, dict):
            return loaded
    except Exception as e:
        log.warning(f'加载已激活策略失败: {e}')

    return None


def _persist_final_test_report(report: Dict):
    """持久化最终测试报告 — 只允许写入一次

    v9新增: 最终测试结果锁定，不允许重复写入
    """
    path = Path(KL8_FINAL_TEST_REPORT_FILE)
    if path.exists():
        log.warning('最终测试报告已存在，不允许重复写入')
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        return True
    except Exception as e:
        log.warning(f'持久化最终测试报告失败: {e}')
        return False


# 模块初始化时加载持久化数据
STRATEGY_TRIAL_RESULTS.extend(_load_trial_results())

# 加载已激活策略（覆盖默认空策略）
loaded_strategies = _load_active_strategies()
if loaded_strategies:
    for play_type, strategy in loaded_strategies.items():
        if play_type in ACTIVE_STRATEGIES:
            ACTIVE_STRATEGIES[play_type] = strategy
    log.info(f'快乐8: 已从持久化文件加载{len(loaded_strategies)}个已激活策略')


# ─── 冲突审核队列 ───

def save_conflict_to_queue(conflict_info: Dict):
    """将数据冲突记录保存到审核队列（不自动覆盖，等待人工确认）"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    queue = []
    if path.exists():
        try:
            queue = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            queue = []

    conflict_info['queued_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    queue.append(conflict_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding='utf-8')


def list_conflict_queue() -> List[Dict]:
    """查看冲突审核队列"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []


# ─── 数据完整性检查 ───

def check_data_integrity(data: List[Dict]) -> Dict:
    """检查历史数据完整性

    检查项:
    1. 数据是否按期号排序（读取后排序而非假设顺序）
    2. 期号连续性（缺失期号报告）
    3. 日期与期号一致性
    4. 号码范围和唯一性
    """
    if not data:
        return {'valid': False, 'error': '数据为空'}

    # 1. 确保排序（读取后排序，不假设顺序）
    data_sorted = sorted(data, key=lambda x: x['issue'], reverse=True)

    issues = [r['issue'] for r in data_sorted]
    total = len(issues)

    # 2. 期号连续性检查
    # 快乐8期号格式: 通常是年份+3位序号(如2026001~2026365)
    missing_issues = []
    issue_ints = []
    for issue in issues:
        try:
            issue_ints.append(int(issue))
        except (ValueError, TypeError):
            continue

    if issue_ints:
        issue_ints_sorted = sorted(issue_ints)
        for i in range(len(issue_ints_sorted) - 1):
            # 检查相邻期号差值
            diff = issue_ints_sorted[i + 1] - issue_ints_sorted[i]
            if diff > 1:
                # 报告缺失期号（但只报告小的gap，跨年的大gap可能正常）
                if diff <= 5:
                    for j in range(1, diff):
                        missing_issues.append(str(issue_ints_sorted[i] + j))

    # 3. 日期与期号一致性（简单检查: 同一天应该有相似期号前缀）
    date_issue_conflicts = []
    seen_dates = {}
    for r in data_sorted:
        date = r.get('date', '')
        issue = r['issue']
        if date:
            year_prefix = issue[:4] if len(issue) >= 4 else ''
            date_year = date[:4] if len(date) >= 4 else ''
            if year_prefix and date_year and year_prefix != date_year:
                date_issue_conflicts.append({
                    'issue': issue,
                    'date': date,
                    'reason': f'期号年份{year_prefix}与日期年份{date_year}不匹配',
                })

    return {
        'valid': True,
        'total_records': total,
        'latest_issue': issues[0] if issues else '',
        'earliest_issue': issues[-1] if issues else '',
        'missing_issues': missing_issues[:20],  # 只报告前20个缺失
        'missing_count': len(missing_issues),
        'date_issue_conflicts': date_issue_conflicts[:10],
        'conflict_count': len(date_issue_conflicts),
    }


class KL8Analyzer:
    """快乐8预测分析器（v5: 严格三段式+预测就绪判断+纯参数化回测）"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('kl8_history.json')
        self.using_simulated_data = False
        self._data_mtime = 0
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()

    # ─── 数据加载（v5: 排序而非假设顺序+完整性检查）───

    def _load_history(self) -> List[Dict]:
        """加载历史开奖数据（v5: 合并后排序+完整性检查+冲突审核队列）"""

        source_records = {}

        # 来源1: doc_store
        try:
            raw_records = doc_store._fallback_load_all('kl8_history')
            if raw_records:
                for r in raw_records:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in source_records:
                        old = source_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'doc_store内期号{issue}号码冲突，保留第一条')
                            save_conflict_to_queue({
                                'source': 'doc_store_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    source_records[issue] = normed
                log.info(f'快乐8: doc_store加载了{len(source_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: doc_store加载失败: {e}')

        # 来源2: JSON文件
        file_records = {}
        path = Path(self.history_file)
        try:
            if path.exists():
                self._data_mtime = path.stat().st_mtime
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    source_list = raw.get('results', raw.get('data', []))
                else:
                    source_list = raw

                for r in source_list:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in file_records:
                        old = file_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'JSON文件内期号{issue}号码冲突')
                            save_conflict_to_queue({
                                'source': 'json_file_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    file_records[issue] = normed
                log.info(f'快乐8: 文件加载了{len(file_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: 文件加载失败: {e}')

        # 合并两个来源，按期号去重，冲突时报错不覆盖
        merged = {}
        for source_name, records in [('doc_store', source_records), ('json', file_records)]:
            for issue, record in records.items():
                if issue in merged:
                    old = merged[issue]
                    if old['numbers'] != record['numbers']:
                        log.error(
                            f'期号{issue}多源号码冲突: '
                            f'{source_name}={record["numbers"]}, '
                            f'已有={old["numbers"]}, 保留旧值待人工确认'
                        )
                        save_conflict_to_queue({
                            'source': source_name,
                            'issue': issue,
                            'old_numbers': old['numbers'],
                            'new_numbers': record['numbers'],
                            'old_source': old.get('source', ''),
                            'new_source': record.get('source', ''),
                            'action': 'kept_old',
                        })
                        continue
                merged[issue] = record

        if not merged:
            self.using_simulated_data = True
            log.error('快乐8: 无真实历史数据')
            return []

        # v5: 排序而非假设顺序（读取后确保按期号降序排列）
        data = sorted(merged.values(), key=lambda x: x['issue'], reverse=True)
        self.using_simulated_data = False

        # v5: 数据完整性检查
        integrity = check_data_integrity(data)
        if integrity.get('missing_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["missing_count"]}个缺失期号')
        if integrity.get('conflict_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["conflict_count"]}个日期期号不一致')

        log.info(f'快乐8: 多源合并后共{len(data)}期有效数据')
        return data

    def _check_data_mtime(self) -> bool:
        """检查数据文件mtime是否变化"""
        path = Path(self.history_file)
        try:
            if path.exists():
                current_mtime = path.stat().st_mtime
                if current_mtime != self._data_mtime:
                    log.info(f'快乐8: 数据文件mtime变化，需重新加载')
                    self._data_mtime = current_mtime
                    return True
        except Exception:
            pass
        return False

    def reload_if_needed(self) -> bool:
        """如果数据文件已更新则重新加载"""
        if self._check_data_mtime():
            self.history_data = self._load_history()
            self.update_statistics()
            return True
        return False

    # ─── 统计计算 ───

    def update_statistics(self):
        """更新所有统计量"""
        if not self.history_data:
            self.statistics = {}
            return

        n = len(self.history_data)
        recent = min(n, KL8_DEFAULT_HISTORY)
        recent_data = self.history_data[:recent]

        freq = Counter()
        for record in recent_data:
            for num in record['numbers']:
                freq[num] += 1

        gap = {}
        for num in range(1, 81):
            gap[num] = 0
            for record in recent_data:
                if num in record['numbers']:
                    break
                gap[num] += 1

        last_numbers = set(recent_data[0]['numbers']) if recent_data else set()

        self.statistics = {
            'frequency': freq,
            'gap': gap,
            'total_periods': recent,
            'expected_freq': recent * KL8_DRAW_COUNT / KL8_NUM_RANGE,
            'expected_gap': KL8_EXPECTED_GAP,
            'last_numbers': last_numbers,
            'freq_by_zone': self._zone_frequency(recent_data),
            'freq_by_road': self._road_frequency(recent_data),
            'freq_by_odd_even': self._odd_even_frequency(recent_data),
            'freq_by_big_small': self._big_small_frequency(recent_data),
        }

    def _zone_frequency(self, data: List[Dict]) -> Dict:
        """8个10码区的频率分布"""
        zone_freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                zone = (num - 1) // 10 + 1
                zone_freq[zone] += 1
        return dict(zone_freq)

    def _road_frequency(self, data: List[Dict]) -> Dict:
        """012路频率分布"""
        road_freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                road = num % 3
                road_freq[road] += 1
        return dict(road_freq)

    def _odd_even_frequency(self, data: List[Dict]) -> Dict:
        """奇偶频率分布"""
        freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                freq['odd' if num % 2 == 1 else 'even'] += 1
        return dict(freq)

    def _big_small_frequency(self, data: List[Dict]) -> Dict:
        """大小频率分布"""
        freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                freq['big' if num > 40 else 'small'] += 1
        return dict(freq)

    # ─── 对称评分函数 ───

    @staticmethod
    def balance_score(actual_ratio: float, target_ratio: float, is_target: bool) -> float:
        """对称平衡评分"""
        imbalance = target_ratio - actual_ratio
        delta = 0.30 * (imbalance if is_target else -imbalance)
        return max(0.2, min(0.8, 0.5 + delta))

    # ─── 特征评分 ───

    def _calculate_feature_score(self, num: int, repeat_direction: str = 'neutral',
                                     repeat_avoid_score: float = 0.10,
                                     repeat_non_avoid_score: float = 0.85,
                                     repeat_follow_score: float = 0.90,
                                     repeat_non_follow_score: float = 0.50) -> Dict[str, float]:
        """计算号码num的各特征得分

        v8改动:
        - position → position_residual: 区内频率 - 全局频率期望(剔除全局频率影响)
        - road → road_residual: 路内频率 - 全局频率期望(剔除全局频率影响)
        - repeat支持3种方向: neutral(0.5), avoid(上期0.10/非上期0.85), follow(上期0.90/非上期0.50)
        """
        scores = {}
        stats = self.statistics
        freq = stats['frequency']
        gap = stats['gap']
        expected_freq = stats['expected_freq']
        expected_gap = stats['expected_gap']
        last_nums = stats['last_numbers']
        total = stats['total_periods']

        # 1. 频率偏离度（全局冷热信号）
        actual_freq = freq.get(num, 0)
        deviation_ratio = actual_freq / max(expected_freq, 0.01)
        if deviation_ratio <= 1.0:
            scores['frequency'] = 0.55 + 0.15 * (1.0 - deviation_ratio)
        else:
            scores['frequency'] = max(0.15, 0.55 * math.exp(-1.8 * (deviation_ratio - 1.0)))

        # 2. 遗漏偏离度 -- 仅展示
        actual_gap_val = gap.get(num, 0)
        gap_ratio = actual_gap_val / max(expected_gap, 0.01)
        if gap_ratio <= 1.0:
            scores['gap'] = 0.25 + 0.60 * (gap_ratio ** 0.7)
        else:
            scores['gap'] = 0.85 - 0.45 * (1.0 - math.exp(-(gap_ratio - 1.0) * 0.8))

        # 3. 区内残差(position_residual): 该号频率 - 区内平均频率 / 全局平均频率
        #    剔除全局频率后，只保留"该号码在区内是否超出全局平均水平"的残差信号
        zone = (num - 1) // 10 + 1
        zone_nums = [z for z in range(((zone-1)*10)+1, zone*10+1)]
        num_freq = freq.get(num, 0)
        # 全局平均频率(每个号码期望出现次数)
        global_avg = expected_freq
        # 区内平均频率
        zone_total = sum(freq.get(z, 0) for z in zone_nums)
        zone_avg = zone_total / len(zone_nums)
        # 残差 = (num_freq - global_avg) vs (zone_avg - global_avg)
        # 如果号码频率 > 全局平均，但在区内也只是平均偏高，则残差为0（这只是区位偏移）
        # 如果号码频率 > 区内平均 + global_avg偏差，则真正是该号码自己的冷热信号
        if global_avg > 0:
            zone_deviation = zone_avg - global_avg  # 区位整体偏移
            num_residual = num_freq - zone_avg  # 号码在区内的偏移
            # 标准化: 冷号(负残差=低于区内平均)得高分，热号(正残差)得低分
            residual_ratio = num_residual / max(global_avg, 0.01)
            if residual_ratio <= 0:
                # 冷号(低于区内平均) → 均值回归加分
                scores['position_residual'] = 0.55 + 0.25 * min(1.0, abs(residual_ratio))
            else:
                # 热号(高于区内平均) → 均值回归降分
                scores['position_residual'] = max(0.15, 0.55 * math.exp(-1.5 * residual_ratio))
        else:
            scores['position_residual'] = 0.50

        # 4. 路内残差(road_residual): 同理，剔除全局频率
        road = num % 3
        road_nums = [r for r in range(1, 81) if r % 3 == road]
        road_total = sum(freq.get(r, 0) for r in road_nums)
        road_avg = road_total / len(road_nums)
        if global_avg > 0:
            road_deviation = road_avg - global_avg  # 路数整体偏移
            num_road_residual = num_freq - road_avg  # 号码在路内的偏移
            residual_ratio = num_road_residual / max(global_avg, 0.01)
            if residual_ratio <= 0:
                scores['road_residual'] = 0.55 + 0.20 * min(1.0, abs(residual_ratio))
            else:
                scores['road_residual'] = max(0.15, 0.55 * math.exp(-1.5 * residual_ratio))
        else:
            scores['road_residual'] = 0.50

        # 5. 和值特征 -- 停用
        scores['sum'] = 0.5

        # 6. 区位近期开出率 -- 停用(追上期模式不优于随机)
        scores['zone'] = 0.5

        # 7. 重号(v8: 支持3种候选方向)
        if repeat_direction == 'avoid':
            # 避开上期号
            scores['repeat'] = repeat_avoid_score if num in last_nums else repeat_non_avoid_score
        elif repeat_direction == 'follow':
            # 适度保留上期号
            scores['repeat'] = repeat_follow_score if num in last_nums else repeat_non_follow_score
        else:
            # 不处理重号(中性)
            scores['repeat'] = 0.50

        # 8. 邻号 -- 停用
        scores['adjacent'] = 0.5

        # 9. 奇偶 -- 对称评分(暂停用)
        scores['odd_even'] = 0.50

        # 10. 大小 -- 对称评分(暂停用)
        scores['big_small'] = 0.50

        return scores

    # ─── 排名模型（v5: 纯参数化，接受外部feature_weights）───

    def get_ensemble_ranking(self, top_n: int = 20, feature_weights: Optional[Dict[str, float]] = None,
                              repeat_direction: str = 'neutral',
                              repeat_avoid_score: float = 0.10,
                              repeat_non_avoid_score: float = 0.85,
                              repeat_follow_score: float = 0.90,
                              repeat_non_follow_score: float = 0.50) -> List[Dict]:
        """综合特征评分排名

        v8改动:
        - feature_weights 支持 position_residual/road_residual（残差特征）
        - repeat_direction 参数支持 neutral/avoid/follow
        """
        # 使用传入权重或全局活跃权重
        weights = feature_weights or get_active_feature_weights()

        # v5: 如果没有有效权重（全部为0），返回空列表
        has_weight = any(w > 0 for w in weights.values())
        if not has_weight:
            return []  # 无信号时不返回[1..20]

        ranking = []
        for num in range(1, 81):
            scores = self._calculate_feature_score(
                num,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )
            total_score = sum(
                scores.get(k, 0) * weights.get(k, 0) for k in scores
            )
            ranking.append({
                'num': num,
                'ranking_score': total_score,
                'score_type': 'heuristic_rank',
                'is_probability': False,
                'scores': scores,
            })
        ranking.sort(key=lambda x: (-x['ranking_score'], x['num']))
        return ranking[:top_n]

    # ─── 贝叶斯模型 ───

    def _model_bayesian(self, top_n: int = 20) -> List[int]:
        """贝叶斯概率模型 -- 目前停用

        v8修正:
        - 先验均值从0.5修正为0.25（符合80选20的理论开出率）
        - 使用 Beta(5, 15) 先验，均值=5/20=0.25
        - 启用前需做概率校准和样本外验证
        """
        stats = self.statistics
        freq = stats['frequency']
        total = stats['total_periods']

        # v8: Beta(5, 15) 先验，均值=0.25
        prior_alpha = 5
        prior_beta = 15
        prior_strength = prior_alpha + prior_beta  # = 20

        scores = {}
        for num in range(1, 81):
            count = freq.get(num, 0)
            # 后验均值 = (count + alpha) / (total + alpha + beta)
            base_prob = (count + prior_alpha) / (total + prior_strength)

            # 均值回归因子：偏离理论率0.25越多，回归越强
            deviation_ratio = base_prob / 0.25  # 理论率=0.25
            reversion_factor = 1.0 / (1.0 + 0.6 * max(0, deviation_ratio - 1.0))
            if deviation_ratio < 1.0:
                reversion_factor = min(1.5, 1.0 + 0.5 * (1.0 - deviation_ratio))

            scores[num] = base_prob * reversion_factor

        return sorted(scores.keys(), key=lambda n: (-scores[n], n))[:top_n]

    # ─── 马尔可夫模型（确定性哈希打破并列）───

    def _model_markov(self, top_n: int = 20) -> List[int]:
        """一阶马尔可夫转移模型 -- 目前停用"""
        if len(self.history_data) < 3:
            return []

        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.history_data) - 1):
            current = set(self.history_data[i]['numbers'])
            prev = set(self.history_data[i + 1]['numbers'])
            for num in prev:
                if num in current:
                    transition_counts[num]['repeat'] += 1
                else:
                    transition_counts[num]['skip'] += 1

        last_nums = set(self.history_data[0]['numbers'])
        based_on_issue = self.history_data[0]['issue']

        scores = {}
        for num in range(1, 81):
            base_score = 0.25
            if num in last_nums:
                repeat_rate = transition_counts[num]['repeat'] / max(
                    transition_counts[num]['repeat'] + transition_counts[num]['skip'], 1)
                base_score = max(0.15, repeat_rate)
            tie_break = int(hashlib.sha256(f'{based_on_issue}_{num}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            scores[num] = base_score + tie_break * 0.001

        return sorted(scores.keys(), key=lambda n: (-scores[n], n))[:top_n]

    # ─── 排名模型(独立) ───

    def _model_rank(self, top_n: int = 20, feature_weights: Optional[Dict[str, float]] = None,
                     repeat_direction: str = 'neutral',
                     repeat_avoid_score: float = 0.10,
                     repeat_non_avoid_score: float = 0.85,
                     repeat_follow_score: float = 0.90,
                     repeat_non_follow_score: float = 0.50) -> List[int]:
        """纯排名模型"""
        ranking = self.get_ensemble_ranking(
            top_n=top_n, feature_weights=feature_weights,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
        )
        return [r['num'] for r in ranking]

    # ─── 多模型投票（v5: 纯参数化 + 预测就绪判断）───

    def multi_model_voting(
        self,
        pick_n: int = 5,
        top_n: int = 20,
        feature_weights: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        repeat_direction: str = 'neutral',
        repeat_avoid_score: float = 0.10,
        repeat_non_avoid_score: float = 0.85,
        repeat_follow_score: float = 0.90,
        repeat_non_follow_score: float = 0.50,
    ) -> Dict:
        """多模型集成投票

        v8改动:
        - 传递 repeat_direction 到 get_ensemble_ranking
        """
        fw = feature_weights or get_active_feature_weights()
        mw = model_weights or get_active_model_weights()

        # ── 预测就绪判断 ──
        has_rank_feature = any(w > 0 for w in fw.values())
        rank_weight = mw.get('rank', 0.0)
        bayesian_weight = mw.get('bayesian', 0.0)
        markov_weight = mw.get('markov', 0.0)

        rank_ready = rank_weight > 0 and has_rank_feature
        bayesian_ready = bayesian_weight > 0
        markov_ready = markov_weight > 0

        if not (rank_ready or bayesian_ready or markov_ready):
            return {
                'selected': [],
                'candidates': [],
                'votes': {},
                'status': 'no_validated_signal',
                'message': '暂无通过回测验证的有效特征，不输出号码推荐',
                'version': KL8_PREDICTOR_VERSION,
            }

        votes = defaultdict(float)

        # 懒加载: 只计算启用模型
        if rank_ready:
            model_result = self._model_rank(
                top_n=top_n, feature_weights=fw,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * rank_weight
                votes[num] += vote_weight

        if bayesian_ready:
            model_result = self._model_bayesian(top_n=top_n)
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * bayesian_weight
                votes[num] += vote_weight

        if markov_ready:
            model_result = self._model_markov(top_n=top_n)
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * markov_weight
                votes[num] += vote_weight

        candidates = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
        selected = [num for num, _ in candidates[:pick_n]]
        candidate_pool = candidates[:max(top_n, 7)]

        return {
            'selected': selected,
            'candidates': candidate_pool,
            'votes': dict(votes),
            'version': KL8_PREDICTOR_VERSION,
        }

    # ─── 选5复式7码 ───

    def get_fu_shi_7(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        repeat_direction: str = 'neutral',
        repeat_avoid_score: float = 0.10,
        repeat_non_avoid_score: float = 0.85,
        repeat_follow_score: float = 0.90,
        repeat_non_follow_score: float = 0.50,
    ) -> Dict:
        """选5复式7码"""
        vote_result = self.multi_model_voting(
            pick_n=7, top_n=7,
            feature_weights=feature_weights,
            model_weights=model_weights,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
        )

        if vote_result.get('status') == 'no_validated_signal':
            return {
                'top7_numbers': [],
                'total_combinations': 0,
                'combinations': [],
                'version': KL8_PREDICTOR_VERSION,
                'source': 'multi_model_voting',
                'status': 'no_validated_signal',
                'message': vote_result.get('message', ''),
            }

        top7 = vote_result['selected']
        combo_list = [sorted(c) for c in combinations(top7, 5)]

        ranking_full = self.get_ensemble_ranking(
            top_n=7,
            feature_weights=feature_weights,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
        )
        top7_details = [r for r in ranking_full if r['num'] in top7]
        top7_details.sort(key=lambda x: (-x['ranking_score'], x['num']))

        return {
            'top7_numbers': top7,
            'top7_scores': [r['ranking_score'] for r in top7_details],
            'total_combinations': len(combo_list),
            'combinations': combo_list,
            'version': KL8_PREDICTOR_VERSION,
            'source': 'multi_model_voting',
        }

    # ─── 预测快照 ───

    def _save_prediction_snapshot(self, prediction_result: Dict) -> Optional[str]:
        """保存预测快照（v9: 唯一约束 + is_experiment标记）

        v9改动:
        - 同一策略+同一目标期只保留一份正式快照
        - 其他重复快照标记 is_experiment=True，不进入正式命中率统计
        """
        if not self.history_data:
            return None

        snapshot_dir = Path(KL8_SNAPSHOT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # v9: 策略指纹 — 用于唯一约束
        # 使用 select_5 的策略作为基准指纹（所有玩法当前使用同一策略）
        resolved = prediction_result.get('resolved_strategies', {})
        base_strategy = resolved.get('select_5', {})
        strategy_fp = _strategy_fingerprint(base_strategy) if base_strategy else 'no_strategy'

        # v9: 检查是否已有同一目标期+策略指纹的正式快照
        target_issue_val = _compute_next_issue(self.history_data[0]['issue'], self.history_data)
        snapshot_key = f'{target_issue_val}_{strategy_fp}'
        is_experiment = False

        # 扫描已有快照
        for existing_file in snapshot_dir.glob('snapshot_*.json'):
            try:
                existing_data = json.loads(existing_file.read_text(encoding='utf-8'))
                existing_key = f'{existing_data.get("target_issue", "")}_{_strategy_fingerprint(existing_data.get("resolved_strategies", {}).get("select_5", {}))}'
                if existing_key == snapshot_key and not existing_data.get('is_experiment', False):
                    # 已有正式快照 → 新快照标记为实验
                    is_experiment = True
                    log.info(f'快乐8: 同期同策略已有正式快照，新快照标记为实验预测')
                    break
            except Exception:
                continue

        # 全窗口SHA256指纹
        recent = min(len(self.history_data), KL8_DEFAULT_HISTORY)
        history_window = self.history_data[:recent]
        history_fingerprint = hashlib.sha256(
            json.dumps(
                [
                    {
                        'issue': r['issue'],
                        'numbers': r['numbers'],
                        'date': r.get('date', ''),
                        'checksum': r.get('checksum', ''),
                    }
                    for r in history_window
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()

        latest_issue = self.history_data[0]['issue']
        # v9: target_issue 推算改进 + target_type 严格校验
        # 不再简单用 int(latest_issue)+1，而是从历史数据推导下一期期号模式
        # 同时保存 target_type='next_draw_after_based_on'，结算时验证actual是based_on的直接下一期
        target_issue = _compute_next_issue(latest_issue, self.history_data)
        target_type = 'next_draw_after_based_on'
        snapshot_id = uuid.uuid4().hex
        snapshot = {
            'snapshot_id': snapshot_id,
            'target_issue': target_issue,  # v9: 从历史模式推算（用于调度器匹配）
            'target_type': target_type,     # v9: 结算时验证actual是based_on的直接下一期
            'based_on_issue': latest_issue,
            'predicted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'history_window_size': recent,
            'history_start_issue': history_window[-1]['issue'] if history_window else '',
            'history_end_issue': latest_issue,
            'history_fingerprint': history_fingerprint,
            'strategy_fingerprint': strategy_fp,  # v9: 策略指纹
            'is_experiment': is_experiment,  # v9: 实验预测标记
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': {k: dict(v) for k, v in FEATURE_CONFIG.items()},
            'model_config': {k: dict(v) for k, v in MODEL_CONFIG.items()},
            'active_strategies': {k: dict(v) for k, v in ACTIVE_STRATEGIES.items()},
            'reference_strategy': dict(REFERENCE_STRATEGY),
            'candidate_strategies': {k: dict(v) for k, v in CANDIDATE_STRATEGIES.items()},
            # v7.1: 每个玩法记录strategy_id和prediction_mode
            'play_strategies': {
                'select_3': prediction_result.get('select_3', {}).get('strategy_id', ''),
                'select_4': prediction_result.get('select_4', {}).get('strategy_id', ''),
                'select_5': prediction_result.get('select_5', {}).get('strategy_id', ''),
                'select_6': prediction_result.get('select_6', {}).get('strategy_id', ''),
                'select_7': prediction_result.get('select_7', {}).get('strategy_id', ''),
                'fu_shi_7': prediction_result.get('fu_shi_7', {}).get('strategy_id', ''),
            },
            'prediction_modes': {
                'select_3': prediction_result.get('select_3', {}).get('prediction_mode', ''),
                'select_4': prediction_result.get('select_4', {}).get('prediction_mode', ''),
                'select_5': prediction_result.get('select_5', {}).get('prediction_mode', ''),
                'select_6': prediction_result.get('select_6', {}).get('prediction_mode', ''),
                'select_7': prediction_result.get('select_7', {}).get('prediction_mode', ''),
                'fu_shi_7': prediction_result.get('fu_shi_7', {}).get('prediction_mode', ''),
            },
            # v9: 保存每种玩法当时实际使用的完整策略配置
            'resolved_strategies': prediction_result.get('resolved_strategies', {}),
            'ranking': prediction_result.get('ranking', []),
            'select_3': prediction_result.get('select_3', {}).get('numbers', []),
            'select_4': prediction_result.get('select_4', {}).get('numbers', []),
            'select_5': prediction_result.get('select_5', {}).get('numbers', []),
            'select_6': prediction_result.get('select_6', {}).get('numbers', []),
            'select_7': prediction_result.get('select_7', {}).get('numbers', []),
            'fu_shi_7': prediction_result.get('fu_shi_7', {}).get('top7_numbers', []),
        }

        snapshot_file = snapshot_dir / f'snapshot_{snapshot_id}.json'

        try:
            with snapshot_file.open('x', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 预测快照已保存 -> {snapshot_file.name}')
            return snapshot_file.name
        except FileExistsError:
            log.error(f'快乐8: 快照UUID冲突 {snapshot_file.name}')
            return None
        except Exception as e:
            log.error(f'快乐8: 保存快照失败: {e}')
            return None

    def settle_prediction(self, snapshot_file: str, actual_issue: str, actual_numbers: List[int]) -> Dict:
        """赛后结算（v6: 无号码不扣投注+ROI统一+期号校验）

        v6改动:
        1. 只有len(numbers)==select_type时才视为placed（已投注）
        2. 未placed的玩法bet=0, prize=0
        3. ROI统一为return_multiple和profit_roi两个字段
        4. 复式7码每期全部21注组合都计算投注和奖金
        5. actual_issue必须晚于based_on_issue
        """

        # 路径安全
        snapshot_file = Path(snapshot_file).name
        path = Path(KL8_SNAPSHOT_DIR) / snapshot_file
        if not path.exists():
            return {'error': f'快照文件不存在: {snapshot_file}'}

        # 校验开奖号码
        actual_normed = normalize_record({
            'issue': actual_issue,
            'numbers': actual_numbers,
        })
        if not actual_normed:
            return {'error': '实际开奖号码非法'}

        # 读取原始快照
        try:
            snapshot = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            return {'error': f'读取快照失败: {e}'}

        # v6: 期号校验 — actual_issue必须晚于based_on_issue
        based_on_issue = snapshot.get('based_on_issue', '')
        target_issue = snapshot.get('target_issue', '')

        # v9: 严格校验 — target_issue必须与实际开奖期号一致
        # 不再只检查"actual > based_on"，而是检查actual == target_issue
        # 或者actual是based_on_issue的直接下一期（target_type校验）
        if target_issue:
            if str(target_issue) != str(actual_normed['issue']):
                # 容许：actual_issue是based_on_issue的直接下一期（跨年/停开/补期场景）
                target_type = snapshot.get('target_type', '')
                if target_type == 'next_draw_after_based_on':
                    # 验证actual_issue确实是based_on_issue在历史数据中的直接下一期
                    analyzer = get_kl8_analyzer()
                    history_asc = sorted(analyzer.history_data, key=lambda x: x['issue'])
                    # 找到based_on_issue的位置
                    based_on_idx = None
                    for idx, rec in enumerate(history_asc):
                        if rec['issue'] == based_on_issue:
                            based_on_idx = idx
                            break
                    if based_on_idx is not None and based_on_idx + 1 < len(history_asc):
                        next_issue = history_asc[based_on_idx + 1]['issue']
                        if str(next_issue) == str(actual_normed['issue']):
                            pass  # 验证通过：actual确实是based_on的直接下一期
                        else:
                            return {
                                'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                                         f'(based_on的直接下一期是{next_issue})'
                            }
                    else:
                        return {
                            'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                        }
                else:
                    return {
                        'error': f'快照目标期号{target_issue}与实际期号{actual_normed["issue"]}不一致'
                    }

        # 旧版兜底：如果没有target_issue，仍保留 based_on < actual 的宽松校验
        if not target_issue and based_on_issue:
            try:
                actual_int = int(actual_normed['issue'])
                based_on_int = int(based_on_issue)
                if actual_int <= based_on_int:
                    return {'error': f'实际开奖期号{actual_normed["issue"]}必须晚于预测基准期号{based_on_issue}'}
            except (ValueError, TypeError):
                if str(actual_normed['issue']) <= str(based_on_issue):
                    return {'error': f'实际开奖期号必须晚于预测基准期号'}

        # 检查是否已结算
        settlements_dir = Path(KL8_SETTLEMENT_DIR)
        settlements_dir.mkdir(parents=True, exist_ok=True)

        existing_settlement = settlements_dir / f'settlement_{snapshot.get("snapshot_id", "")}.json'
        if existing_settlement.exists():
            try:
                old = json.loads(existing_settlement.read_text(encoding='utf-8'))
                return {'error': '快照已结算，不可重复结算', 'settlement': old}
            except Exception:
                pass

        snapshot_sha256 = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        actual_set = set(actual_normed['numbers'])

        # v6: 奖金结算 — 只有placed=true才计投注和奖金
        prize_table = load_prize_table()

        prize_settlement = {}
        cumulative_bet = 0
        cumulative_prize = 0

        for select_type in [3, 4, 5, 6, 7]:
            numbers = _clean_pick_numbers(
                snapshot.get(f'select_{select_type}', []),
                select_type,
            )
            prize_key = f'select_{select_type}'
            prize_info = prize_table.get(prize_key, {})

            # v6: 只有号码完整时才视为placed
            placed = len(numbers) == select_type
            bet = prize_info.get('bet', 2) if placed else 0
            hits = len(set(numbers) & actual_set) if placed else 0
            prize = prize_info.get(str(hits), 0) if placed else 0

            cumulative_bet += bet
            cumulative_prize += prize

            # v6: ROI统一为两个字段
            return_multiple = prize / max(bet, 1) if placed else 0
            profit_roi = (prize - bet) / max(bet, 1) if placed else 0

            prize_settlement[prize_key] = {
                'placed': placed,
                'hits': hits,
                'bet': bet,
                'prize': prize,
                'return_multiple': round(return_multiple, 4),
                'profit_roi': round(profit_roi, 4),
            }

        # v6: 复式7码ROI — 每期全部21注组合都计算
        fu_shi_7_nums = _clean_pick_numbers(snapshot.get('fu_shi_7', []), 7)
        fu7_placed = len(fu_shi_7_nums) == 7

        fu7_prize_info = prize_table.get('fu_shi_7', {})
        bet_per_combo = fu7_prize_info.get('bet_per_combo', 2)

        pool_hits = 0
        combo_hits = []
        fu7_total_bet = 0
        fu7_total_prize = 0

        if fu7_placed and fu_shi_7_nums:
            pool_hits = len(set(fu_shi_7_nums) & actual_set)
            # v6: 每期全部21注组合都计算（无论命中率）
            fu7_total_bet = math.comb(7, 5) * bet_per_combo  # = 21 * 2 = 42
            for combo in combinations(fu_shi_7_nums, 5):
                combo_h = len(set(combo) & actual_set)
                combo_hits.append(combo_h)
                fu7_total_prize += fu7_prize_info.get(str(combo_h), 0)

        cumulative_bet += fu7_total_bet
        cumulative_prize += fu7_total_prize

        hit_distribution = dict(Counter(combo_hits)) if combo_hits else {}
        max_combo_hits = max(combo_hits, default=0)

        # v6: 累计ROI统一
        cumulative_return_multiple = cumulative_prize / max(cumulative_bet, 1)
        cumulative_profit_roi = (cumulative_prize - cumulative_bet) / max(cumulative_bet, 1)

        settlement = {
            'snapshot_id': snapshot.get('snapshot_id', ''),
            'snapshot_file': snapshot_file,
            'snapshot_sha256': snapshot_sha256,
            'settled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'actual_issue': actual_normed['issue'],
            'actual_numbers': actual_normed['numbers'],
            'actual_checksum': _checksum_numbers(actual_normed['numbers']),
            'based_on_issue': snapshot.get('based_on_issue', ''),
            'strategy_ids': dict(snapshot.get('play_strategies', {})),  # v9: 从快照读取
            'prediction_modes': dict(snapshot.get('prediction_modes', {})),  # v9: 从快照读取
            'resolved_strategies': dict(snapshot.get('resolved_strategies', {})),  # v9: 完整策略快照
            'hit_select_3': prize_settlement.get('select_3', {}).get('hits', 0),
            'hit_select_4': prize_settlement.get('select_4', {}).get('hits', 0),
            'hit_select_5': prize_settlement.get('select_5', {}).get('hits', 0),
            'hit_select_6': prize_settlement.get('select_6', {}).get('hits', 0),
            'hit_select_7': prize_settlement.get('select_7', {}).get('hits', 0),
            'fu_shi_7_pool_hits': pool_hits,
            'hit_fu_shi_7_max': max_combo_hits,
            'fu_shi_7_hit_distribution': hit_distribution,
            'prize_settlement': prize_settlement,
            'fu7_total_bet': fu7_total_bet,
            'fu7_total_prize': fu7_total_prize,
            'cumulative_bet': cumulative_bet,
            'cumulative_prize': cumulative_prize,
            'cumulative_return_multiple': round(cumulative_return_multiple, 4),
            'cumulative_profit_roi': round(cumulative_profit_roi, 4),
        }

        try:
            with existing_settlement.open('x', encoding='utf-8') as f:
                json.dump(settlement, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 结算完成 -> {existing_settlement.name}')
            return {'success': True, 'settlement': settlement}
        except FileExistsError:
            return {'error': '结算文件已存在，不可重复结算'}
        except Exception as e:
            return {'error': f'写入结算失败: {e}'}

    # ─── 窗口分析器构造（v7.1新增）───

    def _build_window_analyzer(self, window_size: int):
        """构造临时分析器（与回测逻辑完全一致）

        策略指定window_size时，必须创建临时分析器并调用update_statistics()
        确保线上预测和回测使用相同的统计窗口，而不是只临时计算freq。

        参数:
            window_size: 统计窗口大小，0或None时使用KL8_DEFAULT_HISTORY
        """
        recent = min(len(self.history_data), window_size or KL8_DEFAULT_HISTORY)

        temp = KL8Analyzer.__new__(KL8Analyzer)
        temp.history_data = self.history_data[:recent]
        temp.using_simulated_data = False
        temp.history_file = self.history_file
        temp._data_mtime = self._data_mtime
        temp.statistics = {}
        temp.update_statistics()

        return temp

    # ─── 统一候选池（v9新增）───

    def build_candidate_pool(self) -> Tuple[Dict, Dict]:
        """统一候选池: 所有玩法从同一份 Top20 截取

        v9新增:
        - 统一调用 multi_model_voting(pick_n=20, top_n=20)
        - 所有玩法(选3~选7、复式7码)都从同一份结果截取
        - 确保线上预测与回测管道完全一致

        返回:
            (pool_result, strategy) — 候选池投票结果和使用的策略
        """
        # 使用已验证策略或参考策略（当前所有玩法都用同一策略）
        # 取 select_5 的策略作为基准（最常用的玩法）
        strategy = resolve_play_strategy('select_5')

        repeat_direction = strategy.get('repeat_direction', 'neutral')
        repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
        repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
        repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
        repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)

        predictor = self._build_window_analyzer(
            strategy.get('window_size', KL8_DEFAULT_HISTORY)
        )

        pool_result = predictor.multi_model_voting(
            pick_n=20,
            top_n=20,
            feature_weights=strategy['feature_weights'],
            model_weights=strategy['model_weights'],
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
        )

        return pool_result, strategy

    # ─── 综合预测（v9: 统一候选池 + resolved_strategies）───

    def predict_all(self) -> Dict:
        """生成所有选型的预测结果

        v9改动:
        - 统一候选池: 所有玩法都从同一份 Top20 截取，不再按玩法分别调用
        - resolved_strategies: 快照保存每种玩法当时的完整策略配置
        - 快照唯一约束: target_issue + strategy_fingerprint 唯一
        """
        if not self.history_data or self.using_simulated_data:
            return {
                'error': '历史数据不足，无法进行有效预测。请先抓取真实数据。',
                'using_simulated_data': True,
            }

        prediction_ready = is_prediction_ready()

        results = {}
        resolved_strategies = {}  # v9: 保存每种玩法当时的完整策略

        # v9: 统一候选池 — 所有玩法使用同一份策略
        # 如果已验证策略不同玩法不同，取权重最大的那个策略生成 Top20
        # 实际上当前所有玩法都降级到 REFERENCE_STRATEGY，所以天然统一
        # 未来如果不同玩法有不同策略，仍然用 build_candidate_pool() 统一生成
        pool_result, pool_strategy = build_candidate_pool(self)

        # v9: 从统一候选池截取各玩法号码
        top20 = pool_result.get('selected', [])[:20]
        candidate_pool = pool_result.get('candidates', [])[:20]

        for select_type in [3, 4, 5, 6, 7]:
            config = SELECT_CONFIG[select_type]
            s_key = f'select_{select_type}'

            # v9: 从统一候选池截取
            strategy = resolve_play_strategy(s_key)

            # v9: 保存完整策略配置到 resolved_strategies
            resolved_strategies[s_key] = {
                'strategy_id': strategy['strategy_id'],
                'feature_weights': strategy['feature_weights'],
                'model_weights': strategy['model_weights'],
                'window_size': strategy.get('window_size', KL8_DEFAULT_HISTORY),
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
                'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
                'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
                'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
            }

            if len(top20) >= select_type:
                numbers = top20[:select_type]
            else:
                numbers = top20  # 不够时用全部

            results[s_key] = {
                'desc': config['desc'],
                'pick': config['pick'],
                'numbers': numbers,
                'candidates': candidate_pool[:10],
                'strategy_id': strategy['strategy_id'],
                'prediction_mode': strategy['prediction_mode'],
                'is_validated': strategy['is_validated'],
                'warning': (
                    '' if strategy['is_validated']
                    else '参考预测：当前策略尚未通过严格回测验证，仅供数据观察。'
                ),
            }

        # 复式7码（v9: 从统一候选池截取前7）
        strategy = resolve_play_strategy('fu_shi_7')

        resolved_strategies['fu_shi_7'] = {
            'strategy_id': strategy['strategy_id'],
            'feature_weights': strategy['feature_weights'],
            'model_weights': strategy['model_weights'],
            'window_size': strategy.get('window_size', KL8_DEFAULT_HISTORY),
            'repeat_direction': strategy.get('repeat_direction', 'neutral'),
            'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
            'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
            'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
            'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
            'prediction_mode': strategy['prediction_mode'],
            'is_validated': strategy['is_validated'],
        }

        top7 = top20[:7] if len(top20) >= 7 else top20

        if len(top7) == 7:
            combo_list = [sorted(c) for c in combinations(top7, 5)]
        else:
            combo_list = []

        results['fu_shi_7'] = {
            'top7_numbers': top7,
            'total_combinations': len(combo_list),
            'combinations': combo_list,
            'strategy_id': strategy['strategy_id'],
            'prediction_mode': strategy['prediction_mode'],
            'is_validated': strategy['is_validated'],
            'warning': (
                '' if strategy['is_validated']
                else '参考预测：当前策略尚未通过严格回测验证，仅供数据观察。'
            ),
        }

        # v9: 保存 resolved_strategies 到 results，以便 _save_prediction_snapshot 使用
        results['resolved_strategies'] = resolved_strategies

        recent = self.history_data[:10] if self.history_data else []
        results['recent_results'] = [
            {'issue': r['issue'], 'numbers': r['numbers'], 'date': r['date']}
            for r in recent
        ]

        # v7.1: 状态分三种: validated / reference_unvalidated / no_data
        # 判断整体预测模式
        all_modes = [results.get(f'select_{st}', {}).get('prediction_mode', '') for st in [3,4,5,6,7]]
        all_modes.append(results.get('fu_shi_7', {}).get('prediction_mode', ''))

        if any(m == 'validated' for m in all_modes):
            overall_status = 'validated'
        elif any(m == 'reference_unvalidated' for m in all_modes):
            overall_status = 'reference_unvalidated'
        else:
            overall_status = 'no_data'

        stats = self.statistics
        results['statistics'] = {
            'total_periods': stats.get('total_periods', 0),
            'expected_freq': round(stats.get('expected_freq', 2), 2),
            'expected_gap': round(stats.get('expected_gap', 1), 1),
            'last_numbers': sorted(list(stats.get('last_numbers', set()))),
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': FEATURE_CONFIG,
            'active_feature_weights': get_active_feature_weights(),
            'model_config': MODEL_CONFIG,
            'active_model_weights': get_active_model_weights(),
            'active_strategies': ACTIVE_STRATEGIES,
            'reference_strategy': REFERENCE_STRATEGY,
            'candidate_strategies': CANDIDATE_STRATEGIES,
            'is_prediction_ready': prediction_ready,
            'signal_status': overall_status,
            'note': (
                '当前启用策略已通过回测验证。'
                if overall_status == 'validated'
                else '参考预测模式：当前特征未通过严格回测验证，系统仍会输出基础统计参考号码，但不代表策略已验证有效，请勿将其视为高置信度推荐。'
                if overall_status == 'reference_unvalidated'
                else '历史数据不足，无法进行预测。'
            ),
        }

        results['ranking'] = [
            {
                'num': num,
                'ranking_score': score,
                'score_type': 'candidate_pool_vote',
                'is_probability': False,
            }
            for num, score in candidate_pool
        ]

        results['using_simulated_data'] = self.using_simulated_data

        # 数据完整性信息
        if self.history_data:
            integrity = check_data_integrity(self.history_data)
            results['data_integrity'] = integrity

        snapshot_name = self._save_prediction_snapshot(results)
        if snapshot_name:
            results['snapshot_file'] = snapshot_name

        return results


# ─── 单例与缓存 ───

_analyzer_instance = None
_prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def get_kl8_analyzer() -> KL8Analyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = KL8Analyzer()
    return _analyzer_instance


def build_candidate_pool(analyzer: Optional[KL8Analyzer] = None) -> Tuple[Dict, Dict]:
    """统一候选池入口函数

    v9新增: 所有玩法从同一份 Top20 截取，确保线上与回测一致
    """
    analyzer = analyzer or get_kl8_analyzer()
    return analyzer.build_candidate_pool()


def run_prediction(force_refresh: bool = False) -> Dict:
    """快乐8预测入口（v8: 缓存指纹包含所有策略配置）"""
    analyzer = get_kl8_analyzer()

    if not force_refresh:
        if analyzer.reload_if_needed():
            force_refresh = True

    if not analyzer.history_data:
        return {
            'error': '历史数据不足',
            'using_simulated_data': True,
        }

    # v8: 缓存指纹包含 ACTIVE_STRATEGIES + REFERENCE_STRATEGY + CANDIDATE_STRATEGIES
    config_fingerprint = hashlib.sha256(
        json.dumps(
            {
                'active_strategies': ACTIVE_STRATEGIES,
                'reference_strategy': REFERENCE_STRATEGY,
                'candidate_strategies': CANDIDATE_STRATEGIES,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()[:16]

    cache_key = (
        analyzer.history_data[0]['issue'],
        len(analyzer.history_data),
        KL8_PREDICTOR_VERSION,
        config_fingerprint,
    )

    cache = _prediction_cache
    if not force_refresh and cache['data'] is not None and cache.get('cache_key') == cache_key:
        return cache['data']

    result = analyzer.predict_all()

    cache['data'] = result
    cache['cache_key'] = cache_key
    cache['timestamp'] = time.time()

    return result


def clear_cache():
    global _analyzer_instance, _prediction_cache
    _analyzer_instance = None
    _prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def list_prediction_snapshots() -> List[Dict]:
    snapshot_dir = Path(KL8_SNAPSHOT_DIR)
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for f in sorted(snapshot_dir.glob('snapshot_*.json')):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            snapshots.append({
                'file': f.name,
                'snapshot_id': data.get('snapshot_id', ''),
                'target_issue': data.get('target_issue'),
                'based_on_issue': data.get('based_on_issue'),
                'predicted_at': data.get('predicted_at'),
                'version': data.get('version'),
                'is_experiment': data.get('is_experiment', False),
                'strategy_fingerprint': data.get('strategy_fingerprint', ''),
                'prediction_modes': data.get('prediction_modes', {}),
                'play_strategies': data.get('play_strategies', {}),
                'has_settlement': _check_settlement_exists(data.get('snapshot_id', '')),
            })
        except Exception:
            continue

    return snapshots


def _check_settlement_exists(snapshot_id: str) -> bool:
    if not snapshot_id:
        return False
    return (Path(KL8_SETTLEMENT_DIR) / f'settlement_{snapshot_id}.json').exists()


# ─── 滚动回测模块（v5: 真正三段式+超几何+置换检验+纯参数化+按玩法分开）───

class KL8RollingBacktest:
    """快乐8严格滚动回测 v6

    核心设计:
    - 真正三段式: train→val→final_test，final_test完全冻结
    - 回测使用multi_model_voting管道（model_weights真正生效）
    - 置换检验按玩法分别执行（pick_n=select_type）
    - 多重检验校正（BH FDR）
    - 纯参数化回测，不修改全局配置
    - 策略注册表ACTIVE_STRATEGIES按玩法分别配置
    """

    def __init__(self, analyzer: Optional[KL8Analyzer] = None):
        self.analyzer = analyzer or get_kl8_analyzer()

    # ─── 超几何分布随机基线 ───

    def hypergeom_baseline(self, pick_n: int) -> Dict:
        """超几何分布随机基线

        快乐8从80个号码中不放回开出20个，
        选中pick_n个号码命中k个的概率应使用超几何分布
        """
        expected = hypergeom_expected(pick_n)

        probs = {}
        for k in range(1, pick_n + 1):
            probs[f'>={k}'] = round(hypergeom_p_ge(pick_n, k), 6)

        return {
            'expected_hits': expected,
            'probabilities': probs,
            'distribution': 'hypergeometric',
            'note': '80选20不放回，使用超几何分布而非二项近似',
        }

    # ─── 真正三段式数据分割 ───

    def _split_three_stage(self, total_periods: int, val_periods: int = BACKTEST_MIN_OOS_PERIODS, final_test_periods: int = BACKTEST_FINAL_TEST_PERIODS) -> Dict:
        """真正三段式数据分割（v6: 固定段数，final_test完全冻结）

        train: 生成候选特征/窗口/权重
        val: 只选一个最终策略
        final_test: 完全冻结，不参与任何启用判断，只用于最终成绩报告

        规则:
        - train_end = n - val_periods - final_test_periods
        - 如果train不足100期，数据不够，抛异常
        - final_test的任何结果都不影响is_candidate或recommendations
        """
        train_end = total_periods - val_periods - final_test_periods
        if train_end < 100:
            raise ValueError(
                f'历史数据不足: 需train>=100+val={val_periods}+test={final_test_periods}'
                f'={100+val_periods+final_test_periods}期，现有{total_periods}期'
            )

        return {
            'train': (0, train_end),
            'val': (train_end, train_end + val_periods),
            'final_test': (train_end + val_periods, total_periods),
        }

    # ─── 纯参数化回测（不修改全局配置）───

    def _rolling_backtest_parametric(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        start_idx: int,
        end_idx: int,
        min_train: int = 50,
        window_size: Optional[int] = None,
        repeat_direction: str = 'neutral',
        repeat_avoid_score: float = 0.10,
        repeat_non_avoid_score: float = 0.85,
        repeat_follow_score: float = 0.90,
        repeat_non_follow_score: float = 0.50,
    ) -> Dict:
        """纯参数化滚动回测（v6: 使用multi_model_voting管道，model_weights真正生效）

        不修改全局FEATURE_CONFIG/MODEL_CONFIG，直接传权重参数
        第t期预测只能使用t期之前的历史数据
        使用真实投票管道回测，与线上逻辑一致

        v6关键改动:
        - 使用multi_model_voting()而非get_ensemble_ranking()
        - top20从投票结果截取，后续选3/5/7从同一份top20截取
        - model_weights真正参与（bayesian/markov权重生效）
        """
        history = self.analyzer.history_data
        history_asc = sorted(history, key=lambda x: x['issue'])
        n = len(history_asc)

        actual_end = min(end_idx, n)
        actual_start = max(start_idx, min_train)

        if actual_end - actual_start < 10:
            return {'error': '数据不足'}

        # 使用指定窗口大小（如果不指定，用全部可用历史）
        effective_window = window_size or KL8_DEFAULT_HISTORY

        all_hits = defaultdict(list)
        all_fu_shi_7_pool_hits = []
        all_fu_shi_7_combo_hits_detail = []  # v6: 每期所有组合命中详情（用于ROI）

        for t in range(actual_start, actual_end):
            # 只用t期之前的历史，限制窗口大小
            train_end_idx = t
            train_start_idx = max(0, t - effective_window)
            train_data = history_asc[train_start_idx:train_end_idx]

            if len(train_data) < min_train:
                continue

            actual_numbers = set(history_asc[t]['numbers'])

            # 构造临时分析器（不修改全局配置）
            temp_analyzer = KL8Analyzer.__new__(KL8Analyzer)
            temp_analyzer.history_data = sorted(train_data, key=lambda x: x['issue'], reverse=True)
            temp_analyzer.using_simulated_data = False
            temp_analyzer.history_file = ''
            temp_analyzer._data_mtime = 0
            temp_analyzer.update_statistics()

            # v6: 使用真实投票管道（model_weights真正参与）
            vote = temp_analyzer.multi_model_voting(
                pick_n=20,
                top_n=20,
                feature_weights=feature_weights,
                model_weights=model_weights,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            # 无信号时，该期命中数记录为0
            if vote.get('status') == 'no_validated_signal':
                for select_type in [3, 4, 5, 6, 7]:
                    all_hits[select_type].append(0)
                all_fu_shi_7_pool_hits.append(0)
                all_fu_shi_7_combo_hits_detail.append([])
                continue

            # v6: 从投票结果截取top20（与线上逻辑一致）
            top20 = [num for num, _ in vote.get('candidates', [])]

            if len(top20) < 7:
                # 号码不足7个，无法做复式
                for select_type in [3, 4, 5, 6, 7]:
                    top_nums = top20[:select_type] if len(top20) >= select_type else []
                    hits = len(set(top_nums) & actual_numbers) if top_nums else 0
                    all_hits[select_type].append(hits)
                all_fu_shi_7_pool_hits.append(0)
                all_fu_shi_7_combo_hits_detail.append([])
                continue

            # 后续选3/5/7都从同一份top20截取
            for select_type in [3, 4, 5, 6, 7]:
                top_nums = top20[:select_type]
                hits = len(set(top_nums) & actual_numbers)
                all_hits[select_type].append(hits)

            # 复式7码: 从top20截取前7
            top7 = top20[:7]
            pool_hits = len(set(top7) & actual_numbers)
            all_fu_shi_7_pool_hits.append(pool_hits)

            # v6: 每期所有21组合命中详情（用于ROI精确计算）
            combo_hits = []
            for combo in combinations(top7, 5):
                combo_hits.append(len(set(combo) & actual_numbers))
            all_fu_shi_7_combo_hits_detail.append(combo_hits)

        # 统计结果（使用超几何基线）
        summary = {}

        for select_type in [3, 4, 5, 6, 7]:
            hits_list = all_hits[select_type]
            n_tests = len(hits_list)
            if n_tests == 0:
                summary[f'select_{select_type}'] = {'error': '无测试数据'}
                continue

            mean_hits = sum(hits_list) / n_tests
            expected_random = hypergeom_expected(select_type)

            # >=k概率（实际）
            for_k_probs = {}
            for k in range(1, select_type + 1):
                count = sum(1 for h in hits_list if h >= k)
                for_k_probs[f'>={k}'] = count / n_tests

            # >=k概率（超几何理论）
            theoretical_probs = {}
            for k in range(1, select_type + 1):
                theoretical_probs[f'>={k}'] = hypergeom_p_ge(select_type, k)

            # 95%置信区间
            std_dev = math.sqrt(sum((h - mean_hits) ** 2 for h in hits_list) / n_tests) if n_tests > 1 else 0
            ci_lower = mean_hits - 1.96 * std_dev / math.sqrt(n_tests)
            ci_upper = mean_hits + 1.96 * std_dev / math.sqrt(n_tests)

            lift = (mean_hits - expected_random) / expected_random if expected_random > 0 else 0

            # v6: 奖金ROI（统一为return_multiple和profit_roi两个字段）
            prize_table = load_prize_table()
            prize_info = prize_table.get(f'select_{select_type}', {})
            bet = prize_info.get('bet', 2)
            total_bet = n_tests * bet
            total_prize = sum(prize_info.get(str(h), 0) for h in hits_list)

            # v6: 区分回报倍数和净ROI
            return_multiple = total_prize / max(total_bet, 1)  # 回报倍数(含本金)
            profit_roi = (total_prize - total_bet) / max(total_bet, 1)  # 净ROI(不含本金)

            # 理论随机ROI（超几何期望命中数 * 平均奖金 / 总投注 - 1）
            random_expected_prize = sum(
                hypergeom_pmf(select_type, k) * prize_info.get(str(k), 0)
                for k in range(0, select_type + 1)
            )
            random_return_multiple = random_expected_prize / max(bet, 1)
            random_profit_roi = (random_expected_prize - bet) / max(bet, 1)

            summary[f'select_{select_type}'] = {
                'mean_hits': round(mean_hits, 4),
                'expected_random': round(expected_random, 4),
                'lift': round(lift, 4),
                'probabilities': for_k_probs,
                'theoretical_probs': theoretical_probs,
                'ci_95': [round(ci_lower, 4), round(ci_upper, 4)],
                'std_dev': round(std_dev, 4),
                'n_tests': n_tests,
                'is_significant': ci_lower > expected_random,
                'distribution': 'hypergeometric',
                'bet': bet,
                'total_bet': total_bet,
                'total_prize': total_prize,
                'return_multiple': round(return_multiple, 4),  # 回报倍数(含本金)
                'profit_roi': round(profit_roi, 4),              # 净ROI(不含本金)
                'random_return_multiple': round(random_return_multiple, 4),
                'random_profit_roi': round(random_profit_roi, 4),
            }

        # 复式7码ROI（v6: 每期全部21注组合都计算，无论命中率）
        if all_fu_shi_7_pool_hits:
            pool_mean = sum(all_fu_shi_7_pool_hits) / len(all_fu_shi_7_pool_hits)

            prize_table = load_prize_table()
            fu7_prize_info = prize_table.get('fu_shi_7', {})
            bet_per_combo = fu7_prize_info.get('bet_per_combo', 2)

            # v6: 每期投注= 21注 * 单注金额（无论命中率）
            n_fu7_tests = len(all_fu_shi_7_pool_hits)
            fu7_total_bet = n_fu7_tests * math.comb(7, 5) * bet_per_combo

            # v6: 每期奖金= 所有21组组合的奖金之和
            fu7_total_prize = 0
            for combo_hits_list in all_fu_shi_7_combo_hits_detail:
                if combo_hits_list:
                    fu7_total_prize += sum(
                        fu7_prize_info.get(str(h), 0)
                        for h in combo_hits_list
                    )

            fu7_return_multiple = fu7_total_prize / max(fu7_total_bet, 1)
            fu7_profit_roi = (fu7_total_prize - fu7_total_bet) / max(fu7_total_bet, 1)

            # 理论随机ROI（7码随机选5命中）
            random_fu7_expected_prize_per_combo = sum(
                hypergeom_pmf(5, k) * fu7_prize_info.get(str(k), 0)
                for k in range(0, 6)
            )
            random_fu7_return_multiple = random_fu7_expected_prize_per_combo / max(bet_per_combo, 1)
            random_fu7_profit_roi = (random_fu7_expected_prize_per_combo - bet_per_combo) / max(bet_per_combo, 1)

            summary['fu_shi_7'] = {
                'pool_mean_hits': round(pool_mean, 4),
                'pool_expected_random': round(hypergeom_expected(7), 4),
                'n_tests': n_fu7_tests,
                'total_bet': fu7_total_bet,
                'total_prize': fu7_total_prize,
                'return_multiple': round(fu7_return_multiple, 4),
                'profit_roi': round(fu7_profit_roi, 4),
                'random_return_multiple': round(random_fu7_return_multiple, 4),
                'random_profit_roi': round(random_fu7_profit_roi, 4),
            }

        return summary

    # ─── 置换检验（v9: 打乱实际开奖期 + 加一修正）───

    def _permutation_test(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        start_idx: int,
        end_idx: int,
        pick_n: int = 5,
        metric: str = 'mean_hits',
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
        min_train: int = 50,
        window_size: Optional[int] = None,
        repeat_direction: str = 'neutral',
        repeat_avoid_score: float = 0.10,
        repeat_non_avoid_score: float = 0.85,
        repeat_follow_score: float = 0.90,
        repeat_non_follow_score: float = 0.50,
    ) -> Dict:
        """置换检验: 打乱实际开奖期顺序（v9重大改动）

        v9改动:
        - 不再随机抽号码，而是保持预测不变，打乱实际开奖期的顺序
        - 更准确地模拟"模型预测与实际开奖没有时间关系"的零假设
        - p值使用加一修正: p = (n_ge + 1) / (n_perm + 1)
        """
        history = self.analyzer.history_data
        history_asc = sorted(history, key=lambda x: x['issue'])

        # 先跑一遍真实模型得分
        real_result = self._rolling_backtest_parametric(
            feature_weights, model_weights,
            start_idx=start_idx, end_idx=end_idx,
            min_train=min_train,
            window_size=window_size,
            repeat_direction=repeat_direction,
            repeat_avoid_score=repeat_avoid_score,
            repeat_non_avoid_score=repeat_non_avoid_score,
            repeat_follow_score=repeat_follow_score,
            repeat_non_follow_score=repeat_non_follow_score,
        )

        if 'error' in real_result:
            return real_result

        s_key = f'select_{pick_n}'
        real_lift = real_result.get(s_key, {}).get('lift', 0)
        real_mean_hits = real_result.get(s_key, {}).get('mean_hits', 0)

        # v9: 先收集每一期真实预测号码和对应实际开奖
        actual_start = max(start_idx, min_train)
        actual_end = min(end_idx, len(history_asc))

        predictions = []  # 每期预测号码（top pick_n）
        actual_draws = []  # 每期实际开奖号码

        effective_window = window_size or KL8_DEFAULT_HISTORY

        for t in range(actual_start, actual_end):
            train_data = history_asc[max(0, t - effective_window):t]
            if len(train_data) < min_train:
                continue

            # 构造临时分析器生成预测
            temp_analyzer = KL8Analyzer.__new__(KL8Analyzer)
            temp_analyzer.history_data = sorted(train_data, key=lambda x: x['issue'], reverse=True)
            temp_analyzer.using_simulated_data = False
            temp_analyzer.history_file = ''
            temp_analyzer._data_mtime = 0
            temp_analyzer.update_statistics()

            vote = temp_analyzer.multi_model_voting(
                pick_n=20, top_n=20,
                feature_weights=feature_weights,
                model_weights=model_weights,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            if vote.get('status') == 'no_validated_signal':
                predictions.append([])
                actual_draws.append(set(history_asc[t]['numbers']))
                continue

            top20 = [num for num, _ in vote.get('candidates', [])]
            pred_nums = top20[:pick_n] if len(top20) >= pick_n else top20
            predictions.append(set(pred_nums))
            actual_draws.append(set(history_asc[t]['numbers']))

        if len(predictions) < 10:
            return {'error': '置换检验数据不足'}

        # v9: 真实命中数
        real_hits_list = [
            len(pred & actual) if pred and actual else 0
            for pred, actual in zip(predictions, actual_draws)
        ]
        real_mean = sum(real_hits_list) / len(real_hits_list)
        expected_random = hypergeom_expected(pick_n)
        real_lift_val = (real_mean - expected_random) / expected_random if expected_random > 0 else 0

        # v9: 置换 — 打乱实际开奖期顺序，保持预测不变
        import random as rng

        permutation_lifts = []
        n_greater_or_equal = 0

        for perm_i in range(n_permutations):
            # 确定性seed
            seed = int(hashlib.sha256(f'perm_v9_{perm_i}_{pick_n}'.encode()).hexdigest()[:8], 16)
            rng.seed(seed)

            # 打乱实际开奖期顺序
            shuffled_draws = rng.sample(actual_draws, len(actual_draws))

            # 计算打乱后的命中数
            perm_hits = [
                len(pred & shuffled_actual) if pred and shuffled_actual else 0
                for pred, shuffled_actual in zip(predictions, shuffled_draws)
            ]

            perm_mean = sum(perm_hits) / len(perm_hits)
            perm_lift = (perm_mean - expected_random) / expected_random if expected_random > 0 else 0
            permutation_lifts.append(perm_lift)

            if perm_lift >= real_lift_val:
                n_greater_or_equal += 1

        if not permutation_lifts:
            return {'error': '置换检验数据不足'}

        # v9: 加一修正 p-value
        p_value = (n_greater_or_equal + 1) / (n_permutations + 1)

        # 95%分位数
        sorted_lifts = sorted(permutation_lifts)
        percentile_95 = sorted_lifts[int(len(sorted_lifts) * 0.95)] if len(sorted_lifts) > 20 else 0

        return {
            'play_type': s_key,
            'pick_n': pick_n,
            'real_lift': real_lift_val,
            'real_mean_hits': real_mean,
            'p_value': round(p_value, 6),
            'is_significant_p005': p_value < 0.05,
            'is_significant_p001': p_value < 0.01,
            'permutation_count': len(permutation_lifts),
            'percentile_95_lift': round(percentile_95, 6),
            'permutation_mean_lift': round(sum(permutation_lifts) / len(permutation_lifts), 6),
            'permutation_std_lift': round(
                math.sqrt(sum((l - sum(permutation_lifts) / len(permutation_lifts)) ** 2 for l in permutation_lifts) / len(permutation_lifts)),
                6,
            ) if len(permutation_lifts) > 1 else 0,
            'method': 'shuffle_actual_draws',  # v9: 标记方法
            'plus_one_correction': True,  # v9: 加一修正
        }

    # ─── 特征按玩法分开评估（v6: 按玩法置换检验+多重检验校正）───

    def run_feature_ablation_per_play_type(
        self,
        test_periods: int = BACKTEST_MIN_OOS_PERIODS,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """单特征独立回测 — 按玩法分开评估（v9: 使用独立 ABLATION_FEATURES）

        v9改动:
        - 不再依赖 FEATURE_CONFIG 中 weight>0 的特征
        - 使用独立 ABLATION_FEATURES 试验表
        - repeat_avoid/repeat_follow 作为独立特征参与消融
        - 针对每个玩法单独出结果
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < test_periods + 50:
            return {'error': f'历史数据不足(需{test_periods + 50}期，现有{n}期)'}

        # v6: 真正三段式分割（final_test完全冻结）
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        val_range = split['val']
        final_test_range = split['final_test']

        results = {}

        # v9: 使用独立 ABLATION_FEATURES 而非 FEATURE_CONFIG
        for feature_name, feature_weight in ABLATION_FEATURES.items():
            # 构造单特征权重
            # repeat_avoid/repeat_follow 特殊处理：它们需要 repeat_direction 参数
            single_weights = {k: 0.0 for k in FEATURE_CONFIG}
            repeat_direction = 'neutral'

            if feature_name == 'repeat_avoid':
                single_weights['frequency'] = 0.60
                single_weights['repeat'] = feature_weight
                repeat_direction = 'avoid'
            elif feature_name == 'repeat_follow':
                single_weights['frequency'] = 0.60
                single_weights['repeat'] = feature_weight
                repeat_direction = 'follow'
            elif feature_name in FEATURE_CONFIG:
                single_weights[feature_name] = feature_weight
            else:
                # 未知特征
                continue

            model_weights = {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0}

            # val段回测
            val_result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=val_range[0],
                end_idx=val_range[1],
                min_train=50,
                repeat_direction=repeat_direction,
            )

            # final_test段回测（只用于报告，不影响启用判断）
            final_test_result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=final_test_range[0],
                end_idx=final_test_range[1],
                min_train=50,
                repeat_direction=repeat_direction,
            )

            # v6: 每个玩法单独做置换检验
            perm_results = {}
            for select_type in [3, 4, 5, 6, 7]:
                perm_r = self._permutation_test(
                    single_weights, model_weights,
                    start_idx=val_range[0],
                    end_idx=val_range[1],
                    pick_n=select_type,
                    metric='mean_hits',
                    n_permutations=n_permutations,
                    repeat_direction=repeat_direction,
                )
                perm_results[f'select_{select_type}'] = perm_r

            # v6: 多重检验校正 — 每个玩法的5个特征p值做BH FDR
            # 收集当前特征在所有玩法下的原始p值
            all_p_values_for_correction = []
            for select_type in [3, 4, 5, 6, 7]:
                perm_r = perm_results.get(f'select_{select_type}', {})
                p_val = perm_r.get('p_value', 1.0)
                all_p_values_for_correction.append(p_val)

            # BH FDR校正
            adjusted_p_values = benjamini_hochberg_fdr(all_p_values_for_correction)

            # 按玩法判断有效性（v6: is_candidate条件更严格）
            play_type_recommendations = {}
            for i, select_type in enumerate([3, 4, 5, 6, 7]):
                s_key = f'select_{select_type}'

                val_s = val_result.get(s_key, {})
                test_s = final_test_result.get(s_key, {})

                val_lift = val_s.get('lift', 0)
                test_lift = test_s.get('lift', 0)
                val_significant = val_s.get('is_significant', False)

                # v6: 置换检验p值 + BH FDR校正后p值
                perm_r = perm_results.get(s_key, {})
                raw_p = perm_r.get('p_value', 1.0)
                adjusted_p = adjusted_p_values[i] if i < len(adjusted_p_values) else 1.0

                # v6: ROI指标
                val_profit_roi = val_s.get('profit_roi', 0)
                random_profit_roi = val_s.get('random_profit_roi', 0)
                roi_better = val_profit_roi > random_profit_roi

                # v6: is_candidate = val_lift>0 AND p_adjusted<0.05
                # final_test结果只用于报告，不参与is_candidate判断
                is_candidate = (
                    val_lift > 0
                    and adjusted_p < 0.05
                )

                play_type_recommendations[s_key] = {
                    'val_lift': val_lift,
                    'final_test_lift': test_lift,  # 只报告，不参与判断
                    'val_significant': val_significant,
                    'raw_p_value': raw_p,
                    'adjusted_p_value': adjusted_p,
                    'is_candidate': is_candidate,
                    'val_profit_roi': val_profit_roi,
                    'roi_better_than_random': roi_better,
                    'recommendation': 'enable_for_play_type' if is_candidate else 'keep_disabled',
                }

            results[feature_name] = {
                'val_result': val_result,
                'final_test_result': final_test_result,
                'permutation_tests': perm_results,
                'bh_fdr_adjusted_p_values': {
                    f'select_{st}': adjusted_p_values[i]
                    for i, st in enumerate([3, 4, 5, 6, 7])
                    if i < len(adjusted_p_values)
                },
                'play_type_recommendations': play_type_recommendations,
            }

        return results

    # ─── 窗口长度验证 ───

    def run_window_validation(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        window_sizes: List[int] = [50, 100, 250, 500],
    ) -> Dict:
        """测试不同窗口长度

        在训练段选窗口，验证段确认，测试段验证一次
        如果不同窗口表现互相矛盾，说明信号不稳定
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < 300:
            return {'error': f'数据不足(需300期以上，现有{n}期)'}

        split = self._split_three_stage(n)
        val_range = split['val']

        results = {}
        for ws in window_sizes:
            # 在val段用不同窗口回测
            val_result = self._rolling_backtest_parametric(
                feature_weights, model_weights,
                start_idx=val_range[0],
                end_idx=val_range[1],
                min_train=ws,
                window_size=ws,
            )
            results[f'window_{ws}'] = val_result

        # 一致性检查: 不同窗口对选5的Lift是否一致
        s5_lifts = []
        for ws in window_sizes:
            r = results.get(f'window_{ws}', {})
            s5 = r.get('select_5', {})
            if 'lift' in s5:
                s5_lifts.append(s5['lift'])

        consistency = 'consistent' if all(l > 0 for l in s5_lifts) or all(l <= 0 for l in s5_lifts) else 'contradictory'

        results['consistency'] = {
            's5_lifts_by_window': {f'window_{ws}': s5_lifts[i] for i, ws in enumerate(window_sizes) if i < len(s5_lifts)},
            'consistency': consistency,
            'all_positive': all(l > 0 for l in s5_lifts),
            'recommendation': 'signal_stable' if consistency == 'consistent' and all(l > 0 for l in s5_lifts) else 'signal_unstable',
        }

        return results

    # ─── 稳定性门槛 ───

    def check_stability_gate(
        self,
        feature_name: str,
        feature_weight: float,
    ) -> Dict:
        """稳定性门槛检查

        特征上线需同时满足:
        1. 最近4个独立窗口至少3个Lift>0
        2. val段单侧p-value<0.05
        3. test段Lift>0
        4. 关键中奖档不低于随机
        5. 无严重反向窗口(Lift<-0.1)
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < BACKTEST_MIN_OOS_PERIODS + 50:
            return {'error': f'数据不足(需{BACKTEST_MIN_OOS_PERIODS + 50}期)'}

        single_weights = {k: 0.0 for k in FEATURE_CONFIG}
        single_weights[feature_name] = feature_weight
        model_weights = {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0}

        # 分成4个独立窗口
        window_size = n // BACKTEST_STABILITY_WINDOWS
        window_lifts = []

        for i in range(BACKTEST_STABILITY_WINDOWS):
            start = i * window_size + 50  # 留50期作训练
            end = (i + 1) * window_size

            result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=start, end_idx=end,
                min_train=50,
            )

            s5 = result.get('select_5', {})
            lift = s5.get('lift', 0)
            window_lifts.append(lift)

        # 稳定性判断
        n_positive = sum(1 for l in window_lifts if l > 0)
        n_severe_negative = sum(1 for l in window_lifts if l < -0.1)

        # val段p-value
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        perm = self._permutation_test(
            single_weights, model_weights,
            start_idx=split['val'][0],
            end_idx=split['val'][1],
            pick_n=5,  # 默认用选5做稳定性门槛
            n_permutations=BACKTEST_PERMUTATION_COUNT,
        )

        # v6: final_test段Lift（只用于报告，不参与启用判断）
        final_test_result = self._rolling_backtest_parametric(
            single_weights, model_weights,
            start_idx=split['final_test'][0],
            end_idx=split['final_test'][1],
            min_train=50,
        )
        final_test_lift = final_test_result.get('select_5', {}).get('lift', 0)

        # v6: is_candidate只依赖val段结果，final_test只报告
        # 稳定性门槛仍基于val段判断
        gate_1 = n_positive >= BACKTEST_STABILITY_THRESHOLD  # 4窗口至少3个Lift>0
        gate_2 = perm.get('is_significant_p005', False)       # val p<0.05
        gate_3 = True  # v6: 去掉test_lift>0的条件（final_test不应参与判断）
        gate_4 = n_severe_negative == 0                        # 无严重反向窗口

        all_passed = gate_1 and gate_2 and gate_3 and gate_4

        return {
            'feature': feature_name,
            'window_lifts': window_lifts,
            'n_positive_windows': n_positive,
            'n_severe_negative': n_severe_negative,
            'val_p_value': perm.get('p_value', 1.0),
            'final_test_lift_select_5': final_test_lift,  # 只报告
            'gate_1_stability': gate_1,
            'gate_2_significance': gate_2,
            'gate_3_stability_positive': gate_3,
            'gate_4_no_severe_negative': gate_4,
            'all_gates_passed': all_passed,
            'recommendation': 'enable_candidate' if all_passed else 'keep_disabled',
        }

    # ─── 完整回测 ───

    def run_full_backtest(
        self,
        test_periods: int = BACKTEST_MIN_OOS_PERIODS,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """完整回测: 真正三段式 + 置换检验 + 按玩法分开 + 稳定性门槛

        v6: final_test结果只用于报告，不参与启用判断
        """
        history = self.analyzer.history_data
        n = len(history)

        # 三段式分割（使用默认val_periods=300, final_test_periods=200）
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        result = {
            'total_periods': n,
            'split': split,
            'version': KL8_PREDICTOR_VERSION,
            'distribution': 'hypergeometric',
            'note': 'final_test结果仅供报告，不参与任何启用判断',
        }

        # 1. 单特征按玩法消融回测
        ablation = self.run_feature_ablation_per_play_type(
            test_periods=test_periods,
            n_permutations=n_permutations,
        )
        result['feature_ablation'] = ablation

        # 2. 超几何随机基线
        baseline = {}
        for select_type in [3, 4, 5, 6, 7]:
            baseline[f'select_{select_type}'] = self.hypergeom_baseline(select_type)
        result['random_baseline'] = baseline

        # 3. 稳定性门槛检查（每个有权重的特征）
        stability_checks = {}
        for feature_name, feature_cfg in FEATURE_CONFIG.items():
            if feature_cfg['weight'] > 0:
                stability_checks[feature_name] = self.check_stability_gate(
                    feature_name, feature_cfg['weight'],
                )
        result['stability_checks'] = stability_checks

        # 4. 窗口长度验证
        window_validation = self.run_window_validation(
            feature_weights=get_active_feature_weights(),
            model_weights=get_active_model_weights(),
        )
        result['window_validation'] = window_validation

        # 5. 综合推荐
        recommendations = {}
        for feature_name, stability in stability_checks.items():
            if 'error' in stability:
                recommendations[feature_name] = {'recommendation': 'keep_disabled', 'reason': 'stability_check_error'}
                continue

            all_passed = stability.get('all_gates_passed', False)
            if all_passed:
                # 检查哪些玩法通过了
                ablation_data = ablation.get(feature_name, {})
                play_recs = ablation_data.get('play_type_recommendations', {})
                eligible_play_types = [
                    pt for pt, rec in play_recs.items()
                    if rec.get('is_candidate', False)
                ]
                recommendations[feature_name] = {
                    'recommendation': 'enable_candidate',
                    'eligible_play_types': eligible_play_types,
                    'stability_detail': stability,
                }
            else:
                recommendations[feature_name] = {
                    'recommendation': 'keep_disabled',
                    'failed_gates': {
                        'gate_1': not stability.get('gate_1_stability', False),
                        'gate_2': not stability.get('gate_2_significance', False),
                        'gate_3': not stability.get('gate_3_stability_positive', False),
                        'gate_4': not stability.get('gate_4_no_severe_negative', False),
                    },
                }

        result['recommendations'] = recommendations

        return result

    # ─── 候选策略锦标赛（v9新增）───

    def run_candidate_tournament(
        self,
        candidate_strategies: Optional[Dict] = None,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """候选策略锦标赛 — 训练/验证/最终测试 三段式选型

        v9新增:
        流程固定:
        1. 训练段: 筛掉明显无效策略（Lift<0 或 p>0.5）
        2. 验证段: 从训练段筛出的候选中选最优的1个
        3. 最终测试段: 对胜出策略只跑1次确认
        4. 结果锁定，不允许拿 final_test 再调参数

        参数:
            candidate_strategies: 候选策略字典，默认使用 CANDIDATE_STRATEGIES + REFERENCE_STRATEGY
            n_permutations: 置换检验次数

        返回:
            竞赛结果，包含各段淘汰记录、胜出策略、最终测试确认
        """
        if candidate_strategies is None:
            # 合并参考策略和所有候选策略
            candidate_strategies = {
                'reference': REFERENCE_STRATEGY,
                **CANDIDATE_STRATEGIES,
            }

        history = self.analyzer.history_data
        n = len(history)

        if n < BACKTEST_MIN_OOS_PERIODS + BACKTEST_FINAL_TEST_PERIODS + 100:
            return {'error': f'历史数据不足(需{BACKTEST_MIN_OOS_PERIODS + BACKTEST_FINAL_TEST_PERIODS + 100}期，现有{n}期)'}

        split = self._split_three_stage(n)
        train_range = split['train']
        val_range = split['val']
        final_test_range = split['final_test']

        # ── 第一轮：训练段筛掉明显无效策略 ──
        train_results = {}
        train_survivors = {}

        for name, strategy in candidate_strategies.items():
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
            ws = strategy.get('window_size', KL8_DEFAULT_HISTORY)
            repeat_dir = strategy.get('repeat_direction', 'neutral')
            repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
            repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
            repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
            repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)

            train_result = self._rolling_backtest_parametric(
                fw, mw,
                start_idx=train_range[0],
                end_idx=train_range[1],
                min_train=50,
                window_size=ws,
                repeat_direction=repeat_dir,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            if 'error' in train_result:
                train_results[name] = {'error': train_result['error'], 'survived': False}
                continue

            # 检查所有玩法的 Lift
            all_lifts_positive = True
            best_lift = -999
            for select_type in [3, 4, 5, 6, 7]:
                s_key = f'select_{select_type}'
                lift = train_result.get(s_key, {}).get('lift', 0)
                if lift <= 0:
                    all_lifts_positive = False
                best_lift = max(best_lift, lift)

            # 训练段淘汰条件: Lift 全为负或最大Lift<0
            survived = best_lift > 0

            train_results[name] = {
                'best_lift': round(best_lift, 4),
                'all_lifts_positive': all_lifts_positive,
                'survived': survived,
                'strategy_id': strategy.get('strategy_id', name),
            }

            if survived:
                train_survivors[name] = strategy

        # ── 第二轮：验证段选最优策略 ──
        val_results = {}
        val_best_name = None
        val_best_lift = -999

        for name, strategy in train_survivors.items():
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
            ws = strategy.get('window_size', KL8_DEFAULT_HISTORY)
            repeat_dir = strategy.get('repeat_direction', 'neutral')
            repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
            repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
            repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
            repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)

            val_result = self._rolling_backtest_parametric(
                fw, mw,
                start_idx=val_range[0],
                end_idx=val_range[1],
                min_train=50,
                window_size=ws,
                repeat_direction=repeat_dir,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            if 'error' in val_result:
                val_results[name] = {'error': val_result['error']}
                continue

            # 选5 Lift 作为主指标
            s5_lift = val_result.get('select_5', {}).get('lift', 0)

            # 置换检验
            perm_result = self._permutation_test(
                fw, mw,
                start_idx=val_range[0],
                end_idx=val_range[1],
                pick_n=5,
                n_permutations=n_permutations,
                window_size=ws,
                repeat_direction=repeat_dir,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            raw_p = perm_result.get('p_value', 1.0) if 'error' not in perm_result else 1.0

            # 稳定性检查（4子窗口）
            val_len = val_range[1] - val_range[0]
            sub_window_size = val_len // BACKTEST_STABILITY_WINDOWS
            sub_lifts = []
            for i in range(BACKTEST_STABILITY_WINDOWS):
                sub_start = val_range[0] + i * sub_window_size
                sub_end = val_range[0] + (i + 1) * sub_window_size
                if i == BACKTEST_STABILITY_WINDOWS - 1:
                    sub_end = val_range[1]

                sub_result = self._rolling_backtest_parametric(
                    fw, mw,
                    start_idx=sub_start, end_idx=sub_end,
                    min_train=50, window_size=ws,
                    repeat_direction=repeat_dir,
                    repeat_avoid_score=repeat_avoid_score,
                    repeat_non_avoid_score=repeat_non_avoid_score,
                    repeat_follow_score=repeat_follow_score,
                    repeat_non_follow_score=repeat_non_follow_score,
                )
                sub_lift = sub_result.get('select_5', {}).get('lift', 0) if 'error' not in sub_result else 0
                sub_lifts.append(sub_lift)

            n_positive = sum(1 for l in sub_lifts if l > 0)

            # 记录试验结果（供全量FDR校正）
            trial_record = {
                'strategy_id': strategy.get('strategy_id', name),
                'play_type': 'select_5',
                'feature_weights': fw,
                'model_weights': mw,
                'window_size': ws,
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'raw_p_value': raw_p,
                'validation_lift': round(s5_lift, 4),
                'n_permutations': n_permutations,
                'tested_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'tournament_round': 'validation',
            }
            STRATEGY_TRIAL_RESULTS.append(trial_record)
            _persist_trial_results()

            val_results[name] = {
                's5_lift': round(s5_lift, 4),
                'raw_p_value': raw_p,
                'n_positive_sub_windows': n_positive,
                'sub_window_lifts': [round(l, 4) for l in sub_lifts],
                'strategy_id': strategy.get('strategy_id', name),
            }

            # 选最优: Lift最高 且 p<0.05 且 稳定性>=3/4
            if s5_lift > val_best_lift and raw_p < 0.05 and n_positive >= BACKTEST_STABILITY_THRESHOLD:
                val_best_lift = s5_lift
                val_best_name = name

        # ── 第三轮：最终测试段只跑1次确认 ──
        final_test_result = None
        final_test_report = None

        if val_best_name and val_best_name in train_survivors:
            strategy = train_survivors[val_best_name]
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
            ws = strategy.get('window_size', KL8_DEFAULT_HISTORY)
            repeat_dir = strategy.get('repeat_direction', 'neutral')
            repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
            repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
            repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
            repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)

            final_test_result = self._rolling_backtest_parametric(
                fw, mw,
                start_idx=final_test_range[0],
                end_idx=final_test_range[1],
                min_train=50,
                window_size=ws,
                repeat_direction=repeat_dir,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
            )

            if 'error' not in final_test_result:
                final_test_report = {
                    'strategy_name': val_best_name,
                    's5_lift': round(final_test_result.get('select_5', {}).get('lift', 0), 4),
                    's5_mean_hits': round(final_test_result.get('select_5', {}).get('mean_hits', 0), 4),
                    'note': '最终测试只做确认，结果锁定后不允许再调参数',
                }

        # ── 全量 BH-FDR 校正 ──
        same_play_trials = [t for t in STRATEGY_TRIAL_RESULTS if t['play_type'] == 'select_5']
        same_play_p_values = [t['raw_p_value'] for t in same_play_trials]
        if len(same_play_p_values) > 1:
            fdr_adjusted = benjamini_hochberg_fdr(same_play_p_values)
            for i, trial in enumerate(same_play_trials):
                trial['fdr_adjusted_p'] = round(fdr_adjusted[i], 6)
            _persist_trial_results()

        return {
            'total_candidates': len(candidate_strategies),
            'train_results': train_results,
            'train_survivors': list(train_survivors.keys()),
            'val_results': val_results,
            'val_winner': val_best_name,
            'val_winner_lift': round(val_best_lift, 4) if val_best_lift > -999 else None,
            'final_test_result': final_test_report,
            'tournament_locked': True,
            'note': '锦标赛结果锁定，不允许用final_test再调参数',
            'version': KL8_PREDICTOR_VERSION,
        }


# ─── v9预留: 概率校准框架（暂不启用）───
# 未来目标: 把号码评分升级为可校准概率
# 要求: 80个号码概率和 ≈ 20（KL8开出20个）
# 校准验证: 预测概率0.25的号码，长期是否约25%真正开出

def _brier_score(predicted_probs: List[float], actual_binary: List[int]) -> float:
    """Brier Score — 概率预测准确度度量（预留，暂不启用）

    BS = (1/N) * Σ(forecast - outcome)^2
    完美预测: BS=0; 随机预测: BS≈0.25
    """
    if not predicted_probs or len(predicted_probs) != len(actual_binary):
        return float('inf')
    n = len(predicted_probs)
    return sum((p - o) ** 2 for p, o in zip(predicted_probs, actual_binary)) / n


def _log_loss(predicted_probs: List[float], actual_binary: List[int]) -> float:
    """LogLoss — 概率预测的交叉熵损失（预留，暂不启用）

    LL = -(1/N) * Σ(outcome*log(forecast) + (1-outcome)*log(1-forecast))
    """
    if not predicted_probs or len(predicted_probs) != len(actual_binary):
        return float('inf')
    n = len(predicted_probs)
    total = 0.0
    for p, o in zip(predicted_probs, actual_binary):
        p_clipped = max(1e-10, min(1 - 1e-10, p))
        if o == 1:
            total += -math.log(p_clipped)
        else:
            total += -math.log(1 - p_clipped)
    return total / n


def _calibration_curve_data(
    predicted_probs: List[float],
    actual_binary: List[int],
    n_bins: int = 10,
) -> Dict:
    """校准曲线数据 — 预测概率vs实际频率（预留，暂不启用）

    将预测概率分成n_bins个桶，计算每个桶的平均预测概率和实际频率
    完美校准: 预测0.25的号码，实际25%被开出
    """
    if not predicted_probs:
        return {'error': '无预测概率'}

    # 按预测概率分桶
    bins = [[] for _ in range(n_bins)]
    for p, o in zip(predicted_probs, actual_binary):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, o))

    curve = []
    for i, bin_data in enumerate(bins):
        if not bin_data:
            continue
        mean_pred = sum(p for p, o in bin_data) / len(bin_data)
        mean_actual = sum(o for p, o in bin_data) / len(bin_data)
        curve.append({
            'bin': i,
            'bin_range': f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}',
            'mean_predicted_prob': round(mean_pred, 4),
            'mean_actual_frequency': round(mean_actual, 4),
            'count': len(bin_data),
        })

    return {
        'curve': curve,
        'note': '预留框架，暂不启用。需要模型输出80个号码的入选概率后才能使用。',
    }
