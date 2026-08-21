# -*- coding: utf-8 -*-
"""快乐8滚动回测 KL8RollingBacktest 与校准指标"""

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
    ABLATION_FEATURES, BACKTEST_FINAL_TEST_PERIODS, BACKTEST_MIN_OOS_PERIODS, BACKTEST_PERMUTATION_COUNT, BACKTEST_STABILITY_THRESHOLD, BACKTEST_STABILITY_WINDOWS, BACKTEST_TOTAL_REQUIRED_PERIODS, BACKTEST_TRAIN_PERIODS, CANDIDATE_STRATEGIES, FEATURE_CONFIG, FUSHI_CONFIG, FUSHI_PLAY_KEYS, KL8_DEFAULT_HISTORY, KL8_PREDICTOR_VERSION, REFERENCE_STRATEGY, SELECT_PLAY_KEYS, SELECT_TYPES, VALIDATION_CANDIDATES,
)
from .strategies import (
    get_active_feature_weights, get_active_model_weights,
)
from .stats import (
    _parse_play_pick_n, _play_lift, _practical_validation_score, _prize_tier_thresholds, benjamini_hochberg_fdr, hypergeom_expected, hypergeom_p_ge, hypergeom_pmf,
)
from .candidates import (
    _adaptive_repeat_cap, _select_final_candidate_pool,
)
from .records import (
    load_prize_table,
)
from .analyzer import (
    KL8Analyzer, get_kl8_analyzer,
)


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
        pool_diversify: bool = True,
        pool_max_last_numbers: Optional[int] = None,
        frequency_mode: str = 'mean_reversion',
        final_selection_mode: str = 'balanced',
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
        all_fushi_pool_hits = defaultdict(list)
        all_fushi_combo_hits_detail = defaultdict(list)  # 每期所有组合命中详情（用于ROI）

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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            # 无信号时，该期命中数记录为0
            if vote.get('status') == 'no_validated_signal':
                for select_type in SELECT_TYPES:
                    all_hits[select_type].append(0)
                for fushi_key in FUSHI_PLAY_KEYS:
                    all_fushi_pool_hits[fushi_key].append(0)
                    all_fushi_combo_hits_detail[fushi_key].append([])
                continue

            # v6: 从投票结果截取top20（与线上逻辑一致）
            candidate_items = vote.get('candidates', [])
            top20 = [num for num, _ in candidate_items]

            # 后续各选型都从同一份候选池取号，但按各自选型控制上期重号比例
            for select_type in SELECT_TYPES:
                adaptive_cap = _adaptive_repeat_cap(temp_analyzer.history_data, select_type)
                final_repeat_cap = min(
                    pool_max_last_numbers if pool_max_last_numbers is not None else adaptive_cap,
                    adaptive_cap,
                )
                top_nums = [
                    num for num, _ in _select_final_candidate_pool(
                        candidate_items,
                        select_type,
                        temp_analyzer.statistics.get('last_numbers', set()),
                        max_last_numbers=final_repeat_cap,
                        selection_mode=final_selection_mode,
                    )[0]
                ]
                hits = len(set(top_nums) & actual_numbers)
                all_hits[select_type].append(hits)

            # 复式玩法: 从同一候选池取核心码并控制上期重号比例
            for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
                pool_size = fushi_cfg['pool_size']
                base_pick = fushi_cfg['base_pick']
                adaptive_cap = _adaptive_repeat_cap(temp_analyzer.history_data, pool_size)
                repeat_cap = min(
                    pool_max_last_numbers if pool_max_last_numbers is not None else adaptive_cap,
                    adaptive_cap,
                )
                core_numbers = [
                    num for num, _ in _select_final_candidate_pool(
                        candidate_items,
                        pool_size,
                        temp_analyzer.statistics.get('last_numbers', set()),
                        max_last_numbers=repeat_cap,
                        selection_mode=final_selection_mode,
                    )[0]
                ]
                pool_hits = len(set(core_numbers) & actual_numbers)
                all_fushi_pool_hits[fushi_key].append(pool_hits)

                combo_hits = []
                for combo in combinations(core_numbers, base_pick):
                    combo_hits.append(len(set(combo) & actual_numbers))
                all_fushi_combo_hits_detail[fushi_key].append(combo_hits)

        # 统计结果（使用超几何基线）
        summary = {}

        for select_type in SELECT_TYPES:
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

        # 复式玩法ROI（每期全部组合都计算，无论命中率）
        prize_table = load_prize_table()
        for fushi_key, fushi_cfg in FUSHI_CONFIG.items():
            pool_hits_list = all_fushi_pool_hits[fushi_key]
            if not pool_hits_list:
                continue

            pool_size = fushi_cfg['pool_size']
            base_pick = fushi_cfg['base_pick']
            pool_mean = sum(pool_hits_list) / len(pool_hits_list)
            probabilities = {}
            theoretical_probs = {}
            for k in range(1, pool_size + 1):
                probabilities[f'>={k}'] = sum(1 for h in pool_hits_list if h >= k) / len(pool_hits_list)
                theoretical_probs[f'>={k}'] = hypergeom_p_ge(pool_size, k)

            fushi_prize_info = prize_table.get(fushi_key, {})
            prize_key = fushi_prize_info.get('prize_key', fushi_cfg.get('prize_key', fushi_key))
            combo_prize_info = prize_table.get(prize_key, fushi_prize_info)
            bet_per_combo = fushi_prize_info.get('bet_per_combo', combo_prize_info.get('bet', 2))

            n_tests = len(pool_hits_list)
            total_bet = n_tests * math.comb(pool_size, base_pick) * bet_per_combo

            total_prize = 0
            for combo_hits_list in all_fushi_combo_hits_detail[fushi_key]:
                if combo_hits_list:
                    total_prize += sum(
                        combo_prize_info.get(str(h), 0)
                        for h in combo_hits_list
                    )

            return_multiple = total_prize / max(total_bet, 1)
            profit_roi = (total_prize - total_bet) / max(total_bet, 1)

            random_expected_prize_per_combo = sum(
                hypergeom_pmf(base_pick, k) * combo_prize_info.get(str(k), 0)
                for k in range(0, base_pick + 1)
            )
            random_return_multiple = random_expected_prize_per_combo / max(bet_per_combo, 1)
            random_profit_roi = (random_expected_prize_per_combo - bet_per_combo) / max(bet_per_combo, 1)

            summary[fushi_key] = {
                'pool_mean_hits': round(pool_mean, 4),
                'pool_expected_random': round(hypergeom_expected(pool_size), 4),
                'probabilities': probabilities,
                'theoretical_probs': theoretical_probs,
                'combo_pick': base_pick,
                'pool_size': pool_size,
                'n_tests': n_tests,
                'total_bet': total_bet,
                'total_prize': total_prize,
                'return_multiple': round(return_multiple, 4),
                'profit_roi': round(profit_roi, 4),
                'random_return_multiple': round(random_return_multiple, 4),
                'random_profit_roi': round(random_profit_roi, 4),
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
        pool_diversify: bool = True,
        pool_max_last_numbers: Optional[int] = None,
        frequency_mode: str = 'mean_reversion',
        final_selection_mode: str = 'balanced',
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
            pool_diversify=pool_diversify,
            pool_max_last_numbers=pool_max_last_numbers,
            frequency_mode=frequency_mode,
            final_selection_mode=final_selection_mode,
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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            if vote.get('status') == 'no_validated_signal':
                predictions.append([])
                actual_draws.append(set(history_asc[t]['numbers']))
                continue

            adaptive_cap = _adaptive_repeat_cap(temp_analyzer.history_data, pick_n)
            final_repeat_cap = min(
                pool_max_last_numbers if pool_max_last_numbers is not None else adaptive_cap,
                adaptive_cap,
            )
            pred_nums = [
                num for num, _ in _select_final_candidate_pool(
                    vote.get('candidates', []),
                    pick_n,
                    temp_analyzer.statistics.get('last_numbers', set()),
                    max_last_numbers=final_repeat_cap,
                    selection_mode=final_selection_mode,
                )[0]
            ]
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

        # Only positive lift can support activation; non-positive lift has no
        # evidence of improvement even if it is statistically different.
        if real_lift_val <= 0:
            return {
                'play_type': s_key,
                'pick_n': pick_n,
                'real_lift': real_lift_val,
                'real_mean_hits': real_mean,
                'p_value': 1.0,
                'is_significant_p005': False,
                'is_significant_p001': False,
                'permutation_count': 0,
                'percentile_95_lift': 0,
                'permutation_mean_lift': 0,
                'permutation_std_lift': 0,
                'method': 'shuffle_actual_draws_positive_lift_only',
                'plus_one_correction': True,
                'direction': 'not_better_than_random',
            }

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
            'direction': 'better_than_random',
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
            for select_type in SELECT_TYPES:
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
            for select_type in SELECT_TYPES:
                perm_r = perm_results.get(f'select_{select_type}', {})
                p_val = perm_r.get('p_value', 1.0)
                all_p_values_for_correction.append(p_val)

            # BH FDR校正
            adjusted_p_values = benjamini_hochberg_fdr(all_p_values_for_correction)

            # 按玩法判断有效性（v6: is_candidate条件更严格）
            play_type_recommendations = {}
            for i, select_type in enumerate(SELECT_TYPES):
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
                    for i, st in enumerate(SELECT_TYPES)
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

    def _build_parameter_search_candidates(
        self,
        window_sizes: Optional[List[int]] = None,
        repeat_directions: Optional[List[str]] = None,
        repeat_caps: Optional[List[Optional[int]]] = None,
        pool_diversify_options: Optional[List[bool]] = None,
        final_selection_modes: Optional[List[str]] = None,
        frequency_modes: Optional[List[str]] = None,
        max_candidates: int = 24,
    ) -> Dict[str, Dict]:
        window_sizes = window_sizes or [75, 100, 150]
        repeat_directions = repeat_directions or ['neutral', 'follow']
        repeat_caps = repeat_caps or [None, 2, 3, 4]
        pool_diversify_options = pool_diversify_options or [True]
        final_selection_modes = final_selection_modes or ['prize_floor', 'shape_balanced', 'balanced', 'best_variant', 'low_repeat', 'zone_spread']
        frequency_modes = frequency_modes or ['mean_reversion', 'hot']

        profiles = {
            'random': {'seeded_random': 1.0},
            'random_shape': {'seeded_random': 0.85, 'odd_even': 0.08, 'big_small': 0.07},
            'freq': {'frequency': 1.0, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'freq_gap': {'frequency': 0.55, 'gap': 0.45, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'freq_position': {'frequency': 0.70, 'gap': 0.0, 'position_residual': 0.30, 'road_residual': 0.0, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'freq_road': {'frequency': 0.75, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.25, 'repeat': 0.0, 'odd_even': 0.0, 'big_small': 0.0},
            'freq_repeat': {'frequency': 0.60, 'gap': 0.0, 'position_residual': 0.0, 'road_residual': 0.0, 'repeat': 0.15, 'odd_even': 0.0, 'big_small': 0.0},
            'hot_prize': {'frequency': 0.30, 'adjacent': 0.20, 'trend': 0.15, 'position_residual': 0.15, 'road_residual': 0.10, 'pair_cooccurrence': 0.10},
            'gap_prize': {'frequency': 0.25, 'gap': 0.35, 'trend': 0.15, 'pair_cooccurrence': 0.15, 'position_residual': 0.10},
        }

        candidates = {}
        for window_size in window_sizes:
            for repeat_direction in repeat_directions:
                for repeat_cap in repeat_caps:
                    for pool_diversify in pool_diversify_options:
                        for final_selection_mode in final_selection_modes:
                            for frequency_mode in frequency_modes:
                                for profile_name, feature_weights in profiles.items():
                                    if len(candidates) >= max_candidates:
                                        return candidates
                                    suffix = 'none' if repeat_cap is None else str(repeat_cap)
                                    mode_suffix = final_selection_mode.replace('_', '')
                                    freq_suffix = 'hot' if frequency_mode == 'hot' else 'mr'
                                    name = f'{profile_name}_w{window_size}_{repeat_direction}_cap{suffix}_div{int(pool_diversify)}_{mode_suffix}_{freq_suffix}'
                                    candidates[name] = {
                                        'strategy_id': f'grid_{name}',
                                        'feature_weights': dict(feature_weights),
                                        'model_weights': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
                                        'window_size': window_size,
                                        'repeat_direction': repeat_direction,
                                        'repeat_avoid_score': 0.10,
                                        'repeat_non_avoid_score': 0.85,
                                        'repeat_follow_score': 0.90,
                                        'repeat_non_follow_score': 0.50,
                                        'pool_diversify': pool_diversify,
                                        'pool_max_last_numbers': repeat_cap,
                                        'frequency_mode': frequency_mode,
                                        'final_selection_mode': final_selection_mode,
                                    }
        return candidates

    def run_parameter_search(
        self,
        play_types: Optional[List[str]] = None,
        max_candidates: int = 24,
        top_n: int = 5,
    ) -> Dict:
        history = self.analyzer.history_data
        n = len(history)
        if n < BACKTEST_TOTAL_REQUIRED_PERIODS:
            return {
                'error': f'not enough history, need {BACKTEST_TOTAL_REQUIRED_PERIODS}, got {n}',
                'total_periods': n,
            }

        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e), 'total_periods': n}

        requested = play_types or list(SELECT_PLAY_KEYS) + list(FUSHI_PLAY_KEYS)
        valid = set(SELECT_PLAY_KEYS) | set(FUSHI_PLAY_KEYS)
        target_play_types = [pt for pt in requested if pt in valid]
        if not target_play_types:
            return {'error': 'no valid play_types', 'valid_play_types': sorted(valid)}

        candidates = self._build_parameter_search_candidates(max_candidates=max_candidates)
        rankings = {pt: [] for pt in target_play_types}
        val_range = split['val']
        final_range = split['final_test']
        candidate_runtime = {}

        for name, strategy in candidates.items():
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
            kwargs = {
                'min_train': 50,
                'window_size': strategy.get('window_size', KL8_DEFAULT_HISTORY),
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'repeat_avoid_score': strategy.get('repeat_avoid_score', 0.10),
                'repeat_non_avoid_score': strategy.get('repeat_non_avoid_score', 0.85),
                'repeat_follow_score': strategy.get('repeat_follow_score', 0.90),
                'repeat_non_follow_score': strategy.get('repeat_non_follow_score', 0.50),
                'pool_diversify': strategy.get('pool_diversify', True),
                'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
                'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
                'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
            }
            candidate_runtime[name] = (fw, mw, kwargs)
            val_result = self._rolling_backtest_parametric(
                fw,
                mw,
                start_idx=val_range[0],
                end_idx=val_range[1],
                **kwargs,
            )
            if 'error' in val_result:
                continue

            for play_type in target_play_types:
                val_metrics = val_result.get(play_type, {})
                val_lift = _play_lift(val_result, play_type)
                val_roi = val_metrics.get('profit_roi', 0)
                random_roi = val_metrics.get('random_profit_roi', 0)
                score, practical_detail = _practical_validation_score(val_metrics, play_type, val_lift)
                hit_rate_score = practical_detail['hit_rate_score']
                hit_rate_detail = practical_detail['hit_rate_lifts']

                rankings[play_type].append({
                    'candidate': name,
                    'strategy': dict(strategy),
                    'score': round(score, 6),
                    'validation_lift': round(val_lift, 6),
                    'validation_hit_rate_score': round(hit_rate_score, 6),
                    'validation_hit_rate_lifts': hit_rate_detail,
                    'validation_practical_score_detail': practical_detail,
                    'validation_is_significant': bool(val_metrics.get('is_significant', False)),
                    'final_test_lift': None,
                    'final_test_hit_rate_score': None,
                    'final_test_hit_rate_lifts': None,
                    'validation_profit_roi': val_roi,
                    'random_profit_roi': random_roi,
                    'validation_mean_hits': val_metrics.get('mean_hits', val_metrics.get('pool_mean_hits')),
                    'expected_random': val_metrics.get('expected_random', val_metrics.get('pool_expected_random')),
                    'validation_return_multiple': val_metrics.get('return_multiple'),
                    'final_test_mean_hits': None,
                    'n_tests': val_metrics.get('n_tests'),
                })

        final_candidate_names = set()
        for items in rankings.values():
            items.sort(
                key=lambda item: (
                    item.get('score', 0),
                    item.get('validation_hit_rate_score', 0),
                    item.get('validation_lift', 0),
                ),
                reverse=True,
            )
            final_candidate_names.update(item['candidate'] for item in items[:min(top_n, 3)])

        final_results = {}
        for name in final_candidate_names:
            runtime = candidate_runtime.get(name)
            if not runtime:
                continue
            fw, mw, kwargs = runtime
            final_result = self._rolling_backtest_parametric(
                fw,
                mw,
                start_idx=final_range[0],
                end_idx=final_range[1],
                **kwargs,
            )
            if 'error' not in final_result:
                final_results[name] = final_result

        for play_type, items in rankings.items():
            for item in items:
                final_result = final_results.get(item['candidate'])
                if not final_result:
                    continue
                final_metrics = final_result.get(play_type, {})
                final_lift = _play_lift(final_result, play_type)
                final_score, final_practical_detail = _practical_validation_score(final_metrics, play_type, final_lift)
                final_hit_rate_score = final_practical_detail['hit_rate_score']
                final_hit_rate_detail = final_practical_detail['hit_rate_lifts']
                item['final_test_lift'] = round(final_lift, 6)
                item['final_test_hit_rate_score'] = round(final_hit_rate_score, 6)
                item['final_test_hit_rate_lifts'] = final_hit_rate_detail
                item['final_test_practical_score_detail'] = final_practical_detail
                item['final_test_mean_hits'] = final_metrics.get('mean_hits', final_metrics.get('pool_mean_hits'))
                item['score'] = round(
                    item.get('score', 0)
                    + max(final_score, 0) * 0.25,
                    6,
                )

        best_by_play = {}
        any_significant = False
        for play_type, items in rankings.items():
            items.sort(
                key=lambda item: (
                    item.get('score') or 0,
                    item.get('validation_hit_rate_score') or 0,
                    item.get('final_test_hit_rate_score') or 0,
                    item.get('validation_lift') or 0,
                    item.get('final_test_lift') or 0,
                ),
                reverse=True,
            )
            best = items[0] if items else None
            if best is not None:
                # 排名第一只是候选排序，不等于有 edge：未通过显著性即视为噪声。
                best['likely_noise'] = not best.get('validation_is_significant', False)
                any_significant = any_significant or best.get('validation_is_significant', False)
            best_by_play[play_type] = best
            rankings[play_type] = items[:top_n]

        return {
            'total_periods': n,
            'split': split,
            'candidate_count': len(candidates),
            'play_types': target_play_types,
            'best_by_play': best_by_play,
            'rankings': rankings,
            'any_significant': any_significant,
            'note': 'parameter search only reports candidates; it does not activate strategies',
            'honest_note': (
                '快乐8为公平均匀摇奖：任意 pick_n 个号码的命中数服从超几何分布，'
                '期望恒为 pick_n×0.25，与具体选哪些号无关。因此 lift 的真值为 0，'
                '下面的排名只是候选排序，并不构成"有预测优势"的证据——绝大多数 '
                'validation_is_significant 应为 False。只有通过 validate_and_activate_strategy '
                '的 6 道门(含置换检验 + BH-FDR + 四窗稳定性)才能激活；ROI 依赖奖金表'
                '(可能为占位值)且同属噪声，已不参与打分。'
            ),
            'version': KL8_PREDICTOR_VERSION,
        }

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
        for select_type in SELECT_TYPES:
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

    def run_candidate_tournament_per_play_type(
        self,
        play_type: str,
        candidate_strategies: Optional[Dict] = None,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """v9.2新增: 每个玩法独立验证 — 不再按select_5选胜出策略后强制分给所有玩法

        流程:
        1. 训练段: 筛掉明显无效策略（该玩法Lift<0）
        2. 验证段: 对该玩法所有候选比较 Lift，做BH-FDR校正
        3. 最终封存测试段: 对胜出策略只跑1次确认
        4. 6条激活门槛全通过才写入ACTIVE_STRATEGIES
        5. 最终测试失败 → 直接判定该轮无可激活策略，不重试

        激活门槛:
        - 验证集平均命中 Lift > 0
        - BH-FDR 校正后 p < 0.05
        - 验证集 4 子窗口至少 3 个 Lift > 0
        - 关键中奖档概率 ≥ 随机基线
        - 策略理论回报不低于随机基线
        - 最终封存测试 Lift > 0 且关键中奖档不低于随机

        参数:
            play_type: 玩法名称，如 'select_5', 'fu_shi_7'
            candidate_strategies: 候选策略字典，默认使用 VALIDATION_CANDIDATES
            n_permutations: 置换检验次数
        """
        if candidate_strategies is None:
            candidate_strategies = VALIDATION_CANDIDATES

        # 解析 pick_n
        pick_n = _parse_play_pick_n(play_type)
        if pick_n is None:
            return {'error': f'无效玩法: {play_type}', 'all_failed': True}

        history = self.analyzer.history_data
        n = len(history)

        if n < BACKTEST_TOTAL_REQUIRED_PERIODS:
            return {
                'error': f'历史数据不足(需{BACKTEST_TOTAL_REQUIRED_PERIODS}期，现有{n}期)',
                'all_failed': True,
            }

        # v9.2: 固定拆分 300训练 / 300验证 / 200封存测试
        split = {
            'train': (0, BACKTEST_TRAIN_PERIODS),
            'val': (BACKTEST_TRAIN_PERIODS, BACKTEST_TRAIN_PERIODS + BACKTEST_MIN_OOS_PERIODS),
            'final_test': (BACKTEST_TRAIN_PERIODS + BACKTEST_MIN_OOS_PERIODS, n),
        }
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
            pool_diversify = strategy.get('pool_diversify', True)
            pool_max_last_numbers = strategy.get('pool_max_last_numbers')
            frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
            final_selection_mode = strategy.get('final_selection_mode', 'balanced')

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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            if 'error' in train_result:
                train_results[name] = {'error': train_result['error'], 'survived': False}
                continue

            # 该玩法的 Lift
            s_key = play_type
            lift = _play_lift(train_result, play_type)

            survived = lift > 0

            train_results[name] = {
                'lift': round(lift, 4),
                'survived': survived,
                'strategy_id': strategy.get('strategy_id', name),
            }

            if survived:
                train_survivors[name] = strategy

        # ── 第二轮：验证段 — 对该玩法所有候选比较 ──
        val_results = {}
        val_p_values = []  # 收集所有候选的p值，用于BH-FDR校正
        val_candidates = []

        for name, strategy in train_survivors.items():
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
            ws = strategy.get('window_size', KL8_DEFAULT_HISTORY)
            repeat_dir = strategy.get('repeat_direction', 'neutral')
            repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
            repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
            repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
            repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)
            pool_diversify = strategy.get('pool_diversify', True)
            pool_max_last_numbers = strategy.get('pool_max_last_numbers')
            frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
            final_selection_mode = strategy.get('final_selection_mode', 'balanced')

            # 验证段回测
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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            if 'error' in val_result:
                val_results[name] = {'error': val_result['error']}
                continue

            # 该玩法的验证 Lift
            s_key = play_type
            val_lift = _play_lift(val_result, play_type)

            # 置换检验
            perm_result = self._permutation_test(
                fw, mw,
                start_idx=val_range[0],
                end_idx=val_range[1],
                pick_n=pick_n,
                n_permutations=n_permutations,
                window_size=ws,
                repeat_direction=repeat_dir,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
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
                    pool_diversify=pool_diversify,
                    pool_max_last_numbers=pool_max_last_numbers,
                    frequency_mode=frequency_mode,
                    final_selection_mode=final_selection_mode,
                )

                sub_lift = _play_lift(sub_result, play_type) if 'error' not in sub_result else 0
                sub_lifts.append(sub_lift)

            n_positive = sum(1 for l in sub_lifts if l > 0)

            # 关键奖级概率不低于随机
            threshold_tiers = _prize_tier_thresholds(play_type)
            val_prize_probs = val_result.get(s_key, {}).get('probabilities', {})
            theoretical_probs = val_result.get(s_key, {}).get('theoretical_probs', {})

            prize_tier_passed = True
            for tier in threshold_tiers:
                actual_prob = val_prize_probs.get(tier, 0)
                random_prob = theoretical_probs.get(tier, hypergeom_p_ge(pick_n, int(tier.replace('>=', ''))))
                if actual_prob < random_prob:
                    prize_tier_passed = False

            # ROI不低于随机
            val_roi = val_result.get(s_key, {}).get('profit_roi', 0)
            random_roi = val_result.get(s_key, {}).get('random_profit_roi', 0)
            roi_not_worse = val_roi >= random_roi
            practical_score, practical_detail = _practical_validation_score(
                val_result.get(s_key, {}),
                play_type,
                val_lift,
            )

            # 收集
            val_p_values.append(raw_p)
            val_candidates.append({
                'name': name,
                'strategy': strategy,
                'val_lift': val_lift,
                'raw_p': raw_p,
                'n_positive': n_positive,
                'prize_tier_passed': prize_tier_passed,
                'roi_not_worse': roi_not_worse,
                'practical_score': round(practical_score, 6),
                'practical_detail': practical_detail,
                'sub_lifts': sub_lifts,
            })

            val_results[name] = {
                'lift': round(val_lift, 4),
                'raw_p_value': raw_p,
                'n_positive_sub_windows': n_positive,
                'sub_window_lifts': [round(l, 4) for l in sub_lifts],
                'prize_tier_passed': prize_tier_passed,
                'roi_not_worse': roi_not_worse,
                'strategy_id': strategy.get('strategy_id', name),
            }

            # 记录试验结果
            trial_record = {
                'strategy_id': strategy.get('strategy_id', name),
                'play_type': play_type,
                'feature_weights': fw,
                'model_weights': mw,
                'window_size': ws,
                'repeat_direction': strategy.get('repeat_direction', 'neutral'),
                'pool_diversify': strategy.get('pool_diversify', True),
                'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
                'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
                'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
                'raw_p_value': raw_p,
                'validation_lift': round(val_lift, 4),
                'practical_score': round(practical_score, 6),
                'n_permutations': n_permutations,
                'tested_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'tournament_round': 'per_play_validation',
            }
            _cfg.STRATEGY_TRIAL_RESULTS.append(trial_record)
            _records_mod._persist_trial_results()

        # ── BH-FDR 校正（对同玩法的所有候选p值）───
        if len(val_p_values) > 1:
            adjusted_p_values = benjamini_hochberg_fdr(val_p_values)
        else:
            adjusted_p_values = val_p_values

        # 将FDR校正值写入试验记录
        same_play_trials = [t for t in _cfg.STRATEGY_TRIAL_RESULTS if t['play_type'] == play_type and t.get('tournament_round') == 'per_play_validation']
        same_play_p_values_list = [t['raw_p_value'] for t in same_play_trials]
        if len(same_play_p_values_list) > 1:
            fdr_adjusted = benjamini_hochberg_fdr(same_play_p_values_list)
            for i, trial in enumerate(same_play_trials):
                trial['fdr_adjusted_p'] = round(fdr_adjusted[i], 6)
            _records_mod._persist_trial_results()

        # ── 选验证段最优（Lift最高 + FDR校正后p<0.05 + 稳定性>=3/4 + 奖级>=随机 + ROI>=随机）───
        qualified_candidates = []
        for i, cand in enumerate(val_candidates):
            adjusted_p = adjusted_p_values[i] if i < len(adjusted_p_values) else cand['raw_p']

            if (
                cand['val_lift'] > 0
                and adjusted_p < 0.05
                and cand['n_positive'] >= BACKTEST_STABILITY_THRESHOLD
                and cand['prize_tier_passed']
                and cand['roi_not_worse']
            ):
                qualified_candidates.append({
                    **cand,
                    'adjusted_p': adjusted_p,
                })

        # 没有合格候选
        if not qualified_candidates:
            log.info(f'快乐8: {play_type} 所有候选均未通过验证门槛')
            return {
                'play_type': play_type,
                'all_failed': True,
                'summary': '所有候选均未超过随机基线',
                'train_results': train_results,
                'val_results': val_results,
                'qualified_candidates': [],
            }

        # Pick by prize-threshold score first; mean-hit lift is a tie-breaker.
        best_candidate = max(
            qualified_candidates,
            key=lambda c: (c.get('practical_score', 0), c.get('val_lift', 0)),
        )

        # ── 第三轮：最终封存测试段只跑1次确认 ──
        strategy = best_candidate['strategy']
        fw = strategy.get('feature_weights', {})
        mw = strategy.get('model_weights', {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
        ws = strategy.get('window_size', KL8_DEFAULT_HISTORY)
        repeat_dir = strategy.get('repeat_direction', 'neutral')
        repeat_avoid_score = strategy.get('repeat_avoid_score', 0.10)
        repeat_non_avoid_score = strategy.get('repeat_non_avoid_score', 0.85)
        repeat_follow_score = strategy.get('repeat_follow_score', 0.90)
        repeat_non_follow_score = strategy.get('repeat_non_follow_score', 0.50)
        pool_diversify = strategy.get('pool_diversify', True)
        pool_max_last_numbers = strategy.get('pool_max_last_numbers')
        frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
        final_selection_mode = strategy.get('final_selection_mode', 'balanced')

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
            pool_diversify=pool_diversify,
            pool_max_last_numbers=pool_max_last_numbers,
            frequency_mode=frequency_mode,
            final_selection_mode=final_selection_mode,
        )

        if 'error' in final_test_result:
            return {
                'play_type': play_type,
                'all_failed': True,
                'summary': '最终封存测试失败',
                'best_candidate': best_candidate['name'],
                'final_test_error': final_test_result['error'],
            }

        # 该玩法的最终测试Lift
        final_test_lift = _play_lift(final_test_result, play_type)

        # 最终测试关键奖级不低于随机
        ft_prize_probs = final_test_result.get(play_type, {}).get('probabilities', {})
        ft_theoretical_probs = final_test_result.get(play_type, {}).get('theoretical_probs', {})
        ft_prize_tier_passed = True
        for tier in threshold_tiers:
            ft_actual_prob = ft_prize_probs.get(tier, 0)
            ft_random_prob = ft_theoretical_probs.get(tier, hypergeom_p_ge(pick_n, int(tier.replace('>=', ''))))
            if ft_actual_prob < ft_random_prob:
                ft_prize_tier_passed = False

        # v9.2: 最终测试不用于"挑选策略"，但可以作为"是否允许上线"的门槛
        # 最终测试失败 → 不重试，直接判定该轮无可激活策略
        final_test_passed = final_test_lift > 0 and ft_prize_tier_passed

        if not final_test_passed:
            log.info(
                f'快乐8: {play_type} 最终封存测试未通过 '
                f'(Lift={round(final_test_lift, 4)}, 奖级通过={ft_prize_tier_passed})，'
                f'本轮无可激活策略'
            )
            return {
                'play_type': play_type,
                'all_failed': True,
                'summary': f'最终封存测试未通过(Lift={round(final_test_lift, 4)})',
                'best_candidate': best_candidate['name'],
                'val_lift': round(best_candidate['val_lift'], 4),
                'final_test_lift': round(final_test_lift, 4),
                'final_test_passed': False,
                'note': '最终测试失败，不重新调权重再用同一段测试集试一次',
            }

        # ── 6条门槛全部通过 → 激活 ──
        report = {
            'play_type': play_type,
            'best_candidate': best_candidate['name'],
            'val_lift': round(best_candidate['val_lift'], 4),
            'practical_score': round(best_candidate.get('practical_score', 0), 6),
            'practical_detail': best_candidate.get('practical_detail', {}),
            'adjusted_p': round(best_candidate['adjusted_p'], 6),
            'n_positive_sub_windows': best_candidate['n_positive'],
            'prize_tier_passed': best_candidate['prize_tier_passed'],
            'roi_not_worse': best_candidate['roi_not_worse'],
            'final_test_lift': round(final_test_lift, 4),
            'final_test_prize_tier_passed': ft_prize_tier_passed,
            'frequency_mode': frequency_mode,
            'final_selection_mode': final_selection_mode,
            'data_cutoff_issue': self.analyzer.history_data[0]['issue'] if self.analyzer.history_data else '',
            'data_periods': n,
            'version': KL8_PREDICTOR_VERSION,
        }

        _snapshots_mod.activate_verified_strategy(play_type, strategy, report)

        return {
            'play_type': play_type,
            'activated': True,
            'best_candidate': best_candidate['name'],
            'strategy_id': _cfg.ACTIVE_STRATEGIES[play_type]['strategy_id'],
            'val_lift': round(best_candidate['val_lift'], 4),
            'practical_score': round(best_candidate.get('practical_score', 0), 6),
            'final_test_lift': round(final_test_lift, 4),
            'summary': f'验证通过并激活: {best_candidate["name"]}',
            'train_results': train_results,
            'val_results': val_results,
        }

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
            pool_diversify = strategy.get('pool_diversify', True)
            pool_max_last_numbers = strategy.get('pool_max_last_numbers')
            frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
            final_selection_mode = strategy.get('final_selection_mode', 'balanced')

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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            if 'error' in train_result:
                train_results[name] = {'error': train_result['error'], 'survived': False}
                continue

            # 检查所有玩法的 Lift
            all_lifts_positive = True
            best_lift = -999
            for select_type in SELECT_TYPES:
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
            pool_diversify = strategy.get('pool_diversify', True)
            pool_max_last_numbers = strategy.get('pool_max_last_numbers')
            frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
            final_selection_mode = strategy.get('final_selection_mode', 'balanced')

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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
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
                    pool_diversify=pool_diversify,
                    pool_max_last_numbers=pool_max_last_numbers,
                    frequency_mode=frequency_mode,
                    final_selection_mode=final_selection_mode,
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
                'pool_diversify': strategy.get('pool_diversify', True),
                'pool_max_last_numbers': strategy.get('pool_max_last_numbers'),
                'frequency_mode': strategy.get('frequency_mode', 'mean_reversion'),
                'final_selection_mode': strategy.get('final_selection_mode', 'balanced'),
                'raw_p_value': raw_p,
                'validation_lift': round(s5_lift, 4),
                'n_permutations': n_permutations,
                'tested_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'tournament_round': 'validation',
            }
            _cfg.STRATEGY_TRIAL_RESULTS.append(trial_record)
            _records_mod._persist_trial_results()

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
            pool_diversify = strategy.get('pool_diversify', True)
            pool_max_last_numbers = strategy.get('pool_max_last_numbers')
            frequency_mode = strategy.get('frequency_mode', 'mean_reversion')
            final_selection_mode = strategy.get('final_selection_mode', 'balanced')

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
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                frequency_mode=frequency_mode,
                final_selection_mode=final_selection_mode,
            )

            if 'error' not in final_test_result:
                final_test_report = {
                    'strategy_name': val_best_name,
                    's5_lift': round(final_test_result.get('select_5', {}).get('lift', 0), 4),
                    's5_mean_hits': round(final_test_result.get('select_5', {}).get('mean_hits', 0), 4),
                    'note': '最终测试只做确认，结果锁定后不允许再调参数',
                }

        # ── 全量 BH-FDR 校正 ──
        same_play_trials = [t for t in _cfg.STRATEGY_TRIAL_RESULTS if t['play_type'] == 'select_5']
        same_play_p_values = [t['raw_p_value'] for t in same_play_trials]
        if len(same_play_p_values) > 1:
            fdr_adjusted = benjamini_hochberg_fdr(same_play_p_values)
            for i, trial in enumerate(same_play_trials):
                trial['fdr_adjusted_p'] = round(fdr_adjusted[i], 6)
            _records_mod._persist_trial_results()

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


