# -*- coding: utf-8 -*-
"""足球比分候选生成：集成预测/热度/半全场/推荐挑选/多样化"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')
from . import modeling as _modeling_mod

from .config import (
    AVG_LEAGUE_GOAL, COLD_FILTER_BONUS, DYNAMIC_ELO_AVAILABLE, HEAT_FILTER_PENALTY, HEAT_RATIO_COLD, HEAT_RATIO_HOT, LEAGUE_PROFILES, MARKET_CLUSTERING_AVAILABLE, MAX_GOALS, MOMENTUM_SUPREMACY_WEIGHT, SCORE_1X2_MARKET_ANCHOR_STRENGTH, SCORE_BASELINE_FREQ, SUPREMACY_CONFLICT_GAP, VALUE_BETTING_AVAILABLE, calculate_ev, calculate_value, calibrate_predictions, get_elo_difference, get_market_prior,
)
from .parsing import (
    _blend_close_open, get_close_total_line,
)
from .markets import (
    implied_total_goals, remove_vig,
)
from .modeling import (
    _build_residual_features, _estimate_dc_rho, _fit_lambda_grid, _matrix_margins, _ou_total_distribution, _result_label, apply_handicap_change_adjustment, apply_residual_correction, apply_total_line_change_adjustment, asian_implied_supremacy, bayesian_predict_scores, blend_lambdas_with_market, blend_market_supremacy, calibrate_draw_probability, calibrate_to_euro, estimate_lambdas, euro_implied_supremacy, market_implied_lambdas, team_poisson_lambdas,
)
from .calibrating import (
    calibrate_probabilities, get_league_calibration_data,
)
from .upset import (
    _evaluate_upset_risk,
)

from ..domain.sports.football import scoring as _sc

def _pick_recommendations(*args, **kwargs):
    """挑选推荐；聚类先验的取用要读存储，所以由本层注入（判据 16）"""
    from .market_clustering import get_market_prior
    kwargs.setdefault('market_prior_fn', get_market_prior)
    return _sc._pick_recommendations(*args, **kwargs)


# 34 个纯计算转发给领域层
fit_lambdas_from_markets = _sc.fit_lambdas_from_markets
_baseline_freq = _sc._baseline_freq
score_implied_prob_from_euro = _sc.score_implied_prob_from_euro
score_heat_label = _sc.score_heat_label
_heat_filter_weight = _sc._heat_filter_weight
_half_full_probs_to_dict = _sc._half_full_probs_to_dict
_estimate_score_odds = _sc._estimate_score_odds
_score_entry = _sc._score_entry
_alignment_score = _sc._alignment_score
_recommend_reasons = _sc._recommend_reasons
_get_score_cluster = _sc._get_score_cluster
_get_cluster_name = _sc._get_cluster_name
score_pattern = _sc.score_pattern
_score_total_line_factor = _sc._score_total_line_factor
_common_score_overheat_factor = _sc._common_score_overheat_factor
_total_market_tempo_signal = _sc._total_market_tempo_signal
_joint_market_state = _sc._joint_market_state
_apply_joint_market_state = _sc._apply_joint_market_state
_score_total_movement_factor = _sc._score_total_movement_factor
_adjust_score_probs_with_total_movement = _sc._adjust_score_probs_with_total_movement
_anchor_score_candidates_to_1x2 = _sc._anchor_score_candidates_to_1x2
_anchor_score_candidates_to_goal_mean = _sc._anchor_score_candidates_to_goal_mean
_normalize_goal_dist = _sc._normalize_goal_dist
_implied_total_mean = _sc._implied_total_mean
_anchor_goal_dist_to_total_line = _sc._anchor_goal_dist_to_total_line
_goal_over_under_from_line = _sc._goal_over_under_from_line
_adjust_goal_dist_with_total_movement = _sc._adjust_goal_dist_with_total_movement
_score_result_code = _sc._score_result_code
_candidate_result_support = _sc._candidate_result_support
_adjust_half_full_with_score_context = _sc._adjust_half_full_with_score_context
_adjust_half_full_with_market_context = _sc._adjust_half_full_with_market_context
_assess_market_data_quality = _sc._assess_market_data_quality
_diversify_score_recommendations = _sc._diversify_score_recommendations

def perturb_parameters(base_params):
    """
    对参数进行扰动，生成扰动后的参数组合。
    
    参数:
        base_params: 基础参数 {'max_goals': int, 'rho_init': float, 'league_params': dict}
    
    返回:
        扰动后的参数字典
    """
    perturbed = {}
    
    # 扰动 MAX_GOALS（±1）
    base_max_goals = base_params.get('max_goals', MAX_GOALS)
    perturbed['max_goals'] = base_max_goals + random.randint(-1, 1)
    perturbed['max_goals'] = max(5, min(10, perturbed['max_goals']))
    
    # 扰动 rho 初值（±0.1）
    base_rho = base_params.get('rho_init', 0.0)
    perturbed['rho_init'] = base_rho + random.uniform(-0.1, 0.1)
    perturbed['rho_init'] = max(-0.3, min(0.3, perturbed['rho_init']))
    
    # 扰动联赛参数（场均进球 ±5%）
    league_params = base_params.get('league_params', {})
    perturbed['league_params'] = {}
    for key, value in league_params.items():
        if isinstance(value, (int, float)):
            perturbed['league_params'][key] = value * random.uniform(0.95, 1.05)
        else:
            perturbed['league_params'][key] = value
    
    return perturbed


def ensemble_predict_scores(asian, euro, total, team_strength=None, league_profile=None,
                           num_models=5, method='average', current_time_layer=None):
    """
    多模型集成预测。
    
    参数:
        asian, euro, total: 赔率数据
        team_strength: 球队实力数据
        league_profile: 联赛画像
        num_models: 集成模型数量
        method: 融合方法 'average'（平均）或 'weighted'（加权）
    
    返回:
        (candidates, lam_home, lam_away, meta): 集成后的预测结果
    """
    all_matrices = []
    all_lams = []
    model_types = ['poisson', 'negative_binomial']

    # 2744场离线回测中 Poisson/DC 的比分排序更稳。负二项只用于补充
    # 高比分尾部，避免旧实现用 2/5 固定权重过度放大大比分。
    try:
        total_line = float(total.get('close_line') or total.get('line') or 2.5)
    except (TypeError, ValueError):
        total_line = 2.5
    nb_weight = 0.30 if total_line >= 3.0 else (0.22 if total_line >= 2.75 else 0.15)
    requested_weights = [1.0 - nb_weight, nb_weight]

    for model_type in model_types:
        
        try:
            candidates, lam_home, lam_away, meta = predict_scores(
                asian, euro, total, 
                team_strength=team_strength, 
                league_profile=league_profile,
                model_type=model_type,
                current_time_layer=current_time_layer,
            )
            
            # 将 candidates 转换为矩阵格式
            matrix = {(c[0][0], c[0][1]): c[1] for c in candidates}
            all_matrices.append(matrix)
            all_lams.append((lam_home, lam_away))
            
        except Exception as e:
            log.warning(f"集成模型 {model_type} 失败: {e}")
            continue
    
    if not all_matrices:
        # 如果所有模型都失败，返回基础预测
        return predict_scores(asian, euro, total, team_strength, league_profile, current_time_layer=current_time_layer)
    
    # 融合多个矩阵
    if len(all_matrices) == 2:
        weights = requested_weights
    else:
        weights = [1.0]
    
    # 合并所有矩阵的键
    all_keys = set()
    for m in all_matrices:
        all_keys.update(m.keys())
    
    # 加权平均概率
    ensemble_matrix = {}
    for key in all_keys:
        weighted_sum = 0.0
        weight_total = 0.0
        for i, matrix in enumerate(all_matrices):
            if key in matrix:
                weighted_sum += matrix[key] * weights[i]
                weight_total += weights[i]
        
        if weight_total > 0:
            ensemble_matrix[key] = weighted_sum / weight_total
        else:
            ensemble_matrix[key] = 0.0
    
    # 归一化
    total_prob = sum(ensemble_matrix.values())
    if total_prob > 0:
        ensemble_matrix = {k: v / total_prob for k, v in ensemble_matrix.items()}
    
    # 计算平均 lambda
    avg_lam_home = sum(l[0] for l in all_lams) / len(all_lams)
    avg_lam_away = sum(l[1] for l in all_lams) / len(all_lams)
    
    # 准备返回结果
    candidates = sorted(ensemble_matrix.items(), key=lambda kv: -kv[1])
    
    meta = {
        'ensemble_size': len(all_matrices),
        'ensemble_method': 'adaptive_weighted',
        'ensemble_weights': {
            'poisson': round(weights[0], 4),
            'negative_binomial': round(weights[1], 4) if len(weights) > 1 else 0.0,
        },
        'model_type': 'ensemble',
        'supremacy_asian': meta.get('supremacy_asian'),
        'supremacy_euro': meta.get('supremacy_euro'),
        'supremacy_blended': meta.get('supremacy_blended'),
        'target_total': meta.get('target_total'),
        'calibrated': True,
        'calibration_method': 'platt',
        'market_db_used': meta.get('market_db_used', False),
    }
    
    return candidates, avg_lam_home, avg_lam_away, meta












def calculate_half_full_time_probs(candidates, team_strength=None, asian=None, total=None, home_team='', away_team='', league=''):
    """
    计算半全场概率（集成动态ELO和贝叶斯校准）。
    
    半全场结果共9种：
    HH - 半胜全胜, HD - 半胜全平, HA - 半胜全负
    DH - 半平全胜, DD - 半平全平, DA - 半平全负
    AH - 半负全胜, AD - 半负全平, AA - 半负全负
    
    参数:
        candidates: 比分候选列表，格式为 [((h, a), prob), ...]
        team_strength: 球队实力数据（可选）
        asian: 亚盘数据（可选，用于历史库查询）
        total: 大小球数据（可选，用于历史库查询）
        home_team: 主队名称（用于动态ELO查询）
        away_team: 客队名称（用于动态ELO查询）
    
    返回:
        dict: 包含半全场概率和样本信息的字典
    """
    # 初始化样本信息
    sample_count = 0
    distance = float('inf')
    history_weight = 0.0
    sample_warnings = []
    
    # 处理 candidates 格式：支持 ((h, a), prob) 和 (h, a, prob) 两种格式
    formatted_candidates = []
    for item in candidates:
        if len(item) == 2 and isinstance(item[0], tuple):
            # 格式: ((h, a), prob)
            (h, a), prob = item
            formatted_candidates.append((h, a, prob))
        elif len(item) == 3:
            # 格式: (h, a, prob)
            formatted_candidates.append(item)
    
    candidates = formatted_candidates
    
    # 使用动态ELO调整半场进球比例
    elo_factor = 1.0
    if DYNAMIC_ELO_AVAILABLE and home_team and away_team:
        try:
            from .dynamic_elo import get_elo_difference
            elo_diff = get_elo_difference(home_team, away_team)
            # ELO差距会影响半场进球比例
            elo_factor = 1.0 + (elo_diff / 1000) * 0.1
        except Exception as e:
            log.debug(f"动态ELO计算失败: {e}")
    
    # 根据盘口动态调整半场进球比例
    close_line = (
        total.get('close_line')
        or total.get('line')
        or total.get('close', {}).get('line')
        or 2.5
    ) if total else 2.5
    
    handicap = asian.get('handicap', 0) if asian else 0
    
    # 计算全场进球期望（主客分开计算）
    home_goals_exp = sum(h * prob for h, a, prob in candidates)
    away_goals_exp = sum(a * prob for h, a, prob in candidates)
    total_goals_exp = home_goals_exp + away_goals_exp
    
    # 计算半场比例（可根据联赛、盘口等调整）
    # 优先使用该分桶的真实实测半场比例（≥20 样本），否则回退到硬编码基准 0.42。
    half_time_ratio = 0.42
    half_time_ratio_source = 'default'
    try:
        from .half_time_stats import HalfTimeStatsDB
        _ht_stats = HalfTimeStatsDB().get_stats(league, close_line, handicap,
                                                match_type='league', min_samples=20)
        if _ht_stats and _ht_stats.get('half_time_ratio_avg'):
            _real_ratio = float(_ht_stats['half_time_ratio_avg'])
            if 0.30 <= _real_ratio <= 0.55:
                half_time_ratio = _real_ratio
                half_time_ratio_source = 'real_stats'
    except Exception as e:
        log.debug(f"读取真实半场比例失败，回退默认: {e}")

    if half_time_ratio_source == 'default':
        # 低进球盘口：上半场更谨慎
        if close_line <= 2.0:
            half_time_ratio -= 0.03
        elif close_line <= 2.25:
            half_time_ratio -= 0.015
        # 高进球盘口：上半场进球概率提高
        elif close_line >= 3.0:
            half_time_ratio += 0.025
        elif close_line >= 2.75:
            half_time_ratio += 0.015

        # 深盘强弱明显时，强队上半场领先概率提高
        if abs(handicap) >= 1.0:
            half_time_ratio += 0.015

    half_time_ratio *= elo_factor
    half_time_ratio = max(0.36, min(0.49, half_time_ratio))
    
    # ========== 两阶段模型：半场 + 下半场 ==========
    # 上半场进球期望
    half_home_exp = home_goals_exp * half_time_ratio
    half_away_exp = away_goals_exp * half_time_ratio
    
    # 下半场进球期望（全场 - 上半场）
    second_home_exp = home_goals_exp - half_home_exp
    second_away_exp = away_goals_exp - half_away_exp
    
    # 确保下半场期望为非负数
    second_home_exp = max(0.01, second_home_exp)
    second_away_exp = max(0.01, second_away_exp)
    
    # ========== 下半场条件修正 ==========
    # 根据半场状态调整下半场进球期望
    # 先计算半场平局概率，用于判断是否需要修正
    def poisson_prob(lam, k):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    # 计算半场平局概率
    half_draw_prob = 0.0
    for h in range(4):
        for a in range(4):
            if h == a:
                half_draw_prob += poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
    
    # 下半场修正因子
    second_half_multiplier = 1.0
    second_home_multiplier = 1.0
    second_away_multiplier = 1.0
    
    # 规则1：半场平局时，下半场开放度提升
    if half_draw_prob > 0.35:  # 半场平局概率较高
        second_half_multiplier = 1.08
    
    # 规则2：深盘强队状态修正
    if abs(handicap) >= 0.75:
        # 判断谁是强队
        if handicap > 0:
            # 主队让球，主队是强队
            favorite = 'home'
        else:
            # 客队让球，客队是强队
            favorite = 'away'
        
        # 计算强队半场领先概率
        if favorite == 'home':
            favorite_lead_prob = sum(
                poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
                for h in range(4) for a in range(4) if h > a
            )
            favorite_trail_prob = sum(
                poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
                for h in range(4) for a in range(4) if h < a
            )
        else:
            favorite_lead_prob = sum(
                poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
                for h in range(4) for a in range(4) if a > h
            )
            favorite_trail_prob = sum(
                poisson_prob(half_home_exp, h) * poisson_prob(half_away_exp, a)
                for h in range(4) for a in range(4) if a < h
            )
        
        # 强队落后时，下半场进攻提升
        if favorite_trail_prob > 0.25:
            if favorite == 'home':
                second_home_multiplier = 1.15
            else:
                second_away_multiplier = 1.15
        
        # 强队领先时，下半场保守，落后方进攻提升
        elif favorite_lead_prob > 0.40:
            if favorite == 'home':
                second_home_multiplier = 0.90
                second_away_multiplier = 1.10
            else:
                second_away_multiplier = 0.90
                second_home_multiplier = 1.10
    
    # 应用修正因子
    second_home_exp *= second_half_multiplier * second_home_multiplier
    second_away_exp *= second_half_multiplier * second_away_multiplier
    
    # 定义半场结果映射
    def get_half_result(h, a):
        if h > a:
            return 'H'
        elif h < a:
            return 'A'
        else:
            return 'D'
    
    # 定义全场结果映射
    def get_full_result(h, a):
        if h > a:
            return 'H'
        elif h < a:
            return 'A'
        else:
            return 'D'
    
    # ========== 两阶段条件概率模型 ==========
    # P(HT=h1:a1, FT=h2:a2) = P(HT=h1:a1) * P(2H=(h2-h1):(a2-a1))
    # 其中 FT = HT + 2H
    
    htf_probs = {}
    max_goals = 5  # 最大考虑5球
    max_half_goals = 3  # 半场最多考虑3球
    
    # 遍历所有可能的半场比分和全场比分组合
    for half_h in range(max_half_goals + 1):
        for half_a in range(max_half_goals + 1):
            # 计算半场概率
            half_prob = poisson_prob(half_home_exp, half_h) * poisson_prob(half_away_exp, half_a)
            
            if half_prob < 0.0001:
                continue
            
            # 遍历所有可能的全场比分（必须大于等于半场比分）
            for full_h in range(half_h, max_goals + 1):
                for full_a in range(half_a, max_goals + 1):
                    # 计算下半场进球数
                    second_h = full_h - half_h
                    second_a = full_a - half_a
                    
                    # 计算下半场概率
                    second_prob = poisson_prob(second_home_exp, second_h) * poisson_prob(second_away_exp, second_a)
                    
                    if second_prob < 0.0001:
                        continue
                    
                    # 组合概率
                    total_prob = half_prob * second_prob
                    
                    # 获取半全场结果
                    half_res = get_half_result(half_h, half_a)
                    full_res = get_full_result(full_h, full_a)
                    key = f"{half_res}{full_res}"
                    
                    if key not in htf_probs:
                        htf_probs[key] = 0
                    htf_probs[key] += total_prob
    
    # 归一化半全场概率
    htf_total = sum(htf_probs.values())
    if htf_total > 0:
        htf_probs = {k: v / htf_total for k, v in htf_probs.items()}

    # Blend with real half-time samples when available. These are preferred over
    # inferred half-time history because they come from actual HT scores.
    if asian and total and league:
        try:
            from .half_time_stats import HalfTimeStatsDB

            half_db = HalfTimeStatsDB()
            real_stats = half_db.get_stats(
                league=league,
                total_line=close_line,
                handicap=handicap,
                match_type='league',
                min_samples=20,
            )
            if not real_stats:
                real_stats = half_db.get_nearest_stats(
                    league=league,
                    total_line=close_line,
                    handicap=handicap,
                    match_type='league',
                    min_samples=12,
                    max_distance=0.75,
                )
            real_htf = real_stats.get('half_full_distribution', {}) if real_stats else {}

            if real_htf:
                stats_meta = real_stats.get('_meta', {}) if isinstance(real_stats, dict) else {}
                try:
                    stats_distance = float(stats_meta.get('distance', 0.0) or 0.0)
                except (TypeError, ValueError):
                    stats_distance = 0.0
                try:
                    from .prediction_policy import get_prediction_policy
                    htf_policy = get_prediction_policy(
                        league=league,
                        total_line=close_line,
                        handicap=handicap,
                    )
                    real_weight = htf_policy.get('half_full_real_weight', 0.25)
                except Exception:
                    real_weight = 0.25
                if stats_meta.get('source') == 'nearest':
                    real_weight *= max(0.35, 1 - stats_distance / 0.9)
                blended_htf = {}
                all_keys = set(htf_probs.keys()).union(set(real_htf.keys()))

                for key in all_keys:
                    model_prob = htf_probs.get(key, 0.001)
                    real_prob = real_htf.get(key, 0.001)
                    blended_htf[key] = (1 - real_weight) * model_prob + real_weight * real_prob

                total_prob = sum(blended_htf.values())
                if total_prob > 0:
                    htf_probs = {
                        k: v / total_prob
                        for k, v in sorted(blended_htf.items(), key=lambda x: -x[1])
                    }
                    history_weight = max(history_weight, real_weight)
                    sample_count = max(sample_count, int(stats_meta.get('weighted_sample_count') or 20))
                    distance = min(distance, stats_distance)
                    log.debug("半全场概率融合真实统计: 权重=%.3f", real_weight)
        except Exception as e:
            log.debug(f"无法加载真实半场统计数据调整半全场概率: {e}")
            sample_warnings.append('加载真实半场统计数据失败')
    
    # ========== 结合历史盘口数据调整半全场概率 ==========
    # 注意：当前历史半全场数据是通过全场比分倒推的，不是真实半场数据
    # 倒推逻辑会导致半场平局被严重放大，因此权重需要非常低
    if asian and total:
        try:
            from .market_db import MarketScoreDB
            
            handicap = asian.get('handicap', 0)
            close_line = (
                total.get('close_line')
                or total.get('line')
                or total.get('close', {}).get('line')
                or 2.5
            )
            
            db = MarketScoreDB()
            db.load()
            
            market_result = db.get_htf_probs_with_meta(handicap, close_line)
            market_htf = market_result.get('probabilities', {})
            market_sample_count = market_result.get('sample_count', 0)
            market_distance = market_result.get('distance', float('inf'))
            
            # 当前历史数据是倒推的，权重需大幅降低
            # 在没有真实半场数据前，权重限制在10%以内
            if market_htf and market_sample_count >= 50 and market_distance <= 0.5:
                # 倒推数据权重很低，最多10%
                try:
                    from .prediction_policy import get_prediction_policy
                    htf_policy = get_prediction_policy(
                        league=league,
                        total_line=close_line,
                        handicap=handicap,
                    )
                    market_cap = htf_policy.get('half_full_market_cap', 0.10)
                except Exception:
                    market_cap = 0.10
                market_history_weight = min(market_cap, market_sample_count / 500 * market_cap)
                history_weight = max(history_weight, market_history_weight)
                sample_count = max(sample_count, market_sample_count)
                distance = min(distance, market_distance)
                
                if market_history_weight > 0:
                    blended_htf = {}
                    all_keys = set(htf_probs.keys()).union(set(market_htf.keys()))
                    
                    for key in all_keys:
                        model_prob = htf_probs.get(key, 0.001)
                        market_prob = market_htf.get(key, 0.001)
                        blended_htf[key] = (1 - market_history_weight) * model_prob + market_history_weight * market_prob
                    
                    total_prob = sum(blended_htf.values())
                    if total_prob > 0:
                        htf_probs = {
                            k: v / total_prob
                            for k, v in sorted(blended_htf.items(), key=lambda x: -x[1])
                        }
                
                log.debug("半全场概率融合历史盘口: 权重=%.3f", history_weight)
            else:
                if market_sample_count < 50:
                    sample_warnings.append('历史半场样本不足，不启用历史融合')
                elif market_distance > 0.5:
                    sample_warnings.append('盘口距离过远，不启用历史融合')
        except Exception as e:
            log.debug(f"无法加载历史盘口数据调整半全场概率: {e}")
            sample_warnings.append('加载历史半场数据失败')
    
    # 添加友好名称
    htf_names = {
        'HH': '半胜全胜',
        'HD': '半胜全平',
        'HA': '半胜全负',
        'DH': '半平全胜',
        'DD': '半平全平',
        'DA': '半平全负',
        'AH': '半负全胜',
        'AD': '半负全平',
        'AA': '半负全负',
    }
    
    result = []
    for key in ['HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA']:
        prob = htf_probs.get(key, 0)
        result.append({
            'code': key,
            'name': htf_names[key],
            'probability': round(prob * 100, 1),
            'raw_prob': prob,
        })
    
    # 按概率排序
    result.sort(key=lambda x: -x['probability'])
    
    # 判断样本质量
    quality = 'high' if (sample_count >= 50 and distance <= 0.3) else \
              'medium' if (sample_count >= 30 and distance <= 0.5) else \
              'low' if (sample_count > 0) else 'none'
    
    return {
        'probs': result,
        'sample_info': {
            'sample_count': sample_count,
            'distance': round(distance, 2),
            'blend_weight': round(history_weight, 3),
            'quality': quality,
            'warnings': sample_warnings
        }
    }




def apply_market_change_prior(score_probs: Dict[str, float], asian: Dict, total: Dict,
                              weight: float = 0.08) -> Tuple[Dict[str, float], Dict]:
    """
    用盘口变化历史先验修正比分概率
    """
    try:
        from .market_db import MarketChangeDB, normalize_asian, normalize_ou

        asian_from = asian.get('open_handicap')
        asian_to = asian.get('handicap')

        ou_from = total.get('open_line')
        ou_to = get_close_total_line(total)

        asian_from = normalize_asian(asian_from)
        asian_to = normalize_asian(asian_to)
        ou_from = normalize_ou(ou_from)
        ou_to = normalize_ou(ou_to)

        if asian_from is None or asian_to is None or ou_from is None or ou_to is None:
            return score_probs, {'available': False, 'reason': '缺少开终盘'}

        db = MarketChangeDB()
        stats = db.get_change_stats(asian_from, asian_to, ou_from, ou_to)

        if not stats:
            return score_probs, {'available': False, 'reason': '无历史样本'}

        sample_count = stats.get('sample_count', 0)
        prior = stats.get('probabilities', {})

        # 样本不足只展示，不参与融合
        if sample_count < 30:
            return score_probs, {
                'available': True,
                'used': False,
                'sample_count': sample_count,
                'reason': '样本不足，仅展示',
                'top_scores': list(prior.items())[:5],
            }

        fused = {}
        all_scores = set(score_probs.keys()) | set(prior.keys())

        for score in all_scores:
            model_prob = score_probs.get(score, 0.0)
            prior_prob = prior.get(score, 0.0)
            fused[score] = (1 - weight) * model_prob + weight * prior_prob

        total_prob = sum(fused.values())
        if total_prob > 0:
            fused = {k: v / total_prob for k, v in fused.items()}

        return fused, {
            'available': True,
            'used': True,
            'sample_count': sample_count,
            'weight': weight,
            'top_scores': list(prior.items())[:5],
            'key': stats.get('key'),
        }

    except Exception as e:
        log.debug(f"盘口变化先验融合失败: {e}")
        return score_probs, {'available': False, 'reason': str(e)}


def predict_scores(asian, euro, total, team_strength=None, league_profile=None, 
                   model_type='poisson', enable_draw_calibration=True,
                   enable_calibration=False, calibration_method='platt',
                   enable_ensemble=False, ensemble_size=5, current_time_layer=None):
    """
    比分预测主函数：支持多种模型类型。
    
    参数：
        model_type: 'poisson'（泊松）、'negative_binomial'（负二项）、'bayesian'（贝叶斯）
        enable_draw_calibration: 是否启用平局概率校准
        enable_calibration: 是否启用概率输出校准
        calibration_method: 概率校准方法，'platt' 或 'isotonic'
        enable_ensemble: 是否启用多模型集成
        ensemble_size: 集成模型数量
    """
    # 如果启用多模型集成，直接调用集成函数
    if enable_ensemble:
        return ensemble_predict_scores(asian, euro, total, team_strength, league_profile,
                                      num_models=ensemble_size, method='average',
                                      current_time_layer=current_time_layer)
    p_home = euro['close']['home']
    p_draw = euro['close']['draw']
    p_away = euro['close']['away']
    p_over = total['close_prob']['over']
    line = total['close_line']
    open_line = total.get('open_line')

    target_total_pre = total.get('implied_total') or implied_total_goals(line, p_over)
    sup_asian = asian.get('implied_supremacy')
    if sup_asian is None:
        # 根据让球方向获取正确的概率值
        if asian['handicap'] > 0:
            # 主队让球：home_give是让球方概率，away_recv是受让方概率
            close_hp = asian['close_prob'].get('home_give', asian['close_prob'].get('home', 0.5))
            close_ap = asian['close_prob'].get('away_recv', asian['close_prob'].get('away', 0.5))
            open_hp = asian['open_prob'].get('home_give', asian['open_prob'].get('home', 0.5))
            open_ap = asian['open_prob'].get('away_recv', asian['open_prob'].get('away', 0.5))
        elif asian['handicap'] < 0:
            # 客队让球：home_recv是受让方概率，away_give是让球方概率
            close_hp = asian['close_prob'].get('home_recv', asian['close_prob'].get('home', 0.5))
            close_ap = asian['close_prob'].get('away_give', asian['close_prob'].get('away', 0.5))
            open_hp = asian['open_prob'].get('home_recv', asian['open_prob'].get('home', 0.5))
            open_ap = asian['open_prob'].get('away_give', asian['open_prob'].get('away', 0.5))
        else:
            # 平手盘
            close_hp = asian['close_prob'].get('home', 0.5)
            close_ap = asian['close_prob'].get('away', 0.5)
            open_hp = asian['open_prob'].get('home', 0.5)
            open_ap = asian['open_prob'].get('away', 0.5)
        
        sup_asian = asian_implied_supremacy(
            asian['handicap'], close_hp, close_ap, target_total_pre,
            open_handicap=asian.get('open_handicap'),
            open_hp=open_hp, open_ap=open_ap,
        )
    sup_euro = euro.get('implied_supremacy')
    if sup_euro is None:
        sup_euro = euro_implied_supremacy(p_home, p_draw, p_away, target_total_pre)
    supremacy = blend_market_supremacy(sup_asian, sup_euro)
    mom = euro.get('momentum') or {}
    supremacy += mom.get('shift_supremacy', 0) * MOMENTUM_SUPREMACY_WEIGHT
    if team_strength:
        supremacy += team_strength.get('momentum_supremacy', 0) * 0.35

    # 平局概率校准
    if enable_draw_calibration:
        home_draw_rate = team_strength.get('draw_rate_home', 0.25) if team_strength else 0.25
        away_draw_rate = team_strength.get('draw_rate_away', 0.25) if team_strength else 0.25
        league_draw_rate = league_profile.get('draw_rate', 0.25) if league_profile else 0.25
        
        p_home, p_draw, p_away = calibrate_draw_probability(
            p_home, p_draw, p_away, asian['handicap'],
            home_draw_rate, away_draw_rate, league_draw_rate
        )

    euro_lams = None
    el = euro.get('implied_lambdas')
    if el:
        euro_lams = (el['home'], el['away'])

    # 根据模型类型选择不同的预测方法
    if model_type == 'bayesian':
        # 贝叶斯框架：MCMC 采样后验分布
        targets = [p_home, p_draw, p_away]
        matrix, credible_interval, samples = bayesian_predict_scores(
            targets, target_total_pre, supremacy, league_profile, team_strength
        )
        policy_adjustment = {'applied': False}
        try:
            from .prediction_policy import apply_score_distribution_policy
            matrix, policy_adjustment = apply_score_distribution_policy(
                matrix,
                asian=asian,
                total=total,
                league_profile=league_profile,
            )
        except Exception as e:
            log.debug(f"贝叶斯比分分布策略调整失败: {e}")
        candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
        
        # 从后验均值获取 lambda 值
        lam_home = sum(s[0] for s in samples) / len(samples) if samples else (target_total_pre + supremacy) / 2
        lam_away = sum(s[1] for s in samples) / len(samples) if samples else (target_total_pre - supremacy) / 2
        target_total = target_total_pre
        
        meta = {
            'supremacy_asian': sup_asian,
            'supremacy_euro': sup_euro,
            'supremacy_blended': supremacy,
            'target_total': target_total,
            'credible_interval': credible_interval,
            'model_type': 'bayesian',
            'policy_adjustment': policy_adjustment,
        }
        return candidates, lam_home, lam_away, meta

    # 频率学派方法（泊松或负二项）
    try:
        # 获取盘口数据用于 λ 反推
        handicap = asian.get('handicap')
        open_handicap = asian.get('open_handicap')
        # 获取时间数据用于盘口变化速度分析
        open_time = asian.get('open_time')
        close_time = asian.get('close_time')
        
        lam_home, lam_away, target_total, rho = fit_lambdas_from_markets(
            supremacy, line, p_over, p_home, p_draw, p_away,
            open_total_line=open_line, team_strength=team_strength, euro_lambdas=euro_lams,
            league_profile=league_profile, handicap=handicap, open_handicap=open_handicap,
            open_time=open_time, close_time=close_time,
        )
        
        # fit_lambdas_from_markets 已经根据初终盘变化修正过主客队 λ。
        # analyze_asian/analyze_total 里的 lambda_adjust 仅保留为解释元数据，
        # 此处不再重复叠加同一个升降盘信号。
        
        # ========== 新增：应用博彩公司分歧指数调整 λ ==========
        bookmaker_consensus = asian.get('bookmaker_consensus')
        if bookmaker_consensus and bookmaker_consensus.get('available'):
            adjustment = bookmaker_consensus.get('adjustment', 0)
            if adjustment != 0:
                lam_home += adjustment
                log.debug("博彩公司分歧调整: lam_home += %.3f", adjustment)

        # 确保 λ 值为正
        lam_home = max(0.08, lam_home)
        lam_away = max(0.08, lam_away)
        
        # 选择分布类型
        distribution = 'negative_binomial' if model_type == 'negative_binomial' else 'poisson'
        matrix = _modeling_mod.build_score_matrix(lam_home, lam_away, rho=rho, distribution=distribution,
                                    league_profile=league_profile)

        margins = _matrix_margins(matrix)
        err = sum(
            (margins[k] - t) ** 2
            for k, t in zip(('home', 'draw', 'away'), (p_home, p_draw, p_away))
        )
        if err > 0.012:
            matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
    except (ValueError, ZeroDivisionError, OverflowError):
        lam_home, lam_away = estimate_lambdas(supremacy, line)
        
        # ========== 新增：应用盘口变化调整 λ（异常处理分支）==========
        asian_lambda_adjust = asian.get('lambda_adjust', {})
        if asian_lambda_adjust:
            lam_home += asian_lambda_adjust.get('home', 0)
            lam_away += asian_lambda_adjust.get('away', 0)
        
        total_lambda_adjust = total.get('lambda_adjust', {}).get('total', 0)
        if total_lambda_adjust != 0:
            if lam_home + lam_away > 0:
                lam_home += total_lambda_adjust * (lam_home / (lam_home + lam_away))
                lam_away += total_lambda_adjust * (lam_away / (lam_home + lam_away))
        
        lam_home = max(0.08, lam_home)
        lam_away = max(0.08, lam_away)
        
        distribution = 'negative_binomial' if model_type == 'negative_binomial' else 'poisson'
        rho = _estimate_dc_rho(lam_home, lam_away, p_draw)
        matrix = _modeling_mod.build_score_matrix(lam_home, lam_away, rho=rho, distribution=distribution,
                                    league_profile=league_profile)
        matrix = calibrate_to_euro(matrix, p_home, p_draw, p_away)
        target_total = line

    # ========== 新增：结合历史盘口比分库进行融合 ==========
    market_db_used = False
    change_db_used = False
    try:
        from .market_db import get_market_score_prob, blend_predictions, MarketChangeDB, normalize_asian, normalize_ou
        
        # 获取历史盘口比分概率
        handicap = asian.get('handicap', 0)
        close_line = total.get('close_line', 2.5)
        log.debug("加载历史盘口比分库: 亚盘=%s, 大小球=%s", handicap, close_line)
        
        market_result = get_market_score_prob(handicap, close_line)
        market_probs = market_result.get('probabilities', {})
        sample_count = market_result.get('sample_count', 0)
        distance = market_result.get('distance', float('inf'))
        
        log.debug(
            "历史盘口数据: 样本数=%s, 比分种类=%s, 距离=%.3f",
            sample_count, len(market_probs), distance,
        )
        
        # 将矩阵转换为字典格式
        model_probs = {f"{h}-{a}": prob for (h, a), prob in matrix.items()}
        
        # 融合权重初始化
        try:
            from .prediction_policy import get_prediction_policy
            prediction_policy = get_prediction_policy(
                league=league_profile.get('name') if league_profile else None,
                total_line=close_line,
                handicap=handicap,
                league_profile=league_profile,
            )
        except Exception:
            prediction_policy = {
                'static_market_cap': 0.15,
                'change_market_cap': 0.15,
                'late_market_weight_bias': 0.0,
            }

        model_weight = 0.75
        static_market_weight = prediction_policy.get('static_market_cap', 0.15)
        change_market_weight = min(0.10, prediction_policy.get('change_market_cap', 0.15))
        time_layer_market_adjustment = {'applied': False, 'layer': current_time_layer}
        try:
            late_bias = float(prediction_policy.get('late_market_weight_bias', 0.0) or 0.0)
        except (TypeError, ValueError):
            late_bias = 0.0
        if abs(late_bias) > 1e-9 and current_time_layer:
            if current_time_layer in {'T-1h', 'T-15min', 'final'}:
                layer_factor = 1.0 + late_bias
            elif current_time_layer in {'T-24h', 'T-6h'}:
                layer_factor = 1.0 - (late_bias * 0.5)
            else:
                layer_factor = 1.0
            static_market_weight = max(0.0, min(0.30, static_market_weight * layer_factor))
            change_market_weight = max(0.0, min(0.30, change_market_weight * layer_factor))
            time_layer_market_adjustment = {
                'applied': True,
                'layer': current_time_layer,
                'late_market_weight_bias': late_bias,
                'factor': round(layer_factor, 4),
            }
        market_data_quality = _assess_market_data_quality(asian, euro, total)
        market_quality_factor = market_data_quality.get('weight_factor', 1.0)
        
        # ========== 静态盘口先验 ==========
        if sample_count >= 30 and distance <= 0.5 and market_probs and len(market_probs) >= 3:
            # 计算历史权重：样本越多、盘口越接近，权重越高
            static_cap = static_market_weight
            static_weight = min(static_cap, sample_count / 300 * static_cap) * market_quality_factor
            static_market_weight = static_weight
            
            # 融合预测：模型概率 + 静态历史盘口概率
            blended_probs = blend_predictions(model_probs, market_probs, 
                                             weights={'model': model_weight + (0.15 - static_weight), 'market': static_weight})
            model_probs = blended_probs
            market_db_used = True
            log.debug(
                "静态盘口比分库融合: 模型=%.0f%%, 静态历史=%.0f%%",
                (model_weight + (0.15 - static_weight)) * 100, static_weight * 100,
            )
        else:
            static_market_weight = 0
            if sample_count < 30:
                log.debug("静态盘口样本不足(%s<30)，跳过融合", sample_count)
            elif distance > 0.5:
                log.debug("盘口距离过远(%.3f>0.5)，跳过融合", distance)
        
        # ========== 盘口变化先验 ==========
        # 获取开盘盘口数据
        open_handicap = asian.get('open_handicap')
        open_line = total.get('open_line')
        
        if open_handicap is not None and open_line is not None:
            # 标准化盘口
            asian_open = normalize_asian(open_handicap)
            asian_close = normalize_asian(handicap)
            ou_open = normalize_ou(open_line)
            ou_close = normalize_ou(close_line)
            
            # 查询盘口变化统计
            change_db = MarketChangeDB()
            change_stats = change_db.get_change_stats(asian_open, asian_close, ou_open, ou_close)
            
            if change_stats:
                # 估算样本数：假设最大概率对应的实际样本数
                max_prob = max(change_stats.values()) if change_stats else 0
                change_sample_count = int(round(1 / max_prob)) if max_prob > 0 else 0
                
                # 样本门槛
                if change_sample_count >= 30:
                    # 计算变化权重：5%～15%
                    change_cap = change_market_weight
                    change_weight = min(change_cap, change_sample_count / 300 * change_cap) * market_quality_factor
                    change_market_weight = change_weight
                    
                    # 融合预测：当前概率 + 变化历史概率
                    blended_probs = blend_predictions(model_probs, change_stats,
                                                     weights={'current': 1 - change_weight, 'change': change_weight})
                    model_probs = blended_probs
                    change_db_used = True
                    log.debug(
                        "盘口变化数据库融合: 当前=%.0f%%, 变化历史=%.0f%%",
                        (1 - change_weight) * 100, change_weight * 100,
                    )
                else:
                    log.debug("盘口变化样本不足(%s<30)，跳过融合", change_sample_count)
            else:
                log.debug(
                    "未找到盘口变化记录: %s→%s, %s→%s",
                    asian_open, asian_close, ou_open, ou_close,
                )
        
        # 更新矩阵
        if market_db_used or change_db_used:
            matrix = {}
            for score, prob in model_probs.items():
                h, a = map(int, score.split('-'))
                matrix[(h, a)] = prob
    except Exception as e:
        log.debug(f"无法加载历史盘口比分库进行融合: {e}")

    # 应用残差修正（如果有训练好的模型）
    features = _build_residual_features(asian, euro, total, team_strength, league_profile)
    matrix = apply_residual_correction(matrix, features)

    # 应用概率输出校准
    if enable_calibration:
        # 优先使用贝叶斯校准（基于真实历史预测记录），失败退回 Platt
        try:
            # 获取联赛和盘口信息用于市场环境校准
            league_info = team_strength.get('league', '') if team_strength else ''
            # 转换为字典格式 {"1-1": 0.108, ...}
            score_probs = {f"{h}-{a}": p for (h, a), p in matrix.items()}
            # 使用贝叶斯校准（带市场环境信息）
            score_probs = calibrate_predictions(score_probs, league_info, line, sup_asian or 0)
            # 转换回原始格式
            matrix = {
                tuple(map(int, score.split("-"))): prob
                for score, prob in score_probs.items()
            }
            log.debug("已应用贝叶斯概率校准")
        except Exception as e:
            log.warning(f"贝叶斯校准失败，降级使用Platt校准: {e}")
            # 降级到 Platt 校准
            league_name = league_profile.get('name', 'default') if league_profile else 'default'
            calibration_data = get_league_calibration_data(league_name)
            matrix = calibrate_probabilities(matrix, method=calibration_method, calibration_data=calibration_data)

    policy_adjustment = {'applied': False}
    try:
        from .prediction_policy import apply_score_distribution_policy
        matrix, policy_adjustment = apply_score_distribution_policy(
            matrix,
            asian=asian,
            total=total,
            league_profile=league_profile,
        )
    except Exception as e:
        log.debug(f"比分分布策略调整失败: {e}")

    candidates = sorted(matrix.items(), key=lambda kv: -kv[1])
    meta = {
        'supremacy_asian': sup_asian,
        'supremacy_euro': sup_euro,
        'supremacy_blended': supremacy,
        'target_total': target_total,
        'model_type': model_type,
        'distribution': distribution,
        'calibrated': enable_calibration,
        'calibration_method': calibration_method if enable_calibration else None,
        'handicap_change': asian.get('handicap_change'),
        'line_change': total.get('line_change'),
        'market_db_used': market_db_used,
        'market_data_quality': locals().get('market_data_quality', {'score': 1.0, 'grade': 'unknown'}),
        'market_quality_factor': locals().get('market_quality_factor', 1.0),
        'current_time_layer': current_time_layer,
        'time_layer_market_adjustment': locals().get('time_layer_market_adjustment', {'applied': False}),
        'policy_adjustment': policy_adjustment,
    }
    return candidates, lam_home, lam_away, meta












SCORE_CLUSTERS = {
    # 主胜簇
    'home_win_1': [(1, 0), (2, 1), (3, 2), (4, 3)],      # 主胜1球
    'home_win_2': [(2, 0), (3, 1), (4, 2), (5, 3)],      # 主胜2球
    'home_win_3': [(3, 0), (4, 1), (5, 2)],              # 主胜3球+
    # 平局簇
    'draw': [(0, 0), (1, 1), (2, 2), (3, 3)],           # 平局
    # 客胜簇
    'away_win_1': [(0, 1), (1, 2), (2, 3), (3, 4)],      # 客胜1球
    'away_win_2': [(0, 2), (1, 3), (2, 4), (3, 5)],      # 客胜2球
    'away_win_3': [(0, 3), (1, 4), (2, 5)],              # 客胜3球+
}















































