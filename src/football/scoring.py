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
    AVG_LEAGUE_GOAL, BAYESIAN_CALIBRATION_AVAILABLE, COLD_FILTER_BONUS, DYNAMIC_ELO_AVAILABLE, HEAT_FILTER_PENALTY, HEAT_RATIO_COLD, HEAT_RATIO_HOT, LEAGUE_PROFILES, MARKET_CLUSTERING_AVAILABLE, MAX_GOALS, MOMENTUM_SUPREMACY_WEIGHT, SCORE_1X2_MARKET_ANCHOR_STRENGTH, SCORE_BASELINE_FREQ, SUPREMACY_CONFLICT_GAP, VALUE_BETTING_AVAILABLE, calculate_ev, calculate_value, calibrate_predictions, get_elo_difference, get_market_prior,
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


def fit_lambdas_from_markets(
    supremacy, total_line, p_over,
    p_home, p_draw, p_away,
    open_total_line=None, team_strength=None, euro_lambdas=None,
    league_profile=None, handicap=None, open_handicap=None,
    open_time=None, close_time=None,
):
    """
    大小球反推总进球 + 反推净胜球 + 欧赔/球队先验，网格+坐标下降拟合 λ
    
    核心改进（按优先级）：
    1. 盘口直接反推 λ（主让1.0 + 大小球3.0 → home=2.0, away=1.0）
    2. 亚盘升降盘对 λ 的修正（包含时间因素）
    3. 大小球升降对 λ 的修正（包含时间因素）
    4. ELO xG 直接参与 λ 融合
    """
    # 1. 计算融合后的大小球线
    line = _blend_close_open(total_line, open_total_line)
    lp = league_profile or LEAGUE_PROFILES['default']
    avg_goal = lp.get('avg_goal', AVG_LEAGUE_GOAL)
    
    # 2. 由大小球反推总进球期望
    target_total = implied_total_goals(line, p_over)
    target_total = max(avg_goal * 1.4, min(avg_goal * 3.2, target_total))
    
    # 3. 核心改进：由盘口直接反推 λ（作为主先验）
    market_lams = None
    if handicap is not None:
        market_lams = market_implied_lambdas(handicap, target_total)
    
    # 4. 计算球队攻防 λ
    team_lams = None
    if team_strength:
        team_lams = team_poisson_lambdas(team_strength, target_total, lp)
    
    # 5. 获取 ELO xG（如果可用）
    elo_lams = None
    if team_strength and 'elo_xg_home' in team_strength and 'elo_xg_away' in team_strength:
        elo_lams = (team_strength['elo_xg_home'], team_strength['elo_xg_away'])
    
    # 6. 融合市场、球队和 ELO 的 λ 值
    if market_lams:
        lam_home, lam_away = blend_lambdas_with_market(market_lams, team_lams, elo_lams)
    elif team_lams:
        lam_home, lam_away = team_lams
    else:
        lam_home, lam_away = estimate_lambdas(supremacy, target_total)
    
    # 7. 应用盘口变化调整（亚盘升降盘，包含时间因素）
    lam_home, lam_away = apply_handicap_change_adjustment(
        lam_home, lam_away, open_handicap, handicap,
        open_time, close_time
    )
    
    # 8. 应用大小球变化调整（包含时间因素）
    lam_home, lam_away = apply_total_line_change_adjustment(
        lam_home, lam_away, open_total_line, total_line,
        open_time, close_time
    )
    
    # 9. 使用融合后的 λ 作为先验，进行网格拟合精调
    ou_targets = _ou_total_distribution(target_total)
    fused_lams = (lam_home, lam_away)
    
    lam_home, lam_away = _fit_lambda_grid(
        supremacy, target_total, p_home, p_draw, p_away, rho=0.0,
        ou_targets=ou_targets, team_lambdas=fused_lams, euro_lambdas=euro_lambdas,
    )
    
    # 10. 估计 Dixon-Coles rho 并再次精调
    rho = _estimate_dc_rho(lam_home, lam_away, p_draw)
    lam_home, lam_away = _fit_lambda_grid(
        supremacy, target_total, p_home, p_draw, p_away, rho=rho,
        ou_targets=ou_targets, team_lambdas=fused_lams, euro_lambdas=euro_lambdas,
    )
    
    return lam_home, lam_away, target_total, rho


def _baseline_freq(h, a, league_profile=None):
    """历史基准频率（用于兼容旧逻辑）"""
    base = SCORE_BASELINE_FREQ.get((h, a), 0.018)
    if not league_profile:
        return base
    low_mult = league_profile.get('low_score', 1.0)
    draw_mult = league_profile.get('draw_mult', 1.0)
    if h == a:
        return base * draw_mult
    if h + a <= 2:
        return base * low_mult
    if h + a >= 4:
        return base / max(low_mult, 0.85)
    return base


def score_implied_prob_from_euro(h, a, euro_odds):
    """
    由欧赔计算比分的隐含概率（简化版）。
    使用 Dixon-Coles 风格的近似：先计算 1X2 概率，再按比分分布特征调整。
    """
    home_odds, draw_odds, away_odds = euro_odds['home'], euro_odds['draw'], euro_odds['away']
    
    # 去水概率
    p_home, p_draw, p_away = remove_vig(home_odds, draw_odds, away_odds)
    
    # 比分概率近似计算
    diff = h - a
    
    if diff > 0:  # 主胜
        base_prob = p_home
        # 主胜比分按净胜球分布：净胜1球概率最高，净胜越多概率越低
        if diff == 1:
            base_prob *= 0.55  # 净胜1球占主胜的约55%
        elif diff == 2:
            base_prob *= 0.28  # 净胜2球占主胜的约28%
        elif diff == 3:
            base_prob *= 0.12  # 净胜3球占主胜的约12%
        else:
            base_prob *= 0.05  # 净胜3球以上占主胜的约5%
    elif diff == 0:  # 平局
        base_prob = p_draw
        # 平局比分分布：1-1最高，0-0次之，2-2及以上较少
        if h == 0:
            base_prob *= 0.35  # 0-0 占平局的约35%
        elif h == 1:
            base_prob *= 0.45  # 1-1 占平局的约45%
        elif h == 2:
            base_prob *= 0.15  # 2-2 占平局的约15%
        else:
            base_prob *= 0.05  # 3-3及以上占平局的约5%
    else:  # 客胜
        base_prob = p_away
        # 客胜比分按净胜球分布，对称于主胜
        if diff == -1:
            base_prob *= 0.55
        elif diff == -2:
            base_prob *= 0.28
        elif diff == -3:
            base_prob *= 0.12
        else:
            base_prob *= 0.05
    
    return max(0.001, min(0.5, base_prob))


def score_heat_label(h, a, model_prob, league_profile=None, euro_odds=None, use_implied_prob=True):
    """
    比分冷热：模型概率 vs 赔率隐含概率（或历史基准频率）。
    
    参数：
        h, a: 主客进球数
        model_prob: 模型预测概率
        league_profile: 联赛画像（用于历史基准）
        euro_odds: 欧赔赔率 {'home': x, 'draw': y, 'away': z}
        use_implied_prob: 是否使用赔率隐含概率（默认是）
    
    返回：
        ('cold' | 'hot' | 'neutral', ratio)
        
    冷=模型概率 > 赔率隐含概率（模型更看好但市场忽视）
    热=模型概率 < 赔率隐含概率（市场过热，难出）
    """
    if use_implied_prob and euro_odds:
        # 基于赔率隐含概率计算冷热
        implied_prob = score_implied_prob_from_euro(h, a, euro_odds)
        if implied_prob <= 0:
            return 'neutral', 1.0
        
        # 冷热阈值随概率大小动态调整（小概率事件更容易出现冷热偏差）
        ratio = model_prob / implied_prob
        
        # 动态阈值：概率越小，阈值越宽
        prob_scale = min(1.0, implied_prob * 20)  # 归一化到 0-1
        cold_threshold = 1.25 + (1.45 - 1.25) * (1 - prob_scale)   # 1.25 ~ 1.45
        hot_threshold = 0.75 - (0.75 - 0.65) * (1 - prob_scale)    # 0.65 ~ 0.75
        
        if ratio >= cold_threshold:
            return 'cold', ratio
        if ratio <= hot_threshold:
            return 'hot', ratio
        return 'neutral', ratio
    else:
        # 回退到历史基准频率（兼容旧逻辑）
        base = _baseline_freq(h, a, league_profile)
        if base <= 0:
            return 'neutral', 1.0
        ratio = model_prob / base
        if ratio >= HEAT_RATIO_COLD:
            return 'cold', ratio
        if ratio <= HEAT_RATIO_HOT:
            return 'hot', ratio
        return 'neutral', ratio


def _heat_filter_weight(heat):
    if heat == 'hot':
        return HEAT_FILTER_PENALTY
    if heat == 'cold':
        return COLD_FILTER_BONUS
    return 1.0


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
                    log.info(f"半全场概率已融合真实半场统计数据，权重={real_weight:.3f}")
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
                
                log.info(f"半全场概率已结合历史盘口数据调整（倒推数据），权重={history_weight:.3f}")
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


def _half_full_probs_to_dict(half_full_time):
    """Convert half/full-time display rows into raw probability dict for storage."""
    if not half_full_time:
        return None

    if isinstance(half_full_time, dict):
        if 'distribution' in half_full_time and isinstance(half_full_time['distribution'], dict):
            return half_full_time['distribution']

        probs = half_full_time.get('probs')
        if isinstance(probs, dict):
            return probs
        if isinstance(probs, list):
            result = {}
            for item in probs:
                code = item.get('code')
                if not code:
                    continue
                if 'raw_prob' in item:
                    result[code] = item.get('raw_prob', 0.0)
                else:
                    result[code] = item.get('probability', 0.0) / 100.0
            return result or None

    return None


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
                log.info(f"应用博彩公司分歧指数调整: lam_home += {adjustment:.3f}")

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
        log.info(f"尝试加载历史盘口比分库: 亚盘={handicap}, 大小球={close_line}")
        
        market_result = get_market_score_prob(handicap, close_line)
        market_probs = market_result.get('probabilities', {})
        sample_count = market_result.get('sample_count', 0)
        distance = market_result.get('distance', float('inf'))
        
        log.info(f"历史盘口数据: 样本数={sample_count}, 比分种类={len(market_probs)}, 距离={distance:.3f}")
        
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
            log.info(f"静态盘口比分库融合成功，权重: 模型{(model_weight + (0.15 - static_weight)):.0%} + 静态历史{static_weight:.0%}")
        else:
            static_market_weight = 0
            if sample_count < 30:
                log.info(f"静态盘口样本不足({sample_count}<30)，跳过融合")
            elif distance > 0.5:
                log.info(f"盘口距离过远({distance:.3f}>0.5)，跳过融合")
        
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
                    log.info(f"盘口变化数据库融合成功，权重: 当前{(1 - change_weight):.0%} + 变化历史{change_weight:.0%}")
                else:
                    log.info(f"盘口变化样本不足({change_sample_count}<30)，跳过融合")
            else:
                log.info(f"未找到盘口变化记录: {asian_open}→{asian_close}, {ou_open}→{ou_close}")
        
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
        # 优先使用贝叶斯校准（基于真实历史预测记录）
        if BAYESIAN_CALIBRATION_AVAILABLE:
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
                log.info("已应用贝叶斯概率校准")
            except Exception as e:
                log.warning(f"贝叶斯校准失败，降级使用Platt校准: {e}")
                # 降级到 Platt 校准
                league_name = league_profile.get('name', 'default') if league_profile else 'default'
                calibration_data = get_league_calibration_data(league_name)
                matrix = calibrate_probabilities(matrix, method=calibration_method, calibration_data=calibration_data)
        else:
            # 使用传统 Platt 校准
            league_name = league_profile.get('name', 'default') if league_profile else 'default'
            calibration_data = get_league_calibration_data(league_name)
            log.debug(f"使用联赛 {league_name} 的校准参数: Platt(A={calibration_data['platt_params'][0]:.4f}, B={calibration_data['platt_params'][1]:.4f})")
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


def _estimate_score_odds(h, a, euro_odds):
    """
    估算比分赔率（基于欧赔）
    
    参数：
        h: 主队进球数
        a: 客队进球数
        euro_odds: 欧赔赔率 {'home': x, 'draw': y, 'away': z}
    
    返回：
        估算的比分赔率
    """
    try:
        home_odds = euro_odds.get('home', 2.0)
        draw_odds = euro_odds.get('draw', 3.0)
        away_odds = euro_odds.get('away', 4.0)
        
        # 基于结果类型估算比分赔率
        if h > a:
            base_odds = home_odds
        elif h == a:
            base_odds = draw_odds
        else:
            base_odds = away_odds
        
        # 根据进球数调整
        total_goals = h + a
        if total_goals <= 1:
            return base_odds * 1.5
        elif total_goals <= 3:
            return base_odds * 1.2
        else:
            return base_odds * 1.8
    except Exception:
        return 1.0


def _score_entry(h, a, prob, heat_info=None):
    entry = {'home': h, 'away': a, 'prob': prob, 'result': _result_label(h, a)}
    if heat_info:
        entry['heat'] = heat_info[0]
        entry['heat_ratio'] = round(heat_info[1], 2)
    return entry


def _alignment_score(h, a, asian, euro, total):
    """赔率信号一致性得分（0~1），用于在概率接近时优选更贴合市场的比分"""
    diff = h - a
    favor, diff_range, hcap = asian['favor'], asian['diff_range'], asian['handicap']
    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    lo, hi = total['expected_goals']
    score = 0.0

    if favor == 'home' and diff > 0 and diff_range[0] <= diff <= diff_range[1]:
        score += 0.35
    elif favor == 'away' and diff < 0 and diff_range[0] <= -diff <= diff_range[1]:
        score += 0.35
    elif favor == 'even' and diff == 0:
        score += 0.25

    top = max(p_home, p_draw, p_away)
    if diff > 0 and p_home >= top - 0.03:
        score += 0.3
    elif diff < 0 and p_away >= top - 0.03:
        score += 0.3
    elif diff == 0 and p_draw >= top - 0.03:
        score += 0.3

    goals = h + a
    if lo <= goals <= hi:
        score += 0.25
    elif abs(goals - (lo + hi) / 2) <= 1.0:
        score += 0.12

    if total['lean'] == 'over' and goals >= total['close_line']:
        score += 0.1
    elif total['lean'] == 'under' and goals <= total['close_line']:
        score += 0.1

    return min(1.0, score)


def _recommend_reasons(h, a, asian, euro, total, team=None, heat=None):
    """为单个推荐比分生成理由列表"""
    diff = h - a
    favor, diff_range, hcap = asian['favor'], asian['diff_range'], asian['handicap']
    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']
    lo, hi = total['expected_goals']

    reasons = []
    if team:
        reasons.append(f"攻防强度 λ≈{team['attack_home']:.2f}/{team['attack_away']:.2f}进")
        if team.get('form_diff', 0) > 0.35 and diff > 0:
            reasons.append('主队近期状态更好')
        elif team.get('form_diff', 0) < -0.35 and diff < 0:
            reasons.append('客队近期状态更好')
    mom = euro.get('momentum') or {}
    if mom.get('summary') and mom['summary'] != '欧赔走势平稳':
        reasons.append(mom['summary'])
    if favor == 'home' and diff > 0 and diff_range[0] <= diff <= diff_range[1]:
        reasons.append(f"符合主让{hcap}球盘口预期")
    elif favor == 'away' and diff < 0 and diff_range[0] <= -diff <= diff_range[1]:
        reasons.append(f"符合客让{abs(hcap)}球盘口预期")
    elif diff == 0:
        reasons.append("欧赔平局概率支撑")
    if diff > 0 and p_home > 0.4:
        reasons.append(f"欧赔主胜概率{p_home*100:.0f}%")
    elif diff < 0 and p_away > 0.4:
        reasons.append(f"欧赔客胜概率{p_away*100:.0f}%")
    elif diff == 0 and p_draw > 0.3:
        reasons.append(f"欧赔平局概率{p_draw*100:.0f}%")
    if lo <= h + a <= hi:
        reasons.append(f"总进球{h+a}球在预期区间")
    if heat == 'cold':
        reasons.append("冷门口比分（模型概率高于历史基准）")
    elif heat == 'hot':
        reasons.append("热门比分（已降权）")
    kelly = euro.get('kelly')
    if kelly:
        fav = kelly.get('favored')
        hard = kelly.get('hardest')
        # 只有当不是中性时才进行判断
        if fav != 'neutral':
            if fav == 'home' and diff > 0:
                reasons.append("凯利指数相对看好主胜")
            elif fav == 'away' and diff < 0:
                reasons.append("凯利指数相对看好客胜")
            elif fav == 'draw' and diff == 0:
                reasons.append("凯利指数相对看好平局")
        if hard != 'neutral':
            if hard == 'home' and diff > 0:
                reasons.append("凯利提示主胜打出难度偏大")
            elif hard == 'away' and diff < 0:
                reasons.append("凯利提示客胜打出难度偏大")
    return reasons or ["综合赔率推断"]


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


def _get_score_cluster(h, a):
    """获取比分所属的簇"""
    diff = h - a
    if diff > 0:
        if diff == 1:
            return 'home_win_1'
        elif diff == 2:
            return 'home_win_2'
        else:
            return 'home_win_3'
    elif diff < 0:
        if diff == -1:
            return 'away_win_1'
        elif diff == -2:
            return 'away_win_2'
        else:
            return 'away_win_3'
    else:
        return 'draw'


def _get_cluster_name(cluster):
    """获取簇的中文名称"""
    cluster_names = {
        'home_win_1': '主胜1球',
        'home_win_2': '主胜2球',
        'home_win_3': '主胜3球+',
        'draw': '平局',
        'away_win_1': '客胜1球',
        'away_win_2': '客胜2球',
        'away_win_3': '客胜3球+',
    }
    return cluster_names.get(cluster, cluster)


def score_pattern(h: int, a: int) -> str:
    """
    判断比分属于哪种剧本模式
    
    参数：
        h: 主队进球数
        a: 客队进球数
    
    返回：
        剧本模式标识
    """
    total = h + a
    
    if h > a and total <= 2:
        return 'home_low'      # 主胜小比分
    if h > a and total >= 3:
        return 'home_high'     # 主胜大比分
    if h == a:
        return 'draw'          # 平局
    if h < a and total <= 2:
        return 'away_low'      # 客胜小比分
    return 'away_high'         # 客胜大比分


def _score_total_line_factor(h: int, a: int, total_line: float) -> float:
    """Soft ranking factor for score totals against the closing O/U line."""
    if total_line is None:
        return 1.0
    try:
        line = float(total_line)
    except (TypeError, ValueError):
        return 1.0

    goals = h + a
    distance = abs(goals - line)
    factor = 1.06 - min(distance, 2.5) * 0.08

    if line <= 2.25 and goals >= 4:
        factor *= 0.82
    elif line <= 2.25 and goals <= 2:
        factor *= 1.04
    elif line >= 3.0 and goals <= 1:
        factor *= 0.84
    elif line >= 3.0 and goals >= 3:
        factor *= 1.04

    return max(0.72, min(1.10, factor))


def _common_score_overheat_factor(h: int, a: int, prob: float, total_line: float) -> float:
    """Dampen common scores when their raw probability is too dominant."""
    baselines = {
        (0, 0): 0.095,
        (1, 0): 0.120,
        (0, 1): 0.110,
        (1, 1): 0.135,
    }
    baseline = baselines.get((h, a))
    if baseline is None or prob <= baseline * 1.30:
        return 1.0

    factor = 0.94
    try:
        line = float(total_line) if total_line is not None else None
    except (TypeError, ValueError):
        line = None

    if line is not None:
        goals = h + a
        if line >= 3.0 and goals <= 1:
            factor *= 0.88
        elif line <= 2.25 and goals >= 3:
            factor *= 0.88
    if prob >= baseline * 1.65:
        factor *= 0.92
    return max(0.78, factor)


def _total_market_tempo_signal(total: Dict) -> Dict:
    """Return a conservative tempo signal from O/U line and water movement."""
    total = total or {}
    try:
        close_line = float(get_close_total_line(total))
    except (TypeError, ValueError):
        close_line = None

    try:
        open_line = float(total.get('open_line'))
    except (TypeError, ValueError):
        open_line = close_line

    open_prob = total.get('open_prob') or {}
    close_prob = total.get('close_prob') or {}
    try:
        over_delta = float(close_prob.get('over', 0.0)) - float(open_prob.get('over', 0.0))
    except (TypeError, ValueError):
        over_delta = 0.0

    line_delta = 0.0
    if open_line is not None and close_line is not None:
        line_delta = close_line - open_line

    line_signal = max(-1.0, min(1.0, line_delta / 0.5))
    water_signal = max(-1.0, min(1.0, over_delta / 0.08))
    base_signal = (0.62 * line_signal) + (0.38 * water_signal)

    if close_line is not None:
        if close_line >= 3.0:
            base_signal += 0.22
        elif close_line <= 2.25:
            base_signal -= 0.22

    conflict = line_signal * water_signal < -0.15
    if conflict:
        base_signal *= 0.35

    return {
        'signal': max(-1.0, min(1.0, base_signal)),
        'line': close_line,
        'line_delta': line_delta,
        'over_delta': over_delta,
        'conflict': conflict,
    }


def _joint_market_state(asian: Dict, euro: Dict, total: Dict) -> Dict:
    """Combine handicap, water, 1X2 and O/U movement into one market state."""
    asian = asian or {}
    euro = euro or {}
    tempo = _total_market_tempo_signal(total)

    try:
        handicap_signal = max(-1.0, min(1.0, float(asian.get('handicap_change', 0.0)) / 0.5))
    except (TypeError, ValueError):
        handicap_signal = 0.0
    prob_change = asian.get('prob_change') or {}
    try:
        asian_water_signal = max(-1.0, min(1.0, float(prob_change.get('home', 0.0)) / 0.08))
    except (TypeError, ValueError):
        asian_water_signal = 0.0
    try:
        euro_signal = max(-1.0, min(1.0, float((euro.get('momentum') or {}).get('shift_supremacy', 0.0)) / 0.25))
    except (TypeError, ValueError):
        euro_signal = 0.0

    directional_parts = [handicap_signal, asian_water_signal, euro_signal]
    active = [value for value in directional_parts if abs(value) >= 0.10]
    conflict = bool(active and min(active) < -0.10 and max(active) > 0.10)
    direction_signal = (
        0.45 * handicap_signal + 0.30 * asian_water_signal + 0.25 * euro_signal
    )
    agreement = 1.0
    if conflict:
        agreement = 0.40
        direction_signal *= agreement

    strength = min(1.0, (abs(direction_signal) + abs(tempo['signal'])) / 1.6)
    return {
        'direction_signal': max(-1.0, min(1.0, direction_signal)),
        'tempo_signal': tempo['signal'],
        'handicap_signal': handicap_signal,
        'asian_water_signal': asian_water_signal,
        'euro_signal': euro_signal,
        'conflict': conflict or bool(tempo.get('conflict')),
        'agreement_factor': agreement,
        'strength': strength,
        'tempo': tempo,
    }


def _apply_joint_market_state(candidates, asian: Dict, euro: Dict, total: Dict):
    """Softly fit one score matrix to the closing Asian and O/U prices.

    Closing prices already contain the information accumulated during the
    open-to-close move.  The movement state is therefore used as a reliability
    control (especially when markets conflict), not as another independent
    vote.  A 0.35 constraint was selected on a 2024/25 -> 2025/26 walk-forward
    because it improved exact-score, goal and 1X2 log loss without allowing a
    single market to dominate the prior.
    """
    rows = list(candidates or [])
    if not rows:
        return rows, {'applied': False, 'reason': 'empty_distribution'}
    state = _joint_market_state(asian, euro, total)

    matrix = {}
    for item in rows:
        try:
            if len(item) == 2 and isinstance(item[0], tuple):
                score, probability = item
            else:
                score, probability = (item[0], item[1]), item[2]
            score = int(score[0]), int(score[1])
            matrix[score] = matrix.get(score, 0.0) + max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
    raw_total = sum(matrix.values())
    if raw_total <= 0:
        return rows, {'applied': False, 'reason': 'zero_raw_mass', **state}
    matrix = {score: probability / raw_total for score, probability in matrix.items()}
    before_matrix = dict(matrix)

    def line_parts(line):
        line = round(float(line) * 4.0) / 4.0
        lower = math.floor(line * 2.0) / 2.0
        if abs(line - lower) < 1e-8:
            return (lower,)
        return (lower, lower + 0.5)

    def settlement_profit(value, line, fair_decimal):
        profits = []
        for settlement_line in line_parts(line):
            if value > settlement_line + 1e-8:
                profits.append(fair_decimal - 1.0)
            elif value < settlement_line - 1e-8:
                profits.append(-1.0)
            else:
                profits.append(0.0)
        return sum(profits) / len(profits)

    def constrain(source, feature, strength):
        values = {score: feature(score) for score in source}
        buckets = {}
        for score, probability in source.items():
            value = round(values[score], 12)
            buckets[value] = buckets.get(value, 0.0) + probability

        def expectation(theta):
            weighted = [
                (value, probability * math.exp(max(-20.0, min(20.0, theta * value))))
                for value, probability in buckets.items()
            ]
            denominator = sum(probability for _, probability in weighted)
            return sum(value * probability for value, probability in weighted) / denominator

        lo, hi = -12.0, 12.0
        if expectation(lo) > 0 or expectation(hi) < 0:
            return source, {'applied': False, 'reason': 'target_outside_support'}
        expected_before = expectation(0.0)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if expectation(mid) < 0:
                lo = mid
            else:
                hi = mid
        theta = strength * (lo + hi) / 2.0
        adjusted = {
            score: probability * math.exp(max(-20.0, min(20.0, theta * values[score])))
            for score, probability in source.items()
        }
        denominator = sum(adjusted.values())
        adjusted = {score: probability / denominator for score, probability in adjusted.items()}
        expected_after = sum(adjusted[score] * values[score] for score in adjusted)
        return adjusted, {
            'applied': True,
            'theta': round(theta, 5),
            'fair_profit_before': round(expected_before, 5),
            'fair_profit_after': round(expected_after, 5),
        }

    close_asian = asian.get('close_prob') or {}
    home_probability = next((close_asian.get(key) for key in
                             ('home_give', 'home_recv', 'home')
                             if close_asian.get(key) is not None), None)
    over_probability = (total.get('close_prob') or {}).get('over')
    handicap = asian.get('handicap')
    total_line = total.get('close_line')
    reliability = 0.40 if state.get('conflict') else 1.0
    pass_strength = 0.35 * reliability / 3.0
    asian_meta = {'applied': False, 'reason': 'missing_price_or_line'}
    total_meta = {'applied': False, 'reason': 'missing_price_or_line'}
    for _ in range(3):
        try:
            if handicap is not None and home_probability and 0 < float(home_probability) < 1:
                fair_decimal = 1.0 / float(home_probability)
                matrix, asian_meta = constrain(
                    matrix,
                    lambda score, line=float(handicap), odds=fair_decimal: settlement_profit(
                        score[0] - score[1], line, odds
                    ),
                    pass_strength,
                )
        except (TypeError, ValueError, ZeroDivisionError):
            asian_meta = {'applied': False, 'reason': 'invalid_price_or_line'}
        try:
            if total_line is not None and over_probability and 0 < float(over_probability) < 1:
                fair_decimal = 1.0 / float(over_probability)
                matrix, total_meta = constrain(
                    matrix,
                    lambda score, line=float(total_line), odds=fair_decimal: settlement_profit(
                        score[0] + score[1], line, odds
                    ),
                    pass_strength,
                )
        except (TypeError, ValueError, ZeroDivisionError):
            total_meta = {'applied': False, 'reason': 'invalid_price_or_line'}

    if not asian_meta.get('applied') and not total_meta.get('applied'):
        state.update({'applied': False, 'reason': 'missing_closing_market_prices'})
        return rows, state

    adjusted = sorted(matrix.items(), key=lambda item: -item[1])
    expected_before = sum(sum(score) * probability for score, probability in before_matrix.items())
    home_mass_before = sum(
        probability for (home, away), probability in before_matrix.items() if home > away
    )
    expected_after = sum(sum(score) * probability for score, probability in adjusted)
    home_mass_after = sum(probability for (home, away), probability in adjusted if home > away)
    state.update({
        'applied': True,
        'method': 'maximum_entropy_fair_price_constraint',
        'constraint_strength': round(0.35 * reliability, 3),
        'asian_constraint': asian_meta,
        'total_constraint': total_meta,
        'expected_goals_before': expected_before,
        'expected_goals_after': expected_after,
        'home_win_before': home_mass_before,
        'home_win_after': home_mass_after,
    })
    return adjusted, state


def _score_total_movement_factor(h: int, a: int, total: Dict) -> float:
    """Soft score-ranking factor from O/U movement so score picks follow goal picks."""
    signal_info = _total_market_tempo_signal(total)
    signal = signal_info.get('signal', 0.0)
    if abs(signal) < 0.12:
        return 1.0

    goals = h + a
    line = signal_info.get('line')
    if line is None:
        line = 2.5

    distance = goals - line
    factor = 1.0 + max(-0.18, min(0.18, signal * distance * 0.10))

    if signal > 0 and goals >= math.ceil(line + 0.5):
        factor *= 1.0 + min(0.08, signal * 0.05)
    elif signal < 0 and goals <= math.floor(line):
        factor *= 1.0 + min(0.08, abs(signal) * 0.05)

    if signal > 0 and goals <= 1:
        factor *= 0.93
    elif signal < 0 and goals >= 4:
        factor *= 0.93

    return max(0.82, min(1.12, factor))


def _adjust_score_probs_with_total_movement(score_probs: Dict[str, float], total: Dict) -> Tuple[Dict[str, float], Dict]:
    """Tilt the score probability distribution with the O/U tempo signal."""
    if not isinstance(score_probs, dict) or not score_probs:
        return score_probs, {'applied': False, 'reason': 'empty_distribution'}

    signal_info = _total_market_tempo_signal(total)
    if abs(signal_info.get('signal', 0.0)) < 0.12:
        return score_probs, {
            'applied': False,
            'reason': 'weak_tempo_signal',
            'tempo': signal_info,
        }

    parsed = {}
    raw_total = 0.0
    for score, prob in score_probs.items():
        try:
            h, a = map(int, str(score).split('-'))
            value = max(0.0, float(prob or 0.0))
        except (TypeError, ValueError):
            continue
        parsed[score] = (h, a, value)
        raw_total += value

    if raw_total <= 0:
        return score_probs, {'applied': False, 'reason': 'zero_raw_total'}

    expected_before = sum((h + a) * value for h, a, value in parsed.values()) / raw_total
    adjusted = {
        score: value * _score_total_movement_factor(h, a, total)
        for score, (h, a, value) in parsed.items()
    }

    total_prob = sum(adjusted.values())
    if total_prob <= 0:
        return score_probs, {'applied': False, 'reason': 'zero_adjusted_total'}

    adjusted = {score: prob / total_prob for score, prob in adjusted.items()}
    expected_after = 0.0
    for score, prob in adjusted.items():
        h, a = map(int, score.split('-'))
        expected_after += (h + a) * prob

    return adjusted, {
        'applied': True,
        'tempo': signal_info,
        'expected_before': expected_before,
        'expected_after': expected_after,
        'direction': 'over' if signal_info.get('signal', 0.0) > 0 else 'under',
    }


def _anchor_score_candidates_to_1x2(candidates, euro,
                                    strength=SCORE_1X2_MARKET_ANCHOR_STRENGTH):
    """Partially anchor final score marginals to de-vigged closing 1X2 odds.

    Every upstream score transform is free to improve the within-outcome score
    shape.  This guard only limits aggregate H/D/A drift; a 0.75 geometric
    anchor leaves 25% of the team, Asian-market and context signal intact.
    """
    rows = []
    current = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for score, probability in candidates or []:
        try:
            home_goals, away_goals = int(score[0]), int(score[1])
            probability = max(0.0, float(probability))
        except (TypeError, ValueError, IndexError):
            continue
        outcome = 'home' if home_goals > away_goals else ('away' if home_goals < away_goals else 'draw')
        rows.append(((home_goals, away_goals), probability, outcome))
        current[outcome] += probability
    total = sum(current.values())
    if total <= 0 or any(value <= 0 for value in current.values()):
        return candidates, {'applied': False, 'reason': 'incomplete_score_distribution'}
    current = {key: value / total for key, value in current.items()}

    close = (euro or {}).get('close') or {}
    try:
        target = {
            key: max(0.0, float(close.get(key, 0.0)))
            for key in ('home', 'draw', 'away')
        }
    except (TypeError, ValueError):
        return candidates, {'applied': False, 'reason': 'invalid_market_probabilities'}
    target_total = sum(target.values())
    if target_total <= 0 or any(value <= 0 for value in target.values()):
        return candidates, {'applied': False, 'reason': 'missing_market_probabilities'}
    target = {key: value / target_total for key, value in target.items()}

    weight = max(0.0, min(1.0, float(strength)))
    adjusted = [
        (score, probability * (target[outcome] / current[outcome]) ** weight, outcome)
        for score, probability, outcome in rows
    ]
    adjusted_total = sum(probability for _, probability, _ in adjusted)
    if adjusted_total <= 0:
        return candidates, {'applied': False, 'reason': 'zero_adjusted_total'}
    result = sorted(
        ((score, probability / adjusted_total) for score, probability, _ in adjusted),
        key=lambda item: -item[1],
    )
    after = {
        key: sum(probability for (home_goals, away_goals), probability in result
                 if ('home' if home_goals > away_goals else
                     ('away' if home_goals < away_goals else 'draw')) == key)
        for key in ('home', 'draw', 'away')
    }
    return result, {
        'applied': True,
        'strength': weight,
        'before': {key: round(value, 6) for key, value in current.items()},
        'target': {key: round(value, 6) for key, value in target.items()},
        'after': {key: round(value, 6) for key, value in after.items()},
        'source': 'closing_euro_market',
    }


def _anchor_score_candidates_to_goal_mean(candidates, total, max_shift=0.60):
    """Anchor score expected goals to the O/U target while preserving 1X2 mass."""
    rows = []
    for score, prob in candidates or []:
        try:
            h, a = int(score[0]), int(score[1])
            p = max(0.0, float(prob))
        except (TypeError, ValueError, IndexError):
            continue
        outcome = 'H' if h > a else ('D' if h == a else 'A')
        rows.append(((h, a), p, h + a, outcome))
    total_prob = sum(row[1] for row in rows)
    if total_prob <= 0:
        return candidates, {'applied': False, 'reason': 'empty_distribution'}
    rows = [(score, p / total_prob, goals, outcome) for score, p, goals, outcome in rows]
    expected_before = sum(p * goals for _, p, goals, _ in rows)

    try:
        target = float((total or {}).get('implied_total'))
    except (TypeError, ValueError):
        target = None
    if target is None:
        try:
            target = float(get_close_total_line(total or {}))
        except (TypeError, ValueError):
            target = expected_before
    requested_target = target
    # A fixed 0.60-goal cap was too restrictive when exact-score/history
    # calibration collapsed a genuinely high O/U market back near two goals.
    # Permit a larger upward repair only for a clear 3.0+ market; ordinary and
    # low-total matches retain the conservative cap.
    effective_max_shift = max_shift
    if requested_target >= 3.0 and requested_target > expected_before:
        effective_max_shift = max(max_shift, min(1.25, requested_target - expected_before))
    target = max(
        expected_before - max_shift,
        min(expected_before + effective_max_shift, requested_target),
    )
    if abs(target - expected_before) < 0.08:
        return candidates, {
            'applied': False, 'reason': 'already_aligned',
            'target': target, 'expected_before': expected_before,
            'expected_after': expected_before,
        }

    outcome_mass = {}
    for _, p, _, outcome in rows:
        outcome_mass[outcome] = outcome_mass.get(outcome, 0.0) + p

    def tilt(theta):
        raw = [(score, p * math.exp(theta * goals), goals, outcome)
               for score, p, goals, outcome in rows]
        raw_mass = {}
        for _, p, _, outcome in raw:
            raw_mass[outcome] = raw_mass.get(outcome, 0.0) + p
        adjusted = []
        for score, p, goals, outcome in raw:
            scale = outcome_mass[outcome] / max(raw_mass[outcome], 1e-15)
            adjusted.append((score, p * scale, goals))
        mean = sum(p * goals for _, p, goals in adjusted)
        return adjusted, mean

    low, high = -1.5, 1.5
    for _ in range(50):
        mid = (low + high) / 2.0
        _, mean = tilt(mid)
        if mean < target:
            low = mid
        else:
            high = mid
    theta = (low + high) / 2.0
    adjusted, expected_after = tilt(theta)
    result = sorted(((score, p) for score, p, _ in adjusted), key=lambda item: -item[1])
    return result, {
        'applied': True, 'target': target, 'requested_target': requested_target,
        'max_shift': effective_max_shift, 'theta': theta,
        'expected_before': expected_before, 'expected_after': expected_after,
        'preserved_1x2': True,
    }


def _normalize_goal_dist(goal_dist: Dict) -> Dict[int, float]:
    normalized = {}
    if not isinstance(goal_dist, dict):
        return normalized
    for goals, prob in goal_dist.items():
        try:
            key = int(goals)
            value = max(0.0, float(prob or 0.0))
        except (TypeError, ValueError):
            continue
        normalized[key] = normalized.get(key, 0.0) + value
    total_prob = sum(normalized.values())
    if total_prob > 0:
        normalized = {key: value / total_prob for key, value in normalized.items()}
    return normalized


def _implied_total_mean(line: float, p_over: float) -> float:
    """由大小球盘口(line)与 over 概率反解泊松总进球「期望值」。

    关键：盘口线是 over/under 的平衡点（≈中位数），而总进球分布右偏，均值 > 中位数。
    过去把分布均值直接锚到盘口线，系统性压低了期望总进球（405 场实测：模型对 75%
    的比赛预测「小球」，真实 over 率却约 48%）。这里解 P(Poisson(m) > line) = p_over
    得到 skew-aware 的期望 m，作为进球分布的正确锚点。
    """
    try:
        line = float(line)
        p_over = float(p_over)
    except (TypeError, ValueError):
        return None
    if not (0.0 < p_over < 1.0):
        return None
    k = int(math.floor(line))  # over 表示 total >= k+1
    lo, hi = 0.3, 7.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p_le = sum(math.exp(-mid) * mid ** i / math.factorial(i) for i in range(k + 1))
        if (1.0 - p_le) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _anchor_goal_dist_to_total_line(goal_dist: Dict, total: Dict, max_theta: float = 0.60) -> Tuple[Dict[int, float], Dict]:
    """把总进球分布锚定到「市场隐含期望总进球」（而非盘口线本身）。

    优先用 line + over 概率反解 skew-aware 期望（_implied_total_mean）；无 over 概率时
    退回盘口线。用指数倾斜 exp(theta*goals) 并二分求解 theta 使分布期望「命中」目标
    （旧实现 theta=delta/4 且 0.18 死区会严重欠调）。
    """
    normalized = _normalize_goal_dist(goal_dist)
    if not normalized:
        return normalized, {'applied': False, 'reason': 'empty_distribution'}

    line = get_close_total_line(total)
    if line is None:
        return normalized, {'applied': False, 'reason': 'missing_total_line'}

    p_over = (total or {}).get('close_prob', {}).get('over') if isinstance(total, dict) else None
    target = _implied_total_mean(line, p_over)
    if target is None:
        target = float(line)

    expected_before = sum(goals * prob for goals, prob in normalized.items())
    delta = target - expected_before
    if abs(delta) < 0.05:
        return normalized, {
            'applied': False, 'reason': 'already_aligned',
            'line': float(line), 'target': target,
            'expected_before': expected_before, 'expected_after': expected_before,
        }

    # 二分求解 theta 使倾斜后分布期望命中 target（限幅 max_theta 防极端盘口异常）
    lo, hi = -max_theta, max_theta
    for _ in range(40):
        theta = (lo + hi) / 2.0
        tilted = {g: p * math.exp(theta * g) for g, p in normalized.items()}
        s = sum(tilted.values())
        exp_t = sum(g * p / s for g, p in tilted.items()) if s > 0 else expected_before
        if exp_t < target:
            lo = theta
        else:
            hi = theta
    theta = (lo + hi) / 2.0
    tilted = {g: p * math.exp(theta * g) for g, p in normalized.items()}
    total_prob = sum(tilted.values())
    if total_prob <= 0:
        return normalized, {'applied': False, 'reason': 'zero_tilted_total'}

    adjusted = {goals: prob / total_prob for goals, prob in tilted.items()}
    expected_after = sum(goals * prob for goals, prob in adjusted.items())
    return adjusted, {
        'applied': True,
        'line': float(line),
        'target': target,
        'theta': theta,
        'expected_before': expected_before,
        'expected_after': expected_after,
    }


def _goal_over_under_from_line(goal_dist: Dict[int, float], total: Dict) -> Dict[str, float]:
    line = get_close_total_line(total)
    if line is None:
        line = 2.5
    over = sum(prob for goals, prob in goal_dist.items() if goals > line)
    under = sum(prob for goals, prob in goal_dist.items() if goals < line)
    push = max(0.0, 1.0 - over - under)
    return {'over': over, 'under': under, 'push': push, 'line': line}


def _adjust_goal_dist_with_total_movement(goal_dist: Dict[int, float], total: Dict) -> Tuple[Dict[int, float], Dict]:
    """Softly tilt goal-count distribution with O/U line and water movement."""
    normalized = _normalize_goal_dist(goal_dist)
    if not normalized:
        return goal_dist, {'applied': False, 'reason': 'empty_distribution'}

    total = total or {}
    try:
        open_line = float(total.get('open_line'))
        close_line = float(get_close_total_line(total))
    except (TypeError, ValueError):
        open_line = None
        close_line = None

    open_prob = total.get('open_prob') or {}
    close_prob = total.get('close_prob') or {}
    try:
        over_delta = float(close_prob.get('over', 0.0)) - float(open_prob.get('over', 0.0))
    except (TypeError, ValueError):
        over_delta = 0.0

    line_delta = 0.0
    if open_line is not None and close_line is not None:
        line_delta = close_line - open_line

    if abs(line_delta) < 0.01 and abs(over_delta) < 0.015:
        return normalized, {
            'applied': False,
            'reason': 'stable_total_market',
            'line_delta': round(line_delta, 3),
            'over_delta': round(over_delta, 3),
        }

    line_signal = max(-1.0, min(1.0, line_delta / 0.5))
    water_signal = max(-1.0, min(1.0, over_delta / 0.08))
    conflict = line_signal * water_signal < -0.15

    signal = (0.60 * line_signal) + (0.40 * water_signal)
    if conflict:
        signal *= 0.35

    theta = max(-0.10, min(0.10, signal * 0.08))
    if abs(theta) < 0.003:
        return normalized, {
            'applied': False,
            'reason': 'weak_or_conflicted_signal',
            'line_delta': round(line_delta, 3),
            'over_delta': round(over_delta, 3),
            'conflict': conflict,
        }

    expected_before = sum(goals * prob for goals, prob in normalized.items())
    tilted = {goals: prob * math.exp(theta * (goals - expected_before)) for goals, prob in normalized.items()}
    total_prob = sum(tilted.values())
    if total_prob <= 0:
        return normalized, {'applied': False, 'reason': 'zero_tilted_total'}

    adjusted = {goals: prob / total_prob for goals, prob in tilted.items()}
    expected_after = sum(goals * prob for goals, prob in adjusted.items())
    return adjusted, {
        'applied': True,
        'theta': round(theta, 4),
        'line_delta': round(line_delta, 3),
        'over_delta': round(over_delta, 3),
        'direction': 'over' if theta > 0 else 'under',
        'conflict': conflict,
        'expected_before': expected_before,
        'expected_after': expected_after,
    }


def _score_result_code(h: int, a: int) -> str:
    if h > a:
        return 'H'
    if h < a:
        return 'A'
    return 'D'


def _candidate_result_support(candidates, limit: int = 8) -> Dict[str, float]:
    support = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    for item in candidates[:limit]:
        if len(item) == 2 and isinstance(item[0], tuple):
            (h, a), prob = item
        elif len(item) == 3:
            h, a, prob = item
        else:
            continue
        support[_score_result_code(h, a)] += max(0.0, float(prob or 0.0))
    total_prob = sum(support.values())
    if total_prob > 0:
        support = {key: value / total_prob for key, value in support.items()}
    return support


def _adjust_half_full_with_score_context(half_full_time: Dict, candidates, strength: float = 0.35) -> Dict:
    """Softly align half/full-time final direction with the score candidate distribution."""
    if not half_full_time or not isinstance(half_full_time, dict):
        return half_full_time

    support = _candidate_result_support(candidates)
    if not support or max(support.values()) <= 0:
        return half_full_time

    rows = half_full_time.get('probs')
    if not isinstance(rows, list):
        return half_full_time

    adjusted_rows = []
    adjusted_dist = {}
    for row in rows:
        code = row.get('code')
        if not code or len(code) != 2:
            adjusted_rows.append(row)
            continue
        final_result = code[1]
        factor = 1.0 - strength + strength * (support.get(final_result, 0.0) * 3.0)
        factor = max(0.65, min(1.25, factor))
        raw_prob = max(0.0, float(row.get('raw_prob', 0.0))) * factor
        new_row = row.copy()
        new_row['raw_prob'] = raw_prob
        adjusted_rows.append(new_row)
        adjusted_dist[code] = raw_prob

    total_prob = sum(row.get('raw_prob', 0.0) for row in adjusted_rows)
    if total_prob <= 0:
        return half_full_time

    for row in adjusted_rows:
        raw_prob = row.get('raw_prob', 0.0) / total_prob
        row['raw_prob'] = raw_prob
        row['probability'] = round(raw_prob * 100, 1)

    adjusted_rows.sort(key=lambda item: -item.get('raw_prob', 0.0))
    adjusted = half_full_time.copy()
    adjusted['probs'] = adjusted_rows
    adjusted['distribution'] = {
        row['code']: row['raw_prob']
        for row in adjusted_rows
        if row.get('code')
    }
    adjusted['score_context'] = {
        'applied': True,
        'support': support,
        'strength': strength,
    }
    return adjusted


def _adjust_half_full_with_market_context(half_full_time: Dict,
                                          asian: Dict = None,
                                          total: Dict = None,
                                          strength: float = 0.18) -> Dict:
    """Softly align half/full-time paths with handicap depth and total-goal tempo."""
    if not half_full_time or not isinstance(half_full_time, dict):
        return half_full_time

    rows = half_full_time.get('probs')
    if not isinstance(rows, list):
        return half_full_time

    asian = asian or {}
    total = total or {}
    try:
        handicap = float(asian.get('handicap') or 0.0)
    except (TypeError, ValueError):
        handicap = 0.0
    favor = asian.get('favor') or ('home' if handicap > 0 else 'away' if handicap < 0 else 'even')
    tempo_info = _total_market_tempo_signal(total)
    try:
        total_line = float(tempo_info.get('line') if tempo_info.get('line') is not None else get_close_total_line(total))
    except (TypeError, ValueError):
        total_line = 2.5

    tempo = tempo_info.get('signal', 0.0)

    depth = abs(handicap)
    adjusted_rows = []
    for row in rows:
        code = row.get('code')
        if not code or len(code) != 2:
            adjusted_rows.append(row)
            continue

        half_res, full_res = code[0], code[1]
        factor = 1.0

        if tempo > 0:
            if half_res != 'D':
                factor *= 1.0 + strength * 0.55 * tempo
            if code in {'HH', 'AA'}:
                factor *= 1.0 + strength * 0.35 * tempo
            if code == 'DD':
                factor *= 1.0 - strength * 0.45 * tempo
        elif tempo < 0:
            slow = abs(tempo)
            if half_res == 'D':
                factor *= 1.0 + strength * 0.55 * slow
            if code in {'DD', 'HD', 'AD'}:
                factor *= 1.0 + strength * 0.35 * slow
            if code in {'HA', 'AH'}:
                factor *= 1.0 - strength * 0.35 * slow
            if total_line <= 2.25 and half_res == 'D':
                factor *= 1.0 + strength * 0.30 * slow
            if total_line <= 2.0 and half_res != 'D':
                factor *= 1.0 - strength * 0.18 * slow

        if depth >= 1.0 and favor in {'home', 'away'}:
            fav_res = 'H' if favor == 'home' else 'A'
            if code == f'{fav_res}{fav_res}':
                factor *= 1.0 + strength * min(1.0, depth / 1.5)
            elif full_res != fav_res and half_res != 'D':
                factor *= 1.0 - strength * 0.45
        elif depth <= 0.25:
            if half_res == 'D':
                factor *= 1.0 + strength * 0.35
            if full_res == 'D':
                factor *= 1.0 + strength * 0.25

        new_row = row.copy()
        new_row['raw_prob'] = max(0.0, float(row.get('raw_prob', 0.0))) * max(0.65, min(1.35, factor))
        adjusted_rows.append(new_row)

    total_prob = sum(row.get('raw_prob', 0.0) for row in adjusted_rows)
    if total_prob <= 0:
        return half_full_time

    for row in adjusted_rows:
        raw_prob = row.get('raw_prob', 0.0) / total_prob
        row['raw_prob'] = raw_prob
        row['probability'] = round(raw_prob * 100, 1)
    adjusted_rows.sort(key=lambda item: -item.get('raw_prob', 0.0))

    adjusted = half_full_time.copy()
    adjusted['probs'] = adjusted_rows
    adjusted['distribution'] = {
        row['code']: row['raw_prob']
        for row in adjusted_rows
        if row.get('code')
    }
    adjusted['market_context'] = {
        'applied': True,
        'tempo': round(tempo, 3),
        'tempo_source': tempo_info,
        'handicap': handicap,
        'favor': favor,
        'strength': strength,
    }
    return adjusted


def _assess_market_data_quality(asian: Dict, euro: Dict, total: Dict) -> Dict:
    score = 1.0
    reasons = []

    if asian.get('handicap') is None:
        score -= 0.25
        reasons.append('missing_asian_handicap')
    if total.get('close_line') is None:
        score -= 0.25
        reasons.append('missing_total_line')
    if not euro.get('close') or not all(k in euro.get('close', {}) for k in ('home', 'draw', 'away')):
        score -= 0.25
        reasons.append('missing_euro_close')

    for market_name, market, keys in (
        ('asian', asian, ('open_prob', 'close_prob')),
        ('total', total, ('open_prob', 'close_prob')),
    ):
        for key in keys:
            if not market.get(key):
                score -= 0.08
                reasons.append(f'missing_{market_name}_{key}')

    try:
        sup_a = float(asian.get('implied_supremacy', 0.0))
        sup_e = float(euro.get('implied_supremacy', 0.0))
        if sup_a * sup_e < 0:
            score -= 0.18
            reasons.append('asian_euro_direction_conflict')
        elif abs(sup_a - sup_e) >= SUPREMACY_CONFLICT_GAP:
            score -= 0.10
            reasons.append('asian_euro_supremacy_gap')
    except Exception:
        pass

    score = max(0.0, min(1.0, score))
    if score >= 0.85:
        grade = 'high'
        weight_factor = 1.0
    elif score >= 0.62:
        grade = 'medium'
        weight_factor = 0.75
    elif score >= 0.40:
        grade = 'low'
        weight_factor = 0.45
    else:
        grade = 'reject'
        weight_factor = 0.0

    return {
        'score': round(score, 3),
        'grade': grade,
        'weight_factor': weight_factor,
        'reasons': reasons,
    }


def _pick_recommendations(candidates, asian, euro, total, n=2, pool=12, confidence=None, league_profile=None, team=None, similar_market=None):
    """Top 池内按 概率×一致性×冷热×置信度×相似盘口 选取（基于比分簇）"""
    if confidence:
        n = confidence.get('recommend_count', n)
    pool = min(16, len(candidates))  # 原12，扩大候选池让高比分有更多入选机会
    conf_w = confidence['score'] if confidence else 1.0
    
    # 价值投注列表（单独输出，不参与命中率排序）
    value_bets = []

    # xG 一致性校验：ELO xG 与市场总进球线偏差 >0.5 则降低置信度
    xg_penalty = 1.0
    xg_total = total.get('xg_total')
    total_line = get_close_total_line(total)
    if xg_total is not None and total_line is not None:
        xg_deviation = abs(xg_total - total_line)
        if xg_deviation > 0.5:
            xg_penalty = max(0.75, 1.0 - (xg_deviation - 0.5) * 0.20)  # 偏差>0.5开始渐进扣分

    # 判断球队实力差距：通过亚盘让球判断
    handicap = abs(asian.get('handicap', 0))
    is_clear_favorite = handicap >= 1.0  # 让球>=1球视为强弱分明
    
    # 动态评估爆冷风险
    upset_risk = _evaluate_upset_risk(asian, euro, team)
    
    # 相似盘口比分权重（融合历史数据）
    similar_weight = {}
    similar_confidence = 0.0
    if similar_market and similar_market.get('goals_dist') and similar_market.get('confidence', 0) >= 0.3:
        similar_confidence = similar_market['confidence']
        for score, prob in similar_market['goals_dist'].items():
            h, a = map(int, score.split('-'))
            similar_weight[(h, a)] = prob

    # 盘口聚类先验（与候选比分无关，循环外只计算一次）
    market_prior = {}
    if MARKET_CLUSTERING_AVAILABLE and asian.get('handicap') is not None and total.get('close_line') is not None:
        try:
            from .market_clustering import get_market_prior
            market_prior = get_market_prior(asian['handicap'], total.get('close_line', total.get('line', 2.5))) or {}
        except Exception as e:
            log.debug(f"盘口聚类先验获取失败: {e}")

    scored = []
    favor = asian.get('favor', 'home')
    
    for (h, a), prob in candidates[:pool]:
        align = _alignment_score(h, a, asian, euro, total)
        heat, _ = score_heat_label(h, a, prob, league_profile)
        w = _heat_filter_weight(heat)
        
        # 检查是否是冷门
        diff = h - a
        is_upset = False
        if favor == 'home' and diff < 0:
            is_upset = True  # 主队让球但客队赢
        elif favor == 'away' and diff > 0:
            is_upset = True  # 客队让球但主队赢
        
        # 根据爆冷风险动态调整冷门比分权重
        upset_penalty = 1.0
        if is_upset:
            if is_clear_favorite and upset_risk < 0.3:
                upset_penalty = 0.4
            elif is_clear_favorite and upset_risk < 0.5:
                upset_penalty = 0.7
        
        # 融合相似盘口比分权重
        market_bonus = 1.0
        if (h, a) in similar_weight and similar_confidence > 0:
            market_bonus = 1.0 + similar_weight[(h, a)] * similar_confidence * 0.5

        # 盘口聚类先验权重
        prior_bonus = 1.0
        if MARKET_CLUSTERING_AVAILABLE and asian.get('handicap') is not None and total.get('close_line') is not None:
            try:
                from .market_clustering import get_market_prior
                prior = get_market_prior(asian['handicap'], total.get('close_line', total.get('line', 2.5)))
                if prior:
                    score_key = f"{h}-{a}"
                    prior_prob = prior.get(score_key, 0.0)
                    if prior_prob > 0:
                        prior_bonus = 1.0 + prior_prob * 0.3
            except Exception as e:
                log.debug(f"盘口聚类先验获取失败: {e}")

        # 赔率价值计算（仅记录，不参与命中率排序）
        value_info = None
        if VALUE_BETTING_AVAILABLE and euro.get('raw_odds', {}).get('close'):
            try:
                from .value_betting import calculate_value, calculate_ev
                close_odds = euro['raw_odds']['close']
                score_odds = _estimate_score_odds(h, a, close_odds)
                if score_odds > 1.0:
                    value = calculate_value(prob, score_odds)
                    ev = calculate_ev(prob, score_odds)
                    value_info = {
                        'score': f"{h}-{a}",
                        'value': value,
                        'ev': ev,
                        'odds': score_odds,
                        'probability': prob
                    }
            except Exception as e:
                log.debug(f"赔率价值计算失败: {e}")

        # 计算最终得分（不含价值投注调整，价值投注单独输出）
        total_line_factor = _score_total_line_factor(h, a, total_line)
        common_score_factor = _common_score_overheat_factor(h, a, prob, total_line)
        final_score = (
            prob
            * (1.0 + 0.45 * align)
            * w
            * (0.65 + 0.35 * conf_w)
            * xg_penalty
            * upset_penalty
            * market_bonus
            * prior_bonus
            * total_line_factor
            * common_score_factor
        )
        
        # 记录价值信息
        if value_info:
            value_bets.append(value_info)
        cluster = _get_score_cluster(h, a)
        scored.append(((h, a), prob, align, heat, cluster, final_score))

    scored.sort(key=lambda x: -x[5])
    
    # ========== 比分簇推荐策略 ==========
    # 核心比分：概率最高的比分所在簇
    # 保护比分：同簇内的其他高概率比分 + 邻近簇的比分
    # 冷门覆盖：对立簇的一个比分（如果爆冷风险足够）

    seen = set()
    picked = []
    picked_clusters = set()
    upset_count = 0
    
    # 确定最大冷门数量
    max_upsets = 1 if upset_risk >= 0.3 else 0
    if is_clear_favorite and upset_risk < 0.3:
        max_upsets = 0
    
    # 阶段1：选择核心比分（概率最高的）
    if scored:
        (h, a), prob, _, _, cluster, _ = scored[0]
        pattern = score_pattern(h, a)
        picked.append((h, a, prob, cluster, pattern, 'core'))
        seen.add((h, a))
        picked_clusters.add(cluster)
        picked_patterns = {pattern}
    else:
        picked_patterns = set()

    # 阶段2：选择保护比分（优先覆盖不同剧本模式）
    for (h, a), prob, _, _, cluster, _ in scored[1:]:
        if (h, a) in seen:
            continue
        if len(picked) >= n:
            break

        # 获取当前比分的剧本模式
        pattern = score_pattern(h, a)

        # 检查是否是冷门
        diff = h - a
        is_upset_pick = False
        if favor == 'home' and diff < 0:
            is_upset_pick = True
        elif favor == 'away' and diff > 0:
            is_upset_pick = True
        
        # 冷门限制
        if is_upset_pick:
            if upset_count >= max_upsets:
                continue
            upset_count += 1
            picked.append((h, a, prob, cluster, pattern, 'upset'))
            seen.add((h, a))
            picked_clusters.add(cluster)
            picked_patterns.add(pattern)
            continue

        # 优先选择不同剧本模式的比分
        if pattern not in picked_patterns:
            picked.append((h, a, prob, cluster, pattern, 'protection'))
            seen.add((h, a))
            picked_clusters.add(cluster)
            picked_patterns.add(pattern)
        elif cluster in picked_clusters:
            # 同簇的高概率比分作为保护
            picked.append((h, a, prob, cluster, pattern, 'protection'))
            seen.add((h, a))
        elif len(picked) < n and cluster not in picked_clusters:
            # 添加邻近簇作为分散保护
            picked.append((h, a, prob, cluster, pattern, 'protection'))
            seen.add((h, a))
            picked_clusters.add(cluster)
            picked_patterns.add(pattern)

    # 阶段3：补充剩余推荐（如果还不够）
    if len(picked) < n:
        for (h, a), prob, _, _, cluster, _ in scored:
            if (h, a) in seen:
                continue
            if len(picked) >= n:
                break

            pattern = score_pattern(h, a)
            
            diff = h - a
            is_upset_pick = False
            if favor == 'home' and diff < 0:
                is_upset_pick = True
            elif favor == 'away' and diff > 0:
                is_upset_pick = True

            if is_upset_pick and upset_count >= max_upsets:
                continue

            if is_upset_pick:
                upset_count += 1
                picked.append((h, a, prob, cluster, pattern, 'upset'))
            else:
                picked.append((h, a, prob, cluster, pattern, 'protection'))
            seen.add((h, a))
            picked_patterns.add(pattern)

    # 转换为原有格式 (h, a, prob)
    # 返回推荐列表和价值投注列表（分开输出）
    picked = _diversify_score_recommendations(picked, scored, n, favor, upset_count, max_upsets)
    recommendations = [(h, a, prob) for h, a, prob, _, _, _ in picked]
    
    # 对价值投注按 EV 排序
    value_bets.sort(key=lambda x: -x.get('ev', 0))
    
    return recommendations, value_bets


def _diversify_score_recommendations(picked, scored, n: int, favor: str, upset_count: int, max_upsets: int):
    """Avoid returning recommendations that all tell the same score story."""
    if len(picked) < 3:
        return picked

    def result_code(h, a):
        return 'H' if h > a else 'A' if h < a else 'D'

    def goal_band(h, a):
        goals = h + a
        if goals <= 1:
            return 'low'
        if goals <= 3:
            return 'mid'
        return 'high'

    def replace_last_matching(items, predicate, replacement):
        diversified = items[:]
        for idx in range(len(diversified) - 1, 0, -1):
            if predicate(diversified[idx]):
                diversified[idx] = replacement
                return diversified[:n]
        return items

    pattern_counts = {}
    for _, _, _, _, pattern, _ in picked:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    seen = {(h, a) for h, a, *_ in picked}
    min_prob = min(prob for _, _, prob, *_ in picked)

    overloaded_pattern = next((pattern for pattern, count in pattern_counts.items() if count >= 3), None)
    picked_results = {result_code(h, a) for h, a, *_ in picked}
    picked_bands = {goal_band(h, a) for h, a, *_ in picked}

    target = None
    if overloaded_pattern:
        target = ('pattern', overloaded_pattern)
    elif len(picked_results) == 1:
        target = ('result', next(iter(picked_results)))
    elif len(picked_bands) == 1:
        target = ('goal_band', next(iter(picked_bands)))

    if not target:
        return picked

    for (h, a), prob, _, _, cluster, _ in scored:
        if (h, a) in seen:
            continue
        if prob < min_prob * 0.65:
            continue
        pattern = score_pattern(h, a)
        candidate_result = result_code(h, a)
        candidate_band = goal_band(h, a)
        if target[0] == 'pattern' and pattern == target[1]:
            continue
        if target[0] == 'result' and candidate_result == target[1]:
            continue
        if target[0] == 'goal_band' and candidate_band == target[1]:
            continue

        is_upset = (favor == 'home' and h < a) or (favor == 'away' and h > a)
        if is_upset and upset_count >= max_upsets:
            continue
        replacement = (h, a, prob, cluster, pattern, 'diversity')
        if target[0] == 'pattern':
            return replace_last_matching(picked, lambda item: item[4] == target[1], replacement)
        if target[0] == 'result':
            return replace_last_matching(picked, lambda item: result_code(item[0], item[1]) == target[1], replacement)
        if target[0] == 'goal_band':
            return replace_last_matching(picked, lambda item: goal_band(item[0], item[1]) == target[1], replacement)
        break

    return picked


