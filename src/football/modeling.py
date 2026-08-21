# -*- coding: utf-8 -*-
"""足球比分建模：泊松/负二项/贝叶斯MCMC/λ拟合/比分矩阵/风险评估"""

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

from .config import (
    AVG_LEAGUE_GOAL, CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD, FIT_W_1X2, FIT_W_OU_DIST, FIT_W_SUPREMACY, FIT_W_TEAM, FIT_W_TOTAL, HANDICAP_CHANGE_LAMBDA_BOOST, HOME_VENUE_ATTACK_BOOST, LAMBDA_COARSE_STEP, LAMBDA_FINE_RADIUS, LAMBDA_FINE_STEP, LAMBDA_REFINE_STEP0, LAMBDA_REFINE_STEPS, LAMBDA_WEIGHT_ELO, LAMBDA_WEIGHT_MARKET, LAMBDA_WEIGHT_TEAM, LEAGUE_PROFILES, MAX_GOALS, SUPREMACY_CONFLICT_GAP, SUP_ASIAN_WEIGHT, SUP_EURO_WEIGHT, TOTAL_LINE_CHANGE_LAMBDA_BOOST,
)
from .parsing import (
    _blend_close_open,
)
from .markets import (
    _poisson_pmf,
)

def _negative_binomial_pmf(k, r, p):
    """
    负二项分布概率质量函数 P(X=k)。
    参数：
        k: 成功次数
        r: 失败次数（形状参数）
        p: 每次试验成功概率
    
    负二项分布适合过离散数据（方差 > 期望）
    期望 = r * (1-p) / p
    方差 = r * (1-p) / p^2 = 期望 * (1/p)
    """
    if k < 0 or r <= 0 or p <= 0 or p >= 1:
        return 0.0
    
    # 使用对数计算避免数值溢出
    log_prob = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) +
        r * math.log(p) + k * math.log(1 - p)
    )
    return math.exp(log_prob)


def _nb_params_from_mean_var(mean, var):
    """
    由均值和方差估计负二项分布参数 r 和 p。
    当 var > mean 时（过离散），负二项分布更合适。
    """
    if var <= mean:
        # 接近泊松分布，返回一个近似泊松的负二项
        r = 1000.0
        p = r / (r + mean)
        return r, p
    
    # 形状参数 r
    r = (mean ** 2) / (var - mean)
    # 成功概率 p
    p = r / (r + mean)
    return r, p


def _estimate_nb_overdispersion(league_profile=None):
    """
    估计联赛的进球过离散程度。
    根据历史数据，足球比赛进球的方差通常是均值的 1.3-2.0 倍。
    """
    if league_profile:
        # 不同联赛有不同的过离散程度
        league_overdispersion = {
            '英超': 1.35, '英冠': 1.28, '西甲': 1.25, '意甲': 1.32,
            '德甲': 1.38, '法甲': 1.25, '荷甲': 1.42, '葡超': 1.28,
            '欧冠': 1.22, '欧联': 1.25, '世界杯': 1.18, '欧洲杯': 1.20,
            '中超': 1.30, '日职': 1.25, '韩K': 1.28,
        }
        league_name = league_profile.get('name', '')
        return league_overdispersion.get(league_name, 1.22)
    return 1.22  # 原1.45，降低过离散使分布更紧凑，减少0球堆积概率


def _build_residual_features(asian, euro, total, team, league_profile):
    """
    构建残差学习的特征向量。
    输入：赔率变化、球队实力差、战意、伤停等。
    """
    features = []
    
    # 赔率变化特征
    features.append(euro['close']['home'] - euro['open']['home'])  # 主胜概率变化
    features.append(euro['close']['draw'] - euro['open']['draw'])  # 平局概率变化
    features.append(euro['close']['away'] - euro['open']['away'])  # 客胜概率变化
    
    # 亚盘特征
    features.append(asian['handicap'])  # 让球盘
    # 根据让球方向获取正确的概率值计算变化
    if asian['handicap'] > 0:
        close_hp = asian['close_prob'].get('home_give', asian['close_prob'].get('home', 0.5))
        open_hp = asian['open_prob'].get('home_give', asian['open_prob'].get('home', 0.5))
    elif asian['handicap'] < 0:
        close_hp = asian['close_prob'].get('home_recv', asian['close_prob'].get('home', 0.5))
        open_hp = asian['open_prob'].get('home_recv', asian['open_prob'].get('home', 0.5))
    else:
        close_hp = asian['close_prob'].get('home', 0.5)
        open_hp = asian['open_prob'].get('home', 0.5)
    features.append(close_hp - open_hp)  # 主队方概率变化
    
    # 大小球特征
    features.append(total['close_line'])  # 大小球盘口
    features.append(total['close_prob']['over'] - total['open_prob']['over'])  # 大球概率变化
    
    # 球队实力特征
    if team:
        features.append(team.get('attack_home', 0))
        features.append(team.get('defense_home', 0))
        features.append(team.get('attack_away', 0))
        features.append(team.get('defense_away', 0))
        features.append(team.get('form_home', 0))
        features.append(team.get('form_away', 0))
    else:
        features.extend([0] * 6)
    
    # 联赛特征
    if league_profile:
        features.append(league_profile.get('avg_goal', 1.4))
        features.append(league_profile.get('draw_rate', 0.25))
    else:
        features.extend([1.4, 0.25])
    
    # 欧赔-亚盘分歧特征
    features.append(abs(euro.get('implied_supremacy', 0) - asian.get('implied_supremacy', 0)))
    
    return features


def _train_residual_model(X_train, y_train):
    """
    训练残差学习的 LightGBM 模型。
    目标：真实比分概率 - 基础泊松概率（残差）。
    
    返回：训练好的模型（如果有足够数据），否则返回 None
    """
    if len(X_train) < 100:
        log.warning("训练数据不足，跳过残差模型训练")
        return None
    
    try:
        import lightgbm as lgb
        
        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 参数设置
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'random_state': 42,
        }
        
        # 训练模型
        model = lgb.train(params, train_data, num_boost_round=100)
        
        return model
    except ImportError:
        log.warning("LightGBM 未安装，跳过残差模型")
        return None
    except Exception as e:
        log.error(f"残差模型训练失败: {e}")
        return None


def apply_residual_correction(base_matrix, features, residual_model=None):
    """
    应用残差修正。
    最终概率 = p_base * weight + residual_boost
    
    参数：
        base_matrix: 基础泊松模型输出的比分矩阵
        features: 当前比赛的特征向量
        residual_model: 训练好的残差模型
    
    返回：修正后的比分矩阵
    """
    if residual_model is None:
        return base_matrix
    
    try:
        # 对每个比分计算残差预测
        corrected_matrix = {}
        total_residual = 0.0
        
        for (h, a), prob in base_matrix.items():
            # 使用基础概率和特征预测残差
            # 简化处理：使用比分相关特征
            score_features = features.copy()
            score_features.append(h)
            score_features.append(a)
            score_features.append(h + a)
            score_features.append(h - a)
            
            # 预测残差
            residual = float(residual_model.predict([score_features])[0])
            
            # 应用残差修正（限制范围避免概率异常）
            corrected_prob = prob + residual * 0.1  # 残差权重
            corrected_prob = max(0.001, min(0.999, corrected_prob))
            
            corrected_matrix[(h, a)] = corrected_prob
            total_residual += abs(residual)
        
        # 归一化
        total = sum(corrected_matrix.values())
        if total > 0:
            corrected_matrix = {k: v / total for k, v in corrected_matrix.items()}
        
        return corrected_matrix
    except Exception as e:
        log.error(f"残差修正应用失败: {e}")
        return base_matrix


def _train_draw_calibration_model(X_train, y_train):
    """
    训练平局概率校准的逻辑回归子模型。
    
    输入特征：
        - p_draw_euro: 欧赔平局概率
        - handicap_abs: 亚盘让球绝对值
        - home_draw_rate: 主队近10场平局率
        - away_draw_rate: 客队近10场平局率
        - league_draw_rate: 联赛平均平局率
    
    返回：训练好的模型（如果有足够数据）
    """
    if len(X_train) < 50:
        log.warning("平局校准训练数据不足")
        return None
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
        model.fit(X_scaled, y_train)
        
        return model, scaler
    except ImportError:
        log.warning("scikit-learn 未安装，跳过平局校准")
        return None, None
    except Exception as e:
        log.error(f"平局校准模型训练失败: {e}")
        return None, None


def calibrate_draw_probability(p_home, p_draw, p_away, asian_handicap, 
                               home_draw_rate=0.25, away_draw_rate=0.25, 
                               league_draw_rate=0.25, draw_model=None, scaler=None):
    """
    校准平局概率。
    
    参数：
        p_home, p_draw, p_away: 原始 1X2 概率
        asian_handicap: 亚盘让球（绝对值）
        home_draw_rate: 主队近10场平局率
        away_draw_rate: 客队近10场平局率
        league_draw_rate: 联赛平均平局率
        draw_model: 训练好的平局校准模型
    
    返回：校准后的 (p_home, p_draw, p_away)
    """
    if draw_model is None or scaler is None:
        # 没有训练好的模型，使用启发式校准
        return _heuristic_draw_calibration(p_home, p_draw, p_away, asian_handicap, 
                                           home_draw_rate, away_draw_rate, league_draw_rate)
    
    try:
        # 构建特征向量
        features = [
            p_draw,
            abs(asian_handicap),
            home_draw_rate,
            away_draw_rate,
            league_draw_rate,
        ]
        
        # 预测平局概率的修正系数
        X_scaled = scaler.transform([features])
        draw_prob = float(draw_model.predict_proba(X_scaled)[0][1])
        
        # 重新分配概率
        total_non_draw = p_home + p_away
        if total_non_draw > 0:
            p_home_new = p_home / total_non_draw * (1 - draw_prob)
            p_away_new = p_away / total_non_draw * (1 - draw_prob)
            p_draw_new = draw_prob
        else:
            p_home_new, p_draw_new, p_away_new = p_home, p_draw, p_away
        
        return p_home_new, p_draw_new, p_away_new
    except Exception as e:
        log.error(f"平局校准应用失败: {e}")
        return p_home, p_draw, p_away


def _draw_probability_bounds(asian_handicap) -> Tuple[float, float]:
    handicap_abs = abs(float(asian_handicap or 0.0))
    if handicap_abs <= 0.25:
        return 0.08, 0.42
    if handicap_abs <= 0.75:
        return 0.07, 0.36
    if handicap_abs <= 1.25:
        return 0.06, 0.31
    return 0.05, 0.27


def _redistribute_draw_probability(p_home, p_draw, p_away, asian_handicap) -> Tuple[float, float, float]:
    min_draw, max_draw = _draw_probability_bounds(asian_handicap)
    p_draw_new = max(min_draw, min(max_draw, p_draw))
    non_draw_total = max(p_home + p_away, 1e-12)
    p_home_new = p_home / non_draw_total * (1 - p_draw_new)
    p_away_new = p_away / non_draw_total * (1 - p_draw_new)
    total = p_home_new + p_draw_new + p_away_new
    if total > 0:
        return p_home_new / total, p_draw_new / total, p_away_new / total
    return p_home, p_draw, p_away


def _heuristic_draw_calibration(p_home, p_draw, p_away, asian_handicap, 
                                home_draw_rate, away_draw_rate, league_draw_rate):
    """
    启发式平局校准。
    
    当让球较小时（双方实力接近），平局概率可能被低估。
    根据球队平局历史和联赛平均进行调整。
    """
    # 让球绝对值越小，平局可能性越大
    handicap_abs = abs(asian_handicap)
    
    # 基础平局倾向
    draw_tendency = (home_draw_rate + away_draw_rate) / 2
    
    # 让球调整因子：让球越小，越倾向平局
    if handicap_abs < 0.5:
        # 平手或低水让球，平局概率可能偏低
        adjustment = 1.2 + (league_draw_rate - 0.25) * 2
    elif handicap_abs < 1.0:
        adjustment = 1.1 + (league_draw_rate - 0.25) * 1.5
    else:
        adjustment = 1.0
    
    # 应用调整
    p_draw_new = p_draw * adjustment * draw_tendency / 0.25

    # 重新归一化并按让球分档夹取平局边界
    return _redistribute_draw_probability(p_home, p_draw_new, p_away, asian_handicap)


def _gamma_prior_params(league_profile=None, team_strength=None):
    """
    构建 λ 的 Gamma 先验分布参数（形状参数 α, 尺度参数 β）。
    Gamma(α, β) 的期望为 α/β，方差为 α/β²。
    
    先验信息来源：
    1. 联赛赛季均值（动态更新）
    2. 球队攻防强度作为超参数
    """
    # 默认联赛均值
    default_mean = 1.4
    default_std = 0.5
    
    if league_profile:
        mean_goal = league_profile.get('avg_goal', default_mean)
    else:
        mean_goal = default_mean
    
    # 整合球队实力信息
    if team_strength:
        attack_strength = (team_strength.get('attack_home', 0) + team_strength.get('attack_away', 0)) / 2
        # 球队实力调整均值
        mean_goal = mean_goal * (1 + (attack_strength - mean_goal) * 0.3)
    
    # Gamma 参数：α = (mean/std)^2, β = mean/std^2
    std = default_std
    alpha = (mean_goal / std) ** 2
    beta = mean_goal / (std ** 2)
    
    return max(0.1, alpha), max(0.01, beta)


def _rho_prior_params():
    """
    DC 相关系数 rho 的 Beta 先验参数。
    根据历史数据，rho 通常在 [-0.2, 0.1] 之间，均值接近 -0.05。
    使用 Beta(2, 5) 近似这个分布（均值 ≈ 0.28，转换到 [-0.5, 0.5] 区间后 ≈ -0.06）
    """
    return 2.0, 5.0  # alpha, beta


def _log_posterior(lam_home, lam_away, rho, targets, target_total, supremacy, 
                   prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a):
    """
    计算对数后验概率（不包含归一化常数）。
    
    后验 ∝ 先验 × 似然
    先验：Gamma(α, β) 用于 λ，Beta 用于 rho（转换到 [-0.5, 0.5]）
    似然：泊松-DC 模型拟合欧赔目标
    """
    if lam_home <= 0 or lam_away <= 0 or rho < -0.5 or rho > 0.5:
        return float('-inf')
    
    # 先验对数概率
    # Gamma 先验: p(λ) ∝ λ^(α-1) * exp(-βλ)
    log_prior_h = (prior_alpha_h - 1) * math.log(lam_home) - prior_beta_h * lam_home
    log_prior_a = (prior_alpha_a - 1) * math.log(lam_away) - prior_beta_a * lam_away
    
    # rho 的 Beta 先验（转换到 [-0.5, 0.5]）
    rho_transformed = (rho + 0.5)  # [-0.5, 0.5] -> [0, 1]
    rho_alpha, rho_beta = _rho_prior_params()
    log_prior_rho = (rho_alpha - 1) * math.log(rho_transformed) + (rho_beta - 1) * math.log(1 - rho_transformed)
    
    # 似然：拟合误差的负对数（作为似然的代理）
    matrix = build_score_matrix(lam_home, lam_away, rho=rho)
    margins = _matrix_margins(matrix)
    
    # 拟合误差（越小越好，所以取负）
    err = (
        100 * sum((margins[k] - targets[i]) ** 2 for i, k in enumerate(('home', 'draw', 'away')))
        + 10 * (lam_home + lam_away - target_total) ** 2
        + 5 * (lam_home - lam_away - supremacy) ** 2
    )
    
    log_likelihood = -err
    
    return log_prior_h + log_prior_a + log_prior_rho + log_likelihood


def _mcmc_sample_lambdas(targets, target_total, supremacy, league_profile=None, team_strength=None,
                         n_samples=2000, burn_in=500, step_size=0.05):
    """
    使用 Metropolis-Hastings 算法采样 λ_home, λ_away, rho 的后验分布。
    
    返回：采样结果列表，包含 (lam_home, lam_away, rho, log_prob)
    """
    # 获取先验参数
    prior_alpha_h, prior_beta_h = _gamma_prior_params(league_profile, team_strength)
    prior_alpha_a, prior_beta_a = _gamma_prior_params(league_profile, team_strength)
    
    # 初始化（使用最大似然估计作为初始点）
    lam_h_start = max(0.1, (target_total + supremacy) / 2)
    lam_a_start = max(0.1, (target_total - supremacy) / 2)
    rho_start = 0.0
    
    current = (lam_h_start, lam_a_start, rho_start)
    current_log_prob = _log_posterior(*current, targets, target_total, supremacy,
                                      prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a)
    
    samples = []
    accepted = 0
    
    for i in range(n_samples):
        # 提议新值
        lam_h_new = max(0.01, current[0] + (random.random() - 0.5) * step_size * 2)
        lam_a_new = max(0.01, current[1] + (random.random() - 0.5) * step_size * 2)
        rho_new = max(-0.5, min(0.5, current[2] + (random.random() - 0.5) * 0.02))
        
        new = (lam_h_new, lam_a_new, rho_new)
        new_log_prob = _log_posterior(*new, targets, target_total, supremacy,
                                      prior_alpha_h, prior_beta_h, prior_alpha_a, prior_beta_a)
        
        # Metropolis-Hastings 接受准则
        if new_log_prob > current_log_prob or random.random() < math.exp(new_log_prob - current_log_prob):
            current = new
            current_log_prob = new_log_prob
            accepted += 1
        
        # 收集样本（跳过 burn-in 期）
        if i >= burn_in:
            samples.append((current[0], current[1], current[2], current_log_prob))
    
    acceptance_rate = accepted / n_samples
    log.debug(f"MCMC 采样完成，接受率: {acceptance_rate:.3f}, 样本数: {len(samples)}")
    
    return samples


def bayesian_predict_scores(targets, target_total, supremacy, league_profile=None, team_strength=None):
    """
    贝叶斯框架下的比分概率预测。
    
    返回：
        mean_matrix: 后验均值比分矩阵
        credible_interval: 关键参数的置信区间
        samples: 原始采样结果（用于进一步分析）
    """
    samples = _mcmc_sample_lambdas(targets, target_total, supremacy, league_profile, team_strength)
    
    if not samples:
        # 采样失败，返回点估计
        lam_h = max(0.1, (target_total + supremacy) / 2)
        lam_a = max(0.1, (target_total - supremacy) / 2)
        return build_score_matrix(lam_h, lam_a, rho=0.0), None, None
    
    # 计算后验均值
    n_samples = len(samples)
    mean_lam_h = sum(s[0] for s in samples) / n_samples
    mean_lam_a = sum(s[1] for s in samples) / n_samples
    mean_rho = sum(s[2] for s in samples) / n_samples
    
    # 计算置信区间（95%）
    lh_values = sorted(s[0] for s in samples)
    la_values = sorted(s[1] for s in samples)
    rho_values = sorted(s[2] for s in samples)
    
    credible_interval = {
        'lam_home': (lh_values[int(0.025 * n_samples)], lh_values[int(0.975 * n_samples)]),
        'lam_away': (la_values[int(0.025 * n_samples)], la_values[int(0.975 * n_samples)]),
        'rho': (rho_values[int(0.025 * n_samples)], rho_values[int(0.975 * n_samples)]),
        'total': (mean_lam_h + mean_lam_a, 
                 lh_values[int(0.025 * n_samples)] + la_values[int(0.025 * n_samples)],
                 lh_values[int(0.975 * n_samples)] + la_values[int(0.975 * n_samples)]),
    }
    
    # 构建后验均值矩阵
    mean_matrix = build_score_matrix(mean_lam_h, mean_lam_a, rho=mean_rho)
    
    return mean_matrix, credible_interval, samples


def _outcome(h, a):
    return 'home' if h > a else 'draw' if h == a else 'away'


def market_implied_lambdas(handicap, total_line):
    """
    由盘口直接反推 λ（核心改进）
    
    公式：
        home_lambda = (total_line + handicap) / 2
        away_lambda = (total_line - handicap) / 2
    
    例如：主让1.0，大小球3.0 → home=2.0, away=1.0
    
    参数：
        handicap: 亚盘让球（主队让球为正）
        total_line: 大小球盘口线
    
    返回：
        (lam_home, lam_away)
    """
    lam_home = max(0.08, (total_line + handicap) / 2)
    lam_away = max(0.08, (total_line - handicap) / 2)
    return lam_home, lam_away


def _parse_time(time_str):
    """解析时间字符串为分钟数"""
    if not time_str:
        return None
    try:
        # 格式：06-09 17:20
        parts = time_str.strip().split()
        if len(parts) != 2:
            return None
        date_part = parts[0]
        time_part = parts[1]
        
        month, day = map(int, date_part.split('-'))
        hour, minute = map(int, time_part.split(':'))
        
        # 转换为分钟数（假设在同一个月内）
        return day * 24 * 60 + hour * 60 + minute
    except:
        return None


def apply_handicap_change_adjustment(lam_home, lam_away, open_handicap, close_handicap, 
                                     open_time=None, close_time=None):
    """
    应用亚盘升降盘对 λ 的修正（包含时间因素）
    
    例如：
        初盘主让0.5 → 终盘主让1.0 → 主队被看好
        lambda_home += 0.15, lambda_away -= 0.05
        
    时间因素：
        相同变化幅度下，越临近比赛的变化越有价值
        变化速度越快（单位时间变化量大），信号越强
    
    参数：
        lam_home, lam_away: 当前 λ 值
        open_handicap: 初盘让球
        close_handicap: 终盘让球
        open_time: 初盘时间（格式：06-09 17:20）
        close_time: 终盘时间（格式：06-09 17:20）
    
    返回：
        (adjusted_lam_home, adjusted_lam_away)
    """
    if open_handicap is None or close_handicap is None:
        return lam_home, lam_away
    
    # 让球变化 = 终盘 - 初盘
    # 正数 = 主队让球增加（主队被看好）
    # 负数 = 主队让球减少（客队被看好）
    handicap_change = close_handicap - open_handicap
    
    # 计算时间权重
    # 变化发生得越晚（越临近比赛），权重越高
    time_weight = 1.0
    if open_time and close_time:
        open_minutes = _parse_time(open_time)
        close_minutes = _parse_time(close_time)
        if open_minutes and close_minutes and open_minutes < close_minutes:
            # 时间间隔（分钟）
            time_diff = close_minutes - open_minutes
            # 间隔越短，权重越高（变化速度越快）
            if time_diff > 0:
                # 基准：30分钟内变化权重为2.0，24小时以上变化权重为0.5
                time_weight = min(2.0, max(0.5, 1800 / time_diff + 0.5))
    
    # 根据让球变化调整 λ
    # 主队让球增加 → 主队进球期望增加，客队进球期望减少
    delta_home = handicap_change * HANDICAP_CHANGE_LAMBDA_BOOST * time_weight
    delta_away = -handicap_change * HANDICAP_CHANGE_LAMBDA_BOOST * 0.33 * time_weight  # 客队调整幅度较小
    
    lam_home = max(0.08, lam_home + delta_home)
    lam_away = max(0.08, lam_away + delta_away)
    
    return lam_home, lam_away


def apply_total_line_change_adjustment(lam_home, lam_away, open_total, close_total,
                                       open_time=None, close_time=None):
    """
    应用大小球升降对 λ 的修正（包含时间因素）
    
    例如：
        初盘2.5 → 终盘3.0 → 市场认为比赛更开放
        lambda_total += delta * 0.6
    
    时间因素：
        相同变化幅度下，越临近比赛的变化越有价值
    
    参数：
        lam_home, lam_away: 当前 λ 值
        open_total: 初盘大小球线
        close_total: 终盘大小球线
        open_time: 初盘时间（格式：06-09 17:20）
        close_time: 终盘时间（格式：06-09 17:20）
    
    返回：
        (adjusted_lam_home, adjusted_lam_away)
    """
    if open_total is None or close_total is None:
        return lam_home, lam_away
    
    # 大小球变化
    total_change = close_total - open_total
    
    # 计算时间权重
    time_weight = 1.0
    if open_time and close_time:
        open_minutes = _parse_time(open_time)
        close_minutes = _parse_time(close_time)
        if open_minutes and close_minutes and open_minutes < close_minutes:
            time_diff = close_minutes - open_minutes
            if time_diff > 0:
                time_weight = min(2.0, max(0.5, 1800 / time_diff + 0.5))
    
    # 按比例分配变化到主客队
    total_lambda = lam_home + lam_away
    if total_lambda > 0:
        ratio_home = lam_home / total_lambda
        ratio_away = lam_away / total_lambda
        
        delta_total = total_change * TOTAL_LINE_CHANGE_LAMBDA_BOOST * time_weight
        delta_home = delta_total * ratio_home
        delta_away = delta_total * ratio_away
        
        lam_home = max(0.08, lam_home + delta_home)
        lam_away = max(0.08, lam_away + delta_away)
    
    return lam_home, lam_away


def blend_lambdas_with_market(market_lams, team_lams=None, elo_lams=None):
    """
    融合市场、球队和 ELO 的 λ 值
    
    权重配置：
        market: 0.5（盘口反推，最主要）
        team: 0.3（球队攻防数据）
        elo: 0.2（ELO xG）
    
    参数：
        market_lams: 盘口反推的 λ (lam_home, lam_away)
        team_lams: 球队数据计算的 λ (lam_home, lam_away)
        elo_lams: ELO xG (elo_xg_home, elo_xg_away)
    
    返回：
        (blended_lam_home, blended_lam_away)
    """
    lam_home, lam_away = market_lams
    
    # 初始化加权和
    weighted_home = lam_home * LAMBDA_WEIGHT_MARKET
    weighted_away = lam_away * LAMBDA_WEIGHT_MARKET
    total_weight = LAMBDA_WEIGHT_MARKET
    
    # 添加球队数据权重
    if team_lams:
        weighted_home += team_lams[0] * LAMBDA_WEIGHT_TEAM
        weighted_away += team_lams[1] * LAMBDA_WEIGHT_TEAM
        total_weight += LAMBDA_WEIGHT_TEAM
    
    # 添加 ELO xG 权重
    if elo_lams:
        weighted_home += elo_lams[0] * LAMBDA_WEIGHT_ELO
        weighted_away += elo_lams[1] * LAMBDA_WEIGHT_ELO
        total_weight += LAMBDA_WEIGHT_ELO
    
    # 归一化
    if total_weight > 0:
        lam_home = weighted_home / total_weight
        lam_away = weighted_away / total_weight
    
    return max(0.08, lam_home), max(0.08, lam_away)


def diverse_score_selection(candidates, top_n=3, diversity_threshold=0.5):
    """
    比分多样性选择机制
    
    如果前N个比分过于相似（都是低比分），允许高比分进入推荐池。
    
    参数：
        candidates: 排序后的比分候选列表 [(score, prob), ...]
        top_n: 推荐数量
        diversity_threshold: 多样性阈值，低于此值则增加多样性
    
    返回：
        多样化的比分推荐列表
    """
    if len(candidates) <= top_n:
        return candidates
    
    result = []
    selected_scores = set()
    selected_total_goals = set()
    
    for score, prob in candidates:
        h, a = score
        total_goals = h + a
        
        # 检查是否已经有相似比分
        is_similar = False
        for selected_h, selected_a in selected_scores:
            # 检查比分模式是否相似（同一类结果，进球数相近）
            if _outcome(h, a) == _outcome(selected_h, selected_a):
                if abs(total_goals - (selected_h + selected_a)) <= 1:
                    is_similar = True
                    break
        
        # 如果已经选了太多相似比分，跳过当前比分
        if is_similar and len(result) >= top_n // 2:
            continue
        
        result.append((score, prob))
        selected_scores.add(score)
        selected_total_goals.add(total_goals)
        
        if len(result) >= top_n:
            break
    
    # 如果选中的比分进球数都偏低，尝试加入一个高比分
    if result and len(selected_total_goals) > 0:
        avg_goals = sum(selected_total_goals) / len(selected_total_goals)
        if avg_goals < 2.5 and len(candidates) > top_n:
            # 在剩余候选中找一个高比分
            for score, prob in candidates[top_n:]:
                h, a = score
                if h + a >= 3 and (h, a) not in selected_scores:
                    # 替换概率最低的一个
                    min_idx = min(range(len(result)), key=lambda i: result[i][1])
                    if result[min_idx][1] < prob * 1.2:  # 只有当高比分概率足够时才替换
                        result[min_idx] = (score, prob)
                    break
    
    return result


def _dc_tau(h, a, lam_home, lam_away, rho):
    """Dixon-Coles 相关修正因子（扩展到所有比分，带指数衰减）

    τ 捕捉主客队进球数之间的相关性：
    - ρ < 0（负相关）：不对称比分（如 2-0、3-1）比独立泊松预测的更多
    - ρ > 0（正相关）：对称比分（如 1-1、2-2）比独立泊松预测的更多

    低比分保持原始 D-C 公式，高比分用指数衰减平滑过渡。
    """
    if rho == 0:
        return 1.0
    # 原始 Dixon-Coles 低比分公式（保持兼容）
    if h == 0 and a == 0:
        return 1.0 - lam_home * lam_away * rho
    if h == 0 and a == 1:
        return 1.0 + lam_home * rho
    if h == 1 and a == 0:
        return 1.0 + lam_away * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    # 高比分：负相关效应随总进球数指数衰减
    decay = math.exp(-(h + a) * 0.30)
    if h == a:
        # 对称高分（2-2, 3-3）：负相关是此类比分比预期更少
        return 1.0 + rho * decay
    elif min(h, a) == 0:
        # 零封比分（2-0, 3-0）：负相关 → 不对称比分比预期更多
        return 1.0 - rho * decay
    else:
        # 接近比分（2-1, 3-2）：分差越小效果越弱
        gap_factor = abs(h - a) / (h + a)
        return 1.0 - rho * decay * gap_factor


def _matrix_margins(matrix):
    """从比分矩阵汇总 1X2 边缘概率"""
    margins = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for (h, a), prob in matrix.items():
        margins[_outcome(h, a)] += prob
    return margins


def _asian_payout_home(diff, handicap):
    """亚盘主队结算单位：1=全赢, 0.5=半赢, 0=走水, -0.5=半输, -1=全输"""
    frac = round((handicap * 4) % 4)
    if frac in (1, 3):
        low, high = handicap - 0.25, handicap + 0.25
        return 0.5 * _asian_payout_home(diff, low) + 0.5 * _asian_payout_home(diff, high)
    adj = diff - handicap
    if adj > 1e-9:
        return 1.0
    if abs(adj) <= 1e-9:
        return 0.0
    return -1.0


def _asian_cover_prob(lam_home, lam_away, handicap, rho=0.0):
    """泊松比分矩阵下主队赢盘（含半赢）概率"""
    matrix = build_score_matrix(lam_home, lam_away, rho=rho)
    cover = 0.0
    for (h, a), prob in matrix.items():
        pay = _asian_payout_home(h - a, handicap)
        # 四分盘的半赢只能计入一半概率。原先先判断 pay > 0，导致
        # pay == 0.5 被当成全赢，使 ±0.25/±0.75 等盘口反推偏强。
        cover += max(pay, 0.0) * prob
    return cover


def asian_implied_supremacy(
    handicap, p_home_cover, p_away_cover,
    total_hint=2.5, open_handicap=None, open_hp=None, open_ap=None,
):
    """
    由让球盘 + 上下盘真实概率反推期望净胜球（不再把盘口线当作净胜球）。
    在泊松框架下二分搜索 μ，使 P(主队赢盘) ≈ 去水后主胜概率。
    """
    p_home = max(0.05, min(0.95, p_home_cover))
    if open_handicap is not None and open_hp is not None:
        handicap = _blend_close_open(handicap, open_handicap)
        p_home = _blend_close_open(p_home, open_hp)

    lo, hi = -3.5, 3.5
    for _ in range(52):
        mid = (lo + hi) / 2
        lam_h = max(0.08, (total_hint + mid) / 2)
        lam_a = max(0.08, (total_hint - mid) / 2)
        pc = _asian_cover_prob(lam_h, lam_a, handicap)
        if pc < p_home:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def euro_implied_supremacy(p_home, p_draw, p_away, total_hint=2.5):
    """由欧赔 1X2 真实概率反推期望净胜球（独立于亚盘让球数值）"""
    p_home, p_draw, p_away = max(p_home, 0.02), max(p_draw, 0.02), max(p_away, 0.02)
    lo, hi = -3.5, 3.5
    for _ in range(52):
        mid = (lo + hi) / 2
        lam_h = max(0.08, (total_hint + mid) / 2)
        lam_a = max(0.08, (total_hint - mid) / 2)
        margins = _matrix_margins(build_score_matrix(lam_h, lam_a))
        if margins['home'] < p_home:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def euro_implied_lambdas(p_home, p_draw, p_away, total_hint):
    """由欧赔 1X2 直接拟合主客队 λ（作为球队强度融合的先验）"""
    return _fit_lambda_grid(
        euro_implied_supremacy(p_home, p_draw, p_away, total_hint),
        total_hint, p_home, p_draw, p_away, rho=0.0,
        ou_targets=None, team_lambdas=None,
    )


def blend_market_supremacy(sup_asian, sup_euro):
    """融合亚盘与欧赔反推净胜球；严重分歧时等权避免单边偏差"""
    if sup_asian * sup_euro < 0 or abs(sup_asian - sup_euro) >= SUPREMACY_CONFLICT_GAP:
        return 0.5 * sup_asian + 0.5 * sup_euro
    return SUP_ASIAN_WEIGHT * sup_asian + SUP_EURO_WEIGHT * sup_euro


def compute_prediction_confidence(asian, euro, total, team=None):
    """
    多市场信号一致性 → 置信度 0~1。
    低置信时减少推荐条数并降权排序。
    """
    score = 1.0
    notes = []
    sup_a = asian.get('implied_supremacy', 0.0)
    sup_e = euro.get('implied_supremacy', 0.0)

    if sup_a * sup_e < 0:
        score -= 0.32
        notes.append('亚盘与欧赔净胜球方向相反')
    elif abs(sup_a - sup_e) >= SUPREMACY_CONFLICT_GAP:
        score -= 0.22
        notes.append(f'净胜球分歧较大（亚{sup_a:+.2f}/欧{sup_e:+.2f}）')

    kelly = euro.get('kelly') or {}
    if kelly.get('spread', 99) < 2.5:
        score -= 0.12
        notes.append('凯利三项胶着')

    if team and euro.get('implied_lambdas'):
        target = total.get('implied_total', 2.5)
        tl = team_poisson_lambdas(team, target, team.get('league_profile'))
        el = euro['implied_lambdas']
        gap = abs(el['home'] - tl[0]) + abs(el['away'] - tl[1])
        if gap > 0.85:
            score -= 0.14
            notes.append('球队攻防λ与市场λ偏差大')

    score = max(0.0, min(1.0, score))
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        level, label = 'high', '高置信'
    elif score >= CONFIDENCE_LOW_THRESHOLD:
        level, label = 'medium', '中置信'
    else:
        level, label = 'low', '低置信（谨慎参考）'

    return {
        'score': round(score, 3),
        'level': level,
        'label': label,
        'notes': notes,
        'recommend_count': 2 if level != 'low' else 1,
    }


def team_poisson_lambdas(strength, total_target, league_profile=None):
    """
    用攻防强度构造 λ：主队进攻×客队防守×主场系数。
    defense 为场均失球（对手防守弱则失球多 → 因子更大）。
    
    集成 ELO 评分系统：
    - 使用 ELO 实力因子调整攻防强度
    - ELO 评分高的球队会获得更高的进球期望值

    集成 xG（Expected Goals）：
    - 使用最近5场的 xG/xGA 数据
    - 当实际进球与 xG 有较大差异时，预测回归均值
    """
    lp = league_profile or strength.get('league_profile') or LEAGUE_PROFILES['default']
    avg = lp.get('avg_goal', AVG_LEAGUE_GOAL)
    boost = lp.get('home_boost', HOME_VENUE_ATTACK_BOOST)
    
    # 获取攻防强度
    atk_h = strength['attack_home'] / avg
    def_a = strength['defense_away'] / avg
    atk_a = strength['attack_away'] / avg
    def_h = strength['defense_home'] / avg
    
    # ELO 调整因子
    elo_strength_h = strength.get('elo_strength_home', 1.0)
    elo_strength_a = strength.get('elo_strength_away', 1.0)
    
    # 使用 ELO 实力因子调整攻防强度
    # ELO 评分高的球队进攻能力更强，防守更稳固
    atk_h *= elo_strength_h
    def_h *= elo_strength_a  # 对手ELO高，我方防守压力大（失球可能更多）
    atk_a *= elo_strength_a
    def_a *= elo_strength_h  # 对手ELO高，我方进攻面对更强防守
    
    # 计算基础 lambda
    lam_home = max(0.08, atk_h * def_a * avg * boost)
    lam_away = max(0.08, atk_a * def_h * avg)
    
    # ========== 新增：xG 修正（核心改进）==========
    # 使用最近5场的 xG/xGA 数据进行运气回归修正
    # 当实际进球远低于 xG 时，预测球队可能爆发
    home_xg_last5 = strength.get('home_xg_last5', 0)
    away_xg_last5 = strength.get('away_xg_last5', 0)
    home_xga_last5 = strength.get('home_xga_last5', 0)
    away_xga_last5 = strength.get('away_xga_last5', 0)
    home_recent = strength.get('home_recent', {})
    away_recent = strength.get('away_recent', {})

    # 计算 xG 修正因子
    # 原理：如果球队近期 xG 很高但实际进球少，说明运气差，下一场可能反弹
    if home_xg_last5 > 0 and home_recent:
        home_games = max(1, home_recent.get('games', 5))
        h_gf_per_game = home_recent.get('gf', 0) / home_games

        # xG 均值（最近5场）
        xg_avg_h = home_xg_last5 / min(5, home_games)

        # 计算运气偏差：实际进球 / xG
        # 如果 < 0.7，说明运气差；如果 > 1.3，说明运气好
        luck_ratio = h_gf_per_game / max(xg_avg_h, 0.1)

        # 运气回归修正：运气差的球队增加 λ，运气好的球队减少 λ
        # 修正范围：0.8 ~ 1.4
        xg_factor_h = min(1.4, max(0.8, 1.0 + (1.0 - luck_ratio) * 0.3))
        lam_home *= xg_factor_h

        log.debug(f"主队 xG 修正: xG={xg_avg_h:.2f}, 实际进球={h_gf_per_game:.2f}, 运气因子={xg_factor_h:.2f}")

    if away_xg_last5 > 0 and away_recent:
        away_games = max(1, away_recent.get('games', 5))
        a_gf_per_game = away_recent.get('gf', 0) / away_games

        xg_avg_a = away_xg_last5 / min(5, away_games)
        luck_ratio = a_gf_per_game / max(xg_avg_a, 0.1)
        xg_factor_a = min(1.4, max(0.8, 1.0 + (1.0 - luck_ratio) * 0.3))
        lam_away *= xg_factor_a

        log.debug(f"客队 xG 修正: xG={xg_avg_a:.2f}, 实际进球={a_gf_per_game:.2f}, 运气因子={xg_factor_a:.2f}")

    # 使用 xGA 调整防守端
    # xGA 高说明防守差，对手更容易进球
    if home_xga_last5 > 0:
        # 主队 xGA 越高，客队进球期望越高
        xga_factor_a = 1.0 + (home_xga_last5 / 5.0 - avg) / avg * 0.2
        lam_away *= min(1.3, max(0.7, xga_factor_a))

    if away_xga_last5 > 0:
        # 客队 xGA 越高，主队进球期望越高
        xga_factor_h = 1.0 + (away_xga_last5 / 5.0 - avg) / avg * 0.2
        lam_home *= min(1.3, max(0.7, xga_factor_h))

    # ========== 近期状态衰减加权 ==========
    # 如果 strength 包含近期数据则应用，否则仅依赖长期均值
    if home_recent and away_recent:
        home_games = max(1, home_recent.get('games', 10))
        away_games = max(1, away_recent.get('games', 10))
        # form_pts 范围 0~3 每场，均值≈1.5；>1.5 近期好，<1.5 近期差
        home_form = home_recent.get('form_pts', 0) / (3.0 * home_games)
        away_form = away_recent.get('form_pts', 0) / (3.0 * away_games)
        # 将 form_factor 映射到 ±15% 的 λ 修正（高于均值加分，低于均值减分）
        lam_home *= (1.0 + (home_form - 0.5) * 0.30)
        lam_away *= (1.0 + (away_form - 0.5) * 0.30)
        # 近期进球/失球效率：如果近期场均进球明显偏离长期均值，额外修正
        h_gf_per_game = home_recent.get('gf', 0) / home_games
        h_ga_per_game = home_recent.get('ga', 0) / home_games
        a_gf_per_game = away_recent.get('gf', 0) / away_games
        a_ga_per_game = away_recent.get('ga', 0) / away_games
        # 近期进球比长期预期多/少 → ±10% 微调（在 xG 修正之后应用）
        raw_attack_home = strength.get('attack_home', avg)
        raw_attack_away = strength.get('attack_away', avg)
        lam_home *= (1.0 + (h_gf_per_game - raw_attack_home) / max(raw_attack_home, 0.01) * 0.08)
        lam_away *= (1.0 + (a_gf_per_game - raw_attack_away) / max(raw_attack_away, 0.01) * 0.08)
        lam_home = max(0.06, lam_home)
        lam_away = max(0.06, lam_away)
    
    # 如果有 ELO xG，进行融合
    if 'elo_xg_home' in strength and 'elo_xg_away' in strength:
        elo_weight = 0.25  # ELO 权重（xG 已有较大权重，此处降低）
        lam_home = lam_home * (1 - elo_weight) + strength['elo_xg_home'] * elo_weight
        lam_away = lam_away * (1 - elo_weight) + strength['elo_xg_away'] * elo_weight
    
    # 归一化到目标总进球
    scale = total_target / max(lam_home + lam_away, 0.1)
    return lam_home * scale, lam_away * scale


def _ou_total_distribution(lam_total, max_k=6):
    return {_k: _poisson_pmf(_k, lam_total) for _k in range(max_k + 1)}


def _matrix_total_margins(matrix, max_k=6):
    margins = {k: 0.0 for k in range(max_k + 1)}
    for (h, a), prob in matrix.items():
        t = min(h + a, max_k)
        margins[t] += prob
    return margins


def estimate_lambdas(supremacy, total_line, min_lambda=0.05):
    """由净胜球与总进球快速解 λ（兜底）"""
    lam_home = max(min_lambda, (total_line + supremacy) / 2)
    lam_away = max(min_lambda, (total_line - supremacy) / 2)
    return lam_home, lam_away


def _lambda_fit_error(
    lam_pair, supremacy, target_total, targets, rho,
    ou_targets=None, team_lambdas=None,
):
    """λ 拟合目标函数（越小越好）"""
    lam_h, lam_a = lam_pair
    matrix = build_score_matrix(lam_h, lam_a, rho=rho)
    margins = _matrix_margins(matrix)
    err = (
        FIT_W_1X2 * sum((margins[k] - targets[i]) ** 2 for i, k in enumerate(('home', 'draw', 'away')))
        + FIT_W_TOTAL * (lam_h + lam_a - target_total) ** 2
        + FIT_W_SUPREMACY * (lam_h - lam_a - supremacy) ** 2
    )
    if ou_targets:
        model_ou = _matrix_total_margins(matrix)
        err += FIT_W_OU_DIST * sum((model_ou[k] - ou_targets[k]) ** 2 for k in ou_targets)
    if team_lambdas:
        err += FIT_W_TEAM * (
            (lam_h - team_lambdas[0]) ** 2 + (lam_a - team_lambdas[1]) ** 2
        )
    return err


def _fit_lambda_refine(
    start, supremacy, target_total, targets, rho,
    ou_targets=None, team_lambdas=None,
):
    """网格初解后的坐标下降精调"""
    lh, la = start
    err = _lambda_fit_error((lh, la), supremacy, target_total, targets, rho, ou_targets, team_lambdas)
    step = LAMBDA_REFINE_STEP0
    for _ in range(LAMBDA_REFINE_STEPS):
        improved = False
        for dh in (step, -step, 0):
            for da in (step, -step, 0):
                if dh == 0 and da == 0:
                    continue
                trial = (max(0.08, lh + dh), max(0.08, la + da))
                te = _lambda_fit_error(
                    trial, supremacy, target_total, targets, rho, ou_targets, team_lambdas,
                )
                if te + 1e-9 < err:
                    lh, la, err = trial[0], trial[1], te
                    improved = True
        if not improved:
            step *= 0.55
            if step < 0.008:
                break
    return lh, la


def _fit_lambda_grid(
    supremacy, target_total, p_home, p_draw, p_away, rho=0.0,
    ou_targets=None, team_lambdas=None, euro_lambdas=None,
):
    """λ 网格搜索：1X2 + 反推净胜球 + 大小球分布 + 球队/欧赔先验"""
    targets = (p_home, p_draw, p_away)
    if euro_lambdas:
        best = euro_lambdas
    elif team_lambdas:
        best = team_lambdas
    else:
        best = estimate_lambdas(supremacy, target_total)
    best_err = float('inf')

    def _search(step, center=None, radius=2.5):
        nonlocal best, best_err
        if center is None:
            starts = [i * step for i in range(int(2.6 / step) + 1)]
            pairs = ((lh, la) for lh in starts for la in starts)
        else:
            lh0, la0 = center
            n = int(radius / step) + 1
            pairs = (
                (max(0.08, lh0 + di * step), max(0.08, la0 + dj * step))
                for di in range(-n, n + 1)
                for dj in range(-n, n + 1)
            )
        for lam_h, lam_a in pairs:
            err = _lambda_fit_error(
                (lam_h, lam_a), supremacy, target_total, targets, rho, ou_targets, team_lambdas,
            )
            if err < best_err:
                best_err = err
                best = (lam_h, lam_a)
        return best

    lh, la = _search(LAMBDA_COARSE_STEP)
    lh, la = _search(LAMBDA_FINE_STEP, center=(lh, la), radius=LAMBDA_FINE_RADIUS)
    return _fit_lambda_refine(
        (lh, la), supremacy, target_total, targets, rho, ou_targets, team_lambdas,
    )


def _estimate_dc_rho(lam_home, lam_away, p_draw_target):
    """根据欧赔平局概率估计 Dixon-Coles 相关系数 ρ（负值抬高 0-0/1-1 权重）"""
    base = build_score_matrix(lam_home, lam_away, rho=0.0)
    p_draw_base = _matrix_margins(base)['draw']
    gap = p_draw_target - p_draw_base
    if gap > 0.025:
        return -0.16
    if gap < -0.015:
        return -0.06
    return -0.11


def build_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS, rho=0.0, distribution='poisson',
                       league_profile=None):
    """
    比分矩阵构建；支持泊松分布和负二项分布。
    rho≠0 时施加 Dixon-Coles 低比分修正并归一化。

    参数：
        lam_home, lam_away: 主客队期望进球数
        max_goals: 最大考虑进球数
        rho: Dixon-Coles 相关系数
        distribution: 'poisson' 或 'negative_binomial'
        league_profile: 联赛画像（预留：接入按联赛数据校准的过离散系数）
    """
    cells = {}

    if distribution == 'negative_binomial':
        # 过离散系数保持 1.22：离线回测(1000场,5大联赛)显示，在足球典型 λ 下
        # 把它提高到按联赛表(1.25~1.42)只会增厚 0-0/低比分（0-0 概率 0.087→0.099），
        # 4+ 高比分尾部几乎不变(0.290→0.291)，大小球命中反而略降。故不上调。
        # league_profile 形参预留给未来「用真实赛果标定的过离散」，而非硬编码猜测值。
        overdispersion = 1.22
        var_home = lam_home * overdispersion
        var_away = lam_away * overdispersion
        r_h, p_h = _nb_params_from_mean_var(lam_home, var_home)
        r_a, p_a = _nb_params_from_mean_var(lam_away, var_away)
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tau = _dc_tau(h, a, lam_home, lam_away, rho)
            
            if distribution == 'negative_binomial':
                home_prob = _negative_binomial_pmf(h, r_h, p_h)
                away_prob = _negative_binomial_pmf(a, r_a, p_a)
            else:
                home_prob = _poisson_pmf(h, lam_home)
                away_prob = _poisson_pmf(a, lam_away)
            
            cells[(h, a)] = tau * home_prob * away_prob
    
    total = sum(cells.values())
    if total <= 0:
        return cells
    return {cell: prob / total for cell, prob in cells.items()}


def calibrate_to_euro(matrix, p_home, p_draw, p_away):
    """按欧赔 1X2 缩放矩阵（保留作兜底；主流程已用 λ 拟合替代）"""
    targets = {'home': p_home, 'draw': p_draw, 'away': p_away}
    model = _matrix_margins(matrix)
    adjusted = {}
    for (h, a), prob in matrix.items():
        outcome = _outcome(h, a)
        scale = targets[outcome] / model[outcome] if model[outcome] > 0 else 0.0
        adjusted[(h, a)] = prob * scale
    total = sum(adjusted.values())
    if total <= 0:
        return matrix
    return {cell: prob / total for cell, prob in adjusted.items()}


def _sigmoid(x):
    """Sigmoid 函数：Platt 缩放使用"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _evaluate_risk_level(asian: Dict, euro: Dict, total: Dict,
                         steam_result: Dict, confidence: Dict,
                         similar_market: Dict) -> Dict:
    """
    评估比赛预测的风险等级
    
    风险等级定义：
    - A级：各指标一致 → 推荐3个比分
    - B级：存在冲突因素 → 推荐2个比分
    - C级：风险较高 → 推荐1个比分
    - D级：风险极高 → 不推荐比分
    
    参数：
        asian: 亚盘分析结果
        euro: 欧赔分析结果
        total: 大小球分析结果
        steam_result: 临场资金流结果
        confidence: 置信度
        similar_market: 相似盘口结果
    
    返回：
        风险等级字典
    """
    risk_factors = []
    conflict_count = 0
    risk_score = 0.0
    
    # 因素1：欧亚一致性
    euro_asian_consistent = True
    try:
        if euro.get('implied_home') is not None and asian.get('home_prob') is not None:
            euro_home = euro['implied_home']
            asian_home = asian['home_prob']
            if abs(euro_home - asian_home) > 0.15:
                euro_asian_consistent = False
                conflict_count += 1
                risk_factors.append('欧亚分歧')
                risk_score += 0.2
    except Exception:
        pass
    
    # 因素2：大小球与总进球预期一致性
    total_consistent = True
    try:
        if total.get('close_line') is not None and euro.get('expected_total') is not None:
            market_total = euro['expected_total']
            total_line = total['close_line']
            if abs(market_total - total_line) > 0.5:
                total_consistent = False
                conflict_count += 1
                risk_factors.append('大小球分歧')
                risk_score += 0.15
    except Exception:
        pass
    
    # 因素3：相似盘口置信度
    similar_confident = True
    if similar_market and similar_market.get('confidence', 0) < 0.4:
        similar_confident = False
        conflict_count += 1
        risk_factors.append('相似盘口样本不足')
        risk_score += 0.25
    
    # 因素4：资金流异常
    steam_anomaly = False
    if steam_result and steam_result.get('summary'):
        summary = steam_result['summary']
        if summary.get('has_strong_signal') or summary.get('confidence', 0) > 0.5:
            steam_anomaly = True
            conflict_count += 1
            risk_factors.append('资金流异常')
            risk_score += 0.2
    
    # 因素5：置信度过低
    low_confidence = False
    if confidence:
        conf_score = confidence.get('score', 1.0)
        if conf_score < 0.5:
            low_confidence = True
            conflict_count += 1
            risk_factors.append('置信度过低')
            risk_score += 0.2
    
    # 因素6：盘口变化剧烈（让球反转）
    handicap_reversed = False
    try:
        if asian.get('open_handicap') is not None and asian.get('handicap') is not None:
            open_h = asian['open_handicap']
            close_h = asian['handicap']
            if open_h * close_h < 0:  # 让球方向反转
                handicap_reversed = True
                conflict_count += 1
                risk_factors.append('让球方向反转')
                risk_score += 0.3
    except Exception:
        pass
    
    # 因素7：凯利指数离散度（仅当离散度较高且有明显最难项时）
    kelly = euro.get('kelly', {})
    kelly_spread = kelly.get('spread', 0)
    kelly_hardest = kelly.get('hardest')
    if kelly_spread >= 4.0 and kelly_hardest != 'neutral':
        conflict_count += 1
        risk_factors.append('凯利离散度较高，存在明显分化')
        risk_score += 0.15
    
    # 确定风险等级和推荐数量
    if risk_score < 0.35:
        level = 'A'
        recommend_count = 3
        description = '各指标一致，预测置信度高'
    elif risk_score < 0.55:
        level = 'B'
        recommend_count = 2
        description = f'存在{conflict_count}个冲突因素，需要谨慎'
    elif risk_score < 0.75:
        level = 'C'
        recommend_count = 1
        description = f'冲突因素较多({conflict_count}个)，建议精简投注'
    else:
        level = 'D'
        recommend_count = 0  # 不推荐比分
        description = f'风险较高({conflict_count}个因素)，建议观望'
    
    return {
        'level': level,
        'recommend_count': recommend_count,
        'description': description,
        'risk_factors': risk_factors,
        'risk_score': risk_score,
        'recommend': '正常推荐' if level == 'A' else ('精简推荐' if level == 'B' else ('谨慎推荐' if level == 'C' else '不建议投注比分')),
    }


def _result_label(h, a):
    return "主胜" if h > a else "平局" if h == a else "客胜"


