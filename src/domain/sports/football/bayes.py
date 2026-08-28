# -*- coding: utf-8 -*-
"""贝叶斯 MCMC 采样、残差修正与平局校准。

纯计算——**随机源由调用方注入**（判据 16：不注入的话黄金文件不可复现）。
残差模型走 LightGBM、平局校准走 sklearn，属「配置让它不可达」那一类：
库装不上时降级，**补用例不要删**（判据 9）。
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
import random as _random

from .scoring_model import build_score_matrix, _matrix_margins

log = logging.getLogger('domain.football.bayes')

# 迁移当时 `src/football/config.py` 的真实取值（已实测，不是按命名推测——判据 10）
MAX_GOALS = 7
AVG_LEAGUE_GOAL = 1.35
HOME_VENUE_ATTACK_BOOST = 1.06
LAMBDA_WEIGHT_MARKET = 0.5
LAMBDA_WEIGHT_TEAM = 0.3
LAMBDA_WEIGHT_ELO = 0.2
LAMBDA_COARSE_STEP = 0.12
LAMBDA_FINE_RADIUS = 0.18
LAMBDA_FINE_STEP = 0.04
LAMBDA_REFINE_STEP0 = 0.07
LAMBDA_REFINE_STEPS = 28
FIT_W_1X2 = 3.0
FIT_W_TOTAL = 1.4
FIT_W_TEAM = 1.35
FIT_W_SUPREMACY = 2.0
FIT_W_OU_DIST = 0.9
SUP_ASIAN_WEIGHT = 0.48
SUP_EURO_WEIGHT = 0.52
SUPREMACY_CONFLICT_GAP = 0.75
HANDICAP_CHANGE_LAMBDA_BOOST = 0.15
TOTAL_LINE_CHANGE_LAMBDA_BOOST = 0.6
CONFIDENCE_HIGH_THRESHOLD = 0.72
CONFIDENCE_LOW_THRESHOLD = 0.52

# 迁移当时 config 的联赛画像表（20 个联赛）。领域层只用到 `default` 那一项做兜底，
# 具体画像由调用方传 `league_profile`。
LEAGUE_PROFILES = {'default': {'avg_goal': 1.42, 'home_boost': 1.06,
                               'low_score': 0.92, 'draw_mult': 1.0}}


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
                         n_samples=2000, burn_in=500, step_size=0.05, rng=None):
    """
    使用 Metropolis-Hastings 算法采样 λ_home, λ_away, rho 的后验分布。

    返回：采样结果列表，包含 (lam_home, lam_away, rho, log_prob)

    **`rng` 由调用方注入**（判据 16：副作用不要跟着搬进领域层）。
    不注入的话黄金文件不可复现——采样是这个模块里唯一的随机源。
    默认值是 `random` 模块本身，与迁移前一致。
    """
    random = rng if rng is not None else _random
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


def bayesian_predict_scores(targets, target_total, supremacy, league_profile=None, team_strength=None, rng=None):
    """
    贝叶斯框架下的比分概率预测。
    
    返回：
        mean_matrix: 后验均值比分矩阵
        credible_interval: 关键参数的置信区间
        samples: 原始采样结果（用于进一步分析）
    """
    samples = _mcmc_sample_lambdas(targets, target_total, supremacy, league_profile,
                                   team_strength, rng=rng)
    
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
