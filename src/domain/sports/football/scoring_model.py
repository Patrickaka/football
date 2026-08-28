# -*- coding: utf-8 -*-
"""足球比分分布：泊松 / 负二项 / Dixon-Coles 修正 / 比分矩阵与边际。

纯计算——不读全局配置、不发请求、不看时钟。

**仓库里有三份 Dixon-Coles，它们是三个不同的模型，不是三份拷贝**
（F-4 用双向语料比对确认，详见 `tests/domain/sports/football/test_modeling.py`）：
本模块这份把 τ 修正**扩展到所有比分**（`exp(-(h+a)*0.3)` 衰减），
`src/football/ml.py` 那份只改四格且用的是比值形式（数值与标准公式不同），
`domain/sports/beidan/scoring_model.py` 那份是标准四格形式。
`rho = 0` 时三者一致（≤2.8e-17），非零时不一致——**所以没有合并**。
"""

import math

from .markets import poisson_pmf as _poisson_pmf

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


def _matrix_margins(matrix):
    """从比分矩阵汇总 1X2 边缘概率"""
    margins = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
    for (h, a), prob in matrix.items():
        margins[_outcome(h, a)] += prob
    return margins


def _matrix_total_margins(matrix, max_k=6):
    margins = {k: 0.0 for k in range(max_k + 1)}
    for (h, a), prob in matrix.items():
        t = min(h + a, max_k)
        margins[t] += prob
    return margins


def _ou_total_distribution(lam_total, max_k=6):
    return {_k: _poisson_pmf(_k, lam_total) for _k in range(max_k + 1)}


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


def _outcome(h, a):
    return 'home' if h > a else 'draw' if h == a else 'away'


def _result_label(h, a):
    return "主胜" if h > a else "平局" if h == a else "客胜"


def _sigmoid(x):
    """Sigmoid 函数：Platt 缩放使用"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
