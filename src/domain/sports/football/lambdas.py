# -*- coding: utf-8 -*-
"""期望进球 λ：市场隐含、球队实力推算、盘口变化调整、拟合与融合。

纯计算——不读全局配置、不发请求、不看时钟。
"""

import logging
import math
import re

from .parsing import blend_close_open as _blend_close_open
from .scoring_model import (
    MAX_GOALS, build_score_matrix, _matrix_margins, _matrix_total_margins,
)

log = logging.getLogger('domain.football.lambdas')

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


def estimate_lambdas(supremacy, total_line, min_lambda=0.05):
    """由净胜球与总进球快速解 λ（兜底）"""
    lam_home = max(min_lambda, (total_line + supremacy) / 2)
    lam_away = max(min_lambda, (total_line - supremacy) / 2)
    return lam_home, lam_away
