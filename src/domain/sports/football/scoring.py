# -*- coding: utf-8 -*-
"""比分分布的成形、市场联合状态、半全场修正与推荐多样化。

纯计算——不读全局配置、不碰存储与时钟。留在
`src/football/scoring.py` 的是五个**编排函数**：它们要么碰随机源
（`perturb_parameters`）、要么要注入带存储的协作者
（`predict_scores` 要比分矩阵与校准、`calculate_half_full_time_probs` 要
半场统计库、`_pick_recommendations` 要盘口聚类与价值下注、
`ensemble_predict_scores` 要 `predict_scores` 本身）。
按计划它们归 F-9 与 F-14 用注入分组统一处理。

**推荐理由与推荐结论必须走同一份实现**（判据 11）：曾有数字模型
发现过「为什么推荐这注」加起来对不上旁边的总分，而总分本身是对的
——谁也不会发现解释和结论不是一回事。
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from .lambdas import (
    apply_handicap_change_adjustment, apply_total_line_change_adjustment,
    blend_lambdas_with_market, estimate_lambdas, market_implied_lambdas,
    team_poisson_lambdas, _fit_lambda_grid,
)
from .markets import implied_total_goals, remove_vig
from .parsing import get_close_total_line
from .parsing import blend_close_open as _blend_close_open  # noqa: E402
from .scoring_model import (
    _estimate_dc_rho, _ou_total_distribution, _result_label,
)
from .upset import _evaluate_upset_risk
from .value import calculate_ev, calculate_value

log = logging.getLogger('domain.football.scoring')


# 迁移当时 config.py 的真实取值（原样搬来）
SCORE_1X2_MARKET_ANCHOR_STRENGTH = 1.0
MAX_GOALS = 7
AVG_LEAGUE_GOAL = 1.35
SUPREMACY_CONFLICT_GAP = 0.75
MOMENTUM_SUPREMACY_WEIGHT = 0.22
LEAGUE_PROFILES = {
    'default': {'avg_goal': 1.42, 'home_boost': 1.06, 'low_score': 0.92, 'draw_mult': 1.0},
    '英超': {'avg_goal': 1.52, 'home_boost': 1.08, 'low_score': 0.88, 'draw_mult': 0.95},
    '英冠': {'avg_goal': 1.46, 'home_boost': 1.07, 'low_score': 0.90, 'draw_mult': 0.96},
    '西甲': {'avg_goal': 1.42, 'home_boost': 1.07, 'low_score': 0.95, 'draw_mult': 1.05},
    '意甲': {'avg_goal': 1.32, 'home_boost': 1.05, 'low_score': 1.05, 'draw_mult': 1.08},
    '德甲': {'avg_goal': 1.56, 'home_boost': 1.06, 'low_score': 0.86, 'draw_mult': 0.94},
    '法甲': {'avg_goal': 1.36, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.02},
    '荷甲': {'avg_goal': 1.58, 'home_boost': 1.05, 'low_score': 0.85, 'draw_mult': 0.93},
    '葡超': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '欧冠': {'avg_goal': 1.50, 'home_boost': 1.04, 'low_score': 0.92, 'draw_mult': 0.98},
    '欧联': {'avg_goal': 1.44, 'home_boost': 1.05, 'low_score': 0.94, 'draw_mult': 1.0},
    '世界杯': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '欧洲杯': {'avg_goal': 1.40, 'home_boost': 1.04, 'low_score': 0.98, 'draw_mult': 1.02},
    '友谊': {'avg_goal': 1.44, 'home_boost': 1.02, 'low_score': 0.95, 'draw_mult': 0.97},
    '国际': {'avg_goal': 1.42, 'home_boost': 1.03, 'low_score': 0.96, 'draw_mult': 1.0},
    '巴甲': {'avg_goal': 1.42, 'home_boost': 1.08, 'low_score': 0.92, 'draw_mult': 0.96},
    '阿甲': {'avg_goal': 1.36, 'home_boost': 1.07, 'low_score': 0.96, 'draw_mult': 0.98},
    '中超': {'avg_goal': 1.32, 'home_boost': 1.07, 'low_score': 1.00, 'draw_mult': 1.04},
    '日职': {'avg_goal': 1.34, 'home_boost': 1.06, 'low_score': 1.00, 'draw_mult': 1.03},
    '韩K': {'avg_goal': 1.30, 'home_boost': 1.06, 'low_score': 1.02, 'draw_mult': 1.04},
}
HEAT_RATIO_HOT = 0.70
HEAT_RATIO_COLD = 1.32
HEAT_FILTER_PENALTY = 0.90   # 提高惩罚系数，减少对高比分的压制（原0.75）
COLD_FILTER_BONUS = 1.08     # 原1.18，缩小冷门奖励，防止低比分通过"冷门"机制反复被加权
SCORE_BASELINE_FREQ = {
    (0, 0): 0.075, (1, 0): 0.085, (0, 1): 0.065, (1, 1): 0.110,
    (2, 0): 0.078, (0, 2): 0.055, (2, 1): 0.105, (1, 2): 0.068,
    (2, 2): 0.045, (3, 0): 0.042, (0, 3): 0.025, (3, 1): 0.052,
    (1, 3): 0.032, (3, 2): 0.035, (2, 3): 0.025, (4, 0): 0.020,
    (0, 4): 0.012, (4, 1): 0.025, (1, 4): 0.014, (3, 3): 0.015,
}


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


def _restore_outcome_mass(matrix: Dict, reference: Dict) -> Dict:
    """把每个胜平负分组的质量放回 `reference` 的边际，只保留组内的比分形状。

    胜平负质量在上游已锚到去水收盘价，亚盘与大小球的公平价约束只许改
    比分形状；没有这一步，亚盘那条约束会把主推方向从市场推开。
    """
    target = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    current = {'H': 0.0, 'D': 0.0, 'A': 0.0}
    for (home, away), probability in reference.items():
        target[_score_result_code(home, away)] += probability
    for (home, away), probability in matrix.items():
        current[_score_result_code(home, away)] += probability
    scale = {
        key: target[key] / current[key] if current[key] > 0 else 0.0
        for key in target
    }
    restored = {
        (home, away): probability * scale[_score_result_code(home, away)]
        for (home, away), probability in matrix.items()
    }
    total = sum(restored.values())
    if total <= 0:
        return matrix
    return {score: probability / total for score, probability in restored.items()}


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

    matrix = _restore_outcome_mass(matrix, before_matrix)
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
        'preserved_1x2': True,
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
    """Anchor final score marginals to de-vigged closing 1X2 odds.

    Every upstream score transform is free to improve the within-outcome score
    shape; this guard only controls aggregate H/D/A mass.  Strength 1.0 makes
    the marginals equal the market: without team data the model's departures
    from the closing price carry no information (719 settled matches, the
    market won 24 of 58 disagreements against the model's 17), so any lower
    strength must be re-justified with the same replay before it is tuned.
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
    的比赛预测「小球」，真实 over 率却约 48%）。

    反推交给 `markets.implied_total_goals`：进球分布的锚点必须与比分矩阵的 λ 来自
    同一个函数，整数线 / 四分线 / 平均线上两者才会锚到同一个总进球。极端 p_over
    与超高线会被那边的 IMPLIED_TOTAL_PROB_CLAMP / IMPLIED_TOTAL_BOUNDS 截断。
    """
    try:
        line = float(line)
        p_over = float(p_over)
    except (TypeError, ValueError):
        return None
    if not (0.0 < p_over < 1.0):
        return None
    return implied_total_goals(line, p_over)


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


def _pick_recommendations(candidates, asian, euro, total, n=2, pool=12, confidence=None, league_profile=None, team=None, similar_market=None, market_prior_fn=None):
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
    if market_prior_fn and asian.get('handicap') is not None and total.get('close_line') is not None:
        try:
            market_prior = market_prior_fn(
                asian['handicap'], total.get('close_line', total.get('line', 2.5))) or {}
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
        if market_prior_fn and asian.get('handicap') is not None and total.get('close_line') is not None:
            try:
                prior = market_prior_fn(
                    asian['handicap'], total.get('close_line', total.get('line', 2.5)))
                if prior:
                    score_key = f"{h}-{a}"
                    prior_prob = prior.get(score_key, 0.0)
                    if prior_prob > 0:
                        prior_bonus = 1.0 + prior_prob * 0.3
            except Exception as e:
                log.debug(f"盘口聚类先验获取失败: {e}")

        # 赔率价值计算（仅记录，不参与命中率排序）
        value_info = None
        if euro.get('raw_odds', {}).get('close'):
            try:
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
