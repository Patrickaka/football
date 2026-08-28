# -*- coding: utf-8 -*-
"""预测置信度、比分多样化与风险评级。

纯计算——不读全局配置、不发请求、不看时钟。

`_evaluate_risk_level` 132 行、多档门槛**互相耦合**（判据 28）——
造边界样本前先跑验算把各分量打出来，别猜它会走哪条分支。
"""

import math
from typing import Dict, List, Optional, Tuple

from .lambdas import team_poisson_lambdas
from .scoring_model import _outcome

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
