# -*- coding: utf-8 -*-
"""ML 层的纯计算：特征契约、动态 rho、进球数推荐、按时间切分。

纯计算——**模型训练与推断留在适配层**。判据 20b 是这一批的硬约束：
黄金**不得**放 catboost/xgboost/sklearn 算出来的数（曾有数字模型
因为 `requirements.txt` 用 `>=`、CI 装的版本比本地新，直接红了 5 条）。
所以带 numpy/模型的函数一个都没搬——它们留在 `src/football/ml.py`。

**`get_close_total_line` 不在这里**：它与 `parsing` 里那份**逐字相同**
（F-3 已用 AST 确认），领域层只留 `parsing` 那一份，`ml.py` 改成调它。

**`dixon_coles_*` 是仓库里三份 DC 中的第二份**——只改四格且用**比值形式**，
τ(1,0) 算出来是 `1 + rho*λa/λh` 而不是标准的 `1 + rho*λa`。
F-4 已确认三份是三个不同的模型，`rho=0` 时一致、非零时不一致，**不合并**。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger('domain.football.ml_contract')


CATBOOST_AVAILABLE = False

XGBOOST_AVAILABLE = False

LIGHTGBM_AVAILABLE = False

SKLEARN_AVAILABLE = False

PYTORCH_AVAILABLE = False

DEFAULT_RHO = 0.1

GOAL_COUNT_LABELS = {0: '0球', 1: '1球', 2: '2球', 3: '3球', 
                     4: '4球', 5: '5球', 6: '6球', 7: '7球+'}

_trained_ml_model = None

_trained_ml_metadata = None

_trained_ml_feature_names = []

FEATURE_VERSION = "v2"

ELO_FEATURES = [
    {'name': 'elo_home', 'type': 'float', 'default': 1500.0, 'description': '主队ELO评分'},
    {'name': 'elo_away', 'type': 'float', 'default': 1500.0, 'description': '客队ELO评分'},
    {'name': 'elo_diff', 'type': 'float', 'default': 0.0, 'description': 'ELO差值'},
]

RECENT_FORM_5_FEATURES = [
    {'name': 'home_attack_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场进攻效率'},
    {'name': 'home_defense_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场防守效率'},
    {'name': 'away_attack_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场进攻效率'},
    {'name': 'away_defense_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场防守效率'},
    {'name': 'home_form_points_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5场场均积分'},
    {'name': 'away_form_points_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5场场均积分'},
    {'name': 'home_win_rate_5', 'type': 'float', 'default': 0.33, 'description': '主队最近5场胜率'},
    {'name': 'away_win_rate_5', 'type': 'float', 'default': 0.33, 'description': '客队最近5场胜率'},
    {'name': 'home_draw_rate_5', 'type': 'float', 'default': 0.33, 'description': '主队最近5场平局率'},
    {'name': 'away_draw_rate_5', 'type': 'float', 'default': 0.33, 'description': '客队最近5场平局率'},
]

RECENT_FORM_10_FEATURES = [
    {'name': 'home_win_rate_10', 'type': 'float', 'default': 0.33, 'description': '主队最近10场胜率'},
    {'name': 'away_win_rate_10', 'type': 'float', 'default': 0.33, 'description': '客队最近10场胜率'},
    {'name': 'home_goals_for_10', 'type': 'float', 'default': 1.5, 'description': '主队最近10场场均进球'},
    {'name': 'home_goals_against_10', 'type': 'float', 'default': 1.5, 'description': '主队最近10场场均失球'},
    {'name': 'away_goals_for_10', 'type': 'float', 'default': 1.5, 'description': '客队最近10场场均进球'},
    {'name': 'away_goals_against_10', 'type': 'float', 'default': 1.5, 'description': '客队最近10场场均失球'},
]

HOME_AWAY_FEATURES = [
    {'name': 'home_h_goals_for_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5个主场进球/场'},
    {'name': 'home_h_goals_against_5', 'type': 'float', 'default': 1.5, 'description': '主队最近5个主场失球/场'},
    {'name': 'away_a_goals_for_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5个客场进球/场'},
    {'name': 'away_a_goals_against_5', 'type': 'float', 'default': 1.5, 'description': '客队最近5个客场失球/场'},
]

SAMPLE_COUNT_FEATURES = [
    {'name': 'home_matches_count', 'type': 'int', 'default': 0, 'description': '主队可用历史比赛数'},
    {'name': 'away_matches_count', 'type': 'int', 'default': 0, 'description': '客队可用历史比赛数'},
]

EURO_FEATURES = [
    {'name': 'euro_home_prob', 'type': 'float', 'default': 0.333, 'description': '欧赔去水后主胜概率'},
    {'name': 'euro_draw_prob', 'type': 'float', 'default': 0.333, 'description': '欧赔去水后平局概率'},
    {'name': 'euro_away_prob', 'type': 'float', 'default': 0.334, 'description': '欧赔去水后客胜概率'},
]

ASIAN_FEATURES = [
    {'name': 'asian_handicap', 'type': 'float', 'default': 0.0, 'description': '亚盘让球（主队视角）'},
    {'name': 'asian_home_prob', 'type': 'float', 'default': 0.5, 'description': '亚盘去水后主队概率'},
    {'name': 'asian_away_prob', 'type': 'float', 'default': 0.5, 'description': '亚盘去水后客队概率'},
]

TOTAL_FEATURES = [
    {'name': 'total_line', 'type': 'float', 'default': 2.5, 'description': '大小球盘口线'},
    {'name': 'over_prob', 'type': 'float', 'default': 0.5, 'description': '大球概率'},
    {'name': 'under_prob', 'type': 'float', 'default': 0.5, 'description': '小球概率'},
]

LEAGUE_FEATURES = [
    {'name': 'league_avg_goals_100', 'type': 'float', 'default': 2.7, 'description': '联赛最近100场平均进球'},
    {'name': 'league_draw_rate_100', 'type': 'float', 'default': 0.26, 'description': '联赛最近100场平局率'},
    {'name': 'is_home_favorite', 'type': 'int', 'default': 1, 'description': '是否主队热门（1是0否）'},
]

MISSING_FLAG_FEATURES = [
    {'name': 'has_asian_odds', 'type': 'int', 'default': 1, 'description': '是否有亚盘数据'},
    {'name': 'has_total_odds', 'type': 'int', 'default': 1, 'description': '是否有大小球数据'},
    {'name': 'has_euro_odds', 'type': 'int', 'default': 1, 'description': '是否有欧赔数据'},
]

ALL_FEATURES = (
    ELO_FEATURES +
    RECENT_FORM_5_FEATURES +
    RECENT_FORM_10_FEATURES +
    HOME_AWAY_FEATURES +
    SAMPLE_COUNT_FEATURES +
    EURO_FEATURES +
    ASIAN_FEATURES +
    TOTAL_FEATURES +
    LEAGUE_FEATURES +
    MISSING_FLAG_FEATURES
)


def get_dc_rho(league: str = None, total_line: float = None, handicap: float = None) -> float:
    """
    根据比赛特征动态获取 Dixon-Coles 的 rho 参数
    
    参数：
        league: 联赛名称
        total_line: 大小球盘口
        handicap: 亚盘让球
    
    返回：
        动态 rho 值
    """
    # 默认值
    rho = 0.1
    
    # 低进球联赛（如意甲防守强联赛）使用更高的 rho
    low_goal_leagues = ['意甲', '意乙', '葡超', '希腊超', '阿甲']
    if league and league in low_goal_leagues:
        rho = 0.12
    
    # 亚冠/欧冠等大赛倾向于低 rho
    big_leagues = ['欧冠', '欧联', '世界杯', '欧洲杯', '美洲杯']
    if league and league in big_leagues:
        rho = 0.08
    
    # 根据总进球盘口调整
    if total_line is not None:
        if total_line <= 2.25:
            rho = max(rho, 0.12)  # 低进球盘口，提高 rho
        elif total_line >= 3.0:
            rho = min(rho, 0.04)  # 高进球盘口，降低 rho
    
    # 根据让球调整
    if handicap is not None:
        if abs(handicap) <= 0.25:
            rho = max(rho, 0.10)  # 平手盘，提高 rho
    
    return rho


def poisson_pmf(k: int, lam: float) -> float:
    """泊松概率质量函数 P(X=k)"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def dixon_coles_adjustment(rho: float, lam_home: float, lam_away: float,
                          h_goals: int, a_goals: int) -> float:
    """Dixon-Coles 调整系数"""
    if h_goals > 1 or a_goals > 1:
        return 1.0

    p_home_0 = poisson_pmf(0, lam_home)
    p_home_1 = poisson_pmf(1, lam_home)
    p_away_0 = poisson_pmf(0, lam_away)
    p_away_1 = poisson_pmf(1, lam_away)

    if h_goals == 0 and a_goals == 0:
        return 1 - rho * p_home_1 * p_away_1 / (p_home_0 * p_away_0)
    elif h_goals == 1 and a_goals == 0:
        return 1 + rho * p_home_0 * p_away_1 / (p_home_1 * p_away_0)
    elif h_goals == 0 and a_goals == 1:
        return 1 + rho * p_home_1 * p_away_0 / (p_home_0 * p_away_1)
    elif h_goals == 1 and a_goals == 1:
        return 1 - rho * p_home_0 * p_away_0 / (p_home_1 * p_away_1)
    return 1.0


def dixon_coles_score_prob(h_goals: int, a_goals: int, lam_home: float, 
                           lam_away: float, rho: float = DEFAULT_RHO) -> float:
    """计算 Dixon-Coles 模型下的比分概率"""
    poisson_prob = poisson_pmf(h_goals, lam_home) * poisson_pmf(a_goals, lam_away)
    adjustment = dixon_coles_adjustment(rho, lam_home, lam_away, h_goals, a_goals)
    return poisson_prob * adjustment


def get_goal_count_distribution_from_dist(goal_dist: Dict[int, float]) -> List[Dict]:
    """从进球数分布字典获取分布统计（用于融合后的分布）"""
    return [
        {
            'goals': goals,
            'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
            'probability': prob,
            'percentage': f'{prob * 100:.1f}%'
        }
        for goals, prob in sorted(goal_dist.items())
    ]


def recommend_goal_counts_from_dist(goal_dist: Dict[int, float], top_n: int = 2, 
                                      high_risk: bool = False, low_quality_sample: bool = False) -> List[Dict]:
    """
    从进球数分布字典推荐概率最大的进球数
    
    动态推荐策略：
    - 第一名概率 ≥ 26%，且第二名差距 ≥ 5%，只推 1 个
    - 前两名累计概率 ≥ 45%，推 2 个
    - 高风险盘或盘口冲突，最多推 1 个
    - 低质量历史样本时，只展示分布，推荐基于原始模型
    
    参数：
        goal_dist: 进球数分布字典
        top_n: 最大推荐数量（上限）
        high_risk: 是否为高风险盘
        low_quality_sample: 是否为低质量样本
    
    返回：
        推荐列表
    """
    sorted_counts = sorted(goal_dist.items(), key=lambda x: -x[1])
    
    if not sorted_counts:
        return []
    
    recommendations = []
    
    # 高风险盘或盘口冲突，最多推1个
    if high_risk:
        goals, prob = sorted_counts[0]
        recommendations.append({
            'goals': goals,
            'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
            'probability': prob,
            'rank': 1
        })
        return recommendations
    
    # 第一名概率 ≥ 26%，且第二名差距 ≥ 5%，只推 1 个
    if len(sorted_counts) >= 2:
        first_prob = sorted_counts[0][1]
        second_prob = sorted_counts[1][1]
        
        if first_prob >= 0.26 and (first_prob - second_prob) >= 0.05:
            goals, prob = sorted_counts[0]
            recommendations.append({
                'goals': goals,
                'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
                'probability': prob,
                'rank': 1
            })
            return recommendations
    
    # 前两名累计概率 ≥ 45%，推 2 个
    if len(sorted_counts) >= 2:
        first_second_total = sorted_counts[0][1] + sorted_counts[1][1]
        if first_second_total >= 0.45:
            for i, (goals, prob) in enumerate(sorted_counts[:2], 1):
                recommendations.append({
                    'goals': goals,
                    'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
                    'probability': prob,
                    'rank': i
                })
            return recommendations
    
    # 默认策略：按概率覆盖和差距阈值选择
    recommendations = [sorted_counts[0]]
    
    for goals, prob in sorted_counts[1:]:
        if len(recommendations) >= 3:
            break
        
        # 和第一名差距过大则不推荐
        if prob < sorted_counts[0][1] * 0.72:
            continue
        
        # 推荐累计概率至少覆盖 48%
        recommendations.append((goals, prob))
        if sum(x[1] for x in recommendations) >= 0.48:
            break
    
    # 转换为输出格式
    return [
        {
            'goals': goals,
            'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
            'probability': prob,
            'rank': i + 1
        }
        for i, (goals, prob) in enumerate(recommendations)
    ]


def get_feature_names() -> List[str]:
    """获取所有特征名称列表"""
    return [f['name'] for f in ALL_FEATURES]


def get_feature_defaults() -> Dict[str, Any]:
    """获取特征默认值字典"""
    return {f['name']: f['default'] for f in ALL_FEATURES}


def validate_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证并标准化特征字典
    
    参数：
        features: 原始特征字典
    
    返回：
        标准化后的特征字典（补充缺失字段的默认值）
    """
    result = get_feature_defaults().copy()
    
    for key, value in features.items():
        if key in result:
            # 类型转换
            feature_def = next((f for f in ALL_FEATURES if f['name'] == key), None)
            if feature_def:
                target_type = feature_def['type']
                if target_type == 'float':
                    result[key] = float(value) if value is not None else feature_def['default']
                elif target_type == 'int':
                    result[key] = int(value) if value is not None else feature_def['default']
                else:
                    result[key] = value
    
    return result


def audit_feature_payload(features: Dict[str, Any]) -> Dict[str, Any]:
    """Return a machine-readable contract audit without silently hiding drift."""
    expected = set(get_feature_names())
    supplied = set(features or {})
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    return {
        'feature_version': FEATURE_VERSION,
        'expected_count': len(expected),
        'supplied_count': len(supplied),
        'missing': missing,
        'unknown': unknown,
        'complete': not missing and not unknown,
    }


def get_feature_description(name: str) -> str:
    """获取特征描述"""
    feature = next((f for f in ALL_FEATURES if f['name'] == name), None)
    return feature['description'] if feature else ''


def split_by_time(samples: List[Dict], train_ratio: float = 0.7, 
                  val_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按时间切分数据集
    
    参数：
        samples: 所有样本
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    返回：
        (训练集, 验证集, 测试集)
    """
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_set = samples[:train_end]
    val_set = samples[train_end:val_end]
    test_set = samples[val_end:]
    
    print(f"数据切分完成:")
    print(f"  训练集: {len(train_set)} 场 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_set)} 场 ({val_ratio*100:.0f}%)")
    print(f"  测试集: {len(test_set)} 场 ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_set, val_set, test_set
