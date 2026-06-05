#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球机器学习模块
================

包含：
1. Dixon-Coles 模型（改进泊松分布）
2. 进球数推荐
3. 机器学习预测器（CatBoost/XGBoost/LightGBM）
"""

import math
import os
import pickle
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable

# 尝试导入机器学习库
CATBOOST_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


# ==================== Dixon-Coles 模型 ====================

DEFAULT_RHO = 0.1

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

def dixon_coles_score_matrix(lam_home: float, lam_away: float,
                             max_goals: int = 7, rho: float = DEFAULT_RHO) -> np.ndarray:
    """生成 Dixon-Coles 比分概率矩阵"""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[h, a] = dixon_coles_score_prob(h, a, lam_home, lam_away, rho)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix

def dixon_coles_1x2_prob(lam_home: float, lam_away: float,
                        max_goals: int = 7, rho: float = DEFAULT_RHO) -> Dict[str, float]:
    """计算 Dixon-Coles 模型下的 1X2 概率"""
    matrix = dixon_coles_score_matrix(lam_home, lam_away, max_goals, rho)
    p_home = np.triu(matrix, 1).sum()
    p_draw = np.trace(matrix)
    p_away = np.tril(matrix, -1).sum()
    return {'home': p_home, 'draw': p_draw, 'away': p_away}


# ==================== 进球数推荐 ====================

GOAL_COUNT_LABELS = {0: '0球', 1: '1球', 2: '2球', 3: '3球', 
                     4: '4球', 5: '5球', 6: '6球', 7: '7球+'}

def calculate_goal_counts(prob_matrix: np.ndarray) -> Dict[int, float]:
    """计算各进球数的概率"""
    goal_counts = {}
    max_goals = prob_matrix.shape[0] - 1
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            total_goals = h + a
            goal_counts[total_goals] = goal_counts.get(total_goals, 0.0) + prob_matrix[h, a]
    return goal_counts

def recommend_goal_counts(prob_matrix: np.ndarray, top_n: int = 2) -> List[Dict]:
    """推荐概率最大的进球数"""
    goal_counts = calculate_goal_counts(prob_matrix)
    sorted_counts = sorted(goal_counts.items(), key=lambda x: -x[1])
    recommendations = []
    for i, (goals, prob) in enumerate(sorted_counts[:top_n], 1):
        recommendations.append({
            'goals': goals,
            'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
            'probability': prob,
            'rank': i
        })
    return recommendations

def get_goal_count_distribution(prob_matrix: np.ndarray) -> List[Dict]:
    """获取进球数分布统计"""
    goal_counts = calculate_goal_counts(prob_matrix)
    sorted_counts = sorted(goal_counts.items(), key=lambda x: x[0])
    distribution = []
    for goals, prob in sorted_counts:
        distribution.append({
            'goals': goals,
            'label': GOAL_COUNT_LABELS.get(goals, f'{goals}球'),
            'probability': prob,
            'percentage': f'{prob * 100:.1f}%'
        })
    return distribution

def predict_goal_counts_from_candidates(candidates: List[Tuple], max_goals: int = 7) -> Dict:
    """从候选比分列表计算进球数推荐"""
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for item in candidates:
        if len(item) == 2 and isinstance(item[0], tuple):
            (h, a), prob = item
        elif len(item) == 3:
            h, a, prob = item
        else:
            continue
        if h <= max_goals and a <= max_goals:
            prob_matrix[h, a] = prob
    total_prob = prob_matrix.sum()
    if total_prob > 0:
        prob_matrix = prob_matrix / total_prob
    
    return {
        'recommendations': recommend_goal_counts(prob_matrix, top_n=2),
        'distribution': get_goal_count_distribution(prob_matrix),
        'over_under': {'over': np.sum(prob_matrix[np.triu_indices(max_goals+1, k=3)]),
                       'under': np.sum(prob_matrix[np.tril_indices(max_goals+1, k=2)])},
        'matrix': prob_matrix.tolist()
    }


# ==================== 机器学习预测器 ====================

class MLFootballPredictor:
    """机器学习足球预测器"""
    
    def __init__(self, model_type: str = 'auto'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
        if model_type == 'auto':
            if CATBOOST_AVAILABLE:
                self.model_type = 'catboost'
            elif XGBOOST_AVAILABLE:
                self.model_type = 'xgboost'
            elif LIGHTGBM_AVAILABLE:
                self.model_type = 'lightgbm'
            elif SKLEARN_AVAILABLE:
                self.model_type = 'randomforest'
            else:
                self.model_type = 'none'

    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            return CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6,
                                     loss_function='MultiClass', random_seed=42, verbose=False)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6,
                                random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=6,
                                 random_state=42, verbose=-1)
        elif self.model_type == 'randomforest' and SKLEARN_AVAILABLE:
            return RandomForestClassifier(n_estimators=200, max_depth=10,
                                         min_samples_split=10, random_state=42)
        return None

    def extract_features(self, match_data: Dict) -> List[float]:
        """提取比赛特征"""
        features = []
        elo_home = match_data.get('elo_home', 1500)
        elo_away = match_data.get('elo_away', 1500)
        features.extend([elo_home, elo_away, elo_home - elo_away])
        
        euro_home = match_data.get('euro_home', 2.5)
        euro_draw = match_data.get('euro_draw', 3.0)
        euro_away = match_data.get('euro_away', 3.0)
        features.extend([euro_home, euro_draw, euro_away])
        
        features.extend([match_data.get('asian_handicap', 0),
                        match_data.get('asian_home_water', 0.95),
                        match_data.get('asian_away_water', 0.95)])
        
        features.extend([match_data.get('total_line', 2.5),
                        match_data.get('total_over_water', 0.95),
                        match_data.get('total_under_water', 0.95)])
        
        features.extend([match_data.get('home_attack', 1.3),
                        match_data.get('home_defense', 1.2),
                        match_data.get('away_attack', 1.2),
                        match_data.get('away_defense', 1.3)])
        
        features.extend([match_data.get('home_form', 0.5),
                        match_data.get('away_form', 0.5),
                        match_data.get('home_form', 0.5) - match_data.get('away_form', 0.5)])
        
        features.append(1.0)  # 主场标志
        return features

    def train(self, matches: List[Dict]):
        """训练模型"""
        if not matches:
            return
        
        X = []
        y = []
        label_map = {'home': 0, 'draw': 1, 'away': 2}
        
        for match in matches:
            features = self.extract_features(match.get('features', {}))
            result = match.get('result', 'draw')
            X.append(features)
            y.append(label_map.get(result, 1))
        
        X = np.array(X)
        y_encoded = np.array(y)
        
        self.feature_names = [
            'elo_home', 'elo_away', 'elo_diff',
            'euro_home', 'euro_draw', 'euro_away',
            'asian_handicap', 'asian_home_water', 'asian_away_water',
            'total_line', 'total_over_water', 'total_under_water',
            'home_attack', 'home_defense', 'away_attack', 'away_defense',
            'home_form', 'away_form', 'form_diff', 'is_home'
        ]
        
        if self.model_type != 'catboost' and self.scaler is None and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        self.model = self._create_model()
        if self.model:
            self.model.fit(X, y_encoded)
            self.is_trained = True

    def predict(self, match_data: Dict) -> Dict[str, float]:
        """预测比赛结果"""
        if not self.is_trained or self.model is None:
            return {'home': 0.333, 'draw': 0.334, 'away': 0.333}
        
        try:
            features = self.extract_features(match_data)
            X = np.array([features])
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            probs = self.model.predict_proba(X)[0]
            return {'home': float(probs[0]), 'draw': float(probs[1]), 'away': float(probs[2])}
        except Exception:
            return {'home': 0.333, 'draw': 0.334, 'away': 0.333}

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            return
        model_data = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'model': self.model,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model_type = model_data.get('model_type', 'randomforest')
            self.is_trained = model_data.get('is_trained', False)
            self.feature_names = model_data.get('feature_names', [])
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
        except FileNotFoundError:
            self.is_trained = False