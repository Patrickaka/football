#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球机器学习模块
================

包含：
1. Dixon-Coles 模型（改进泊松分布）
2. 进球数推荐
3. 机器学习预测器（CatBoost/XGBoost/LightGBM）
4. 盘口时序神经网络（LSTM/XGBoost/LightGBM时序模型）
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
PYTORCH_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
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

def recommend_goal_counts_from_dist(goal_dist: Dict[int, float], top_n: int = 2) -> List[Dict]:
    """从进球数分布字典推荐概率最大的进球数"""
    sorted_counts = sorted(goal_dist.items(), key=lambda x: -x[1])
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

def predict_goal_counts_from_candidates(candidates: List[Tuple], max_goals: int = 7, asian=None, total=None) -> Dict:
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
    
    # 获取进球数分布（字典格式）
    goal_dist = calculate_goal_counts(prob_matrix)
    
    # ========== 新增：结合历史盘口数据调整进球数分布 ==========
    if asian and total:
        try:
            from .market_db import MarketScoreDB
            
            handicap = asian.get('handicap', 0)
            close_line = total.get('close_line', 2.5)
            
            db = MarketScoreDB()
            db.load()
            
            market_goal_dist = db.get_goal_count_dist(handicap, close_line)
            
            if market_goal_dist and len(market_goal_dist) >= 3:
                # 融合模型分布和历史盘口分布（60%模型 + 40%历史）
                blended_dist = {}
                all_keys = set(goal_dist.keys()).union(set(market_goal_dist.keys()))
                
                for key in all_keys:
                    model_prob = goal_dist.get(key, 0.001)
                    market_prob = market_goal_dist.get(key, 0.001)
                    blended_dist[key] = 0.6 * model_prob + 0.4 * market_prob
                
                # 归一化
                blended_total = sum(blended_dist.values())
                if blended_total > 0:
                    goal_dist = {k: v / blended_total for k, v in sorted(blended_dist.items())}
                
                import logging
                log = logging.getLogger('football')
                log.info(f"进球数分布已结合历史盘口数据调整")
        except Exception as e:
            import logging
            log = logging.getLogger('football')
            log.debug(f"无法加载历史盘口数据调整进球数分布: {e}")
    
    # 基于调整后的分布重新计算推荐
    return {
        'recommendations': recommend_goal_counts_from_dist(goal_dist, top_n=2),
        'distribution': get_goal_count_distribution(prob_matrix),
        'over_under': {'over': sum(v for k, v in goal_dist.items() if k >= 3),
                       'under': sum(v for k, v in goal_dist.items() if k <= 2)},
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


# ==================== 盘口时序神经网络 ====================

class OddsSequenceProcessor:
    """盘口时序数据处理器"""
    
    TIME_POINTS = ['T-48h', 'T-24h', 'T-12h', 'T-6h', 'T-1h']
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if SKLEARN_AVAILABLE else None
        self.is_fitted = False
    
    def create_sequence(self, odds_history: List[Dict]) -> np.ndarray:
        """
        从盘口历史数据创建时序序列
        
        序列结构（每个时间点5个特征）：
        [主胜赔率, 平局赔率, 客胜赔率, 亚盘, 水位]
        
        参数:
            odds_history: 包含各时间点盘口数据的列表
                        每个元素为 {'time': 'T-XXh', 'euro': {...}, 'asian': {...}}
        
        返回:
            shape: (时间点数, 特征数) = (5, 5)
        """
        sequence = np.zeros((len(self.TIME_POINTS), 5))
        
        for i, time_point in enumerate(self.TIME_POINTS):
            data = next((h for h in odds_history if h.get('time') == time_point), None)
            if data:
                euro = data.get('euro', {})
                asian = data.get('asian', {})
                
                sequence[i, 0] = euro.get('home', 2.5)      # 主胜赔率
                sequence[i, 1] = euro.get('draw', 3.0)      # 平局赔率
                sequence[i, 2] = euro.get('away', 3.0)      # 客胜赔率
                sequence[i, 3] = asian.get('handicap', 0)   # 亚盘
                sequence[i, 4] = asian.get('water', 0.95)   # 水位
        
        return sequence
    
    def fit_scaler(self, sequences: List[np.ndarray]):
        """拟合归一化器"""
        if not SKLEARN_AVAILABLE:
            return
        
        all_data = np.vstack([seq.flatten() for seq in sequences])
        self.scaler.fit(all_data)
        self.is_fitted = True
    
    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """归一化序列"""
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return sequence
        
        flat = sequence.flatten().reshape(1, -1)
        scaled = self.scaler.transform(flat)
        return scaled.reshape(sequence.shape)
    
    def fit_transform(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """拟合并转换"""
        self.fit_scaler(sequences)
        return [self.transform(seq) for seq in sequences]


# 仅在 PyTorch 可用时定义 OddsLSTM 类
if PYTORCH_AVAILABLE:
    class OddsLSTM(nn.Module):
        """盘口时序LSTM神经网络"""
        
        def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=14):
            super(OddsLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.3)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, output_size)
            )
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


class MarketTimingPredictor:
    """
    盘口时序预测器
    
    支持：
    1. LightGBM - 提取时序特征后用GBDT预测
    2. XGBoost - 梯度提升树
    3. LSTM - 长短期记忆网络
    
    预测目标：真实比分类别（14种常见比分）
    """
    
    SCORE_LABELS = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1), (1, 3), (2, 3), (3, 2)
    ]
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type.lower()
        self.model = None
        self.sequence_processor = OddsSequenceProcessor()
        self.scaler = None
        self.is_trained = False
        self._validate_model_type()
    
    def _validate_model_type(self):
        """验证模型类型"""
        available_types = []
        if LIGHTGBM_AVAILABLE:
            available_types.append('lightgbm')
        if XGBOOST_AVAILABLE:
            available_types.append('xgboost')
        if PYTORCH_AVAILABLE:
            available_types.append('lstm')
        
        if self.model_type not in available_types:
            if available_types:
                self.model_type = available_types[0]
            else:
                self.model_type = 'none'
    
    def _extract_temporal_features(self, sequence: np.ndarray) -> List[float]:
        """
        从时序序列中提取特征用于传统ML模型
        
        特征包括：
        - 各时间点的原始特征
        - 趋势特征（变化率）
        - 波动率特征
        - 差分特征
        """
        features = []
        
        # 原始特征
        features.extend(sequence.flatten().tolist())
        
        # 趋势特征（从T-48h到T-1h的变化）
        if sequence.shape[0] >= 2:
            first = sequence[0]
            last = sequence[-1]
            features.extend((last - first).tolist())
            
            # 变化率
            for i in range(len(first)):
                if first[i] != 0:
                    features.append((last[i] - first[i]) / abs(first[i]))
                else:
                    features.append(0)
        
        # 波动率特征
        std_features = np.std(sequence, axis=0)
        features.extend(std_features.tolist())
        
        # 斜率特征（线性回归斜率）
        if sequence.shape[0] >= 3:
            x = np.arange(sequence.shape[0])
            for i in range(sequence.shape[1]):
                y = sequence[:, i]
                if np.std(y) > 0:
                    slope, _ = np.polyfit(x, y, 1)
                    features.append(slope)
                else:
                    features.append(0)
        
        return features
    
    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=64,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'lstm' and PYTORCH_AVAILABLE:
            return OddsLSTM(input_size=5, hidden_size=128, num_layers=2, 
                           output_size=len(self.SCORE_LABELS))
        return None
    
    def _score_to_label(self, h_goals: int, a_goals: int) -> int:
        """将比分转换为标签"""
        try:
            return self.SCORE_LABELS.index((h_goals, a_goals))
        except ValueError:
            return len(self.SCORE_LABELS) // 2  # 返回中间类别
    
    def _label_to_score(self, label: int) -> Tuple[int, int]:
        """将标签转换为比分"""
        if 0 <= label < len(self.SCORE_LABELS):
            return self.SCORE_LABELS[label]
        return (1, 1)  # 默认平局
    
    def train(self, train_data: List[Dict], epochs: int = 50, batch_size: int = 32):
        """
        训练模型
        
        参数:
            train_data: 训练数据列表，每个元素包含：
                       {'sequence': 时序序列, 'result': (h_goals, a_goals)}
            epochs: 训练轮数（仅LSTM）
            batch_size: 批次大小（仅LSTM）
        """
        if not train_data:
            return
        
        # 准备数据
        sequences = [d['sequence'] for d in train_data]
        labels = [self._score_to_label(d['result'][0], d['result'][1]) for d in train_data]
        
        # 预处理
        sequences = self.sequence_processor.fit_transform(sequences)
        
        if self.model_type == 'lstm' and PYTORCH_AVAILABLE:
            # LSTM训练
            X = np.array(sequences)
            y = np.array(labels)
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
            
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            self.model = self._create_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            best_val_acc = 0
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证
                self.model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = self.model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_acc = val_correct / val_total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            self.is_trained = True
        
        else:
            # LightGBM/XGBoost训练
            X = np.array([self._extract_temporal_features(seq) for seq in sequences])
            y = np.array(labels)
            
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            
            self.model = self._create_model()
            if self.model:
                self.model.fit(X, y)
                self.is_trained = True
    
    def predict(self, sequence: np.ndarray) -> Dict:
        """
        预测比赛结果
        
        参数:
            sequence: 时序序列 (5, 5)
        
        返回:
            包含各比分概率的字典
        """
        if not self.is_trained or self.model is None:
            return self._default_prediction()
        
        try:
            # 预处理
            sequence = self.sequence_processor.transform(sequence)
            
            if self.model_type == 'lstm' and PYTORCH_AVAILABLE:
                # LSTM预测
                self.model.eval()
                X = torch.FloatTensor(sequence).unsqueeze(0)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                X = X.to(device)
                self.model.to(device)
                
                with torch.no_grad():
                    outputs = self.model(X)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            else:
                # LightGBM/XGBoost预测
                features = self._extract_temporal_features(sequence)
                X = np.array([features])
                
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                
                probs = self.model.predict_proba(X)[0]
            
            # 构建结果
            result = {}
            total_prob = 0
            for i, (h, a) in enumerate(self.SCORE_LABELS):
                if i < len(probs):
                    result[(h, a)] = float(probs[i])
                    total_prob += float(probs[i])
            
            # 归一化
            if total_prob > 0:
                result = {k: v / total_prob for k, v in result.items()}
            
            return result
        
        except Exception as e:
            import logging
            log = logging.getLogger('football')
            log.error(f"盘口时序预测失败: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> Dict:
        """默认预测（均匀分布）"""
        result = {}
        prob = 1.0 / len(self.SCORE_LABELS)
        for h, a in self.SCORE_LABELS:
            result[(h, a)] = prob
        return result
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            return
        
        model_data = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'sequence_processor': self.sequence_processor,
            'scaler': self.scaler
        }
        
        if self.model_type == 'lstm' and PYTORCH_AVAILABLE:
            model_data['model_state_dict'] = self.model.state_dict()
            model_data['model_config'] = {
                'input_size': 5,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': len(self.SCORE_LABELS)
            }
        else:
            model_data['model'] = self.model
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data.get('model_type', 'lightgbm')
            self.is_trained = model_data.get('is_trained', False)
            self.sequence_processor = model_data.get('sequence_processor', OddsSequenceProcessor())
            self.scaler = model_data.get('scaler')
            
            if self.model_type == 'lstm' and PYTORCH_AVAILABLE:
                config = model_data.get('model_config', {})
                self.model = OddsLSTM(
                    input_size=config.get('input_size', 5),
                    hidden_size=config.get('hidden_size', 128),
                    num_layers=config.get('num_layers', 2),
                    output_size=config.get('output_size', len(self.SCORE_LABELS))
                )
                self.model.load_state_dict(model_data.get('model_state_dict'))
                self.model.eval()
            else:
                self.model = model_data.get('model')
        
        except FileNotFoundError:
            self.is_trained = False