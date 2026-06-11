#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态权重模块 - Meta Model实现
==============================

功能：
1. 使用机器学习模型（Meta Model）根据比赛特征预测最优权重
2. 特征包括：联赛、让球深度、欧赔离散度、凯利离散度、盘口变化次数、ELO差、总进球盘口
3. 输出：market_weight, team_weight, elo_weight, ml_weight

这是一个真正的动态权重系统，而非固定规则
"""

import pickle
import numpy as np
from typing import Dict, Tuple, Optional

# 尝试导入机器学习库
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


# 支持的联赛列表（用于编码）
LEAGUES = [
    '英超', '英冠', '英甲', '英乙',
    '西甲', '西乙',
    '德甲', '德乙',
    '意甲', '意乙',
    '法甲', '法乙',
    '西甲',
    '荷甲', '荷乙',
    '葡超',
    '俄超',
    '土超',
    '巴甲', '巴乙',
    '阿甲',
    '日职联', '日乙',
    'K联赛',
    '中超',
    '澳超',
    '美职联',
    '墨超',
    '欧冠', '欧联', '欧协联',
    '世界杯', '欧洲杯', '美洲杯', '亚洲杯',
    '其他'
]


class MetaWeightModel:
    """
    Meta模型 - 根据比赛特征预测最优权重
    
    输入特征：
    1. league_encoded - 联赛编码
    2. handicap_depth - 让球深度（绝对值）
    3. euro_std - 欧赔离散度
    4. kelly_std - 凯利离散度
    5. odds_changes - 盘口变化次数
    6. elo_diff - ELO差值
    7. total_line - 总进球盘口
    
    输出：
    [market_weight, team_weight, elo_weight, ml_weight]
    """
    
    def __init__(self, model_type: str = 'auto'):
        self.model_type = model_type.lower()
        self.models = {}  # 四个输出的模型
        self.league_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self._validate_model_type()
        
        # 默认权重（当模型不可用时使用）
        self.default_weights = {
            'market': 0.5,
            'team': 0.3,
            'elo': 0.2,
            'ml': 0.0
        }
    
    def _validate_model_type(self):
        """验证模型类型"""
        available_types = []
        if LIGHTGBM_AVAILABLE:
            available_types.append('lightgbm')
        if XGBOOST_AVAILABLE:
            available_types.append('xgboost')
        
        if self.model_type not in available_types:
            if available_types:
                self.model_type = available_types[0]
            else:
                self.model_type = 'none'
    
    def _create_regressor(self):
        """创建回归模型实例"""
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=64,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        return None
    
    def _extract_features(self, match_data: Dict) -> np.ndarray:
        """
        从比赛数据中提取特征
        
        参数：
            match_data: 包含比赛特征的字典
            
        返回：
            特征向量 (7,)
        """
        features = []
        
        # 1. 联赛编码
        league = match_data.get('league', '其他')
        if self.league_encoder is not None:
            try:
                features.append(float(self.league_encoder.transform([league])[0]))
            except ValueError:
                features.append(float(len(self.league_encoder.classes_)))
        else:
            features.append(float(LEAGUES.index(league) if league in LEAGUES else len(LEAGUES)))
        
        # 2. 让球深度（绝对值）
        handicap = match_data.get('handicap', 0)
        features.append(abs(float(handicap)))
        
        # 3. 欧赔离散度
        euro_std = match_data.get('euro_std', 0.05)
        features.append(float(euro_std))
        
        # 4. 凯利离散度
        kelly_std = match_data.get('kelly_std', 0.02)
        features.append(float(kelly_std))
        
        # 5. 盘口变化次数
        odds_changes = match_data.get('odds_changes', 3)
        features.append(float(odds_changes))
        
        # 6. ELO差值（绝对值）
        elo_diff = match_data.get('elo_diff', 0)
        features.append(abs(float(elo_diff)))
        
        # 7. 总进球盘口
        total_line = match_data.get('total_line', 2.5)
        features.append(float(total_line))
        
        return np.array(features)
    
    def train(self, train_data: List[Dict]):
        """
        训练Meta模型
        
        参数：
            train_data: 训练数据列表，每个元素包含：
                       {
                           'features': {...},  # 比赛特征
                           'weights': {
                               'market': float,
                               'team': float,
                               'elo': float,
                               'ml': float
                           },
                           'performance': float  # 该权重下的预测准确率
                       }
        """
        if not train_data or not SKLEARN_AVAILABLE:
            return
        
        # 准备特征和目标
        X = []
        y_market = []
        y_team = []
        y_elo = []
        y_ml = []
        
        for data in train_data:
            features = self._extract_features(data['features'])
            weights = data['weights']
            
            X.append(features)
            y_market.append(weights.get('market', 0.5))
            y_team.append(weights.get('team', 0.3))
            y_elo.append(weights.get('elo', 0.2))
            y_ml.append(weights.get('ml', 0.0))
        
        X = np.array(X)
        y_market = np.array(y_market)
        y_team = np.array(y_team)
        y_elo = np.array(y_elo)
        y_ml = np.array(y_ml)
        
        # 拟合编码器和归一化器
        self.league_encoder.fit(LEAGUES)
        X[:, 1:] = self.scaler.fit_transform(X[:, 1:])
        
        # 为每个权重输出训练一个模型
        targets = {
            'market': y_market,
            'team': y_team,
            'elo': y_elo,
            'ml': y_ml
        }
        
        for weight_type, y in targets.items():
            model = self._create_regressor()
            if model:
                model.fit(X, y)
                self.models[weight_type] = model
        
        self.is_trained = True
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """
        根据比赛特征预测最优权重
        
        参数：
            match_data: 包含比赛特征的字典
            
        返回：
            {'market': float, 'team': float, 'elo': float, 'ml': float}
        """
        if not self.is_trained or not self.models:
            return self.default_weights.copy()
        
        try:
            # 提取特征
            features = self._extract_features(match_data)
            X = features.reshape(1, -1)
            
            # 归一化（跳过联赛编码）
            X[:, 1:] = self.scaler.transform(X[:, 1:])
            
            # 预测各权重
            weights = {}
            for weight_type, model in self.models.items():
                pred = model.predict(X)[0]
                # 限制在合理范围内
                weights[weight_type] = max(0.0, min(1.0, float(pred)))
            
            # 归一化权重和为1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = self.default_weights.copy()
            
            return weights
        
        except Exception as e:
            import logging
            log = logging.getLogger('football')
            log.error(f"Meta权重预测失败: {e}")
            return self.default_weights.copy()
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            return
        
        model_data = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'models': self.models,
            'league_encoder_classes': self.league_encoder.classes_ if self.league_encoder else [],
            'scaler_mean': self.scaler.mean_ if self.scaler else [],
            'scaler_scale': self.scaler.scale_ if self.scaler else []
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data.get('model_type', 'lightgbm')
            self.is_trained = model_data.get('is_trained', False)
            self.models = model_data.get('models', {})
            
            if SKLEARN_AVAILABLE:
                self.league_encoder = LabelEncoder()
                self.league_encoder.classes_ = model_data.get('league_encoder_classes', LEAGUES)
                
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(model_data.get('scaler_mean', [0]*6))
                self.scaler.scale_ = np.array(model_data.get('scaler_scale', [1]*6))
        
        except FileNotFoundError:
            self.is_trained = False


# ==================== 兼容旧接口的函数 ====================

def get_dynamic_weights(confidence: float = 0.5, match_data: Optional[Dict] = None) -> Tuple[float, float, float, float]:
    """
    获取动态权重（兼容旧接口，新增ML权重）
    
    参数：
        confidence: 置信度（0~1）- 备用方案
        match_data: 比赛特征数据 - 用于Meta模型
        
    返回：
        (market_weight, team_weight, elo_weight, ml_weight)
    """
    # 尝试使用Meta模型
    if match_data and hasattr(get_dynamic_weights, '_meta_model'):
        meta_model = getattr(get_dynamic_weights, '_meta_model')
        if meta_model.is_trained:
            weights = meta_model.predict(match_data)
            return weights['market'], weights['team'], weights['elo'], weights['ml']
    
    # 备用：基于置信度的固定规则（保持向后兼容）
    base_market, base_team, base_elo = 0.5, 0.3, 0.2
    high_market, high_team, high_elo = 0.7, 0.2, 0.1
    low_market, low_team, low_elo = 0.3, 0.4, 0.3
    ml_weight = 0.0  # 默认无ML权重
    
    if confidence >= 0.7:
        return high_market, high_team, high_elo, ml_weight
    elif confidence <= 0.3:
        return low_market, low_team, low_elo, ml_weight
    else:
        if confidence <= 0.5:
            t = (confidence - 0.3) / 0.2
            market = low_market + t * (base_market - low_market)
            team = low_team + t * (base_team - low_team)
            elo = low_elo + t * (base_elo - low_elo)
        else:
            t = (confidence - 0.5) / 0.2
            market = base_market + t * (high_market - base_market)
            team = base_team + t * (high_team - base_team)
            elo = base_elo + t * (high_elo - base_elo)
        
        return market, team, elo, ml_weight


def fuse_predictions(market_pred: Dict[str, float],
                    team_pred: Dict[str, float],
                    elo_pred: Dict[str, float],
                    ml_pred: Optional[Dict[str, float]] = None,
                    confidence: float = 0.5,
                    match_data: Optional[Dict] = None) -> Dict[str, float]:
    """
    根据动态权重融合多个预测源（支持ML预测）
    
    参数：
        market_pred: 市场数据预测
        team_pred: 球队实力预测
        elo_pred: ELO预测
        ml_pred: 机器学习预测（可选）
        confidence: 置信度（备用）
        match_data: 比赛特征数据（用于Meta模型）
    
    返回：
        融合后的预测
    """
    market_w, team_w, elo_w, ml_w = get_dynamic_weights(confidence, match_data)
    
    # 获取所有可能的比分
    all_scores = set(market_pred.keys()) | set(team_pred.keys()) | set(elo_pred.keys())
    if ml_pred:
        all_scores |= set(ml_pred.keys())
    
    fused = {}
    for score in all_scores:
        m_prob = market_pred.get(score, 0.0)
        t_prob = team_pred.get(score, 0.0)
        e_prob = elo_pred.get(score, 0.0)
        ml_prob = ml_pred.get(score, 0.0) if ml_pred else 0.0
        
        # 加权融合
        fused[score] = market_w * m_prob + team_w * t_prob + elo_w * e_prob + ml_w * ml_prob
    
    # 归一化
    total = sum(fused.values())
    if total > 0:
        return {k: v / total for k, v in fused.items()}
    return fused


# 全局Meta模型实例
_global_meta_model = MetaWeightModel()


def init_meta_model(model_path: Optional[str] = None):
    """
    初始化全局Meta模型
    
    参数：
        model_path: 预训练模型路径
    """
    global _global_meta_model
    
    if model_path:
        _global_meta_model.load_model(model_path)
    
    # 将Meta模型绑定到get_dynamic_weights函数
    get_dynamic_weights._meta_model = _global_meta_model
    
    return _global_meta_model