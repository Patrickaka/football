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

from __future__ import annotations

import pickle
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

# 机器学习库：只有 MetaWeightModel 用得到，而它是惰性构造的，所以这里只声明
# 名字，真正的 import 推到 _load_ml_libs()。三个库合计约 0.76 秒，而整个
# import src.football 原本才 0.85 秒——九成的导入代价花在这上面。
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SKLEARN_AVAILABLE = False

XGBRegressor = None
LGBMRegressor = None
StandardScaler = None
LabelEncoder = None
train_test_split = None
mean_squared_error = None

_ML_LIBS_LOADED = False


def _load_ml_libs():
    """真正要用 ML 库时才导入。

    语义与原来的模块级 try-import 完全一致：真的去 import，失败就把对应的
    *_AVAILABLE 留在 False。**只有时机变了**——从「import src.football」
    推迟到「第一次构造 MetaWeightModel」。

    幂等：失败过一次就不再重试，与模块级 import 只执行一次的行为对齐。
    """
    global _ML_LIBS_LOADED
    global XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE
    global XGBRegressor, LGBMRegressor
    global StandardScaler, LabelEncoder, train_test_split, mean_squared_error

    if _ML_LIBS_LOADED:
        return
    _ML_LIBS_LOADED = True

    try:
        from xgboost import XGBRegressor as _XGBRegressor
        XGBRegressor = _XGBRegressor
        XGBOOST_AVAILABLE = NUMPY_AVAILABLE
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor as _LGBMRegressor
        LGBMRegressor = _LGBMRegressor
        LIGHTGBM_AVAILABLE = NUMPY_AVAILABLE
    except Exception:
        pass

    try:
        from sklearn.preprocessing import StandardScaler as _StandardScaler
        from sklearn.preprocessing import LabelEncoder as _LabelEncoder
        from sklearn.model_selection import train_test_split as _train_test_split
        from sklearn.metrics import mean_squared_error as _mean_squared_error
        StandardScaler = _StandardScaler
        LabelEncoder = _LabelEncoder
        train_test_split = _train_test_split
        mean_squared_error = _mean_squared_error
        SKLEARN_AVAILABLE = NUMPY_AVAILABLE
    except Exception:
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
    
    训练方式：
    - 为每场比赛生成多个候选权重组合
    - 用真实结果计算各组合的LogLoss/Brier
    - 选择最优组合作为训练标签
    - 按时间切分训练/验证集
    """
    
    def __init__(self, model_type: str = 'auto'):
        _load_ml_libs()
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
        
        # 候选权重组合生成器配置
        self.weight_candidates = [
            {'market': 0.6, 'team': 0.25, 'elo': 0.15, 'ml': 0.0},
            {'market': 0.5, 'team': 0.35, 'elo': 0.15, 'ml': 0.0},
            {'market': 0.4, 'team': 0.4, 'elo': 0.2, 'ml': 0.0},
            {'market': 0.7, 'team': 0.2, 'elo': 0.1, 'ml': 0.0},
            {'market': 0.3, 'team': 0.5, 'elo': 0.2, 'ml': 0.0},
            {'market': 0.55, 'team': 0.3, 'elo': 0.15, 'ml': 0.0},
            {'market': 0.45, 'team': 0.35, 'elo': 0.2, 'ml': 0.0},
            {'market': 0.5, 'team': 0.3, 'elo': 0.2, 'ml': 0.0},
        ]
    
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
    
    def _calculate_logloss(self, predictions: Dict[str, float], actual_result: str) -> float:
        """
        计算多分类LogLoss
        
        参数：
            predictions: {结果: 概率} 字典
            actual_result: 实际结果
        
        返回：
            LogLoss值（越小越好）
        """
        prob = predictions.get(actual_result, 1e-15)
        return -math.log(max(prob, 1e-15))
    
    def _calculate_brier(self, predictions: Dict[str, float], actual_result: str) -> float:
        """
        计算多分类Brier Score
        
        参数：
            predictions: {结果: 概率} 字典
            actual_result: 实际结果
        
        返回：
            Brier Score值（越小越好）
        """
        brier = 0.0
        for result, prob in predictions.items():
            target = 1.0 if result == actual_result else 0.0
            brier += (prob - target) ** 2
        return brier
    
    def fuse_with_weights(self, market_pred: Dict[str, float], team_pred: Dict[str, float],
                          elo_pred: Dict[str, float], ml_pred: Dict[str, float],
                          weights: Dict[str, float]) -> Dict[str, float]:
        """
        使用指定权重融合多个预测源
        
        参数：
            market_pred: 市场预测
            team_pred: 球队预测
            elo_pred: ELO预测
            ml_pred: ML预测
            weights: 权重组合
        
        返回：
            融合后的预测分布
        """
        fused = {}
        all_keys = set(market_pred.keys()) | set(team_pred.keys()) | set(elo_pred.keys())
        
        if ml_pred:
            all_keys |= set(ml_pred.keys())
        
        for key in all_keys:
            fused[key] = (
                weights['market'] * market_pred.get(key, 0.0) +
                weights['team'] * team_pred.get(key, 0.0) +
                weights['elo'] * elo_pred.get(key, 0.0) +
                weights['ml'] * ml_pred.get(key, 0.0) if ml_pred else 0.0
            )
        
        # 归一化
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}
        
        return fused
    
    def find_best_weight_combination(self, market_pred: Dict[str, float],
                                     team_pred: Dict[str, float], elo_pred: Dict[str, float],
                                     ml_pred: Dict[str, float], actual_result: str,
                                     metric: str = 'brier') -> Dict[str, float]:
        """
        为单场比赛找到最优权重组合
        
        参数：
            market_pred: 市场预测分布
            team_pred: 球队预测分布
            elo_pred: ELO预测分布
            ml_pred: ML预测分布
            actual_result: 实际结果
            metric: 'brier' 或 'logloss'
        
        返回：
            最优权重组合
        """
        best_weights = None
        best_score = float('inf')
        
        for weights in self.weight_candidates:
            fused = self.fuse_with_weights(market_pred, team_pred, elo_pred, ml_pred, weights)
            
            if metric == 'logloss':
                score = self._calculate_logloss(fused, actual_result)
            else:
                score = self._calculate_brier(fused, actual_result)
            
            if score < best_score:
                best_score = score
                best_weights = weights.copy()
        
        return best_weights or self.default_weights.copy()
    
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
    
    def train(self, train_data: List[Dict], auto_select_weights: bool = True, 
              train_ratio: float = 0.8, metric: str = 'brier') -> Dict:
        """
        训练Meta模型（支持自动寻找最优权重）
        
        参数：
            train_data: 训练数据列表，每个元素包含：
                       {
                           'match_time': 'YYYY-MM-DD HH:MM',  # 比赛时间（用于时间切分）
                           'features': {...},  # 比赛特征
                           'market_pred': {...},  # 市场预测分布
                           'team_pred': {...},    # 球队预测分布
                           'elo_pred': {...},     # ELO预测分布
                           'ml_pred': {...},      # ML预测分布（可选）
                           'actual_result': str,  # 实际结果
                           # 以下为旧格式支持（已不推荐）
                           'weights': {
                               'market': float,
                               'team': float,
                               'elo': float,
                               'ml': float
                           }
                       }
            auto_select_weights: 是否自动为每场比赛选择最优权重组合
            train_ratio: 训练集比例（前train_ratio用于训练，后(1-train_ratio)用于验证）
            metric: 评估指标 'brier' 或 'logloss'
        
        返回：
            训练结果统计 {'train_samples': int, 'val_samples': int, 'val_metrics': {...}}
        """
        if not train_data or not SKLEARN_AVAILABLE:
            return {'error': '训练数据不足或SKLearn不可用'}
        
        # 按时间排序（确保时间切分正确）
        train_data.sort(key=lambda x: x.get('match_time', ''))
        
        # 时间切分训练/验证集
        split_idx = int(len(train_data) * train_ratio)
        train_set = train_data[:split_idx]
        val_set = train_data[split_idx:]
        
        # 准备训练数据
        X_train = []
        y_market_train = []
        y_team_train = []
        y_elo_train = []
        y_ml_train = []
        
        for data in train_set:
            features = self._extract_features(data['features'])
            
            if auto_select_weights and all(key in data for key in 
                                          ['market_pred', 'team_pred', 'elo_pred', 'actual_result']):
                # 自动寻找最优权重组合
                best_weights = self.find_best_weight_combination(
                    data['market_pred'],
                    data['team_pred'],
                    data['elo_pred'],
                    data.get('ml_pred'),
                    data['actual_result'],
                    metric
                )
            else:
                # 使用已有权重（向后兼容）
                best_weights = data.get('weights', self.default_weights)
            
            X_train.append(features)
            y_market_train.append(best_weights.get('market', 0.5))
            y_team_train.append(best_weights.get('team', 0.3))
            y_elo_train.append(best_weights.get('elo', 0.2))
            y_ml_train.append(best_weights.get('ml', 0.0))
        
        X_train = np.array(X_train)
        y_market_train = np.array(y_market_train)
        y_team_train = np.array(y_team_train)
        y_elo_train = np.array(y_elo_train)
        y_ml_train = np.array(y_ml_train)
        
        # 拟合编码器和归一化器
        self.league_encoder.fit(LEAGUES)
        X_train[:, 1:] = self.scaler.fit_transform(X_train[:, 1:])
        
        # 为每个权重输出训练一个模型
        targets = {
            'market': y_market_train,
            'team': y_team_train,
            'elo': y_elo_train,
            'ml': y_ml_train
        }
        
        for weight_type, y in targets.items():
            model = self._create_regressor()
            if model:
                model.fit(X_train, y)
                self.models[weight_type] = model
        
        self.is_trained = True
        
        # 验证集评估
        val_results = self._evaluate(val_set, metric)
        
        return {
            'train_samples': len(train_set),
            'val_samples': len(val_set),
            'val_metrics': val_results
        }
    
    def _evaluate(self, val_data: List[Dict], metric: str = 'brier') -> Dict:
        """
        在验证集上评估模型
        
        参数：
            val_data: 验证数据列表
            metric: 'brier' 或 'logloss'
        
        返回：
            评估指标字典
        """
        if not val_data or not self.is_trained:
            return {}
        
        total_score = 0.0
        correct_count = 0
        
        for data in val_data:
            # 预测权重
            weights = self.predict(data['features'])
            
            # 融合预测
            fused = self.fuse_with_weights(
                data['market_pred'],
                data['team_pred'],
                data['elo_pred'],
                data.get('ml_pred'),
                weights
            )
            
            # 计算指标
            actual = data['actual_result']
            if metric == 'logloss':
                score = self._calculate_logloss(fused, actual)
            else:
                score = self._calculate_brier(fused, actual)
            
            total_score += score
            
            # 判断是否命中Top1
            if fused:
                pred_top1 = max(fused.items(), key=lambda x: x[1])[0]
                if pred_top1 == actual:
                    correct_count += 1
        
        return {
            'avg_score': total_score / len(val_data) if val_data else 0,
            'accuracy': correct_count / len(val_data) if val_data else 0
        }
    
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
    
    # 检查 ML 是否有资格参与融合
    try:
        from .result_sync import get_history, check_ml_fusion_eligibility, get_ml_fusion_weight
        import src.football.ml as ml_module
        
        # 确保模型已加载并获取测试集样本数
        ml_module.load_trained_ml_model()
        test_set_samples = ml_module._trained_ml_metadata.get('test_count', 0) if ml_module._trained_ml_metadata else 0
        
        history = get_history()
        ml_stats = history.get_ml_evaluation_stats()
        eligibility = check_ml_fusion_eligibility(ml_stats, test_set_samples)
        
        if eligibility['eligible']:
            shadow_samples = eligibility['shadow_samples']
            ml_weight = get_ml_fusion_weight(True, shadow_samples, 0.0)
        else:
            ml_weight = 0.0
    except Exception:
        ml_weight = 0.0  # 默认无ML权重
    
    # 根据置信度确定基础权重
    if confidence >= 0.7:
        market_w, team_w, elo_w = high_market, high_team, high_elo
    elif confidence <= 0.3:
        market_w, team_w, elo_w = low_market, low_team, low_elo
    else:
        if confidence <= 0.5:
            t = (confidence - 0.3) / 0.2
            market_w = low_market + t * (base_market - low_market)
            team_w = low_team + t * (base_team - low_team)
            elo_w = low_elo + t * (base_elo - low_elo)
        else:
            t = (confidence - 0.5) / 0.2
            market_w = base_market + t * (high_market - base_market)
            team_w = base_team + t * (high_team - base_team)
            elo_w = base_elo + t * (high_elo - base_elo)
    
    # 如果有ML权重，需要从其他权重中按比例扣除
    if ml_weight > 0 and ml_weight < 1.0:
        # 计算其他权重的总和
        base_total = market_w + team_w + elo_w
        if base_total > 0:
            # 按比例缩减其他权重，腾出ML权重空间
            scale_factor = (1.0 - ml_weight) / base_total
            market_w *= scale_factor
            team_w *= scale_factor
            elo_w *= scale_factor
    
    return market_w, team_w, elo_w, ml_weight


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


# 全局Meta模型实例（惰性）：构造它会拉起三个 ML 库，而线上从来走不到这里
# ——init_meta_model 在全仓没有任何调用者，所以 get_dynamic_weights._meta_model
# 永远不存在，Meta 模型那条分支恒为假。这一条留给 F-1 的活死清单定性，
# 本批只把导入代价挪走，不删任何东西。
_META_MODEL = None


def _meta_model():
    """首次访问时才构造全局 Meta 模型"""
    global _META_MODEL
    if _META_MODEL is None:
        _META_MODEL = MetaWeightModel()
    return _META_MODEL


def __getattr__(name):
    """保住 `dynamic_weights._global_meta_model` 这个既有写法（PEP 562）"""
    if name == '_global_meta_model':
        return _meta_model()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def init_meta_model(model_path: Optional[str] = None):
    """
    初始化全局Meta模型
    
    参数：
        model_path: 预训练模型路径
    """
    model = _meta_model()

    if model_path:
        model.load_model(model_path)

    # 将Meta模型绑定到get_dynamic_weights函数
    get_dynamic_weights._meta_model = model

    return model
