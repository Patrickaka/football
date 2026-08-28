#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML模型训练器
=============

功能：
1. 加载构建好的训练数据
2. 按时间切分训练/验证/测试集
3. 训练CatBoost分类模型
4. 评估模型性能
5. 保存模型和元数据
"""

import os
import json
import pickle
import math
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

from ..common import kv_store

from ..domain.sports.football import ml_contract as _mlc

# 纯计算转发给领域层
split_by_time = _mlc.split_by_time


# ==================== 常量配置 ====================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'ml_training_data.jsonl')
MODEL_FILE = os.path.join(DATA_DIR, 'ml_model.pkl')
SCALER_FILE = os.path.join(DATA_DIR, 'ml_scaler.pkl')
METADATA_FILE = os.path.join(DATA_DIR, 'ml_metadata.json')

# 训练/验证/测试集比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 目标标签映射
LABEL_MAP = {'H': 0, 'D': 1, 'A': 2}
REVERSE_LABEL_MAP = {0: 'H', 1: 'D', 2: 'A'}


# ==================== 工具函数 ====================

def load_training_data(filepath: str) -> List[Dict]:
    """
    加载JSONL格式的训练数据
    
    参数：
        filepath: 训练数据文件路径
    
    返回：
        训练样本列表
    """
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except Exception as e:
                    print(f"解析JSON失败: {e}")
    
    # 按日期排序
    samples.sort(key=lambda x: x['match_date'])
    return samples




def prepare_features_target(samples: List[Dict], feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备特征矩阵和目标向量
    
    参数：
        samples: 样本列表
        feature_names: 特征名称列表
    
    返回：
        (特征矩阵, 目标向量)
    """
    X = []
    y = []
    
    for sample in samples:
        features = sample['features']
        target = sample['target']['result']
        
        # 构建特征向量
        feature_vec = []
        for name in feature_names:
            feature_vec.append(features.get(name, 0.0))
        
        X.append(feature_vec)
        y.append(LABEL_MAP.get(target, 1))  # 默认平局
    
    return np.array(X), np.array(y)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
    """
    计算评估指标
    
    参数：
        y_true: 真实标签
        y_pred: 预测标签
        y_proba: 预测概率（可选）
    
    返回：
        指标字典
    """
    # 准确率
    accuracy = np.mean(y_true == y_pred)
    
    # LogLoss
    logloss = 0.0
    if y_proba is not None:
        eps = 1e-15
        y_proba = np.clip(y_proba, eps, 1 - eps)
        logloss = -np.mean(np.log(y_proba[np.arange(len(y_true)), y_true]))
    
    # Brier Score（多分类版本）
    brier = 0.0
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        one_hot = np.zeros((len(y_true), n_classes))
        one_hot[np.arange(len(y_true)), y_true] = 1
        brier = np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))
    
    return {
        'accuracy': accuracy,
        'logloss': logloss,
        'brier': brier
    }


# ==================== 模型训练器 ====================

class MLModelTrainer:
    """ML模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metadata = {}
    
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              feature_names: List[str]) -> Dict:
        """
        训练模型
        
        参数：
            train_data: 训练数据
            val_data: 验证数据
            feature_names: 特征名称列表
        
        返回：
            训练指标
        """
        self.feature_names = feature_names
        
        # 准备数据
        X_train, y_train = prepare_features_target(train_data, feature_names)
        X_val, y_val = prepare_features_target(val_data, feature_names)
        
        # 尝试导入CatBoost
        try:
            from catboost import CatBoostClassifier, Pool
            
            print("使用CatBoost训练...")
            
            # 创建数据集
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_val, y_val)
            
            # 定义模型参数
            params = {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'early_stopping_rounds': 50,
                'verbose': 100,
                'random_seed': 42,
            }
            
            # 训练模型
            self.model = CatBoostClassifier(**params)
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True
            )
            
            # 评估验证集
            val_preds = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)
            val_metrics = calculate_metrics(y_val, val_preds, val_proba)
            
            print(f"\n验证集指标:")
            print(f"  准确率: {val_metrics['accuracy']:.4f}")
            print(f"  LogLoss: {val_metrics['logloss']:.4f}")
            print(f"  Brier: {val_metrics['brier']:.4f}")
            
            return val_metrics
        
        except ImportError:
            # CatBoost不可用，尝试LightGBM
            try:
                from lightgbm import LGBMClassifier
                
                print("CatBoost不可用，使用LightGBM训练...")
                
                self.model = LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    verbose=100,
                    random_state=42,
                )
                
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=100
                )
                
                val_preds = self.model.predict(X_val)
                val_proba = self.model.predict_proba(X_val)
                val_metrics = calculate_metrics(y_val, val_preds, val_proba)
                
                print(f"\n验证集指标:")
                print(f"  准确率: {val_metrics['accuracy']:.4f}")
                print(f"  LogLoss: {val_metrics['logloss']:.4f}")
                print(f"  Brier: {val_metrics['brier']:.4f}")
                
                return val_metrics
            
            except ImportError:
                # 尝试XGBoost
                try:
                    from xgboost import XGBClassifier
                    
                    print("LightGBM不可用，使用XGBoost训练...")
                    
                    self.model = XGBClassifier(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=6,
                        verbosity=1,
                        random_state=42,
                    )
                    
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=100
                    )
                    
                    val_preds = self.model.predict(X_val)
                    val_proba = self.model.predict_proba(X_val)
                    val_metrics = calculate_metrics(y_val, val_preds, val_proba)
                    
                    print(f"\n验证集指标:")
                    print(f"  准确率: {val_metrics['accuracy']:.4f}")
                    print(f"  LogLoss: {val_metrics['logloss']:.4f}")
                    print(f"  Brier: {val_metrics['brier']:.4f}")
                    
                    return val_metrics
                
                except ImportError:
                    raise ImportError("请安装CatBoost、LightGBM或XGBoost")
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        在测试集上评估模型
        
        参数：
            test_data: 测试数据
        
        返回：
            测试指标
        """
        if not self.model:
            return {'error': '模型未训练'}
        
        X_test, y_test = prepare_features_target(test_data, self.feature_names)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        print(f"\n测试集指标:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  LogLoss: {metrics['logloss']:.4f}")
        print(f"  Brier: {metrics['brier']:.4f}")
        
        return metrics
    
    def save(self, test_metrics: Dict = None):
        """
        保存模型和元数据
        
        参数：
            test_metrics: 测试集指标
        """
        if not self.model:
            print("警告：模型未训练，无法保存")
            return
        
        # 确保目录存在
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # 保存模型
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存到: {MODEL_FILE}")
        
        # 保存元数据
        from .ml_feature_schema import FEATURE_VERSION

        dataset_sha256 = None
        if os.path.exists(TRAINING_DATA_FILE):
            digest = hashlib.sha256()
            with open(TRAINING_DATA_FILE, 'rb') as dataset:
                for chunk in iter(lambda: dataset.read(1024 * 1024), b''):
                    digest.update(chunk)
            dataset_sha256 = digest.hexdigest()

        self.metadata = {
            'model_version': f"ml-{FEATURE_VERSION}-{datetime.now().strftime('%Y%m%d')}",
            'model_type': type(self.model).__name__,
            'trained_at': datetime.now().isoformat(),
            'feature_version': FEATURE_VERSION,
            'features': self.feature_names,
            'train_count': self.metadata.get('train_count', 0),
            'validation_count': self.metadata.get('validation_count', 0),
            'test_count': self.metadata.get('test_count', 0),
            'metrics': test_metrics or {},
            'dataset': {
                'path': os.path.basename(TRAINING_DATA_FILE),
                'sha256': dataset_sha256,
                'split_method': 'chronological-70-15-15',
            },
        }
        
        kv_store.save('ml_metadata', self.metadata)
        # Keep a portable sidecar next to the pickle even when MySQL/KV storage
        # is unavailable. It is part of the deployable model artifact.
        temp_metadata = METADATA_FILE + '.tmp'
        with open(temp_metadata, 'w', encoding='utf-8') as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)
        os.replace(temp_metadata, METADATA_FILE)
        print("元数据已保存到 MySQL kv_store: ml_metadata")
    
    def load(self) -> bool:
        """
        加载已训练的模型
        
        返回：
            是否加载成功
        """
        if not os.path.exists(MODEL_FILE):
            print(f"模型文件不存在: {MODEL_FILE}")
            return False
        
        try:
            with open(MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型加载成功")
            
            _meta = kv_store.load('ml_metadata')
            if _meta is None and os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, encoding='utf-8') as handle:
                    _meta = json.load(handle)
            if _meta is not None:
                self.metadata = _meta
                self.feature_names = self.metadata.get('features', [])
                print(f"元数据加载成功")
            
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def predict(self, features: Dict) -> Dict:
        """
        预测单场比赛
        
        参数：
            features: 特征字典
        
        返回：
            预测结果
        """
        if not self.model:
            return {
                'available': False,
                'reason': 'model_not_trained'
            }
        
        # 构建特征向量
        feature_vec = []
        for name in self.feature_names:
            feature_vec.append(features.get(name, 0.0))
        
        # 预测
        X = np.array([feature_vec])
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0]
        
        # 归一化概率
        total = y_proba.sum()
        if total > 0:
            y_proba = y_proba / total
        
        return {
            'H': float(y_proba[0]),
            'D': float(y_proba[1]),
            'A': float(y_proba[2]),
            'predicted_label': REVERSE_LABEL_MAP.get(y_pred, 'D'),
            'model_version': self.metadata.get('model_version', 'unknown'),
            'available': True
        }


# ==================== 主函数 ====================

def main():
    """主函数"""
    from datetime import datetime
    
    # 加载训练数据
    print("加载训练数据...")
    samples = load_training_data(TRAINING_DATA_FILE)
    
    if not samples:
        print(f"没有找到训练数据: {TRAINING_DATA_FILE}")
        return
    
    # 获取特征名称
    from .ml_feature_schema import get_feature_names
    feature_names = get_feature_names()
    
    # 按时间切分数据集
    train_set, val_set, test_set = split_by_time(samples, TRAIN_RATIO, VAL_RATIO)
    
    # 创建训练器
    trainer = MLModelTrainer()
    
    # 记录样本数量
    trainer.metadata['train_count'] = len(train_set)
    trainer.metadata['validation_count'] = len(val_set)
    trainer.metadata['test_count'] = len(test_set)
    
    # 训练模型
    print("\n开始训练模型...")
    val_metrics = trainer.train(train_set, val_set, feature_names)
    
    # 评估测试集
    print("\n评估测试集...")
    test_metrics = trainer.evaluate(test_set)
    
    # 保存模型
    print("\n保存模型...")
    trainer.save(test_metrics)
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()