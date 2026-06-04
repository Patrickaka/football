#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态阈值调整模块
基于滑动窗口实现置信度阈值的动态调整
"""

import os
import json
import logging
from collections import deque
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 阈值配置存储路径
THRESHOLD_CONFIG_PATH = 'dynamic_threshold_config.json'


class DynamicThresholdManager:
    """
    动态阈值管理器
    基于滑动窗口的预测准确率动态调整置信度阈值
    """
    
    def __init__(self, window_size: int = 20, 
                 initial_low_threshold: float = 0.5,
                 initial_high_threshold: float = 0.8,
                 min_low_threshold: float = 0.3,
                 max_low_threshold: float = 0.7,
                 sensitivity: float = 0.1):
        """
        初始化动态阈值管理器
        
        参数:
            window_size: 滑动窗口大小（最近N场比赛）
            initial_low_threshold: 初始低置信度阈值
            initial_high_threshold: 初始高置信度阈值
            min_low_threshold: 低置信度阈值最小值
            max_low_threshold: 低置信度阈值最大值
            sensitivity: 调整灵敏度（0-1，越大越敏感）
        """
        self.window_size = window_size
        self.initial_low_threshold = initial_low_threshold
        self.initial_high_threshold = initial_high_threshold
        self.min_low_threshold = min_low_threshold
        self.max_low_threshold = max_low_threshold
        self.sensitivity = sensitivity
        
        # 滑动窗口存储最近的预测结果
        self.prediction_window = deque(maxlen=window_size)
        
        # 当前阈值
        self.current_low_threshold = initial_low_threshold
        self.current_high_threshold = initial_high_threshold
        
        # 历史阈值记录
        self.threshold_history = []
        
        # 加载配置
        self._load_config()
    
    def record_prediction(self, predicted: bool, actual: bool, confidence: float):
        """
        记录一次预测结果
        
        参数:
            predicted: 预测结果（True=命中，False=未命中）
            actual: 实际结果（True=预测事件发生，False=未发生）
            confidence: 预测时的置信度
        """
        self.prediction_window.append({
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'correct': predicted == actual
        })
        
        # 自动调整阈值
        self._adjust_thresholds()
        
        # 保存配置
        self._save_config()
    
    def get_current_accuracy(self) -> float:
        """
        获取当前滑动窗口内的准确率
        
        返回:
            准确率 (0-1)
        """
        if not self.prediction_window:
            return 0.5  # 默认值
        
        correct_count = sum(1 for p in self.prediction_window if p['correct'])
        return correct_count / len(self.prediction_window)
    
    def _adjust_thresholds(self):
        """
        根据近期准确率调整阈值
        """
        if len(self.prediction_window) < self.window_size:
            return  # 窗口未满，不调整
        
        # 计算当前准确率
        accuracy = self.get_current_accuracy()
        
        # 目标准确率（期望达到的准确率）
        target_accuracy = 0.65  # 目标65%准确率
        
        # 计算准确率偏差
        accuracy_diff = accuracy - target_accuracy
        
        # 根据偏差调整低置信度阈值
        # 当准确率下降时，提高低置信阈值，减少推荐
        adjustment = accuracy_diff * self.sensitivity * (-1)  # 反向调整
        new_low_threshold = self.current_low_threshold + adjustment
        
        # 限制在范围内
        self.current_low_threshold = max(self.min_low_threshold, 
                                         min(self.max_low_threshold, new_low_threshold))
        
        # 高置信度阈值跟随调整（保持一定差距）
        threshold_gap = self.initial_high_threshold - self.initial_low_threshold
        self.current_high_threshold = self.current_low_threshold + threshold_gap
        
        # 记录调整历史
        self.threshold_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'low_threshold': self.current_low_threshold,
            'high_threshold': self.current_high_threshold
        })
        
        # 保留最近100条历史记录
        if len(self.threshold_history) > 100:
            self.threshold_history = self.threshold_history[-100:]
        
        logger.info(f"阈值调整: 准确率={accuracy:.4f}, "
                    f"低阈值={self.current_low_threshold:.4f}, "
                    f"高阈值={self.current_high_threshold:.4f}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        获取当前阈值
        
        返回:
            包含低阈值和高阈值的字典
        """
        return {
            'low_threshold': self.current_low_threshold,
            'high_threshold': self.current_high_threshold
        }
    
    def should_recommend(self, confidence: float) -> bool:
        """
        判断是否应该推荐
        
        参数:
            confidence: 预测置信度
        
        返回:
            True=应该推荐，False=不应该推荐
        """
        return confidence >= self.current_low_threshold
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        获取置信度等级
        
        参数:
            confidence: 预测置信度
        
        返回:
            置信度等级字符串
        """
        if confidence >= self.current_high_threshold:
            return 'high'
        elif confidence >= self.current_low_threshold:
            return 'medium'
        else:
            return 'low'
    
    def reset_thresholds(self):
        """
        重置阈值为初始值
        """
        self.current_low_threshold = self.initial_low_threshold
        self.current_high_threshold = self.initial_high_threshold
        self.prediction_window.clear()
        logger.info("阈值已重置为初始值")
    
    def _save_config(self):
        """
        保存配置到文件
        """
        config = {
            'window_size': self.window_size,
            'initial_low_threshold': self.initial_low_threshold,
            'initial_high_threshold': self.initial_high_threshold,
            'min_low_threshold': self.min_low_threshold,
            'max_low_threshold': self.max_low_threshold,
            'sensitivity': self.sensitivity,
            'current_low_threshold': self.current_low_threshold,
            'current_high_threshold': self.current_high_threshold,
            'threshold_history': self.threshold_history,
            'prediction_window': list(self.prediction_window)
        }
        
        with open(THRESHOLD_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def _load_config(self):
        """
        从文件加载配置
        """
        if os.path.exists(THRESHOLD_CONFIG_PATH):
            try:
                with open(THRESHOLD_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 恢复状态
                self.current_low_threshold = config.get('current_low_threshold', 
                                                       self.initial_low_threshold)
                self.current_high_threshold = config.get('current_high_threshold', 
                                                         self.initial_high_threshold)
                self.threshold_history = config.get('threshold_history', [])
                
                # 恢复预测窗口
                window_data = config.get('prediction_window', [])
                for item in window_data:
                    self.prediction_window.append(item)
                
                logger.info(f"已加载动态阈值配置")
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
            统计信息字典
        """
        stats = {
            'window_size': self.window_size,
            'window_filled': len(self.prediction_window) == self.window_size,
            'current_accuracy': self.get_current_accuracy(),
            'current_low_threshold': self.current_low_threshold,
            'current_high_threshold': self.current_high_threshold,
            'recommendation_count': sum(1 for p in self.prediction_window 
                                       if p['confidence'] >= self.current_low_threshold),
            'correct_recommendations': sum(1 for p in self.prediction_window 
                                          if p['confidence'] >= self.current_low_threshold and p['correct']),
            'history_length': len(self.threshold_history)
        }
        
        return stats


# 全局阈值管理器实例
_threshold_manager = None


def get_threshold_manager() -> DynamicThresholdManager:
    """
    获取全局阈值管理器实例（单例模式）
    
    返回:
        动态阈值管理器实例
    """
    global _threshold_manager
    
    if _threshold_manager is None:
        _threshold_manager = DynamicThresholdManager()
    
    return _threshold_manager


def update_prediction_result(predicted: bool, actual: bool, confidence: float):
    """
    更新预测结果（便捷函数）
    
    参数:
        predicted: 预测结果
        actual: 实际结果
        confidence: 置信度
    """
    manager = get_threshold_manager()
    manager.record_prediction(predicted, actual, confidence)


def get_current_recommendation_threshold() -> float:
    """
    获取当前推荐阈值（便捷函数）
    
    返回:
        当前低置信度阈值
    """
    manager = get_threshold_manager()
    return manager.current_low_threshold


def should_make_recommendation(confidence: float) -> bool:
    """
    判断是否应该推荐（便捷函数）
    
    参数:
        confidence: 置信度
    
    返回:
        是否应该推荐
    """
    manager = get_threshold_manager()
    return manager.should_recommend(confidence)


if __name__ == '__main__':
    # 示例：测试动态阈值调整
    manager = DynamicThresholdManager(window_size=5)
    
    # 模拟一些预测结果（前几次准确率低）
    test_predictions = [
        (True, False, 0.7),   # 错误
        (True, False, 0.6),   # 错误
        (True, True, 0.8),    # 正确
        (True, False, 0.75),  # 错误
        (True, False, 0.65),  # 错误
    ]
    
    for predicted, actual, confidence in test_predictions:
        manager.record_prediction(predicted, actual, confidence)
        print(f"预测: {predicted}, 实际: {actual}, 置信度: {confidence}, "
              f"当前准确率: {manager.get_current_accuracy():.4f}, "
              f"低阈值: {manager.current_low_threshold:.4f}")
    
    print("\n统计信息:")
    print(json.dumps(manager.get_statistics(), ensure_ascii=False, indent=2))