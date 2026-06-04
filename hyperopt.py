#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超参数优化模块
集成Optuna实现自动超参数搜索
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import optuna

from backtest import run_backtest

logger = logging.getLogger(__name__)

# Optuna 存储目录
OPTUNA_STUDY_DIR = 'optuna_studies'
os.makedirs(OPTUNA_STUDY_DIR, exist_ok=True)


class HyperParameterOptimizer:
    """
    超参数优化器类
    """
    
    def __init__(self, study_name: str = 'football_model'):
        """
        初始化优化器
        
        参数:
            study_name: Optuna study 名称
        """
        self.study_name = study_name
        self.storage = f'sqlite:///{os.path.join(OPTUNA_STUDY_DIR, f"{study_name}.db")}'
        self.study = None
    
    def _define_search_space(self, trial: optuna.Trial) -> Dict[str, float]:
        """
        定义超参数搜索空间
        
        参数:
            trial: Optuna trial 对象
        
        返回:
            超参数字典
        """
        params = {
            # 权重参数
            'FIT_W_1X2': trial.suggest_float('FIT_W_1X2', 0.5, 2.0),
            'FIT_W_OVER': trial.suggest_float('FIT_W_OVER', 0.3, 1.5),
            'FIT_W_ASIAN': trial.suggest_float('FIT_W_ASIAN', 0.3, 1.5),
            'FIT_W_FORM': trial.suggest_float('FIT_W_FORM', 0.1, 1.0),
            
            # Lambda 优化参数
            'LAMBDA_REFINE_STEPS': trial.suggest_int('LAMBDA_REFINE_STEPS', 3, 15),
            'LAMBDA_TOLERANCE': trial.suggest_float('LAMBDA_TOLERANCE', 1e-6, 1e-3, log=True),
            
            # 冷热过滤参数
            'HEAT_RATIO_HOT': trial.suggest_float('HEAT_RATIO_HOT', 0.5, 1.5),
            'HEAT_RATIO_COLD': trial.suggest_float('HEAT_RATIO_COLD', 0.8, 2.0),
            'HEAT_FILTER_WEIGHT': trial.suggest_float('HEAT_FILTER_WEIGHT', 0.5, 1.0),
            
            # 置信度阈值
            'CONFIDENCE_LOW_THRESHOLD': trial.suggest_float('CONFIDENCE_LOW_THRESHOLD', 0.4, 0.7),
            'CONFIDENCE_HIGH_THRESHOLD': trial.suggest_float('CONFIDENCE_HIGH_THRESHOLD', 0.7, 0.95),
            
            # 凯利指数参数
            'KELLY_DEVIATION_THRESHOLD': trial.suggest_float('KELLY_DEVIATION_THRESHOLD', 2.0, 10.0),
            'KELLY_SPREAD_THRESHOLD': trial.suggest_float('KELLY_SPREAD_THRESHOLD', 0.1, 2.0),
            
            # 泊松模型参数
            'RHO_INITIAL': trial.suggest_float('RHO_INITIAL', -0.3, 0.0),
            'MAX_GOALS': trial.suggest_int('MAX_GOALS', 5, 10),
            
            # 残差学习参数
            'RESIDUAL_WEIGHT': trial.suggest_float('RESIDUAL_WEIGHT', 0.1, 0.5),
            
            # 平局校准参数
            'DRAW_CALIBRATION_WEIGHT': trial.suggest_float('DRAW_CALIBRATION_WEIGHT', 0.1, 0.5),
        }
        
        return params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        目标函数：最小化对数损失
        
        参数:
            trial: Optuna trial 对象
        
        返回:
            对数损失值（越小越好）
        """
        # 获取超参数
        params = self._define_search_space(trial)
        
        try:
            # 执行回测
            result = run_backtest(
                league='英超',
                start_date='2024-01-01',
                end_date='2024-06-30',
                model_params=params
            )
            
            # 返回对数损失作为优化目标
            log_loss = result.get('log_loss', 10.0)
            
            # 记录其他指标
            trial.set_user_attr('brier_score', result.get('brier_score', 1.0))
            trial.set_user_attr('top1_accuracy', result.get('top1_accuracy', 0.0))
            trial.set_user_attr('top2_accuracy', result.get('top2_accuracy', 0.0))
            trial.set_user_attr('top3_accuracy', result.get('top3_accuracy', 0.0))
            
            logger.info(f"Trial {trial.number}: log_loss={log_loss:.4f}, top1_acc={result.get('top1_accuracy', 0):.4f}")
            
            return log_loss
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 失败: {str(e)}")
            return 10.0  # 返回较大的损失值
    
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, float]:
        """
        执行超参数优化
        
        参数:
            n_trials: 尝试次数
            timeout: 超时时间（秒）
        
        返回:
            最优超参数字典
        """
        logger.info(f"开始超参数优化，study: {self.study_name}, trials: {n_trials}")
        
        # 创建或加载 study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='minimize'  # 最小化对数损失
        )
        
        # 设置剪枝策略
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
        self.study.pruner = pruner
        
        # 执行优化
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"优化完成！最佳对数损失: {self.study.best_value:.4f}")
        logger.info(f"最佳超参数: {json.dumps(self.study.best_params, ensure_ascii=False, indent=2)}")
        
        return self.study.best_params
    
    def get_best_params(self) -> Dict[str, float]:
        """
        获取最佳超参数
        
        返回:
            最佳超参数字典
        """
        if self.study is None:
            # 尝试加载已存在的 study
            try:
                self.study = optuna.load_study(
                    study_name=self.study_name,
                    storage=self.storage
                )
            except Exception as e:
                logger.error(f"无法加载 study: {e}")
                return {}
        
        return self.study.best_params if self.study else {}
    
    def save_best_params(self, filepath: str):
        """
        保存最佳超参数到文件
        
        参数:
            filepath: 保存路径
        """
        best_params = self.get_best_params()
        if best_params:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)
            logger.info(f"最佳超参数已保存: {filepath}")
    
    def load_best_params(self, filepath: str) -> Dict[str, float]:
        """
        从文件加载超参数
        
        参数:
            filepath: 文件路径
        
        返回:
            超参数字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_results(self):
        """
        分析优化结果
        """
        if self.study is None:
            logger.warning("Study 未初始化")
            return
        
        # 打印统计信息
        print("\n=== 优化结果分析 ===")
        print(f"总尝试次数: {len(self.study.trials)}")
        print(f"最佳对数损失: {self.study.best_value:.4f}")
        print(f"最佳参数:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")
        
        # 可视化（如果安装了 plotly）
        try:
            import optuna.visualization as vis
            
            # 参数重要性
            fig = vis.plot_param_importances(self.study)
            fig.write_html(os.path.join(OPTUNA_STUDY_DIR, 'param_importance.html'))
            
            # 优化历史
            fig = vis.plot_optimization_history(self.study)
            fig.write_html(os.path.join(OPTUNA_STUDY_DIR, 'optimization_history.html'))
            
            # 参数关系
            fig = vis.plot_parallel_coordinate(self.study)
            fig.write_html(os.path.join(OPTUNA_STUDY_DIR, 'parallel_coordinate.html'))
            
            logger.info("可视化结果已保存到 HTML 文件")
            
        except ImportError:
            logger.warning("plotly 未安装，跳过可视化")


def optimize_hyperparameters(study_name: str = 'football_model', 
                             n_trials: int = 50) -> Dict[str, float]:
    """
    便捷函数：执行超参数优化
    
    参数:
        study_name: study 名称
        n_trials: 尝试次数
    
    返回:
        最佳超参数
    """
    optimizer = HyperParameterOptimizer(study_name)
    best_params = optimizer.optimize(n_trials)
    optimizer.save_best_params(os.path.join(OPTUNA_STUDY_DIR, f'{study_name}_best_params.json'))
    optimizer.analyze_results()
    return best_params


if __name__ == '__main__':
    # 示例：执行超参数优化
    best_params = optimize_hyperparameters('football_model_v1', n_trials=30)
    print("\n最佳超参数:")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))