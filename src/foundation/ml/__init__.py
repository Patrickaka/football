"""ML 训练骨架：验证集切分、eval_set 形状适配、融合权重。

只放各领域通用的训练脚手架，不含任何特征工程或业务模型。
"""
from .training import (
    TrainValidSplit,
    blend_weights,
    lightgbm_eval_set,
    split_by_group,
    xgboost_eval_set,
)

__all__ = [
    'TrainValidSplit',
    'blend_weights',
    'lightgbm_eval_set',
    'split_by_group',
    'xgboost_eval_set',
]
