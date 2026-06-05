"""入口模块 - 福彩3D机器学习"""
from src.lottery3d.ml import (
    fetch_data,
    predict_current,
    N_TREES,
)

__all__ = ['fetch_data', 'predict_current', 'N_TREES']
