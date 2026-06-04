"""福彩3D预测模块"""
from . import predictor as _predictor

run_prediction = _predictor.run_prediction
print_report = _predictor.print_report
main = _predictor.main

__all__ = ['run_prediction', 'print_report', 'main']
