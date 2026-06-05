#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试排列五新功能 - 排名模型、特征贡献度、多模型集成投票、周期识别"""

import pailie5

def main():
    analyzer = pailie5.get_pailie5_analyzer()
    
    print('=== 测试排名模型 (Top-N排序) ===')
    ranked = analyzer.rank_model(top_n=5)
    print(f'Top-5 排名号码: {ranked}')
    
    print('\n=== 测试特征贡献度 ===')
    contributions = analyzer.feature_contribution()
    print(f'特征列表: {list(contributions.keys())}')
    print(f'频率贡献: {contributions["frequency"]}')
    
    print('\n=== 测试动态权重调整 ===')
    backtest = analyzer.backtest(method='bayesian', test_periods=10)
    weights = analyzer.dynamic_weight_adjustment(backtest['predictions'])
    print(f'动态调整后的权重: {weights}')
    
    print('\n=== 测试周期与状态识别 ===')
    cycles = analyzer.identify_cycles()
    print(f'热门号码: {cycles["hot"]}')
    print(f'冷门号码: {cycles["cold"]}')
    print(f'升温状态: {cycles["warming"]}')
    print(f'降温状态: {cycles["cooling"]}')
    
    print('\n=== 测试多模型集成投票 ===')
    prediction = analyzer.multi_model_voting(n_votes=3)
    print(f'投票结果: {prediction}')
    
    print('\n=== 测试集成预测 ===')
    ensemble = analyzer.ensemble_predict(method='voting')
    print(f'预测方法: {ensemble["method"]}')
    print(f'预测号码: {ensemble["prediction"]}')
    print(f'排名号码: {ensemble["ranked_numbers"]}')

if __name__ == '__main__':
    main()
