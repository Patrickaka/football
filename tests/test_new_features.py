#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试排列五新功能"""

import pailie5

def main():
    analyzer = pailie5.get_pailie5_analyzer()
    
    print('=== 测试二阶马尔可夫链 ===')
    markov = analyzer.build_second_order_markov()
    print(f'状态数: {len(markov)}')
    
    print('\n=== 测试权重优化 ===')
    weights = analyzer.optimize_weights()
    print(f'最优权重: {weights}')
    
    print('\n=== 测试历史回测 ===')
    backtest = analyzer.backtest(method='bayesian', test_periods=30)
    print(f'回测方法: {backtest["method"]}')
    print(f'测试期数: {backtest["test_periods"]}')
    print(f'命中率: {backtest["hit_rate"]}%')
    
    print('\n=== 测试多条件缩水 ===')
    candidates = [
        [1,2,3,4,5], [6,7,8,9,0], [1,3,5,7,9], [2,4,6,8,0]
    ]
    conditions = {
        'sum_range': (15, 25),
        'odd_count': (2, 3)
    }
    filtered = analyzer.multi_condition_filter(candidates, conditions)
    print(f'原始候选: {candidates}')
    print(f'筛选后: {filtered}')

if __name__ == '__main__':
    main()
