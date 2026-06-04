#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测命令行工具
支持手动触发回测和超参数调优
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta

from backtest import run_backtest, evaluate_model_performance, save_backtest_results
from hyperopt import HyperParameterOptimizer
from dynamic_threshold import DynamicThresholdManager, get_threshold_manager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('backtest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_backtest_command(args):
    """
    执行回测命令
    """
    logger.info(f"开始回测: {args.league} {args.start_date} ~ {args.end_date}")
    
    result = run_backtest(
        league=args.league,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"联赛: {result['league']}")
    print(f"时间范围: {result['start_date']} ~ {result['end_date']}")
    print(f"总比赛数: {result['total_matches']}")
    print(f"处理比赛数: {result['processed_matches']}")
    print(f"Top-1 命中: {result['top1_correct']}/{result['processed_matches']}")
    
    if 'log_loss' in result:
        print(f"对数损失: {result['log_loss']:.4f}")
    if 'brier_score' in result:
        print(f"Brier分数: {result['brier_score']:.4f}")
    if 'top1_accuracy' in result:
        print(f"Top-1 准确率: {result['top1_accuracy']:.4f}")
    if 'top2_accuracy' in result:
        print(f"Top-2 准确率: {result['top2_accuracy']:.4f}")
    if 'top3_accuracy' in result:
        print(f"Top-3 准确率: {result['top3_accuracy']:.4f}")
    if 'avg_confidence' in result:
        print(f"平均置信度: {result['avg_confidence']:.4f}")
    
    print("="*60)
    
    # 保存结果
    if args.save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{args.league}_{timestamp}.json"
        save_backtest_results(result, filename)
        print(f"\n结果已保存到: {filename}")


def run_hyperopt_command(args):
    """
    执行超参数优化命令
    """
    logger.info(f"开始超参数优化: {args.study_name}, 尝试次数: {args.n_trials}")
    
    optimizer = HyperParameterOptimizer(study_name=args.study_name)
    best_params = optimizer.optimize(n_trials=args.n_trials)
    
    # 输出最佳参数
    print("\n" + "="*60)
    print("最佳超参数")
    print("="*60)
    for key, value in sorted(best_params.items()):
        print(f"{key}: {value}")
    print("="*60)
    
    # 分析结果
    optimizer.analyze_results()


def run_threshold_command(args):
    """
    执行阈值管理命令
    """
    manager = get_threshold_manager()
    
    if args.action == 'show':
        stats = manager.get_statistics()
        print("\n" + "="*60)
        print("动态阈值状态")
        print("="*60)
        print(f"窗口大小: {stats['window_size']}")
        print(f"窗口已满: {'是' if stats['window_filled'] else '否'}")
        print(f"当前准确率: {stats['current_accuracy']:.4f}")
        print(f"当前低阈值: {stats['current_low_threshold']:.4f}")
        print(f"当前高阈值: {stats['current_high_threshold']:.4f}")
        print(f"推荐次数: {stats['recommendation_count']}")
        print(f"正确推荐: {stats['correct_recommendations']}")
        print(f"历史记录数: {stats['history_length']}")
        print("="*60)
    
    elif args.action == 'reset':
        manager.reset_thresholds()
        print("阈值已重置为初始值")
    
    elif args.action == 'update':
        # 模拟添加预测结果
        import random
        for _ in range(args.count):
            predicted = random.choice([True, False])
            actual = random.choice([True, False])
            confidence = round(random.uniform(0.4, 0.95), 2)
            manager.record_prediction(predicted, actual, confidence)
        
        print(f"已添加 {args.count} 条模拟预测记录")
        stats = manager.get_statistics()
        print(f"当前准确率: {stats['current_accuracy']:.4f}")
        print(f"当前低阈值: {stats['current_low_threshold']:.4f}")


def run_compare_command(args):
    """
    执行模型对比命令
    """
    print("\n" + "="*60)
    print("模型对比")
    print("="*60)
    
    results = []
    
    # 回测不同联赛
    for league in args.leagues.split(','):
        league = league.strip()
        if league:
            result = run_backtest(
                league=league,
                start_date=args.start_date,
                end_date=args.end_date
            )
            results.append(result)
    
    # 评估整体性能
    overall = evaluate_model_performance(results)
    
    print("\n各联赛结果:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['league']}:")
        print(f"   比赛数: {result['processed_matches']}")
        print(f"   Top-1 命中: {result['top1_correct']}")
        print(f"   Top-1 准确率: {result.get('top1_accuracy', 0):.4f}")
        print(f"   对数损失: {result.get('log_loss', 0):.4f}")
    
    print("\n综合评估:")
    print(f"总联赛数: {overall.get('total_leagues', 0)}")
    print(f"总比赛数: {overall.get('total_processed', 0)}")
    print(f"平均 Top-1 准确率: {overall.get('avg_top1_accuracy', 0):.4f}")
    print(f"平均对数损失: {overall.get('avg_log_loss', 0):.4f}")
    
    print("="*60)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='足球预测模型回测工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='执行历史回测')
    backtest_parser.add_argument('--league', '-l', default='英超', help='联赛名称')
    backtest_parser.add_argument('--start-date', '-s', default='2024-01-01', help='开始日期')
    backtest_parser.add_argument('--end-date', '-e', default='2024-12-31', help='结束日期')
    backtest_parser.add_argument('--save', '-o', action='store_true', help='保存结果')
    
    # 超参数优化命令
    hyperopt_parser = subparsers.add_parser('hyperopt', help='执行超参数优化')
    hyperopt_parser.add_argument('--study-name', '-n', default='football_model', help='Study名称')
    hyperopt_parser.add_argument('--n-trials', '-t', type=int, default=50, help='尝试次数')
    
    # 阈值管理命令
    threshold_parser = subparsers.add_parser('threshold', help='动态阈值管理')
    threshold_parser.add_argument('action', choices=['show', 'reset', 'update'], help='操作')
    threshold_parser.add_argument('--count', '-c', type=int, default=10, help='更新数量')
    
    # 模型对比命令
    compare_parser = subparsers.add_parser('compare', help='对比不同联赛/模型')
    compare_parser.add_argument('--leagues', '-l', default='英超,西甲,德甲', help='联赛列表（逗号分隔）')
    compare_parser.add_argument('--start-date', '-s', default='2024-01-01', help='开始日期')
    compare_parser.add_argument('--end-date', '-e', default='2024-06-30', help='结束日期')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        run_backtest_command(args)
    elif args.command == 'hyperopt':
        run_hyperopt_command(args)
    elif args.command == 'threshold':
        run_threshold_command(args)
    elif args.command == 'compare':
        run_compare_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()