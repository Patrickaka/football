#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
历史回测引擎模块
提供历史数据抓取、回测逻辑、评估指标计算功能
"""

import os
import json
import time
import pickle
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from src.football import (
    analyze_match, fetch_ouzhi, resolve_league_profile,
    CONFIDENCE_LOW_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD
)
from src.common.paths import DATA_DIR

logger = logging.getLogger(__name__)

# 回测数据存储目录
BACKTEST_DATA_DIR = str(DATA_DIR / 'backtest_data')
os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)


def fetch_historical_matches(league: str, start_date: str, end_date: str) -> List[Dict]:
    """
    抓取历史比赛数据
    
    参数:
        league: 联赛名称
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
        历史比赛列表
    """
    logger.info(f"抓取历史比赛数据: {league} {start_date} ~ {end_date}")
    
    # 模拟历史数据抓取（实际应用中需要接入历史数据API）
    # 这里使用示例数据
    historical_data = [
        {
            'match_id': f'hist_{i}',
            'home': ['曼联', '利物浦', '阿森纳', '切尔西', '曼城'][i % 5],
            'away': ['热刺', '埃弗顿', '纽卡斯尔', '莱斯特', '南安普顿'][i % 5],
            'league': league,
            'time': f'2024-{str(i//30 + 1).zfill(2)}-{str(i%30 + 1).zfill(2)} 20:00',
            'home_score': np.random.randint(0, 5),
            'away_score': np.random.randint(0, 4),
        }
        for i in range(50)
    ]
    
    return historical_data


def load_historical_data(league: str, start_date: str, end_date: str) -> List[Dict]:
    """
    加载历史数据（优先从缓存读取）
    
    参数:
        league: 联赛名称
        start_date: 开始日期
        end_date: 结束日期
    
    返回:
        历史比赛列表
    """
    cache_path = os.path.join(BACKTEST_DATA_DIR, f'{league}_{start_date}_{end_date}.pkl')
    
    if os.path.exists(cache_path):
        logger.info(f"从缓存加载历史数据: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # 抓取新数据
    matches = fetch_historical_matches(league, start_date, end_date)
    
    # 保存到缓存
    with open(cache_path, 'wb') as f:
        pickle.dump(matches, f)
    
    return matches


def calculate_log_loss(predictions: List[float], actuals: List[int]) -> float:
    """
    计算对数损失 (Log Loss)
    
    参数:
        predictions: 预测概率列表
        actuals: 实际结果列表 (0或1)
    
    返回:
        对数损失值
    """
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean([actuals[i] * np.log(predictions[i]) + (1 - actuals[i]) * np.log(1 - predictions[i]) 
                     for i in range(len(predictions))])


def calculate_brier_score(predictions: List[float], actuals: List[int]) -> float:
    """
    计算Brier分数
    
    参数:
        predictions: 预测概率列表
        actuals: 实际结果列表 (0或1)
    
    返回:
        Brier分数值
    """
    return np.mean([(predictions[i] - actuals[i]) ** 2 for i in range(len(predictions))])


def calculate_top_n_accuracy(predictions: List[List[Tuple[int, int, float]]], 
                              actuals: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    计算Top-N准确率
    
    参数:
        predictions: 预测结果列表，每个元素是 (home_score, away_score, prob) 的列表
        actuals: 实际比分列表
    
    返回:
        Top-1/2/3准确率字典
    """
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    
    for pred, actual in zip(predictions, actuals):
        actual_tuple = (actual[0], actual[1])
        pred_sorted = sorted(pred, key=lambda x: -x[2])
        
        if len(pred_sorted) >= 1 and pred_sorted[0][:2] == actual_tuple:
            top1_correct += 1
            top2_correct += 1
            top3_correct += 1
        elif len(pred_sorted) >= 2 and pred_sorted[1][:2] == actual_tuple:
            top2_correct += 1
            top3_correct += 1
        elif len(pred_sorted) >= 3 and pred_sorted[2][:2] == actual_tuple:
            top3_correct += 1
    
    total = len(predictions)
    return {
        'top1_accuracy': top1_correct / total if total > 0 else 0,
        'top2_accuracy': top2_correct / total if total > 0 else 0,
        'top3_accuracy': top3_correct / total if total > 0 else 0,
    }


def run_backtest(league: str, start_date: str, end_date: str, 
                 model_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    执行历史回测
    
    参数:
        league: 联赛名称
        start_date: 开始日期
        end_date: 结束日期
        model_params: 模型参数（用于超参数调优）
    
    返回:
        回测结果字典
    """
    logger.info(f"开始回测: {league} {start_date} ~ {end_date}")
    
    # 加载历史数据
    matches = load_historical_data(league, start_date, end_date)
    
    # 存储预测结果和实际结果
    predictions = []
    actuals = []
    score_predictions = []
    score_actuals = []
    confidence_scores = []
    correct_count = 0
    total_count = 0
    
    # 逐场预测
    for i, match in enumerate(matches):
        try:
            # 模拟分析比赛（使用当前模型）
            result = analyze_match({
                'match_id': match['match_id'],
                'home': match['home'],
                'away': match['away'],
                'league': match['league'],
                'time': match['time'],
            })
            
            # 获取预测概率（主胜概率）
            if result['model']['top_scores']:
                pred_prob = result['model']['top_scores'][0]['prob']
                predictions.append(pred_prob)
                
                # 实际结果：主胜=1，平局或客胜=0
                actual = 1 if match['home_score'] > match['away_score'] else 0
                actuals.append(actual)
                
                # 比分预测
                score_preds = [(s['home'], s['away'], s['prob']) 
                              for s in result['model']['top_scores']]
                score_predictions.append(score_preds)
                score_actuals.append((match['home_score'], match['away_score']))
                
                # 置信度
                if result['confidence'] and result['confidence']['score']:
                    confidence_scores.append(result['confidence']['score'])
                
                # 统计Top-1命中
                if score_preds and score_preds[0][:2] == (match['home_score'], match['away_score']):
                    correct_count += 1
                
                total_count += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i + 1}/{len(matches)} 场比赛")
            
            # 避免请求过快
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"处理比赛 {match['match_id']} 时出错: {str(e)}")
            continue
    
    # 计算评估指标
    results = {
        'league': league,
        'start_date': start_date,
        'end_date': end_date,
        'total_matches': len(matches),
        'processed_matches': total_count,
        'top1_correct': correct_count,
    }
    
    if predictions:
        results.update({
            'log_loss': calculate_log_loss(predictions, actuals),
            'brier_score': calculate_brier_score(predictions, actuals),
        })
    
    if score_predictions:
        results.update(calculate_top_n_accuracy(score_predictions, score_actuals))
    
    if confidence_scores:
        results.update({
            'avg_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores),
        })
    
    logger.info(f"回测完成: {json.dumps(results, ensure_ascii=False, indent=2)}")
    return results


def evaluate_model_performance(results: List[Dict]) -> Dict[str, Any]:
    """
    评估模型整体性能
    
    参数:
        results: 多个回测结果列表
    
    返回:
        综合评估结果
    """
    if not results:
        return {}
    
    # 合并结果
    combined = {
        'total_leagues': len(results),
        'total_matches': sum(r['total_matches'] for r in results),
        'total_processed': sum(r['processed_matches'] for r in results),
        'total_top1_correct': sum(r['top1_correct'] for r in results),
    }
    
    # 平均指标
    log_losses = [r['log_loss'] for r in results if 'log_loss' in r]
    brier_scores = [r['brier_score'] for r in results if 'brier_score' in r]
    top1_accs = [r['top1_accuracy'] for r in results if 'top1_accuracy' in r]
    
    if log_losses:
        combined['avg_log_loss'] = np.mean(log_losses)
        combined['std_log_loss'] = np.std(log_losses)
    
    if brier_scores:
        combined['avg_brier_score'] = np.mean(brier_scores)
        combined['std_brier_score'] = np.std(brier_scores)
    
    if top1_accs:
        combined['avg_top1_accuracy'] = np.mean(top1_accs)
        combined['std_top1_accuracy'] = np.std(top1_accs)
    
    return combined


def save_backtest_results(results: Dict, filepath: str):
    """
    保存回测结果到文件
    
    参数:
        results: 回测结果
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"回测结果已保存: {filepath}")


def load_backtest_results(filepath: str) -> Dict:
    """
    从文件加载回测结果
    
    参数:
        filepath: 文件路径
    
    返回:
        回测结果字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    # 示例：执行回测
    result = run_backtest('英超', '2024-01-01', '2024-06-30')
    save_backtest_results(result, 'backtest_result.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))