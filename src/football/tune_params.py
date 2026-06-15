#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数自动寻优脚本
================

功能：
1. 使用历史回填数据自动搜索最优参数组合
2. 综合考虑多个指标的目标函数
3. 支持并行搜索和结果可视化

目标函数：
score = Top3命中率 * 0.5
       + Top5命中率 * 0.2
       - LogLoss * 0.2
       + 推荐覆盖率 * 0.1
"""

import os
import json
import math
import itertools
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('tune_params.log'), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def load_historical_data(data_dir: str = None) -> List[Dict]:
    """
    加载历史回填数据
    
    参数：
        data_dir: 数据目录，默认为当前目录
    
    返回：
        历史记录列表
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    records = []
    
    if not os.path.exists(data_dir):
        log.warning(f"数据目录不存在: {data_dir}")
        return records
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and 'history' in filename.lower():
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict) and 'records' in data:
                        records.extend(data['records'])
            except Exception as e:
                log.warning(f"读取文件失败 {filename}: {e}")
    
    log.info(f"加载了 {len(records)} 条历史记录")
    return records


def calculate_log_loss(predicted: Dict[str, float], actual: str) -> float:
    """
    计算对数损失（LogLoss）
    
    参数：
        predicted: 预测概率字典，如 {'H': 0.5, 'D': 0.3, 'A': 0.2}
        actual: 实际结果，'H'/'D'/'A'
    
    返回：
        LogLoss 值
    """
    epsilon = 1e-15
    p = predicted.get(actual, epsilon)
    p = max(epsilon, min(1 - epsilon, p))
    return -math.log(p)


def evaluate_prediction(prediction: Dict, actual: Dict) -> Dict[str, float]:
    """
    评估单个预测的效果
    
    参数：
        prediction: 预测结果
        actual: 实际结果
    
    返回：
        评估指标字典
    """
    actual_score = actual.get('score', '')
    actual_result = actual.get('result', '')
    
    # 获取预测的比分列表
    predicted_scores = prediction.get('predicted_scores', {})
    top_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
    
    # 获取预测的胜平负概率
    predicted_1x2 = prediction.get('predicted_1x2', {})
    
    # Top3 命中率
    top3_hit = 0
    if actual_score:
        top3_scores = [s[0] for s in top_scores[:3]]
        if actual_score in top3_scores:
            top3_hit = 1
    
    # Top5 命中率
    top5_hit = 0
    if actual_score:
        top5_scores = [s[0] for s in top_scores[:5]]
        if actual_score in top5_scores:
            top5_hit = 1
    
    # LogLoss
    log_loss = calculate_log_loss(predicted_1x2, actual_result) if actual_result else 0
    
    # 推荐覆盖率（Top3概率和）
    top3_sum = sum(s[1] for s in top_scores[:3]) if top_scores else 0
    
    return {
        'top3_hit': top3_hit,
        'top5_hit': top5_hit,
        'log_loss': log_loss,
        'top3_sum': top3_sum,
    }


def evaluate_params(params: Dict[str, float], records: List[Dict], 
                   predict_func) -> Dict[str, float]:
    """
    使用指定参数评估模型效果
    
    参数：
        params: 参数字典
        records: 历史记录列表
        predict_func: 预测函数，签名为 predict_func(record, **params) -> prediction
    
    返回：
        综合评估结果
    """
    total_top3_hit = 0
    total_top5_hit = 0
    total_log_loss = 0
    total_top3_sum = 0
    count = 0
    
    for record in records:
        actual_score = record.get('actual_score')
        actual_result = record.get('actual_result')
        
        if not actual_score or not actual_result:
            continue
        
        try:
            # 使用当前参数进行预测
            prediction = predict_func(record, **params)
            
            # 评估预测效果
            eval_result = evaluate_prediction(prediction, {
                'score': actual_score,
                'result': actual_result
            })
            
            total_top3_hit += eval_result['top3_hit']
            total_top5_hit += eval_result['top5_hit']
            total_log_loss += eval_result['log_loss']
            total_top3_sum += eval_result['top3_sum']
            count += 1
            
        except Exception as e:
            log.warning(f"处理记录失败 {record.get('match_id')}: {e}")
            continue
    
    if count == 0:
        return {
            'top3_accuracy': 0,
            'top5_accuracy': 0,
            'avg_log_loss': float('inf'),
            'avg_top3_sum': 0,
            'score': float('-inf'),
            'sample_count': 0,
        }
    
    # 计算各项指标
    top3_accuracy = total_top3_hit / count
    top5_accuracy = total_top5_hit / count
    avg_log_loss = total_log_loss / count
    avg_top3_sum = total_top3_sum / count
    
    # 计算综合得分
    score = (
        top3_accuracy * 0.5 +
        top5_accuracy * 0.2 -
        avg_log_loss * 0.2 +
        avg_top3_sum * 0.1
    )
    
    return {
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'avg_log_loss': avg_log_loss,
        'avg_top3_sum': avg_top3_sum,
        'score': score,
        'sample_count': count,
    }


def grid_search(param_grid: Dict[str, List[float]], records: List[Dict],
               predict_func) -> List[Dict]:
    """
    网格搜索最优参数
    
    参数：
        param_grid: 参数网格
        records: 历史记录列表
        predict_func: 预测函数
    
    返回：
        所有参数组合的评估结果，按得分排序
    """
    results = []
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    total_combinations = math.prod(len(v) for v in param_values)
    log.info(f"开始网格搜索，共 {total_combinations} 种参数组合")
    
    for i, values in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, values))
        
        log.info(f"正在测试参数组合 {i+1}/{total_combinations}: {params}")
        
        try:
            result = evaluate_params(params, records, predict_func)
            result['params'] = params
            results.append(result)
            
            log.info(f"  得分: {result['score']:.4f}, "
                    f"Top3命中率: {result['top3_accuracy']:.2%}, "
                    f"Top5命中率: {result['top5_accuracy']:.2%}, "
                    f"LogLoss: {result['avg_log_loss']:.4f}, "
                    f"覆盖率: {result['avg_top3_sum']:.4f}")
                    
        except Exception as e:
            log.error(f"参数组合 {params} 评估失败: {e}")
            continue
    
    # 按得分排序
    results.sort(key=lambda x: -x['score'])
    
    return results


def print_results(results: List[Dict], top_n: int = 10):
    """
    打印搜索结果
    
    参数：
        results: 结果列表
        top_n: 显示前N个结果
    """
    print("\n" + "="*80)
    print("参数寻优结果")
    print("="*80)
    print(f"{'排名':^6} | {'得分':^10} | {'Top3命中率':^12} | {'Top5命中率':^12} | {'LogLoss':^10} | {'覆盖率':^10} | 参数")
    print("-"*80)
    
    for i, result in enumerate(results[:top_n], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in result['params'].items())
        print(f"{i:^6} | {result['score']:^10.4f} | {result['top3_accuracy']:^12.2%} | {result['top5_accuracy']:^12.2%} | {result['avg_log_loss']:^10.4f} | {result['avg_top3_sum']:^10.4f} | {params_str}")
    
    print("="*80)


def save_results(results: List[Dict], filepath: str = None):
    """
    保存搜索结果到文件
    
    参数：
        results: 结果列表
        filepath: 保存路径
    """
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'param_tuning_result_{timestamp}.json'
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_combinations': len(results),
        'best_score': results[0]['score'] if results else None,
        'best_params': results[0]['params'] if results else None,
        'results': results,
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    log.info(f"结果已保存到 {filepath}")


def default_predict_func(record: Dict, **params) -> Dict:
    """
    默认预测函数（示例）
    
    参数：
        record: 比赛记录
        params: 参数
    
    返回：
        预测结果
    """
    # 这里应该调用实际的预测逻辑
    # 目前返回一个模拟结果作为示例
    market_weight = params.get('market_weight', 0.3)
    similar_weight = params.get('similar_weight', 0.1)
    bayes_weight = params.get('bayes_weight', 0.2)
    top3_sum_threshold = params.get('top3_sum_threshold', 0.28)
    
    # 模拟预测结果
    return {
        'predicted_scores': {
            '1-1': 0.25 * market_weight + 0.1 * similar_weight,
            '2-1': 0.2 * market_weight + 0.15 * similar_weight,
            '1-0': 0.18 * market_weight,
            '2-0': 0.15 * bayes_weight,
            '0-0': 0.12 * market_weight,
            '3-1': 0.05,
            '0-1': 0.03,
            '1-2': 0.02,
        },
        'predicted_1x2': {
            'H': 0.45 + bayes_weight * 0.1,
            'D': 0.3,
            'A': 0.25 - bayes_weight * 0.05,
        },
    }


def main():
    """
    主函数
    """
    log.info("="*60)
    log.info("参数自动寻优脚本启动")
    log.info("="*60)
    
    # 定义参数网格
    param_grid = {
        'market_weight': [0.1, 0.2, 0.3, 0.4],
        'similar_weight': [0.0, 0.1, 0.2, 0.3],
        'bayes_weight': [0.1, 0.2, 0.3],
        'steam_weight': [0.0, 0.05, 0.1],
        'low_score_bonus': [0.0, 0.05, 0.1],
        'draw_mult': [1.0, 1.1, 1.2],
        'cold_score_bonus': [0.0, 0.03, 0.05],
        'risk_threshold': [0.3, 0.35, 0.4],
        'top3_sum_threshold': [0.24, 0.28, 0.32],
    }
    
    # 加载历史数据
    records = load_historical_data()
    
    if not records:
        log.warning("没有加载到历史数据，使用模拟数据进行演示")
        # 创建一些模拟数据
        records = [
            {'match_id': f'test_{i}', 'actual_score': '1-1', 'actual_result': 'D'}
            for i in range(100)
        ]
    
    # 运行网格搜索
    results = grid_search(param_grid, records, default_predict_func)
    
    # 打印结果
    print_results(results, top_n=10)
    
    # 保存结果
    save_results(results)
    
    if results:
        log.info(f"\n最优参数组合:")
        log.info(f"  得分: {results[0]['score']:.4f}")
        log.info(f"  参数: {results[0]['params']}")
        log.info(f"  Top3命中率: {results[0]['top3_accuracy']:.2%}")
        log.info(f"  Top5命中率: {results[0]['top5_accuracy']:.2%}")
        log.info(f"  LogLoss: {results[0]['avg_log_loss']:.4f}")
        log.info(f"  覆盖率: {results[0]['avg_top3_sum']:.4f}")
    
    log.info("="*60)
    log.info("参数自动寻优脚本完成")
    log.info("="*60)


if __name__ == '__main__':
    main()