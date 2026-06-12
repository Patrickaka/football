#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一回测模块
============

功能：
1. 对历史记录逐场复盘
2. 用赛前盘口生成预测
3. 和真实比分比较
4. 输出命中率、Brier Score、LogLoss

核心指标：
- Top1 比分命中率
- Top3 比分命中率
- Top5 比分命中率
- 胜平负命中率
- 总进球 Top2 命中率
- 让球方向命中率
- Brier Score
- LogLoss
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime


log = logging.getLogger('football')


class BacktestRunner:
    """回测运行器"""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.league_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total': 0, 'top1': 0, 'top3': 0, 'top5': 0,
            'correct_1x2': 0, 'correct_total': 0, 'correct_handicap': 0,
            'brier_scores': [], 'log_losses': []
        })
    
    def add_result(self, record: Dict, prediction: Dict, actual: Dict):
        """
        添加一场回测结果
        
        参数：
            record: 原始预测记录
            prediction: 模型预测结果
            actual: 实际比赛结果
        """
        # 计算各项指标
        predicted_scores = record.get('predicted_scores', {})
        actual_score = actual.get('score', '')
        actual_result = actual.get('result', '')
        
        # 排序预测比分
        sorted_scores = sorted(predicted_scores.items(), key=lambda x: -x[1])
        top1_score = sorted_scores[0][0] if sorted_scores else None
        top3_scores = [s[0] for s in sorted_scores[:3]]
        top5_scores = [s[0] for s in sorted_scores[:5]]
        
        # 计算命中
        hit_top1 = (top1_score == actual_score)
        hit_top3 = (actual_score in top3_scores)
        hit_top5 = (actual_score in top5_scores)
        
        # 胜平负
        predicted_1x2 = record.get('predicted_1x2', {})
        pred_result = max(predicted_1x2.items(), key=lambda x: x[1])[0] if predicted_1x2 else None
        hit_1x2 = (pred_result == actual_result)
        
        # 总进球
        actual_goals = 0
        if actual_score:
            try:
                parts = actual_score.split('-')
                actual_goals = int(parts[0]) + int(parts[1])
            except:
                pass
        
        predicted_totals = prediction.get('goal_count', {})
        sorted_totals = sorted(predicted_totals.items(), key=lambda x: -x[1])
        top2_totals = [t[0] for t in sorted_totals[:2]]
        hit_total = str(actual_goals) in top2_totals
        
        # 让球方向
        asian = record.get('asian')
        hit_handicap = False
        if asian is not None and actual_score:
            home_score, away_score = map(int, actual_score.split('-'))
            handicap_margin = home_score - away_score - asian
            
            if asian > 0:
                # 主队让球，主队赢盘需要净胜超过盘口
                hit_handicap = handicap_margin > 0
            elif asian < 0:
                # 客队让球（主队受让），主队赢盘条件
                hit_handicap = handicap_margin > 0
            else:
                # 平手盘，主队赢盘即胜负分
                hit_handicap = home_score != away_score
        
        # 计算比分 LogLoss
        actual_prob = predicted_scores.get(actual_score, 1e-15)
        score_logloss = -math.log(max(1e-15, min(1 - 1e-15, actual_prob)))
        
        # 计算比分 Brier Score（多分类版本）
        all_scores = set(predicted_scores.keys()) | {actual_score}
        score_brier = sum(
            (predicted_scores.get(score, 0.0) - (1.0 if score == actual_score else 0.0)) ** 2
            for score in all_scores
        )
        
        # 计算胜平负 LogLoss 和 Brier
        predicted_1x2 = record.get('predicted_1x2', {})
        result_logloss = 0.0
        result_brier = 0.0
        if actual_result and predicted_1x2:
            actual_prob_1x2 = predicted_1x2.get(actual_result, 1e-15)
            result_logloss = -math.log(max(1e-15, min(1 - 1e-15, actual_prob_1x2)))
            
            all_results = {'H', 'D', 'A'}
            result_brier = sum(
                (predicted_1x2.get(res, 0.0) - (1.0 if res == actual_result else 0.0)) ** 2
                for res in all_results
            )
        
        result = {
            'match_id': record.get('match_id'),
            'league': record.get('league'),
            'home': record.get('home'),
            'away': record.get('away'),
            'actual_score': actual_score,
            'actual_result': actual_result,
            'top1_score': top1_score,
            'hit_top1': hit_top1,
            'hit_top3': hit_top3,
            'hit_top5': hit_top5,
            'hit_1x2': hit_1x2,
            'hit_total': hit_total,
            'hit_handicap': hit_handicap,
            'score_logloss': score_logloss,
            'score_brier': score_brier,
            'result_logloss': result_logloss,
            'result_brier': result_brier,
        }
        
        self.results.append(result)
        
        # 更新联赛统计
        league = record.get('league', '未知')
        stats = self.league_stats[league]
        stats['total'] += 1
        if hit_top1: stats['top1'] += 1
        if hit_top3: stats['top3'] += 1
        if hit_top5: stats['top5'] += 1
        if hit_1x2: stats['correct_1x2'] += 1
        if hit_total: stats['correct_total'] += 1
        if hit_handicap: stats['correct_handicap'] += 1
        stats['brier_scores'].append(brier_score)
        stats['log_losses'].append(log_loss)
        
        return result
    
    def get_summary(self) -> Dict:
        """获取回测汇总"""
        if not self.results:
            return {'error': '没有回测结果'}
        
        total = len(self.results)
        
        # 总体统计
        top1_hits = sum(1 for r in self.results if r['hit_top1'])
        top3_hits = sum(1 for r in self.results if r['hit_top3'])
        top5_hits = sum(1 for r in self.results if r['hit_top5'])
        hits_1x2 = sum(1 for r in self.results if r['hit_1x2'])
        hits_total = sum(1 for r in self.results if r['hit_total'])
        hits_handicap = sum(1 for r in self.results if r['hit_handicap'])
        
        brier_scores = [r['brier_score'] for r in self.results]
        log_losses = [r['log_loss'] for r in self.results]
        
        summary = {
            'total_matches': total,
            'top1_hit_rate': top1_hits / total,
            'top3_hit_rate': top3_hits / total,
            'top5_hit_rate': top5_hits / total,
            'hit_rate_1x2': hits_1x2 / total,
            'hit_rate_total': hits_total / total,
            'hit_rate_handicap': hits_handicap / total,
            'brier_score': sum(brier_scores) / total,
            'log_loss': sum(log_losses) / total,
            'by_league': {},
        }
        
        # 联赛统计
        for league, stats in self.league_stats.items():
            t = stats['total']
            if t > 0:
                summary['by_league'][league] = {
                    'total': t,
                    'top1_hit_rate': stats['top1'] / t,
                    'top3_hit_rate': stats['top3'] / t,
                    'top5_hit_rate': stats['top5'] / t,
                    'hit_rate_1x2': stats['correct_1x2'] / t,
                    'brier_score': sum(stats['brier_scores']) / t,
                    'log_loss': sum(stats['log_losses']) / t,
                }
        
        return summary
    
    def print_summary(self):
        """打印回测汇总"""
        summary = self.get_summary()
        
        if 'error' in summary:
            print(summary['error'])
            return
        
        print("=" * 60)
        print("  回测结果汇总")
        print("=" * 60)
        print(f"总比赛数: {summary['total_matches']}")
        print("-" * 60)
        print(f"Top1 比分命中率: {summary['top1_hit_rate']:.2%}")
        print(f"Top3 比分命中率: {summary['top3_hit_rate']:.2%}")
        print(f"Top5 比分命中率: {summary['top5_hit_rate']:.2%}")
        print(f"胜平负命中率:    {summary['hit_rate_1x2']:.2%}")
        print(f"总进球 Top2:     {summary['hit_rate_total']:.2%}")
        print(f"让球方向命中率:  {summary['hit_rate_handicap']:.2%}")
        print("-" * 60)
        print(f"Brier Score:     {summary['brier_score']:.4f}")
        print(f"LogLoss:         {summary['log_loss']:.4f}")
        
        if summary['by_league']:
            print("-" * 60)
            print("按联赛统计:")
            for league, stats in summary['by_league'].items():
                print(f"  {league} ({stats['total']}场): Top1={stats['top1_hit_rate']:.2%}, "
                      f"Top3={stats['top3_hit_rate']:.2%}, 1x2={stats['hit_rate_1x2']:.2%}")
        
        print("=" * 60)


def run_backtest(records: List[Dict], 
                predict_func: Optional[Callable] = None,
                verbose: bool = True) -> Dict:
    """
    运行回测
    
    参数：
        records: 历史预测记录列表，每条记录需要包含：
                 - match_id: 比赛ID
                 - league: 联赛
                 - home: 主队
                 - away: 客队
                 - predicted_scores: 预测比分 {"1-1": 0.108, ...}
                 - predicted_1x2: 预测胜平负 {"home": 0.46, ...}
                 - asian: 亚盘让球
                 - total_line: 大小球盘口
                 - actual_score: 实际比分
                 - actual_result: 实际结果 H/D/A
        predict_func: 可选的预测函数，用于重新预测
        verbose: 是否打印详细信息
    
    返回：
        回测汇总结果
    """
    runner = BacktestRunner()
    
    for record in records:
        # 检查是否有实际结果
        actual_score = record.get('actual_score')
        if not actual_score:
            continue
        
        actual_result = record.get('actual_result')
        if not actual_result:
            # 根据比分计算结果
            try:
                parts = actual_score.split('-')
                home_g = int(parts[0])
                away_g = int(parts[1])
                if home_g > away_g:
                    actual_result = 'H'
                elif home_g < away_g:
                    actual_result = 'A'
                else:
                    actual_result = 'D'
            except:
                continue
        
        actual = {
            'score': actual_score,
            'result': actual_result,
        }
        
        # 添加结果
        result = runner.add_result(record, {}, actual)
        
        if verbose and result['hit_top1']:
            log.info(f"命中: {record['home']} vs {record['away']} -> "
                    f"{actual_score} (预测: {result['top1_score']})")
    
    return runner.get_summary()


def backtest_from_history(league: str = None, limit: int = None) -> Dict:
    """
    从预测历史中运行回测
    
    参数：
        league: 只回测指定联赛
        limit: 限制回测数量
    
    返回：
        回测汇总结果
    """
    try:
        from .result_sync import _global_history
        
        if league:
            records = [r for r in _global_history.records 
                      if r.get('settled') and r.get('league') == league]
        else:
            records = [r for r in _global_history.records if r.get('settled')]
        
        if limit:
            records = records[-limit:]
        
        runner = BacktestRunner()
        
        for record in records:
            actual_score = record.get('actual_score')
            if not actual_score:
                continue
            
            actual_result = record.get('actual_result', '')
            try:
                parts = actual_score.split('-')
                home_g = int(parts[0])
                away_g = int(parts[1])
                if home_g > away_g:
                    actual_result = 'H'
                elif home_g < away_g:
                    actual_result = 'A'
                else:
                    actual_result = 'D'
            except:
                continue
            
            actual = {'score': actual_score, 'result': actual_result}
            runner.add_result(record, {}, actual)
        
        return runner.get_summary()
        
    except ImportError:
        return {'error': 'result_sync 模块未导入'}


def compare_parameters(records: List[Dict], 
                     param_sets: List[Dict],
                     param_name: str = 'params') -> Dict:
    """
    对比不同参数集的回测效果
    
    参数：
        records: 历史记录
        param_sets: 参数集列表，如 [{'heat_threshold': 0.1}, {'heat_threshold': 0.2}]
        param_name: 参数名称
    
    返回：
        各参数集的回测结果对比
    """
    results = {}
    
    for i, params in enumerate(param_sets):
        # 这里简化处理，实际应用中应该用不同的参数重新预测
        summary = run_backtest(records, verbose=False)
        results[f'{param_name}_{i}'] = {
            'params': params,
            'summary': summary
        }
    
    return results


# ==================== 测试 ====================

def main():
    print("=== 回测模块测试 ===")
    
    # 测试数据
    test_records = [
        {
            'match_id': 'test_001',
            'league': '英超',
            'home': '曼城',
            'away': '曼联',
            'predicted_scores': {'1-1': 0.25, '2-1': 0.20, '1-0': 0.15, '0-0': 0.10},
            'predicted_1x2': {'home': 0.60, 'draw': 0.25, 'away': 0.15},
            'asian': -1.0,
            'total_line': 2.5,
            'actual_score': '2-1',
            'actual_result': 'H',
        },
        {
            'match_id': 'test_002',
            'league': '英超',
            'home': '阿森纳',
            'away': '切尔西',
            'predicted_scores': {'1-1': 0.30, '0-0': 0.20, '2-1': 0.15, '1-0': 0.12},
            'predicted_1x2': {'home': 0.40, 'draw': 0.35, 'away': 0.25},
            'asian': -0.5,
            'total_line': 2.5,
            'actual_score': '1-1',
            'actual_result': 'D',
        },
    ]
    
    # 运行回测
    summary = run_backtest(test_records, verbose=True)
    
    # 打印结果
    runner = BacktestRunner()
    for record in test_records:
        actual_score = record['actual_score']
        parts = actual_score.split('-')
        home_g, away_g = int(parts[0]), int(parts[1])
        actual_result = 'H' if home_g > away_g else ('A' if home_g < away_g else 'D')
        runner.add_result(record, {}, {'score': actual_score, 'result': actual_result})
    
    runner.print_summary()


if __name__ == '__main__':
    main()
