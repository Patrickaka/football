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

import itertools
import math
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime


log = logging.getLogger('football')


def _quality_filter(records: List[Dict],
                    enabled: bool = False,
                    min_grade: str = 'medium',
                    exclude_friendly: bool = True) -> Tuple[List[Dict], Dict]:
    if not enabled:
        return records, {
            'enabled': False,
            'input_count': len(records),
            'kept_count': len(records),
            'rejected_count': 0,
        }
    try:
        from .sample_quality import filter_quality_records

        kept, report = filter_quality_records(records, min_grade=min_grade, exclude_friendly=exclude_friendly)
        report['enabled'] = True
        return kept, report
    except Exception as e:
        log.warning(f"样本质量过滤失败，使用原始样本: {e}")
        return records, {
            'enabled': False,
            'error': str(e),
            'input_count': len(records),
            'kept_count': len(records),
            'rejected_count': 0,
        }


def _normalize_1x2_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize 1X2 probability keys to H/D/A."""
    if not probs:
        return {}

    normalized = {
        'H': probs.get('H', probs.get('home', 0.0)),
        'D': probs.get('D', probs.get('draw', 0.0)),
        'A': probs.get('A', probs.get('away', 0.0)),
    }
    total = sum(normalized.values())
    if total > 0:
        normalized = {key: value / total for key, value in normalized.items()}
    return normalized


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
        predicted_scores = prediction.get('predicted_scores') or record.get('predicted_scores', {})
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
        predicted_1x2 = _normalize_1x2_probs(prediction.get('predicted_1x2') or record.get('predicted_1x2', {}))
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
        
        # 从goal_count对象中获取distribution字段（统一格式）
        goal_count_obj = prediction.get('goal_count') or record.get('goal_count', {})
        goal_dist = goal_count_obj.get('distribution_dict', {}) if isinstance(goal_count_obj, dict) else {}
        if not goal_dist and isinstance(goal_count_obj, dict):
            goal_dist = goal_count_obj.get('distribution', {})
        if not goal_dist:
            # 兼容旧格式
            goal_dist = prediction.get('goal_count', {}) or record.get('goal_count', {})
        if isinstance(goal_dist, list):
            goal_dist = {
                item.get('goals'): item.get('probability', 0.0)
                for item in goal_dist
                if isinstance(item, dict) and item.get('goals') is not None
            }
        
        # 按概率排序，取Top2
        sorted_totals = sorted(goal_dist.items(), key=lambda x: -x[1])
        top2_totals = [int(goals) for goals, _ in sorted_totals[:2]]
        hit_total = actual_goals in top2_totals
        
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
        predicted_1x2 = _normalize_1x2_probs(prediction.get('predicted_1x2') or record.get('predicted_1x2', {}))
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
        
        # 计算半全场指标
        predicted_htf = prediction.get('predicted_half_full') or record.get('predicted_half_full', {})
        actual_htf = record.get('actual_half_full')
        hit_htf_top1 = False
        hit_htf_top3 = False
        htf_logloss = 0.0
        htf_brier = 0.0
        
        if predicted_htf and actual_htf:
            sorted_htf = sorted(predicted_htf.items(), key=lambda x: -x[1])
            htf_top1 = sorted_htf[0][0] if sorted_htf else None
            htf_top3 = [k for k, _ in sorted_htf[:3]]
            
            hit_htf_top1 = (actual_htf == htf_top1)
            hit_htf_top3 = (actual_htf in htf_top3)
            
            # 半全场 LogLoss
            actual_prob_htf = predicted_htf.get(actual_htf, 1e-15)
            htf_logloss = -math.log(max(1e-15, min(1 - 1e-15, actual_prob_htf)))
            
            # 半全场 Brier Score
            all_htf = {'HH', 'HD', 'HA', 'DH', 'DD', 'DA', 'AH', 'AD', 'AA'}
            htf_brier = sum(
                (predicted_htf.get(res, 0.0) - (1.0 if res == actual_htf else 0.0)) ** 2
                for res in all_htf
            )
        
        result = {
            'match_id': record.get('match_id'),
            'league': record.get('league'),
            'home': record.get('home'),
            'away': record.get('away'),
            'asian': record.get('asian'),
            'total_line': record.get('total_line'),
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
            # 半全场指标
            'hit_htf_top1': hit_htf_top1,
            'hit_htf_top3': hit_htf_top3,
            'htf_logloss': htf_logloss,
            'htf_brier': htf_brier,
            'has_htf_data': bool(predicted_htf and actual_htf),
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
        stats['brier_scores'].append(score_brier)
        stats['log_losses'].append(score_logloss)
        
        # 半全场统计（只统计有真实半场数据的比赛）
        if result['has_htf_data']:
            if 'htf_top1' not in stats:
                stats['htf_top1'] = 0
                stats['htf_top3'] = 0
                stats['htf_logloss'] = []
                stats['htf_brier'] = []
            if hit_htf_top1: stats['htf_top1'] += 1
            if hit_htf_top3: stats['htf_top3'] += 1
            stats['htf_logloss'].append(htf_logloss)
            stats['htf_brier'].append(htf_brier)
        
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
        
        # 半全场统计（只统计有真实半场数据的比赛）
        htf_results = [r for r in self.results if r['has_htf_data']]
        htf_total = len(htf_results)
        htf_top1_hits = sum(1 for r in htf_results if r['hit_htf_top1'])
        htf_top3_hits = sum(1 for r in htf_results if r['hit_htf_top3'])
        htf_brier_scores = [r['htf_brier'] for r in htf_results]
        htf_log_losses = [r['htf_logloss'] for r in htf_results]
        
        brier_scores = [r['score_brier'] for r in self.results]
        log_losses = [r['score_logloss'] for r in self.results]
        
        result_brier_scores = [r['result_brier'] for r in self.results]
        result_log_losses = [r['result_logloss'] for r in self.results]
        
        summary = {
            'total_matches': total,
            'top1_hit_rate': top1_hits / total,
            'top3_hit_rate': top3_hits / total,
            'top5_hit_rate': top5_hits / total,
            'hit_rate_1x2': hits_1x2 / total,
            'hit_rate_total': hits_total / total,
            'hit_rate_handicap': hits_handicap / total,
            'score_brier': sum(brier_scores) / total,
            'score_logloss': sum(log_losses) / total,
            'result_brier': sum(result_brier_scores) / total,
            'result_logloss': sum(result_log_losses) / total,
            # 半全场指标
            'htf_total': htf_total,
            'htf_top1_hit_rate': htf_top1_hits / htf_total if htf_total > 0 else 0,
            'htf_top3_hit_rate': htf_top3_hits / htf_total if htf_total > 0 else 0,
            'htf_brier': sum(htf_brier_scores) / htf_total if htf_total > 0 else 0,
            'htf_logloss': sum(htf_log_losses) / htf_total if htf_total > 0 else 0,
            'by_league': {},
        }
        
        # 联赛统计
        for league, stats in self.league_stats.items():
            t = stats['total']
            if t > 0:
                league_summary = {
                    'total': t,
                    'top1_hit_rate': stats['top1'] / t,
                    'top3_hit_rate': stats['top3'] / t,
                    'top5_hit_rate': stats['top5'] / t,
                    'hit_rate_1x2': stats['correct_1x2'] / t,
                    'score_brier': sum(stats['brier_scores']) / t,
                    'score_logloss': sum(stats['log_losses']) / t,
                }
                # 半全场统计
                if 'htf_top1' in stats:
                    htf_t = len(stats['htf_logloss'])
                    league_summary['htf_total'] = htf_t
                    league_summary['htf_top1_hit_rate'] = stats['htf_top1'] / htf_t if htf_t > 0 else 0
                    league_summary['htf_top3_hit_rate'] = stats['htf_top3'] / htf_t if htf_t > 0 else 0
                    league_summary['htf_brier'] = sum(stats['htf_brier']) / htf_t if htf_t > 0 else 0
                    league_summary['htf_logloss'] = sum(stats['htf_logloss']) / htf_t if htf_t > 0 else 0
                
                summary['by_league'][league] = league_summary
        
        return summary

    def get_detailed_report(self) -> Dict:
        """Return unified metrics with league, total-line, and handicap buckets."""
        summary = self.get_summary()
        if 'error' in summary:
            return summary

        def bucket_total_line(value):
            if value is None:
                return 'unknown'
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 'unknown'
            if value <= 2.25:
                return '<=2.25'
            if value <= 2.75:
                return '2.5-2.75'
            return '>=3.0'

        def bucket_asian(value):
            if value is None:
                return 'unknown'
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 'unknown'
            abs_value = abs(value)
            if abs_value <= 0.25:
                return 'level_or_quarter'
            if abs_value <= 0.75:
                return 'half_to_0.75'
            return 'deep'

        def summarize(rows):
            total = len(rows)
            if total == 0:
                return {}
            htf_rows = [r for r in rows if r['has_htf_data']]
            return {
                'total': total,
                'score_top1': sum(1 for r in rows if r['hit_top1']) / total,
                'score_top3': sum(1 for r in rows if r['hit_top3']) / total,
                'score_top5': sum(1 for r in rows if r['hit_top5']) / total,
                'goal_top2': sum(1 for r in rows if r['hit_total']) / total,
                'result_1x2': sum(1 for r in rows if r['hit_1x2']) / total,
                'score_logloss': sum(r['score_logloss'] for r in rows) / total,
                'score_brier': sum(r['score_brier'] for r in rows) / total,
                'htf_total': len(htf_rows),
                'htf_top1': (
                    sum(1 for r in htf_rows if r['hit_htf_top1']) / len(htf_rows)
                    if htf_rows else 0
                ),
                'htf_top3': (
                    sum(1 for r in htf_rows if r['hit_htf_top3']) / len(htf_rows)
                    if htf_rows else 0
                ),
            }

        def group_by(key_func):
            grouped = defaultdict(list)
            for row in self.results:
                grouped[key_func(row)].append(row)
            return {key: summarize(rows) for key, rows in sorted(grouped.items())}

        report = {
            'summary': summary,
            'by_league': group_by(lambda r: r.get('league') or 'unknown'),
            'by_total_line': group_by(lambda r: bucket_total_line(r.get('total_line'))),
            'by_asian_bucket': group_by(lambda r: bucket_asian(r.get('asian'))),
        }
        return report
    
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
        print(f"比分 Brier Score:     {summary['score_brier']:.4f}")
        print(f"比分 LogLoss:         {summary['score_logloss']:.4f}")
        print(f"胜平负 Brier Score:   {summary['result_brier']:.4f}")
        print(f"胜平负 LogLoss:       {summary['result_logloss']:.4f}")
        print("-" * 60)
        if summary['htf_total'] > 0:
            print(f"半全场统计 (有真实半场数据 {summary['htf_total']} 场):")
            print(f"  半全场 Top1 命中率: {summary['htf_top1_hit_rate']:.2%}")
            print(f"  半全场 Top3 命中率: {summary['htf_top3_hit_rate']:.2%}")
            print(f"  半全场 Brier Score: {summary['htf_brier']:.4f}")
            print(f"  半全场 LogLoss:     {summary['htf_logloss']:.4f}")
        else:
            print("半全场统计: 暂无真实半场数据")
        
        if summary['by_league']:
            print("-" * 60)
            print("按联赛统计:")
            for league, stats in summary['by_league'].items():
                htf_info = ""
                if stats.get('htf_total', 0) > 0:
                    htf_info = f", HTF Top1={stats['htf_top1_hit_rate']:.2%}"
                print(f"  {league} ({stats['total']}场): Top1={stats['top1_hit_rate']:.2%}, "
                      f"Top3={stats['top3_hit_rate']:.2%}, 1x2={stats['hit_rate_1x2']:.2%}{htf_info}")
        
        print("=" * 60)


def run_backtest(records: List[Dict], 
                predict_func: Optional[Callable] = None,
                verbose: bool = True,
                quality_filter: bool = False,
                min_quality_grade: str = 'medium') -> Dict:
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
    records, quality_report = _quality_filter(records, quality_filter, min_quality_grade)
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
        prediction = {}
        if predict_func:
            try:
                prediction = predict_func(record)
            except Exception as e:
                log.warning(f"预测失败: {e}")
                continue

        result = runner.add_result(record, prediction, actual)
        
        if verbose and result['hit_top1']:
            log.info(f"命中: {record['home']} vs {record['away']} -> "
                    f"{actual_score} (预测: {result['top1_score']})")
    
    summary = runner.get_summary()
    summary['sample_quality'] = quality_report
    return summary


def run_backtest_report(records: List[Dict],
                        predict_func: Optional[Callable] = None,
                        verbose: bool = False,
                        quality_filter: bool = False,
                        min_quality_grade: str = 'medium') -> Dict:
    """Run backtest and return unified detailed report."""
    records, quality_report = _quality_filter(records, quality_filter, min_quality_grade)
    runner = BacktestRunner()

    for record in records:
        actual_score = record.get('actual_score')
        if not actual_score:
            continue

        actual_result = record.get('actual_result')
        if not actual_result:
            try:
                home_g, away_g = map(int, actual_score.split('-'))
                actual_result = 'H' if home_g > away_g else 'A' if home_g < away_g else 'D'
            except Exception:
                continue

        prediction = {}
        if predict_func:
            try:
                prediction = predict_func(record)
            except Exception as e:
                log.warning(f"预测失败: {e}")
                continue

        result = runner.add_result(record, prediction, {'score': actual_score, 'result': actual_result})
        if verbose and result['hit_top1']:
            log.info(f"命中: {record.get('home')} vs {record.get('away')} -> {actual_score}")

    report = runner.get_detailed_report()
    report['sample_quality'] = quality_report
    return report


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


def backtest_report_from_history(league: str = None,
                                 limit: int = None,
                                 quality_filter: bool = True,
                                 min_quality_grade: str = 'medium') -> Dict:
    """Build detailed report from settled prediction history."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return run_backtest_report(
            records,
            verbose=False,
            quality_filter=quality_filter,
            min_quality_grade=min_quality_grade,
        )
    except Exception as e:
        return {'error': str(e)}


def _expand_param_grid(param_grid: Dict[str, List]) -> List[Dict]:
    keys = list(param_grid.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, values)) for values in itertools.product(*(param_grid[key] for key in keys))]


def _objective_score(summary: Dict, objective: str = 'balanced') -> float:
    if summary.get('error'):
        return float('inf')

    score_logloss = summary.get('score_logloss', 10.0) or 10.0
    score_brier = summary.get('score_brier', 10.0) or 10.0
    result_logloss = summary.get('result_logloss', 10.0) or 10.0
    top3 = summary.get('top3_hit_rate', 0.0) or 0.0
    total_hit = summary.get('hit_rate_total', 0.0) or 0.0
    htf_top3 = summary.get('htf_top3_hit_rate', 0.0) or 0.0

    if objective == 'score':
        return score_logloss + 0.35 * score_brier - 0.20 * top3
    if objective == 'goals':
        return score_logloss + 0.25 * score_brier - 0.45 * total_hit
    if objective == 'half_full':
        return score_logloss + 0.25 * result_logloss - 0.40 * htf_top3
    if objective == '1x2':
        return result_logloss + 0.25 * score_brier

    return (
        score_logloss
        + 0.30 * score_brier
        + 0.25 * result_logloss
        - 0.20 * top3
        - 0.25 * total_hit
        - 0.15 * htf_top3
    )


def optimize_prediction_parameters(records: List[Dict],
                                   predict_func: Callable,
                                   param_grid: Optional[Dict[str, List]] = None,
                                   objective: str = 'balanced',
                                   validation_ratio: float = 0.25,
                                   min_samples: int = 30,
                                   quality_filter: bool = True,
                                   min_quality_grade: str = 'medium') -> Dict:
    """
    Search parameter combinations on quality-filtered historical records.

    predict_func must accept: predict_func(record, **params) -> prediction
    """
    if predict_func is None:
        return {'error': 'predict_func is required for parameter optimization'}

    records, quality_report = _quality_filter(records, quality_filter, min_quality_grade)
    records = [record for record in records if record.get('actual_score')]
    if len(records) < min_samples:
        return {
            'error': f'not enough quality samples: {len(records)} < {min_samples}',
            'sample_quality': quality_report,
        }

    default_grid = {
        'market_db_weight': [0.10, 0.15, 0.20, 0.25],
        'bayesian_weight': [0.00, 0.05, 0.10],
        'goal_calibration_weight': [0.10, 0.20, 0.30],
        'half_full_history_weight': [0.10, 0.20, 0.30],
    }
    param_sets = _expand_param_grid(param_grid or default_grid)

    split_at = max(1, int(len(records) * (1 - validation_ratio)))
    train_records = records[:split_at]
    validation_records = records[split_at:] or records[-max(1, len(records) // 5):]

    trials = []
    for params in param_sets:
        report = run_backtest_report(
            validation_records,
            predict_func=lambda record, p=params: predict_func(record, **p),
            verbose=False,
            quality_filter=False,
        )
        summary = report.get('summary', {})
        trials.append({
            'params': params,
            'objective_score': _objective_score(summary, objective),
            'summary': summary,
        })

    trials.sort(key=lambda item: item['objective_score'])
    best = trials[0] if trials else None
    return {
        'objective': objective,
        'best_params': best['params'] if best else {},
        'best_score': best['objective_score'] if best else None,
        'best_summary': best['summary'] if best else {},
        'trial_count': len(trials),
        'train_count': len(train_records),
        'validation_count': len(validation_records),
        'sample_quality': quality_report,
        'top_trials': trials[:10],
    }


def optimize_parameters_from_history(predict_func: Callable,
                                     league: str = None,
                                     limit: int = None,
                                     **kwargs) -> Dict:
    """Load settled prediction history and run parameter optimization."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return optimize_prediction_parameters(records, predict_func, **kwargs)
    except Exception as e:
        return {'error': str(e)}


def compare_parameters(records: List[Dict], 
                     param_sets: List[Dict],
                     predict_func: Optional[Callable] = None,
                     param_name: str = 'params') -> Dict:
    """
    对比不同参数集的回测效果（真正的参数回测）
    
    参数：
        records: 历史记录，每条记录需要包含进行预测所需的特征数据
        param_sets: 参数集列表，如 [{'market_weight': 0.1}, {'market_weight': 0.2}]
        predict_func: 预测函数，签名为 predict_func(record, **params) -> prediction
        param_name: 参数名称
    
    返回：
        各参数集的回测结果对比
    """
    results = {}
    
    for i, params in enumerate(param_sets):
        runner = BacktestRunner()
        params_str = ', '.join(f'{k}={v}' for k, v in params.items())
        
        log.info(f"正在回测参数集 {i+1}/{len(param_sets)}: {params_str}")
        
        for record in records:
            actual_score = record.get('actual_score')
            if not actual_score:
                continue
            
            actual_result = record.get('actual_result')
            if not actual_result:
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
            
            if predict_func:
                try:
                    prediction = predict_func(record, **params)
                    runner.add_result(record, prediction, actual)
                except Exception as e:
                    log.warning(f"预测失败: {e}")
                    continue
            else:
                runner.add_result(record, {}, actual)
        
        summary = runner.get_summary()
        results[f'{param_name}_{i}'] = {
            'params': params,
            'params_str': params_str,
            'summary': summary
        }
        
        log.info(f"参数集 {i+1} 回测完成: Top1命中率={summary['top1_hit_rate']:.2%}, "
                f"Top3命中率={summary['top3_hit_rate']:.2%}, "
                f"Brier={summary['score_brier']:.4f}")
    
    return results


def compare_key_parameters(records: List[Dict], 
                          predict_func: Callable) -> Dict:
    """
    对比关键参数的回测效果（预设参数集）
    
    参数：
        records: 历史记录
        predict_func: 预测函数
    
    返回：
        参数对比结果
    """
    comparisons = {}
    
    # 1. market_db 融合权重对比
    log.info("=== 对比 market_db 融合权重 ===")
    market_weights = [
        {'market_db_weight': 0.1},
        {'market_db_weight': 0.2},
        {'market_db_weight': 0.3},
        {'market_db_weight': 0.4},
    ]
    comparisons['market_db_weight'] = compare_parameters(
        records, market_weights, predict_func, 'market_db_weight'
    )
    
    # 2. 欧赔融合权重对比
    log.info("=== 对比 CLOSE_BLEND_WEIGHT ===")
    blend_weights = [
        {'close_blend_weight': 0.65},
        {'close_blend_weight': 0.72},
        {'close_blend_weight': 0.80},
    ]
    comparisons['close_blend_weight'] = compare_parameters(
        records, blend_weights, predict_func, 'close_blend_weight'
    )
    
    # 3. 热门比分过滤惩罚对比
    log.info("=== 对比 HEAT_FILTER_PENALTY ===")
    heat_penalties = [
        {'heat_filter_penalty': 0.0},
        {'heat_filter_penalty': 0.1},
        {'heat_filter_penalty': 0.2},
        {'heat_filter_penalty': 0.3},
    ]
    comparisons['heat_filter_penalty'] = compare_parameters(
        records, heat_penalties, predict_func, 'heat_filter_penalty'
    )
    
    # 4. 冷门比分奖励对比
    log.info("=== 对比 COLD_FILTER_BONUS ===")
    cold_bonuses = [
        {'cold_filter_bonus': 0.0},
        {'cold_filter_bonus': 0.05},
        {'cold_filter_bonus': 0.1},
        {'cold_filter_bonus': 0.15},
    ]
    comparisons['cold_filter_bonus'] = compare_parameters(
        records, cold_bonuses, predict_func, 'cold_filter_bonus'
    )
    
    # 5. 置信度阈值对比
    log.info("=== 对比 CONFIDENCE_LOW_THRESHOLD ===")
    confidence_thresholds = [
        {'confidence_low_threshold': 0.3},
        {'confidence_low_threshold': 0.4},
        {'confidence_low_threshold': 0.5},
        {'confidence_low_threshold': 0.6},
    ]
    comparisons['confidence_low_threshold'] = compare_parameters(
        records, confidence_thresholds, predict_func, 'confidence_low_threshold'
    )
    
    # 6. 先验奖励系数对比
    log.info("=== 对比 prior_bonus ===")
    prior_bonuses = [
        {'prior_bonus': 0.5},
        {'prior_bonus': 1.0},
        {'prior_bonus': 1.5},
        {'prior_bonus': 2.0},
    ]
    comparisons['prior_bonus'] = compare_parameters(
        records, prior_bonuses, predict_func, 'prior_bonus'
    )
    
    # 7. 强弱分明比赛过滤冷门比分
    log.info("=== 对比 强弱分明比赛过滤冷门 ===")
    upset_filters = [
        {'filter_upset_in_imbalanced': False},
        {'filter_upset_in_imbalanced': True},
    ]
    comparisons['filter_upset'] = compare_parameters(
        records, upset_filters, predict_func, 'filter_upset'
    )
    
    return comparisons


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
