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


def _normalize_goal_distribution(goal_dist) -> Dict[int, float]:
    """Normalize a total-goals probability distribution to int goal keys."""
    if isinstance(goal_dist, list):
        goal_dist = {
            item.get('goals'): item.get('probability', 0.0)
            for item in goal_dist
            if isinstance(item, dict) and item.get('goals') is not None
        }
    if not isinstance(goal_dist, dict):
        return {}

    normalized = {}
    for goals, probability in goal_dist.items():
        try:
            goal_key = int(goals)
            prob_value = max(0.0, float(probability or 0.0))
        except (TypeError, ValueError):
            continue
        normalized[goal_key] = normalized.get(goal_key, 0.0) + prob_value

    total = sum(normalized.values())
    if total > 0:
        normalized = {key: value / total for key, value in normalized.items()}
    return normalized


def _result_quality_is_usable(record: Dict) -> bool:
    quality = record.get('result_quality') or {}
    grade = quality.get('grade')
    if not grade:
        return True
    return grade not in {'reject', 'low'}


def _has_real_half_full_sample(record: Dict) -> bool:
    return (
        record.get('half_time_data_quality') == 'real'
        and _result_quality_is_usable(record)
    )


def _actual_goals(score: str) -> int:
    try:
        home, away = str(score).split('-')
        return int(home) + int(away)
    except Exception:
        return 0


def _is_draw_score(score: str) -> bool:
    try:
        home, away = str(score).split('-')
        return int(home) == int(away)
    except Exception:
        return False


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
        goal_dist = _normalize_goal_distribution(goal_dist)
        
        # 按概率排序，取Top2
        sorted_totals = sorted(goal_dist.items(), key=lambda x: -x[1])
        top2_totals = [goals for goals, _ in sorted_totals[:2]]
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
        goal_logloss = 0.0
        goal_brier = 0.0
        has_goal_count_data = bool(goal_dist)
        if has_goal_count_data:
            actual_goal_prob = goal_dist.get(actual_goals, 1e-15)
            goal_logloss = -math.log(max(1e-15, min(1 - 1e-15, actual_goal_prob)))
            all_goal_counts = set(goal_dist.keys()) | {actual_goals}
            goal_brier = sum(
                (goal_dist.get(goals, 0.0) - (1.0 if goals == actual_goals else 0.0)) ** 2
                for goals in all_goal_counts
            )

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
        has_htf_data = bool(predicted_htf and actual_htf and _has_real_half_full_sample(record))
        hit_htf_top1 = False
        hit_htf_top3 = False
        htf_logloss = 0.0
        htf_brier = 0.0
        
        if has_htf_data:
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
            'goal_logloss': goal_logloss,
            'goal_brier': goal_brier,
            'has_goal_count_data': has_goal_count_data,
            'result_logloss': result_logloss,
            'result_brier': result_brier,
            # 半全场指标
            'hit_htf_top1': hit_htf_top1,
            'hit_htf_top3': hit_htf_top3,
            'htf_logloss': htf_logloss,
            'htf_brier': htf_brier,
            'has_htf_data': has_htf_data,
            '_predicted_scores': predicted_scores,
            '_time_layers': record.get('time_layers') or {},
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
        if has_goal_count_data:
            if 'goal_logloss' not in stats:
                stats['goal_logloss'] = []
                stats['goal_brier'] = []
            stats['goal_logloss'].append(goal_logloss)
            stats['goal_brier'].append(goal_brier)
        
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

        goal_results = [r for r in self.results if r.get('has_goal_count_data')]
        goal_total = len(goal_results)
        goal_brier_scores = [r['goal_brier'] for r in goal_results]
        goal_log_losses = [r['goal_logloss'] for r in goal_results]
        
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
            'goal_count_total': goal_total,
            'goal_brier': sum(goal_brier_scores) / goal_total if goal_total > 0 else 0,
            'goal_logloss': sum(goal_log_losses) / goal_total if goal_total > 0 else 0,
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
                    'goal_count_total': len(stats.get('goal_logloss', [])),
                    'goal_brier': (
                        sum(stats.get('goal_brier', [])) / len(stats.get('goal_brier', []))
                        if stats.get('goal_brier') else 0
                    ),
                    'goal_logloss': (
                        sum(stats.get('goal_logloss', [])) / len(stats.get('goal_logloss', []))
                        if stats.get('goal_logloss') else 0
                    ),
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
            goal_rows = [r for r in rows if r.get('has_goal_count_data')]
            return {
                'total': total,
                'score_top1': sum(1 for r in rows if r['hit_top1']) / total,
                'score_top3': sum(1 for r in rows if r['hit_top3']) / total,
                'score_top5': sum(1 for r in rows if r['hit_top5']) / total,
                'goal_top2': sum(1 for r in rows if r['hit_total']) / total,
                'goal_count_total': len(goal_rows),
                'goal_logloss': (
                    sum(r['goal_logloss'] for r in goal_rows) / len(goal_rows)
                    if goal_rows else 0
                ),
                'goal_brier': (
                    sum(r['goal_brier'] for r in goal_rows) / len(goal_rows)
                    if goal_rows else 0
                ),
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

        def summarize_time_layers():
            try:
                from .result_sync import time_layer_weight
            except Exception:
                def time_layer_weight(layer):
                    return 1.0 if layer == 'final' else 0.5

            layers = ['T-24h', 'T-6h', 'T-1h', 'T-15min', 'final']
            layer_stats = {}
            for layer in layers:
                total_layer = 0
                top1_hits = 0
                top3_hits = 0
                top5_hits = 0
                weighted_total = 0.0
                weighted_top1_hits = 0.0
                weighted_top3_hits = 0.0
                weighted_top5_hits = 0.0
                score_loglosses = []
                score_briers = []
                weight = time_layer_weight(layer)

                for row in self.results:
                    predicted_scores = (row.get('_time_layers') or {}).get(layer)
                    if layer == 'final' and not predicted_scores:
                        predicted_scores = row.get('_predicted_scores') or {}
                    if not predicted_scores:
                        continue

                    actual_score = row.get('actual_score')
                    if not actual_score:
                        continue
                    sorted_scores = sorted(predicted_scores.items(), key=lambda item: -item[1])
                    top1_score = sorted_scores[0][0] if sorted_scores else None
                    top3_scores = [score for score, _ in sorted_scores[:3]]
                    top5_scores = [score for score, _ in sorted_scores[:5]]

                    total_layer += 1
                    weighted_total += weight
                    if top1_score == actual_score:
                        top1_hits += 1
                        weighted_top1_hits += weight
                    if actual_score in top3_scores:
                        top3_hits += 1
                        weighted_top3_hits += weight
                    if actual_score in top5_scores:
                        top5_hits += 1
                        weighted_top5_hits += weight

                    actual_prob = predicted_scores.get(actual_score, 1e-15)
                    score_loglosses.append(-math.log(max(1e-15, min(1 - 1e-15, actual_prob))))
                    all_scores = set(predicted_scores.keys()) | {actual_score}
                    score_briers.append(sum(
                        (predicted_scores.get(score, 0.0) - (1.0 if score == actual_score else 0.0)) ** 2
                        for score in all_scores
                    ))

                layer_stats[layer] = {
                    'total': total_layer,
                    'weight': weight,
                    'top1': top1_hits / total_layer if total_layer else 0.0,
                    'top3': top3_hits / total_layer if total_layer else 0.0,
                    'top5': top5_hits / total_layer if total_layer else 0.0,
                    'weighted_top1': weighted_top1_hits / weighted_total if weighted_total else 0.0,
                    'weighted_top3': weighted_top3_hits / weighted_total if weighted_total else 0.0,
                    'weighted_top5': weighted_top5_hits / weighted_total if weighted_total else 0.0,
                    'weighted_total': round(weighted_total, 3),
                    'score_logloss': sum(score_loglosses) / len(score_loglosses) if score_loglosses else 0.0,
                    'score_brier': sum(score_briers) / len(score_briers) if score_briers else 0.0,
                }
            return layer_stats

        report = {
            'summary': summary,
            'by_league': group_by(lambda r: r.get('league') or 'unknown'),
            'by_total_line': group_by(lambda r: bucket_total_line(r.get('total_line'))),
            'by_asian_bucket': group_by(lambda r: bucket_asian(r.get('asian'))),
            'by_time_layer': summarize_time_layers(),
        }
        report['diagnostics'] = self.get_bias_diagnostics(report)
        report['diagnostic_suggestions'] = get_diagnostic_tuning_suggestions(report['diagnostics'])
        return report

    def get_bias_diagnostics(self, report: Dict = None) -> Dict:
        if not self.results:
            return {}

        total = len(self.results)
        draw_actual = sum(1 for r in self.results if r.get('actual_result') == 'D') / total
        draw_pred = sum(1 for r in self.results if _is_draw_score(r.get('top1_score'))) / total

        common_scores = {'0-0', '1-0', '0-1', '1-1'}
        common_top1 = sum(1 for r in self.results if r.get('top1_score') in common_scores) / total
        common_actual = sum(1 for r in self.results if r.get('actual_score') in common_scores) / total

        goal_rows = [r for r in self.results if r.get('has_goal_count_data')]
        goal_bias = 0.0
        if goal_rows:
            over_hits = sum(1 for r in goal_rows if _actual_goals(r.get('actual_score')) >= 3) / len(goal_rows)
            over_pred = sum(1 for r in goal_rows if r.get('hit_total') and _actual_goals(r.get('actual_score')) >= 3) / len(goal_rows)
            goal_bias = over_pred - over_hits

        weak_buckets = []
        bucket_tuning_candidates = []

        def time_layer_signal(layer_metrics):
            if not isinstance(layer_metrics, dict):
                return {'available': False, 'reason': 'missing_time_layer_metrics'}

            early_layers = ['T-24h', 'T-6h']
            late_layers = ['T-1h', 'T-15min', 'final']

            def weighted_average(layers, field, min_total=1):
                total_weight = 0.0
                value_sum = 0.0
                sample_total = 0
                for layer in layers:
                    metrics = layer_metrics.get(layer) or {}
                    total_count = metrics.get('total', 0) or 0
                    if total_count < min_total:
                        continue
                    weight = metrics.get('weighted_total') or metrics.get('weight') or 1.0
                    value_sum += (metrics.get(field, 0.0) or 0.0) * weight
                    total_weight += weight
                    sample_total += total_count
                if total_weight <= 0:
                    return None, sample_total
                return value_sum / total_weight, sample_total

            early_top3, early_total = weighted_average(early_layers, 'top3')
            late_top3, late_total = weighted_average(late_layers, 'top3')
            early_logloss, _ = weighted_average(early_layers, 'score_logloss')
            late_logloss, _ = weighted_average(late_layers, 'score_logloss')

            if early_top3 is None or late_top3 is None:
                return {
                    'available': False,
                    'reason': 'not_enough_layer_samples',
                    'early_total': early_total,
                    'late_total': late_total,
                }

            top3_lift = late_top3 - early_top3
            logloss_delta = (early_logloss - late_logloss) if early_logloss is not None and late_logloss is not None else 0.0
            action = 'keep_time_layer_weights'
            confidence = 'low'
            reasons = []

            if top3_lift >= 0.08 and logloss_delta >= -0.05:
                action = 'raise_late_market_weight'
                reasons.append('late_layers_score_top3_better')
                confidence = 'medium' if early_total >= 8 and late_total >= 8 else 'low'
            elif top3_lift <= -0.08 and logloss_delta <= 0.05:
                action = 'lower_late_market_weight'
                reasons.append('early_layers_score_top3_better')
                confidence = 'medium' if early_total >= 8 and late_total >= 8 else 'low'

            return {
                'available': True,
                'action': action,
                'confidence': confidence,
                'early_total': early_total,
                'late_total': late_total,
                'early_top3': round(early_top3, 3),
                'late_top3': round(late_top3, 3),
                'top3_lift': round(top3_lift, 3),
                'early_logloss': round(early_logloss, 3) if early_logloss is not None else None,
                'late_logloss': round(late_logloss, 3) if late_logloss is not None else None,
                'logloss_delta': round(logloss_delta, 3),
                'reasons': reasons,
            }

        def bucket_candidate(section, key, metrics):
            total_count = metrics.get('total', 0)
            score_top3 = metrics.get('score_top3', 0.0) or 0.0
            goal_top2 = metrics.get('goal_top2', 0.0) or 0.0
            htf_total_count = metrics.get('htf_total', 0) or 0
            htf_top3 = metrics.get('htf_top3', 0.0) or 0.0
            deltas = defaultdict(float)
            reasons = []

            if score_top3 < 0.35:
                reasons.append('score_top3_weak')
                if section == 'by_total_line':
                    if key == '<=2.25':
                        deltas['low_score_bias'] += 0.02
                        deltas['high_score_bias'] -= 0.02
                    elif key == '>=3.0':
                        deltas['high_score_bias'] += 0.02
                        deltas['low_score_bias'] -= 0.02
                    else:
                        deltas['draw_bias'] -= 0.015
                elif section == 'by_asian_bucket' and key == 'deep':
                    deltas['draw_bias'] -= 0.02
                    deltas['high_score_bias'] += 0.015
                else:
                    deltas['low_score_bias'] -= 0.015

            if metrics.get('goal_count_total', total_count) >= 5 and goal_top2 < 0.40:
                reasons.append('goal_top2_weak')
                if section == 'by_total_line' and key == '>=3.0':
                    deltas['high_score_bias'] += 0.025
                elif section == 'by_total_line' and key == '<=2.25':
                    deltas['low_score_bias'] += 0.025
                else:
                    deltas['high_score_bias'] += 0.015 if metrics.get('goal_logloss', 0) > 1.2 else 0.0

            if htf_total_count >= 5 and htf_top3 < 0.35:
                reasons.append('half_full_top3_weak')
                deltas['half_full_real_weight'] += 0.03

            if not deltas:
                return None

            severity = max(0.0, 0.35 - score_top3) + max(0.0, 0.40 - goal_top2)
            confidence = 'medium' if total_count >= 12 and severity >= 0.20 else 'low'
            return {
                'section': section,
                'bucket': key,
                'scope': 'league' if section == 'by_league' else 'bucket',
                'total': total_count,
                'score_top3': round(score_top3, 3),
                'goal_top2': round(goal_top2, 3),
                'htf_top3': round(htf_top3, 3),
                'reasons': reasons,
                'param_deltas': {
                    name: round(max(-0.04, min(0.04, value)), 4)
                    for name, value in sorted(deltas.items())
                    if abs(value) >= 0.0001
                },
                'confidence': confidence,
            }

        if report:
            for section in ('by_total_line', 'by_asian_bucket', 'by_league'):
                for key, metrics in report.get(section, {}).items():
                    if not metrics or metrics.get('total', 0) < 5:
                        continue
                    if metrics.get('score_top3', 1.0) < 0.35 or metrics.get('goal_top2', 1.0) < 0.40:
                        weak_buckets.append({
                            'section': section,
                            'bucket': key,
                            'total': metrics.get('total'),
                            'score_top3': round(metrics.get('score_top3', 0), 3),
                            'goal_top2': round(metrics.get('goal_top2', 0), 3),
                            'score_logloss': round(metrics.get('score_logloss', 0), 3),
                        })
                    candidate = bucket_candidate(section, key, metrics)
                    if candidate:
                        bucket_tuning_candidates.append(candidate)

        notes = []
        if draw_pred - draw_actual > 0.06:
            notes.append('draw_top1_overheated')
        elif draw_actual - draw_pred > 0.06:
            notes.append('draw_top1_underweighted')
        if common_top1 - common_actual > 0.08:
            notes.append('common_scores_overheated')
        if abs(goal_bias) > 0.08:
            notes.append('goal_direction_bias')

        return {
            'draw': {
                'actual_rate': round(draw_actual, 3),
                'top1_draw_rate': round(draw_pred, 3),
                'bias': round(draw_pred - draw_actual, 3),
            },
            'common_scores': {
                'actual_rate': round(common_actual, 3),
                'top1_rate': round(common_top1, 3),
                'bias': round(common_top1 - common_actual, 3),
            },
            'goal_direction_bias': round(goal_bias, 3),
            'weak_buckets': sorted(weak_buckets, key=lambda item: (item['score_top3'], item['goal_top2']))[:10],
            'bucket_tuning_candidates': sorted(
                bucket_tuning_candidates,
                key=lambda item: (
                    item['confidence'] != 'medium',
                    item['score_top3'],
                    item['goal_top2'],
                    -item['total'],
                ),
            )[:10],
            'time_layer_signal': time_layer_signal(report.get('by_time_layer', {})) if report else {
                'available': False,
                'reason': 'missing_report',
            },
            'notes': notes,
        }
    
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
        if summary.get('goal_count_total', 0) > 0:
            print(f"Goal Brier Score:       {summary['goal_brier']:.4f}")
            print(f"Goal LogLoss:           {summary['goal_logloss']:.4f}")
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


def get_diagnostic_tuning_suggestions(diagnostics: Dict) -> Dict:
    """Translate backtest bias diagnostics into conservative tuning hints."""
    if not diagnostics:
        return {
            'suggestions': [],
            'param_deltas': {},
            'bucket_reviews': [],
            'bucket_tuning_candidates': [],
            'time_layer_signal': {},
        }

    suggestions = []
    param_deltas = defaultdict(float)
    notes = set(diagnostics.get('notes') or [])

    draw_bias = diagnostics.get('draw', {}).get('bias', 0.0) or 0.0
    common_bias = diagnostics.get('common_scores', {}).get('bias', 0.0) or 0.0
    goal_bias = diagnostics.get('goal_direction_bias', 0.0) or 0.0

    if 'draw_top1_overheated' in notes or draw_bias > 0.06:
        param_deltas['draw_bias'] -= min(0.05, max(0.02, abs(draw_bias) * 0.50))
        suggestions.append({
            'area': 'score_1x2',
            'action': 'lower_draw_bias',
            'reason': 'Top1 draw rate is higher than actual draw rate.',
            'confidence': 'medium',
        })
    elif 'draw_top1_underweighted' in notes or draw_bias < -0.06:
        param_deltas['draw_bias'] += min(0.05, max(0.02, abs(draw_bias) * 0.50))
        suggestions.append({
            'area': 'score_1x2',
            'action': 'raise_draw_bias',
            'reason': 'Actual draw rate is higher than Top1 draw rate.',
            'confidence': 'medium',
        })

    if 'common_scores_overheated' in notes or common_bias > 0.08:
        param_deltas['low_score_bias'] -= min(0.06, max(0.02, common_bias * 0.40))
        suggestions.append({
            'area': 'score_distribution',
            'action': 'reduce_common_low_score_heat',
            'reason': '0-0/1-0/0-1/1-1 are appearing too often as Top1 picks.',
            'confidence': 'medium',
        })

    if 'goal_direction_bias' in notes or abs(goal_bias) > 0.08:
        if goal_bias > 0:
            param_deltas['high_score_bias'] -= min(0.05, max(0.02, abs(goal_bias) * 0.40))
            action = 'lower_over_goal_bias'
            reason = 'High-goal direction is winning less often than the model expects.'
        else:
            param_deltas['high_score_bias'] += min(0.05, max(0.02, abs(goal_bias) * 0.40))
            action = 'raise_over_goal_bias'
            reason = 'High-goal direction is underrepresented in winning goal-count picks.'
        suggestions.append({
            'area': 'goal_count',
            'action': action,
            'reason': reason,
            'confidence': 'medium',
        })

    bucket_reviews = diagnostics.get('weak_buckets') or []
    bucket_candidates = diagnostics.get('bucket_tuning_candidates') or []
    time_signal = diagnostics.get('time_layer_signal') or {}
    if bucket_reviews:
        suggestions.append({
            'area': 'bucket_policy',
            'action': 'review_weak_buckets',
            'reason': 'Some league/handicap/total-line buckets have weak Top3 or goal-count hit rates.',
            'confidence': 'low',
        })
    if bucket_candidates:
        suggestions.append({
            'area': 'bucket_policy',
            'action': 'apply_bucket_specific_micro_tuning',
            'reason': 'Weak buckets have conservative candidate deltas for score, goal-count, or half/full-time policy.',
            'confidence': 'medium' if any(c.get('confidence') == 'medium' for c in bucket_candidates) else 'low',
        })
    if time_signal.get('available') and time_signal.get('action') != 'keep_time_layer_weights':
        suggestions.append({
            'area': 'time_layer_policy',
            'action': time_signal.get('action'),
            'reason': 'Time-layer backtest shows a meaningful difference between early and late market snapshots.',
            'confidence': time_signal.get('confidence', 'low'),
        })

    return {
        'suggestions': suggestions,
        'param_deltas': {key: round(value, 4) for key, value in sorted(param_deltas.items())},
        'bucket_reviews': bucket_reviews[:10],
        'bucket_tuning_candidates': bucket_candidates[:10],
        'time_layer_signal': time_signal,
    }


def _records_with_actual_scores(records: List[Dict]) -> List[Dict]:
    return [record for record in records if record.get('actual_score')]


def rolling_backtest_report(records: List[Dict],
                            windows: Tuple[int, ...] = (30, 60, 90),
                            predict_func: Optional[Callable] = None,
                            verbose: bool = False,
                            quality_filter: bool = True,
                            min_quality_grade: str = 'medium') -> Dict:
    """Run recent-window backtests and return diagnostics for trend checks."""
    records, quality_report = _quality_filter(records, quality_filter, min_quality_grade)
    settled_records = _records_with_actual_scores(records)
    if not settled_records:
        return {
            'error': 'no settled records with actual_score',
            'sample_quality': quality_report,
            'available_samples': 0,
            'windows': {},
            'latest_window': None,
            'diagnostic_suggestions': get_diagnostic_tuning_suggestions({}),
        }

    unique_windows = sorted({int(window) for window in windows if int(window) > 0})
    window_reports = {}
    latest_key = None
    for window in unique_windows:
        window_records = settled_records[-window:]
        report = run_backtest_report(
            window_records,
            predict_func=predict_func,
            verbose=verbose,
            quality_filter=False,
        )
        report['window_size'] = window
        report['sample_count'] = len(window_records)
        window_reports[str(window)] = report
        latest_key = str(window)

    latest_report = window_reports.get(latest_key, {})
    return {
        'available_samples': len(settled_records),
        'sample_quality': quality_report,
        'windows': window_reports,
        'latest_window': latest_key,
        'diagnostic_suggestions': latest_report.get(
            'diagnostic_suggestions',
            get_diagnostic_tuning_suggestions(latest_report.get('diagnostics', {})),
        ),
    }


def rolling_backtest_from_history(league: str = None,
                                  limit: int = None,
                                  windows: Tuple[int, ...] = (30, 60, 90),
                                  predict_func: Optional[Callable] = None,
                                  **kwargs) -> Dict:
    """Load settled prediction history and run rolling-window diagnostics."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return rolling_backtest_report(
            records,
            windows=windows,
            predict_func=predict_func,
            **kwargs,
        )
    except Exception as e:
        return {'error': str(e)}


def build_diagnostic_tuning_plan(rolling_report: Dict,
                                 min_consistent_windows: int = 2,
                                 min_window_samples: int = 30,
                                 max_abs_delta: float = 0.04) -> Dict:
    """Build a guarded tuning plan from rolling-window diagnostic suggestions."""
    if not rolling_report or rolling_report.get('error'):
        return {
            'ready': False,
            'reason': rolling_report.get('error', 'missing_rolling_report') if isinstance(rolling_report, dict) else 'missing_rolling_report',
            'param_deltas': {},
            'window_count': 0,
        }

    windows = rolling_report.get('windows') or {}
    eligible = []
    for key, report in sorted(windows.items(), key=lambda item: int(item[0])):
        sample_count = report.get('sample_count') or report.get('summary', {}).get('total_matches', 0)
        if sample_count < min_window_samples:
            continue
        eligible.append((key, report))

    if len(eligible) < min_consistent_windows:
        return {
            'ready': False,
            'reason': 'not_enough_eligible_windows',
            'required_windows': min_consistent_windows,
            'eligible_windows': len(eligible),
            'min_window_samples': min_window_samples,
            'param_deltas': {},
            'window_count': len(eligible),
        }

    supported_params = {'draw_bias', 'low_score_bias', 'high_score_bias', 'late_market_weight_bias'}
    by_param = defaultdict(list)
    by_time_layer_action = defaultdict(list)
    for key, report in eligible:
        deltas = report.get('diagnostic_suggestions', {}).get('param_deltas') or {}
        for param, delta in deltas.items():
            if param not in supported_params:
                continue
            try:
                delta_value = float(delta)
            except (TypeError, ValueError):
                continue
            if abs(delta_value) < 1e-9:
                continue
            by_param[param].append((key, delta_value))
        time_signal = report.get('diagnostic_suggestions', {}).get('time_layer_signal') or {}
        action = time_signal.get('action')
        if action in {'raise_late_market_weight', 'lower_late_market_weight'}:
            by_time_layer_action[action].append((key, time_signal))

    planned = {}
    consistency = {}
    for param, values in sorted(by_param.items()):
        signs = [1 if delta > 0 else -1 for _, delta in values]
        same_direction = len(set(signs)) == 1
        if len(values) < min_consistent_windows or not same_direction:
            consistency[param] = {
                'accepted': False,
                'windows': [key for key, _ in values],
                'reason': 'insufficient_or_conflicting_direction',
            }
            continue
        avg_delta = sum(delta for _, delta in values) / len(values)
        avg_delta = max(-max_abs_delta, min(max_abs_delta, avg_delta))
        planned[param] = round(avg_delta, 4)
        consistency[param] = {
            'accepted': True,
            'windows': [key for key, _ in values],
            'direction': 'up' if avg_delta > 0 else 'down',
        }

    time_layer_action = None
    time_layer_consistency = {}
    for action, values in sorted(by_time_layer_action.items()):
        if len(values) < min_consistent_windows:
            time_layer_consistency[action] = {
                'accepted': False,
                'windows': [key for key, _ in values],
                'reason': 'not_enough_consistent_windows',
            }
            continue
        opposite = 'lower_late_market_weight' if action == 'raise_late_market_weight' else 'raise_late_market_weight'
        if by_time_layer_action.get(opposite):
            time_layer_consistency[action] = {
                'accepted': False,
                'windows': [key for key, _ in values],
                'reason': 'conflicting_time_layer_direction',
            }
            continue
        avg_top3_lift = sum(float(signal.get('top3_lift', 0.0) or 0.0) for _, signal in values) / len(values)
        avg_logloss_delta = sum(float(signal.get('logloss_delta', 0.0) or 0.0) for _, signal in values) / len(values)
        time_layer_action = {
            'action': action,
            'windows': [key for key, _ in values],
            'avg_top3_lift': round(avg_top3_lift, 4),
            'avg_logloss_delta': round(avg_logloss_delta, 4),
            'confidence': 'medium' if len(values) >= min_consistent_windows else 'low',
        }
        delta_sign = 1 if action == 'raise_late_market_weight' else -1
        planned.setdefault(
            'late_market_weight_bias',
            round(delta_sign * min(max_abs_delta, max(0.01, abs(avg_top3_lift) * 0.20)), 4),
        )
        time_layer_consistency[action] = {
            'accepted': True,
            'windows': [key for key, _ in values],
        }

    return {
        'ready': bool(planned or time_layer_action),
        'reason': 'ready' if (planned or time_layer_action) else 'no_consistent_param_deltas',
        'param_deltas': planned,
        'consistency': consistency,
        'time_layer_action': time_layer_action,
        'time_layer_consistency': time_layer_consistency,
        'eligible_windows': [key for key, _ in eligible],
        'window_count': len(eligible),
        'guards': {
            'min_consistent_windows': min_consistent_windows,
            'min_window_samples': min_window_samples,
            'max_abs_delta': max_abs_delta,
        },
    }


def apply_diagnostic_tuning(records: List[Dict],
                            windows: Tuple[int, ...] = (30, 60, 90),
                            predict_func: Optional[Callable] = None,
                            quality_filter: bool = True,
                            min_quality_grade: str = 'medium',
                            min_consistent_windows: int = 2,
                            min_window_samples: int = 30,
                            max_abs_delta: float = 0.04,
                            scope: str = 'bucket',
                            league: str = None,
                            total_line: float = None,
                            handicap: float = None,
                            dry_run: bool = True) -> Dict:
    """Plan and optionally persist conservative tuning based on rolling diagnostics."""
    rolling_report = rolling_backtest_report(
        records,
        windows=windows,
        predict_func=predict_func,
        verbose=False,
        quality_filter=quality_filter,
        min_quality_grade=min_quality_grade,
    )
    plan = build_diagnostic_tuning_plan(
        rolling_report,
        min_consistent_windows=min_consistent_windows,
        min_window_samples=min_window_samples,
        max_abs_delta=max_abs_delta,
    )
    if not plan.get('ready'):
        return {
            'applied': False,
            'dry_run': dry_run,
            'plan': plan,
            'rolling_report': rolling_report,
        }

    settled_records = _records_with_actual_scores(records)
    sample = settled_records[-1] if settled_records else {}
    target_league = league if league is not None else sample.get('league')
    target_total_line = total_line if total_line is not None else sample.get('total_line')
    target_handicap = handicap if handicap is not None else sample.get('asian')

    try:
        from .prediction_policy import PARAM_RANGES, get_prediction_policy, save_tuning_params

        current = get_prediction_policy(
            league=target_league,
            total_line=target_total_line,
            handicap=target_handicap,
        )
        new_params = {}
        for param, delta in plan['param_deltas'].items():
            base_value = float(current.get(param, 1.0))
            new_value = base_value + float(delta)
            low, high = PARAM_RANGES.get(param, (0.0, 10.0))
            new_params[param] = round(max(low, min(high, new_value)), 4)

        if not new_params:
            return {
                'applied': False,
                'dry_run': dry_run,
                'scope': scope,
                'target': {
                    'league': target_league,
                    'total_line': target_total_line,
                    'handicap': target_handicap,
                },
                'current_params': {},
                'new_params': {},
                'save_result': None,
                'plan': plan,
                'rolling_report': rolling_report,
                'reason': 'plan_contains_non_persisted_actions_only',
            }

        save_result = None
        if not dry_run:
            save_result = save_tuning_params(
                new_params,
                league=target_league,
                total_line=target_total_line,
                handicap=target_handicap,
                scope=scope,
                metrics={
                    'source': 'rolling_diagnostics',
                    'plan': plan,
                    'latest_window': rolling_report.get('latest_window'),
                    'available_samples': rolling_report.get('available_samples'),
                },
            )
        return {
            'applied': not dry_run and bool(save_result and save_result.get('saved')),
            'dry_run': dry_run,
            'scope': scope,
            'target': {
                'league': target_league,
                'total_line': target_total_line,
                'handicap': target_handicap,
            },
            'current_params': {param: current.get(param) for param in plan['param_deltas']},
            'new_params': new_params,
            'save_result': save_result,
            'plan': plan,
            'rolling_report': rolling_report,
        }
    except Exception as e:
        return {
            'applied': False,
            'dry_run': dry_run,
            'error': str(e),
            'plan': plan,
            'rolling_report': rolling_report,
        }


def apply_diagnostic_tuning_from_history(league: str = None,
                                         limit: int = None,
                                         **kwargs) -> Dict:
    """Load settled history and plan/apply guarded diagnostic tuning."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return apply_diagnostic_tuning(records, league=league, **kwargs)
    except Exception as e:
        return {'applied': False, 'error': str(e)}


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
    goal_logloss = summary.get('goal_logloss')
    goal_brier = summary.get('goal_brier')
    if not summary.get('goal_count_total'):
        goal_logloss = score_logloss
        goal_brier = score_brier
    goal_logloss = goal_logloss if goal_logloss is not None else score_logloss
    goal_brier = goal_brier if goal_brier is not None else score_brier
    result_logloss = summary.get('result_logloss', 10.0) or 10.0
    top3 = summary.get('top3_hit_rate', 0.0) or 0.0
    total_hit = summary.get('hit_rate_total', 0.0) or 0.0
    htf_top3 = summary.get('htf_top3_hit_rate', 0.0) or 0.0

    if objective == 'score':
        return score_logloss + 0.35 * score_brier - 0.20 * top3
    if objective == 'goals':
        return goal_logloss + 0.35 * goal_brier - 0.35 * total_hit
    if objective == 'half_full':
        return score_logloss + 0.25 * result_logloss - 0.40 * htf_top3
    if objective == '1x2':
        return result_logloss + 0.25 * score_brier

    return (
        score_logloss
        + 0.30 * score_brier
        + 0.20 * goal_logloss
        + 0.10 * goal_brier
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
                                   min_quality_grade: str = 'medium',
                                   save_best: bool = False,
                                   tuning_scope: str = 'bucket',
                                   tuning_league: str = None,
                                   tuning_total_line: float = None,
                                   tuning_handicap: float = None) -> Dict:
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
        'static_market_cap': [0.10, 0.15, 0.20],
        'change_market_cap': [0.10, 0.15, 0.20],
        'half_full_real_weight': [0.20, 0.25, 0.30],
        'draw_bias': [0.96, 1.00, 1.04],
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
    save_result = None
    if save_best and best:
        try:
            from .prediction_policy import save_tuning_params

            context_records = validation_records or records
            sample = context_records[-1] if context_records else {}
            save_result = save_tuning_params(
                best['params'],
                league=tuning_league if tuning_league is not None else sample.get('league'),
                total_line=tuning_total_line if tuning_total_line is not None else sample.get('total_line'),
                handicap=tuning_handicap if tuning_handicap is not None else sample.get('asian'),
                scope=tuning_scope,
                metrics={
                    'objective': objective,
                    'best_score': best['objective_score'],
                    'validation_count': len(validation_records),
                    'summary': best['summary'],
                },
            )
        except Exception as e:
            save_result = {'saved': False, 'error': str(e)}

    return {
        'objective': objective,
        'best_params': best['params'] if best else {},
        'best_score': best['objective_score'] if best else None,
        'best_summary': best['summary'] if best else {},
        'save_result': save_result,
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


def optimize_policy_buckets(records: List[Dict],
                            predict_func: Callable,
                            group_by: str = 'bucket',
                            min_samples: int = 30,
                            limit_groups: int = None,
                            save_best: bool = True,
                            **kwargs) -> Dict:
    """
    Optimize and optionally persist policy params for multiple leagues/buckets.

    group_by:
        - 'league': one policy per league
        - 'bucket': one policy per league + total bucket + handicap bucket
        - 'market': one policy per total bucket + handicap bucket across leagues
    """
    if predict_func is None:
        return {'error': 'predict_func is required for bucket optimization'}

    try:
        from .prediction_policy import get_handicap_bucket, get_total_bucket
    except Exception as e:
        return {'error': f'prediction_policy unavailable: {e}'}

    groups = defaultdict(list)
    for record in records:
        if not record.get('actual_score'):
            continue

        league = record.get('league') or '*'
        total_bucket = get_total_bucket(record.get('total_line'))
        handicap_bucket = get_handicap_bucket(record.get('asian'))

        if group_by == 'league':
            key = (league, None, None)
        elif group_by == 'market':
            key = ('*', total_bucket, handicap_bucket)
        else:
            key = (league, total_bucket, handicap_bucket)
        groups[key].append(record)

    sortable_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    if limit_groups:
        sortable_groups = sortable_groups[:limit_groups]

    results = {}
    skipped = {}
    for (league, total_bucket, handicap_bucket), group_records in sortable_groups:
        label = f"{league}|{total_bucket or '*'}|{handicap_bucket or '*'}"
        if len(group_records) < min_samples:
            skipped[label] = {
                'reason': 'not_enough_samples',
                'sample_count': len(group_records),
                'required': min_samples,
            }
            continue

        sample = group_records[-1]
        tuning_scope = 'league' if group_by == 'league' else 'bucket'
        tuning_league = None if group_by == 'market' else league

        result = optimize_prediction_parameters(
            group_records,
            predict_func,
            min_samples=min_samples,
            save_best=save_best,
            tuning_scope=tuning_scope,
            tuning_league=tuning_league,
            tuning_total_line=sample.get('total_line'),
            tuning_handicap=sample.get('asian'),
            **kwargs,
        )
        results[label] = result

    return {
        'group_by': group_by,
        'group_count': len(groups),
        'optimized_count': len(results),
        'skipped_count': len(skipped),
        'results': results,
        'skipped': skipped,
    }


def optimize_policy_buckets_from_history(predict_func: Callable,
                                         league: str = None,
                                         limit: int = None,
                                         **kwargs) -> Dict:
    """Load settled history and optimize multiple policy groups."""
    try:
        from .result_sync import get_prediction_records

        records = [r for r in get_prediction_records(include_hidden=True) if r.get('settled')]
        if league:
            records = [r for r in records if r.get('league') == league]
        if limit:
            records = records[-limit:]
        return optimize_policy_buckets(records, predict_func, **kwargs)
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
