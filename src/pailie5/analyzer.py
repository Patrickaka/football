#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排列五主分析器：历史数据管理、统计分析、集成预测、滚动回测"""

import logging
import random
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from ..common import repositories
from .caching import _load_backtest_cache, _save_backtest_cache
from .config import HISTORY_URL, NUMBERS, RECOMMEND_GROUPS
from .features import (
    _recent_slice, average_miss_cycle_5, build_markov_pos,
    default_window_weights, ensemble_digit_scores_multi_window,
    ensemble_sum_span_5, exp_weighted_counts, load_window_weights,
    markov_prob_smoothed, miss_value_pos, pick_dan_kill,
    save_window_weights,
)
from .pool import backtest_window_weights, generate_pool

logger = logging.getLogger(__name__)


class Pailie5Analyzer:
    """
    排列五分析器 V2.0
    """

    def __init__(self):
        self.history: List[Dict] = []
        self._load_history()
        if not self.history:
            self.fetch_history_data(90)

    def _load_history(self):
        try:
            self.history = repositories.pailie5_load()
            logger.info(f"已加载 {len(self.history)} 期排列五历史数据")
        except Exception as e:
            logger.error(f"加载排列五历史数据失败: {e}")
            self.history = []

    def _save_history(self):
        try:
            repositories.pailie5_save(self.history)
        except Exception as e:
            logger.error(f"保存排列五历史数据失败: {e}")

    def _fetch_history_data_internal(self, days: int = 30):
        try:
            url = HISTORY_URL
            logger.info(f"正在抓取排列五历史数据: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")

            issue_tags = re.findall(r'<td>[^<]*期[^<]*</td>', html)
            issues = []
            for tag in issue_tags:
                match = re.search(r'(\d{7})', tag)
                if match:
                    issues.append(match.group(1))

            dates = re.findall(r'<td>\s*(\d{4}-\d{2}-\d{2})\s*</td>', html)
            balls = re.findall(r'<span class="ball">(\d)</span>', html)

            if not issues or not balls:
                logger.error("无法解析排列五数据，未找到期号或数字球")
                return 0

            count = 0
            for i in range(min(len(issues), len(dates)) - 1, -1, -1):
                ball_start = i * 5
                ball_end = ball_start + 5
                if ball_end <= len(balls):
                    numbers = [int(b) for b in balls[ball_start:ball_end]]
                    date = dates[i] if i < len(dates) else None
                    if self.add_result(issues[i], numbers, date):
                        count += 1

            logger.info(f"成功抓取 {count} 期排列五数据")
            return count

        except Exception as e:
            logger.error(f"抓取排列五历史数据失败：{e}")
            return 0

    def fetch_history_data(self, days: int = 30, force_refresh: bool = False):
        try:
            if not force_refresh:
                from ..common.data_cache import is_cache_valid
                if is_cache_valid('pailie5'):
                    logger.info("使用缓存的排列五数据")
                    self._load_history()
                    return len(self.history)
            count = self._fetch_history_data_internal(days)
            if count > 0:
                self._save_history()
                from ..common.data_cache import save_cached_data
                save_cached_data('pailie5', self.history)
            return count
        except Exception as e:
            logger.warning(f"缓存/抓取失败，使用已有历史数据: {e}")
            self._load_history()
            return len(self.history)

    def _calculate_date_from_issue(self, issue: str) -> str:
        try:
            if len(issue) != 7:
                return datetime.now().strftime('%Y-%m-%d')
            year = int(issue[:4])
            day_of_year = int(issue[4:])
            date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            return date.strftime('%Y-%m-%d')
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')

    def add_result(self, issue: str, numbers: List[int], date: str = None):
        if len(numbers) != 5:
            return False
        for n in numbers:
            if n < 0 or n > 9:
                return False
        if not date:
            date = self._calculate_date_from_issue(issue)
        for r in self.history:
            if r['issue'] == issue:
                r['numbers'] = numbers
                r['date'] = date
                r['timestamp'] = datetime.now().isoformat()
                self._save_history()
                return True
        self.history.append({
            'issue': issue,
            'numbers': numbers,
            'date': date,
            'timestamp': datetime.now().isoformat()
        })
        self.history.sort(key=lambda x: x['issue'], reverse=True)
        self._save_history()
        return True

    def fetch_latest_results(self, count: int = 10, force_refresh: bool = False) -> Dict:
        try:
            fetched_count = self.fetch_history_data(days=1, force_refresh=force_refresh)
            recent = self.get_recent_results(count)
            return {
                'success': fetched_count > 0,
                'source': 'web' if fetched_count > 0 else 'local',
                'count': fetched_count if fetched_count > 0 else len(recent),
                'message': f'成功抓取 {fetched_count} 期数据' if fetched_count > 0 else '使用本地数据',
                'latest_issue': recent[0]['issue'] if recent else None,
                'results': recent
            }
        except Exception:
            logger.error("排列五抓取失败", exc_info=True)
            recent = self.get_recent_results(count)
            return {'success': False, 'source': 'local', 'count': len(recent),
                    'message': '抓取异常，使用本地数据',
                    'latest_issue': recent[0]['issue'] if recent else None, 'results': recent}

    def get_recent_results(self, count: int = 10) -> List[Dict]:
        return self.history[:count]

    # ==================== 统计分析 ====================

    def _get_numbers_list(self):
        """获取历史号码列表（按开奖时间从旧到新）"""
        return [r["numbers"] for r in self.history]

    def analyze_position_frequency(self) -> List[Dict[int, int]]:
        position_freq = []
        for pos in range(5):
            freq = {n: 0 for n in NUMBERS}
            for result in self.history:
                freq[result['numbers'][pos]] += 1
            position_freq.append(freq)
        return position_freq

    def analyze_frequency(self) -> Dict[int, int]:
        freq = {n: 0 for n in NUMBERS}
        for result in self.history:
            for n in result['numbers']:
                freq[n] += 1
        return freq

    def get_hot_numbers(self, top_n: int = 5) -> List[Tuple[int, int]]:
        numbers_list = self._get_numbers_list()
        recent = _recent_slice(numbers_list, 90)
        freq = exp_weighted_counts([d for n in recent for d in n])
        return freq.most_common(top_n)

    def get_cold_numbers(self, top_n: int = 5) -> List[Tuple[int, int]]:
        numbers_list = self._get_numbers_list()
        recent = _recent_slice(numbers_list, 90)
        freq = exp_weighted_counts([d for n in recent for d in n])
        all_digits = [(d, freq.get(d, 0)) for d in range(10)]
        return sorted(all_digits, key=lambda x: x[1])[:top_n]

    def analyze_current_gaps(self) -> Dict[int, int]:
        numbers_list = self._get_numbers_list()
        gaps = {}
        for d in range(10):
            gaps[d] = miss_value_pos(numbers_list, d)
        return gaps

    def analyze_average_gaps(self) -> Dict[int, float]:
        numbers_list = self._get_numbers_list()
        avg_gaps = {}
        for d in range(10):
            avg_gaps[d] = round(average_miss_cycle_5(numbers_list, d), 2)
        return avg_gaps

    def analyze_sum(self) -> Dict:
        sums = [sum(r['numbers']) for r in self.history]
        if not sums:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        sum_counts = Counter(sums)
        return {
            'min': min(sums), 'max': max(sums),
            'avg': round(sum(sums) / len(sums), 2),
            'most_common': sum_counts.most_common(5),
            'distribution': dict(sum_counts)
        }

    def analyze_span(self) -> Dict:
        spans = [max(r['numbers']) - min(r['numbers']) for r in self.history]
        if not spans:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        span_counts = Counter(spans)
        return {
            'min': min(spans), 'max': max(spans),
            'avg': round(sum(spans) / len(spans), 2),
            'most_common': span_counts.most_common(5),
            'distribution': dict(span_counts)
        }

    def analyze_odd_even(self) -> Dict:
        odd_counts = [sum(1 for n in r['numbers'] if n % 2 == 1) for r in self.history]
        if not odd_counts:
            return {'distribution': {}, 'most_common': []}
        dist = Counter(odd_counts)
        return {'distribution': dict(dist), 'most_common': dist.most_common(3)}

    def analyze_size(self) -> Dict:
        small_counts = [sum(1 for n in r['numbers'] if n <= 4) for r in self.history]
        if not small_counts:
            return {'distribution': {}, 'most_common': []}
        dist = Counter(small_counts)
        return {'distribution': dict(dist), 'most_common': dist.most_common(3)}

    def analyze_road(self) -> Dict:
        road_counts = {0: 0, 1: 0, 2: 0}
        for result in self.history:
            for n in result['numbers']:
                road_counts[n % 3] += 1
        total = sum(road_counts.values()) or 1
        return {
            'distribution': road_counts,
            'most_common': sorted(road_counts.items(), key=lambda x: -x[1]),
            'road_numbers': {
                0: [n for n in NUMBERS if n % 3 == 0],
                1: [n for n in NUMBERS if n % 3 == 1],
                2: [n for n in NUMBERS if n % 3 == 2]
            }
        }

    def analyze_transition_matrix(self) -> List[List[int]]:
        matrix = [[0] * 10 for _ in range(10)]
        for i in range(1, len(self.history)):
            prev_nums = self.history[i - 1]['numbers']
            curr_nums = self.history[i]['numbers']
            for p in prev_nums:
                for c in curr_nums:
                    matrix[p][c] += 1
        return matrix

    def bayesian_score(self) -> Dict[int, float]:
        """增强版贝叶斯评分（整合多窗口评分）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        min_s = min(score)
        max_s = max(score) - min_s + 1e-9
        return {d: round((score[d] - min_s) / max_s, 4) for d in range(10)}

    def multi_model_voting(self) -> List[int]:
        """多模型投票（使用增强版评分）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)

        votes = Counter()

        # 模型1：全局评分 Top5
        top5_global = [d for d, _ in sorted(enumerate(score), key=lambda x: -x[1])[:5]]
        for d in top5_global:
            votes[d] += 2

        # 模型2：位置级马尔可夫 Top5（每个位置的Top预测）
        last = numbers_list[-1] if numbers_list else [0] * 5
        for pos in range(5):
            trans = build_markov_pos(numbers_list, pos)
            prev_d = last[pos]
            row = trans.get(prev_d, Counter())
            probs = markov_prob_smoothed(row, range(10))
            top_markov = [d for d, _ in sorted(probs.items(), key=lambda x: -x[1])[:3]]
            for d in top_markov:
                votes[d] += 1

        # 模型3：热号 Top5
        hot = [d for d, _ in self.get_hot_numbers(5)]
        for d in hot:
            votes[d] += 1

        # 返回得票最多的5个数字
        return [d for d, _ in votes.most_common(5)]

    def generate_recommendation(self, method: str = 'balanced') -> List[int]:
        """生成推荐号码（5个数字）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)

        if method == 'random':
            return [random.randint(0, 9) for _ in range(5)]

        sorted_digits = [d for d, _ in sorted(enumerate(score), key=lambda x: -x[1])]
        hot = sorted_digits[:5]
        cold = sorted_digits[-5:]

        if method == 'hot':
            return random.sample(hot, 5)
        if method == 'cold':
            return random.sample(cold, 5)

        # balanced: 热冷混合
        hot_count = random.choice([2, 3])
        cold_count = 5 - hot_count
        result = random.sample(hot, hot_count) + random.sample(cold, cold_count)
        random.shuffle(result)
        return result

    def rank_model(self, top_n: int = 5) -> List[Tuple[int, float]]:
        """排名模型：综合多特征评分"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        return sorted(enumerate(score), key=lambda x: -x[1])[:top_n]

    def identify_cycles(self) -> Dict[str, List]:
        """识别冷热周期状态"""
        freq = self.analyze_frequency()
        gaps = self.analyze_current_gaps()
        avg_gaps = self.analyze_average_gaps()
        avg_freq = sum(freq.values()) / len(freq) if freq else 1
        cycles = {'hot': [], 'cold': [], 'warming': [], 'cooling': [], 'stable': []}
        for n in NUMBERS:
            freq_dev = freq[n] / avg_freq if avg_freq > 0 else 1
            if freq_dev > 1.10:
                cycles['hot'].append(n)
            elif freq_dev < 0.90:
                cycles['cold'].append(n)
            else:
                cycles['stable'].append(n)
            if avg_gaps[n] > 0 and gaps[n] < avg_gaps[n] * 0.7:
                cycles['warming'].append(n)
            elif avg_gaps[n] > 0 and gaps[n] > avg_gaps[n] * 1.5:
                cycles['cooling'].append(n)
        return cycles

    def ensemble_predict(self) -> Dict:
        """集成预测（使用增强评分系统）"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()

        # 生成推荐池
        pool = generate_pool(numbers_list, window_weights, top_n=RECOMMEND_GROUPS, apply_dedup=True)
        top30_str = [num for _, num in pool]
        top5 = [list(map(int, num)) for num in top30_str[:5]]

        # 综合数字评分
        score = ensemble_digit_scores_multi_window(numbers_list, window_weights)
        dan, kill = pick_dan_kill(score, top_dan=2, top_kill=2)

        # 周期分析
        cycles = self.identify_cycles()

        return {
            'prediction': top5[0] if top5 else [0, 1, 2, 3, 4],
            'top5_combos': top5,
            'top30': top30_str,
            'dan': dan,
            'kill': kill,
            'cycles': cycles,
            'ranked_numbers': self.rank_model(10),
        }

    def rolling_backtest(self, trials: int = 50, use_cache: bool = True) -> Dict:
        """滚动回测（改进版：评估推荐池覆盖率）

        use_cache: 是否使用磁盘缓存（默认True，今日已有缓存则直接返回）
        """
        # 尝试从磁盘缓存加载
        if use_cache:
            cached_result, _ = _load_backtest_cache(len(self.history))
            if cached_result is not None:
                logger.info("rolling_backtest：使用磁盘缓存，跳过计算")
                return cached_result

        numbers_list = self._get_numbers_list()
        n = len(numbers_list)
        if n < 30:
            logger.warning(f"排列五回测数据不足，仅 {n} 期")
            return {
                'trials': 0, 'top30_hit': 0, 'top30_rate': 0,
                'ge2_digit_hit': 0, 'ge2_digit_rate': 0,
                'ge3_digit_hit': 0, 'ge3_digit_rate': 0,
                'avg_digit_coverage': 0.0, 'predictions': []
            }
        if n < trials + 10:
            trials = max(20, n - 10)

        start = n - trials
        window_weights = default_window_weights()

        top30_hit = ge2_hit = ge3_hit = 0
        total_coverage = 0.0
        predictions = []

        for i in range(start, n):
            train = numbers_list[:i]
            actual = numbers_list[i]
            actual_str = ''.join(map(str, actual))
            actual_set = set(actual)

            pool = generate_pool(train, window_weights, top_n=30, apply_dedup=False)
            pool_strs = [num for _, num in pool]

            if actual_str in pool_strs:
                top30_hit += 1

            coverage = 0
            for num_str in pool_strs:
                overlap = len({int(c) for c in num_str} & actual_set)
                coverage = max(coverage, overlap)

            total_coverage += coverage / 5.0

            # 至少覆盖2/3个数字
            best_overlap = max((len({int(c) for c in s} & actual_set) for s in pool_strs), default=0)
            if best_overlap >= 2:
                ge2_hit += 1
            if best_overlap >= 3:
                ge3_hit += 1

            predictions.append({
                'actual': actual_str,
                'top30_hit': actual_str in pool_strs[:30],
                'best_overlap': best_overlap,
            })

        result = {
            'trials': trials,
            'top30_hit': top30_hit,
            'top30_rate': round(top30_hit / trials, 4),
            'ge2_digit_hit': ge2_hit,
            'ge2_digit_rate': round(ge2_hit / trials, 4),
            'ge3_digit_hit': ge3_hit,
            'ge3_digit_rate': round(ge3_hit / trials, 4),
            'avg_digit_coverage': round(total_coverage / trials, 4),
            'predictions': predictions[-10:],  # 最近10条
        }

        # 保存回测结果到磁盘缓存
        _save_backtest_cache(result, time.time(), len(numbers_list))

        return result

    def optimize_window_weights(self, trials: int = 80):
        """通过回测优化窗口权重并持久化"""
        numbers_list = self._get_numbers_list()
        weights = backtest_window_weights(numbers_list, trials=trials)
        latest_issue = self.history[0]['issue'] if self.history else None
        save_window_weights(weights, period=latest_issue)
        logger.info(f"窗口权重已更新: {weights}")
        return weights

    def get_statistics(self) -> Dict:
        """获取完整统计信息"""
        numbers_list = self._get_numbers_list()
        window_weights = load_window_weights()
        ss = ensemble_sum_span_5(numbers_list, window_weights)
        return {
            'total_issues': len(self.history),
            'frequency': self.analyze_frequency(),
            'position_frequency': self.analyze_position_frequency(),
            'hot_numbers': self.get_hot_numbers(5),
            'cold_numbers': self.get_cold_numbers(5),
            'current_gaps': self.analyze_current_gaps(),
            'average_gaps': self.analyze_average_gaps(),
            'sum_analysis': self.analyze_sum(),
            'span_analysis': self.analyze_span(),
            'odd_even_analysis': self.analyze_odd_even(),
            'size_analysis': self.analyze_size(),
            'road_analysis': self.analyze_road(),
            'bayesian_scores': self.bayesian_score(),
            'sum_center': round(ss['sum_center'], 2),
            'span_center': round(ss['span_center'], 2),
            'window_weights': window_weights,
        }


# ==================== 全局实例 ====================

_pailie5_analyzer = None


def get_pailie5_analyzer() -> Pailie5Analyzer:
    global _pailie5_analyzer
    if _pailie5_analyzer is None:
        _pailie5_analyzer = Pailie5Analyzer()
    return _pailie5_analyzer
