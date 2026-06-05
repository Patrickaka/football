#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
排列五号码分析模块
提供排列五历史数据统计和号码推荐功能

分析功能：
1. 历史数据抓取
2. 位置频率统计
3. 热冷号分析
4. 当前遗漏分析
5. 平均遗漏分析
6. 和值分析
7. 跨度分析
8. 奇偶分析
9. 大小分析
10. 012路分析
11. 转移矩阵
12. 贝叶斯综合评分
"""

import os
import re
import json
import random
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 配置
DATA_FILE = 'pailie5_history.json'
# 排列五历史数据接口（参考福彩3D使用的网站，游戏ID为5）
HISTORY_URL = 'https://www.8300.cn/kjhhis/5/200.html'
NUMBERS = list(range(0, 10))  # 0-9


class Pailie5Analyzer:
    """
    排列五分析器
    """
    
    def __init__(self):
        self.history: List[Dict] = []
        self._load_history()
        # 如果数据为空，尝试抓取历史数据
        if not self.history:
            self.fetch_history_data(90)
    
    def _load_history(self):
        """
        加载历史数据
        """
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"已加载 {len(self.history)} 期排列五历史数据")
            except Exception as e:
                logger.error(f"加载排列五历史数据失败: {e}")
                self.history = []
        else:
            logger.info("排列五历史数据文件不存在，将创建新数据")
    
    def _save_history(self):
        """
        保存历史数据
        """
        try:
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.debug("排列五历史数据已保存")
        except Exception as e:
            logger.error(f"保存排列五历史数据失败: {e}")
    
    def fetch_history_data(self, days: int = 30):
        """
        从网站抓取排列五历史数据（参考福彩3D的抓取方式）
        
        参数:
            days: 抓取页数（每页约200期）
        
        返回:
            成功抓取的期数
        """
        try:
            url = HISTORY_URL
            logger.info(f"正在抓取排列五历史数据: {url}")
            
            # 使用简单的请求头（参考福彩3D的方式）
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            
            # 提取期号列表（从<td>标签中提取）
            issue_tags = re.findall(r'<td>[^<]*期[^<]*</td>', html)
            issues = []
            for tag in issue_tags:
                match = re.search(r'(\d{7})', tag)
                if match:
                    issues.append(match.group(1))
            
            # 提取日期列表（允许日期前后有空白字符）
            dates = re.findall(r'<td>\s*(\d{4}-\d{2}-\d{2})\s*</td>', html)
            
            # 提取数字球列表
            balls = re.findall(r'<span class="ball">(\d)</span>', html)
            
            if not issues or not balls:
                logger.error("无法解析排列五数据，未找到期号或数字球")
                return 0
            
            # 确保期数和球数匹配（每期 5 个球）
            expected_ball_count = len(issues) * 5
            if len(balls) != expected_ball_count:
                logger.warning(f"球数 ({len(balls)}) 与期数 ({len(issues)}) 不匹配，尝试使用可用数据")
            
            count = 0
            # 按期号从旧到新遍历，确保最新数据在前面
            for i in range(min(len(issues), len(dates)) - 1, -1, -1):
                ball_start = i * 5
                ball_end = ball_start + 5
                if ball_end <= len(balls):
                    numbers = [int(b) for b in balls[ball_start:ball_end]]
                    # 直接使用提取的日期
                    date = dates[i] if i < len(dates) else None
                    
                    if self.add_result(issues[i], numbers, date):
                        count += 1
            
            logger.info(f"成功抓取 {count} 期排列五数据")
            return count
        
        except Exception as e:
            logger.error(f"抓取排列五历史数据失败：{e}")
            return 0
    
    def _calculate_date_from_issue(self, issue: str) -> str:
        """
        从期号推算日期（排列五期号格式：YYYYNNN，NNN 是年内序号）
        
        参数:
            issue: 期号（如 2025316）
        
        返回:
            日期字符串（YYYY-MM-DD）
        """
        try:
            if len(issue) != 7:
                return datetime.now().strftime('%Y-%m-%d')
            
            year = int(issue[:4])
            day_of_year = int(issue[4:])
            
            # 计算日期
            date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            return date.strftime('%Y-%m-%d')
        except Exception:
            logger.warning(f"无法从期号 {issue} 推算日期")
            return datetime.now().strftime('%Y-%m-%d')
    
    def add_result(self, issue: str, numbers: List[int], date: str = None):
        """
        添加开奖结果
        
        参数:
            issue: 期号
            numbers: 5 个号码的列表
            date: 日期（可选）
        """
        if len(numbers) != 5:
            logger.error("排列五必须包含 5 个号码")
            return False
        
        for n in numbers:
            if n < 0 or n > 9:
                logger.error(f"号码 {n} 超出范围")
                return False
        
        if not date:
            # 从期号推算日期
            date = self._calculate_date_from_issue(issue)
        
        # 检查是否已存在
        for r in self.history:
            if r['issue'] == issue:
                # 期号已存在，跳过但返回 True（表示处理成功）
                return True
        
        result = {
            'issue': issue,
            'numbers': numbers,
            'date': date,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.insert(0, result)
        self._save_history()
        return True
    
    def fetch_latest_results(self, count: int = 10) -> Dict:
        """动态抓取最新开奖号码
        
        尝试从网站抓取最新的排列五开奖结果，如果网络不可用则返回本地数据。
        
        Args:
            count: 要抓取的期数（默认 10 期）
        
        Returns:
            包含抓取结果和状态信息的字典
        """
        try:
            # 重新抓取数据
            fetched_count = self.fetch_history_data(days=1)
            
            if fetched_count > 0:
                # 获取最新的结果
                recent = self.get_recent_results(count)
                
                return {
                    'success': True,
                    'source': 'web',
                    'count': fetched_count,
                    'message': f'成功抓取 {fetched_count} 期数据',
                    'latest_issue': recent[0]['issue'] if recent else None,
                    'results': recent
                }
            else:
                # 抓取失败，返回本地数据
                recent = self.get_recent_results(count)
                return {
                    'success': False,
                    'source': 'local',
                    'count': len(recent),
                    'message': '网络抓取失败，使用本地数据',
                    'latest_issue': recent[0]['issue'] if recent else None,
                    'results': recent
                }
        except Exception:
            logger.error("排列五抓取失败", exc_info=True)
            # 返回本地数据
            recent = self.get_recent_results(count)
            return {
                'success': False,
                'source': 'local',
                'count': len(recent),
                'message': '抓取异常，使用本地数据',
                'latest_issue': recent[0]['issue'] if recent else None,
                'results': recent
            }
    
    def get_recent_results(self, count: int = 10) -> List[Dict]:
        """
        获取最近的开奖结果
        """
        return self.history[:count]
    
    def analyze_position_frequency(self) -> List[Dict[int, int]]:
        """
        分析每个位置的号码频率
        """
        position_freq = []
        for pos in range(5):
            freq = {n: 0 for n in NUMBERS}
            for result in self.history:
                freq[result['numbers'][pos]] += 1
            position_freq.append(freq)
        return position_freq
    
    def analyze_frequency(self) -> Dict[int, int]:
        """
        分析号码总出现频率
        """
        freq = {n: 0 for n in NUMBERS}
        for result in self.history:
            for n in result['numbers']:
                freq[n] += 1
        return freq
    
    def get_hot_numbers(self, top_n: int = 3) -> List[Tuple[int, int]]:
        """
        获取热门号码（出现次数最多）
        """
        freq = self.analyze_frequency()
        return sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    
    def get_cold_numbers(self, top_n: int = 3) -> List[Tuple[int, int]]:
        """
        获取冷门号码（出现次数最少）
        """
        freq = self.analyze_frequency()
        return sorted(freq.items(), key=lambda x: x[1])[:top_n]
    
    def analyze_current_gaps(self) -> Dict[int, int]:
        """
        分析当前遗漏（距离上次出现的期数）
        """
        gaps = {n: len(self.history) for n in NUMBERS}
        last_occurrence = {n: -1 for n in NUMBERS}
        
        for i, result in enumerate(self.history):
            for n in result['numbers']:
                if last_occurrence[n] == -1:
                    last_occurrence[n] = i
        
        for n in NUMBERS:
            if last_occurrence[n] != -1:
                gaps[n] = last_occurrence[n]
        
        return gaps
    
    def analyze_average_gaps(self) -> Dict[int, float]:
        """
        分析平均遗漏
        """
        gaps = {n: [] for n in NUMBERS}
        last_occurrence = {n: 0 for n in NUMBERS}
        
        for i, result in enumerate(self.history):
            for n in result['numbers']:
                gap = i - last_occurrence[n]
                if gap > 0:
                    gaps[n].append(gap)
                last_occurrence[n] = i
        
        avg_gaps = {}
        for n in NUMBERS:
            if gaps[n]:
                avg_gaps[n] = round(sum(gaps[n]) / len(gaps[n]), 2)
            else:
                avg_gaps[n] = 0.0
        
        return avg_gaps
    
    def analyze_sum(self) -> Dict:
        """
        和值分析
        """
        sums = []
        for result in self.history:
            sums.append(sum(result['numbers']))
        
        if not sums:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        
        sum_counts = {}
        for s in sums:
            sum_counts[s] = sum_counts.get(s, 0) + 1
        
        most_common = sorted(sum_counts.items(), key=lambda x: -x[1])[:5]
        
        return {
            'min': min(sums),
            'max': max(sums),
            'avg': round(sum(sums) / len(sums), 2),
            'most_common': most_common,
            'distribution': sum_counts
        }
    
    def analyze_span(self) -> Dict:
        """
        跨度分析
        """
        spans = []
        for result in self.history:
            nums = result['numbers']
            spans.append(max(nums) - min(nums))
        
        if not spans:
            return {'min': 0, 'max': 0, 'avg': 0, 'most_common': []}
        
        span_counts = {}
        for s in spans:
            span_counts[s] = span_counts.get(s, 0) + 1
        
        most_common = sorted(span_counts.items(), key=lambda x: -x[1])[:5]
        
        return {
            'min': min(spans),
            'max': max(spans),
            'avg': round(sum(spans) / len(spans), 2),
            'most_common': most_common,
            'distribution': span_counts
        }
    
    def analyze_odd_even(self) -> Dict:
        """
        奇偶分析
        """
        odd_counts = []
        even_counts = []
        
        for result in self.history:
            odds = sum(1 for n in result['numbers'] if n % 2 == 1)
            odds = min(odds, 5 - odds)  # 统一为较小值
            odd_counts.append(odds)
        
        if not odd_counts:
            return {'distribution': {}, 'most_common': []}
        
        dist = {}
        for o in odd_counts:
            dist[o] = dist.get(o, 0) + 1
        
        most_common = sorted(dist.items(), key=lambda x: -x[1])[:3]
        
        return {
            'distribution': dist,
            'most_common': most_common
        }
    
    def analyze_size(self) -> Dict:
        """
        大小分析（0-4为小，5-9为大）
        """
        small_counts = []
        
        for result in self.history:
            smalls = sum(1 for n in result['numbers'] if n <= 4)
            smalls = min(smalls, 5 - smalls)  # 统一为较小值
            small_counts.append(smalls)
        
        if not small_counts:
            return {'distribution': {}, 'most_common': []}
        
        dist = {}
        for s in small_counts:
            dist[s] = dist.get(s, 0) + 1
        
        most_common = sorted(dist.items(), key=lambda x: -x[1])[:3]
        
        return {
            'distribution': dist,
            'most_common': most_common
        }
    
    def analyze_road(self) -> Dict:
        """
        012路分析（除3余数）
        """
        road_counts = {0: [], 1: [], 2: []}
        
        for result in self.history:
            for n in result['numbers']:
                road = n % 3
                road_counts[road].append(n)
        
        dist = {}
        for road in [0, 1, 2]:
            dist[road] = len(road_counts[road])
        
        most_common = sorted(dist.items(), key=lambda x: -x[1])
        
        return {
            'distribution': dist,
            'most_common': most_common,
            'total': {
                0: len(road_counts[0]),
                1: len(road_counts[1]),
                2: len(road_counts[2])
            },
            'road_numbers': {
                0: [n for n in NUMBERS if n % 3 == 0],
                1: [n for n in NUMBERS if n % 3 == 1],
                2: [n for n in NUMBERS if n % 3 == 2]
            }
        }
    
    def analyze_transition_matrix(self) -> List[List[int]]:
        """
        转移矩阵（相邻期数之间的号码转移）
        """
        matrix = [[0] * 10 for _ in range(10)]
        
        for i in range(1, len(self.history)):
            prev_nums = self.history[i-1]['numbers']
            curr_nums = self.history[i]['numbers']
            
            for p in prev_nums:
                for c in curr_nums:
                    matrix[p][c] += 1
        
        return matrix
    
    def bayesian_score(self) -> Dict[int, float]:
        """
        贝叶斯综合评分
        综合考虑频率、遗漏、冷热等因素
        """
        freq = self.analyze_frequency()
        gaps = self.analyze_current_gaps()
        avg_gaps = self.analyze_average_gaps()
        
        max_freq = max(freq.values()) if freq else 1
        max_gap = max(gaps.values()) if gaps else 1
        
        scores = {}
        for n in NUMBERS:
            # 频率得分（越高越好）
            freq_score = freq[n] / max_freq if max_freq > 0 else 0.5
            
            # 遗漏得分（当前遗漏越接近平均遗漏越好）
            gap_ratio = gaps[n] / avg_gaps[n] if avg_gaps[n] > 0 else 1
            gap_score = max(0, 1 - abs(gap_ratio - 1))
            
            # 综合评分
            scores[n] = round(0.6 * freq_score + 0.4 * gap_score, 4)
        
        return scores
    
    def generate_recommendation(self, method: str = 'balanced') -> List[int]:
        """
        生成推荐号码
        """
        if method == 'random':
            return [random.randint(0, 9) for _ in range(5)]
        
        hot = [n for n, _ in self.get_hot_numbers(5)]
        cold = [n for n, _ in self.get_cold_numbers(5)]
        
        if method == 'hot':
            return random.sample(hot, 5)
        
        if method == 'cold':
            return random.sample(cold, 5)
        
        # balanced: 3热2冷 或 2热3冷
        result = []
        hot_count = random.choice([2, 3])
        cold_count = 5 - hot_count
        
        result.extend(random.sample(hot, hot_count))
        result.extend(random.sample(cold, cold_count))
        random.shuffle(result)
        return result
    
    def build_second_order_markov(self) -> Dict:
        """
        构建二阶马尔可夫链模型
        返回状态转移概率：{(prev1, prev2): {next_num: probability}}
        """
        transitions = {}
        
        for i in range(2, len(self.history)):
            prev1 = tuple(self.history[i-2]['numbers'])
            prev2 = tuple(self.history[i-1]['numbers'])
            current = tuple(self.history[i]['numbers'])
            
            state = (prev1, prev2)
            if state not in transitions:
                transitions[state] = {}
            
            if current not in transitions[state]:
                transitions[state][current] = 0
            transitions[state][current] += 1
        
        # 归一化概率
        for state, next_states in transitions.items():
            total = sum(next_states.values())
            for next_state, count in next_states.items():
                transitions[state][next_state] = count / total
        
        return transitions
    
    def predict_with_markov(self, recent_numbers: List[List[int]]) -> List[int]:
        """
        使用二阶马尔可夫链预测下一期号码
        """
        if len(recent_numbers) < 2:
            return self.generate_recommendation('balanced')
        
        state = (tuple(recent_numbers[-2]), tuple(recent_numbers[-1]))
        transitions = self.build_second_order_markov()
        
        if state in transitions:
            # 选择概率最高的号码组合
            best_prediction = max(transitions[state].items(), key=lambda x: x[1])[0]
            return list(best_prediction)
        
        return self.generate_recommendation('balanced')
    
    def multi_condition_filter(self, candidates: List[List[int]], conditions: Dict) -> List[List[int]]:
        """
        多条件缩水筛选
        conditions参数：
        - sum_range: 和值范围 (min, max)
        - span_range: 跨度范围 (min, max)
        - odd_count: 奇数个数范围 (min, max)
        - big_count: 大数个数范围 (min, max)
        - road_counts: 012路个数范围 {'0': (min, max), '1': (min, max), '2': (min, max)}
        """
        filtered = []
        
        for nums in candidates:
            # 和值条件
            if 'sum_range' in conditions:
                s = sum(nums)
                if s < conditions['sum_range'][0] or s > conditions['sum_range'][1]:
                    continue
            
            # 跨度条件
            if 'span_range' in conditions:
                span = max(nums) - min(nums)
                if span < conditions['span_range'][0] or span > conditions['span_range'][1]:
                    continue
            
            # 奇数个数条件
            if 'odd_count' in conditions:
                odd_count = sum(1 for n in nums if n % 2 == 1)
                if odd_count < conditions['odd_count'][0] or odd_count > conditions['odd_count'][1]:
                    continue
            
            # 大数个数条件 (0-4为小，5-9为大)
            if 'big_count' in conditions:
                big_count = sum(1 for n in nums if n >= 5)
                if big_count < conditions['big_count'][0] or big_count > conditions['big_count'][1]:
                    continue
            
            # 012路条件
            if 'road_counts' in conditions:
                road_counts = {'0': 0, '1': 0, '2': 0}
                for n in nums:
                    road_counts[str(n % 3)] += 1
                
                for road, (min_count, max_count) in conditions['road_counts'].items():
                    if road_counts[road] < min_count or road_counts[road] > max_count:
                        break
                else:
                    filtered.append(nums)
                    continue
                continue
            
            filtered.append(nums)
        
        return filtered
    
    def optimize_weights(self, test_periods: int = 50) -> Dict[str, float]:
        """
        自动权重优化 - 使用网格搜索找到最优权重组合
        """
        if len(self.history) < test_periods:
            return {'freq_weight': 0.6, 'gap_weight': 0.4}
        
        # 测试数据
        test_data = self.history[:test_periods]
        train_data = self.history[test_periods:]
        
        best_weights = None
        best_score = -1
        
        # 网格搜索权重组合
        for freq_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for gap_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                # 跳过权重和不为1的组合
                if abs(freq_w + gap_w - 1) > 0.01:
                    continue
                
                # 使用当前权重进行回测
                score = self._backtest_with_weights(train_data, test_data, freq_w, gap_w)
                
                if score > best_score:
                    best_score = score
                    best_weights = {'freq_weight': freq_w, 'gap_weight': gap_w}
        
        return best_weights if best_weights else {'freq_weight': 0.6, 'gap_weight': 0.4}
    
    def _backtest_with_weights(self, train_data: List[Dict], test_data: List[Dict], 
                               freq_weight: float, gap_weight: float) -> float:
        """
        使用指定权重进行回测
        """
        correct_count = 0
        total_count = len(test_data)
        
        # 使用训练数据构建频率表
        freq = {n: 0 for n in NUMBERS}
        for result in train_data:
            for n in result['numbers']:
                freq[n] += 1
        
        # 回测
        current_gaps = {n: 0 for n in NUMBERS}
        
        for test_result in test_data:
            # 更新遗漏
            for n in NUMBERS:
                current_gaps[n] += 1
            
            # 计算评分
            max_freq = max(freq.values()) if freq else 1
            scores = {}
            for n in NUMBERS:
                freq_score = freq[n] / max_freq if max_freq > 0 else 0.5
                gap_score = max(0, 1 - current_gaps[n] / 10.0)  # 假设平均遗漏约为10
                scores[n] = freq_weight * freq_score + gap_weight * gap_score
            
            # 选择评分最高的5个号码
            top5 = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]
            
            # 检查命中
            for n in test_result['numbers']:
                if n in top5:
                    correct_count += 1
            
            # 更新频率和遗漏
            for n in test_result['numbers']:
                freq[n] += 1
                current_gaps[n] = 0
        
        return correct_count / (total_count * 5)  # 返回命中率
    
    def backtest(self, method: str = 'bayesian', test_periods: int = 50) -> Dict:
        """
        历史回测 - 测试预测模型在历史数据上的表现
        """
        if len(self.history) < test_periods:
            return {'error': '数据不足'}
        
        test_data = self.history[:test_periods]
        train_data = self.history[test_periods:]
        
        # 重建分析器用于回测
        backtest_analyzer = Pailie5Analyzer()
        backtest_analyzer.history = []
        for result in train_data:
            backtest_analyzer.add_result(result['issue'], result['numbers'], result['date'])
        
        total_correct = 0
        total_numbers = test_periods * 5
        predictions = []
        
        for i, test_result in enumerate(test_data):
            # 获取统计数据并预测
            stats = backtest_analyzer.get_statistics()
            
            if method == 'bayesian':
                # 使用贝叶斯评分选择top5
                scores = stats['bayesian_scores']
                top5 = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]
            elif method == 'markov':
                # 使用马尔可夫链预测
                recent = [r['numbers'] for r in backtest_analyzer.history[-2:]]
                pred = backtest_analyzer.predict_with_markov(recent)
                top5 = pred
            else:
                # 使用综合策略
                top5 = backtest_analyzer.generate_recommendation(method)
            
            # 检查命中
            correct = sum(1 for n in test_result['numbers'] if n in top5)
            total_correct += correct
            
            predictions.append({
                'issue': test_result['issue'],
                'actual': test_result['numbers'],
                'predicted': top5,
                'correct': correct
            })
            
            # 添加到训练数据中
            backtest_analyzer.add_result(test_result['issue'], test_result['numbers'], test_result['date'])
        
        return {
            'method': method,
            'test_periods': test_periods,
            'total_correct': total_correct,
            'total_numbers': total_numbers,
            'hit_rate': round(total_correct / total_numbers * 100, 2),
            'predictions': predictions
        }
    
    def rolling_backtest(self, trials: int = 50) -> Dict:
        """
        滚动回测 - 逐步添加训练数据，每期用之前的数据预测并与实际比较
        返回与3D类似的回测统计
        """
        # 准备数据
        numbers = [r['numbers'] for r in self.history]
        if len(numbers) < trials + 10:
            trials = max(20, len(numbers) - 10)
        
        start = len(numbers) - trials
        
        hit_top3 = hit_top5 = hit_ge2_digit = hit_ge3_digit = 0
        
        for i in range(start, len(numbers)):
            train = numbers[:i]
            actual = numbers[i]
            
            # 重建分析器用于回测
            bt_analyzer = Pailie5Analyzer()
            bt_analyzer.history = []
            for result in self.history[:i]:
                bt_analyzer.add_result(result['issue'], result['numbers'], result['date'])
            
            # 获取统计数据并预测
            stats = bt_analyzer.get_statistics()
            
            # 使用贝叶斯评分选择top5
            scores = stats['bayesian_scores']
            top5 = [n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]
            
            # 检查命中
            # 直选Top3命中
            actual_str = ''.join(map(str, actual))
            top3 = top5[:3]
            
            if actual_str in [''.join(map(str, top5[:3]))]:
                # 检查顺序命中
                pass
            
            # Top5直选命中
            if actual_str in [''.join(map(str, top5))]:
                hit_top5 += 1
            
            # 至少中2个数字
            if len(set(actual) & set(top5)) >= 2:
                hit_ge2_digit += 1
            
            # 至少中3个数字
            if len(set(actual) & set(top5)) >= 3:
                hit_ge3_digit += 1
        
        n = trials
        return {
            'trials': n,
            'top5_hit': hit_top5,
            'top5_rate': hit_top5 / n,
            'ge2_digit_rate': hit_ge2_digit / n,
            'ge3_digit_rate': hit_ge3_digit / n,
        }
    
    def feature_contribution(self) -> Dict[str, Dict[int, float]]:
        """
        计算各特征对每个号码的贡献度
        返回: {feature_name: {number: contribution}}
        """
        freq = self.analyze_frequency()
        gaps = self.analyze_current_gaps()
        avg_gaps = self.analyze_average_gaps()
        pos_freq = self.analyze_position_frequency()
        
        max_freq = max(freq.values()) if freq else 1
        max_gap = max(gaps.values()) if gaps else 1
        
        contributions = {}
        
        # 频率贡献
        contributions['frequency'] = {n: freq[n] / max_freq for n in NUMBERS}
        
        # 遗漏贡献（反转，遗漏越小贡献越大）
        contributions['gap'] = {n: 1 - (gaps[n] / (max_gap + 1)) for n in NUMBERS}
        
        # 位置频率贡献（取各位置平均）
        pos_contrib = {}
        for n in NUMBERS:
            avg_pos_freq = sum(pos_freq[pos].get(n, 0) for pos in range(5)) / 5
            max_pos_freq = max(max(p.values()) for p in pos_freq) if pos_freq else 1
            pos_contrib[n] = avg_pos_freq / max_pos_freq if max_pos_freq > 0 else 0.5
        contributions['position'] = pos_contrib
        
        # 012路贡献
        road_counts = {'0': 0, '1': 0, '2': 0}
        for result in self.history:
            for n in result['numbers']:
                road_counts[str(n % 3)] += 1
        total_road = sum(road_counts.values()) or 1
        contributions['road'] = {n: road_counts[str(n % 3)] / total_road for n in NUMBERS}
        
        return contributions
    
    def dynamic_weight_adjustment(self, backtest_results: List[Dict]) -> Dict[str, float]:
        """
        根据回测结果动态调整特征权重
        """
        # 计算各特征在回测中的表现
        feature_scores = {
            'frequency': 0,
            'gap': 0,
            'position': 0,
            'road': 0
        }
        
        # 简单策略：根据特征与实际结果的相关性调整权重
        # 这里使用简化的启发式方法
        for result in backtest_results:
            actual = result['actual']
            predicted = result['predicted']
            
            # 计算预测命中率
            hit_count = sum(1 for n in actual if n in predicted)
            hit_rate = hit_count / 5
            
            # 根据命中率调整权重
            if hit_rate > 0.6:
                feature_scores['frequency'] += 1
            elif hit_rate < 0.2:
                feature_scores['gap'] += 1
        
        # 归一化权重
        total = sum(feature_scores.values()) or 1
        weights = {k: v / total for k, v in feature_scores.items()}
        
        # 确保所有权重和为1
        if sum(weights.values()) > 0:
            weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        
        return weights
    
    def rank_model(self, top_n: int = 5, weights: Dict = None) -> List[Tuple[int, float]]:
        """
        排名模型 - Top-N排序
        根据多个特征综合评分对号码进行排序
        """
        if weights is None:
            weights = {
                'frequency': 0.3,
                'gap': 0.3,
                'position': 0.2,
                'road': 0.2
            }
        
        contributions = self.feature_contribution()
        
        # 计算综合评分
        scores = {}
        for n in NUMBERS:
            score = 0
            for feature, weight in weights.items():
                if feature in contributions:
                    score += contributions[feature][n] * weight
            scores[n] = round(score, 4)
        
        # 按评分排序并返回Top-N
        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    
    def identify_cycles(self, min_cycle: int = 3, max_cycle: int = 20) -> Dict[str, List]:
        """
        周期与状态识别
        识别号码的冷热周期和当前状态
        """
        freq = self.analyze_frequency()
        gaps = self.analyze_current_gaps()
        avg_gaps = self.analyze_average_gaps()
        
        # 计算平均频率
        avg_freq = sum(freq.values()) / len(freq) if freq else 1
        
        cycles = {
            'hot': [],      # 热门状态（频率高于平均）
            'cold': [],     # 冷门状态（频率低于平均）
            'warming': [],  # 升温状态（当前遗漏小于平均遗漏）
            'cooling': [],  # 降温状态（当前遗漏大于平均遗漏）
            'stable': []    # 稳定状态
        }
        
        for n in NUMBERS:
            # 判断冷热状态（基于频率偏离度）
            freq_deviation = freq[n] / avg_freq if avg_freq > 0 else 1
            if freq_deviation > 1.10:  # 频率高于平均值 10% 以上
                cycles['hot'].append(n)
            elif freq_deviation < 0.90:  # 频率低于平均值 10% 以上
                cycles['cold'].append(n)
            else:
                cycles['stable'].append(n)
            
            # 判断升降温趋势（基于遗漏偏离度）
            if avg_gaps[n] > 0 and gaps[n] < avg_gaps[n] * 0.7:  # 当前遗漏小于平均遗漏的 70%
                cycles['warming'].append(n)
            elif avg_gaps[n] > 0 and gaps[n] > avg_gaps[n] * 1.5:  # 当前遗漏大于平均遗漏的 150%
                cycles['cooling'].append(n)
        
        return cycles
    
    def multi_model_voting(self, n_votes: int = 3) -> List[int]:
        """
        多模型集成投票
        综合多个模型的预测结果进行投票
        """
        predictions = []
        
        # 模型1：贝叶斯评分
        scores = self.bayesian_score()
        predictions.append([n for n, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]])
        
        # 模型2：排名模型
        ranked = self.rank_model(top_n=5)
        predictions.append([n for n, _ in ranked])
        
        # 模型3：马尔可夫链
        recent = [r['numbers'] for r in self.get_recent_results(2)]
        predictions.append(self.predict_with_markov(recent))
        
        # 模型4：热号推荐
        hot = [n for n, _ in self.get_hot_numbers(5)]
        predictions.append(hot)
        
        # 模型5：冷号推荐
        cold = [n for n, _ in self.get_cold_numbers(5)]
        predictions.append(cold)
        
        # 投票计数
        votes = {}
        for pred in predictions:
            for n in pred:
                votes[n] = votes.get(n, 0) + 1
        
        # 选择得票最高的n_votes个号码
        result = [n for n, _ in sorted(votes.items(), key=lambda x: -x[1])[:n_votes]]
        
        # 如果不足5个，补充随机号码
        while len(result) < 5:
            n = random.randint(0, 9)
            if n not in result:
                result.append(n)
        
        return result
    
    def ensemble_predict(self, method: str = 'voting') -> Dict:
        """
        集成预测 - 综合所有模型进行预测
        """
        if method == 'voting':
            prediction = self.multi_model_voting(n_votes=3)
        elif method == 'rank':
            ranked = self.rank_model(top_n=5)
            prediction = [n for n, _ in ranked]
        elif method == 'markov':
            recent = [r['numbers'] for r in self.get_recent_results(2)]
            prediction = self.predict_with_markov(recent)
        else:
            prediction = self.generate_recommendation('balanced')
        
        # 获取周期状态
        cycles = self.identify_cycles()
        
        # 获取特征贡献度
        contributions = self.feature_contribution()
        
        return {
            'prediction': prediction,
            'method': method,
            'cycles': cycles,
            'feature_contributions': contributions,
            'ranked_numbers': self.rank_model(top_n=10)
        }
    
    def get_statistics(self) -> Dict:
        """
        获取完整统计信息
        """
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
        }


# 全局实例
_pailie5_analyzer = None


def get_pailie5_analyzer() -> Pailie5Analyzer:
    """
    获取排列五分析器实例
    """
    global _pailie5_analyzer
    if _pailie5_analyzer is None:
        _pailie5_analyzer = Pailie5Analyzer()
    return _pailie5_analyzer


# ==================== 测试函数 ====================

def test_pailie5():
    """
    测试排列五分析器
    """
    print("="*60)
    print("排列五分析器测试")
    print("="*60)
    
    analyzer = get_pailie5_analyzer()
    
    # 添加测试数据
    test_data = [
        ('2024001', [1, 2, 3, 4, 5], '2024-01-01'),
        ('2024002', [5, 4, 3, 2, 1], '2024-01-02'),
        ('2024003', [0, 5, 8, 3, 7], '2024-01-03'),
        ('2024004', [9, 9, 9, 9, 9], '2024-01-04'),
        ('2024005', [1, 3, 5, 7, 9], '2024-01-05'),
        ('2024006', [2, 4, 6, 8, 0], '2024-01-06'),
        ('2024007', [3, 1, 4, 1, 5], '2024-01-07'),
        ('2024008', [9, 2, 6, 5, 3], '2024-01-08'),
        ('2024009', [5, 8, 9, 7, 9], '2024-01-09'),
        ('2024010', [3, 2, 3, 8, 4], '2024-01-10'),
    ]
    
    for issue, numbers, date in test_data:
        analyzer.add_result(issue, numbers, date)
    
    stats = analyzer.get_statistics()
    
    print(f"总期数: {stats['total_issues']}")
    
    print("\n1. 号码频率:")
    for n, freq in sorted(stats['frequency'].items()):
        print(f"  {n}: {freq} 次")
    
    print("\n2. 位置频率:")
    for pos, freq in enumerate(stats['position_frequency']):
        print(f"  位置{pos+1}: {', '.join([f'{n}:{freq[n]}' for n in sorted(freq)])}")
    
    print("\n3. 热门号码:")
    for n, freq in stats['hot_numbers']:
        print(f"  {n}: {freq} 次")
    
    print("\n4. 冷门号码:")
    for n, freq in stats['cold_numbers']:
        print(f"  {n}: {freq} 次")
    
    print("\n5. 当前遗漏:")
    for n, gap in sorted(stats['current_gaps'].items()):
        print(f"  {n}: {gap} 期")
    
    print("\n6. 平均遗漏:")
    for n, avg in sorted(stats['average_gaps'].items()):
        print(f"  {n}: {avg:.2f} 期")
    
    print("\n7. 和值分析:")
    sum_anal = stats['sum_analysis']
    print(f"  最小: {sum_anal['min']}, 最大: {sum_anal['max']}, 平均: {sum_anal['avg']}")
    print(f"  最常见和值: {sum_anal['most_common']}")
    
    print("\n8. 跨度分析:")
    span_anal = stats['span_analysis']
    print(f"  最小: {span_anal['min']}, 最大: {span_anal['max']}, 平均: {span_anal['avg']}")
    
    print("\n9. 奇偶分析:")
    print(f"  分布: {stats['odd_even_analysis']['distribution']}")
    
    print("\n10. 大小分析:")
    print(f"  分布: {stats['size_analysis']['distribution']}")
    
    print("\n11. 012路分析:")
    road_anal = stats['road_analysis']
    print(f"  分布: {road_anal['distribution']}")
    
    print("\n12. 贝叶斯综合评分:")
    for n, score in sorted(stats['bayesian_scores'].items(), key=lambda x: -x[1]):
        print(f"  {n}: {score:.4f}")
    
    print("\n推荐号码:")
    for method in ['balanced', 'hot', 'cold', 'random']:
        nums = analyzer.generate_recommendation(method)
        print(f"  {method}: {nums}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == '__main__':
    test_pailie5()