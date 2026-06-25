"""
快乐8预测模块
=============

快乐8玩法：从1-80中开出20个号码，玩家可选1-10个号码进行投注。
本模块提供选3/选4/选5/选6/选7预测，以及选5复式7码预测。

预测策略：
1. 综合特征评分（频率、遗漏、区位、路数等）
2. 均值回归修正（热号降分、冷号加分）
3. 多模型投票（贝叶斯+排名+马尔可夫）
4. 选5复式：从Top7号码中生成C(7,5)=21组组合

版本: kl8-v1.0-ensemble
"""

import math
import json
import time
import random
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')

KL8_PREDICTOR_VERSION = "kl8-v1.0-ensemble"

# ─── 快乐8常量 ───
KL8_NUM_RANGE = 80       # 号码范围 1-80
KL8_DRAW_COUNT = 20      # 每期开出20个号码
KL8_DEFAULT_HISTORY = 250  # 默认使用最近250期

# ─── 选型配置：选3~选7各选多少号码 ───
SELECT_CONFIG = {
    3: {'pick': 3, 'top_n': 10,  'desc': '选3'},
    4: {'pick': 4, 'top_n': 12,  'desc': '选4'},
    5: {'pick': 5, 'top_n': 15,  'desc': '选5'},
    6: {'pick': 6, 'top_n': 15,  'desc': '选6'},
    7: {'pick': 7, 'top_n': 18,  'desc': '选7'},
}

# ─── 特征权重（基于大乐透v3经验 + 快乐880选20特性调整）───
# 快乐8开出20/80=25%，远高于大乐透5/35=14.3%，所以频率/遗漏信号可能更强
FEATURE_WEIGHTS = {
    'frequency': 0.08,    # 频率偏离度(冷号加分,热号降分)
    'gap': 0.12,          # 遗漏偏离度(遗漏越大越可能开出)
    'position': 0.06,     # 位置特征(前/中/后区分布)
    'road': 0.10,         # 路数特征(012路分布)
    'sum': 0.08,          # 和值特征
    'zone': 0.15,         # 区位特征(8个10码区分布)
    'repeat': 0.20,       # 重号特征(与上期重复)
    'adjacent': 0.12,     # 邻号特征(与上期号码±1)
    'odd_even': 0.04,     # 奇偶特征
    'big_small': 0.04,    # 大小特征(1-40小,41-80大)
}


class KL8Analyzer:
    """快乐8预测分析器"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('kl8_history.json')
        self.using_simulated_data = False
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()

    # ─── 数据加载 ───

    def _load_history(self) -> List[Dict]:
        """加载历史开奖数据"""
        try:
            # 尝试从doc_store加载
            records = doc_store._fallback_load_all('kl8_history')
            if records:
                data = []
                for r in records:
                    nums = r.get('numbers') or r.get('draw_numbers')
                    if isinstance(nums, str):
                        nums = json.loads(nums)
                    data.append({
                        'issue': r.get('issue', ''),
                        'numbers': nums,
                        'date': r.get('date') or r.get('draw_date', ''),
                    })
                # 按期号降序排列(最新在前)
                data.sort(key=lambda x: x.get('issue', ''), reverse=True)
                self.using_simulated_data = False
                log.info(f'快乐8: 加载了{len(data)}期历史数据')
                return data
        except Exception as e:
            log.warning(f'快乐8: doc_store加载失败: {e}')

        # 尝试从JSON文件加载
        try:
            path = Path(self.history_file)
            if path.exists():
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    data = raw.get('results', raw.get('data', []))
                else:
                    data = raw
                data.sort(key=lambda x: x.get('issue', ''), reverse=True)
                self.using_simulated_data = False
                log.info(f'快乐8: 从文件加载了{len(data)}期数据')
                return data
        except Exception as e:
            log.warning(f'快乐8: 文件加载失败: {e}')

        # 生成模拟数据
        log.info('快乐8: 生成模拟数据')
        return self._generate_simulated_data()

    def _generate_simulated_data(self) -> List[Dict]:
        """生成模拟数据用于测试"""
        self.using_simulated_data = True
        data = []
        for i in range(100):
            # 从1-80中随机选20个号码
            nums = sorted(random.sample(range(1, 81), 20))
            data.append({
                'issue': f'2026{100 - i:03d}',
                'numbers': nums,
                'date': f'2026-{(100 - i) // 30 + 1:02d}-{(100 - i) % 30 + 1:02d}',
            })
        return data

    def _save_history(self):
        """保存历史数据到文件"""
        try:
            path = Path(self.history_file)
            path.write_text(
                json.dumps({'results': self.history_data}, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            log.info(f'快乐8: 保存了{len(self.history_data)}期数据')
        except Exception as e:
            log.error(f'快乐8: 保存数据失败: {e}')

    # ─── 统计计算 ───

    def update_statistics(self):
        """更新所有统计量"""
        if not self.history_data:
            return

        n = len(self.history_data)
        recent = min(n, KL8_DEFAULT_HISTORY)
        recent_data = self.history_data[:recent]

        # 全号码频率统计
        freq = Counter()
        for record in recent_data:
            for num in record['numbers']:
                freq[num] += 1

        # 全号码遗漏统计
        gap = {}
        for num in range(1, 81):
            gap[num] = 0
            for record in recent_data:
                if num in record['numbers']:
                    break
                gap[num] += 1

        # 上期号码
        last_numbers = set(recent_data[0]['numbers']) if recent_data else set()
        last2_numbers = set(recent_data[1]['numbers']) if len(recent_data) > 1 else set()

        self.statistics = {
            'frequency': freq,
            'gap': gap,
            'total_periods': recent,
            'expected_freq': recent * KL8_DRAW_COUNT / KL8_NUM_RANGE,  # 期望频率=期数*20/80
            'expected_gap': KL8_NUM_RANGE / KL8_DRAW_COUNT,  # 期望遗漏=80/20=4
            'last_numbers': last_numbers,
            'last2_numbers': last2_numbers,
            'freq_by_zone': self._zone_frequency(recent_data),
            'freq_by_road': self._road_frequency(recent_data),
            'freq_by_odd_even': self._odd_even_frequency(recent_data),
            'freq_by_big_small': self._big_small_frequency(recent_data),
        }

    def _zone_frequency(self, data: List[Dict]) -> Dict:
        """8个10码区的频率分布"""
        zone_freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                zone = (num - 1) // 10 + 1  # 1-8区
                zone_freq[zone] += 1
        return dict(zone_freq)

    def _road_frequency(self, data: List[Dict]) -> Dict:
        """012路频率分布"""
        road_freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                road = num % 3  # 0路、1路、2路
                road_freq[road] += 1
        return dict(road_freq)

    def _odd_even_frequency(self, data: List[Dict]) -> Dict:
        """奇偶频率分布"""
        freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                freq['odd' if num % 2 == 1 else 'even'] += 1
        return dict(freq)

    def _big_small_frequency(self, data: List[Dict]) -> Dict:
        """大小频率分布(1-40小,41-80大)"""
        freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                freq['big' if num > 40 else 'small'] += 1
        return dict(freq)

    # ─── 特征评分 ───

    def _calculate_feature_score(self, num: int) -> Dict[str, float]:
        """计算号码num的各特征得分"""
        scores = {}
        stats = self.statistics
        freq = stats['frequency']
        gap = stats['gap']
        expected_freq = stats['expected_freq']
        expected_gap = stats['expected_gap']
        last_nums = stats['last_numbers']
        last2_nums = stats['last2_numbers']
        total = stats['total_periods']

        # 1. 频率偏离度 — 均值回归(冷号加分,热号降分)
        actual_freq = freq.get(num, 0)
        deviation_ratio = actual_freq / max(expected_freq, 0.01)
        if deviation_ratio <= 1.0:
            # 冷号: 偏离越小(越冷) → 越加分(均值回归)
            scores['frequency'] = 0.55 + 0.15 * (1.0 - deviation_ratio)
        else:
            # 热号: 偏离越大 → 指数衰减降分
            scores['frequency'] = max(0.15, 0.55 * math.exp(-1.8 * (deviation_ratio - 1.0)))

        # 2. 遗漏偏离度 — 非对称平滑函数
        actual_gap = gap.get(num, 0)
        gap_ratio = actual_gap / max(expected_gap, 0.01)
        if gap_ratio <= 1.0:
            scores['gap'] = 0.25 + 0.60 * (gap_ratio ** 0.7)
        else:
            scores['gap'] = 0.85 - 0.45 * (1.0 - math.exp(-(gap_ratio - 1.0) * 0.8))

        # 3. 位置特征 — 8个10码区的均衡性
        zone = (num - 1) // 10 + 1
        zone_freq = stats['freq_by_zone']
        expected_zone = total * KL8_DRAW_COUNT / KL8_NUM_RANGE * 10 / KL8_NUM_RANGE
        zone_ratio = zone_freq.get(zone, 0) / max(expected_zone, 0.01)
        scores['position'] = 0.4 + 0.6 * max(0, 1.0 - abs(zone_ratio - 1.0))

        # 4. 路数特征 — 012路均衡
        road = num % 3
        road_freq = stats['freq_by_road']
        expected_road = total * KL8_DRAW_COUNT / 3
        road_ratio = road_freq.get(road, 0) / max(expected_road, 0.01)
        scores['road'] = 0.4 + 0.6 * max(0, 1.0 - abs(road_ratio - 1.0) * 0.5)

        # 5. 和值特征 — 与上期和值的偏离
        last_sum = sum(last_nums) if last_nums else 400
        # 理论期望和值 = 20 * (1+80)/2 = 810
        expected_sum = KL8_DRAW_COUNT * (1 + KL8_NUM_RANGE) / 2  # = 810
        # 号码num对和值的贡献偏离
        num_deviation = abs(num - (1 + KL8_NUM_RANGE) / 2) / ((KL8_NUM_RANGE - 1) / 2)
        # 如果上期和值偏高,本期倾向偏低 → 小号码加分;反之大号码加分
        sum_bias = (last_sum - expected_sum) / expected_sum
        if sum_bias > 0:
            scores['sum'] = 0.5 + 0.3 * max(0, 1.0 - num_deviation)
        else:
            scores['sum'] = 0.5 + 0.3 * min(1.0, num_deviation)

        # 6. 区位特征 — 号码所在区的近期开出率
        zone_nums = [z for z in range(((zone-1)*10)+1, zone*10+1)]
        zone_hit_rate = len([n for n in zone_nums if n in last_nums]) / len(zone_nums)
        # 期望每区开出2.5个号码(20/8=2.5)
        expected_hit = KL8_DRAW_COUNT / 8 / 10
        zone_deviation = zone_hit_rate / max(expected_hit, 0.01)
        # 上期开出多的区,本期可能减少(均值回归)
        scores['zone'] = max(0.2, 0.6 - 0.3 * max(0, zone_deviation - 1.0))

        # 7. 重号特征 — 与上期号码重复
        is_repeat = num in last_nums
        # 快乐8开出20/80=25%, 重号期望约5个(20*25%≈5)
        repeat_rate = len([n for n in last_nums if n in set(range(1,81))]) / KL8_NUM_RANGE
        if is_repeat:
            scores['repeat'] = 0.6  # 重号加分(快乐8重号率高)
        else:
            scores['repeat'] = 0.35

        # 8. 邻号特征 — 与上期号码±1
        last_neighbors = set()
        for n in last_nums:
            if n - 1 >= 1: last_neighbors.add(n - 1)
            if n + 1 <= 80: last_neighbors.add(n + 1)
        is_adjacent = num in last_neighbors
        if is_adjacent:
            scores['adjacent'] = 0.6
        else:
            scores['adjacent'] = 0.30

        # 9. 奇偶特征 — 上期奇偶分布的回归
        oe = stats['freq_by_odd_even']
        odd_ratio = oe.get('odd', 0) / max(oe.get('odd', 0) + oe.get('even', 0), 1)
        is_odd = num % 2 == 1
        # 期望奇偶比约1:1(80个号码40奇40偶,开出20个约10奇10偶)
        if is_odd:
            scores['odd_even'] = 0.5 + 0.3 * max(0, 1.0 - odd_ratio)
        else:
            scores['odd_even'] = 0.5 + 0.3 * max(0, odd_ratio - 0.5)

        # 10. 大小特征 — 上期大小分布的回归
        bs = stats['freq_by_big_small']
        big_ratio = bs.get('big', 0) / max(bs.get('big', 0) + bs.get('small', 0), 1)
        is_big = num > 40
        if is_big:
            scores['big_small'] = 0.5 + 0.3 * max(0, 1.0 - big_ratio)
        else:
            scores['big_small'] = 0.5 + 0.3 * max(0, big_ratio - 0.5)

        return scores

    # ─── 排名模型 ───

    def get_ensemble_ranking(self, top_n: int = 20) -> List[Dict]:
        """综合特征评分排名"""
        ranking = []
        for num in range(1, 81):
            scores = self._calculate_feature_score(num)
            total_score = sum(scores.get(k, 0) * FEATURE_WEIGHTS.get(k, 0) for k in scores)
            ranking.append({
                'num': num,
                'score': total_score,
                'scores': scores,
            })
        ranking.sort(key=lambda x: -x['score'])
        return ranking[:top_n]

    # ─── 贝叶斯模型 ───

    def _model_bayesian(self, top_n: int = 20) -> List[int]:
        """贝叶斯概率模型(含均值回归修正)"""
        stats = self.statistics
        freq = stats['frequency']
        expected_freq = stats['expected_freq']
        total = stats['total_periods']

        scores = {}
        for num in range(1, 81):
            actual_rate = freq.get(num, 0) / max(total, 1)
            expected_rate = KL8_DRAW_COUNT / KL8_NUM_RANGE  # = 0.25
            base_prob = (freq.get(num, 0) + 1) / (total + 2)

            # 均值回归修正
            deviation_ratio = actual_rate / max(expected_rate, 0.01)
            reversion_factor = 1.0 / (1.0 + 0.6 * max(0, deviation_ratio - 1.0))
            if deviation_ratio < 1.0:
                reversion_factor = min(1.5, 1.0 + 0.5 * (1.0 - deviation_ratio))

            scores[num] = base_prob * reversion_factor

        return sorted(scores.keys(), key=lambda n: -scores[n])[:top_n]

    # ─── 马尔可夫模型 ───

    def _model_markov(self, top_n: int = 20) -> List[int]:
        """一阶马尔可夫转移模型"""
        if len(self.history_data) < 3:
            return list(range(1, 21))

        # 构建转移概率矩阵(简化: 上期开出的号码→本期也开出的概率)
        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.history_data) - 1):
            current = set(self.history_data[i]['numbers'])
            prev = set(self.history_data[i + 1]['numbers'])  # 注意: 最新在前
            for num in prev:
                if num in current:
                    transition_counts[num]['repeat'] += 1
                else:
                    transition_counts[num]['skip'] += 1

        # 计算每个号码的马尔可夫得分
        last_nums = set(self.history_data[0]['numbers'])
        scores = {}
        for num in range(1, 81):
            base_score = 0.25  # 基线概率
            if num in last_nums:
                repeat_rate = transition_counts[num]['repeat'] / max(
                    transition_counts[num]['repeat'] + transition_counts[num]['skip'], 1)
                base_score = max(0.15, repeat_rate)
            else:
                # 上期未开出 → 本期开出的概率也约25%
                base_score = 0.25
            scores[num] = base_score

        return sorted(scores.keys(), key=lambda n: -scores[n])[:top_n]

    # ─── 排名模型(独立) ───

    def _model_rank(self, top_n: int = 20) -> List[int]:
        """纯排名模型"""
        ranking = self.get_ensemble_ranking(top_n=top_n)
        return [r['num'] for r in ranking]

    # ─── 多模型投票 ───

    def multi_model_voting(self, pick_n: int = 5, top_n: int = 20, skip_ml: bool = False) -> Dict:
        """多模型集成投票

        Args:
            pick_n: 选几号(3/4/5/6/7)
            top_n: 候选池大小
            skip_ml: 是否跳过ML模型
        """
        votes = defaultdict(float)

        # 模型列表与权重(v3.3经验: 排名绝对主导)
        models = [
            self._model_bayesian(top_n=top_n),
            self._model_rank(top_n=top_n),
            self._model_markov(top_n=top_n),
        ]
        # v3.3策略: 排名绝对主导
        model_weights = [0.1, 4.0, 0.2]  # [bayesian, rank, markov]

        for model_idx, model_result in enumerate(models):
            mw = model_weights[model_idx]
            for rank, num in enumerate(model_result):
                weight = (1.0 - (rank / max(len(model_result), 1))) * mw
                votes[num] += weight

        # 按投票得分排序
        candidates = sorted(votes.items(), key=lambda x: -x[1])

        # 选出pick_n个号码
        selected = [num for num, _ in candidates[:pick_n]]

        # 候选池(用于选5复式等)
        candidate_pool = candidates[:max(top_n, 7)]

        return {
            'selected': selected,
            'candidates': candidate_pool,
            'votes': dict(votes),
            'version': KL8_PREDICTOR_VERSION,
        }

    # ─── 选5复式7码 ───

    def get_fu_shi_7(self) -> Dict:
        """选5复式7码 — 从Top7号码中生成C(7,5)=21组组合"""
        ranking = self.get_ensemble_ranking(top_n=7)
        top7 = [r['num'] for r in ranking]

        # 生成所有C(7,5)=21组组合
        combinations_list = list(combinations(top7, 5))
        combo_list = [list(c) for c in combinations_list]

        return {
            'top7_numbers': top7,
            'top7_scores': [r['score'] for r in ranking],
            'top7_details': ranking,
            'total_combinations': len(combo_list),
            'combinations': combo_list,
            'version': KL8_PREDICTOR_VERSION,
        }

    # ─── 综合预测 ───

    def predict_all(self) -> Dict:
        """生成所有选型的预测结果"""
        results = {}

        # 选3~选7的推荐
        for select_type in [3, 4, 5, 6, 7]:
            config = SELECT_CONFIG[select_type]
            vote_result = self.multi_model_voting(
                pick_n=config['pick'],
                top_n=config['top_n'],
                skip_ml=True
            )
            results[f'select_{select_type}'] = {
                'desc': config['desc'],
                'pick': config['pick'],
                'numbers': vote_result['selected'],
                'candidates': vote_result['candidates'][:10],
                'version': vote_result['version'],
            }

        # 选5复式7码
        results['fu_shi_7'] = self.get_fu_shi_7()

        # 最近开奖数据
        recent = self.history_data[:10] if self.history_data else []
        results['recent_results'] = [
            {
                'issue': r['issue'],
                'numbers': r['numbers'],
                'date': r['date'],
            }
            for r in recent
        ]

        # 统计信息
        results['statistics'] = {
            'total_periods': self.statistics.get('total_periods', 0),
            'expected_freq': self.statistics.get('expected_freq', 0),
            'expected_gap': round(self.statistics.get('expected_gap', 4), 1),
            'last_numbers': sorted(list(self.statistics.get('last_numbers', set()))),
            'version': KL8_PREDICTOR_VERSION,
        }

        # 号码排名(Top20)
        results['ranking'] = self.get_ensemble_ranking(top_n=20)

        results['using_simulated_data'] = self.using_simulated_data

        return results


# ─── 单例与缓存 ───

_analyzer_instance = None
_prediction_cache = {'data': None, 'timestamp': 0}


def get_kl8_analyzer() -> KL8Analyzer:
    """获取KL8Analyzer单例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = KL8Analyzer()
    return _analyzer_instance


def run_prediction(force_refresh: bool = False) -> Dict:
    """运行快乐8预测(含缓存)"""
    now = time.time()

    # 检查缓存(同一天内有效)
    cache = _prediction_cache
    if not force_refresh and cache['data'] is not None:
        from datetime import date
        if date.fromtimestamp(cache['timestamp']) == date.today():
            return cache['data']

    analyzer = get_kl8_analyzer()
    result = analyzer.predict_all()

    _prediction_cache['data'] = result
    _prediction_cache['timestamp'] = time.time()

    return result


def clear_cache():
    """清除缓存"""
    global _analyzer_instance, _prediction_cache
    _analyzer_instance = None
    _prediction_cache = {'data': None, 'timestamp': 0}
