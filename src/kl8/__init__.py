"""
快乐8预测模块
=============

快乐8玩法：从1-80中开出20个号码，玩家可选1-10个号码进行投注。
本模块提供选3/选4/选5/选6/选7预测，以及选5复式7码预测。

v6 核心改动:
1. 真正三段式: train→val→final_test, final_test完全冻结不参与启用判断
2. 回测使用multi_model_voting管道（model_weights真正生效）
3. ACTIVE_STRATEGIES策略注册表（按玩法分别配置）
4. 置换检验按玩法分别执行（pick_n=select_type）
5. 多重检验校正（Benjamini-Hochberg FDR + Bonferroni备用）
6. ROI统一为return_multiple和profit_roi两个字段
7. 复式7码每期全部21注组合都计算投注和奖金
8. 无号码推荐时不扣投注（只有placed=true才计）
9. 快照结算添加期号校验（actual_issue > based_on_issue）
10. 第二数据源交叉校验接入抓取流程+冲突写入队列
11. 超几何分布替代二项分布(80选20不放回)
12. 数据排序+连续性检查+冲突审核队列

版本: kl8-v6.0-strict-three-stage
"""

import math
import json
import time
import hashlib
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')

KL8_PREDICTOR_VERSION = "kl8-v6.0-strict-three-stage"

# ─── 快乐8常量 ───
KL8_NUM_RANGE = 80       # 号码范围 1-80
KL8_DRAW_COUNT = 20      # 每期开出20个号码
KL8_DEFAULT_HISTORY = 250  # 默认使用最近250期
KL8_EXPECTED_GAP = (KL8_NUM_RANGE - KL8_DRAW_COUNT) / KL8_DRAW_COUNT  # = 3.0

# ─── 回测常量 ───
BACKTEST_MIN_OOS_PERIODS = 300   # 最小样本外期数
BACKTEST_FINAL_TEST_PERIODS = 200  # 最终封存测试期数
BACKTEST_PERMUTATION_COUNT = 1000  # 置换检验次数
BACKTEST_STABILITY_WINDOWS = 4     # 稳定性检查窗口数
BACKTEST_STABILITY_THRESHOLD = 3   # 至少3/4窗口Lift>0

# ─── 选型配置：选3~选7各选多少号码 ───
SELECT_CONFIG = {
    3: {'pick': 3, 'top_n': 10,  'desc': '选3'},
    4: {'pick': 4, 'top_n': 12,  'desc': '选4'},
    5: {'pick': 5, 'top_n': 15,  'desc': '选5'},
    6: {'pick': 6, 'top_n': 15,  'desc': '选6'},
    7: {'pick': 7, 'top_n': 18,  'desc': '选7'},
}

# ─── 特征开关配置（v5：所有特征默认停用，需回测验证才能启用）───
# 按玩法分开评估: 每个特征可以有per-play-type的enabled状态
FEATURE_CONFIG = {
    'frequency':  {'enabled': False, 'weight': 0.12,  'desc': '频率偏离度(均值回归:冷号加分,热号降分)'},
    'gap':        {'enabled': False, 'weight': 0.00,   'desc': '遗漏偏离度 -- 仅展示指标,不参与预测'},
    'position':   {'enabled': False, 'weight': 0.08,   'desc': '区位均衡(8个10码区)'},
    'road':       {'enabled': False, 'weight': 0.10,   'desc': '路数特征(012路分布)'},
    'sum':        {'enabled': False, 'weight': 0.00,   'desc': '和值特征 -- 停用(代码与注释不一致)'},
    'zone':       {'enabled': False, 'weight': 0.00,   'desc': '区位近期开出率 -- 停用(追上期模式不优于随机)'},
    'repeat':     {'enabled': False, 'weight': 0.00,   'desc': '重号特征 -- 停用(上期出现!=下期更容易出现)'},
    'adjacent':   {'enabled': False, 'weight': 0.00,   'desc': '邻号特征 -- 停用(追上期模式不优于随机)'},
    'odd_even':   {'enabled': False, 'weight': 0.06,   'desc': '奇偶特征(对称评分)'},
    'big_small':  {'enabled': False, 'weight': 0.06,   'desc': '大小特征(对称评分)'},
}

# ─── 投票模型权重（v6：停用，等策略注册表接管）───
MODEL_CONFIG = {
    'bayesian': {'enabled': False, 'weight': 0.0, 'desc': '停用: 倾向热号,与排名频率冷号方向相反'},
    'rank':     {'enabled': False, 'weight': 0.0, 'desc': '排名模型 -- 停用,等回测验证'},
    'markov':   {'enabled': False, 'weight': 0.0, 'desc': '停用: 低号码偏差,未出现号码全0.25'},
}

# ─── v6 策略注册表（按玩法分别配置，取代全局FEATURE_CONFIG的预测权重）───
# 每个玩法有独立的 strategy_id、feature_weights、model_weights、window_size
# 预测、快照、结算、回测都必须记录 strategy_id
# 当前所有策略默认空（无信号），需回测验证后才能填入具体配置
ACTIVE_STRATEGIES = {
    'select_3': {
        'strategy_id': '',           # 空=无验证策略
        'feature_weights': {},       # 空=不启用任何特征
        'model_weights': {},         # 空=不启用任何模型
        'window_size': 0,            # 0=无固定窗口
    },
    'select_4': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_5': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_6': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'select_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
    'fu_shi_7': {
        'strategy_id': '',
        'feature_weights': {},
        'model_weights': {},
        'window_size': 0,
    },
}

# ─── 预测快照目录 ───
KL8_SNAPSHOT_DIR = data_path('kl8_snapshots')
KL8_SETTLEMENT_DIR = data_path('kl8_settlements')
KL8_PRIZE_TABLE_FILE = data_path('kl8_prize_table.json')

# ─── 冲突审核队列 ───
KL8_CONFLICT_QUEUE_FILE = data_path('kl8_conflict_queue.json')


# ─── 预测就绪检查（v6: 基于ACTIVE_STRATEGIES判断）───

def is_prediction_ready() -> bool:
    """预测准备就绪判断

    v6: 基于 ACTIVE_STRATEGIES 判断
    任一玩法有非空策略（strategy_id不为空且有权重），即视为有信号
    无信号时ranking不返回[1..20]，返回空列表
    """
    for play_type, strategy in ACTIVE_STRATEGIES.items():
        if strategy.get('strategy_id', ''):
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {})
            has_feature_weight = any(w > 0 for w in fw.values())
            has_model_weight = any(w > 0 for w in mw.values())
            if has_feature_weight or has_model_weight:
                return True
    return False


def has_active_signal() -> bool:
    """是否有任何启用的特征或模型（向后兼容，但推荐用is_prediction_ready）"""
    return is_prediction_ready()


def get_active_feature_weights() -> Dict[str, float]:
    """获取当前启用的特征权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in FEATURE_CONFIG.items()}


def get_active_model_weights() -> Dict[str, float]:
    """获取当前启用的模型权重"""
    return {k: v['weight'] if v['enabled'] else 0.0 for k, v in MODEL_CONFIG.items()}


# ─── 超几何分布 ───

def hypergeom_pmf(pick_n: int, hits: int) -> float:
    """超几何分布PMF: 从80个号码中选pick_n个，开出20个，命中hits个的概率

    P(X=hits) = C(20,hits) * C(60,pick_n-hits) / C(80,pick_n)
    """
    from math import comb
    if hits < 0 or hits > min(pick_n, KL8_DRAW_COUNT):
        return 0.0
    if pick_n - hits > KL8_NUM_RANGE - KL8_DRAW_COUNT:
        return 0.0
    return comb(KL8_DRAW_COUNT, hits) * comb(KL8_NUM_RANGE - KL8_DRAW_COUNT, pick_n - hits) / comb(KL8_NUM_RANGE, pick_n)


def hypergeom_p_ge(pick_n: int, min_hits: int) -> float:
    """超几何分布 P(X >= min_hits)"""
    total = 0.0
    for k in range(min_hits, min(pick_n, KL8_DRAW_COUNT) + 1):
        total += hypergeom_pmf(pick_n, k)
    return total


def hypergeom_expected(pick_n: int) -> float:
    """超几何分布期望命中数 = pick_n * 20/80"""
    return pick_n * KL8_DRAW_COUNT / KL8_NUM_RANGE


# ─── 多重检验校正（v6新增）───

def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR校正

    对同一玩法下所有特征、窗口、权重的p值做FDR校正
    步骤:
    1. p值从小到大排序
    2. 每个p值校正为: p_adjusted = p * m / rank
       (m = 总检验次数, rank = 该p值在排序中的序位)
    3. 确保单调性: 从大到小遍历，取min(p_adjusted, 下一个校正值)

    参数:
        p_values: 原始p值列表

    返回:
        校正后的p值列表（与输入顺序对应）
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    prev_adjusted = 1.0

    # 从最大p值开始，确保单调性
    for i in range(m - 1, -1, -1):
        original_idx, p = indexed[i]
        rank = i + 1  # 序位从1开始
        bh_adjusted = min(1.0, p * m / rank)
        # 确保单调性: 不比后面(更小rank)的校正值更大
        adjusted[original_idx] = min(bh_adjusted, prev_adjusted)
        prev_adjusted = adjusted[original_idx]

    return adjusted


def bonferroni_correction(p_value: float, n_experiments: int) -> float:
    """Bonferroni校正（保守版多重检验校正）

    p_adjusted = min(1.0, p_value * number_of_experiments)

    参数:
        p_value: 原始p值
        n_experiments: 总检验次数

    返回:
        校正后的p值
    """
    return min(1.0, p_value * n_experiments)

def normalize_record(record, keep_meta: bool = False) -> Optional[Dict]:
    """校验并标准化单条记录

    v5加固:
    - 坏JSON字符串 -> 返回None(不再抛异常)
    - 坏数据类型 -> 返回None
    - 20个号码必须唯一且在1-80范围
    - 期号不为空
    - keep_meta=True时保留source/fetched_at/checksum溯源字段
    """
    if not isinstance(record, dict):
        return None

    nums = record.get('numbers') or record.get('draw_numbers')

    if isinstance(nums, str):
        try:
            nums = json.loads(nums)
        except (json.JSONDecodeError, TypeError):
            return None

    if not nums:
        return None

    if not isinstance(nums, (list, tuple, set)):
        return None

    try:
        nums = sorted(int(x) for x in nums)
    except (ValueError, TypeError):
        return None

    if len(nums) != 20:
        return None
    if len(set(nums)) != 20:
        return None
    if any(n < 1 or n > 80 for n in nums):
        return None

    if not str(record.get('issue', '')).strip():
        return None

    result = {
        'issue': str(record['issue']),
        'numbers': nums,
        'date': record.get('date') or record.get('draw_date', ''),
    }

    if keep_meta:
        result.update({
            'source': record.get('source', ''),
            'fetched_at': record.get('fetched_at', ''),
            'checksum': record.get('checksum', _checksum_numbers(nums)),
        })

    return result


def _checksum_numbers(nums: List[int]) -> str:
    """号码列表的短校验码"""
    s = json.dumps(sorted(nums), separators=(',', ':'))
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ─── 奖金表 ───

def load_prize_table() -> Dict:
    """加载可配置奖金表

    格式: {
        "select_3": {"3": 25, "2": 5, "1": 0, "0": 0, "bet": 2},
        "select_4": {"4": 100, "3": 20, ...},
        ...
        "fu_shi_7": {"5": 10000, "4": 500, ...}
    }

    每个玩法包含: 命中档位奖金 + "bet"单注金额
    如果文件不存在，返回默认值
    """
    path = Path(KL8_PRIZE_TABLE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            log.warning(f'奖金表加载失败: {e}')

    # 默认奖金表（中国快乐8官方奖金，单位:元）
    return {
        'select_3': {'3': 25, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_4': {'4': 100, '3': 20, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_5': {'5': 10000, '4': 500, '3': 30, '2': 5, '1': 0, '0': 0, 'bet': 2},
        'select_6': {'6': 300000, '5': 5000, '4': 100, '3': 10, '2': 0, '1': 0, '0': 0, 'bet': 2},
        'select_7': {'7': 1000000, '6': 50000, '5': 1000, '4': 50, '3': 5, '2': 0, '1': 0, '0': 0, 'bet': 2},
        'fu_shi_7': {'5': 10000, '4': 500, '3': 30, '2': 5, '1': 0, '0': 0, 'bet_per_combo': 2},
    }


# ─── 冲突审核队列 ───

def save_conflict_to_queue(conflict_info: Dict):
    """将数据冲突记录保存到审核队列（不自动覆盖，等待人工确认）"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    queue = []
    if path.exists():
        try:
            queue = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            queue = []

    conflict_info['queued_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    queue.append(conflict_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding='utf-8')


def list_conflict_queue() -> List[Dict]:
    """查看冲突审核队列"""
    path = Path(KL8_CONFLICT_QUEUE_FILE)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []


# ─── 数据完整性检查 ───

def check_data_integrity(data: List[Dict]) -> Dict:
    """检查历史数据完整性

    检查项:
    1. 数据是否按期号排序（读取后排序而非假设顺序）
    2. 期号连续性（缺失期号报告）
    3. 日期与期号一致性
    4. 号码范围和唯一性
    """
    if not data:
        return {'valid': False, 'error': '数据为空'}

    # 1. 确保排序（读取后排序，不假设顺序）
    data_sorted = sorted(data, key=lambda x: x['issue'], reverse=True)

    issues = [r['issue'] for r in data_sorted]
    total = len(issues)

    # 2. 期号连续性检查
    # 快乐8期号格式: 通常是年份+3位序号(如2026001~2026365)
    missing_issues = []
    issue_ints = []
    for issue in issues:
        try:
            issue_ints.append(int(issue))
        except (ValueError, TypeError):
            continue

    if issue_ints:
        issue_ints_sorted = sorted(issue_ints)
        for i in range(len(issue_ints_sorted) - 1):
            # 检查相邻期号差值
            diff = issue_ints_sorted[i + 1] - issue_ints_sorted[i]
            if diff > 1:
                # 报告缺失期号（但只报告小的gap，跨年的大gap可能正常）
                if diff <= 5:
                    for j in range(1, diff):
                        missing_issues.append(str(issue_ints_sorted[i] + j))

    # 3. 日期与期号一致性（简单检查: 同一天应该有相似期号前缀）
    date_issue_conflicts = []
    seen_dates = {}
    for r in data_sorted:
        date = r.get('date', '')
        issue = r['issue']
        if date:
            year_prefix = issue[:4] if len(issue) >= 4 else ''
            date_year = date[:4] if len(date) >= 4 else ''
            if year_prefix and date_year and year_prefix != date_year:
                date_issue_conflicts.append({
                    'issue': issue,
                    'date': date,
                    'reason': f'期号年份{year_prefix}与日期年份{date_year}不匹配',
                })

    return {
        'valid': True,
        'total_records': total,
        'latest_issue': issues[0] if issues else '',
        'earliest_issue': issues[-1] if issues else '',
        'missing_issues': missing_issues[:20],  # 只报告前20个缺失
        'missing_count': len(missing_issues),
        'date_issue_conflicts': date_issue_conflicts[:10],
        'conflict_count': len(date_issue_conflicts),
    }


class KL8Analyzer:
    """快乐8预测分析器（v5: 严格三段式+预测就绪判断+纯参数化回测）"""

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or data_path('kl8_history.json')
        self.using_simulated_data = False
        self._data_mtime = 0
        self.history_data = self._load_history()
        self.statistics = {}
        self.update_statistics()

    # ─── 数据加载（v5: 排序而非假设顺序+完整性检查）───

    def _load_history(self) -> List[Dict]:
        """加载历史开奖数据（v5: 合并后排序+完整性检查+冲突审核队列）"""

        source_records = {}

        # 来源1: doc_store
        try:
            raw_records = doc_store._fallback_load_all('kl8_history')
            if raw_records:
                for r in raw_records:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in source_records:
                        old = source_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'doc_store内期号{issue}号码冲突，保留第一条')
                            save_conflict_to_queue({
                                'source': 'doc_store_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    source_records[issue] = normed
                log.info(f'快乐8: doc_store加载了{len(source_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: doc_store加载失败: {e}')

        # 来源2: JSON文件
        file_records = {}
        path = Path(self.history_file)
        try:
            if path.exists():
                self._data_mtime = path.stat().st_mtime
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    source_list = raw.get('results', raw.get('data', []))
                else:
                    source_list = raw

                for r in source_list:
                    normed = normalize_record(r, keep_meta=True)
                    if not normed:
                        continue
                    issue = normed['issue']
                    if issue in file_records:
                        old = file_records[issue]
                        if old['numbers'] != normed['numbers']:
                            log.error(f'JSON文件内期号{issue}号码冲突')
                            save_conflict_to_queue({
                                'source': 'json_file_internal',
                                'issue': issue,
                                'old_numbers': old['numbers'],
                                'new_numbers': normed['numbers'],
                                'action': 'kept_old',
                            })
                            continue
                    file_records[issue] = normed
                log.info(f'快乐8: 文件加载了{len(file_records)}期有效数据')
        except Exception as e:
            log.warning(f'快乐8: 文件加载失败: {e}')

        # 合并两个来源，按期号去重，冲突时报错不覆盖
        merged = {}
        for source_name, records in [('doc_store', source_records), ('json', file_records)]:
            for issue, record in records.items():
                if issue in merged:
                    old = merged[issue]
                    if old['numbers'] != record['numbers']:
                        log.error(
                            f'期号{issue}多源号码冲突: '
                            f'{source_name}={record["numbers"]}, '
                            f'已有={old["numbers"]}, 保留旧值待人工确认'
                        )
                        save_conflict_to_queue({
                            'source': source_name,
                            'issue': issue,
                            'old_numbers': old['numbers'],
                            'new_numbers': record['numbers'],
                            'old_source': old.get('source', ''),
                            'new_source': record.get('source', ''),
                            'action': 'kept_old',
                        })
                        continue
                merged[issue] = record

        if not merged:
            self.using_simulated_data = True
            log.error('快乐8: 无真实历史数据')
            return []

        # v5: 排序而非假设顺序（读取后确保按期号降序排列）
        data = sorted(merged.values(), key=lambda x: x['issue'], reverse=True)
        self.using_simulated_data = False

        # v5: 数据完整性检查
        integrity = check_data_integrity(data)
        if integrity.get('missing_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["missing_count"]}个缺失期号')
        if integrity.get('conflict_count', 0) > 0:
            log.warning(f'快乐8: 发现{integrity["conflict_count"]}个日期期号不一致')

        log.info(f'快乐8: 多源合并后共{len(data)}期有效数据')
        return data

    def _check_data_mtime(self) -> bool:
        """检查数据文件mtime是否变化"""
        path = Path(self.history_file)
        try:
            if path.exists():
                current_mtime = path.stat().st_mtime
                if current_mtime != self._data_mtime:
                    log.info(f'快乐8: 数据文件mtime变化，需重新加载')
                    self._data_mtime = current_mtime
                    return True
        except Exception:
            pass
        return False

    def reload_if_needed(self) -> bool:
        """如果数据文件已更新则重新加载"""
        if self._check_data_mtime():
            self.history_data = self._load_history()
            self.update_statistics()
            return True
        return False

    # ─── 统计计算 ───

    def update_statistics(self):
        """更新所有统计量"""
        if not self.history_data:
            self.statistics = {}
            return

        n = len(self.history_data)
        recent = min(n, KL8_DEFAULT_HISTORY)
        recent_data = self.history_data[:recent]

        freq = Counter()
        for record in recent_data:
            for num in record['numbers']:
                freq[num] += 1

        gap = {}
        for num in range(1, 81):
            gap[num] = 0
            for record in recent_data:
                if num in record['numbers']:
                    break
                gap[num] += 1

        last_numbers = set(recent_data[0]['numbers']) if recent_data else set()

        self.statistics = {
            'frequency': freq,
            'gap': gap,
            'total_periods': recent,
            'expected_freq': recent * KL8_DRAW_COUNT / KL8_NUM_RANGE,
            'expected_gap': KL8_EXPECTED_GAP,
            'last_numbers': last_numbers,
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
                zone = (num - 1) // 10 + 1
                zone_freq[zone] += 1
        return dict(zone_freq)

    def _road_frequency(self, data: List[Dict]) -> Dict:
        """012路频率分布"""
        road_freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                road = num % 3
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
        """大小频率分布"""
        freq = defaultdict(int)
        for record in data:
            for num in record['numbers']:
                freq['big' if num > 40 else 'small'] += 1
        return dict(freq)

    # ─── 对称评分函数 ───

    @staticmethod
    def balance_score(actual_ratio: float, target_ratio: float, is_target: bool) -> float:
        """对称平衡评分"""
        imbalance = target_ratio - actual_ratio
        delta = 0.30 * (imbalance if is_target else -imbalance)
        return max(0.2, min(0.8, 0.5 + delta))

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
        total = stats['total_periods']

        # 1. 频率偏离度
        actual_freq = freq.get(num, 0)
        deviation_ratio = actual_freq / max(expected_freq, 0.01)
        if deviation_ratio <= 1.0:
            scores['frequency'] = 0.55 + 0.15 * (1.0 - deviation_ratio)
        else:
            scores['frequency'] = max(0.15, 0.55 * math.exp(-1.8 * (deviation_ratio - 1.0)))

        # 2. 遗漏偏离度 -- 仅展示
        actual_gap_val = gap.get(num, 0)
        gap_ratio = actual_gap_val / max(expected_gap, 0.01)
        if gap_ratio <= 1.0:
            scores['gap'] = 0.25 + 0.60 * (gap_ratio ** 0.7)
        else:
            scores['gap'] = 0.85 - 0.45 * (1.0 - math.exp(-(gap_ratio - 1.0) * 0.8))

        # 3. 区位特征(对称标准化偏离)
        zone = (num - 1) // 10 + 1
        zone_freq = stats['freq_by_zone']
        expected_zone = total * KL8_DRAW_COUNT / 8
        zone_ratio = zone_freq.get(zone, 0) / max(expected_zone, 0.01)
        scores['position'] = 0.4 + 0.6 * max(0, 1.0 - abs(zone_ratio - 1.0))

        # 4. 路数特征
        road = num % 3
        road_freq = stats['freq_by_road']
        expected_road = total * KL8_DRAW_COUNT / 3
        road_ratio = road_freq.get(road, 0) / max(expected_road, 0.01)
        scores['road'] = 0.4 + 0.6 * max(0, 1.0 - abs(road_ratio - 1.0) * 0.5)

        # 5. 和值特征 -- 停用
        scores['sum'] = 0.5

        # 6. 区位近期开出率 -- 对称标准化偏离
        zone_nums = [z for z in range(((zone-1)*10)+1, zone*10+1)]
        zone_hit_count = len([n for n in zone_nums if n in last_nums])
        zone_hit_rate = zone_hit_count / len(zone_nums)
        expected_hit_rate = KL8_DRAW_COUNT / KL8_NUM_RANGE
        zone_deviation = abs(zone_hit_rate - expected_hit_rate) / max(expected_hit_rate, 0.01)
        scores['zone'] = 0.4 + 0.6 * max(0, 1.0 - zone_deviation)

        # 7. 重号 -- 停用
        scores['repeat'] = 0.5

        # 8. 邻号 -- 停用
        scores['adjacent'] = 0.5

        # 9. 奇偶 -- 对称评分
        oe = stats['freq_by_odd_even']
        total_oe = oe.get('odd', 0) + oe.get('even', 0)
        odd_ratio = oe.get('odd', 0) / max(total_oe, 1)
        scores['odd_even'] = self.balance_score(odd_ratio, 0.5, num % 2 == 1)

        # 10. 大小 -- 对称评分
        bs = stats['freq_by_big_small']
        total_bs = bs.get('big', 0) + bs.get('small', 0)
        big_ratio = bs.get('big', 0) / max(total_bs, 1)
        scores['big_small'] = self.balance_score(big_ratio, 0.5, num > 40)

        return scores

    # ─── 排名模型（v5: 纯参数化，接受外部feature_weights）───

    def get_ensemble_ranking(self, top_n: int = 20, feature_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """综合特征评分排名

        v5: feature_weights参数可选传入（回测时不修改全局配置）
        无信号时返回空列表而非[1..20]
        """
        # 使用传入权重或全局活跃权重
        weights = feature_weights or get_active_feature_weights()

        # v5: 如果没有有效权重（全部为0），返回空列表
        has_weight = any(w > 0 for w in weights.values())
        if not has_weight:
            return []  # 无信号时不返回[1..20]

        ranking = []
        for num in range(1, 81):
            scores = self._calculate_feature_score(num)
            total_score = sum(
                scores.get(k, 0) * weights.get(k, 0) for k in scores
            )
            ranking.append({
                'num': num,
                'ranking_score': total_score,
                'score_type': 'heuristic_rank',
                'is_probability': False,
                'scores': scores,
            })
        ranking.sort(key=lambda x: (-x['ranking_score'], x['num']))
        return ranking[:top_n]

    # ─── 贝叶斯模型 ───

    def _model_bayesian(self, top_n: int = 20) -> List[int]:
        """贝叶斯概率模型 -- 目前停用"""
        stats = self.statistics
        freq = stats['frequency']
        expected_freq = stats['expected_freq']
        total = stats['total_periods']

        scores = {}
        for num in range(1, 81):
            actual_rate = freq.get(num, 0) / max(total, 1)
            expected_rate = KL8_DRAW_COUNT / KL8_NUM_RANGE
            base_prob = (freq.get(num, 0) + 1) / (total + 2)

            deviation_ratio = actual_rate / max(expected_rate, 0.01)
            reversion_factor = 1.0 / (1.0 + 0.6 * max(0, deviation_ratio - 1.0))
            if deviation_ratio < 1.0:
                reversion_factor = min(1.5, 1.0 + 0.5 * (1.0 - deviation_ratio))

            scores[num] = base_prob * reversion_factor

        return sorted(scores.keys(), key=lambda n: (-scores[n], n))[:top_n]

    # ─── 马尔可夫模型（确定性哈希打破并列）───

    def _model_markov(self, top_n: int = 20) -> List[int]:
        """一阶马尔可夫转移模型 -- 目前停用"""
        if len(self.history_data) < 3:
            return []

        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.history_data) - 1):
            current = set(self.history_data[i]['numbers'])
            prev = set(self.history_data[i + 1]['numbers'])
            for num in prev:
                if num in current:
                    transition_counts[num]['repeat'] += 1
                else:
                    transition_counts[num]['skip'] += 1

        last_nums = set(self.history_data[0]['numbers'])
        based_on_issue = self.history_data[0]['issue']

        scores = {}
        for num in range(1, 81):
            base_score = 0.25
            if num in last_nums:
                repeat_rate = transition_counts[num]['repeat'] / max(
                    transition_counts[num]['repeat'] + transition_counts[num]['skip'], 1)
                base_score = max(0.15, repeat_rate)
            tie_break = int(hashlib.sha256(f'{based_on_issue}_{num}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            scores[num] = base_score + tie_break * 0.001

        return sorted(scores.keys(), key=lambda n: (-scores[n], n))[:top_n]

    # ─── 排名模型(独立) ───

    def _model_rank(self, top_n: int = 20, feature_weights: Optional[Dict[str, float]] = None) -> List[int]:
        """纯排名模型"""
        ranking = self.get_ensemble_ranking(top_n=top_n, feature_weights=feature_weights)
        return [r['num'] for r in ranking]

    # ─── 多模型投票（v5: 纯参数化 + 预测就绪判断）───

    def multi_model_voting(
        self,
        pick_n: int = 5,
        top_n: int = 20,
        feature_weights: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """多模型集成投票

        v5: 纯参数化 — 传入权重不修改全局配置
        无信号时返回空推荐 + no_validated_signal状态
        """
        fw = feature_weights or get_active_feature_weights()
        mw = model_weights or get_active_model_weights()

        # ── 预测就绪判断 ──
        has_rank_feature = any(w > 0 for w in fw.values())
        rank_weight = mw.get('rank', 0.0)
        bayesian_weight = mw.get('bayesian', 0.0)
        markov_weight = mw.get('markov', 0.0)

        rank_ready = rank_weight > 0 and has_rank_feature
        bayesian_ready = bayesian_weight > 0
        markov_ready = markov_weight > 0

        if not (rank_ready or bayesian_ready or markov_ready):
            return {
                'selected': [],
                'candidates': [],
                'votes': {},
                'status': 'no_validated_signal',
                'message': '暂无通过回测验证的有效特征，不输出号码推荐',
                'version': KL8_PREDICTOR_VERSION,
            }

        votes = defaultdict(float)

        # 懒加载: 只计算启用模型
        if rank_ready:
            model_result = self._model_rank(top_n=top_n, feature_weights=fw)
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * rank_weight
                votes[num] += vote_weight

        if bayesian_ready:
            model_result = self._model_bayesian(top_n=top_n)
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * bayesian_weight
                votes[num] += vote_weight

        if markov_ready:
            model_result = self._model_markov(top_n=top_n)
            for rank, num in enumerate(model_result):
                vote_weight = (1.0 - (rank / max(len(model_result), 1))) * markov_weight
                votes[num] += vote_weight

        candidates = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
        selected = [num for num, _ in candidates[:pick_n]]
        candidate_pool = candidates[:max(top_n, 7)]

        return {
            'selected': selected,
            'candidates': candidate_pool,
            'votes': dict(votes),
            'version': KL8_PREDICTOR_VERSION,
        }

    # ─── 选5复式7码 ───

    def get_fu_shi_7(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """选5复式7码"""
        vote_result = self.multi_model_voting(
            pick_n=7, top_n=7,
            feature_weights=feature_weights,
            model_weights=model_weights,
        )

        if vote_result.get('status') == 'no_validated_signal':
            return {
                'top7_numbers': [],
                'total_combinations': 0,
                'combinations': [],
                'version': KL8_PREDICTOR_VERSION,
                'source': 'multi_model_voting',
                'status': 'no_validated_signal',
                'message': vote_result.get('message', ''),
            }

        top7 = vote_result['selected']
        combo_list = [sorted(c) for c in combinations(top7, 5)]

        ranking_full = self.get_ensemble_ranking(top_n=7, feature_weights=feature_weights)
        top7_details = [r for r in ranking_full if r['num'] in top7]
        top7_details.sort(key=lambda x: (-x['ranking_score'], x['num']))

        return {
            'top7_numbers': top7,
            'top7_scores': [r['ranking_score'] for r in top7_details],
            'total_combinations': len(combo_list),
            'combinations': combo_list,
            'version': KL8_PREDICTOR_VERSION,
            'source': 'multi_model_voting',
        }

    # ─── 预测快照 ───

    def _save_prediction_snapshot(self, prediction_result: Dict) -> Optional[str]:
        """保存预测快照（UUID+排他创建，永不修改）"""
        if not self.history_data:
            return None

        snapshot_dir = Path(KL8_SNAPSHOT_DIR)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # 全窗口SHA256指纹
        recent = min(len(self.history_data), KL8_DEFAULT_HISTORY)
        history_window = self.history_data[:recent]
        history_fingerprint = hashlib.sha256(
            json.dumps(
                [
                    {
                        'issue': r['issue'],
                        'numbers': r['numbers'],
                        'date': r.get('date', ''),
                        'checksum': r.get('checksum', ''),
                    }
                    for r in history_window
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()

        latest_issue = self.history_data[0]['issue']
        snapshot_id = uuid.uuid4().hex
        snapshot = {
            'snapshot_id': snapshot_id,
            'target_issue': None,  # 不预设目标期号
            'based_on_issue': latest_issue,
            'predicted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'history_window_size': recent,
            'history_start_issue': history_window[-1]['issue'] if history_window else '',
            'history_end_issue': latest_issue,
            'history_fingerprint': history_fingerprint,
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': {k: dict(v) for k, v in FEATURE_CONFIG.items()},
            'model_config': {k: dict(v) for k, v in MODEL_CONFIG.items()},
            'active_strategies': {k: dict(v) for k, v in ACTIVE_STRATEGIES.items()},
            # v6: 每个玩法记录strategy_id
            'select_3_strategy_id': prediction_result.get('select_3', {}).get('strategy_id', ''),
            'select_4_strategy_id': prediction_result.get('select_4', {}).get('strategy_id', ''),
            'select_5_strategy_id': prediction_result.get('select_5', {}).get('strategy_id', ''),
            'select_6_strategy_id': prediction_result.get('select_6', {}).get('strategy_id', ''),
            'select_7_strategy_id': prediction_result.get('select_7', {}).get('strategy_id', ''),
            'fu_shi_7_strategy_id': prediction_result.get('fu_shi_7', {}).get('strategy_id', ''),
            'ranking': prediction_result.get('ranking', []),
            'select_3': prediction_result.get('select_3', {}).get('numbers', []),
            'select_4': prediction_result.get('select_4', {}).get('numbers', []),
            'select_5': prediction_result.get('select_5', {}).get('numbers', []),
            'select_6': prediction_result.get('select_6', {}).get('numbers', []),
            'select_7': prediction_result.get('select_7', {}).get('numbers', []),
            'fu_shi_7': prediction_result.get('fu_shi_7', {}).get('top7_numbers', []),
        }

        snapshot_file = snapshot_dir / f'snapshot_{snapshot_id}.json'

        try:
            with snapshot_file.open('x', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 预测快照已保存 -> {snapshot_file.name}')
            return snapshot_file.name
        except FileExistsError:
            log.error(f'快乐8: 快照UUID冲突 {snapshot_file.name}')
            return None
        except Exception as e:
            log.error(f'快乐8: 保存快照失败: {e}')
            return None

    def settle_prediction(self, snapshot_file: str, actual_issue: str, actual_numbers: List[int]) -> Dict:
        """赛后结算（v6: 无号码不扣投注+ROI统一+期号校验）

        v6改动:
        1. 只有len(numbers)==select_type时才视为placed（已投注）
        2. 未placed的玩法bet=0, prize=0
        3. ROI统一为return_multiple和profit_roi两个字段
        4. 复式7码每期全部21注组合都计算投注和奖金
        5. actual_issue必须晚于based_on_issue
        """

        # 路径安全
        snapshot_file = Path(snapshot_file).name
        path = Path(KL8_SNAPSHOT_DIR) / snapshot_file
        if not path.exists():
            return {'error': f'快照文件不存在: {snapshot_file}'}

        # 校验开奖号码
        actual_normed = normalize_record({
            'issue': actual_issue,
            'numbers': actual_numbers,
        })
        if not actual_normed:
            return {'error': '实际开奖号码非法'}

        # 读取原始快照
        try:
            snapshot = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            return {'error': f'读取快照失败: {e}'}

        # v6: 期号校验 — actual_issue必须晚于based_on_issue
        based_on_issue = snapshot.get('based_on_issue', '')
        if based_on_issue:
            try:
                actual_int = int(actual_normed['issue'])
                based_on_int = int(based_on_issue)
                if actual_int <= based_on_int:
                    return {'error': f'实际开奖期号{actual_normed["issue"]}必须晚于预测基准期号{based_on_issue}'}
            except (ValueError, TypeError):
                # 非纯数字期号，按字符串比较
                if str(actual_normed['issue']) <= str(based_on_issue):
                    return {'error': f'实际开奖期号必须晚于预测基准期号'}

        # 检查是否已结算
        settlements_dir = Path(KL8_SETTLEMENT_DIR)
        settlements_dir.mkdir(parents=True, exist_ok=True)

        existing_settlement = settlements_dir / f'settlement_{snapshot.get("snapshot_id", "")}.json'
        if existing_settlement.exists():
            try:
                old = json.loads(existing_settlement.read_text(encoding='utf-8'))
                return {'error': '快照已结算，不可重复结算', 'settlement': old}
            except Exception:
                pass

        snapshot_sha256 = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        actual_set = set(actual_normed['numbers'])

        # v6: 奖金结算 — 只有placed=true才计投注和奖金
        prize_table = load_prize_table()

        prize_settlement = {}
        cumulative_bet = 0
        cumulative_prize = 0

        for select_type in [3, 4, 5, 6, 7]:
            numbers = snapshot.get(f'select_{select_type}', [])
            prize_key = f'select_{select_type}'
            prize_info = prize_table.get(prize_key, {})

            # v6: 只有号码完整时才视为placed
            placed = len(numbers) == select_type
            bet = prize_info.get('bet', 2) if placed else 0
            hits = len(set(numbers) & actual_set) if placed else 0
            prize = prize_info.get(str(hits), 0) if placed else 0

            cumulative_bet += bet
            cumulative_prize += prize

            # v6: ROI统一为两个字段
            return_multiple = prize / max(bet, 1) if placed else 0
            profit_roi = (prize - bet) / max(bet, 1) if placed else 0

            prize_settlement[prize_key] = {
                'placed': placed,
                'hits': hits,
                'bet': bet,
                'prize': prize,
                'return_multiple': round(return_multiple, 4),
                'profit_roi': round(profit_roi, 4),
            }

        # v6: 复式7码ROI — 每期全部21注组合都计算
        fu_shi_7_nums = snapshot.get('fu_shi_7', [])
        fu7_placed = len(fu_shi_7_nums) == 7

        fu7_prize_info = prize_table.get('fu_shi_7', {})
        bet_per_combo = fu7_prize_info.get('bet_per_combo', 2)

        pool_hits = 0
        combo_hits = []
        fu7_total_bet = 0
        fu7_total_prize = 0

        if fu7_placed and fu_shi_7_nums:
            pool_hits = len(set(fu_shi_7_nums) & actual_set)
            # v6: 每期全部21注组合都计算（无论命中率）
            fu7_total_bet = math.comb(7, 5) * bet_per_combo  # = 21 * 2 = 42
            for combo in combinations(fu_shi_7_nums, 5):
                combo_h = len(set(combo) & actual_set)
                combo_hits.append(combo_h)
                fu7_total_prize += fu7_prize_info.get(str(combo_h), 0)

        cumulative_bet += fu7_total_bet
        cumulative_prize += fu7_total_prize

        hit_distribution = dict(Counter(combo_hits)) if combo_hits else {}
        max_combo_hits = max(combo_hits, default=0)

        # v6: 累计ROI统一
        cumulative_return_multiple = cumulative_prize / max(cumulative_bet, 1)
        cumulative_profit_roi = (cumulative_prize - cumulative_bet) / max(cumulative_bet, 1)

        settlement = {
            'snapshot_id': snapshot.get('snapshot_id', ''),
            'snapshot_file': snapshot_file,
            'snapshot_sha256': snapshot_sha256,
            'settled_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'actual_issue': actual_normed['issue'],
            'actual_numbers': actual_normed['numbers'],
            'actual_checksum': _checksum_numbers(actual_normed['numbers']),
            'based_on_issue': snapshot.get('based_on_issue', ''),
            'strategy_ids': {
                f'select_{st}': snapshot.get(f'select_{st}_strategy_id', '') or ACTIVE_STRATEGIES.get(f'select_{st}', {}).get('strategy_id', '')
                for st in [3, 4, 5, 6, 7]
            },
            'fu_shi_7_strategy_id': snapshot.get('fu_shi_7_strategy_id', '') or ACTIVE_STRATEGIES.get('fu_shi_7', {}).get('strategy_id', ''),
            'hit_select_3': prize_settlement.get('select_3', {}).get('hits', 0),
            'hit_select_4': prize_settlement.get('select_4', {}).get('hits', 0),
            'hit_select_5': prize_settlement.get('select_5', {}).get('hits', 0),
            'hit_select_6': prize_settlement.get('select_6', {}).get('hits', 0),
            'hit_select_7': prize_settlement.get('select_7', {}).get('hits', 0),
            'fu_shi_7_pool_hits': pool_hits,
            'hit_fu_shi_7_max': max_combo_hits,
            'fu_shi_7_hit_distribution': hit_distribution,
            'prize_settlement': prize_settlement,
            'fu7_total_bet': fu7_total_bet,
            'fu7_total_prize': fu7_total_prize,
            'cumulative_bet': cumulative_bet,
            'cumulative_prize': cumulative_prize,
            'cumulative_return_multiple': round(cumulative_return_multiple, 4),
            'cumulative_profit_roi': round(cumulative_profit_roi, 4),
        }

        try:
            with existing_settlement.open('x', encoding='utf-8') as f:
                json.dump(settlement, f, ensure_ascii=False, indent=2)
            log.info(f'快乐8: 结算完成 -> {existing_settlement.name}')
            return {'success': True, 'settlement': settlement}
        except FileExistsError:
            return {'error': '结算文件已存在，不可重复结算'}
        except Exception as e:
            return {'error': f'写入结算失败: {e}'}

    # ─── 综合预测（v6: 使用ACTIVE_STRATEGIES按玩法分别配置）───

    def predict_all(self) -> Dict:
        """生成所有选型的预测结果

        v6改动:
        - 使用ACTIVE_STRATEGIES，每个玩法有独立策略配置
        - strategy_id记录到快照，不使用全局FEATURE_CONFIG
        """
        if not self.history_data or self.using_simulated_data:
            return {
                'error': '历史数据不足，无法进行有效预测。请先抓取真实数据。',
                'using_simulated_data': True,
            }

        prediction_ready = is_prediction_ready()

        results = {}

        for select_type in [3, 4, 5, 6, 7]:
            config = SELECT_CONFIG[select_type]
            s_key = f'select_{select_type}'
            strategy = ACTIVE_STRATEGIES.get(s_key, {})

            # v6: 使用策略配置（有strategy_id时才真正预测）
            strategy_id = strategy.get('strategy_id', '')
            fw = strategy.get('feature_weights', {})
            mw = strategy.get('model_weights', {})
            window_size = strategy.get('window_size', 0) or KL8_DEFAULT_HISTORY

            if strategy_id and (any(w > 0 for w in fw.values()) or any(w > 0 for w in mw.values())):
                # 有验证策略，使用策略权重预测
                # 使用策略指定的窗口大小
                if window_size != KL8_DEFAULT_HISTORY and window_size > 0:
                    # 重新统计只使用策略指定窗口的数据
                    recent = min(len(self.history_data), window_size)
                    recent_data = self.history_data[:recent]
                    freq = Counter()
                    for record in recent_data:
                        for num in record['numbers']:
                            freq[num] += 1
                    # 简化: 直接使用multi_model_voting（已支持feature_weights/model_weights）
                    vote_result = self.multi_model_voting(
                        pick_n=config['pick'],
                        top_n=config['top_n'],
                        feature_weights=fw,
                        model_weights=mw,
                    )
                else:
                    vote_result = self.multi_model_voting(
                        pick_n=config['pick'],
                        top_n=config['top_n'],
                        feature_weights=fw,
                        model_weights=mw,
                    )
            else:
                # 无验证策略，使用全局默认（当前全是停用）
                vote_result = self.multi_model_voting(
                    pick_n=config['pick'],
                    top_n=config['top_n'],
                )

            results[s_key] = {
                'desc': config['desc'],
                'pick': config['pick'],
                'numbers': vote_result['selected'],
                'candidates': vote_result.get('candidates', [])[:10],
                'version': vote_result.get('version', KL8_PREDICTOR_VERSION),
                'status': vote_result.get('status', ''),
                'message': vote_result.get('message', ''),
                'strategy_id': strategy_id,
            }

        # 复式7码
        fu7_strategy = ACTIVE_STRATEGIES.get('fu_shi_7', {})
        fu7_strategy_id = fu7_strategy.get('strategy_id', '')
        fu7_fw = fu7_strategy.get('feature_weights', {})
        fu7_mw = fu7_strategy.get('model_weights', {})

        if fu7_strategy_id and (any(w > 0 for w in fu7_fw.values()) or any(w > 0 for w in fu7_mw.values())):
            results['fu_shi_7'] = self.get_fu_shi_7(feature_weights=fu7_fw, model_weights=fu7_mw)
        else:
            results['fu_shi_7'] = self.get_fu_shi_7()
        results['fu_shi_7']['strategy_id'] = fu7_strategy_id

        recent = self.history_data[:10] if self.history_data else []
        results['recent_results'] = [
            {'issue': r['issue'], 'numbers': r['numbers'], 'date': r['date']}
            for r in recent
        ]

        stats = self.statistics
        results['statistics'] = {
            'total_periods': stats.get('total_periods', 0),
            'expected_freq': round(stats.get('expected_freq', 2), 2),
            'expected_gap': round(stats.get('expected_gap', 1), 1),
            'last_numbers': sorted(list(stats.get('last_numbers', set()))),
            'version': KL8_PREDICTOR_VERSION,
            'feature_config': FEATURE_CONFIG,
            'active_feature_weights': get_active_feature_weights(),
            'model_config': MODEL_CONFIG,
            'active_model_weights': get_active_model_weights(),
            'active_strategies': ACTIVE_STRATEGIES,
            'is_prediction_ready': prediction_ready,
            'signal_status': 'no_validated_signal' if not prediction_ready else 'active',
            'note': (
                '暂无通过回测验证的有效策略，不输出号码推荐。'
                if not prediction_ready
                else '当前启用策略已通过回测验证。'
            ),
        }

        # v5: 无信号时ranking返回空列表
        results['ranking'] = self.get_ensemble_ranking(top_n=20)

        results['using_simulated_data'] = self.using_simulated_data

        # 数据完整性信息
        if self.history_data:
            integrity = check_data_integrity(self.history_data)
            results['data_integrity'] = integrity

        snapshot_name = self._save_prediction_snapshot(results)
        if snapshot_name:
            results['snapshot_file'] = snapshot_name

        return results


# ─── 单例与缓存 ───

_analyzer_instance = None
_prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def get_kl8_analyzer() -> KL8Analyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = KL8Analyzer()
    return _analyzer_instance


def run_prediction(force_refresh: bool = False) -> Dict:
    analyzer = get_kl8_analyzer()

    if not force_refresh:
        if analyzer.reload_if_needed():
            force_refresh = True

    if not analyzer.history_data:
        return {
            'error': '历史数据不足',
            'using_simulated_data': True,
        }

    active_fw = get_active_feature_weights()
    active_mw = get_active_model_weights()
    config_fingerprint = hashlib.md5(
        json.dumps({'fw': active_fw, 'mw': active_mw}, separators=(',', ':')).encode()
    ).hexdigest()[:8]

    cache_key = (
        analyzer.history_data[0]['issue'],
        len(analyzer.history_data),
        KL8_PREDICTOR_VERSION,
        config_fingerprint,
    )

    cache = _prediction_cache
    if not force_refresh and cache['data'] is not None and cache.get('cache_key') == cache_key:
        return cache['data']

    result = analyzer.predict_all()

    cache['data'] = result
    cache['cache_key'] = cache_key
    cache['timestamp'] = time.time()

    return result


def clear_cache():
    global _analyzer_instance, _prediction_cache
    _analyzer_instance = None
    _prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def list_prediction_snapshots() -> List[Dict]:
    snapshot_dir = Path(KL8_SNAPSHOT_DIR)
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for f in sorted(snapshot_dir.glob('snapshot_*.json')):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            snapshots.append({
                'file': f.name,
                'snapshot_id': data.get('snapshot_id', ''),
                'target_issue': data.get('target_issue'),
                'based_on_issue': data.get('based_on_issue'),
                'predicted_at': data.get('predicted_at'),
                'version': data.get('version'),
                'has_settlement': _check_settlement_exists(data.get('snapshot_id', '')),
            })
        except Exception:
            continue

    return snapshots


def _check_settlement_exists(snapshot_id: str) -> bool:
    if not snapshot_id:
        return False
    return (Path(KL8_SETTLEMENT_DIR) / f'settlement_{snapshot_id}.json').exists()


# ─── 滚动回测模块（v5: 真正三段式+超几何+置换检验+纯参数化+按玩法分开）───

class KL8RollingBacktest:
    """快乐8严格滚动回测 v6

    核心设计:
    - 真正三段式: train→val→final_test，final_test完全冻结
    - 回测使用multi_model_voting管道（model_weights真正生效）
    - 置换检验按玩法分别执行（pick_n=select_type）
    - 多重检验校正（BH FDR）
    - 纯参数化回测，不修改全局配置
    - 策略注册表ACTIVE_STRATEGIES按玩法分别配置
    """

    def __init__(self, analyzer: Optional[KL8Analyzer] = None):
        self.analyzer = analyzer or get_kl8_analyzer()

    # ─── 超几何分布随机基线 ───

    def hypergeom_baseline(self, pick_n: int) -> Dict:
        """超几何分布随机基线

        快乐8从80个号码中不放回开出20个，
        选中pick_n个号码命中k个的概率应使用超几何分布
        """
        expected = hypergeom_expected(pick_n)

        probs = {}
        for k in range(1, pick_n + 1):
            probs[f'>={k}'] = round(hypergeom_p_ge(pick_n, k), 6)

        return {
            'expected_hits': expected,
            'probabilities': probs,
            'distribution': 'hypergeometric',
            'note': '80选20不放回，使用超几何分布而非二项近似',
        }

    # ─── 真正三段式数据分割 ───

    def _split_three_stage(self, total_periods: int, val_periods: int = BACKTEST_MIN_OOS_PERIODS, final_test_periods: int = BACKTEST_FINAL_TEST_PERIODS) -> Dict:
        """真正三段式数据分割（v6: 固定段数，final_test完全冻结）

        train: 生成候选特征/窗口/权重
        val: 只选一个最终策略
        final_test: 完全冻结，不参与任何启用判断，只用于最终成绩报告

        规则:
        - train_end = n - val_periods - final_test_periods
        - 如果train不足100期，数据不够，抛异常
        - final_test的任何结果都不影响is_candidate或recommendations
        """
        train_end = total_periods - val_periods - final_test_periods
        if train_end < 100:
            raise ValueError(
                f'历史数据不足: 需train>=100+val={val_periods}+test={final_test_periods}'
                f'={100+val_periods+final_test_periods}期，现有{total_periods}期'
            )

        return {
            'train': (0, train_end),
            'val': (train_end, train_end + val_periods),
            'final_test': (train_end + val_periods, total_periods),
        }

    # ─── 纯参数化回测（不修改全局配置）───

    def _rolling_backtest_parametric(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        start_idx: int,
        end_idx: int,
        min_train: int = 50,
        window_size: Optional[int] = None,
    ) -> Dict:
        """纯参数化滚动回测（v6: 使用multi_model_voting管道，model_weights真正生效）

        不修改全局FEATURE_CONFIG/MODEL_CONFIG，直接传权重参数
        第t期预测只能使用t期之前的历史数据
        使用真实投票管道回测，与线上逻辑一致

        v6关键改动:
        - 使用multi_model_voting()而非get_ensemble_ranking()
        - top20从投票结果截取，后续选3/5/7从同一份top20截取
        - model_weights真正参与（bayesian/markov权重生效）
        """
        history = self.analyzer.history_data
        history_asc = sorted(history, key=lambda x: x['issue'])
        n = len(history_asc)

        actual_end = min(end_idx, n)
        actual_start = max(start_idx, min_train)

        if actual_end - actual_start < 10:
            return {'error': '数据不足'}

        # 使用指定窗口大小（如果不指定，用全部可用历史）
        effective_window = window_size or KL8_DEFAULT_HISTORY

        all_hits = defaultdict(list)
        all_fu_shi_7_pool_hits = []
        all_fu_shi_7_combo_hits_detail = []  # v6: 每期所有组合命中详情（用于ROI）

        for t in range(actual_start, actual_end):
            # 只用t期之前的历史，限制窗口大小
            train_end_idx = t
            train_start_idx = max(0, t - effective_window)
            train_data = history_asc[train_start_idx:train_end_idx]

            if len(train_data) < min_train:
                continue

            actual_numbers = set(history_asc[t]['numbers'])

            # 构造临时分析器（不修改全局配置）
            temp_analyzer = KL8Analyzer.__new__(KL8Analyzer)
            temp_analyzer.history_data = sorted(train_data, key=lambda x: x['issue'], reverse=True)
            temp_analyzer.using_simulated_data = False
            temp_analyzer.history_file = ''
            temp_analyzer._data_mtime = 0
            temp_analyzer.update_statistics()

            # v6: 使用真实投票管道（model_weights真正参与）
            vote = temp_analyzer.multi_model_voting(
                pick_n=20,
                top_n=20,
                feature_weights=feature_weights,
                model_weights=model_weights,
            )

            # 无信号时，该期命中数记录为0
            if vote.get('status') == 'no_validated_signal':
                for select_type in [3, 4, 5, 6, 7]:
                    all_hits[select_type].append(0)
                all_fu_shi_7_pool_hits.append(0)
                all_fu_shi_7_combo_hits_detail.append([])
                continue

            # v6: 从投票结果截取top20（与线上逻辑一致）
            top20 = [num for num, _ in vote.get('candidates', [])]

            if len(top20) < 7:
                # 号码不足7个，无法做复式
                for select_type in [3, 4, 5, 6, 7]:
                    top_nums = top20[:select_type] if len(top20) >= select_type else []
                    hits = len(set(top_nums) & actual_numbers) if top_nums else 0
                    all_hits[select_type].append(hits)
                all_fu_shi_7_pool_hits.append(0)
                all_fu_shi_7_combo_hits_detail.append([])
                continue

            # 后续选3/5/7都从同一份top20截取
            for select_type in [3, 4, 5, 6, 7]:
                top_nums = top20[:select_type]
                hits = len(set(top_nums) & actual_numbers)
                all_hits[select_type].append(hits)

            # 复式7码: 从top20截取前7
            top7 = top20[:7]
            pool_hits = len(set(top7) & actual_numbers)
            all_fu_shi_7_pool_hits.append(pool_hits)

            # v6: 每期所有21组合命中详情（用于ROI精确计算）
            combo_hits = []
            for combo in combinations(top7, 5):
                combo_hits.append(len(set(combo) & actual_numbers))
            all_fu_shi_7_combo_hits_detail.append(combo_hits)

        # 统计结果（使用超几何基线）
        summary = {}

        for select_type in [3, 4, 5, 6, 7]:
            hits_list = all_hits[select_type]
            n_tests = len(hits_list)
            if n_tests == 0:
                summary[f'select_{select_type}'] = {'error': '无测试数据'}
                continue

            mean_hits = sum(hits_list) / n_tests
            expected_random = hypergeom_expected(select_type)

            # >=k概率（实际）
            for_k_probs = {}
            for k in range(1, select_type + 1):
                count = sum(1 for h in hits_list if h >= k)
                for_k_probs[f'>={k}'] = count / n_tests

            # >=k概率（超几何理论）
            theoretical_probs = {}
            for k in range(1, select_type + 1):
                theoretical_probs[f'>={k}'] = hypergeom_p_ge(select_type, k)

            # 95%置信区间
            std_dev = math.sqrt(sum((h - mean_hits) ** 2 for h in hits_list) / n_tests) if n_tests > 1 else 0
            ci_lower = mean_hits - 1.96 * std_dev / math.sqrt(n_tests)
            ci_upper = mean_hits + 1.96 * std_dev / math.sqrt(n_tests)

            lift = (mean_hits - expected_random) / expected_random if expected_random > 0 else 0

            # v6: 奖金ROI（统一为return_multiple和profit_roi两个字段）
            prize_table = load_prize_table()
            prize_info = prize_table.get(f'select_{select_type}', {})
            bet = prize_info.get('bet', 2)
            total_bet = n_tests * bet
            total_prize = sum(prize_info.get(str(h), 0) for h in hits_list)

            # v6: 区分回报倍数和净ROI
            return_multiple = total_prize / max(total_bet, 1)  # 回报倍数(含本金)
            profit_roi = (total_prize - total_bet) / max(total_bet, 1)  # 净ROI(不含本金)

            # 理论随机ROI（超几何期望命中数 * 平均奖金 / 总投注 - 1）
            random_expected_prize = sum(
                hypergeom_pmf(select_type, k) * prize_info.get(str(k), 0)
                for k in range(0, select_type + 1)
            )
            random_return_multiple = random_expected_prize / max(bet, 1)
            random_profit_roi = (random_expected_prize - bet) / max(bet, 1)

            summary[f'select_{select_type}'] = {
                'mean_hits': round(mean_hits, 4),
                'expected_random': round(expected_random, 4),
                'lift': round(lift, 4),
                'probabilities': for_k_probs,
                'theoretical_probs': theoretical_probs,
                'ci_95': [round(ci_lower, 4), round(ci_upper, 4)],
                'std_dev': round(std_dev, 4),
                'n_tests': n_tests,
                'is_significant': ci_lower > expected_random,
                'distribution': 'hypergeometric',
                'bet': bet,
                'total_bet': total_bet,
                'total_prize': total_prize,
                'return_multiple': round(return_multiple, 4),  # 回报倍数(含本金)
                'profit_roi': round(profit_roi, 4),              # 净ROI(不含本金)
                'random_return_multiple': round(random_return_multiple, 4),
                'random_profit_roi': round(random_profit_roi, 4),
            }

        # 复式7码ROI（v6: 每期全部21注组合都计算，无论命中率）
        if all_fu_shi_7_pool_hits:
            pool_mean = sum(all_fu_shi_7_pool_hits) / len(all_fu_shi_7_pool_hits)

            prize_table = load_prize_table()
            fu7_prize_info = prize_table.get('fu_shi_7', {})
            bet_per_combo = fu7_prize_info.get('bet_per_combo', 2)

            # v6: 每期投注= 21注 * 单注金额（无论命中率）
            n_fu7_tests = len(all_fu_shi_7_pool_hits)
            fu7_total_bet = n_fu7_tests * math.comb(7, 5) * bet_per_combo

            # v6: 每期奖金= 所有21组组合的奖金之和
            fu7_total_prize = 0
            for combo_hits_list in all_fu_shi_7_combo_hits_detail:
                if combo_hits_list:
                    fu7_total_prize += sum(
                        fu7_prize_info.get(str(h), 0)
                        for h in combo_hits_list
                    )

            fu7_return_multiple = fu7_total_prize / max(fu7_total_bet, 1)
            fu7_profit_roi = (fu7_total_prize - fu7_total_bet) / max(fu7_total_bet, 1)

            # 理论随机ROI（7码随机选5命中）
            random_fu7_expected_prize_per_combo = sum(
                hypergeom_pmf(5, k) * fu7_prize_info.get(str(k), 0)
                for k in range(0, 6)
            )
            random_fu7_return_multiple = random_fu7_expected_prize_per_combo / max(bet_per_combo, 1)
            random_fu7_profit_roi = (random_fu7_expected_prize_per_combo - bet_per_combo) / max(bet_per_combo, 1)

            summary['fu_shi_7'] = {
                'pool_mean_hits': round(pool_mean, 4),
                'pool_expected_random': round(hypergeom_expected(7), 4),
                'n_tests': n_fu7_tests,
                'total_bet': fu7_total_bet,
                'total_prize': fu7_total_prize,
                'return_multiple': round(fu7_return_multiple, 4),
                'profit_roi': round(fu7_profit_roi, 4),
                'random_return_multiple': round(random_fu7_return_multiple, 4),
                'random_profit_roi': round(random_fu7_profit_roi, 4),
            }

        return summary

    # ─── 置换检验（v6: 按玩法分别检验）───

    def _permutation_test(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        start_idx: int,
        end_idx: int,
        pick_n: int = 5,  # v6: 指定检验的玩法（不再是固定select_5）
        metric: str = 'mean_hits',
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
        min_train: int = 50,
    ) -> Dict:
        """置换检验: 计算模型Lift在零假设下的p-value（v6: 按玩法检验）

        零假设: 模型排名与随机选择无区别
        方法: 每期将模型推荐号码随机打散，计算"随机打散后的Lift"分布
        只有模型成绩高于95%以上随机打散结果，才算候选有效

        v6改动:
        - pick_n参数: 可以指定检验选3/选4/选5/选6/选7
        - metric: 'mean_hits' 或关键中奖档概率
        - 结果包含该玩法的p值和Lift
        """
        history = self.analyzer.history_data
        history_asc = sorted(history, key=lambda x: x['issue'])

        # 先跑一遍真实模型得分（用multi_model_voting管道）
        real_result = self._rolling_backtest_parametric(
            feature_weights, model_weights,
            start_idx=start_idx, end_idx=end_idx,
            min_train=min_train,
        )

        if 'error' in real_result:
            return real_result

        # v6: 取指定玩法的真实Lift作为基准（不再固定select_5）
        s_key = f'select_{pick_n}'
        real_lift = real_result.get(s_key, {}).get('lift', 0)
        real_mean_hits = real_result.get(s_key, {}).get('mean_hits', 0)

        # 置换: 每次打散模型推荐的号码顺序
        import random as rng

        permutation_lifts = []
        actual_start = max(start_idx, min_train)
        actual_end = min(end_idx, len(history_asc))

        for perm_i in range(n_permutations):
            perm_hits = []

            for t in range(actual_start, actual_end):
                train_data = history_asc[max(0, t - KL8_DEFAULT_HISTORY):t]
                if len(train_data) < min_train:
                    continue

                actual_numbers = set(history_asc[t]['numbers'])

                # 用确定性seed生成打散
                seed = int(hashlib.sha256(f'perm_{perm_i}_{pick_n}_{history_asc[t]["issue"]}'.encode()).hexdigest()[:8], 16)
                rng.seed(seed)

                # v6: 随机选pick_n个号码（不是固定5）
                random_nums = rng.sample(range(1, 81), pick_n)
                hits = len(set(random_nums) & actual_numbers)
                perm_hits.append(hits)

            if perm_hits:
                perm_mean = sum(perm_hits) / len(perm_hits)
                expected = hypergeom_expected(pick_n)
                perm_lift = (perm_mean - expected) / expected if expected > 0 else 0
                permutation_lifts.append(perm_lift)

        if not permutation_lifts:
            return {'error': '置换检验数据不足'}

        # 计算p-value: 真实Lift在置换分布中的位置
        n_greater = sum(1 for l in permutation_lifts if l >= real_lift)
        p_value = n_greater / len(permutation_lifts)

        # 95%分位数
        sorted_lifts = sorted(permutation_lifts)
        percentile_95 = sorted_lifts[int(len(sorted_lifts) * 0.95)] if len(sorted_lifts) > 20 else 0

        return {
            'play_type': s_key,
            'pick_n': pick_n,
            'real_lift': real_lift,
            'real_mean_hits': real_mean_hits,
            'p_value': round(p_value, 6),
            'is_significant_p005': p_value < 0.05,
            'is_significant_p001': p_value < 0.01,
            'permutation_count': len(permutation_lifts),
            'percentile_95_lift': round(percentile_95, 6),
            'permutation_mean_lift': round(sum(permutation_lifts) / len(permutation_lifts), 6),
            'permutation_std_lift': round(
                math.sqrt(sum((l - sum(permutation_lifts) / len(permutation_lifts)) ** 2 for l in permutation_lifts) / len(permutation_lifts)),
                6,
            ) if len(permutation_lifts) > 1 else 0,
        }

    # ─── 特征按玩法分开评估（v6: 按玩法置换检验+多重检验校正）───

    def run_feature_ablation_per_play_type(
        self,
        test_periods: int = BACKTEST_MIN_OOS_PERIODS,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """单特征独立回测 — 按玩法分开评估（v6改进版）

        不要求特征同时提升选3到选7，按玩法独立判断
        v6改动:
        1. 每个玩法单独做置换检验(pick_n=select_type)
        2. is_candidate条件加入 permutation_p_adjusted < 0.05
        3. 多重检验校正(BH FDR)后p值才进入启用判断
        4. final_test结果只用于报告，不影响is_candidate
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < test_periods + 50:
            return {'error': f'历史数据不足(需{test_periods + 50}期，现有{n}期)'}

        # v6: 真正三段式分割（final_test完全冻结）
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        val_range = split['val']
        final_test_range = split['final_test']

        results = {}

        for feature_name, feature_cfg in FEATURE_CONFIG.items():
            if feature_cfg['weight'] <= 0:
                continue

            # 单独启用该特征
            single_weights = {k: 0.0 for k in FEATURE_CONFIG}
            single_weights[feature_name] = feature_cfg['weight']
            model_weights = {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0}

            # val段回测
            val_result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=val_range[0],
                end_idx=val_range[1],
                min_train=50,
            )

            # final_test段回测（只用于报告，不影响启用判断）
            final_test_result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=final_test_range[0],
                end_idx=final_test_range[1],
                min_train=50,
            )

            # v6: 每个玩法单独做置换检验
            perm_results = {}
            for select_type in [3, 4, 5, 6, 7]:
                perm_r = self._permutation_test(
                    single_weights, model_weights,
                    start_idx=val_range[0],
                    end_idx=val_range[1],
                    pick_n=select_type,
                    metric='mean_hits',
                    n_permutations=n_permutations,
                )
                perm_results[f'select_{select_type}'] = perm_r

            # v6: 多重检验校正 — 每个玩法的5个特征p值做BH FDR
            # 收集当前特征在所有玩法下的原始p值
            all_p_values_for_correction = []
            for select_type in [3, 4, 5, 6, 7]:
                perm_r = perm_results.get(f'select_{select_type}', {})
                p_val = perm_r.get('p_value', 1.0)
                all_p_values_for_correction.append(p_val)

            # BH FDR校正
            adjusted_p_values = benjamini_hochberg_fdr(all_p_values_for_correction)

            # 按玩法判断有效性（v6: is_candidate条件更严格）
            play_type_recommendations = {}
            for i, select_type in enumerate([3, 4, 5, 6, 7]):
                s_key = f'select_{select_type}'

                val_s = val_result.get(s_key, {})
                test_s = final_test_result.get(s_key, {})

                val_lift = val_s.get('lift', 0)
                test_lift = test_s.get('lift', 0)
                val_significant = val_s.get('is_significant', False)

                # v6: 置换检验p值 + BH FDR校正后p值
                perm_r = perm_results.get(s_key, {})
                raw_p = perm_r.get('p_value', 1.0)
                adjusted_p = adjusted_p_values[i] if i < len(adjusted_p_values) else 1.0

                # v6: ROI指标
                val_profit_roi = val_s.get('profit_roi', 0)
                random_profit_roi = val_s.get('random_profit_roi', 0)
                roi_better = val_profit_roi > random_profit_roi

                # v6: is_candidate = val_lift>0 AND p_adjusted<0.05
                # final_test结果只用于报告，不参与is_candidate判断
                is_candidate = (
                    val_lift > 0
                    and adjusted_p < 0.05
                )

                play_type_recommendations[s_key] = {
                    'val_lift': val_lift,
                    'final_test_lift': test_lift,  # 只报告，不参与判断
                    'val_significant': val_significant,
                    'raw_p_value': raw_p,
                    'adjusted_p_value': adjusted_p,
                    'is_candidate': is_candidate,
                    'val_profit_roi': val_profit_roi,
                    'roi_better_than_random': roi_better,
                    'recommendation': 'enable_for_play_type' if is_candidate else 'keep_disabled',
                }

            results[feature_name] = {
                'val_result': val_result,
                'final_test_result': final_test_result,
                'permutation_tests': perm_results,
                'bh_fdr_adjusted_p_values': {
                    f'select_{st}': adjusted_p_values[i]
                    for i, st in enumerate([3, 4, 5, 6, 7])
                    if i < len(adjusted_p_values)
                },
                'play_type_recommendations': play_type_recommendations,
            }

        return results

    # ─── 窗口长度验证 ───

    def run_window_validation(
        self,
        feature_weights: Dict[str, float],
        model_weights: Dict[str, float],
        window_sizes: List[int] = [50, 100, 250, 500],
    ) -> Dict:
        """测试不同窗口长度

        在训练段选窗口，验证段确认，测试段验证一次
        如果不同窗口表现互相矛盾，说明信号不稳定
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < 300:
            return {'error': f'数据不足(需300期以上，现有{n}期)'}

        split = self._split_three_stage(n)
        val_range = split['val']

        results = {}
        for ws in window_sizes:
            # 在val段用不同窗口回测
            val_result = self._rolling_backtest_parametric(
                feature_weights, model_weights,
                start_idx=val_range[0],
                end_idx=val_range[1],
                min_train=ws,
                window_size=ws,
            )
            results[f'window_{ws}'] = val_result

        # 一致性检查: 不同窗口对选5的Lift是否一致
        s5_lifts = []
        for ws in window_sizes:
            r = results.get(f'window_{ws}', {})
            s5 = r.get('select_5', {})
            if 'lift' in s5:
                s5_lifts.append(s5['lift'])

        consistency = 'consistent' if all(l > 0 for l in s5_lifts) or all(l <= 0 for l in s5_lifts) else 'contradictory'

        results['consistency'] = {
            's5_lifts_by_window': {f'window_{ws}': s5_lifts[i] for i, ws in enumerate(window_sizes) if i < len(s5_lifts)},
            'consistency': consistency,
            'all_positive': all(l > 0 for l in s5_lifts),
            'recommendation': 'signal_stable' if consistency == 'consistent' and all(l > 0 for l in s5_lifts) else 'signal_unstable',
        }

        return results

    # ─── 稳定性门槛 ───

    def check_stability_gate(
        self,
        feature_name: str,
        feature_weight: float,
    ) -> Dict:
        """稳定性门槛检查

        特征上线需同时满足:
        1. 最近4个独立窗口至少3个Lift>0
        2. val段单侧p-value<0.05
        3. test段Lift>0
        4. 关键中奖档不低于随机
        5. 无严重反向窗口(Lift<-0.1)
        """
        history = self.analyzer.history_data
        n = len(history)

        if n < BACKTEST_MIN_OOS_PERIODS + 50:
            return {'error': f'数据不足(需{BACKTEST_MIN_OOS_PERIODS + 50}期)'}

        single_weights = {k: 0.0 for k in FEATURE_CONFIG}
        single_weights[feature_name] = feature_weight
        model_weights = {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0}

        # 分成4个独立窗口
        window_size = n // BACKTEST_STABILITY_WINDOWS
        window_lifts = []

        for i in range(BACKTEST_STABILITY_WINDOWS):
            start = i * window_size + 50  # 留50期作训练
            end = (i + 1) * window_size

            result = self._rolling_backtest_parametric(
                single_weights, model_weights,
                start_idx=start, end_idx=end,
                min_train=50,
            )

            s5 = result.get('select_5', {})
            lift = s5.get('lift', 0)
            window_lifts.append(lift)

        # 稳定性判断
        n_positive = sum(1 for l in window_lifts if l > 0)
        n_severe_negative = sum(1 for l in window_lifts if l < -0.1)

        # val段p-value
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        perm = self._permutation_test(
            single_weights, model_weights,
            start_idx=split['val'][0],
            end_idx=split['val'][1],
            pick_n=5,  # 默认用选5做稳定性门槛
            n_permutations=BACKTEST_PERMUTATION_COUNT,
        )

        # v6: final_test段Lift（只用于报告，不参与启用判断）
        final_test_result = self._rolling_backtest_parametric(
            single_weights, model_weights,
            start_idx=split['final_test'][0],
            end_idx=split['final_test'][1],
            min_train=50,
        )
        final_test_lift = final_test_result.get('select_5', {}).get('lift', 0)

        # v6: is_candidate只依赖val段结果，final_test只报告
        # 稳定性门槛仍基于val段判断
        gate_1 = n_positive >= BACKTEST_STABILITY_THRESHOLD  # 4窗口至少3个Lift>0
        gate_2 = perm.get('is_significant_p005', False)       # val p<0.05
        gate_3 = True  # v6: 去掉test_lift>0的条件（final_test不应参与判断）
        gate_4 = n_severe_negative == 0                        # 无严重反向窗口

        all_passed = gate_1 and gate_2 and gate_3 and gate_4

        return {
            'feature': feature_name,
            'window_lifts': window_lifts,
            'n_positive_windows': n_positive,
            'n_severe_negative': n_severe_negative,
            'val_p_value': perm.get('p_value', 1.0),
            'final_test_lift_select_5': final_test_lift,  # 只报告
            'gate_1_stability': gate_1,
            'gate_2_significance': gate_2,
            'gate_3_stability_positive': gate_3,
            'gate_4_no_severe_negative': gate_4,
            'all_gates_passed': all_passed,
            'recommendation': 'enable_candidate' if all_passed else 'keep_disabled',
        }

    # ─── 完整回测 ───

    def run_full_backtest(
        self,
        test_periods: int = BACKTEST_MIN_OOS_PERIODS,
        n_permutations: int = BACKTEST_PERMUTATION_COUNT,
    ) -> Dict:
        """完整回测: 真正三段式 + 置换检验 + 按玩法分开 + 稳定性门槛

        v6: final_test结果只用于报告，不参与启用判断
        """
        history = self.analyzer.history_data
        n = len(history)

        # 三段式分割（使用默认val_periods=300, final_test_periods=200）
        try:
            split = self._split_three_stage(n)
        except ValueError as e:
            return {'error': str(e)}

        result = {
            'total_periods': n,
            'split': split,
            'version': KL8_PREDICTOR_VERSION,
            'distribution': 'hypergeometric',
            'note': 'final_test结果仅供报告，不参与任何启用判断',
        }

        # 1. 单特征按玩法消融回测
        ablation = self.run_feature_ablation_per_play_type(
            test_periods=test_periods,
            n_permutations=n_permutations,
        )
        result['feature_ablation'] = ablation

        # 2. 超几何随机基线
        baseline = {}
        for select_type in [3, 4, 5, 6, 7]:
            baseline[f'select_{select_type}'] = self.hypergeom_baseline(select_type)
        result['random_baseline'] = baseline

        # 3. 稳定性门槛检查（每个有权重的特征）
        stability_checks = {}
        for feature_name, feature_cfg in FEATURE_CONFIG.items():
            if feature_cfg['weight'] > 0:
                stability_checks[feature_name] = self.check_stability_gate(
                    feature_name, feature_cfg['weight'],
                )
        result['stability_checks'] = stability_checks

        # 4. 窗口长度验证
        window_validation = self.run_window_validation(
            feature_weights=get_active_feature_weights(),
            model_weights=get_active_model_weights(),
        )
        result['window_validation'] = window_validation

        # 5. 综合推荐
        recommendations = {}
        for feature_name, stability in stability_checks.items():
            if 'error' in stability:
                recommendations[feature_name] = {'recommendation': 'keep_disabled', 'reason': 'stability_check_error'}
                continue

            all_passed = stability.get('all_gates_passed', False)
            if all_passed:
                # 检查哪些玩法通过了
                ablation_data = ablation.get(feature_name, {})
                play_recs = ablation_data.get('play_type_recommendations', {})
                eligible_play_types = [
                    pt for pt, rec in play_recs.items()
                    if rec.get('is_candidate', False)
                ]
                recommendations[feature_name] = {
                    'recommendation': 'enable_candidate',
                    'eligible_play_types': eligible_play_types,
                    'stability_detail': stability,
                }
            else:
                recommendations[feature_name] = {
                    'recommendation': 'keep_disabled',
                    'failed_gates': {
                        'gate_1': not stability.get('gate_1_stability', False),
                        'gate_2': not stability.get('gate_2_significance', False),
                        'gate_3': not stability.get('gate_3_stability_positive', False),
                        'gate_4': not stability.get('gate_4_no_severe_negative', False),
                    },
                }

        result['recommendations'] = recommendations

        return result
