# 福彩 3D 预测器 V7.0 - 多模型集成版（CatBoost/XGBoost/LightGBM + 增强特征 + 分层采样 + early stopping）
# Python 3.10+
"""
基于多模型集成的福彩 3D 预测系统（V7.0 增强版）

主要改进（v7 vs v6）：
1. 更深的模型架构（depth 2→4, iterations 30→100），捕获复杂非线性关系
2. Early stopping 防止过拟合
3. 二阶马尔可夫转移特征（捕获更长程的模式依赖）
4. 分层负采样（按和值分组，确保负例分布均匀）
5. 滚动窗口趋势特征（和值/跨度趋势、数字间隔统计）
6. 数字冷热等级特征
7. 形态连续趋势特征
8. 更大的训练窗口（120→200期）

特征工程（54维 vs 原36维）：
  - 每个数字的热度得分（3个数字的各自热度）
  - 每个位置的热度得分
  - 每个位置的一阶+二阶马尔可夫转移概率
  - 数字的遗漏值
  - 数字间隔统计（期望回补间隔、出现次数）
  - 数字冷热等级
  - 和值、跨度、和值偏离度、跨度偏离度
  - 奇偶比、大小比、012路匹配度等组合特征
  - 与上期/上两期号码的相似度
  - 形态趋势特征（连续次数、切换标记）

训练策略：
  - 正例：历史每一期的开奖号码
  - 负例：每期分层抽取 100 个未开出的号码
  - 时序验证：前 80% 期训练，后 20% 期验证（按期切分）
  - Early stopping 防止过拟合
  - 多模型集成：CatBoost > XGBoost > LightGBM > 纯 Python

预测：
  - 对所有 1000 个直选组合预测概率
  - 使用动态权重集成多模型预测
  - 取 Top K 作为推荐
"""
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations, product
from ..common.logger import setup_logger
from ..common.data_cache import cached_fetch
from ..common import kv_store

log = setup_logger('lottery3d_ml')

# 尝试导入机器学习库
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log.info("3D-ML 后端可用性: catboost=%s xgboost=%s lightgbm=%s sklearn=%s",
         HAS_CATBOOST, HAS_XGBOOST, HAS_LIGHTGBM, HAS_SKLEARN)

# 在整个进程生命周期内，对「ML 后端不可用→降级」只告警一次，避免每次预测都刷屏。
_ML_FALLBACK_WARNED = False

URL = "https://www.8300.cn/kjhhis/3/2000.html"  # 与规则模型统一拉取约 2000 期历史

# 模型参数（v7优化版：更深模型、更大窗口、更丰富特征）
BACKTEST_TRIALS = 60  # 增加回测期数，更稳定评估模型表现
TRAIN_RATIO = 0.8  # 时序划分比例
NEGATIVE_SAMPLES_PER_PERIOD = 100  # 每期负例采样数（增加覆盖，分层采样降低偏差）
TOP_K = 15  # 推荐注数
FEATURE_SUBSET_RATIO = 0.85  # 特征选择保留比例（略提高，保留更多信号）
MIN_VARIANCE = 0.0005  # 方差过滤阈值（放宽，保留弱信号特征）
TRAINING_WINDOW = 200  # 滚动训练样本窗口（增加至200期，捕获更长周期模式）
FEATURE_HISTORY_WINDOW = 100  # 特征工程历史窗口（增加至100期）

# 时间衰减训练参数（更激进的新近期权重）
TIME_DECAY_RECENT = 30  # 最近 30 期
TIME_DECAY_RECENT_WEIGHT = 2.0  # 最近 30 期权重（提升至2.0，更重视近期模式）
TIME_DECAY_MID = 60  # 最近 60 期
TIME_DECAY_MID_WEIGHT = 1.4  # 最近 60 期权重
TIME_DECAY_OLD_WEIGHT = 1.0  # 历史数据权重


def _native_number(x):
    """numpy 标量 → Python int/float，便于 JSON 序列化"""
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, float):
        return float(x)
    return x


def _fetch_data_internal(url=URL):
    """内部数据抓取函数"""
    log.debug('fetch 3D-ML data')
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    compact = re.sub(r"\s+", " ", html)
    pattern = re.compile(
        r'<td>(\d{7})期</td>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>'
        r'\s*<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>\s*'
        r'<span\s+class="ball">(\d)</span>'
    )
    rows = pattern.findall(compact)
    data = [(pid, dt, (int(a), int(b), int(c))) for pid, dt, a, b, c in rows]
    data.reverse()
    return data


def fetch_data(url=URL, force_refresh=False):
    """获取历史开奖数据（带缓存，每天只抓取一次）"""
    return cached_fetch('lottery3d_ml', lambda: _fetch_data_internal(url), force_refresh)


def miss_value(numbers, digit, position=None):
    """计算遗漏值"""
    for i in range(len(numbers) - 1, -1, -1):
        n = numbers[i]
        if position is None:
            if digit in n:
                return len(numbers) - 1 - i
        elif n[position] == digit:
            return len(numbers) - 1 - i
    return len(numbers)


def neighbor(d):
    """返回邻号"""
    return {(d - 1) % 10, (d + 1) % 10}


def road(d):
    """返回 012 路"""
    return d % 3


def exp_weighted_counts(series, decay=0.96):
    """指数加权计数"""
    cnt = Counter()   
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def build_markov(numbers, position):
    """构建马尔可夫转移矩阵"""
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        a, b = numbers[i][position], numbers[i + 1][position]
        trans[a][b] += 1
    return trans


def markov_prob_smoothed(row, states, alpha=1.0):
    """转移概率（拉普拉斯平滑）"""
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def odd_even_key(triple):
    """奇偶比"""
    odds = sum(1 for d in triple if d % 2 == 1)
    return odds, 3 - odds


def big_small_key(triple):
    """大小比（0-4 小，5-9 大）"""
    big = sum(1 for d in triple if d >= 5)
    return big, 3 - big


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def position_repeat_count(triple, last_draw):
    """与上期同位置重复个数"""
    return sum(1 for i in range(3) if triple[i] == last_draw[i])


def digit_overlap_count(triple, last_draw):
    """任意位置相同数字个数（集合交集）"""
    return len(set(triple) & set(last_draw))


def classify_form(triple):
    """分类号码形态：zu6（组六/3个不同）、zu3（组三/2个相同）、baozi（豹子/3个相同）"""
    unique = len(set(triple))
    if unique == 3:
        return "zu6"
    elif unique == 2:
        return "zu3"
    else:
        return "baozi"


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, numbers, window=100):
        self.numbers = numbers
        self.window = window
        self.recent = numbers[-window:] if len(numbers) > window else list(numbers)
        self.last_draw = numbers[-1] if numbers else None
        self.last_two = numbers[-2] if len(numbers) >= 2 else None
        
        # 预计算统计量
        self._precompute()
    
    def _precompute(self):
        """预计算常用统计量"""
        # 全局热度（指数加权）
        self.freq_global = exp_weighted_counts(
            [d for n in self.recent for d in n]
        )
        self.total_global = sum(self.freq_global.values()) or 1.0
        
        # 分位热度
        self.freq_pos = []
        self.total_pos = []
        for pos in range(3):
            freq = exp_weighted_counts([n[pos] for n in self.recent])
            self.freq_pos.append(freq)
            self.total_pos.append(sum(freq.values()) or 1.0)
        
        # 马尔可夫转移（一阶 + 二阶）
        self.markov_trans = []
        self.markov2_trans = []
        for pos in range(3):
            trans = build_markov(self.numbers, pos)
            self.markov_trans.append(trans)
            if len(self.numbers) >= 3:
                trans2 = defaultdict(Counter)
                for i in range(len(self.numbers) - 2):
                    a, b, c = self.numbers[i][pos], self.numbers[i + 1][pos], self.numbers[i + 2][pos]
                    trans2[(a, b)][c] += 1
                self.markov2_trans.append(trans2)
            else:
                self.markov2_trans.append(defaultdict(Counter))
        
        # 遗漏值
        self.miss_global = {d: miss_value(self.numbers, d) for d in range(10)}
        self.miss_pos = [
            {d: miss_value(self.numbers, d, position=pos) for d in range(10)}
            for pos in range(3)
        ]
        
        # 最近N期数字出现间隔统计
        self.interval_stats = {}
        for d in range(10):
            intervals = []
            last_seen = None
            for i, n in enumerate(self.recent):
                if d in n:
                    if last_seen is not None:
                        intervals.append(i - last_seen)
                    last_seen = i
            if intervals:
                self.interval_stats[d] = {
                    "mean": sum(intervals) / len(intervals),
                    "min": min(intervals),
                    "max": max(intervals),
                    "count": len(intervals),
                }
            else:
                self.interval_stats[d] = {"mean": len(self.recent), "min": len(self.recent), "max": len(self.recent), "count": 0}
        
        # 形态统计
        self.oe_freq = Counter()
        self.bs_freq = Counter()
        self.forms = []  # 形态序列
        self.form_streaks = []  # 连续出现次数
        current_form = None
        current_streak = 0
        for n in self.recent:
            self.oe_freq[odd_even_key(n)] += 1
            self.bs_freq[big_small_key(n)] += 1
            f = classify_form(n)
            self.forms.append(f)
            if f == current_form:
                current_streak += 1
            else:
                if current_form:
                    self.form_streaks.append((current_form, current_streak))
                current_form = f
                current_streak = 1
        if current_form:
            self.form_streaks.append((current_form, current_streak))
        
        # 连号频率
        self.consec_count = sum(1 for n in self.recent if has_consecutive_digits(*n))
        self.consec_rate = self.consec_count / len(self.recent) if self.recent else 0.0
        
        # 和值跨度统计
        self.sums = [sum(n) for n in self.recent]
        self.spans = [max(n) - min(n) for n in self.recent]
        self.sum_freq = Counter(self.sums)
        self.span_freq = Counter(self.spans)
        
        # 和值趋势统计（最近20期的移动平均和标准差）
        if len(self.sums) >= 20:
            recent_sums = self.sums[-20:]
            self.sum_mean_20 = sum(recent_sums) / 20
            self.sum_std_20 = (sum((s - self.sum_mean_20) ** 2 for s in recent_sums) / 20) ** 0.5
            self.sum_trend = sum(
                (i - 9.5) * (recent_sums[i] - self.sum_mean_20) for i in range(20)
            ) / sum((i - 9.5) ** 2 for i in range(20)) if self.sum_std_20 > 0 else 0.0
        else:
            self.sum_mean_20 = sum(self.sums) / len(self.sums) if self.sums else 13.5
            self.sum_std_20 = 1.0
            self.sum_trend = 0.0
        
        # 跨度趋势
        if len(self.spans) >= 20:
            recent_spans = self.spans[-20:]
            self.span_mean_20 = sum(recent_spans) / 20
            self.span_std_20 = (sum((s - self.span_mean_20) ** 2 for s in recent_spans) / 20) ** 0.5
        else:
            self.span_mean_20 = sum(self.spans) / len(self.spans) if self.spans else 4.5
            self.span_std_20 = 1.0
        
        # 数字冷热等级（基于近期出现频率）
        total_appearances = sum(self.freq_global.values()) or 1
        avg_appearance = total_appearances / 10
        self.digit_tier = {}
        for d in range(10):
            f = self.freq_global.get(d, 0)
            if f >= avg_appearance * 1.3:
                self.digit_tier[d] = 2  # 热号
            elif f >= avg_appearance * 0.7:
                self.digit_tier[d] = 1  # 温号
            else:
                self.digit_tier[d] = 0  # 冷号
        
        # 012路频率统计
        self.road_freq = Counter()
        for n in self.recent:
            for d in n:
                self.road_freq[d % 3] += 1
        self.road_total = sum(self.road_freq.values()) or 1
    
    def build_features(self, a, b, c):
        """为单个直选组合 (a,b,c) 构造特征向量（v7增强版：36维→54维）"""
        features = []
        triple = (a, b, c)
        
        # 1. 每个数字的全局热度得分（3个特征）
        for d in triple:
            features.append(self.freq_global.get(d, 0) / self.total_global)
        
        # 2. 每个位置的热度得分（3个特征）
        for pos, d in enumerate(triple):
            features.append(self.freq_pos[pos].get(d, 0) / self.total_pos[pos])
        
        # 3. 每个位置的一阶马尔可夫转移概率（3个特征）
        for pos, d in enumerate(triple):
            prev_d = self.last_draw[pos] if self.last_draw else 0
            trans = self.markov_trans[pos].get(prev_d, Counter())
            prob = markov_prob_smoothed(trans, range(10)).get(d, 0.1)
            features.append(prob)
        
        # 4. 每个位置的二阶马尔可夫转移概率（3个特征）★ 新增
        for pos, d in enumerate(triple):
            if self.last_two and self.last_draw:
                key = (self.last_two[pos], self.last_draw[pos])
                trans2 = self.markov2_trans[pos].get(key, Counter())
                prob2 = markov_prob_smoothed(trans2, range(10)).get(d, 0.1)
            else:
                prob2 = 0.1
            features.append(prob2)
        
        # 5. 每个数字的遗漏值（3个特征）
        for d in triple:
            features.append(self.miss_global.get(d, 0))
        
        # 6. 每个位置的遗漏值（3个特征）
        for pos, d in enumerate(triple):
            features.append(self.miss_pos[pos].get(d, 0))
        
        # 7. 数字间隔统计（每个数字3个特征：均值间隔、最小间隔、出现次数）★ 新增
        for d in triple:
            stats = self.interval_stats.get(d, {"mean": 100, "min": 100, "max": 100, "count": 0})
            features.append(max(0, stats["mean"] - self.miss_global.get(d, 0)))  # 期望回补间隔
            features.append(stats["count"])  # 窗口内出现次数
        
        # 8. 数字冷热等级（3个特征）★ 新增
        for d in triple:
            features.append(self.digit_tier.get(d, 1))
        
        # 9. 和值特征（2个特征）
        s = a + b + c
        features.append(s)  # 和值
        features.append(self.sum_freq.get(s, 0))  # 和值频次
        
        # 10. 和值偏离度（和值与近期趋势的偏离）★ 新增
        sum_deviation = abs(s - self.sum_mean_20) / max(self.sum_std_20, 1.0)
        features.append(sum_deviation)
        
        # 11. 跨度特征（2个特征）
        span = max(a, b, c) - min(a, b, c)
        features.append(span)  # 跨度
        features.append(self.span_freq.get(span, 0))  # 跨度频次
        
        # 12. 跨度偏离度 ★ 新增
        span_deviation = abs(span - self.span_mean_20) / max(self.span_std_20, 1.0)
        features.append(span_deviation)
        
        # 13. 奇偶比（2个特征）
        oe = odd_even_key(triple)
        features.append(oe[0])  # 奇数个数
        features.append(self.oe_freq.get(oe, 0))  # 奇偶比频次
        
        # 14. 大小比（2个特征）
        bs = big_small_key(triple)
        features.append(bs[0])  # 大数个数
        features.append(self.bs_freq.get(bs, 0))  # 大小比频次
        
        # 15. 连号特征（1个特征）
        features.append(1 if has_consecutive_digits(a, b, c) else 0)
        
        # 16. 与上期的相似度（3个特征）
        if self.last_draw:
            features.append(position_repeat_count(triple, self.last_draw))  # 同位重复个数
            features.append(digit_overlap_count(triple, self.last_draw))  # 集合交集大小
            # 跨位复用：数字在上期出现但不在同位
            features.append(sum(
                1 for i in range(3)
                if triple[i] in self.last_draw and triple[i] != self.last_draw[i]
            ))
        else:
            features.extend([0, 0, 0])
        
        # 17. 与上两期的关系 ★ 新增
        if self.last_two:
            features.append(digit_overlap_count(triple, self.last_two))
        else:
            features.append(0)
        
        # 18. 012 路特征（2个特征）
        roads = [road(d) for d in triple]
        features.append(sum(roads))  # 012 路和
        features.append(max(Counter(roads).values()))  # 最集中路数
        
        # 19. 012路匹配度（与近期路数分布的匹配度）★ 新增
        road_match = sum(self.road_freq.get(r % 3, 0) / self.road_total for r in triple) / 3
        features.append(road_match)
        
        # 20. 邻号特征（1个特征）
        if self.last_draw:
            nb = set()
            for d in self.last_draw:
                nb.update(neighbor(d))
            features.append(len(set(triple) & nb))  # 与上期邻号重叠数
        else:
            features.append(0)
        
        # 21. 豹子/组三/组六特征（2个特征）
        unique_count = len(set(triple))
        features.append(1 if unique_count == 1 else 0)  # 豹子
        features.append(1 if unique_count == 2 else 0)  # 组三
        
        # 22. 形态匹配度（与当前形态趋势的匹配）★ 新增
        if self.forms:
            last_form = self.forms[-1]
            current_form = "zu6" if unique_count == 3 else ("zu3" if unique_count == 2 else "baozi")
            # 最近形态的连续次数
            if self.form_streaks:
                last_streak_form, last_streak_len = self.form_streaks[-1]
                features.append(last_streak_len)  # 当前形态连续次数
                features.append(1 if current_form != last_streak_form else 0)  # 形态是否切换
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # 23. 质数个数（1个特征）
        primes = {2, 3, 5, 7}
        features.append(sum(1 for d in triple if d in primes))
        
        # 24. 大中小分布（3个特征）
        for d in triple:
            if d <= 2:
                features.append(1)
            elif d <= 5:
                features.append(2)
            else:
                features.append(3)
        
        return features
    
    def get_feature_names(self):
        """返回特征名称列表（v7增强版：54维）"""
        names = []
        
        # 1-3. 数字全局热度
        for i in range(3):
            names.append(f"digit_{i}_global_freq")
        
        # 4-6. 位置热度
        for i in range(3):
            names.append(f"pos_{i}_freq")
        
        # 7-9. 一阶马尔可夫概率
        for i in range(3):
            names.append(f"pos_{i}_markov1")
        
        # 10-12. 二阶马尔可夫概率
        for i in range(3):
            names.append(f"pos_{i}_markov2")
        
        # 13-15. 全局遗漏
        for i in range(3):
            names.append(f"digit_{i}_miss_global")
        
        # 16-18. 分位遗漏
        for i in range(3):
            names.append(f"pos_{i}_miss")
        
        # 19-24. 数字间隔统计
        for i in range(3):
            names.append(f"digit_{i}_exp_interval")
            names.append(f"digit_{i}_appear_count")
        
        # 25-27. 数字冷热等级
        for i in range(3):
            names.append(f"digit_{i}_tier")
        
        # 28-29. 和值
        names.extend(["sum", "sum_freq"])
        
        # 30. 和值偏离度
        names.append("sum_deviation")
        
        # 31-32. 跨度
        names.extend(["span", "span_freq"])
        
        # 33. 跨度偏离度
        names.append("span_deviation")
        
        # 34-35. 奇偶
        names.extend(["odd_count", "oe_freq"])
        
        # 36-37. 大小
        names.extend(["big_count", "bs_freq"])
        
        # 38. 连号
        names.append("has_consecutive")
        
        # 39-41. 上期相似度
        names.extend(["pos_repeat", "digit_overlap", "repeat_count"])
        
        # 42. 与上两期交集
        names.append("overlap_last2")
        
        # 43-44. 012 路
        names.extend(["road_sum", "road_max"])
        
        # 45. 012路匹配度
        names.append("road_match")
        
        # 46. 邻号
        names.append("neighbor_overlap")
        
        # 47-48. 形态
        names.extend(["is_baozi", "is_zu3"])
        
        # 49-50. 形态趋势
        names.extend(["form_streak", "form_switch"])
        
        # 51. 质数
        names.append("prime_count")
        
        # 52-54. 大中小
        for i in range(3):
            names.append(f"digit_{i}_size_cat")
        
        return names


class SimpleDecisionTree:
    """简易决策树（用于分类，优化版：分位数分裂加速）"""
    
    def __init__(self, max_depth=4, min_samples_split=10, feature_subset_ratio=1.0, rng=None, max_splits_per_feature=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.rng = rng if rng else random.Random(42)
        self.tree = None
        self.feature_indices = None
        self.max_splits_per_feature = max_splits_per_feature  # 每个特征最多评估的分裂点数量
    
    def _gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        p1 = sum(y) / len(y)
        p0 = 1 - p1
        return 1 - p0**2 - p1**2
    
    def _best_split(self, X, y, feature_indices):
        """寻找最佳分裂点（分位数加速版）"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini(y)
        n = len(y)
        
        for feature_idx in feature_indices:
            values = [X[i][feature_idx] for i in range(n)]
            sorted_indices = sorted(range(n), key=lambda i: values[i])
            
            # 使用分位数采样式分裂点（大幅加速）
            step = max(1, n // (self.max_splits_per_feature + 1))
            split_indices = list(range(step, n - 1, step))
            
            for si in split_indices:
                threshold = (values[sorted_indices[si]] + values[sorted_indices[si + 1]]) / 2
                
                # 在已排序数组中切分
                left_y = [y[sorted_indices[j]] for j in range(si + 1)]
                right_y = [y[sorted_indices[j]] for j in range(si + 1, n)]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                gain = current_gini - (
                    len(left_y) / n * self._gini(left_y) +
                    len(right_y) / n * self._gini(right_y)
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        # 停止条件
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        
        if len(set(y)) == 1:
            return {"leaf": True, "value": y[0]}
        
        # 随机选择特征子集
        n_features = len(X[0])
        n_select = max(1, int(n_features * self.feature_subset_ratio))
        feature_indices = self.rng.sample(range(n_features), n_select)
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_feature is None or best_gain <= 0:
            return {"leaf": True, "value": sum(y) / len(y) if y else 0.5}
        
        # 分裂
        left_idx = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
        
        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]
        
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }
    
    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x, node):
        """预测单个样本"""
        if node["leaf"]:
            return node["value"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
    
    def predict(self, X):
        """预测多个样本"""
        return [self._predict_one(x, self.tree) for x in X]


class SimpleRandomForest:
    """简易随机森林（多棵树投票，优化版）"""
    
    def __init__(self, n_trees=50, max_depth=4, min_samples_split=10, feature_subset_ratio=0.6, max_splits_per_feature=20, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_ratio = feature_subset_ratio
        self.max_splits_per_feature = max_splits_per_feature
        self.rng = rng if rng else random.Random(42)
        self.trees = []
    
    def fit(self, X, y):
        """训练随机森林（使用 bootstrap 采样）"""
        n = len(X)
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap 采样
            indices = [self.rng.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[j] for j in indices]
            y_boot = [y[j] for j in indices]
            
            # 训练树
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_subset_ratio=self.feature_subset_ratio,
                max_splits_per_feature=self.max_splits_per_feature,
                rng=self.rng
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """预测（多棵树平均）"""
        all_preds = [tree.predict(X) for tree in self.trees]
        
        # 平均所有树的预测
        n_samples = len(X)
        avg_preds = []
        for i in range(n_samples):
            avg = sum(preds[i] for preds in all_preds) / len(all_preds)
            avg_preds.append(avg)
        
        return avg_preds


def build_training_data(numbers, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD, rng=None):
    """
    构造训练数据（v7优化版 - 分层负采样 + 时间衰减）
    正例：历史每一期的开奖号码
    负例：每期分层抽取 neg_samples 个未开出的号码
    
    分层策略：将1000个组合按和值分组，每层按比例采样，确保负例分布均匀
    
    时间衰减：
    - 最近 30 期：权重 2.0
    - 最近 60 期：权重 1.4
    - 历史数据：权重 1.0
    
    返回：
        X: 特征矩阵
        y: 标签
        sample_weights: 样本权重
        group_ids: 期号分组（用于按期切分验证集）
    """
    if rng is None:
        rng = random.Random(len(numbers) * 10007)
    
    X = []
    y = []
    sample_weights = []
    group_ids = []
    
    min_history = 60
    if len(numbers) <= min_history:
        return None, None, None, None
    
    window_size = FEATURE_HISTORY_WINDOW
    
    for i in range(min_history, len(numbers)):
        start = max(0, i - window_size)
        train_numbers = numbers[start:i]
        actual = numbers[i]
        
        fe = FeatureEngineer(train_numbers)
        
        # 计算时间衰减权重
        periods_ago = len(numbers) - i
        if periods_ago <= TIME_DECAY_RECENT:
            weight = TIME_DECAY_RECENT_WEIGHT
        elif periods_ago <= TIME_DECAY_MID:
            weight = TIME_DECAY_MID_WEIGHT
        else:
            weight = TIME_DECAY_OLD_WEIGHT
        
        # 正例：1 个
        features = fe.build_features(*actual)
        X.append(features)
        y.append(1)
        sample_weights.append(weight)
        group_ids.append(i)
        
        # 负例：分层采样（按和值分组，确保分布均匀）
        actual_sum = sum(actual)
        neg_count = 0
        
        # 预计算所有可能负例并按和值分组
        sum_groups = {s: [] for s in range(28)}  # 和值范围 0-27
        actual_tuple = tuple(actual) if not isinstance(actual, tuple) else actual
        tried = {actual_tuple}
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    combo = (a, b, c)
                    if combo in tried:
                        continue
                    s = a + b + c
                    sum_groups[s].append(combo)
        
        # 计算每层采样配额（按该和值在全部组合中的比例分配）
        total_combos = 1000 - 1  # 排除实际开奖号
        sum_group_sizes = {s: len(combos) for s, combos in sum_groups.items()}
        
        # 先确保每个有组合的和值组至少采样1个（如果配额允许）
        valid_groups = [s for s, combos in sum_groups.items() if combos]
        samples_per_group = {}
        remaining = neg_samples
        for s in valid_groups:
            quota = max(1, int(neg_samples * sum_group_sizes[s] / total_combos))
            actual_quota = min(quota, sum_group_sizes[s], remaining)
            samples_per_group[s] = actual_quota
            remaining -= actual_quota
        
        # 随机打乱和值组列表，追加剩余配额
        shuffled_groups = list(valid_groups)
        rng.shuffle(shuffled_groups)
        for s in shuffled_groups:
            if remaining <= 0:
                break
            available = sum_group_sizes[s] - samples_per_group.get(s, 0)
            extra = min(remaining, available)
            samples_per_group[s] = samples_per_group.get(s, 0) + extra
            remaining -= extra
        
        # 从每组随机采样
        for s, quota in samples_per_group.items():
            if quota <= 0:
                continue
            combos = sum_groups[s]
            rng.shuffle(combos)
            for combo in combos[:quota]:
                if neg_count >= neg_samples:
                    break
                if combo in tried:
                    continue
                tried.add(combo)
                features = fe.build_features(*combo)
                X.append(features)
                y.append(0)
                sample_weights.append(weight)
                group_ids.append(i)
                neg_count += 1
    
    return X, y, sample_weights, group_ids


def select_features(X, y, feature_names, keep_ratio=FEATURE_SUBSET_RATIO):
    """
    特征选择：结合方差过滤和互信息选择
    """
    if not HAS_SKLEARN or len(X) < 50:
        return list(range(len(feature_names))), feature_names
    
    import numpy as np
    X_np = np.array(X)
    y_np = np.array(y)
    
    # 步骤1：方差过滤
    var_filter = VarianceThreshold(threshold=MIN_VARIANCE)
    X_filtered = var_filter.fit_transform(X_np)
    selected_mask = var_filter.get_support()
    
    # 步骤2：互信息选择
    n_features = int(len(feature_names) * keep_ratio)
    if n_features < 5:
        n_features = max(5, len(feature_names) // 2)
    
    try:
        mi_scores = mutual_info_classif(X_filtered, y_np, random_state=42)
        # 获取互信息得分高的特征索引
        mi_indices = np.argsort(mi_scores)[::-1][:n_features]
        
        # 映射回原始特征索引
        original_indices = np.where(selected_mask)[0][mi_indices]
        selected_indices = original_indices.tolist()
        selected_names = [feature_names[i] for i in selected_indices]
    except Exception:
        # 如果互信息计算失败，使用所有通过方差过滤的特征
        selected_indices = np.where(selected_mask)[0].tolist()[:n_features]
        selected_names = [feature_names[i] for i in selected_indices]
    
    return selected_indices, selected_names


def train_single_model(X, y, model_name, sample_weight=None, eval_set=None):
    """训练单个模型（v7增强版：更深的树、更多迭代、early stopping）"""
    y_list = list(y)
    n_neg = sum(1 for yi in y_list if yi == 0)
    n_pos = sum(1 for yi in y_list if yi == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    try:
        if model_name == "catboost" and HAS_CATBOOST:
            model = CatBoostClassifier(
                iterations=100,    # 更多迭代次数，配合early_stopping
                depth=4,           # 更深的树，捕获复杂交互
                learning_rate=0.03, # 更小的学习率，配合更多迭代
                l2_leaf_reg=15,    # L2正则化
                random_strength=1.5,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=False,
                task_type="CPU",
                early_stopping_rounds=20,  # early stopping防过拟合
                # 注：eval_set在外部传入
            )
            if eval_set is not None:
                model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
            else:
                model.fit(X, y, sample_weight=sample_weight)
            return model, "catboost"
        
        elif model_name == "xgboost" and HAS_XGBOOST:
            model = XGBClassifier(
                n_estimators=100,  # 更多迭代
                max_depth=4,       # 更深的树
                learning_rate=0.03, # 更小的学习率
                reg_lambda=15,     # L2正则化
                reg_alpha=5,       # L1正则化
                subsample=0.8,     # 行采样
                colsample_bytree=0.8,  # 列采样
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                early_stopping_rounds=20,
            )
            if eval_set is not None:
                model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
            else:
                model.fit(X, y, sample_weight=sample_weight)
            return model, "xgboost"
        
        elif model_name == "lightgbm" and HAS_LIGHTGBM:
            try:
                import numpy as np
                X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
                y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
                eval_set_data = None
                if eval_set is not None:
                    eval_X = np.array(eval_set[0]) if not isinstance(eval_set[0], np.ndarray) else eval_set[0]
                    eval_y = np.array(eval_set[1]) if not isinstance(eval_set[1], np.ndarray) else eval_set[1]
                    eval_set_data = [(eval_X, eval_y)]
            except Exception:
                eval_set_data = None
            
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.03,
                reg_lambda=15,
                reg_alpha=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
                force_col_wise=True,
            )
            if eval_set_data is not None:
                model.fit(X_arr, y_arr, sample_weight=sample_weight, eval_set=eval_set_data)
            else:
                model.fit(X_arr, y_arr, sample_weight=sample_weight)
            return model, "lightgbm"
    except Exception as e:
        log.warning(f"训练 {model_name} 失败: {e}")
    
    return None, None


def _validation_score(y_true, probs):
    """验证集得分（越高越好）：基于 LogLoss 的反函数"""
    import math
    if not y_true:
        return 0.5
    eps = 1e-15
    ll = 0.0
    for yt, p in zip(y_true, probs):
        p = max(eps, min(1 - eps, p))
        ll -= yt * math.log(p) + (1 - yt) * math.log(1 - p)
    ll /= len(y_true)
    return 1.0 / (1.0 + ll)


def train_ensemble(X, y, models_to_try=None, sample_weight=None, group_ids=None):
    """
    训练多模型集成（支持样本权重和按期切分）
    
    流程：先时序切分 → 训练集选特征 → 验证集评权重 → 全量数据重训
    
    参数：
        X: 特征矩阵
        y: 标签
        models_to_try: 尝试的模型列表
        sample_weight: 样本权重
        group_ids: 期号分组（用于按期切分验证集，避免数据泄漏）
    """
    if models_to_try is None:
        models_to_try = ["catboost", "xgboost", "lightgbm"]
    
    fe = FeatureEngineer([])
    feature_names = fe.get_feature_names()

    # 按期号切分验证集（严格时序验证，避免数据泄漏）
    if group_ids and len(group_ids) == len(X):
        unique_periods = sorted(set(group_ids))
        if len(unique_periods) >= 10:
            split_period = unique_periods[int(len(unique_periods) * 0.8)]
            train_idx = [j for j, gid in enumerate(group_ids) if gid <= split_period]
            valid_idx = [j for j, gid in enumerate(group_ids) if gid > split_period]
            use_valid = len(valid_idx) >= 5
        else:
            use_valid = False
    else:
        # 降级为按样本切分（不推荐）
        split = max(10, int(len(X) * 0.8))
        train_idx = list(range(split))
        valid_idx = list(range(split, len(X)))
        use_valid = len(X) - split >= 5
    
    if use_valid:
        X_train_raw = [X[j] for j in train_idx]
        y_train = [y[j] for j in train_idx]
        X_valid_raw = [X[j] for j in valid_idx]
        y_valid = [y[j] for j in valid_idx]
        w_train = [sample_weight[j] for j in train_idx] if sample_weight is not None else None
        w_full = sample_weight
    else:
        X_train_raw = X
        y_train = y
        X_valid_raw = None
        y_valid = None
        w_train = sample_weight
        w_full = sample_weight

    selected_indices, _ = select_features(X_train_raw, y_train, feature_names)
    X_train = [[x[i] for i in selected_indices] for x in X_train_raw]
    X_valid = [[x[i] for i in selected_indices] for x in X_valid_raw] if X_valid_raw else None
    X_full = [[x[i] for i in selected_indices] for x in X]

    # 构建eval_set用于early stopping
    eval_set = None
    if X_valid and y_valid:
        eval_set = (X_valid, y_valid)

    model_scores = {}

    for model_name in models_to_try:
        probe, used_name = train_single_model(X_train, y_train, model_name, sample_weight=w_train, eval_set=eval_set)
        if not probe:
            continue
        try:
            if X_valid and hasattr(probe, 'predict_proba'):
                probs = probe.predict_proba(X_valid)[:, 1]
                model_scores[used_name] = _validation_score(y_valid, probs)
            elif hasattr(probe, 'predict_proba'):
                probs = probe.predict_proba(X_train)[:, 1]
                model_scores[used_name] = _validation_score(y_train, probs)
            else:
                model_scores[used_name] = 0.5
        except Exception:
            model_scores[used_name] = 0.5
        log.info(f"验证 {used_name} 得分: {model_scores[used_name]:.4f}")

    trained_models = []
    for model_name in models_to_try:
        model, used_name = train_single_model(X_full, y, model_name, sample_weight=w_full, eval_set=eval_set)
        if model:
            score = model_scores.get(used_name, 0.5)
            trained_models.append((model, used_name, score))
            log.info(f"全量重训 {used_name} 完成，融合权重依据验证得分: {score:.4f}")

    if not trained_models:
        if not _ML_FALLBACK_WARNED:
            log.warning(
                "ML 集成后端不可用（catboost=%s xgboost=%s lightgbm=%s sklearn=%s），"
                "已降级为纯 Python 随机森林；若期望启用 ML 集成，请用已安装这些库的虚拟环境运行服务。",
                HAS_CATBOOST, HAS_XGBOOST, HAS_LIGHTGBM, HAS_SKLEARN,
            )
            _ML_FALLBACK_WARNED = True
        model = SimpleRandomForest(n_trees=30, max_depth=4, min_samples_split=15)
        model.fit(X_full, y)
        trained_models.append((model, "random_forest", 0.5))
    
    return trained_models, selected_indices


def ensemble_predict(models, X):
    """
    动态权重集成预测
    
    参数：
        models: [(model, model_name, score), ...]
        X: 特征矩阵
    
    返回：
        集成后的概率预测
    """
    if len(models) == 1:
        model, _, _ = models[0]
        return predict_single(model, X)
    
    # 动态权重：根据模型得分分配权重
    total_score = sum(score for _, _, score in models)
    if total_score == 0:
        weights = [1.0 / len(models)] * len(models)
    else:
        weights = [score / total_score for _, _, score in models]
    
    # 加权融合
    all_probs = []
    for model, model_name, _ in models:
        probs = predict_single(model, X)
        all_probs.append(probs)
    
    # 加权平均
    n_samples = len(all_probs[0])
    final_probs = []
    for i in range(n_samples):
        prob = sum(w * all_probs[j][i] for j, w in enumerate(weights))
        final_probs.append(prob)
    
    return final_probs


def predict_single(model, X):
    """预测单个模型"""
    if hasattr(model, 'predict_proba'):
        try:
            import numpy as np
            X_np = np.array(X) if not isinstance(X, np.ndarray) else X
            return model.predict_proba(X_np)[:, 1].tolist()
        except Exception:
            pass
    return model.predict(X)


def backtest_ml(numbers, trials=BACKTEST_TRIALS, train_window=TRAINING_WINDOW, base_period=None, neg_samples=NEGATIVE_SAMPLES_PER_PERIOD):
    """
    时序回测（滚动窗口版本）
    
    使用滚动窗口训练：每次预测时使用最近 train_window 期历史数据训练模型，
    与实盘逻辑保持一致。
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
        train_window: 训练窗口大小（使用最近多少期数据训练）
        base_period: 当前最后一期期号（用于标记回测记录）
    
    返回：
        包含多维度命中率指标的回测结果
    """
    if len(numbers) < trials + train_window:
        return {"error": f"数据量不足，需要至少 {trials + train_window} 期"}
    
    start_idx = len(numbers) - trials
    hit_top3 = hit_top15 = hit_top30 = hit_top100 = 0
    actual_ranks = []
    
    for i in range(start_idx, len(numbers)):
        # 准备数据
        history = numbers[:i]
        actual = numbers[i]
        
        # 滚动窗口：使用最近 train_window 期数据训练
        train_nums = history[-train_window:] if len(history) > train_window else history
        
        if len(train_nums) < 60:
            continue
        
        train_rng = random.Random(len(train_nums) * 10007)
        # 构建训练数据（包含样本权重和期号分组）
        result = build_training_data(train_nums, neg_samples=neg_samples, rng=train_rng)
        if result is None or len(result) < 4:
            continue
        X, y, sample_weights, group_ids = result
        if len(X) < 100:
            continue
        
        # 训练模型（使用集成方法，传递样本权重）
        try:
            models, selected_indices = train_ensemble(X, y, sample_weight=sample_weights, group_ids=group_ids)
        except Exception as e:
            print(f"训练失败：{e}")
            continue
        
        # 对所有 1000 个直选组合预测
        feature_history = history[-FEATURE_HISTORY_WINDOW:]
        fe = FeatureEngineer(feature_history)
        all_probs = []
        
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    features = fe.build_features(a, b, c)
                    features_selected = [features[j] for j in selected_indices]
                    all_probs.append((a, b, c, features_selected))
        
        # 预测（使用集成方法）
        X_all = [x[3] for x in all_probs]
        probs = ensemble_predict(models, X_all)
        
        # 排序
        ranked = sorted(
            [(probs[i], all_probs[i][0], all_probs[i][1], all_probs[i][2]) 
             for i in range(len(probs))],
            key=lambda x: -x[0]
        )
        
        # 计算真实号码排名
        actual_str = f"{actual[0]}{actual[1]}{actual[2]}"
        rank_map = {f"{a}{b}{c}": idx + 1 for idx, (_, a, b, c) in enumerate(ranked)}
        actual_rank = rank_map.get(actual_str, 1001)
        actual_ranks.append(actual_rank)
        
        # 检查命中（统一口径：Top3/Top15/Top30/Top100）
        top3_nums = {f"{a}{b}{c}" for _, a, b, c in ranked[:3]}
        top15_nums = {f"{a}{b}{c}" for _, a, b, c in ranked[:15]}
        top30_nums = {f"{a}{b}{c}" for _, a, b, c in ranked[:30]}
        top100_nums = {f"{a}{b}{c}" for _, a, b, c in ranked[:100]}
        
        if actual_str in top3_nums:
            hit_top3 += 1
        if actual_str in top15_nums:
            hit_top15 += 1
        if actual_str in top30_nums:
            hit_top30 += 1
        if actual_str in top100_nums:
            hit_top100 += 1
    
    n = len(actual_ranks)
    actual_rank_avg = sum(actual_ranks) / n if n > 0 else 0.0
    actual_rank_median = sorted(actual_ranks)[n // 2] if actual_ranks else 0
    
    result = {
        "trials": n,
        "evaluated": n,
        "model_type": "ensemble",
        "train_window": train_window,
        "base_period": base_period,  # 标记回测对应的期号
        # 统一口径的命中率指标
        "top3_hit": hit_top3,
        "top3_rate": round(hit_top3 / n if n > 0 else 0, 4),
        "top15_hit": hit_top15,
        "top15_rate": round(hit_top15 / n if n > 0 else 0, 4),
        "top30_hit": hit_top30,
        "top30_rate": round(hit_top30 / n if n > 0 else 0, 4),
        "top100_hit": hit_top100,
        "top100_rate": round(hit_top100 / n if n > 0 else 0, 4),
        # 排名指标
        "actual_rank_avg": round(actual_rank_avg, 1),
        "actual_rank_median": actual_rank_median,
        # 基准率（用于计算Lift）
        "baseline_top3_rate": 3 / 1000,
        "baseline_top30_rate": 30 / 1000,
    }
    
    # 保存回测结果到历史记录
    save_ml_backtest_history(result)
    
    return result


def save_ml_backtest_history(result):
    """保存ML回测结果到历史记录"""
    try:
        history = kv_store.load("lottery3d_ml_backtest_history", [])
        record = {
            "base_period": result.get("base_period"),
            "model_version": "ml-v7",
            "top30_rate": result["top30_rate"],
            "top3_rate": result["top3_rate"],
            "top100_rate": result["top100_rate"],
            "actual_rank_avg": result["actual_rank_avg"],
            "actual_rank_median": result["actual_rank_median"],
            "trials": result["trials"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        history.append(record)
        # 保留最近30次回测记录
        kv_store.save("lottery3d_ml_backtest_history", history[-30:])
        log.info("ML回测历史已保存")
    except Exception as e:
        log.error(f"保存ML回测历史失败: {e}")


def predict_current(numbers, top_k=TOP_K, model_type="ensemble", neg_samples=NEGATIVE_SAMPLES_PER_PERIOD):
    """
    预测当前期（多模型集成版）
    返回推荐号码

    参数：
        model_type: "ensemble" (多模型集成), "catboost", "xgboost", "lightgbm", "random_forest"
        neg_samples: 每期负例采样数
    """
    if len(numbers) < 100:
        return {"error": "历史数据不足"}

    train_numbers = numbers[-TRAINING_WINDOW:]
    feature_history = numbers[-FEATURE_HISTORY_WINDOW:]
    train_rng = random.Random(len(train_numbers) * 10007)

    result = build_training_data(
        train_numbers,
        neg_samples=neg_samples,
        rng=train_rng,
    )
    if result is None or len(result) < 4:
        return {"error": "训练数据不足"}
    X, y, sample_weights, group_ids = result
    if len(X) < 100:
        return {"error": "训练数据不足"}

    try:
        if model_type == "ensemble":
            # 多模型集成（传递样本权重和分组信息）
            models, selected_indices = train_ensemble(X, y, sample_weight=sample_weights, group_ids=group_ids)
            model_names = [name for _, name, _ in models]
            model_weights = [score for _, _, score in models]
            log.info(f'3D-ML 多模型集成完成：{model_names}')
        else:
            # 单模型（传递样本权重）
            model, used_name = train_single_model(X, y, model_type, sample_weight=sample_weights)
            if not model:
                log.warning(f"{model_type} 不可用，降级为集成模式")
                models, selected_indices = train_ensemble(X, y, sample_weight=sample_weights, group_ids=group_ids)
                model_names = [name for _, name, _ in models]
                model_weights = [score for _, _, score in models]
            else:
                models = [(model, used_name, 1.0)]
                selected_indices = list(range(len(X[0])))
                model_names = [used_name]
                model_weights = [1.0]
    except Exception as e:
        log.error(f'3D-ML 训练失败：{e}', exc_info=True)
        return {'error': '训练失败'}
    
    # 对所有 1000 个直选组合预测（批量处理）
    fe = FeatureEngineer(feature_history)
    
    # 批量构建特征
    all_probs = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                features = fe.build_features(a, b, c)
                # 筛选特征
                features_selected = [features[i] for i in selected_indices]
                all_probs.append((a, b, c, features_selected))
    
    # 批量预测（使用集成）
    X_all = [x[3] for x in all_probs]
    probs = ensemble_predict(models, X_all)
    
    # 排序
    ranked = sorted(
        [(probs[i], all_probs[i][0], all_probs[i][1], all_probs[i][2]) 
         for i in range(len(probs))],
        key=lambda x: -x[0]
    )
    
    # 计算 Top K 的相对占比（使用排序后的概率，而非原始顺序）
    top_probs = [item[0] for item in ranked[:top_k]]
    total_top_prob = sum(top_probs) or 1.0
    
    # 返回 Top K（model_score 为排序分，topk_score_share 为 TopK 内占比，非真实概率）
    # 注意：由于训练集是 1:30 正负采样，而非真实 1:999 比例，
    # model_score 只能用于排序，不能当作真实概率使用
    recommendations = []
    for prob, a, b, c in ranked[:top_k]:
        score_share = round(_native_number(prob) / total_top_prob, 4)
        recommendations.append({
            "num": f"{a}{b}{c}",
            "model_score": round(_native_number(prob), 4),
            "topk_score_share": score_share,  # TopK 内部分数占比（非真实概率）
        })

    # 特征重要性（取第一个模型的）
    feature_names = fe.get_feature_names()
    selected_feature_names = [feature_names[i] for i in selected_indices]
    top_features = []
    
    if models:
        model, model_name, _ = models[0]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features = sorted(
                [(selected_feature_names[i], _native_number(importances[i])) 
                 for i in range(len(selected_feature_names))],
                key=lambda x: -x[1],
            )[:10]
        else:
            top_features = [(selected_feature_names[i], float(i)) 
                           for i in range(min(10, len(selected_feature_names)))]
    else:
        top_features = [(selected_feature_names[i], float(i)) 
                       for i in range(min(10, len(selected_feature_names)))]

    pos_n = int(_native_number(sum(y)))
    
    # 计算模型权重
    total_weight = sum(model_weights) if model_weights else 1
    normalized_weights = [w / total_weight for w in model_weights] if model_weights else []
    
    return {
        "recommendations": recommendations,
        "top3": recommendations[:3],
        "feature_importance": [[name, round(float(score), 4)] for name, score in top_features],
        "total_samples": int(len(X)),
        "pos_samples": pos_n,
        "neg_samples": int(len(y) - pos_n),
        "model_type": "+".join(model_names),
        "model_weights": [round(w, 4) for w in normalized_weights],
        "model_info": f"多模型集成 ({', '.join(model_names)})",
        "num_models": len(models),
    }


def print_ml_report(result, top_k=TOP_K):
    """打印 ML 预测结果"""
    if result.get("error"):
        print(result["error"])
        return
    
    print("\n" + "=" * 70)
    print("【福彩 3D ML 预测结果 V7.0 - 多模型集成版】")
    print("=" * 70)
    model_type = result.get('model_type', 'unknown')
    model_info = result.get('model_info', '未知模型')
    num_models = result.get('num_models', 1)
    print(f"  模型类型：{model_info}")
    print(f"  模型数量：{num_models} 个")
    
    # 显示模型权重
    weights = result.get('model_weights', [])
    if weights:
        print(f"  模型权重：{', '.join(f'{w:.2f}' for w in weights)}")
    
    if "catboost" in model_type.lower() or "xgboost" in model_type.lower():
        print(f"  ⚡ 速度：快 (3-8 秒) | 准确率：高")
    elif "lightgbm" in model_type.lower():
        print(f"  ⚡ 速度：快 (5-10 秒) | 准确率：中高")
    else:
        print(f"   速度：慢 (15-30 秒) | 准确率：中")
    
    print(f"  训练样本：{result.get('total_samples', 0)} (正例：{result.get('pos_samples', 0)}, 负例：{result.get('neg_samples', 0)})")
    
    print("\n" + "=" * 70)
    print(f"【直选推荐 {top_k} 注】（按模型分排序）")
    print("=" * 70)
    for idx, rec in enumerate(result["recommendations"], start=1):
        marker = "★" if idx <= 3 else " "
        print(f"  {marker} {idx:02d}. {rec['num']}  模型分={rec['model_score']:.4f}")
    
    print("\n" + "=" * 70)
    print("【特征重要性（前 10 个）】")
    print("=" * 70)
    for i, (name, score) in enumerate(result["feature_importance"], start=1):
        print(f"  {i:2d}. {name} (重要性：{score:.2f})" if isinstance(score, (int, float)) else f"  {i:2d}. {name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="福彩 3D 预测器 V7.0 (多模型集成版：CatBoost/XGBoost/LightGBM)")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="运行回测"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=BACKTEST_TRIALS,
        help="回测期数"
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=TRAINING_WINDOW,
        help="滚动训练窗口大小（期数）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="推荐注数"
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=NEGATIVE_SAMPLES_PER_PERIOD,
        help="每期负例采样数"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "catboost", "xgboost", "lightgbm", "random_forest"],
        help="模型类型"
    )
    
    args = parser.parse_args()
    
    print("抓取数据中...")
    data = fetch_data()
    if not data:
        print("未获取到数据")
        return
    
    numbers = [x[2] for x in data]
    print(f"共 {len(numbers)} 期数据")
    
    if args.backtest:
        print(f"\n运行回测（{args.trials}期，训练窗口={args.train_window}，负例数={args.neg_samples}）...")
        result = backtest_ml(
            numbers,
            trials=args.trials,
            train_window=args.train_window,
            base_period=data[-1][0],  # 传入最新期号
            neg_samples=args.neg_samples,
        )
        if result.get("error"):
            print(result["error"])
        else:
            print("\n" + "=" * 70)
            print("【回测结果】")
            print("=" * 70)
            print(f"  回测期数：{result['trials']}")
            print(f"  Top3 命中：{result['top3_hit']} ({result['top3_rate']*100:.2f}%)")
            print(f"  Top15 命中：{result['top15_hit']} ({result['top15_rate']*100:.2f}%)")
            print(f"  Top30 命中：{result['top30_hit']} ({result['top30_rate']*100:.2f}%)")
            print(f"  Top100 覆盖率：{result['top100_hit']} ({result['top100_rate']*100:.2f}%)")
            print(f"  平均真实排名：{result['actual_rank_avg']}")
            print(f"  中位真实排名：{result['actual_rank_median']}")
    else:
        print("\n运行预测...")
        result = predict_current(
            numbers,
            top_k=args.top_k,
            model_type=args.model,
            neg_samples=args.neg_samples,
        )
        print_ml_report(result, top_k=args.top_k)


# ML 缓存配置
ML_CACHE_KEY = "lottery3d_ml_prediction_cache"


def load_ml_cache():
    """加载 ML 预测缓存"""
    try:
        return kv_store.load(ML_CACHE_KEY, None)
    except Exception as e:
        log.error(f"加载 ML 缓存失败: {e}")
        return None


def save_ml_cache(data):
    """保存 ML 预测缓存"""
    try:
        kv_store.save(ML_CACHE_KEY, data)
        log.info("ML 缓存已保存")
    except Exception as e:
        log.error(f"保存 ML 缓存失败: {e}")

def clear_ml_cache():
    """Clear persisted ML prediction cache."""
    try:
        kv_store.save(ML_CACHE_KEY, None)
        log.info("ML prediction cache cleared")
    except Exception as e:
        log.error(f"Clear ML prediction cache failed: {e}")


if __name__ == "__main__":
    main()
