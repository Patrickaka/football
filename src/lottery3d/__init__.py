# 福彩3D预测器 V3.1+（标准库版，准确率优化）
# Python 3.10+
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from itertools import combinations, product
from ..common.logger import setup_logger
from ..common.data_cache import cached_fetch

log = setup_logger('lottery3d')

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

URL = "https://www.8300.cn/kjhhis/3/200.html"

RECENT_WINDOWS = (30, 45, 60, 90)
RECENT_WINDOW = 90  # 展示用最大窗口
WINDOW_BACKTEST_TRIALS = 40
EXP_DECAY = 0.96
BACKTEST_TRIALS = 80
PERMUTATION_SHUFFLES = 20  # 置换检验打乱次数，评估命中率是否显著优于随机

# 缓存配置
_prediction_cache = None
_cache_time = 0

def _is_today_cache(cache_timestamp):
    """检查缓存是否是今天的（按自然天判断）"""
    if cache_timestamp is None or cache_timestamp == 0:
        return False
    
    import datetime
    cache_date = datetime.date.fromtimestamp(cache_timestamp)
    today = datetime.date.today()
    return cache_date == today

def clear_cache():
    """清除缓存"""
    global _prediction_cache, _cache_time
    _prediction_cache = None
    _cache_time = 0
    log.info("3D模块缓存已清除")

W_HOT_GLOBAL = 2.5   # 原 4.0；降低热号全局权重，减少同一号码长期霸榜
W_HOT_POS = 3.0     # 原 5.0；降低分位热号权重，让转移概率有更多发言权
# 冷号遗漏加分：W_MISS_HIGH 对待极高遗漏值（≥20 期），W_MISS_MID 对待中等遗漏值（≥12 期）
W_MISS_HIGH = 6.0   # 遗漏 20 期加 12 分，30 期加 18 分，40 期加 24 分（增强冷号补偿）
W_MISS_MID = 7.0    # 中等遗漏值加分增强
W_MARKOV = 5.0       # 原 8.0；降低马尔可夫转移权重，避免过度依赖最近一期
W_MARKOV2 = 1.5      # 原 4.0；降低二阶马尔可夫转移权重
MARKOV_MAX_SCORE = 6.0  # 马尔可夫转移得分上限，避免主导推荐结果
W_LAST_APPEAR = 2.5
W_NEIGHBOR = 2.0
W_ROAD_MATCH = 1.5
W_DANMA_HIT = 4.0
W_KILL_PENALTY = 6.0  # 杀码出现在组合中时每码扣分（软约束，非硬杀）
W_CONSECUTIVE = 1.5   # 含相邻连号（如 12、67）
W_POS_REPEAT = 1.2    # 与上期同位重复（直选复刻），每码；实际强度由 lag1 动态缩放
W_RATIO_MATCH = 1.8   # 奇偶比 / 大小比与近期热门匹配
# 随机基准：单位置复刻 10%；指定数字在下一期三码中出现 ≈ 27.1%
RANDOM_POS_REPEAT = 0.10
RANDOM_DIGIT_REUSE = 1 - (9 / 10) ** 3
SUM_SOFT_SIGMA = 3.2
SPAN_SOFT_SIGMA = 1.4

# 探索机制：推荐时有一定概率从候选池中随机选择
EXPLORATION_RATE = 0.15  # 15%概率进行探索，85%概率选择最高分号码

# 动态胆码机制：70%概率选 Top2，30%概率从 Top6 中随机选 2 个
DANMA_TOP_POOL = 6  # 胆码候选池大小
DANMA_RANDOM_RATE = 0.30  # 30%概率随机选择胆码

# 推荐注数（直选为带顺序的三位数）
RECOMMEND_GROUPS = 30  # 推荐池扩大至 30 注
ZHIXUAN_TOP3 = 3
ZU6_POOL_SIZE = 5
ZU6_FOUR_SIZE = 4

# Top50 随机扰动：避免同分号长期霸榜
RANDOM_NOISE = 0.3  # 随机噪声范围 [-0.3, 0.3]

# 近期回补模型：统计最近 30 期，严重欠账的号码额外加分
RECENT_WINDOW_REBOUND = 30  # 回补统计窗口
REBOUND_BONUS = 3.0  # 严重欠账号码加分
REBOUND_THRESHOLD = 0.5  # 欠账阈值（实际值/理论值 < 0.5 认为严重欠账）

# 冷热平衡模型：推荐池号码类型比例
HOT_RATIO = 0.40   # 热号比例 40%
WARM_RATIO = 0.40  # 温号比例 40%
COLD_RATIO = 0.20  # 冷号比例 20%
HOT_WINDOW = 20    # 冷热判断窗口

# 和值趋势模型：统计最近 20 期和值趋势，动态调整和值中心
SUM_TREND_WINDOW = 20  # 和值趋势统计窗口
SUM_TREND_ADJUST = 2.0  # 和值中心调整幅度

# 遗漏周期模型：计算平均遗漏周期和超期倍率
MISS_CYCLE_WINDOW = 200  # 统计平均遗漏周期的窗口大小
MISS_OVER_RATIO_THRESHOLD = 2.5  # 超期倍率阈值
MISS_OVER_BONUS = 5.0  # 超期额外加分

# 数字配对模型：统计数字对出现频率
PAIR_FREQ_WINDOWS = [50, 100, 200]  # 统计窗口
PAIR_HIGH_FREQ_THRESHOLD = 0.15  # 高频对子阈值（出现频率 > 15%）
PAIR_BONUS = 2.5  # 高频对子加分

# 组三组六切换模型：根据连续出现调整权重
FORM_SWITCH_WEIGHT = 4.0  # 切换奖励权重
ZU6_STREAK_THRESHOLD = 8  # 组六连续出现阈值，超过此值增加组三权重
ZU3_STREAK_THRESHOLD = 4  # 组三连续出现阈值，超过此值增加组六权重

# 和值区间回归模型：预测区间而非具体和值
SUM_INTERVAL_WINDOW = 5  # 计算中心的窗口大小
SUM_INTERVAL_WIDTH = 3  # 区间宽度（中心 ± width）
SUM_INTERVAL_BONUS = 3.0  # 区间内加分
SUM_EXTREME_PENALTY = 2.0  # 极端区间降权（0-5, 25-27）

# 最近5期排除机制：对重复推荐进行惩罚
RECENT_RECOMMEND_WINDOW = 5  # 最近推荐窗口大小
RECENT_RECOMMEND_PENALTY = 2.0  # 最近推荐过的号码惩罚
RECENT_RECOMMEND_CONSECUTIVE_PENALTY = 4.0  # 连续推荐的号码惩罚

# 推荐池多样性控制：最大化数字覆盖率
DIVERSITY_WEIGHT = 1.5  # 多样性权重
DIVERSITY_TARGET_COVERAGE = 0.8  # 目标数字覆盖率（0-1）

# 回测目标调整：优化综合评分
COMPOSITE_WEIGHT_TOP_HIT = 0.4  # top_hit 权重
COMPOSITE_WEIGHT_GE2_RATE = 0.3  # ge2_rate 权重
COMPOSITE_WEIGHT_ZU6_RATE = 0.2  # zu6_rate 权重
COMPOSITE_WEIGHT_KILL_RATE = 0.1  # kill_rate 权重

# 贝叶斯融合：融合多模型预测
BAYESIAN_PRIOR_WEIGHT = 0.3  # 先验权重
BAYESIAN_LIKELIHOOD_WEIGHT = 0.7  # 似然权重

# 推荐号码去相关：减少高度相关推荐
CORRELATION_THRESHOLD = 2  # 重合数字阈值
CORRELATION_PENALTY = 3.0  # 相关惩罚分数

# 自动淘汰失效特征：定期评估特征贡献
FEATURE_EVAL_PERIOD = 30  # 特征评估周期（期数）
FEATURE_MIN_CONTRIBUTION = 0.01  # 最小贡献率阈值（1%）
FEATURE_DOWNGRADE_FACTOR = 0.5  # 降权因子

# 马尔可夫转移：拉普拉斯平滑系数 α（加法平滑，α=1 即标准 Laplace）
MARKOV_LAPLACE_ALPHA = 1.0

# 可调评分权重（供 search_weights 搜索）
TUNABLE_WEIGHTS = (
    "W_HOT_GLOBAL",
    "W_HOT_POS",
    "W_MISS_MID",
    "W_MARKOV",
    "W_MARKOV2",
    "W_LAST_APPEAR",
    "W_NEIGHBOR",
    "W_ROAD_MATCH",
    "W_DANMA_HIT",
    "W_KILL_PENALTY",
    "SUM_SOFT_SIGMA",
    "SPAN_SOFT_SIGMA",
)

# 随机搜索时各参数相对默认值的缩放范围 (low, high)
WEIGHT_SEARCH_RANGES = {
    "W_HOT_GLOBAL": (0.5, 2.0),
    "W_HOT_POS": (0.5, 2.0),
    "W_MISS_MID": (0.4, 2.5),
    "W_MARKOV": (0.5, 2.0),
    "W_MARKOV2": (0.3, 1.5),
    "W_LAST_APPEAR": (0.3, 2.5),
    "W_NEIGHBOR": (0.3, 2.5),
    "W_ROAD_MATCH": (0.0, 3.0),
    "W_DANMA_HIT": (0.5, 2.5),
    "W_KILL_PENALTY": (3.0, 12.0),
    "SUM_SOFT_SIGMA": (2.0, 5.0),
    "SPAN_SOFT_SIGMA": (0.8, 2.2),
}


def default_weights():
    """当前默认评分权重快照"""
    return {k: globals()[k] for k in TUNABLE_WEIGHTS}


@contextmanager
def patch_weights(weights):
    """临时覆盖模块级权重，供回测/搜索使用"""
    saved = {k: globals()[k] for k in TUNABLE_WEIGHTS}
    for k in TUNABLE_WEIGHTS:
        if k in weights:
            globals()[k] = weights[k]
    try:
        yield
    finally:
        for k, v in saved.items():
            globals()[k] = v


def _fetch_data_internal(url=URL):
    """内部数据抓取函数"""
    log.debug('fetch 3D data')
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
    return cached_fetch('lottery3d', lambda: _fetch_data_internal(url), force_refresh)


def calc_span(n):
    return max(n) - min(n)


def miss_value(numbers, digit, position=None):
    for i in range(len(numbers) - 1, -1, -1):
        n = numbers[i]
        if position is None:
            if digit in n:
                return len(numbers) - 1 - i
        elif n[position] == digit:
            return len(numbers) - 1 - i
    return len(numbers)


def neighbor(d):
    return {(d - 1) % 10, (d + 1) % 10}


def road(d):
    return d % 3


def exp_weighted_counts(series, decay=EXP_DECAY):
    cnt = Counter()
    w = 1.0
    for item in reversed(series):
        cnt[item] += w
        w *= decay
    return cnt


def build_markov(numbers, position):
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        a, b = numbers[i][position], numbers[i + 1][position]
        trans[a][b] += 1
    return trans


def build_markov2(numbers, position):
    """二阶马尔可夫转移矩阵：P(next | prev2, prev1) → Counter[(prev2, prev1)][next]"""
    trans2 = defaultdict(Counter)
    for i in range(len(numbers) - 2):
        p2, p1 = numbers[i][position], numbers[i + 1][position]
        nx = numbers[i + 2][position]
        trans2[(p2, p1)][nx] += 1
    return trans2


def markov_prob_smoothed(row, states, alpha=MARKOV_LAPLACE_ALPHA):
    """转移概率 P(next|prev)，拉普拉斯平滑：(count + α) / (total + α·|S|)"""
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def gaussian_score(value, center, sigma):
    if sigma <= 0:
        return 0.0
    z = (value - center) / sigma
    return math.exp(-0.5 * z * z)


def _recent_slice(series, window):
    return series[-window:] if len(series) > window else list(series)


def odd_even_key(triple):
    """奇偶比 (奇数个数, 偶数个数)"""
    odds = sum(1 for d in triple if d % 2 == 1)
    return odds, 3 - odds


def big_small_key(triple):
    """大小比 (大数个数, 小数个数)，0-4 小、5-9 大"""
    big = sum(1 for d in triple if d >= 5)
    return big, 3 - big


def ratio_label(key, kind="oe"):
    a, b = key
    if kind == "oe":
        return f"{a}奇{b}偶"
    return f"{a}大{b}小"


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号（差值为 1，不含 9-0）"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def entropy_model(numbers, min_appear_window=30):
    """熵值模型：统计数字熵、和值熵、跨度熵，计算长期未出现号码的奖励
    
    参数：
        numbers: 历史开奖号码列表
        min_appear_window: 最小统计窗口（期数）
    
    返回：
        熵值奖励字典 {digit: entropy_bonus}
    """
    if len(numbers) < min_appear_window:
        return {d: 0.0 for d in range(10)}
    
    # 统计数字出现频率
    digit_counts = Counter()
    sum_counts = Counter()
    span_counts = Counter()
    
    for n in numbers[-min_appear_window:]:
        for d in n:
            digit_counts[d] += 1
        sum_counts[sum(n)] += 1
        span_counts[calc_span(n)] += 1
    
    # 计算熵值奖励
    entropy_bonus = {}
    total_digits = sum(digit_counts.values())
    total_sums = sum(sum_counts.values())
    total_spans = sum(span_counts.values())
    
    for d in range(10):
        bonus = 0.0
        
        # 数字熵：计算该数字的出现频率
        digit_freq = digit_counts.get(d, 0) / total_digits if total_digits > 0 else 0
        
        # 若某数字长期未进入推荐池（出现频率低于期望值的 50%）
        expected_freq = 1.0 / 10  # 期望频率 10%
        if digit_freq < expected_freq * 0.5:
            bonus += 2.0  # 长期未出现奖励
        
        # 若长期未进入前三推荐（通过遗漏值判断）
        mv = miss_value(numbers, d)
        if mv >= 25:  # 遗漏 25 期以上
            bonus += 3.0  # 长期未进前三奖励
        
        entropy_bonus[d] = bonus
    
    return entropy_bonus


def rebound_model(numbers, window=RECENT_WINDOW_REBOUND):
    """近期回补模型：统计最近 N 期数字出现次数，严重欠账的号码额外加分
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口（期数）
    
    返回：
        回补奖励字典 {digit: rebound_bonus}
    """
    if len(numbers) < window:
        return {d: 0.0 for d in range(10)}
    
    # 统计最近 window 期数字出现次数
    digit_counts = Counter()
    for n in numbers[-window:]:
        for d in n:
            digit_counts[d] += 1
    
    # 计算理论值：每期待 3 个数字，window 期共 3*window 个数字，10 个数字平均分配
    theoretical = (3 * window) / 10.0  # 理论出现次数
    
    # 计算回补奖励
    rebound_bonus = {}
    for d in range(10):
        actual = digit_counts.get(d, 0)
        ratio = actual / theoretical if theoretical > 0 else 0
        
        # 严重欠账：实际值/理论值 < 阈值
        if ratio < REBOUND_THRESHOLD:
            rebound_bonus[d] = REBOUND_BONUS
        else:
            rebound_bonus[d] = 0.0
    
    return rebound_bonus


def classify_digits_by_hot(numbers, window=HOT_WINDOW):
    """将数字分为热号、温号、冷号三类
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口
    
    返回：
        (hot_digits, warm_digits, cold_digits)
    """
    if len(numbers) < window:
        return list(range(10)), [], []
    
    # 统计最近 window 期数字出现次数
    digit_counts = Counter()
    for n in numbers[-window:]:
        for d in n:
            digit_counts[d] += 1
    
    # 计算理论值
    theoretical = (3 * window) / 10.0
    
    hot_digits = []
    warm_digits = []
    cold_digits = []
    
    for d in range(10):
        actual = digit_counts.get(d, 0)
        ratio = actual / theoretical if theoretical > 0 else 0
        
        if ratio >= 1.2:  # 超过理论值 20% 为热号
            hot_digits.append(d)
        elif ratio >= 0.8:  # 理论值 80%-120% 为温号
            warm_digits.append(d)
        else:  # 低于理论值 80% 为冷号
            cold_digits.append(d)
    
    return hot_digits, warm_digits, cold_digits


def sum_trend_model(numbers, window=SUM_TREND_WINDOW):
    """和值趋势模型：统计最近 N 期和值趋势，动态调整和值中心
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口
    
    返回：
        adjusted_sum_center: 调整后的和值中心
        trend_direction: 趋势方向 ('up', 'down', 'oscillate')
    """
    if len(numbers) < window:
        return 13.5, 'oscillate'  # 默认和值中心（0-27 的中间值）
    
    # 计算最近 window 期的和值
    recent_sums = [sum(n) for n in numbers[-window:]]
    
    # 计算前一半和后一半的平均和值
    half = window // 2
    first_half_avg = sum(recent_sums[:half]) / half if half > 0 else 0
    second_half_avg = sum(recent_sums[half:]) / (window - half) if (window - half) > 0 else 0
    
    # 计算整体平均和值
    overall_avg = sum(recent_sums) / window
    
    # 判断趋势
    if second_half_avg > first_half_avg + 1.5:
        trend_direction = 'up'
        adjusted_sum_center = overall_avg + SUM_TREND_ADJUST
    elif second_half_avg < first_half_avg - 1.5:
        trend_direction = 'down'
        adjusted_sum_center = overall_avg - SUM_TREND_ADJUST
    else:
        trend_direction = 'oscillate'
        adjusted_sum_center = overall_avg
    
    # 限制和值中心在合理范围内（0-27）
    adjusted_sum_center = max(0, min(27, adjusted_sum_center))
    
    return adjusted_sum_center, trend_direction


def average_miss_cycle(numbers, digit, window=MISS_CYCLE_WINDOW):
    """计算单个数字的平均遗漏周期
    
    参数：
        numbers: 历史开奖号码列表
        digit: 目标数字
        window: 统计窗口大小
    
    返回：
        avg_cycle: 平均遗漏周期（期数），如果数据不足返回默认值 7
    """
    if len(numbers) < 10:
        return 7.0  # 默认平均遗漏周期
    
    # 使用最近 window 期数据
    recent_numbers = numbers[-window:] if len(numbers) > window else numbers
    
    miss_periods = []
    current_miss = 0
    
    for n in recent_numbers:
        if digit in n:
            miss_periods.append(current_miss)
            current_miss = 0
        else:
            current_miss += 1
    
    # 如果最后还有未结束的遗漏，不计入
    if miss_periods:
        return sum(miss_periods) / len(miss_periods)
    else:
        return 7.0  # 默认值


def miss_cycle_bonus(numbers):
    """遗漏周期模型：计算超期遗漏奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        bonus: 各数字的超期奖励 {digit: bonus}
    """
    bonus = {}
    
    for d in range(10):
        current_miss = miss_value(numbers, d)
        avg_miss = average_miss_cycle(numbers, d)
        
        if avg_miss > 0:
            ratio = current_miss / avg_miss
            if ratio > MISS_OVER_RATIO_THRESHOLD:
                # 超期倍率越高，奖励越多
                bonus[d] = MISS_OVER_BONUS * (ratio - MISS_OVER_RATIO_THRESHOLD + 1)
            else:
                bonus[d] = 0.0
        else:
            bonus[d] = 0.0
    
    return bonus


def pair_frequency(numbers, window=50):
    """统计数字对出现频率
    
    参数：
        numbers: 历史开奖号码列表
        window: 统计窗口大小
    
    返回：
        pair_freq: 数字对频率字典 {(a, b): freq}，a <= b
    """
    recent_numbers = numbers[-window:] if len(numbers) > window else numbers
    total_draws = len(recent_numbers)
    
    if total_draws == 0:
        return {}
    
    pair_counts = Counter()
    
    for n in recent_numbers:
        # 生成所有不重复的数字对（不考虑顺序）
        digits = sorted(set(n))  # 去重并排序
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                pair_counts[(digits[i], digits[j])] += 1
    
    # 计算频率
    pair_freq = {}
    for pair, count in pair_counts.items():
        pair_freq[pair] = count / total_draws
    
    return pair_freq


def high_freq_pairs(numbers):
    """获取高频数字对
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        high_pairs: 高频数字对集合 {(a, b), ...}
    """
    high_pairs = set()
    
    for window in PAIR_FREQ_WINDOWS:
        pair_freq = pair_frequency(numbers, window)
        for pair, freq in pair_freq.items():
            if freq > PAIR_HIGH_FREQ_THRESHOLD:
                high_pairs.add(pair)
    
    return high_pairs


def pair_bonus(triple, numbers):
    """计算号码组合中的数字配对奖励
    
    参数：
        triple: 三位数号码 (d1, d2, d3)
        numbers: 历史开奖号码列表
    
    返回：
        bonus: 配对奖励分数
    """
    bonus = 0.0
    high_pairs = high_freq_pairs(numbers)
    
    # 生成号码中的所有数字对
    digits = sorted(set(triple))
    for i in range(len(digits)):
        for j in range(i + 1, len(digits)):
            if (digits[i], digits[j]) in high_pairs:
                bonus += PAIR_BONUS
    
    return bonus


def form_switch_bonus(numbers):
    """组三组六切换模型：根据连续出现次数计算切换奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        bonus: {"zu3": 组三奖励, "zu6": 组六奖励}
    """
    if len(numbers) < 5:
        return {"zu3": 0.0, "zu6": 0.0}
    
    # 统计最近的形式序列
    forms = [classify_form(n) for n in numbers]
    last_form = forms[-1]
    
    # 计算连续出现次数
    streak = 1
    for i in range(len(forms) - 2, -1, -1):
        if forms[i] == last_form:
            streak += 1
        else:
            break
    
    bonus = {"zu3": 0.0, "zu6": 0.0}
    
    # 如果组六连续出现过多，增加组三权重
    if last_form == "zu6" and streak >= ZU6_STREAK_THRESHOLD:
        # 连续次数越多，切换奖励越大
        bonus["zu3"] = FORM_SWITCH_WEIGHT * (streak - ZU6_STREAK_THRESHOLD + 1)
    
    # 如果组三连续出现过多，增加组六权重
    elif last_form == "zu3" and streak >= ZU3_STREAK_THRESHOLD:
        bonus["zu6"] = FORM_SWITCH_WEIGHT * (streak - ZU3_STREAK_THRESHOLD + 1)
    
    return bonus


def sum_interval_bonus(numbers):
    """和值区间回归模型：计算和值区间奖励
    
    参数：
        numbers: 历史开奖号码列表
    
    返回：
        interval_info: {"center": 和值中心, "low": 区间下限, "high": 区间上限}
    """
    if len(numbers) < SUM_INTERVAL_WINDOW:
        return {"center": 13.5, "low": 10, "high": 17, "bonus": {}}
    
    # 计算最近 SUM_INTERVAL_WINDOW 期的和值中心
    recent_numbers = numbers[-SUM_INTERVAL_WINDOW:]
    recent_sums = [sum(n) for n in recent_numbers]
    sum_center = sum(recent_sums) / len(recent_sums)
    
    # 定义区间
    interval_low = max(0, int(sum_center - SUM_INTERVAL_WIDTH))
    interval_high = min(27, int(sum_center + SUM_INTERVAL_WIDTH))
    
    # 构建奖励字典
    bonus = {}
    for s in range(28):
        if interval_low <= s <= interval_high:
            bonus[s] = SUM_INTERVAL_BONUS
        elif s <= 5 or s >= 25:
            bonus[s] = -SUM_EXTREME_PENALTY
        else:
            bonus[s] = 0.0
    
    return {"center": sum_center, "low": interval_low, "high": interval_high, "bonus": bonus}


def recent_recommend_penalty(pool, recent_recommendations):
    """最近5期排除机制：对重复推荐进行惩罚
    
    参数：
        pool: 当前推荐池 [(权重, 号码字符串), ...]
        recent_recommendations: 最近推荐历史列表 [[号码字符串, ...], ...]
    
    返回：
        penalized_pool: 应用惩罚后的推荐池
    """
    if not recent_recommendations:
        return pool
    
    # 扁平化最近推荐历史
    recent_set = set()
    consecutive_count = {}
    
    for rec_list in recent_recommendations[-RECENT_RECOMMEND_WINDOW:]:
        for num_str in rec_list:
            recent_set.add(num_str)
            consecutive_count[num_str] = consecutive_count.get(num_str, 0) + 1
    
    # 应用惩罚
    penalized_pool = []
    for w, num_str in pool:
        penalty = 0.0
        
        # 如果最近推荐过
        if num_str in recent_set:
            penalty -= RECENT_RECOMMEND_PENALTY
        
        # 如果连续推荐过（出现多次）
        if consecutive_count.get(num_str, 0) >= 2:
            penalty -= RECENT_RECOMMEND_CONSECUTIVE_PENALTY
        
        penalized_pool.append((w + penalty, num_str))
    
    return penalized_pool


def diversity_filter(pool, top_n=10):
    """推荐池多样性控制：最大化数字覆盖率
    
    参数：
        pool: 当前推荐池 [(权重, 号码字符串), ...]
        top_n: 目标推荐数量
    
    返回：
        diverse_pool: 多样性优化后的推荐池
    """
    if len(pool) <= top_n:
        return pool
    
    # 使用贪心算法选择多样性最大的组合
    selected = []
    covered_digits = set()
    
    # 先按原始权重排序
    sorted_pool = sorted(pool, key=lambda x: -x[0])
    
    for _ in range(top_n):
        best_candidate = None
        best_score = -float('inf')
        
        for w, num_str in sorted_pool:
            if num_str in [s[1] for s in selected]:
                continue
            
            # 计算数字覆盖率收益
            digits = set(num_str)
            new_digits = digits - covered_digits
            coverage_gain = len(new_digits) * DIVERSITY_WEIGHT
            
            # 综合评分：原始权重 + 多样性收益
            total_score = w + coverage_gain
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = (w, num_str)
        
        if best_candidate:
            selected.append(best_candidate)
            covered_digits.update(set(best_candidate[1]))
        else:
            break
    
    # 如果还没选够，从剩余中补充
    if len(selected) < top_n:
        remaining = [item for item in sorted_pool if item[1] not in [s[1] for s in selected]]
        selected.extend(remaining[:top_n - len(selected)])
    
    return selected


def composite_score(metrics):
    """计算综合评分（用于回测目标优化）
    
    参数：
        metrics: 指标字典，包含 top_hit, ge2_rate, zu6_rate, kill_rate
    
    返回：
        score: 综合评分
    """
    top_hit = metrics.get("top_hit", 0.0)
    ge2_rate = metrics.get("ge2_rate", 0.0)
    zu6_rate = metrics.get("zu6_rate", 0.0)
    kill_rate = metrics.get("kill_rate", 0.0)
    
    score = (
        COMPOSITE_WEIGHT_TOP_HIT * top_hit
        + COMPOSITE_WEIGHT_GE2_RATE * ge2_rate
        + COMPOSITE_WEIGHT_ZU6_RATE * zu6_rate
        + COMPOSITE_WEIGHT_KILL_RATE * kill_rate
    )
    
    return score


def bayesian_adjust(scores, model_probs):
    """贝叶斯融合：融合多模型预测
    
    参数：
        scores: 各模型的分数字典 {"hot": score, "miss": score, "markov": score, "ml": prob}
        model_probs: 各模型的先验概率权重
    
    返回：
        adjusted_score: 贝叶斯调整后的分数
    """
    # 提取各模型的分数
    hot_score = scores.get("hot", 0.0)
    miss_score = scores.get("miss", 0.0)
    markov_score = scores.get("markov", 0.0)
    ml_prob = scores.get("ml", 0.5)  # ML 概率
    
    # 归一化分数（转换为概率形式）
    total_score = hot_score + miss_score + markov_score + 1e-9
    hot_prob = hot_score / total_score if total_score > 0 else 0.33
    miss_prob = miss_score / total_score if total_score > 0 else 0.33
    markov_prob = markov_score / total_score if total_score > 0 else 0.34
    
    # 计算先验（基于历史统计的平均概率）
    prior = 0.1  # 3D 号码中奖的先验概率（约 1/1000，这里简化为 0.1）
    
    # 计算似然（各模型的加权平均）
    likelihood = (
        0.25 * hot_prob
        + 0.25 * miss_prob
        + 0.25 * markov_prob
        + 0.25 * ml_prob
    )
    
    # 贝叶斯公式：Posterior ∝ Prior × Likelihood
    # 为了数值稳定性，使用加权组合
    posterior = (
        BAYESIAN_PRIOR_WEIGHT * prior
        + BAYESIAN_LIKELIHOOD_WEIGHT * likelihood
    )
    
    return posterior


def correlation_penalty(pool):
    """推荐号码去相关：减少高度相关推荐
    
    参数：
        pool: 当前推荐池 [(权重, 号码字符串), ...]
    
    返回：
        penalized_pool: 应用去相关惩罚后的推荐池
    """
    if len(pool) <= 1:
        return pool
    
    # 先按原始权重排序
    sorted_pool = sorted(pool, key=lambda x: -x[0])
    
    # 记录已选中的号码的数字集合
    selected_digits = []
    penalized_pool = []
    
    for w, num_str in sorted_pool:
        current_digits = set(num_str)
        penalty = 0.0
        
        # 计算与已选中号码的相关性
        for selected in selected_digits:
            overlap = len(current_digits & selected)
            if overlap >= CORRELATION_THRESHOLD:
                penalty += CORRELATION_PENALTY
        
        penalized_pool.append((w - penalty, num_str))
        selected_digits.append(current_digits)
    
    # 重新排序
    penalized_pool.sort(key=lambda x: -x[0])
    
    return penalized_pool


class FeatureEvaluator:
    """自动淘汰失效特征：定期评估特征贡献"""
    
    def __init__(self):
        self.feature_contributions = {
            "hot": [],
            "miss": [],
            "markov": [],
            "neighbor": [],
            "road": []
        }
        self.current_period = 0
        self.dynamic_weights = {
            "hot": W_HOT_GLOBAL,
            "miss": W_MISS_HIGH,
            "markov": W_MARKOV,
            "neighbor": W_NEIGHBOR,
            "road": W_ROAD_MATCH
        }
    
    def record_contribution(self, contributions):
        """记录本期各特征的贡献
        
        参数：
            contributions: 各特征贡献字典 {"hot": value, "miss": value, ...}
        """
        for feature, value in contributions.items():
            if feature in self.feature_contributions:
                self.feature_contributions[feature].append(value)
        
        self.current_period += 1
        
        # 每 FEATURE_EVAL_PERIOD 期评估一次
        if self.current_period % FEATURE_EVAL_PERIOD == 0:
            self.evaluate_features()
    
    def evaluate_features(self):
        """评估各特征的贡献并自动调整权重"""
        total_contribution = 0.0
        feature_totals = {}
        
        for feature, values in self.feature_contributions.items():
            if values:
                feature_totals[feature] = sum(values[-FEATURE_EVAL_PERIOD:])
                total_contribution += feature_totals[feature]
        
        # 计算各特征的贡献率
        if total_contribution > 0:
            for feature, total in feature_totals.items():
                contribution_rate = total / total_contribution
                
                # 如果贡献率低于阈值，降权
                if contribution_rate < FEATURE_MIN_CONTRIBUTION:
                    self.dynamic_weights[feature] *= FEATURE_DOWNGRADE_FACTOR
                    log.info(f"特征 {feature} 贡献率 {contribution_rate:.4f} < {FEATURE_MIN_CONTRIBUTION}, 权重调整为 {self.dynamic_weights[feature]:.4f}")
                else:
                    # 恢复权重（如果之前被降权）
                    default_weights = {
                        "hot": W_HOT_GLOBAL,
                        "miss": W_MISS_HIGH,
                        "markov": W_MARKOV,
                        "neighbor": W_NEIGHBOR,
                        "road": W_ROAD_MATCH
                    }
                    if self.dynamic_weights[feature] < default_weights[feature]:
                        self.dynamic_weights[feature] = min(
                            self.dynamic_weights[feature] / FEATURE_DOWNGRADE_FACTOR,
                            default_weights[feature]
                        )
        
        # 重置贡献记录
        for feature in self.feature_contributions:
            self.feature_contributions[feature] = []
    
    def get_weights(self):
        """获取当前动态权重"""
        return self.dynamic_weights


def position_repeat_count(triple, last_draw):
    """与上期同位置重复个数（直选复刻）"""
    return sum(1 for i in range(3) if triple[i] == last_draw[i])


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _empty_lag1():
    return {
        "pairs": 0,
        "pos_repeat_rate": [RANDOM_POS_REPEAT] * 3,
        "avg_pos_repeat": RANDOM_POS_REPEAT,
        "repeat_dist": {0: 1.0},
        "full_repeat_rate": 0.0,
        "same_set_rate": 0.0,
        "ge2_overlap_rate": 0.0,
        "digit_reuse_rate": RANDOM_DIGIT_REUSE,
    }


def analyze_lag1_dynamics(numbers, window=RECENT_WINDOW):
    """分析近窗「上期→本期」转移：同位复刻、重号、全同号等"""
    if len(numbers) < 2:
        return _empty_lag1()

    pairs = list(zip(numbers[:-1], numbers[1:]))
    recent_pairs = pairs[-window:] if len(pairs) > window else pairs

    pos_w = [0.0] * 3
    repeat_dist = Counter()
    full_w = same_set_w = ge2_w = digit_hit = digit_total = 0.0
    total_w = 0.0
    w = 1.0
    for prev, cur in reversed(recent_pairs):
        rep = position_repeat_count(cur, prev)
        repeat_dist[rep] += w
        for j in range(3):
            if prev[j] == cur[j]:
                pos_w[j] += w
        if prev == cur:
            full_w += w
        if set(prev) == set(cur):
            same_set_w += w
        if len(set(prev) & set(cur)) >= 2:
            ge2_w += w
        for d in set(prev):
            digit_total += w
            if d in cur:
                digit_hit += w
        total_w += w
        w *= EXP_DECAY

    total_w = total_w or 1.0
    return {
        "pairs": len(recent_pairs),
        "pos_repeat_rate": [pos_w[i] / total_w for i in range(3)],
        "avg_pos_repeat": sum(pos_w) / (3 * total_w),
        "repeat_dist": {k: v / total_w for k, v in sorted(repeat_dist.items())},
        "full_repeat_rate": full_w / total_w,
        "same_set_rate": same_set_w / total_w,
        "ge2_overlap_rate": ge2_w / total_w,
        "digit_reuse_rate": digit_hit / digit_total if digit_total else RANDOM_DIGIT_REUSE,
    }


def ensemble_lag1_dynamics(numbers, window_weights):
    """多窗口加权集成上期→本期转移统计"""
    acc = _empty_lag1()
    if len(numbers) < 2:
        return acc

    pos_rate = [0.0] * 3
    repeat_dist = Counter()
    full = same_set = ge2 = digit_hit = digit_total = avg_rep = 0.0
    pairs_n = 0

    for w, wt in window_weights.items():
        lag = analyze_lag1_dynamics(numbers, window=w)
        pairs_n = max(pairs_n, lag["pairs"])
        for i in range(3):
            pos_rate[i] += wt * lag["pos_repeat_rate"][i]
        for k, v in lag["repeat_dist"].items():
            repeat_dist[k] += wt * v
        full += wt * lag["full_repeat_rate"]
        same_set += wt * lag["same_set_rate"]
        ge2 += wt * lag["ge2_overlap_rate"]
        digit_hit += wt * lag["digit_reuse_rate"]
        digit_total += wt
        avg_rep += wt * lag["avg_pos_repeat"]

    return {
        "pairs": pairs_n,
        "pos_repeat_rate": pos_rate,
        "avg_pos_repeat": avg_rep,
        "repeat_dist": dict(repeat_dist),
        "full_repeat_rate": full,
        "same_set_rate": same_set,
        "ge2_overlap_rate": ge2,
        "digit_reuse_rate": digit_hit / digit_total if digit_total else RANDOM_DIGIT_REUSE,
    }


def derive_dynamic_weights(lag1, consec_rate):
    """根据历史转移统计动态缩放评分权重与惩罚项"""
    avg_rep = lag1["avg_pos_repeat"]
    w_pos = W_POS_REPEAT * _clamp(avg_rep / RANDOM_POS_REPEAT, 0.2, 1.6)
    pos_mult = [_clamp(r / RANDOM_POS_REPEAT, 0.3, 2.0) for r in lag1["pos_repeat_rate"]]
    w_last = W_LAST_APPEAR * _clamp(lag1["digit_reuse_rate"] / RANDOM_DIGIT_REUSE, 0.3, 1.4)
    consec_base = max(consec_rate, 0.15)
    w_consec = W_CONSECUTIVE * _clamp(consec_rate / consec_base, 0.6, 1.2)
    w_full_pen = _clamp(12.0 * (1.0 - lag1["full_repeat_rate"] * 80), 4.0, 15.0)
    w_perm_pen = _clamp(6.0 * (1.0 - lag1["same_set_rate"] * 40), 1.5, 8.0)
    return {
        "w_pos_repeat": w_pos,
        "pos_mult": pos_mult,
        "w_last_appear": w_last,
        "w_consecutive": w_consec,
        "w_full_repeat_penalty": w_full_pen,
        "w_same_set_penalty": w_perm_pen,
    }


def analyze_patterns(numbers, window=RECENT_WINDOW):
    """统计近窗连号占比、奇偶比/大小比频次"""
    recent = _recent_slice(numbers, window)
    oe_freq = Counter()
    bs_freq = Counter()
    consec_w = 0.0
    w = 1.0
    for n in reversed(recent):
        oe_freq[odd_even_key(n)] += w
        bs_freq[big_small_key(n)] += w
        if has_consecutive_digits(*n):
            consec_w += w
        w *= EXP_DECAY
    total = sum(oe_freq.values()) or 1.0
    return {
        "oe_freq": oe_freq,
        "bs_freq": bs_freq,
        "consec_rate": consec_w / total,
    }


def ensemble_patterns(numbers, window_weights):
    """多窗口加权集成形态模式统计"""
    oe_acc = Counter()
    bs_acc = Counter()
    consec_rate = 0.0
    for w, wt in window_weights.items():
        p = analyze_patterns(numbers, window=w)
        for k, v in p["oe_freq"].items():
            oe_acc[k] += wt * v
        for k, v in p["bs_freq"].items():
            bs_acc[k] += wt * v
        consec_rate += wt * p["consec_rate"]
    oe_total = sum(oe_acc.values()) or 1.0
    bs_total = sum(bs_acc.values()) or 1.0
    return {
        "oe_freq": oe_acc,
        "bs_freq": bs_acc,
        "oe_total": oe_total,
        "bs_total": bs_total,
        "hot_oe_set": {k for k, _ in oe_acc.most_common(3)},
        "hot_bs_set": {k for k, _ in bs_acc.most_common(3)},
        "consec_rate": consec_rate,
    }


def analyze_sum_span(sums, spans, window=RECENT_WINDOW):
    recent_s = _recent_slice(sums, window)
    recent_p = _recent_slice(spans, window)
    w_s = exp_weighted_counts(recent_s)
    w_p = exp_weighted_counts(recent_p)

    sum_center = sum(k * v for k, v in w_s.items()) / max(sum(w_s.values()), 1e-9)
    span_center = sum(k * v for k, v in w_p.items()) / max(sum(w_p.values()), 1e-9)

    # 加入近期趋势感知：最近5期的移动均值偏移
    if len(recent_s) >= 5:
        recent5_s = recent_s[-5:]
        recent5_p = recent_p[-5:]
        avg5_s = sum(recent5_s) / 5
        avg5_p = sum(recent5_p) / 5
        # 趋势偏向近期：center 以 0.35 的权重向最近5期移动
        sum_center = sum_center * 0.65 + avg5_s * 0.35
        span_center = span_center * 0.65 + avg5_p * 0.35

    return {
        "sum_center": sum_center,
        "span_center": span_center,
        "hot_sums": [x for x, _ in w_s.most_common(6)],
        "hot_spans": [x for x, _ in w_p.most_common(4)],
        "sum_tail_freq": Counter(s % 10 for s in recent_s),
    }


def ensemble_sum_span(sums, spans, window_weights):
    """多窗口加权集成和值/跨度中心与热号"""
    sum_center = span_center = 0.0
    hot_sums_vote = Counter()
    hot_spans_vote = Counter()
    tail_acc = Counter()
    for w, wt in window_weights.items():
        r = analyze_sum_span(sums, spans, window=w)
        sum_center += wt * r["sum_center"]
        span_center += wt * r["span_center"]
        for s in r["hot_sums"]:
            hot_sums_vote[s] += wt
        for s in r["hot_spans"]:
            hot_spans_vote[s] += wt
        for tail, cnt in r["sum_tail_freq"].items():
            tail_acc[tail] += wt * cnt
    return {
        "sum_center": sum_center,
        "span_center": span_center,
        "hot_sums": [x for x, _ in hot_sums_vote.most_common(6)],
        "hot_spans": [x for x, _ in hot_spans_vote.most_common(4)],
        "sum_tail_freq": tail_acc,
    }


def digit_scores(numbers, window=RECENT_WINDOW, dynamic=None):
    recent = _recent_slice(numbers, window)
    last = numbers[-1]
    score = [0.0] * 10
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)

    freq_all = exp_weighted_counts([d for n in recent for d in n])
    for d, _ in freq_all.most_common(4):
        score[d] += W_HOT_GLOBAL

    for pos in range(3):
        pos_freq = exp_weighted_counts([n[pos] for n in recent])
        for d, _ in pos_freq.most_common(3):
            score[d] += W_HOT_POS

        trans = build_markov(numbers, pos)
        prev_d = last[pos]
        row = trans.get(prev_d, Counter())
        for d, p in markov_prob_smoothed(row, range(10)).items():
            markov_score = W_MARKOV * p
            score[d] += min(markov_score, MARKOV_MAX_SCORE)

        # 二阶马尔可夫：基于最近两期的转移概率
        if len(numbers) >= 2:
            trans2 = build_markov2(numbers, pos)
            prev2 = numbers[-2][pos]
            prev1 = last[pos]
            row2 = trans2.get((prev2, prev1), Counter())
            for d, p in markov_prob_smoothed(row2, range(10)).items():
                markov2_score = W_MARKOV2 * p
                score[d] += min(markov2_score, MARKOV_MAX_SCORE)

    for d in range(10):
        mv = miss_value(numbers, d)
        if mv >= 20:
            score[d] += W_MISS_HIGH * (1 + mv / 20)  # 遗漏20期加6分，30期加9分，40期加12分
        elif mv >= 12:
            score[d] += W_MISS_MID
    
    # 遗漏周期奖励：超期倍率超过阈值的号码获得额外加分
    miss_cycle_bonus_scores = miss_cycle_bonus(numbers)
    for d in range(10):
        score[d] += miss_cycle_bonus_scores.get(d, 0.0)

    for d in set(last):
        score[d] += w_last

    nb = set()
    for d in last:
        nb.update(neighbor(d))
    for d in nb:
        score[d] += W_NEIGHBOR

    last_roads = {road(d) for d in last}
    for d in range(10):
        if road(d) in last_roads:
            score[d] += W_ROAD_MATCH

    # 熵值奖励：长期未出现的号码获得额外加分
    entropy_bonus = entropy_model(numbers)
    for d in range(10):
        score[d] += entropy_bonus.get(d, 0.0)

    # 近期回补奖励：严重欠账的号码获得额外加分
    rebound_bonus = rebound_model(numbers)
    for d in range(10):
        score[d] += rebound_bonus.get(d, 0.0)

    return score, freq_all


def ensemble_digit_scores(numbers, window_weights, dynamic=None):
    combined = [0.0] * 10
    freq_combined = Counter()
    for w, wt in window_weights.items():
        sc, freq = digit_scores(numbers, window=w, dynamic=dynamic)
        for d in range(10):
            combined[d] += wt * sc[d]
        for d, c in freq.items():
            freq_combined[d] += wt * c
    
    # 熵值奖励：长期未出现的号码获得额外加分（只在最终评分中添加一次）
    entropy_bonus = entropy_model(numbers)
    for d in range(10):
        combined[d] += entropy_bonus.get(d, 0.0)
    
    # 近期回补奖励：严重欠账的号码获得额外加分（只在最终评分中添加一次）
    rebound_bonus = rebound_model(numbers)
    for d in range(10):
        combined[d] += rebound_bonus.get(d, 0.0)
    
    return combined, freq_combined


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    """单码分位评分（百/十/个）"""
    recent = [n[position] for n in _recent_slice(numbers, window)]
    last_d = numbers[-1][position]
    sc = [0.0] * 10
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
    for d, _ in exp_weighted_counts(recent).most_common(4):
        sc[d] += W_HOT_POS + 1
    trans = build_markov(numbers, position)
    row = trans.get(last_d, Counter())
    for d, p in markov_prob_smoothed(row, range(10)).items():
        markov_score = W_MARKOV * p
        sc[d] += min(markov_score, MARKOV_MAX_SCORE)
    # 二阶马尔可夫
    if len(numbers) >= 2:
        trans2 = build_markov2(numbers, position)
        prev2_d = numbers[-2][position]
        row2 = trans2.get((prev2_d, last_d), Counter())
        for d, p in markov_prob_smoothed(row2, range(10)).items():
            markov2_score = W_MARKOV2 * p
            sc[d] += min(markov2_score, MARKOV_MAX_SCORE)
    mv = miss_value(numbers, None, position=position)
    for d in range(10):
        miss_p = miss_value(numbers, d, position=position)
        if miss_p >= 20:
            sc[d] += W_MISS_HIGH * (1 + miss_p / 20)  # 遗漏20期加6分，30期加9分，40期加12分
        elif miss_p >= 12:
            sc[d] += W_MISS_MID
    sc[last_d] += w_last * pos_mult[position]
    for d in neighbor(last_d):
        sc[d] += W_NEIGHBOR
    return sc


def ensemble_position_digit_scores(numbers, position, window_weights, dynamic=None):
    sc = [0.0] * 10
    for w, wt in window_weights.items():
        ps = position_digit_scores(numbers, position, window=w, dynamic=dynamic)
        for d in range(10):
            sc[d] += wt * ps[d]
    return sc


# 窗口权重缓存
_window_weights_cache = None
_window_weights_cache_time = 0
_window_weights_cache_numbers_hash = None

def default_window_weights():
    n = len(RECENT_WINDOWS)
    return {w: 1.0 / n for w in RECENT_WINDOWS}


def compute_window_weights(numbers, trials=WINDOW_BACKTEST_TRIALS, enable_cache=True):
    """回测各窗口 Top3 命中表现，拉普拉斯先验后归一化为集成权重
    
    参数：
        numbers: 历史号码数据
        trials: 回测次数
        enable_cache: 是否启用缓存（默认 True）
    
    返回：
        (weights, scores): 窗口权重字典和原始分数字典
    """
    global _window_weights_cache, _window_weights_cache_time, _window_weights_cache_numbers_hash
    
    max_w = max(RECENT_WINDOWS)
    if len(numbers) < max_w + 10:
        return default_window_weights(), {}
    
    # 检查缓存
    numbers_hash = hash(tuple(tuple(n) for n in numbers[-max_w-10:]))
    if enable_cache and _window_weights_cache is not None:
        elapsed = time.time() - _window_weights_cache_time
        if elapsed < 3600 and _window_weights_cache_numbers_hash == numbers_hash:
            log.debug("使用缓存的窗口权重")
            return _window_weights_cache
    
    trials = min(trials, len(numbers) - max_w - 5)
    trials = max(10, trials)
    raw = {w: 0.0 for w in RECENT_WINDOWS}
    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"
        for w in RECENT_WINDOWS:
            if len(train) < w:
                continue
            sums = [sum(x) for x in train]
            spans = [calc_span(x) for x in train]
            meta = build_ranking_meta(train, {w: 1.0}, sums, spans, tail_top=4)
            sc, _ = digit_scores(train, window=w, dynamic=meta.get("dynamic"))
            dan, _, kill, _ = pick_dan_tuo_kill(sc, enable_danma_random=False)
            top = rank_triplets(sc, dan, kill, meta, top_n=ZHIXUAN_TOP3, enable_exploration=False)
            top_nums = [t[1] for t in top]
            if act_s in top_nums:
                raw[w] += 1.0
            elif len({int(c) for s in top_nums for c in s} & set(actual)) >= 2:
                raw[w] += 0.25

    prior = 1.0
    total = sum(raw[w] + prior for w in RECENT_WINDOWS)
    weights = {w: (raw[w] + prior) / total for w in RECENT_WINDOWS}
    
    # 更新缓存
    if enable_cache:
        _window_weights_cache = (weights, {w: round(raw[w], 1) for w in RECENT_WINDOWS})
        _window_weights_cache_time = time.time()
        _window_weights_cache_numbers_hash = numbers_hash
    
    return weights, {w: round(raw[w], 1) for w in RECENT_WINDOWS}


def classify_form(triple):
    """形态：组六 / 组三 / 豹子"""
    n = len(set(triple))
    if n == 3:
        return "zu6"
    if n == 2:
        return "zu3"
    return "baozi"


FORM_LABELS = {"zu6": "组六", "zu3": "组三", "baozi": "豹子"}
THEORY_FORM_P = {"zu6": 0.72, "zu3": 0.27, "baozi": 0.01}


def form_miss(forms, target):
    """距上次出现 target 形态的期数"""
    for i in range(len(forms) - 1, -1, -1):
        if forms[i] == target:
            return len(forms) - 1 - i
    return len(forms)


def _form_recent_p(forms, window):
    recent = _recent_slice(forms, window)
    w_cnt = exp_weighted_counts(recent)
    w_total = sum(w_cnt.values()) or 1.0
    return {k: w_cnt.get(k, 0) / w_total for k in THEORY_FORM_P}


def analyze_form_probability(numbers, window_weights=None):
    """估算本期开出组六/组三/豹子的概率（多源融合）"""
    forms = [classify_form(n) for n in numbers]
    last_form = forms[-1]

    if window_weights:
        recent_p = {k: 0.0 for k in THEORY_FORM_P}
        for w, wt in window_weights.items():
            rp = _form_recent_p(forms, w)
            for k in THEORY_FORM_P:
                recent_p[k] += wt * rp[k]
    else:
        recent_p = _form_recent_p(forms, RECENT_WINDOW)

    hist_cnt = Counter(forms)
    hist_total = len(forms)
    hist_p = {k: hist_cnt.get(k, 0) / hist_total for k in THEORY_FORM_P}

    trans = defaultdict(Counter)
    for i in range(len(forms) - 1):
        trans[forms[i]][forms[i + 1]] += 1
    row = trans.get(last_form, Counter())
    row_total = sum(row.values())
    markov_p = markov_prob_smoothed(row, THEORY_FORM_P)

    blend = {}
    for k in THEORY_FORM_P:
        blend[k] = (
            0.40 * recent_p[k]
            + 0.35 * markov_p[k]
            + 0.15 * hist_p[k]
            + 0.10 * THEORY_FORM_P[k]
        )
    total = sum(blend.values()) or 1.0
    blend = {k: v / total for k, v in blend.items()}

    streak = 1
    for i in range(len(forms) - 2, -1, -1):
        if forms[i] == last_form:
            streak += 1
        else:
            break

    return {
        "last_form": last_form,
        "streak": streak,
        "miss_zu6": form_miss(forms, "zu6"),
        "miss_zu3": form_miss(forms, "zu3"),
        "recent_p": recent_p,
        "hist_p": hist_p,
        "markov_p": markov_p,
        "blend_p": blend,
        "markov_samples": row_total,
    }


def pick_dan_tuo_kill(score, enable_danma_random=True):
    """动态选择胆码、拖码和杀码
    
    参数：
        score: 各数字评分
        enable_danma_random: 是否启用胆码随机选择
    
    返回：
        (胆码，拖码，杀码，排名列表)
    """
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    # 动态胆码机制：70%选 Top2，30%从 Top6 中随机选 2 个
    danma = select_danma(rank, enable_random=enable_danma_random)
    tuoma = [x[0] for x in rank[2:6]]
    kill = [rank[-1][0]] if rank[-1][1] + 3 < rank[-2][1] else [x[0] for x in rank[-2:]]
    return danma, tuoma, kill, rank


def pick_zu6_four(score, kill=None):
    """组六四码：按有效分选 4 个号（杀码降权）"""
    return pick_zu6_pool(score, kill, pool_size=ZU6_FOUR_SIZE)


def zu6_notes_from_digits(digits):
    """N 码组六 → C(N,3) 注组六组合"""
    combos = [tuple(sorted(c)) for c in combinations(digits, 3)]
    return combos, ["".join(map(str, c)) for c in combos]


def _effective_digit_score(score, digit, kill=None):
    """单码有效分：杀码降权而非排除"""
    kill_set = set(kill or [])
    return score[digit] - (W_KILL_PENALTY if digit in kill_set else 0.0)


def pick_zu6_pool(score, kill=None, pool_size=ZU6_POOL_SIZE):
    """组六复式选号：按有效分取 top N（杀码降权）"""
    rank = sorted(range(10), key=lambda d: -_effective_digit_score(score, d, kill))
    return sorted(rank[:pool_size])


def rank_zu6_groups(score, digits, danma, kill, meta, top_n=RECOMMEND_GROUPS):
    """从复式号码中按评分排出 top_n 注组六（五码时恰好 10 注）"""
    ranked = []
    for combo in combinations(digits, 3):
        a, b, c = combo
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        ranked.append((w, "".join(map(str, (a, b, c)))))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:top_n]


def is_zu6_draw(triple):
    """开奖号为组六（三码各不相同）"""
    return len(set(triple)) == 3


def triplet_weight(a, b, c, score, danma, kill, meta):
    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    w = score[a] + score[b] + score[c]
    for x in (a, b, c):
        if x in danma:
            w += W_DANMA_HIT
        if x in kill_set:
            w -= W_KILL_PENALTY

    s = a + b + c
    w += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

    span = max(a, b, c) - min(a, b, c)
    w += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

    if s in meta["hot_sum_set"]:
        w += 2.0
    if span in meta["hot_span_set"]:
        w += 1.5
    if (s % 10) in meta["sum_tail_top"]:
        w += 1.0

    if has_consecutive_digits(a, b, c):
        w += dyn.get("w_consecutive", W_CONSECUTIVE)

    last_draw = meta.get("last_draw")
    w_pos = dyn.get("w_pos_repeat", W_POS_REPEAT)
    pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
    if last_draw:
        triple = (a, b, c)
        for i in range(3):
            if triple[i] == last_draw[i]:
                w += w_pos * pos_mult[i]
        if triple == tuple(last_draw):
            w -= dyn.get("w_full_repeat_penalty", 0.0)
        elif set(triple) == set(last_draw):
            w -= dyn.get("w_same_set_penalty", 0.0)

    oe = odd_even_key((a, b, c))
    bs = big_small_key((a, b, c))
    oe_freq = meta.get("oe_freq")
    bs_freq = meta.get("bs_freq")
    if oe_freq:
        w += W_RATIO_MATCH * oe_freq.get(oe, 0) / meta.get("oe_total", 1)
    if bs_freq:
        w += W_RATIO_MATCH * bs_freq.get(bs, 0) / meta.get("bs_total", 1)
    
    # 数字配对奖励：高频数字对加分
    numbers = meta.get("numbers", [])
    if len(numbers) >= 50:
        w += pair_bonus((a, b, c), numbers)
    
    # 组三组六切换奖励：连续同形式出现后增加切换概率
    if len(numbers) >= 5:
        form_bonus = form_switch_bonus(numbers)
        # 判断当前号码是组三还是组六
        if a == b or a == c or b == c:
            # 组三或豹子
            w += form_bonus.get("zu3", 0.0)
        else:
            # 组六
            w += form_bonus.get("zu6", 0.0)
    
    # 和值区间回归奖励：区间内加分，极端区间降权
    if len(numbers) >= SUM_INTERVAL_WINDOW:
        sum_interval_info = sum_interval_bonus(numbers)
        w += sum_interval_info["bonus"].get(s, 0.0)
    
    return w


def select_danma(score_rank, enable_random=True):
    """动态选择胆码
    
    参数：
        score_rank: 按评分排序的数字列表 [(分数，数字), ...]
        enable_random: 是否启用随机选择
    
    返回：
        胆码列表（2 个数字）
    """
    # 提取 Top6 数字
    top6_digits = [d for _, d in score_rank[:DANMA_TOP_POOL]]
    
    if enable_random and random.random() < DANMA_RANDOM_RATE:
        # 30%概率：从 Top6 中随机选 2 个
        return random.sample(top6_digits, 2)
    else:
        # 70%概率：选择前 2 个
        return top6_digits[:2]


def rank_triplets(score, danma, kill, meta, top_n=20, enable_exploration=True, apply_noise=True, enable_cold_hot_balance=True, recent_recommendations=None, enable_diversity=True, enable_correlation=True):
    """对三位数组合进行评分排序，支持探索机制、随机扰动和冷热平衡
    
    参数：
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
        top_n: 返回前 N 个推荐
        enable_exploration: 是否启用探索机制
        apply_noise: 是否应用随机噪声扰动
        enable_cold_hot_balance: 是否启用冷热平衡
        recent_recommendations: 最近推荐历史列表，用于排除重复推荐
        enable_diversity: 是否启用多样性控制
        enable_correlation: 是否启用到相关惩罚
    
    返回：
        排序后的推荐列表 [(权重，号码), ...]
    """
    pool = []
    for a, b, c in product(range(10), repeat=3):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        pool.append((w, f"{a}{b}{c}"))
    
    # Top50 随机扰动：避免同分号长期霸榜
    if apply_noise:
        # 对 Top50 应用随机噪声
        top50_count = min(50, len(pool))
        for i in range(top50_count):
            noise = random.uniform(-RANDOM_NOISE, RANDOM_NOISE)
            pool[i] = (pool[i][0] + noise, pool[i][1])
    
    # 重新排序（应用噪声后）
    pool.sort(key=lambda x: -x[0])
    
    # 最近5期排除机制：对重复推荐进行惩罚
    if recent_recommendations:
        pool = recent_recommend_penalty(pool, recent_recommendations)
        # 重新排序（应用惩罚后）
        pool.sort(key=lambda x: -x[0])
    
    # 冷热平衡模型：确保推荐池包含 40% 热号、40% 温号、20% 冷号
    if enable_cold_hot_balance:
        numbers = meta.get("numbers", [])
        if len(numbers) >= HOT_WINDOW:
            hot_digits, warm_digits, cold_digits = classify_digits_by_hot(numbers, HOT_WINDOW)
            
            # 计算各类别需要的号码数量
            hot_needed = max(1, int(top_n * HOT_RATIO))
            warm_needed = max(1, int(top_n * WARM_RATIO))
            cold_needed = max(1, int(top_n * COLD_RATIO))
            
            # 从各类别中选取最佳组合
            hot_pool = []
            warm_pool = []
            cold_pool = []
            
            for w, num_str in pool:
                digits = set(int(c) for c in num_str)
                hot_count = len(digits & set(hot_digits))
                warm_count = len(digits & set(warm_digits))
                cold_count = len(digits & set(cold_digits))
                
                # 根据组合中冷热号的比例分类
                if hot_count >= 2:
                    hot_pool.append((w, num_str))
                elif cold_count >= 1 and warm_count >= 1:
                    cold_pool.append((w, num_str))
                else:
                    warm_pool.append((w, num_str))
            
            # 合并并重新排序
            balanced_pool = []
            balanced_pool.extend(sorted(hot_pool, key=lambda x: -x[0])[:hot_needed])
            balanced_pool.extend(sorted(warm_pool, key=lambda x: -x[0])[:warm_needed])
            balanced_pool.extend(sorted(cold_pool, key=lambda x: -x[0])[:cold_needed])
            
            # 如果平衡池不足，从原池补充
            if len(balanced_pool) < top_n:
                remaining = [item for item in pool if item not in balanced_pool]
                balanced_pool.extend(remaining[:top_n - len(balanced_pool)])
            
            pool = balanced_pool[:top_n]
    
    # 探索机制：15%概率从 Top50 中随机选择，85%概率选择最高分
    if enable_exploration and random.random() < EXPLORATION_RATE:
        # 探索模式：从 Top50 中随机抽取
        top_50 = pool[:50] if len(pool) >= 50 else pool
        # 确保至少返回 top_n 个
        if len(top_50) >= top_n:
            # 随机打乱后取前 top_n 个
            random.shuffle(top_50)
            return top_50[:top_n]
        else:
            # 如果候选不足，返回全部
            return top_50
    
    # 正常模式：返回最高分的 top_n 个
    result = pool[:top_n]
    
    # 推荐池多样性控制：最大化数字覆盖率
    if enable_diversity and len(result) >= 5:
        result = diversity_filter(result, top_n)
    
    # 推荐号码去相关：减少高度相关推荐
    if enable_correlation and len(result) >= 2:
        result = correlation_penalty(result)
    
    return result


def _meta_from_raw(meta_raw, tail_top=5):
    return {
        **meta_raw,
        "hot_sum_set": set(meta_raw["hot_sums"]),
        "hot_span_set": set(meta_raw["hot_spans"]),
        "sum_tail_top": {t for t, _ in meta_raw["sum_tail_freq"].most_common(tail_top)},
    }


def build_ranking_meta(numbers, window_weights, sums=None, spans=None, tail_top=5):
    """和值/跨度 + 模式 + 上期→本期转移，供直选排序使用"""
    if sums is None:
        sums = [sum(x) for x in numbers]
    if spans is None:
        spans = [calc_span(x) for x in numbers]
    meta = _meta_from_raw(ensemble_sum_span(sums, spans, window_weights), tail_top=tail_top)
    pat = ensemble_patterns(numbers, window_weights)
    meta.update(pat)
    lag1 = ensemble_lag1_dynamics(numbers, window_weights)
    meta["lag1"] = lag1
    meta["dynamic"] = derive_dynamic_weights(lag1, pat["consec_rate"])
    meta["last_draw"] = numbers[-1]
    meta["numbers"] = numbers  # 用于冷热平衡模型
    
    # 和值趋势模型：动态调整和值中心
    adjusted_sum_center, trend_direction = sum_trend_model(numbers, SUM_TREND_WINDOW)
    meta["sum_center"] = adjusted_sum_center
    meta["sum_trend"] = trend_direction
    
    return meta


def backtest(numbers, trials=BACKTEST_TRIALS, window_weights=None):
    max_w = max(RECENT_WINDOWS)
    if len(numbers) < trials + max_w + 5:
        trials = max(20, len(numbers) - max_w - 5)

    hit_top = hit_top3 = hit_ge2 = hit_sum_band = hit_zu6_pool = hit_zu6_four = zu6_draws = 0
    start = len(numbers) - trials
    ww = window_weights or compute_window_weights(numbers[:start])[0]

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta = build_ranking_meta(train, ww, sums, spans, tail_top=4)
        sc, _ = ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        dan, _, kill, _ = pick_dan_tuo_kill(sc, enable_danma_random=False)
        top = rank_triplets(sc, dan, kill, meta, top_n=RECOMMEND_GROUPS, enable_exploration=False)
        top_nums = [t[1] for t in top]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"

        if act_s in top_nums:
            hit_top += 1
        if act_s in top_nums[:ZHIXUAN_TOP3]:
            hit_top3 += 1
        pred_digits = {int(ch) for s in top_nums for ch in s}
        if len(pred_digits & set(actual)) >= 2:
            hit_ge2 += 1
        if abs(sum(actual) - meta["sum_center"]) <= 4:
            hit_sum_band += 1

        z6 = pick_zu6_pool(sc, kill)
        z4 = pick_zu6_four(sc, kill)
        if is_zu6_draw(actual):
            zu6_draws += 1
            if set(actual).issubset(set(z6)):
                hit_zu6_pool += 1
            if set(actual).issubset(set(z4)):
                hit_zu6_four += 1

    n = trials
    return {
        "trials": n,
        "top_hit": hit_top,
        "top_rate": hit_top / n,
        "top_rate_baseline": RECOMMEND_GROUPS / 1000.0,
        "top3_hit": hit_top3,
        "top3_rate": hit_top3 / n,
        "top3_rate_baseline": ZHIXUAN_TOP3 / 1000.0,
        "recommend_groups": RECOMMEND_GROUPS,
        "ge2_digit_rate": hit_ge2 / n,
        "sum_band_rate": hit_sum_band / n,
        "zu6_draws": zu6_draws,
        "zu6_pool_hit": hit_zu6_pool,
        "zu6_pool_rate": hit_zu6_pool / zu6_draws if zu6_draws else 0.0,
        "zu6_four_hit": hit_zu6_four,
        "zu6_four_rate": hit_zu6_four / zu6_draws if zu6_draws else 0.0,
    }


def permutation_test(numbers, observed_rate, trials=BACKTEST_TRIALS,
                     window_weights=None, shuffles=PERMUTATION_SHUFFLES, seed=20):
    """打乱历史顺序重跑回测，估计直选命中率优于随机的显著性。

    福彩3D 为独立均匀摇奖，期间无时序可学。若打乱顺序后命中率不降，
    说明模型未抓到真实信号；p 值为打乱样本命中率 >= 实际命中率的比例。
    """
    seq = [list(n) for n in numbers]
    rng = random.Random(seed)
    perm_rates = []
    for _ in range(shuffles):
        rng.shuffle(seq)
        perm_rates.append(backtest(seq, trials=trials, window_weights=window_weights)["top_rate"])
    ge = sum(1 for r in perm_rates if r >= observed_rate)
    mean = sum(perm_rates) / len(perm_rates) if perm_rates else 0.0
    pvalue = (ge + 1) / (shuffles + 1)
    return {
        "shuffles": shuffles,
        "observed_rate": observed_rate,
        "shuffled_mean_rate": mean,
        "shuffled_max_rate": max(perm_rates) if perm_rates else 0.0,
        "baseline_rate": RECOMMEND_GROUPS / 1000.0,
        "pvalue": pvalue,
        "significant": pvalue < 0.05,
    }


def backtest_objective(bt, metric="top3_rate"):
    """从回测结果提取优化目标"""
    if metric == "composite":
        return (
            0.55 * bt["top3_rate"]
            + 0.30 * bt["top_rate"]
            + 0.15 * bt["ge2_digit_rate"]
        )
    if metric not in bt:
        raise ValueError(f"未知 metric: {metric}")
    return bt[metric]


def evaluate_weights(
    numbers,
    weights,
    trials=60,
    window_weights=None,
    metric="top3_rate",
    recompute_window_weights=False,
):
    """给定权重在历史数据上跑滚动回测，返回 (目标值, 回测详情)"""
    with patch_weights(weights):
        if recompute_window_weights or window_weights is None:
            ww, _ = compute_window_weights(
                numbers, trials=min(WINDOW_BACKTEST_TRIALS, max(20, trials // 2))
            )
        else:
            ww = window_weights
        bt = backtest(numbers, trials=trials, window_weights=ww)
    return backtest_objective(bt, metric), bt


def _sample_random_weights(base, rng):
    """在默认权重附近随机采样一组候选参数"""
    candidate = {}
    for k in TUNABLE_WEIGHTS:
        lo, hi = WEIGHT_SEARCH_RANGES.get(k, (0.5, 2.0))
        if k.endswith("_SIGMA"):
            candidate[k] = rng.uniform(lo, hi)
        else:
            candidate[k] = base[k] * rng.uniform(lo, hi)
    return candidate


def _mutate_weights(weights, base, rng, scale=0.15):
    """在最优解附近做局部扰动"""
    candidate = dict(weights)
    k = rng.choice(TUNABLE_WEIGHTS)
    lo, hi = WEIGHT_SEARCH_RANGES.get(k, (0.5, 2.0))
    if k.endswith("_SIGMA"):
        delta = (hi - lo) * scale * rng.uniform(-1, 1)
        candidate[k] = max(lo, min(hi, candidate[k] + delta))
    else:
        candidate[k] = max(0.1, candidate[k] * (1 + scale * rng.uniform(-1, 1)))
    return candidate


def search_weights(
    numbers=None,
    iterations=80,
    backtest_trials=60,
    metric="top3_rate",
    seed=42,
    refine_rounds=30,
    verbose=True,
):
    """
    随机搜索 + 局部 refine，最大化历史回测命中率。

    metric: top3_rate | top_rate | ge2_digit_rate | composite
    返回 dict：baseline / best / improvement / history
    """
    if numbers is None:
        numbers = [x[2] for x in fetch_data()]
    if not numbers:
        return {"error": "未获取到数据"}

    rng = random.Random(seed)
    base = default_weights()
    fixed_ww, _ = compute_window_weights(numbers)

    if verbose:
        print(f"参数搜索: {iterations} 次随机采样 + {refine_rounds} 次局部 refine")
        print(f"回测期数={backtest_trials}, 目标={metric}, 窗口权重固定（加速搜索）")

    _, baseline_bt = evaluate_weights(
        numbers, base, trials=backtest_trials, window_weights=fixed_ww, metric=metric
    )
    baseline_score = backtest_objective(baseline_bt, metric)
    best_weights = dict(base)
    best_score = baseline_score
    best_bt = baseline_bt
    history = []

    for i in range(iterations):
        candidate = _sample_random_weights(base, rng)
        score, bt = evaluate_weights(
            numbers, candidate, trials=backtest_trials, window_weights=fixed_ww, metric=metric
        )
        history.append({"phase": "random", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [random {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    for i in range(refine_rounds):
        candidate = _mutate_weights(best_weights, base, rng)
        score, bt = evaluate_weights(
            numbers, candidate, trials=backtest_trials, window_weights=fixed_ww, metric=metric
        )
        history.append({"phase": "refine", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [refine {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    return {
        "metric": metric,
        "backtest_trials": backtest_trials,
        "baseline": {"weights": base, "score": baseline_score, "backtest": baseline_bt},
        "best": {"weights": best_weights, "score": best_score, "backtest": best_bt},
        "improvement": best_score - baseline_score,
        "history_len": len(history),
    }


def print_search_report(result):
    """打印权重搜索结果"""
    if result.get("error"):
        print(result["error"])
        return

    base_w = result["baseline"]["weights"]
    best_w = result["best"]["weights"]
    base_bt = result["baseline"]["backtest"]
    best_bt = result["best"]["backtest"]

    print("\n" + "=" * 70)
    print("【评分权重搜索】")
    print("=" * 70)
    print(f"  目标指标: {result['metric']}  |  回测期数: {result['backtest_trials']}")
    print(f"  基线 {result['baseline']['score'] * 100:.2f}%  →  最优 {result['best']['score'] * 100:.2f}%  "
          f"(+{result['improvement'] * 100:.2f}%)")

    print("\n  回测对比:")
    for label, bt in ("基线", base_bt), ("最优", best_bt):
        print(
            f"    {label}: Top3 {bt['top3_rate'] * 100:.1f}% ({bt['top3_hit']}/{bt['trials']})  "
            f"| Top{RECOMMEND_GROUPS} {bt['top_rate'] * 100:.1f}%  "
            f"| ≥2码 {bt['ge2_digit_rate'] * 100:.1f}%"
        )

    print("\n  权重变化 (默认 → 最优):")
    for k in TUNABLE_WEIGHTS:
        b, n = base_w[k], best_w[k]
        delta = ((n / b - 1) * 100) if b else 0
        print(f"    {k:16s}  {b:6.2f}  →  {n:6.2f}  ({delta:+.0f}%)")

    print("\n  可复制到 lottery3d.py 顶部:")
    for k in TUNABLE_WEIGHTS:
        v = best_w[k]
        fmt = f"{v:.2f}" if isinstance(v, float) and not v.is_integer() else str(int(v) if v == int(v) else v)
        print(f"    {k} = {fmt}")


def _transition_for_api(lag1, dynamic, pos_names=("百", "十", "个")):
    """序列化上期→本期转移统计与动态权重"""
    dyn_out = {}
    for k, v in dynamic.items():
        if isinstance(v, list):
            dyn_out[k] = [round(x, 3) for x in v]
        else:
            dyn_out[k] = round(v, 3)
    return {
        "pairs_analyzed": lag1["pairs"],
        "pos_repeat_rate": [
            {
                "name": pos_names[i],
                "rate": round(lag1["pos_repeat_rate"][i], 4),
                "vs_random": round(lag1["pos_repeat_rate"][i] / RANDOM_POS_REPEAT, 2),
            }
            for i in range(3)
        ],
        "repeat_dist": {f"{k}位同": round(v * 100, 1) for k, v in lag1["repeat_dist"].items()},
        "digit_reuse_rate": round(lag1["digit_reuse_rate"], 4),
        "full_repeat_rate": round(lag1["full_repeat_rate"], 4),
        "same_set_rate": round(lag1["same_set_rate"], 4),
        "ge2_overlap_rate": round(lag1["ge2_overlap_rate"], 4),
        "dynamic": dyn_out,
    }


def run_prediction(data=None, force_refresh=False, enable_backtest=False, enable_permutation=False, compute_weights=False):
    """运行预测，返回 JSON 可序列化 dict；data 为 None 时自动抓取。
    
    Args:
        data: 可选的数据列表，如果为 None 则自动抓取
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
        enable_backtest: 是否启用回测（默认 False，大幅提升速度）
        enable_permutation: 是否启用排列测试（默认 False，仅在 enable_backtest=True 时生效）
        compute_weights: 是否重新计算窗口权重（默认 False，使用缓存或默认权重，提升速度）
    """
    global _prediction_cache, _cache_time
    
    # 检查缓存（按自然天判断）
    if not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            log.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            log.info("缓存已过期（非今日数据），重新抓取")
    
    try:
        if data is None:
            data = fetch_data()
    except Exception:
        log.error('3D 数据抓取失败', exc_info=True)
        return {'error': '数据抓取失败'}
    if not data:
        return {"error": "未获取到数据"}

    periods = [x[0] for x in data]
    numbers = [x[2] for x in data]
    sums = [sum(x) for x in numbers]
    spans = [calc_span(x) for x in numbers]

    # 窗口权重计算（可选择跳过以提升速度）
    if compute_weights:
        window_weights, window_scores = compute_window_weights(numbers)
    else:
        window_weights, window_scores = default_window_weights(), {}
    
    meta_raw = ensemble_sum_span(sums, spans, window_weights)
    meta = build_ranking_meta(numbers, window_weights, sums, spans, tail_top=5)
    pat = {k: meta[k] for k in ("consec_rate", "oe_freq", "bs_freq", "oe_total", "bs_total")}

    score, freq_all = ensemble_digit_scores(numbers, window_weights, dynamic=meta.get("dynamic"))
    danma, tuoma, kill, rank = pick_dan_tuo_kill(score, enable_danma_random=True)
    form_prob = analyze_form_probability(numbers, window_weights=window_weights)
    zu6_four = pick_zu6_four(score, kill)
    _, z6_straight = zu6_notes_from_digits(zu6_four)
    zhixuan_top = rank_triplets(score, danma, kill, meta, top_n=RECOMMEND_GROUPS, enable_exploration=True, apply_noise=True)
    
    # 可选：回测分析（耗时操作）
    bt = None
    if enable_backtest:
        bt = backtest(numbers, window_weights=window_weights)
        if enable_permutation:
            bt["significance"] = permutation_test(
                numbers, bt["top_rate"], window_weights=window_weights
            )

    last_num = numbers[-1]
    pos_names = ("百", "十", "个")
    position_top = []
    for pos, name in enumerate(pos_names):
        pr = sorted(enumerate(ensemble_position_digit_scores(numbers, pos, window_weights, dynamic=meta.get("dynamic"))), key=lambda x: -x[1])[:5]
        position_top.append({
            "name": name,
            "digits": [{"digit": d, "score": round(s, 1)} for d, s in pr],
        })

    miss_global = []
    for d in range(10):
        mv = miss_value(numbers, d)
        if mv >= 8:
            miss_global.append({"digit": d, "miss": mv})
    miss_global.sort(key=lambda x: -x["miss"])

    miss_position = []
    for pos, name in enumerate(pos_names):
        top = sorted(range(10), key=lambda x: -miss_value(numbers, x, position=pos))[:3]
        miss_position.append({
            "name": name,
            "digits": [{"digit": d, "miss": miss_value(numbers, d, position=pos)} for d in top],
        })

    sum_tails = [{"tail": t, "count": c} for t, c in meta_raw["sum_tail_freq"].most_common(5)]

    result = {
        "period": periods[-1],
        "total_periods": len(numbers),
        "avg_sum": round(sum(sums) / len(sums), 2),
        "last_draw": "".join(map(str, last_num)),
        "neighbors": sorted(set().union(*[neighbor(d) for d in last_num])),
        "hot_digits": [{"digit": d, "weight": round(c, 1)} for d, c in freq_all.most_common(5)],
        "danma": danma,
        "tuoma": tuoma,
        "kill": kill,
        "rank_top10": [{"digit": d, "score": round(s, 1)} for d, s in rank[:10]],
        "position_top": position_top,
        "miss_global": miss_global,
        "miss_position": miss_position,
        "sum_tails": sum_tails,
        "recommend_groups": RECOMMEND_GROUPS,
        "recent_windows": list(RECENT_WINDOWS),
        "window_weights": {str(k): round(v, 4) for k, v in window_weights.items()},
        "window_scores": window_scores,
        "sum_span": {
            "sum_center": round(meta["sum_center"], 1),
            "hot_sums": meta["hot_sums"],
            "span_center": round(meta["span_center"], 1),
            "hot_spans": meta["hot_spans"],
        },
        "patterns": {
            "consecutive_rate": round(pat["consec_rate"], 4),
            "odd_even_top": [
                {"label": ratio_label(k, "oe"), "weight": round(v, 2)}
                for k, v in pat["oe_freq"].most_common(4)
            ],
            "big_small_top": [
                {"label": ratio_label(k, "bs"), "weight": round(v, 2)}
                for k, v in pat["bs_freq"].most_common(4)
            ],
            "last_odd_even": ratio_label(odd_even_key(last_num), "oe"),
            "last_big_small": ratio_label(big_small_key(last_num), "bs"),
            "last_has_consecutive": has_consecutive_digits(*last_num),
        },
        "transition": _transition_for_api(meta["lag1"], meta["dynamic"], pos_names),
        "form": {
            "last_label": FORM_LABELS[form_prob["last_form"]],
            "streak": form_prob["streak"],
            "miss_zu6": form_prob["miss_zu6"],
            "miss_zu3": form_prob["miss_zu3"],
            "recent": {k: round(v, 4) for k, v in form_prob["recent_p"].items()},
            "hist": {k: round(v, 4) for k, v in form_prob["hist_p"].items()},
            "markov": {k: round(v, 4) for k, v in form_prob["markov_p"].items()},
            "blend": {k: round(v, 4) for k, v in form_prob["blend_p"].items()},
            "theory": THEORY_FORM_P,
            "markov_samples": int(form_prob["markov_samples"]),
        },
        "zu6_four": {
            "digits_str": "".join(map(str, zu6_four)),
            "combos": z6_straight,
        },
        "zhixuan_top3": [{"num": num, "score": round(w, 1)} for w, num in zhixuan_top[:ZHIXUAN_TOP3]],
        "zhixuan": [{"num": num, "score": round(w, 1)} for w, num in zhixuan_top],
    }
    
    # 可选：添加回测结果
    if bt is not None:
        result["backtest"] = bt
    
    # 保存到缓存
    _prediction_cache = result
    _cache_time = time.time()
    log.info("预测结果已缓存")
    
    return result


def print_report(result):
    """终端格式化输出"""
    if result.get("error"):
        print(result["error"])
        return

    form = result["form"]
    lf = form["last_label"]
    z6 = result["zu6_four"]

    print("\n" + "=" * 70)
    print("【本期摘要】")
    print("=" * 70)
    print(f"  上期 {result['period']} 期: {result['last_draw']}  ({lf}，连出 {form['streak']} 期)")
    print(f"  形态预估 → 组六 {form['blend']['zu6']*100:.1f}%  |  组三 {form['blend']['zu3']*100:.1f}%  |  豹子 {form['blend']['baozi']*100:.1f}%")
    print(f"  组六四码 → {z6['digits_str']}  (覆盖: {', '.join(z6['combos'])})")
    if result["zhixuan_top3"]:
        top3 = ", ".join(x["num"] for x in result["zhixuan_top3"])
        print(f"  直选Top3 → {top3}")

    ww = result.get("window_weights", {})
    ws = result.get("window_scores", {})
    if ww:
        parts = [
            f"{k}期权重{float(ww[k])*100:.0f}%"
            + (f"(得分{ws.get(int(k), ws.get(k))})" if ws.get(int(k), ws.get(k)) is not None else "")
            for k in ww
        ]
        print(f"  动态窗口集成: {', '.join(parts)}")

    print("\n" + "=" * 70)
    print(f"热号分析（多窗口集成 {list(result.get('recent_windows', RECENT_WINDOWS))}）")
    print("=" * 70)
    for item in result["hot_digits"]:
        print(f"  热号 {item['digit']} -> 加权{item['weight']:.1f}")

    print("\n遗漏分析（分位+全局）")
    for item in result.get("miss_global", []):
        print(f"  数字{item['digit']} 全局遗漏{item['miss']}期")
    for block in result.get("miss_position", []):
        for item in block["digits"]:
            print(f"  {block['name']}位 数字{item['digit']} 遗漏{item['miss']}期")

    print("\n上期号码:", result["last_draw"])
    print("邻号:", result["neighbors"])

    print("\n" + "=" * 70)
    print("【本期形态概率】（组六 / 组三 / 豹子）")
    print("=" * 70)
    print(f"  上期形态: {lf}（已连续 {form['streak']} 期）")
    print(f"  形态遗漏: 组六 {form['miss_zu6']} 期  |  组三 {form['miss_zu3']} 期")
    print(f"  近态(多窗口集成): 组六 {form['recent']['zu6']*100:.1f}%  "
          f"组三 {form['recent']['zu3']*100:.1f}%  "
          f"豹子 {form['recent']['baozi']*100:.1f}%")
    print(
        f"  上期{lf}→下期(样本{form['markov_samples']}): "
        f"组六 {form['markov']['zu6']*100:.1f}%  "
        f"组三 {form['markov']['zu3']*100:.1f}%  "
        f"豹子 {form['markov']['baozi']*100:.1f}%"
    )
    print("  综合预估(近态+转移+历史+理论):")
    print(f"    ★ 组六 {form['blend']['zu6']*100:.1f}%  "
          f"★ 组三 {form['blend']['zu3']*100:.1f}%  "
          f"  豹子 {form['blend']['baozi']*100:.1f}%")
    print(f"  理论基准: 组六 {form['theory']['zu6']*100:.0f}%  "
          f"组三 {form['theory']['zu3']*100:.0f}%  "
          f"豹子 {form['theory']['baozi']*100:.0f}%")

    ss = result["sum_span"]
    print("\n和值/跨度（软约束中心）")
    print(f"  和值中心 {ss['sum_center']}，推荐区间 {ss['hot_sums']}")
    print(f"  跨度中心 {ss['span_center']}，推荐 {ss['hot_spans']}")
    if result.get("sum_tails"):
        print("  和值尾TOP5:", [(x["tail"], x["count"]) for x in result["sum_tails"]])

    pat = result.get("patterns")
    if pat:
        print("\n模式特征（连号 / 奇偶 / 大小 / 同位复刻）")
        print(f"  近态连号占比: {pat['consecutive_rate']*100:.1f}%")
        print(f"  上期: {pat['last_odd_even']} · {pat['last_big_small']}"
              f"{' · 含连号' if pat['last_has_consecutive'] else ''}")
        oe_top = ", ".join(f"{x['label']}({x['weight']})" for x in pat.get("odd_even_top", [])[:3])
        bs_top = ", ".join(f"{x['label']}({x['weight']})" for x in pat.get("big_small_top", [])[:3])
        print(f"  热门奇偶比: {oe_top}")
        print(f"  热门大小比: {bs_top}")

    tr = result.get("transition")
    if tr:
        print("\n上期→本期转移（近{}对，动态调权）".format(tr["pairs_analyzed"]))
        pos_line = "  ".join(
            f"{x['name']}位同位复刻 {x['rate']*100:.1f}%（随机10%，×{x['vs_random']:.2f}）"
            for x in tr["pos_repeat_rate"]
        )
        print(f"  {pos_line}")
        dist = ", ".join(f"{k} {v}%" for k, v in tr.get("repeat_dist", {}).items())
        print(f"  同位个数分布: {dist}")
        print(f"  重号出现率 {tr['digit_reuse_rate']*100:.1f}%（随机27%）"
              f"  |  全同号 {tr['full_repeat_rate']*100:.2f}%  |  同号不同序 {tr['same_set_rate']*100:.2f}%")
        dyn = tr.get("dynamic", {})
        print(f"  动态权重: 同位复刻 {dyn.get('w_pos_repeat', W_POS_REPEAT):.2f}"
              f"  上期重号 {dyn.get('w_last_appear', W_LAST_APPEAR):.2f}"
              f"  全同惩罚 -{dyn.get('w_full_repeat_penalty', 0):.1f}"
              f"  同集惩罚 -{dyn.get('w_same_set_penalty', 0):.1f}")

    print("\n综合评分 TOP10")
    for item in result["rank_top10"]:
        print(f"  {item['digit']}: {item['score']:.1f}分")

    print("\n分位推荐（各位 Top5）")
    for block in result["position_top"]:
        print(f"  {block['name']}位:", [f"{x['digit']}({x['score']:.0f})" for x in block["digits"]])

    print("\n" + "=" * 70)
    print("【组六四码推荐】（选 4 个号打组六复式即可）")
    print("=" * 70)
    print("  投注号码:", z6["digits_str"])
    print("  杀码参考:", result["kill"], "（四码中已尽量避开）")
    print("  覆盖 4 注组六:", ", ".join(z6["combos"]))

    print("\n" + "=" * 70)
    print("【直选Top3推荐】（百十个位顺序一致）")
    print("=" * 70)
    for idx, item in enumerate(result.get("zhixuan_top3", []), start=1):
        print(f"  {idx}. {item['num']}  评分={item['score']:.1f}")

    print("\n" + "=" * 70)
    print(f"【直选推荐 {RECOMMEND_GROUPS} 注】（百十个位顺序一致）")
    print("=" * 70)
    print("  杀码参考:", result["kill"], f"（含杀码组合每码 -{W_KILL_PENALTY} 分降权）")
    print("-" * 70)
    for idx, item in enumerate(result["zhixuan"], start=1):
        print(f"  {idx:02d}. {item['num']}  评分={item['score']:.1f}")

    bt = result["backtest"]
    print("\n" + "=" * 70)
    print("滚动回测（仅供参考）")
    print("=" * 70)
    print(f"  回测期数: {bt['trials']}")
    print(f"  直选{RECOMMEND_GROUPS}注命中: {bt['top_rate']*100:.1f}%  ({bt['top_hit']}/{bt['trials']})")
    print(f"  直选Top3命中: {bt['top3_rate']*100:.1f}%  ({bt['top3_hit']}/{bt['trials']})")
    print(f"  推荐池至少中2个数字: {bt['ge2_digit_rate']*100:.1f}%")
    print(f"  和值落在预测带±4: {bt['sum_band_rate']*100:.1f}%")
    if bt["zu6_draws"]:
        print(
            f"  组六四码命中(开奖为组六时): {bt['zu6_four_rate']*100:.1f}%  "
            f"({bt['zu6_four_hit']}/{bt['zu6_draws']})"
        )
        print(
            f"  (参考)五码组六池含开奖三码: {bt['zu6_pool_rate']*100:.1f}%  "
            f"({bt['zu6_pool_hit']}/{bt['zu6_draws']})"
        )

    print("\n统计信息")
    print("  总期数:", result["total_periods"])
    print("  最近一期:", result["period"])
    print("  平均和值:", result["avg_sum"])
    print("\n  说明: 3D 开奖具有随机性，回测用于观察候选池收缩效果，不构成投注建议。")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="福彩3D预测器 V3.1+")
    parser.add_argument(
        "--search-weights",
        action="store_true",
        help="在历史数据上搜索最优评分权重（随机搜索+局部 refine）",
    )
    parser.add_argument("--search-iters", type=int, default=80, help="随机搜索次数")
    parser.add_argument("--search-refine", type=int, default=30, help="局部 refine 次数")
    parser.add_argument("--search-trials", type=int, default=60, help="每次评估的回测期数")
    parser.add_argument(
        "--search-metric",
        default="top3_rate",
        choices=("top3_rate", "top_rate", "ge2_digit_rate", "composite"),
        help="优化目标",
    )
    parser.add_argument("--search-seed", type=int, default=42, help="随机种子")
    args = parser.parse_args(argv)

    print("抓取数据中...")
    data = fetch_data()
    numbers = [x[2] for x in data] if data else []

    if args.search_weights:
        if not numbers:
            print("未获取到数据")
            return
        result = search_weights(
            numbers=numbers,
            iterations=args.search_iters,
            backtest_trials=args.search_trials,
            metric=args.search_metric,
            seed=args.search_seed,
            refine_rounds=args.search_refine,
        )
        print_search_report(result)
        return

    print_report(run_prediction(data))


if __name__ == "__main__":
    main()
