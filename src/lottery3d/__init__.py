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
from .advanced import (
    WINDOW_CANDIDATES,
    detect_position_cycles,
    transfer_entropy_probs_simple,
)

log = setup_logger('lottery3d')

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

URL = "https://www.8300.cn/kjhhis/3/200.html"

RECENT_WINDOWS = (30, 45, 60, 90)  # 兼容旧逻辑；运行时由回测动态选择
RECENT_WINDOW = 90
DYNAMIC_WINDOW_TOP_K = 3  # 回测选最强的 K 个窗口集成
WINDOW_BACKTEST_TRIALS = 40
EXP_DECAY = 0.96
BACKTEST_TRIALS = 80
PERMUTATION_SHUFFLES = 20  # 置换检验打乱次数，评估命中率是否显著优于随机

# 缓存配置
CACHE_EXPIRE_SECONDS = 86400  # 24小时缓存过期时间（当天有效）
_prediction_cache = None
_cache_time = 0

W_HOT_GLOBAL = 4.0
W_HOT_POS = 3.0
W_MISS_HIGH = 8.0
W_MISS_MID = 4.5
W_MARKOV = 3.5

# 冷热平衡评分：最终得分 = 0.35热度 + 0.25马尔可夫 + 0.25遗漏 + 0.15周期
BALANCE_W_HOT = 0.35
BALANCE_W_MARKOV = 0.25
BALANCE_W_MISS = 0.25
BALANCE_W_CYCLE = 0.15
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

# 多样性增强配置
DIVERSITY_NOISE = 0.15  # 随机噪声幅度（0-1之间，0表示无噪声）
DIVERSITY_COOL_BONUS = 1.5  # 冷门号码额外加分系数
USE_DIVERSITY = True  # 是否启用多样性增强

# 推荐池去重与探索
POOL_DEDUP_ENABLED = True
POOL_MIN_HAMMING = 2          # 与已选号码至少相差位数
POOL_MAX_SAME_MULTISORT = 2   # 同号集（组选等价）最多保留注数
POOL_POS_DIGIT_CAP = 6        # 每位同一数字在推荐池中最多出现次数
EXPLORE_RATIO = 0.27          # 探索槽位占比（约 4/15 注）
EXPLORE_NOVELTY_BONUS = 3.0   # 探索阶段新颖度加分

# 推荐注数（直选为带顺序的三位数）
RECOMMEND_GROUPS = 15
ZHIXUAN_TOP3 = 3
ZU6_POOL_SIZE = 5
ZU6_FOUR_SIZE = 4

# 分位独立 + Monte Carlo 采样（不再枚举固定笛卡尔积）
POSITION_CANDIDATE_N = 10       # 候选扩容：各位 Top10 按权重抽样
POSITION_SAMPLE_N = 5           # 分位独立采样：各位 Top5 概率抽样
MONTE_CARLO_TRIALS = 5000       # MC 随机生成组数
MONTE_CARLO_KEEP = 100          # MC 评分后保留 Top
POSITION_SAMPLE_TRIALS = 1000   # 分位 Top5 概率抽样次数
BLEND_MAIN_WEIGHT = 0.8         # 主模型权重
BLEND_CONTRARIAN_WEIGHT = 0.2   # 反相关冷门模型权重
# 兼容旧字段名
POSITION_TOP_N = POSITION_SAMPLE_N
POSITION_COMBO_POOL = MONTE_CARLO_KEEP + POSITION_SAMPLE_TRIALS

# 马尔可夫转移：拉普拉斯平滑系数 α（加法平滑，α=1 即标准 Laplace）
MARKOV_LAPLACE_ALPHA = 1.0

# 贝叶斯冷热周期：近窗样本 + Beta(2,2) 先验
CYCLE_WINDOW = 20
CYCLE_FREQ_WINDOW = 30
CYCLE_HOT_TOP = 4
CYCLE_COLD_MISS = 12
CYCLE_BETA_PRIOR = 2.0
CYCLE_P_HOT_CAP = 0.85  # 热周期置信上限，防止单边碾压
CYCLE_P_HOT_FLOOR = 0.35

# 可调评分权重（供 search_weights 搜索）
TUNABLE_WEIGHTS = (
    "W_HOT_GLOBAL",
    "W_HOT_POS",
    "W_MISS_MID",
    "W_MARKOV",
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


def build_second_order_markov(numbers, position):
    """二阶马尔可夫：(前两期, 前一期) → 下一期，如 (3,7) → 6"""
    trans = defaultdict(Counter)
    for i in range(len(numbers) - 2):
        state = (numbers[i][position], numbers[i + 1][position])
        nxt = numbers[i + 2][position]
        trans[state][nxt] += 1
    return trans


def second_order_markov_probs(numbers, position, states=None):
    """P(next | prev2, prev1)，由转移熵过滤替代纯马尔可夫"""
    return transfer_entropy_probs_simple(numbers, position, states=states)


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


def bayesian_hot_cold_cycle(numbers, window=CYCLE_WINDOW):
    """贝叶斯推断当前处于热周期还是冷周期，用于动态缩放热号/冷号权重"""
    if len(numbers) < window + 3:
        return {
            "phase": "neutral",
            "p_hot": 0.5,
            "hot_hits": 0,
            "cold_hits": 0,
            "w_hot_scale": 1.0,
            "w_miss_scale": 1.0,
        }

    hot_hits = cold_hits = 0
    start = len(numbers) - window
    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        recent = train[-CYCLE_FREQ_WINDOW:] if len(train) >= CYCLE_FREQ_WINDOW else train
        freq = exp_weighted_counts([d for n in recent for d in n])
        hot_set = {d for d, _ in freq.most_common(CYCLE_HOT_TOP)}
        for d in actual:
            if d in hot_set:
                hot_hits += 1
            if miss_value(train, d) >= CYCLE_COLD_MISS:
                cold_hits += 1

    alpha = CYCLE_BETA_PRIOR
    denom = 2 * alpha + hot_hits + cold_hits
    p_hot_raw = (alpha + hot_hits) / denom if denom else 0.5
    p_hot = _clamp(p_hot_raw, CYCLE_P_HOT_FLOOR, CYCLE_P_HOT_CAP)

    if p_hot >= 0.58:
        phase = "hot"
    elif p_hot <= 0.42:
        phase = "cold"
    else:
        phase = "neutral"

    w_hot_scale = _clamp(0.75 + 0.5 * p_hot, 0.75, 1.25)
    w_miss_scale = _clamp(0.75 + 0.5 * (1.0 - p_hot), 0.75, 1.25)

    return {
        "phase": phase,
        "p_hot": p_hot,
        "p_hot_raw": p_hot_raw,
        "hot_hits": hot_hits,
        "cold_hits": cold_hits,
        "w_hot_scale": w_hot_scale,
        "w_miss_scale": w_miss_scale,
    }


def derive_dynamic_weights(lag1, consec_rate, cycle=None):
    """根据历史转移统计 + 贝叶斯冷热周期动态缩放评分权重"""
    avg_rep = lag1["avg_pos_repeat"]
    w_pos = W_POS_REPEAT * _clamp(avg_rep / RANDOM_POS_REPEAT, 0.2, 1.6)
    pos_mult = [_clamp(r / RANDOM_POS_REPEAT, 0.3, 2.0) for r in lag1["pos_repeat_rate"]]
    w_last = W_LAST_APPEAR * _clamp(lag1["digit_reuse_rate"] / RANDOM_DIGIT_REUSE, 0.3, 1.4)
    consec_base = max(consec_rate, 0.15)
    w_consec = W_CONSECUTIVE * _clamp(consec_rate / consec_base, 0.6, 1.2)
    w_full_pen = _clamp(12.0 * (1.0 - lag1["full_repeat_rate"] * 80), 4.0, 15.0)
    w_perm_pen = _clamp(6.0 * (1.0 - lag1["same_set_rate"] * 40), 1.5, 8.0)

    cycle = cycle or {}
    w_hot_scale = cycle.get("w_hot_scale", 1.0)
    w_miss_scale = cycle.get("w_miss_scale", 1.0)

    return {
        "w_pos_repeat": w_pos,
        "pos_mult": pos_mult,
        "w_last_appear": w_last,
        "w_consecutive": w_consec,
        "w_full_repeat_penalty": w_full_pen,
        "w_same_set_penalty": w_perm_pen,
        "w_hot_global": W_HOT_GLOBAL * w_hot_scale,
        "w_hot_pos": W_HOT_POS * w_hot_scale,
        "w_miss_high": W_MISS_HIGH * w_miss_scale,
        "w_miss_mid": W_MISS_MID * w_miss_scale,
        "cycle_phase": cycle.get("phase", "neutral"),
        "p_hot_cycle": round(cycle.get("p_hot", 0.5), 4),
        "p_hot_cycle_raw": round(cycle.get("p_hot_raw", cycle.get("p_hot", 0.5)), 4),
        "w_hot_scale": round(w_hot_scale, 3),
        "w_miss_scale": round(w_miss_scale, 3),
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
        "hot_sums": [x for x, _ in hot_spans_vote.most_common(6)],
        "hot_spans": [x for x, _ in hot_spans_vote.most_common(4)],
        "sum_tail_freq": tail_acc,
    }


def _norm_scale_10(raw):
    """将分量原始值线性映射到 0–10，消除量纲失衡"""
    if not raw:
        return [5.0] * 10
    mn, mx = min(raw), max(raw)
    if mx - mn < 1e-9:
        return [5.0] * 10
    return [10.0 * (v - mn) / (mx - mn) for v in raw]


def _heat_raw(freq, digit, scale):
    mx = max(freq.values()) if freq else 1.0
    return (freq.get(digit, 0) / mx) * scale


def _markov_raw(probs, digit):
    """概率×10，不做 max 归一（避免塌缩为同分）"""
    return probs.get(digit, 0.1) * 10.0


def _miss_raw(miss_val, w_high):
    """遗漏连续梯度 0–10"""
    return min(miss_val / 15.0, 1.5) * (w_high / max(W_MISS_HIGH, 0.1)) * 6.0


def _cycle_raw(digit, dynamic, freq, miss_val):
    """周期连续梯度：热度亲和 + 遗漏亲和 按周期态混合"""
    phase = dynamic.get("cycle_phase", "neutral")
    p_hot = dynamic.get("p_hot_cycle", 0.5)
    mx = max(freq.values()) if freq else 1.0
    hot_aff = (freq.get(digit, 0) / mx) * 10.0
    cold_aff = min(miss_val / 18.0, 1.0) * 10.0
    if phase == "hot":
        blend = 0.5 + 0.35 * p_hot
        return blend * hot_aff + (1.0 - blend) * cold_aff
    if phase == "cold":
        blend = 0.5 + 0.35 * (1.0 - p_hot)
        return (1.0 - blend) * hot_aff + blend * cold_aff
    return 0.5 * hot_aff + 0.5 * cold_aff


def balance_digit_score(hot, markov, miss, cycle):
    """冷热平衡：各分量已归一化到 0–10 再加权"""
    return (
        BALANCE_W_HOT * hot
        + BALANCE_W_MARKOV * markov
        + BALANCE_W_MISS * miss
        + BALANCE_W_CYCLE * cycle
    )


def _balance_scores_normalized(numbers, window, dynamic, position=None):
    """计算 10 个号码的归一化平衡分"""
    dyn = dynamic or {}
    recent = _recent_slice(numbers, window)
    w_hot_global = dyn.get("w_hot_global", W_HOT_GLOBAL)
    w_hot_pos = dyn.get("w_hot_pos", W_HOT_POS)
    w_miss_high = dyn.get("w_miss_high", W_MISS_HIGH)

    hot_raw = [0.0] * 10
    markov_raw = [0.0] * 10
    miss_raw = [0.0] * 10
    cycle_raw = [0.0] * 10

    if position is not None:
        pos_freq = exp_weighted_counts([n[position] for n in recent])
        probs = second_order_markov_probs(numbers, position)
        for d in range(10):
            hot_raw[d] = _heat_raw(pos_freq, d, w_hot_pos)
            markov_raw[d] = _markov_raw(probs, d)
            mv = miss_value(numbers, d, position=position)
            miss_raw[d] = _miss_raw(mv, w_miss_high)
            cycle_raw[d] = _cycle_raw(d, dyn, pos_freq, mv)
    else:
        freq_all = exp_weighted_counts([d for n in recent for d in n])
        pos_freqs = [exp_weighted_counts([n[p] for n in recent]) for p in range(3)]
        markov_by_pos = [second_order_markov_probs(numbers, p) for p in range(3)]
        for d in range(10):
            hot_parts = [_heat_raw(freq_all, d, w_hot_global)]
            hot_parts += [_heat_raw(pf, d, w_hot_pos) for pf in pos_freqs]
            hot_raw[d] = sum(hot_parts) / len(hot_parts)
            markov_raw[d] = sum(_markov_raw(mp, d) for mp in markov_by_pos) / 3
            mv = miss_value(numbers, d)
            miss_raw[d] = _miss_raw(mv, w_miss_high)
            cycle_raw[d] = _cycle_raw(d, dyn, freq_all, mv)

    hot_n = _norm_scale_10(hot_raw)
    markov_n = _norm_scale_10(markov_raw)
    miss_n = _norm_scale_10(miss_raw)
    cycle_n = _norm_scale_10(cycle_raw)
    return [
        balance_digit_score(hot_n[d], markov_n[d], miss_n[d], cycle_n[d])
        for d in range(10)
    ]


def digit_scores(numbers, window=RECENT_WINDOW, dynamic=None):
    recent = _recent_slice(numbers, window)
    last = numbers[-1]
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    score = _balance_scores_normalized(numbers, window, dyn, position=None)
    freq_all = exp_weighted_counts([d for n in recent for d in n])

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
    return combined, freq_combined


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    """单码分位评分（百/十/个）— 冷热平衡（归一化四分量）"""
    last_d = numbers[-1][position]
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
    sc = _balance_scores_normalized(numbers, window, dyn, position=position)
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


def ensemble_all_position_scores(numbers, window_weights, dynamic=None):
    """百/十/个分位独立评分（各 10 维）"""
    return [
        ensemble_position_digit_scores(numbers, pos, window_weights, dynamic=dynamic)
        for pos in range(3)
    ]


def position_top_pools(pos_scores, top_n=POSITION_SAMPLE_N):
    """各位置 Top-N 号码池（仅取前 N，供展示/兼容）"""
    pools = []
    for sc in pos_scores:
        ranked = sorted(enumerate(sc), key=lambda x: -x[1])[:top_n]
        pools.append([d for d, _ in ranked])
    return pools


def position_weighted_pools(pos_scores, top_n=POSITION_CANDIDATE_N):
    """各位置 Top-N 候选 + 归一化抽样权重"""
    pools, weights = [], []
    for sc in pos_scores:
        ranked = sorted(enumerate(sc), key=lambda x: -x[1])[:top_n]
        digits = [d for d, _ in ranked]
        wts = [max(s, 0.01) for _, s in ranked]
        total = sum(wts)
        pools.append(digits)
        weights.append([w / total for w in wts])
    return pools, weights


def contrarian_position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    """反相关冷门模型：偏高遗漏、低热度、低转移概率（寻找易被忽略的号）"""
    recent = _recent_slice(numbers, window)
    pos_freq = exp_weighted_counts([n[position] for n in recent])
    probs = second_order_markov_probs(numbers, position)
    mx = max(pos_freq.values()) if pos_freq else 1.0
    raw = [0.0] * 10
    for d in range(10):
        miss_part = min(miss_value(numbers, d, position=position) / 12.0, 1.0) * 10.0
        cold_part = (1.0 - pos_freq.get(d, 0) / mx) * 10.0 if mx else 5.0
        low_te = (1.0 - probs.get(d, 0.1)) * 10.0
        raw[d] = 0.45 * miss_part + 0.35 * cold_part + 0.20 * low_te
    return _norm_scale_10(raw)


def ensemble_contrarian_position_scores(numbers, window_weights, dynamic=None):
    out = []
    for pos in range(3):
        sc = [0.0] * 10
        for w, wt in window_weights.items():
            ps = contrarian_position_digit_scores(numbers, pos, window=w, dynamic=dynamic)
            for d in range(10):
                sc[d] += wt * ps[d]
        out.append(sc)
    return out


def blend_position_scores(main_scores, contra_scores, w_main=BLEND_MAIN_WEIGHT, w_contra=BLEND_CONTRARIAN_WEIGHT):
    return [
        [w_main * main_scores[pos][d] + w_contra * contra_scores[pos][d] for d in range(10)]
        for pos in range(3)
    ]


def _sample_triplet_from_pools(pools, weights, rng):
    a = rng.choices(pools[0], weights=weights[0], k=1)[0]
    b = rng.choices(pools[1], weights=weights[1], k=1)[0]
    c = rng.choices(pools[2], weights=weights[2], k=1)[0]
    return a, b, c, f"{a}{b}{c}"


def _triplet_score_with_noise(a, b, c, pos_scores, danma, kill, meta, rng, noise_range):
    w = triplet_weight(a, b, c, None, danma, kill, meta, pos_scores=pos_scores)
    if USE_DIVERSITY and noise_range > 0:
        w += rng.uniform(-noise_range, noise_range)
    return w


def monte_carlo_triplet_pool(
    pos_scores, danma, kill, meta, pools, weights, rng,
    trials=MONTE_CARLO_TRIALS, keep=MONTE_CARLO_KEEP,
):
    """Top10 按权重随机抽样 → 评分 → 保留 Top-N"""
    max_score = max(max(sc) for sc in pos_scores) * 3 if pos_scores else 1.0
    noise_range = max_score * DIVERSITY_NOISE if USE_DIVERSITY else 0.0
    best = {}
    for _ in range(trials):
        a, b, c, num = _sample_triplet_from_pools(pools, weights, rng)
        w = _triplet_score_with_noise(a, b, c, pos_scores, danma, kill, meta, rng, noise_range)
        if num not in best or best[num] < w:
            best[num] = w
    ranked = sorted(((w, n) for n, w in best.items()), key=lambda x: -x[0])
    return ranked[:keep], len(best)


def position_sample_triplet_pool(
    pos_scores, danma, kill, meta, pools, weights, rng,
    trials=POSITION_SAMPLE_TRIALS,
):
    """分位 Top5 概率独立抽样 → 评分 → 去重保留"""
    max_score = max(max(sc) for sc in pos_scores) * 3 if pos_scores else 1.0
    noise_range = max_score * DIVERSITY_NOISE if USE_DIVERSITY else 0.0
    best = {}
    for _ in range(trials):
        a, b, c, num = _sample_triplet_from_pools(pools, weights, rng)
        w = _triplet_score_with_noise(a, b, c, pos_scores, danma, kill, meta, rng, noise_range)
        if num not in best or best[num] < w:
            best[num] = w
    return sorted(((w, n) for n, w in best.items()), key=lambda x: -x[0]), len(best)


def default_window_weights():
    """数据不足时的默认窗口（取得分靠前的候选）"""
    fallback = tuple(sorted(WINDOW_CANDIDATES, reverse=True)[:DYNAMIC_WINDOW_TOP_K])
    return {w: 1.0 / len(fallback) for w in fallback}


def compute_window_weights(numbers, trials=WINDOW_BACKTEST_TRIALS, top_k=DYNAMIC_WINDOW_TOP_K):
    """回测 WINDOW_CANDIDATES 全部候选，自动选最强 top_k 窗口集成"""
    candidates = [w for w in WINDOW_CANDIDATES if len(numbers) >= w + 10]
    if not candidates:
        return default_window_weights(), {}

    max_w = max(candidates)
    trials = min(trials, len(numbers) - max_w - 5)
    trials = max(10, trials)
    raw = {w: 0.0 for w in candidates}
    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"
        for w in candidates:
            if len(train) < w:
                continue
            sums = [sum(x) for x in train]
            spans = [calc_span(x) for x in train]
            meta = build_ranking_meta(train, {w: 1.0}, sums, spans, tail_top=4)
            sc, _ = digit_scores(train, window=w, dynamic=meta.get("dynamic"))
            dan, _, kill, _ = pick_dan_tuo_kill(sc)
            top, _, _, _ = rank_position_triplets(
                train, {w: 1.0}, dan, kill, meta, top_n=ZHIXUAN_TOP3
            )
            top_nums = [t[1] if len(t) == 2 else t[1] for t in top]
            if act_s in top_nums:
                raw[w] += 1.0
            elif len({int(c) for s in top_nums for c in s} & set(actual)) >= 2:
                raw[w] += 0.25

    ranked = sorted(candidates, key=lambda w: raw[w], reverse=True)
    selected = ranked[:top_k]
    prior = 1.0
    total = sum(raw[w] + prior for w in selected)
    weights = {w: (raw[w] + prior) / total for w in selected}
    return weights, {w: round(raw[w], 1) for w in candidates}


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
    prev2_form = forms[-2] if len(forms) >= 2 else None

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
    for i in range(len(forms) - 2):
        state = (forms[i], forms[i + 1])
        trans[state][forms[i + 2]] += 1
    if len(forms) >= 2:
        markov_state = (forms[-2], forms[-1])
        row = trans.get(markov_state, Counter())
    else:
        markov_state = (last_form,)
        row = Counter()
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
        "prev2_form": prev2_form,
        "markov_state": markov_state,
        "streak": streak,
        "miss_zu6": form_miss(forms, "zu6"),
        "miss_zu3": form_miss(forms, "zu3"),
        "recent_p": recent_p,
        "hist_p": hist_p,
        "markov_p": markov_p,
        "blend_p": blend,
        "markov_samples": row_total,
    }


def pick_dan_tuo_kill(score):
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    danma = [rank[0][0], rank[1][0]]
    tuoma = [x[0] for x in rank[2:6]]
    kill = [rank[-1][0]] if rank[-1][1] + 3 < rank[-2][1] else [x[0] for x in rank[-2:]]
    return danma, tuoma, kill, rank


def pick_zu6_four(score, kill=None):
    """组六四码：按有效分选 4 个号（杀码降权）+ 多样性增强"""
    digits = pick_zu6_pool(score, kill, pool_size=ZU6_FOUR_SIZE)
    
    # 多样性增强：如果有更多有效候选，随机替换1-2个位置
    if USE_DIVERSITY and len(score) >= 10:
        seed = int(time.time() // 86400) + 1  # 与rank_triplets不同的种子
        rng = random.Random(seed)
        
        # 获取所有候选数字及其分数
        kill_set = set(kill or [])
        candidates = []
        for d in range(10):
            eff_score = score[d] - (W_KILL_PENALTY if d in kill_set else 0.0)
            candidates.append((eff_score, d))
        
        # 按分数排序，选取Top 6作为候选池
        candidates.sort(key=lambda x: -x[0])
        top_pool = [d for _, d in candidates[:6]]
        
        # 随机替换1个位置（保持核心热号的前提下增加变化）
        if len(top_pool) > ZU6_FOUR_SIZE and rng.random() < 0.5:
            # 随机选择一个要替换的位置
            replace_pos = rng.randint(0, ZU6_FOUR_SIZE - 1)
            # 从剩余候选中选择一个不在当前四码中的
            current_set = set(digits)
            replacements = [d for d in top_pool if d not in current_set]
            if replacements:
                new_digit = rng.choice(replacements)
                digits = list(digits)
                digits[replace_pos] = new_digit
                digits = sorted(digits)
    
    return digits


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


def triplet_weight(a, b, c, score, danma, kill, meta, pos_scores=None):
    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    if pos_scores is not None:
        w = pos_scores[0][a] + pos_scores[1][b] + pos_scores[2][c]
    else:
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

    return w


def _pos_hamming(num_a, num_b):
    return sum(1 for x, y in zip(num_a, num_b) if x != y)


def _multisort_key(num_str):
    return tuple(sorted(int(c) for c in num_str))


def _pool_pos_digit_counts(selected):
    counts = [Counter() for _ in range(3)]
    for _, num in selected:
        for pos, ch in enumerate(num):
            counts[pos][int(ch)] += 1
    return counts


def _pool_can_add(num, selected, pos_counts):
    if any(num == n for _, n in selected):
        return False, "exact_dup"
    ms = _multisort_key(num)
    if sum(1 for _, n in selected if _multisort_key(n) == ms) >= POOL_MAX_SAME_MULTISORT:
        return False, "multisort_cap"
    if selected:
        min_dist = min(_pos_hamming(num, n) for _, n in selected)
        if min_dist < POOL_MIN_HAMMING:
            return False, "too_similar"
    for pos, ch in enumerate(num):
        if pos_counts[pos][int(ch)] >= POOL_POS_DIGIT_CAP:
            return False, "pos_digit_cap"
    return True, ""


def _zhixuan_api_item(entry):
    if len(entry) >= 3:
        w, num, slot = entry[0], entry[1], entry[2]
    else:
        w, num = entry[0], entry[1]
        slot = "exploit"
    labels = {"exploit": "利用", "explore": "探索", "fill": "补足"}
    return {
        "num": num,
        "score": round(w, 1),
        "slot": slot,
        "slot_label": labels.get(slot, slot),
    }


def _pool_novelty_score(num, selected):
    if not selected:
        return 10.0
    min_h = min(_pos_hamming(num, n) for _, n in selected)
    ms_pen = sum(1 for _, n in selected if _multisort_key(n) == _multisort_key(num))
    used_digits = Counter(int(n[pos]) for _, n in selected for pos in range(3))
    fresh = sum(1 for pos in range(3) if used_digits[int(num[pos])] <= 1)
    return min_h * 2.5 - ms_pen * 1.5 + fresh * 0.5


def select_diverse_recommendations(ranked_pool, top_n=RECOMMEND_GROUPS, explore_ratio=EXPLORE_RATIO):
    """
    推荐池去重 + 探索选取：
    - 去重：同号集上限、最小海明距、分位数字频次上限
    - 探索：预留槽位从高分候选外挑选新颖组合
    """
    if not POOL_DEDUP_ENABLED or not ranked_pool:
        items = [(w, n, "exploit") for w, n in ranked_pool[:top_n]]
        return items, {"enabled": False}

    explore_n = max(1, int(round(top_n * explore_ratio))) if top_n >= 5 else 0
    exploit_n = top_n - explore_n
    selected = []
    rejected = Counter()
    used = set()
    exploit_count = 0

    def try_add(candidates, limit, slot, prefer_novelty=False):
        nonlocal exploit_count
        if prefer_novelty:
            candidates = sorted(
                candidates,
                key=lambda x: x[0] + EXPLORE_NOVELTY_BONUS * _pool_novelty_score(x[1], [(a, b) for a, b, _ in selected]),
                reverse=True,
            )
        for w, num in candidates:
            if len(selected) >= limit or num in used:
                continue
            ok, reason = _pool_can_add(num, [(a, b) for a, b, _ in selected], _pool_pos_digit_counts([(a, b) for a, b, _ in selected]))
            if ok:
                selected.append((w, num, slot))
                used.add(num)
                if slot == "exploit":
                    exploit_count += 1
            else:
                rejected[reason] += 1

    try_add(ranked_pool, exploit_n, "exploit", prefer_novelty=False)
    exploit_done = len(selected)

    if explore_n > 0:
        remainder = [(w, n) for w, n in ranked_pool if n not in used]
        try_add(remainder, exploit_n + explore_n, "explore", prefer_novelty=True)

    for w, num in ranked_pool:
        if len(selected) >= top_n or num in used:
            continue
        if any(num == n for _, n, _ in selected):
            continue
        if selected and min(_pos_hamming(num, n) for _, n, _ in selected) < 1:
            continue
        selected.append((w, num, "fill"))
        used.add(num)

    for w, num in ranked_pool:
        if len(selected) >= top_n or num in used:
            continue
        selected.append((w, num, "fill"))
        used.add(num)

    explore_nums = [n for _, n, s in selected if s == "explore"]
    meta = {
        "enabled": True,
        "exploit_slots": exploit_n,
        "explore_slots": explore_n,
        "exploit_picked": exploit_done,
        "explore_picked": len(explore_nums),
        "explore_nums": explore_nums,
        "selected": len(selected),
        "rejected": dict(rejected),
        "rejected_labels": {
            "too_similar": "位差过近",
            "multisort_cap": "同号集已满",
            "pos_digit_cap": "分位频次上限",
            "exact_dup": "完全重复",
        },
        "rules": {
            "min_hamming": POOL_MIN_HAMMING,
            "max_same_multisort": POOL_MAX_SAME_MULTISORT,
            "pos_digit_cap": POOL_POS_DIGIT_CAP,
            "explore_ratio": explore_ratio,
        },
        "avg_hamming": 0.0,
        "unique_multisort": 0,
    }
    nums_only = [n for _, n, _ in selected]
    if len(nums_only) >= 2:
        dists = [
            _pos_hamming(nums_only[i], nums_only[j])
            for i in range(len(nums_only))
            for j in range(i + 1, len(nums_only))
        ]
        meta["avg_hamming"] = round(sum(dists) / len(dists), 2)
    meta["unique_multisort"] = len({_multisort_key(n) for n in nums_only})
    meta["position_spread"] = {
        name: dict(Counter(nums_only[i][pos] for i in range(len(nums_only))))
        for pos, name in enumerate(("百", "十", "个"))
    }
    return selected[:top_n], meta


def rank_triplets(score, danma, kill, meta, top_n=20):
    """全空间 10×10×10 直选排序（组六复式等仍使用）"""
    pool = []
    max_score = max(score) if score else 1.0
    noise_range = max_score * DIVERSITY_NOISE if USE_DIVERSITY else 0.0
    seed = int(time.time() // 86400)
    rng = random.Random(seed)

    for a, b, c in product(range(10), repeat=3):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        if USE_DIVERSITY and noise_range > 0:
            w += rng.uniform(-noise_range, noise_range)
        pool.append((w, f"{a}{b}{c}"))
    pool.sort(key=lambda x: -x[0])
    return pool[:top_n]


def rank_position_triplets(
    numbers,
    window_weights,
    danma,
    kill,
    meta,
    top_n=RECOMMEND_GROUPS,
    pos_top_n=POSITION_SAMPLE_N,
    pos_scores=None,
):
    """
    分位独立 + 反相关融合 + Monte Carlo：
    - 0.8×主模型 + 0.2×冷门模型
    - Top10 按权重 MC 5000 组 → 保留 Top100
    - Top5 概率抽样 1000 次 → 合并排序
    """
    main_pos = ensemble_all_position_scores(
        numbers, window_weights, dynamic=meta.get("dynamic")
    )
    contra_pos = ensemble_contrarian_position_scores(
        numbers, window_weights, dynamic=meta.get("dynamic")
    )
    if pos_scores is None:
        pos_scores = blend_position_scores(main_pos, contra_pos)

    seed = int(time.time() // 86400)
    rng = random.Random(seed)

    cand_pools, cand_weights = position_weighted_pools(pos_scores, top_n=POSITION_CANDIDATE_N)
    samp_pools, samp_weights = position_weighted_pools(pos_scores, top_n=pos_top_n)

    mc_pool, mc_unique = monte_carlo_triplet_pool(
        pos_scores, danma, kill, meta, cand_pools, cand_weights, rng,
    )
    pos_pool, pos_unique = position_sample_triplet_pool(
        pos_scores, danma, kill, meta, samp_pools, samp_weights, rng,
    )

    merged = {}
    for w, num in mc_pool + pos_pool:
        if num not in merged or merged[num] < w:
            merged[num] = w
    pool = sorted(((w, n) for n, w in merged.items()), key=lambda x: -x[0])

    candidate_n = min(len(pool), max(top_n * 4, top_n + 20))
    selected, pool_meta = select_diverse_recommendations(pool[:candidate_n], top_n=top_n)
    pool_meta["sampling"] = {
        "method": "monte_carlo_blend",
        "blend_main": BLEND_MAIN_WEIGHT,
        "blend_contrarian": BLEND_CONTRARIAN_WEIGHT,
        "candidate_top_n": POSITION_CANDIDATE_N,
        "position_sample_top_n": pos_top_n,
        "monte_carlo_trials": MONTE_CARLO_TRIALS,
        "monte_carlo_keep": MONTE_CARLO_KEEP,
        "position_sample_trials": POSITION_SAMPLE_TRIALS,
        "mc_unique": mc_unique,
        "pos_sample_unique": pos_unique,
        "merged_unique": len(pool),
    }
    pools_info = {
        "candidate": cand_pools,
        "sample": samp_pools,
        "weights": cand_weights,
        "main": main_pos,
        "contrarian": contra_pos,
    }
    return selected, pos_scores, pools_info, pool_meta


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
    cycle = bayesian_hot_cold_cycle(numbers)
    meta["cycle"] = cycle
    meta["dynamic"] = derive_dynamic_weights(lag1, pat["consec_rate"], cycle)
    meta["last_draw"] = numbers[-1]
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
        dan, _, kill, _ = pick_dan_tuo_kill(sc)
        top, _, _, _ = rank_position_triplets(
            train, ww, dan, kill, meta, top_n=RECOMMEND_GROUPS
        )
        top_nums = [t[1] if len(t) == 2 else t[1] for t in top]
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
        "top_rate_baseline": RECOMMEND_GROUPS / 1000,
        "top3_hit": hit_top3,
        "top3_rate": hit_top3 / n,
        "top3_rate_baseline": ZHIXUAN_TOP3 / 1000,
        "combo_pool": MONTE_CARLO_KEEP + POSITION_SAMPLE_TRIALS,
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
        "baseline_rate": RECOMMEND_GROUPS / POSITION_COMBO_POOL,
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


GA_TUNABLE = ("W_HOT_GLOBAL", "W_MARKOV", "W_MISS_HIGH")


def _ga_random_individual(base, rng):
    ind = dict(base)
    for k in GA_TUNABLE:
        lo, hi = WEIGHT_SEARCH_RANGES.get(k, (0.5, 2.0))
        ind[k] = base[k] * rng.uniform(lo, hi)
    return ind


def _ga_crossover(p1, p2, base, rng):
    child = dict(base)
    for k in GA_TUNABLE:
        child[k] = p1[k] if rng.random() < 0.5 else p2[k]
    return child


def _ga_mutate(ind, rng, rate=0.25):
    child = dict(ind)
    for k in GA_TUNABLE:
        if rng.random() < rate:
            child[k] = max(0.1, child[k] * rng.uniform(0.8, 1.25))
    return child


def genetic_search_weights(
    numbers=None,
    population_size=24,
    generations=12,
    backtest_trials=60,
    metric="top3_rate",
    seed=42,
    elite=4,
    verbose=True,
):
    """遗传算法优化 W_HOT_GLOBAL / W_MARKOV / W_MISS_HIGH 等核心权重"""
    if numbers is None:
        numbers = [x[2] for x in fetch_data()]
    if not numbers:
        return {"error": "未获取到数据"}

    rng = random.Random(seed)
    base = default_weights()
    fixed_ww, _ = compute_window_weights(numbers)

    if verbose:
        print(f"遗传算法: 种群={population_size}, 代数={generations}, 基因={GA_TUNABLE}")

    _, baseline_bt = evaluate_weights(
        numbers, base, trials=backtest_trials, window_weights=fixed_ww, metric=metric
    )
    baseline_score = backtest_objective(baseline_bt, metric)

    population = [_ga_random_individual(base, rng) for _ in range(population_size)]
    best_weights = dict(base)
    best_score = baseline_score
    best_bt = baseline_bt

    for gen in range(generations):
        scored = []
        for ind in population:
            score, bt = evaluate_weights(
                numbers, ind, trials=backtest_trials, window_weights=fixed_ww, metric=metric
            )
            scored.append((score, ind, bt))
        scored.sort(key=lambda x: -x[0])

        if scored[0][0] > best_score:
            best_score, best_weights, best_bt = scored[0]
            if verbose:
                print(
                    f"  [gen {gen + 1:2d}] 新最优 {best_score * 100:.2f}%  "
                    f"hot={best_weights['W_HOT_GLOBAL']:.2f} "
                    f"te={best_weights['W_MARKOV']:.2f} "
                    f"miss={best_weights['W_MISS_HIGH']:.2f}"
                )

        elites = [x[1] for x in scored[:elite]]
        next_pop = list(elites)
        while len(next_pop) < population_size:
            p1, p2 = rng.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = _ga_crossover(p1, p2, base, rng)
            child = _ga_mutate(child, rng)
            next_pop.append(child)
        population = next_pop

    return {
        "method": "genetic",
        "metric": metric,
        "genes": GA_TUNABLE,
        "generations": generations,
        "population_size": population_size,
        "baseline": {"weights": base, "score": baseline_score, "backtest": baseline_bt},
        "best": {"weights": best_weights, "score": best_score, "backtest": best_bt},
        "improvement": best_score - baseline_score,
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
        elif isinstance(v, str):
            dyn_out[k] = v
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


def run_prediction(data=None, force_refresh=False):
    """运行预测，返回 JSON 可序列化 dict；data 为 None 时自动抓取。
    
    Args:
        data: 可选的数据列表，如果为 None 则自动抓取
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
    """
    global _prediction_cache, _cache_time
    
    # 检查缓存
    if not force_refresh and _prediction_cache is not None:
        elapsed = time.time() - _cache_time
        if elapsed < CACHE_EXPIRE_SECONDS:
            log.info(f"使用缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
    
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

    window_weights, window_scores = compute_window_weights(numbers)
    cycle_fft = detect_position_cycles(numbers)
    meta_raw = ensemble_sum_span(sums, spans, window_weights)
    meta = build_ranking_meta(numbers, window_weights, sums, spans, tail_top=5)
    pat = {k: meta[k] for k in ("consec_rate", "oe_freq", "bs_freq", "oe_total", "bs_total")}

    score, freq_all = ensemble_digit_scores(numbers, window_weights, dynamic=meta.get("dynamic"))
    danma, tuoma, kill, rank = pick_dan_tuo_kill(score)
    form_prob = analyze_form_probability(numbers, window_weights=window_weights)
    zu6_four = pick_zu6_four(score, kill)
    _, z6_straight = zu6_notes_from_digits(zu6_four)
    zhixuan_top, pos_scores, pools_info, pool_meta = rank_position_triplets(
        numbers, window_weights, danma, kill, meta, top_n=RECOMMEND_GROUPS
    )
    bt = backtest(numbers, window_weights=window_weights)
    bt["significance"] = permutation_test(
        numbers, bt["top_rate"], window_weights=window_weights
    )

    last_num = numbers[-1]
    pos_names = ("百", "十", "个")
    cand_pools = pools_info.get("candidate", [[]] * 3)
    cand_weights = pools_info.get("weights", [[]] * 3)
    contra_pos = pools_info.get("contrarian", [[]] * 3)
    position_top = [
        {
            "name": name,
            "digits": [
                {
                    "digit": d,
                    "score": round(pos_scores[pos][d], 1),
                    "weight_pct": round(cand_weights[pos][i] * 100, 1) if i < len(cand_weights[pos]) else 0,
                }
                for i, d in enumerate(cand_pools[pos])
            ],
        }
        for pos, name in enumerate(pos_names)
    ]
    contrarian_top = [
        {
            "name": name,
            "digits": [
                {"digit": d, "score": round(contra_pos[pos][d], 1)}
                for d in sorted(range(10), key=lambda x: -contra_pos[pos][x])[:POSITION_CANDIDATE_N]
            ],
        }
        for pos, name in enumerate(pos_names)
    ]

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

    return {
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
        "contrarian_top": contrarian_top,
        "zhixuan_model": {
            "method": "monte_carlo_blend",
            "position_top_n": POSITION_SAMPLE_N,
            "candidate_top_n": POSITION_CANDIDATE_N,
            "combo_pool": pool_meta.get("sampling", {}).get("merged_unique", POSITION_COMBO_POOL),
            "pools": {
                name: pools_info.get("sample", [[]] * 3)[i]
                for i, name in enumerate(pos_names)
            },
            "candidate_pools": {
                name: cand_pools[i]
                for i, name in enumerate(pos_names)
            },
            "candidate_weights": {
                name: [round(w * 100, 1) for w in cand_weights[i]]
                for i, name in enumerate(pos_names)
            },
            "sampling": pool_meta.get("sampling", {}),
            "blend": {
                "main": BLEND_MAIN_WEIGHT,
                "contrarian": BLEND_CONTRARIAN_WEIGHT,
            },
            "pool_dedup": pool_meta,
        },
        "miss_global": miss_global,
        "miss_position": miss_position,
        "sum_tails": sum_tails,
        "recommend_groups": RECOMMEND_GROUPS,
        "recent_windows": list(window_weights.keys()),
        "window_weights": {str(k): round(v, 4) for k, v in window_weights.items()},
        "window_scores": window_scores,
        "dynamic_windows": {
            "method": "backtest_top_k",
            "top_k": DYNAMIC_WINDOW_TOP_K,
            "selected": list(window_weights.keys()),
            "candidates": list(WINDOW_CANDIDATES),
            "best_single": max(window_scores, key=window_scores.get) if window_scores else None,
        },
        "cycle_detection": {
            "global_dominant": cycle_fft.get("global_dominant"),
            "targets": cycle_fft.get("targets"),
            "positions": {
                name: {
                    "dominant": info.get("dominant"),
                    "periods": info.get("periods"),
                }
                for name, info in cycle_fft.get("positions", {}).items()
            },
        },
        "transfer_model": "transfer_entropy",
        "score_formula": {
            "method": "balance_blend",
            "weights": {
                "hot": BALANCE_W_HOT,
                "markov": BALANCE_W_MARKOV,
                "miss": BALANCE_W_MISS,
                "cycle": BALANCE_W_CYCLE,
            },
            "blend": {
                "main": BLEND_MAIN_WEIGHT,
                "contrarian": BLEND_CONTRARIAN_WEIGHT,
            },
            "scales": {
                "W_HOT_POS": W_HOT_POS,
                "W_MARKOV": W_MARKOV,
                "W_MISS_HIGH": W_MISS_HIGH,
            },
        },
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
        "cycle": {
            "phase": meta["cycle"]["phase"],
            "phase_label": {"hot": "热周期", "cold": "冷周期", "neutral": "均衡"}[meta["cycle"]["phase"]],
            "p_hot": round(meta["cycle"]["p_hot"], 4),
            "p_hot_raw": round(meta["cycle"].get("p_hot_raw", meta["cycle"]["p_hot"]), 4),
            "hot_hits": meta["cycle"]["hot_hits"],
            "cold_hits": meta["cycle"]["cold_hits"],
            "w_hot_scale": meta["dynamic"]["w_hot_scale"],
            "w_miss_scale": meta["dynamic"]["w_miss_scale"],
            "w_hot_global": round(meta["dynamic"]["w_hot_global"], 2),
            "w_miss_mid": round(meta["dynamic"]["w_miss_mid"], 2),
        },
        "transition": _transition_for_api(meta["lag1"], meta["dynamic"], pos_names),
        "form": {
            "last_label": FORM_LABELS[form_prob["last_form"]],
            "prev2_label": FORM_LABELS[form_prob["prev2_form"]] if form_prob.get("prev2_form") else None,
            "markov_state_label": tuple(FORM_LABELS[f] for f in form_prob["markov_state"]),
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
        "zhixuan_top3": [_zhixuan_api_item(t) for t in zhixuan_top[:ZHIXUAN_TOP3]],
        "zhixuan": [_zhixuan_api_item(t) for t in zhixuan_top],
        "backtest": bt,
    }
    
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

    dw = result.get("dynamic_windows", {})
    ww = result.get("window_weights", {})
    ws = result.get("window_scores", {})
    if dw.get("selected"):
        print(f"  动态窗口(回测Top{dw.get('top_k', 3)}): {dw['selected']}"
              f"  最强单窗={dw.get('best_single')}")
    if ww:
        parts = [
            f"{k}期权重{float(ww[k])*100:.0f}%"
            + (f"(得分{ws.get(int(k), ws.get(k))})" if ws.get(int(k), ws.get(k)) is not None else "")
            for k in ww
        ]
        print(f"  窗口权重: {', '.join(parts)}")

    cd = result.get("cycle_detection", {})
    if cd.get("global_dominant"):
        print(f"  周期检测(FFT+自相关): 全局主导 {cd['global_dominant']} 期")
        for name, info in cd.get("positions", {}).items():
            dom = info.get("dominant")
            if dom and dom in (info.get("periods") or {}):
                st = info["periods"][dom].get("strength", 0)
                print(f"    {name}位: {dom}期周期 强度={st:.3f}")
    if result.get("transfer_model"):
        print(f"  转移模型: {result['transfer_model']}（替代马尔可夫）")

    print("\n" + "=" * 70)
    cycle = result.get("cycle")
    if cycle:
        print(f"  冷热周期: {cycle['phase_label']}（P(热)={cycle['p_hot']*100:.1f}%）"
              f"  热号权重×{cycle['w_hot_scale']}  冷号权重×{cycle['w_miss_scale']}")

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
    ms = form.get("markov_state_label") or (lf,)
    ms_str = ",".join(ms) if len(ms) >= 2 else lf
    print(
        f"  前两期({ms_str})→下期(样本{form['markov_samples']}): "
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
        phase = dyn.get("cycle_phase", "neutral")
        phase_cn = {"hot": "热周期", "cold": "冷周期", "neutral": "均衡"}.get(phase, phase)
        print(f"  冷热周期: {phase_cn}（P(热)={dyn.get('p_hot_cycle', 0.5)*100:.1f}%）"
              f"  热号×{dyn.get('w_hot_scale', 1):.2f}  冷号×{dyn.get('w_miss_scale', 1):.2f}")
        print(f"  动态权重: 热号 {dyn.get('w_hot_global', W_HOT_GLOBAL):.2f}"
              f"  冷号 {dyn.get('w_miss_mid', W_MISS_MID):.2f}"
              f"  同位复刻 {dyn.get('w_pos_repeat', W_POS_REPEAT):.2f}"
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

    zm = result.get("zhixuan_model", {})
    if zm:
        pools = zm.get("pools", {})
        samp = zm.get("sampling", {})
        cand = zm.get("candidate_pools", {})
        print(f"  MC采样: Top10权重×{samp.get('monte_carlo_trials', MONTE_CARLO_TRIALS)}"
              f" → Top{samp.get('monte_carlo_keep', MONTE_CARLO_KEEP)}"
              f" + 分位Top5×{samp.get('position_sample_trials', POSITION_SAMPLE_TRIALS)}"
              f" = {samp.get('merged_unique', '—')} 注合并池")
        print(f"  候选Top10: 百{''.join(map(str, cand.get('百', [])))}"
              f" 十{''.join(map(str, cand.get('十', [])))}"
              f" 个{''.join(map(str, cand.get('个', [])))}")
        blend = zm.get("blend", {})
        print(f"  融合评分: {blend.get('main', BLEND_MAIN_WEIGHT)}×主模型"
              f" + {blend.get('contrarian', BLEND_CONTRARIAN_WEIGHT)}×冷门模型")
        pd = zm.get("pool_dedup", {})
        if pd.get("enabled"):
            print(f"  推荐池去重+探索: 利用{pd.get('exploit_slots')} + 探索{pd.get('explore_slots')} "
                  f"| 平均位差{pd.get('avg_hamming')} | 号集{pd.get('unique_multisort')}种 "
                  f"| 拦截{pd.get('rejected')}")

    print("\n" + "=" * 70)
    print("【直选Top3推荐】（MC采样 + 0.8主+0.2冷门）")
    print("=" * 70)
    for idx, item in enumerate(result.get("zhixuan_top3", []), start=1):
        print(f"  {idx}. {item['num']}  评分={item['score']:.1f}")

    print("\n" + "=" * 70)
    print(f"【直选推荐 {RECOMMEND_GROUPS} 注】（MC采样 + 分位概率抽样）")
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
    parser.add_argument(
        "--genetic-search",
        action="store_true",
        help="遗传算法优化 W_HOT_GLOBAL / W_MARKOV / W_MISS_HIGH",
    )
    parser.add_argument("--ga-pop", type=int, default=24, help="遗传算法种群大小")
    parser.add_argument("--ga-gen", type=int, default=12, help="遗传算法代数")
    args = parser.parse_args(argv)

    print("抓取数据中...")
    data = fetch_data()
    numbers = [x[2] for x in data] if data else []

    if args.genetic_search:
        if not numbers:
            print("未获取到数据")
            return
        result = genetic_search_weights(
            numbers=numbers,
            population_size=args.ga_pop,
            generations=args.ga_gen,
            backtest_trials=args.search_trials,
            metric=args.search_metric,
            seed=args.search_seed,
        )
        print_search_report(result)
        return

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
