# 福彩3D 高级信号模块：周期检测 / 转移熵 / 动态窗口候选
"""第三梯队算法：FFT + 自相关 + 滑动窗口周期检测，转移熵替代马尔可夫。"""
import math
from collections import Counter, defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# 目标周期（期）
CYCLE_TARGETS = (7, 14, 28)

# 动态窗口候选（回测自动选择，替代固定 30/45/60/90）
WINDOW_CANDIDATES = tuple(sorted({
    7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 90,
    30, 45, 60,
}))

# 转移熵：仅当有效态足够时才过滤，否则回退完整二阶马尔可夫
TE_MIN_CONTRIB = 0.005
TE_MIN_FILTERED = 3
MARKOV_LAPLACE_ALPHA = 1.0

POS_NAMES = ("百", "十", "个")


def _detrend(series):
    if not series:
        return []
    mean = sum(series) / len(series)
    return [x - mean for x in series]


def autocorrelation_at_lag(series, lag):
    """滞后 lag 的自相关系数"""
    n = len(series)
    if lag <= 0 or lag >= n:
        return 0.0
    mean = sum(series) / n
    var = sum((x - mean) ** 2 for x in series)
    if var <= 1e-12:
        return 0.0
    cov = sum(
        (series[i] - mean) * (series[i + lag] - mean)
        for i in range(n - lag)
    )
    return cov / var


def fft_power_at_period(series, period):
    """FFT 在目标周期处的功率"""
    if not HAS_NUMPY or len(series) < period * 2:
        return 0.0
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    n = len(y)
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0)
    target = 1.0 / period
    idx = int(np.argmin(np.abs(freqs - target)))
    return float(np.abs(fft[idx]) ** 2 / max(n, 1))


def sliding_autocorr_score(series, period, window=None):
    """滑动窗口内目标周期自相关强度"""
    window = window or max(period * 4, 28)
    if len(series) < period + 2:
        return 0.0
    seg = series[-min(window, len(series)):]
    return abs(autocorrelation_at_lag(seg, period))


def _normalize_strength(raw_scores):
    """归一化各周期强度到 [0,1]"""
    if not raw_scores:
        return {}
    mx = max(raw_scores.values()) or 1.0
    return {k: v / mx for k, v in raw_scores.items()}


def detect_series_cycles(series, targets=CYCLE_TARGETS):
    """单序列：FFT + 自相关 + 滑动窗口 → 7/14/28 期周期强度"""
    if len(series) < 14:
        return {"dominant": None, "periods": {}, "series_len": len(series)}

    detrended = _detrend(series)
    periods = {}
    combined = {}

    for p in targets:
        fft_p = fft_power_at_period(detrended, p)
        ac_p = abs(autocorrelation_at_lag(detrended, p))
        sw_p = sliding_autocorr_score(detrended, p)
        strength = 0.35 * (fft_p / (fft_p + 1.0)) + 0.35 * ac_p + 0.30 * sw_p
        periods[p] = {
            "fft_power": round(fft_p, 4),
            "autocorr": round(ac_p, 4),
            "sliding": round(sw_p, 4),
            "strength": round(strength, 4),
        }
        combined[p] = strength

    norm = _normalize_strength(combined)
    for p in targets:
        periods[p]["norm_strength"] = round(norm.get(p, 0.0), 4)

    dominant = max(combined, key=combined.get) if combined else None
    return {
        "dominant": dominant,
        "periods": periods,
        "series_len": len(series),
    }


def detect_position_cycles(numbers):
    """百/十/个位 + 和值 的周期检测"""
    if not numbers:
        return {"positions": {}, "global_dominant": None}

    sums = [sum(n) for n in numbers]
    series_map = {
        "百": [n[0] for n in numbers],
        "十": [n[1] for n in numbers],
        "个": [n[2] for n in numbers],
        "和值": sums,
    }

    positions = {}
    dominant_votes = Counter()
    for name, series in series_map.items():
        info = detect_series_cycles(series)
        positions[name] = info
        if info.get("dominant"):
            dominant_votes[info["dominant"]] += info["periods"][info["dominant"]]["strength"]

    global_dominant = dominant_votes.most_common(1)[0][0] if dominant_votes else None
    return {
        "positions": positions,
        "global_dominant": global_dominant,
        "targets": list(CYCLE_TARGETS),
    }


def _discrete_entropy(counter, total=None):
    total = total or sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _mutual_information_xy(joint, x_vals, y_vals):
    """I(X;Y) from joint count dict {(x,y): count}"""
    n = sum(joint.values())
    if n <= 0:
        return 0.0
    px = Counter()
    py = Counter()
    for (x, y), c in joint.items():
        px[x] += c
        py[y] += c
    mi = 0.0
    for (x, y), c in joint.items():
        if c <= 0:
            continue
        pxy = c / n
        pxv = px[x] / n
        pyv = py[y] / n
        if pxv > 0 and pyv > 0:
            mi += pxy * math.log(pxy / (pxv * pyv))
    return max(0.0, mi)


def transfer_entropy_second_order(numbers, position):
    """
    二阶转移熵 TE: I(X_{t-2}, X_{t-1}; X_t) - I(X_{t-1}; X_t)
    衡量 (t-2,t-1) 对 t 的增量信息，过滤纯随机共现。
    """
    if len(numbers) < 4:
        return 0.0, {}

    joint3 = Counter()
    joint2 = Counter()
    cnt_ab = Counter()
    cnt_b = Counter()

    for i in range(len(numbers) - 2):
        a = numbers[i][position]
        b = numbers[i + 1][position]
        c = numbers[i + 2][position]
        joint3[(a, b, c)] += 1
        joint2[(b, c)] += 1
        cnt_ab[(a, b)] += 1
        cnt_b[b] += 1

    n = sum(cnt_ab.values())
    if n <= 0:
        return 0.0, {}

    mi_full = _mutual_information_xy(
        {(ab, c): joint3[(*ab, c)] for ab in cnt_ab for c in range(10) if joint3.get((*ab, c), 0)},
        list(cnt_ab.keys()),
        list(range(10)),
    )
    mi_reduced = _mutual_information_xy(joint2, list(cnt_b.keys()), list(range(10)))
    te = max(0.0, mi_full - mi_reduced)

    state = (numbers[-2][position], numbers[-1][position])
    per_target = {}
    ab_total = cnt_ab.get(state, 0)
    b_only = numbers[-1][position]
    b_total = cnt_b.get(b_only, 0)

    for c in range(10):
        n_abc = joint3.get((state[0], state[1], c), 0)
        n_bc = joint2.get((b_only, c), 0)
        p_abc = n_abc / ab_total if ab_total else 0.0
        p_bc = n_bc / b_total if b_total else 0.0
        if p_abc > 0 and p_bc > 0:
            contrib = p_abc * math.log(p_abc / p_bc)
        elif p_abc > 0:
            contrib = p_abc * math.log(max(p_abc, 1e-9))
        else:
            contrib = 0.0
        per_target[c] = max(0.0, contrib)

    return te, per_target


def markov_prob_smoothed(row, states, alpha=MARKOV_LAPLACE_ALPHA):
    states = list(states)
    row_total = sum(row.values())
    denom = row_total + alpha * len(states)
    return {s: (row.get(s, 0) + alpha) / denom for s in states}


def _second_order_row(numbers, position, state):
    """当前二阶状态 (a,b) 下各下一码计数"""
    row = Counter()
    for i in range(len(numbers) - 2):
        if (numbers[i][position], numbers[i + 1][position]) == state:
            row[numbers[i + 2][position]] += 1
    return row


def _first_order_row(numbers, position, prev_digit):
    """一阶回退：P(next | prev)"""
    row = Counter()
    for i in range(len(numbers) - 1):
        if numbers[i][position] == prev_digit:
            row[numbers[i + 1][position]] += 1
    return row


def transfer_entropy_probs(numbers, position, states=None, te_threshold=TE_MIN_CONTRIB):
    """
    转移熵指导的条件概率 P(next | prev2, prev1)。
    有效转移态不足时回退完整马尔可夫行，避免概率塌缩为两档。
    """
    states = list(range(10) if states is None else states)
    if len(numbers) < 2:
        return markov_prob_smoothed(Counter(), states), {"te": 0.0, "filtered_states": 0, "fallback": "empty"}

    te_global, te_per = transfer_entropy_second_order(numbers, position)
    state = (numbers[-2][position], numbers[-1][position])
    full_row = _second_order_row(numbers, position, state)

    te_filtered = Counter()
    for s in states:
        if full_row.get(s, 0) > 0 and te_per.get(s, 0.0) >= te_threshold:
            te_filtered[s] = full_row[s]

    fallback = "second_order"
    if sum(full_row.values()) == 0:
        full_row = _first_order_row(numbers, position, numbers[-1][position])
        fallback = "first_order"
    if sum(full_row.values()) == 0:
        probs = markov_prob_smoothed(Counter(), states)
        return probs, {"te": round(te_global, 4), "filtered_states": 0, "fallback": "uniform"}

    if len(te_filtered) >= TE_MIN_FILTERED:
        use_row = te_filtered
        fallback = "te_filtered"
    else:
        use_row = full_row

    probs = markov_prob_smoothed(use_row, states)
    # TE 作为软增益而非硬过滤，拉开概率差距
    if te_global > te_threshold:
        boosted = {}
        for s in states:
            boost = 1.0 + min(te_per.get(s, 0.0) * 4.0, 0.5)
            boosted[s] = probs[s] * boost
        total = sum(boosted.values()) or 1.0
        probs = {s: boosted[s] / total for s in states}

    return probs, {
        "te": round(te_global, 4),
        "filtered_states": len(te_filtered),
        "fallback": fallback,
    }


def transfer_entropy_probs_simple(numbers, position, states=None):
    """仅返回概率 dict（兼容旧接口）"""
    probs, _ = transfer_entropy_probs(numbers, position, states=states)
    return probs
