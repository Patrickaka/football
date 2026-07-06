# 福彩3D预测器 V3.1+（标准库版，准确率优化）
# Python 3.10+
import json
import math
import os
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
from ..common import kv_store

log = setup_logger('lottery3d')

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 拉取约 2000 期历史（≈6 年）。更长的历史让频率/转移/和值估计与回测显著更稳定，
# 减少 200 期小样本下的噪声。预测路径仍按窗口（≤90 期）截取，故不影响线上速度。
URL = "https://www.8300.cn/kjhhis/3/2000.html"

RECENT_WINDOWS = (30, 45, 60, 90)
RECENT_WINDOW = 90  # 展示用最大窗口
# 窗口权重回测参数（保守收缩）
WINDOW_BACKTEST_TRIALS = 100  # 增加回测期数，减少偶然因素影响
WINDOW_WEIGHT_PRIOR = 5.0  # 增大先验权重，平滑窗口权重波动
EXP_DECAY = 0.96
BACKTEST_TRIALS = 80
PERMUTATION_SHUFFLES = 200  # 置换检验打乱次数，评估命中率是否显著优于随机（建议线上至少200，离线验证1000）
# 和值/跨度近期偏移参数
RECENT_SUM_SPAN_SHIFT = 0.0  # 关闭近期5期偏移，避免追涨杀跌

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
# 实盘保守版本：大幅降权冷号遗漏奖励，避免追冷
W_MISS_HIGH = 1.5   # 遗漏 20 期加 1.5*(1+20/20)=3分（大幅降权）
W_MISS_MID = 1.0    # 中等遗漏值加分（降权）
W_MARKOV = 5.0       # 原 8.0；降低马尔可夫转移权重，避免过度依赖最近一期
W_MARKOV2 = 1.5      # 原 4.0；降低二阶马尔可夫转移权重
MARKOV_MAX_SCORE = 6.0  # 马尔可夫转移得分上限，避免主导推荐结果
W_LAST_APPEAR = 2.5
W_NEIGHBOR = 2.0
W_ROAD_MATCH = 1.5
W_DANMA_HIT = 4.0
W_KILL_PENALTY = 4.0  # 杀码软降权（原6.0过强，易误伤开奖数字）
W_CONSECUTIVE = 1.5   # 含相邻连号（如 12、67）
W_POS_REPEAT = 1.2    # 与上期同位重复（直选复刻），每码；实际强度由 lag1 动态缩放
W_RATIO_MATCH = 1.8   # 奇偶比 / 大小比与近期热门匹配
# 随机基准：单位置复刻 10%；指定数字在下一期三码中出现 ≈ 27.1%
RANDOM_POS_REPEAT = 0.10
RANDOM_DIGIT_REUSE = 1 - (9 / 10) ** 3
SUM_SOFT_SIGMA = 3.2
SPAN_SOFT_SIGMA = 1.4
# 形态先验：按真实形态概率(组六72%/组三27%/豹子1%)给分，使推荐池形态分布贴合真实开奖。
# 实测原推荐组六偏少(64% vs 真实71%)、组三偏多，此项把组六占比拉回~72%。
W_FORM_PRIOR = 6.0

# 直选组合：分位评分为主、全局评分为辅（直选带顺序，分位信号更关键）
W_TRIPLET_POS = 1.0
W_TRIPLET_GLOBAL = 0.30
ZHXUAN_POS_TOPK = 5  # Top3/Top5 分位候选池每位置取前 N 码

# 探索机制：推荐时有一定概率从候选池中随机选择
# 稳定基础版：关闭探索机制
EXPLORATION_RATE = 0.0  # 关闭探索，始终选择最高分号码

# 动态胆码机制：70%概率选 Top2，30%概率从 Top6 中随机选 2 个
# 稳定基础版：关闭随机胆码
DANMA_TOP_POOL = 6  # 胆码候选池大小
DANMA_RANDOM_RATE = 0.0  # 关闭随机选择胆码

# 推荐注数（直选为带顺序的三位数）
RECOMMEND_GROUPS = 30  # 推荐池扩大至 30 注
ZHIXUAN_TOP3 = 3
ZU6_POOL_SIZE = 5
ZU6_FOUR_SIZE = 4

# Top50 随机扰动：避免同分号长期霸榜
# 稳定基础版：关闭随机噪声
RANDOM_NOISE = 0.0  # 关闭随机噪声

# 近期回补模型：统计最近 30 期，严重欠账的号码额外加分
RECENT_WINDOW_REBOUND = 30  # 回补统计窗口
REBOUND_BONUS = 0.5  # 严重欠账号码加分（实盘保守版本：大幅降权）
REBOUND_THRESHOLD = 0.5  # 欠账阈值（实际值/理论值 < 0.5 认为严重欠账）

# 冷热平衡模型：推荐池号码类型比例
HOT_RATIO = 0.40   # 热号比例 40%
WARM_RATIO = 0.40  # 温号比例 40%

# 特征开关：用于消融测试
# 稳定基础版配置：关闭形态切换、冷热平衡、数字配对和遗漏（待消融验证）
FEATURE_FLAGS = {
    "hot": True,           # 热号得分
    "miss": False,         # 遗漏加分（关闭：待消融回测验证有效性）
    "markov": True,        # 马尔可夫转移
    "neighbor": False,     # 邻号加分（关闭：待消融回测验证有效性）
    "road": False,         # 012 路匹配（关闭：待消融回测验证有效性）
    "sum_span": True,      # 和值跨度
    "pair": True,          # 数字配对（high_pairs 已在 meta 预计算）
    "form_switch": False,  # 形态切换（关闭：避免赌徒谬误）
    "cold_hot_balance": False,  # 冷热平衡（关闭：避免干扰模型排序）
    "consecutive": True,   # 连号奖励（开启：待消融回测验证有效性）
    "lag1_repeat": True,   # 上期同位重复、全重复、同集合惩罚（开启：待消融回测验证有效性）
    "ratio": True,         # 奇偶比、大小比奖励（开启：待消融回测验证有效性）
    "slope": True,         # 斜连走势（同位/跨期，辅助加分）
}
COLD_RATIO = 0.20  # 冷号比例 20%
HOT_WINDOW = 20    # 冷热判断窗口

# 和值趋势模型：统计最近 20 期和值趋势，动态调整和值中心
# 实盘版本：关闭和值趋势调整（避免过度约束）
SUM_TREND_WINDOW = 20  # 和值趋势统计窗口
SUM_TREND_ADJUST = 0.0  # 和值中心调整幅度（关闭）

# 遗漏周期模型：计算平均遗漏周期和超期倍率
MISS_CYCLE_WINDOW = 200  # 统计平均遗漏周期的窗口大小
MISS_OVER_RATIO_THRESHOLD = 2.5  # 超期倍率阈值
MISS_OVER_BONUS = 1.0  # 超期额外加分（实盘保守版本：降权）

# 数字配对模型：统计数字对出现频率
PAIR_FREQ_WINDOWS = [50, 100, 200]  # 统计窗口
PAIR_HIGH_FREQ_THRESHOLD = 0.15  # 高频对子阈值（出现频率 > 15%）
PAIR_BONUS = 2.5  # 高频对子加分

# 斜连走势：同位连续 ±1、跨期百→十→个对角（样本少，仅轻量加分）
SLOPE_MIN_CHAIN = 3
SLOPE_MAX_CHAIN = 6
W_SLOPE_MATCH = 1.2
POS_NAMES_3D = ("百", "十", "个")

# 组三组六切换模型：根据连续出现调整权重
# 实盘版本：关闭组三组六强制切换（典型赌徒谬误，连续出现不代表下期一定要切换）
FORM_SWITCH_WEIGHT = 0.0  # 切换奖励权重（关闭）
ZU6_STREAK_THRESHOLD = 8  # 组六连续出现阈值
ZU3_STREAK_THRESHOLD = 4  # 组三连续出现阈值

# 和值区间回归模型：预测区间而非具体和值
# 实盘版本：关闭和值区间回归奖励（和值适合软排序，不适合硬约束）
SUM_INTERVAL_WINDOW = 5  # 计算中心的窗口大小
SUM_INTERVAL_WIDTH = 3  # 区间宽度（中心 ± width）
SUM_INTERVAL_BONUS = 0.0  # 区间内加分（关闭）
SUM_EXTREME_PENALTY = 0.0  # 极端区间降权（关闭）

# 最近N期去重机制：对重复推荐降权，使每日推荐池在天与天之间轮换。
# 3D 为公平摇奖，任意 30 注互异组合命中率恒为 3%，故轮换不改变命中期望(实测900期均落在
# 3%±1.1% 噪声带内)，但能把日间重复度从 ~54% 降到 ~14%。惩罚值取较大以形成"近窗内基本不重复"。
RECENT_RECOMMEND_WINDOW = 5  # 最近推荐窗口大小
RECENT_RECOMMEND_PENALTY = 5.0  # 日间轮换惩罚（略降，减少对 Top30 命中率的干扰）
RECENT_RECOMMEND_CONSECUTIVE_PENALTY = 16.0  # 连续推荐的号码惩罚

# 组六四码逐期轮换（默认关闭：轮换不为命中率服务，会主动换掉相对高分码）
ZU6_RECENT_WINDOW = 5
ZU6_RECENT_PENALTY = 0.0
ZU6_RECENT_DECAY = 0.6

# 组六选码：不用杀码降权（杀码误伤开奖数字的风险大于收益）
ZU6_USE_KILL = False
ZU6_CANDIDATE_SIZE = 8

# 组六专用单码评分权重
W_ZU6_HOT = 3.0
W_ZU6_MISS = 1.5
W_ZU6_POS = 1.2
W_ZU6_PAIR = 2.0
W_ZU6_BLEND = 1.5

# 窗口权重持久化键
WINDOW_WEIGHTS_KV_KEY = "lottery3d_window_weights"

# 预测版本号
PREDICTOR_VERSION = "3d-v4.5-slope"
ML_MODEL_VERSION = "ml-v6"
MIN_DATA_PERIODS_FOR_ML_FUSION = 300
ML_CACHE_MAX_AGE_SECONDS = 36 * 3600

# 线上实盘记录文件
ONLINE_PREDICTION_FILE = "data/lottery3d_online_predictions.json"

# 推荐池多样性控制：最大化数字覆盖率
DIVERSITY_WEIGHT = 1.5  # 多样性权重
SERVED_POOL_CANDIDATE_SIZE = 150  # 贪心选池候选范围


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
# 精简版本：只保留已启用特征的权重，避免浪费计算在无效参数上
TUNABLE_WEIGHTS = (
    "W_HOT_GLOBAL",
    "W_HOT_POS",
    "W_MARKOV",
    "W_MARKOV2",
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


def _fetch_data_internal(url=URL, retries=3, timeout=30):
    """内部数据抓取函数（带重试，应对上游瞬时超时）"""
    log.debug('fetch 3D data')
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
            break
        except Exception as e:
            last_err = e
            log.warning('3D 抓取第 %d/%d 次失败: %s', attempt + 1, retries, e)
            time.sleep(2 * (attempt + 1))
    else:
        # 全部重试失败，向上抛出由 cached_fetch 兜底（旧缓存）或 run_prediction 处理。
        raise last_err
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


def has_consecutive_digits(a, b, c):
    """是否存在相邻连号（差值为 1，不含 9-0）"""
    digits = (a, b, c)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(digits[i] - digits[j]) == 1:
                return True
    return False


def _slope_step(prev_digit, cur_digit):
    """斜连步长：仅认可 ±1（不含 9↔0 绕回）。"""
    diff = cur_digit - prev_digit
    return diff if diff in (-1, 1) else None


def _detect_position_slope_chain(digits_at_pos, min_len=SLOPE_MIN_CHAIN, max_len=SLOPE_MAX_CHAIN):
    """同一位上最近若干期是否形成等差斜连，返回最长有效链。"""
    if len(digits_at_pos) < min_len:
        return None
    best = None
    upper = min(max_len, len(digits_at_pos))
    for length in range(upper, min_len - 1, -1):
        seq = digits_at_pos[-length:]
        step = None
        valid = True
        for i in range(1, len(seq)):
            s = _slope_step(seq[i - 1], seq[i])
            if s is None:
                valid = False
                break
            if step is None:
                step = s
            elif s != step:
                valid = False
                break
        if not valid or step is None:
            continue
        nxt = seq[-1] + step
        if 0 <= nxt <= 9:
            best = {
                "chain": seq,
                "step": step,
                "predict_digit": nxt,
                "length": length,
            }
            break
    return best


def _cross_period_slope_signals(numbers):
    """跨期斜连：近三期在百→十→个（及轮换）上形成对角走势。"""
    if len(numbers) < 3:
        return []
    signals = []
    draws = numbers[-3:]
    # 三种位次轮换的对角：起始位 offset ∈ {0,1,2}
    for offset in range(3):
        vals = []
        for k in range(3):
            pos = (offset + k) % 3
            vals.append(draws[k][pos])
        step = _slope_step(vals[0], vals[1])
        if step is None or _slope_step(vals[1], vals[2]) != step:
            continue
        predict_pos = offset
        predict_digit = vals[-1] + step
        if not (0 <= predict_digit <= 9):
            continue
        route = "→".join(
            POS_NAMES_3D[(offset + k) % 3] for k in range(3)
        )
        signals.append({
            "type": "cross_period_slope",
            "position": predict_pos,
            "position_name": POS_NAMES_3D[predict_pos],
            "chain": vals,
            "route": route,
            "step": step,
            "predict_digit": predict_digit,
            "length": 3,
            "label": (
                f"跨期斜连 {route} {'+' if step > 0 else ''}{step} "
                f"({'→'.join(map(str, vals))}) → 下期{POS_NAMES_3D[predict_pos]}位关注 {predict_digit}"
            ),
            "strength": 1.0,
        })
    return signals


def analyze_slope_patterns(numbers, min_len=SLOPE_MIN_CHAIN):
    """识别斜连走势并给出下期分位关注码（辅助参考，非主预测）。"""
    signals = []
    position_hints = {i: [] for i in range(3)}

    for pos in range(3):
        hist = [n[pos] for n in numbers]
        det = _detect_position_slope_chain(hist, min_len=min_len)
        if not det:
            continue
        chain_s = "→".join(map(str, det["chain"]))
        step = det["step"]
        pred = det["predict_digit"]
        strength = 1.0 + (det["length"] - min_len) * 0.25
        sig = {
            "type": "position_slope",
            "position": pos,
            "position_name": POS_NAMES_3D[pos],
            "chain": det["chain"],
            "step": step,
            "predict_digit": pred,
            "length": det["length"],
            "label": (
                f"同位斜连 {POS_NAMES_3D[pos]}位 {'+' if step > 0 else ''}{step} "
                f"({chain_s}) → 关注 {pred}"
            ),
            "strength": round(strength, 2),
        }
        signals.append(sig)
        position_hints[pos].append({
            "digit": pred,
            "strength": strength,
            "type": "position_slope",
        })

    for sig in _cross_period_slope_signals(numbers):
        signals.append(sig)
        pos = sig["position"]
        position_hints[pos].append({
            "digit": sig["predict_digit"],
            "strength": sig["strength"],
            "type": "cross_period_slope",
        })

    # 上期三位本身呈斜连（百→十→个等差），下期同向延伸作弱提示
    if len(numbers) >= 1:
        last = numbers[-1]
        s01 = _slope_step(last[0], last[1])
        s12 = _slope_step(last[1], last[2])
        if s01 is not None and s01 == s12:
            for pos in range(3):
                nxt = last[pos] + s01
                if 0 <= nxt <= 9:
                    position_hints[pos].append({
                        "digit": nxt,
                        "strength": 0.6,
                        "type": "in_draw_slope",
                    })
            signals.append({
                "type": "in_draw_slope",
                "chain": list(last),
                "step": s01,
                "label": (
                    f"上期位内斜连 {'+' if s01 > 0 else ''}{s01} "
                    f"({last[0]}→{last[1]}→{last[2]})，下期各位可顺势延伸"
                ),
                "position_hints": [
                    {"position_name": POS_NAMES_3D[i], "digit": last[i] + s01}
                    for i in range(3)
                    if 0 <= last[i] + s01 <= 9
                ],
            })

    return {
        "active": len(signals) > 0,
        "signal_count": len(signals),
        "signals": signals,
        "position_hints": {
            POS_NAMES_3D[i]: position_hints[i] for i in range(3)
        },
        "note": "斜连为走势辅助信号；历史回测命中率接近随机，请与和值/共现等一并参考。",
    }


def slope_triplet_bonus(a, b, c, meta):
    """直选组合与斜连关注码吻合时的加分。"""
    slope = meta.get("slope") or {}
    hints = slope.get("position_hints") or {}
    bonus = 0.0
    digits = (a, b, c)
    for pos, name in enumerate(POS_NAMES_3D):
        for hint in hints.get(name, []):
            if hint.get("digit") == digits[pos]:
                bonus += W_SLOPE_MATCH * float(hint.get("strength", 1.0))
    return bonus


def backtest_slope_patterns(numbers, trials=100):
    """斜连信号独立回测（分位预测是否命中）。"""
    pos_hit = pos_total = 0
    cross_hit = cross_total = 0
    start = max(SLOPE_MIN_CHAIN + 1, len(numbers) - trials)

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        slope = analyze_slope_patterns(train)
        for sig in slope.get("signals", []):
            if sig["type"] == "position_slope":
                pos_total += 1
                if actual[sig["position"]] == sig["predict_digit"]:
                    pos_hit += 1
            elif sig["type"] == "cross_period_slope":
                cross_total += 1
                if actual[sig["position"]] == sig["predict_digit"]:
                    cross_hit += 1

    return {
        "trials": trials,
        "position_slope_hit": pos_hit,
        "position_slope_total": pos_total,
        "position_slope_rate": round(pos_hit / pos_total, 4) if pos_total else 0.0,
        "cross_slope_hit": cross_hit,
        "cross_slope_total": cross_total,
        "cross_slope_rate": round(cross_hit / cross_total, 4) if cross_total else 0.0,
        "baseline_single_pos": 0.10,
    }


def entropy_model(numbers, min_appear_window=30):
    """熵值模型：统计数字熵、和值熵、跨度熵，计算长期未出现号码的奖励
    
    参数：
        numbers: 历史开奖号码列表
        min_appear_window: 最小统计窗口（期数）
    
    返回：
        熵值奖励字典 {digit: entropy_bonus}
    
    注意：实盘版本已关闭熵值奖励，所谓"长期未出现"并不会提高下一期出现概率，容易形成追冷号。
    """
    # 实盘版本：关闭熵值奖励，避免追冷
    return {d: 0.0 for d in range(10)}


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


def pair_bonus(triple, high_pairs):
    """计算号码组合中的数字配对奖励（使用预计算的高频对子）"""
    bonus = 0.0
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
        recent_recommendations: 最近推荐历史列表（新格式：[{"period": ..., "recommendations": [...]}, ...]）
    
    返回：
        penalized_pool: 应用惩罚后的推荐池
    """
    if not recent_recommendations:
        return pool
    
    # 扁平化最近推荐历史
    recent_set = set()
    consecutive_count = {}
    
    for entry in recent_recommendations[-RECENT_RECOMMEND_WINDOW:]:
        # 兼容新格式（字典）和旧格式（列表）
        if isinstance(entry, dict):
            rec_list = entry.get("recommendations", [])
        else:
            rec_list = entry
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


def load_recent_3d_recommendations():
    """加载最近推荐历史"""
    try:
        return kv_store.load('lottery3d_recent_recommend', [])
    except Exception as e:
        log.error(f"加载推荐历史失败: {e}")
        return []


def recent_zu6_digit_penalty(score, recent_zu6, base=ZU6_RECENT_PENALTY, decay=ZU6_RECENT_DECAY):
    """对近期组六四码用过的数字按新近度降权，返回调整后的数字评分列表。

    3D 选哪些码无 edge（任意 4 互异码组六覆盖率恒为 4*6/1000），故轮换零命中代价。
    最近一期惩罚最重，越久远越轻；连续多期出现的数字累计惩罚最大，优先被换出。
    """
    base = min(float(base), 3.0)
    adj = list(score)
    if not recent_zu6:
        return adj
    for age, entry in enumerate(reversed(recent_zu6[-ZU6_RECENT_WINDOW:])):
        digits = entry.get("digits", []) if isinstance(entry, dict) else entry
        w = base * (decay ** age)
        for d in digits:
            if 0 <= d < len(adj):
                adj[d] -= w
    return adj


def load_recent_zu6_four():
    """加载最近组六四码历史"""
    try:
        return kv_store.load('lottery3d_recent_zu6', [])
    except Exception as e:
        log.error(f"加载组六四码历史失败: {e}")
        return []


def save_recent_zu6_four(period, digits):
    """保存组六四码历史（按期号去重，仅保留最近 N 期）"""
    try:
        history = load_recent_zu6_four()
        if history and isinstance(history[-1], dict) and history[-1].get("period") == period:
            history[-1]["digits"] = list(digits)
        else:
            history.append({"period": period, "digits": list(digits)})
        history = history[-ZU6_RECENT_WINDOW:]
        kv_store.save('lottery3d_recent_zu6', history)
    except Exception as e:
        log.error(f"保存组六四码历史失败: {e}")


def save_recent_3d_recommendations(period, recommendations):
    """保存推荐历史（按期号去重）
    
    参数：
        period: 目标期号
        recommendations: 推荐号码列表
    
    说明：同一期号多次调用时，只会保存最后一次的推荐，避免重复写入。
    推荐历史必须以"期"为单位，不要以"页面调用次数"为单位。
    """
    try:
        # 加载现有历史
        history = load_recent_3d_recommendations()

        # 按期号去重：如果已有相同期号，更新推荐；否则添加新记录
        if (
            history
            and isinstance(history[-1], dict)
            and history[-1].get("period") == period
        ):
            # 更新当前期的推荐（覆盖）
            history[-1]["recommendations"] = recommendations
            history[-1]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 添加新期记录
            history.append({
                "period": period,
                "recommendations": recommendations,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        # 保持最近 N 期
        history = history[-RECENT_RECOMMEND_WINDOW:]

        kv_store.save('lottery3d_recent_recommend', history)
        log.info(f"推荐历史已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存推荐历史失败: {e}")


def load_online_predictions():
    """加载线上预测记录"""
    try:
        return kv_store.load('lottery3d_online_predictions', [])
    except Exception as e:
        log.error(f"加载线上预测记录失败: {e}")
        return []


def save_online_prediction(period, last_draw, zhixuan_top3, zhixuan, danma, kill):
    """保存线上预测记录"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(ONLINE_PREDICTION_FILE), exist_ok=True)
        
        # 加载现有记录
        records = load_online_predictions()
        
        # 创建新记录
        record = {
            "version": PREDICTOR_VERSION,
            "period": period,
            "last_draw": last_draw,
            "zhixuan_top3": [item["num"] for item in zhixuan_top3],
            "zhixuan": [item["num"] for item in zhixuan],
            "danma": danma,
            "kill": kill,
            "actual": None,
            "settled": False,
            "hit_top3": False,
            "hit_top30": False,
            "ge2_digit": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 检查是否已存在相同期数的记录
        existing_index = None
        for i, r in enumerate(records):
            if r["period"] == period:
                existing_index = i
                break
        
        if existing_index is not None:
            old_record = records[existing_index]
            # 如果已结算，不覆盖（保留原始预测记录）
            if old_record.get("settled"):
                log.info(f"预测记录已结算，跳过更新: {period}")
                return
            # 如果未结算，只补充空字段，不覆盖 zhixuan / top3（保留首次发布）
            record["zhixuan_top3"] = old_record["zhixuan_top3"]
            record["zhixuan"] = old_record["zhixuan"]
            record["danma"] = old_record["danma"]
            record["kill"] = old_record["kill"]
            record["created_at"] = old_record["created_at"]  # 保留首次创建时间
            records[existing_index] = record
        else:
            records.append(record)
        
        kv_store.save('lottery3d_online_predictions', records)
        log.info(f"线上预测记录已保存: {period}")
    except Exception as e:
        log.error(f"保存线上预测记录失败: {e}")


def max_digit_overlap(actual_s, candidates):
    """候选号码中与开奖号的最大数字重合数（ multiset 计数）"""
    actual_counter = Counter(actual_s)
    if not candidates:
        return 0
    return max(
        sum((actual_counter & Counter(num)).values())
        for num in candidates
    )


def settle_prediction(record, actual):
    """结算预测记录（赛后回填）"""
    top3 = record["zhixuan_top3"]
    top30 = record["zhixuan"]

    actual_s = "".join(map(str, actual))

    record["actual"] = actual_s
    record["settled"] = True
    record["hit_top3"] = actual_s in top3
    record["hit_top30"] = actual_s in top30
    record["ge2_digit"] = max_digit_overlap(actual_s, top30) >= 2

    return record


def settle_pending_online_predictions(periods, numbers):
    """根据最新开奖数据结算未回填的线上预测记录"""
    records = load_online_predictions()
    if not records:
        return 0

    period_index = {p: i for i, p in enumerate(periods)}
    changed = False
    settled_count = 0

    for record in records:
        if record.get("settled"):
            continue
        base_period = record.get("period")
        idx = period_index.get(base_period)
        if idx is None or idx + 1 >= len(numbers):
            continue
        settle_prediction(record, numbers[idx + 1])
        record["draw_period"] = periods[idx + 1]
        changed = True
        settled_count += 1

    if changed:
        try:
            kv_store.save('lottery3d_online_predictions', records)
            log.info(f"线上预测已结算 {settled_count} 条")
        except Exception as e:
            log.error(f"保存线上预测结算结果失败: {e}")

    # 结算三路策略记录
    settle_strategy_records(periods, numbers)

    if settled_count > 0:
        try:
            refresh_persisted_window_weights(numbers, periods[-1] if periods else None)
        except Exception as e:
            log.warning(f"回填后刷新窗口权重失败: {e}")

    return settled_count


def calculate_online_stats():
    """计算线上实盘命中率统计"""
    records = load_online_predictions()
    
    settled = [r for r in records if r["settled"]]
    unsettled = [r for r in records if not r["settled"]]
    
    n = len(settled)
    if n == 0:
        return {
            "total_records": len(records),
            "settled_count": 0,
            "unsettled_count": len(unsettled),
            "hit_top3_rate": 0.0,
            "hit_top30_rate": 0.0,
            "ge2_digit_rate": 0.0,
            "by_version": {},
        }
    
    hit_top3 = sum(1 for r in settled if r["hit_top3"])
    hit_top30 = sum(1 for r in settled if r["hit_top30"])
    ge2_digit = sum(1 for r in settled if r["ge2_digit"])
    
    # 按版本统计
    by_version = {}
    for r in settled:
        version = r["version"]
        if version not in by_version:
            by_version[version] = {"count": 0, "hit_top3": 0, "hit_top30": 0}
        by_version[version]["count"] += 1
        if r["hit_top3"]:
            by_version[version]["hit_top3"] += 1
        if r["hit_top30"]:
            by_version[version]["hit_top30"] += 1
    
    for v in by_version:
        by_version[v]["hit_top3_rate"] = by_version[v]["hit_top3"] / by_version[v]["count"]
        by_version[v]["hit_top30_rate"] = by_version[v]["hit_top30"] / by_version[v]["count"]
    
    return {
        "total_records": len(records),
        "settled_count": n,
        "unsettled_count": len(unsettled),
        "hit_top3_count": hit_top3,
        "hit_top3_rate": hit_top3 / n,
        "hit_top30_count": hit_top30,
        "hit_top30_rate": hit_top30 / n,
        "ge2_digit_count": ge2_digit,
        "ge2_digit_rate": ge2_digit / n,
        "by_version": by_version,
    }


def recommendation_stability(current, history):
    """计算推荐稳定度（最近7次推荐的重叠率）
    
    参数：
        current: 当前推荐号码列表
        history: 历史推荐列表（新格式：[{"period": ..., "recommendations": [...]}, ...]）
    
    返回：
        stability: 稳定度分数 (0.0-1.0)
    """
    current_set = set(current)
    scores = []

    for old_entry in history[-7:]:
        # 兼容新格式（字典）和旧格式（列表）
        if isinstance(old_entry, dict):
            old = old_entry.get("recommendations", [])
        else:
            old = old_entry
        old_set = set(old)
        if not old_set:
            continue
        overlap = len(current_set & old_set) / len(current_set)
        scores.append(overlap)

    return sum(scores) / len(scores) if scores else 0.0


def get_stability_level(stability):
    """获取稳定度等级"""
    if stability > 0.8:
        return "high"  # 过度稳定
    elif stability < 0.3:
        return "low"   # 过度随机
    else:
        return "normal"  # 正常


def adjust_exploration_rate(stability):
    """根据稳定度调整探索率"""
    if stability > 0.8:
        return 0.25
    elif stability < 0.3:
        return 0.08
    else:
        return EXPLORATION_RATE


def fuse_rule_ml(rule_list, ml_list, top_n=30, rule_weight=0.55, ml_weight=0.45, score=None, danma=None, kill=None, meta=None):
    """融合规则模型和ML模型的推荐结果（支持动态权重）
    
    参数：
        rule_list: 规则模型推荐列表 [{"num": "...", "score": ..., "detail": {...}}, ...]
        ml_list: ML模型推荐列表 [{"num": "...", "model_score": ...}, ...]
        top_n: 最终推荐数量
        rule_weight: 规则模型权重（基于回测表现动态计算）
        ml_weight: ML模型权重（基于回测表现动态计算）
        score: 数字评分数组（用于构建detail）
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
    
    返回：
        fused: 融合后的推荐列表，包含置信度标签和detail
    """
    rule_rank = {x["num"]: i for i, x in enumerate(rule_list)}
    ml_rank = {x["num"]: i for i, x in enumerate(ml_list)}
    
    # 保留规则模型的detail映射
    rule_detail = {x["num"]: x.get("detail") for x in rule_list}

    all_nums = set(rule_rank) | set(ml_rank)

    fused = []
    for num in all_nums:
        r = rule_rank.get(num, 999)
        m = ml_rank.get(num, 999)

        fuse_score = 0.0
        fuse_score += max(0, 100 - r) * rule_weight
        fuse_score += max(0, 100 - m) * ml_weight

        in_rule = num in rule_rank
        in_ml = num in ml_rank
        if in_rule and in_ml:
            fuse_score += 20
            tag = "high_confidence"
        elif in_rule:
            tag = "rule_preferred"
        elif in_ml:
            tag = "exploration"
        else:
            tag = "other"

        fused.append((fuse_score, num, tag, in_rule))

    fused.sort(reverse=True)
    
    result = []
    for fuse_score, num, tag, in_rule in fused[:top_n]:
        # 获取规则模型的detail
        detail = rule_detail.get(num)
        
        # 如果没有detail且提供了必要参数，尝试构建detail
        if detail is None and score is not None and danma is not None and kill is not None and meta is not None:
            a, b, c = int(num[0]), int(num[1]), int(num[2])
            detail = triplet_weight_detail(a, b, c, score, danma, kill, meta)
        
        result.append({
            "num": num,
            "score": round(fuse_score, 2),  # 统一字段名，兼容页面打印
            "fuse_score": round(fuse_score, 2),
            "tag": tag,
            "in_rule": num in rule_rank,
            "in_ml": num in ml_rank,
            "rule_rank": rule_rank.get(num),
            "ml_rank": ml_rank.get(num),
            "detail": detail,
        })
    
    return result


def load_recent_rule_performance():
    """加载缓存的规则模型最近表现（避免每次都跑回测）
    
    返回：
        dict: 包含 top30_rate, top3_rate, top100_rate, actual_rank_avg 等指标
    """
    try:
        perf = kv_store.load('lottery3d_rule_performance', {})
        return perf
    except Exception as e:
        log.error(f"加载规则模型表现失败: {e}")
        return {
            "top30_rate": 0.03,
            "top3_rate": 0.003,
            "top100_rate": 0.1,
            "actual_rank_avg": 500,
            "actual_rank_median": 500,
        }


def load_latest_ml_performance():
    """加载最近一次ML回测表现（用于融合权重计算）
    
    返回：
        包含top30_rate、top3_rate、actual_rank_avg等指标的字典
    """
    try:
        history = kv_store.load('lottery3d_ml_backtest_history', [])
        return history[-1] if history else {}
    except Exception as e:
        log.error(f"加载ML回测表现失败: {e}")
        return {}


def is_ml_eligible_from_backtest(period):
    """基于已保存的滚动回测结果判断ML是否符合准入条件
    
    准入条件：
        1. 存在最近的ML回测记录
        2. 回测记录未过期（模型版本、训练窗口、期号校验）
        3. Top30命中率高于随机基准(3%)
        4. 平均真实排名优于500
    
    参数：
        period: 当前期号
    
    返回：
        eligible: 是否符合准入条件
    """
    try:
        # 读取ML回测历史记录
        ml_backtest = kv_store.load('lottery3d_ml_backtest_history', [])
        
        if not ml_backtest:
            return False
        
        # 检查最近的回测结果
        recent = ml_backtest[-1] if ml_backtest else None
        if not recent:
            return False
        
        # 版本和期号校验
        record_period = recent.get('base_period')
        model_version = recent.get('model_version')
        
        # 检查回测记录是否过期（期号差异超过20期认为过期）
        if record_period and period:
            try:
                period_diff = abs(int(period) - int(record_period))
                if period_diff > 20:
                    log.info(f"ML回测记录过期（期号差异: {period_diff}期）")
                    return False
            except:
                pass
        
        # 检查模型版本是否匹配（当前使用ml-v6）
        if model_version != "ml-v6":
            log.info(f"ML模型版本不匹配（记录: {model_version}, 当前: ml-v6）")
            return False
        
        # 检查命中率是否高于基准
        top30_rate = recent.get('top30_rate', 0.0)
        actual_rank_avg = recent.get('actual_rank_avg', 1000)
        
        baseline_rate = RECOMMEND_GROUPS / 1000.0  # 3%
        
        # 准入条件：Top30命中率高于基准，且平均排名优于500
        if top30_rate > baseline_rate and actual_rank_avg < 500:
            return True
        
        return False
    except Exception as e:
        log.error(f"检查ML准入条件失败: {e}")
        return False


def save_strategy_records(period, rule_only, ml_only, fused):
    """保存三套策略记录（用于后续对比分析）
    
    参数：
        period: 期号
        rule_only: 规则模型推荐列表
        ml_only: ML模型推荐列表
        fused: 融合推荐列表
    
    注意：首次发布的策略记录不会被覆盖（即使未结算），确保策略对比的准确性
    """
    try:
        history = kv_store.load('lottery3d_strategy_records', [])
        
        # 检查是否已存在同一期的记录
        existing_record = next((h for h in history if h["period"] == period), None)
        
        if existing_record:
            # 如果已存在，检查是否已结算
            if existing_record.get("settled"):
                # 已结算，不覆盖
                log.info(f"策略记录已结算，跳过更新（期号: {period}）")
                return
            # 未结算也保留首次发布，不覆盖
            log.info(f"策略记录已存在，保留首次发布（期号: {period}）")
            return
        
        # 不存在记录，创建新记录
        record = {
            "period": period,
            "rule_only": rule_only,
            "ml_only": ml_only,
            "fused": fused,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settled": False,
            "revision": 1,  # 首次发布版本
        }
        
        history.append(record)
        
        # 只保留最近200期记录
        history = history[-200:]
        
        kv_store.save('lottery3d_strategy_records', history)
        log.info(f"策略记录已保存（期号: {period}）")
    except Exception as e:
        log.error(f"保存策略记录失败: {e}")


def settle_strategy_records(periods, numbers):
    """结算三路策略记录（规则模型/ML模型/融合模型分别统计）
    
    参数：
        periods: 期号列表
        numbers: 号码列表
    """
    try:
        history = kv_store.load('lottery3d_strategy_records', [])
        if not history:
            return
        
        index_map = {period: i for i, period in enumerate(periods)}
        changed = False
        
        for row in history:
            if row.get("settled"):
                continue
            
            idx = index_map.get(row["period"])
            if idx is None or idx + 1 >= len(numbers):
                continue
            
            actual = "".join(map(str, numbers[idx + 1]))
            row["actual"] = actual
            row["settled"] = True
            row["draw_period"] = periods[idx + 1]
            
            # 分别统计三条策略的表现
            for name in ("rule_only", "ml_only", "fused"):
                nums = row.get(name, [])
                row[f"{name}_hit_top3"] = actual in nums[:3]
                row[f"{name}_hit_top30"] = actual in nums[:30]
                row[f"{name}_hit_top100"] = actual in nums[:100]
                row[f"{name}_rank"] = nums.index(actual) + 1 if actual in nums else 1001
            
            changed = True
        
        if changed:
            kv_store.save('lottery3d_strategy_records', history)
            log.info("三路策略记录已结算")
    except Exception as e:
        log.error(f"结算策略记录失败: {e}")


def generate_strategy_recommendations(rule_list, ml_list, danma, kill):
    """生成三套推荐策略
    
    参数：
        rule_list: 规则模型推荐列表
        ml_list: ML模型推荐列表
        danma: 胆码列表
        kill: 杀码列表
    
    返回：
        strategy_recommendations: 包含三套策略的推荐结果
    """
    rule_set = set(r["num"] for r in rule_list)
    ml_set = set(m["num"] for m in ml_list)
    
    # 保守策略：规则 + ML 交集
    conservative = [r for r in rule_list if r["num"] in ml_set][:10]
    
    # 均衡策略：规则主导，少量探索
    balanced = []
    rule_added = set()
    for r in rule_list[:20]:
        balanced.append({"num": r["num"], "score": r.get("score", 0), "source": "rule"})
        rule_added.add(r["num"])
    
    # 补充少量ML独有号码
    ml_only = [m for m in ml_list if m["num"] not in rule_added][:5]
    for m in ml_only:
        balanced.append({"num": m["num"], "score": m.get("model_score", 0), "source": "ml"})
    
    # 探索策略：ML独有 + 冷号特征
    explore = []
    ml_explore = [m for m in ml_list if m["num"] not in rule_set][:8]
    for m in ml_explore:
        explore.append({"num": m["num"], "score": m.get("model_score", 0), "source": "ml_only"})
    
    return {
        "conservative": conservative,
        "balanced": balanced[:20],
        "explore": explore[:10],
    }


def select_strategy_mode(stability, model_lift, recent_hit_rate, actual_rank_avg):
    """根据模型表现自动选择推荐模式
    
    参数：
        stability: 推荐稳定度
        model_lift: 模型相对随机基准的提升
        recent_hit_rate: 最近线上命中率
        actual_rank_avg: 真实号码平均排名
    
    返回：
        mode: 推荐模式（conservative/balanced/explore）
        reason: 选择理由
    """
    if model_lift <= 0:
        return "explore", "模型未明显优于随机基准，需要探索"

    if stability > 0.8 and recent_hit_rate < 0.03:
        return "explore", "推荐过度稳定且命中率偏低，增加探索"

    if actual_rank_avg <= 250 and model_lift > 0.01:
        return "conservative", "模型排名表现优秀，采用保守策略"

    return "balanced", "模型有提升但需保持多样性，采用均衡策略"


def recommend_budget_level(model_lift, stability, recent_online_rate):
    """根据模型表现推荐资金/注数等级
    
    参数：
        model_lift: 模型相对随机基准的提升
        stability: 推荐稳定度
        recent_online_rate: 最近线上命中率
    
    返回：
        budget_info: 资金建议信息
    """
    if model_lift <= 0:
        return {
            "level": "低",
            "suggest_count": 10,
            "reason": "模型未明显优于随机基准"
        }

    if model_lift > 0.015 and recent_online_rate >= 0.03:
        return {
            "level": "中",
            "suggest_count": 20,
            "reason": "模型近期表现略优于随机"
        }

    return {
        "level": "观察",
        "suggest_count": 10,
        "reason": "样本不足或优势不稳定"
    }


def auto_recommend_count(model_lift, rank_top100_rate, online_hit_rate):
    """根据模型表现自动调整推荐注数
    
    参数：
        model_lift: 模型相对随机基准的提升
        rank_top100_rate: Top100覆盖率
        online_hit_rate: 线上命中率
    
    返回：
        count: 推荐注数
        reason: 调整理由
    """
    if model_lift <= 0:
        return 10, "模型无明显优势，减少推荐注数"

    if rank_top100_rate >= 0.18 and online_hit_rate >= 0.03:
        return 30, "Top100覆盖率和线上命中率均良好"

    if rank_top100_rate >= 0.12:
        return 20, "Top100覆盖率尚可"

    return 15, "模型优势有限，保持适中注数"


def backtest_dan_kill(numbers, trials=100):
    """胆码/杀码独立回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 胆码和杀码的回测统计
    """
    dan_hit1 = 0
    dan_hit2 = 0
    kill_fail = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_set = set(actual)

        ww = default_window_weights()
        meta = build_ranking_meta(train, ww)
        sc, _ = ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        dan, _, kill, _ = pick_dan_tuo_kill(sc, enable_danma_random=False)

        hit_count = len(set(dan) & actual_set)

        if hit_count >= 1:
            dan_hit1 += 1
        if hit_count >= 2:
            dan_hit2 += 1

        if set(kill) & actual_set:
            kill_fail += 1

    return {
        "trials": trials,
        "dan_hit1_rate": dan_hit1 / trials,
        "dan_hit2_rate": dan_hit2 / trials,
        "kill_fail_rate": kill_fail / trials,
    }


def backtest_form_prediction(numbers, trials=100):
    """形态预测命中率回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 形态预测回测统计
    """
    hit = 0
    zu6_hit = 0
    zu6_total = 0
    zu3_hit = 0
    zu3_total = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual_form = classify_form(numbers[i])

        ww = default_window_weights()
        pred = analyze_form_probability(train, window_weights=ww)
        pred_form = max(pred["blend_p"].items(), key=lambda x: x[1])[0]

        if pred_form == actual_form:
            hit += 1

        if pred_form == "zu6":
            zu6_total += 1
            if actual_form == "zu6":
                zu6_hit += 1

        if pred_form == "zu3":
            zu3_total += 1
            if actual_form == "zu3":
                zu3_hit += 1

    return {
        "trials": trials,
        "form_top1_rate": hit / trials,
        "zu6_precision": zu6_hit / zu6_total if zu6_total else 0,
        "zu3_precision": zu3_hit / zu3_total if zu3_total else 0,
    }


def backtest_sum_span_interval(numbers, trials=100):
    """和值/跨度区间独立回测
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
    
    返回：
        result: 和值/跨度区间回测统计
    """
    sum_hit_2 = 0
    sum_hit_3 = 0
    sum_hit_4 = 0
    span_hit_1 = 0
    span_hit_2 = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_sum = sum(actual)
        actual_span = max(actual) - min(actual)

        ww = default_window_weights()
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta = build_ranking_meta(train, ww, sums, spans)

        sum_center = meta["sum_center"]
        span_center = meta["span_center"]

        if abs(actual_sum - sum_center) <= 2:
            sum_hit_2 += 1
        if abs(actual_sum - sum_center) <= 3:
            sum_hit_3 += 1
        if abs(actual_sum - sum_center) <= 4:
            sum_hit_4 += 1

        if abs(actual_span - span_center) <= 1:
            span_hit_1 += 1
        if abs(actual_span - span_center) <= 2:
            span_hit_2 += 1

    return {
        "trials": trials,
        "sum_hit_2_rate": sum_hit_2 / trials,
        "sum_hit_3_rate": sum_hit_3 / trials,
        "sum_hit_4_rate": sum_hit_4 / trials,
        "span_hit_1_rate": span_hit_1 / trials,
        "span_hit_2_rate": span_hit_2 / trials,
    }


def select_diverse_pool(
    pool,
    top_n=30,
    candidate_size=SERVED_POOL_CANDIDATE_SIZE,
    use_diversity=True,
    use_correlation=True,
):
    """贪心选池：从更大候选集中兼顾原始分、数字覆盖与去相关
    
    参数：
        pool: 候选池 [(权重, 号码字符串), ...]
        top_n: 目标推荐数量
        candidate_size: 候选集大小
        use_diversity: 是否启用数字覆盖奖励
        use_correlation: 是否启用重合惩罚
    """
    candidates = sorted(pool, key=lambda x: -x[0])[:candidate_size]
    selected = []
    selected_sets = []

    while candidates and len(selected) < top_n:
        best_item = None
        best_score = -float("inf")
        union_digits = set().union(*selected_sets) if selected_sets else set()

        for w, num in candidates:
            digits = set(num)
            
            # 重合惩罚（可选）
            overlap_penalty = (
                sum(
                    CORRELATION_PENALTY
                    for old_digits in selected_sets
                    if len(digits & old_digits) >= CORRELATION_THRESHOLD
                )
                if use_correlation else 0.0
            )
            
            # 数字覆盖奖励（可选）
            new_cover = (
                len(digits - union_digits)
                if use_diversity and selected_sets
                else 0.0
            )
            
            final_score = w + new_cover * DIVERSITY_WEIGHT - overlap_penalty

            if final_score > best_score:
                best_score = final_score
                best_item = (w, num)

        if best_item is None:
            break
        selected.append(best_item)
        selected_sets.append(set(best_item[1]))
        candidates.remove(best_item)

    return selected


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

    # 近期趋势偏移（可配置开关）
    # 默认关闭，避免追涨杀跌，等消融回测证明有效再开启
    if RECENT_SUM_SPAN_SHIFT > 0 and len(recent_s) >= 5:
        recent5_s = recent_s[-5:]
        recent5_p = recent_p[-5:]
        avg5_s = sum(recent5_s) / 5
        avg5_p = sum(recent5_p) / 5
        sum_center = (
            sum_center * (1 - RECENT_SUM_SPAN_SHIFT)
            + avg5_s * RECENT_SUM_SPAN_SHIFT
        )
        span_center = (
            span_center * (1 - RECENT_SUM_SPAN_SHIFT)
            + avg5_p * RECENT_SUM_SPAN_SHIFT
        )

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
        # 和值/跨度都是整数统计量，中心必须取整：用整数容差(±k)去框一个分数中心会
        # 不对称地少框一个取值（如 |v-4.5|<=1 只含{4,5}，而 |v-5|<=1 含{4,5,6}）。
        # 实测取整后 跨度±1 命中 30.8%→45%、和值±2 28.8%→34.6%。四舍五入到最近整数
        # 同时贴近分布众数(和值13/14、跨度5)，对平滑高斯打分几乎无影响。
        "sum_center": float(round(sum_center)),
        "span_center": float(round(span_center)),
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
    flags = FEATURE_FLAGS

    freq_all = exp_weighted_counts([d for n in recent for d in n])

    if flags.get("hot", True):
        for d, _ in freq_all.most_common(4):
            score[d] += W_HOT_GLOBAL

        for pos in range(3):
            pos_freq = exp_weighted_counts([n[pos] for n in recent])
            for d, _ in pos_freq.most_common(3):
                score[d] += W_HOT_POS

    if flags.get("markov", True):
        for pos in range(3):
            trans = build_markov(numbers, pos)
            prev_d = last[pos]
            row = trans.get(prev_d, Counter())
            for d, p in markov_prob_smoothed(row, range(10)).items():
                markov_score = W_MARKOV * p
                score[d] += min(markov_score, MARKOV_MAX_SCORE)

            if len(numbers) >= 2:
                trans2 = build_markov2(numbers, pos)
                prev2 = numbers[-2][pos]
                prev1 = last[pos]
                row2 = trans2.get((prev2, prev1), Counter())
                for d, p in markov_prob_smoothed(row2, range(10)).items():
                    markov2_score = W_MARKOV2 * p
                    score[d] += min(markov2_score, MARKOV_MAX_SCORE)

    if flags.get("miss", True):
        for d in range(10):
            mv = miss_value(numbers, d)
            if mv >= 20:
                score[d] += W_MISS_HIGH * (1 + mv / 20)
            elif mv >= 12:
                score[d] += W_MISS_MID

        miss_cycle_bonus_scores = miss_cycle_bonus(numbers)
        for d in range(10):
            score[d] += miss_cycle_bonus_scores.get(d, 0.0)

        entropy_bonus = entropy_model(numbers)
        for d in range(10):
            score[d] += entropy_bonus.get(d, 0.0)

        rebound_bonus = rebound_model(numbers)
        for d in range(10):
            score[d] += rebound_bonus.get(d, 0.0)

    if flags.get("neighbor", True):
        for d in set(last):
            score[d] += w_last

        nb = set()
        for d in last:
            nb.update(neighbor(d))
        for d in nb:
            score[d] += W_NEIGHBOR

    if flags.get("road", True):
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
    
    # 注意：熵值奖励和回补奖励已经在 digit_scores() 内添加过，
    # 这里不再重复添加，避免双重加权
    # entropy_model() 和 rebound_model() 的奖励已在 digit_scores() 中处理
    
    return combined, freq_combined


def zu6_digit_scores(numbers, window_weights=None, dynamic=None):
    """组六单码评分：全局模型为主，叠加组六历史频率（回测优于纯组六特征）。"""
    if window_weights is None:
        window_weights = default_window_weights()

    score, _ = ensemble_digit_scores(numbers, window_weights, dynamic=dynamic)

    zu6_draws = [n for n in numbers if classify_form(n) == "zu6"]
    if zu6_draws:
        recent_zu6 = _recent_slice(zu6_draws, min(90, len(zu6_draws)))
        freq = exp_weighted_counts([d for n in recent_zu6 for d in set(n)])
        peak = max(freq.values()) if freq else 1.0
        for d, cnt in freq.items():
            score[d] += W_ZU6_HOT * 0.35 * (cnt / peak)

        for pos in range(3):
            pos_sc = ensemble_position_digit_scores(
                numbers, pos, window_weights, dynamic=dynamic
            )
            for d, _ in sorted(enumerate(pos_sc), key=lambda x: -x[1])[:3]:
                score[d] += W_ZU6_POS * 0.25

    return score


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    """单码分位评分（百/十/个），与主模型共用 FEATURE_FLAGS"""
    recent = [n[position] for n in _recent_slice(numbers, window)]
    last_d = numbers[-1][position]
    sc = [0.0] * 10
    dyn = dynamic or {}
    w_last = dyn.get("w_last_appear", W_LAST_APPEAR)
    pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
    flags = FEATURE_FLAGS

    if flags.get("hot", True):
        for d, _ in exp_weighted_counts(recent).most_common(4):
            sc[d] += W_HOT_POS + 1

    if flags.get("markov", True):
        trans = build_markov(numbers, position)
        row = trans.get(last_d, Counter())
        for d, p in markov_prob_smoothed(row, range(10)).items():
            markov_score = W_MARKOV * p
            sc[d] += min(markov_score, MARKOV_MAX_SCORE)
        if len(numbers) >= 2:
            trans2 = build_markov2(numbers, position)
            prev2_d = numbers[-2][position]
            row2 = trans2.get((prev2_d, last_d), Counter())
            for d, p in markov_prob_smoothed(row2, range(10)).items():
                markov2_score = W_MARKOV2 * p
                sc[d] += min(markov2_score, MARKOV_MAX_SCORE)

    if flags.get("miss", True):
        for d in range(10):
            miss_p = miss_value(numbers, d, position=position)
            if miss_p >= 20:
                sc[d] += W_MISS_HIGH * (1 + miss_p / 20)
            elif miss_p >= 12:
                sc[d] += W_MISS_MID

    if flags.get("neighbor", True):
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


def load_persisted_window_weights():
    """读取持久化的动态窗口权重"""
    try:
        data = kv_store.load(WINDOW_WEIGHTS_KV_KEY)
        if not data or not isinstance(data.get("weights"), dict):
            return None
        weights = {int(k): float(v) for k, v in data["weights"].items()}
        scores = {int(k): float(v) for k, v in (data.get("scores") or {}).items()}
        return {"weights": weights, "scores": scores, "period": data.get("period")}
    except Exception as e:
        log.debug(f"读取窗口权重失败: {e}")
        return None


def save_persisted_window_weights(weights, scores, period=None):
    """持久化动态窗口权重"""
    try:
        payload = {
            "weights": {str(k): round(v, 6) for k, v in weights.items()},
            "scores": {str(k): round(v, 4) for k, v in (scores or {}).items()},
            "period": period,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        kv_store.save(WINDOW_WEIGHTS_KV_KEY, payload)
        log.info(f"窗口权重已持久化: period={period}")
    except Exception as e:
        log.warning(f"保存窗口权重失败: {e}")


def refresh_persisted_window_weights(numbers, period=None):
    """重新计算并持久化窗口权重（回填后或手动刷新时调用）"""
    weights, scores = compute_window_weights(numbers, enable_cache=False)
    save_persisted_window_weights(weights, scores, period)
    return weights, scores


def resolve_window_weights(numbers, compute_weights=False, period=None):
    """获取预测用窗口权重：优先持久化缓存，必要时重算"""
    if compute_weights:
        weights, scores = compute_window_weights(numbers, enable_cache=False)
        save_persisted_window_weights(weights, scores, period)
        return weights, scores

    persisted = load_persisted_window_weights()
    if persisted:
        return persisted["weights"], persisted.get("scores", {})

    if len(numbers) >= max(RECENT_WINDOWS) + 10:
        weights, scores = compute_window_weights(numbers, enable_cache=True)
        save_persisted_window_weights(weights, scores, period)
        return weights, scores

    return default_window_weights(), {}


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
            top = rank_triplets(
                sc, dan, kill, meta,
                top_n=ZHIXUAN_TOP3,
                enable_exploration=False,
                apply_noise=False,
                enable_cold_hot_balance=False,
                enable_diversity=False,
                enable_correlation=False,
                recent_recommendations=None,
            )
            top_nums = [t[1] for t in top]
            if act_s in top_nums:
                raw[w] += 1.0
            elif max_digit_overlap(act_s, top_nums) >= 2:
                raw[w] += 0.25

    prior = WINDOW_WEIGHT_PRIOR
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


def recommend_form_bet(form_prob, numbers):
    """形态主推：固定主推「组六」。

    组六约占全部开奖的 72%，是唯一数学最优的单形态投注。实测(600期)「跟随
    短期 markov/recent 概率最大者」与「永远押组六」命中率完全相同(均 73.8%)——
    短期信号无法击败 base rate，连续组六后"该出组三了"是赌徒谬误。故主推固定为
    组六，组三/豹子仅作小注分散（按其真实概率，不建议作主投）。
    """
    forms = [classify_form(n) for n in numbers]
    n = len(forms) or 1
    hist_cnt = Counter(forms)
    emp = {k: hist_cnt.get(k, 0) / n for k in THEORY_FORM_P}
    return {
        "primary": "zu6",
        "primary_label": FORM_LABELS["zu6"],
        "expected_hit_rate": round(emp["zu6"], 4),       # 经验 base rate(全历史组六占比)
        "theory_hit_rate": THEORY_FORM_P["zu6"],          # 理论 72%
        "empirical_form_p": {k: round(v, 4) for k, v in emp.items()},
        "blend_p": {k: round(v, 4) for k, v in form_prob["blend_p"].items()},
        "secondary": "zu3",
        "note": "主推组六(≈72%为数学最优单形态投注)；实测短期信号无法超越此基准，"
                "组三/豹子按真实概率仅作小注分散，不建议主投。投注组选6即覆盖该形态全部组合。",
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


def pick_zu6_four(score, kill=None, use_kill=ZU6_USE_KILL, numbers=None, pair_freq=None):
    """组六四码：在 Top 候选中组合优化选 4 码"""
    return pick_zu6_pool(
        score, kill, pool_size=ZU6_FOUR_SIZE,
        use_kill=use_kill, numbers=numbers, pair_freq=pair_freq,
    )


def zu6_notes_from_digits(digits):
    """N 码组六 → C(N,3) 注组六组合"""
    combos = [tuple(sorted(c)) for c in combinations(digits, 3)]
    return combos, ["".join(map(str, c)) for c in combos]


# 组六复式覆盖档位：单注价格（元）
TICKET_PRICE = 2


def build_zu6_coverage_tiers(score, kill=None, sizes=(4, 5, 6, 7), numbers=None):
    """组六复式覆盖档位：N 码 → C(N,3) 注，给出注数/成本/理论命中率。

    3D 为公平均匀摇奖，选哪些码无 edge（实测评分选码≈随机选码），
    唯一的杠杆是覆盖多少注：持有 K 注互异组六，无条件命中率 = K*6/1000
    （命中需开奖为组六且三码全在所选码内）。本函数把各档位摊开，供按预算选择。
    """
    tiers = []
    for n in sizes:
        digits = pick_zu6_pool(score, kill, pool_size=n, numbers=numbers)
        combos, combo_strs = zu6_notes_from_digits(digits)
        notes = len(combos)
        tiers.append({
            "size": n,
            "digits_str": "".join(map(str, digits)),
            "notes": notes,
            "cost": notes * TICKET_PRICE,
            "hit_rate": round(notes * 6 / 1000.0, 4),  # 理论=实测命中率（无 edge）
            "combos": combo_strs,
        })
    return tiers


def _zu6_four_payload(label, digits):
    digits = sorted(int(d) for d in digits)
    combos, combo_strs = zu6_notes_from_digits(digits)
    return {
        "label": label,
        "digits": digits,
        "digits_str": "".join(map(str, digits)),
        "notes": len(combos),
        "cost": len(combos) * TICKET_PRICE,
        "hit_rate": round(len(combos) * 6 / 1000.0, 4),
        "combos": combo_strs,
    }


def _zu6_four_balance_score(combo, score, kill=None):
    digits = tuple(sorted(combo))
    kill_set = set(kill or [])
    base = sum(_effective_digit_score(score, d, kill) for d in digits)
    odd = sum(1 for d in digits if d % 2)
    big = sum(1 for d in digits if d >= 5)
    span = digits[-1] - digits[0]
    adjacent_pairs = sum(1 for a, b in zip(digits, digits[1:]) if b - a == 1)
    kill_count = sum(1 for d in digits if d in kill_set)
    return (
        base
        - abs(odd - 2) * 1.0
        - abs(big - 2) * 0.8
        + min(span, 8) * 0.15
        - adjacent_pairs * 0.35
        - kill_count * 1.2
    )


def build_zu6_four_variants(score, kill=None, limit=4, numbers=None):
    """Build several deterministic four-digit zu6 groups for coverage comparison."""
    kill_eff = kill if ZU6_USE_KILL else None
    rank = sorted(range(10), key=lambda d: -_effective_digit_score(score, d, kill_eff))
    candidate_pool = rank[:8]
    primary = tuple(pick_zu6_four(score, kill, numbers=numbers))
    variants = []
    seen = set()

    def add(label, digits):
        key = tuple(sorted(digits))
        if key in seen or len(key) != 4:
            return
        seen.add(key)
        variants.append(_zu6_four_payload(label, key))

    add("主推", primary)
    balanced = max(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill),
    )
    add("均衡", balanced)

    kill_set = set(kill or [])
    no_kill_pool = [d for d in rank if d not in kill_set][:6]
    if len(no_kill_pool) >= 4:
        add("避杀", no_kill_pool[:4])

    wide = max(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill) + (max(c) - min(c)) * 0.3,
    )
    add("扩散", wide)

    for combo in sorted(
        combinations(candidate_pool, 4),
        key=lambda c: _zu6_four_balance_score(c, score, kill),
        reverse=True,
    ):
        add("备选", combo)
        if len(variants) >= limit:
            break

    return variants[:limit]


def _effective_digit_score(score, digit, kill=None):
    """单码有效分：杀码降权而非排除"""
    kill_set = set(kill or [])
    return score[digit] - (W_KILL_PENALTY if digit in kill_set else 0.0)


def _zu6_combo_score(combo, score, kill=None, pair_freq=None):
    """组六 N 码组合得分：单码分 + 对内共现 + 奇偶大小均衡。"""
    digits = tuple(sorted(combo))
    val = sum(_effective_digit_score(score, d, kill) for d in digits)
    if pair_freq:
        for i in range(len(digits)):
            for j in range(i + 1, len(digits)):
                val += pair_freq.get((digits[i], digits[j]), 0.0) * W_ZU6_PAIR
    odd = sum(1 for d in digits if d % 2)
    big = sum(1 for d in digits if d >= 5)
    val -= abs(odd - len(digits) / 2) * 0.5
    val -= abs(big - len(digits) / 2) * 0.4
    return val


def pick_zu6_pool(
    score, kill=None, pool_size=ZU6_POOL_SIZE,
    use_kill=ZU6_USE_KILL, pair_freq=None, numbers=None,
):
    """组六复式选号：按组六专用分取 Top N（默认不用杀码）。"""
    kill_eff = kill if use_kill else None
    rank = sorted(range(10), key=lambda d: -_effective_digit_score(score, d, kill_eff))
    return sorted(rank[:pool_size])


def _blend_dan_score(score, meta):
    """胆码/杀码用分位融合评分，与直选分位排序一致。"""
    dan_score = list(score)
    if meta.get("pos_scores"):
        for d in range(10):
            pos_sum = sum(meta["pos_scores"][p][d] for p in range(3))
            dan_score[d] = score[d] * 0.45 + pos_sum * 0.55
    return dan_score


def _triplet_digit_base(a, b, c, score, meta):
    """直选三位基础分：分位评分为主，全局评分为辅。"""
    pos_scores = meta.get("pos_scores")
    if pos_scores and len(pos_scores) == 3:
        return (
            W_TRIPLET_POS * (pos_scores[0][a] + pos_scores[1][b] + pos_scores[2][c])
            + W_TRIPLET_GLOBAL * (score[a] + score[b] + score[c])
        )
    return score[a] + score[b] + score[c]


def triplet_weight(a, b, c, score, danma, kill, meta, features=None):
    """计算三位数组合的评分权重
    
    参数：
        a, b, c: 百位、十位、个位数字
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
        features: 特征开关字典（可选，默认为全局 FEATURE_FLAGS）
    """
    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    flags = features if features is not None else FEATURE_FLAGS
    numbers = meta.get("numbers", [])

    w = _triplet_digit_base(a, b, c, score, meta)
    for x in (a, b, c):
        if x in danma:
            w += W_DANMA_HIT
        if x in kill_set:
            w -= W_KILL_PENALTY

    s = a + b + c
    
    # 和值跨度特征
    if flags.get("sum_span", True):
        w += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

        span = max(a, b, c) - min(a, b, c)
        w += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

        if s in meta["hot_sum_set"]:
            w += 2.0
        if span in meta["hot_span_set"]:
            w += 1.5
        if (s % 10) in meta["sum_tail_top"]:
            w += 1.0

    # 连号奖励
    if flags.get("consecutive", True) and has_consecutive_digits(a, b, c):
        w += dyn.get("w_consecutive", W_CONSECUTIVE)

    # 上期同位重复、全重复、同集合惩罚
    if flags.get("lag1_repeat", True):
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

    # 奇偶比、大小比奖励
    if flags.get("ratio", True):
        oe = odd_even_key((a, b, c))
        bs = big_small_key((a, b, c))
        oe_freq = meta.get("oe_freq")
        bs_freq = meta.get("bs_freq")
        if oe_freq:
            w += W_RATIO_MATCH * oe_freq.get(oe, 0) / meta.get("oe_total", 1)
        if bs_freq:
            w += W_RATIO_MATCH * bs_freq.get(bs, 0) / meta.get("bs_total", 1)

    if flags.get("slope", True):
        w += slope_triplet_bonus(a, b, c, meta)
    
    # 数字配对奖励：使用 meta 预计算的高频对子
    high_pairs = meta.get("high_pairs") or set()
    if flags.get("pair", True) and high_pairs:
        w += pair_bonus((a, b, c), high_pairs)
    
    # 组三组六切换奖励：连续同形式出现后增加切换概率
    if flags.get("form_switch", True) and len(numbers) >= 5:
        form_bonus = form_switch_bonus(numbers)
        if a == b or a == c or b == c:
            w += form_bonus.get("zu3", 0.0)
        else:
            w += form_bonus.get("zu6", 0.0)

    # 形态先验：按真实形态概率加分(组六0.72/组三0.27/豹子0.01)，使推荐池形态分布贴合真实开奖。
    # 选哪些具体号无 edge(直选恒3%)，此项只调整推荐"长得像不像真实开奖"的形态构成。
    nd = len({a, b, c})
    w += W_FORM_PRIOR * (THEORY_FORM_P["zu6"] if nd == 3
                         else THEORY_FORM_P["zu3"] if nd == 2
                         else THEORY_FORM_P["baozi"])
    
    # 和值区间回归奖励：区间内加分，极端区间降权
    if flags.get("sum_span", True) and len(numbers) >= SUM_INTERVAL_WINDOW:
        sum_interval_info = sum_interval_bonus(numbers)
        w += sum_interval_info["bonus"].get(s, 0.0)
    
    return w


def triplet_weight_detail(a, b, c, score, danma, kill, meta):
    """计算三位数组合的详细得分分解，用于解释推荐原因
    
    参数：
        a, b, c: 百位、十位、个位数字
        score: 各数字评分数组
        danma: 胆码列表
        kill: 杀码列表
        meta: 元数据
    
    返回：
        detail: 包含各特征得分的字典
    """
    detail = {
        "base_digit": _triplet_digit_base(a, b, c, score, meta),
        "danma": 0.0,
        "kill": 0.0,
        "sum_span": 0.0,
        "pattern": 0.0,
        "last_repeat": 0.0,
        "ratio_match": 0.0,
        "pair": 0.0,
        "slope": 0.0,
        "form_switch": 0.0,
        "sum_interval": 0.0,
        "total": 0.0,
    }

    kill_set = set(kill or [])
    dyn = meta.get("dynamic") or {}
    flags = FEATURE_FLAGS

    for x in (a, b, c):
        if x in danma:
            detail["danma"] += W_DANMA_HIT
        if x in kill_set:
            detail["kill"] -= W_KILL_PENALTY

    s = a + b + c
    if flags.get("sum_span", True):
        detail["sum_span"] += 8.0 * gaussian_score(s, meta["sum_center"], SUM_SOFT_SIGMA)

        span = max(a, b, c) - min(a, b, c)
        detail["sum_span"] += 5.0 * gaussian_score(span, meta["span_center"], SPAN_SOFT_SIGMA)

        if s in meta["hot_sum_set"]:
            detail["sum_span"] += 2.0
        if span in meta["hot_span_set"]:
            detail["sum_span"] += 1.5
        if (s % 10) in meta["sum_tail_top"]:
            detail["sum_span"] += 1.0

    # 连号奖励
    if flags.get("consecutive", True) and has_consecutive_digits(a, b, c):
        detail["pattern"] += dyn.get("w_consecutive", W_CONSECUTIVE)

    # 上期同位重复、全重复、同集合惩罚
    if flags.get("lag1_repeat", True):
        last_draw = meta.get("last_draw")
        w_pos = dyn.get("w_pos_repeat", W_POS_REPEAT)
        pos_mult = dyn.get("pos_mult", [1.0, 1.0, 1.0])
        if last_draw:
            triple = (a, b, c)
            for i in range(3):
                if triple[i] == last_draw[i]:
                    detail["last_repeat"] += w_pos * pos_mult[i]
            if triple == tuple(last_draw):
                detail["last_repeat"] -= dyn.get("w_full_repeat_penalty", 0.0)
            elif set(triple) == set(last_draw):
                detail["last_repeat"] -= dyn.get("w_same_set_penalty", 0.0)

    # 奇偶比、大小比奖励
    if flags.get("ratio", True):
        oe = odd_even_key((a, b, c))
        bs = big_small_key((a, b, c))
        oe_freq = meta.get("oe_freq")
        bs_freq = meta.get("bs_freq")
        if oe_freq:
            detail["ratio_match"] += W_RATIO_MATCH * oe_freq.get(oe, 0) / meta.get("oe_total", 1)
        if bs_freq:
            detail["ratio_match"] += W_RATIO_MATCH * bs_freq.get(bs, 0) / meta.get("bs_total", 1)

    high_pairs = meta.get("high_pairs") or set()
    if flags.get("pair", True) and high_pairs:
        detail["pair"] += pair_bonus((a, b, c), high_pairs)

    if flags.get("slope", True):
        detail["slope"] += slope_triplet_bonus(a, b, c, meta)

    numbers = meta.get("numbers", [])
    if flags.get("form_switch", True) and len(numbers) >= 5:
        form_bonus = form_switch_bonus(numbers)
        if a == b or a == c or b == c:
            detail["form_switch"] += form_bonus.get("zu3", 0.0)
        else:
            detail["form_switch"] += form_bonus.get("zu6", 0.0)

    if flags.get("sum_span", True) and len(numbers) >= SUM_INTERVAL_WINDOW:
        sum_interval_info = sum_interval_bonus(numbers)
        detail["sum_interval"] += sum_interval_info["bonus"].get(s, 0.0)

    detail["total"] = sum(detail.values())
    return detail


def build_detail_list(items, score, danma, kill, meta):
    """为推荐号码列表构建带得分拆解的详情"""
    result = []
    for w, num in items:
        a, b, c = map(int, num)
        detail = triplet_weight_detail(a, b, c, score, danma, kill, meta)
        result.append({
            "num": num,
            "score": round(w, 1),
            "detail": {
                "base_digit": round(detail["base_digit"], 1),
                "danma": round(detail["danma"], 1),
                "kill": round(detail["kill"], 1),
                "sum_span": round(detail["sum_span"], 1),
                "pattern": round(detail["pattern"], 1),
                "last_repeat": round(detail["last_repeat"], 1),
                "ratio_match": round(detail["ratio_match"], 1),
                "pair": round(detail["pair"], 1),
                "slope": round(detail["slope"], 1),
                "form_switch": round(detail["form_switch"], 1),
                "sum_interval": round(detail["sum_interval"], 1),
            }
        })
    return result


def select_danma(score_rank, enable_random=True):
    """动态选择胆码
    
    参数：
        score_rank: 按评分排序的数字列表 [(数字, 分数), ...]
        enable_random: 是否启用随机选择
    
    返回：
        胆码列表（2 个数字）
    """
    top6_digits = [digit for digit, score in score_rank[:DANMA_TOP_POOL]]
    
    if enable_random and random.random() < DANMA_RANDOM_RATE:
        # 30%概率：从 Top6 中随机选 2 个
        return random.sample(top6_digits, 2)
    else:
        # 70%概率：选择前 2 个
        return top6_digits[:2]


def _position_constrained_pool(score, danma, kill, meta, per_pos=ZHXUAN_POS_TOPK):
    """百/十/个分位 Top 码笛卡尔积，用于精炼 Top3/Top5。"""
    pos_scores = meta.get("pos_scores")
    if not pos_scores:
        return []
    tops = [
        sorted(range(10), key=lambda d: -pos_scores[i][d])[:per_pos]
        for i in range(3)
    ]
    pool = []
    for a, b, c in product(*tops):
        w = triplet_weight(a, b, c, score, danma, kill, meta)
        pool.append((w, f"{a}{b}{c}"))
    pool.sort(key=lambda x: -x[0])
    return pool


def _merge_rank_pools(*pools, top_n):
    seen = set()
    merged = []
    for pool in pools:
        for item in pool:
            if item[1] not in seen:
                seen.add(item[1])
                merged.append(item)
    merged.sort(key=lambda x: -x[0])
    return merged[:top_n]


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
    
    # 先排序
    pool.sort(key=lambda x: -x[0])
    
    # Top50 随机扰动：避免同分号长期霸榜
    if apply_noise:
        top50 = pool[:50]
        rest = pool[50:]
        top50 = [
            (w + random.uniform(-RANDOM_NOISE, RANDOM_NOISE), num)
            for w, num in top50
        ]
        pool = sorted(top50 + rest, key=lambda x: -x[0])
    
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
            
            # 冷热平衡先保留较大的候选池，不直接砍到 top_n
            balance_keep = max(top_n * 4, 100)
            
            # 计算各类别需要的号码数量
            hot_needed = max(1, int(balance_keep * HOT_RATIO))
            warm_needed = max(1, int(balance_keep * WARM_RATIO))
            cold_needed = max(1, int(balance_keep * COLD_RATIO))
            
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
            if len(balanced_pool) < balance_keep:
                remaining = [item for item in pool if item not in balanced_pool]
                balanced_pool.extend(remaining[:balance_keep - len(balanced_pool)])
            
            pool = balanced_pool[:balance_keep]
    
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
    
    # 正常模式：贪心选池或纯排序
    if enable_diversity or enable_correlation:
        result = select_diverse_pool(
            pool,
            top_n=top_n,
            candidate_size=max(top_n * 5, SERVED_POOL_CANDIDATE_SIZE),
            use_diversity=enable_diversity,
            use_correlation=enable_correlation,
        )
    else:
        result = pool[:top_n]

    # Top3/Top5：合并分位候选池，避免全量排序漏掉「分位热号组合」
    if (
        top_n <= 5
        and meta.get("pos_scores")
        and not enable_exploration
        and not enable_diversity
        and not enable_correlation
    ):
        pos_pool = _position_constrained_pool(score, danma, kill, meta)
        if pos_pool:
            result = _merge_rank_pools(pos_pool, pool, top_n=top_n)

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
    meta["high_pairs"] = high_freq_pairs(numbers) if len(numbers) >= 50 else set()
    meta["pos_scores"] = [
        ensemble_position_digit_scores(
            numbers, pos, window_weights, dynamic=meta.get("dynamic")
        )
        for pos in range(3)
    ]
    meta["slope"] = analyze_slope_patterns(numbers)
    
    # 和值趋势模型：仅在开启调整时才融合，否则保留多窗口中心
    base_sum_center = meta["sum_center"]
    adjusted_sum_center, trend_direction = sum_trend_model(numbers, SUM_TREND_WINDOW)
    if SUM_TREND_ADJUST != 0:
        meta["sum_center"] = (
            base_sum_center * 0.85
            + adjusted_sum_center * 0.15
        )
    else:
        meta["sum_center"] = base_sum_center
    meta["sum_trend"] = trend_direction
    
    return meta


def evaluate_strategy_admission(
    served_last100_rate,
    raw_last100_rate,
    actual_rank_avg,
    random_baseline=None,
    significance=None,
):
    """策略准入检查：仅当多项指标同时达标才建议进入实盘融合
    
    参数：
        random_baseline: 随机基准命中率（可选，默认使用理论基准 3%）
    """
    # 使用固定理论基准 3%（30/1000），避免单次随机抽样波动
    baseline_rate = random_baseline if random_baseline is not None else 0.03
    
    checks = {
        "served_top30_last100_above_baseline": {
            "passed": served_last100_rate >= baseline_rate,
            "actual": round(served_last100_rate, 4),
            "required": round(baseline_rate, 4),
            "reason": f"近100期 served Top30 不低于理论基准({baseline_rate*100:.1f}%)",
        },
        "raw_top30_last100_above_baseline": {
            "passed": raw_last100_rate >= baseline_rate,
            "actual": round(raw_last100_rate, 4),
            "required": round(baseline_rate, 4),
            "reason": f"近100期 raw Top30 不低于理论基准({baseline_rate*100:.1f}%)",
        },
        "avg_rank_below_500": {
            "passed": actual_rank_avg < 500,
            "actual": actual_rank_avg,
            "required": 500,
            "reason": "平均真实号码排名 < 500",
        },
    }
    if significance is not None:
        checks["permutation_significant"] = {
            "passed": significance.get("pvalue", 1.0) < 0.10,
            "actual": significance.get("pvalue"),
            "required": 0.10,
            "reason": "置换检验 p 值 < 0.10",
        }

    eligible = all(item["passed"] for item in checks.values())
    return {"eligible": eligible, "checks": checks}


def backtest(numbers, trials=BACKTEST_TRIALS, window_weights=None):
    """
    增强版滚动回测（稳定基础版）
    
    核心指标：
        - Top3 命中率
        - Top30 命中率
        - Top100 覆盖率
        - 平均真实号码排名
        - 真实号码中位排名
        - Top30 至少命中两个数字比例
    
    使用滚动窗口训练，与实盘逻辑保持一致。
    窗口权重每 10 期更新一次，更接近实际线上运行逻辑。
    """
    max_w = max(RECENT_WINDOWS)
    if len(numbers) < trials + max_w + 5:
        trials = max(20, len(numbers) - max_w - 5)

    hit_top = hit_top3 = hit_top100 = hit_ge2 = 0
    hit_raw_top30 = hit_served_top30 = 0
    raw_top30_hits = []
    served_top30_hits = []
    actual_ranks = []
    zu6_four_hit = zu6_pool_hit = zu6_draws = zu6_ge2_hit = 0
    random_zu6_four_hit = random_zu6_pool_hit = 0
    rng_zu6 = random.Random(42)
    start = len(numbers) - trials
    
    # 如果传入了固定窗口权重，使用它（用于参数搜索）
    # 否则动态计算（用于正常滚动回测）
    ww = dict(window_weights) if window_weights else None

    rank_kw_pure = dict(
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=False,
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=None,
    )
    rank_kw_served = dict(
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=FEATURE_FLAGS.get("cold_hot_balance", False),
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=None,
    )

    for i in range(start, len(numbers)):
        # 使用滚动窗口：每次只用当前可用的数据
        train = numbers[:i]
        actual = numbers[i]
        
        # 每 10 期更新一次窗口权重（模拟实盘"回填后刷新权重"逻辑）
        # 只有当 window_weights 为 None 时才动态更新（正常滚动回测）
        # 如果传入了固定窗口权重（参数搜索），则保持不变
        if window_weights is None and (ww is None or (i - start) % 10 == 0):
            ww, _ = compute_window_weights(
                train,
                trials=WINDOW_BACKTEST_TRIALS,
                enable_cache=False,
            )
        
        sums = [sum(x) for x in train]
        spans = [calc_span(x) for x in train]
        meta = build_ranking_meta(train, ww, sums, spans, tail_top=4)
        sc, _ = ensemble_digit_scores(train, ww, dynamic=meta.get("dynamic"))
        dan, _, kill, _ = pick_dan_tuo_kill(_blend_dan_score(sc, meta), enable_danma_random=False)
        
        # 纯模型排名（1000 候选）
        all_ranked = rank_triplets(
            sc, dan, kill, meta,
            top_n=1000,
            **rank_kw_pure,
        )
        rank_map = {num: idx + 1 for idx, (_, num) in enumerate(all_ranked)}
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"
        actual_rank = rank_map.get(act_s, 1001)
        actual_ranks.append(actual_rank)

        # 纯模型 Top3
        top3 = rank_triplets(sc, dan, kill, meta, top_n=ZHIXUAN_TOP3, **rank_kw_pure)
        top3_nums = [t[1] for t in top3]

        # raw Top30：纯排序能力
        raw_top30 = rank_triplets(
            sc, dan, kill, meta,
            top_n=RECOMMEND_GROUPS,
            **rank_kw_pure,
        )
        raw_top30_nums = [t[1] for t in raw_top30]

        # served Top30：模拟实盘推荐池（多样性+去相关，近期惩罚关闭）
        served_top30 = rank_triplets(
            sc, dan, kill, meta,
            top_n=RECOMMEND_GROUPS,
            **rank_kw_served,
        )
        served_top30_nums = [t[1] for t in served_top30]

        # Top100（用于覆盖率统计）
        top100_nums = [t[1] for t in all_ranked[:100]]

        raw_hit = act_s in raw_top30_nums
        served_hit = act_s in served_top30_nums
        raw_top30_hits.append(1 if raw_hit else 0)
        served_top30_hits.append(1 if served_hit else 0)
        if raw_hit:
            hit_raw_top30 += 1
        if served_hit:
            hit_served_top30 += 1
        if served_hit:
            hit_top += 1
        if act_s in top3_nums:
            hit_top3 += 1
        if act_s in top100_nums:
            hit_top100 += 1

        if max_digit_overlap(act_s, served_top30_nums) >= 2:
            hit_ge2 += 1

        if classify_form(actual) == "zu6":
            zu6_draws += 1
            actual_set = set(actual)
            z6_sc = zu6_digit_scores(train, ww, dynamic=meta.get("dynamic"))
            z4 = set(pick_zu6_four(z6_sc, numbers=train))
            z5 = set(pick_zu6_pool(z6_sc, pool_size=ZU6_POOL_SIZE, numbers=train))
            if actual_set <= z4:
                zu6_four_hit += 1
            if actual_set <= z5:
                zu6_pool_hit += 1
            if len(actual_set & z4) >= 2:
                zu6_ge2_hit += 1
            if actual_set <= set(rng_zu6.sample(range(10), ZU6_FOUR_SIZE)):
                random_zu6_four_hit += 1
            if actual_set <= set(rng_zu6.sample(range(10), ZU6_POOL_SIZE)):
                random_zu6_pool_hit += 1

    n = trials
    
    # 计算真实号码排名统计
    sorted_ranks = sorted(actual_ranks)
    actual_rank_avg = sum(actual_ranks) / len(actual_ranks) if actual_ranks else 0.0
    actual_rank_median = sorted_ranks[len(sorted_ranks) // 2] if sorted_ranks else 0
    actual_rank_top100_rate = sum(1 for r in actual_ranks if r <= 100) / n if n > 0 else 0.0
    actual_rank_top300_rate = sum(1 for r in actual_ranks if r <= 300) / n if n > 0 else 0.0
    
    # 计算随机基准
    random_result = random_baseline_backtest(numbers, trials=trials, top_n=RECOMMEND_GROUPS)

    last100 = min(100, n)
    raw_last100_rate = sum(raw_top30_hits[-last100:]) / last100 if last100 else 0.0
    served_last100_rate = sum(served_top30_hits[-last100:]) / last100 if last100 else 0.0
    random_baseline = round(random_result["random_rate"], 4)

    return {
        "trials": n,
        "strategy": "stable_baseline",
        # TopK 命中率
        "top3_hit": hit_top3,
        "top3_rate": round(hit_top3 / n, 4) if n > 0 else 0.0,
        "top3_rate_baseline": round(ZHIXUAN_TOP3 / 1000.0, 4),
        # raw / served Top30
        "raw_top30_hit": hit_raw_top30,
        "raw_top30_rate": round(hit_raw_top30 / n, 4) if n > 0 else 0.0,
        "served_top30_hit": hit_served_top30,
        "served_top30_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "raw_top30_last100_rate": round(raw_last100_rate, 4),
        "served_top30_last100_rate": round(served_last100_rate, 4),
        # 主展示指标 = 实盘推荐池
        "top30_hit": hit_served_top30,
        "top30_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "top30_rate_baseline": round(RECOMMEND_GROUPS / 1000.0, 4),
        # 兼容旧字段名
        "top_hit": hit_served_top30,
        "top_rate": round(hit_served_top30 / n, 4) if n > 0 else 0.0,
        "top_rate_baseline": round(RECOMMEND_GROUPS / 1000.0, 4),
        "top100_hit": hit_top100,
        "top100_rate": round(hit_top100 / n, 4) if n > 0 else 0.0,
        # 真实号码排名指标（核心）
        "actual_rank_avg": round(actual_rank_avg, 1),
        "actual_rank_median": actual_rank_median,
        "actual_rank_top100_rate": round(actual_rank_top100_rate, 4),
        "actual_rank_top300_rate": round(actual_rank_top300_rate, 4),
        # 数字命中比例
        "ge2_digit_rate": round(hit_ge2 / n, 4) if n > 0 else 0.0,
        # 组六四码/五码（仅组六开奖期统计）
        "zu6_draws": zu6_draws,
        "zu6_four_hit": zu6_four_hit,
        "zu6_four_rate": round(zu6_four_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_pool_hit": zu6_pool_hit,
        "zu6_pool_rate": round(zu6_pool_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_ge2_hit": zu6_ge2_hit,
        "zu6_ge2_rate": round(zu6_ge2_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_random_four_rate": round(random_zu6_four_hit / zu6_draws, 4) if zu6_draws else 0.0,
        "zu6_random_pool_rate": round(random_zu6_pool_hit / zu6_draws, 4) if zu6_draws else 0.0,
        # 随机基准（仅用于页面展示，准入使用固定理论基准 3%）
        "random_rate": random_baseline,
        "random_hit": random_result["random_hit"],
        "admission": evaluate_strategy_admission(
            served_last100_rate,
            raw_last100_rate,
            actual_rank_avg,
            # 不传 random_baseline，使用默认理论基准 3%
        ),
    }


def random_baseline_backtest(numbers, trials=80, top_n=30, seed=42):
    """随机基准回测：作为模型效果的对照基准
    
    参数：
        numbers: 历史号码数据
        trials: 回测期数
        top_n: 推荐数量
        seed: 随机种子（固定以保证可重复）
    
    返回：
        result: 随机基准回测结果
    """
    rng = random.Random(seed)
    hit = 0

    start = len(numbers) - trials

    for i in range(start, len(numbers)):
        actual = numbers[i]
        act_s = f"{actual[0]}{actual[1]}{actual[2]}"

        pool = [f"{a}{b}{c}" for a in range(10) for b in range(10) for c in range(10)]
        picks = set(rng.sample(pool, top_n))

        if act_s in picks:
            hit += 1

    return {
        "trials": trials,
        "random_hit": hit,
        "random_rate": hit / trials if trials > 0 else 0.0,
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
        perm_rates.append(backtest(seq, trials=trials, window_weights=window_weights)["top30_rate"])
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
    if metric == "top_rate":
        metric = "top30_rate"
    if metric == "composite":
        return (
            0.55 * bt["top3_rate"]
            + 0.30 * bt["top30_rate"]
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
):
    """给定权重在历史数据上跑滚动回测，返回 (目标值, 回测详情)

    参数：
        window_weights: 固定窗口权重（用于参数搜索时公平比较）。
                       设为 None 时，由 backtest() 内部按时间滚动计算，
                       避免训练集内部前视。
    """
    with patch_weights(weights):
        bt = backtest(
            numbers,
            trials=trials,
            window_weights=window_weights,
        )
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
    test_ratio=0.15,  # 预留测试集比例，不参与搜索
):
    """
    随机搜索 + 局部 refine，最大化历史回测命中率。

    参数：
        test_ratio: 预留测试集比例，用于最终验收，不参与参数搜索（防止数据泄漏）
    
    metric: top3_rate | top_rate | ge2_digit_rate | composite
    返回 dict：baseline / best / improvement / history / test_result
    """
    if numbers is None:
        numbers = [x[2] for x in fetch_data()]
    if not numbers:
        return {"error": "未获取到数据"}

    # 时序切分：训练集用于参数搜索，测试集用于最终验收
    train_size = int(len(numbers) * (1 - test_ratio))
    train_numbers = numbers[:train_size]
    test_numbers = numbers[train_size:]
    
    if verbose:
        print(f"数据切分: 训练集 {len(train_numbers)} 期, 测试集 {len(test_numbers)} 期")
        print(f"参数搜索: {iterations} 次随机采样 + {refine_rounds} 次局部 refine")
        print(f"回测期数={backtest_trials}, 目标={metric}")

    rng = random.Random(seed)
    base = default_weights()
    
    # 不预先计算窗口权重，让 backtest() 内部按时间滚动计算
    # 这样训练集前面的预测不会看到训练集后段的开奖结果
    _, baseline_bt = evaluate_weights(
        train_numbers, base, trials=backtest_trials, window_weights=None, metric=metric
    )
    baseline_score = backtest_objective(baseline_bt, metric)
    best_weights = dict(base)
    best_score = baseline_score
    best_bt = baseline_bt
    history = []

    for i in range(iterations):
        candidate = _sample_random_weights(base, rng)
        # 不传固定窗口权重，让回测内部按时间滚动计算
        score, bt = evaluate_weights(
            train_numbers, candidate, trials=backtest_trials, window_weights=None, metric=metric
        )
        history.append({"phase": "random", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [random {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    for i in range(refine_rounds):
        candidate = _mutate_weights(best_weights, base, rng)
        # 不传固定窗口权重，避免训练集内未来信息泄漏
        score, bt = evaluate_weights(
            train_numbers, candidate, trials=backtest_trials, window_weights=None, metric=metric
        )
        history.append({"phase": "refine", "score": score, "weights": candidate})
        if score > best_score:
            best_score, best_weights, best_bt = score, candidate, bt
            if verbose:
                print(f"  [refine {i + 1:3d}] 新最优 {score * 100:.2f}%  top3={bt['top3_rate'] * 100:.1f}%")

    # 在测试集上验收最优参数（测试集从未参与搜索）
    # 注意：传入完整数据（训练集+测试集），但只统计测试段的最后 N 期
    # 这样测试期有真实的历史上下文，与线上逻辑一致
    test_result = None
    test_trials = min(len(test_numbers), backtest_trials)
    if test_trials >= 20:  # 至少20期才有统计意义
        _, test_result = evaluate_weights(
            numbers, best_weights, trials=test_trials, window_weights=None, metric=metric
        )
        if verbose:
            print(f"\n测试集验收（测试段 {test_trials} 期，使用完整历史上下文）:")
            print(f"  Top3 命中率: {test_result['top3_rate'] * 100:.2f}%")
            print(f"  Top30 命中率: {test_result['top30_rate'] * 100:.2f}%")
            print(f"  平均排名: {test_result['actual_rank_avg']}")

    return {
        "metric": metric,
        "backtest_trials": backtest_trials,
        "train_size": len(train_numbers),
        "test_size": len(test_numbers),
        "baseline": {"weights": base, "score": baseline_score, "backtest": baseline_bt},
        "best": {"weights": best_weights, "score": best_score, "backtest": best_bt},
        "improvement": best_score - baseline_score,
        "history_len": len(history),
        "test_result": test_result,
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
            f"| Top{RECOMMEND_GROUPS} {bt['top30_rate'] * 100:.1f}%  "
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


def assess_data_quality(data):
    """Return a compact quality summary for the history used by the 3D models."""
    periods = [str(x[0]) for x in data if x]
    dates = [str(x[1]) for x in data if len(x) > 1]
    duplicate_periods = len(periods) - len(set(periods))
    numeric_gaps = 0

    for prev, curr in zip(periods, periods[1:]):
        try:
            prev_year, prev_seq = int(prev[:4]), int(prev[4:])
            curr_year, curr_seq = int(curr[:4]), int(curr[4:])
        except Exception:
            continue
        if curr_year == prev_year and curr_seq - prev_seq != 1:
            numeric_gaps += 1
        elif curr_year == prev_year + 1 and curr_seq != 1:
            numeric_gaps += 1

    warnings = []
    if len(periods) < MIN_DATA_PERIODS_FOR_ML_FUSION:
        warnings.append("history_too_short_for_ml_fusion")
    if duplicate_periods:
        warnings.append("duplicate_periods")
    if numeric_gaps:
        warnings.append("period_gaps")

    return {
        "periods": len(periods),
        "first_period": periods[0] if periods else None,
        "last_period": periods[-1] if periods else None,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "duplicate_periods": duplicate_periods,
        "period_gaps": numeric_gaps,
        "ml_fusion_allowed": (
            len(periods) >= MIN_DATA_PERIODS_FOR_ML_FUSION
            and duplicate_periods == 0
            and numeric_gaps == 0
        ),
        "warnings": warnings,
    }


def is_ml_prediction_cache_valid(cache, current_period):
    """Guard against stale ML predictions being reused after data/model changes."""
    if not cache or cache.get("base_period") != current_period:
        return False
    if cache.get("model_version") != ML_MODEL_VERSION:
        return False

    created_at = cache.get("created_at")
    if created_at:
        try:
            created_ts = time.mktime(time.strptime(created_at, "%Y-%m-%d %H:%M:%S"))
            if time.time() - created_ts > ML_CACHE_MAX_AGE_SECONDS:
                return False
        except Exception:
            return False
    return True


def run_prediction(data=None, force_refresh=False, enable_backtest=False, enable_permutation=False, compute_weights=False, use_prediction_cache=False):
    """运行预测，返回 JSON 可序列化 dict；data 为 None 时自动抓取。
    
    Args:
        data: 可选的数据列表，如果为 None 则自动抓取
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
        enable_backtest: 是否启用回测（默认 False，大幅提升速度）
        enable_permutation: 是否启用排列测试（默认 False，仅在 enable_backtest=True 时生效）
        compute_weights: 是否重新计算窗口权重（默认 False，使用缓存或默认权重，提升速度）
        use_prediction_cache: 是否使用预测结果缓存（默认 False，避免页面整天显示相同结果）
    """
    global _prediction_cache, _cache_time
    
    # 检查缓存（按自然天判断）
    if use_prediction_cache and not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            log.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            log.info("缓存已过期（非今日数据），重新抓取")
    
    try:
        if data is None:
            data = fetch_data(force_refresh=force_refresh)
    except Exception:
        log.error('3D 数据抓取失败', exc_info=True)
        return {'error': '数据抓取失败'}
    if not data:
        return {"error": "未获取到数据"}

    data_quality = assess_data_quality(data)
    periods = [x[0] for x in data]
    numbers = [x[2] for x in data]
    settle_pending_online_predictions(periods, numbers)
    sums = [sum(x) for x in numbers]
    spans = [calc_span(x) for x in numbers]

    # 窗口权重：优先读取持久化结果，compute_weights=True 时强制重算
    window_weights, window_scores = resolve_window_weights(
        numbers,
        compute_weights=compute_weights,
        period=periods[-1] if periods else None,
    )
    
    meta_raw = ensemble_sum_span(sums, spans, window_weights)
    meta = build_ranking_meta(numbers, window_weights, sums, spans, tail_top=5)
    pat = {k: meta[k] for k in ("consec_rate", "oe_freq", "bs_freq", "oe_total", "bs_total")}

    score, freq_all = ensemble_digit_scores(numbers, window_weights, dynamic=meta.get("dynamic"))
    danma, tuoma, kill, rank = pick_dan_tuo_kill(
        _blend_dan_score(score, meta), enable_danma_random=False
    )
    form_prob = analyze_form_probability(numbers, window_weights=window_weights)
    zu6_score = zu6_digit_scores(numbers, window_weights, dynamic=meta.get("dynamic"))
    if ZU6_RECENT_PENALTY > 0:
        current_period_zu6 = periods[-1] if periods else None
        recent_zu6 = [
            e for e in load_recent_zu6_four()
            if not (isinstance(e, dict) and e.get("period") == current_period_zu6)
        ]
        zu6_score = recent_zu6_digit_penalty(zu6_score, recent_zu6)
    zu6_four = pick_zu6_four(zu6_score, numbers=numbers)
    _, z6_straight = zu6_notes_from_digits(zu6_four)
    save_recent_zu6_four(periods[-1] if periods else None, zu6_four)
    
    # 加载最近推荐历史（用于排除重复推荐）
    recent_recommendations = load_recent_3d_recommendations()
    current_period = periods[-1] if periods else None
    # 仅对「之前期」的推荐做去重惩罚，排除当前期自身——否则同一天多次调用(本期推荐已被保存)
    # 会自我惩罚导致结果漂移，破坏当日稳定性。
    prior_recommendations = [
        e for e in recent_recommendations
        if not (isinstance(e, dict) and e.get("period") == current_period)
    ]
    
    # 实盘版本：关闭随机探索和随机噪声，确保结果稳定
    # Top3：纯模型排序，不应用冷热平衡、多样性和去相关
    zhixuan_top3 = rank_triplets(
        score, 
        danma, 
        kill, 
        meta, 
        top_n=3, 
        enable_exploration=False, 
        apply_noise=False,
        enable_cold_hot_balance=False,
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=None
    )
    
    # Top30：服务模型评分最高的 30 注（纯排序），并施加「近窗去重惩罚」使日间轮换。
    # 3D 为独立均匀摇奖，任意 30 注互异组合命中率恒为 30/1000=3%——多样性/去相关重排会
    # 用真实命中换无奖金价值的"2 码重合"(实测把 served 从 3.4% 拉到 2.2%)，故仍关闭。
    # 但「近窗去重」只是在等价的 30 注之间轮换，不损失命中期望(实测900期落在3%±1.1%噪声带内)，
    # 却把日间重复度从 ~54% 降到 ~14%，解决"每天推荐高度雷同"的问题。
    zhixuan_top = rank_triplets(
        score,
        danma,
        kill,
        meta,
        top_n=RECOMMEND_GROUPS,
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=FEATURE_FLAGS.get("cold_hot_balance", False),
        enable_diversity=False,
        enable_correlation=False,
        recent_recommendations=prior_recommendations,
    )
    
    zhixuan_top3_detail = build_detail_list(
        zhixuan_top3, score, danma, kill, meta
    )
    rule_top3_detail = zhixuan_top3_detail.copy()
    zhixuan_with_detail = build_detail_list(
        zhixuan_top, score, danma, kill, meta
    )
    
    # 保存融合前的规则模型推荐（用于策略推荐展示）
    rule_only_detail = zhixuan_with_detail.copy()
    
    # 先初始化回测结果（放在ML逻辑之前，避免提前引用）
    bt = None
    
    # 获取ML预测结果（带缓存，避免每次重新训练）
    ml_result = None
    ml_list = []
    try:
        from .ml import predict_current, load_ml_cache, save_ml_cache, ML_CACHE_KEY
        # 尝试加载缓存
        ml_cache = load_ml_cache()
        current_period = periods[-1] if periods else None
        
        # 检查缓存是否有效
        cache_valid = is_ml_prediction_cache_valid(ml_cache, current_period)
        
        if cache_valid and not force_refresh:
            ml_result = ml_cache
            ml_list = ml_cache.get("recommendations", [])
            log.info(f"使用ML缓存（期号: {current_period}）")
        else:
            # 需要重新训练
            ml_result = predict_current(numbers, top_k=100)
            ml_list = ml_result.get("recommendations", []) if not ml_result.get("error") else []
            
            # 保存缓存
            if not ml_result.get("error") and current_period:
                ml_result["base_period"] = current_period
                ml_result["model_version"] = ML_MODEL_VERSION
                ml_result["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_ml_cache(ml_result)
            
            log.info(f"ML预测完成，推荐{len(ml_list)}注")
    except Exception as e:
        log.warning(f"ML预测失败: {e}")
    
    # 获取缓存的回测表现（不依赖本次是否开启回测）
    rule_perf = load_recent_rule_performance()
    rule_top30_rate = rule_perf.get("top30_rate", 0.03)
    
    baseline_rate = RECOMMEND_GROUPS / 1000.0
    rule_lift = rule_top30_rate - baseline_rate
    
    # ML准入：只基于已保存的滚动回测结果，不使用当前预测去匹配历史数据（避免数据泄漏）
    ml_eligible = data_quality.get("ml_fusion_allowed", False) and is_ml_eligible_from_backtest(current_period)
    ml_weight = 0.0
    rule_weight = 1.0
    
    # 如果ML符合准入条件且有正Lift，计算动态权重
    # 从已保存的回测历史读取ML表现（与准入判断使用同一份数据）
    if ml_eligible:
        ml_perf = load_latest_ml_performance()
        ml_top30_rate = ml_perf.get("top30_rate", 0.0)
        ml_lift = ml_top30_rate - baseline_rate
        
        if ml_lift > 0 and ml_top30_rate > rule_top30_rate:
            total_lift = max(rule_lift, 0) + ml_lift
            rule_weight = max(max(rule_lift, 0) / total_lift, 0.55)
            ml_weight = ml_lift / total_lift
        else:
            ml_eligible = False
            ml_weight = 0.0
            rule_weight = 1.0
    
    # 融合规则模型和ML模型
    # 当ML不准入、权重为0或推荐列表为空时，直接使用规则模型，避免无意义的重排
    ml_status = "eligible"
    ml_error = None
    ml_eligible_reason = ""

    if not ml_list:
        ml_status = "no_recommendations"
        ml_eligible_reason = "ML推荐列表为空"
        fused = rule_only_detail
        log.info("ML推荐列表为空，使用纯规则模型推荐")
    elif not data_quality.get("ml_fusion_allowed", False):
        ml_status = "insufficient_history"
        ml_eligible_reason = f"ML fusion requires at least {MIN_DATA_PERIODS_FOR_ML_FUSION} periods"
        fused = rule_only_detail
        log.info("ML fusion skipped because history is too short")
    elif ml_weight <= 0:
        ml_status = "low_weight"
        ml_eligible_reason = "ML权重为0"
        fused = rule_only_detail
        log.info("ML权重为0，使用纯规则模型推荐")
    elif not ml_eligible:
        ml_status = "not_eligible"
        ml_eligible_reason = "ML未通过准入检查"
        fused = rule_only_detail
        log.info("ML未准入，使用纯规则模型推荐")
    else:
        fused = fuse_rule_ml(
            rule_list=zhixuan_with_detail,
            ml_list=ml_list,
            top_n=RECOMMEND_GROUPS,
            rule_weight=rule_weight,
            ml_weight=ml_weight,
            score=score,
            danma=danma,
            kill=kill,
            meta=meta,
        )
        ml_eligible_reason = f"ML准入成功，规则权重={rule_weight:.2f}, ML权重={ml_weight:.2f}"
        log.info(f"ML融合完成，规则权重={rule_weight:.2f}, ML权重={ml_weight:.2f}")
    
    # 保存ML状态信息（等最后构造result时再加入）
    ml_status_info = {
        "status": ml_status,
        "eligible": ml_eligible,
        "weight": round(ml_weight, 4),
        "error": ml_error,
        "reason": ml_eligible_reason,
    }

    # 保存三套策略记录
    save_strategy_records(
        period=periods[-1],
        rule_only=[r["num"] for r in rule_only_detail],
        ml_only=[m["num"] for m in ml_list[:RECOMMEND_GROUPS]],
        fused=[f["num"] for f in fused],
    )
    
    # 使用融合结果作为最终 Top30；Top3 始终保留纯规则模型排序（ML 融合易拉低 Top3）
    zhixuan_with_detail = fused
    zhixuan_top3_detail = rule_top3_detail
    
    # 保存本次推荐历史（按期号去重）
    current_recommendations = [f["num"] for f in fused]
    save_recent_3d_recommendations(periods[-1], current_recommendations)
    
    # 计算推荐稳定度
    stability = recommendation_stability(current_recommendations, recent_recommendations)
    stability_level = get_stability_level(stability)
    adjusted_exploration_rate = adjust_exploration_rate(stability)
    
    # 可选：回测分析（耗时操作）
    bt = None
    if enable_backtest:
        bt = backtest(numbers, window_weights=window_weights)
        
        # 保存规则模型表现到缓存（用于动态融合权重计算）
        try:
            kv_store.save("lottery3d_rule_performance", {
                "base_period": periods[-1],
                "top30_rate": bt.get("top30_rate", 0.0),
                "top3_rate": bt.get("top3_rate", 0.0),
                "top100_rate": bt.get("top100_rate", 0.0),
                "actual_rank_avg": bt.get("actual_rank_avg", 500),
                "actual_rank_median": bt.get("actual_rank_median", 500),
                "trials": bt.get("trials", 0),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log.info("规则模型表现已保存")
        except Exception as e:
            log.error(f"保存规则模型表现失败: {e}")
        
        if enable_permutation:
            sig = permutation_test(
                numbers, bt["raw_top30_rate"], window_weights=window_weights
            )
            bt["significance"] = sig
            bt["admission"] = evaluate_strategy_admission(
                bt["served_top30_last100_rate"],
                bt["raw_top30_last100_rate"],
                bt["actual_rank_avg"],
                # 不传 random_rate，使用默认理论基准 3%
                significance=sig,
            )

    last_num = numbers[-1]
    
    # 保存线上预测记录
    save_online_prediction(
        period=periods[-1],
        last_draw="".join(map(str, last_num)),
        zhixuan_top3=zhixuan_top3_detail,
        zhixuan=zhixuan_with_detail,
        danma=danma,
        kill=kill,
    )
    
    # 计算线上实盘统计
    online_stats = calculate_online_stats()
    
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
        "slope": meta.get("slope") or analyze_slope_patterns(numbers),
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
            "recommendation": recommend_form_bet(form_prob, numbers),
        },
        "zu6_four": {
            "digits_str": "".join(map(str, zu6_four)),
            "combos": z6_straight,
        },
        "zu6_digit_scores": [
            {"digit": d, "score": round(zu6_score[d], 2)}
            for d in sorted(range(10), key=lambda x: -zu6_score[x])
        ],
        "zu6_four_variants": build_zu6_four_variants(zu6_score, kill=None, numbers=numbers),
        "zu6_coverage": build_zu6_coverage_tiers(zu6_score, kill=None, numbers=numbers),
        "zhixuan_top3": zhixuan_top3_detail,
        "zhixuan": zhixuan_with_detail,
        "stability": {
            "score": round(stability, 2),
            "level": stability_level,
            "adjusted_exploration_rate": round(adjusted_exploration_rate, 2),
        },
        "version": PREDICTOR_VERSION,
        "online_stats": online_stats,
        "ml_status": ml_status_info,
        "data_quality": data_quality,
    }
    
    # 添加策略推荐（保守/均衡/探索）
    # 使用融合前的规则列表和ML列表，而非融合后的结果
    result["strategy_recommendations"] = generate_strategy_recommendations(
        rule_only_detail,
        ml_list,
        danma,
        kill,
    )
    
    # 添加策略模式选择
    # 优先使用本次回测结果，否则读取缓存的规则模型表现
    if bt:
        top30_rate = bt["top30_rate"]
        actual_rank_avg = bt.get("actual_rank_avg", 500)
        rank_top100_rate = bt.get("actual_rank_top100_rate", 0.0)
    else:
        # 从缓存读取规则模型表现（不依赖本次是否开启回测）
        rule_perf = load_recent_rule_performance()
        # load_recent_rule_performance 返回的是 top30_rate 值，需要调整
        if isinstance(rule_perf, dict):
            top30_rate = rule_perf.get("top30_rate", 0.03)
            actual_rank_avg = rule_perf.get("actual_rank_avg", 500)
            rank_top100_rate = rule_perf.get("top100_rate", 0.0)
        else:
            top30_rate = rule_perf  # 兼容旧版本返回值
            actual_rank_avg = 500
            rank_top100_rate = 0.0
    
    # 使用固定理论基准 3%（30/1000）
    model_lift = top30_rate - 0.03
    recent_hit_rate = online_stats.get("hit_top30_rate", 0.0)
    
    strategy_mode, strategy_reason = select_strategy_mode(
        stability,
        model_lift,
        recent_hit_rate,
        actual_rank_avg,
    )
    result["strategy_mode"] = {
        "mode": strategy_mode,
        "reason": strategy_reason,
    }
    
    # 添加资金建议
    budget_info = recommend_budget_level(model_lift, stability, recent_hit_rate)
    result["budget_recommendation"] = budget_info
    
    # 添加自动推荐注数
    auto_count, count_reason = auto_recommend_count(model_lift, rank_top100_rate, recent_hit_rate)
    result["auto_recommend_count"] = {
        "count": auto_count,
        "reason": count_reason,
    }
    
    # 添加额外回测统计
    if bt is not None:
        result["backtest"] = bt
        
        # 添加胆码/杀码回测
        result["backtest"]["dan_kill"] = backtest_dan_kill(numbers, trials=min(100, len(numbers) - 50))
        
        # 添加形态预测回测
        result["backtest"]["form_prediction"] = backtest_form_prediction(numbers, trials=min(100, len(numbers) - 50))
        
        # 添加和值/跨度区间回测
        result["backtest"]["sum_span_interval"] = backtest_sum_span_interval(numbers, trials=min(100, len(numbers) - 50))
        result["backtest"]["slope_patterns"] = backtest_slope_patterns(numbers, trials=min(200, len(numbers) - 50))
    
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

    tiers = result.get("zu6_coverage")
    if tiers:
        print("\n  组六复式覆盖档位（选号无 edge，按预算选覆盖）:")
        print("    码数  注数  成本   命中率   复式码")
        for t in tiers:
            print(f"    {t['size']:>2d}码  {t['notes']:>3d}注  {t['cost']:>3d}元  "
                  f"{t['hit_rate']*100:>5.1f}%   {t['digits_str']}")
        print("    注：纯组六复式命中率上限 72.8%（组三/豹子开奖无法覆盖）")

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

    bt = result.get("backtest")
    if bt:
        print("\n" + "=" * 70)
        print("滚动回测（稳定基础版）")
        print("=" * 70)
        print(f"  回测期数: {bt['trials']}")
        print(f"  Top3 命中: {bt['top3_rate'] * 100:.1f}% "
              f"({bt['top3_hit']}/{bt['trials']})")
        print(f"  Top30 命中（served）: {bt['served_top30_rate'] * 100:.1f}% "
              f"({bt['served_top30_hit']}/{bt['trials']})")
        print(f"  Top30 命中（raw）: {bt['raw_top30_rate'] * 100:.1f}% "
              f"({bt['raw_top30_hit']}/{bt['trials']})")
        print(f"  Top100 覆盖: {bt['top100_rate'] * 100:.1f}% "
              f"({bt['top100_hit']}/{bt['trials']})")
        print(f"  平均真实号码排名: {bt['actual_rank_avg']}")
        print(f"  中位真实号码排名: {bt['actual_rank_median']}")
        print(f"  Top30 至少一注重合2码: {bt['ge2_digit_rate'] * 100:.1f}%")
        print(f"  随机 Top30 基准: {bt['random_rate'] * 100:.1f}%")

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
