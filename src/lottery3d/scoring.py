# -*- coding: utf-8 -*-
"""福彩3D评分与选号：窗口权重、数字评分、组三/组六、直选排名"""

import math
import time
from collections import Counter, defaultdict
from itertools import combinations

from ..common.logger import setup_logger
from ..common import kv_store

log = setup_logger('lottery3d')

from .config import (
    COLD_RATIO, CORRELATION_PENALTY, CORRELATION_THRESHOLD, DANMA_RANDOM_RATE, DANMA_TOP_POOL, DIVERSITY_WEIGHT, EXPLORATION_RATE, EXP_DECAY, FEATURE_FLAGS, HOT_RATIO, HOT_WINDOW, MARKOV_MAX_SCORE, RANDOM_DIGIT_REUSE, RANDOM_NOISE, RANDOM_POS_REPEAT, RECENT_SUM_SPAN_SHIFT, RECENT_WINDOW, RECENT_WINDOWS, SERVED_POOL_CANDIDATE_SIZE, SPAN_SOFT_SIGMA, SUM_INTERVAL_WINDOW, SUM_SOFT_SIGMA, SUM_TREND_ADJUST, SUM_TREND_WINDOW, WARM_RATIO, WINDOW_BACKTEST_TRIALS, WINDOW_WEIGHTS_KV_KEY, WINDOW_WEIGHT_PRIOR, W_CONSECUTIVE, W_DANMA_HIT, W_FORM_PRIOR, W_HOT_GLOBAL, W_HOT_POS, W_KILL_PENALTY, W_LAST_APPEAR, W_MARKOV, W_MARKOV2, W_MISS_HIGH, W_MISS_MID, W_NEIGHBOR, W_POS_REPEAT, W_RATIO_MATCH, W_ROAD_MATCH, W_TRIPLET_GLOBAL, W_TRIPLET_POS, W_ZU6_PAIR, ZHIXUAN_TOP3, ZHXUAN_POS_TOPK, ZU3_MIN_SAMPLES, ZU3_PAIRS_COUNT, ZU3_PRESENCE_WINDOWS, ZU3_TIER_SIZES, ZU6_FOUR_SIZE, ZU6_POOL_SIZE, ZU6_PRESENCE_WINDOWS, ZU6_USE_KILL,
    MARKOV_LAPLACE_ALPHA, PAIR_BONUS, W_SLOPE_MATCH,
)
from .features import (
    FORM_LABELS, THEORY_FORM_P, _form_recent_p, analyze_slope_patterns, calc_span, classify_digits_by_hot, classify_form, entropy_model, form_miss, form_switch_bonus, high_freq_pairs, markov_prob_smoothed, max_digit_overlap, miss_cycle_bonus, neighbor, pair_bonus, rebound_model, recent_recommend_penalty, sum_interval_bonus, sum_trend_model,
)

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
    """动态形态主推：本期更可能是组六还是组三。

    v4.9 变更：原实现固定主推「组六」（注释称短期信号无法击败 base rate）。
    现改为动态——主推 = blend 融合概率最大者，并给出组三概率相对其基准(27%)的
    抬升/回落信号。诚实说明：组六基准概率 72% 远高于组三 27%，500期 walk-forward
    实测动态 max 选组六占 100%，即"动态判断"在大多数时候仍指向组六；其真正价值
    在于量化展示组三概率何时抬升（如连续组六后 markov 信号偏组三），供加注参考，
    而非声称能预测形态（形态无短期可预测性，追涨杀跌是赌徒谬误）。
    """
    blend = form_prob["blend_p"]
    primary = max(blend, key=blend.get)
    secondary = "zu3" if primary != "zu3" else "zu6"
    zu3_elevation = blend["zu3"] - THEORY_FORM_P["zu3"]
    forms = [classify_form(n) for n in numbers]
    n = len(forms) or 1
    hist_cnt = Counter(forms)
    emp = {k: hist_cnt.get(k, 0) / n for k in THEORY_FORM_P}
    return {
        "primary": primary,
        "primary_label": FORM_LABELS[primary],
        "primary_prob": round(blend[primary], 4),
        "secondary": secondary,
        "zu6_prob": round(blend["zu6"], 4),
        "zu3_prob": round(blend["zu3"], 4),
        "zu3_base_rate": THEORY_FORM_P["zu3"],
        "zu3_elevation": round(zu3_elevation, 4),
        "zu3_signal": (
            "elevated" if zu3_elevation > 0.03
            else "depressed" if zu3_elevation < -0.03
            else "normal"
        ),
        "expected_hit_rate": round(emp[primary], 4),  # 主推形态的历史 base rate
        "theory_hit_rate": THEORY_FORM_P[primary],
        "empirical_form_p": {k: round(v, 4) for k, v in emp.items()},
        "blend_p": {k: round(v, 4) for k, v in blend.items()},
        "note": (
            "主推=blend概率最大形态（组六以72%基准概率占绝对优势，500期实测动态选组六100%）；"
            "zu3_elevation>0 表示组三概率高于其27%基准，可作为加注组三的参考，"
            "但形态本身无短期可预测性，概率波动属噪声。"
        ),
    }


def zu3_digit_presence(numbers, window=None):
    """组三条件下的数字出现率：最近组三开奖去重后各数字出现的比例（每期只计一次）。

    与组六 presence 模型（ZU6_PRESENCE_WINDOWS）同思路：只预测"数字是否进入
    组三开奖号集合"，不预测位置/重复位。组三样本不足时自动扩大到 60 期；
    仍无组三数据则返回均匀 0.2（无信息先验）。
    """
    window = window if window is not None else ZU3_PRESENCE_WINDOWS[0]
    zu3 = [set(n) for n in numbers[-window:] if classify_form(n) == "zu3"]
    if len(zu3) < ZU3_MIN_SAMPLES:
        zu3 = [set(n) for n in numbers[-60:] if classify_form(n) == "zu3"]
    if not zu3:
        return {d: 0.2 for d in range(10)}
    cnt = Counter()
    for s in zu3:
        cnt.update(s)
    total = len(zu3)
    return {d: cnt.get(d, 0) / total for d in range(10)}


def zu3_pair_scores(presence):
    """45 个无序数对的组三条件概率（独立性假设）：P({a,b}|zu3) ∝ r_a·r_b，归一化。"""
    scored = []
    total = 0.0
    for a in range(10):
        for b in range(a + 1, 10):
            s = presence[a] * presence[b]
            scored.append(((a, b), s))
            total += s
    total = total or 1.0
    return [(pair, s / total) for pair, s in scored]


def zu3_combos_from_pair(pair):
    """组选三对子 {a,b} 覆盖的全部 6 注单选（aab/aba/baa/abb/bab/bba）。"""
    a, b = sorted(pair)
    combos = set()
    for rep, single in ((a, b), (b, a)):
        for p in {0, 1, 2}:
            slot = [rep] * 3
            slot[p] = single
            combos.add("".join(map(str, slot)))
    return sorted(combos)


def zu3_zu_notes_from_pair(pair):
    """对子 {a,b} 的组选三表达：2 注覆盖全部 6 种排列（4 元），与 6 注单选（12 元）等价。

    福彩3D 规则：组选3 一注 = 3 码含一重复位 → 3 种排列（如 225 → 225/252/522）。
    对子 {2,5} 有双号 2（225）与双号 5（552）两个方向，共 6 种排列 → 2 注组选三即可。
    命中概率与 6 注单选完全相同（EV 相同），成本仅 1/3。
    """
    a, b = sorted(pair)
    return sorted({f"{rep}{rep}{single}" for rep, single in ((a, b), (b, a))})


def pick_zu3_pairs(numbers, limit=ZU3_PAIRS_COUNT, presence=None):
    """组三推荐：取组三条件概率最高的 4 个对子（四组）。

    每组 = 一个组选三对子 {a,b}：组选三 2 注（4 元）覆盖 6 种排列（v4.10 高效口径，
    原 6 注单选 = 12 元仅作对比保留）。任取 K 组（不要求互异），给定开奖为组三的
    条件命中率 = K/C(10,2) = K/45 —— 与选哪些码无关，顶部对子的概率差异
    （0.17~0.24 的数字率）只带来 1% 量级的微小偏移，属噪声。
    """
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    scored = zu3_pair_scores(presence)
    scored.sort(key=lambda x: -x[1])
    top = scored[:limit]
    pairs = []
    for (a, b), pr in top:
        combos = zu3_combos_from_pair((a, b))
        zu_notes = zu3_zu_notes_from_pair((a, b))
        pairs.append({
            "digits": [a, b],
            "digits_str": f"{a}{b}",
            "prob": round(pr, 4),
            "notes": len(zu_notes),               # 组选三注数 = 2
            "cost": len(zu_notes) * TICKET_PRICE,  # 组选三成本 = 4 元
            "zu_notes": zu_notes,                 # 2 注组选三（高效口径，主推）
            "combos": combos,                     # 6 注单选（直选口径，对比）
            "direct_notes": len(combos),
            "direct_cost": len(combos) * TICKET_PRICE,
        })
    cond_hit = sum(pr for _, pr in top)
    return {
        "method": "zu3_conditional_presence",
        "window": ZU3_PRESENCE_WINDOWS[0],
        "presence": {d: round(v, 4) for d, v in presence.items()},
        "pairs": pairs,
        # 模型内样本估计：top4 对子概率和（presence 噪声被取顶放大，系过拟合，
        # 500期回测实测 ≈ 随机基准，勿当作真实命中率）
        "conditional_hit_rate": round(cond_hit, 4),
        # 数学精确基准：任取 K 组对子条件命中 = K/45（与选哪些码无关），回测实测 ≈ 此值
        "random_conditional_hit_rate": round(limit / 45.0, 4),
        "notes_total": sum(p["notes"] for p in pairs),          # 组选三 8 注
        "total_cost": sum(p["cost"] for p in pairs),            # 组选三 16 元（v4.10 主口径）
        "direct_notes_total": sum(p["direct_notes"] for p in pairs),  # 单选 24 注
        "direct_total_cost": sum(p["direct_cost"] for p in pairs),    # 单选 48 元（v4.9 口径）
        "note": (
            f"组选三：每组={pairs[0]['digits_str'] if pairs else ''}式对子 = 2 注组选三/4 元"
            f"（覆盖 6 种排列，与 6 注单选 12 元等价，EV 相同）；"
            f"任取{limit}组条件命中率={limit}/45≈{limit/45:.1%}（与选哪些码基本无关，"
            "数字出现率0.17~0.24差异为噪声；conditional_hit_rate 为模型内样本估计，"
            "500期回测实测≈随机基准，属过拟合）。"
        ),
    }


def zu3_coverage_tiers(numbers, sizes=ZU3_TIER_SIZES, presence=None):
    """组三覆盖档位：K 组对子 → 组选三 2K 注/4K 元，条件命中率 K/45（线性）。

    与组六 build_zu6_coverage_tiers 对称：K 组对子 = 排序后 top-K 前缀（复用同一
    presence/评分），任取 K 组条件命中 = K/C(10,2)，与选哪些码无关（回测实测 ≈ K/45）。
    直选口径（6K 注/12K 元）一并给出作对比：同样的 K 组覆盖，组选三成本仅 1/3。
    """
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    scored = zu3_pair_scores(presence)
    scored.sort(key=lambda x: -x[1])
    tiers = []
    for k in sizes:
        k = min(k, 45)
        top = scored[:k]
        pairs = [list(p) for p, _ in top]
        tiers.append({
            "size": k,
            "pairs": pairs,
            "pairs_str": " ".join(f"{a}{b}" for a, b in top),
            "notes": k * 2,                # 组选三注数
            "cost": k * 4,                 # 组选三成本（元）
            "conditional_hit_rate": round(k / 45.0, 4),
            "direct_notes": k * 6,         # 直选注数（对比）
            "direct_cost": k * 12,         # 直选成本（对比）
        })
    return tiers


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
            "hit_rate": round(notes * 6 / 1000.0, 4),  # 无条件命中率（含"开奖须为组六"）
            "conditional_hit_rate": round(notes / 120.0, 4),  # 给定开奖为组六时 = notes/C(10,3)
            "is_primary": n == ZU6_POOL_SIZE,
            "combos": combo_strs,
        })
    return tiers


def build_zu6_primary(score, kill=None, numbers=None, size=ZU6_POOL_SIZE):
    """组六主推池：默认 6 码 → C(6,3)=20 注组六。

    与 build_zu6_coverage_tiers 中同尺寸档位取号一致（同一 pick_zu6_pool），
    供前端 zu6_primary 直接渲染，避免退化回四码却仍标注五码。
    """
    digits = pick_zu6_pool(score, kill, pool_size=size, numbers=numbers)
    combos, combo_strs = zu6_notes_from_digits(digits)
    notes = len(combos)
    return {
        "size": size,
        "digits": digits,
        "digits_str": "".join(map(str, digits)),
        "notes": notes,
        "cost": notes * TICKET_PRICE,
        "hit_rate": round(notes * 6 / 1000.0, 4),
        "conditional_hit_rate": round(notes / 120.0, 4),
        "is_primary": True,
        "combos": combo_strs,
    }


def evaluate_zu6_pool_recent(numbers, sizes=(5, 6), trials=100):
    """最近 N 期逐期样本外检验号码池，专门衡量“中几个数字”。

    每一期只使用它之前的数据选码，避免把当期开奖泄漏进评分。完整命中只在
    组六期统计；ge2_rate 则回答用户最直观的“至少覆盖两个不同开奖号”频率。
    """
    sizes = tuple(sorted({int(s) for s in sizes if 3 <= int(s) <= 10}))
    minimum_train = max(ZU6_PRESENCE_WINDOWS) + 5
    if len(numbers) <= minimum_train or not sizes:
        return {"trials": 0, "zu6_draws": 0, "tiers": {}}
    start = max(minimum_train, len(numbers) - max(1, int(trials)))
    stats = {
        size: {"full_hit": 0, "ge2_hit": 0, "overlap_sum": 0}
        for size in sizes
    }
    zu6_draws = 0
    evaluated = 0
    for i in range(start, len(numbers)):
        train = numbers[:i]
        actual = numbers[i]
        actual_set = set(actual)
        is_zu6 = len(actual_set) == 3
        zu6_draws += int(is_zu6)
        evaluated += 1
        scores = zu6_digit_scores(train)
        for size in sizes:
            pool = set(pick_zu6_pool(scores, pool_size=size, numbers=train))
            overlap = len(actual_set & pool)
            stats[size]["overlap_sum"] += overlap
            stats[size]["ge2_hit"] += int(overlap >= 2)
            stats[size]["full_hit"] += int(is_zu6 and actual_set <= pool)

    tiers = {}
    for size, item in stats.items():
        notes = math.comb(size, 3)
        tiers[str(size)] = {
            "size": size,
            "trials": evaluated,
            "zu6_draws": zu6_draws,
            "full_hit": item["full_hit"],
            "conditional_full_rate": round(
                item["full_hit"] / zu6_draws, 4
            ) if zu6_draws else 0.0,
            "unconditional_full_rate": round(
                item["full_hit"] / evaluated, 4
            ) if evaluated else 0.0,
            "ge2_rate": round(item["ge2_hit"] / evaluated, 4) if evaluated else 0.0,
            "avg_unique_overlap": round(
                item["overlap_sum"] / evaluated, 3
            ) if evaluated else 0.0,
            "theoretical_conditional_rate": round(notes / 120.0, 4),
            "theoretical_unconditional_rate": round(notes * 6 / 1000.0, 4),
        }
    return {"trials": evaluated, "zu6_draws": zu6_draws, "tiers": tiers}


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




# ─── 领域层适配 ───
#
# 窗口集成、数字评分、直选排名三段已迁入 `src/domain/numeric/lottery3d/`。
# 下面这一层只做两件事：把配置常量装进 `DigitWeights` / `TripletWeights`，
# 以及保住旧的函数名与签名（`prediction.py` / `backtest.py` / `__init__.py`
# 都按旧名字导入）。

from src.domain.numeric.lottery3d import digit_scoring as _digits
from src.domain.numeric.lottery3d import ranking as _ranking
from src.domain.numeric.lottery3d import weights as _weights
from src.domain.numeric.lottery3d import windows as _windows

_DIGIT_WEIGHTS = _weights.DigitWeights(
    hot_global=W_HOT_GLOBAL, hot_position=W_HOT_POS,
    markov=W_MARKOV, markov2=W_MARKOV2, markov_max=MARKOV_MAX_SCORE,
    markov_alpha=MARKOV_LAPLACE_ALPHA,
    miss_high=W_MISS_HIGH, miss_mid=W_MISS_MID,
    last_appear=W_LAST_APPEAR, neighbor=W_NEIGHBOR, road_match=W_ROAD_MATCH,
    decay=EXP_DECAY,
)

_TRIPLET_WEIGHTS = _weights.TripletWeights(
    danma_hit=W_DANMA_HIT, kill_penalty=W_KILL_PENALTY,
    sum_sigma=SUM_SOFT_SIGMA, span_sigma=SPAN_SOFT_SIGMA,
    consecutive=W_CONSECUTIVE, position_repeat=W_POS_REPEAT,
    ratio_match=W_RATIO_MATCH, slope_match=W_SLOPE_MATCH, pair_bonus=PAIR_BONUS,
    form_prior=W_FORM_PRIOR,
    triplet_position=W_TRIPLET_POS, triplet_global=W_TRIPLET_GLOBAL,
    diversity=DIVERSITY_WEIGHT,
    correlation_penalty=CORRELATION_PENALTY,
    correlation_threshold=CORRELATION_THRESHOLD,
    noise=RANDOM_NOISE, exploration_rate=EXPLORATION_RATE,
    danma_top_pool=DANMA_TOP_POOL, danma_random_rate=DANMA_RANDOM_RATE,
)

_BASELINES = _weights.Baselines(
    position_repeat=RANDOM_POS_REPEAT, digit_reuse=RANDOM_DIGIT_REUSE,
)

# 动态权重缩放的基准。领域层不读全局配置，所以这几个静态值由这里给。
_DYNAMIC_BASE = {
    'position_repeat': W_POS_REPEAT,
    'last_appear': W_LAST_APPEAR,
    'consecutive': W_CONSECUTIVE,
}

_clamp = _windows.clamp
def _empty_lag1():
    return _windows.empty_lag1(_BASELINES)
position_repeat_count = _windows.position_repeat_count


def analyze_lag1_dynamics(numbers, window=RECENT_WINDOW):
    return _windows.analyze_lag1(numbers, window, EXP_DECAY, _BASELINES)


def ensemble_lag1_dynamics(numbers, window_weights):
    return _windows.ensemble_lag1(numbers, window_weights, EXP_DECAY, _BASELINES)


def derive_dynamic_weights(lag1, consec_rate):
    return _windows.derive_dynamic_weights(lag1, consec_rate, _DYNAMIC_BASE, _BASELINES)


def analyze_patterns(numbers, window=RECENT_WINDOW):
    return _windows.analyze_patterns(numbers, window, EXP_DECAY)


def ensemble_patterns(numbers, window_weights):
    return _windows.ensemble_patterns(numbers, window_weights, EXP_DECAY)


def analyze_sum_span(sums, spans, window=RECENT_WINDOW):
    return _windows.analyze_sum_span(sums, spans, window, EXP_DECAY,
                                     RECENT_SUM_SPAN_SHIFT)


def ensemble_sum_span(sums, spans, window_weights):
    return _windows.ensemble_sum_span(sums, spans, window_weights, EXP_DECAY,
                                      RECENT_SUM_SPAN_SHIFT)


def _meta_from_raw(meta_raw, tail_top=5):
    return _windows.with_hot_sets(meta_raw, tail_top)


def digit_scores(numbers, window=RECENT_WINDOW, dynamic=None):
    """三份弱先验（遗漏周期、回补、熵值）在这里算好再喂进去：
    它们各自有窗口与阈值，那是配置问题。"""
    return _digits.digit_scores(
        numbers, window, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic,
        miss_cycle=miss_cycle_bonus(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        rebound=rebound_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        entropy=entropy_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
    )


def ensemble_digit_scores(numbers, window_weights, dynamic=None):
    return _digits.ensemble_digit_scores(
        numbers, window_weights, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic,
        miss_cycle=miss_cycle_bonus(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        rebound=rebound_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
        entropy=entropy_model(numbers) if FEATURE_FLAGS.get('miss', True) else None,
    )


def position_digit_scores(numbers, position, window=RECENT_WINDOW, dynamic=None):
    return _digits.position_digit_scores(numbers, position, window,
                                         _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic)


def ensemble_position_digit_scores(numbers, position, window_weights, dynamic=None):
    return _digits.ensemble_position_digit_scores(
        numbers, position, window_weights, _DIGIT_WEIGHTS, FEATURE_FLAGS, dynamic)


def zu6_digit_scores(numbers, window_weights=None, dynamic=None):
    """`window_weights` 与 `dynamic` 留在签名里只为兼容旧调用方：
    组六选池模型刻意不复用分位直选模型。"""
    return _digits.zu6_digit_scores(numbers, ZU6_PRESENCE_WINDOWS)


def _triplet_context(danma, kill, meta):
    """建一次上下文。`form_switch` 与 `sum_interval` 只依赖历史——迁移前它们
    是在一千注的循环里各算一遍的。"""
    numbers = meta.get('numbers', [])
    return _ranking.build_context(
        meta, _TRIPLET_WEIGHTS, FEATURE_FLAGS, danma, kill,
        form_switch=(form_switch_bonus(numbers)
                     if len(numbers) >= 5 else None),
        sum_interval=(sum_interval_bonus(numbers)
                      if len(numbers) >= SUM_INTERVAL_WINDOW else None),
    )


def _blend_dan_score(score, meta):
    return _ranking.blend_dan_score(score, meta)


def _triplet_digit_base(a, b, c, score, meta):
    return _ranking._term_base((a, b, c), score, meta,
                               _triplet_context([], [], meta))


def triplet_weight(a, b, c, score, danma, kill, meta, features=None):
    context = _ranking.build_context(
        meta, _TRIPLET_WEIGHTS, features if features is not None else FEATURE_FLAGS,
        danma, kill,
        form_switch=(form_switch_bonus(meta.get('numbers', []))
                     if len(meta.get('numbers', [])) >= 5 else None),
        sum_interval=(sum_interval_bonus(meta.get('numbers', []))
                      if len(meta.get('numbers', [])) >= SUM_INTERVAL_WINDOW else None),
    )
    return _ranking.weight((a, b, c), score, meta, context)


def triplet_weight_detail(a, b, c, score, danma, kill, meta):
    return _ranking.detail((a, b, c), score, meta, _triplet_context(danma, kill, meta))


def build_detail_list(items, score, danma, kill, meta):
    return _ranking.build_detail_list(items, score, meta,
                                      _triplet_context(danma, kill, meta))


def select_danma(score_rank, enable_random=True):
    return _ranking.select_danma(score_rank, DANMA_TOP_POOL, DANMA_RANDOM_RATE,
                                 enable_random)


def select_diverse_pool(pool, top_n=30, candidate_size=SERVED_POOL_CANDIDATE_SIZE,
                        use_diversity=True, use_correlation=True):
    return _ranking.select_diverse_pool(
        pool, top_n, candidate_size, DIVERSITY_WEIGHT,
        CORRELATION_PENALTY, CORRELATION_THRESHOLD, use_diversity, use_correlation)


def _position_constrained_pool(score, danma, kill, meta, per_pos=ZHXUAN_POS_TOPK):
    return _ranking.position_constrained_pool(
        score, meta, _triplet_context(danma, kill, meta), per_pos)


def _merge_rank_pools(*pools, top_n):
    return _ranking.merge_pools(*pools, top_n=top_n)


def rank_triplets(score, danma, kill, meta, top_n=20, enable_exploration=True,
                  apply_noise=True, enable_cold_hot_balance=True,
                  recent_recommendations=None, enable_diversity=True,
                  enable_correlation=False):
    numbers = meta.get('numbers', [])
    hot_cold = None
    if enable_cold_hot_balance and len(numbers) >= HOT_WINDOW:
        hot, warm, cold = classify_digits_by_hot(numbers, HOT_WINDOW)
        hot_cold = {'hot': hot, 'warm': warm, 'cold': cold,
                    'hot_share': HOT_RATIO, 'warm_share': WARM_RATIO,
                    'cold_share': COLD_RATIO}
    return _ranking.rank_triplets(
        score, meta, _triplet_context(danma, kill, meta), top_n,
        hot_cold=hot_cold,
        recent_recommendations=recent_recommendations,
        penalise_recent=recent_recommend_penalty,
        diversity={'candidate_size': SERVED_POOL_CANDIDATE_SIZE},
        enable_exploration=enable_exploration, apply_noise=apply_noise,
        enable_diversity=enable_diversity, enable_correlation=enable_correlation,
        position_top_k=ZHXUAN_POS_TOPK,
    )
