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


TICKET_PRICE = 2


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


# ─── 选号层适配 ───
#
# 组三 / 组六 / 胆拖杀 / 形态概率 / 策略准入已迁入
# `src/domain/numeric/lottery3d/{selection,admission}.py`。这里只把配置
# 常量喂进去，并保住旧的函数名与签名。

from src.domain.numeric.lottery3d import admission as _admission
from src.domain.numeric.lottery3d import selection as _selection

TICKET_PRICE = _selection.TICKET_PRICE

_effective_digit_score = lambda score, digit, kill=None: _selection.effective_digit_score(
    score, digit, kill, W_KILL_PENALTY)
zu6_notes_from_digits = _selection.zu6_notes
zu3_combos_from_pair = _selection.zu3_straight_combos
zu3_zu_notes_from_pair = _selection.zu3_group_notes
zu3_pair_scores = _selection.zu3_pair_scores


def pick_dan_tuo_kill(score, enable_danma_random=True):
    return _selection.pick_dan_tuo_kill(
        score, lambda rank: select_danma(rank, enable_random=enable_danma_random))


def pick_zu6_pool(score, kill=None, pool_size=ZU6_POOL_SIZE, use_kill=ZU6_USE_KILL):
    """**签名比迁移前少了 `pair_freq` 与 `numbers`**：那两个参数函数体里从来
    没用过，而回测与两个分析脚本都在认真地传。删掉之后，再传就是 TypeError，
    不会再有人以为它们起了作用。"""
    return _selection.zu6_pool(score, pool_size,
                               kill if use_kill else None, W_KILL_PENALTY)


def pick_zu6_four(score, kill=None, use_kill=ZU6_USE_KILL):
    return pick_zu6_pool(score, kill, pool_size=ZU6_FOUR_SIZE, use_kill=use_kill)


def build_zu6_primary(score, kill=None, numbers=None, size=ZU6_POOL_SIZE):
    return _selection.zu6_primary(score, size, kill if ZU6_USE_KILL else None,
                                  W_KILL_PENALTY)


def build_zu6_coverage_tiers(score, kill=None, sizes=(4, 5, 6, 7), numbers=None):
    return _selection.zu6_coverage_tiers(
        score, sizes, ZU6_POOL_SIZE, kill if ZU6_USE_KILL else None, W_KILL_PENALTY)


def _zu6_four_payload(label, digits):
    return _selection.zu6_payload(digits, label=label)


def _zu6_four_balance_score(combo, score, kill=None):
    return _selection.zu6_balance_score(combo, score, kill, W_KILL_PENALTY)


def build_zu6_four_variants(score, kill=None, limit=4, numbers=None):
    return _selection.zu6_four_variants(score, limit, kill, W_KILL_PENALTY, ZU6_USE_KILL)


def evaluate_zu6_pool_recent(numbers, sizes=(5, 6), trials=100):
    return _selection.evaluate_zu6_pool(
        numbers, sizes, trials, zu6_digit_scores,
        min_train=max(ZU6_PRESENCE_WINDOWS) + 5)


def zu3_digit_presence(numbers, window=None):
    return _selection.zu3_presence(
        numbers, window if window is not None else ZU3_PRESENCE_WINDOWS[0],
        ZU3_MIN_SAMPLES)


def pick_zu3_pairs(numbers, limit=ZU3_PAIRS_COUNT, presence=None):
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    pairs, conditional = _selection.zu3_pairs(presence, limit)
    return {
        'method': 'zu3_conditional_presence',
        'window': ZU3_PRESENCE_WINDOWS[0],
        'presence': {d: round(v, 4) for d, v in presence.items()},
        'pairs': pairs,
        # 模型内样本估计：top-K 概率和。presence 的噪声被取顶放大，属过拟合，
        # 500 期回测实测 ≈ 随机基准，**别当真实命中率看**。
        'conditional_hit_rate': round(conditional, 4),
        # 数学精确基准：任取 K 组条件命中 = K/45，与选哪些码无关
        'random_conditional_hit_rate': round(limit / _selection.ZU3_TOTAL_PAIRS, 4),
        'notes_total': sum(p['notes'] for p in pairs),
        'total_cost': sum(p['cost'] for p in pairs),
        'direct_notes_total': sum(p['direct_notes'] for p in pairs),
        'direct_total_cost': sum(p['direct_cost'] for p in pairs),
        'note': (
            f"组选三：每组={pairs[0]['digits_str'] if pairs else ''}式对子 = 2 注组选三/4 元"
            f"（覆盖 6 种排列，与 6 注单选 12 元等价，EV 相同）；"
            f"任取{limit}组条件命中率={limit}/45≈{limit/45:.1%}（与选哪些码基本无关，"
            "数字出现率0.17~0.24差异为噪声；conditional_hit_rate 为模型内样本估计，"
            "500期回测实测≈随机基准，属过拟合）。"
        ),
    }


def zu3_coverage_tiers(numbers, sizes=ZU3_TIER_SIZES, presence=None):
    presence = presence if presence is not None else zu3_digit_presence(numbers)
    return _selection.zu3_coverage_tiers(presence, sizes)


def analyze_form_probability(numbers, window_weights=None):
    forms = [classify_form(n) for n in numbers]
    if window_weights:
        recent_p = {k: 0.0 for k in THEORY_FORM_P}
        for window, weight in window_weights.items():
            part = _form_recent_p(forms, window)
            for key in THEORY_FORM_P:
                recent_p[key] += weight * part[key]
    else:
        recent_p = _form_recent_p(forms, RECENT_WINDOW)

    transitions = defaultdict(Counter)
    for i in range(len(forms) - 1):
        transitions[forms[i]][forms[i + 1]] += 1
    row = transitions.get(forms[-1], Counter())
    markov_p = markov_prob_smoothed(row, THEORY_FORM_P)

    historical, blended = _selection.form_probability(forms, recent_p, markov_p)
    return {
        'last_form': forms[-1],
        'streak': _selection.form_streak(forms),
        'miss_zu6': form_miss(forms, 'zu6'),
        'miss_zu3': form_miss(forms, 'zu3'),
        'recent_p': recent_p,
        'hist_p': historical,
        'markov_p': markov_p,
        'blend_p': blended,
        'markov_samples': sum(row.values()),
    }


def recommend_form_bet(form_prob, numbers):
    """动态形态主推：本期更可能是组六还是组三。

    **诚实说明**：组六基准 72% 远高于组三 27%，500 期 walk-forward 实测
    动态 max 选组六占 100%——所谓「动态判断」大多数时候仍指向组六。它真正
    的用处是量化展示组三概率何时抬升，供加注参考，不是声称能预测形态。
    """
    blend = form_prob['blend_p']
    primary = max(blend, key=blend.get)
    elevation = blend['zu3'] - THEORY_FORM_P['zu3']
    forms = [classify_form(n) for n in numbers]
    counts = Counter(forms)
    empirical = {k: counts.get(k, 0) / (len(forms) or 1) for k in THEORY_FORM_P}
    return {
        'primary': primary,
        'primary_label': FORM_LABELS[primary],
        'primary_prob': round(blend[primary], 4),
        'secondary': 'zu3' if primary != 'zu3' else 'zu6',
        'zu6_prob': round(blend['zu6'], 4),
        'zu3_prob': round(blend['zu3'], 4),
        'zu3_base_rate': THEORY_FORM_P['zu3'],
        'zu3_elevation': round(elevation, 4),
        'zu3_signal': _selection.form_signal(elevation),
        'expected_hit_rate': round(empirical[primary], 4),
        'theory_hit_rate': THEORY_FORM_P[primary],
        'empirical_form_p': {k: round(v, 4) for k, v in empirical.items()},
        'blend_p': {k: round(v, 4) for k, v in blend.items()},
        'note': (
            "主推=blend概率最大形态（组六以72%基准概率占绝对优势，500期实测动态选组六100%）；"
            "zu3_elevation>0 表示组三概率高于其27%基准，可作为加注组三的参考，"
            "但形态本身无短期可预测性，概率波动属噪声。"
        ),
    }


def evaluate_strategy_admission(served_last100_rate, raw_last100_rate,
                                actual_rank_avg, random_baseline=None,
                                significance=None):
    return _admission.evaluate(served_last100_rate, raw_last100_rate,
                               actual_rank_avg, random_baseline, significance)
