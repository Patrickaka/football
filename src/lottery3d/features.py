# -*- coding: utf-8 -*-
"""3D 基础特征：旧名字对领域层的适配层。

算法本身在 `src/domain/numeric/lottery3d/` 下的 `draw`（一注自身的属性）、
`history`（历史序列上的统计）、`slope`（斜连）、`recommendations`
（最近推荐的处理）四个模块里。这里留下的只有两样东西：

1. **旧的函数名与签名**——`scoring.py`、`prediction.py`、`backtest.py`
   与 `src/lottery3d/__init__.py` 都按这些名字导入。
2. **把配置常量喂进去**。领域层的函数不读全局配置，窗口、阈值、权重一律
   由调用方传入；哪个常量配哪个参数是配置问题，所以留在这里。
"""
import math
from collections import Counter, defaultdict

from src.domain.numeric.lottery3d import backtest as _bt
from src.domain.numeric.lottery3d import component_backtest as _component
from src.domain.numeric.lottery3d import draw as _draw
from src.domain.numeric.lottery3d import history as _history
from src.domain.numeric.lottery3d import recommendations as _recommend
from src.domain.numeric.lottery3d import slope as _slope

from .config import (
    EXP_DECAY, FORM_SWITCH_WEIGHT, HOT_WINDOW, MARKOV_LAPLACE_ALPHA,
    MISS_CYCLE_WINDOW, MISS_OVER_BONUS, MISS_OVER_RATIO_THRESHOLD,
    PAIR_BONUS, PAIR_FREQ_WINDOWS, PAIR_HIGH_FREQ_THRESHOLD,
    REBOUND_BONUS, REBOUND_THRESHOLD, RECENT_RECOMMEND_CONSECUTIVE_PENALTY,
    RECENT_RECOMMEND_PENALTY, RECENT_RECOMMEND_WINDOW, RECENT_WINDOW_REBOUND,
    SLOPE_MAX_CHAIN, SLOPE_MIN_CHAIN, SUM_EXTREME_PENALTY, SUM_INTERVAL_BONUS,
    SUM_INTERVAL_WIDTH, SUM_INTERVAL_WINDOW, SUM_TREND_ADJUST, SUM_TREND_WINDOW,
    W_SLOPE_MATCH, ZU3_STREAK_THRESHOLD, ZU6_STREAK_THRESHOLD,
)

FORM_LABELS = _draw.FORM_LABELS
THEORY_FORM_P = _draw.THEORY_FORM_P

calc_span = _draw.span
classify_form = _draw.classify_form
odd_even_key = _draw.odd_even_key
big_small_key = _draw.big_small_key
ratio_label = _draw.ratio_label
neighbor = _draw.neighbor
road = _draw.road
has_consecutive_digits = _draw.has_consecutive_digits

miss_value = _history.miss_value
form_miss = _history.form_miss
build_markov = _history.build_markov
build_markov2 = _history.build_markov2
gaussian_score = _history.gaussian_score
_slope_step = _slope.step_between
max_digit_overlap = _recommend.max_digit_overlap


def _recent_slice(series, window):
    return _history._recent(series, window)


def exp_weighted_counts(series, decay=EXP_DECAY):
    return _history.exp_weighted_counts(series, decay)


def markov_prob_smoothed(row, states, alpha=MARKOV_LAPLACE_ALPHA):
    return _history.markov_prob_smoothed(row, states, alpha)


def entropy_model(numbers, min_appear_window=30):
    """恒为 0。

    「长期未出现」不会提高下一期出现的概率，实盘版本据此关掉了熵值奖励。
    保留这个函数是因为 `digit_scores` 仍在把它加进去，删掉它等于把这条
    结论也一并删掉——留在这里，下次有人想「加个冷号奖励」时能先读到。
    """
    return {d: 0.0 for d in _history.digits()}


def rebound_model(numbers, window=RECENT_WINDOW_REBOUND):
    return _history.rebound_bonus(numbers, window, REBOUND_THRESHOLD, REBOUND_BONUS)


def classify_digits_by_hot(numbers, window=HOT_WINDOW):
    return _history.classify_by_hot(numbers, window)


def sum_trend_model(numbers, window=SUM_TREND_WINDOW):
    return _history.sum_trend(numbers, window, SUM_TREND_ADJUST)


def average_miss_cycle(numbers, digit, window=MISS_CYCLE_WINDOW):
    return _history.average_miss_cycle(numbers, digit, window)


def miss_cycle_bonus(numbers):
    return _history.miss_cycle_bonus(numbers, MISS_CYCLE_WINDOW,
                                     MISS_OVER_RATIO_THRESHOLD, MISS_OVER_BONUS)


def pair_frequency(numbers, window=50):
    return _history.pair_frequency(numbers, window)


def high_freq_pairs(numbers):
    return _history.high_freq_pairs(numbers, PAIR_FREQ_WINDOWS,
                                    PAIR_HIGH_FREQ_THRESHOLD)


def pair_bonus(triple, high_pairs):
    return _history.pair_bonus(triple, high_pairs, PAIR_BONUS)


def form_switch_bonus(numbers):
    return _history.form_switch_bonus(numbers, FORM_SWITCH_WEIGHT,
                                      ZU6_STREAK_THRESHOLD, ZU3_STREAK_THRESHOLD)


def sum_interval_bonus(numbers):
    return _history.sum_interval(numbers, SUM_INTERVAL_WINDOW, SUM_INTERVAL_WIDTH,
                                 SUM_INTERVAL_BONUS, SUM_EXTREME_PENALTY)


def _form_recent_p(forms, window):
    return _history.form_recent_p(forms, window, EXP_DECAY)


def _detect_position_slope_chain(digits_at_pos, min_len=SLOPE_MIN_CHAIN,
                                 max_len=SLOPE_MAX_CHAIN):
    return _slope.detect_chain(digits_at_pos, min_len, max_len)


def _cross_period_slope_signals(numbers):
    return _slope.cross_period_signals(numbers)


def analyze_slope_patterns(numbers, min_len=SLOPE_MIN_CHAIN):
    return _slope.analyze(numbers, min_len, SLOPE_MAX_CHAIN)


def slope_triplet_bonus(a, b, c, meta):
    return _slope.triplet_bonus((a, b, c), meta.get('slope'), W_SLOPE_MATCH)


def recent_recommend_penalty(pool, recent_recommendations):
    return _recommend.penalise_repeats(
        pool, recent_recommendations, RECENT_RECOMMEND_WINDOW,
        RECENT_RECOMMEND_PENALTY, RECENT_RECOMMEND_CONSECUTIVE_PENALTY)


def backtest_slope_patterns(numbers, trials=100):
    """斜连信号独立回测（分位预测是否命中）。

    起点不只受 `trials` 约束：链还没形成的那几期算不出任何信号，
    `SLOPE_MIN_CHAIN + 1` 是能出第一条信号的最早位置。
    """
    reachable = max(0, len(numbers) - SLOPE_MIN_CHAIN - 1)
    accumulator = _component.SlopeBacktest()
    for train, actual in _bt.rolling_slices(numbers, min(trials, reachable)):
        analysis = analyze_slope_patterns(train)
        accumulator.observe(actual, analysis.get('signals', []))
    return accumulator.summarise()
