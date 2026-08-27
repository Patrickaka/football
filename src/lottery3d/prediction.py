# -*- coding: utf-8 -*-
"""福彩3D预测入口 run_prediction / 报告 / CLI"""

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

from .config import (
    FEATURE_FLAGS, MIN_DATA_PERIODS_FOR_ML_FUSION, ML_CACHE_MAX_AGE_SECONDS, ML_MODEL_VERSION, PREDICTOR_VERSION, RANDOM_POS_REPEAT, RECENT_WINDOWS, RECOMMEND_GROUPS, W_KILL_PENALTY, W_LAST_APPEAR, W_POS_REPEAT, ZU6_POOL_SIZE, ZU6_RECENT_PENALTY,
)
from .fetching import (
    fetch_data,
)
from .features import (
    FORM_LABELS, THEORY_FORM_P, analyze_slope_patterns, backtest_slope_patterns, big_small_key, calc_span, has_consecutive_digits, miss_value, neighbor, odd_even_key, ratio_label,
)
from .scoring import (
    _blend_dan_score, analyze_form_probability, backtest_dan_kill, backtest_form_prediction, backtest_sum_span_interval, build_detail_list, build_ranking_meta, build_zu6_coverage_tiers, build_zu6_four_variants, build_zu6_primary, ensemble_digit_scores, ensemble_position_digit_scores, ensemble_sum_span, evaluate_strategy_admission, evaluate_zu6_pool_recent, pick_dan_tuo_kill, pick_zu3_pairs, pick_zu6_four, rank_triplets, recommend_form_bet, resolve_window_weights, zu3_coverage_tiers, zu6_digit_scores, zu6_notes_from_digits,
)
from .fusion import (
    auto_recommend_count, fuse_rule_ml, generate_strategy_recommendations, is_ml_eligible_from_backtest, load_latest_ml_performance, load_recent_rule_performance, recommend_budget_level, save_strategy_records, select_strategy_mode,
)
from .records import (
    adjust_exploration_rate, calculate_online_stats, get_stability_level, load_recent_3d_recommendations, load_recent_zu6_four, recent_zu6_digit_penalty, recommendation_stability, save_online_prediction, save_recent_3d_recommendations, save_recent_zu6_four, settle_pending_online_predictions,
)
from .backtest import (
    backtest, permutation_test, print_search_report, search_weights,
)

# ─── 领域层适配 ───
#
# 展示层的取舍与四舍五入在 `domain/numeric/lottery3d/presentation.py`，
# 数据与缓存的可用性判断在 `quality.py`。这里只把配置常量与时钟喂进去。

from src.domain.numeric.lottery3d import presentation as _view
from src.domain.numeric.lottery3d import quality as _quality
from src.domain.numeric.lottery3d.space import POSITION_NAMES

POS_NAMES = POSITION_NAMES
# 遗漏超过这么多期才值得单独列出来。十个数字全列的话那张表没有信息量——
# 遗漏三五期是常态。
LONG_MISS_THRESHOLD = 8
TOP_DIGITS_PER_POSITION = 5
TOP_MISS_PER_POSITION = 3
TOP_HOT_DIGITS = 5
TOP_SUM_TAILS = 5


def _transition_for_api(lag1, dynamic, pos_names=POS_NAMES):
    return _view.transition_view(lag1, dynamic, pos_names, RANDOM_POS_REPEAT)


def assess_data_quality(data):
    """历史数据的质量摘要。"""
    return _quality.assess_history(
        [str(row[0]) for row in data if row],
        [str(row[1]) for row in data if len(row) > 1],
        MIN_DATA_PERIODS_FOR_ML_FUSION)


def is_ml_prediction_cache_valid(cache, current_period):
    """ML 预测缓存还能不能用。时钟由这里给——读时钟是副作用。"""
    return _quality.is_cache_valid(
        cache, current_period, ML_MODEL_VERSION, ML_CACHE_MAX_AGE_SECONDS,
        time.time(),
        lambda stamp: time.mktime(time.strptime(stamp, '%Y-%m-%d %H:%M:%S')))


def run_prediction(data=None, force_refresh=False, enable_backtest=False,
                   enable_permutation=False, compute_weights=False,
                   train_ml_if_stale=True):
    """运行预测，返回 JSON 可序列化 dict；data 为 None 时自动抓取。

    **签名比迁移前少了 `use_prediction_cache`**：它默认 False 且没有任何
    调用方传过 True，而结果照样每次都写进一个模块级缓存——**只写不读**。
    连同那个缓存与它的 `clear_cache`（同样零调用方）一起删了。
    接口层自己有缓存（`webapp/caching.py`），这一层不必再存一份。

    Args:
        data: 可选的数据列表，如果为 None 则自动抓取
        force_refresh: 是否强制刷新抓取缓存
        enable_backtest: 是否启用回测（默认 False，大幅提升速度）
        enable_permutation: 是否启用排列测试（仅在 enable_backtest=True 时生效）
        compute_weights: 是否重新计算窗口权重（默认 False，用缓存或默认权重）
        train_ml_if_stale: ML缓存失效时是否立即训练；普通刷新设 False
    """
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
    zu6_four = pick_zu6_four(zu6_score)
    _, z6_straight = zu6_notes_from_digits(zu6_four)
    save_recent_zu6_four(periods[-1] if periods else None, zu6_four)
    # v4.9/v4.10: 组三推荐（四组对子）——动态形态分析的组三侧，概率透明标注
    zu3_rec = pick_zu3_pairs(numbers)
    # v4.10: 组三覆盖档位（K 组对子 → K/45 线性），无条件命中率按本期组三概率联动
    zu3_tiers = zu3_coverage_tiers(numbers)
    for _t in zu3_tiers:
        _t["unconditional_hit_rate"] = round(
            _t["conditional_hit_rate"] * form_prob["blend_p"]["zu3"], 4
        )
    
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
    # 优化配置(v4.6)：启用多样性控制提升数字覆盖率，关闭去相关避免精确率损失
    # 诊断结论：diversity ON + correlation OFF → Top30=3.88%(Lift+29.3%) 最优
    zhixuan_top = rank_triplets(
        score,
        danma,
        kill,
        meta,
        top_n=RECOMMEND_GROUPS,
        enable_exploration=False,
        apply_noise=False,
        enable_cold_hot_balance=FEATURE_FLAGS.get("cold_hot_balance", False),
        enable_diversity=True,   # v4.6: 开启 diversity 提升 Top30 命中率
        enable_correlation=False, # 保持关闭：correlation 会降低精确率
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
    ml_deferred = False
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
        elif not train_ml_if_stale:
            ml_result = None
            ml_list = []
            ml_deferred = True
            log.info("ML缓存不可用，快速刷新模式跳过重训")
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
        ml_status = "deferred" if ml_deferred else "no_recommendations"
        ml_eligible_reason = (
            "快速刷新已跳过ML重训，可用‘运行ML预测’单独计算"
            if ml_deferred else "ML推荐列表为空"
        )
        fused = rule_only_detail
        log.info("ML未参与本次快速刷新，使用纯规则模型推荐")
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

    sum_tails = [{"tail": t, "count": round(c, 2)} for t, c in meta_raw["sum_tail_freq"].most_common(5)]
    zu6_recent_validation = evaluate_zu6_pool_recent(
        numbers, sizes=(5, ZU6_POOL_SIZE), trials=100
    )
    primary_validation = (
        zu6_recent_validation.get("tiers", {}).get(str(ZU6_POOL_SIZE), {})
    )
    budget_validation = zu6_recent_validation.get("tiers", {}).get("5", {})

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
            "notes": len(z6_straight),
            "conditional_hit_rate": round(len(z6_straight) / 120.0, 4),
            # v4.9: 无条件命中率 = 组六条件命中 × 本期组六概率（动态形态概率联动）
            "unconditional_hit_rate": round(
                len(z6_straight) / 120.0 * form_prob["blend_p"]["zu6"], 4
            ),
            "form_prob": round(form_prob["blend_p"]["zu6"], 4),
        },
        "zu3_recommendation": {
            **zu3_rec,
            "form_prob": round(form_prob["blend_p"]["zu3"], 4),
            # 诚实无条件命中率：用回测验证的数学基准（4/45）× 本期组三概率，
            # 而非过拟合的 conditional_hit_rate（500期实测条件命中 8.0% ≈ 随机 8.9%）
            "unconditional_hit_rate": round(
                zu3_rec["random_conditional_hit_rate"] * form_prob["blend_p"]["zu3"], 4
            ),
            # 模型内样本估计（过拟合上限，仅供对比）
            "model_unconditional_hit_rate": round(
                zu3_rec["conditional_hit_rate"] * form_prob["blend_p"]["zu3"], 4
            ),
            # v4.10: 覆盖档位（K 组对子 → K/45 线性；组选三口径 4K 元 vs 直选 12K 元）
            "tiers": zu3_tiers,
        },
        "zu6_digit_scores": [
            {"digit": d, "score": round(zu6_score[d], 2)}
            for d in sorted(range(10), key=lambda x: -zu6_score[x])
        ],
        "zu6_primary": build_zu6_primary(zu6_score, kill=None, numbers=numbers),
        "zu6_strategy_evidence": {
            "method": "recent_walk_forward",
            "window": 25,
            "recent_trials": zu6_recent_validation.get("trials", 0),
            "validation_zu6_draws": zu6_recent_validation.get("zu6_draws", 0),
            "validation_hit_rate": primary_validation.get("conditional_full_rate", 0.0),
            "validation_ge2_rate": primary_validation.get("ge2_rate", 0.0),
            "previous_validation_hit_rate": budget_validation.get("conditional_full_rate", 0.0),
            "previous_validation_ge2_rate": budget_validation.get("ge2_rate", 0.0),
            "theoretical_conditional_hit_rate": primary_validation.get(
                "theoretical_conditional_rate", 0.0
            ),
            "theoretical_unconditional_hit_rate": primary_validation.get(
                "theoretical_unconditional_rate", 0.0
            ),
            "pool_size": ZU6_POOL_SIZE,
            "budget_pool_size": 5,
            "tiers": zu6_recent_validation.get("tiers", {}),
            "statistically_validated": False,
        },
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
    budget_info = recommend_budget_level(model_lift, recent_hit_rate)
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


