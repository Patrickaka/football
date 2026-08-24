#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透预测入口 run_prediction 及其缓存"""

import time
from typing import Any, Dict

from ..common.logger import setup_logger
from .config import (
    BACK_FEATURE_WEIGHTS, FEATURE_WEIGHTS, FULL_HISTORY_FETCH_COUNT,
    LOTTERY_PREDICTOR_VERSION, MIN_FULL_HISTORY_ISSUES, ML_BACKTEST_TRIALS,
    ROLLING_BACKTEST_TRIALS,
)
from .analyzer import get_lottery_analyzer
from .fusion import compute_fusion_weights, fuse_rule_ml
from .records import (
    _next_issue, calculate_online_stats, load_online_predictions,
    save_online_prediction, settle_predictions,
)

log = setup_logger('lottery')

# 预测结果缓存配置
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
    log.info("大乐透模块缓存已清除")


def _needs_full_history_bootstrap(data_quality: Dict[str, Any]) -> bool:
    """生产环境未提交运行时 JSON 时，自动引导全量历史。"""
    warnings = set(data_quality.get('warnings') or [])
    return (
        data_quality.get('using_simulated_data')
        or int(data_quality.get('issues') or 0) < MIN_FULL_HISTORY_ISSUES
        or 'issue_gaps' in warnings
        or 'date_anomalies' in warnings
        or not data_quality.get('ranking_allowed', False)
    )


def run_prediction(force_refresh=False, enable_backtest=True,
                   enable_ml=True, enable_fusion=True,
                   compute_weights=False, network_fetch_timeout=None):
    """运行大乐透预测，返回 JSON 可序列化 dict。

    Args:
        force_refresh: 是否强制刷新缓存（默认 False，使用缓存）
        enable_backtest: 是否启用滚动回测（默认 True）
        enable_ml: 是否启用 ML 模型预测（默认 True）
        enable_fusion: 是否启用规则+ML 融合推荐（默认 True）
        compute_weights: 是否计算动态权重（默认 False；特征回测较慢，仅排障时开启）
    """
    global _prediction_cache, _cache_time

    # 检查模块级内存缓存（按自然天判断）
    if not force_refresh and _prediction_cache is not None:
        if _is_today_cache(_cache_time):
            elapsed = time.time() - _cache_time
            log.info(f"使用今日缓存数据（缓存时间：{elapsed:.1f}秒前）")
            return _prediction_cache
        else:
            log.info("缓存已过期（非今日数据），重新计算")

    try:
        analyzer = get_lottery_analyzer()

        # 仅在历史不足/脏数据时全量引导；日常 force_refresh 只增量抓近20期
        initial_quality = analyzer.assess_data_quality()
        full_bootstrap = _needs_full_history_bootstrap(initial_quality)
        fetch_count = FULL_HISTORY_FETCH_COUNT if full_bootstrap else 20
        fetch_result = analyzer.fetch_latest_results(
            count=fetch_count,
            force_refresh=True if force_refresh else full_bootstrap,
            network_timeout=network_fetch_timeout,
        )
        if full_bootstrap:
            log.info(
                "大乐透全量历史引导完成: source=%s count=%s latest=%s",
                fetch_result.get('source'),
                fetch_result.get('count'),
                fetch_result.get('latest_issue'),
            )
        elif force_refresh:
            log.info(
                "大乐透增量抓取完成: source=%s count=%s latest=%s",
                fetch_result.get('source'),
                fetch_result.get('count'),
                fetch_result.get('latest_issue'),
            )

        # 获取统计数据
        stats = analyzer.get_statistics()
        recent = analyzer.get_recent_results(10)
        data_quality = analyzer.assess_data_quality()

        # 滚动回测（默认30期，兼顾显著性与耗时）
        if enable_backtest:
            backtest = analyzer.rolling_backtest(trials=ROLLING_BACKTEST_TRIALS)
        else:
            backtest = {'trials': 0, 'note': 'backtest disabled', 'baseline_comparison': {}}

        # 动态权重：默认关闭重型特征回测；开启时用缩短的 FEATURE_BACKTEST_TRIALS
        if compute_weights and enable_backtest:
            optimized_weights = analyzer.dynamic_weight_adjustment()
            weight_diff = {
                k: round(optimized_weights.get(k, 0) - FEATURE_WEIGHTS.get(k, 0), 4)
                for k in FEATURE_WEIGHTS
            }
        else:
            optimized_weights = dict(FEATURE_WEIGHTS)
            weight_diff = {k: 0.0 for k in FEATURE_WEIGHTS}

        # ML 先跑（投票内若启用 ML 可复用今日缓存）
        ml_prediction = None
        ml_backtest_result = None
        fusion_result = None
        if enable_ml:
            try:
                from .ml import (
                    predict_with_ml, backtest_ml, TRAINING_WINDOW as _ML_TW,
                )
                ml_prediction = predict_with_ml(
                    analyzer.history_data, force_retrain=False
                )
                if enable_backtest:
                    ml_trials = min(
                        ML_BACKTEST_TRIALS,
                        max(3, len(analyzer.history_data) - _ML_TW),
                    )
                    ml_backtest_result = backtest_ml(
                        analyzer.history_data, trials=ml_trials
                    )
            except Exception as e:
                log.warning(f"ML模型预测失败（不影响整体功能）: {e}")

        # 多模型投票一次；多策略推荐组间互斥，避免主推/均衡/排名三组重号
        # 快速页面/刷新路径必须真正跳过 ML。此前即使 enable_ml=False，
        # multi_model_voting 仍会冷启动 CatBoost，生产机器可能额外耗时数十秒。
        voting = analyzer.multi_model_voting(
            front_n=20,
            back_n=10,
            skip_ml=not enable_ml,
        )
        multi = analyzer.generate_multi_strategy_recommendations(voting_result=voting)
        recommendations = {}
        for item in multi.get('recommendations') or []:
            key = item.get('strategy') or item.get('method') or 'unknown'
            rec_entry = {
                'front': item.get('front', []),
                'back': item.get('back', []),
                'method': key,
                'label': item.get('method'),
                'core_front': item.get('core_front', []),
                'core_back': item.get('core_back', []),
                'based_on_issue': item.get('based_on_issue'),
            }
            # 透传精选一注的投票详情等额外字段
            for extra in ('picked_reason', 'selected_from', 'validation_evidence', 'front_vote_detail', 'back_vote_detail', 'cover_reason'):
                if extra in item:
                    rec_entry[extra] = item[extra]
            recommendations[key] = rec_entry

        # ML推荐 (v2.2新增策略)
        if ml_prediction and ml_prediction.get('front_top'):
            front_top5 = ml_prediction['front_top'][:5]
            back_top2 = ml_prediction['back_top'][:2]
            recommendations['ml'] = {
                'front': front_top5,
                'back': back_top2,
                'method': 'ml',
                'front_probs': ml_prediction.get('front_probs', {}),
                'back_probs': ml_prediction.get('back_probs', {}),
                'front_model_scores': ml_prediction.get('front_model_scores', {}),
                'back_model_scores': ml_prediction.get('back_model_scores', {}),
            }

        # v3.3: 规则+ML 融合推荐
        if enable_fusion and enable_ml and ml_prediction and ml_prediction.get('front_top'):
            try:
                front_ranked, back_ranked = analyzer.rank_model(top_n=35)
                rule_w, ml_w = compute_fusion_weights(backtest, ml_backtest_result or {})
                fusion_result = fuse_rule_ml(
                    front_ranked, back_ranked, ml_prediction,
                    rule_weight=rule_w, ml_weight=ml_w
                )
                # v4.1: 新增前区大底池和后区扩展池，达成"前区≥4码、后区1-2码"目标
                # 诊断: 融合Top15前区ge4=11.7%, 融合Top5后区ge1=69%
                recommendations['fusion'] = {
                    'front': fusion_result['front_top12'][:5],       # 主推5码
                    'front_pool': fusion_result['front_ranked'][:15], # 前区大底15码 (ge4=11.7%)
                    'back': fusion_result['back_top6'][:2],          # 主推2码
                    'back_pool': fusion_result['back_top6'][:5],     # 后区扩展5码 (ge1=69%)
                    'method': 'fusion',
                    'front_top12': fusion_result['front_top12'],
                    'back_top6': fusion_result['back_top6'],
                    'front_fused': fusion_result['front_ranked'][:20],
                    'back_fused': fusion_result['back_ranked'][:10],
                    'fusion_weights': {
                        'rule': fusion_result['rule_weight'],
                        'ml': fusion_result['ml_weight'],
                    },
                }
            except Exception as e:
                log.warning(f"规则+ML融合失败: {e}")

        # 保存线上预测记录（对下一期的预测，不是已开奖期号）
        latest_issue = data_quality.get('latest_issue', '')
        if latest_issue and not analyzer.using_simulated_data:
            try:
                next_issue = _next_issue(latest_issue, analyzer.history_data)
                save_online_prediction(
                    next_issue,
                    recommendations,
                    fusion_result,
                    based_on_issue=latest_issue,
                )
            except Exception as e:
                log.warning(f"保存预测记录失败: {e}")

        # 结算待回填的预测
        try:
            settled = settle_predictions(analyzer.history_data)
            if settled > 0:
                log.info(f"已结算 {settled} 条大乐透预测")
        except Exception as e:
            log.warning(f"结算预测失败: {e}")

        # 线上统计
        online_stats = calculate_online_stats()

        algorithm_summary = {
            'version': LOTTERY_PREDICTOR_VERSION,
            'history_source': '500.com 全量历史 + 本地 doc_store 缓存',
            'history_issues': data_quality.get('issues'),
            'latest_issue': data_quality.get('latest_issue'),
            'latest_date': data_quality.get('latest_date'),
            'ranking_allowed': data_quality.get('ranking_allowed'),
            'scoring': [
                '前区使用衰减频率、遗漏、位置、012路、和值、趋势、区间、重号、邻号综合排名。',
                '后区使用独立的衰减频率、遗漏、位置、012路、趋势、重号、邻号与和值评分。',
                'v3.3新增规则+ML动态权重融合，基于回测表现自动分配权重。',
                'v3.3新增ML滚动回测和线上预测记录，支持闭环学习。',
            ],
            'front_weights': FEATURE_WEIGHTS,
            'back_weights': BACK_FEATURE_WEIGHTS,
            'portfolio_policy': multi.get('portfolio_policy'),
            'rolling_backtest': {
                'trials': backtest.get('trials'),
                'baseline_comparison': backtest.get('baseline_comparison'),
                'note': backtest.get('note'),
            },
            'ml_backtest': ml_backtest_result,
            'fusion_weights': {
                'rule': fusion_result['rule_weight'] if fusion_result else 0.55,
                'ml': fusion_result['ml_weight'] if fusion_result else 0.45,
            } if fusion_result else None,
        }

        result = {
            'statistics': stats,
            'recent_results': recent,
            'backtest': backtest,
            'voting': voting,
            'recommendations': recommendations,
            'portfolio_policy': multi.get('portfolio_policy'),
            'back_coverage_profile': multi.get('back_coverage_profile'),
            'data_quality': data_quality,
            'algorithm_summary': algorithm_summary,
            'optimized_weights': optimized_weights,
            'weight_adjustment': weight_diff,
            'ml_prediction': ml_prediction,
            'ml_backtest': ml_backtest_result,
            'fusion': fusion_result,
            'online_stats': online_stats,
            'prediction_records': list(reversed(load_online_predictions()[-20:])),
            'version': LOTTERY_PREDICTOR_VERSION,
        }

        # 保存到模块级内存缓存
        _prediction_cache = result
        _cache_time = time.time()
        log.info("大乐透预测结果已缓存")

        return result
    except Exception:
        log.error('大乐透预测失败', exc_info=True)
        return {'error': '大乐透预测失败'}


if __name__ == '__main__':
    analyzer = get_lottery_analyzer()

    print("=== 大乐透分析器 (v2) ===")
    stats = analyzer.get_statistics()
    print(f"总期数: {stats.get('total_issues', 0)}")

    # 新增分析维度
    print("\n【AC值分析】")
    ac = stats.get('ac_analysis', {})
    print(f"  平均AC值: {ac.get('avg_ac', 0):.2f}")
    print(f"  常见AC值: {ac.get('most_common_ac', [])}")

    print("\n【连号分析】")
    ca = stats.get('consecutive_analysis', {})
    print(f"  含连号比例: {ca.get('pct_with_consecutive', 0):.1%}")

    print("\n【重号分析】")
    da = stats.get('duplicate_analysis', {})
    print(f"  平均重号数: {da.get('avg_duplicates', 0):.2f}")
    print(f"  有重号比例: {da.get('pct_has_duplicate', 0):.1%}")

    print("\n【和值趋势】")
    st = stats['sum_analysis'].get('trend', {})
    print(f"  方向: {st.get('direction', 'N/A')}")
    print(f"  5期MA斜率: {st.get('ma5_slope', 0)}")

    print("\n【升温降温轨迹 (Top5上升)】")
    traj = stats.get('temperature_trajectory', {})
    rising = sorted(
        [(k, v) for k, v in traj.items() if v.get('direction') == 'rising'],
        key=lambda x: x[1]['recent_hits'], reverse=True
    )[:5]
    for num, info in rising:
        print(f"  {num:02d}: {info['direction']} (近期{info['recent_hits']}次 vs 前期{info['prior_hits']}次)")

    print("\n【降温轨迹 (Top5下降)】")
    falling = sorted(
        [(k, v) for k, v in traj.items() if v.get('direction') == 'falling'],
        key=lambda x: x[1]['prior_hits'], reverse=True
    )[:5]
    for num, info in falling:
        print(f"  {num:02d}: {info['direction']} (近期{info['recent_hits']}次 vs 前期{info['prior_hits']}次)")

    # 排名模型
    front_ranked, back_ranked = analyzer.rank_model(top_n=10)
    print("\n前区排名 Top-10:")
    for num, score, features in front_ranked[:10]:
        print(f"  {num:02d}: {score:.4f}")

    print("\n后区排名 Top-6:")
    for num, score, features in back_ranked[:6]:
        print(f"  {num:02d}: {score:.4f}")

    # 集成投票
    print("\n=== 多模型集成投票推荐 (含二阶马尔可夫) ===")
    result = analyzer.multi_model_voting()
    print(f"前区推荐: {[f'{n:02d}' for n in result['front']]}")
    print(f"后区推荐: {[f'{n:02d}' for n in result['back']]}")

    # 约束推荐
    print("\n=== 约束推荐 ===")
    for method in ['balanced', 'hot', 'cold', 'rank']:
        rec = analyzer.generate_recommendation(method)
        print(f"  {method}: 前区{[f'{n:02d}' for n in rec['front']]} + 后区{[f'{n:02d}' for n in rec['back']]}")
