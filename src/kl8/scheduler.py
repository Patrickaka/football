#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快乐8定时调度模块
================

每小时自动抓取最新开奖数据，仅当发现新期号时才重新预测并保存快照。
同一期号不会重复生成快照，避免浪费存储。

调度策略:
- 服务启动时立即执行一次
- 之后每小时自动检查
- 有新期号 → 抓取 + 清缓存 + 重新预测 + 快照自动保存
- 无新期号 → 仅日志记录，不重复预测/快照
- 抓取失败 → 保留旧数据，下次重试

与 football/result_sync.py 共享 APScheduler（如已安装），
否则回退到简单守护线程。
"""

import json

from src.kl8.fetch import fetch_or_load_kl8_data, check_need_backfill, fetch_kl8_history_backfill
from src.kl8 import get_kl8_analyzer, run_prediction, clear_cache
from src.common.logger import setup_logger

log = setup_logger('kl8_scheduler')

# 记录最近已处理的期号，防止同一期内因缓存失效重复触发
_last_processed_issue = ''


def settle_previous_period(old_latest_issue: str):
    """自动结算上一期未结算快照

    v8新增:
    - 找到所有基于 old_latest_issue 且未结算的快照
    - 用当前最新数据(即old_latest_issue对应的开奖号码)进行结算
    - 记录命中率、奖金、ROI、策略ID
    - 更新滚动表现统计

    参数:
        old_latest_issue: 上一期期号（即需要结算的目标期号）
    """
    from src.kl8 import list_prediction_snapshots, KL8Analyzer
    from pathlib import Path
    from src.kl8 import data_path

    # 找未结算快照
    snapshots = list_prediction_snapshots()
    unsettled = [s for s in snapshots if not s.get('has_settlement', False)]

    if not unsettled:
        log.info('快乐8: 无未结算快照')
        return

    # 获取开奖号码 — old_latest_issue就是最新已开奖期号
    analyzer = get_kl8_analyzer()
    if not analyzer.history_data:
        log.warning('快乐8: 无历史数据，无法结算')
        return

    # 找到对应期号的开奖号码
    actual_data = None
    for record in analyzer.history_data:
        if record['issue'] == old_latest_issue:
            actual_data = record
            break

    if not actual_data:
        log.warning(f'快乐8: 未找到期号{old_latest_issue}的开奖数据，无法结算')
        return

    actual_numbers = actual_data['numbers']

    # 对每个未结算快照尝试结算
    settled_count = 0
    for snap_info in unsettled:
        # 只有based_on_issue < actual_issue的快照才能结算
        based_on = snap_info.get('based_on_issue', '')
        if not based_on:
            continue

        try:
            based_on_int = int(based_on)
            actual_int = int(old_latest_issue)
            if based_on_int >= actual_int:
                continue  # 快照基准期号不早于开奖期号，不能结算
        except (ValueError, TypeError):
            if str(based_on) >= str(old_latest_issue):
                continue

        # 执行结算
        analyzer_instance = get_kl8_analyzer()
        result = analyzer_instance.settle_prediction(
            snap_info['file'],
            old_latest_issue,
            actual_numbers,
        )

        if result.get('success', False):
            settled_count += 1
            settlement = result.get('settlement', {})
            log.info(
                f'快乐8自动结算: 快照{snap_info.get("snapshot_id", "")[:8]}, '
                f'期号{old_latest_issue}, '
                f'选5命中={settlement.get("hit_select_5", 0)}, '
                f'累计ROI={settlement.get("cumulative_profit_roi", 0)}'
            )
        elif 'error' in result and '已结算' in result.get('error', ''):
            # 已结算过，忽略
            pass
        else:
            log.warning(f'快乐8结算失败: {result.get("error", "unknown")}')

    if settled_count > 0:
        log.info(f'快乐8自动结算完成: {settled_count}个快照已结算')

        # v8: 更新滚动表现 — 检查是否需要降级策略
        _check_strategy_degradation()


def _check_strategy_degradation():
    """检查策略是否持续低于随机基线，需要降级

    v8新增:
    - 统计最近100期结算结果的平均命中率
    - 与理论随机基线对比
    - 如果正式策略持续低于基线，自动降级为黄色参考预测
    """
    from src.kl8 import list_prediction_snapshots, data_path, hypergeom_expected, ACTIVE_STRATEGIES
    from pathlib import Path

    settlements_dir = Path(data_path('kl8_settlements'))
    if not settlements_dir.exists():
        return

    # 统计最近100期结算
    recent_settlements = []
    for f in sorted(settlements_dir.glob('settlement_*.json'), reverse=True)[:100]:
        try:
            s = json.loads(f.read_text(encoding='utf-8'))
            recent_settlements.append(s)
        except Exception:
            continue

    if len(recent_settlements) < 30:
        log.info(f'快乐8: 结算数据不足({len(recent_settlements)}期)，暂不评估策略降级')
        return

    # 计算各玩法的平均命中率
    for play_type, strategy in ACTIVE_STRATEGIES.items():
        if not strategy.get('strategy_id', ''):
            continue  # 空策略不需要检查

        # 提取select_type
        if play_type == 'fu_shi_7':
            pick_n = 7
        elif play_type.startswith('select_'):
            try:
                pick_n = int(play_type.split('_')[1])
            except (ValueError, IndexError):
                continue
        else:
            continue

        expected_random = hypergeom_expected(pick_n)
        hit_key = f'hit_{play_type}' if play_type != 'fu_shi_7' else 'fu_shi_7_pool_hits'

        # 计算最近30期平均命中
        recent_hits = []
        for s in recent_settlements[:30]:
            hits = s.get(hit_key, 0)
            if isinstance(hits, (int, float)):
                recent_hits.append(hits)

        if len(recent_hits) < 10:
            continue

        mean_hits = sum(recent_hits) / len(recent_hits)

        # 如果命中率持续低于随机基线的80%，降级
        if mean_hits < expected_random * 0.8:
            log.warning(
                f'快乐8策略降级: {play_type} 最近{len(recent_hits)}期平均命中={mean_hits:.2f}, '
                f'低于随机基线80%({expected_random * 0.8:.2f}), '
                f'降级为参考预测'
            )
            # 清空策略 → 自动回退到REFERENCE_STRATEGY
            ACTIVE_STRATEGIES[play_type] = {
                'strategy_id': '',
                'feature_weights': {},
                'model_weights': {},
                'window_size': 0,
            }
            clear_cache()


def refresh_kl8_and_predict():
    """定时抓取；仅出现新一期时重新预测。

    v8改动:
    - 启动时先检查是否需要历史补数(不足1500期时自动补数到2000期)
    - 发现新期号后，先结算上一期未结算快照，再生成新预测
    - 新预测快照记录 target_issue = 下一期期号(不再为None)

    流程:
    1. 检查历史补数需求（不足1500期时触发补数）
    2. 读取当前分析器中的最新期号 (old_issue)
    3. 强制抓取最新数据 (fetch_or_load_kl8_data(force_refresh=True))
    4. 比对新旧期号
    5. 有新期号 → 结算上一期 → 清缓存 + 重新预测
    6. 无新期号 → 仅记录日志
    """
    global _last_processed_issue

    # v8: 检查是否需要历史补数
    backfill_check = check_need_backfill()
    if backfill_check.get('need_backfill', False):
        log.info(
            f'快乐8历史数据不足({backfill_check.get("current_periods", 0)}期)，'
            f'开始自动补数(目标{backfill_check.get("recommended_target", 2000)}期)'
        )
        backfill_result = fetch_kl8_history_backfill()
        if backfill_result.get('success', False):
            log.info(
                f'快乐8补数完成: {backfill_result.get("periods_before", 0)}期 -> '
                f'{backfill_result.get("periods_after", 0)}期'
            )
            clear_cache()
            # 补数后重建分析器
            analyzer = get_kl8_analyzer()
        else:
            log.warning(f'快乐8补数失败: {backfill_result.get("error", "unknown")}')

    analyzer = get_kl8_analyzer()
    old_issue = analyzer.history_data[0]['issue'] if analyzer.history_data else ''

    # 如果当前分析器无数据，先尝试加载一次
    if not old_issue:
        log.warning('当前无历史数据，尝试首次加载')
        merged_data = fetch_or_load_kl8_data(force_refresh=True)
        if merged_data:
            new_issue = merged_data[0]['issue'] if merged_data else ''
            _last_processed_issue = new_issue
            clear_cache()
            result = run_prediction(force_refresh=True)
            log.info(
                f'快乐8首次数据加载完成，最新期号={new_issue}，'
                f'模式={result.get("statistics", {}).get("signal_status", "unknown")}'
            )
            return
        else:
            log.warning('快乐8首次数据加载失败，下次重试')
            return

    # 已有数据 → 检查是否有新期号
    # v8: 日常只抓最近1-2页（不抓大量数据）
    merged_data = fetch_or_load_kl8_data(force_refresh=True)
    if not merged_data:
        log.warning('快乐8自动抓取失败，本次保留旧数据')
        return

    new_issue = merged_data[0]['issue'] if merged_data else ''

    if new_issue != old_issue and new_issue != _last_processed_issue:
        # v8: 先结算上一期未结算快照
        try:
            settle_previous_period(old_issue)
        except Exception as e:
            log.warning(f'快乐8自动结算上一期失败: {e}（继续生成新预测）')

        clear_cache()
        result = run_prediction(force_refresh=True)
        _last_processed_issue = new_issue
        log.info(
            f'快乐8发现新期开奖：{old_issue} -> {new_issue}，已自动生成新预测，'
            f'模式={result.get("statistics", {}).get("signal_status", "unknown")}'
        )
    elif new_issue == old_issue:
        log.info(f'快乐8暂无新期开奖，当前最新期号={new_issue}')
    else:
        # new_issue == _last_processed_issue 但 != old_issue（理论上不应发生）
        log.info(f'快乐8期号{new_issue}已处理过，跳过重复预测')


def start_kl8_scheduler(interval_hours: int = 1):
    """启动快乐8定时调度（与 football result_sync 共享 APScheduler 或回退到线程）

    参数:
        interval_hours: 检查间隔（小时），默认1
    """
    interval_seconds = interval_hours * 3600

    # 启动时立即执行一次
    log.info('快乐8调度器启动，立即执行首次同步')
    try:
        refresh_kl8_and_predict()
    except Exception as e:
        log.error(f'快乐8首次同步异常: {e}')

    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(timezone='Asia/Shanghai')

        scheduler.add_job(
            refresh_kl8_and_predict,
            trigger='interval',
            hours=interval_hours,
            id='kl8_auto_refresh',
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )

        scheduler.start()
        log.info(
            f'快乐8 APScheduler 调度器已启动，间隔 {interval_hours} 小时'
        )
        return scheduler

    except ImportError:
        log.warning('APScheduler 未安装，快乐8使用简单线程调度')

        import threading
        import time

        def _loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    refresh_kl8_and_predict()
                except Exception as e:
                    log.error(f'快乐8定时调度异常: {e}')

        thread = threading.Thread(target=_loop, daemon=True, name='KL8SchedulerThread')
        thread.start()
        log.info(
            f'快乐8简单线程调度器已启动，间隔 {interval_hours} 小时'
        )
        return thread
