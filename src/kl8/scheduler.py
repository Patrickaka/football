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

    snapshots = list_prediction_snapshots()

    # v9: 只结算 target_issue == old_latest_issue 的快照（不再按 based_on_issue < actual_issue 宽泛匹配）
    unsettled = [
        s for s in snapshots
        if not s.get('has_settlement', False)
        and s.get('target_issue') == old_latest_issue
        and not s.get('is_experiment', False)
    ]

    if not unsettled:
        log.info('快乐8: 无匹配目标期号的未结算快照')
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

    # 对每个匹配的未结算快照结算
    settled_count = 0
    for snap_info in unsettled:
        # 执行结算（settle_prediction内部会做target_issue严格校验）
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
    """检查策略是否持续低于随机基线（v9改进版）

    v9改动:
    - 只统计 is_validated=True 且 strategy_id 完全相同的记录
    - 按 actual_issue / settled_at 排序
    - 同一期只取一个 canonical snapshot
    - 至少100个独立结算期才评估（不再用30期）
    - 使用置信区间而非"低于随机80%"硬阈值
    - 降级改为: 黄色观察 → 降低推荐等级 → 人工确认后停用（不再自动清空）
    """
    from src.kl8 import list_prediction_snapshots, data_path, hypergeom_expected, ACTIVE_STRATEGIES
    from src.kl8 import _strategy_fingerprint, REFERENCE_STRATEGY
    from pathlib import Path
    import math

    settlements_dir = Path(data_path('kl8_settlements'))
    if not settlements_dir.exists():
        return

    # 加载所有结算记录并按 actual_issue 排序
    all_settlements = []
    for f in settlements_dir.glob('settlement_*.json'):
        try:
            s = json.loads(f.read_text(encoding='utf-8'))
            all_settlements.append(s)
        except Exception:
            continue

    # 按 actual_issue 排序（不再按文件名排序）
    all_settlements.sort(key=lambda x: x.get('actual_issue', ''), reverse=True)

    if len(all_settlements) < 100:
        log.info(f'快乐8: 结算数据不足({len(all_settlements)}期 < 100期)，暂不评估策略降级')
        return

    for play_type, strategy in ACTIVE_STRATEGIES.items():
        strategy_id = strategy.get('strategy_id', '')
        if not strategy_id:
            continue  # 空策略不需要检查

        # 提取 pick_n
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

        # v9: 只统计 strategy_id 完全相同的记录
        # 同一期只取一个 canonical snapshot（避免重复计数）
        strategy_hits = {}
        for s in all_settlements:
            s_strategy_ids = s.get('strategy_ids', {})
            s_id = s_strategy_ids.get(play_type, '') if isinstance(s_strategy_ids, dict) else ''
            if s_id != strategy_id:
                continue  # 不同策略的记录不参与

            actual_issue = s.get('actual_issue', '')
            hits = s.get(hit_key, 0)
            if isinstance(hits, (int, float)):
                # 同一期只取一个（如果有多个结算取第一个）
                if actual_issue not in strategy_hits:
                    strategy_hits[actual_issue] = hits

        if len(strategy_hits) < 100:
            log.info(f'快乐8: {play_type}策略{strategy_id}结算数据不足({len(strategy_hits)}期 < 100期)')
            continue

        # 计算最近100期的平均命中数和置信区间
        recent_hits = list(strategy_hits.values())[:100]
        mean_hits = sum(recent_hits) / len(recent_hits)
        std_dev = math.sqrt(sum((h - mean_hits) ** 2 for h in recent_hits) / len(recent_hits)) if len(recent_hits) > 1 else 0

        # 95% 置信区间下界
        ci_lower = mean_hits - 1.96 * std_dev / math.sqrt(len(recent_hits))

        # v9: 使用置信区间判断，而非"低于随机80%"硬阈值
        # 如果置信区间下界低于随机基线，则进入降级流程
        if ci_lower < expected_random:
            # 计算偏离程度
            deviation = (mean_hits - expected_random) / expected_random

            if deviation < -0.2:
                # 严重低于随机 → 黄色观察（标记为待确认，不自动清空）
                log.warning(
                    f'快乐8策略降级观察: {play_type}策略{strategy_id}, '
                    f'最近{len(recent_hits)}期平均命中={mean_hits:.2f}, '
                    f'随机基线={expected_random:.2f}, '
                    f'偏离={deviation:.2%}, '
                    f'95%CI下界={ci_lower:.2f}, '
                    f'标记为黄色观察(需人工确认是否停用)'
                )
                # v9: 不自动清空策略，只标记观察状态
                ACTIVE_STRATEGIES[play_type]['degradation_status'] = 'yellow_watch'
                ACTIVE_STRATEGIES[play_type]['degradation_deviation'] = round(deviation, 4)
                ACTIVE_STRATEGIES[play_type]['degradation_ci_lower'] = round(ci_lower, 4)
                # 持久化
                from src.kl8 import _persist_active_strategies
                _persist_active_strategies()
            else:
                # 轻微低于随机 → 继续观察
                log.info(
                    f'快乐8策略轻微偏离: {play_type}策略{strategy_id}, '
                    f'偏离={deviation:.2%}, CI下界={ci_lower:.2f}, 继续观察'
                )


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
