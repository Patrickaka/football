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

from src.kl8.fetch import fetch_or_load_kl8_data
from src.kl8 import get_kl8_analyzer, run_prediction, clear_cache
from src.common.logger import setup_logger

log = setup_logger('kl8_scheduler')

# 记录最近已处理的期号，防止同一期内因缓存失效重复触发
_last_processed_issue = ''


def refresh_kl8_and_predict():
    """定时抓取；仅出现新一期时重新预测。

    流程:
    1. 读取当前分析器中的最新期号 (old_issue)
    2. 强制抓取最新数据 (fetch_or_load_kl8_data(force_refresh=True))
    3. 比对新旧期号
    4. 有新期号 → 清缓存 + 重新预测 (run_prediction(force_refresh=True))
    5. 无新期号 → 仅记录日志，不做额外计算
    """
    global _last_processed_issue

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
    merged_data = fetch_or_load_kl8_data(force_refresh=True)
    if not merged_data:
        log.warning('快乐8自动抓取失败，本次保留旧数据')
        return

    new_issue = merged_data[0]['issue'] if merged_data else ''

    if new_issue != old_issue and new_issue != _last_processed_issue:
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
