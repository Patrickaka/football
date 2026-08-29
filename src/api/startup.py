# -*- coding: utf-8 -*-
"""进程启动时的编排：磁盘清理、缓存恢复、后台任务、缓存预热。

**新旧两个入口共用同一份**（判据 11）。这七件事漏掉任何一件都不会让服务
起不来——它只会安静地少干活：后台不再回填赛果、缓存不再跨重启保留、
用户重新承担每天第一次的冷计算。**「零消费者时测试全绿」正是这么发生的。**

顺序是有讲究的：
1. **磁盘清理必须最先**，且是同步的。生产盘长期在 91%，先回收可再生的
   日志/报告与过期 binlog，否则后面的预热与落盘只会继续放大压力。
2. 缓存恢复要早于预热——恢复得到的当天结果能让预热直接跳过。
3. 周期任务**全部登记完再统一启动**：调度器一旦 start()，submit() 就会
   RuntimeError。
"""

import logging
import os
import threading

log = logging.getLogger('api.startup')

#: 预热是纯粹的"提前算好"，失败了只影响首次访问的延迟，不该拖垮启动。
WARMUP_THREADS = (
    ('Warm3DThread', 'src.api.runtime.caching', '_warm_3d_caches', '3D'),
    ('WarmFootballThread', 'src.api.runtime.jobs', '_warm_football_caches', '足球'),
    ('WarmBeidanThread', 'src.api.runtime.jobs', '_warm_beidan_caches', '北单'),
)


def run_startup_maintenance():
    """磁盘清理。**必须早于缓存恢复和各类预热**。

    生产盘长期 91%，启动任务只会继续放大压力；先回收可再生的日志/报告
    与过期 binlog。`MAINTENANCE_EMERGENCY_ON_STARTUP=0` 可关掉紧急模式。
    """
    try:
        from src.common.maintenance import run_maintenance

        emergency = os.getenv('MAINTENANCE_EMERGENCY_ON_STARTUP', '1'
                              ).strip().lower() not in ('0', 'false', 'no', 'off')
        run_maintenance(force_emergency=emergency)
    except Exception as exc:
        log.warning('启动前磁盘清理失败: %s', exc)


def restore_persisted_caches():
    """恢复当天有效的落盘结果，重启后无需冷计算。"""
    try:
        from src.api.runtime.caching import _load_persisted_caches

        _load_persisted_caches()
    except Exception as exc:
        log.warning('恢复落盘缓存失败: %s', exc)


def register_background_tasks():
    """登记三族周期任务，登记完再统一启动调度器。

    迁移前它们分散在三处（kl8 用 APScheduler、篮球采样自建调度器、
    另有裸线程），没有任何一个地方能回答「现在后台在跑什么」。

    每一族**各自 try**：一族登记失败不该连累另外两族——那会把"少一个任务"
    放大成"后台全停"。
    """
    from src.api.runtime import background

    try:
        from src.kl8.scheduler import register_kl8_tasks

        register_kl8_tasks(background.submit_periodic, interval_hours=1)
    except Exception as exc:
        log.warning('登记快乐8后台任务失败: %s', exc)

    try:
        from src.api.runtime.basketball_service import (
            ODDS_TRACKING_INTERVAL_MINUTES, register_odds_tracking,
        )

        # 夜里没人看的时候盘口照样在动，而那段变化正是开盘到临场的主要部分，
        # 所以必须周期采样，不能只在有人请求时才采。
        if register_odds_tracking():
            log.info('篮球赔率采样已登记: 每 %s 分钟', ODDS_TRACKING_INTERVAL_MINUTES)
        else:
            log.warning('篮球赔率采样未登记（数据库不可用）')
    except Exception as exc:
        log.warning('登记篮球赔率采样失败: %s', exc)

    try:
        from src.football.result_sync import register_football_tasks

        register_football_tasks(background.submit_periodic)
    except Exception as exc:
        log.warning('登记足球后台任务失败: %s', exc)

    try:
        background.start()
        log.info('后台调度器已启动: %s 个周期任务', background.task_count())
    except Exception as exc:
        log.warning('启动后台调度器失败: %s', exc)


def start_cache_warmups():
    """把每日首次打开的全量冷分析挪到后台。

    北单一次请求要算完整页、冷算 12 秒以上；3D 的 ML 更久。
    线程都是 daemon——**预热没算完不该拦着进程退出**。
    """
    for thread_name, module_path, function_name, label in WARMUP_THREADS:
        try:
            module = __import__(module_path, fromlist=[function_name])
            threading.Thread(target=getattr(module, function_name),
                             daemon=True, name=thread_name).start()
            log.info('%s 缓存预热线程已启动', label)
        except Exception as exc:
            log.warning('启动 %s 缓存预热失败: %s', label, exc)


def start_maintenance_schedule():
    """后续的周期清理。启动时那一轮已经同步跑过，这里不再立即执行。"""
    try:
        from src.common.maintenance import start_maintenance_scheduler

        start_maintenance_scheduler(run_immediately=False)
    except Exception as exc:
        log.warning('启动定时维护线程失败: %s', exc)


def run_all():
    """按顺序做完全部七件事。顺序见模块说明，别调换。"""
    run_startup_maintenance()
    restore_persisted_caches()
    register_background_tasks()
    start_cache_warmups()
    start_maintenance_schedule()
