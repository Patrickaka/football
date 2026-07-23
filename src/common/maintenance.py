"""定时维护清理，防止磁盘被写满。

背景：生产曾因 MySQL binlog 无限增长（叠加历史上 football_prediction 的整表重写）
把磁盘写满。整表重写已在写入层修复；本模块负责「按周期自动回收可再生数据」，
免去人工手动清理。

清理边界（务必守住）：
- 只删「可再生 / 已轮转」的东西：过期 binlog、旧的滚动日志文件。
- 绝不碰业务数据：预测记录、kl8 快照/结算、开奖历史、校准库等一律不动。
- 每步独立 try/except：一步失败不影响其余，且绝不让维护线程崩溃退出。

binlog 的主策略应是 MySQL 服务端配置 `binlog_expire_logs_seconds`（由 MySQL
自身滚动过期，最可靠）；本模块的 purge 只是兜底，防止配置缺失时 binlog 失控。
"""
import os
import shutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

from . import db
from .logger import setup_logger, LOG_DIR

log = setup_logger('maintenance')

# 保留窗口与调度间隔，均可用环境变量覆盖
BINLOG_RETENTION_DAYS = int(os.getenv('MYSQL_BINLOG_RETENTION_DAYS', '3'))
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '3'))
MAINTENANCE_INTERVAL_HOURS = float(os.getenv('MAINTENANCE_INTERVAL_HOURS', '24'))
# 启动后延迟首次执行，避开与预热/同步线程争抢启动资源
MAINTENANCE_STARTUP_DELAY_SECONDS = int(os.getenv('MAINTENANCE_STARTUP_DELAY_SECONDS', '60'))
ARTIFACT_RETENTION_DAYS = int(os.getenv('ARTIFACT_RETENTION_DAYS', '3'))
DISK_MIN_FREE_GB = float(os.getenv('DISK_MIN_FREE_GB', '2'))
DISK_MIN_FREE_PERCENT = float(os.getenv('DISK_MIN_FREE_PERCENT', '10'))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / 'src' / 'football' / 'cache'
REPORTS_DIR = PROJECT_ROOT / 'reports'
CATBOOST_INFO_DIR = PROJECT_ROOT / 'catboost_info'

# Only these reproducible artifacts may be deleted automatically. Training
# data, model files, prediction history and calibration databases are excluded.
REGENERABLE_TARGETS = (
    (CACHE_DIR, ('*.pkl', '*.tmp')),
    (REPORTS_DIR, (
        'football_bayes_*.html', 'beidan_bayes_*.html',
        '_snap_*.json', 'live_context_*.json', '_agent_findings_*.json',
    )),
    (CATBOOST_INFO_DIR, ('*.log', '*.tsv', '*.json', '*.tmp')),
    (Path(LOG_DIR), ('football.log.*', '*.txt', '*.tmp')),
)


def _is_within(path: Path, root: Path) -> bool:
    """Reject symlinks and path traversal before any automatic deletion."""
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def disk_status(path: Path = PROJECT_ROOT) -> dict:
    usage = shutil.disk_usage(path)
    free_percent = usage.free / usage.total * 100 if usage.total else 0.0
    return {
        'total_bytes': usage.total,
        'used_bytes': usage.used,
        'free_bytes': usage.free,
        'free_gb': round(usage.free / (1024 ** 3), 3),
        'free_percent': round(free_percent, 2),
        'under_pressure': (
            usage.free < DISK_MIN_FREE_GB * 1024 ** 3
            or free_percent < DISK_MIN_FREE_PERCENT
        ),
    }


def cleanup_regenerable_artifacts(
    retention_days: int = None,
    dry_run: bool = False,
) -> dict:
    """Delete only allow-listed, reproducible files older than the cutoff."""
    retention_days = ARTIFACT_RETENTION_DAYS if retention_days is None else max(0, retention_days)
    cutoff = time.time() - retention_days * 86400
    removed = []
    errors = []
    bytes_freed = 0
    for root, patterns in REGENERABLE_TARGETS:
        if not root.exists() or root.is_symlink():
            continue
        seen = set()
        for pattern in patterns:
            for path in root.glob(pattern):
                if path in seen or not path.is_file() or not _is_within(path, root):
                    continue
                seen.add(path)
                try:
                    stat = path.stat()
                    if stat.st_mtime >= cutoff:
                        continue
                    item = {
                        'path': str(path.relative_to(PROJECT_ROOT)),
                        'bytes': stat.st_size,
                    }
                    if not dry_run:
                        path.unlink()
                    removed.append(item)
                    bytes_freed += stat.st_size
                except OSError as exc:
                    errors.append({'path': str(path), 'error': str(exc)})
    return {
        'dry_run': dry_run,
        'retention_days': retention_days,
        'removed_count': len(removed),
        'bytes_freed': bytes_freed,
        'removed': removed,
        'errors': errors,
    }


def purge_binlogs(retention_days: int = None) -> bool:
    """兜底清理过期 binlog，保留最近 retention_days 天。

    需 MySQL 账号具备 BINLOG_ADMIN/SUPER 权限；未开 binlog 或无权限时安全跳过。
    """
    retention_days = BINLOG_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime('%Y-%m-%d %H:%M:%S')
    try:
        db.execute("PURGE BINARY LOGS BEFORE %s", (cutoff,))
        log.info(f"binlog 清理完成：已删除 {cutoff} 之前的日志（保留最近 {retention_days} 天）")
        return True
    except Exception as e:
        # 常见原因：无权限 / 未启用 binlog / MySQL 降级中。记录即可，不阻断维护。
        log.warning(f"binlog 清理跳过（{e}）——若长期出现，请改用 MySQL 的 binlog_expire_logs_seconds")
        return False


def cleanup_rotated_logs(retention_days: int = None) -> int:
    """清理旧的滚动日志文件（football.log.*），返回删除数量。

    日志 handler 本身已限制 24 个滚动文件，但 Windows 文件占用会吞掉轮转删除、
    偶尔留下老文件；这里做一次兜底扫描。只匹配已轮转文件，绝不动当前活动的
    football.log，也不动 logs/ 下其它人工文件。
    """
    retention_days = LOG_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = time.time() - retention_days * 86400
    removed = 0
    try:
        entries = list(Path(LOG_DIR).glob('football.log.*'))
    except Exception as e:
        log.debug(f"扫描日志目录失败：{e}")
        return 0
    for path in entries:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as e:
            log.debug(f"删除旧日志失败 {path.name}：{e}")
    if removed:
        log.info(f"清理旧滚动日志 {removed} 个（早于 {retention_days} 天）")
    return removed


def run_maintenance() -> dict:
    """执行一轮维护清理。各步互不影响。"""
    log.info("开始定时维护清理")
    result = {}
    try:
        result['binlog_purged'] = purge_binlogs()
    except Exception as e:
        log.error(f"binlog 清理异常：{e}")
        result['binlog_purged'] = False
    try:
        result['logs_removed'] = cleanup_rotated_logs()
    except Exception as e:
        log.error(f"日志清理异常：{e}")
        result['logs_removed'] = 0
    try:
        before = disk_status()
        # Under pressure, remove reproducible artifacts older than one day;
        # otherwise apply the normal three-day retention window.
        retention = 1 if before['under_pressure'] else ARTIFACT_RETENTION_DAYS
        result['artifacts'] = cleanup_regenerable_artifacts(retention)
        result['disk_before'] = before
        result['disk_after'] = disk_status()
    except Exception as e:
        log.error(f"可再生文件清理异常：{e}")
        result['artifacts'] = {'removed_count': 0, 'bytes_freed': 0, 'errors': [str(e)]}
    log.info(f"维护清理完成：{result}")
    return result


def start_maintenance_scheduler(interval_hours: float = None) -> threading.Thread:
    """启动后台维护线程（守护线程，随主进程退出）。"""
    interval_hours = MAINTENANCE_INTERVAL_HOURS if interval_hours is None else interval_hours
    interval_seconds = max(60.0, interval_hours * 3600)

    def _loop():
        if MAINTENANCE_STARTUP_DELAY_SECONDS > 0:
            time.sleep(MAINTENANCE_STARTUP_DELAY_SECONDS)
        while True:
            try:
                run_maintenance()
            except Exception as e:
                log.error(f"维护线程异常：{e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_loop, daemon=True, name='MaintenanceThread')
    thread.start()
    log.info(f"维护清理线程已启动：间隔 {interval_hours} 小时，binlog 保留 {BINLOG_RETENTION_DAYS} 天，日志保留 {LOG_RETENTION_DAYS} 天")
    return thread
