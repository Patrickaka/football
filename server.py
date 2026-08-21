"""
预测服务 - 网页服务
========================
标准库 http.server 实现，零第三方依赖。
集成：足球比分预测 + 各彩种预测

运行：python3 server.py
然后浏览器打开 http://localhost:9000

业务实现在 src/webapp/ 包（routing/各域 *_api mixin/caching/jobs/lazy_modules），
本文件仅保留进程入口与启动编排；同时 re-export 常用符号以兼容 `import server` 的用法。
"""

import os
import sys
import socket
import time
import json
import threading
import webbrowser
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.common.logger import setup_logger

from src.webapp.routing import Handler
from src.webapp.settings import HOST, PORT, INDEX_FILE, AUTH_ENABLED, CREDENTIALS, CORS_ORIGIN
from src.webapp.http_util import _json_default, _sanitize_json
from src.webapp.caching import (
    _CACHE, _CACHE_LOCKS, _serve_cached, _persist_cache, _load_persisted_caches,
    _is_cache_payload_current, _is_cache_valid, _compute_3d, _compute_3d_ml,
    _warm_3d_caches,
)
from src.webapp.jobs import (
    LOTTERY_BACKGROUND_JOBS, LOTTERY_BACKGROUND_LOCK, REPORTS_DIR,
    _run_3d_refresh_job, _start_3d_refresh_job, _run_lottery_refresh_job,
    _start_lottery_refresh_job, _set_lottery_background_job,
    _warm_football_caches, _warm_beidan_caches,
)
from src.webapp.lazy_modules import (
    _get_football_module, _get_lottery3d_module, _get_lottery3d_ml_module,
    fetch_match_list, get_match_list_status, analyze_match,
    run_prediction, fetch_data, predict_current,
    ssq_run_prediction, ssq_clear_cache,
    get_lottery_analyzer, lottery_run_prediction,
    kl8_run_prediction, kl8_clear_cache,
)

log = setup_logger('server')


def _is_private_lan(ip):
    """是否为常见家庭/办公局域网段（排除代理/VPN 虚拟段如 198.18.x）"""
    if ip.startswith('192.168.') or ip.startswith('10.'):
        return True
    parts = ip.split('.')
    return len(parts) == 4 and parts[0] == '172' and parts[1].isdigit() and 16 <= int(parts[1]) <= 31


def _candidate_ips():
    """收集本机所有非回环 IPv4，私有局域网段排在前面"""
    ips = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(('10.255.255.255', 1))
            ips.add(s.getsockname()[0])
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips.discard('127.0.0.1')
    return sorted(ips, key=lambda ip: (not _is_private_lan(ip), ip))


def _start_background_sync():
    """启动后台自动同步线程（football + KL8 + basketball）"""
    try:
        from src.football.result_sync import start_background_sync
        import threading

        # 使用后台线程启动同步（非阻塞）
        sync_thread = threading.Thread(
            target=start_background_sync,
            args=(7200,),  # 2小时间隔
            daemon=True,
            name='ResultSyncThread'
        )
        sync_thread.start()
        log.info('后台自动同步线程已启动')
    except Exception as e:
        log.warning(f"启动后台同步失败: {e}")

    # 快乐8定时调度（每小时检查新期号）
    try:
        from src.kl8.scheduler import start_kl8_scheduler
        start_kl8_scheduler(interval_hours=1)
        log.info('快乐8定时调度器已启动')
    except Exception as e:
        log.warning(f"启动快乐8调度器失败: {e}")

    # 篮球盘口/水位自动采样，为开盘 -> 即时盘反推持续积累真实快照。
    try:
        from src.basketball.odds_movement import start_basketball_odds_scheduler
        start_basketball_odds_scheduler(interval_minutes=15)
        log.info('篮球赔率自动追踪器已启动')
    except Exception as e:
        log.warning(f"启动篮球赔率追踪器失败: {e}")

    # 3D 缓存预热：启动后台线程提前算好规则 + ML 结果，用户永不承担冷计算
    threading.Thread(target=_warm_3d_caches, daemon=True, name='Warm3DThread').start()
    log.info('3D 缓存预热线程已启动')

    # 足球缓存预热：同理，把每日首次打开的全量冷分析挪到后台
    threading.Thread(target=_warm_football_caches, daemon=True, name='WarmFootballThread').start()
    log.info('足球缓存预热线程已启动')

    # 北单缓存预热：北单一次请求要算完整页，冷算 12 秒以上，同样挪到后台
    threading.Thread(target=_warm_beidan_caches, daemon=True, name='WarmBeidanThread').start()
    log.info('北单缓存预热线程已启动')

    # 定时维护：兜底清理过期 binlog 与旧滚动日志，防止磁盘被写满（无需人工）
    try:
        from src.common.maintenance import start_maintenance_scheduler
        start_maintenance_scheduler()
    except Exception as e:
        log.warning(f"启动定时维护线程失败: {e}")

def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    local_url = f'http://localhost:{PORT}'
    candidates = _candidate_ips()
    log.info('=' * 50)
    log.info('预测服务启动 端口=%s', PORT)
    if candidates:
        log.info('候选地址: %s %s', local_url,
                 ' '.join(f'http://{ip}:{PORT}' for ip in candidates))
    if AUTH_ENABLED:
        log.info('鉴权: 已启用 (用户: %s)', ', '.join(sorted(CREDENTIALS)))
    else:
        log.warning('鉴权: 未启用 — 公网暴露前请设置 FOOTBALL_USERS')
    
    # 启动后台自动同步
    _load_persisted_caches()  # 恢复当天有效的落盘结果，重启后无需冷计算
    _start_background_sync()
    
    log.info('=' * 50)
    try:
        webbrowser.open(local_url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info('服务已停止')
        server.shutdown()


if __name__ == '__main__':
    main()
