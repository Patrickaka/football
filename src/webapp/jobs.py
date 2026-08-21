# -*- coding: utf-8 -*-
"""后台任务：报告同步、足球预分析、彩票/3D刷新、快乐8参数搜索"""

import os
import sys
import json
import math
import hmac
import base64
import socket
import time
import re
from datetime import datetime, timedelta
import importlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from src.common.logger import setup_logger
from src.common.paths import data_path

log = setup_logger('server')

from .http_util import (
    _sanitize_json,
)
from .lazy_modules import (
    KL8RollingBacktest, analyze_match, data_path, ensure_football_report, fetch_match_list, football_reportable_ids, get_kl8_analyzer, refresh_football_cache_index, sync_beidan_reports, sync_football_reports,
)
from .caching import (
    _CACHE, _is_cache_payload_current,
)
from . import caching as _caching_mod
from . import lazy_modules as _lazy_mod

REPORTS_DIR = Path(__file__).resolve().parents[2] / 'reports'


BAYES_MANIFEST_PATH = REPORTS_DIR / 'football_bayes_manifest.json'


_REPORT_SYNC_LOCK = threading.Lock()


_LAST_FOOTBALL_SYNC = 0.0


_LAST_BEIDAN_SYNC = 0.0


_REPORT_SYNC_INTERVAL = 30  # 秒


def _trigger_football_report_sync(mids):
    """对一批足球比赛在后台线程里同步深度报告（无则生成、变盘则重生成）。"""
    if not mids:
        return
    now = time.time()
    with _REPORT_SYNC_LOCK:
        global _LAST_FOOTBALL_SYNC
        if now - _LAST_FOOTBALL_SYNC < _REPORT_SYNC_INTERVAL:
            return
        _LAST_FOOTBALL_SYNC = now
    threading.Thread(
        target=sync_football_reports, args=(list(mids),),
        daemon=True, name='ReportSyncFootball',
    ).start()


def _trigger_beidan_report_sync(recs):
    """对一批北单 rec 在后台线程里同步深度报告（无则生成、变盘则重生成）。"""
    if not recs:
        return
    now = time.time()
    with _REPORT_SYNC_LOCK:
        global _LAST_BEIDAN_SYNC
        if now - _LAST_BEIDAN_SYNC < _REPORT_SYNC_INTERVAL:
            return
        _LAST_BEIDAN_SYNC = now
    threading.Thread(
        target=sync_beidan_reports, args=(recs,),
        daemon=True, name='ReportSyncBeidan',
    ).start()


_ANALYZE_LOCK = threading.Lock()


_LAST_FOOTBALL_ANALYZE = 0.0


_ANALYZE_SYNC_INTERVAL = 60  # 秒（比报告同步更保守）


_ANALYZE_BATCH_MAX = 15


def _match_started(match):
    """判断比赛是否已开赛（时间缺失时保守判为未开赛）。"""
    raw = match.get('time', '') or ''
    parsed = re.match(r'(\d{2})-(\d{2})\s+(\d{2}):(\d{2})', raw)
    if not parsed:
        return False
    month, day, hour, minute = (int(x) for x in parsed.groups())
    now = datetime.now()
    try:
        kickoff = datetime(now.year, month, day, hour, minute)
    except ValueError:
        return False
    # 跨年修正：若按今年解析明显在过去很久，视为明年（如 12 月看 1 月场）
    if kickoff < now - timedelta(days=180):
        try:
            kickoff = datetime(now.year + 1, month, day, hour, minute)
        except ValueError:
            return False
    return kickoff <= now


FOOTBALL_WARM_INTERVAL = int(os.getenv('FOOTBALL_WARM_INTERVAL', '1800'))


def _warm_football_caches():
    """后台预热当天未开赛场次的分析缓存，让用户不必承担冷分析。"""
    while True:
        try:
            matches = [m for m in fetch_match_list() if not _match_started(m)]
            warmed = failed = 0
            for match in matches:
                try:
                    analyze_match(match)
                    warmed += 1
                except Exception:
                    failed += 1
            log.info('足球缓存预热完成: 成功=%d, 失败=%d, 共%d场', warmed, failed, len(matches))
        except Exception:
            log.warning('足球缓存预热失败', exc_info=True)
        time.sleep(FOOTBALL_WARM_INTERVAL)


def _trigger_football_analysis(matches):
    """对名单内「没有分析缓存 pkl」的比赛，后台补跑 analyze_match，使其可生成深度报告。

    仅对当前未开赛列表中、且不在可生成集合（无 pkl）的比赛触发；已能生成报告的
    比赛不需要再分析。补分析完成后直接 ensure_football_report 出报告，并刷新索引。
    """
    if not matches:
        return
    now = time.time()
    with _ANALYZE_LOCK:
        global _LAST_FOOTBALL_ANALYZE
        if now - _LAST_FOOTBALL_ANALYZE < _ANALYZE_SYNC_INTERVAL:
            return
        _LAST_FOOTBALL_ANALYZE = now
    try:
        reportable = football_reportable_ids()
    except Exception:
        reportable = set()
    need = [m for m in matches if str(m.get('match_id')) not in reportable]
    if not need:
        return
    if len(need) > _ANALYZE_BATCH_MAX:
        need = need[:_ANALYZE_BATCH_MAX]
    threading.Thread(
        target=_run_football_analysis, args=(need,),
        daemon=True, name='FootballAnalyze',
    ).start()


def _run_football_analysis(matches):
    """后台线程：逐场补分析并生成深度报告。

    传入的 matches 均为「名单内、且当前无分析缓存 pkl」的比赛。补分析后
    ensure_football_report 会自动强制刷新索引并重生成报告（自愈）。
    """
    done = False
    for m in matches:
        mid = str(m.get('match_id') or '')
        if not mid:
            continue
        try:
            # 已生成报告则跳过
            if os.path.exists(os.path.join(REPORTS_DIR, f"football_bayes_{mid}.html")):
                continue
            log.info("后台补分析比赛 %s %s vs %s",
                     mid, m.get('home', ''), m.get('away', ''))
            analyze_match(m, force_refresh=False)
            done = True
            # 分析后直接生成深度报告（ensure_football_report 内部会强制刷新索引定位 pkl）
            ensure_football_report(mid)
        except Exception as e:
            log.warning("后台补分析失败 %s: %s", mid, e)
    # 本轮有新增 pkl，强制刷新索引，使按钮可见性及时更新
    if done:
        try:
            refresh_football_cache_index()
        except Exception:
            pass


def _load_bayes_manifest():
    """加载足球单场深度报告清单，返回 match_id -> report 的映射。"""
    if not BAYES_MANIFEST_PATH.exists():
        return {}
    try:
        with open(BAYES_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {str(k): v for k, v in (data.get('reports') or {}).items()}
    except Exception:
        return {}


def _attach_bayes_report_url(matches, kind='football'):
    """为比赛列表附加贝叶斯深度报告 URL。

    两类来源都会触发按钮显示：
    1) 清单（football_bayes_manifest.json）中已登记（说明报告文件已存在）；
    2) 可生成：足球有对应缓存 pkl、北单有已持久化 rec（即便 HTML 尚未生成，
       点击时由 server 按需生成）。

    兼容两种结构：足球比赛 match_id 在顶层；北单 rec 的 match_id 嵌套在 spf 内。
    """
    if not matches:
        return matches
    manifest = _load_bayes_manifest()
    reportable = set()
    if _lazy_mod._BAYES_REPORT_AVAILABLE and kind == 'football':
        try:
            reportable = football_reportable_ids()
        except Exception:
            reportable = set()
    prefix = 'football' if kind == 'football' else 'beidan'
    for m in matches:
        mid = str(m.get('match_id') or (m.get('spf') or {}).get('match_id') or '')
        if not mid:
            continue
        if mid in manifest:
            m['bayes_report_url'] = manifest[mid].get('url')
        elif mid in reportable:
            m['bayes_report_url'] = f"/reports/{prefix}_bayes_{mid}.html"
    return matches


KL8_PARAMETER_SEARCH_JOBS = {}


KL8_PARAMETER_SEARCH_LOCK = threading.Lock()


KL8_PARAMETER_SEARCH_REPORT_DIR = Path(data_path('kl8_parameter_search_reports'))


LOTTERY_BACKGROUND_JOBS = {}


LOTTERY_BACKGROUND_LOCK = threading.Lock()


def _set_lottery_background_job(job_id, updates):
    with LOTTERY_BACKGROUND_LOCK:
        job = LOTTERY_BACKGROUND_JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def _run_lottery_refresh_job(job_id):
    _set_lottery_background_job(job_id, {
        'status': 'processing',
        'started_at': time.time(),
        'message': '正在增量抓取并快速计算',
    })
    try:
        from src.lottery import clear_cache
        clear_cache()
        started = time.time()
        result = _lazy_mod.lottery_run_prediction(
            force_refresh=True,
            enable_backtest=False,
            enable_ml=False,
            enable_fusion=False,
            compute_weights=False,
            network_fetch_timeout=8,
        )
        if result.get('error'):
            raise RuntimeError(result['error'])
        _CACHE['lottery']['data'] = result
        _CACHE['lottery']['timestamp'] = time.time()
        _set_lottery_background_job(job_id, {
            'status': 'done',
            'success': True,
            'finished_at': time.time(),
            'elapsed': round(time.time() - started, 2),
            'message': '大乐透刷新完成',
        })
    except Exception as exc:
        log.exception('大乐透后台刷新失败')
        _set_lottery_background_job(job_id, {
            'status': 'error',
            'success': False,
            'finished_at': time.time(),
            'error': str(exc),
            'message': '大乐透刷新失败',
        })


def _start_lottery_refresh_job():
    with LOTTERY_BACKGROUND_LOCK:
        for existing_id, job in LOTTERY_BACKGROUND_JOBS.items():
            if job.get('kind') == 'lottery_refresh' and job.get('status') == 'processing':
                # A failed upstream request must not permanently block all later
                # refreshes.  Keep the old thread isolated and allow a new job.
                age = time.time() - float(job.get('started_at') or job.get('created_at') or time.time())
                if age <= 180:
                    return dict(job)
                job.update({
                    'status': 'error',
                    'success': False,
                    'finished_at': time.time(),
                    'error': 'upstream refresh timed out',
                    'message': '刷新超时，已保留上次结果',
                })
        job_id = uuid.uuid4().hex
        LOTTERY_BACKGROUND_JOBS[job_id] = {
            'task_id': job_id,
            'kind': 'lottery_refresh',
            'status': 'processing',
            'created_at': time.time(),
            'message': '后台任务已启动',
        }
    threading.Thread(
        target=_run_lottery_refresh_job,
        args=(job_id,),
        daemon=True,
        name=f'LotteryRefresh-{job_id[:8]}',
    ).start()
    return dict(LOTTERY_BACKGROUND_JOBS[job_id])


def _run_3d_refresh_job(job_id, enable_backtest=False):
    """Refresh 3D data outside the request thread so proxies never see a 504."""
    _set_lottery_background_job(job_id, {
        'status': 'processing',
        'started_at': time.time(),
        'message': '正在抓取福彩3D数据并计算',
    })
    try:
        started = time.time()
        rule_module = _lazy_mod._get_lottery3d_module()
        # 单次强制抓取后把同一份历史同时交给规则和 ML，避免两个模块重复访问上游。
        fresh_data = rule_module.fetch_data(force_refresh=True)
        if not fresh_data:
            raise RuntimeError('未获取到福彩3D数据')
        latest_period = fresh_data[-1][0]
        previous = _CACHE['3d'].get('data') or {}
        period_changed = previous.get('period') != latest_period

        result = rule_module.run_prediction(
            data=fresh_data,
            # 同一期可复用 ML 缓存；新期号只更新规则结果，ML 按钮再单独训练。
            force_refresh=False,
            enable_backtest=bool(enable_backtest),
            # 普通刷新不再强制跑 100 轮窗口权重回测。
            compute_weights=bool(enable_backtest),
            train_ml_if_stale=False,
        )
        if result.get('error'):
            raise RuntimeError(result['error'])

        _CACHE['3d']['data'] = result
        _CACHE['3d']['timestamp'] = time.time()
        _CACHE['3d_data']['data'] = fresh_data
        _CACHE['3d_data']['timestamp'] = time.time()
        _caching_mod._persist_cache('3d')
        # 普通刷新不训练 ML；新期或旧模型仅使 ML 缓存失效，显式 ML 按钮/后台
        # 预热再单飞训练，确保规则结果先快速返回。
        if (
            period_changed
            or not _is_cache_payload_current('3d_ml', _CACHE['3d_ml']['data'])
        ):
            _CACHE['3d_ml']['data'] = None
            _CACHE['3d_ml']['timestamp'] = 0

        _set_lottery_background_job(job_id, {
            'status': 'done',
            'success': True,
            'finished_at': time.time(),
            'elapsed': round(time.time() - started, 2),
            'data_count': result.get('total_periods', 0),
            'ml_data_count': len(fresh_data),
            'period': latest_period,
            'period_changed': period_changed,
            'message': '福彩3D刷新完成',
        })
    except Exception as exc:
        log.exception('福彩3D后台刷新失败')
        _set_lottery_background_job(job_id, {
            'status': 'error',
            'success': False,
            'finished_at': time.time(),
            'error': str(exc),
            'message': '福彩3D刷新失败',
        })


def _start_3d_refresh_job(enable_backtest=False):
    with LOTTERY_BACKGROUND_LOCK:
        for job in LOTTERY_BACKGROUND_JOBS.values():
            if job.get('kind') == '3d_refresh' and job.get('status') == 'processing':
                age = time.time() - float(job.get('started_at') or job.get('created_at') or time.time())
                if age <= 1800:
                    return dict(job)
                job.update({
                    'status': 'error',
                    'success': False,
                    'finished_at': time.time(),
                    'error': 'background refresh timed out',
                })
        job_id = uuid.uuid4().hex
        LOTTERY_BACKGROUND_JOBS[job_id] = {
            'task_id': job_id,
            'kind': '3d_refresh',
            'status': 'processing',
            'created_at': time.time(),
            'backtest_enabled': bool(enable_backtest),
            'message': '福彩3D后台任务已启动',
        }
    threading.Thread(
        target=_run_3d_refresh_job,
        args=(job_id, bool(enable_backtest)),
        daemon=True,
        name=f'Lottery3DRefresh-{job_id[:8]}',
    ).start()
    return dict(LOTTERY_BACKGROUND_JOBS[job_id])


def _set_kl8_parameter_search_job(job_id, updates):
    with KL8_PARAMETER_SEARCH_LOCK:
        job = KL8_PARAMETER_SEARCH_JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def _get_kl8_parameter_search_job(job_id):
    with KL8_PARAMETER_SEARCH_LOCK:
        job = KL8_PARAMETER_SEARCH_JOBS.get(job_id)
        return dict(job) if job else None


def _save_kl8_parameter_search_report(job_id, result, options):
    KL8_PARAMETER_SEARCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'job_id': job_id,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'options': options,
        'result': result,
    }
    report_file = KL8_PARAMETER_SEARCH_REPORT_DIR / f'kl8_parameter_search_{job_id}.json'
    report_file.write_text(
        json.dumps(_sanitize_json(report), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return str(report_file)


def _run_kl8_parameter_search_job(job_id, options):
    _set_kl8_parameter_search_job(job_id, {
        'status': 'running',
        'started_at': time.time(),
        'message': 'running parameter search',
    })
    try:
        analyzer = get_kl8_analyzer()
        if not analyzer.history_data:
            raise RuntimeError('no KL8 history data')
        bt = KL8RollingBacktest(analyzer)
        result = bt.run_parameter_search(
            play_types=options.get('play_types'),
            max_candidates=options.get('max_candidates', 80),
            top_n=options.get('top_n', 5),
        )
        report_file = _save_kl8_parameter_search_report(job_id, result, options)
        _set_kl8_parameter_search_job(job_id, {
            'status': 'completed',
            'finished_at': time.time(),
            'message': 'completed',
            'result': result,
            'report_file': report_file,
        })
    except Exception as e:
        _set_kl8_parameter_search_job(job_id, {
            'status': 'failed',
            'finished_at': time.time(),
            'message': str(e),
            'error': str(e),
        })
        log.exception('KL8 parameter search job failed')

