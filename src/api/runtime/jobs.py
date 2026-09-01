# -*- coding: utf-8 -*-
"""后台任务：报告同步、足球预分析与快乐8长任务。"""

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
from .beidan_cache import beidan_cache_key, write_beidan_cache
from .lazy_modules import (
    KL8RollingBacktest, _load_beidan_helpers, analyze_match, data_path, ensure_football_report, fetch_match_list, football_reportable_ids, get_kl8_analyzer, refresh_football_cache_index, sync_beidan_reports, sync_football_reports,
)
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


def finalize_beidan_recs(recs):
    """落盘北单 rec 并附上深度报告 URL，然后触发报告同步。

    只能在「真正算出了新数据」时调用。这一步一次要写三百多个 JSON、
    还会触发整批深度报告生成；此前它挂在每次请求上，北单变快之后被反复触发，
    直接把服务器 CPU 与磁盘打满、SSH 都连不上。读缓存时不该重复做这件事，
    缓存里的 rec 已经带着上一次生成的 bayes_report_url。
    """
    if not isinstance(recs, list) or not recs:
        return
    if _lazy_mod._BAYES_REPORT_AVAILABLE:
        persisted = set(_lazy_mod.persist_beidan_recs(recs))
        for rec in recs:
            mid = str(rec.get('match_id') or '')
            if mid and mid in persisted:
                rec['bayes_report_url'] = f"/reports/beidan_bayes_{mid}.html"
    else:
        _attach_bayes_report_url(recs, kind='beidan')
    _trigger_beidan_report_sync(recs)


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


BEIDAN_WARM_INTERVAL = int(os.getenv('BEIDAN_WARM_INTERVAL', '1800'))
BEIDAN_WARM_SOURCE = os.getenv('BEIDAN_WARM_SOURCE', 'zgzcw')
BEIDAN_WARM_TYPES = os.getenv('BEIDAN_WARM_TYPES', 'spf,rqspf,zjq')
# 北单与足球预热打的是同一个 odds.500.com。两者共用限速令牌流后不会再互相推进 429，
# 但同时起跑仍会从第一秒起互抢配额、把两边都拖慢。错开启动让足球那轮先跑完。
BEIDAN_WARM_START_DELAY = int(os.getenv('BEIDAN_WARM_START_DELAY', '150'))


def _warm_beidan_caches():
    """后台预热当天北单推荐。

    北单是「一次请求算完整页」的结构，冷算实测 12~15 秒全部压在用户那一次点击上，
    所以预热的收益比足球更直接：命中后前端几乎是秒开。
    预热用的键必须与接口层一致（date 缺省同为当天、bet_types 同序），否则热不到点上。
    """
    bet_types = [item for item in BEIDAN_WARM_TYPES.split(',') if item]
    if BEIDAN_WARM_START_DELAY > 0:
        time.sleep(BEIDAN_WARM_START_DELAY)
    while True:
        try:
            generate_beidan_recommendations, _, _ = _load_beidan_helpers()
            started = time.perf_counter()
            result = generate_beidan_recommendations(
                date=None, bet_types=bet_types, source=BEIDAN_WARM_SOURCE)
            if 'error' in result:
                log.warning('北单缓存预热跳过: %s', result['error'])
            else:
                finalize_beidan_recs(result.get('recommendations'))
                write_beidan_cache(beidan_cache_key(None, BEIDAN_WARM_SOURCE, bet_types), result)
                log.info('北单缓存预热完成: 推荐=%d条, 耗时 %.1fs',
                         len(result.get('recommendations') or []),
                         time.perf_counter() - started)
        except Exception:
            log.warning('北单缓存预热失败', exc_info=True)
        time.sleep(BEIDAN_WARM_INTERVAL)


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


KL8_REFRESH_JOBS = {}


KL8_REFRESH_THREADS = {}


KL8_REFRESH_LOCK = threading.Lock()


KL8_REFRESH_MAX_JOBS = 32


KL8_REFRESH_STALE_SECONDS = 300


def _copy_kl8_refresh_job(job):
    return dict(job) if job else None


def _prune_kl8_refresh_jobs_locked(preserve_id):
    if len(KL8_REFRESH_JOBS) <= KL8_REFRESH_MAX_JOBS:
        return
    terminal = sorted(
        (
            (key, value)
            for key, value in KL8_REFRESH_JOBS.items()
            if key != preserve_id
            and value.get('status') in ('completed', 'failed')
        ),
        key=lambda item: float(
            item[1].get('finished_at')
            or item[1].get('created_at')
            or 0
        ),
    )
    while len(KL8_REFRESH_JOBS) > KL8_REFRESH_MAX_JOBS and terminal:
        old_id, _ = terminal.pop(0)
        KL8_REFRESH_JOBS.pop(old_id, None)
        KL8_REFRESH_THREADS.pop(old_id, None)


def _set_kl8_refresh_job(job_id, updates):
    """更新刷新任务，并限制已完成任务的内存占用。"""
    with KL8_REFRESH_LOCK:
        job = KL8_REFRESH_JOBS.setdefault(job_id, {})
        job.update(updates)
        _prune_kl8_refresh_jobs_locked(job_id)
        return _copy_kl8_refresh_job(job)


def _get_kl8_refresh_job(job_id):
    with KL8_REFRESH_LOCK:
        return _copy_kl8_refresh_job(KL8_REFRESH_JOBS.get(job_id))


def _transition_kl8_refresh_job(job_id, allowed_statuses, updates):
    """仅让仍处于预期状态的 worker 推进任务，禁止超时任务复活。"""
    with KL8_REFRESH_LOCK:
        job = KL8_REFRESH_JOBS.get(job_id)
        if not job or job.get('status') not in allowed_statuses:
            return _copy_kl8_refresh_job(job)
        job.update(updates)
        _prune_kl8_refresh_jobs_locked(job_id)
        return _copy_kl8_refresh_job(job)


def _run_kl8_refresh_job(job_id, refresh_fn):
    """在线程中执行完整预测；HTTP 请求只查询这里的轻量状态。"""
    running = _transition_kl8_refresh_job(job_id, {'queued'}, {
        'status': 'running',
        'started_at': time.time(),
        'message': '正在重新计算快乐8预测',
    })
    if not running or running.get('status') != 'running':
        return
    try:
        payload = refresh_fn()
        if not isinstance(payload, dict):
            raise RuntimeError('快乐8重新计算返回格式错误')
        if payload.get('error'):
            raise RuntimeError(str(payload['error']))
        result = payload.get('result')
        if not payload.get('success') or not isinstance(result, dict):
            raise RuntimeError('快乐8重新计算未返回有效结果')
        if result.get('error'):
            raise RuntimeError(str(result['error']))
        _transition_kl8_refresh_job(job_id, {'running'}, {
            'status': 'completed',
            'finished_at': time.time(),
            'message': '重新计算完成',
            # 前端契约直接 renderKL8(job.result)，不要再包一层 result。
            'result': result,
        })
    except Exception as exc:
        _transition_kl8_refresh_job(job_id, {'queued', 'running'}, {
            'status': 'failed',
            'finished_at': time.time(),
            'message': str(exc),
            'error': str(exc),
        })
        log.exception('快乐8重新计算任务失败')
    finally:
        with KL8_REFRESH_LOCK:
            worker = KL8_REFRESH_THREADS.get(job_id)
            if worker is threading.current_thread():
                KL8_REFRESH_THREADS.pop(job_id, None)


def _start_kl8_refresh_job(refresh_fn):
    """单飞启动刷新任务；重复点击复用正在运行的同一个任务。"""
    with KL8_REFRESH_LOCK:
        now = time.time()
        for existing_id, job in KL8_REFRESH_JOBS.items():
            if job.get('status') not in ('queued', 'running'):
                continue
            active_since = float(
                job.get('started_at') or job.get('created_at') or now
            )
            if now - active_since > KL8_REFRESH_STALE_SECONDS:
                worker = KL8_REFRESH_THREADS.get(existing_id)
                if worker is not None and worker.is_alive():
                    # Python 线程不能安全强杀。继续复用真实存活的单飞任务，
                    # 不能假装失败后再启动一个必然等同一把预测锁的重试线程。
                    job['message'] = '重新计算耗时较长，仍在后台执行，请勿重复提交'
                else:
                    job.update({
                        'status': 'failed',
                        'finished_at': now,
                        'message': '重新计算任务已停止，请重试',
                        'error': '重新计算任务已停止，请重试',
                    })

        active = [
            job for job in KL8_REFRESH_JOBS.values()
            if job.get('status') in ('queued', 'running')
        ]
        if active:
            active.sort(
                key=lambda job: float(job.get('created_at') or 0),
                reverse=True,
            )
            return _copy_kl8_refresh_job(active[0])

        job_id = uuid.uuid4().hex
        queued = {
            'job_id': job_id,
            'status': 'queued',
            'created_at': now,
            'started_at': None,
            'finished_at': None,
            'message': '已进入重新计算队列',
        }
        KL8_REFRESH_JOBS[job_id] = dict(queued)

    thread = threading.Thread(
        target=_run_kl8_refresh_job,
        args=(job_id, refresh_fn),
        daemon=True,
        name=f'KL8Refresh-{job_id[:8]}',
    )
    with KL8_REFRESH_LOCK:
        KL8_REFRESH_THREADS[job_id] = thread
    try:
        thread.start()
    except Exception as exc:
        with KL8_REFRESH_LOCK:
            KL8_REFRESH_THREADS.pop(job_id, None)
        return _set_kl8_refresh_job(job_id, {
            'status': 'failed',
            'finished_at': time.time(),
            'message': f'无法启动重新计算任务: {exc}',
            'error': f'无法启动重新计算任务: {exc}',
        })
    # 返回启动前的快照，保证接口契约稳定为 queued；状态由轮询读取。
    return queued


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

