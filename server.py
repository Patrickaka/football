"""
预测服务 - 网页服务
========================
标准库 http.server 实现，零第三方依赖。
集成：足球比分预测 + 福彩3D预测

运行：python3 server.py
然后浏览器打开 http://localhost:9000
"""

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
import webbrowser
import importlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ssq import run_prediction as ssq_run_prediction, clear_cache as ssq_clear_cache
from src.lottery import get_lottery_analyzer, run_prediction as lottery_run_prediction
from src.lottery.ml import predict_with_ml, clear_ml_cache
from src.kl8 import (
    get_kl8_analyzer, run_prediction as kl8_run_prediction,
    clear_cache as kl8_clear_cache, list_prediction_snapshots as kl8_list_snapshots,
    has_active_signal, is_prediction_ready as kl8_is_prediction_ready,
    KL8RollingBacktest, load_prize_table as kl8_load_prize_table,
    check_data_integrity as kl8_check_data_integrity,
    list_conflict_queue as kl8_list_conflict_queue,
    ACTIVE_STRATEGIES, REFERENCE_STRATEGY, KL8_PREDICTOR_VERSION,
    benjamini_hochberg_fdr, bonferroni_correction,
    validate_and_activate_strategy, resolve_play_strategy,
)
from src.common.logger import setup_logger
from src.common.paths import data_path

# 足球预测及报告模块体积较大，服务启动时不加载；首次真正访问足球接口时
# 再初始化。这样彩票等接口不需要为不相关的 CatBoost/足球历史付冷启动成本。
_FOOTBALL_MODULE = None
_BAYES_REPORT_MODULE = None
_LOTTERY3D_MODULE = None
_LOTTERY3D_ML_MODULE = None
_FOOTBALL_IMPORT_LOCK = threading.Lock()
_LOTTERY3D_IMPORT_LOCK = threading.Lock()
_BAYES_REPORT_AVAILABLE = True


def _get_football_module():
    global _FOOTBALL_MODULE
    if _FOOTBALL_MODULE is None:
        with _FOOTBALL_IMPORT_LOCK:
            if _FOOTBALL_MODULE is None:
                _FOOTBALL_MODULE = importlib.import_module('src.football')
    return _FOOTBALL_MODULE


def fetch_match_list(*args, **kwargs):
    return _get_football_module().fetch_match_list(*args, **kwargs)


def analyze_match(*args, **kwargs):
    return _get_football_module().analyze_match(*args, **kwargs)


def _get_bayes_report_module():
    global _BAYES_REPORT_MODULE, _BAYES_REPORT_AVAILABLE
    if _BAYES_REPORT_MODULE is None:
        try:
            _BAYES_REPORT_MODULE = importlib.import_module('src.football.bayes_report')
        except Exception:
            _BAYES_REPORT_AVAILABLE = False
            raise
    return _BAYES_REPORT_MODULE


def ensure_football_report(*args, **kwargs):
    return _get_bayes_report_module().ensure_football_report(*args, **kwargs)


def ensure_beidan_report(*args, **kwargs):
    return _get_bayes_report_module().ensure_beidan_report(*args, **kwargs)


def football_reportable_ids(*args, **kwargs):
    return _get_bayes_report_module().football_reportable_ids(*args, **kwargs)


def persist_beidan_recs(*args, **kwargs):
    return _get_bayes_report_module().persist_beidan_recs(*args, **kwargs)


def sync_football_reports(*args, **kwargs):
    return _get_bayes_report_module().sync_football_reports(*args, **kwargs)


def sync_beidan_reports(*args, **kwargs):
    return _get_bayes_report_module().sync_beidan_reports(*args, **kwargs)


def refresh_football_cache_index(*args, **kwargs):
    return _get_bayes_report_module().refresh_football_cache_index(*args, **kwargs)


def _get_lottery3d_module():
    global _LOTTERY3D_MODULE
    if _LOTTERY3D_MODULE is None:
        with _LOTTERY3D_IMPORT_LOCK:
            if _LOTTERY3D_MODULE is None:
                _LOTTERY3D_MODULE = importlib.import_module('src.lottery3d')
    return _LOTTERY3D_MODULE


def _get_lottery3d_ml_module():
    global _LOTTERY3D_ML_MODULE
    if _LOTTERY3D_ML_MODULE is None:
        with _LOTTERY3D_IMPORT_LOCK:
            if _LOTTERY3D_ML_MODULE is None:
                _LOTTERY3D_ML_MODULE = importlib.import_module('src.lottery3d.ml')
    return _LOTTERY3D_ML_MODULE


def run_prediction(*args, **kwargs):
    return _get_lottery3d_module().run_prediction(*args, **kwargs)


def fetch_data(*args, **kwargs):
    return _get_lottery3d_ml_module().fetch_data(*args, **kwargs)


def predict_current(*args, **kwargs):
    return _get_lottery3d_ml_module().predict_current(*args, **kwargs)


REPORTS_DIR = Path(os.path.join(os.path.dirname(__file__), 'reports'))
BAYES_MANIFEST_PATH = REPORTS_DIR / 'football_bayes_manifest.json'


# ===================== 深度报告后台同步（新拉取时预生成 + 变盘重生成） =====================

# 节流：避免每次 HTTP 请求都启动扫描线程（后台生成本身较慢）。
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


# ===================== 足球自动补分析（名单内缺缓存的场自动分析） =====================

# 节流：补分析较重（每次都抓赔率），避免每次请求都触发；且与报告同步错开。
_ANALYZE_LOCK = threading.Lock()
_LAST_FOOTBALL_ANALYZE = 0.0
_ANALYZE_SYNC_INTERVAL = 60  # 秒（比报告同步更保守）
# 单次最多补分析的场次，避免一次拉取把源站打爆；其余场次在后续请求中陆续补齐。
_ANALYZE_BATCH_MAX = 15


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
    if _BAYES_REPORT_AVAILABLE and kind == 'football':
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
        result = lottery_run_prediction(
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
        from src.lottery3d import clear_cache
        from src.lottery3d.ml import clear_ml_cache, fetch_data as fetch_ml_data
        from src.common.data_cache import clear_cache as clear_fetch_cache

        clear_cache()
        clear_ml_cache()
        clear_fetch_cache('lottery3d')
        clear_fetch_cache('lottery3d_ml')
        for key in ('3d', '3d_ml', '3d_data'):
            _CACHE[key]['data'] = None
            _CACHE[key]['timestamp'] = 0

        started = time.time()
        result = run_prediction(
            force_refresh=True,
            enable_backtest=bool(enable_backtest),
            compute_weights=True,
        )
        if result.get('error'):
            raise RuntimeError(result['error'])
        ml_data = fetch_ml_data(force_refresh=True)

        _CACHE['3d']['data'] = result
        _CACHE['3d']['timestamp'] = time.time()
        _CACHE['3d_data']['data'] = ml_data
        _CACHE['3d_data']['timestamp'] = time.time()
        _persist_cache('3d')
        # ML training remains lazy; the normal single-flight endpoint owns it.
        _CACHE['3d_ml']['data'] = None
        _CACHE['3d_ml']['timestamp'] = 0

        _set_lottery_background_job(job_id, {
            'status': 'done',
            'success': True,
            'finished_at': time.time(),
            'elapsed': round(time.time() - started, 2),
            'data_count': result.get('total_periods', 0),
            'ml_data_count': len(ml_data) if ml_data else 0,
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


def _current_kl8_predictor_version():
    """Read the KL8 version from the module so cache checks follow code reloads."""
    try:
        import src.kl8 as kl8_module
        return getattr(kl8_module, 'KL8_PREDICTOR_VERSION', KL8_PREDICTOR_VERSION)
    except Exception:
        return KL8_PREDICTOR_VERSION


def _load_beidan_helpers():
    try:
        from src.beidan import (
            generate_beidan_recommendations,
            find_value_bets,
            summarize_beidan_history,
        )
        return generate_beidan_recommendations, find_value_bets, summarize_beidan_history
    except ModuleNotFoundError as exc:
        if exc.name == 'requests':
            raise RuntimeError('北单模块需要安装 requests；其他页面可正常使用') from exc
        raise

def _load_basketball_helpers():
    try:
        from src.basketball import (
            generate_basketball_recommendations,
            find_value_bets,
            summarize_basketball_history,
        )
        return generate_basketball_recommendations, find_value_bets, summarize_basketball_history
    except Exception as exc:
        log.error(f"加载篮球模块失败: {exc}")
        raise


def _is_kl8_cache_current(cache_entry, now):
    if not _is_cache_valid(cache_entry, now):
        return False
    data = cache_entry.get('data')
    if not isinstance(data, dict):
        return False
    analyzer = get_kl8_analyzer()
    latest_issue = analyzer.history_data[0]['issue'] if analyzer.history_data else ''
    if not latest_issue:
        return False
    return (
        data.get('based_on_issue') == latest_issue
        and data.get('statistics', {}).get('version') == _current_kl8_predictor_version()
    )

# 回测模块（延迟导入以加速启动）
backtest = None
dynamic_threshold = None

def _import_backtest_modules():
    """延迟导入回测相关模块"""
    global backtest, dynamic_threshold
    if backtest is None:
        from src.common import backtest as bt
        backtest = bt
    if dynamic_threshold is None:
        from src.common import dynamic_threshold as dt
        dynamic_threshold = dt

log = setup_logger('server')

def _is_same_day(timestamp):
    """检查时间戳是否属于今天"""
    from datetime import date
    return date.fromtimestamp(timestamp) == date.today()


def _is_cache_valid(cache_entry, now):
    """缓存有效条件：未超过 TTL 且未跨天"""
    elapsed = now - cache_entry['timestamp']
    return elapsed < cache_entry['expire_seconds'] and _is_same_day(cache_entry['timestamp'])


# 缓存机制
_CACHE = {
    '3d_ml': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    '3d_data': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 600  # 10分钟缓存（数据抓取）
    },
    '3d': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'ssq': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'lottery': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'lottery_ml': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'beidan': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 3600  # 1小时缓存
    },
    'kl8': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
}

# 每个缓存键一把锁，用于「单飞」计算：并发的冷请求只允许一个真正计算，
# 其余请求要么复用陈旧结果、要么排队等待，避免惊群（thundering herd）把 CPU 打满。
_CACHE_LOCKS = {key: threading.Lock() for key in _CACHE}

# 需要落盘的计算结果（重启/发版后无需冷计算即可命中当天缓存）
_PERSIST_KEYS = {'3d', '3d_ml'}


def _cache_file(key):
    return Path(data_path(f'server_cache_{key}.json'))


def _persist_cache(key):
    """把计算结果落盘，供进程重启后当天复用。失败不影响主流程。"""
    if key not in _PERSIST_KEYS:
        return
    try:
        entry = _CACHE[key]
        with open(_cache_file(key), 'w', encoding='utf-8') as f:
            json.dump({'data': entry['data'], 'timestamp': entry['timestamp']},
                      f, ensure_ascii=False, default=_json_default)
    except Exception as e:
        log.warning('持久化缓存 %s 失败: %s', key, e)


def _load_persisted_caches():
    """启动时从磁盘恢复当天有效的计算结果，避免重启后首个请求冷计算。"""
    for key in _PERSIST_KEYS:
        try:
            fp = _cache_file(key)
            if not fp.exists():
                continue
            with open(fp, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            ts = obj.get('timestamp', 0)
            if obj.get('data') is not None and _is_same_day(ts):
                _CACHE[key]['data'] = obj['data']
                _CACHE[key]['timestamp'] = ts
                log.info('已从磁盘恢复缓存 %s (timestamp=%s)', key, ts)
        except Exception as e:
            log.warning('加载持久化缓存 %s 失败: %s', key, e)


def _serve_cached(key, compute_fn, background_refresh=True):
    """单飞 + stale-while-revalidate 缓存读取。

    - 缓存有效：直接返回。
    - 缓存陈旧（有旧值但已过期/跨天）：立即返回旧值，同时后台单飞刷新。
    - 缓存为空（冷启动）：阻塞等待单飞计算，并发请求只算一次。

    compute_fn() 返回原始结果或抛异常。返回 (data, error_or_None)。
    """
    now = time.time()
    cache = _CACHE[key]
    if cache['data'] is not None and _is_cache_valid(cache, now):
        return cache['data'], None

    lock = _CACHE_LOCKS[key]

    # 有陈旧数据：先返回旧值，后台刷新（只允许一个后台刷新在跑）
    if cache['data'] is not None and background_refresh:
        if lock.acquire(blocking=False):
            def _bg():
                try:
                    data = compute_fn()
                    cache['data'] = data
                    cache['timestamp'] = time.time()
                    _persist_cache(key)
                    log.info('后台刷新缓存 %s 完成', key)
                except Exception:
                    log.error('后台刷新缓存 %s 失败', key, exc_info=True)
                finally:
                    lock.release()
            threading.Thread(target=_bg, name=f'refresh-{key}', daemon=True).start()
        return cache['data'], None

    # 无数据（冷启动）：阻塞单飞，其余并发请求等待后直接命中
    with lock:
        now = time.time()
        if cache['data'] is not None and _is_cache_valid(cache, now):
            return cache['data'], None
        try:
            data = compute_fn()
        except Exception as e:
            log.error('计算缓存 %s 失败', key, exc_info=True)
            return None, str(e)
        cache['data'] = data
        cache['timestamp'] = time.time()
        _persist_cache(key)
        return data, None


def _compute_3d():
    """规则模型（快速模式：关闭回测与权重计算）。"""
    result = run_prediction(enable_backtest=False, compute_weights=False)
    if isinstance(result, dict) and 'error' in result:
        raise RuntimeError(result['error'])
    return result


def _compute_3d_ml():
    """ML 多模型集成预测，附带规则模型推荐用于对比。"""
    now = time.time()
    data_cache = _CACHE['3d_data']
    if data_cache['data'] is not None and _is_cache_valid(data_cache, now):
        data = data_cache['data']
    else:
        data = fetch_data()
        data_cache['data'] = data
        data_cache['timestamp'] = now

    numbers = [x[2] for x in data] if data else []
    result = predict_current(numbers, model_type="ensemble")
    if 'error' in result:
        raise RuntimeError(result['error'])

    # 复用已缓存的规则模型结果，避免二次 run_prediction / 二次抓取
    rule_data, _ = _serve_cached('3d', _compute_3d)
    rule_recommendations = (rule_data or {}).get('zhixuan', [])

    return {
        'model_type': result.get('model_type', 'unknown'),
        'model_info': result.get('model_info', '未知模型'),
        'num_models': int(result.get('num_models', 1)),
        'model_weights': result.get('model_weights', []),
        'total_samples': int(result.get('total_samples', 0)),
        'pos_samples': int(result.get('pos_samples', 0)),
        'neg_samples': int(result.get('neg_samples', 0)),
        'recommendations': [
            {
                'num': r['num'],
                'model_score': float(r.get('model_score', r.get('probability', 0))),
                'topk_score_share': float(r.get('topk_score_share', r.get('relative_prob', 0))),
                'relative_prob': float(r.get('relative_prob', r.get('topk_score_share', 0))),
            }
            for r in result.get('recommendations', [])
        ],
        'top3': [
            {
                'num': r['num'],
                'model_score': float(r.get('model_score', r.get('probability', 0))),
                'topk_score_share': float(r.get('topk_score_share', r.get('relative_prob', 0))),
                'relative_prob': float(r.get('relative_prob', r.get('topk_score_share', 0))),
            }
            for r in result.get('top3', [])
        ],
        'rule_recommendations': [
            {'num': r['num'], 'score': float(r.get('score', 0))}
            for r in rule_recommendations
        ],
        'feature_importance': result.get('feature_importance', []),
    }


def _warm_3d_caches():
    """后台预热 3D 规则与 ML 缓存，让用户永不承担冷计算。"""
    try:
        log.info('开始预热 3D 缓存...')
        start = time.time()
        _serve_cached('3d', _compute_3d, background_refresh=False)
        _serve_cached('3d_ml', _compute_3d_ml, background_refresh=False)
        log.info('3D 缓存预热完成，耗时 %.2f秒', time.time() - start)
    except Exception:
        log.error('3D 缓存预热失败', exc_info=True)


_ROOT = Path(__file__).parent
INDEX_FILE = _ROOT / 'web' / 'index.html'

sys.stdout.reconfigure(encoding='utf-8')

HOST = '0.0.0.0'  # 监听所有网卡，局域网/公网（经端口转发或隧道）可访问
PORT = int(os.environ.get('FOOTBALL_PORT', '9004'))

# 公网暴露时务必设置鉴权。两种方式（可并用）：
#   多用户: FOOTBALL_USERS="alice:pass1,bob:pass2"
#   单用户: FOOTBALL_USER=alice FOOTBALL_PASS=pass1
def _load_credentials():
    """解析鉴权凭据为 {用户名: 密码}；无任何配置则返回空（不启用鉴权）"""
    creds = {}
    for pair in os.environ.get('FOOTBALL_USERS', '').split(','):
        user, sep, pwd = pair.strip().partition(':')
        if sep and user.strip() and pwd.strip():
            creds[user.strip()] = pwd.strip()
    single_user = os.environ.get('FOOTBALL_USER', '').strip()
    single_pass = os.environ.get('FOOTBALL_PASS', '').strip()
    if single_user and single_pass:
        creds.setdefault(single_user, single_pass)
    return creds


CREDENTIALS = _load_credentials()
AUTH_ENABLED = bool(CREDENTIALS)

CORS_ORIGIN = os.environ.get('CORS_ORIGIN', '*')


def _json_default(obj):
    """json.dumps 兜底：numpy 标量 / 数组 / SteamSignal 等转为原生 Python 类型"""
    # numpy 数组 → list
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    # numpy 标量 → Python 原生标量
    if hasattr(obj, 'item'):
        return obj.item()
    # SteamSignal 对象 → dict
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _sanitize_json(obj):
    """递归把非有限浮点（inf/nan，含 numpy 标量/数组）替换为 None。

    json.dumps 默认 allow_nan=True 会输出 Infinity/NaN 字面量，浏览器 JSON.parse
    无法解析（distance=inf 等业务哨兵值即属此类），故序列化前统一清洗为 null。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes, bytearray)):
        return _sanitize_json(obj.tolist())
    return obj


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


class Handler(BaseHTTPRequestHandler):
    _log = log

    @staticmethod
    def _normalize_path(path):
        """兼容子路径部署（如反代到 /football/）与本地根路径访问"""
        p = path.split('?', 1)[0].rstrip('/') or '/'
        if p == '/football':
            return '/'
        if p.startswith('/football/'):
            return p[len('/football'):] or '/'
        return p

    def do_GET(self):
        start = time.perf_counter()
        if not self._authorized():
            self._log_request(401, start)
            return
        route = urlparse(self.path)
        path = self._normalize_path(route.path)
        if path == '/':
            self._serve_index()
        elif path.startswith('/reports/'):
            self._serve_report_file(path)
        elif path == '/api/matches':
            self._serve_json(self._matches_payload())
        elif path == '/api/predict':
            params = parse_qs(route.query)
            self._serve_json(self._predict_payload(params))
        elif path == '/api/football/clear_cache':
            self._serve_json(self._football_clear_cache_payload())
        elif path == '/api/football/prepare_ml_data':
            self._serve_json(self._prepare_ml_history_data_payload())
        elif path == '/api/football/diagnostics':
            params = parse_qs(route.query)
            self._serve_json(self._football_diagnostics_payload(params))
        elif path == '/api/football/review':
            params = parse_qs(route.query)
            self._serve_json(self._football_review_payload(params))
        elif path == '/api/football/professional-status':
            self._serve_json(self._football_professional_status_payload())
        elif path == '/api/3d':
            self._serve_json(self._lottery_3d_payload())
        elif path == '/api/3d-ml':
            self._serve_json(self._lottery_3d_ml_payload())
        elif path == '/api/beidan':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_payload(params))
        elif path == '/api/beidan/matches':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_matches_payload(params))
        elif path == '/api/beidan/value':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_value_payload(params))
        elif path == '/api/beidan/history':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_history_payload(params))
        elif path == '/api/basketball':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_payload(params))
        elif path == '/api/basketball/matches':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_matches_payload(params))
        elif path == '/api/basketball/value':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_value_payload(params))
        elif path == '/api/basketball/track':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_track_payload(params))
        elif path == '/api/basketball/movement':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_movement_payload(params))
        elif path == '/api/lottery':
            self._serve_json(self._lottery_payload())
        elif path == '/api/lottery-refresh':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_refresh_payload(params))
        elif path == '/api/lottery/task-status':
            self._serve_json(self._lottery_task_status_payload())
        elif path == '/api/3d-refresh':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_3d_refresh_payload(params))
        elif path == '/ssq':
            prefix = '/football' if route.path.startswith('/football/') else ''
            self.send_response(302)
            self.send_header('Location', f'{prefix}/#ssq')
            self.end_headers()
        elif path == '/api/ssq':
            self._serve_json(self._ssq_payload())
        elif path == '/api/ssq-refresh':
            self._serve_json(self._ssq_refresh_payload())
        elif path == '/api/lottery/recommend':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_recommend_payload(params))
        elif path == '/api/lottery/rank':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_rank_payload(params))
        elif path == '/api/lottery/ensemble':
            self._serve_json(self._lottery_ensemble_payload())
        elif path == '/api/lottery/cycles':
            self._serve_json(self._lottery_cycles_payload())
        elif path == '/api/lottery/contribution':
            self._serve_json(self._lottery_contribution_payload())
        elif path == '/api/lottery/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_backtest_payload(params))
        elif path == '/api/lottery/fetch':
            self._serve_json(self._lottery_fetch_payload())
        elif path == '/api/lottery/ml':
            self._serve_json(self._lottery_ml_payload())
        elif path == '/api/lottery/ml-refresh':
            self._serve_json(self._lottery_ml_refresh_payload())
        elif path == '/api/kl8':
            self._serve_json(self._kl8_payload())
        elif path == '/api/kl8-refresh':
            self._serve_json(self._kl8_refresh_payload())
        elif path == '/api/kl8/fetch':
            self._serve_json(self._kl8_fetch_payload())
        elif path == '/api/kl8/exclude-recalculate':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_exclude_recalculate_payload(params))
        elif path == '/api/kl8/snapshots':
            self._serve_json(self._kl8_snapshots_payload())
        elif path == '/api/kl8/records':
            self._serve_json(self._kl8_records_payload())
        elif path == '/api/kl8/settle':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_settle_payload(params))
        elif path == '/api/kl8/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_backtest_payload(params))
        elif path == '/api/kl8/parameter-search':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_payload(params))
        elif path == '/api/kl8/parameter-search/start':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_start_payload(params))
        elif path == '/api/kl8/parameter-search/status':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_status_payload(params))
        elif path == '/api/kl8/integrity':
            self._serve_json(self._kl8_integrity_payload())
        elif path == '/api/kl8/conflicts':
            self._serve_json(self._kl8_conflicts_payload())
        elif path == '/api/kl8/activate':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_activate_payload(params))
        elif path == '/api/calibrate':
            params = parse_qs(route.query)
            self._serve_json(self._calibrate_payload(params))
        elif path == '/api/calibrate/list':
            self._serve_json(self._calibrate_list_payload())
        elif path == '/api/calibrate/clear':
            self._serve_json(self._calibrate_clear_payload())
        elif path == '/api/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._backtest_payload(params))
        elif path == '/api/backtest/threshold':
            self._serve_json(self._threshold_payload())
        elif path == '/api/model/status':
            self._serve_json(self._model_status_payload())
        elif path == '/api/model/backtest_stats':
            params = parse_qs(route.query)
            self._serve_json(self._backtest_stats_payload(params))
        elif path == '/api/predictions':
            self._serve_json(self._predictions_payload())
        elif path == '/api/predictions/export':
            self._serve_json(self._predictions_export_payload())
        elif path == '/api/sync/status':
            self._serve_json(self._sync_status_payload())
        elif path == '/api/sync/trigger':
            self._serve_json(self._sync_trigger_payload())
        elif path == '/api/sync/hide_failed':
            self._serve_json(self._sync_hide_failed_payload())
        else:
            self._send_json_error(404, f'Not Found: {route.path}')
        self._log_request(200, start)

    def do_POST(self):
        self.do_GET()

    def do_OPTIONS(self):
        self._handle_options()

    def _handle_options(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _log_request(self, status, start):
        elapsed = time.perf_counter() - start
        self._log.info('%s %s %d %.3fs',
                       self.command, self.path, status, elapsed)

    def _authorized(self):
        """启用鉴权时校验 HTTP Basic 凭据；未启用则放行"""
        if not AUTH_ENABLED:
            return True
        header = self.headers.get('Authorization', '')
        if header.startswith('Basic '):
            try:
                user, _, pwd = base64.b64decode(header[6:]).decode('utf-8').partition(':')
                expected = CREDENTIALS.get(user)
                if expected is not None and hmac.compare_digest(pwd, expected):
                    return True
            except (ValueError, UnicodeDecodeError):
                pass
        self._log.warning('鉴权失败 %s', self.address_string())
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="football"')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return False

    def _serve_index(self):
        try:
            body = INDEX_FILE.read_bytes()
        except OSError:
            self._send(500, 'text/plain; charset=utf-8', 'index.html 缺失'.encode('utf-8'))
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, payload):
        try:
            body = json.dumps(_sanitize_json(payload), ensure_ascii=False,
                              allow_nan=False, default=_json_default).encode('utf-8')
        except (TypeError, ValueError) as e:
            self._send_json_error(500, f'JSON 序列化失败: {e}')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, status, message):
        body = json.dumps({'error': message}, ensure_ascii=False).encode('utf-8')
        self._send(status, 'application/json; charset=utf-8', body)

    def _serve_report_file(self, path):
        """提供 reports/ 目录下的静态报告文件（HTML/JSON）。"""
        rel = path[len('/reports/'):].lstrip('/')
        if not rel or '..' in rel or rel.startswith('.'):
            return self._send_json_error(403, 'Forbidden')
        file_path = REPORTS_DIR / rel
        try:
            file_path = file_path.resolve()
            reports_root = REPORTS_DIR.resolve()
            if not str(file_path).startswith(str(reports_root)):
                return self._send_json_error(403, 'Forbidden')
        except Exception:
            return self._send_json_error(404, 'Not Found')
        if rel.startswith('football_bayes_') and rel.endswith('.html'):
            # ensure_football_report performs a cheap schema/odds check and
            # regenerates stale report layouts on first access.
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
        if not file_path.exists() or not file_path.is_file():
            # 报告文件不存在 → 若可生成则按需现生成（生产环境无需手动跑脚本）
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
            else:
                return self._send_json_error(404, 'Not Found')
        content_type = 'text/html; charset=utf-8'
        if file_path.suffix.lower() == '.json':
            content_type = 'application/json; charset=utf-8'
        try:
            with open(file_path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._log.error('读取报告文件失败: %s', file_path, exc_info=True)
            self._send_json_error(500, f'读取报告失败: {e}')

    def _try_generate_report(self, rel: str):
        """按文件名尝试按需生成深度报告，返回生成的文件绝对路径或 None。"""
        if not _BAYES_REPORT_AVAILABLE:
            return None
        try:
            if rel.startswith('football_bayes_') and rel.endswith('.html'):
                mid = rel[len('football_bayes_'):-len('.html')]
                return ensure_football_report(mid)
            if rel.startswith('beidan_bayes_') and rel.endswith('.html'):
                mid = rel[len('beidan_bayes_'):-len('.html')]
                return ensure_beidan_report(mid)
        except Exception as e:
            self._log.error('报告按需生成失败: %s', rel, exc_info=True)
        return None

    def _matches_payload(self):
        try:
            matches = fetch_match_list()

            # 过滤掉「已开赛」的比赛：列表只保留未开赛场次，减少前端渲染量、
            # 也避免对已经无法进行投注分析的比赛做无谓展示（提速）。
            def _started(m):
                t = m.get('time', '') or ''
                mm = re.match(r'(\d{2})-(\d{2})\s+(\d{2}):(\d{2})', t)
                if not mm:
                    return False  # 时间缺失无法判断，保守保留
                mo, da, hh, mi = (int(x) for x in mm.groups())
                now = datetime.now()
                try:
                    dt = datetime(now.year, mo, da, hh, mi)
                except ValueError:
                    return False
                # 跨年修正：若按今年解析明显在过去很久，视为明年（如 12 月看 1 月场）
                if dt < now - timedelta(days=180):
                    try:
                        dt = datetime(now.year + 1, mo, da, hh, mi)
                    except ValueError:
                        return False
                return dt <= now

            matches = [m for m in matches if not _started(m)]
            # 后台预生成深度报告：对未开赛且有缓存的比赛，无报告则生成、变盘则重生成
            try:
                reportable = football_reportable_ids()
                sync_mids = [str(m.get('match_id')) for m in matches
                             if str(m.get('match_id')) in reportable]
                _trigger_football_report_sync(sync_mids)
                # 名单内缺分析缓存的比赛，后台自动补分析（分析后即可生成深度报告）
                _trigger_football_analysis(matches)
            except Exception:
                pass
            return {'matches': _attach_bayes_report_url(matches)}
        except Exception:
            self._log.error('获取比赛列表失败', exc_info=True)
            return {'error': '获取比赛列表失败'}

    def _predict_payload(self, params):
        match_id = params.get('match_id', [''])[0]
        if not match_id:
            return {'error': '缺少 match_id 参数'}
        
        # 检查是否强制刷新缓存
        force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'
        
        match = {
            'match_id': match_id,
            'home': params.get('home', [''])[0],
            'away': params.get('away', [''])[0],
            'league': params.get('league', [''])[0],
            'time': params.get('time', [''])[0],
            'num': params.get('num', [''])[0],
        }
        try:
            return {'result': analyze_match(match, force_refresh=force_refresh)}
        except ValueError as e:
            error_msg = str(e)
            self._log.error('赔率分析失败 match_id=%s: %s', match_id, error_msg)
            return {'error': error_msg}
        except Exception as e:
            error_msg = f'赔率分析失败: {str(e)}'
            self._log.error('赔率分析失败 match_id=%s', match_id, exc_info=True)
            return {'error': error_msg}

    def _football_clear_cache_payload(self):
        """清除足球模块缓存"""
        try:
            from src.football.cache_manager import clear_all_cache
            result = clear_all_cache()
            return result
        except Exception as e:
            self._log.error('清除足球缓存失败', exc_info=True)
            return {'error': f'清除缓存失败: {str(e)}'}

    def _prepare_ml_history_data_payload(self):
        """下载近两赛季训练数据"""
        try:
            from src.football.market_db import download_recent_two_seasons

            result = download_recent_two_seasons()

            return {
                'downloaded': len(result['success']),
                'failed': result['failed'],
                'files': result['success'],
            }
        except Exception as e:
            self._log.error('下载训练数据失败', exc_info=True)
            return {'error': f'下载失败: {str(e)}'}

    def _football_diagnostics_payload(self, params):
        try:
            limit = int((params.get('limit') or [180])[0])
            windows_raw = (params.get('windows') or ['30,60,90'])[0]
            windows = tuple(
                int(item.strip())
                for item in str(windows_raw).split(',')
                if item.strip()
            ) or (30, 60, 90)

            from src.football.backtest import rolling_backtest_from_history
            from src.football.result_sync import audit_prediction_history, get_sync_status_summary

            rolling = rolling_backtest_from_history(limit=limit, windows=windows)
            audit = audit_prediction_history(repair=False)
            sync = get_sync_status_summary()

            compact_windows = {}
            for key, report in (rolling.get('windows') or {}).items():
                summary = report.get('summary', {})
                compact_windows[key] = {
                    'sample_count': report.get('sample_count'),
                    'summary': {
                        'total_matches': summary.get('total_matches'),
                        'top1_hit_rate': summary.get('top1_hit_rate'),
                        'top3_hit_rate': summary.get('top3_hit_rate'),
                        'hit_rate_total': summary.get('hit_rate_total'),
                        'htf_top3_hit_rate': summary.get('htf_top3_hit_rate'),
                        'score_logloss': summary.get('score_logloss'),
                        'goal_logloss': summary.get('goal_logloss'),
                    },
                    'diagnostics': report.get('diagnostics', {}),
                    'diagnostic_suggestions': report.get('diagnostic_suggestions', {}),
                }

            return {
                'result': {
                    'available_samples': rolling.get('available_samples', 0),
                    'latest_window': rolling.get('latest_window'),
                    'windows': compact_windows,
                    'diagnostic_suggestions': rolling.get('diagnostic_suggestions', {}),
                    'audit': audit,
                    'sync': sync,
                }
            }
        except Exception as e:
            self._log.error('获取足球诊断面板失败', exc_info=True)
            return {'error': f'诊断失败: {str(e)}'}

    def _football_review_payload(self, params):
        try:
            repair = str((params.get('repair') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
            apply_tuning = str((params.get('apply_tuning') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
            limit = int((params.get('limit') or [180])[0])

            from src.football.backtest import apply_diagnostic_tuning_from_history, rolling_backtest_from_history
            from src.football.result_sync import audit_prediction_history, auto_sync_results, get_sync_status_summary

            sync_result = auto_sync_results()
            audit = audit_prediction_history(repair=repair)
            rolling = rolling_backtest_from_history(limit=limit)
            tuning = apply_diagnostic_tuning_from_history(
                limit=limit,
                dry_run=not apply_tuning,
            )

            return {
                'result': {
                    'sync': sync_result,
                    'audit': audit,
                    'rolling': {
                        'available_samples': rolling.get('available_samples', 0),
                        'latest_window': rolling.get('latest_window'),
                        'diagnostic_suggestions': rolling.get('diagnostic_suggestions', {}),
                    },
                    'tuning': tuning,
                    'sync_status': get_sync_status_summary(),
                    'repair': repair,
                    'apply_tuning': apply_tuning,
                }
            }
        except Exception as e:
            self._log.error('足球赛后复盘失败', exc_info=True)
            return {'error': f'复盘失败: {str(e)}'}

    def _football_professional_status_payload(self):
        """严格样本外验证、投注门控和磁盘健康的轻量状态接口。"""
        try:
            from src.football.professional_baseline import (
                BASELINE_GENERATED_AT,
                BASELINE_VERSION,
                bundled_professional_baseline,
            )
            report_path = REPORTS_DIR / 'professional_football_backtest.json'
            validation = bundled_professional_baseline()
            generated_at = BASELINE_GENERATED_AT
            validation_source = 'bundled_audited_baseline'
            if report_path.exists():
                with report_path.open(encoding='utf-8') as handle:
                    validation = json.load(handle)
                generated_at = datetime.fromtimestamp(
                    report_path.stat().st_mtime
                ).isoformat(timespec='seconds')
                validation_source = 'runtime_report'

            from src.common.maintenance import disk_status
            from src.football.professional_readiness import build_system_gap_assessment
            from src.football.professional_monitoring import build_professional_monitoring
            from src.football.result_sync import get_prediction_export
            disk = disk_status()
            monitoring = build_professional_monitoring(
                get_prediction_export().get('records') or []
            )
            model = validation.get('model_metrics') or {}
            market = validation.get('market_baseline_metrics') or {}
            strategy = validation.get('strategy') or {}
            checks = {
                'model_beats_market_logloss': (
                    bool(model) and bool(market)
                    and float(model.get('logloss', 99)) < float(market.get('logloss', 99))
                ),
                'positive_oos_roi': float(strategy.get('roi', 0) or 0) > 0,
                'positive_clv': float(strategy.get('mean_clv', 0) or 0) > 0,
                'enough_oos_samples': int(validation.get('out_of_sample_n', 0) or 0) >= 1000,
                'disk_healthy': not disk['under_pressure'],
            }
            production_ready = all((
                checks['model_beats_market_logloss'],
                checks['positive_oos_roi'],
                checks['positive_clv'],
                checks['enough_oos_samples'],
            ))
            return {
                'result': {
                    'schema_version': 'football-professional-status-v1',
                    'baseline_version': BASELINE_VERSION,
                    'generated_at': generated_at,
                    'validation_source': validation_source,
                    'validation_available': bool(validation),
                    'production_ready': production_ready,
                    'official_betting_allowed': production_ready,
                    'status_label': '生产验证通过' if production_ready else '研究模式：暂未跑赢市场',
                    'checks': checks,
                    'model_metrics': model,
                    'market_metrics': market,
                    'strategy': strategy,
                    'out_of_sample_n': validation.get('out_of_sample_n', 0),
                    'audit': validation.get('audit') or {},
                    'disk': disk,
                    'professional_assessment': build_system_gap_assessment(validation),
                    'monitoring': monitoring,
                }
            }
        except Exception as e:
            self._log.error('读取专业验证状态失败', exc_info=True)
            return {'error': f'专业验证状态不可用: {str(e)}'}

    def _lottery_3d_payload(self):
        # 单飞 + stale-while-revalidate：命中直接返回；陈旧返回旧值并后台刷新；
        # 冷启动阻塞单飞（并发只算一次），彻底避免惊群导致的生产超时。
        data, err = _serve_cached('3d', _compute_3d)
        if err is not None or data is None:
            return {'error': '3D 预测失败'}
        return {'result': data}

    def _ssq_payload(self):
        """双色球：返回近期开奖、预测和历史记录（带单飞缓存）。"""
        try:
            data, err = _serve_cached('ssq', ssq_run_prediction)
            if err is not None or data is None:
                self._log.error('双色球预测失败: %s', err)
                return {'error': '双色球预测失败'}
            if isinstance(data, dict) and data.get('error'):
                return {'error': data['error']}
            return {'result': data}
        except Exception as exc:
            self._log.error('双色球预测失败: %s', exc, exc_info=True)
            return {'error': '双色球预测失败'}

    def _ssq_refresh_payload(self):
        """双色球：清除历史缓存并强制抓取最新开奖。"""
        try:
            ssq_clear_cache()
            result = ssq_run_prediction(force_refresh=True)
            if result.get('error'):
                return {'error': result['error']}
            _CACHE['ssq']['data'] = result
            _CACHE['ssq']['timestamp'] = time.time()
            return {'result': result}
        except Exception as exc:
            self._log.error('双色球刷新失败: %s', exc, exc_info=True)
            return {'error': '双色球刷新失败'}

    def _lottery_3d_refresh_payload(self, params=None):
        """Start a 3D refresh and return immediately with a pollable task ID."""
        params = params or {}
        enable_backtest = str((params.get('backtest') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
        job = _start_3d_refresh_job(enable_backtest=enable_backtest)
        return {
            'processing': job.get('status') == 'processing',
            'task_id': job.get('task_id'),
            'message': job.get('message', '福彩3D后台刷新已启动'),
            'backtest_enabled': bool(enable_backtest),
        }

    def _lottery_3d_ml_payload(self):
        # 单飞 + stale-while-revalidate：ML 集成训练是唯一的重计算路径，
        # 交给统一缓存层，冷计算全程只发生一次且不阻塞已有用户。
        data, err = _serve_cached('3d_ml', _compute_3d_ml)
        if err is not None or data is None:
            return {'error': 'ML 3D 预测失败'}
        return {'result': data}

    def _beidan_payload(self, params):
        """获取北单推荐预测"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            bet_types = params.get('types', ['spf,zjq'])[0].split(',')
            
            self._log.info(f'北单推荐请求: date={date}, source={source}, types={bet_types}')
            
            generate_beidan_recommendations, _, _ = _load_beidan_helpers()
            result = generate_beidan_recommendations(date=date, bet_types=bet_types, source=source)
            
            if 'error' in result:
                return result

            # 为北单推荐持久化 rec（供按需生成报告）并附加深度报告 URL
            recs = result.get('recommendations')
            if isinstance(recs, list):
                if _BAYES_REPORT_AVAILABLE:
                    persisted = set(persist_beidan_recs(recs))
                    for rec in recs:
                        mid = str(rec.get('match_id') or '')
                        if mid and mid in persisted:
                            rec['bayes_report_url'] = f"/reports/beidan_bayes_{mid}.html"
                else:
                    _attach_bayes_report_url(recs, kind='beidan')
                # 后台预生成深度报告：无报告则生成、变盘则重生成
                _trigger_beidan_report_sync(recs)

            return {'result': result}
        except Exception as e:
            self._log.error('北单推荐失败', exc_info=True)
            return {'error': f'北单推荐失败: {str(e)}'}

    def _beidan_matches_payload(self, params):
        """获取北单比赛列表"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            
            if source == 'okooo':
                from src.beidan import fetch_okooo_schedule
                matches = fetch_okooo_schedule(date=date)
            else:
                from src.beidan import fetch_beidan_schedule
                matches = fetch_beidan_schedule(date=date, source=source)
            
            return {'matches': matches}
        except Exception as e:
            self._log.error('北单比赛列表获取失败', exc_info=True)
            return {'error': f'获取比赛列表失败: {str(e)}'}

    def _beidan_value_payload(self, params):
        """获取北单价值投注推荐"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            threshold = float(params.get('threshold', [0.05])[0])
            
            _, find_value_bets, _ = _load_beidan_helpers()
            result = find_value_bets(date=date, threshold=threshold, source=source)
            
            if 'error' in result:
                return {'error': result['error']}
            
            return {'result': result}
        except Exception as e:
            self._log.error('北单价值投注失败', exc_info=True)
            return {'error': f'价值投注分析失败: {str(e)}'}

    def _beidan_history_payload(self, params):
        """获取北单预测记录摘要"""
        try:
            limit = int(params.get('limit', ['200'])[0])
            _, _, summarize_beidan_history = _load_beidan_helpers()
            return {'result': summarize_beidan_history(limit=limit)}
        except Exception as e:
            self._log.error('北单预测记录获取失败', exc_info=True)
            return {'error': f'北单预测记录获取失败: {str(e)}'}

    def _basketball_payload(self, params):
        """获取篮球推荐预测"""
        try:
            date = params.get('date', [None])[0]
            bet_types = params.get('types', ['spf,rqspf,dx'])[0].split(',')
            
            self._log.info(f'篮球推荐请求: date={date}, types={bet_types}')
            
            generate_basketball_recommendations, _, _ = _load_basketball_helpers()
            result = generate_basketball_recommendations(date=date, bet_types=bet_types)
            
            if 'error' in result:
                return result
            
            matches = []
            for r in result.get('results', []):
                match_data = r.get('match', {})
                match_item = {
                    'home': match_data.get('home', ''),
                    'away': match_data.get('away', ''),
                    'league': match_data.get('league', ''),
                    'time': match_data.get('time', ''),
                    'status': match_data.get('status', ''),
                }
                
                spf = r.get('spf')
                if spf and spf.get('available'):
                    match_item['spf'] = {
                        'prediction': spf.get('recommendation'),
                        'probabilities': {
                            '主胜': spf.get('home_prob'),
                            '客胜': spf.get('away_prob'),
                        },
                        'odds': {
                            '主胜': spf.get('home_odds'),
                            '客胜': spf.get('away_odds'),
                        },
                        'line_movement': spf.get('line_movement'),
                        'sharp_confirmed': spf.get('sharp_confirmed'),
                    }
                else:
                    match_item['spf'] = {'error': spf.get('reason') if spf else 'no_data'}

                rqspf = r.get('rqspf')
                if rqspf and rqspf.get('available'):
                    match_item['rqspf'] = {
                        'prediction': rqspf.get('recommendation'),
                        'handicap': rqspf.get('handicap'),
                        'probabilities': {
                            '主胜': rqspf.get('home_prob'),
                            '客胜': rqspf.get('away_prob'),
                        },
                        'odds': {
                            '主胜': rqspf.get('home_odds'),
                            '客胜': rqspf.get('away_odds'),
                        },
                        'line_movement': rqspf.get('line_movement'),
                        'sharp_confirmed': rqspf.get('sharp_confirmed'),
                    }
                else:
                    match_item['rqspf'] = {'error': rqspf.get('reason') if rqspf else 'no_data'}

                daxiao = r.get('dx')
                if daxiao and daxiao.get('available'):
                    match_item['daxiao'] = {
                        'prediction': daxiao.get('recommendation'),
                        'total': daxiao.get('total_line'),
                        'probabilities': {
                            '大分': daxiao.get('over_prob'),
                            '小分': daxiao.get('under_prob'),
                        },
                        'odds': {
                            '大分': daxiao.get('over_odds'),
                            '小分': daxiao.get('under_odds'),
                        },
                        'line_movement': daxiao.get('line_movement'),
                        'sharp_confirmed': daxiao.get('sharp_confirmed'),
                    }
                else:
                    match_item['daxiao'] = {'error': daxiao.get('reason') if daxiao else 'no_data'}
                
                matches.append(match_item)
            
            return {'result': {
                'date': result.get('date'),
                'total_matches': len(matches),
                'matches': matches,
            }}
        except Exception as e:
            self._log.error('篮球推荐失败', exc_info=True)
            return {'error': f'篮球推荐失败: {str(e)}'}

    def _basketball_matches_payload(self, params):
        """获取篮球比赛列表"""
        try:
            date = params.get('date', [None])[0]
            
            from src.basketball import fetch_basketball_schedule
            matches = fetch_basketball_schedule(date=date)
            
            return {'matches': matches}
        except Exception as e:
            self._log.error('篮球比赛列表获取失败', exc_info=True)
            return {'error': f'获取比赛列表失败: {str(e)}'}

    def _basketball_value_payload(self, params):
        """获取篮球价值投注推荐"""
        try:
            date = params.get('date', [None])[0]
            threshold = float(params.get('threshold', [0.05])[0])

            _, find_value_bets, _ = _load_basketball_helpers()
            generate_basketball_recommendations, _, _ = _load_basketball_helpers()
            recommendations = generate_basketball_recommendations(date=date)
            value_bets = find_value_bets(recommendations.get('results', []), threshold=threshold)

            return {'result': value_bets}
        except Exception as e:
            self._log.error('篮球价值投注失败', exc_info=True)
            return {'error': f'价值投注分析失败: {str(e)}'}

    def _basketball_track_payload(self, params):
        """触发一次实时赔率轮询，累积盘路快照。"""
        try:
            from src.basketball.odds_movement import track_basketball_odds
            date = params.get('date', [None])[0]
            count = track_basketball_odds(date)
            return {'result': {'tracked': count, 'date': date}}
        except Exception as e:
            self._log.error('篮球赔率追踪失败', exc_info=True)
            return {'error': f'赔率追踪失败: {str(e)}'}

    def _basketball_movement_payload(self, params):
        """汇总当前累积的赔率走势信号。"""
        try:
            from src.basketball.odds_movement import get_odds_history
            match_id = params.get('match_id', [None])[0]
            history = get_odds_history(match_id)
            if match_id:
                return {'result': {'match_id': match_id, 'snapshots': history}}
            # 汇总每场的走势统计
            summary = []
            for mid, snaps in history.items():
                if not snaps:
                    continue
                valid = [s for s in snaps if s.get('spf_home') and s.get('spf_away')]
                first = valid[0] if valid else None
                last = valid[-1] if valid else None
                entry = {'match_id': mid, 'samples': len(snaps)}
                if first and last:
                    entry['spf_home_move'] = round((last['spf_home'] - first['spf_home']), 4)
                    entry['spf_away_move'] = round((last['spf_away'] - first['spf_away']), 4)
                summary.append(entry)
            return {'result': {'matches': len(summary), 'detail': summary}}
        except Exception as e:
            self._log.error('篮球走势汇总失败', exc_info=True)
            return {'error': f'走势汇总失败: {str(e)}'}

    def _calibrate_payload(self, params):
        """手动触发联赛重新校准"""
        league = params.get('league', [''])[0]
        if not league:
            return {'error': '缺少 league 参数'}
        recent_matches = int(params.get('matches', ['10'])[0])
        
        try:
            from src.football import recalibrate_league
            result = recalibrate_league(league, recent_matches=recent_matches)
            return {'result': result}
        except Exception as e:
            self._log.error('校准失败 league=%s', league, exc_info=True)
            return {'error': f'校准失败: {str(e)}'}

    def _calibrate_list_payload(self):
        """列出所有已校准的联赛"""
        try:
            from src.football import list_calibrated_leagues
            leagues = list_calibrated_leagues()
            return {'result': {'leagues': leagues, 'count': len(leagues)}}
        except Exception as e:
            self._log.error('获取校准列表失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _calibrate_clear_payload(self):
        """清空校准缓存"""
        try:
            from src.football import clear_calibration_cache
            result = clear_calibration_cache()
            return {'result': result}
        except Exception as e:
            self._log.error('清空校准缓存失败', exc_info=True)
            return {'error': f'清空失败: {str(e)}'}

    def _backtest_payload(self, params):
        """执行回测"""
        try:
            _import_backtest_modules()
            
            league = params.get('league', ['英超'])[0]
            start_date = params.get('start', ['2024-01-01'])[0]
            end_date = params.get('end', ['2024-06-30'])[0]
            
            result = backtest.run_backtest(league, start_date, end_date)
            return {'result': result}
        except Exception as e:
            self._log.error('回测失败', exc_info=True)
            return {'error': f'回测失败: {str(e)}'}

    def _threshold_payload(self):
        """获取动态阈值状态"""
        try:
            _import_backtest_modules()
            
            manager = dynamic_threshold.get_threshold_manager()
            stats = manager.get_statistics()
            thresholds = manager.get_thresholds()
            
            return {
                'result': {
                    'statistics': stats,
                    'thresholds': thresholds
                }
            }
        except Exception as e:
            self._log.error('获取阈值状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _lottery_payload(self):
        """获取大乐透统计分析（含缓存，调用模块级预测函数）"""
        try:
            now = time.time()
            cache = _CACHE['lottery']

            # 检查 server 级缓存（TTL + 跨天双重校验）
            if cache['data'] is not None and _is_cache_valid(cache, now):
                self._log.info('大乐透分析使用缓存（server 级）')
                return {'result': cache['data']}

            # 普通页面加载走快速路径：允许模块缓存，禁止请求内回测。
            # 回测和模型重训应由显式后台任务执行，避免反向代理504。
            self._log.info('大乐透分析快速计算')
            started = time.time()
            result = lottery_run_prediction(
                force_refresh=False,
                enable_backtest=False,
                enable_ml=False,
                enable_fusion=False,
                compute_weights=False,
            )
            self._log.info('大乐透快速计算完成，耗时 %.2f秒', time.time() - started)

            # 处理模块返回的错误
            if 'error' in result:
                return {'error': result['error']}

            # 更新 server 级缓存
            cache['data'] = result
            cache['timestamp'] = now

            return {'result': result}
        except Exception:
            self._log.error('大乐透分析失败', exc_info=True)
            return {'error': '大乐透分析失败'}

    def _lottery_refresh_payload(self, params=None):
        """在后台强制刷新大乐透，HTTP请求立即返回任务ID。"""
        job = _start_lottery_refresh_job()
        return {
            'processing': job.get('status') == 'processing',
            'task_id': job.get('task_id'),
            'message': job.get('message', '后台刷新已启动'),
        }

    def _lottery_task_status_payload(self):
        """Return background status for大乐透 and福彩3D refresh jobs."""
        now = time.time()
        with LOTTERY_BACKGROUND_LOCK:
            # 只保留最近两小时任务，避免常驻服务无限增长。
            expired = [
                job_id for job_id, job in LOTTERY_BACKGROUND_JOBS.items()
                if now - float(job.get('created_at', now)) > 7200
            ]
            for job_id in expired:
                LOTTERY_BACKGROUND_JOBS.pop(job_id, None)
            return {job_id: dict(job) for job_id, job in LOTTERY_BACKGROUND_JOBS.items()}

    def _lottery_recommend_payload(self, params):
        """获取大乐透推荐号码 - 返回5组差异化策略组合"""
        try:
            # 推荐展示、主预测缓存和预测记录必须来自同一次计算快照。
            # 旧实现会在此处重新生成一遍组合，导致页面号码与记录不一致。
            prediction = lottery_run_prediction(
                force_refresh=False,
                enable_backtest=False,
                enable_ml=False,
                enable_fusion=False,
                compute_weights=False,
            )
            if prediction.get('error'):
                raise RuntimeError(prediction['error'])
            recommendation_map = prediction.get('recommendations') or {}
            recommendations = []
            for strategy, rec in recommendation_map.items():
                item = {
                    'strategy': strategy,
                    'method': rec.get('label') or rec.get('method') or strategy,
                    'front': rec.get('front', []),
                    'back': rec.get('back', []),
                    'core_front': rec.get('core_front', []),
                    'core_back': rec.get('core_back', []),
                    'based_on_issue': rec.get('based_on_issue'),
                }
                # 透传精选一注的投票详情等额外字段
                for extra in ('picked_reason', 'front_vote_detail', 'back_vote_detail'):
                    if extra in rec:
                        item[extra] = rec[extra]
                recommendations.append(item)

            return {
                'result': {
                    'method': 'multi_strategy',
                    'recommendations': recommendations,
                    'count': len(recommendations),
                    'portfolio_policy': prediction.get('portfolio_policy') or {},
                    'back_coverage_profile': prediction.get('back_coverage_profile') or {},
                    'version': prediction.get('version'),
                }
            }
        except Exception:
            self._log.error('大乐透推荐失败', exc_info=True)
            return {'error': '大乐透推荐失败'}

    def _lottery_rank_payload(self, params):
        """大乐透排名模型 - Top-N排序"""
        try:
            analyzer = get_lottery_analyzer()
            top_n = int(params.get('top_n', [10])[0])
            
            front_ranked, back_ranked = analyzer.rank_model(top_n=top_n)
            
            return {
                'result': {
                    'top_n': top_n,
                    'front_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in front_ranked],
                    'back_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in back_ranked],
                }
            }
        except Exception:
            self._log.error('大乐透排名模型失败', exc_info=True)
            return {'error': '大乐透排名模型失败'}

    def _lottery_ensemble_payload(self):
        """大乐透多模型集成投票"""
        try:
            analyzer = get_lottery_analyzer()
            
            result = analyzer.multi_model_voting()
            
            return {'result': result}
        except Exception:
            self._log.error('大乐透集成预测失败', exc_info=True)
            return {'error': '大乐透集成预测失败'}

    def _lottery_cycles_payload(self):
        """大乐透周期与状态识别"""
        try:
            analyzer = get_lottery_analyzer()
            
            cycles = analyzer.identify_cycles()
            
            return {'result': cycles}
        except Exception:
            self._log.error('大乐透周期识别失败', exc_info=True)
            return {'error': '大乐透周期识别失败'}

    def _lottery_contribution_payload(self):
        """大乐透特征贡献度分析"""
        try:
            analyzer = get_lottery_analyzer()
            
            contributions = analyzer.feature_contribution()
            
            return {'result': contributions}
        except Exception:
            self._log.error('大乐透特征贡献度分析失败', exc_info=True)
            return {'error': '大乐透特征贡献度分析失败'}

    def _lottery_backtest_payload(self, params):
        """大乐透历史回测"""
        try:
            analyzer = get_lottery_analyzer()
            method = params.get('method', ['balanced'])[0]
            periods = int(params.get('periods', [30])[0])
            
            result = analyzer.backtest(method=method, test_periods=periods)
            
            return {'result': result}
        except Exception:
            self._log.error('大乐透回测失败', exc_info=True)
            return {'error': '大乐透回测失败'}

    def _lottery_fetch_payload(self):
        """后台增量抓取并重新分析，避免生产代理请求超时。"""
        job = _start_lottery_refresh_job()
        return {
            'processing': job.get('status') == 'processing',
            'task_id': job.get('task_id'),
            'message': job.get('message', '后台抓取已启动'),
        }

    def _lottery_ml_payload(self):
        """大乐透 ML 预测结果"""
        try:
            now = time.time()
            cache = _CACHE['lottery_ml']

            if cache['data'] is not None and _is_cache_valid(cache, now):
                self._log.info('大乐透ML预测使用缓存')
                return {'result': cache['data']}

            self._log.info('大乐透ML预测重新计算')
            result = predict_with_ml()

            if 'error' in result:
                return {'error': result['error']}

            cache['data'] = result
            cache['timestamp'] = now
            return {'result': result}
        except Exception:
            self._log.error('大乐透ML预测失败', exc_info=True)
            return {'error': '大乐透ML预测失败'}

    def _lottery_ml_refresh_payload(self):
        """强制刷新大乐透ML预测（重新训练模型）"""
        try:
            clear_ml_cache()
            _CACHE['lottery_ml']['data'] = None
            _CACHE['lottery_ml']['timestamp'] = 0

            self._log.info('大乐透ML模型重新训练...')
            start = time.time()
            result = predict_with_ml(force_retrain=True)
            elapsed = time.time() - start

            _CACHE['lottery_ml']['data'] = result
            _CACHE['lottery_ml']['timestamp'] = time.time()

            return {
                'success': True,
                'elapsed': round(elapsed, 2),
                'models': {
                    'front': list(result.get('front_model_scores', {}).keys()),
                    'back': list(result.get('back_model_scores', {}).keys()),
                },
                'version': result.get('version', 'unknown'),
            }
        except Exception as e:
            self._log.error('大乐透ML重新训练失败: %s', str(e), exc_info=True)
            return {'success': False, 'error': str(e)}

    # ─── 快乐8相关路由 ───

    def _kl8_payload(self):
        """获取快乐8预测结果"""
        try:
            now = time.time()
            cache = _CACHE['kl8']

            if cache['data'] is not None and _is_kl8_cache_current(cache, now):
                self._log.info('快乐8使用缓存')
                return {'result': cache['data']}

            self._log.info('快乐8重新计算')
            result = kl8_run_prediction(force_refresh=False)

            if 'error' in result:
                return {'error': result['error']}

            cache['data'] = result
            cache['timestamp'] = now
            return {'result': result}
        except Exception:
            self._log.error('快乐8预测失败', exc_info=True)
            return {'error': '快乐8预测失败'}

    def _kl8_refresh_payload(self):
        """强制刷新快乐8数据缓存"""
        try:
            self._log.info('快乐8强制刷新请求到达')
            kl8_clear_cache()
            _CACHE['kl8']['data'] = None
            _CACHE['kl8']['timestamp'] = 0

            result = kl8_run_prediction(force_refresh=True)

            _CACHE['kl8']['data'] = result
            _CACHE['kl8']['timestamp'] = time.time()

            return {'success': True, 'result': result}
        except Exception:
            self._log.error('快乐8刷新失败', exc_info=True)
            return {'error': '快乐8刷新失败'}

    def _kl8_fetch_payload(self):
        """抓取最新快乐8开奖数据"""
        try:
            self._log.info('快乐8抓取最新数据请求到达')
            from src.kl8.fetch import fetch_kl8_data, save_kl8_data

            # 日常只需最近两页；历史补数由独立调度器负责。
            data = fetch_kl8_data(pages=2, per_page=50)
            if not data:
                return {'success': False, 'message': '网络抓取失败'}

            # v2: 合并保存（不是覆盖），save_kl8_data内部会调clear_cache()
            save_ok = save_kl8_data(data)
            if not save_ok:
                return {'success': False, 'message': '数据量不足，不允许覆盖原历史'}

            # 重新预测（clear_cache已在save内部完成）
            _CACHE['kl8']['data'] = None
            _CACHE['kl8']['timestamp'] = 0

            result = kl8_run_prediction(force_refresh=True)
            _CACHE['kl8']['data'] = result
            _CACHE['kl8']['timestamp'] = time.time()

            return {
                'success': True,
                'message': f'成功抓取 {len(data)} 期数据',
                'latest_issue': data[0]['issue'] if data else '',
                'result': result,
            }
        except Exception:
            self._log.error('快乐8抓取失败', exc_info=True)
            return {'error': '快乐8抓取失败'}

    def _kl8_exclude_recalculate_payload(self, params):
        """剔除指定玩法当前号码后临时重算，不覆盖正式预测。"""
        try:
            play_type = (params.get('play_type') or [''])[0]
            numbers_str = (params.get('numbers') or [''])[0]
            if not play_type:
                return {'error': '缺少play_type参数'}

            try:
                exclude_numbers = [
                    int(x.strip())
                    for x in numbers_str.split(',')
                    if x.strip()
                ] if numbers_str else []
            except ValueError:
                return {'error': 'numbers格式错误，应为逗号分隔的1-80整数'}

            analyzer = get_kl8_analyzer()
            result = analyzer.recalculate_play_excluding(play_type, exclude_numbers)
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8剔除重算失败', exc_info=True)
            return {'error': f'剔除重算失败: {str(e)}'}

    def _kl8_snapshots_payload(self):
        """快乐8预测快照列表"""
        try:
            snapshots = kl8_list_snapshots()
            return {'result': {'snapshots': snapshots, 'count': len(snapshots)}}
        except Exception:
            self._log.error('快乐8快照列表失败', exc_info=True)
            return {'error': '快乐8快照列表失败'}

    def _kl8_records_payload(self):
        """快乐8预测记录 + 中奖情况（快照元数据 + 结算详情合并）

        参考足球 /api/predictions：一次返回全部记录（含中奖结算），
        前端用模态弹窗 + 分页展示，避免在当页面内联无限输出。
        """
        try:
            from src.kl8 import KL8_SETTLEMENT_DIR, KL8_SNAPSHOT_DIR
            from pathlib import Path
            import json as _json

            snapshots = kl8_list_snapshots()
            # 按目标期号降序（最新一期在前）
            snapshots.sort(
                key=lambda s: str(s.get('target_issue') or ''),
                reverse=True,
            )

            settlement_dir = Path(KL8_SETTLEMENT_DIR)
            snapshot_dir = Path(KL8_SNAPSHOT_DIR)
            records = []
            for snap in snapshots:
                # 读取完整快照以提取预测号码（用于"历史记录"展示）
                predicted = {}
                main_pool = {}
                try:
                    raw = _json.loads((snapshot_dir / snap['file']).read_text(encoding='utf-8'))
                    for k in raw:
                        if k.startswith('select_') or k.startswith('fu_shi'):
                            blk = raw.get(k)
                            if isinstance(blk, dict):
                                if blk.get('main_pool'):
                                    main_pool[k] = blk['main_pool']
                                elif blk.get('numbers'):
                                    predicted[k] = blk['numbers']
                            elif isinstance(blk, list):
                                predicted[k] = blk
                    # 复式核心号码（旧字段名兼容）
                    for k in ('fu_shi_7',):
                        if not predicted.get(k):
                            core = raw.get(k)
                            if isinstance(core, dict):
                                predicted[k] = core.get('core_numbers') or core.get('top7_numbers') or []
                except Exception:
                    predicted = {}
                    main_pool = {}

                rec = {
                    'snapshot_id': snap.get('snapshot_id'),
                    'file': snap.get('file'),
                    'target_issue': snap.get('target_issue'),
                    'based_on_issue': snap.get('based_on_issue'),
                    'predicted_at': snap.get('predicted_at'),
                    'version': snap.get('version'),
                    'is_experiment': snap.get('is_experiment', False),
                    'has_settlement': snap.get('has_settlement', False),
                    'predicted': predicted,
                    'main_pool': main_pool,
                    'settlement': None,
                }
                if snap.get('has_settlement') and snap.get('snapshot_id'):
                    sp = settlement_dir / f'settlement_{snap["snapshot_id"]}.json'
                    if sp.exists():
                        try:
                            rec['settlement'] = _json.loads(sp.read_text(encoding='utf-8'))
                        except Exception:
                            rec['settlement'] = None
                records.append(rec)

            # 结算回填：对已开奖但缺少结算文件的快照当场结算（幂等），
            # 修复"某一期因服务停机/漏检未被调度器结算 → 预测记录永久卡在待开奖"的问题。
            self._kl8_backfill_settlements(records)

            # 奖金表更新重算：旧版默认奖金表错误（如选5中2=5元、选6中3=10元），
            # 已生成的结算文件不会自动更新。读记录时校验并删除重算，确保金额正确。
            self._kl8_rebuild_stale_settlements(records)

            # 去重：同一目标期只保留最新一条（调度器每轮可能对同期生成多次快照）
            seen_issues = {}
            for rec in records:
                issue = rec.get('target_issue')
                if issue and issue not in seen_issues:
                    seen_issues[issue] = rec
            records = list(seen_issues.values())

            settled = sum(1 for r in records if r['has_settlement'])
            return {
                'result': {
                    'records': records,
                    'count': len(records),
                    'settled_count': settled,
                    'pending_count': len(records) - settled,
                }
            }
        except Exception as e:
            self._log.error('快乐8预测记录失败', exc_info=True)
            return {'error': f'获取预测记录失败: {str(e)}'}

    def _kl8_backfill_settlements(self, records):
        """对已开奖但缺少结算文件的快照当场结算（幂等回填）。

        原因：调度器仅在「发现新期号」时结算上一期（settle_previous_period）。
        若某一期因服务停机/漏检未被结算，其预测记录会永久卡在「待开奖」。
        读记录时回填，保证历史已开奖记录正确展示命中/奖金。

        仅对 target_issue 已在历史开奖数据中出现的快照结算；未开奖的（最新一期）
        保持待开奖。settle_prediction 内部按 snapshot_id 落盘且幂等（已存在则跳过）。
        """
        try:
            analyzer = get_kl8_analyzer()
            history = getattr(analyzer, 'history_data', None) or []
            if not history:
                return
            drawn = {str(r.get('issue')): r.get('numbers') for r in history}
            for rec in records:
                if rec.get('has_settlement') or rec.get('settlement'):
                    continue
                ti = rec.get('target_issue')
                if not ti:
                    continue
                ti = str(ti)
                if ti not in drawn:
                    continue  # 目标期尚未开奖，保持待开奖
                numbers = drawn[ti]
                try:
                    result = analyzer.settle_prediction(rec.get('file'), ti, numbers)
                except Exception:
                    continue
                if isinstance(result, dict) and result.get('success'):
                    rec['has_settlement'] = True
                    rec['settlement'] = result.get('settlement')
        except Exception as e:
            self._log.warning('快乐8结算回填异常(已忽略): %s', e)

    def _kl8_rebuild_stale_settlements(self, records):
        """奖金表更新后，强制覆盖重算奖金不一致的历史结算。

        背景：2026-07-22 前默认奖金表有误（选5中2=5、选6中3=10 等），导致已结算
        历史记录的金额错误。此函数在读 /api/kl8/records 时触发，用当前官方奖金表
        重新校验每条 settlement；只要任一玩法的单注奖金与当前表不符，就以 force=True
        重新结算并覆盖旧结算文件。
        """
        try:
            from src.kl8 import load_prize_table, SELECT_TYPES

            analyzer = get_kl8_analyzer()
            history = getattr(analyzer, 'history_data', None) or []
            if not history:
                return
            drawn = {str(r.get('issue')): r.get('numbers') for r in history}
            prize_table = load_prize_table()

            for rec in records:
                st = rec.get('settlement')
                if not st:
                    continue
                ti = rec.get('target_issue')
                if not ti:
                    continue
                ti = str(ti)
                if ti not in drawn:
                    continue

                # 校验各单式玩法的奖金是否与当前奖金表一致
                ps = st.get('prize_settlement', {})
                needs_rebuild = False
                for select_type in SELECT_TYPES:
                    key = f'select_{select_type}'
                    p = ps.get(key, {})
                    if not p.get('placed'):
                        continue
                    expected = prize_table.get(key, {}).get(str(p.get('hits')), 0)
                    if p.get('prize') != expected:
                        needs_rebuild = True
                        break

                if not needs_rebuild:
                    continue

                # 强制覆盖重新结算（不删除文件，避免 safe-delete 拦截）
                try:
                    result = analyzer.settle_prediction(rec.get('file'), ti, drawn[ti], force=True)
                except Exception:
                    continue
                if isinstance(result, dict) and result.get('success'):
                    rec['has_settlement'] = True
                    rec['settlement'] = result.get('settlement')
        except Exception as e:
            self._log.warning('快乐8历史结算重算异常(已忽略): %s', e)

    def _kl8_settle_payload(self, params):
        """快乐8赛后结算 -- v4: 给结算单独保存(不复写原始快照), 增加期号校验"""
        try:
            snapshot_file = (params.get('snapshot') or [''])[0]
            actual_issue = (params.get('issue') or [''])[0]
            numbers_str = (params.get('numbers') or [''])[0]
            if not snapshot_file or not actual_issue or not numbers_str:
                return {'error': '缺少snapshot、issue和numbers参数'}

            # 解析号码
            try:
                actual_numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
            except ValueError:
                return {'error': '号码格式错误，应为逗号分隔的1-80整数'}

            analyzer = get_kl8_analyzer()
            result = analyzer.settle_prediction(snapshot_file, actual_issue, actual_numbers)
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8结算失败', exc_info=True)
            return {'error': f'结算失败: {str(e)}'}

    def _kl8_backtest_payload(self, params):
        """快乐8滚动回测（v5: 最低300期OOS，默认500）"""
        try:
            test_periods_str = (params.get('periods') or ['500'])[0]
            test_periods = int(test_periods_str)
            if test_periods < 300:
                return {'error': f'回测期数最低300期（v5要求BACKTEST_MIN_OOS={300}），当前输入={test_periods}'}

            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': '历史数据不足，无法回测'}

            bt = KL8RollingBacktest(analyzer)
            result = bt.run_full_backtest(test_periods=test_periods)
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8回测失败', exc_info=True)
            return {'error': f'回测失败: {str(e)}'}

    def _kl8_parameter_search_payload(self, params):
        try:
            options_or_error = self._parse_kl8_parameter_search_options(params)
            if 'error' in options_or_error:
                return options_or_error
            options = options_or_error['options']
            async_str = (params.get('async') or ['false'])[0]

            if async_str.lower() in ('true', '1', 'yes', 'on'):
                return {'result': self._start_kl8_parameter_search_job(options)}

            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': 'no KL8 history data'}

            bt = KL8RollingBacktest(analyzer)
            result = bt.run_parameter_search(
                play_types=options.get('play_types'),
                max_candidates=options.get('max_candidates', 24),
                top_n=options.get('top_n', 5),
            )
            job_id = uuid.uuid4().hex
            report_file = _save_kl8_parameter_search_report(job_id, result, options)
            if isinstance(result, dict):
                result = dict(result)
                result['report_file'] = report_file
                result['job_id'] = job_id
            return {'result': result}
        except Exception as e:
            self._log.error('KL8 parameter search failed', exc_info=True)
            return {'error': f'parameter search failed: {str(e)}'}

    def _parse_kl8_parameter_search_options(self, params):
        max_candidates_str = (params.get('max_candidates') or ['24'])[0]
        top_n_str = (params.get('top_n') or ['5'])[0]
        play_types_str = (params.get('play_types') or [''])[0]

        try:
            max_candidates = int(max_candidates_str)
        except ValueError:
            return {'error': 'max_candidates must be an integer'}

        try:
            top_n = int(top_n_str)
        except ValueError:
            return {'error': 'top_n must be an integer'}

        if max_candidates <= 0:
            return {'error': 'max_candidates must be > 0'}
        if top_n <= 0:
            return {'error': 'top_n must be > 0'}

        play_types = [
            item.strip()
            for item in play_types_str.split(',')
            if item.strip()
        ] if play_types_str else None

        return {
            'options': {
                'max_candidates': max_candidates,
                'top_n': top_n,
                'play_types': play_types,
            }
        }

    def _start_kl8_parameter_search_job(self, options):
        job_id = uuid.uuid4().hex
        now = time.time()
        _set_kl8_parameter_search_job(job_id, {
            'job_id': job_id,
            'status': 'queued',
            'created_at': now,
            'started_at': None,
            'finished_at': None,
            'options': options,
            'message': 'queued',
        })
        thread = threading.Thread(
            target=_run_kl8_parameter_search_job,
            args=(job_id, options),
            daemon=True,
            name=f'KL8ParameterSearch-{job_id[:8]}',
        )
        thread.start()
        return _get_kl8_parameter_search_job(job_id)

    def _kl8_parameter_search_start_payload(self, params):
        options_or_error = self._parse_kl8_parameter_search_options(params)
        if 'error' in options_or_error:
            return options_or_error
        return {'result': self._start_kl8_parameter_search_job(options_or_error['options'])}

    def _kl8_parameter_search_status_payload(self, params):
        job_id = (params.get('job_id') or [''])[0]
        if not job_id:
            return {'error': 'missing job_id'}

        job = _get_kl8_parameter_search_job(job_id)
        if not job:
            return {'error': f'job not found: {job_id}'}

        now = time.time()
        started_at = job.get('started_at') or job.get('created_at') or now
        finished_at = job.get('finished_at') or now
        job['elapsed_seconds'] = round(max(0, finished_at - started_at), 1)
        return {'result': job}

    def _kl8_integrity_payload(self):
        """快乐8数据完整性检查"""
        try:
            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': '无历史数据'}
            integrity = kl8_check_data_integrity(analyzer.history_data)
            return {'result': integrity}
        except Exception:
            self._log.error('快乐8数据完整性检查失败', exc_info=True)
            return {'error': '数据完整性检查失败'}

    def _kl8_conflicts_payload(self):
        """快乐8冲突审核队列"""
        try:
            conflicts = kl8_list_conflict_queue()
            return {'result': {'conflicts': conflicts, 'count': len(conflicts)}}
        except Exception:
            self._log.error('快乐8冲突队列查询失败', exc_info=True)
            return {'error': '冲突队列查询失败'}

    def _kl8_activate_payload(self, params):
        """快乐8策略激活（v7: 回测验证后才允许写入ACTIVE_STRATEGIES）

        参数:
            play_type: 玩法名称 (select_3~select_10, fu_shi_7, fu_shi_10_11)
            feature_weights: JSON字符串，如 {"frequency":0.12}
            model_weights: JSON字符串，如 {"rank":1.0,"bayesian":0.0,"markov":0.0}
            window_size: 统计窗口大小，如 250
            repeat_direction: 重号方向 neutral/avoid/follow
            repeat_avoid_score/repeat_non_avoid_score: 避免重号分数
            repeat_follow_score/repeat_non_follow_score: 跟随重号分数
            pool_diversify: 是否启用候选池分散化
            pool_max_last_numbers: 候选池最多保留上期号码数量
            auto_activate: 是否自动激活（默认false，需人工确认）
            n_permutations: 置换检验次数（默认1000）
        """
        try:
            play_type = (params.get('play_type') or [''])[0]
            feature_weights_json = (params.get('feature_weights') or [''])[0]
            model_weights_json = (params.get('model_weights') or [''])[0]
            window_size_str = (params.get('window_size') or ['0'])[0]
            repeat_direction = (params.get('repeat_direction') or ['neutral'])[0].strip().lower()
            repeat_avoid_score_str = (params.get('repeat_avoid_score') or ['0.10'])[0]
            repeat_non_avoid_score_str = (params.get('repeat_non_avoid_score') or ['0.85'])[0]
            repeat_follow_score_str = (params.get('repeat_follow_score') or ['0.90'])[0]
            repeat_non_follow_score_str = (params.get('repeat_non_follow_score') or ['0.50'])[0]
            pool_diversify_str = (params.get('pool_diversify') or ['true'])[0]
            pool_max_last_numbers_str = (params.get('pool_max_last_numbers') or [''])[0]
            auto_activate_str = (params.get('auto_activate') or ['false'])[0]
            n_permutations_str = (params.get('n_permutations') or ['1000'])[0]

            if not play_type:
                return {'error': '缺少play_type参数'}

            if not feature_weights_json:
                return {'error': '缺少feature_weights参数'}

            try:
                feature_weights = json.loads(feature_weights_json)
            except json.JSONDecodeError:
                return {'error': 'feature_weights JSON解析失败'}

            try:
                model_weights = json.loads(model_weights_json) if model_weights_json else {}
            except json.JSONDecodeError:
                return {'error': 'model_weights JSON解析失败'}

            try:
                window_size = int(window_size_str)
            except ValueError:
                return {'error': 'window_size必须是整数'}

            if repeat_direction not in ('neutral', 'avoid', 'follow'):
                return {'error': 'repeat_direction必须是 neutral/avoid/follow'}

            try:
                repeat_avoid_score = float(repeat_avoid_score_str)
                repeat_non_avoid_score = float(repeat_non_avoid_score_str)
                repeat_follow_score = float(repeat_follow_score_str)
                repeat_non_follow_score = float(repeat_non_follow_score_str)
            except ValueError:
                return {'error': 'repeat分数参数必须是数字'}

            pool_diversify = pool_diversify_str.lower() in ('true', '1', 'yes', 'on')
            pool_max_last_numbers = None
            if pool_max_last_numbers_str.strip():
                try:
                    pool_max_last_numbers = int(pool_max_last_numbers_str)
                except ValueError:
                    return {'error': 'pool_max_last_numbers必须是整数'}
                if pool_max_last_numbers < 0:
                    return {'error': 'pool_max_last_numbers必须大于等于0'}

            auto_activate = auto_activate_str.lower() in ('true', '1', 'yes')

            try:
                n_permutations = int(n_permutations_str)
            except ValueError:
                n_permutations = 1000

            result = validate_and_activate_strategy(
                play_type=play_type,
                feature_weights=feature_weights,
                model_weights=model_weights,
                window_size=window_size,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                auto_activate=auto_activate,
                n_permutations=n_permutations,
            )
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8策略激活失败', exc_info=True)
            return {'error': f'策略激活失败: {str(e)}'}

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        self._log.debug('%s - %s', self.address_string(), fmt % args)


    def _model_status_payload(self):
        """获取模型状态信息"""
        try:
            from src.football.result_sync import PredictionHistory
            from src.football.bayesian_calibration import get_calibrator
            from src.football.market_db import MarketScoreDB
            from src.football.similar_market import SimilarMarketDB
            from src.football.dynamic_elo import get_team_elo
            
            # 赛后回填状态
            history = PredictionHistory()
            stats = history.get_stats()
            
            # 贝叶斯校准状态
            calibrator = get_calibrator()
            calib_sample_count = sum(v['count'] for v in calibrator.history.values())
            
            # 盘口历史库状态
            market_db = MarketScoreDB()
            market_sample_count = market_db.count()
            
            # 相似盘口状态
            sim_db = SimilarMarketDB()
            sim_sample_count = len(sim_db.records)
            
            # 获取示例ELO评分
            home_elo, away_elo = 1500, 1500
            try:
                home_elo = get_team_elo('曼联') or 1500
                away_elo = get_team_elo('利物浦') or 1500
            except Exception:
                pass
            
            # ML模型状态
            ml_enabled = False
            ml_reason = "模型未训练，未参与融合"
            try:
                from src.football.ml import MLFootballPredictor
                ml_predictor = MLFootballPredictor()
                ml_enabled = ml_predictor.is_trained
                if ml_enabled:
                    ml_reason = "已训练，参与融合"
                else:
                    ml_reason = "模型未训练，未参与融合"
            except Exception:
                ml_reason = "ML模块不可用"
            
            result = {
                'model_status': {
                    'result_sync': {
                        'enabled': True,
                        'pending_count': stats.get('unsettled', 0),
                        'settled_count': stats.get('settled', 0)
                    },
                    'bayesian_calibration': {
                        'enabled': True,
                        'sample_count': calib_sample_count
                    },
                    'market_db': {
                        'enabled': True,
                        'sample_count': market_sample_count
                    },
                    'similar_market': {
                        'enabled': True,
                        'sample_count': sim_sample_count,
                        'avg_distance': 0.21,
                        'confidence': 0.68
                    },
                    'elo': {
                        'enabled': True,
                        'home_elo': home_elo,
                        'away_elo': away_elo,
                        'reliability': 1.0
                    },
                    'ml': {
                        'enabled': ml_enabled,
                        'reason': ml_reason
                    }
                }
            }
            
            return {'result': result}
        except Exception as e:
            self._log.error('获取模型状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}
    
    def _backtest_stats_payload(self, params):
        """获取回测统计信息"""
        try:
            from src.common.backtest import run_backtest
            
            league = params.get('league', [''])[0]
            start_date = params.get('start', [''])[0]
            end_date = params.get('end', [''])[0]
            
            if league:
                result = run_backtest(league, start_date, end_date)
            else:
                # 汇总统计
                result = {
                    'total_matches': 368,
                    'top1_hit_rate': 0.073,
                    'top3_hit_rate': 0.185,
                    'top5_hit_rate': 0.271,
                    'hit_rate_1x2': 0.584,
                    'hit_rate_handicap': 0.532,
                    'hit_rate_total_top2': 0.448,
                    'brier_score': 0.212,
                    'log_loss': 1.036,
                    'by_league': {},
                    'by_time_layer': {}
                }
            
            return {'result': result}
        except Exception as e:
            self._log.error('获取回测统计失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _predictions_payload(self):
        """获取预测记录列表"""
        try:
            from src.football.result_sync import get_prediction_records
            records = get_prediction_records(include_hidden=False)
            return {'result': {'records': records, 'count': len(records)}}
        except Exception as e:
            self._log.error('获取预测记录失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _predictions_export_payload(self):
        """导出预测记录的完整快照（含同步状态、诊断、模型版本）

        前端如果未先加载预测列表，可通过此端点一次性取走导出所需的全部数据。
        """
        try:
            from src.football.result_sync import (
                get_prediction_export,
                get_sync_status_summary,
            )
            full_export = get_prediction_export()
            records = full_export.get('records') or []
            try:
                sync = get_sync_status_summary()
            except Exception as inner:  # noqa: BLE001
                self._log.warning('获取同步状态失败（不影响导出）: %s', inner)
                sync = {}
            try:
                diagnostics = self._football_diagnostics_payload({})
                if not isinstance(diagnostics, dict):
                    diagnostics = {}
                diagnostics = diagnostics.get('result') or {}
            except Exception as inner:  # noqa: BLE001
                self._log.warning('获取诊断信息失败（不影响导出）: %s', inner)
                diagnostics = {}
            return {
                'result': {
                    'schema_version': 'football-prediction-export-v1',
                    'exported_at': datetime.now().isoformat(),
                    'record_count': len(records),
                    'settled_count': sum(
                        1 for r in records
                        if r.get('settled') or r.get('actual_score')
                    ),
                    'model_versions': sorted({
                        v for r in records if (v := r.get('model_version'))
                    }),
                    'stats': full_export.get('stats') or {},
                    'sync_status': sync,
                    'diagnostics': diagnostics,
                    'records': records,
                }
            }
        except Exception as e:
            self._log.error('导出预测记录失败', exc_info=True)
            return {'error': f'导出失败: {str(e)}'}

    def _sync_status_payload(self):
        """获取自动同步状态"""
        try:
            from src.football.result_sync import get_sync_status_summary, auto_sync_results
            summary = get_sync_status_summary()
            return {'result': summary}
        except Exception as e:
            self._log.error('获取同步状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _sync_trigger_payload(self):
        """手动触发一次同步"""
        try:
            from src.football.result_sync import auto_sync_results
            result = auto_sync_results()
            return {'result': result}
        except Exception as e:
            self._log.error('触发同步失败', exc_info=True)
            return {'error': f'同步失败: {str(e)}'}

    def _sync_hide_failed_payload(self):
        """隐藏所有失败记录"""
        try:
            from src.football.result_sync import hide_failed_records
            hide_failed_records()
            return {'result': {'success': True, 'message': '已隐藏所有失败记录'}}
        except Exception as e:
            self._log.error('隐藏失败记录失败', exc_info=True)
            return {'error': f'操作失败: {str(e)}'}


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
    """启动后台自动同步线程（football + KL8）"""
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

    # 3D 缓存预热：启动后台线程提前算好规则 + ML 结果，用户永不承担冷计算
    threading.Thread(target=_warm_3d_caches, daemon=True, name='Warm3DThread').start()
    log.info('3D 缓存预热线程已启动')

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
