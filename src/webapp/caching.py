# -*- coding: utf-8 -*-
"""接口缓存框架：内存缓存、磁盘持久化、后台刷新"""

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
    _json_default,
)
from .lazy_modules import (
    KL8_PREDICTOR_VERSION, data_path, fetch_data, get_kl8_analyzer, run_prediction,
)
from . import lazy_modules as _lazy_mod

def _current_kl8_predictor_version():
    """Read the KL8 version from the module so cache checks follow code reloads."""
    try:
        import src.kl8 as kl8_module
        return getattr(kl8_module, 'KL8_PREDICTOR_VERSION', KL8_PREDICTOR_VERSION)
    except Exception:
        return KL8_PREDICTOR_VERSION


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


def _is_same_day(timestamp):
    """检查时间戳是否属于今天"""
    from datetime import date
    return date.fromtimestamp(timestamp) == date.today()


def _is_cache_valid(cache_entry, now):
    """缓存有效条件：未超过 TTL 且未跨天"""
    elapsed = now - cache_entry['timestamp']
    return elapsed < cache_entry['expire_seconds'] and _is_same_day(cache_entry['timestamp'])


def _is_cache_payload_current(key, data):
    """Reject persisted predictions produced by an older code version."""
    if data is None:
        return False
    if key == '3d':
        return data.get('version') == _lazy_mod._get_lottery3d_module().PREDICTOR_VERSION
    if key == '3d_ml':
        return data.get('model_version') == _lazy_mod._get_lottery3d_module().ML_MODEL_VERSION
    if key == 'ssq':
        import src.ssq as _ssq
        return data.get('version') == _ssq.SSQ_PREDICTION_VERSION
    return True


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


_CACHE_LOCKS = {key: threading.Lock() for key in _CACHE}


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
            if (
                obj.get('data') is not None
                and _is_same_day(ts)
                and _is_cache_payload_current(key, obj['data'])
            ):
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
    if cache['data'] is not None and not _is_cache_payload_current(key, cache['data']):
        log.info('缓存 %s 版本已过期，强制重新计算', key)
        cache['data'] = None
        cache['timestamp'] = 0
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
    result = run_prediction(
        enable_backtest=False, compute_weights=False, train_ml_if_stale=False
    )
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
    current_period = data[-1][0] if data else None
    rule_module = _lazy_mod._get_lottery3d_module()
    ml_module = _lazy_mod._get_lottery3d_ml_module()
    persisted = ml_module.load_ml_cache()
    cache_reused = bool(
        current_period
        and rule_module.is_ml_prediction_cache_valid(persisted, current_period)
    )
    if cache_reused:
        result = persisted
    else:
        result = _lazy_mod.predict_current(numbers, top_k=15, model_type="ensemble")
        if not result.get('error') and current_period:
            result['base_period'] = current_period
            result['model_version'] = rule_module.ML_MODEL_VERSION
            result['created_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            ml_module.save_ml_cache(result)
    if 'error' in result:
        raise RuntimeError(result['error'])

    selected = list(result.get('recommendations', []))[:15]
    selected_total = sum(
        float(r.get('model_score', r.get('probability', 0)) or 0)
        for r in selected
    ) or 1.0

    def _ml_item(row):
        model_score = float(row.get('model_score', row.get('probability', 0)))
        share = model_score / selected_total
        return {
            'num': row['num'],
            'model_score': model_score,
            'topk_score_share': share,
            'relative_prob': share,
        }

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
        'recommendations': [_ml_item(r) for r in selected],
        'top3': [_ml_item(r) for r in selected[:3]],
        'rule_recommendations': [
            {'num': r['num'], 'score': float(r.get('score', 0))}
            for r in rule_recommendations
        ],
        'feature_importance': result.get('feature_importance', []),
        'cache_reused': cache_reused,
        'base_period': current_period,
        'model_version': rule_module.ML_MODEL_VERSION,
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

