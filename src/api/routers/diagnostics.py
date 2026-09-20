# -*- coding: utf-8 -*-
"""进程内存构成诊断。

RSS 高既可能是某个常驻结构大，也可能只是堆碎片——Python 的高水位不回落，
glibc 释放的内存未必还给内核，而这两种情况的处置方向相反（削数据结构
vs 调分配器）。所以这里并列报告「具名结构的活对象字节」和「RSS 减去它们
之后的余额」，余额既不叫泄漏，也不摊到任何一个结构头上。
"""
import gc
import sys
import types

from fastapi import APIRouter, Query, Request

from src.api.deps import run_blocking

router = APIRouter(tags=['diagnostics'])

#: 这些对象是全局图谱的入口，跟进去会把整个解释器都算成某条记录的体积。
_UNTRAVERSABLE = (type, types.ModuleType, types.FunctionType,
                  types.BuiltinFunctionType)

_STATUS_FIELDS = {'VmRSS': 'rss_bytes', 'VmSwap': 'swap_bytes'}

_NOTES = (
    'unaccounted_bytes = RSS - tracked_total_bytes，含解释器与已导入库的常驻、'
    '堆碎片，以及没被探针覆盖的对象；它不等于泄漏。',
    'gc 直方图只包含 GC 跟踪的容器对象，str/int/float 等原子对象不在其中，'
    '且按浅层计算，仅用于发现数量级异常，不能当作字节数依据。',
)


def _own_size(obj):
    """单个对象自身的字节数；numpy 数组按 nbytes 计，否则退回 getsizeof。

    **不能用 getattr 探测 nbytes**：堆里有惰性模块代理（`six.MovedModule`
    就是一个），任意实例属性访问都会触发 import 或别的副作用，而诊断只是
    读一眼内存。所以 numpy 按类型判定——那只读 type 的属性，不碰实例。
    `__sizeof__` 同样可能是第三方实现，异常按 0 计，不让一个古怪对象
    把整个端点带成 500。
    """
    try:
        cls = type(obj)
        if cls.__name__ == 'ndarray' and cls.__module__.startswith('numpy'):
            return int(obj.nbytes)
        return sys.getsizeof(obj, 0)
    except Exception:
        return 0


def _deep_size(root):
    """沿引用图累加可达对象的字节数，按 id 去重，跳过全局图谱入口。"""
    seen = set()
    pending = [root]
    total = 0
    while pending:
        obj = pending.pop()
        if isinstance(obj, _UNTRAVERSABLE):
            continue
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        total += _own_size(obj)
        pending.extend(gc.get_referents(obj))
    return total


def _process_status():
    try:
        with open('/proc/self/status', encoding='utf-8') as handle:
            raw = handle.read()
    except OSError:
        return {'rss_bytes': None, 'swap_bytes': None, 'threads': None,
                'note': '本机没有 /proc（如 macOS），进程级数字不可得'}

    status = {'rss_bytes': None, 'swap_bytes': None, 'threads': None}
    for line in raw.splitlines():
        name, _, value = line.partition(':')
        if name in _STATUS_FIELDS:
            status[_STATUS_FIELDS[name]] = int(value.split()[0]) * 1024
        elif name == 'Threads':
            status['threads'] = int(value.strip())
    return status


def _prediction_history_section():
    """预测历史按骨架与时间线分开计——时间线是可卸载的那一半。"""
    from src.football import result_sync

    history = result_sync._global_history
    with history._records_lock:
        records = history.records
        total = _deep_size(records)
        timeline_bytes = _deep_size([r.get('market_timeline') for r in records])
        offloaded = sum(1 for r in records
                        if r.get(result_sync.TIMELINE_OFFLOADED))
        count = len(records)
    return {
        'records': count,
        'offloaded_records': offloaded,
        'bytes': total,
        'timeline_bytes': timeline_bytes,
        'skeleton_bytes': total - timeline_bytes,
    }


def _kl8_trials_section():
    from src.kl8 import config as kl8_config

    trials = kl8_config.STRATEGY_TRIAL_RESULTS
    return {'entries': len(trials), 'bytes': _deep_size(trials)}


def _ml_model_section():
    from src.football import ml

    model = ml._trained_ml_model
    return {
        'loaded': model is not None,
        'bytes': _deep_size(model) if model is not None else 0,
        'note': 'LightGBM/XGBoost/CatBoost 的权重在 C 层，getsizeof 看不到，'
                '这里只是 Python 壳的大小',
    }


def _cache_l1_section(state):
    entries = getattr(getattr(getattr(state, 'cache', None), 'l1', None),
                      '_data', None)
    if entries is None:
        return {'available': False}
    return {'entries': len(entries), 'bytes': _deep_size(entries)}


def _section(probe):
    try:
        return probe()
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}


def _gc_histogram(limit=20):
    totals = {}
    counts = {}
    for obj in gc.get_objects():
        name = type(obj).__name__
        totals[name] = totals.get(name, 0) + _own_size(obj)
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{'type': name, 'bytes': size, 'count': counts[name]}
            for name, size in ranked[:limit]]


def memory_payload(state, with_gc=False):
    tracked = {
        'prediction_history': _section(_prediction_history_section),
        'kl8_strategy_trials': _section(_kl8_trials_section),
        'ml_model': _section(_ml_model_section),
        'cache_l1': _section(lambda: _cache_l1_section(state)),
    }
    tracked_total = sum(section.get('bytes', 0) for section in tracked.values())
    process = _process_status()
    rss = process.get('rss_bytes')
    return {
        'process': process,
        'tracked': tracked,
        'tracked_total_bytes': tracked_total,
        'unaccounted_bytes': None if rss is None else rss - tracked_total,
        'gc_histogram': _section(_gc_histogram) if with_gc else None,
        'notes': list(_NOTES),
    }


@router.get('/api/diagnostics/memory')
async def memory(request: Request,
                 with_gc: bool = Query(False, alias='gc')):
    return await run_blocking(memory_payload, request.app.state, with_gc)
