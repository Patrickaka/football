# -*- coding: utf-8 -*-
"""快乐8预测入口 run_prediction、缓存、策略激活与快照列表"""

import math
import copy
import json
import os
import time
import threading
import uuid
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from pathlib import Path

from src.common.paths import data_path
from src.common.repositories import doc_store
from src.common.logger import setup_logger

log = setup_logger('kl8')
from . import config as _cfg

from .config import KL8_PREDICTOR_VERSION
from .records import (
    _checksum_numbers, _persist_active_strategies,
    _prediction_config_fingerprint, _strategy_fingerprint,
)
from .analyzer import (
    get_kl8_analyzer,
)


_PREDICTION_CACHE_FILE = Path(data_path('kl8_prediction_cache.json'))
_PREDICTION_CACHE_SCHEMA = 1
_prediction_cache_lock = threading.Lock()
_prediction_run_lock = threading.RLock()
_record_index_lock = threading.Lock()
_snapshot_index_cache = {'signature': None, 'records': []}
_recalculation_index_cache = {'signature': None, 'records': []}


def _directory_signature(directory: Path):
    """不可变记录目录的轻量版本号；新增/删除文件时目录 mtime 会变化。"""
    try:
        stat = directory.stat()
        return str(directory.resolve()), stat.st_mtime_ns
    except OSError:
        return str(directory), None


def _strategy_config_fingerprint() -> str:
    """返回会改变推荐号码的完整策略指纹。"""
    return _prediction_config_fingerprint()


def _history_signature():
    """用便宜的文件元数据判断开奖历史是否变化，不初始化完整分析器。"""
    path = Path(data_path('kl8_history.json'))
    try:
        stat = path.stat()
        return [stat.st_mtime_ns, stat.st_size]
    except OSError:
        return None


def _load_persisted_prediction(history_signature, config_fingerprint):
    """读取跨进程缓存；任一失效条件不同都视为未命中。"""
    if history_signature is None:
        return None
    try:
        payload = json.loads(_PREDICTION_CACHE_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('schema') != _PREDICTION_CACHE_SCHEMA:
        return None
    if payload.get('version') != KL8_PREDICTOR_VERSION:
        return None
    if payload.get('config_fingerprint') != config_fingerprint:
        return None
    if payload.get('history_signature') != history_signature:
        return None
    result = payload.get('result')
    if isinstance(result, dict) and 'error' not in result:
        result.setdefault('strategy_config_fingerprint', config_fingerprint)
        return result
    return None


def _persist_prediction(result, history_signature, config_fingerprint):
    """原子落盘完整预测结果，供服务重启后的首个请求直接读取。"""
    if history_signature is None or not isinstance(result, dict) or 'error' in result:
        return
    payload = {
        'schema': _PREDICTION_CACHE_SCHEMA,
        'version': KL8_PREDICTOR_VERSION,
        'config_fingerprint': config_fingerprint,
        'history_signature': history_signature,
        'stored_at': time.time(),
        'result': result,
    }
    path = _PREDICTION_CACHE_FILE
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        log.warning(f'快乐8: 预测磁盘缓存写入失败: {exc}')
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

def activate_verified_strategy(play_type: str, strategy: Dict, report: Dict):
    """统一激活已验证策略 — 所有验证流程通过后走此方法写入ACTIVE_STRATEGIES

    v9.2新增:
    - 不再在 validate_and_activate_strategy() 内直接写 _cfg.ACTIVE_STRATEGIES
    - 所有激活统一走此方法
    - 写入 status='validated', validated_at, validated_on_issue
    - 设置 degradation_status='normal'
    - 持久化 + 清缓存

    参数:
        play_type: 玩法名称
        strategy: 完整策略配置（含 feature_weights, model_weights, window_size 等）
        report: 验证报告（含 data_cutoff_issue 等元信息）
    """
    fingerprint = _strategy_fingerprint(strategy)

    # 策略替换、持久化与预测共用一把可重入锁。否则 predict_all 逐玩法解析
    # 策略时可能前半读取旧配置、后半读取新配置，最终生成一份混合版本结果。
    with _prediction_run_lock:
        _cfg.ACTIVE_STRATEGIES[play_type] = {
            **strategy,
            'strategy_id': f'{play_type}_{fingerprint}',
            'status': 'validated',
            'validated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'validated_on_issue': report.get('data_cutoff_issue', ''),
            'validation_report': report,
            'degradation_status': 'normal',
        }

        _persist_active_strategies()
        clear_cache()
        strategy_id = _cfg.ACTIVE_STRATEGIES[play_type]['strategy_id']
    log.info(f'快乐8: 策略已激活 {play_type} -> {strategy_id}')


def mark_strategy_degradation(
    play_type: str,
    expected_strategy_id: str,
    deviation: float,
    ci_lower: float,
) -> bool:
    """原子标记策略降级，并让后续预测立即使用新的策略状态。

    ``expected_strategy_id`` 防止评估完成到写入之间策略已被替换，误把新策略
    标成黄色观察。策略修改与预测、历史更新共用同一把锁，避免保存出混合配置
    的正式快照。
    """
    with _prediction_run_lock:
        strategy = _cfg.ACTIVE_STRATEGIES.get(play_type)
        if (
            not isinstance(strategy, dict)
            or str(strategy.get('strategy_id') or '')
            != str(expected_strategy_id or '')
        ):
            return False

        strategy['degradation_status'] = 'yellow_watch'
        strategy['degradation_deviation'] = round(deviation, 4)
        strategy['degradation_ci_lower'] = round(ci_lower, 4)
        _persist_active_strategies()
        clear_cache()
        return True


_prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def run_prediction(force_refresh: bool = False) -> Dict:
    """快乐8预测入口（v8: 缓存指纹包含所有策略配置）"""
    # scheduler、手动刷新和抓取接口最终都走这里。串行化完整计算，避免两个
    # 调用同时扫描快照目录后各自写成“正式”记录，也避免 clear_cache 在
    # 另一轮 predict_all 执行中途替换全局分析器。
    with _prediction_run_lock:
        return _run_prediction_locked(force_refresh)


def _run_prediction_locked(force_refresh: bool = False) -> Dict:
    config_fingerprint = _strategy_config_fingerprint()
    history_signature = _history_signature()
    fast_cache_key = (
        tuple(history_signature) if history_signature else None,
        KL8_PREDICTOR_VERSION,
        config_fingerprint,
    )

    if not force_refresh:
        with _prediction_cache_lock:
            cache = _prediction_cache
            if cache['data'] is not None and cache.get('cache_key') == fast_cache_key:
                return cache['data']
            persisted = _load_persisted_prediction(
                history_signature,
                config_fingerprint,
            )
            if persisted is not None:
                cache['data'] = persisted
                cache['cache_key'] = fast_cache_key
                cache['timestamp'] = time.time()
                return persisted

    analyzer = get_kl8_analyzer()

    if not force_refresh:
        if analyzer.reload_if_needed():
            force_refresh = True

    if not analyzer.history_data:
        return {
            'error': '历史数据不足',
            'using_simulated_data': True,
        }

    history_signature = _history_signature()
    cache_key = (
        tuple(history_signature) if history_signature else (
            analyzer.history_data[0]['issue'],
            _checksum_numbers(analyzer.history_data[0]['numbers']),
            len(analyzer.history_data),
        ),
        KL8_PREDICTOR_VERSION,
        config_fingerprint,
    )

    cache = _prediction_cache
    if not force_refresh and cache['data'] is not None and cache.get('cache_key') == cache_key:
        return cache['data']

    result = analyzer.predict_all()
    if isinstance(result, dict) and 'error' not in result:
        # API 的跨进程缓存也必须区分策略配置；仅靠代码版本无法覆盖同版本
        # 激活新策略的场景。
        result['strategy_config_fingerprint'] = config_fingerprint

    cache['data'] = result
    cache['cache_key'] = cache_key
    cache['timestamp'] = time.time()
    _persist_prediction(result, history_signature, config_fingerprint)

    return result


def clear_cache():
    global _prediction_cache
    with _prediction_run_lock:
        from . import analyzer as _analyzer_mod
        _analyzer_mod._analyzer_instance = None
        _prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}
        try:
            _PREDICTION_CACHE_FILE.unlink(missing_ok=True)
        except OSError as exc:
            log.warning(f'快乐8: 清理预测磁盘缓存失败: {exc}')


def list_prediction_snapshots() -> List[Dict]:
    snapshot_dir = Path(_cfg.KL8_SNAPSHOT_DIR)
    if not snapshot_dir.exists():
        return []

    settlement_dir = Path(_cfg.KL8_SETTLEMENT_DIR)
    signature = (
        _directory_signature(snapshot_dir),
        _directory_signature(settlement_dir),
    )
    with _record_index_lock:
        if _snapshot_index_cache['signature'] == signature:
            return copy.deepcopy(_snapshot_index_cache['records'])

    snapshots = []
    for f in sorted(snapshot_dir.glob('snapshot_*.json')):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            snapshots.append({
                'file': f.name,
                'snapshot_id': data.get('snapshot_id', ''),
                'target_issue': data.get('target_issue'),
                'based_on_issue': data.get('based_on_issue'),
                'predicted_at': data.get('predicted_at'),
                'predicted_at_ns': data.get('predicted_at_ns', 0),
                'version': data.get('version'),
                'strategy_config_fingerprint': data.get(
                    'strategy_config_fingerprint', ''
                ),
                'is_experiment': data.get('is_experiment', False),
                'strategy_fingerprint': data.get('strategy_fingerprint', ''),
                'prediction_modes': data.get('prediction_modes', {}),
                'play_strategies': data.get('play_strategies', {}),
                'has_settlement': _check_settlement_exists(data.get('snapshot_id', '')),
            })
        except Exception:
            continue

    with _record_index_lock:
        _snapshot_index_cache['signature'] = signature
        _snapshot_index_cache['records'] = snapshots
    return copy.deepcopy(snapshots)


def list_exclude_recalculations() -> List[Dict]:
    directory = Path(_cfg.KL8_RECALCULATION_DIR)
    if not directory.exists():
        return []
    signature = _directory_signature(directory)
    with _record_index_lock:
        if _recalculation_index_cache['signature'] == signature:
            return copy.deepcopy(_recalculation_index_cache['records'])
    records = []
    for path in directory.glob('recalculation_*.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(record, dict):
                records.append(record)
        except Exception:
            continue
    records.sort(key=lambda item: (
        str(item.get('target_issue') or ''),
        str(item.get('play_type') or ''),
        int(item.get('round', 0)),
    ), reverse=True)
    with _record_index_lock:
        _recalculation_index_cache['signature'] = signature
        _recalculation_index_cache['records'] = records
    return copy.deepcopy(records)


def _check_settlement_exists(snapshot_id: str) -> bool:
    if not snapshot_id:
        return False
    return (Path(_cfg.KL8_SETTLEMENT_DIR) / f'settlement_{snapshot_id}.json').exists()


