# -*- coding: utf-8 -*-
"""快乐8预测入口 run_prediction、缓存、策略激活与快照列表"""

import math
import json
import time
import hashlib
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

from .config import (
    CANDIDATE_STRATEGIES, KL8_PREDICTOR_VERSION, REFERENCE_STRATEGY,
)
from .records import (
    _checksum_numbers, _persist_active_strategies, _strategy_fingerprint,
)
from .analyzer import (
    get_kl8_analyzer,
)

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
    log.info(f'快乐8: 策略已激活 {play_type} -> {_cfg.ACTIVE_STRATEGIES[play_type]["strategy_id"]}')


_prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def run_prediction(force_refresh: bool = False) -> Dict:
    """快乐8预测入口（v8: 缓存指纹包含所有策略配置）"""
    analyzer = get_kl8_analyzer()

    if not force_refresh:
        if analyzer.reload_if_needed():
            force_refresh = True

    if not analyzer.history_data:
        return {
            'error': '历史数据不足',
            'using_simulated_data': True,
        }

    # v8: 缓存指纹包含 _cfg.ACTIVE_STRATEGIES + REFERENCE_STRATEGY + CANDIDATE_STRATEGIES
    config_fingerprint = hashlib.sha256(
        json.dumps(
            {
                'active_strategies': _cfg.ACTIVE_STRATEGIES,
                'reference_strategy': REFERENCE_STRATEGY,
                'candidate_strategies': CANDIDATE_STRATEGIES,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()[:16]

    cache_key = (
        analyzer.history_data[0]['issue'],
        _checksum_numbers(analyzer.history_data[0]['numbers']),
        len(analyzer.history_data),
        KL8_PREDICTOR_VERSION,
        config_fingerprint,
    )

    cache = _prediction_cache
    if not force_refresh and cache['data'] is not None and cache.get('cache_key') == cache_key:
        return cache['data']

    result = analyzer.predict_all()

    cache['data'] = result
    cache['cache_key'] = cache_key
    cache['timestamp'] = time.time()

    return result


def clear_cache():
    global _prediction_cache
    from . import analyzer as _analyzer_mod
    _analyzer_mod._analyzer_instance = None
    _prediction_cache = {'data': None, 'timestamp': 0, 'cache_key': None}


def list_prediction_snapshots() -> List[Dict]:
    snapshot_dir = Path(_cfg.KL8_SNAPSHOT_DIR)
    if not snapshot_dir.exists():
        return []

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
                'version': data.get('version'),
                'is_experiment': data.get('is_experiment', False),
                'strategy_fingerprint': data.get('strategy_fingerprint', ''),
                'prediction_modes': data.get('prediction_modes', {}),
                'play_strategies': data.get('play_strategies', {}),
                'has_settlement': _check_settlement_exists(data.get('snapshot_id', '')),
            })
        except Exception:
            continue

    return snapshots


def list_exclude_recalculations() -> List[Dict]:
    directory = Path(_cfg.KL8_RECALCULATION_DIR)
    if not directory.exists():
        return []
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
    return records


def _check_settlement_exists(snapshot_id: str) -> bool:
    if not snapshot_id:
        return False
    return (Path(_cfg.KL8_SETTLEMENT_DIR) / f'settlement_{snapshot_id}.json').exists()


