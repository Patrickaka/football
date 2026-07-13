#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球预测记录模块
================
保存每日预测记录，支持赛后结果回填和ELO/校准器反馈。
与 football/result_sync.py 同架构。
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, date

from ..common import kv_store

logger = logging.getLogger(__name__)

BB_PREDICTION_KEY = 'basketball_prediction_records'
BB_RESULT_KEY = 'basketball_match_results'
MAX_RECORDS = 500


def _today() -> str:
    return date.today().isoformat()


def save_predictions(date_str: str, matches: List[Dict], version: str = ''):
    """
    保存预测记录

    参数:
        date_str: 日期
        matches: 预测结果列表（每个含 match, spf, rqspf, dx 字段）
        version: 模型版本
    """
    existing = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(existing, list):
        existing = []

    # 移除同一日期的旧记录
    existing = [r for r in existing if r.get('date') != date_str]

    for m in matches:
        match_data = m.get('match', {})
        spf = m.get('spf', {})
        rqspf = m.get('rqspf', {})
        dx = m.get('dx', {})

        record = {
            'date': date_str,
            'match_id': match_data.get('id', ''),
            'num': match_data.get('num', ''),
            'league': match_data.get('league', ''),
            'home': match_data.get('home', ''),
            'away': match_data.get('away', ''),
            'time': match_data.get('time', ''),
            'version': version,
            'created_at': datetime.now().isoformat(),

            'spf': {
                'available': spf.get('available', False),
                'recommendation': spf.get('recommendation'),
                'home_prob': spf.get('home_prob'),
                'away_prob': spf.get('away_prob'),
                'confidence': spf.get('confidence'),
                'elo_home_prob': spf.get('elo_home_prob'),
            } if spf else None,

            'rqspf': {
                'available': rqspf.get('available', False),
                'recommendation': rqspf.get('recommendation'),
                'handicap': rqspf.get('handicap'),
                'home_prob': rqspf.get('home_prob'),
                'away_prob': rqspf.get('away_prob'),
                'confidence': rqspf.get('confidence'),
                'elo_margin': rqspf.get('elo_margin'),
            } if rqspf else None,

            'dx': {
                'available': dx.get('available', False),
                'recommendation': dx.get('recommendation'),
                'total_line': dx.get('total_line'),
                'over_prob': dx.get('over_prob'),
                'under_prob': dx.get('under_prob'),
                'confidence': dx.get('confidence'),
                'elo_total': dx.get('elo_total'),
            } if dx else None,

            # 赛后结果（初始为 None）
            'result': None,
        }
        existing.append(record)

    # 限制记录数
    if len(existing) > MAX_RECORDS:
        existing = existing[-MAX_RECORDS:]

    kv_store.save(BB_PREDICTION_KEY, existing)
    logger.info(f"篮球预测记录已保存: {date_str}, {len(matches)} 场")


def get_predictions(date_str: str = None, limit: int = 50) -> List[Dict]:
    """
    获取预测记录

    参数:
        date_str: 指定日期，None 返回最近
        limit: 最大返回数
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return []

    if date_str:
        records = [r for r in records if r.get('date') == date_str]

    return records[-limit:]


def get_unsettled_predictions() -> List[Dict]:
    """获取尚未结算的预测记录"""
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return []
    return [r for r in records if r.get('result') is None]


def save_match_result(match_id: str, result: Dict):
    """
    保存比赛结果并更新预测记录

    参数:
        match_id: 比赛ID
        result: {
            'home_score': int,
            'away_score': int,
            'status': str,
            'settled_at': str,
        }
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return

    updated = False
    for r in records:
        if r.get('match_id') == match_id:
            r['result'] = result
            updated = True
            break

    if updated:
        kv_store.save(BB_PREDICTION_KEY, records)
        logger.info(f"篮球比赛结果已更新: {match_id}")


def get_prediction_stats() -> Dict:
    """
    获取预测统计（用于评估准确率）

    只统计已结算的预测记录。
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        records = []

    settled = [r for r in records if r.get('result')]

    stats = {
        'total_predictions': len(records),
        'settled_count': len(settled),
        'spf': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'rqspf': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'dx': {'total': 0, 'correct': 0, 'accuracy': 0.0},
    }

    for r in settled:
        result = r['result']
        home_score = result.get('home_score', 0)
        away_score = result.get('away_score', 0)
        total_score = home_score + away_score

        # SPF 准确率
        spf = r.get('spf') or {}
        if spf.get('available'):
            stats['spf']['total'] += 1
            predicted = spf.get('recommendation')
            actual = '主胜' if home_score > away_score else '客胜'
            if predicted == actual:
                stats['spf']['correct'] += 1

        # RQSPF 准确率
        rqspf = r.get('rqspf') or {}
        if rqspf.get('available'):
            stats['rqspf']['total'] += 1
            handicap_str = rqspf.get('handicap', '')
            try:
                handicap = float(handicap_str) if handicap_str else 0.0
            except (TypeError, ValueError):
                handicap = 0.0

            adjusted_diff = (home_score + handicap) - away_score
            actual_rq = '让胜' if adjusted_diff > 0 else '让负'
            predicted_rq = rqspf.get('recommendation')
            if predicted_rq == actual_rq:
                stats['rqspf']['correct'] += 1

        # DX 准确率
        dx = r.get('dx') or {}
        if dx.get('available'):
            stats['dx']['total'] += 1
            total_line = dx.get('total_line', 0)
            if total_line is None:
                total_line = 0
            try:
                total_line = float(total_line)
            except (TypeError, ValueError):
                total_line = 0

            actual_dx = '大分' if total_score > total_line else '小分'
            predicted_dx = dx.get('recommendation')
            if predicted_dx == actual_dx:
                stats['dx']['correct'] += 1

    # 计算准确率
    for key in ['spf', 'rqspf', 'dx']:
        if stats[key]['total'] > 0:
            stats[key]['accuracy'] = round(stats[key]['correct'] / stats[key]['total'], 4)

    return stats


def feed_calibration():
    """
    将已结算预测反馈给校准器

    读取所有已结算记录，将每条预测的实际命中情况录入校准器。
    """
    from .calibration import get_calibrator

    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return 0

    calibrator = get_calibrator()
    count = 0

    for r in records:
        result = r.get('result')
        if not result:
            continue

        home_score = result.get('home_score', 0)
        away_score = result.get('away_score', 0)
        total_score = home_score + away_score
        league = r.get('league', '')

        # SPF
        spf = r.get('spf') or {}
        if spf.get('available'):
            predicted = spf.get('recommendation')
            actual = '主胜' if home_score > away_score else '客胜'
            prob = spf.get('home_prob', 0.5) if predicted == '主胜' else spf.get('away_prob', 0.5)
            calibrator.record('spf', prob, predicted == actual, league, spf.get('confidence', 'medium'))
            count += 1

        # RQSPF
        rqspf = r.get('rqspf') or {}
        if rqspf.get('available'):
            handicap_str = rqspf.get('handicap', '')
            try:
                handicap = float(handicap_str) if handicap_str else 0.0
            except (TypeError, ValueError):
                handicap = 0.0
            adjusted_diff = (home_score + handicap) - away_score
            actual_rq = '让胜' if adjusted_diff > 0 else '让负'
            predicted_rq = rqspf.get('recommendation')
            prob = rqspf.get('home_prob', 0.5) if predicted_rq == '让胜' else rqspf.get('away_prob', 0.5)
            calibrator.record('rqspf', prob, predicted_rq == actual_rq, league, rqspf.get('confidence', 'medium'))
            count += 1

        # DX
        dx = r.get('dx') or {}
        if dx.get('available'):
            total_line = dx.get('total_line', 0)
            if total_line is None:
                total_line = 0
            try:
                total_line = float(total_line)
            except (TypeError, ValueError):
                total_line = 0
            actual_dx = '大分' if total_score > total_line else '小分'
            predicted_dx = dx.get('recommendation')
            prob = dx.get('over_prob', 0.5) if predicted_dx == '大分' else dx.get('under_prob', 0.5)
            calibrator.record('dx', prob, predicted_dx == actual_dx, league, dx.get('confidence', 'medium'))
            count += 1

    calibrator.save()
    logger.info(f"校准反馈完成: {count} 条记录")
    return count
