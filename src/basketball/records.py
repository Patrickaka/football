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

    # Refreshing the recommendation endpoint must not erase settled games or
    # other matches from the same date (sources can return partial schedules).
    by_key = {
        (r.get('date'), r.get('match_id')): i
        for i, r in enumerate(existing)
        if r.get('match_id')
    }

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
                'pick_prob': spf.get('pick_prob'),
                'playable': spf.get('playable', True),
                'official': spf.get('official', spf.get('playable', True)),
                'skip_reason': spf.get('skip_reason'),
                'home_prob': spf.get('home_prob'),
                'away_prob': spf.get('away_prob'),
                'confidence': spf.get('confidence'),
                'elo_home_prob': spf.get('elo_home_prob'),
                'elo_trust': spf.get('elo_trust'),
                'market_home_prob': spf.get('market_home_prob'),
            } if spf else None,

            'rqspf': {
                'available': rqspf.get('available', False),
                'recommendation': rqspf.get('recommendation'),
                'pick_prob': rqspf.get('pick_prob'),
                'playable': rqspf.get('playable', True),
                'official': rqspf.get('official', rqspf.get('playable', True)),
                'skip_reason': rqspf.get('skip_reason'),
                'handicap': rqspf.get('handicap'),
                'home_prob': rqspf.get('home_prob'),
                'away_prob': rqspf.get('away_prob'),
                'confidence': rqspf.get('confidence'),
                'elo_margin': rqspf.get('elo_margin'),
                'elo_trust': rqspf.get('elo_trust'),
                'market_home_prob': rqspf.get('market_home_prob'),
                'line_movement': rqspf.get('line_movement'),
                'water_inference': rqspf.get('water_inference'),
                'movement_led': rqspf.get('movement_led', False),
                'sharp_confirmed': rqspf.get('sharp_confirmed', False),
            } if rqspf else None,

            'dx': {
                'available': dx.get('available', False),
                'recommendation': dx.get('recommendation'),
                'pick_prob': dx.get('pick_prob'),
                'playable': dx.get('playable', True),
                'official': dx.get('official', dx.get('playable', True)),
                'skip_reason': dx.get('skip_reason'),
                'total_line': dx.get('total_line'),
                'over_prob': dx.get('over_prob'),
                'under_prob': dx.get('under_prob'),
                'confidence': dx.get('confidence'),
                'elo_total': dx.get('elo_total'),
                'elo_trust': dx.get('elo_trust'),
                'market_over_prob': dx.get('market_over_prob'),
                'line_movement': dx.get('line_movement'),
                'water_inference': dx.get('water_inference'),
                'movement_led': dx.get('movement_led', False),
                'sharp_confirmed': dx.get('sharp_confirmed', False),
            } if dx else None,

            # 赛后结果（初始为 None）
            'result': None,
        }
        key = (date_str, record['match_id'])
        old_index = by_key.get(key)
        if old_index is None:
            by_key[key] = len(existing)
            existing.append(record)
        else:
            old_result = existing[old_index].get('result')
            if old_result:
                record['result'] = old_result
            existing[old_index] = record

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


def _evaluate_markets(record: Dict, home_score: int, away_score: int) -> Dict:
    """评估各玩法命中；走盘/平推记为 void，不计入准确率与校准。"""
    total_score = home_score + away_score
    out = {
        'spf_hit': None,
        'rqspf_hit': None,
        'dx_hit': None,
        'spf_void': False,
        'rqspf_void': False,
        'dx_void': False,
    }

    spf = record.get('spf') or {}
    if spf.get('available') and spf.get('playable', True):
        if home_score == away_score:
            out['spf_void'] = True
        else:
            actual = '主胜' if home_score > away_score else '客胜'
            out['spf_hit'] = spf.get('recommendation') == actual

    rqspf = record.get('rqspf') or {}
    if rqspf.get('available') and rqspf.get('playable', True):
        handicap_str = rqspf.get('handicap', '')
        try:
            handicap = float(handicap_str) if handicap_str not in (None, '') else 0.0
        except (TypeError, ValueError):
            handicap = 0.0
        adjusted_diff = (home_score + handicap) - away_score
        if abs(adjusted_diff) < 1e-9:
            out['rqspf_void'] = True
        else:
            actual_rq = '让胜' if adjusted_diff > 0 else '让负'
            out['rqspf_hit'] = rqspf.get('recommendation') == actual_rq

    dx = record.get('dx') or {}
    if dx.get('available') and dx.get('playable', True):
        total_line = dx.get('total_line', 0)
        try:
            total_line = float(total_line) if total_line is not None else 0.0
        except (TypeError, ValueError):
            total_line = 0.0
        if abs(total_score - total_line) < 1e-9:
            out['dx_void'] = True
        else:
            actual_dx = '大分' if total_score > total_line else '小分'
            out['dx_hit'] = dx.get('recommendation') == actual_dx

    return out


def settle_and_learn(match_id: str, home_score: int, away_score: int,
                     league: str = '', status: str = 'finished') -> Dict:
    """
    赛果回填 + ELO 更新 + 校准反馈（幂等）。

    同一 match_id 重复调用不会重复更新 ELO / 重复喂校准器。
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return {'ok': False, 'error': 'no_records'}

    target = None
    for r in records:
        if r.get('match_id') == match_id:
            target = r
            break
    if target is None:
        return {'ok': False, 'error': 'match_not_found', 'match_id': match_id}

    prev = target.get('result') if isinstance(target.get('result'), dict) else {}
    league_name = league or target.get('league', '') or 'NBA'
    eval_hits = _evaluate_markets(target, int(home_score), int(away_score))

    result = {
        'home_score': int(home_score),
        'away_score': int(away_score),
        'status': status,
        'settled_at': datetime.now().isoformat(),
        'elo_updated': bool(prev.get('elo_updated')),
        'calibration_fed': bool(prev.get('calibration_fed')),
        **eval_hits,
    }

    # ELO：仅首次写入
    if not result['elo_updated']:
        try:
            from .elo import get_elo_system
            elo = get_elo_system()
            elo.update_ratings(
                target.get('home', ''),
                target.get('away', ''),
                int(home_score),
                int(away_score),
                league_name,
            )
            result['elo_updated'] = True
        except Exception as e:
            logger.warning(f"结算 ELO 更新失败: {e}")

    target['result'] = result
    kv_store.save(BB_PREDICTION_KEY, records)

    # 校准：仅首次写入
    fed = 0
    if not result['calibration_fed']:
        fed = _feed_one_record(target)
        result['calibration_fed'] = True
        target['result'] = result
        kv_store.save(BB_PREDICTION_KEY, records)

    return {
        'ok': True,
        'match_id': match_id,
        'result': result,
        'calibration_samples': fed,
    }


def get_prediction_stats() -> Dict:
    """
    获取预测统计（用于评估准确率）

    只统计已结算且非走盘的预测记录。
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        records = []

    settled = [r for r in records if r.get('result')]

    stats = {
        'total_predictions': len(records),
        'settled_count': len(settled),
        'official_predictions': 0,
        'spf': {'total': 0, 'correct': 0, 'void': 0, 'accuracy': 0.0},
        'rqspf': {'total': 0, 'correct': 0, 'void': 0, 'accuracy': 0.0},
        'dx': {'total': 0, 'correct': 0, 'void': 0, 'accuracy': 0.0},
        'water_inference': {
            'rqspf': {'total': 0, 'correct': 0, 'accuracy': 0.0},
            'dx': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        },
    }

    for r in settled:
        result = r['result']
        home_score = result.get('home_score', 0)
        away_score = result.get('away_score', 0)
        hits = _evaluate_markets(r, home_score, away_score)

        spf = r.get('spf') or {}
        if spf.get('available') and spf.get('playable', True):
            stats['official_predictions'] += 1
            if hits['spf_void']:
                stats['spf']['void'] += 1
            else:
                stats['spf']['total'] += 1
                if hits['spf_hit']:
                    stats['spf']['correct'] += 1

        rqspf = r.get('rqspf') or {}
        if rqspf.get('available') and rqspf.get('playable', True):
            stats['official_predictions'] += 1
            if hits['rqspf_void']:
                stats['rqspf']['void'] += 1
            else:
                stats['rqspf']['total'] += 1
                if hits['rqspf_hit']:
                    stats['rqspf']['correct'] += 1
                if rqspf.get('movement_led'):
                    stats['water_inference']['rqspf']['total'] += 1
                    if hits['rqspf_hit']:
                        stats['water_inference']['rqspf']['correct'] += 1

        dx = r.get('dx') or {}
        if dx.get('available') and dx.get('playable', True):
            stats['official_predictions'] += 1
            if hits['dx_void']:
                stats['dx']['void'] += 1
            else:
                stats['dx']['total'] += 1
                if hits['dx_hit']:
                    stats['dx']['correct'] += 1
                if dx.get('movement_led'):
                    stats['water_inference']['dx']['total'] += 1
                    if hits['dx_hit']:
                        stats['water_inference']['dx']['correct'] += 1

    for key in ['spf', 'rqspf', 'dx']:
        if stats[key]['total'] > 0:
            stats[key]['accuracy'] = round(stats[key]['correct'] / stats[key]['total'], 4)
    for key in ('rqspf', 'dx'):
        item = stats['water_inference'][key]
        if item['total'] > 0:
            item['accuracy'] = round(item['correct'] / item['total'], 4)

    return stats


def _feed_one_record(r: Dict) -> int:
    """将单条已结算记录写入校准器（跳过 void）。"""
    from .calibration import get_calibrator

    result = r.get('result') or {}
    home_score = result.get('home_score', 0)
    away_score = result.get('away_score', 0)
    league = r.get('league', '')
    hits = _evaluate_markets(r, home_score, away_score)
    calibrator = get_calibrator()
    count = 0

    spf = r.get('spf') or {}
    if spf.get('available') and spf.get('playable', True) and not hits['spf_void'] and hits['spf_hit'] is not None:
        predicted = spf.get('recommendation')
        prob = spf.get('home_prob', 0.5) if predicted == '主胜' else spf.get('away_prob', 0.5)
        calibrator.record('spf', prob, bool(hits['spf_hit']), league, spf.get('confidence', 'medium'))
        count += 1

    rqspf = r.get('rqspf') or {}
    if rqspf.get('available') and rqspf.get('playable', True) and not hits['rqspf_void'] and hits['rqspf_hit'] is not None:
        predicted_rq = rqspf.get('recommendation')
        prob = rqspf.get('home_prob', 0.5) if predicted_rq == '让胜' else rqspf.get('away_prob', 0.5)
        calibrator.record('rqspf', prob, bool(hits['rqspf_hit']), league, rqspf.get('confidence', 'medium'))
        count += 1

    dx = r.get('dx') or {}
    if dx.get('available') and dx.get('playable', True) and not hits['dx_void'] and hits['dx_hit'] is not None:
        predicted_dx = dx.get('recommendation')
        prob = dx.get('over_prob', 0.5) if predicted_dx == '大分' else dx.get('under_prob', 0.5)
        calibrator.record('dx', prob, bool(hits['dx_hit']), league, dx.get('confidence', 'medium'))
        count += 1

    if count:
        calibrator.save()
    return count


def feed_calibration():
    """
    将已结算且尚未喂过校准器的预测反馈（幂等）。
    """
    records = kv_store.load(BB_PREDICTION_KEY, [])
    if not isinstance(records, list):
        return 0

    count = 0
    dirty = False
    for r in records:
        result = r.get('result')
        if not result:
            continue
        if result.get('calibration_fed'):
            continue
        fed = _feed_one_record(r)
        result['calibration_fed'] = True
        r['result'] = result
        count += fed
        dirty = True

    if dirty:
        kv_store.save(BB_PREDICTION_KEY, records)

    logger.info(f"校准反馈完成: {count} 条样本")
    return count
