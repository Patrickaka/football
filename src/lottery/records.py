#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大乐透线上预测记录：期号推导、记录存取、奖级结算、统计"""

import math
import time
from collections import Counter
from typing import Dict, List

from ..common.logger import setup_logger
from ..common import kv_store
from .config import LOTTERY_PREDICTOR_VERSION

log = setup_logger('lottery')

# ==================== 线上预测记录系统 ====================

DALETOU_PREDICTIONS_KEY = 'lottery_dlt_online_predictions'


def _next_issue(issue: str, history_data: List[Dict] = None) -> str:
    """推算下一期期号（YYYYNNN 格式，跨年自动进位）。

    v4.5 修复：原实现 `str(int(issue)+1)` 在年末产生永不存在的期号
    （如 2025150+1=2025151，而 2025 年最后一期就是 2025150），
    导致年末最后一条预测记录永远无法结算（pending 悬挂）。
    与双色球 v3.1 `_next_period` 同类 bug。

    跨年判断规则（近年大乐透每年约 150 期）：
    - 当年不是历史最后一年 → 年份已完结，n 达到当年最大期号即跨年；
    - 当年是历史最后一年（数据可能未抓全）→ 仅当已开期数 >= 145
      且 n 已达当年最大值时才跨年，否则正常 +1。
    """
    s = str(issue)
    if not (s.isdigit() and len(s) >= 6):
        try:
            return str(int(issue) + 1).zfill(len(s))
        except Exception:
            return ''
    y, n = int(s[:4]), int(s[4:])
    if history_data:
        same_year = [int(str(h.get('issue', ''))[4:]) for h in history_data
                     if str(h.get('issue', '')).isdigit()
                     and str(h.get('issue', ''))[:4] == s[:4]]
        max_n = max(same_year) if same_year else 0
        if max_n:
            years = sorted({int(str(h.get('issue', ''))[:4]) for h in history_data
                            if str(h.get('issue', '')).isdigit()})
            is_last_year = years and (y == years[-1])
            if not is_last_year:
                if n >= max_n:
                    return f'{y + 1}001'
                return f'{y}{n + 1:03d}'
            if n >= max_n and max_n >= 145:
                return f'{y + 1}001'
            return f'{y}{n + 1:03d}'
    if n >= 155:  # 无历史兜底：大乐透近年每年最多154期
        return f'{y + 1}001'
    return f'{y}{n + 1:03d}'


def load_online_predictions() -> List[Dict]:
    """加载线上预测记录"""
    try:
        return kv_store.load(DALETOU_PREDICTIONS_KEY, [])
    except Exception as e:
        log.error(f"加载大乐透预测记录失败: {e}")
        return []


def save_online_prediction(period: str, recommendations: Dict,
                           fusion_result: Dict = None,
                           based_on_issue: str = None) -> None:
    """保存线上预测记录

    Args:
        period: 目标期号
        recommendations: 各策略推荐 {method: {front, back}}
        fusion_result: 融合结果 (可选)
    """
    try:
        records = load_online_predictions()

        record = {
            'version': LOTTERY_PREDICTOR_VERSION,
            'period': period,
            'based_on_issue': str(based_on_issue or ''),
            'integrity_status': 'pending',
            'recommendations': {
                method: {
                    'front': rec.get('front', []),
                    'back': rec.get('back', []),
                }
                for method, rec in recommendations.items()
            },
            'actual': None,
            'settled': False,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        if fusion_result:
            record['fusion'] = {
                'front_top12': fusion_result.get('front_top12', []),
                'back_top6': fusion_result.get('back_top6', []),
                'weights': {
                    'rule': fusion_result.get('rule_weight', 0.55),
                    'ml': fusion_result.get('ml_weight', 0.45),
                },
            }

        # 按期号去重
        existing_idx = None
        for i, r in enumerate(records):
            if r.get('period') == period:
                existing_idx = i
                break

        if existing_idx is not None:
            if records[existing_idx].get('settled'):
                log.info(f"预测记录已结算，跳过更新: {period}")
                return
            record['created_at'] = records[existing_idx].get('created_at', record['created_at'])
            records[existing_idx] = record
        else:
            records.append(record)

        # 保留最近200期
        records = records[-200:]
        kv_store.save(DALETOU_PREDICTIONS_KEY, records)
        log.info(f"大乐透预测记录已保存: {period}")
    except Exception as e:
        log.error(f"保存大乐透预测记录失败: {e}")


def dlt_prize_tier(front_hits: int, back_hits: int) -> int:
    """大乐透奖级判定（0=未中奖）。

    一等 5+2 / 二等 5+1 / 三等 5+0 / 四等 4+2 / 五等 4+1 / 六等 3+2 /
    七等 4+0 / 八等 3+1,2+2 / 九等 3+0,2+1,1+2,0+2。
    注意：前区命中≤2 且后区未中时奖金为 0 —— 前后区命中数必须联合看奖级。
    """
    if front_hits == 5 and back_hits == 2:
        return 1
    if front_hits == 5 and back_hits == 1:
        return 2
    if front_hits == 5:
        return 3
    if front_hits == 4 and back_hits == 2:
        return 4
    if front_hits == 4 and back_hits == 1:
        return 5
    if front_hits == 3 and back_hits == 2:
        return 6
    if front_hits == 4:
        return 7
    if (front_hits == 3 and back_hits == 1) or (front_hits == 2 and back_hits == 2):
        return 8
    if ((front_hits == 3 and back_hits == 0) or (front_hits == 2 and back_hits == 1)
            or (front_hits == 1 and back_hits == 2) or (front_hits == 0 and back_hits == 2)):
        return 9
    return 0


def settle_predictions(history_data: List[Dict]) -> int:
    """结算未回填的预测记录

    Args:
        history_data: 历史开奖数据 (idx=0最新)

    Returns:
        结算的记录数
    """
    records = load_online_predictions()
    if not records:
        return 0

    changed = False
    settled_count = 0

    period_index = {h['issue']: i for i, h in enumerate(history_data)}

    for record in records:
        if record.get('settled'):
            continue

        period = record.get('period')
        if period not in period_index:
            continue

        idx = period_index[period]
        actual_data = history_data[idx]
        actual_front = set(actual_data['front'])
        actual_back = set(actual_data['back'])

        # A real forward prediction must prove which already-drawn issue it was
        # based on. Legacy rows without this field cannot be distinguished from
        # hindsight-generated predictions and must not be reported as hits.
        based_on = str(record.get('based_on_issue') or '')
        try:
            forward_valid = bool(based_on) and int(based_on) < int(period)
        except (TypeError, ValueError):
            forward_valid = False
        if not forward_valid:
            record['actual'] = {
                'front': sorted(actual_front),
                'back': sorted(actual_back),
            }
            record['settled'] = True
            record['settled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            record['integrity_status'] = 'legacy_unverified'
            record['integrity_note'] = '缺少开奖前生成凭证，不计入实盘命中'
            changed = True
            settled_count += 1
            continue

        record['integrity_status'] = 'verified_forward'

        record['actual'] = {
            'front': list(actual_front),
            'back': list(actual_back),
        }
        record['settled'] = True
        record['settled_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # 结算各策略命中
        for method, rec in record.get('recommendations', {}).items():
            pred_front = set(rec.get('front', []))
            pred_back = set(rec.get('back', []))
            front_hit = len(pred_front & actual_front)
            back_hit = len(pred_back & actual_back)
            record[f'{method}_front_hit'] = front_hit
            record[f'{method}_back_hit'] = back_hit
            # v4.5: 奖级结算（0=未中奖）——命中数≠中奖，奖级才对应真实奖金
            record[f'{method}_prize'] = dlt_prize_tier(front_hit, back_hit)

        # 结算融合命中
        if record.get('fusion'):
            fusion_front = set(record['fusion'].get('front_top12', []))
            fusion_back = set(record['fusion'].get('back_top6', []))
            record['fusion_front_hit'] = len(fusion_front & actual_front)
            record['fusion_back_hit'] = len(fusion_back & actual_back)

        changed = True
        settled_count += 1

    if changed:
        kv_store.save(DALETOU_PREDICTIONS_KEY, records)
        log.info(f"大乐透预测已结算 {settled_count} 条")

    return settled_count


def calculate_online_stats() -> Dict:
    """计算线上实盘命中率统计

    Returns:
        {
            'total_records': int,
            'settled_count': int,
            'by_method': {method: {front_ge1_rate, front_ge2_rate, back_ge1_rate, back_ge2_rate}},
            'fusion': {front_ge1_rate, front_ge2_rate, back_ge1_rate, back_ge2_rate},
            'baseline': 随机基准,
        }
    """
    records = load_online_predictions()
    settled = [
        r for r in records
        if r.get('settled') and r.get('integrity_status') == 'verified_forward'
    ]
    n = len(settled)

    if n == 0:
        return {
            'total_records': len(records),
            'settled_count': 0,
            'by_method': {},
            'fusion': {},
        }

    methods = set()
    for r in settled:
        methods.update(r.get('recommendations', {}).keys())

    by_method = {}
    for method in sorted(methods):
        # Strategies were added over time.  A method that did not exist in an
        # older version is not a miss and must not dilute its hit rate.
        method_records = [r for r in settled if method in (r.get('recommendations') or {})]
        method_n = len(method_records)
        front_hits = [r.get(f'{method}_front_hit', 0) for r in method_records]
        back_hits = [r.get(f'{method}_back_hit', 0) for r in method_records]
        by_method[method] = {
            'count': method_n,
            'front_ge1_rate': round(sum(1 for h in front_hits if h >= 1) / method_n, 4),
            'front_ge2_rate': round(sum(1 for h in front_hits if h >= 2) / method_n, 4),
            'front_ge3_rate': round(sum(1 for h in front_hits if h >= 3) / method_n, 4),
            'front_avg': round(sum(front_hits) / method_n, 2),
            'back_ge1_rate': round(sum(1 for h in back_hits if h >= 1) / method_n, 4),
            'back_ge2_rate': round(sum(1 for h in back_hits if h >= 2) / method_n, 4),
            'back_avg': round(sum(back_hits) / method_n, 2),
            # v4.5: 单策略中奖率（按命中数回算奖级，兼容旧记录）
            'prize_rate': round(
                sum(1 for f, b in zip(front_hits, back_hits) if dlt_prize_tier(f, b) > 0) / method_n, 4
            ),
        }

    def _portfolio_stats(rows):
        if not rows:
            return {}
        best_front = []
        best_back = []
        same_ticket_3p1 = []
        any_prize = []
        for record in rows:
            names = (record.get('recommendations') or {}).keys()
            pairs = [
                (record.get(f'{name}_front_hit', 0), record.get(f'{name}_back_hit', 0))
                for name in names
            ]
            best_front.append(max((front for front, _ in pairs), default=0))
            best_back.append(max((back for _, back in pairs), default=0))
            same_ticket_3p1.append(any(front >= 3 and back >= 1 for front, back in pairs))
            # v4.5: 任1注中奖率（旧记录无 prize 字段，按命中数回算）
            any_prize.append(any(dlt_prize_tier(f, b) > 0 for f, b in pairs))
        count = len(rows)
        return {
            'count': count,
            'front_any_ge2_rate': round(sum(hit >= 2 for hit in best_front) / count, 4),
            'front_any_ge3_rate': round(sum(hit >= 3 for hit in best_front) / count, 4),
            'back_any_ge1_rate': round(sum(hit >= 1 for hit in best_back) / count, 4),
            'back_any_ge2_rate': round(sum(hit >= 2 for hit in best_back) / count, 4),
            'same_ticket_front3_back1_rate': round(sum(same_ticket_3p1) / count, 4),
            'any_prize_rate': round(sum(any_prize) / count, 4),
            'avg_ticket_count': round(sum(len(r.get('recommendations') or {}) for r in rows) / count, 2),
        }

    versions = sorted({r.get('version') or 'legacy-unversioned' for r in settled})
    by_version = {
        version: _portfolio_stats([
            r for r in settled if (r.get('version') or 'legacy-unversioned') == version
        ])
        for version in versions
    }

    # 融合统计
    fusion_records = [r for r in settled if r.get('fusion')]
    fusion_stats = {}
    if fusion_records:
        fk = len(fusion_records)
        f_front_hits = [r.get('fusion_front_hit', 0) for r in fusion_records]
        f_back_hits = [r.get('fusion_back_hit', 0) for r in fusion_records]
        fusion_stats = {
            'count': fk,
            'front_ge1_rate': round(sum(1 for h in f_front_hits if h >= 1) / fk, 4),
            'front_ge2_rate': round(sum(1 for h in f_front_hits if h >= 2) / fk, 4),
            'front_ge3_rate': round(sum(1 for h in f_front_hits if h >= 3) / fk, 4),
            'front_avg': round(sum(f_front_hits) / fk, 2),
            'back_ge1_rate': round(sum(1 for h in f_back_hits if h >= 1) / fk, 4),
            'back_ge2_rate': round(sum(1 for h in f_back_hits if h >= 2) / fk, 4),
            'back_avg': round(sum(f_back_hits) / fk, 2),
        }

    return {
        'total_records': len(records),
        'settled_count': n,
        'unsettled_count': len(records) - n,
        'by_method': by_method,
        'portfolio': _portfolio_stats(settled),
        'by_version': by_version,
        'fusion': fusion_stats,
        'baseline': {
            'front_ge1': round(1 - math.comb(30, 5) / math.comb(35, 5), 4),
            'front_ge2': 0.1389,
            'front_ge3': 0.0139,
            'back_ge1': 0.4545,
            'back_ge2': 0.0455,
            # v4.5: 单注中奖率随机基准（公平摇奖理论值，不可系统性超越）
            'single_ticket_any_prize': 0.0666,
            'note': '单注任意奖级≈6.66%（九等奖为主）；组合5注后区覆盖10码后'
                    '任1注后区≥1=98.5%（已达结构上界）',
        },
    }


