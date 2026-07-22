#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
篮球实时赔率变化分析（Line Movement / Sharp Money）
==================================================

把竞彩篮球的「赔率实时变化」转成可量化的交易信号，并接入推荐管线：

1. 500 源轮询追踪：周期性调用 fetch_basketball_schedule，把每场快照写入
   kv_store（basketball_odds_history），累积出真实盘路变化。
2. 走势计算：从快照序列（或澳客 rf_trend / dx_trend / ml bundle）算出两路
   movement —— 哪一侧被资金追捧（home/away/over/under/flat）、强度、是否
   steam（急单）、是否 stale（长时间无变化）。
3. 信号映射：把 movement 映射成 okooo.analyze_line_trend 的 trend 字典，
   复用 adjust_two_way_by_trend 微调双边概率；并对「聪明钱确认本方推荐」
   的场次打 sharp_confirmed 标、提升置信度。

设计原则：
- 赔率走势只做「增强/排序」，绝不推翻模型方向；模型与走势矛盾时降权而非反转。
- 所有网络抓取都有降级：澳客 WAF / 无历史时 movement=None，推荐回退到原逻辑。
- 默认 source=500 时若尚未累积历史，movement=None，零回归。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..common import kv_store

log = logging.getLogger(__name__)

ODDS_HISTORY_KEY = 'basketball_odds_history'
HISTORY_CAP = 240  # 每场最多保留快照数

# steam：短时间内出现明显且单向的赔率位移（资金急涌）
STEAM_STRENGTH = 0.45
# 视为 stale 的静止时长（分钟）：盘口长期不动，信号价值低
STALE_MINUTES = 240
# 微调概率的最大幅度（防止走势喧宾夺主）
MAX_ADJUST_FACTOR = 0.10


def _parse_ts(ts) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def track_basketball_odds(date: str = None) -> int:
    """轮询 500 源赛程，把当前赔率快照追加进 kv_store 历史。

    返回本次追踪到的场次数量。建议由定时任务（比赛日每 10~15 分钟）调用，
    以累积出真实的盘路变化序列。
    """
    try:
        from . import fetch_basketball_schedule
        matches = fetch_basketball_schedule(date)
    except Exception as exc:
        log.warning(f"篮球赔率追踪抓取失败: {exc}")
        return 0
    if not matches:
        return 0

    history = kv_store.load(ODDS_HISTORY_KEY, {})
    now_iso = datetime.now().isoformat()
    now = _parse_ts(now_iso)
    count = 0
    for m in matches:
        mid = m.get('id')
        if not mid:
            continue
        snap = {
            'ts': now_iso,
            'spf_home': m.get('spf_home'),
            'spf_away': m.get('spf_away'),
            'rqspf_home': m.get('rqspf_home'),
            'rqspf_away': m.get('rqspf_away'),
            'dx_over': m.get('dx_over'),
            'dx_under': m.get('dx_under'),
            'handicap': m.get('handicap'),
            'total_line': m.get('total_line'),
        }
        # 跳过全空快照（无赔率场次）
        if not any(v is not None for k, v in snap.items() if k != 'ts'):
            continue
        seq = history.setdefault(mid, [])
        # 去重：与上一个快照完全相同则跳过
        if seq and all(seq[-1].get(k) == snap.get(k) for k in (
                'spf_home', 'spf_away', 'rqspf_home', 'rqspf_away',
                'dx_over', 'dx_under')):
            # 仅更新 ts，便于 staleness 判定
            seq[-1]['ts'] = now_iso
        else:
            seq.append(snap)
            if len(seq) > HISTORY_CAP:
                seq[:] = seq[-HISTORY_CAP:]
        count += 1

    kv_store.save(ODDS_HISTORY_KEY, history)
    if now:
        log.info(f"篮球赔率追踪完成: {count} 场, 当前 {now.strftime('%H:%M')}")
    return count


def _movement_from_snapshots(snaps: List[Dict], hk: str, ak: str) -> Optional[Dict]:
    """从快照序列计算两路 movement。hk/ak 为两侧赔率字段名。

    返回标准化 movement 字典；样本不足返回 None。
    """
    valid = [s for s in snaps if s.get(hk) and s.get(ak)]
    if len(valid) < 2:
        return None

    first, last = valid[0], valid[-1]
    h0, a0 = float(first[hk]), float(first[ak])
    h1, a1 = float(last[hk]), float(last[ak])
    # 赔率下降 = 该侧被买入 = 资金追捧
    hm = h1 - h0
    am = a1 - a0
    raw_strength = abs(hm) + abs(am)

    if hm < -0.02 and am > 0.01:
        side = 'home'
    elif am < -0.02 and hm > 0.01:
        side = 'away'
    else:
        side = 'flat'

    strength = min(1.0, raw_strength / 0.4)

    # 时间维度
    t_first = _parse_ts(first.get('ts'))
    t_last = _parse_ts(last.get('ts'))
    now = datetime.now()
    window_min = (now - t_first).total_seconds() / 60.0 if t_first else 0.0

    # 找出最后一次变化的时间
    last_move_ts = None
    for i in range(len(valid) - 1, 0, -1):
        if (valid[i].get(hk) != valid[i - 1].get(hk) or
                valid[i].get(ak) != valid[i - 1].get(ak)):
            last_move_ts = _parse_ts(valid[i].get('ts'))
            break
    last_move_age = (now - last_move_ts).total_seconds() / 60.0 if last_move_ts else None

    steam = strength >= STEAM_STRENGTH and (last_move_age is None or last_move_age <= 90)
    stale = (last_move_age is not None and last_move_age >= STALE_MINUTES) or len(valid) < 3

    return {
        'available': True,
        'side': side,
        'strength': round(strength, 4),
        'raw_strength': round(raw_strength, 4),
        'steam': steam,
        'stale': stale,
        'samples': len(valid),
        'window_min': round(max(0.0, window_min), 1),
        'last_move_age_min': round(last_move_age, 1) if last_move_age is not None else None,
        'home_move': round(hm, 4),
        'away_move': round(am, 4),
    }


def _normalize_okooo_trend(trend: Optional[Dict], kind: str) -> Optional[Dict]:
    """把 okooo.analyze_line_trend 的输出规范化成统一 movement 字典。

    kind: 'ah'(让分) / 'ou'(大小) / 'ml'(胜负)
    """
    if not trend or not isinstance(trend, dict):
        return None
    direction = trend.get('direction')
    if direction in (None, 'stable'):
        side = 'flat'
    elif direction in ('home_backing', 'over_backing'):
        side = 'home' if kind in ('ah', 'ml') else 'over'
    elif direction in ('away_backing', 'under_backing'):
        side = 'away' if kind in ('ah', 'ml') else 'under'
    elif direction == 'line_up':
        side = 'home' if kind in ('ah', 'ml') else 'over'
    elif direction == 'line_down':
        side = 'away' if kind in ('ah', 'ml') else 'under'
    else:
        side = 'flat'

    strength = min(1.0, float(trend.get('strength', 0) or 0) / 0.2)
    return {
        'available': True,
        'side': side,
        'strength': round(strength, 4),
        'raw_strength': round(float(trend.get('strength', 0) or 0), 4),
        'steam': strength >= STEAM_STRENGTH,
        'stale': False,
        'samples': int(trend.get('samples', 0) or 0),
        'window_min': None,
        'last_move_age_min': None,
        'home_move': round(float(trend.get('home_move', 0) or 0), 4),
        'away_move': round(float(trend.get('away_move', 0) or 0), 4),
    }


def build_movement_for_match(match: Dict, kv_history: Dict = None,
                             okooo_bundle: Dict = None) -> Dict:
    """为单场比赛构建 {spf, rqspf, dx} 三个玩法的 movement 字典。

    优先级：
    - rqspf / dx：优先用 okooo 赛程自带的 rf_trend / dx_trend（已含完整盘路）。
    - spf：优先用 okooo 详情 ml bundle 的 trend（需传入 okooo_bundle）。
    - 500 源（无 okooo 数据时）：用 kv_history 中该 match_id 的快照序列计算。

    任意玩法无数据则对应值为 None（调用方回退原逻辑）。
    """
    out = {'spf': None, 'rqspf': None, 'dx': None}
    source = match.get('source')

    # —— 让分胜负 rqspf ——
    if source == 'okooo' and match.get('rf_trend'):
        out['rqspf'] = _normalize_okooo_trend(match.get('rf_trend'), 'ah')
    elif okooo_bundle and okooo_bundle.get('ah', {}).get('available'):
        out['rqspf'] = _normalize_okooo_trend(okooo_bundle['ah'].get('trend'), 'ah')

    # —— 大小分 dx ——
    if source == 'okooo' and match.get('dx_trend'):
        out['dx'] = _normalize_okooo_trend(match.get('dx_trend'), 'ou')
    elif okooo_bundle and okooo_bundle.get('ou', {}).get('available'):
        out['dx'] = _normalize_okooo_trend(okooo_bundle['ou'].get('trend'), 'ou')

    # —— 胜负 spf ——
    if okooo_bundle and okooo_bundle.get('ml', {}).get('available'):
        out['spf'] = _normalize_okooo_trend(okooo_bundle['ml'].get('trend'), 'ml')

    # 500 源回退：用 kv_store 历史序列
    if kv_history is not None:
        seq = kv_history.get(match.get('id'))
        if seq:
            if out['spf'] is None:
                out['spf'] = _movement_from_snapshots(seq, 'spf_home', 'spf_away')
            if out['rqspf'] is None:
                out['rqspf'] = _movement_from_snapshots(seq, 'rqspf_home', 'rqspf_away')
            if out['dx'] is None:
                out['dx'] = _movement_from_snapshots(seq, 'dx_over', 'dx_under')

    return out


def movement_to_trend(movement: Optional[Dict]) -> Optional[Dict]:
    """把统一 movement 映射回 okooo.analyze_line_trend 的 trend 字典，
    供 adjust_two_way_by_trend 复用。"""
    if not movement or not movement.get('available'):
        return None
    side = movement.get('side')
    mapping = {
        'home': 'home_backing',
        'away': 'away_backing',
        'over': 'over_backing',
        'under': 'under_backing',
        'flat': 'stable',
    }
    return {
        'direction': mapping.get(side, 'stable'),
        'strength': movement.get('strength', 0.0),
        'home_move': movement.get('home_move', 0.0),
        'away_move': movement.get('away_move', 0.0),
        'line_move': 0.0,
        'kind': 'ml',
        'samples': movement.get('samples', 0),
    }


def apply_movement(p_home: float, p_away: float, movement: Optional[Dict],
                   factor: float = None) -> tuple:
    """用赔率走势微调双边概率。

    返回 (p_home, p_away) 微调后（已归一化）。无 movement 时原样返回。
    """
    if not movement or not movement.get('available') or movement.get('side') == 'flat':
        return p_home, p_away
    try:
        from . import okooo
        trend = movement_to_trend(movement)
        if factor is None:
            factor = MAX_ADJUST_FACTOR
        f = max(0.0, min(MAX_ADJUST_FACTOR, factor))
        ph, pa = okooo.adjust_two_way_by_trend(p_home, p_away, trend, factor=f)
        return ph, pa
    except Exception as exc:
        log.warning(f"篮球走势微调失败: {exc}")
        return p_home, p_away


def sharp_confirmation(movement: Optional[Dict], recommendation: str) -> Dict:
    """判断「聪明钱」是否确认本方的推荐方向。

    recommendation 取值：主胜/客胜(对应 home/away)、让胜/让负(对应 home/away)、
    大分/小分(对应 over/under)。
    返回 {confirmed: bool, reason: str, boost: float}
    boost 为置信度提升档位（0/0.5/1），供调用方调高 confidence。
    """
    if not movement or not movement.get('available') or movement.get('side') == 'flat':
        return {'confirmed': False, 'reason': 'no_movement', 'boost': 0.0}

    rec_side = None
    if recommendation in ('主胜', '让胜'):
        rec_side = 'home'
    elif recommendation in ('客胜', '让负'):
        rec_side = 'away'
    elif recommendation == '大分':
        rec_side = 'over'
    elif recommendation == '小分':
        rec_side = 'under'

    if rec_side is None:
        return {'confirmed': False, 'reason': 'unknown_pick', 'boost': 0.0}

    if movement['side'] != rec_side:
        # 资金流向与本方推荐相反：降权但不反转
        return {'confirmed': False, 'reason': 'contrary_flow', 'boost': -0.5}

    strength = movement.get('strength', 0.0)
    if movement.get('steam'):
        return {'confirmed': True, 'reason': 'steam_confirm', 'boost': 1.0}
    if strength >= 0.3:
        return {'confirmed': True, 'reason': 'strong_confirm', 'boost': 0.5}
    return {'confirmed': True, 'reason': 'mild_confirm', 'boost': 0.0}


def get_odds_history(match_id: str = None) -> Dict:
    """读取累积的赔率历史（调试/前端用）。"""
    history = kv_store.load(ODDS_HISTORY_KEY, {})
    if match_id:
        return history.get(match_id, [])
    return history
