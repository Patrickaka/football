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
3. 水位反推：分别识别水位和盘口的指向，在信号新鲜、样本充分且不冲突时，
   允许走势改变弱模型的原始方向；强模型与走势冲突时仍然放弃而不是硬翻。

设计原则：
- 让分使用「主队加分值」约定：-3.5 -> -5.5 是主队让深，指向主队；
  大小分升盘指向大分、降盘指向小分。
- 只有至少两个有效快照、未过期、强度达标且盘口/水位不冲突，才允许反推。
- 所有网络抓取都有降级：澳客 WAF / 无历史时 movement=None，推荐回退到原逻辑。
- 默认 source=500 时若尚未累积历史，movement=None，零回归。
"""
from __future__ import annotations

import logging
import threading
import time
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
INFERENCE_MIN_STRENGTH = 0.25
_scheduler_started = False


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
                'dx_over', 'dx_under', 'handicap', 'total_line')):
            # 保留真正发生变化的 ts；否则每次轮询都会把旧信号伪装成新信号。
            seq[-1]['observed_ts'] = now_iso
        else:
            seq.append(snap)
            if len(seq) > HISTORY_CAP:
                seq[:] = seq[-HISTORY_CAP:]
        count += 1

    kv_store.save(ODDS_HISTORY_KEY, history)
    if now:
        log.info(f"篮球赔率追踪完成: {count} 场, 当前 {now.strftime('%H:%M')}")
    return count


def start_basketball_odds_scheduler(interval_minutes: int = 15) -> bool:
    """后台自动累积篮球盘口快照；重复调用不会启动多个线程。"""
    global _scheduler_started
    if _scheduler_started:
        return False
    _scheduler_started = True
    interval_seconds = max(300, int(interval_minutes) * 60)

    def _loop():
        while True:
            try:
                track_basketball_odds(datetime.now().strftime('%Y-%m-%d'))
            except Exception as exc:
                log.warning(f"篮球赔率自动追踪失败: {exc}")
            time.sleep(interval_seconds)

    threading.Thread(
        target=_loop, daemon=True, name='BasketballOddsTracker'
    ).start()
    log.info(f"篮球赔率自动追踪已启动: 每 {interval_seconds // 60} 分钟")
    return True


def _movement_from_snapshots(snaps: List[Dict], hk: str, ak: str,
                             line_key: str = None, kind: str = 'ml') -> Optional[Dict]:
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
    opening_line = first.get(line_key) if line_key else None
    current_line = last.get(line_key) if line_key else None
    try:
        line_move = float(current_line) - float(opening_line)
    except (TypeError, ValueError):
        line_move = 0.0
    line_weight = 0.06 if kind in ('ah', 'ou') else 0.0
    raw_strength = abs(hm) + abs(am) + min(abs(line_move), 5.0) * line_weight

    first_side = 'over' if kind == 'ou' else 'home'
    second_side = 'under' if kind == 'ou' else 'away'
    water_side = 'flat'
    if hm < -0.02 and am > 0.01:
        water_side = first_side
    elif am < -0.02 and hm > 0.01:
        water_side = second_side
    elif hm < -0.04 and am >= -0.005:
        water_side = first_side
    elif am < -0.04 and hm >= -0.005:
        water_side = second_side

    line_side = 'flat'
    if abs(line_move) >= 0.5:
        if kind == 'ou':
            line_side = 'over' if line_move > 0 else 'under'
        elif kind == 'ah':
            # handicap 是加在主队得分上的有符号值。数值下降（例如
            # -3.5 -> -5.5，或 +5.5 -> +3.5）都代表主队变强。
            line_side = 'home' if line_move < 0 else 'away'

    signal_conflict = (
        water_side != 'flat' and line_side != 'flat' and water_side != line_side
    )
    signal_agreement = (
        water_side != 'flat' and line_side != 'flat' and water_side == line_side
    )
    if signal_conflict:
        # 冲突时保留较直接的水位方向用于展示，但禁止后续反推。
        side = water_side
    elif water_side != 'flat':
        side = water_side
    else:
        side = line_side

    if signal_agreement:
        raw_strength += 0.05

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
                valid[i].get(ak) != valid[i - 1].get(ak) or
                (line_key and valid[i].get(line_key) != valid[i - 1].get(line_key))):
            last_move_ts = _parse_ts(valid[i].get('ts'))
            break
    last_move_age = (now - last_move_ts).total_seconds() / 60.0 if last_move_ts else None

    steam = strength >= STEAM_STRENGTH and (last_move_age is None or last_move_age <= 90)
    stale = last_move_age is not None and last_move_age >= STALE_MINUTES

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
        'line_move': round(line_move, 4),
        'opening_line': opening_line,
        'current_line': current_line,
        'kind': kind,
        'water_side': water_side,
        'line_side': line_side,
        'signal_agreement': signal_agreement,
        'signal_conflict': signal_conflict,
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
        side = 'over' if kind == 'ou' else 'home'
    elif direction in ('away_backing', 'under_backing'):
        side = 'under' if kind == 'ou' else 'away'
    elif direction == 'line_up':
        side = 'away' if kind == 'ah' else ('home' if kind == 'ml' else 'over')
    elif direction == 'line_down':
        side = 'home' if kind == 'ah' else ('away' if kind == 'ml' else 'under')
    else:
        side = 'flat'

    home_move = float(trend.get('home_move', 0) or 0)
    away_move = float(trend.get('away_move', 0) or 0)
    line_move = float(trend.get('line_move', 0) or 0)
    first_side = 'over' if kind == 'ou' else 'home'
    second_side = 'under' if kind == 'ou' else 'away'
    water_side = 'flat'
    if home_move < -0.02 and away_move > 0.01:
        water_side = first_side
    elif away_move < -0.02 and home_move > 0.01:
        water_side = second_side
    elif home_move < -0.04 and away_move >= -0.005:
        water_side = first_side
    elif away_move < -0.04 and home_move >= -0.005:
        water_side = second_side
    line_side = 'flat'
    if abs(line_move) >= 0.5:
        if kind == 'ou':
            line_side = 'over' if line_move > 0 else 'under'
        elif kind == 'ah':
            line_side = 'home' if line_move < 0 else 'away'
    signal_conflict = (
        water_side != 'flat' and line_side != 'flat' and water_side != line_side
    )
    signal_agreement = (
        water_side != 'flat' and line_side != 'flat' and water_side == line_side
    )
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
        'home_move': round(home_move, 4),
        'away_move': round(away_move, 4),
        'line_move': round(line_move, 4),
        'opening_line': trend.get('opening_line'),
        'current_line': trend.get('current_line'),
        'kind': kind,
        'water_side': water_side,
        'line_side': line_side,
        'signal_agreement': signal_agreement,
        'signal_conflict': signal_conflict,
    }


def infer_market_from_movement(movement: Optional[Dict], market: str) -> Dict:
    """把盘口/水位变化转成可审计的独立反推结论。"""
    names = {
        'rqspf': {'home': '让胜', 'away': '让负'},
        'dx': {'over': '大分', 'under': '小分'},
    }
    mapping = names.get(market, {})
    side = (movement or {}).get('side', 'flat')
    strength = float((movement or {}).get('strength', 0) or 0)
    samples = int((movement or {}).get('samples', 0) or 0)
    stale = bool((movement or {}).get('stale'))
    conflict = bool((movement or {}).get('signal_conflict'))
    available = bool(movement and movement.get('available') and side in mapping)

    reason = 'movement_unavailable'
    if available:
        if samples < 2:
            reason = 'movement_samples_insufficient'
        elif stale:
            reason = 'movement_stale'
        elif conflict:
            reason = 'water_line_conflict'
        elif strength < INFERENCE_MIN_STRENGTH:
            reason = 'movement_signal_weak'
        else:
            reason = 'water_line_inference'
    actionable = available and reason == 'water_line_inference'
    confidence = (
        'high' if actionable and (strength >= 0.65 or movement.get('steam'))
        else 'medium' if actionable else 'low'
    )
    return {
        'available': available,
        'actionable': actionable,
        'side': side,
        'recommendation': mapping.get(side),
        'strength': round(strength, 4),
        'confidence': confidence,
        'reason': reason,
        'samples': samples,
        'stale': stale,
        'steam': bool((movement or {}).get('steam')),
        'water_side': (movement or {}).get('water_side', side),
        'line_side': (movement or {}).get('line_side', 'flat'),
        'signal_agreement': bool((movement or {}).get('signal_agreement')),
        'signal_conflict': conflict,
        'opening_line': (movement or {}).get('opening_line'),
        'current_line': (movement or {}).get('current_line'),
        'line_move': float((movement or {}).get('line_move', 0) or 0),
        'first_water_move': float((movement or {}).get('home_move', 0) or 0),
        'second_water_move': float((movement or {}).get('away_move', 0) or 0),
    }


def apply_market_inference(p_first: float, p_second: float,
                           movement: Optional[Dict], market: str) -> tuple:
    """融合模型与水位反推；强走势可翻转弱模型，但不能碾压强模型。"""
    inference = infer_market_from_movement(movement, market)
    model_side = 'home' if market == 'rqspf' and p_first >= p_second else None
    if market == 'rqspf' and model_side is None:
        model_side = 'away'
    elif market == 'dx':
        model_side = 'over' if p_first >= p_second else 'under'
    inference['model_side_before'] = model_side
    inference['model_recommendation_before'] = (
        {'home': '让胜', 'away': '让负', 'over': '大分', 'under': '小分'}
        .get(model_side)
    )
    if not inference['actionable']:
        inference['reversed_model'] = False
        inference['probability_shift'] = 0.0
        return p_first, p_second, inference

    strength = inference['strength']
    movement_edge = 0.025 + 0.075 * strength
    if inference['signal_agreement']:
        movement_edge += 0.015
    if inference['steam']:
        movement_edge += 0.01
    movement_first = inference['side'] in ('home', 'over')
    original_first = float(p_first)
    # 保留 55% 的原模型边际，让高确定性基本盘不被一次异常跳水硬翻。
    final_edge = (float(p_first) - 0.5) * 0.55
    final_edge += movement_edge if movement_first else -movement_edge
    final_first = max(0.34, min(0.66, 0.5 + final_edge))
    final_second = 1.0 - final_first
    final_side = (
        ('home' if final_first >= final_second else 'away') if market == 'rqspf'
        else ('over' if final_first >= final_second else 'under')
    )
    inference['reversed_model'] = final_side != model_side
    inference['final_side'] = final_side
    inference['probability_shift'] = round(final_first - original_first, 4)
    return final_first, final_second, inference


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
                out['rqspf'] = _movement_from_snapshots(
                    seq, 'rqspf_home', 'rqspf_away', 'handicap', 'ah'
                )
            if out['dx'] is None:
                out['dx'] = _movement_from_snapshots(
                    seq, 'dx_over', 'dx_under', 'total_line', 'ou'
                )

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
        'line_move': movement.get('line_move', 0.0),
        'kind': movement.get('kind', 'ml'),
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


def describe_market_movement(movements: Dict, bets: Dict) -> Dict:
    """Build a concise, user-facing joint interpretation of all three markets."""
    movements = movements or {}
    bets = bets or {}
    labels = {
        'home': '主队', 'away': '客队', 'over': '大分', 'under': '小分', 'flat': '稳定',
    }
    market_names = {'spf': '胜负', 'rqspf': '让分', 'dx': '大小分'}
    signals = []
    aligned = contrary = steam = 0
    for key in ('spf', 'rqspf', 'dx'):
        movement = movements.get(key) or {}
        if not movement.get('available'):
            continue
        side = movement.get('side', 'flat')
        bet = bets.get(key) or {}
        confirmed = (bet.get('line_movement') or {}).get('confirmed')
        if side != 'flat':
            aligned += int(bool(confirmed))
            contrary += int(confirmed is False)
        steam += int(bool(movement.get('steam')))
        line_move = float(movement.get('line_move', 0) or 0)
        line_text = ''
        if abs(line_move) >= 0.01:
            line_text = f"，盘口{'升' if line_move > 0 else '降'}{abs(line_move):g}分"
        signals.append({
            'market': key,
            'label': market_names[key],
            'side': side,
            'side_label': labels.get(side, side),
            'strength': round(float(movement.get('strength', 0) or 0), 3),
            'steam': bool(movement.get('steam')),
            'stale': bool(movement.get('stale')),
            'samples': int(movement.get('samples', 0) or 0),
            'line_move': line_move,
            'summary': f"{market_names[key]}资金偏{labels.get(side, side)}{line_text}",
            'aligned_with_model': confirmed,
        })
    if not signals:
        return {'available': False, 'verdict': '暂无有效盘口变化样本', 'signals': []}
    if steam and aligned:
        verdict, level = '急速资金与模型方向一致，属于强确认信号', 'strong'
    elif aligned > contrary:
        verdict, level = '盘口变化总体确认模型方向', 'positive'
    elif contrary > aligned:
        verdict, level = '盘口资金与模型方向冲突，建议降低置信度', 'warning'
    else:
        verdict, level = '各盘口信号分化，维持谨慎判断', 'neutral'
    return {
        'available': True, 'level': level, 'verdict': verdict, 'signals': signals,
        'aligned_count': aligned, 'contrary_count': contrary, 'steam_count': steam,
    }


def get_odds_history(match_id: str = None) -> Dict:
    """读取累积的赔率历史（调试/前端用）。"""
    history = kv_store.load(ODDS_HISTORY_KEY, {})
    if match_id:
        return history.get(match_id, [])
    return history
