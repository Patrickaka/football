#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""篮球赔率走势优化 - 离线逻辑验证（不依赖网络）。

验证点：
1. 快照序列 → movement 方向/强度/steam/stale 判定
2. apply_movement 微调概率（聪明钱侧提升）
3. sharp_confirmation 对本方推荐方向的确认/反对
4. build_movement_for_match 对 okooo rf_trend/dx_trend 的归一化
5. analyze_* 接入 movement 后输出 line_movement + sharp_confirmed
6. find_value_bets 在 movement_edge 上的优先排序
"""
import sys
import time
sys.path.insert(0, '.')

from src.basketball.odds_movement import (
    _movement_from_snapshots, apply_movement, sharp_confirmation,
    build_movement_for_match, movement_to_trend,
)
import src.basketball as bb


def iso(offset_min):
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(minutes=offset_min)).isoformat()


def snap(ts, spf_h, spf_a, rq_h, rq_a, dx_o, dx_u):
    return {'ts': ts, 'spf_home': spf_h, 'spf_away': spf_a,
            'rqspf_home': rq_h, 'rqspf_away': rq_a,
            'dx_over': dx_o, 'dx_under': dx_u}


print('=' * 64)
print('1) 快照序列 → movement 判定')
print('=' * 64)

# 主胜赔率从 1.70 降到 1.55，客胜从 1.95 升到 2.10 → 主队被追捧
seq = [
    snap(iso(120), 1.70, 1.95, 1.68, 2.05, 1.80, 1.90),
    snap(iso(60), 1.62, 2.02, 1.60, 2.12, 1.76, 1.95),
    snap(iso(10), 1.55, 2.10, 1.52, 2.20, 1.72, 2.00),
]
mv = _movement_from_snapshots(seq, 'spf_home', 'spf_away')
print('主队追捧: side=%s strength=%.3f steam=%s stale=%s' % (
    mv['side'], mv['strength'], mv['steam'], mv['stale']))
assert mv['side'] == 'home', '应判定为主队被追捧'
assert mv['steam'] is True, '短时间大幅位移应判 steam'

# 长时间无变化 → stale：变化发生在 ~490 分钟前，之后一直静止
seq_stale = [
    snap(iso(500), 1.70, 1.95, 1.68, 2.05, 1.80, 1.90),
    snap(iso(490), 1.55, 2.10, 1.52, 2.20, 1.72, 2.00),
    snap(iso(10), 1.55, 2.10, 1.52, 2.20, 1.72, 2.00),
]
mv_stale = _movement_from_snapshots(seq_stale, 'spf_home', 'spf_away')
print('长期静止盘: side=%s stale=%s' % (mv_stale['side'], mv_stale['stale']))
assert mv_stale['stale'] is True
# 全平序列（从未变化）应判为 flat 但非 stale
seq_flat = [
    snap(iso(400), 1.70, 1.95, 1.68, 2.05, 1.80, 1.90),
    snap(iso(300), 1.70, 1.95, 1.68, 2.05, 1.80, 1.90),
    snap(iso(200), 1.70, 1.95, 1.68, 2.05, 1.80, 1.90),
]
mv_flat = _movement_from_snapshots(seq_flat, 'spf_home', 'spf_away')
print('全平盘: side=%s stale=%s' % (mv_flat['side'], mv_flat['stale']))
assert mv_flat['side'] == 'flat' and mv_flat['stale'] is False

print()
print('=' * 64)
print('2) apply_movement 微调概率')
print('=' * 64)
p_h, p_a = apply_movement(0.55, 0.45, mv)
print('原 主%.3f/客%.3f → 微调后 主%.3f/客%.3f' % (0.55, 0.45, p_h, p_a))
assert p_h > 0.55, '主队被追捧应提升主胜概率'
# flat 走势不改变
p_h2, p_a2 = apply_movement(0.55, 0.45, {'available': True, 'side': 'flat'})
assert p_h2 == 0.55 and p_a2 == 0.45

print()
print('=' * 64)
print('3) sharp_confirmation 对本方推荐方向的确认/反对')
print('=' * 64)
sc_ok = sharp_confirmation(mv, '主胜')
print('推荐主胜 + 主队追捧: confirmed=%s reason=%s boost=%s' % (
    sc_ok['confirmed'], sc_ok['reason'], sc_ok['boost']))
assert sc_ok['confirmed'] and sc_ok['boost'] > 0
sc_no = sharp_confirmation(mv, '客胜')
print('推荐客胜 + 主队追捧: confirmed=%s reason=%s boost=%s' % (
    sc_no['confirmed'], sc_no['reason'], sc_no['boost']))
assert not sc_no['confirmed'] and sc_no['boost'] < 0

print()
print('=' * 64)
print('4) build_movement_for_match 归一化 okooo 走势')
print('=' * 64)
ok_match = {
    'id': '123', 'source': 'okooo',
    'rf_trend': {'direction': 'home_backing', 'strength': 0.6, 'samples': 5,
                 'home_move': -0.15, 'away_move': 0.18, 'line_move': 0, 'kind': 'ah'},
    'dx_trend': {'direction': 'over_backing', 'strength': 0.3, 'samples': 4,
                 'home_move': -0.08, 'away_move': 0.05, 'line_move': 0, 'kind': 'ou'},
}
mv_b = build_movement_for_match(ok_match)
print('rqspf side=%s dx side=%s' % (mv_b['rqspf']['side'], mv_b['dx']['side']))
assert mv_b['rqspf']['side'] == 'home'   # 让分主侧
assert mv_b['dx']['side'] == 'over'      # 大分侧

print()
print('=' * 64)
print('5) analyze_* 接入 movement 输出 line_movement + sharp_confirmed')
print('=' * 64)
m = {
    'home': '湖人', 'away': '勇士', 'league': 'NBA',
    'spf_home': 1.70, 'spf_away': 2.10,
    'rqspf_home': 1.52, 'rqspf_away': 2.20, 'handicap': '-5.5',
    'dx_over': 1.72, 'dx_under': 2.00, 'total_line': 220.0,
}
res_rq = bb.analyze_rqspf(m, mv_b['rqspf'])
res_dx = bb.analyze_daxiao(m, mv_b['dx'])
print('让分推荐=%s sharp=%s | line_movement.side=%s' % (
    res_rq['recommendation'], res_rq['sharp_confirmed'],
    (res_rq['line_movement'] or {}).get('side')))
print('大小推荐=%s sharp=%s' % (res_dx['recommendation'], res_dx['sharp_confirmed']))
assert res_rq['line_movement'] is not None and res_rq['sharp_confirmed'] is True
assert res_dx['line_movement'] is not None and res_dx['sharp_confirmed'] is True

print()
print('=' * 64)
print('6) find_value_bets 按 movement_edge 优先排序')
print('=' * 64)
res = [{
    'match': {'home': 'A', 'away': 'B', 'handicap': '-3.5', 'total_line': 200},
    'spf': {'available': True, 'playable': True, 'recommendation': '主胜',
            'home_prob': 0.60, 'away_prob': 0.40, 'sharp_confirmed': True,
            'line_movement': {'side': 'home', 'confirmed': True}},
    'rqspf': None, 'dx': None,
}, {
    'match': {'home': 'C', 'away': 'D', 'handicap': None, 'total_line': None},
    'spf': {'available': True, 'playable': True, 'recommendation': '客胜',
            'home_prob': 0.42, 'away_prob': 0.58, 'sharp_confirmed': False,
            'line_movement': None},
    'rqspf': None, 'dx': None,
}]
vbs = bb.find_value_bets(res, threshold=0.05)
print('排序后首推: %s %s (movement_edge=%.3f)' % (
    vbs[0]['type'], vbs[0]['recommendation'], vbs[0]['movement_edge']))
assert vbs[0]['sharp_confirmed'] is True, '聪明钱确认项应排第一'

print()
print('✅ 全部离线逻辑验证通过')
