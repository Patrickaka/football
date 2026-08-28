# -*- coding: utf-8 -*-
"""生成回测族（样本质量 / 策略口径 / 历史校准 / 动态权重 / 回测）的黄金语料。

**记录语料固定在文件里**，不读生产历史——读了的话黄金会随线上数据漂，
每次回填都要重生成，也就不再是参照物了。固定语料是从生产历史抽的形状
（`predicted_scores` 是 dict、`time_layer` 可能为 None、`asian_line` 常缺），
再补 24 条造出来的边角。
"""
import gzip
import json
import pathlib
import random

from src.domain.sports.football import backtest, calibration_history, policy
from src.domain.sports.football import quality, weights

MODULES = {'sample_quality': quality, 'prediction_policy': policy,
           'history_calibration': calibration_history,
           'dynamic_weights': weights, 'backtest': backtest}

FIXED_RECORDS = json.load(gzip.open(
    pathlib.Path(__file__).resolve().parents[1]
    / 'tests/fixtures/football_backtest_records.json.gz', 'rt', encoding='utf-8'))

_SEEN = {}


def _key(fn, label):
    """键必须唯一——同名 label 会互相覆盖，黄金里少掉的条目一声不响。"""
    base = f'{fn}:{label}'
    _SEEN[base] = _SEEN.get(base, 0) + 1
    return base if _SEEN[base] == 1 else f'{base}#{_SEEN[base]}'


def _y(mod, fn, label, *args, **kwargs):
    name = f'{mod}.{fn}'
    try:
        yield _key(name, label), getattr(MODULES[mod], fn)(*args, **kwargs)
    except Exception as exc:
        yield _key(name, label), f'{type(exc).__name__}: {exc}'


def entries():
    _SEEN.clear()
    random.seed(20260829)

    RECORDS = list(FIXED_RECORDS)

    # 真实记录都是"已结算、字段齐全"的那一类；边角要另外造，
    # 否则半数分支根本走不到（判据 23）
    for i in range(24):
        RECORDS.append({
            'match_id': f'synth{i}', 'home': f'主队{i}', 'away': f'客队{i}', 'league': ['英超', '德乙', '友谊赛', '欧冠杯', ''][i % 5],
            'match_time': f'2026-08-{10 + i % 18:02d} 20:00',
            'settled': i % 3 != 0,
            'sync_status': ['synced', 'failed', 'ignored', ''][i % 4],
            'actual_score': f'{i % 4}-{i % 3}' if i % 3 else '',
            'actual_result': ['H', 'D', 'A'][i % 3] if i % 3 else '',
            'actual_half_score': f'{i % 2}-0' if i % 5 else None,
            'predicted_scores': {'1-0': 0.3, '2-1': 0.2, '1-1': 0.5} if i % 7 else {},
            'predicted_1x2': {'H': 0.5, 'D': 0.3, 'A': 0.2} if i % 6 else {},
            'time_layer': ['T-24h', 'T-6h', 'T-1h', 'T-15min', 'final', None][i % 6],
            'asian_line': [-0.5, 0.25, 1.0, None][i % 4],
            'total_line': [2.25, 2.5, 3.0, None][i % 4],
            'odds_snapshot': {'asian': {'handicap': 0.8}} if i % 3 else {},
            'result_quality': {'score': 1.0, 'grade': ['high', 'medium', 'low', 'reject'][i % 4],
                               'source': ['live_fid', 'live_team', 'shuju'][i % 3],
                               'reasons': [], 'usable_for_calibration': i % 2 == 0} if i % 4 else {},
            'exclude_from_calibration': i % 11 == 0,
            'half_time_data_quality': 'real' if i % 2 else 'estimated',
            'updated_at': f'2026-08-{10 + i % 18:02d}T20:00:00',
        })

    # ---- sample_quality ----
    for i, rec in enumerate(RECORDS):
        yield from _y('sample_quality', 'assess_record_quality', str(i), dict(rec))
        yield from _y('sample_quality', '_is_friendly', str(i), dict(rec))
        yield from _y('sample_quality', '_has_any', str(i), dict(rec), ('actual_score', 'league'))
    for grade in ('reject', 'low', 'medium', 'high'):
        yield from _y('sample_quality', 'filter_quality_records', grade, [dict(r) for r in RECORDS], grade)

    # ---- prediction_policy ----
    for league in ('英超', '德乙', '欧冠杯', '友谊赛', '', None):
        yield from _y('prediction_policy', '_league_text', str(league), {'league': league})
        yield from _y('prediction_policy', 'get_prediction_policy', str(league), league)
    for line in (0.5, 2.25, 2.5, 3.0, 4.5, None, -1.0):
        yield from _y('prediction_policy', 'get_total_bucket', str(line), line)
        yield from _y('prediction_policy', 'get_handicap_bucket', str(line), line)
        for h in (-0.5, 0.25, None):
            yield from _y('prediction_policy', 'policy_bucket_key', f'{line}/{h}', h, line)
    MATRIX = {'1-0': 0.2, '2-1': 0.3, '0-0': 0.1, '1-1': 0.25, '3-0': 0.15}
    yield from _y('prediction_policy', 'normalize_score_matrix', 'm', dict(MATRIX))
    yield from _y('prediction_policy', 'normalize_score_matrix', 'empty', {})
    for w in (0.0, 0.3, 1.0):
        yield from _y('prediction_policy', 'blend_score_matrices', str(w), dict(MATRIX),
            {'2-1': 1.0}, w)
    SCENARIOS = [((1, 0), 0.2), ((2, 1), 0.3), ((0, 0), 0.1), ((1, 1), 0.25), ((3, 0), 0.15)]
    for n in (1, 3, 5, 10):
        yield from _y('prediction_policy', 'select_diverse_score_scenarios', str(n), SCENARIOS, n)
    yield from _y('prediction_policy', 'select_diverse_score_scenarios', 'empty', [], 3)
    for params in ({'draw_bias': 0.1}, {'market_db_weight': 0.5},
                   {'draw_bias': 99}, {'draw_bias': -99}, {'draw_bias': 'x'},
                   {}, {'unknown': 1}):
        yield from _y('prediction_policy', '_canonical_params', str(params), dict(params))
    yield from _y('prediction_policy', '_empty_tuning_config', '-')

    # ---- history_calibration ----
    for i, rec in enumerate(RECORDS):
        yield from _y('history_calibration', '_quality_weight', str(i), dict(rec))
        yield from _y('history_calibration', '_normalized_scores', str(i), dict(rec))
        yield from _y('history_calibration', '_outcome', str(i), (i % 4, i % 3))
        yield from _y('history_calibration', '_score_tuple', str(i), rec.get('actual_score'))
    SCORE_ROWS = [(('H', 'x'), {(1, 0): 0.3, (2, 1): 0.2, (1, 1): 0.5}, 1.0),
                  (('A', 'y'), {(0, 2): 0.6, (1, 1): 0.4}, 0.35)]
    for beta in (0.0, 0.05, 0.18, 0.5):
        yield from _y('history_calibration', '_mean_after_beta', str(beta), SCORE_ROWS, beta)
        yield from _y('history_calibration', '_mean_after_beta', f'{beta}/skew', SCORE_ROWS, beta,
            {'H': 1.4, 'D': 0.8, 'A': 1.0})
    for cut in (0, 20, 60, len(RECORDS)):
        yield from _y('history_calibration', 'estimate_history_calibration', str(cut),
            [dict(r) for r in RECORDS[:cut]])
    PROFILE = calibration_history.estimate_history_calibration([dict(r) for r in RECORDS])
    yield from _y('history_calibration', 'apply_history_calibration', 'p', dict(MATRIX), PROFILE)
    yield from _y('history_calibration', 'apply_history_calibration', 'empty', dict(MATRIX), {})

    # ---- dynamic_weights ----
    for c in (0.0, 0.15, 0.3, 0.31, 0.45, 0.5, 0.55, 0.69, 0.7, 0.85, 1.0):
        yield from _y('dynamic_weights', 'get_dynamic_weights', str(c), c)
        yield from _y('dynamic_weights', 'fuse_predictions', str(c), dict(MATRIX),
            {'2-1': 0.6, '0-0': 0.4}, {'1-1': 1.0}, None, c)

    # ---- backtest ----
    for i, rec in enumerate(RECORDS):
        yield from _y('backtest', '_actual_goals', str(i), rec.get('actual_score'))
        yield from _y('backtest', '_is_draw_score', str(i), rec.get('actual_score'))
        yield from _y('backtest', '_result_quality_is_usable', str(i), dict(rec))
        yield from _y('backtest', '_has_real_half_full_sample', str(i), dict(rec))
        yield from _y('backtest', '_normalize_1x2_probs', str(i), dict(rec.get('predicted_1x2') or {}))
        yield from _y('backtest', '_normalize_goal_distribution', str(i), {'0': 0.2, '1': 0.5, '2': 0.3})
    yield from _y('backtest', '_records_with_actual_scores', '-', [dict(r) for r in RECORDS])
    for grade in ('reject', 'medium', 'high'):
        yield from _y('backtest', '_quality_filter', grade, [dict(r) for r in RECORDS], True, grade)
    yield from _y('backtest', 'run_backtest', '-', [dict(r) for r in RECORDS])
    yield from _y('backtest', 'run_backtest_report', '-', [dict(r) for r in RECORDS])
    REPORT = backtest.run_backtest_report([dict(r) for r in RECORDS])
    yield from _y('backtest', 'get_diagnostic_tuning_suggestions', '-', REPORT)
    yield from _y('backtest', 'build_diagnostic_tuning_plan', '-', REPORT)
    for w in ((5,), (20, 60), (30, 60, 90)):
        yield from _y('backtest', 'rolling_backtest_report', str(w), [dict(r) for r in RECORDS], w)
    yield from _y('backtest', '_expand_param_grid', '-', {'a': [1, 2], 'b': [3]})
    yield from _y('backtest', '_objective_score', '-', REPORT)

