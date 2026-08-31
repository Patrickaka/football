# -*- coding: utf-8 -*-
"""生成报告层（监控 / 闸门 / 就绪度 / 验证 / 临场 / 渲染）的黄金语料。

**时钟固定在 `NOW`**：`assess_live_context` 判「这条情报够不够新」要跟当前
时间比，不注入的话黄金隔天就红。记录语料复用回测族那份冻结的历史。
"""
import gzip
import json
import pathlib
import random
from datetime import datetime, timedelta, timezone

from src.domain.sports.football import accuracy_gate, context, league_gate
from src.domain.sports.football import monitoring, readiness, reporting
from src.domain.sports.football import stats, validation
from tests.domain.golden import describe_exception

NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

MODULES = {
    'professional_monitoring': monitoring, 'professional_baseline': monitoring,
    'production_league_gate': league_gate, 'professional_readiness': readiness,
    'professional_validation': validation, 'accuracy_gate': accuracy_gate,
    'live_context_quality': context, 'contextual_fusion': context,
    'bayes_report': reporting,
}

_SEEN = {}


def _key(fn, label):
    """键必须唯一——同名 label 会互相覆盖，黄金里少掉的条目一声不响。"""
    base = f'{fn}:{label}'
    _SEEN[base] = _SEEN.get(base, 0) + 1
    return base if _SEEN[base] == 1 else f'{base}#{_SEEN[base]}'


def _resolve(mod, fn):
    """`wilson_interval` 住在 `stats`——转发清单要对着领域层的真实导出核。"""
    target = MODULES[mod]
    return getattr(target, fn) if hasattr(target, fn) else getattr(stats, fn)


def _serialisable(value):
    """`datetime` 存进黄金会被 `default=str` 转成字符串，读回来就对不上了——
    在这里先转成 ISO 文本，两边形状才一致。
    """
    if isinstance(value, datetime):
        return value.isoformat(sep=' ')
    if isinstance(value, dict):
        return {k: _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    return value


def _y(mod, fn, label, *args, **kwargs):
    name = f'{mod}.{fn}'
    try:
        yield _key(name, label), _serialisable(_resolve(mod, fn)(*args, **kwargs))
    except Exception as exc:
        yield _key(name, label), describe_exception(exc)


def entries():
    _SEEN.clear()
    RECORDS = json.load(gzip.open(
        pathlib.Path(__file__).resolve().parents[1]
        / 'tests/fixtures/football_backtest_records.json.gz', 'rt', encoding='utf-8'))
    random.seed(20260829)

    PROBS = {'H': 0.55, 'D': 0.25, 'A': 0.20}
    ODDS = {'H': 1.9, 'D': 3.4, 'A': 4.2}
    SPF_RECORDS = []
    for i, base in enumerate(RECORDS):
        SPF_RECORDS.append(dict(
            base,
            league=['英超', '德乙', '西甲', '意甲', ''][i % 5],
            probabilities={'H': 0.30 + (i % 7) * 0.06, 'D': 0.30, 'A': 0.40 - (i % 7) * 0.06},
            spf_probabilities={'H': 0.4, 'D': 0.3, 'A': 0.3},
            rqspf_probabilities={'H': 0.35, 'D': 0.30, 'A': 0.35},
            odds=dict(ODDS), spf_odds=dict(ODDS),
            rqspf_odds={'H': 2.0, 'D': 3.2, 'A': 3.9},
            actual='HDA'[i % 3], actual_spf='HDA'[i % 3],
            actual_result='HDA'[i % 3], actual_score=f'{i % 4}-{i % 3}',
            lottery_handicap=[-1, 0, 1][i % 3],
            professional_snapshot={'accuracy_gate': {'spf': {
                'candidate': '胜平负'[i % 3],
                'market_probability': 0.35 + (i % 6) * 0.09,
                'probability': 0.4 + (i % 6) * 0.08,
                'reasons': ['样本充足'] if i % 2 else [],
                'eligible': i % 2 == 0}}},
            rqspf_actual='HDA'[(i + 1) % 3],
            handicap=[-0.5, 0.0, 0.25, 1.0][i % 4],
            settled=True,
        ))

    LIVE = {'lineup': {'confirmed': True, 'verified': True, 'source': '官方',
                       'ts': (NOW - timedelta(hours=2)).isoformat(),
                       'updated_at': (NOW - timedelta(hours=2)).isoformat()},
            'injuries': [{'player': '9号', 'team': '主', 'role': '前锋',
                          'status': '伤停', 'impact': '大', 'source': '官方',
                          'ts': (NOW - timedelta(hours=1)).isoformat(),
                          'verified': True,
                          'updated_at': (NOW - timedelta(hours=1)).isoformat()}],
            'weather': {'updated_at': (NOW - timedelta(hours=30)).isoformat()},
            'possession': {'home': 0.62, 'away': 0.38},
            'motivation': {'home': 'must_win', 'away': 'nothing'},
            'h2h': {'home_wins': 4, 'draws': 1, 'away_wins': 1}}

    # ---- stats / monitoring ----
    for hits, n in ((0, 0), (0, 10), (8, 10), (10, 10), (5, 100), (-1, 10)):
        yield from _y('professional_monitoring', 'wilson_interval', f'{hits}/{n}', hits, n)
        yield from _y('professional_monitoring', 'wilson_interval', f'{hits}/{n}/z1', hits, n, 1.0)
    for market in ('spf', 'rqspf'):
        for width in (0.05, 0.1, 0.25):
            yield from _y('professional_monitoring', 'calibration_report', f'{market}/{width}',
                SPF_RECORDS, market, width)
    yield from _y('professional_monitoring', 'calibration_report', 'empty', [])
    yield from _y('professional_monitoring', 'upset_alert_report', '-', SPF_RECORDS)
    yield from _y('professional_monitoring', 'upset_alert_report', 'empty', [])
    for recent, base in ((10, 40), (50, 200)):
        yield from _y('professional_monitoring', 'build_professional_monitoring', f'{recent}/{base}',
            SPF_RECORDS, recent, base)
    yield from _y('professional_monitoring', '_window_metrics', '-', SPF_RECORDS[:20])
    for i, rec in enumerate(SPF_RECORDS[:20]):
        yield from _y('professional_monitoring', '_actual_rqspf', str(i), rec)
    for labels in (('H', 'D', 'A'), ('H', 'A')):
        yield from _y('professional_monitoring', '_normalise', str(labels), dict(PROBS), labels)
        yield from _y('professional_monitoring', '_normalise', f'{labels}/none', None, labels)
    yield from _y('professional_baseline', 'bundled_professional_baseline', '-')

    # ---- league_gate ----
    for value in ('胜', '平', '负', 'H', '', None, 0):
        yield from _y('production_league_gate', '_candidate_label', f'label/{value!r}', value)
    for league in ('英超', '德乙', '', None, '  英超  '):
        yield from _y('production_league_gate', '_normalise_league', str(league), league)
        yield from _y('production_league_gate', '_candidate_label', str(league), league)
        yield from _y('production_league_gate', 'validate_league_spf_policy', str(league),
            SPF_RECORDS, league)
    yield from _y('production_league_gate', 'build_production_league_spf_policies', '-', SPF_RECORDS)
    yield from _y('production_league_gate', 'build_production_league_spf_policies', 'empty', [])
    for i, rec in enumerate(SPF_RECORDS[:20]):
        yield from _y('production_league_gate', '_gate_row', str(i), rec)
    ROWS = [r for r in (league_gate._gate_row(x) for x in SPF_RECORDS) if r]
    for threshold in (0.0, 0.5, 0.65, 0.9):
        yield from _y('production_league_gate', '_metrics', str(threshold), ROWS, threshold)

    # ---- validation ----
    for i, rec in enumerate(SPF_RECORDS[:20]):
        yield from _y('professional_validation', '_rqspf_actual', str(i), rec)
        yield from _y('professional_validation', '_rqspf_odds', str(i), rec)
        for w in (0.0, 0.3, 1.0):
            yield from _y('professional_validation', 'blend_record_with_market', f'{i}/{w}', rec, w)
    yield from _y('professional_validation', 'normalize_probabilities', '-', dict(PROBS))
    yield from _y('professional_validation', 'normalize_probabilities', 'zero', {'H': 0, 'D': 0, 'A': 0})
    yield from _y('professional_validation', 'probabilities_from_odds', '-', dict(ODDS))
    yield from _y('professional_validation', 'probabilities_from_odds', 'zero', {'H': 0, 'D': 0, 'A': 0})
    for key in ('probabilities', 'spf_probabilities', 'missing'):
        yield from _y('professional_validation', 'multiclass_metrics', key, SPF_RECORDS, key)
    for p, e in ((0.0, 0.0), (0.5, 0.03), (0.9, 0.5)):
        yield from _y('professional_validation', 'evaluate_strategy', f'{p}/{e}', SPF_RECORDS, p, e)
        yield from _y('professional_validation', 'evaluate_rqspf_records', f'{p}/{e}', SPF_RECORDS, p, e)
    yield from _y('professional_validation', 'select_threshold', '-', SPF_RECORDS)
    yield from _y('professional_validation', 'select_market_residual_weight', '-', SPF_RECORDS)
    for equity in ([1, 2, 1.5, 3, 0.5], [], [1], [3, 2, 1]):
        yield from _y('professional_validation', '_max_drawdown', str(equity), equity)
    for train, test in ((20, 10), (40, 20)):
        yield from _y('professional_validation', 'walk_forward_evaluate', f'{train}/{test}',
            SPF_RECORDS, train, test)
    yield from _y('professional_validation', '_normalize_labeled', '-', dict(PROBS), ('H', 'D', 'A'))

    # ---- readiness ----
    VALIDATION = validation.evaluate_strategy(SPF_RECORDS, 0.5, 0.03)
    MONITORING = monitoring.build_professional_monitoring(SPF_RECORDS)
    AUDITED = {'source': '官方', 'ts': NOW.isoformat()}
    for value in (None, '', 0, [], {}, 'x', 1.0, dict(AUDITED), [dict(AUDITED)],
                  [dict(AUDITED), {'source': '官方'}], {'source': '官方'}, {'ts': 'x'}):
        yield from _y('professional_readiness', '_present', repr(value)[:24], value)
        yield from _y('professional_readiness', '_live_item_verified', repr(value)[:24], value)
    yield from _y('professional_readiness', '_probability_divergence', '-', dict(PROBS), {'H': 0.4, 'D': 0.3, 'A': 0.3})
    yield from _y('professional_readiness', '_probability_divergence', 'none', None, None)
    RESULT = {'probabilities': dict(PROBS), 'odds': dict(ODDS), 'live': dict(LIVE),
              'league': '英超', 'confidence': {'score': 0.7}, 'match_id': 'm1'}
    yield from _y('professional_readiness', 'build_match_evidence_profile', '-', RESULT)
    yield from _y('professional_readiness', 'build_match_evidence_profile', 'empty', {})
    yield from _y('professional_readiness', 'build_system_gap_assessment', '-', VALIDATION)
    EVIDENCE = readiness.build_match_evidence_profile(RESULT)
    yield from _y('professional_readiness', 'build_professional_decision_gate', '-', VALIDATION, EVIDENCE)
    yield from _y('professional_readiness', 'build_professional_decision_gate', 'none', None)

    # ---- accuracy_gate ----
    LOTTERY = {'probabilities': dict(PROBS), 'prediction': 'H', 'league': '英超',
               'odds': dict(ODDS)}
    POLICIES = league_gate.build_production_league_spf_policies(SPF_RECORDS)
    for league in ('西甲', 'SP1', 'La Liga', '德甲', 'Bundesliga', '英超', '', None):
        yield from _y('accuracy_gate', 'has_static_spf_policy', str(league), league)
        yield from _y('accuracy_gate', '_static_spf_policy', str(league), league)
        yield from _y('accuracy_gate', 'build_accuracy_gate', str(league), dict(LOTTERY, league=league))
        yield from _y('accuracy_gate', '_spf_policy', str(league), league)
        yield from _y('accuracy_gate', '_league_policy', str(league), league, POLICIES)
    yield from _y('accuracy_gate', 'build_accuracy_gate', 'with_conf', LOTTERY,
        confidence={'score': 0.8}, anomaly={'level': 'high'})
    yield from _y('accuracy_gate', 'build_total_goals_gate', '-',
        {'probabilities': {'over': 0.6, 'under': 0.4}, 'prediction': 'over'}, league='英超')
    yield from _y('accuracy_gate', 'build_total_goals_gate', 'none', None)
    for probs in (dict(PROBS), {}, None, {'H': 0.34, 'D': 0.33, 'A': 0.33}):
        yield from _y('accuracy_gate', '_top_pick', str(probs)[:20], probs)
    for market in ({'H': 1.9, 'D': 3.4, 'A': 4.2}, None, {}):
        for pick in ('H', 'A', None):
            yield from _y('accuracy_gate', '_market_agrees', f'{market}/{pick}', market, pick)
    for prob in (0.0, 0.4, 0.65, 1.0):
        for completeness in (0.0, 0.5, 1.0):
            yield from _y('accuracy_gate', 'prediction_reliability', f'{prob}/{completeness}',
                prob, completeness)

    # ---- context ----
    for hours in (0.5, 3, 25, 100):
        ctx = {'lineup': {'confirmed': True, 'verified': True,
                          'updated_at': (NOW - timedelta(hours=hours)).isoformat()}}
        yield from _y('live_context_quality', 'assess_live_context', f'{hours}', ctx, NOW)
        yield from _y('live_context_quality', 'assess_live_context', f'{hours}/strict', ctx, NOW, True)
        yield from _y('live_context_quality', 'assess_live_context', f'{hours}/12h', ctx, NOW, False, 12.0)
    yield from _y('live_context_quality', 'assess_live_context', 'full', LIVE, NOW)
    yield from _y('live_context_quality', 'assess_live_context', 'empty', {}, NOW)
    for value in (NOW.isoformat(), '2026-08-29T12:00:00Z', 'bad', '', None, 123):
        yield from _y('live_context_quality', '_parse_timestamp', repr(value)[:20], value)
    CANDIDATES = [((1, 0), 0.25), ((2, 1), 0.30), ((1, 1), 0.20), ((0, 1), 0.15), ((3, 0), 0.10)]
    for ctx in (LIVE, {}, {'motivation': {'home': 'must_win', 'away': 'nothing'}},
                {'h2h': {'home_wins': 5, 'draws': 0, 'away_wins': 0}}):
        yield from _y('contextual_fusion', 'apply_contextual_fusion', str(ctx)[:24],
            list(CANDIDATES), ctx)
    for value in ('1.5', 1.5, None, '', 'x', []):
        yield from _y('contextual_fusion', '_number', repr(value)[:14], value)
        yield from _y('contextual_fusion', '_number', f'{value!r}/d', value, 9.0)

    # ---- reporting ----
    MODULE = {'wdl': {'home': 0.55, 'draw': 0.25, 'away': 0.20}, 'risk': {'level': 'medium', 'reasons': ['x']},
              'league_profile': {'avg_goals': 2.7, 'home_advantage': 0.3},
              'euro_open': {'home': 1.9, 'draw': 3.4, 'away': 4.2}, 'scores': {'1-0': 0.2, '2-1': 0.3}}
    IMPLIED = {'home': 1.9, 'draw': 3.4, 'away': 4.2}
    for odds in (dict(IMPLIED), {'胜': 1.9, '平': 3.4, '负': 4.2},
                 {'home': 0.5, 'draw': 0.3, 'away': 0.2}, {}, None,
                 {'home': 0, 'draw': 0, 'away': 0}):
        yield from _y('bayes_report', 'derive_prior_p0', str(odds)[:24], odds)
    for x in (0.0, 0.5, 1.0, 0.12345, -0.1):
        yield from _y('bayes_report', 'pct', str(x), x)
    for odds in (dict(IMPLIED), {'胜': 1.9, '平': 3.4, '负': 4.2},
                 {'home': 0.5, 'draw': 0.3, 'away': 0.2},
                 {'home': 0, 'draw': 0, 'away': 0}, {}, None, {'home': 'x'}):
        yield from _y('bayes_report', '_to_implied', str(odds)[:24], odds)
    yield from _y('bayes_report', 'tactical_context', '-', LIVE)
    yield from _y('bayes_report', 'tactical_context', 'empty', {})
    yield from _y('bayes_report', 'possession_trap_warning', '-', LIVE)
    yield from _y('bayes_report', 'possession_trap_warning', 'empty', {})
    WDL = {'home': 0.55, 'draw': 0.25, 'away': 0.20}
    for league in ('英超', '德乙', ''):
        yield from _y('bayes_report', 'league_specifics', league, MODULE['league_profile'], league)
        yield from _y('bayes_report', 'likelihood_update', league, dict(WDL), LIVE, league)
        yield from _y('bayes_report', 'likelihood_update', f'{league}/nolive', dict(WDL), {}, league)
    TACTICAL = reporting.tactical_context(LIVE)
    yield from _y('bayes_report', 'build_scripts', '-', MODULE, TACTICAL, '主队', '客队')
    yield from _y('bayes_report', 'risk_list', '-', MODULE['risk'], TACTICAL, LIVE)
    yield from _y('bayes_report', '_beidan_p0_p1', '-',
        {'euro_open': dict(IMPLIED), 'wdl': dict(WDL)})
    yield from _y('bayes_report', '_beidan_scripts', '-', {'wdl': dict(WDL)}, TACTICAL, '主队', '客队')
    for path in ('/a/b/reports/football_bayes_12345.html',
                 'beidan_bayes_abc.html', '/x/football_12345.html', 'x.html', ''):
        yield from _y('bayes_report', '_extract_mid_from_report_path', path or 'empty', path)
        yield from _y('bayes_report', 'report_url_from_path', path or 'empty', path, '12345')

