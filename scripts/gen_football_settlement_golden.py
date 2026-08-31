# -*- coding: utf-8 -*-
"""生成赛果判定与结算质量的黄金语料。

**时间解析的「当前年」注入固定时钟**——不注入的话黄金跨年就红。
"""
import inspect
import itertools
import re as _re
from datetime import datetime, timedelta

from src.domain.sports.football import settlement as new
from tests.domain.golden import describe_exception

NOW = datetime(2026, 8, 28, 12, 0, 0)


def _nan_safe(value):
    """NaN 不等于自身——存成字符串 `'nan'`，否则黄金比对永远红。

    `calculate_logloss` 对非 `H`/`D`/`A` 的赛果**故意返回 NaN**，
    所以这不是边角情况，是正常路径上的返回值。
    """
    if isinstance(value, float) and value != value:
        return 'nan'
    if isinstance(value, datetime):
        return value.isoformat(sep=' ')
    if isinstance(value, dict):
        return {k: _nan_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_safe(v) for v in value]
    return value


_SEEN = {}


def _key(fn, label):
    """键必须唯一——同名 label 会在字典里互相覆盖，
    黄金里少掉的那几十条比对时**一声不响**。
    """
    base = f'{fn}:{label}'
    _SEEN[base] = _SEEN.get(base, 0) + 1
    return base if _SEEN[base] == 1 else f'{base}#{_SEEN[base]}'


def _clock_kwargs(func, given):
    """签名里带 `now` 就自动注入固定时钟。

    漏一个就是「本地绿、CI 红」——CI 跑在 UTC，本地跑在东八区，
    `_assess_result_quality` 判「比赛到期没」的结果两边不一样。
    按名字维护一张清单迟早会漏，按签名判才不会。
    """
    if 'now' in given:
        return {}
    try:
        return {'now': NOW} if 'now' in inspect.signature(func).parameters else {}
    except (TypeError, ValueError):
        return {}


def _y(fn, label, *a, **kw):
    func = getattr(new, fn)
    kw = dict(kw, **_clock_kwargs(func, kw))
    try:
        yield _key(fn, label), _nan_safe(func(*a, **kw))
    except Exception as exc:
        yield _key(fn, label), describe_exception(exc)


def _yc(fn, label, args_old, args_new):
    """注入型函数：黄金只记新实现在**固定时钟**下的结果"""
    func = getattr(new, fn)
    fixed = tuple(NOW if isinstance(x, datetime) else x for x in args_new)
    try:
        yield _key(fn, label), _nan_safe(func(*fixed, **_clock_kwargs(func, {})))
    except Exception as exc:
        yield _key(fn, label), describe_exception(exc)


def entries():
    _SEEN.clear()
    PROBS = [{'H':0.5,'D':0.3,'A':0.2},{'H':0.34,'D':0.33,'A':0.33},
             {'H':0.9,'D':0.05,'A':0.05},{'H':0.0,'D':0.0,'A':1.0},
             {'H':0,'D':0,'A':0},{}]
    for i,p in enumerate(PROBS):
        yield from _y('normalize_1x2_probs',str(i),dict(p))
        for res in ('H','D','A','home','',None):
            yield from _y('calculate_logloss',f'{i}/{res}',dict(p),res)
            yield from _y('calculate_brier_score',f'{i}/{res}',dict(p),res)
    for score in ('2-1','0-0','1-2','','bad',None,'10-0','2:1'):
        yield from _y('_score_to_result',repr(score),score)
        yield from _y('_parse_score_string',repr(score),score)
    for i,p in enumerate(PROBS):
        for res in ('H','D','A','',None):
            yield from _y('calculate_hit',f'{i}/{res}',dict(p),res)
    import re as _re
    for raw in ('2-1','比分 3:0','','bad',None):
        m = _re.search(r'(\d+)\s*[-:]\s*(\d+)', str(raw) if raw else '')
        yield from _y('_parse_score_result',repr(raw),m)
    for mid in ('1430311','','abc',None,'12',12345):
        yield from _y('_is_valid_match_id',repr(mid),mid)
    NOW = datetime(2026,8,28,12,0,0)
    def run_clocked(fn, label, args_old, args_new):
        '''注入型函数：旧的不带 now、新的带——两边喂同一个真实时钟'''
        def call(m, args):
            try: return ('ok', _norm(getattr(m,fn)(*args)), 1)
            except Exception as e: return ('exc', describe_exception(e), None)
        o, n = call(old, args_old), call(new, args_new)
        c = cov[fn]; c['n'] += 1; c['exc' if n[0]=='exc' else 'ok'] += 1
        if o[:2] != n[:2]: diffs.append((f'{fn}:{label}', o[1], n[1]))

    for t in ('2026-08-28 20:00:00','2026-08-28 20:00','2026/08/28 20:00','08-28 20:00',
              '12-31 20:00','01-01 20:00','','bad',None):
        real_now = NOW
        yield from _yc('_parse_match_datetime',repr(t),(t,),(t, real_now))
        yield from _yc('_is_match_settle_due',repr(t),(t,180),(t,180,real_now))
        yield from _yc('infer_time_layer',repr(t),(t,),(t, real_now))
        yield from _yc('_live_query_dates',repr(t),(t,),(t, real_now))
    for layer in ('T24h','T6h','T2h','final','unknown',None):
        yield from _y('time_layer_weight',repr(layer),layer)
    for a,b,w in itertools.product(PROBS[:3],PROBS[:3],(0.0,0.3,0.5,1.0)):
        yield from _y('fuse_probabilities',f'{w}',dict(a),dict(b),w)
    STATS = [{'count':0},{'count':50,'logloss':1.0,'brier':0.2},
             {'count':200,'logloss':0.9,'brier':0.18},{'count':200,'logloss':1.2,'brier':0.3},{}]
    for i,st in enumerate(STATS):
        for n_ in (0,50,200,1000):
            yield from _y('check_ml_fusion_eligibility',f'{i}/{n_}',dict(st),n_)
    for elig,shadow,extra in itertools.product((True,False),(0,50,200,1000),(0.0,0.1)):
        yield from _y('get_ml_fusion_weight',f'{elig}/{shadow}',elig,shadow,extra)
    for i,q in enumerate(({'usable':True},{'usable':False},{},None)):
        yield from _y('_is_result_quality_usable',str(i),q)
    # 结算质量 / 决策快照 / 内容签名
    REC = {'match_id':'1430311','home':'A','away':'B','match_time':'2026-08-28 20:00:00',
           'predicted_probs':{'H':0.5,'D':0.3,'A':0.2},'actual_score':'2-1','settled':True,
           'recommend':[{'home':2,'away':1,'prob':0.12}],'created_at':'2026-08-27 10:00:00',
           'sample_count':50,'source':'model'}
    for i,kw in enumerate(({}, {'actual_score':''}, {'settled':False}, {'match_time':''},
                           {'predicted_probs':{}}, {'sample_count':0})):
        rec = dict(REC, **kw)
        for src_ in (None, 'live', 'shuju'):
            yield from _y('_assess_result_quality',f'{i}/{src_}',rec,'2-1','H',src_)
        yield from _y('_prediction_decision_snapshot',str(i),{'H':0.5,'D':0.3,'A':0.2})
        for prof in (None, {'grade':'A','score':0.8}):
            yield from _y('_audited_decision_snapshot',f'{i}/{bool(prof)}',{'H':0.5,'D':0.3,'A':0.2},prof)
        yield from _y('_prediction_content_sig',str(i),
            [{'home':2,'away':1,'prob':0.12}], {'H':0.5,'D':0.3,'A':0.2},
            {'handicap':0.5}, 2.5, {'home':2.0}, [{'code':'HH','p':0.3}], 'v2')
        yield from _y('_calibration_sample_weight',str(i),rec)
        for ml_on in (True, False):
            yield from _y('evaluate_ml_prediction',f'{i}/{ml_on}',
                dict(rec, actual_result='H', base_1x2={'H':0.45,'D':0.3,'A':0.25},
                     ml_1x2={'H':0.6,'D':0.25,'A':0.15}, ml_available=ml_on))
        yield from _y('evaluate_ml_prediction',f'{i}/no_result', dict(rec, actual_result=''))
    # HTML 抽取
    for raw in ('2-1', '2 - 1', '比分 3:0', '(2-1)', '', 'bad', None):
        yield from _y('_extract_score_text',repr(raw),raw)
    SHUJU = ('<a href="shuju-1430311.shtml" class="x">x</a>'
             '<em class="l">A队</em><span class="gray">2-1</span><em class="r">B队</em>')
    LIVE_ROW = ('<div class="pk"><a class="clt1">2</a><span>-</span><a class="clt3">1</a></div>'
                '<td class="red">0-1</td>')
    for h in (SHUJU, '<div></div>', '', None):
        yield from _y('_parse_shuju_score',repr(h)[:14],h,'1430311')
    LIVE_TEAM_ROW = ('<td>A队</td><td class="pk"><a class="clt1">2</a>'
                     '<a class="clt3">1</a></td><td>B队</td>')
    for row in (LIVE_ROW, LIVE_TEAM_ROW, '<td></td>', '', None):
        yield from _y('_parse_live_row_final_score',repr(row)[:16],row)
        yield from _y('_parse_live_row_score',repr(row)[:16],row,'A队','B队')

