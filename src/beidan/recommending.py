# -*- coding: utf-8 -*-
"""北单玩法分析与推荐生成、快照、CLI"""

import sys
import math
import re
from collections import defaultdict
import time
import json
import urllib.request
import urllib.error
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from ..common.logger import setup_logger
from ..common.paths import data_path
from ..common import kv_store

log = setup_logger('beidan')

from .config import (
    BASE_URL, BEIDAN_VERSION, BET_TYPES, LEAGUE_PROFILES, MAX_GOALS,
)

# ─── 领域层适配 ───
#
# 赛果解读、分组、筛选的算法在 `src/domain/sports/beidan/analysis.py`。
# 留在这里的是抓取、缓存、时钟与快照写入。
from src.domain.sports.beidan import analysis as _analysis


from .modeling import (
    _profile as _league_profile,
    BEIDAN_STRONG_MIN_LEAD, BEIDAN_STRONG_MIN_PROBABILITY, aggregate_goals_from_scores, anchor_score_outcomes, build_dixon_coles_matrix, calibrate_draw_probability, match_lambdas, match_target_total, parse_beidan_handicap, predict_scores_by_poisson, rqspf_probs_from_score_probs,
)
from .fetching import (
    fetch_beidan_schedule, fetch_json, fetch_zgzcw_asian_history,
    fetch_zgzcw_cs_history, fetch_zgzcw_goals_history, fetch_zgzcw_schedule,
)
from .markets import (
    _beidan_market_snapshot, _latest_ou_market, adjust_zjq_by_goals, analyze_asian_trend, analyze_cs_trend, analyze_goals_trend, apply_beidan_joint_market_state, build_beidan_market_admission, build_water_market_prediction, calculate_asian_goal_factor, calculate_goals_factor, enhance_scores_with_cs,
)
from .schedules import (
    fetch_beidan_bifen, fetch_beidan_bqc, fetch_beidan_zjq,
)
from .settling import (
    _actual_rqspf_from_record, _actual_zjq_from_record, _beidan_record_key, _load_beidan_history, _save_beidan_history, apply_beidan_history_calibration, calculate_implied_probability,
)
from .quality import (
    assess_recommendation_quality,
)
from .upset import (
    _result_from_score, assess_score_consistency, assess_upset_risk, pick_upset_scores,
)

# ─── 领域层适配（推荐组装）───
#
# 四种玩法的组装顺序在 `src/domain/sports/beidan/recommendation.py`。
# 那一层不认识配置，所以把**已经配好配置的操作**打成两组注入进去：
# `_MODEL` 是比分模型（联赛档案、DC 的 rho、最大进球数、锚定强度都在里面），
# `_MARKET` 是盘口层（十几个阈值都在里面）。
#
# 属性名与领域层用到的名字必须一一对上，拼错要到运行时才炸——
# `test_recommendation.py` 里有一道按 AST 比对的守卫盯着这件事。
import types

from src.domain.sports.beidan import recommendation as _recommendation

_MODEL = types.SimpleNamespace(
    predict_scores=predict_scores_by_poisson,
    calibrate_draw=calibrate_draw_probability,
    league_profile=_league_profile,
    target_total=match_target_total,
    lambdas=match_lambdas,
    score_matrix=build_dixon_coles_matrix,
    anchor_outcomes=anchor_score_outcomes,
    aggregate_goals=aggregate_goals_from_scores,
    rqspf_from_scores=rqspf_probs_from_score_probs,
)

_MARKET = types.SimpleNamespace(
    latest_total=_latest_ou_market,
    apply_joint=apply_beidan_joint_market_state,
    analyze_asian=analyze_asian_trend,
    analyze_goals=analyze_goals_trend,
    analyze_correct_score=analyze_cs_trend,
    blend_scores_with_market=enhance_scores_with_cs,
    asian_goal_factor=calculate_asian_goal_factor,
    goals_factor=calculate_goals_factor,
    adjust_goal_buckets=adjust_zjq_by_goals,
)


def _history_calibration(probabilities, bet_type, league=None):
    """把历史校准包成领域层要的形状。

    **仍然走模块全局**：黄金生成脚本与测试都靠打桩 `apply_beidan_history_calibration`
    来隔开存储，改成直接引用函数对象会让那些桩失效。
    """
    return apply_beidan_history_calibration(probabilities, bet_type, league=league)


def build_zjq_group_recommendation(zjq_probs):
    """总进球的三个可投注分组。分组定义在领域层，这里只透传。"""
    return _analysis.zjq_groups(zjq_probs)


def _compact_beidan_record(match, source, professional_snapshot=None):
    spf = match.get('spf') or {}
    zjq = match.get('zjq') or {}
    rqspf = match.get('rqspf') or {}
    return {
        'key': _beidan_record_key(match),
        'source': source,
        'match_id': match.get('match_id'),
        'date': match.get('date'),
        'num': match.get('num'),
        'time': match.get('time'),
        'league': match.get('league'),
        'home': match.get('home'),
        'away': match.get('away'),
        'handicap': match.get('handicap'),
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'settled': False,
        'professional_snapshot': professional_snapshot,
        'spf': {
            'prediction': spf.get('prediction'),
            'confidence': spf.get('confidence'),
            'quality': spf.get('quality'),
            'probabilities': spf.get('probabilities'),
            'score_consistency': spf.get('score_consistency'),
            'odds': dict(spf.get('odds') or {}),
        },
        'zjq': {
            'prediction': zjq.get('prediction'),
            'confidence': zjq.get('confidence'),
            'quality': zjq.get('quality'),
            'goal_groups': zjq.get('goal_groups'),
            'probabilities': zjq.get('probabilities'),
            'market_admission': zjq.get('market_admission'),
            'accuracy_gate': zjq.get('accuracy_gate'),
        },
        'rqspf': {
            'prediction': rqspf.get('prediction'),
            'confidence': rqspf.get('confidence'),
            'quality': rqspf.get('quality'),
            'probabilities': rqspf.get('probabilities'),
            'market_admission': rqspf.get('market_admission'),
            'odds': dict(rqspf.get('odds') or {}),
        },
        'market_snapshot': _beidan_market_snapshot(match),
    }


def save_beidan_prediction_snapshot(result):
    if not isinstance(result, dict) or 'error' in result:
        return {'saved': 0, 'total': 0}

    records = _load_beidan_history()
    by_key = {r.get('key'): r for r in records if r.get('key')}
    professional_snapshot = {
        'decision_gate': result.get('decision_gate'),
        'validation': result.get('professional_validation'),
        'gap_assessment': result.get('professional_gap_assessment'),
    }
    saved = 0
    for match in result.get('recommendations') or []:
        key = _beidan_record_key(match)
        if not key.strip('|'):
            continue
        compact = _compact_beidan_record(
            match, result.get('source'), professional_snapshot=professional_snapshot,
        )
        if key in by_key and by_key[key].get('settled'):
            compact['settled'] = True
            compact['actual'] = by_key[key].get('actual')
            compact['settlement'] = by_key[key].get('settlement')
            compact['created_at'] = by_key[key].get('created_at') or compact['created_at']
        previous = by_key.get(key, {})
        layers = list(previous.get('market_layers') or [])
        snapshot = compact.pop('market_snapshot')
        signature = json.dumps(
            {
                'asian': snapshot.get('asian'),
                'total': snapshot.get('total'),
                'spf_odds': snapshot.get('spf_odds'),
                'rqspf_odds': snapshot.get('rqspf_odds'),
            },
            sort_keys=True, ensure_ascii=False,
        )
        previous_signature = layers[-1].get('signature') if layers else None
        if signature != previous_signature:
            layers.append({**snapshot, 'signature': signature})
        compact['market_layers'] = layers[-30:]
        by_key[key] = {**previous, **compact}
        saved += 1

    persistence_backend = _save_beidan_history(list(by_key.values()))
    return {
        'saved': saved,
        'total': len(by_key),
        'persistence_backend': persistence_backend or 'unknown',
    }


def summarize_beidan_history(limit=200):
    records = _load_beidan_history()
    recent = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)[:limit]
    levels = {}
    for record in recent:
        for bet_type in ('spf', 'zjq', 'rqspf'):
            level = ((record.get(bet_type) or {}).get('quality') or {}).get('level') or 'unknown'
            levels[level] = levels.get(level, 0) + 1

    settled = [r for r in recent if r.get('settled')]
    market_accuracy = {
        'rqspf': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'zjq': {'total': 0, 'correct': 0, 'accuracy': 0.0},
    }
    for record in settled:
        for bet_type, actual_fn in (
            ('rqspf', _actual_rqspf_from_record),
            ('zjq', _actual_zjq_from_record),
        ):
            section = record.get(bet_type) or {}
            admission = section.get('market_admission') or {}
            if not admission.get('official'):
                continue
            actual = actual_fn(record)
            prediction = section.get('prediction')
            if actual is None or prediction is None:
                continue
            market_accuracy[bet_type]['total'] += 1
            market_accuracy[bet_type]['correct'] += int(str(prediction) == str(actual))
    for stats in market_accuracy.values():
        if stats['total']:
            stats['accuracy'] = round(stats['correct'] / stats['total'], 4)
    try:
        from ..football.professional_validation import evaluate_rqspf_records
        rqspf_rows = []
        for record in settled:
            actual = record.get('actual') if isinstance(record.get('actual'), dict) else {}
            settlement = record.get('settlement') if isinstance(record.get('settlement'), dict) else {}
            rqspf_rows.append({
                'actual_score': (
                    record.get('actual_score') or actual.get('score')
                    or actual.get('actual_score') or settlement.get('score')
                    or settlement.get('actual_score')
                ),
                'lottery_handicap': parse_beidan_handicap(record.get('handicap')),
                'predicted_rqspf': (record.get('rqspf') or {}).get('probabilities'),
                'rqspf_odds': (record.get('rqspf') or {}).get('odds'),
            })
        rqspf_professional = evaluate_rqspf_records(
            rqspf_rows, min_probability=0.65, min_edge=0.03,
        )
    except Exception as exc:
        rqspf_professional = {
            'market': 'rqspf', 'n': 0, 'production_ready': False,
            'reason': 'internal_error', 'error_type': type(exc).__name__,
        }
    return {
        'total_records': len(records),
        'recent_records': len(recent),
        'settled_records': len(settled),
        'pending_records': len(recent) - len(settled),
        'quality_levels': levels,
        'market_accuracy': market_accuracy,
        'rqspf_professional_validation': rqspf_professional,
        'latest': recent[:30],
    }


_ouzhi_cache = {}


def fetch_ouzhi_odds(match_id):
    match_id = str(match_id)
    if match_id in _ouzhi_cache:
        return _ouzhi_cache[match_id]
    
    url = f'{BASE_URL}/fenxi1/json/ouzhi.php?fid={match_id}&cid=0&type=europe&r=1'
    referer = f'{BASE_URL}/fenxi/ouzhi-{match_id}.shtml'
    
    try:
        series = fetch_json(url, referer=referer)
        if not isinstance(series, list) or len(series) == 0:
            _ouzhi_cache[match_id] = None
            return None
        
        close = series[0]
        if not isinstance(close, (list, tuple)) or len(close) < 3:
            _ouzhi_cache[match_id] = None
            return None
        
        result = {
            'home': float(close[0]),
            'draw': float(close[1]),
            'away': float(close[2]),
        }
        _ouzhi_cache[match_id] = result
        return result
    except Exception as e:
        log.warning(f"获取欧赔数据失败 match_id={match_id}: {e}")
        _ouzhi_cache[match_id] = None
        return None


def _clear_ouzhi_cache():
    """就地清空，**不重新绑定**。

    迁移前这里是 `global _ouzhi_cache; _ouzhi_cache = {}`，而
    `__init__.py` 有一行 `from .recommending import _ouzhi_cache`——
    重绑定之后那行导出的就是一份没人再写的孤儿副本（§五·2）。
    改成就地清空，导出与源模块从此指向同一个对象。
    """
    _ouzhi_cache.clear()


def build_beidan_match_analysis(spf_result):
    """本地 AI 式赛果分析（对齐足球模块），用于北单胜平负推荐。

    算法在 `src/domain/sports/beidan/analysis.py`。**兜异常留在这一层**：
    迁移前整个函数体裹在一个 `except Exception` 里，任何缺陷都会被压成
    一句 warning 加一个 `None`，与「这场数据不全」长得一模一样（判据 6）。
    领域层现在算不出来才返回 `None`，别的照抛；要不要吞由这里决定。
    """
    try:
        return _analysis.build_match_analysis(
            spf_result,
            min_single=BEIDAN_STRONG_MIN_PROBABILITY,
            min_margin=BEIDAN_STRONG_MIN_LEAD)
    except Exception as e:
        log.warning(f"build_beidan_match_analysis 失败: {e}")
        return None


def analyze_spf(match, asian_data=None, cs_data=None, goals_data=None):
    """胜平负推荐。抓欧赔在这里，组装在领域层。"""
    return _recommendation.spf(
        match, fetch_ouzhi_odds(match['id']), _MODEL, _MARKET,
        asian_data=asian_data, cs_data=cs_data, goals_data=goals_data,
        calibrate=_history_calibration)


def analyze_rqspf(match, asian_data=None, goals_data=None):
    """让球胜平负推荐。

    **让球值先解析、解析不出来就不抓欧赔**——迁移前那道守卫排在抓取之前，
    顺序保住：没有盘口的场次不该白白打一次网络。
    """
    handicap_value = parse_beidan_handicap(match.get('handicap'))
    ouzhi = fetch_ouzhi_odds(match['id']) if handicap_value is not None else None
    return _recommendation.rqspf(
        match, ouzhi, handicap_value, _MODEL, _MARKET,
        asian_data=asian_data, goals_data=goals_data,
        calibrate=_history_calibration)


def build_beidan_total_goals_accuracy_gate(section, goals_data, league):
    """Map Beidan HK-water O/U data into the frozen league O/U 2.5 gate."""
    line, over_water, under_water = _latest_ou_market(goals_data)
    market, model_over, model_under = _analysis.total_goals_gate_inputs(
        section, (over_water, under_water), max_goals=MAX_GOALS)

    from ..football.accuracy_gate import build_total_goals_gate
    return build_total_goals_gate(
        {"close_line": line, "close_prob": market},
        league=league,
        goal_count={"over_under": {"over": model_over, "under": model_under}},
    )


def analyze_bifen(match, bifen_odds=None, asian_data=None, goals_data=None):
    """比分推荐。市场报价按场次取出后传进去。"""
    market_odds = ((bifen_odds or {}).get(match['id'])
                   if isinstance(bifen_odds, dict) else None)
    return _recommendation.bifen(
        match, fetch_ouzhi_odds(match['id']), _MODEL, _MARKET,
        market_odds=market_odds, asian_data=asian_data, goals_data=goals_data,
        calibrate=_history_calibration)


def analyze_zjq(match, zjq_odds=None, asian_data=None, goals_data=None):
    """总进球推荐。市场报价按场次取出后传进去。"""
    market_odds = ((zjq_odds or {}).get(match['id'])
                   if isinstance(zjq_odds, dict) else None)
    return _recommendation.zjq(
        match, fetch_ouzhi_odds(match['id']), _MODEL, _MARKET,
        market_odds=market_odds, asian_data=asian_data, goals_data=goals_data,
        calibrate=_history_calibration)


def analyze_bqc(match, bqc_odds):
    """半全场推荐。纯计算，赔率由调用方取好后传进来。"""
    return _analysis.bqc(match, bqc_odds)


def _candidate_beidan_dates(date, allow_fallback=True, days=2):
    return _analysis.candidate_dates(date, allow_fallback=allow_fallback, days=days)


def _fetch_beidan_matches_with_fallback(date, source, allow_date_fallback=True):
    sources = [source]
    if source != 'zgzcw':
        sources.append('zgzcw')

    attempts = []
    for candidate_source in sources:
        for candidate_date in _candidate_beidan_dates(date, allow_date_fallback):
            if candidate_source == 'zgzcw':
                matches = fetch_zgzcw_schedule(candidate_date)
            else:
                matches = fetch_beidan_schedule(candidate_date, source=candidate_source)

            attempts.append({
                'source': candidate_source,
                'date': candidate_date,
                'match_count': len(matches),
            })
            if matches:
                return matches, {
                    'requested_source': source,
                    'requested_date': date,
                    'source': candidate_source,
                    'date': candidate_date,
                    'source_fallback': candidate_source != source,
                    'date_fallback': candidate_date != date,
                    'attempts': attempts,
                }

    log.error(f"所有数据源均失败，返回空结果。尝试记录: {attempts}")
    return [], {
        'requested_source': source,
        'requested_date': date,
        'source': source,
        'date': date,
        'source_fallback': False,
        'date_fallback': False,
        'attempts': attempts,
    }


def generate_beidan_recommendations(date=None, bet_types=None, source='zgzcw', save_history=True):
    if bet_types is None:
        bet_types = ['spf', 'rqspf']
    
    allow_date_fallback = not date
    date = date or time.strftime('%Y-%m-%d')
    matches, match_meta = _fetch_beidan_matches_with_fallback(
        date,
        source,
        allow_date_fallback=allow_date_fallback
    )
    
    if not matches:
        return {
            'error': '未获取到比赛数据',
            'date': date,
            'source': source,
            'attempts': match_meta.get('attempts', []),
        }
    date = match_meta.get('date', date)
    source = match_meta.get('source', source)
    
    bifen_odds = {}
    zjq_odds = {}
    bqc_odds = {}
    
    if 'bifen' in bet_types and source != 'zgzcw':
        bifen_odds = fetch_beidan_bifen(date)
    if 'zjq' in bet_types and source != 'zgzcw':
        zjq_odds = fetch_beidan_zjq(date)
    if 'bqc' in bet_types and source != 'zgzcw':
        bqc_odds = fetch_beidan_bqc(date)
    
    _clear_ouzhi_cache()
    
    match_ids = [str(m['id']) for m in matches]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        ouzhi_futures = {executor.submit(fetch_ouzhi_odds, mid): mid for mid in match_ids}
        for future in as_completed(ouzhi_futures):
            pass
    
    actual_source = matches[0].get('source', source) if matches else source
    if actual_source == 'zgzcw' and bet_types:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for match in matches:
                if any(bet_type in bet_types for bet_type in ('spf', 'rqspf', 'bifen', 'zjq')):
                    futures[executor.submit(fetch_zgzcw_asian_history, match['id'])] = ('asian', match['id'])
                if any(bet_type in bet_types for bet_type in ('spf', 'rqspf', 'bifen', 'zjq')):
                    futures[executor.submit(fetch_zgzcw_goals_history, match['id'])] = ('goals', match['id'])
                if 'spf' in bet_types:
                    futures[executor.submit(fetch_zgzcw_cs_history, match['id'])] = ('cs', match['id'])
            
            asian_cache = {}
            goals_cache = {}
            cs_cache = {}
            
            for future in as_completed(futures):
                data_type, match_id = futures[future]
                try:
                    data = future.result()
                    if data_type == 'asian':
                        asian_cache[match_id] = data
                    elif data_type == 'goals':
                        goals_cache[match_id] = data
                    elif data_type == 'cs':
                        cs_cache[match_id] = data
                except Exception as e:
                    log.warning(f"并行获取数据失败 {data_type} {match_id}: {e}")
    else:
        asian_cache = {}
        goals_cache = {}
        cs_cache = {}
    
    recommendations = []
    
    for match in matches:
        rec = {
            'match_id': match['id'],
            'num': match['num'],
            'date': match['date'],
            'time': match['time'],
            'league': match['league'],
            'home': match['home'],
            'away': match['away'],
            'handicap': match['handicap'],
            'rqspf_odds': match.get('rqspf_odds') or match.get('lottery_rqspf_odds'),
        }
        
        asian_data = asian_cache.get(match['id'])
        goals_data = goals_cache.get(match['id'])
        cs_data = cs_cache.get(match['id'])
        
        if asian_data and asian_data.get('history'):
            rec['asian'] = asian_data
        if goals_data and goals_data.get('history'):
            rec['goals'] = goals_data
        if cs_data and cs_data.get('history'):
            rec['cs'] = cs_data
        
        if 'spf' in bet_types:
            rec['spf'] = analyze_spf(match, asian_data, cs_data, goals_data)
            spf_a = build_beidan_match_analysis(rec['spf'])
            if spf_a:
                rec['spf']['analysis'] = spf_a
        
        if 'rqspf' in bet_types:
            rec['rqspf'] = analyze_rqspf(match, asian_data, goals_data)
            rec['rqspf']['market_admission'] = build_beidan_market_admission(
                rec['rqspf'], 'rqspf', asian_data, goals_data
            )
        
        if 'bifen' in bet_types:
            rec['bifen'] = analyze_bifen(match, bifen_odds, asian_data, goals_data)
        
        if 'zjq' in bet_types:
            rec['zjq'] = analyze_zjq(match, zjq_odds, asian_data, goals_data)
            rec['zjq']['market_admission'] = build_beidan_market_admission(
                rec['zjq'], 'zjq', asian_data, goals_data
            )
            rec['zjq']['accuracy_gate'] = build_beidan_total_goals_accuracy_gate(
                rec['zjq'], goals_data, match.get('league'),
            )
            if not rec['zjq']['accuracy_gate']['selected']:
                rec['zjq']['market_admission']['official'] = False
                rec['zjq']['market_admission']['playable'] = False
                rec['zjq']['market_admission']['skip_reason'] = 'total_accuracy_gate_rejected'

        if rec.get('spf') and not rec['spf'].get('error'):
            rec['water_market_prediction'] = build_water_market_prediction(
                rec['spf'], match.get('handicap')
            )
        
        if 'bqc' in bet_types:
            rec['bqc'] = analyze_bqc(match, bqc_odds)
        
        recommendations.append(rec)
    
    result = {
        'model_version': BEIDAN_VERSION,
        'date': date,
        'total_matches': len(matches),
        'pending_matches': len(recommendations),
        'recommendations': recommendations,
        'source': source,
        'match_fetch': match_meta,
        'history_summary': summarize_beidan_history(limit=200),
    }
    # Web/API 主推聚合：以“已验证的可投市场”为单位，而不是把所有
    # 模型 Top1 都当成推荐。SPF 只收 70% 高精度层；大小球按已冻结的 O/U 2.5
    # 跨赛季门禁收录；RQSPF 因尚缺独立样本外验证，即使盘口可玩也只进观察。
    top_picks = []
    watch_picks = []
    pick_levels = {'strong': 0, 'medium': 0, 'split': 0, 'low': 0, 'unknown': 0}
    for rec in recommendations:
        for bet_type in ('spf', 'rqspf', 'zjq', 'bifen', 'bqc'):
            sec = rec.get(bet_type)
            if not isinstance(sec, dict):
                continue
            q = sec.get('quality') or {}
            level = q.get('level', 'unknown')
            pick_levels[level] = pick_levels.get(level, 0) + 1
            admission = sec.get('market_admission') or {}
            market_required = bet_type in ('rqspf', 'zjq')
            pick = {
                'num': rec.get('num'),
                'league': rec.get('league'),
                'home': rec.get('home'),
                'away': rec.get('away'),
                'bet_type': bet_type,
                'prediction': sec.get('prediction'),
                'confidence': sec.get('confidence'),
                'level': level,
                'advice': q.get('advice'),
                'high_precision': bool(q.get('high_precision')),
            }
            if bet_type == 'zjq':
                gate = sec.get('accuracy_gate') or {}
                playable = bool(admission.get('official') and admission.get('playable'))
                if gate.get('selected') and playable:
                    direction = gate.get('candidate')
                    pick.update({
                        'prediction': (
                            '大2.5' if direction == 'over'
                            else '小2.5' if direction == 'under'
                            else direction
                        ),
                        'confidence': gate.get('probability'),
                        'level': 'validated_gate',
                        'high_precision': True,
                        'selection_basis': 'dual_season_ou_2_5_accuracy_gate',
                        'validation': gate.get('validation'),
                    })
                    top_picks.append(pick)
                continue
            if bet_type == 'rqspf':
                if admission.get('official') and level in ('strong', 'medium'):
                    pick['selection_basis'] = 'pending_independent_rqspf_validation'
                    watch_picks.append(pick)
                continue
            if bet_type == 'spf':
                if level == 'strong' and q.get('high_precision'):
                    pick['selection_basis'] = 'walk_forward_spf_probability_gte_70pct'
                    top_picks.append(pick)
                elif level in ('strong', 'medium'):
                    pick['selection_basis'] = 'below_high_precision_spf_gate'
                    watch_picks.append(pick)
                continue
            if level in ('strong', 'medium') and not market_required:
                pick['selection_basis'] = 'model_watch_only'
                watch_picks.append(pick)
    result['top_picks'] = top_picks
    result['watch_picks'] = watch_picks
    result['pick_levels'] = pick_levels

    # 爆冷预警聚合：从各场比分分析里拎出 high/medium 爆冷风险场次，
    # 附上"如爆冷则最可能的比分候选"，方便前端单独展示"冷门雷达"。
    upset_watch = []
    for rec in recommendations:
        # 优先用比分分析(bifen)的爆冷块，回退到胜平负(spf)——
        # 默认面板只请求 spf，故必须兼容 spf 才能让前端看到爆冷雷达。
        bf = rec.get('bifen') if isinstance(rec.get('bifen'), dict) else None
        sp = rec.get('spf') if isinstance(rec.get('spf'), dict) else None
        src = None
        if bf and isinstance(bf.get('upset'), dict) and bf['upset'].get('alert'):
            src, up = bf, bf['upset']
        elif sp and isinstance(sp.get('upset'), dict) and sp['upset'].get('alert'):
            src, up = sp, sp['upset']
        else:
            continue
        upset_watch.append({
            'num': rec.get('num'),
            'league': rec.get('league'),
            'home': rec.get('home'),
            'away': rec.get('away'),
            'level': up.get('level'),
            'label': up.get('label'),
            'favorite': up.get('favorite'),
            'favorite_prob': up.get('favorite_prob'),
            'upset_prob': up.get('upset_prob'),
            'chalk_result': up.get('chalk_result'),
            'chalk_score': bf.get('prediction') if bf else None,
            'upset_candidates': up.get('candidates', []),
        })
    # 高风险优先、其次按爆冷概率降序
    upset_watch.sort(key=lambda x: (0 if x['level'] == 'high' else 1,
                                    -(x.get('upset_prob') or 0)))
    result['upset_watch'] = upset_watch

    # Beidan currently lacks an independently priced RQSPF walk-forward set.
    # Expose that limitation in the API instead of presenting strong research
    # picks as professionally validated betting instructions.
    try:
        from ..football.bayes_report import load_professional_validation_summary
        from ..football.professional_readiness import (
            build_professional_decision_gate,
            build_system_gap_assessment,
        )
        professional_validation = load_professional_validation_summary()
        result['professional_validation'] = professional_validation
        result['professional_gap_assessment'] = build_system_gap_assessment({
            'model_metrics': professional_validation.get('model') or {},
            'market_baseline_metrics': professional_validation.get('market') or {},
            'strategy': professional_validation.get('strategy') or {},
        })
        result['decision_gate'] = build_professional_decision_gate(
            professional_validation,
        )
        if '北单让球胜平负尚缺独立赔率样本外验证' not in result['decision_gate']['reasons']:
            result['decision_gate']['reasons'].append('北单让球胜平负尚缺独立赔率样本外验证')
        result['decision_gate']['official_bet_allowed'] = False
        result['decision_gate']['mode'] = 'research_only'
    except Exception as e:
        result['professional_validation'] = {
            'available': False, 'production_ready': False,
            'reason': 'internal_error', 'error_type': type(e).__name__,
        }
        result['decision_gate'] = {
            'official_bet_allowed': False,
            'mode': 'research_only',
            'reasons': ['专业决策闸门不可用，按失败关闭处理'],
        }

    if save_history:
        result['history_save'] = save_beidan_prediction_snapshot(result)
        result['history_summary'] = summarize_beidan_history(limit=200)
    return result


def find_value_bets(date=None, threshold=0.05, source='zgzcw'):
    """抓一轮推荐，再从中挑出模型概率高于市场隐含概率的注。

    抓取在这里，筛选在领域层——迁移前两件事写在同一个函数里，
    想测「怎么挑」就得先联网跑一整轮。
    """
    result = generate_beidan_recommendations(
        date, bet_types=['spf', 'rqspf', 'zjq'], source=source)
    if 'error' in result:
        return result
    return {
        'date': result['date'],
        'total_matches': result['total_matches'],
        'value_bets': _analysis.value_bets(result['recommendations'], threshold),
    }


def print_recommendations(result):
    if 'error' in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"📅 北单推荐 ({result['date']})")
    print(f"场次总数: {result['total_matches']}, 未开赛: {result['pending_matches']}")
    print("=" * 80)
    
    for match in result['recommendations']:
        print(f"\n⚽ [{match['num']}] {match['league']}")
        print(f"   {match['home']} VS {match['away']}")
        time_display = match['time'] if match['time'] else match['date']
        print(f"   时间: {time_display}")
        
        if match.get('handicap'):
            print(f"   让球: {'主队让' if match['handicap'] > 0 else '客队让'} {abs(match['handicap'])}球")
        
        for bet_type in ['spf', 'rqspf', 'bifen', 'zjq', 'bqc']:
            if bet_type not in match:
                continue
            
            data = match[bet_type]
            if 'error' in data:
                continue
            
            name = BET_TYPES.get(bet_type, {}).get('name', bet_type)
            print(f"\n   🎯 {name}:")
            
            if bet_type == 'bifen':
                for score, prob in data.get('top3', []):
                    odd = data['odds'].get(score)
                    print(f"      {score}: 概率 {prob:.2%}, 赔率 {odd}")
            elif bet_type == 'zjq':
                for count, prob in data.get('top3', []):
                    odd = data['odds'].get(count) if 'odds' in data else None
                    print(f"      {count}球: 概率 {prob:.2%}")
                print(f"      大球(>2.5): {data.get('over25_prob', 0):.2%}")
                print(f"      小球(<2.5): {data.get('under25_prob', 0):.2%}")
            elif bet_type == 'bqc':
                for bqc, prob in data.get('top3', []):
                    odd = data['odds'].get(bqc)
                    print(f"      {bqc}: 概率 {prob:.2%}, 赔率 {odd}")
            else:
                for key, prob in data.get('probabilities', {}).items():
                    odd = data['odds'].get(key)
                    print(f"      {key}: 概率 {prob:.2%}, 赔率 {odd}")
                print(f"      推荐: {data.get('prediction')} (置信度 {data.get('confidence', 0):.2%})")
        
        print("-" * 60)


def main():
    print("=" * 80)
    print("              北单足球彩票分析系统")
    print("=" * 80)
    
    source = input("\n请选择数据源 (dc=竞彩单场, jczq=竞彩足球): ").strip()
    if not source:
        source = 'dc'
    if source not in ['dc', 'jczq']:
        source = 'dc'
    
    date = input("请输入日期(格式: YYYY-MM-DD, 回车为今天): ").strip()
    if not date:
        date = time.strftime('%Y-%m-%d')
    
    print(f"\n正在获取 {date} 的{'竞彩单场' if source == 'dc' else '竞彩足球'}数据...")
    
    result = generate_beidan_recommendations(date, bet_types=['spf', 'zjq'], source=source)
    
    print_recommendations(result)


if __name__ == '__main__':
    main()


