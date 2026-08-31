# -*- coding: utf-8 -*-
"""足球分析编排：analyze_match / build_match_analysis 与缓存版本"""

import sys
import os
import math
import re
import time
import gzip
import json
import urllib.request
import urllib.error
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

from ..common.logger import setup_logger
from ..common.paths import data_path

log = setup_logger('football')
from . import parsing as _parsing_mod
from . import fetching as _fetching_mod

from ..domain.sports.football.analysis_result import build_analysis_result
from ..domain.sports.football.market_anchoring import anchor_candidates_to_market
from .config import (
    ACTIONABLE_1X2_MIN_MARGIN, ACTIONABLE_1X2_MIN_PROBABILITY, AVG_LEAGUE_GOAL, BAYESIAN_CALIBRATION_AVAILABLE, CACHE_AVAILABLE, DYNAMIC_ELO_AVAILABLE, DYNAMIC_WEIGHTS_AVAILABLE, FOOTBALL_PREDICTION_LOGIC_VERSION, MAX_GOALS, SIMILAR_MARKET_AVAILABLE, STEAM_MOVE_AVAILABLE, calibrate_predictions, get_cache, get_calibrator, get_dynamic_weights, set_cache, similar_market_match, steam_move_detector,
)
from .fetching import (
    clear_fetch_cache,
)
from .parsing import (
    _apply_lottery_market_availability, calculate_bookmaker_consensus, lottery_market_probabilities, parse_lottery_handicap, resolve_league_profile,
)
from .markets import (
    analyze_asian, analyze_euro, analyze_kelly_trend, analyze_total, compute_euro_asian_deviation, compute_joint_anomaly,
)
from .modeling import (
    _evaluate_risk_level, _result_label, asian_implied_supremacy, compute_prediction_confidence, euro_implied_lambdas, euro_implied_supremacy,
)
from .upset import (
    _evaluate_upset_risk, assess_football_upset,
)
from .scoring import (
    _adjust_goal_dist_with_total_movement, _adjust_half_full_with_market_context, _adjust_half_full_with_score_context, _adjust_score_probs_with_total_movement, _anchor_goal_dist_to_total_line, _anchor_score_candidates_to_1x2, _anchor_score_candidates_to_goal_mean, _apply_joint_market_state, _goal_over_under_from_line, _half_full_probs_to_dict, _pick_recommendations, _recommend_reasons, _score_entry, apply_market_change_prior, calculate_half_full_time_probs, predict_scores, score_heat_label,
)

def _cached_prediction_logic_version(result: Dict) -> str:
    if not isinstance(result, dict):
        return ''
    model = result.get('model') or {}
    status = result.get('model_status') or {}
    return (
        model.get('prediction_logic_version')
        or status.get('prediction_logic_version')
        or ''
    )


def analysis_cache_key(match: Dict) -> str:
    """一场比赛的分析缓存键。

    **只读缓存的地方（BFF 首屏）必须和写入方共用这一份**：key 算错不会
    报错，只会永远 miss——那个接口就变成一个永远返回"计算中"的空壳，
    看起来在工作、实际什么也没做。
    """
    return f"{match['match_id']}_{match.get('home', '')}_{match.get('away', '')}"


def _is_prediction_cache_current(result: Dict) -> bool:
    return _cached_prediction_logic_version(result) == FOOTBALL_PREDICTION_LOGIC_VERSION


def _is_lottery_cache_current(result: Dict, match: Dict) -> bool:
    """中国足彩网核实销售玩法后，使旧的未核实分析缓存失效。"""
    # 中国足彩网临时失败时不能丢弃此前已核实的销售玩法。
    if not match.get('lottery_offer_matched'):
        return True
    cached = result.get('lottery') or {}
    if not cached.get('offer_matched'):
        return False
    expected_market = match.get('lottery_primary_market') or None
    if cached.get('primary_market') != expected_market:
        return False
    if bool(cached.get('spf_available')) != bool(match.get('lottery_spf_available')):
        return False
    if bool(cached.get('rqspf_available')) != bool(match.get('lottery_rqspf_available')):
        return False
    expected_handicap = parse_lottery_handicap(match.get('lottery_handicap'))
    cached_handicap = parse_lottery_handicap((cached.get('handicap') or {}).get('handicap'))
    return expected_handicap == cached_handicap


def build_match_analysis(result):
    """本地 AI 式赛果分析：胜负倾向 → 比分区间 → 进球数 → 理由 → 决策。

    关键原则（对齐元宝/主流 AI 范式）：
    1. 胜负倾向、比分、进球数 **全部从同一个完整比分分布(model.candidates) 边际化得出**，
       三者天然自洽（解决「比分推荐与进球数不一致」问题）。
    2. 采用「比分区间法」而非单一比分：给出 首推 / 次选 / 防冷 三个覆盖不同场景的比分选项，
       其中 防冷 直接复用 ``upset`` 的反向比分候选。
    3. 带中文理由叙述（攻防强度 / 盘口 / 近期状态 / ELO / 赔率异动 / 爆冷提示），像分析师一样解释为何这么看。
    """
    try:
        model = result.get('model', {}) or {}
        candidates = model.get('candidates') or []
        if not candidates:
            return None
        asian = result.get('asian') or {}
        euro = result.get('euro') or {}
        total = result.get('total') or {}
        team = result.get('team') or {}
        upset = result.get('upset') or {}
        confidence = result.get('confidence') or {}
        risk = result.get('risk_level') or {}
        match_info = result.get('match', {}) or {}

        conf_score = confidence.get('score', 0.5) if isinstance(confidence, dict) else (confidence or 0.5)

        # ---- 1. 胜负倾向（从完整比分分布边际化，与比分/进球数同源）----
        w = d = l = 0.0
        goals_map = {}
        for (h, a), prob in candidates:
            if h > a:
                w += prob
            elif h == a:
                d += prob
            else:
                l += prob
            g = h + a
            goals_map[g] = goals_map.get(g, 0.0) + prob
        s = w + d + l
        if s > 0:
            w, d, l = w / s, d / s, l / s

        from src.common.local_match_analysis import (
            LOCAL_ANALYST_VERSION, build_decision, normalize_probabilities,
            pick_high_score_scenario, build_score_strategy,
        )
        probs = normalize_probabilities({'home': w, 'draw': d, 'away': l})
        w, d, l = probs['home'], probs['draw'], probs['away']
        fav = max(probs, key=probs.get)
        fav_p = probs[fav]
        sec_p = sorted(probs.values(), reverse=True)[1]
        margin = fav_p - sec_p

        # ---- 2. 比分区间法：首推 / 次选 / 防冷 ----
        ranked = sorted(candidates, key=lambda x: -x[1])
        primary = ranked[0]
        ph, pa = primary[0]
        # 次选：取「结果方向(胜/平/负)不同」的最高概率比分。
        # 旧逻辑用「方向不同 或 总进球不同」，导致首推为平局(如 1-1)时，把同为平局
        # 的 0-0 当作合法次选——对高大小球盘尤其荒谬(市场预期近 3 球却推 0-0)。
        # 改为强制结果方向不同：首推平局 → 次选必为最可能的胜/负比分(2-1/1-0/1-2)，
        # 既排除了 0-0，又让次选成为真正对冲另一种赛果的有用选项。
        def _res_sign(h, a):
            return 1 if h > a else (-1 if h < a else 0)
        primary_sign = _res_sign(ph, pa)
        secondary = None
        for (h, a), prob in ranked[1:]:
            if _res_sign(h, a) != primary_sign:
                secondary = ((h, a), prob)
                break
        if secondary is None and len(ranked) > 1:
            secondary = ranked[1]

        score_picks = [{
            'type': '首推',
            'score': f"{ph}-{pa}",
            'home': ph, 'away': pa,
            'result': _result_label(ph, pa),
            'probability': primary[1],
        }]
        if secondary:
            sh, sa = secondary[0]
            score_picks.append({
                'type': '次选',
                'score': f"{sh}-{sa}",
                'home': sh, 'away': sa,
                'result': _result_label(sh, sa),
                'probability': secondary[1],
            })
        # 防冷：取爆冷候选中概率最高者（来自 upset 的反向比分）
        upset_cands = upset.get('candidates') or []
        if upset.get('alert') and upset_cands:
            try:
                uh, ua = (int(x) for x in upset_cands[0]['score'].split('-'))
                uprob = next((p for (h, a), p in candidates if h == uh and a == ua), 0.0)
                score_picks.append({
                    'type': '爆冷' if upset_cands[0].get('scenario') == 'outright_upset' else '防冷平',
                    'score': f"{uh}-{ua}",
                    'home': uh, 'away': ua,
                    'result': _result_label(uh, ua),
                    'probability': uprob,
                })
            except (ValueError, KeyError):
                pass

        # ---- 3. 进球数方向（从比分分布边际化）----
        goals_sorted = sorted(goals_map.items(), key=lambda x: -x[1])
        line = total.get('close_line') or total.get('line') or 2.5
        over_p = sum(v for k, v in goals_map.items() if k > line)
        under_p = sum(v for k, v in goals_map.items() if k < line)
        expected = sum(k * v for k, v in goals_map.items())
        top_goals = goals_sorted[:2]
        interval = f"{top_goals[0][0]}-{top_goals[1][0]}球" if len(top_goals) > 1 else f"{top_goals[0][0]}球"
        if over_p >= under_p:
            ou_dir, ou_p = '大球', over_p
        else:
            ou_dir, ou_p = '小球', under_p

        goals_read = {
            'expected': expected,
            'line': line,
            'over_prob': over_p,
            'under_prob': under_p,
            'direction': ou_dir,
            'direction_prob': ou_p,
            'most_likely_interval': interval,
            'top_goals': [{'goals': k, 'probability': v} for k, v in goals_sorted[:3]],
        }

        # BTTS（双方进球）：从完整比分分布边际化。2744 场离线回测校准良好
        # （预测 0.4/0.5/0.6 桶对应真实 50%/52%/60%），方向命中 55.5%，是专业模型常见输出。
        btts_yes = sum(prob for (h, a), prob in candidates if h > 0 and a > 0)
        goals_read['btts_yes'] = btts_yes
        goals_read['btts_no'] = max(0.0, 1.0 - btts_yes)
        goals_read['btts_pick'] = '双方进球' if btts_yes >= 0.5 else '有球队零封'

        high_scenario = pick_high_score_scenario(candidates)
        high_signal = over_p >= 0.52 or float(line or 0) >= 2.75
        if high_signal and high_scenario:
            hh, ha = high_scenario['score']
            if not any(pick['home'] == hh and pick['away'] == ha for pick in score_picks):
                score_picks.append({
                    'type': '大比分', 'score': f"{hh}-{ha}",
                    'home': hh, 'away': ha, 'result': _result_label(hh, ha),
                    'probability': high_scenario['probability'],
                    'scenario_probability': high_scenario['tail_probability'],
                })
        goals_read['high_score_probability'] = (
            high_scenario['tail_probability'] if high_scenario else 0.0
        )

        # ---- 4. 理由叙述（像分析师一样解释）----
        lam_home = model.get('lam_home')
        lam_away = model.get('lam_away')
        reasons = []
        if lam_home is not None and lam_away is not None:
            bias = '主队进攻占优' if lam_home > lam_away else ('客队进攻占优' if lam_away > lam_home else '双方攻防均衡')
            reasons.append(f"模型预期进球 主{lam_home:.1f}/客{lam_away:.1f}，{bias}")
        hcap = asian.get('handicap')
        if hcap is not None:
            favor_text = {'home': '主队', 'away': '客队', 'even': '双方'}.get(asian.get('favor'), '双方')
            reasons.append(f"亚盘开{hcap}球，机构倾向{favor_text}")
        hf = team.get('home_recent', {}).get('form_pts')
        af = team.get('away_recent', {}).get('form_pts')
        if hf is not None and af is not None:
            st = '主队状态更佳' if hf > af else ('客队状态更佳' if af > hf else '双方状态接近')
            reasons.append(f"近期状态（近5场场均积分）主{hf}/客{af}，{st}")
        eh = team.get('elo_home')
        ea = team.get('elo_away')
        if eh and ea:
            reasons.append(f"ELO 实力差 {eh - ea:+.0f}（主{eh:.0f}/客{ea:.0f}）")
        changes = euro.get('changes') or []
        if changes:
            reasons.append(f"赔率异动：{'；'.join(changes[:3])}")
        if upset.get('alert'):
            reasons.append(f"⚠️ 爆冷预警（{upset.get('label')}）：热门{upset.get('favorite')}仅{fav_p:.0%}，关注反向比分")
        elif upset.get('confident'):
            reasons.append(f"✅ 热门稳胆：{upset.get('favorite')}{fav_p:.0%} 且领先次选 {upset.get('gap', 0):.0%}，真实冷门率约 30%")
        # BTTS 叙述
        reasons.append(f"双方进球（BTTS）{goals_read['btts_yes']:.0%}，倾向{goals_read['btts_pick']}")
        # 势均力敌提示：比分本质开放，避免把单一比分当定论
        if fav_p < 0.45 and margin < 0.12:
            reasons.append("势均力敌·比分开放：胜平负接近，精确比分参考区间即可，建议主看进球数/大小球/BTTS")

        # ---- 5. 结论句 ----
        if fav == 'home':
            verdict = f"{match_info.get('home', '主队')}胜面最高 {fav_p:.0%}，优势{'明显' if margin >= 0.12 else '有限'}"
        elif fav == 'away':
            verdict = f"{match_info.get('away', '客队')}胜面最高 {fav_p:.0%}，优势{'明显' if margin >= 0.12 else '有限'}"
        else:
            verdict = f"平局概率 {d:.0%}，双方势均力敌"

        lottery_info = result.get('lottery') or {}
        lottery_primary = lottery_info.get('primary') or {}
        lottery_verdict = None
        if lottery_info.get('primary_market') == 'rqspf':
            rq_probs = lottery_primary.get('probabilities') or {}
            standard_market = lottery_info.get('standard') or {}
            handicap_market = lottery_info.get('handicap') or {}
            linked_pick = lottery_info.get('linked_recommendation') or {}
            standard_pick = linked_pick.get('standard_prediction') or standard_market.get('prediction')
            rq_pick = linked_pick.get('handicap_prediction') or handicap_market.get('prediction')
            if rq_probs and standard_pick and rq_pick:
                rq_value = linked_pick.get('conditional_probability', rq_probs.get(rq_pick, 0.0))
                rq_handicap = handicap_market.get('handicap')
                handicap_text = f"{rq_handicap:+d}" if isinstance(rq_handicap, int) else str(rq_handicap)
                lottery_verdict = (
                    f"先按不让球最高概率选择{standard_pick}；主队{handicap_text}球口径下，"
                    f"兼容结果为{'/'.join(linked_pick.get('compatible_handicap_predictions') or [rq_pick])}，"
                    f"条件分析首选{rq_pick} {rq_value:.0%}"
                )

        conf_level = confidence.get('level') if isinstance(confidence, dict) else None
        if conf_level is None:
            conf_level = 'high' if conf_score >= 0.68 else ('medium' if conf_score >= 0.55 else 'low')
        decision = build_decision(
            probs, confidence=conf_level, upset_alert=bool(upset.get('alert')),
            min_single=ACTIONABLE_1X2_MIN_PROBABILITY,
            min_margin=ACTIONABLE_1X2_MIN_MARGIN,
        )
        score_strategy = build_score_strategy(
            candidates, confidence=conf_level, upset_alert=bool(upset.get('alert')),
        )
        return {
            'analysis_model': LOCAL_ANALYST_VERSION,
            'verdict': verdict,
            'lottery_verdict': lottery_verdict,
            'favorite': fav,
            'favorite_prob': fav_p,
            'margin': margin,
            'open_match': bool(fav_p < 0.45 and margin < 0.12),
            'wdl': {'home': w, 'draw': d, 'away': l},
            'score_picks': score_picks,
            'goals': goals_read,
            'reasons': reasons,
            'confidence': conf_score,
            'risk_level': risk.get('level') if isinstance(risk, dict) else None,
            'upset_alert': bool(upset.get('alert')),
            'upset_confident': bool(upset.get('confident')),
            'decision': decision,
            'score_strategy': score_strategy,
        }
    except Exception as e:
        log.warning(f"build_match_analysis 失败: {e}")
        return None


def analyze_match(match, force_refresh=False):
    """抓取赔率 + 球队攻防 + 泊松模型，返回完整结果 dict
    
    参数：
        match: 比赛信息字典
        force_refresh: 是否强制刷新缓存（重新抓取数据）
    """
    mid = match['match_id']
    home, away = match.get('home', ''), match.get('away', '')
    league_profile = resolve_league_profile(match.get('league', ''))
    log.debug('分析比赛 %s vs %s (id=%s)', home, away, mid)
    
    # 尝试从缓存获取结果
    cache_key = analysis_cache_key(match)
    match_time = match.get('time', '')
    
    if force_refresh:
        # 强制刷新必须穿透页面级缓存，否则只会拿到刚抓过的同一份 HTML
        clear_fetch_cache()

    if not force_refresh and CACHE_AVAILABLE:
        cached_result = get_cache('match_analysis', cache_key, match_time)
        if cached_result is not None and not _is_prediction_cache_current(cached_result):
            log.info(
                "cached prediction logic stale: %s -> %s, recomputing %s vs %s",
                _cached_prediction_logic_version(cached_result) or 'missing',
                FOOTBALL_PREDICTION_LOGIC_VERSION,
                home,
                away,
            )
            cached_result = None
        if cached_result is not None and not _is_lottery_cache_current(cached_result, match):
            log.info(
                "cached lottery offer stale, recomputing %s vs %s: market=%s handicap=%s",
                home, away, match.get('lottery_primary_market'), match.get('lottery_handicap'),
            )
            cached_result = None
        if cached_result is not None:
            log.debug("使用缓存的比赛分析结果: %s vs %s", home, away)
            # 即使使用缓存，也要确保预测记录被保存
            try:
                from .result_sync import save_prediction
                
                model = cached_result.get('model', {})
                top_scores = model.get('top_scores', [])
                candidates = model.get('candidates', [])
                
                predicted_scores = {
                    f"{h}-{a}": prob
                    for (h, a), prob in candidates[:30]
                }
                
                predicted_half_full = _half_full_probs_to_dict(model.get('half_full_time'))
                cached_lottery = cached_result.get('lottery') or lottery_market_probabilities(
                    candidates,
                    match.get('lottery_handicap'),
                    spf_odds=match.get('lottery_spf_odds'),
                    rqspf_odds=match.get('lottery_rqspf_odds'),
                )
                cached_spf_enabled = (
                    not cached_lottery.get('offer_matched')
                    or bool(cached_lottery.get('spf_available'))
                )
                predicted_1x2 = ({
                    'H': sum(prob for (h, a), prob in candidates if h > a),
                    'D': sum(prob for (h, a), prob in candidates if h == a),
                    'A': sum(prob for (h, a), prob in candidates if h < a),
                } if cached_spf_enabled else {})
                
                persistence_result = save_prediction(
                    match_id=mid,
                    league=match.get('league', ''),
                    home=home,
                    away=away,
                    match_time=match.get('time', ''),
                    predicted_scores=predicted_scores,
                    predicted_1x2=predicted_1x2,
                    asian=cached_result.get('asian', {}).get('handicap'),
                    total_line=cached_result.get('total', {}).get('close_line'),
                    predicted_half_full=predicted_half_full,
                    odds_data={
                        'asian': cached_result.get('asian'),
                        'euro': cached_result.get('euro'),
                        'total': cached_result.get('total'),
                        'lottery': cached_lottery,
                    },
                    lottery_handicap=(cached_lottery.get('handicap') or {}).get('handicap'),
                    predicted_rqspf=(
                        (cached_lottery.get('linked_recommendation') or {}).get('handicap_conditional_probabilities')
                        or (cached_lottery.get('handicap') or {}).get('probabilities')
                    ),
                    goal_count=model.get('goal_count'),
                    professional_snapshot={
                        'decision_gate': cached_result.get('decision_gate'),
                        'validation': cached_result.get('professional_validation'),
                        'evidence': cached_result.get('professional_evidence'),
                        'live_context_quality': cached_result.get('live_context_quality'),
                        'accuracy_gate': (cached_lottery or {}).get('accuracy_gate'),
                    },
                )
                model_status = cached_result.get('model_status')
                # 仅在标记真正翻转时才回写缓存：否则每次命中都要 pickle 整个
                # 分析结果并落盘，54 场一轮就是 54 次无谓的整对象序列化。
                if model_status is not None and not model_status.get('prediction_saved'):
                    model_status['prediction_saved'] = True
                    model_status['persistence_backend'] = (
                        (persistence_result or {}).get('persistence_backend')
                    )
                    set_cache('match_analysis', cache_key, cached_result, match_time)
            except Exception as e:
                log.error(f"保存缓存结果的预测记录失败: {e}")
            return cached_result

    zgzcw_only = (
        match.get('schedule_source') == 'zgzcw'
        and not match.get('analysis_source_id_available')
    )
    if zgzcw_only:
        # 中国足彩网独立降级：官方胜平负决定方向；缺少连续亚盘/大小球时
        # 使用中性盘与联赛基准，后续置为低信息完整度。
        spf = match.get('lottery_spf_odds') or {}
        home_odd = float(spf.get('胜') or 2.50)
        draw_odd = float(spf.get('平') or 3.20)
        away_odd = float(spf.get('负') or 2.80)
        asian_raw = {
            'open': {'handicap': 0.0, 'home_odds': 1.0, 'away_odds': 1.0},
            'close': {'handicap': 0.0, 'home_odds': 1.0, 'away_odds': 1.0},
        }
        euro_raw = {
            'open': {'home': home_odd, 'draw': draw_odd, 'away': away_odd},
            'close': {'home': home_odd, 'draw': draw_odd, 'away': away_odd},
            'series': [],
        }
        total_raw = {
            'open': {'line': 2.5, 'over_odds': 1.0, 'under_odds': 1.0},
            'close': {'line': 2.5, 'over_odds': 1.0, 'under_odds': 1.0},
        }
        yazhi_raw = asian_raw
        daxiao_raw = total_raw
        asian = analyze_asian(asian_raw)
        euro = analyze_euro(euro_raw)
        total = analyze_total(total_raw)
        team = None
        log.warning('比赛 %s 使用中国足彩网独立降级模型', mid)
    else:
        # 五组抓取彼此独立，并发发起把单场耗时从「往返之和」降到「最慢的一次」；
        # 对源站的实际压力由 _fetching_mod.fetch() 的发号器统一控速，重复 URL 也只会打一次。
        pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix='FootballOdds')
        try:
            yazhi_task = pool.submit(_parsing_mod.fetch_yazhi, mid)
            euro_task = pool.submit(_parsing_mod.fetch_ouzhi, mid)
            daxiao_task = pool.submit(_parsing_mod.fetch_daxiao, mid)
            team_task = pool.submit(_parsing_mod.fetch_team_strength, mid, home, away, league_profile)
            single_odds_task = pool.submit(_parsing_mod.fetch_single_company_odds, mid)
        finally:
            pool.shutdown(wait=False)
        # 解析顺序与串行版一致，保证失败时抛出的仍是最先失败那一环的错误
        try:
            yazhi_raw = yazhi_task.result()
            asian = analyze_asian(yazhi_raw)
            log.debug(f"亚盘数据获取成功: keys={list(asian.keys())}")
        except Exception as e:
            raise ValueError(f"亚盘数据获取失败: {e}")
        try:
            euro_raw = euro_task.result()
            euro = analyze_euro(euro_raw)
        except Exception as e:
            raise ValueError(f"欧赔数据获取/分析失败: {e}")
        try:
            daxiao_raw = daxiao_task.result()
            total = analyze_total(daxiao_raw)
        except Exception as e:
            raise ValueError(f"大小球数据获取失败: {e}")
        team = team_task.result()
    if team:
        team['league_profile'] = league_profile
    
    # ========== 新增：抓取 Bet365 和 Pinnacle 独赔数据 ==========
    single_odds = None
    if not zgzcw_only:
        try:
            single_odds = single_odds_task.result()
            log.debug(
                "独赔数据抓取结果: Bet365=%s, Pinnacle=%s",
                '有' if single_odds.get('bet365') else '无',
                '有' if single_odds.get('pinnacle') else '无',
            )
        except Exception as e:
            log.warning(f"抓取独赔数据失败: {e}")
    
    # ========== 计算博彩公司分歧指数（在替换之前保存原始平均盘口） ==========
    bookmaker_consensus = None
    original_handicap = asian.get('handicap')
    if single_odds and single_odds.get('bet365') and single_odds.get('pinnacle') and original_handicap:
        bookmaker_consensus = calculate_bookmaker_consensus(
            single_odds['bet365'],
            single_odds['pinnacle'],
            original_handicap
        )
        log.debug(
            "博彩公司分歧指数: 可用=%s, Sharp方向=%s, 调整=%.3f",
            bookmaker_consensus['available'], bookmaker_consensus['sharp_bias'],
            bookmaker_consensus['adjustment'],
        )

    # ========== 保存独赔数据（用于分析，不直接替换平均盘）==========
    # 平均盘作为主模型基准
    # Pinnacle 用于方向修正（Sharp Money）
    # Bet365 用于大众热度参考
    if single_odds:
        if single_odds.get('bet365'):
            # 保存 Bet365 数据作为大众盘参考
            asian['bet365'] = single_odds['bet365'].get('asian')
            total['bet365'] = single_odds['bet365'].get('total')
        
        if single_odds.get('pinnacle'):
            # 保存 Pinnacle 数据作为 Sharp 信号
            asian['pinnacle'] = single_odds['pinnacle'].get('asian')
            total['pinnacle'] = single_odds['pinnacle'].get('total')
    
    # 确保时间字段始终存在（用于盘口变化速度分析）
    if 'close_time' not in asian:
        asian['close_time'] = None
    if 'open_time' not in asian:
        asian['open_time'] = None
    if 'close_time' not in total:
        total['close_time'] = None
    if 'open_time' not in total:
        total['open_time'] = None

    # 保存博彩公司分歧指数到 asian 字典（已在前面计算）
    if 'bookmaker_consensus' not in asian and bookmaker_consensus:
        asian['bookmaker_consensus'] = bookmaker_consensus

    # 注入 ELO xG 数据到 total 字典，供后续 xG 一致性校验使用
    if team and 'elo_xg_home' in team and 'elo_xg_away' in team:
        total['xg_home'] = team['elo_xg_home']
        total['xg_away'] = team['elo_xg_away']
        total['xg_total'] = team['elo_xg_home'] + team['elo_xg_away']

    target_total = total['implied_total']
    lp_avg = league_profile.get('avg_goal', AVG_LEAGUE_GOAL)
    target_total = max(lp_avg * 1.6, min(lp_avg * 3.5, target_total))  # 原1.4/3.2，上调下限使λ更真实
    total['implied_total'] = target_total

    p_home, p_draw, p_away = euro['close']['home'], euro['close']['draw'], euro['close']['away']

    # 根据让球方向获取正确的概率值
    if asian['handicap'] > 0:
        close_hp = asian['close_prob'].get('home_give', asian['close_prob'].get('home', 0.5))
        close_ap = asian['close_prob'].get('away_recv', asian['close_prob'].get('away', 0.5))
        open_hp = asian['open_prob'].get('home_give', asian['open_prob'].get('home', 0.5))
        open_ap = asian['open_prob'].get('away_recv', asian['open_prob'].get('away', 0.5))
    elif asian['handicap'] < 0:
        close_hp = asian['close_prob'].get('home_recv', asian['close_prob'].get('home', 0.5))
        close_ap = asian['close_prob'].get('away_give', asian['close_prob'].get('away', 0.5))
        open_hp = asian['open_prob'].get('home_recv', asian['open_prob'].get('home', 0.5))
        open_ap = asian['open_prob'].get('away_give', asian['open_prob'].get('away', 0.5))
    else:
        close_hp = asian['close_prob'].get('home', 0.5)
        close_ap = asian['close_prob'].get('away', 0.5)
        open_hp = asian['open_prob'].get('home', 0.5)
        open_ap = asian['open_prob'].get('away', 0.5)
    
    asian['implied_supremacy'] = asian_implied_supremacy(
        asian['handicap'], close_hp, close_ap,
        target_total, open_handicap=asian.get('open_handicap'),
        open_hp=open_hp, open_ap=open_ap,
    )
    euro['implied_supremacy'] = euro_implied_supremacy(p_home, p_draw, p_away, target_total)
    euro['implied_lambdas'] = dict(
        zip(('home', 'away'), euro_implied_lambdas(p_home, p_draw, p_away, target_total))
    )

    # 新增：联合异常特征
    joint_anomaly = compute_joint_anomaly(yazhi_raw, daxiao_raw)
    euro_asian_dev = compute_euro_asian_deviation(euro['close'], asian['handicap'])
    
    # 新增：凯利时序趋势分析
    kelly_trend = analyze_kelly_trend(euro_raw.get('series', []))
    if 'kelly' in euro:
        euro['kelly']['trend'] = kelly_trend

    confidence = compute_prediction_confidence(asian, euro, total, team)
    if zgzcw_only:
        confidence = {
            'score': 0.44,
            'level': 'low',
            'label': '中国足彩网单源·低置信',
            'notes': ['500盘口不可用', '缺少连续大小球与亚盘'],
            'recommend_count': 1,
        }

    # 新增：机器学习模型预测
    ml_result = None
    ml_1x2 = None
    ml_model_version = "unknown"
    ml_available = False
    ml_feature_snapshot = {}
    
    try:
        if zgzcw_only:
            raise RuntimeError('中国足彩网单源降级不启用 ML 融合')
        # 准备特征
        ml_features = {
            'elo_home': team.get('elo_home', 1500) if team else 1500,
            'elo_away': team.get('elo_away', 1500) if team else 1500,
            'euro_home': euro['raw_odds']['close']['home'],
            'euro_draw': euro['raw_odds']['close']['draw'],
            'euro_away': euro['raw_odds']['close']['away'],
            'asian_handicap': asian['handicap'],
            'asian_home_water': asian['close_water']['home'],
            'asian_away_water': asian['close_water']['away'],
            'total_line': total['close_line'],
            'total_over_water': total['close_water']['over'],
            'total_under_water': total['close_water']['under'],
            'home_attack': team.get('attack_home', 1.3) if team else 1.3,
            'home_defense': team.get('defense_home', 1.2) if team else 1.2,
            'away_attack': team.get('attack_away', 1.2) if team else 1.2,
            'away_defense': team.get('defense_away', 1.3) if team else 1.3,
            'home_form': team['home_recent']['form_pts'] / 3.0 if team else 0.5,
            'away_form': team['away_recent']['form_pts'] / 3.0 if team else 0.5
        }
        
        # 构建特征快照（用于后续排查）
        elo_diff = ml_features['elo_home'] - ml_features['elo_away']
        ml_feature_snapshot = {
            'elo_diff': elo_diff,
            'total_line': ml_features['total_line'],
            'asian_handicap': ml_features['asian_handicap'],
            'euro_home': ml_features['euro_home'],
            'euro_draw': ml_features['euro_draw'],
            'euro_away': ml_features['euro_away'],
            'home_attack': ml_features['home_attack'],
            'home_defense': ml_features['home_defense'],
            'away_attack': ml_features['away_attack'],
            'away_defense': ml_features['away_defense'],
            'home_form': ml_features['home_form'],
            'away_form': ml_features['away_form'],
            'home_elo': ml_features['elo_home'],
            'away_elo': ml_features['elo_away']
        }
        
        from .ml import predict_1x2_by_ml
        
        ml_response = predict_1x2_by_ml(ml_features)
        
        if ml_response.get('available'):
            ml_available = True
            ml_model_version = ml_response.get('model_version', 'unknown')
            
            ml_1x2 = {
                'H': ml_response.get('H', 0.0),
                'D': ml_response.get('D', 0.0),
                'A': ml_response.get('A', 0.0),
            }
            
            ml_probs = {
                'home': ml_1x2['H'],
                'draw': ml_1x2['D'],
                'away': ml_1x2['A'],
            }
        else:
            ml_available = False
            ml_probs = None
            
        ml_result = {
            'probabilities': ml_probs,
            'model_type': ml_response.get('model_type', 'unknown'),
            'is_trained': ml_available,
            'ml_1x2': ml_1x2,
            'ml_model_version': ml_model_version,
            'ml_available': ml_available,
            'ml_feature_snapshot': ml_feature_snapshot
        }
        
        if ml_available and ml_1x2:
            log.debug(
                "机器学习预测: 主胜=%.3f, 平局=%.3f, 客胜=%.3f, 版本=%s",
                ml_1x2['H'], ml_1x2['D'], ml_1x2['A'], ml_model_version,
            )
        else:
            log.debug("机器学习模型未训练或不可用")
    except Exception as e:
        log.warning(f"机器学习模型预测失败: {e}")

    try:
        from .result_sync import infer_time_layer
        current_time_layer = infer_time_layer(match_time)
    except Exception:
        current_time_layer = None

    candidates, lam_home, lam_away, meta = predict_scores(
        asian, euro, total, team_strength=team, league_profile=league_profile,
        model_type='negative_binomial',
        enable_draw_calibration=True,
        enable_calibration=True,
        calibration_method='platt',
        enable_ensemble=True,
        ensemble_size=2,
        current_time_layer=current_time_layer,
    )
    meta['prediction_logic_version'] = FOOTBALL_PREDICTION_LOGIC_VERSION

    # Mature score models correct the independent-goals assumption around
    # low scores. A chronological 2,744-match five-league backtest selected a
    # 50/50 market-calibrated ensemble: it improved Top3 on both held-out cuts
    # while keeping the blend conservative.
    try:
        from .ml import dixon_coles_score_matrix, get_dc_rho
        from .prediction_policy import blend_score_matrices
        dc_rho = get_dc_rho(
            league=match.get('league', ''),
            total_line=total.get('close_line') or total.get('line', 2.5),
            handicap=asian.get('handicap', 0.0),
        )
        dc_matrix = dixon_coles_score_matrix(
            lam_home, lam_away, max_goals=MAX_GOALS, rho=dc_rho,
        )
        dc_matrix = {
            (home_goals, away_goals): float(dc_matrix[home_goals][away_goals])
            for home_goals in range(MAX_GOALS + 1)
            for away_goals in range(MAX_GOALS + 1)
        }
        primary_matrix = dict(candidates)
        ensemble_matrix = blend_score_matrices(primary_matrix, dc_matrix, 0.50)
        candidates = sorted(ensemble_matrix.items(), key=lambda item: -item[1])
        meta['dixon_coles_ensemble'] = {
            'applied': True, 'weight': 0.50, 'rho': dc_rho,
            'selection': 'chronological_holdout_top3',
        }
    except Exception as e:
        meta['dixon_coles_ensemble'] = {'applied': False, 'reason': str(e)}
        log.warning(f"Dixon-Coles ensemble failed: {e}")

    # 盘口变化先验融合
    market_change_result = None
    try:
        # 将 candidates 转换为 score_probs 字典
        score_probs = {
            f"{h}-{a}": prob
            for (h, a), prob in candidates
        }
        
        score_probs, market_change_result = apply_market_change_prior(
            score_probs,
            asian,
            total,
            weight=0.08
        )
        
        # 转换回 candidates 格式
        candidates = [
            (tuple(map(int, score.split('-'))), prob)
            for score, prob in score_probs.items()
        ]
        candidates.sort(key=lambda x: -x[1])
        
        if market_change_result.get('used'):
            log.debug(
                "盘口变化先验融合: sample=%s, weight=%s",
                market_change_result['sample_count'], market_change_result['weight'],
            )
    except Exception as e:
        log.warning(f"盘口变化先验融合失败: {e}")

    # 新增：Dixon-Coles 模型预测（依赖 predict_scores 产出的 lam_home/lam_away）
    # Apply Bayesian score calibration to the live candidate distribution so
    # downstream score ranking, recommendations, and goal-count aggregation all
    # use the calibrated probabilities.
    if BAYESIAN_CALIBRATION_AVAILABLE:
        try:
            score_probs = {
                f"{h}-{a}": prob
                for (h, a), prob in candidates
            }
            score_probs = calibrate_predictions(
                score_probs,
                league=match.get('league', ''),
                total_line=total.get('close_line') or total.get('line', 2.5),
                asian=asian.get('handicap', 0.0),
            )
            candidates = [
                (tuple(map(int, score.split('-'))), prob)
                for score, prob in score_probs.items()
            ]
            candidates.sort(key=lambda x: -x[1])
            meta['bayesian_candidate_calibrated'] = True
            log.debug("贝叶斯比分校准已写回候选比分排序")
        except Exception as e:
            meta['bayesian_candidate_calibrated'] = False
            log.warning(f"贝叶斯比分候选校准失败: {e}")

    score_total_movement_result = None
    try:
        score_probs = {
            f"{h}-{a}": prob
            for (h, a), prob in candidates
        }
        score_probs, score_total_movement_result = _adjust_score_probs_with_total_movement(score_probs, total)
        if score_total_movement_result.get('applied'):
            candidates = [
                (tuple(map(int, score.split('-'))), prob)
                for score, prob in score_probs.items()
            ]
            candidates.sort(key=lambda x: -x[1])
            meta['score_total_movement_adjusted'] = True
            meta['score_total_movement'] = score_total_movement_result
            log.debug(
                "score total movement adjusted: %s %.3f -> %.3f",
                score_total_movement_result.get('direction'),
                score_total_movement_result.get('expected_before', 0),
                score_total_movement_result.get('expected_after', 0),
            )
        else:
            meta['score_total_movement_adjusted'] = False
            meta['score_total_movement'] = score_total_movement_result
    except Exception as e:
        meta['score_total_movement_adjusted'] = False
        meta['score_total_movement'] = {'applied': False, 'reason': str(e)}
        log.warning(f"score total movement adjustment failed: {e}")

    # Structured H2H / motivation context participates in the same score
    # distribution as 1X2 and totals. Free-form previews are never converted
    # into probabilities; only sourced, quality-scored fields are accepted.
    live_context = match.get('live_context') or {}
    live_context_quality = {}
    try:
        if not live_context:
            safe_mid = re.sub(r'[^0-9A-Za-z_-]', '', str(mid))
            context_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'reports', f'live_context_{safe_mid}.json',
            )
            if safe_mid and os.path.exists(context_path):
                with open(context_path, encoding='utf-8') as context_file:
                    live_context = json.load(context_file)
        from .contextual_fusion import apply_contextual_fusion
        from .live_context_quality import assess_live_context
        live_context_quality = assess_live_context(live_context)
        candidates, contextual_adjustment = apply_contextual_fusion(candidates, live_context)
        meta['contextual_fusion'] = contextual_adjustment
    except Exception as e:
        meta['contextual_fusion'] = {'applied': False, 'reason': str(e)}
        log.warning(f"contextual fusion failed: {e}")

    # Exact-score calibration can change the marginal goal mean. Re-anchor the
    # final score matrix to the market-implied total without changing 1X2 mass.
    # 依次锚定到市场信号——顺序有讲究，见领域层的模块说明。
    # 生产历史档案要读存储、带缓存，在这一层取好注入进去（判据 16）。
    try:
        from .history_calibration import get_runtime_history_profile
        history_profile = get_runtime_history_profile()
    except Exception as e:
        history_profile = None
        log.warning(f"production history profile unavailable: {e}")

    candidates, anchor_meta = anchor_candidates_to_market(
        candidates, total, euro, asian, history_profile)
    meta.update(anchor_meta)

    try:
        from .ml import dixon_coles_score_matrix, dixon_coles_1x2_prob, get_dc_rho
        dc_rho = get_dc_rho(
            league=match.get('league', ''),
            total_line=total.get('close_line') or total.get('line', 2.5),
            handicap=asian.get('handicap', 0.0),
        )
        dc_matrix = dixon_coles_score_matrix(lam_home, lam_away, max_goals=MAX_GOALS, rho=dc_rho)
        dc_1x2 = dixon_coles_1x2_prob(lam_home, lam_away, max_goals=MAX_GOALS, rho=dc_rho)
        dixon_coles_result = {
            'matrix': dc_matrix,
            '1x2': dc_1x2,
            'rho': dc_rho
        }
        log.debug(
            "Dixon-Coles 预测: 主胜=%.3f, 平局=%.3f, 客胜=%.3f",
            dc_1x2['home'], dc_1x2['draw'], dc_1x2['away'],
        )
    except Exception as e:
        log.warning(f"Dixon-Coles 模型预测失败: {e}")

    # 准备欧赔赔率用于冷热计算（基于赔率隐含概率）
    euro_odds_for_heat = {
        'home': euro['raw_odds']['close']['home'],
        'draw': euro['raw_odds']['close']['draw'],
        'away': euro['raw_odds']['close']['away'],
    }

    # ========== 相似盘口数据库匹配（提前到比分推荐之前）==========
    similar_market_result = None
    if SIMILAR_MARKET_AVAILABLE:
        try:
            asian_handicap = asian.get('handicap', 0)
            total_line = total.get('close_line') or total.get('line', 2.5)
            euro_home_odds = euro['raw_odds']['close']['home']
            euro_draw_odds = euro['raw_odds']['close']['draw']
            euro_away_odds = euro['raw_odds']['close']['away']
            
            similar_market_result = similar_market_match(
                asian=asian_handicap,
                total=total_line,
                euro_home=euro_home_odds,
                euro_draw=euro_draw_odds,
                euro_away=euro_away_odds,
                k=1000,
                league=match.get('league', '')
            )
            log.debug(
                "相似盘口匹配: %s 场, 置信度 %.2f%%",
                similar_market_result['count'], similar_market_result['confidence'] * 100,
            )
        except Exception as e:
            log.warning(f"相似盘口匹配失败: {e}")

    # 判断球队实力差距：通过亚盘让球判断
    handicap = abs(asian.get('handicap', 0))
    is_clear_favorite = handicap >= 1.0  # 让球>=1球视为强弱分明
    favor = asian.get('favor', 'home')
    
    # 动态评估爆冷可能性
    upset_risk = _evaluate_upset_risk(asian, euro, team)
    
    # 根据爆冷风险决定最大冷门数量
    # 风险高：允许1个冷门；风险中：允许1个冷门；风险低：不允许冷门
    max_upsets = 0
    if upset_risk >= 0.4:
        max_upsets = 1  # 爆冷风险较高，允许1个冷门提示
    elif is_clear_favorite:
        max_upsets = 0  # 强弱分明且无爆冷迹象，不给出冷门
    
    # “最高概率比分”必须保持真实概率顺序。此前为了展示不同比赛剧本而
    # 重排这里，会把较低概率比分挤进 Top5，历史样本中 Top5 命中率由
    # 52.48% 降至 50.44%。多样化剧本仍由 analyst.score_strategy 单独提供，
    # 不再混入可结算、可回测的概率排名。
    display_candidates = candidates
    filtered_candidates = []
    upset_count = 0
    
    for (h, a), prob in display_candidates:
        # 检查是否是冷门
        diff = h - a
        is_upset = False
        if favor == 'home' and diff < 0:
            is_upset = True  # 主队让球但客队赢
        elif favor == 'away' and diff > 0:
            is_upset = True  # 客队让球但主队赢
        
        # 根据爆冷风险限制冷门数量
        if is_upset:
            if upset_count >= max_upsets:
                continue
            upset_count += 1
        
        filtered_candidates.append(((h, a), prob))
        if len(filtered_candidates) >= 5:
            break
    
    # 更新比分冷热计算：使用赔率隐含概率 vs 模型概率
    top_scores = [
        _score_entry(h, a, prob, score_heat_label(h, a, prob, league_profile, euro_odds_for_heat))
        for (h, a), prob in filtered_candidates
    ]
    recommend = []
    value_bets = []
    
    # 计算半全场概率（集成动态ELO）
    half_full_time = calculate_half_full_time_probs(
        candidates, team, asian, total, home_team=home, away_team=away, league=match.get('league', '')
    )
    half_full_time = _adjust_half_full_with_score_context(half_full_time, candidates)
    half_full_time = _adjust_half_full_with_market_context(half_full_time, asian, total)

    # 新增：进球数推荐（结合历史盘口数据 + 校准器）
    goal_count_result = None
    goal_dist_before_calibration = None
    goal_dist_after_calibration = None
    try:
        from .ml import predict_goal_counts_from_candidates
        goal_count_result = predict_goal_counts_from_candidates(candidates, max_goals=MAX_GOALS, asian=asian, total=total)
        
        # 保存校准前的分布
        goal_dist_before_calibration = goal_count_result.get('distribution_dict', {}).copy()
        if not goal_dist_before_calibration:
            raw_distribution = goal_count_result.get('distribution', {})
            if isinstance(raw_distribution, list):
                goal_dist_before_calibration = {
                    item.get('goals'): item.get('probability', 0.0)
                    for item in raw_distribution
                    if item.get('goals') is not None
                }
            elif isinstance(raw_distribution, dict):
                goal_dist_before_calibration = raw_distribution.copy()
        
        # 使用总球数校准器校准
        try:
            from .goal_count_calibrator import GoalCountCalibrator
            calibrator = GoalCountCalibrator()
            
            # 计算期望总进球数
            expected_total = sum(k * v for k, v in goal_dist_before_calibration.items())
            
            # 获取盘口参数
            total_line = total.get('close_line', 2.5) if total else 2.5
            asian_handicap = asian.get('handicap', 0) if asian else 0.0
            league_name = match.get('league', '其他')
            
            # 应用校准
            calibrated_dist = calibrator.calibrate_goal_dist(
                league=league_name,
                total_line=total_line,
                goal_dist=goal_dist_before_calibration,
                expected_total=expected_total,
                asian=asian_handicap,
                min_samples=10
            )
            
            # 保存校准后的分布
            calibrated_dist, goal_line_anchor = _anchor_goal_dist_to_total_line(calibrated_dist, total)
            calibrated_dist, goal_movement_adjustment = _adjust_goal_dist_with_total_movement(calibrated_dist, total)
            goal_dist_after_calibration = calibrated_dist
            
            # 更新结果中的分布
            goal_count_result['distribution_dict'] = calibrated_dist
            goal_count_result['line_anchor'] = goal_line_anchor
            goal_count_result['movement_adjustment'] = goal_movement_adjustment
            
            # 重新计算推荐（基于校准后的分布）
            from .ml import recommend_goal_counts_from_dist, get_goal_count_distribution_from_dist
            high_risk = goal_count_result.get('sample_info', {}).get('quality', 'none') in ['low', 'none']
            low_quality_sample = goal_count_result.get('sample_info', {}).get('quality', 'none') in ['low', 'none']
            if goal_movement_adjustment.get('conflict'):
                high_risk = True
            
            goal_count_result['recommendations'] = recommend_goal_counts_from_dist(
                calibrated_dist, top_n=3, high_risk=high_risk, low_quality_sample=low_quality_sample
            )
            goal_count_result['distribution'] = get_goal_count_distribution_from_dist(calibrated_dist)
            
            # 更新大小球概率
            goal_count_result['over_under'] = _goal_over_under_from_line(calibrated_dist, total)
            
            log.debug("进球数校准完成: 期望总进球 %.2f", expected_total)
        except Exception as e:
            log.warning(f"进球数校准失败: {e}，使用原始分布")
        
        log.debug("进球数推荐: %s", goal_count_result['recommendations'])
    except Exception as e:
        log.warning(f"进球数推荐失败: {e}")

    # ========== 临场资金流检测 ==========
    steam_result = None
    if STEAM_MOVE_AVAILABLE:
        try:
            steam_result = steam_move_detector(asian, total, match.get('time'))
            log.debug("资金流检测: %d 个信号", len(steam_result['signals']))
        except Exception as e:
            log.warning(f"资金流检测失败: {e}")

    # ========== 爆冷识别（对齐北单：显式暴露爆冷风险 + 反向比分候选）==========
    try:
        upset = assess_football_upset(
            asian, euro, team, candidates, total=total,
            anomaly={
                'joint_water': joint_anomaly,
                'euro_asian_deviation': euro_asian_dev,
            },
            steam_result=steam_result,
        )
    except Exception as e:
        log.warning(f"爆冷识别失败: {e}")
        upset = None

    # ========== 风险等级评估（提前到推荐之前）==========
    risk = _evaluate_risk_level(asian, euro, total, steam_result, confidence, similar_market_result)
    recommend_count = risk.get('recommend_count', 2)
    
    # ========== 根据风险等级获取推荐比分和价值投注 ==========
    if recommend_count > 0:
        rec_list, value_bets = _pick_recommendations(
            candidates, asian, euro, total, n=recommend_count,
            confidence=confidence, league_profile=league_profile, team=team, similar_market=similar_market_result,
        )
        
        for h, a, prob in rec_list:
            heat, _ = score_heat_label(h, a, prob, league_profile, euro_odds_for_heat)
            recommend.append({
                **_score_entry(h, a, prob, (heat, _)),
                'reasons': _recommend_reasons(h, a, asian, euro, total, team, heat=heat),
            })
    else:
        log.debug("风险等级 %s，不推荐具体比分", risk['level'])
    
    # ========== 构建概率排序（纯模型概率）==========
    probability_rank = []
    for (h, a), prob in candidates[:5]:
        probability_rank.append({
            'score': f"{h}-{a}",
            'home': h,
            'away': a,
            'result': _result_label(h, a),
            'prob': prob
        })
    
    # ========== 构建推荐排序（带推荐原因）==========
    recommend_rank = []
    for rec in recommend:
        h, a = rec['home'], rec['away']
        recommend_rank.append({
            'score': f"{h}-{a}",
            'home': h,
            'away': a,
            'result': _result_label(h, a),
            'prob': rec['prob'],
            'recommend_score': rec.get('prob', 0),
            'reasons': rec.get('reasons', [])
        })
    
    # ========== 获取模型融合权重 ==========
    model_weights = {
        'market': 0.55,
        'team': 0.18,
        'elo': 0.17,
        'similar': 0.10,
        'ml': 0.0
    }
    try:
        if DYNAMIC_WEIGHTS_AVAILABLE:
            match_data = {
                'league': match.get('league', '其他'),
                'handicap': asian.get('handicap', 0),
                'euro_std': euro.get('kelly', {}).get('spread', 0.05),
                'kelly_std': euro.get('kelly', {}).get('spread', 0.02),
                'odds_changes': len(euro_raw.get('series', [])),
                'elo_diff': abs(team.get('elo_home', 1500) - team.get('elo_away', 1500)) if team else 0,
                'total_line': total.get('close_line', 2.5),
            }
            log.debug(f"构建动态权重特征: {match_data}")
            
            market_w, team_w, elo_w, ml_w = get_dynamic_weights(confidence.get('score', 0.5), match_data)
            model_weights = {
                'market': market_w,
                'team': team_w,
                'elo': elo_w,
                'similar': 0.10,
                'ml': ml_w
            }
            log.debug(
                "动态权重: market=%.3f, team=%.3f, elo=%.3f, ml=%.3f",
                market_w, team_w, elo_w, ml_w,
            )
    except Exception as e:
        log.warning(f"获取动态权重失败: {e}")
    
    # ========== 贝叶斯校准影响分析 ==========
    calibration_effect = []
    if BAYESIAN_CALIBRATION_AVAILABLE:
        try:
            calibrator = get_calibrator()
            # 计算校准前后的概率变化
            for (h, a), prob in candidates[:6]:
                score_str = f"{h}-{a}"
                calibrated_prob = calibrator.calibrate(score_str, prob)
                delta = calibrated_prob - prob
                record = calibrator.history.get(score_str, {})
                calibration_effect.append({
                    'score': score_str,
                    'before': prob,
                    'after': calibrated_prob,
                    'delta': delta,
                    'sample_count': record.get('count', 0)
                })
        except ImportError as e:
            log.warning(f"贝叶斯校准模块导入失败: {e}")
        except Exception as e:
            log.warning(f"计算贝叶斯校准影响失败: {e}")
    
    # ========== 相似盘口样本质量详情 ==========
    similar_market_detail = {}
    if similar_market_result:
        detail = similar_market_result.get('sample_quality', {})
        result_dist = similar_market_result.get('result_dist', {})
        top_scores_list = []
        if 'goals_dist' in similar_market_result:
            for score, prob in sorted(similar_market_result['goals_dist'].items(), key=lambda x: -x[1])[:3]:
                top_scores_list.append({'score': score, 'prob': prob})
        
        similar_market_detail = {
            'count': similar_market_result.get('count', 0),
            'avg_distance': similar_market_result.get('avg_distance', 0),
            'confidence': similar_market_result.get('confidence', 0),
            'same_league_ratio': detail.get('same_league_ratio', 0),
            'recent_season_ratio': detail.get('recent_season_ratio', 0),
            'result_dist': {
                'home': result_dist.get('H', 0),
                'draw': result_dist.get('D', 0),
                'away': result_dist.get('A', 0)
            },
            'top_scores': top_scores_list
        }
    
    # ========== 赛后回填状态 ==========
    settlement = {
        'status': 'pending',
        'actual_score': None,
        'hit': None,
        'updated_modules': []
    }
    try:
        from .result_sync import PredictionHistory, _is_match_settle_due
        ph = PredictionHistory()
        for rec in ph.records:
            if rec.get('match_id') == mid:
                if rec.get('settled') and _is_match_settle_due(rec.get('match_time'), minutes=180):
                    settlement = {
                        'status': 'settled',
                        'actual_score': rec.get('actual_score'),
                        'hit': {
                            'top1': rec.get('hit_top1', False),
                            'top3': rec.get('hit_top3', False),
                            'result_1x2': rec.get('hit_1x2', False),
                            'goal_count': rec.get('hit_total', False)
                        },
                        'updated_modules': [
                            'bayesian_calibration',
                            'elo',
                            'market_cluster',
                            'market_score_db'
                        ]
                    }
                break
    except Exception as e:
        log.debug(f"获取赛后回填状态失败: {e}")
    
    # ========== 模型状态汇总 ==========
    ml_enabled = False
    ml_reason = "模型未训练，未参与融合"
    ml_participating = False
    ml_fusion_weight = 0.0
    ml_eligibility = None
    
    try:
        from .result_sync import get_history, check_ml_fusion_eligibility, get_ml_fusion_weight
        import src.football.ml as ml_module
        
        ml_enabled = ml_module.load_trained_ml_model()
        
        if ml_enabled:
            # 获取测试集样本数
            test_set_samples = ml_module._trained_ml_metadata.get('test_count', 0) if ml_module._trained_ml_metadata else 0
            
            # 检查是否满足融合条件
            history = get_history()
            ml_stats = history.get_ml_evaluation_stats()
            eligibility = check_ml_fusion_eligibility(ml_stats, test_set_samples)
            ml_eligibility = eligibility
            shadow_samples = eligibility['shadow_samples']
            
            if eligibility['eligible']:
                ml_fusion_weight = get_ml_fusion_weight(True, shadow_samples, 0.0)
                if ml_fusion_weight > 0:
                    metrics_hint = '，指标已达标' if eligibility.get('metrics_passed') else '，指标待观察'
                    ml_reason = f"已参与融合，权重 {ml_fusion_weight*100:.1f}%{metrics_hint}"
                    ml_participating = True
                else:
                    ml_reason = "已训练，等待权重分配"
            else:
                pending = []
                conds = eligibility['conditions']
                if not conds['test_set_samples']['passed']:
                    pending.append(
                        f"测试集 {conds['test_set_samples']['actual']}/{conds['test_set_samples']['required']}"
                    )
                if not conds['shadow_samples']['passed']:
                    pending.append(
                        f"影子样本 {conds['shadow_samples']['actual']}/{conds['shadow_samples']['required']}"
                    )
                ml_reason = f"已训练，样本收集中（{', '.join(pending)}）" if pending else "已训练，样本收集中"
        else:
            ml_reason = "模型文件不存在或加载失败"
    except Exception as e:
        ml_reason = "ML模块不可用"
    
    # 将参与状态保存到状态字典中供前端显示
    ml_enabled = ml_participating
    
    # 获取真实统计数据
    pending_count = 0
    settled_count = 0
    calib_sample_count = 0
    market_sample_count = 0
    prediction_saved = False  # 预测记录保存状态
    
    try:
        from .result_sync import get_history_stats
        history_stats = get_history_stats()
        pending_count = history_stats.get('unsettled', 0)
        settled_count = history_stats.get('settled', 0)
    except Exception as e:
        log.debug(f"获取历史统计失败: {e}")
    
    try:
        if BAYESIAN_CALIBRATION_AVAILABLE:
            calibrator = get_calibrator()
            calib_sample_count = sum(v.get('count', 0) for v in calibrator.history.values())
    except Exception as e:
        log.debug(f"获取校准器统计失败: {e}")
    
    try:
        from .market_db import MarketScoreDB
        db = MarketScoreDB()
        market_sample_count = sum(db.sample_counts.values())
    except Exception as e:
        log.debug(f"获取盘口库统计失败: {e}")
    
    # 判断相似盘口质量等级
    similar_quality = '-'
    if similar_market_result:
        avg_dist = similar_market_result.get('avg_distance', 1)
        count = similar_market_result.get('count', 0)
        if count >= 100 and avg_dist <= 0.25:
            similar_quality = '高'
        elif count >= 50 and avg_dist <= 0.4:
            similar_quality = '中'
        else:
            similar_quality = '低'
    
    model_status = {
        'prediction_saved': False,  # 稍后更新
        'prediction_logic_version': FOOTBALL_PREDICTION_LOGIC_VERSION,
        'result_sync': {
            'enabled': True,
            'pending_count': pending_count,
            'settled_count': settled_count
        },
        'bayesian_calibration': {
            'enabled': BAYESIAN_CALIBRATION_AVAILABLE,
            'sample_count': calib_sample_count
        },
        'market_db': {
            'enabled': True,
            'sample_count': market_sample_count
        },
        'similar_market': {
            'enabled': SIMILAR_MARKET_AVAILABLE,
            'quality': similar_quality,
            'sample_count': similar_market_result.get('count', 0) if similar_market_result else 0,
            'avg_distance': similar_market_result.get('avg_distance', 0) if similar_market_result else 0,
            'confidence': similar_market_result.get('confidence', 0) if similar_market_result else 0
        },
        'elo': {
            'enabled': DYNAMIC_ELO_AVAILABLE,
            'home_elo': team.get('elo_home', 1500) if team else 1500,
            'away_elo': team.get('elo_away', 1500) if team else 1500,
            'reliability': 1.0
        },
        'ml': {
            'enabled': ml_enabled,
            'reason': ml_reason,
            'weight': ml_fusion_weight,
            'metrics_passed': ml_eligibility.get('metrics_passed') if ml_eligibility else None,
        }
    }
    
    lottery = lottery_market_probabilities(
        candidates,
        match.get('lottery_handicap'),
        spf_odds=match.get('lottery_spf_odds'),
        rqspf_odds=match.get('lottery_rqspf_odds'),
    )
    lottery.update({
        'source': match.get('lottery_source') or 'unavailable',
        'offer_matched': bool(match.get('lottery_offer_matched')),
        'unavailable_reason': match.get('lottery_unavailable_reason'),
        'available_markets': match.get('lottery_available_markets') or [],
        'spf_available': bool(match.get('lottery_spf_available')),
        'rqspf_available': bool(match.get('lottery_rqspf_available')),
        'spf_odds': match.get('lottery_spf_odds'),
        'rqspf_odds': match.get('lottery_rqspf_odds'),
    })
    spf_prediction_enabled = _apply_lottery_market_availability(lottery)
    official_primary = match.get('lottery_primary_market')
    lottery['primary_market'] = official_primary
    if official_primary == 'rqspf' and lottery.get('handicap'):
        lottery['primary'] = lottery['handicap']
    elif official_primary == 'spf':
        lottery['primary'] = lottery['standard']
    else:
        lottery['primary'] = None
    from .accuracy_gate import (
        build_accuracy_gate,
        build_total_goals_gate,
        has_static_spf_policy,
    )
    production_spf_policy = None
    if not has_static_spf_policy(match.get('league')):
        try:
            from .production_league_gate import load_production_league_spf_policy
            production_spf_policy = load_production_league_spf_policy(match.get('league'))
        except Exception as e:
            log.warning('生产联赛门禁读取失败，继续使用全局门禁规则: %s', e)
    # 输出契约住在领域层——字段名与嵌套形状下游都按名字取（判据 12）。
    result = build_analysis_result(
        asian=asian,
        calibration_effect=calibration_effect,
        candidates=candidates,
        confidence=confidence,
        dixon_coles_result=dixon_coles_result,
        euro=euro,
        euro_asian_dev=euro_asian_dev,
        goal_count_result=goal_count_result,
        goal_dist_after_calibration=goal_dist_after_calibration,
        goal_dist_before_calibration=goal_dist_before_calibration,
        half_full_time=half_full_time,
        joint_anomaly=joint_anomaly,
        lam_away=lam_away,
        lam_home=lam_home,
        league_profile=league_profile,
        live_context=live_context,
        live_context_quality=live_context_quality,
        lottery=lottery,
        market_change_result=market_change_result,
        match=match,
        meta=meta,
        ml_result=ml_result,
        model_status=model_status,
        model_weights=model_weights,
        probability_rank=probability_rank,
        production_spf_policy=production_spf_policy,
        recommend=recommend,
        recommend_rank=recommend_rank,
        risk=risk,
        settlement=settlement,
        similar_market_detail=similar_market_detail,
        similar_market_result=similar_market_result,
        single_odds=single_odds,
        steam_result=steam_result,
        team=team,
        top_scores=top_scores,
        total=total,
        upset=upset,
        value_bets=value_bets,
    )


    try:
        from .bayes_report import load_professional_validation_summary
        from .professional_readiness import (
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
            evidence=result.get('professional_evidence'),
            live_quality=result.get('live_context_quality'),
            accuracy_gate=(result.get('lottery') or {}).get('accuracy_gate'),
        )
    except Exception as e:
        log.warning(f"专业决策闸门构建失败: {e}")
        result['professional_validation'] = {
            'available': False, 'production_ready': False, 'reason': str(e),
        }
        result['decision_gate'] = {
            'official_bet_allowed': False,
            'mode': 'research_only',
            'reasons': ['专业决策闸门不可用，按失败关闭处理'],
        }

    # ========== 元宝式赛果分析（胜负倾向→比分区间→进球数方向→理由，三者同源自洽）==========
    try:
        result['analysis'] = build_match_analysis(result)
    except Exception as e:
        log.warning(f"赛果分析注入失败: {e}")
        result['analysis'] = None
    
    # 保存结果到缓存
    if CACHE_AVAILABLE:
        set_cache('match_analysis', cache_key, result, match_time)
        log.debug(f"比赛分析结果已缓存: {home} vs {away}")
    
    # 保存预测记录用于赛后回填
    try:
        from .result_sync import save_prediction

        model = result.get('model', {})
        candidates = model.get('candidates', [])

        predicted_scores = {
            f"{h}-{a}": prob
            for (h, a), prob in candidates[:30]
        }

        predicted_1x2 = ({
            'H': sum(prob for (h, a), prob in candidates if h > a),
            'D': sum(prob for (h, a), prob in candidates if h == a),
            'A': sum(prob for (h, a), prob in candidates if h < a),
        } if spf_prediction_enabled else {})

        # 影子预测：base_1x2 是现有基础模型的预测结果
        predicted_half_full = _half_full_probs_to_dict(model.get('half_full_time'))
        base_1x2 = predicted_1x2.copy()

        persistence_result = save_prediction(
            match_id=mid,
            league=match.get('league', ''),
            home=home,
            away=away,
            match_time=match.get('time', ''),
            predicted_scores=predicted_scores,
            predicted_1x2=predicted_1x2,
            asian=asian.get('handicap'),
            total_line=total.get('close_line'),
            predicted_half_full=predicted_half_full,
            odds_data={
                'asian': asian,
                'euro': euro,
                'total': total,
                'lottery': lottery,
            },
            # 影子预测相关字段
            base_1x2=base_1x2,
            ml_1x2=ml_1x2 if spf_prediction_enabled else {},
            ml_model_version=ml_model_version,
            ml_available=ml_available,
            ml_feature_snapshot=ml_feature_snapshot,
            lottery_handicap=(lottery.get('handicap') or {}).get('handicap'),
            predicted_rqspf=(
                (lottery.get('linked_recommendation') or {}).get('handicap_conditional_probabilities')
                or (lottery.get('handicap') or {}).get('probabilities')
            ),
            goal_count=goal_count_result,
            professional_snapshot={
                'decision_gate': result.get('decision_gate'),
                'validation': result.get('professional_validation'),
                'evidence': result.get('professional_evidence'),
                'live_context_quality': result.get('live_context_quality'),
                'accuracy_gate': (lottery or {}).get('accuracy_gate'),
                'upset': result.get('upset'),
            },
        )
        prediction_saved = True
        # 更新模型状态中的保存状态
        if 'model_status' in result:
            result['model_status']['prediction_saved'] = True
            result['model_status']['persistence_backend'] = (
                (persistence_result or {}).get('persistence_backend')
            )
            # 更新缓存以包含最新的 prediction_saved 状态
            if CACHE_AVAILABLE:
                set_cache('match_analysis', cache_key, result, match_time)
    except Exception as e:
        log.error(f"保存预测记录失败: {e}", exc_info=True)
    
    return result


