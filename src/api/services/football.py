# -*- coding: utf-8 -*-
"""足球接口的业务装配。

**新旧两个入口共用同一份**（判据 11）。

`_serve_report_file` **没有**提升：它直接往 `self.wfile` 写响应流，
是 HTTP 层的文件服务而不是业务装配。新入口用 FastAPI 的 `FileResponse`
另做一条，两边各自处理自己那套响应机制。
"""

import logging

import os
import sys
import json
import time
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from src.common.paths import data_path
from src.api.runtime import lazy_modules as _lazy_mod
from src.api.runtime.lazy_modules import _import_backtest_modules, analyze_match, ensure_beidan_report, ensure_football_report, fetch_match_list, football_reportable_ids, get_match_list_status
from src.api.runtime.jobs import _attach_bayes_report_url, _match_started, _trigger_football_analysis, _trigger_football_report_sync
from src.api.runtime import jobs as _jobs_mod

log = logging.getLogger('api.services.football')

# 提升时随函数体一起带过来的模块级常量——函数引用的是模块全局，
# 漏掉不会在导入时报错，运行到那一行才 NameError。
FOOTBALL_BATCH_LIMIT = max(1, int(os.getenv('FOOTBALL_BATCH_LIMIT', '30')))
FOOTBALL_BATCH_CONCURRENCY = max(1, int(os.getenv('FOOTBALL_BATCH_CONCURRENCY', '4')))


def try_generate_report(rel: str):
    """按文件名尝试按需生成深度报告，返回生成的文件绝对路径或 None。"""
    if not _lazy_mod._BAYES_REPORT_AVAILABLE:
        return None
    try:
        if rel.startswith('football_bayes_') and rel.endswith('.html'):
            mid = rel[len('football_bayes_'):-len('.html')]
            return ensure_football_report(mid)
        if rel.startswith('beidan_bayes_') and rel.endswith('.html'):
            mid = rel[len('beidan_bayes_'):-len('.html')]
            return ensure_beidan_report(mid)
    except Exception as e:
        log.error('报告按需生成失败: %s', rel, exc_info=True)
    return None


def matches_payload():
    try:
        matches = fetch_match_list()

        # 过滤掉「已开赛」的比赛：列表只保留未开赛场次，减少前端渲染量、
        # 也避免对已经无法进行投注分析的比赛做无谓展示（提速）。
        matches = [m for m in matches if not _match_started(m)]
        # 后台预生成深度报告：对未开赛且有缓存的比赛，无报告则生成、变盘则重生成
        try:
            reportable = football_reportable_ids()
            sync_mids = [str(m.get('match_id')) for m in matches
                         if str(m.get('match_id')) in reportable]
            _trigger_football_report_sync(sync_mids)
            # 名单内缺分析缓存的比赛，后台自动补分析（分析后即可生成深度报告）
            _trigger_football_analysis(matches)
        except Exception:
            pass
        return {
            'matches': _attach_bayes_report_url(matches),
            'source_status': get_match_list_status(),
        }
    except Exception as exc:
        log.error('获取比赛列表失败', exc_info=True)
        return {
            'error': '获取比赛列表失败',
            'source_status': get_match_list_status(),
            'error_type': type(exc).__name__,
        }


def match_from_params(params):
    """把 GET 查询参数解析成 analyze_match 需要的 match 字典"""

    def _json_param(name):
        raw = params.get(name, [''])[0]
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    def _number_param(name):
        raw = params.get(name, [''])[0]
        try:
            value = float(raw)
            return int(value) if value.is_integer() else value
        except (TypeError, ValueError):
            return None

    return {
        'match_id': params.get('match_id', [''])[0],
        'home': params.get('home', [''])[0],
        'away': params.get('away', [''])[0],
        'league': params.get('league', [''])[0],
        'time': params.get('time', [''])[0],
        'num': params.get('num', [''])[0],
        'schedule_source': params.get('schedule_source', [''])[0],
        'analysis_source_id_available': params.get('analysis_source_id_available', ['true'])[0].lower() == 'true',
        'zgzcw_id': params.get('zgzcw_id', [''])[0],
        'analysis_id': params.get('analysis_id', [''])[0],
        'lottery_handicap': _number_param('lottery_handicap'),
        'lottery_primary_market': params.get('lottery_primary_market', [''])[0] or None,
        'lottery_source': params.get('lottery_source', ['unavailable'])[0],
        'lottery_offer_matched': params.get('lottery_offer_matched', ['false'])[0].lower() == 'true',
        'lottery_available_markets': [
            item for item in params.get('lottery_available_markets', [''])[0].split(',') if item
        ],
        'lottery_spf_available': params.get('lottery_spf_available', ['false'])[0].lower() == 'true',
        'lottery_rqspf_available': params.get('lottery_rqspf_available', ['false'])[0].lower() == 'true',
        'lottery_spf_odds': _json_param('lottery_spf_odds'),
        'lottery_rqspf_odds': _json_param('lottery_rqspf_odds'),
    }


def match_from_json(raw):
    """把批量接口 JSON body 里的一场比赛归一化成与查询参数路径同构的字典"""

    def _dict_or_none(value):
        return value if isinstance(value, dict) else None

    def _number_or_none(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if number.is_integer() else number

    markets = raw.get('lottery_available_markets') or []
    if isinstance(markets, str):
        markets = [item for item in markets.split(',') if item]

    return {
        'match_id': str(raw.get('match_id') or ''),
        'home': raw.get('home') or '',
        'away': raw.get('away') or '',
        'league': raw.get('league') or '',
        'time': raw.get('time') or '',
        'num': raw.get('num') or '',
        'schedule_source': raw.get('schedule_source') or '',
        'analysis_source_id_available': bool(raw.get('analysis_source_id_available', True)),
        'zgzcw_id': raw.get('zgzcw_id') or '',
        'analysis_id': raw.get('analysis_id') or '',
        'lottery_handicap': _number_or_none(raw.get('lottery_handicap')),
        'lottery_primary_market': raw.get('lottery_primary_market') or None,
        'lottery_source': raw.get('lottery_source') or 'unavailable',
        'lottery_offer_matched': bool(raw.get('lottery_offer_matched')),
        'lottery_available_markets': [item for item in markets if item],
        'lottery_spf_available': bool(raw.get('lottery_spf_available')),
        'lottery_rqspf_available': bool(raw.get('lottery_rqspf_available')),
        'lottery_spf_odds': _dict_or_none(raw.get('lottery_spf_odds')),
        'lottery_rqspf_odds': _dict_or_none(raw.get('lottery_rqspf_odds')),
    }


def analyze_one(match, force_refresh=False):
    """分析单场并把异常归一化成 error 字段，供单场与批量接口共用"""
    match_id = match.get('match_id', '')
    try:
        return {'result': analyze_match(match, force_refresh=force_refresh)}
    except ValueError as e:
        error_msg = str(e)
        log.error('赔率分析失败 match_id=%s: %s', match_id, error_msg)
        return {'error': error_msg}
    except Exception as e:
        error_msg = f'赔率分析失败: {str(e)}'
        log.error('赔率分析失败 match_id=%s', match_id, exc_info=True)
        return {'error': error_msg}


def predict_payload(params):
    match_id = params.get('match_id', [''])[0]
    if not match_id:
        return {'error': '缺少 match_id 参数'}
    force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'
    return analyze_one(match_from_params(params), force_refresh)


def predict_batch_payload(body):
    """批量预测：一次请求分析多场，把每场一次 HTTP 往返压成一次。

    逐场结果按入参顺序原样返回（失败场次带 error 而非整批失败），
    前端据此保持渐进渲染，不必等整批算完。
    """
    if not isinstance(body, dict):
        return {'error': '请求体缺失或不是合法 JSON 对象'}
    raw_matches = body.get('matches')
    if not isinstance(raw_matches, list) or not raw_matches:
        return {'error': '缺少 matches 参数'}
    if len(raw_matches) > FOOTBALL_BATCH_LIMIT:
        return {'error': f'单批最多 {FOOTBALL_BATCH_LIMIT} 场，收到 {len(raw_matches)} 场'}
    force_refresh = bool(body.get('force_refresh'))

    matches = [match_from_json(item) for item in raw_matches if isinstance(item, dict)]
    if not matches:
        return {'error': 'matches 中没有有效的比赛对象'}

    workers = min(FOOTBALL_BATCH_CONCURRENCY, len(matches))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='PredictBatch') as pool:
        outcomes = list(pool.map(
            lambda m: analyze_one(m, force_refresh), matches
        ))

    results = []
    for match, outcome in zip(matches, outcomes):
        entry = {'match_id': match['match_id']}
        entry.update(outcome)
        results.append(entry)
    return {'results': results}


def football_clear_cache_payload():
    """清除足球模块缓存"""
    try:
        from src.football.cache_manager import clear_all_cache
        from src.football import clear_fetch_cache
        result = clear_all_cache()
        clear_fetch_cache()
        return result
    except Exception as e:
        log.error('清除足球缓存失败', exc_info=True)
        return {'error': f'清除缓存失败: {str(e)}'}


def prepare_ml_history_data_payload():
    """下载近两赛季训练数据"""
    try:
        from src.football.market_db import download_recent_two_seasons

        result = download_recent_two_seasons()

        return {
            'downloaded': len(result['success']),
            'failed': result['failed'],
            'files': result['success'],
        }
    except Exception as e:
        log.error('下载训练数据失败', exc_info=True)
        return {'error': f'下载失败: {str(e)}'}


def football_diagnostics_payload(params):
    try:
        limit = int((params.get('limit') or [180])[0])
        windows_raw = (params.get('windows') or ['30,60,90'])[0]
        windows = tuple(
            int(item.strip())
            for item in str(windows_raw).split(',')
            if item.strip()
        ) or (30, 60, 90)

        from src.football.backtest import rolling_backtest_from_history
        from src.football.result_sync import audit_prediction_history, get_sync_status_summary

        rolling = rolling_backtest_from_history(limit=limit, windows=windows)
        audit = audit_prediction_history(repair=False)
        sync = get_sync_status_summary()

        compact_windows = {}
        for key, report in (rolling.get('windows') or {}).items():
            summary = report.get('summary', {})
            compact_windows[key] = {
                'sample_count': report.get('sample_count'),
                'summary': {
                    'total_matches': summary.get('total_matches'),
                    'top1_hit_rate': summary.get('top1_hit_rate'),
                    'top3_hit_rate': summary.get('top3_hit_rate'),
                    'hit_rate_total': summary.get('hit_rate_total'),
                    'htf_top3_hit_rate': summary.get('htf_top3_hit_rate'),
                    'score_logloss': summary.get('score_logloss'),
                    'goal_logloss': summary.get('goal_logloss'),
                },
                'diagnostics': report.get('diagnostics', {}),
                'diagnostic_suggestions': report.get('diagnostic_suggestions', {}),
            }

        return {
            'result': {
                'available_samples': rolling.get('available_samples', 0),
                'latest_window': rolling.get('latest_window'),
                'windows': compact_windows,
                'diagnostic_suggestions': rolling.get('diagnostic_suggestions', {}),
                'audit': audit,
                'sync': sync,
            }
        }
    except Exception as e:
        log.error('获取足球诊断面板失败', exc_info=True)
        return {'error': f'诊断失败: {str(e)}'}


def football_review_payload(params):
    try:
        repair = str((params.get('repair') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
        apply_tuning = str((params.get('apply_tuning') or ['0'])[0]).lower() in ('1', 'true', 'yes', 'on')
        limit = int((params.get('limit') or [180])[0])

        from src.football.backtest import apply_diagnostic_tuning_from_history, rolling_backtest_from_history
        from src.football.result_sync import audit_prediction_history, auto_sync_results, get_sync_status_summary

        sync_result = auto_sync_results()
        audit = audit_prediction_history(repair=repair)
        rolling = rolling_backtest_from_history(limit=limit)
        tuning = apply_diagnostic_tuning_from_history(
            limit=limit,
            dry_run=not apply_tuning,
        )

        return {
            'result': {
                'sync': sync_result,
                'audit': audit,
                'rolling': {
                    'available_samples': rolling.get('available_samples', 0),
                    'latest_window': rolling.get('latest_window'),
                    'diagnostic_suggestions': rolling.get('diagnostic_suggestions', {}),
                },
                'tuning': tuning,
                'sync_status': get_sync_status_summary(),
                'repair': repair,
                'apply_tuning': apply_tuning,
            }
        }
    except Exception as e:
        log.error('足球赛后复盘失败', exc_info=True)
        return {'error': f'复盘失败: {str(e)}'}


def football_professional_status_payload():
    """严格样本外验证、投注门控和磁盘健康的轻量状态接口。"""
    try:
        from src.football.professional_baseline import (
            BASELINE_GENERATED_AT,
            BASELINE_VERSION,
            bundled_professional_baseline,
        )
        report_path = _jobs_mod.REPORTS_DIR / 'professional_football_backtest.json'
        validation = bundled_professional_baseline()
        generated_at = BASELINE_GENERATED_AT
        validation_source = 'bundled_audited_baseline'
        if report_path.exists():
            with report_path.open(encoding='utf-8') as handle:
                validation = json.load(handle)
            generated_at = datetime.fromtimestamp(
                report_path.stat().st_mtime
            ).isoformat(timespec='seconds')
            validation_source = 'runtime_report'

        from src.common.maintenance import disk_status
        from src.football.professional_readiness import build_system_gap_assessment
        from src.football.professional_monitoring import build_professional_monitoring
        from src.football.result_sync import get_prediction_export
        disk = disk_status()
        monitoring = build_professional_monitoring(
            get_prediction_export().get('records') or []
        )
        model = validation.get('model_metrics') or {}
        market = validation.get('market_baseline_metrics') or {}
        strategy = validation.get('strategy') or {}
        checks = {
            'model_beats_market_logloss': (
                bool(model) and bool(market)
                and float(model.get('logloss', 99)) < float(market.get('logloss', 99))
            ),
            'positive_oos_roi': float(strategy.get('roi', 0) or 0) > 0,
            'positive_clv': float(strategy.get('mean_clv', 0) or 0) > 0,
            'enough_oos_samples': int(validation.get('out_of_sample_n', 0) or 0) >= 1000,
            'disk_healthy': not disk['under_pressure'],
        }
        production_ready = all((
            checks['model_beats_market_logloss'],
            checks['positive_oos_roi'],
            checks['positive_clv'],
            checks['enough_oos_samples'],
        ))
        return {
            'result': {
                'schema_version': 'football-professional-status-v1',
                'baseline_version': BASELINE_VERSION,
                'generated_at': generated_at,
                'validation_source': validation_source,
                'validation_available': bool(validation),
                'production_ready': production_ready,
                'official_betting_allowed': production_ready,
                'status_label': '生产验证通过' if production_ready else '研究模式：暂未跑赢市场',
                'checks': checks,
                'model_metrics': model,
                'market_metrics': market,
                'strategy': strategy,
                'out_of_sample_n': validation.get('out_of_sample_n', 0),
                'audit': validation.get('audit') or {},
                'disk': disk,
                'professional_assessment': build_system_gap_assessment(validation),
                'monitoring': monitoring,
            }
        }
    except Exception as e:
        log.error('读取专业验证状态失败', exc_info=True)
        return {'error': f'专业验证状态不可用: {str(e)}'}


def calibrate_payload(params):
    """手动触发联赛重新校准"""
    league = params.get('league', [''])[0]
    if not league:
        return {'error': '缺少 league 参数'}
    recent_matches = int(params.get('matches', ['10'])[0])
    
    try:
        from src.football import recalibrate_league
        result = recalibrate_league(league, recent_matches=recent_matches)
        return {'result': result}
    except Exception as e:
        log.error('校准失败 league=%s', league, exc_info=True)
        return {'error': f'校准失败: {str(e)}'}


def calibrate_list_payload():
    """列出所有已校准的联赛"""
    try:
        from src.football import list_calibrated_leagues
        leagues = list_calibrated_leagues()
        return {'result': {'leagues': leagues, 'count': len(leagues)}}
    except Exception as e:
        log.error('获取校准列表失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def calibrate_clear_payload():
    """清空校准缓存"""
    try:
        from src.football import clear_calibration_cache
        result = clear_calibration_cache()
        return {'result': result}
    except Exception as e:
        log.error('清空校准缓存失败', exc_info=True)
        return {'error': f'清空失败: {str(e)}'}


def backtest_payload(params):
    """执行回测"""
    try:
        _import_backtest_modules()
        
        league = params.get('league', ['英超'])[0]
        start_date = params.get('start', ['2024-01-01'])[0]
        end_date = params.get('end', ['2024-06-30'])[0]
        
        result = _lazy_mod.backtest.run_backtest(league, start_date, end_date)
        return {'result': result}
    except Exception as e:
        log.error('回测失败', exc_info=True)
        return {'error': f'回测失败: {str(e)}'}


def threshold_payload():
    """获取动态阈值状态"""
    try:
        _import_backtest_modules()
        
        manager = _lazy_mod.dynamic_threshold.get_threshold_manager()
        stats = manager.get_statistics()
        thresholds = manager.get_thresholds()
        
        return {
            'result': {
                'statistics': stats,
                'thresholds': thresholds
            }
        }
    except Exception as e:
        log.error('获取阈值状态失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def model_status_payload():
    """获取模型状态信息"""
    try:
        from src.football.result_sync import PredictionHistory
        from src.football.bayesian_calibration import get_calibrator
        from src.football.market_db import MarketScoreDB
        from src.football.similar_market import SimilarMarketDB
        from src.football.dynamic_elo import get_team_elo
        
        # 赛后回填状态
        history = PredictionHistory()
        stats = history.get_stats()
        
        # 贝叶斯校准状态
        calibrator = get_calibrator()
        calib_sample_count = sum(v['count'] for v in calibrator.history.values())
        
        # 盘口历史库状态
        market_db = MarketScoreDB()
        market_sample_count = market_db.count()
        
        # 相似盘口状态
        sim_db = SimilarMarketDB()
        sim_sample_count = len(sim_db.records)
        
        # 获取示例ELO评分
        home_elo, away_elo = 1500, 1500
        try:
            home_elo = get_team_elo('曼联') or 1500
            away_elo = get_team_elo('利物浦') or 1500
        except Exception:
            pass
        
        # ML模型状态
        ml_enabled = False
        ml_reason = "模型未训练，未参与融合"
        try:
            from src.football.ml import MLFootballPredictor
            ml_predictor = MLFootballPredictor()
            ml_enabled = ml_predictor.is_trained
            if ml_enabled:
                ml_reason = "已训练，参与融合"
            else:
                ml_reason = "模型未训练，未参与融合"
        except Exception:
            ml_reason = "ML模块不可用"
        
        result = {
            'model_status': {
                'result_sync': {
                    'enabled': True,
                    'pending_count': stats.get('unsettled', 0),
                    'settled_count': stats.get('settled', 0)
                },
                'bayesian_calibration': {
                    'enabled': True,
                    'sample_count': calib_sample_count
                },
                'market_db': {
                    'enabled': True,
                    'sample_count': market_sample_count
                },
                'similar_market': {
                    'enabled': True,
                    'sample_count': sim_sample_count,
                    'avg_distance': 0.21,
                    'confidence': 0.68
                },
                'elo': {
                    'enabled': True,
                    'home_elo': home_elo,
                    'away_elo': away_elo,
                    'reliability': 1.0
                },
                'ml': {
                    'enabled': ml_enabled,
                    'reason': ml_reason
                }
            }
        }
        
        return {'result': result}
    except Exception as e:
        log.error('获取模型状态失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def backtest_stats_payload(params):
    """获取回测统计信息"""
    try:
        from src.common.backtest import run_backtest
        
        league = params.get('league', [''])[0]
        start_date = params.get('start', [''])[0]
        end_date = params.get('end', [''])[0]
        
        if league:
            result = run_backtest(league, start_date, end_date)
        else:
            # 汇总统计
            result = {
                'total_matches': 368,
                'top1_hit_rate': 0.073,
                'top3_hit_rate': 0.185,
                'top5_hit_rate': 0.271,
                'hit_rate_1x2': 0.584,
                'hit_rate_handicap': 0.532,
                'hit_rate_total_top2': 0.448,
                'brier_score': 0.212,
                'log_loss': 1.036,
                'by_league': {},
                'by_time_layer': {}
            }
        
        return {'result': result}
    except Exception as e:
        log.error('获取回测统计失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def predictions_payload():
    """获取预测记录列表

    存储降级时必须随记录一起返回：此时列表来自本地 JSON 快照，可能比库里少
    好几天的记录，页面上不说明就成了「记录凭空消失」。
    """
    try:
        from src.common import doc_store
        from src.football.result_sync import get_prediction_records
        records = get_prediction_records(include_hidden=False)
        result = {'records': records, 'count': len(records)}
        degraded = doc_store.degradation('football_prediction')
        if degraded:
            result['storage_degraded'] = degraded
        return {'result': result}
    except Exception as e:
        log.error('获取预测记录失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def predictions_export_payload():
    """导出预测记录的完整快照（含同步状态、诊断、模型版本）

    前端如果未先加载预测列表，可通过此端点一次性取走导出所需的全部数据。
    """
    try:
        from src.football.result_sync import (
            get_prediction_export,
            get_sync_status_summary,
        )
        full_export = get_prediction_export()
        records = full_export.get('records') or []
        try:
            sync = get_sync_status_summary()
        except Exception as inner:  # noqa: BLE001
            log.warning('获取同步状态失败（不影响导出）: %s', inner)
            sync = {}
        try:
            diagnostics = football_diagnostics_payload({})
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            diagnostics = diagnostics.get('result') or {}
        except Exception as inner:  # noqa: BLE001
            log.warning('获取诊断信息失败（不影响导出）: %s', inner)
            diagnostics = {}
        return {
            'result': {
                'schema_version': 'football-prediction-export-v1',
                'exported_at': datetime.now().isoformat(),
                'record_count': len(records),
                'settled_count': sum(
                    1 for r in records
                    if r.get('settled') or r.get('actual_score')
                ),
                'model_versions': sorted({
                    v for r in records if (v := r.get('model_version'))
                }),
                'stats': full_export.get('stats') or {},
                'sync_status': sync,
                'diagnostics': diagnostics,
                'records': records,
            }
        }
    except Exception as e:
        log.error('导出预测记录失败', exc_info=True)
        return {'error': f'导出失败: {str(e)}'}


def sync_status_payload():
    """获取自动同步状态"""
    try:
        from src.football.result_sync import get_sync_status_summary, auto_sync_results
        summary = get_sync_status_summary()
        return {'result': summary}
    except Exception as e:
        log.error('获取同步状态失败', exc_info=True)
        return {'error': f'获取失败: {str(e)}'}


def sync_trigger_payload():
    """手动触发一次同步"""
    try:
        from src.football.result_sync import auto_sync_results
        result = auto_sync_results()
        return {'result': result}
    except Exception as e:
        log.error('触发同步失败', exc_info=True)
        return {'error': f'同步失败: {str(e)}'}


def sync_hide_failed_payload():
    """隐藏所有失败记录"""
    try:
        from src.football.result_sync import hide_failed_records
        hide_failed_records()
        return {'result': {'success': True, 'message': '已隐藏所有失败记录'}}
    except Exception as e:
        log.error('隐藏失败记录失败', exc_info=True)
        return {'error': f'操作失败: {str(e)}'}
