# -*- coding: utf-8 -*-
"""足球接口 handler（mixin）"""

import os
import sys
import json
import math
import hmac
import base64
import socket
import time
import re
from datetime import datetime, timedelta
import importlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from src.common.logger import setup_logger
from src.common.paths import data_path

log = setup_logger('server')
from . import lazy_modules as _lazy_mod

from .lazy_modules import (
    _import_backtest_modules, analyze_match, ensure_beidan_report, ensure_football_report, fetch_match_list, football_reportable_ids, get_match_list_status,
)
from .jobs import (
    _attach_bayes_report_url, _match_started, _trigger_football_analysis, _trigger_football_report_sync,
)
from . import jobs as _jobs_mod

class FootballApiMixin:
    def _serve_report_file(self, path):
        """提供 reports/ 目录下的静态报告文件（HTML/JSON）。"""
        rel = path[len('/reports/'):].lstrip('/')
        if not rel or '..' in rel or rel.startswith('.'):
            return self._send_json_error(403, 'Forbidden')
        file_path = _jobs_mod.REPORTS_DIR / rel
        try:
            file_path = file_path.resolve()
            reports_root = _jobs_mod.REPORTS_DIR.resolve()
            if not str(file_path).startswith(str(reports_root)):
                return self._send_json_error(403, 'Forbidden')
        except Exception:
            return self._send_json_error(404, 'Not Found')
        if rel.startswith('football_bayes_') and rel.endswith('.html'):
            # ensure_football_report performs a cheap schema/odds check and
            # regenerates stale report layouts on first access.
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
        if not file_path.exists() or not file_path.is_file():
            # 报告文件不存在 → 若可生成则按需现生成（生产环境无需手动跑脚本）
            generated = self._try_generate_report(rel)
            if generated and os.path.exists(generated):
                file_path = Path(generated)
            else:
                return self._send_json_error(404, 'Not Found')
        content_type = 'text/html; charset=utf-8'
        if file_path.suffix.lower() == '.json':
            content_type = 'application/json; charset=utf-8'
        try:
            with open(file_path, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._log.error('读取报告文件失败: %s', file_path, exc_info=True)
            self._send_json_error(500, f'读取报告失败: {e}')


    def _try_generate_report(self, rel: str):
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
            self._log.error('报告按需生成失败: %s', rel, exc_info=True)
        return None


    def _matches_payload(self):
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
            self._log.error('获取比赛列表失败', exc_info=True)
            return {
                'error': '获取比赛列表失败',
                'source_status': get_match_list_status(),
                'error_type': type(exc).__name__,
            }


    def _predict_payload(self, params):
        match_id = params.get('match_id', [''])[0]
        if not match_id:
            return {'error': '缺少 match_id 参数'}
        
        # 检查是否强制刷新缓存
        force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'
        
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

        match = {
            'match_id': match_id,
            'home': params.get('home', [''])[0],
            'away': params.get('away', [''])[0],
            'league': params.get('league', [''])[0],
            'time': params.get('time', [''])[0],
            'num': params.get('num', [''])[0],
            'schedule_source': params.get('schedule_source', [''])[0],
            'analysis_source_id_available': params.get('analysis_source_id_available', ['true'])[0].lower() == 'true',
            'okooo_id': params.get('okooo_id', [''])[0],
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
        try:
            return {'result': analyze_match(match, force_refresh=force_refresh)}
        except ValueError as e:
            error_msg = str(e)
            self._log.error('赔率分析失败 match_id=%s: %s', match_id, error_msg)
            return {'error': error_msg}
        except Exception as e:
            error_msg = f'赔率分析失败: {str(e)}'
            self._log.error('赔率分析失败 match_id=%s', match_id, exc_info=True)
            return {'error': error_msg}


    def _football_clear_cache_payload(self):
        """清除足球模块缓存"""
        try:
            from src.football.cache_manager import clear_all_cache
            from src.football import clear_fetch_cache
            result = clear_all_cache()
            clear_fetch_cache()
            return result
        except Exception as e:
            self._log.error('清除足球缓存失败', exc_info=True)
            return {'error': f'清除缓存失败: {str(e)}'}


    def _prepare_ml_history_data_payload(self):
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
            self._log.error('下载训练数据失败', exc_info=True)
            return {'error': f'下载失败: {str(e)}'}


    def _football_diagnostics_payload(self, params):
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
            self._log.error('获取足球诊断面板失败', exc_info=True)
            return {'error': f'诊断失败: {str(e)}'}


    def _football_review_payload(self, params):
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
            self._log.error('足球赛后复盘失败', exc_info=True)
            return {'error': f'复盘失败: {str(e)}'}


    def _football_professional_status_payload(self):
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
            self._log.error('读取专业验证状态失败', exc_info=True)
            return {'error': f'专业验证状态不可用: {str(e)}'}


    def _calibrate_payload(self, params):
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
            self._log.error('校准失败 league=%s', league, exc_info=True)
            return {'error': f'校准失败: {str(e)}'}


    def _calibrate_list_payload(self):
        """列出所有已校准的联赛"""
        try:
            from src.football import list_calibrated_leagues
            leagues = list_calibrated_leagues()
            return {'result': {'leagues': leagues, 'count': len(leagues)}}
        except Exception as e:
            self._log.error('获取校准列表失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _calibrate_clear_payload(self):
        """清空校准缓存"""
        try:
            from src.football import clear_calibration_cache
            result = clear_calibration_cache()
            return {'result': result}
        except Exception as e:
            self._log.error('清空校准缓存失败', exc_info=True)
            return {'error': f'清空失败: {str(e)}'}


    def _backtest_payload(self, params):
        """执行回测"""
        try:
            _import_backtest_modules()
            
            league = params.get('league', ['英超'])[0]
            start_date = params.get('start', ['2024-01-01'])[0]
            end_date = params.get('end', ['2024-06-30'])[0]
            
            result = _lazy_mod.backtest.run_backtest(league, start_date, end_date)
            return {'result': result}
        except Exception as e:
            self._log.error('回测失败', exc_info=True)
            return {'error': f'回测失败: {str(e)}'}


    def _threshold_payload(self):
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
            self._log.error('获取阈值状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _model_status_payload(self):
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
            self._log.error('获取模型状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _backtest_stats_payload(self, params):
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
            self._log.error('获取回测统计失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _predictions_payload(self):
        """获取预测记录列表"""
        try:
            from src.football.result_sync import get_prediction_records
            records = get_prediction_records(include_hidden=False)
            return {'result': {'records': records, 'count': len(records)}}
        except Exception as e:
            self._log.error('获取预测记录失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _predictions_export_payload(self):
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
                self._log.warning('获取同步状态失败（不影响导出）: %s', inner)
                sync = {}
            try:
                diagnostics = self._football_diagnostics_payload({})
                if not isinstance(diagnostics, dict):
                    diagnostics = {}
                diagnostics = diagnostics.get('result') or {}
            except Exception as inner:  # noqa: BLE001
                self._log.warning('获取诊断信息失败（不影响导出）: %s', inner)
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
            self._log.error('导出预测记录失败', exc_info=True)
            return {'error': f'导出失败: {str(e)}'}


    def _sync_status_payload(self):
        """获取自动同步状态"""
        try:
            from src.football.result_sync import get_sync_status_summary, auto_sync_results
            summary = get_sync_status_summary()
            return {'result': summary}
        except Exception as e:
            self._log.error('获取同步状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}


    def _sync_trigger_payload(self):
        """手动触发一次同步"""
        try:
            from src.football.result_sync import auto_sync_results
            result = auto_sync_results()
            return {'result': result}
        except Exception as e:
            self._log.error('触发同步失败', exc_info=True)
            return {'error': f'同步失败: {str(e)}'}


    def _sync_hide_failed_payload(self):
        """隐藏所有失败记录"""
        try:
            from src.football.result_sync import hide_failed_records
            hide_failed_records()
            return {'result': {'success': True, 'message': '已隐藏所有失败记录'}}
        except Exception as e:
            self._log.error('隐藏失败记录失败', exc_info=True)
            return {'error': f'操作失败: {str(e)}'}

