# -*- coding: utf-8 -*-
"""快乐8接口 handler（mixin）"""

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

from .lazy_modules import (
    KL8RollingBacktest, get_kl8_analyzer, kl8_check_data_integrity, kl8_clear_cache, kl8_list_conflict_queue, kl8_list_recalculations, kl8_list_snapshots, kl8_run_prediction, validate_and_activate_strategy,
)
from .caching import (
    _CACHE, _is_kl8_cache_current,
)
from .jobs import (
    _get_kl8_parameter_search_job, _run_kl8_parameter_search_job, _save_kl8_parameter_search_report, _set_kl8_parameter_search_job,
)

class KL8ApiMixin:
    def _kl8_payload(self):
        """获取快乐8预测结果"""
        try:
            now = time.time()
            cache = _CACHE['kl8']

            if cache['data'] is not None and _is_kl8_cache_current(cache, now):
                self._log.info('快乐8使用缓存')
                return {'result': cache['data']}

            self._log.info('快乐8重新计算')
            result = kl8_run_prediction(force_refresh=False)

            if 'error' in result:
                return {'error': result['error']}

            cache['data'] = result
            cache['timestamp'] = now
            return {'result': result}
        except Exception:
            self._log.error('快乐8预测失败', exc_info=True)
            return {'error': '快乐8预测失败'}


    def _kl8_refresh_payload(self):
        """强制刷新快乐8数据缓存"""
        try:
            self._log.info('快乐8强制刷新请求到达')
            kl8_clear_cache()
            _CACHE['kl8']['data'] = None
            _CACHE['kl8']['timestamp'] = 0

            result = kl8_run_prediction(force_refresh=True)

            _CACHE['kl8']['data'] = result
            _CACHE['kl8']['timestamp'] = time.time()

            return {'success': True, 'result': result}
        except Exception:
            self._log.error('快乐8刷新失败', exc_info=True)
            return {'error': '快乐8刷新失败'}


    def _kl8_fetch_payload(self):
        """抓取最新快乐8开奖数据"""
        try:
            self._log.info('快乐8抓取最新数据请求到达')
            from src.kl8.fetch import fetch_kl8_data, save_kl8_data

            # 日常只需最近两页；历史补数由独立调度器负责。
            data = fetch_kl8_data(pages=2, per_page=50)
            if not data:
                return {'success': False, 'message': '网络抓取失败'}

            # v2: 合并保存（不是覆盖），save_kl8_data内部会调clear_cache()
            save_ok = save_kl8_data(data)
            if not save_ok:
                return {'success': False, 'message': '数据量不足，不允许覆盖原历史'}

            # 重新预测（clear_cache已在save内部完成）
            _CACHE['kl8']['data'] = None
            _CACHE['kl8']['timestamp'] = 0

            result = kl8_run_prediction(force_refresh=True)
            _CACHE['kl8']['data'] = result
            _CACHE['kl8']['timestamp'] = time.time()

            return {
                'success': True,
                'message': f'成功抓取 {len(data)} 期数据',
                'latest_issue': data[0]['issue'] if data else '',
                'result': result,
            }
        except Exception:
            self._log.error('快乐8抓取失败', exc_info=True)
            return {'error': '快乐8抓取失败'}


    def _kl8_exclude_recalculate_payload(self, params):
        """剔除指定玩法当前号码后临时重算，不覆盖正式预测。"""
        try:
            play_type = (params.get('play_type') or [''])[0]
            numbers_str = (params.get('numbers') or [''])[0]
            if not play_type:
                return {'error': '缺少play_type参数'}

            try:
                exclude_numbers = [
                    int(x.strip())
                    for x in numbers_str.split(',')
                    if x.strip()
                ] if numbers_str else []
            except ValueError:
                return {'error': 'numbers格式错误，应为逗号分隔的1-80整数'}

            analyzer = get_kl8_analyzer()
            current = _CACHE.get('kl8', {}).get('data') or {}
            snapshot_file = str(current.get('snapshot_file') or '')
            source_snapshot_id = Path(snapshot_file).stem.replace('snapshot_', '', 1) if snapshot_file else ''
            current_play = current.get(play_type) or {}
            initial_numbers = (
                current_play.get('numbers')
                or current_play.get('core_numbers')
                or current_play.get('top8_numbers')
                or current_play.get('top11_numbers')
                or []
            ) if isinstance(current_play, dict) else []
            result = analyzer.recalculate_play_excluding(
                play_type,
                exclude_numbers,
                record_context={
                    'source_snapshot_id': source_snapshot_id,
                    'source_version': current.get('statistics', {}).get('version') or current.get('version') or '',
                    'generation_mode': 'manual',
                    'initial_numbers': initial_numbers,
                },
            )
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8剔除重算失败', exc_info=True)
            return {'error': f'剔除重算失败: {str(e)}'}


    def _kl8_snapshots_payload(self):
        """快乐8预测快照列表"""
        try:
            snapshots = kl8_list_snapshots()
            return {'result': {'snapshots': snapshots, 'count': len(snapshots)}}
        except Exception:
            self._log.error('快乐8快照列表失败', exc_info=True)
            return {'error': '快乐8快照列表失败'}


    def _kl8_records_payload(self):
        """快乐8预测记录 + 中奖情况（快照元数据 + 结算详情合并）

        参考足球 /api/predictions：一次返回全部记录（含中奖结算），
        前端用模态弹窗 + 分页展示，避免在当页面内联无限输出。
        """
        try:
            from src.kl8 import KL8_SETTLEMENT_DIR, KL8_SNAPSHOT_DIR
            from pathlib import Path
            import json as _json

            snapshots = kl8_list_snapshots()
            # 按目标期号降序（最新一期在前）
            snapshots.sort(
                key=lambda s: (
                    str(s.get('target_issue') or ''),
                    str(s.get('predicted_at') or ''),
                ),
                reverse=True,
            )

            settlement_dir = Path(KL8_SETTLEMENT_DIR)
            snapshot_dir = Path(KL8_SNAPSHOT_DIR)
            records = []
            for snap in snapshots:
                # 读取完整快照以提取预测号码（用于"历史记录"展示）
                predicted = {}
                main_pool = {}
                try:
                    raw = _json.loads((snapshot_dir / snap['file']).read_text(encoding='utf-8'))
                    for k in raw:
                        if k.startswith('select_') or k.startswith('fu_shi'):
                            blk = raw.get(k)
                            if isinstance(blk, dict):
                                if blk.get('main_pool'):
                                    main_pool[k] = blk['main_pool']
                                elif blk.get('numbers'):
                                    predicted[k] = blk['numbers']
                            elif isinstance(blk, list):
                                predicted[k] = blk
                    # 复式核心号码（旧字段名兼容）
                    for k in ('fu_shi_7',):
                        if not predicted.get(k):
                            core = raw.get(k)
                            if isinstance(core, dict):
                                predicted[k] = core.get('core_numbers') or core.get('top7_numbers') or []
                except Exception:
                    predicted = {}
                    main_pool = {}

                rec = {
                    'snapshot_id': snap.get('snapshot_id'),
                    'file': snap.get('file'),
                    'target_issue': snap.get('target_issue'),
                    'based_on_issue': snap.get('based_on_issue'),
                    'predicted_at': snap.get('predicted_at'),
                    'version': snap.get('version'),
                    'is_experiment': snap.get('is_experiment', False),
                    'has_settlement': snap.get('has_settlement', False),
                    'predicted': predicted,
                    'main_pool': main_pool,
                    'settlement': None,
                }
                if snap.get('has_settlement') and snap.get('snapshot_id'):
                    sp = settlement_dir / f'settlement_{snap["snapshot_id"]}.json'
                    if sp.exists():
                        try:
                            rec['settlement'] = _json.loads(sp.read_text(encoding='utf-8'))
                        except Exception:
                            rec['settlement'] = None
                records.append(rec)

            # 结算回填：对已开奖但缺少结算文件的快照当场结算（幂等），
            # 修复"某一期因服务停机/漏检未被调度器结算 → 预测记录永久卡在待开奖"的问题。
            self._kl8_backfill_settlements(records)

            # 奖金表更新重算：旧版默认奖金表错误（如选5中2=5元、选6中3=10元），
            # 已生成的结算文件不会自动更新。读记录时校验并删除重算，确保金额正确。
            self._kl8_rebuild_stale_settlements(records)

            # 删号重算必须绑定来源快照，不能把同一期不同模型版本的轨迹串在一起。
            recalculations_by_snapshot = {}
            for item in kl8_list_recalculations():
                source_id = str(item.get('source_snapshot_id') or '')
                if source_id:
                    recalculations_by_snapshot.setdefault(source_id, []).append(item)
            for rec in records:
                rounds = recalculations_by_snapshot.get(str(rec.get('snapshot_id') or ''), [])
                actual = set((rec.get('settlement') or {}).get('actual_numbers') or [])
                enriched = []
                for item in rounds:
                    row = dict(item)
                    nums = row.get('numbers') or []
                    row['hits'] = len(set(nums) & actual) if actual and nums else None
                    enriched.append(row)
                enriched.sort(key=lambda row: (str(row.get('play_type') or ''), int(row.get('round', 0))))
                rec['exclude_recalculations'] = enriched

            # 去重：同一目标期只保留最新一条（调度器每轮可能对同期生成多次快照）
            seen_issues = {}
            for rec in records:
                issue = rec.get('target_issue')
                if issue and issue not in seen_issues:
                    seen_issues[issue] = rec
            records = list(seen_issues.values())

            settled = sum(1 for r in records if r['has_settlement'])
            return {
                'result': {
                    'records': records,
                    'count': len(records),
                    'settled_count': settled,
                    'pending_count': len(records) - settled,
                }
            }
        except Exception as e:
            self._log.error('快乐8预测记录失败', exc_info=True)
            return {'error': f'获取预测记录失败: {str(e)}'}


    def _kl8_backfill_settlements(self, records):
        """对已开奖但缺少结算文件的快照当场结算（幂等回填）。

        原因：调度器仅在「发现新期号」时结算上一期（settle_previous_period）。
        若某一期因服务停机/漏检未被结算，其预测记录会永久卡在「待开奖」。
        读记录时回填，保证历史已开奖记录正确展示命中/奖金。

        仅对 target_issue 已在历史开奖数据中出现的快照结算；未开奖的（最新一期）
        保持待开奖。settle_prediction 内部按 snapshot_id 落盘且幂等（已存在则跳过）。
        """
        try:
            analyzer = get_kl8_analyzer()
            history = getattr(analyzer, 'history_data', None) or []
            if not history:
                return
            drawn = {str(r.get('issue')): r.get('numbers') for r in history}
            for rec in records:
                if rec.get('has_settlement') or rec.get('settlement'):
                    continue
                ti = rec.get('target_issue')
                if not ti:
                    continue
                ti = str(ti)
                if ti not in drawn:
                    continue  # 目标期尚未开奖，保持待开奖
                numbers = drawn[ti]
                try:
                    result = analyzer.settle_prediction(rec.get('file'), ti, numbers)
                except Exception:
                    continue
                if isinstance(result, dict) and result.get('success'):
                    rec['has_settlement'] = True
                    rec['settlement'] = result.get('settlement')
        except Exception as e:
            self._log.warning('快乐8结算回填异常(已忽略): %s', e)


    def _kl8_rebuild_stale_settlements(self, records):
        """奖金表更新后，强制覆盖重算奖金不一致的历史结算。

        背景：2026-07-22 前默认奖金表有误（选5中2=5、选6中3=10 等），导致已结算
        历史记录的金额错误。此函数在读 /api/kl8/records 时触发，用当前官方奖金表
        重新校验每条 settlement；只要任一玩法的单注奖金与当前表不符，就以 force=True
        重新结算并覆盖旧结算文件。
        """
        try:
            from src.kl8 import load_prize_table, SELECT_TYPES

            analyzer = get_kl8_analyzer()
            history = getattr(analyzer, 'history_data', None) or []
            if not history:
                return
            drawn = {str(r.get('issue')): r.get('numbers') for r in history}
            prize_table = load_prize_table()

            for rec in records:
                st = rec.get('settlement')
                if not st:
                    continue
                ti = rec.get('target_issue')
                if not ti:
                    continue
                ti = str(ti)
                if ti not in drawn:
                    continue

                # 校验各单式玩法的奖金是否与当前奖金表一致
                ps = st.get('prize_settlement', {})
                needs_rebuild = False
                for select_type in SELECT_TYPES:
                    key = f'select_{select_type}'
                    p = ps.get(key, {})
                    if not p.get('placed'):
                        continue
                    expected = prize_table.get(key, {}).get(str(p.get('hits')), 0)
                    if p.get('prize') != expected:
                        needs_rebuild = True
                        break

                if not needs_rebuild:
                    continue

                # 强制覆盖重新结算（不删除文件，避免 safe-delete 拦截）
                try:
                    result = analyzer.settle_prediction(rec.get('file'), ti, drawn[ti], force=True)
                except Exception:
                    continue
                if isinstance(result, dict) and result.get('success'):
                    rec['has_settlement'] = True
                    rec['settlement'] = result.get('settlement')
        except Exception as e:
            self._log.warning('快乐8历史结算重算异常(已忽略): %s', e)


    def _kl8_settle_payload(self, params):
        """快乐8赛后结算 -- v4: 给结算单独保存(不复写原始快照), 增加期号校验"""
        try:
            snapshot_file = (params.get('snapshot') or [''])[0]
            actual_issue = (params.get('issue') or [''])[0]
            numbers_str = (params.get('numbers') or [''])[0]
            if not snapshot_file or not actual_issue or not numbers_str:
                return {'error': '缺少snapshot、issue和numbers参数'}

            # 解析号码
            try:
                actual_numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
            except ValueError:
                return {'error': '号码格式错误，应为逗号分隔的1-80整数'}

            analyzer = get_kl8_analyzer()
            result = analyzer.settle_prediction(snapshot_file, actual_issue, actual_numbers)
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8结算失败', exc_info=True)
            return {'error': f'结算失败: {str(e)}'}


    def _kl8_backtest_payload(self, params):
        """快乐8滚动回测（v5: 最低300期OOS，默认500）"""
        try:
            test_periods_str = (params.get('periods') or ['500'])[0]
            test_periods = int(test_periods_str)
            if test_periods < 300:
                return {'error': f'回测期数最低300期（v5要求BACKTEST_MIN_OOS={300}），当前输入={test_periods}'}

            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': '历史数据不足，无法回测'}

            bt = KL8RollingBacktest(analyzer)
            result = bt.run_full_backtest(test_periods=test_periods)
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8回测失败', exc_info=True)
            return {'error': f'回测失败: {str(e)}'}


    def _kl8_parameter_search_payload(self, params):
        try:
            options_or_error = self._parse_kl8_parameter_search_options(params)
            if 'error' in options_or_error:
                return options_or_error
            options = options_or_error['options']
            async_str = (params.get('async') or ['false'])[0]

            if async_str.lower() in ('true', '1', 'yes', 'on'):
                return {'result': self._start_kl8_parameter_search_job(options)}

            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': 'no KL8 history data'}

            bt = KL8RollingBacktest(analyzer)
            result = bt.run_parameter_search(
                play_types=options.get('play_types'),
                max_candidates=options.get('max_candidates', 24),
                top_n=options.get('top_n', 5),
            )
            job_id = uuid.uuid4().hex
            report_file = _save_kl8_parameter_search_report(job_id, result, options)
            if isinstance(result, dict):
                result = dict(result)
                result['report_file'] = report_file
                result['job_id'] = job_id
            return {'result': result}
        except Exception as e:
            self._log.error('KL8 parameter search failed', exc_info=True)
            return {'error': f'parameter search failed: {str(e)}'}


    def _parse_kl8_parameter_search_options(self, params):
        max_candidates_str = (params.get('max_candidates') or ['24'])[0]
        top_n_str = (params.get('top_n') or ['5'])[0]
        play_types_str = (params.get('play_types') or [''])[0]

        try:
            max_candidates = int(max_candidates_str)
        except ValueError:
            return {'error': 'max_candidates must be an integer'}

        try:
            top_n = int(top_n_str)
        except ValueError:
            return {'error': 'top_n must be an integer'}

        if max_candidates <= 0:
            return {'error': 'max_candidates must be > 0'}
        if top_n <= 0:
            return {'error': 'top_n must be > 0'}

        play_types = [
            item.strip()
            for item in play_types_str.split(',')
            if item.strip()
        ] if play_types_str else None

        return {
            'options': {
                'max_candidates': max_candidates,
                'top_n': top_n,
                'play_types': play_types,
            }
        }


    def _start_kl8_parameter_search_job(self, options):
        job_id = uuid.uuid4().hex
        now = time.time()
        _set_kl8_parameter_search_job(job_id, {
            'job_id': job_id,
            'status': 'queued',
            'created_at': now,
            'started_at': None,
            'finished_at': None,
            'options': options,
            'message': 'queued',
        })
        thread = threading.Thread(
            target=_run_kl8_parameter_search_job,
            args=(job_id, options),
            daemon=True,
            name=f'KL8ParameterSearch-{job_id[:8]}',
        )
        thread.start()
        return _get_kl8_parameter_search_job(job_id)


    def _kl8_parameter_search_start_payload(self, params):
        options_or_error = self._parse_kl8_parameter_search_options(params)
        if 'error' in options_or_error:
            return options_or_error
        return {'result': self._start_kl8_parameter_search_job(options_or_error['options'])}


    def _kl8_parameter_search_status_payload(self, params):
        job_id = (params.get('job_id') or [''])[0]
        if not job_id:
            return {'error': 'missing job_id'}

        job = _get_kl8_parameter_search_job(job_id)
        if not job:
            return {'error': f'job not found: {job_id}'}

        now = time.time()
        started_at = job.get('started_at') or job.get('created_at') or now
        finished_at = job.get('finished_at') or now
        job['elapsed_seconds'] = round(max(0, finished_at - started_at), 1)
        return {'result': job}


    def _kl8_integrity_payload(self):
        """快乐8数据完整性检查"""
        try:
            analyzer = get_kl8_analyzer()
            if not analyzer.history_data:
                return {'error': '无历史数据'}
            integrity = kl8_check_data_integrity(analyzer.history_data)
            return {'result': integrity}
        except Exception:
            self._log.error('快乐8数据完整性检查失败', exc_info=True)
            return {'error': '数据完整性检查失败'}


    def _kl8_conflicts_payload(self):
        """快乐8冲突审核队列"""
        try:
            conflicts = kl8_list_conflict_queue()
            return {'result': {'conflicts': conflicts, 'count': len(conflicts)}}
        except Exception:
            self._log.error('快乐8冲突队列查询失败', exc_info=True)
            return {'error': '冲突队列查询失败'}


    def _kl8_activate_payload(self, params):
        """快乐8策略激活（v7: 回测验证后才允许写入ACTIVE_STRATEGIES）

        参数:
            play_type: 玩法名称 (select_3~select_10, fu_shi_7, fu_shi_10_11)
            feature_weights: JSON字符串，如 {"frequency":0.12}
            model_weights: JSON字符串，如 {"rank":1.0,"bayesian":0.0,"markov":0.0}
            window_size: 统计窗口大小，如 250
            repeat_direction: 重号方向 neutral/avoid/follow
            repeat_avoid_score/repeat_non_avoid_score: 避免重号分数
            repeat_follow_score/repeat_non_follow_score: 跟随重号分数
            pool_diversify: 是否启用候选池分散化
            pool_max_last_numbers: 候选池最多保留上期号码数量
            auto_activate: 是否自动激活（默认false，需人工确认）
            n_permutations: 置换检验次数（默认1000）
        """
        try:
            play_type = (params.get('play_type') or [''])[0]
            feature_weights_json = (params.get('feature_weights') or [''])[0]
            model_weights_json = (params.get('model_weights') or [''])[0]
            window_size_str = (params.get('window_size') or ['0'])[0]
            repeat_direction = (params.get('repeat_direction') or ['neutral'])[0].strip().lower()
            repeat_avoid_score_str = (params.get('repeat_avoid_score') or ['0.10'])[0]
            repeat_non_avoid_score_str = (params.get('repeat_non_avoid_score') or ['0.85'])[0]
            repeat_follow_score_str = (params.get('repeat_follow_score') or ['0.90'])[0]
            repeat_non_follow_score_str = (params.get('repeat_non_follow_score') or ['0.50'])[0]
            pool_diversify_str = (params.get('pool_diversify') or ['true'])[0]
            pool_max_last_numbers_str = (params.get('pool_max_last_numbers') or [''])[0]
            auto_activate_str = (params.get('auto_activate') or ['false'])[0]
            n_permutations_str = (params.get('n_permutations') or ['1000'])[0]

            if not play_type:
                return {'error': '缺少play_type参数'}

            if not feature_weights_json:
                return {'error': '缺少feature_weights参数'}

            try:
                feature_weights = json.loads(feature_weights_json)
            except json.JSONDecodeError:
                return {'error': 'feature_weights JSON解析失败'}

            try:
                model_weights = json.loads(model_weights_json) if model_weights_json else {}
            except json.JSONDecodeError:
                return {'error': 'model_weights JSON解析失败'}

            try:
                window_size = int(window_size_str)
            except ValueError:
                return {'error': 'window_size必须是整数'}

            if repeat_direction not in ('neutral', 'avoid', 'follow'):
                return {'error': 'repeat_direction必须是 neutral/avoid/follow'}

            try:
                repeat_avoid_score = float(repeat_avoid_score_str)
                repeat_non_avoid_score = float(repeat_non_avoid_score_str)
                repeat_follow_score = float(repeat_follow_score_str)
                repeat_non_follow_score = float(repeat_non_follow_score_str)
            except ValueError:
                return {'error': 'repeat分数参数必须是数字'}

            pool_diversify = pool_diversify_str.lower() in ('true', '1', 'yes', 'on')
            pool_max_last_numbers = None
            if pool_max_last_numbers_str.strip():
                try:
                    pool_max_last_numbers = int(pool_max_last_numbers_str)
                except ValueError:
                    return {'error': 'pool_max_last_numbers必须是整数'}
                if pool_max_last_numbers < 0:
                    return {'error': 'pool_max_last_numbers必须大于等于0'}

            auto_activate = auto_activate_str.lower() in ('true', '1', 'yes')

            try:
                n_permutations = int(n_permutations_str)
            except ValueError:
                n_permutations = 1000

            result = validate_and_activate_strategy(
                play_type=play_type,
                feature_weights=feature_weights,
                model_weights=model_weights,
                window_size=window_size,
                repeat_direction=repeat_direction,
                repeat_avoid_score=repeat_avoid_score,
                repeat_non_avoid_score=repeat_non_avoid_score,
                repeat_follow_score=repeat_follow_score,
                repeat_non_follow_score=repeat_non_follow_score,
                pool_diversify=pool_diversify,
                pool_max_last_numbers=pool_max_last_numbers,
                auto_activate=auto_activate,
                n_permutations=n_permutations,
            )
            return {'result': result}
        except Exception as e:
            self._log.error('快乐8策略激活失败', exc_info=True)
            return {'error': f'策略激活失败: {str(e)}'}

