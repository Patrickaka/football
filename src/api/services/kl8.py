# -*- coding: utf-8 -*-
"""快乐 8 接口的业务装配。

**新旧两个入口共用同一份**（判据 11）。原来 mixin 里的方法互调（结算回填、
陈旧结算重建、参数搜索的选项解析与起任务）提升后都是模块级函数，
直接按函数名互调。
"""

import logging

import threading
import uuid
import re
import json
import time
from pathlib import Path

from src.common.paths import data_path
from src.api.runtime.lazy_modules import (
    KL8RollingBacktest, get_kl8_analyzer, kl8_check_data_integrity,
    kl8_clear_cache, kl8_list_conflict_queue, kl8_list_recalculations,
    kl8_list_snapshots, kl8_run_prediction, validate_and_activate_strategy,
)
from src.api.runtime import kl8_cache
from src.api.runtime.shared_cache import get_cache as get_shared_cache
from src.api.runtime.caching import (
    _CACHE, _current_kl8_predictor_version,
)
from src.api.runtime.jobs import (
    _get_kl8_refresh_job, _start_kl8_refresh_job,
    _get_kl8_parameter_search_job, _run_kl8_parameter_search_job,
    _save_kl8_parameter_search_report, _set_kl8_parameter_search_job,
)

log = logging.getLogger('api.services.kl8')

_KL8_RECORDS_DEFAULT_PAGE_SIZE = 8
_KL8_RECORDS_MAX_PAGE_SIZE = 50
_KL8_RECORDS_MAINTENANCE_INTERVAL = 300.0
_kl8_records_maintenance_lock = threading.Lock()
_kl8_compat_cache_lock = threading.Lock()
_kl8_refresh_execution_lock = threading.Lock()
_kl8_records_maintenance_running = False
_kl8_records_maintenance_last_finished = 0.0


def _current_kl8_config_fingerprint():
    """取得当前完整策略指纹，供跨进程预测缓存键使用。"""
    try:
        from src.kl8.snapshots import _strategy_config_fingerprint

        return str(_strategy_config_fingerprint() or '')
    except Exception:
        log.warning('读取快乐8策略指纹失败，本次仅按代码版本缓存', exc_info=True)
        return ''


def _kl8_cache_version(version=None, config_fingerprint=None):
    """把同代码版本内的策略变更也编码进缓存身份。"""
    version = str(version or _current_kl8_predictor_version())
    fingerprint = str(
        config_fingerprint
        if config_fingerprint is not None
        else _current_kl8_config_fingerprint()
    )
    return f'{version}:{fingerprint}' if fingerprint else version


def _kl8_prediction_freshness(data):
    if not isinstance(data, dict):
        return ('', False, False, 0, '')
    statistics = data.get('statistics') or {}
    version = str(statistics.get('version') or data.get('version') or '')
    config_fingerprint = str(
        data.get('strategy_config_fingerprint')
        or statistics.get('strategy_config_fingerprint')
        or ''
    )
    issue = str(
        statistics.get('based_on_issue')
        or data.get('based_on_issue')
        or ''
    )
    generated_at = str(
        statistics.get('prediction_generated_at')
        or data.get('prediction_generated_at')
        or ''
    )
    try:
        generated_at_ns = int(
            statistics.get('prediction_generated_at_ns')
            or data.get('prediction_generated_at_ns')
            or 0
        )
    except (TypeError, ValueError):
        generated_at_ns = 0
    current_config = _current_kl8_config_fingerprint()
    latest_issue = str(kl8_latest_issue() or '')
    return (
        issue,
        version == _current_kl8_predictor_version()
        and (not config_fingerprint or config_fingerprint == current_config),
        not latest_issue or issue == latest_issue,
        generated_at_ns,
        generated_at,
    )


def _sync_kl8_compat_cache(result):
    """按版本、期号和生成时间单调更新旧进程状态。"""
    if not isinstance(result, dict) or result.get('error'):
        return

    with _kl8_compat_cache_lock:
        previous = _CACHE['kl8'].get('data')
        if (
            previous is not None
            and _kl8_prediction_freshness(result)
            <= _kl8_prediction_freshness(previous)
        ):
            return
        _CACHE['kl8']['data'] = result
        _CACHE['kl8']['timestamp'] = time.time()


def _publish_kl8_prediction(result):
    """成功计算后同时替换兼容缓存和两层共享缓存。"""
    if not isinstance(result, dict) or result.get('error'):
        raise RuntimeError(
            str(result.get('error') if isinstance(result, dict) else '')
            or '快乐8重新计算未返回有效结果'
        )

    statistics = result.get('statistics') or {}
    result_issue = str(
        statistics.get('based_on_issue')
        or result.get('based_on_issue')
        or ''
    )
    latest_issue = str(kl8_latest_issue() or '')
    if result_issue and latest_issue and result_issue != latest_issue:
        raise RuntimeError(
            f'开奖号已从{result_issue}更新到{latest_issue}，旧预测已丢弃，请重试'
        )
    cache_issue = result_issue or latest_issue
    version = str(
        statistics.get('version')
        or result.get('version')
        or _current_kl8_predictor_version()
    )
    result_config = str(
        result.get('strategy_config_fingerprint')
        or statistics.get('strategy_config_fingerprint')
        or ''
    )
    current_config = _current_kl8_config_fingerprint()
    if result_config and current_config and result_config != current_config:
        raise RuntimeError('快乐8策略已更新，较旧计算结果已丢弃，请重试')
    with _kl8_compat_cache_lock:
        previous = _CACHE['kl8'].get('data')
        if (
            previous is not None
            and _kl8_prediction_freshness(result)
            < _kl8_prediction_freshness(previous)
        ):
            raise RuntimeError('检测到更新的快乐8预测，较旧结果已丢弃')

        cache = get_shared_cache()
        if cache is not None and cache_issue:
            key = kl8_cache.cache_key(
                cache_issue,
                _kl8_cache_version(version, result_config or current_config),
            )
            # invalidate 先推进缓存纪元，避免更早启动的 SWR 线程在 set 后写回旧值。
            cache.invalidate(key)
            cache.set(key, result, ttl=kl8_cache.TTL_SECONDS)
        _CACHE['kl8']['data'] = result
        _CACHE['kl8']['timestamp'] = time.time()


def kl8_payload():
    """获取快乐8预测结果。

    走 foundation/cache：并发请求单飞，L2 是 Redis 因而**跨重启保留**。
    迁移前这份缓存纯进程内存，每次部署清零，用户在发版后的第一个请求要等
    5.6 秒（线上实测冷启动 3.5~5.6s，命中 0.01s）。

    失效条件编进 key（最新期号 + 预测器版本），新开奖或版本变更自然产生
    新 key——不必读回来再逐字段校验版本，那条路径本身就是错误来源。
    """
    def _compute():
        result = kl8_run_prediction(force_refresh=False)
        if isinstance(result, dict) and 'error' in result:
            raise RuntimeError(result['error'])
        return result

    try:
        data = kl8_cache.predict(
            compute_fn=_compute,
            latest_issue=kl8_latest_issue(),
            version=_kl8_cache_version(),
            cache=get_shared_cache())
        # 手动删号重算仍通过兼容状态取得当前正式快照的上下文。主读取路径
        # 命中 Redis/L1 时不会经过 refresh/fetch，若不在这里同步，后续重算
        # 会以空 source_snapshot_id 落盘，预测记录便无法归入对应快照。
        # 这里只复用已经取得的结果，不触发第二次预测计算。
        _sync_kl8_compat_cache(data)
        return {'result': data}
    except Exception:
        log.error('快乐8预测失败', exc_info=True)
        return {'error': '快乐8预测失败'}


def kl8_latest_issue():
    """最新一期期号。优先轻量读取历史文件，避免仅为缓存 key 初始化分析器。

    取不到就返回空字符串——调用方据此绕过缓存。历史文件异常时才回退到
    分析器，以兼容只写入 store/doc_store 的旧部署。
    """
    issue = _latest_issue_from_history_file()
    if issue:
        return issue
    try:
        analyzer = get_kl8_analyzer()
        history = getattr(analyzer, 'history_data', None) or []
        return history[0].get('issue', '') if history else ''
    except Exception as exc:
        log.warning('读取 kl8 最新期号失败，本次绕过缓存: %s', exc)
        return ''


def _latest_issue_from_history_file():
    """不创建 KL8Analyzer，直接从可重建的本地历史缓存读取最新期号。"""
    try:
        raw = json.loads(Path(data_path('kl8_history.json')).read_text(encoding='utf-8'))
        records = raw.get('results', raw.get('data', [])) if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            return ''
        issues = [
            str(record.get('issue') or '')
            for record in records
            if isinstance(record, dict) and record.get('issue')
        ]
        return max(issues) if issues else ''
    except (OSError, ValueError, TypeError):
        return ''


def _kl8_draw_map_from_history_file():
    """轻量读取已开奖期号；读取失败返回 None，让调用方走兼容回退。"""
    try:
        raw = json.loads(Path(data_path('kl8_history.json')).read_text(encoding='utf-8'))
        records = raw.get('results', raw.get('data', [])) if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            return None
        return {
            str(record.get('issue')): record.get('numbers')
            for record in records
            if isinstance(record, dict) and record.get('issue')
        }
    except (OSError, ValueError, TypeError):
        return None


def kl8_refresh_payload():
    """强制刷新快乐8数据缓存"""
    try:
        with _kl8_refresh_execution_lock:
            log.info('快乐8强制刷新请求到达')
            kl8_clear_cache()
            result = kl8_run_prediction(force_refresh=True)
            _publish_kl8_prediction(result)
            return {'success': True, 'result': result}
    except Exception as exc:
        log.error('快乐8刷新失败', exc_info=True)
        return {'error': f'快乐8刷新失败: {exc}'}


def kl8_refresh_start_payload():
    """立即返回后台任务；完整计算不再占用网关请求生命周期。"""
    return {
        'success': True,
        'result': _start_kl8_refresh_job(kl8_refresh_payload),
    }


def kl8_refresh_status_payload(params):
    """查询快乐8重新计算任务状态。"""
    job_id = (params.get('job_id') or [''])[0]
    if not job_id:
        return {'error': '缺少job_id参数'}
    job = _get_kl8_refresh_job(job_id)
    if not job:
        return {'error': f'重新计算任务不存在: {job_id}'}

    now = time.time()
    started_at = job.get('started_at') or job.get('created_at') or now
    finished_at = job.get('finished_at') or now
    job['elapsed_seconds'] = round(max(0, finished_at - started_at), 1)
    return {'success': True, 'result': job}


def kl8_fetch_payload():
    """抓取最新快乐8开奖数据"""
    try:
        with _kl8_refresh_execution_lock:
            log.info('快乐8抓取最新数据请求到达')
            from src.kl8.fetch import fetch_kl8_data, save_kl8_data

            # 日常只需最近两页；历史补数由独立调度器负责。
            data = fetch_kl8_data(pages=2, per_page=50)
            if not data:
                return {'success': False, 'message': '网络抓取失败'}

            # v2: 合并保存（不是覆盖），save_kl8_data内部会调clear_cache()
            save_ok = save_kl8_data(data)
            if not save_ok:
                return {'success': False, 'message': '数据量不足，不允许覆盖原历史'}

            # 重新预测（clear_cache已在save内部完成）。兼容缓存不能先裸清空，
            # 否则计算窗口内的手动杀号会失去来源快照；成功后由 publish 原子替换。
            result = kl8_run_prediction(force_refresh=True)
            _publish_kl8_prediction(result)

            return {
                'success': True,
                'message': f'成功抓取 {len(data)} 期数据',
                'latest_issue': data[0]['issue'] if data else '',
                'result': result,
            }
    except Exception:
        log.error('快乐8抓取失败', exc_info=True)
        return {'error': '快乐8抓取失败'}


def kl8_exclude_recalculate_payload(params):
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

        requested_context = {
            'source_snapshot_id': (params.get('source_snapshot_id') or [''])[0],
            'source_version': (params.get('source_version') or [''])[0],
            'source_target_issue': (params.get('source_target_issue') or [''])[0],
            'source_based_on_issue': (params.get('source_based_on_issue') or [''])[0],
            'source_config_fingerprint': (
                params.get('source_config_fingerprint') or ['']
            )[0],
        }
        if not all(str(value or '') for value in requested_context.values()):
            return {'error': '页面预测上下文不完整，请刷新快乐8页面后重试'}

        with _kl8_compat_cache_lock:
            current = _CACHE.get('kl8', {}).get('data') or {}

        # 进程刚重启、模型版本或策略刚升级时，兼容状态可能为空/过期；先通过
        # 正常缓存入口取得当前结果，禁止写出 source_snapshot_id 为空的孤儿记录。
        freshness = _kl8_prediction_freshness(current)
        if not current or not (freshness[1] and freshness[2]):
            current_payload = kl8_payload()
            if current_payload.get('error'):
                return {'error': '当前预测尚未就绪，请先刷新快乐8数据'}

        # 从核对页面上下文到计算落盘都持有预测锁，避免刷新/定时任务在中途
        # 更换 analyzer、历史期号或快照。kl8_payload 已在锁外完成，避免与
        # foundation/cache 的单飞锁形成反向等待。
        from src.kl8.snapshots import _prediction_run_lock

        with _prediction_run_lock:
            with _kl8_compat_cache_lock:
                current = _CACHE.get('kl8', {}).get('data') or {}
            freshness = _kl8_prediction_freshness(current)
            if not current or not (freshness[1] and freshness[2]):
                return {'error': '预测期号或版本已更新，请刷新快乐8页面后重试'}

            snapshot_file = str(current.get('snapshot_file') or '')
            source_snapshot_id = (
                Path(snapshot_file).stem.replace('snapshot_', '', 1)
                if snapshot_file else ''
            )
            statistics = current.get('statistics') or {}
            source_version = str(
                statistics.get('version') or current.get('version') or ''
            )
            current_target = str(
                statistics.get('target_issue')
                or current.get('target_issue')
                or ''
            )
            current_based_on = str(
                statistics.get('based_on_issue')
                or current.get('based_on_issue')
                or ''
            )
            latest_issue = str(kl8_latest_issue() or '')
            if not latest_issue:
                return {'error': '开奖历史暂时不可读取，请稍后重试'}
            if latest_issue != current_based_on:
                return {'error': '开奖历史已更新，请刷新快乐8页面后再进行杀号计算'}
            current_config = str(
                current.get('strategy_config_fingerprint')
                or statistics.get('strategy_config_fingerprint')
                or ''
            )
            actual_context = {
                'source_snapshot_id': source_snapshot_id,
                'source_version': source_version,
                'source_target_issue': current_target,
                'source_based_on_issue': current_based_on,
                'source_config_fingerprint': current_config,
            }
            if any(
                str(requested_context[key]) != str(actual_context[key])
                for key in requested_context
            ):
                return {'error': '页面显示的预测已经更新，请刷新后再进行杀号计算'}

            try:
                snapshots = kl8_list_snapshots()
            except Exception:
                log.warning('快乐8读取重算来源快照失败', exc_info=True)
                return {'error': '预测记录暂时不可读取，请稍后重试'}
            matching = [
                item for item in snapshots
                if str(item.get('target_issue') or '') == current_target
                and str(item.get('based_on_issue') or '') == current_based_on
                and str(item.get('version') or '') == source_version
                and str(item.get('strategy_config_fingerprint') or '')
                == current_config
            ]
            if not matching:
                return {'error': '当前预测记录不存在，请重新计算快乐8预测'}
            latest = max(matching, key=lambda item: (
                int(item.get('predicted_at_ns') or 0),
                str(item.get('predicted_at') or ''),
                str(item.get('snapshot_id') or item.get('file') or ''),
            ))
            latest_snapshot_id = str(
                latest.get('snapshot_id')
                or Path(str(latest.get('file') or '')).stem.replace(
                    'snapshot_', '', 1,
                )
            )
            if latest_snapshot_id != source_snapshot_id:
                return {'error': '预测记录已更新，请刷新快乐8页面后再进行杀号计算'}

            analyzer = get_kl8_analyzer()
            history = getattr(analyzer, 'history_data', None) or []
            analyzer_issue = str(history[0].get('issue') or '') if history else ''
            if analyzer_issue != current_based_on:
                return {'error': '开奖历史已更新，请刷新快乐8页面后再进行杀号计算'}

            current_play = current.get(play_type) or {}
            initial_numbers = (
                current_play.get('numbers')
                or current_play.get('core_numbers')
                or current_play.get('top7_numbers')
                # 兼容升级前已经落盘的 8 码缓存；新预测只会生成 top7_numbers。
                or current_play.get('top8_numbers')
                or current_play.get('top11_numbers')
                or []
            ) if isinstance(current_play, dict) else []
            record_context = {
                'source_snapshot_id': source_snapshot_id,
                'source_version': source_version,
                'generation_mode': 'manual',
                'initial_numbers': initial_numbers,
            }
            if play_type == 'fu_shi_7':
                # 手动逐轮剔除时复用自动链的选6基准，避免同一排除集算出不同号码。
                chain = current.get('fu_shi_7_recalculation_chain') or {}
                for row in chain.get('records') or []:
                    if set(row.get('excluded_numbers') or []) == set(exclude_numbers):
                        if row.get('select6_round'):
                            record_context['select6_round'] = row['select6_round']
                        break
            result = analyzer.recalculate_play_excluding(
                play_type,
                exclude_numbers,
                record_context=record_context,
            )
            return {'result': result}
    except Exception as e:
        log.error('快乐8剔除重算失败', exc_info=True)
        return {'error': f'剔除重算失败: {str(e)}'}


def kl8_snapshots_payload():
    """快乐8预测快照列表"""
    try:
        snapshots = kl8_list_snapshots()
        return {'result': {'snapshots': snapshots, 'count': len(snapshots)}}
    except Exception:
        log.error('快乐8快照列表失败', exc_info=True)
        return {'error': '快乐8快照列表失败'}


def _kl8_records_page_options(params, total):
    """解析记录分页；旧调用不传分页参数时仍返回全部，保持接口兼容。"""
    params = params or {}
    paginated = 'page' in params or 'page_size' in params
    if not paginated:
        return 1, max(total, 1), 1, False

    def _positive_int(name, default):
        raw = (params.get(name) or [str(default)])[0]
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return default

    page_size = min(
        _positive_int('page_size', _KL8_RECORDS_DEFAULT_PAGE_SIZE),
        _KL8_RECORDS_MAX_PAGE_SIZE,
    )
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(_positive_int('page', 1), total_pages)
    return page, page_size, total_pages, True


def _dedupe_kl8_snapshots(snapshots):
    """同一目标期优先显示当前代码及当前策略配置的最新预测。"""
    current_version = _current_kl8_predictor_version()
    current_config = _current_kl8_config_fingerprint()

    def _selection_key(item):
        is_current = str(item.get('version') or '') == current_version
        is_current_config = (
            str(item.get('strategy_config_fingerprint') or '') == current_config
        )
        try:
            predicted_at_ns = int(item.get('predicted_at_ns') or 0)
        except (TypeError, ValueError):
            predicted_at_ns = 0
        return (
            str(item.get('target_issue') or ''),
            is_current,
            is_current_config,
            predicted_at_ns,
            str(item.get('predicted_at') or ''),
            str(item.get('snapshot_id') or item.get('file') or ''),
        )

    ordered = sorted(
        snapshots,
        key=_selection_key,
        reverse=True,
    )
    seen = set()
    result = []
    for snapshot in ordered:
        issue = str(snapshot.get('target_issue') or '')
        if not issue or issue in seen:
            continue
        seen.add(issue)
        result.append(snapshot)
    return result


def _load_kl8_record(snapshot, snapshot_dir, settlement_dir, fushi_config,
                     clean_pick_numbers):
    """只读取一个可见页所需的完整预测和结算。"""
    predicted = {}
    main_pool = {}
    try:
        raw = json.loads(
            (snapshot_dir / snapshot['file']).read_text(encoding='utf-8')
        )
        for key, block in raw.items():
            if not (key.startswith('select_') or key.startswith('fu_shi')):
                continue
            if isinstance(block, dict):
                if block.get('main_pool'):
                    main_pool[key] = block['main_pool']
                elif block.get('numbers'):
                    predicted[key] = block['numbers']
            elif isinstance(block, list):
                predicted[key] = block

        block = raw.get('fu_shi_7')
        if not predicted.get('fu_shi_7') and isinstance(block, dict):
            predicted['fu_shi_7'] = (
                block.get('core_numbers') or block.get('top7_numbers') or []
            )
        predicted['fu_shi_7'] = clean_pick_numbers(
            predicted.get('fu_shi_7', []),
            fushi_config['fu_shi_7']['pool_size'],
        )
    except Exception:
        predicted = {}
        main_pool = {}

    record = {
        'snapshot_id': snapshot.get('snapshot_id'),
        'file': snapshot.get('file'),
        'target_issue': snapshot.get('target_issue'),
        'based_on_issue': snapshot.get('based_on_issue'),
        'predicted_at': snapshot.get('predicted_at'),
        'version': snapshot.get('version'),
        'is_experiment': snapshot.get('is_experiment', False),
        'has_settlement': snapshot.get('has_settlement', False),
        'predicted': predicted,
        'main_pool': main_pool,
        'settlement': None,
    }
    snapshot_id = snapshot.get('snapshot_id')
    if snapshot.get('has_settlement') and snapshot_id:
        settlement_path = settlement_dir / f'settlement_{snapshot_id}.json'
        if settlement_path.exists():
            try:
                record['settlement'] = json.loads(
                    settlement_path.read_text(encoding='utf-8')
                )
            except Exception:
                record['settlement'] = None
    return record


def _kl8_records_maintenance_worker(snapshots):
    """在展示请求之外补结算并校验旧奖金，避免 GET 被分析器冷启动阻塞。"""
    global _kl8_records_maintenance_running
    global _kl8_records_maintenance_last_finished
    try:
        from src.kl8 import KL8_SETTLEMENT_DIR

        settlement_dir = Path(KL8_SETTLEMENT_DIR)
        records = []
        for snapshot in snapshots:
            snapshot_id = snapshot.get('snapshot_id')
            settlement = None
            if snapshot.get('has_settlement') and snapshot_id:
                path = settlement_dir / f'settlement_{snapshot_id}.json'
                if path.exists():
                    try:
                        settlement = json.loads(path.read_text(encoding='utf-8'))
                    except Exception:
                        settlement = None
            records.append({
                'snapshot_id': snapshot_id,
                'file': snapshot.get('file'),
                'target_issue': snapshot.get('target_issue'),
                'has_settlement': bool(settlement),
                'settlement': settlement,
            })
        kl8_backfill_settlements(records)
        kl8_rebuild_stale_settlements(records)
    except Exception:
        log.warning('快乐8记录后台维护失败', exc_info=True)
    finally:
        with _kl8_records_maintenance_lock:
            _kl8_records_maintenance_running = False
            _kl8_records_maintenance_last_finished = time.monotonic()


def _schedule_kl8_records_maintenance(snapshots):
    """至多启动一个低频后台维护任务；返回当前是否仍在维护。"""
    global _kl8_records_maintenance_running
    now = time.monotonic()
    with _kl8_records_maintenance_lock:
        if _kl8_records_maintenance_running:
            return True
        if (_kl8_records_maintenance_last_finished and
                now - _kl8_records_maintenance_last_finished <
                _KL8_RECORDS_MAINTENANCE_INTERVAL):
            return False
        _kl8_records_maintenance_running = True
    threading.Thread(
        target=_kl8_records_maintenance_worker,
        args=(list(snapshots),),
        daemon=True,
        name='KL8RecordsMaintenance',
    ).start()
    return True


def kl8_records_payload(params=None):
    """快乐8预测记录；分页读取，结算修复在后台执行。"""
    try:
        from src.kl8 import (
            FUSHI_CONFIG, KL8_SETTLEMENT_DIR, KL8_SNAPSHOT_DIR,
            _clean_pick_numbers,
        )

        snapshots = _dedupe_kl8_snapshots(kl8_list_snapshots())
        total = len(snapshots)
        page, page_size, total_pages, paginated = _kl8_records_page_options(
            params, total,
        )
        if paginated:
            start = (page - 1) * page_size
            visible_snapshots = snapshots[start:start + page_size]
        else:
            visible_snapshots = snapshots

        settlement_dir = Path(KL8_SETTLEMENT_DIR)
        snapshot_dir = Path(KL8_SNAPSHOT_DIR)
        records = [
            _load_kl8_record(
                snapshot, snapshot_dir, settlement_dir,
                FUSHI_CONFIG, _clean_pick_numbers,
            )
            for snapshot in visible_snapshots
        ]

        # 删号重算必须绑定来源快照，不能把同一期不同模型版本的轨迹串在一起。
        visible_ids = {
            str(record.get('snapshot_id') or '') for record in records
        }
        recalculations_by_snapshot = {}
        for item in kl8_list_recalculations():
            source_id = str(item.get('source_snapshot_id') or '')
            if source_id and source_id in visible_ids:
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

        settled = sum(1 for snapshot in snapshots if snapshot.get('has_settlement'))
        maintenance_running = _schedule_kl8_records_maintenance(snapshots)
        return {
            'result': {
                'records': records,
                'count': total,
                'settled_count': settled,
                'pending_count': total - settled,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'has_more': page < total_pages,
                'maintenance_running': maintenance_running,
            }
        }
    except Exception as e:
        log.error('快乐8预测记录失败', exc_info=True)
        return {'error': f'获取预测记录失败: {str(e)}'}


def kl8_backfill_settlements(records):
    """对已开奖但缺少结算文件的快照当场结算（幂等回填）。

    原因：调度器仅在「发现新期号」时结算上一期（settle_previous_period）。
    若某一期因服务停机/漏检未被结算，其预测记录会永久卡在「待开奖」。
    记录接口在后台触发回填，保证历史已开奖记录最终正确展示命中/奖金，
    同时不阻塞首屏。

    仅对 target_issue 已在历史开奖数据中出现的快照结算；未开奖的（最新一期）
    保持待开奖。settle_prediction 内部按 snapshot_id 落盘且幂等（已存在则跳过）。
    """
    try:
        pending = [
            rec for rec in records
            if not rec.get('has_settlement') and not rec.get('settlement')
            and rec.get('target_issue')
        ]
        if not pending:
            return

        # 绝大多数请求只有尚未开奖的最新一期。先查轻量历史文件，确认确有
        # 已开奖但漏结算的目标期后才初始化分析器（冷启动会尝试数据库并做统计）。
        file_drawn = _kl8_draw_map_from_history_file()
        if file_drawn is not None and not any(
            str(rec.get('target_issue')) in file_drawn for rec in pending
        ):
            return

        analyzer = get_kl8_analyzer()
        history = getattr(analyzer, 'history_data', None) or []
        if not history:
            return
        drawn = {str(r.get('issue')): r.get('numbers') for r in history}
        for rec in pending:
            ti = rec.get('target_issue')
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
        log.warning('快乐8结算回填异常(已忽略): %s', e)


def kl8_rebuild_stale_settlements(records):
    """奖金表更新后，强制覆盖重算奖金不一致的历史结算。

    背景：2026-07-22 前默认奖金表有误（选5中2=5、选6中3=10 等），导致已结算
    历史记录的金额错误。后台维护任务用当前官方奖金表重新校验每条
    settlement；只要任一玩法的单注奖金与当前表不符，就以 force=True 重新结算
    并覆盖旧结算文件。
    """
    try:
        from src.kl8 import load_prize_table, SELECT_TYPES

        prize_table = load_prize_table()
        stale_records = []
        for rec in records:
            st = rec.get('settlement')
            if not st:
                continue
            ti = rec.get('target_issue')
            if not ti:
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

            if needs_rebuild:
                stale_records.append(rec)

        # 正常记录是绝大多数；没有过期奖金时不要为了“确认没事”初始化分析器。
        if not stale_records:
            return

        analyzer = get_kl8_analyzer()
        history = getattr(analyzer, 'history_data', None) or []
        if not history:
            return
        drawn = {str(r.get('issue')): r.get('numbers') for r in history}

        for rec in stale_records:
            ti = str(rec.get('target_issue') or '')
            if ti not in drawn:
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
        log.warning('快乐8历史结算重算异常(已忽略): %s', e)


def kl8_settle_payload(params):
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
        log.error('快乐8结算失败', exc_info=True)
        return {'error': f'结算失败: {str(e)}'}


def kl8_backtest_payload(params):
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
        log.error('快乐8回测失败', exc_info=True)
        return {'error': f'回测失败: {str(e)}'}


def kl8_parameter_search_payload(params):
    try:
        options_or_error = parse_kl8_parameter_search_options(params)
        if 'error' in options_or_error:
            return options_or_error
        options = options_or_error['options']
        async_str = (params.get('async') or ['false'])[0]

        if async_str.lower() in ('true', '1', 'yes', 'on'):
            return {'result': start_kl8_parameter_search_job(options)}

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
        log.error('KL8 parameter search failed', exc_info=True)
        return {'error': f'parameter search failed: {str(e)}'}


def parse_kl8_parameter_search_options(params):
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


def start_kl8_parameter_search_job(options):
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


def kl8_parameter_search_start_payload(params):
    options_or_error = parse_kl8_parameter_search_options(params)
    if 'error' in options_or_error:
        return options_or_error
    return {'result': start_kl8_parameter_search_job(options_or_error['options'])}


def kl8_parameter_search_status_payload(params):
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


def kl8_integrity_payload():
    """快乐8数据完整性检查"""
    try:
        analyzer = get_kl8_analyzer()
        if not analyzer.history_data:
            return {'error': '无历史数据'}
        integrity = kl8_check_data_integrity(analyzer.history_data)
        return {'result': integrity}
    except Exception:
        log.error('快乐8数据完整性检查失败', exc_info=True)
        return {'error': '数据完整性检查失败'}


def kl8_conflicts_payload():
    """快乐8冲突审核队列"""
    try:
        conflicts = kl8_list_conflict_queue()
        return {'result': {'conflicts': conflicts, 'count': len(conflicts)}}
    except Exception:
        log.error('快乐8冲突队列查询失败', exc_info=True)
        return {'error': '冲突队列查询失败'}


def kl8_activate_payload(params):
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
        if isinstance(result, dict) and result.get('activated'):
            # 内部预测缓存已由 activate_verified_strategy 清理；兼容状态也要
            # 清掉。跨进程缓存会因策略指纹进入新 key 而自然失效。
            with _kl8_compat_cache_lock:
                _CACHE['kl8']['data'] = None
                _CACHE['kl8']['timestamp'] = 0
        return {'result': result}
    except Exception as e:
        log.error('快乐8策略激活失败', exc_info=True)
        return {'error': f'策略激活失败: {str(e)}'}
