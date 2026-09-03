# -*- coding: utf-8 -*-
"""北单赛果提取、历史校准、预测记录与自动赛果回填。"""

from datetime import datetime, timedelta
import os
import re
import threading

from ..common.logger import setup_logger
from ..common import kv_store

log = setup_logger('beidan')

from .config import (
    BEIDAN_HISTORY_KEY, BEIDAN_HISTORY_LIMIT, MAX_GOALS,
)

# ─── 领域层适配 ───
#
# 赛果判定与校准算术在 `src/domain/sports/beidan/settlement.py`。
# 这里只做三件领域层不该碰的事：读写 kv、解析页面来的盘口文本、
# 按玩法名挑一个提取器。旧名字全部保住——`__init__.py` 导出了它们，
# `recommending` 按名字导入。

from .fetching import fetch_zgzcw_finished_results
from src.domain.sports.beidan import handicap as _handicap
from src.domain.sports.beidan import settlement as _settlement

# 历史校准的默认门槛。**迁移前它们是函数签名里的裸默认值**：
# 至少 8 个加权样本才动手（少于这个数，因子基本由先验决定，动了也是噪声），
# 只看最近 200 条（更早的记录来自改过的模型，拿它们校准当前模型是刻舟求剑）。
BEIDAN_CALIBRATION_MIN_SAMPLES = 8
BEIDAN_CALIBRATION_LIMIT = 200
BEIDAN_SYNC_INTERVAL_SECONDS = max(
    60, int(os.getenv('BEIDAN_SYNC_INTERVAL_SECONDS', '600')),
)
BEIDAN_SYNC_BATCH_SIZE = max(
    1, int(os.getenv('BEIDAN_SYNC_BATCH_SIZE', '100')),
)

# 周期任务和手动补数可能同时触发。同步过程是「整份历史读出 -> 修改 -> 整份写回」，
# 两轮并发会互相覆盖，所以进程内只允许一轮实际抓取。手动接口另有一个很轻的
# 状态锁，用于立即返回任务进度，不让几百场补数占住 HTTP 请求直到网关超时。
_BEIDAN_SYNC_RUN_LOCK = threading.Lock()
_BEIDAN_SYNC_JOB_LOCK = threading.Lock()
_BEIDAN_HISTORY_UPDATE_LOCK = threading.RLock()
_BEIDAN_SYNC_JOB = {
    'running': False,
    'started_at': None,
    'finished_at': None,
    'processed': 0,
    'total': 0,
    'synced': 0,
    'failed': 0,
    'remaining': 0,
    'message': '',
}

_BEIDAN_SYNC_STATE_FIELDS = (
    'settled', 'sync_status', 'sync_attempts', 'last_sync_at',
    'last_sync_error', 'next_sync_at', 'settled_at', 'actual_score',
    'actual_spf', 'actual_rqspf', 'actual_zjq', 'actual', 'settlement',
    'hit_spf', 'hit_rqspf', 'hit_zjq', 'result_source',
)


def calculate_implied_probability(odds_dict):
    """欧赔 → 去水后的隐含概率。"""
    return _settlement.implied_probability(odds_dict)


def _actual_spf_from_record(record):
    return _settlement.actual_spf(record)


def _actual_zjq_from_record(record):
    return _settlement.actual_zjq(record, max_goals=MAX_GOALS)


def _actual_bifen_from_record(record):
    """从已结算快照解析实际比分字符串 'h-a'（用于比分历史校准）"""
    return _settlement.actual_bifen(record)


def _actual_rqspf_from_record(record):
    """从已结算快照的实际比分 + 让球值，推导让球胜平负实际结果。

    **盘口在历史记录里存的是页面文本**：线上 500 条里 143 条有值，形如
    `'(-1)'`、`'(+1)'`、`'(-2)'`、`'(+2)'`，其余为 `None`。迁移前这里写的是
    `float(hc)`，对这四种形态**无一例外抛 ValueError**，被 except 吞成
    「按平手盘算」——分盘全部退化成不让球，`让平` 因此被系统性地误判成
    让胜或让负。同一个包里另外两处（`markets.py`、`recommending.py`）
    读的是同一个字段，用的都是 `parse_beidan_handicap`。

    改成同一个解析器。**这不改变任何线上行为**：这个函数的两个调用方
    （历史校准、`summarize_beidan_history`）都以 `settled` 为前提，
    而线上 500 条 `settled` 全是 False，仓库里也没有任何一处会把它置 True。
    """
    return _settlement.actual_rqspf(record, _handicap.parse(record.get('handicap')))


# 玩法 → 实际结果提取器。迁移前是函数体里一串 if/elif，未知玩法落到
# `actual = None`，于是一条样本都攒不上——保住这个语义：查不到就给个
# 恒返回 None 的提取器，而不是抛错。
_ACTUAL_EXTRACTORS = {
    'spf': _actual_spf_from_record,
    'zjq': _actual_zjq_from_record,
    'bifen': _actual_bifen_from_record,
    'rqspf': _actual_rqspf_from_record,
}


def apply_beidan_history_calibration(probabilities, bet_type, league=None,
                                     min_samples=BEIDAN_CALIBRATION_MIN_SAMPLES,
                                     limit=BEIDAN_CALIBRATION_LIMIT):
    """Use settled Beidan snapshots as a conservative reliability correction."""
    # 概率为空时不读存储：迁移前那道守卫排在 kv 读取之前。比分那一路
    # （`recommending.py:852`）传进来的矩阵确实可能是空的，顺序保住。
    records = _load_beidan_history() if probabilities else []
    return _settlement.apply_history_calibration(
        probabilities, records,
        _ACTUAL_EXTRACTORS.get(bet_type, lambda record: None), bet_type,
        league=league, min_samples=min_samples, limit=limit)


def _beidan_record_key(match):
    return _settlement.record_key(match)


def _load_beidan_history():
    data = kv_store.load(BEIDAN_HISTORY_KEY, [])
    return data if isinstance(data, list) else []


def _save_beidan_history(records):
    records = sorted(records, key=lambda r: r.get('created_at', ''), reverse=True)
    return kv_store.save(BEIDAN_HISTORY_KEY, records[:BEIDAN_HISTORY_LIMIT])


def _save_beidan_sync_updates(original_records, updated_records):
    """把同步字段合并进最新历史，避免长任务覆盖期间新增的预测。"""
    with _BEIDAN_HISTORY_UPDATE_LOCK:
        latest = _load_beidan_history() or original_records
        by_key = {record.get('key'): record for record in latest if record.get('key')}
        for updated in updated_records:
            key = updated.get('key')
            current = by_key.get(key)
            if current is None:
                latest.append(updated)
                if key:
                    by_key[key] = updated
                continue
            for field in _BEIDAN_SYNC_STATE_FIELDS:
                if field in updated:
                    current[field] = updated[field]
        return _save_beidan_history(latest)


def _record_match_time(record):
    """把北单分开的 date/time 字段转换为竞彩足球同步器可用的时间。"""
    date = str(record.get('date') or '').strip()
    clock = str(record.get('time') or '').strip()
    clock_match = re.search(r'(\d{1,2}:\d{2})', clock)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) or not clock_match:
        return ''
    hour, minute = clock_match.group(1).split(':')
    return f'{date} {int(hour):02d}:{minute}'


def _ready_for_result_sync(record, now=None, wait_minutes=180,
                           force_retry=False):
    if record.get('settled') or record.get('sync_status') in ('synced', 'ignored'):
        return False
    if record.get('sync_status') == 'failed' and not force_retry:
        return False
    now = now or datetime.now()
    match_time = _record_match_time(record)
    if not match_time:
        return False
    try:
        kickoff = datetime.strptime(match_time, '%Y-%m-%d %H:%M')
    except ValueError:
        return False
    if now < kickoff + timedelta(minutes=wait_minutes):
        return False
    next_sync_at = None if force_retry else record.get('next_sync_at')
    if next_sync_at:
        try:
            if now < datetime.fromisoformat(str(next_sync_at)):
                return False
        except ValueError:
            pass
    return True


def _prediction_hit(section, actual):
    prediction = (section or {}).get('prediction')
    if prediction in (None, '') or actual in (None, ''):
        return None
    return str(prediction) == str(actual)


def _settle_beidan_record(record, score, source=None, now=None):
    """以最终比分结算北单已保存的三个玩法。"""
    now = now or datetime.now()
    score = str(score or '').strip().replace(':', '-')
    if not re.fullmatch(r'\d{1,2}-\d{1,2}', score):
        return False
    record['actual_score'] = score
    record['actual_spf'] = _actual_spf_from_record({'actual_score': score})
    record['actual_rqspf'] = _actual_rqspf_from_record({
        'actual_score': score, 'handicap': record.get('handicap'),
    })
    record['actual_zjq'] = _actual_zjq_from_record({'actual_score': score})
    record['actual'] = {
        'score': score,
        'spf': record['actual_spf'],
        'rqspf': record['actual_rqspf'],
        'zjq': record['actual_zjq'],
    }
    record['settlement'] = dict(record['actual'])
    record['hit_spf'] = _prediction_hit(record.get('spf'), record['actual_spf'])
    record['hit_rqspf'] = _prediction_hit(record.get('rqspf'), record['actual_rqspf'])
    record['hit_zjq'] = _prediction_hit(record.get('zjq'), record['actual_zjq'])
    record['settled'] = True
    record['sync_status'] = 'synced'
    record['sync_attempts'] = int(record.get('sync_attempts') or 0)
    record['last_sync_error'] = None
    record['next_sync_at'] = None
    record['result_source'] = source
    record['last_sync_at'] = now.isoformat(timespec='seconds')
    record['settled_at'] = record['last_sync_at']
    return True


def _mark_beidan_sync_failure(record, error, now=None):
    now = now or datetime.now()
    attempts = int(record.get('sync_attempts') or 0) + 1
    retry_hours = {1: 2, 2: 6, 3: 24, 4: 48}
    record['sync_attempts'] = attempts
    record['last_sync_at'] = now.isoformat(timespec='seconds')
    record['last_sync_error'] = str(error or '未找到赛果')
    if attempts >= 5:
        record['sync_status'] = 'failed'
        record['next_sync_at'] = None
    else:
        record['sync_status'] = 'retry'
        record['next_sync_at'] = (
            now + timedelta(hours=retry_hours.get(attempts, 24))
        ).isoformat(timespec='seconds')


def _sync_result_for_record(record, match_time, fetch_by_id, fetch_by_team,
                            finished_map=None):
    """按记录来源选择有效的赛果查询顺序。

    500.com 的 dc/jczq 赛程 ID 可以直接查 shuju 页面；okooo 与 zgzcw 的数字
    是各自站内 ID，拿它拼 500.com URL 只会制造一场一次的无效请求。

    中国足彩网单场页的完场比分排在最前：北单赛程本来就抓自那张页，按同一个
    ID 对齐没有队名歧义，也不用赌 500.com 收不收录这场。
    """
    hit = (finished_map or {}).get(str(record.get('match_id') or ''))
    if hit and hit.get('score'):
        return {'score': hit['score'], 'source': 'zgzcw_finished'}

    source = str(record.get('source') or '').lower()
    if source in ('okooo', 'zgzcw'):
        return fetch_by_team(
            record.get('home') or '', record.get('away') or '', match_time,
        )

    result = None
    match_id = str(record.get('match_id') or '')
    if match_id.isdigit():
        result = fetch_by_id(match_id, match_time)
    if not result:
        result = fetch_by_team(
            record.get('home') or '', record.get('away') or '', match_time,
        )
    return result


def _sync_busy_result():
    return {
        'synced': 0,
        'failed': 0,
        'total': 0,
        'remaining': 0,
        'busy': True,
        'message': '已有一轮北单赛果同步正在执行',
    }


def sync_beidan_results(fetch_by_id=None, fetch_by_team=None, now=None,
                        batch_size=BEIDAN_SYNC_BATCH_SIZE, force_retry=False,
                        progress=None, wait_for_lock=False):
    """回填已结束北单比赛；复用竞彩足球的双来源赛果抓取。

    ``batch_size=None`` 表示处理当前全部可同步记录，供手动后台追赶使用；
    周期任务仍使用受控批次，避免一次长期占用调度器 worker。
    """
    if not _BEIDAN_SYNC_RUN_LOCK.acquire(blocking=wait_for_lock):
        return _sync_busy_result()
    try:
        return _sync_beidan_results_locked(
            fetch_by_id=fetch_by_id,
            fetch_by_team=fetch_by_team,
            now=now,
            batch_size=batch_size,
            force_retry=force_retry,
            progress=progress,
        )
    finally:
        _BEIDAN_SYNC_RUN_LOCK.release()


def _sync_beidan_results_locked(fetch_by_id=None, fetch_by_team=None, now=None,
                                batch_size=BEIDAN_SYNC_BATCH_SIZE,
                                force_retry=False, progress=None):
    now = now or datetime.now()
    if fetch_by_id is None or fetch_by_team is None:
        from src.football.result_sync import (
            fetch_result_by_match_id, fetch_result_by_team_and_date,
        )
        fetch_by_id = fetch_by_id or fetch_result_by_match_id
        fetch_by_team = fetch_by_team or fetch_result_by_team_and_date

    finished_map = {}
    try:
        finished_map = fetch_zgzcw_finished_results() or {}
    except Exception as exc:
        # 源站抖动只该让这一轮少一条捷径，不该把整轮回填带崩。
        log.warning('中国足彩网完场比分不可用，本轮回退 500.com: %s', exc)

    records = _load_beidan_history()
    ready = [
        r for r in records
        if _ready_for_result_sync(r, now=now, force_retry=force_retry)
    ]
    # 新完赛且尚未尝试的记录优先，避免上线时几百条旧历史把当天比赛堵在队尾。
    ready.sort(key=lambda r: _record_match_time(r), reverse=True)
    ready.sort(key=lambda r: int(r.get('sync_attempts') or 0))
    if batch_size is not None:
        ready = ready[:max(1, int(batch_size))]
    synced = failed = 0
    for processed, record in enumerate(ready, start=1):
        match_time = _record_match_time(record)
        result = None
        try:
            result = _sync_result_for_record(
                record, match_time, fetch_by_id, fetch_by_team, finished_map,
            )
            if result and _settle_beidan_record(
                    record, result.get('score'), result.get('source'), now=now):
                synced += 1
            else:
                _mark_beidan_sync_failure(record, '未找到赛果', now=now)
                failed += 1
        except Exception as exc:
            _mark_beidan_sync_failure(record, exc, now=now)
            failed += 1
        if progress:
            progress({
                'processed': processed,
                'total': len(ready),
                'synced': synced,
                'failed': failed,
            })
    if ready:
        _save_beidan_sync_updates(records, ready)
    return {
        'synced': synced,
        'failed': failed,
        'total': len(ready),
        'remaining': sum(
            1 for r in records
            if _ready_for_result_sync(r, now=now, force_retry=force_retry)
        ),
        'message': f'北单赛果回填完成：成功 {synced} 场，失败 {failed} 场',
    }


def _set_beidan_sync_job(**values):
    with _BEIDAN_SYNC_JOB_LOCK:
        _BEIDAN_SYNC_JOB.update(values)


def get_beidan_sync_job_status():
    with _BEIDAN_SYNC_JOB_LOCK:
        return dict(_BEIDAN_SYNC_JOB)


def start_beidan_sync_job():
    """启动一轮不受批次/退避限制的手动追赶，并立即返回。"""
    with _BEIDAN_SYNC_JOB_LOCK:
        if _BEIDAN_SYNC_JOB['running']:
            return {
                **_BEIDAN_SYNC_JOB,
                'started': False,
                'running': True,
                'message': '已有一轮北单赛果同步正在执行',
            }
        started_at = datetime.now().isoformat(timespec='seconds')
        _BEIDAN_SYNC_JOB.update({
            'running': True,
            'started_at': started_at,
            'finished_at': None,
            'processed': 0,
            'total': 0,
            'synced': 0,
            'failed': 0,
            'remaining': 0,
            'message': '北单全量同步任务已启动',
        })

    def _progress(values):
        _set_beidan_sync_job(**values)

    def _run():
        try:
            result = sync_beidan_results(
                batch_size=None, force_retry=True, progress=_progress,
                wait_for_lock=True,
            )
            _set_beidan_sync_job(
                running=False,
                finished_at=datetime.now().isoformat(timespec='seconds'),
                **{key: result.get(key, 0) for key in (
                    'synced', 'failed', 'total', 'remaining',
                )},
                processed=result.get('total', 0),
                message=result.get('message') or '北单全量同步完成',
            )
        except Exception as exc:
            log.error('北单手动全量同步失败', exc_info=True)
            _set_beidan_sync_job(
                running=False,
                finished_at=datetime.now().isoformat(timespec='seconds'),
                message=f'北单全量同步失败：{exc}',
            )

    threading.Thread(
        target=_run, daemon=True, name='BeidanManualResultSync',
    ).start()
    return {**get_beidan_sync_job_status(), 'started': True}


def get_beidan_sync_status_summary(records=None):
    records = _load_beidan_history() if records is None else records
    counts = {'pending': 0, 'ready': 0, 'synced': 0, 'retry': 0,
              'failed': 0, 'ignored': 0}
    last_settled = None
    for record in records:
        status = 'synced' if record.get('settled') else record.get('sync_status', 'pending')
        if status not in counts:
            status = 'pending'
        counts[status] += 1
        if status == 'synced':
            stamp = record.get('settled_at') or record.get('last_sync_at')
            if stamp and (last_settled is None or str(stamp) > str(last_settled)):
                last_settled = stamp
    return {
        'total': len(records),
        'settled': counts['synced'],
        'pending_sync': counts['pending'] + counts['ready'],
        'retry': counts['retry'],
        'failed': counts['failed'],
        'ignored': counts['ignored'],
        'last_settled_at': last_settled,
    }


def get_beidan_prediction_records():
    """返回记录页所需字段；不暴露体积很大的盘口分层快照。"""
    records = _load_beidan_history()
    result = []
    for source in records:
        record = dict(source)
        record.pop('market_layers', None)
        record['match_num'] = record.get('num')
        record['match_time'] = _record_match_time(record)
        record['sync_status'] = (
            'synced' if record.get('settled') else record.get('sync_status', 'pending')
        )
        result.append(record)
    result.sort(key=lambda r: (
        r.get('match_time') or r.get('date') or '', r.get('num') or '',
    ), reverse=True)
    return result


def register_beidan_tasks(submit, interval_seconds=BEIDAN_SYNC_INTERVAL_SECONDS):
    registered = submit('beidan_result_sync', sync_beidan_results, interval_seconds)
    if registered:
        log.info('北单赛果回填任务已登记: 每 %s 秒', interval_seconds)
        return ['beidan_result_sync']
    return []
