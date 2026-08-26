"""赔率快照的存取与周期采集。

竞彩的赔率一天里会动很多次，而任何一次请求只看得到当下的那一个值。要算出
「资金往哪边走」，就得自己把每次看到的值攒成序列——这个模块负责攒，
`movement` 模块负责从序列里读出信号。

**只在盘口真的变了时才追加快照**。没变时改写 `observed_ts`（最后一次确认
仍然没变的时刻），不动 `ts`（变化发生的时刻）。两者必须分开：若每轮采集
都追加一条，一个几小时没动过的旧信号会被伪装成刚刚发生的变化，走势的
新鲜度判断（steam / stale）会全线失真。
"""
import logging
from datetime import datetime

from src.domain.sports.basketball.repository import OddsSnapshotRepository

log = logging.getLogger('domain.basketball.odds_history')

# 单场保留的快照上限。一个赛季不设限会无限膨胀，而走势只看首尾与最后一次
# 变化，留太久的历史没有额外价值。
HISTORY_CAP = 240

SNAPSHOT_FIELDS = ('spf_home', 'spf_away', 'rqspf_home', 'rqspf_away',
                   'dx_over', 'dx_under', 'handicap', 'total_line')
# 让分盘存字符串：线上是 '+9.5' 这种带符号写法，转成数值会丢掉符号语义。
# 但上游偶尔会给来数值，落库后读回就变成了字符串——两次采集因此永远判定
# 「变了」，每轮都追加一条新快照，走势的新鲜度随之失真。所以在**构造快照
# 时**就统一成字符串，让「写进去的」和「读出来的」是同一个东西。
_TEXT_FIELDS = ('handicap',)


class OddsHistoryStore:
    """把 {match_key: [快照]} 存进 foundation/store。

    整体替换语义：save 用传入内容取代库中已有的。历史会被截断（HISTORY_CAP），
    只做 upsert 的话被截掉的旧快照会留在库里，读出来的序列比实际更长。
    """

    def __init__(self, db):
        self.db = db
        self._repo = OddsSnapshotRepository(db)

    def load(self):
        history = {}
        for row in self._repo.find_all(order_by=['match_key', 'seq']):
            history.setdefault(row['match_key'], []).append(_row_to_snapshot(row))
        return history

    def history_for(self, match_key):
        return self.load().get(match_key, [])

    def save(self, history):
        self._repo.delete_all()
        rows = [_snapshot_to_row(match_key, seq, snapshot)
                for match_key, snapshots in (history or {}).items()
                for seq, snapshot in enumerate(snapshots or [])]
        self._repo.insert_many(rows)


def _row_to_snapshot(row):
    """把行还原成快照。

    八个盘口字段**一律带上，哪怕值是 None**：线上 171 条快照全都八键齐全，
    只是部分取值为空（未开售的玩法）。按「有值才写键」还原会悄悄改变数据
    形状，比丢一个字段更难察觉。observed_ts 相反，它本就是可选的
    （171 条里 52 条没有），有才写。
    """
    snapshot = {'ts': row['captured_at']}
    if row.get('observed_ts'):
        snapshot['observed_ts'] = row['observed_ts']
    snapshot.update({field: row.get(field) for field in SNAPSHOT_FIELDS})
    return snapshot


def _snapshot_to_row(match_key, seq, snapshot):
    row = {
        'match_key': match_key,
        'seq': seq,
        'captured_at': snapshot.get('ts') or '',
        'observed_ts': snapshot.get('observed_ts'),
    }
    for field in SNAPSHOT_FIELDS:
        row[field] = _normalize(field, snapshot.get(field))
    return row


class OddsTracker:
    """采集一轮赔率快照。

    schedule_fetcher 注入而非内建：采集与「从哪儿取赛程」是两件事，
    500 源和澳客源都能喂给它。
    """

    def __init__(self, schedule_fetcher, store, now_fn=None, cap=HISTORY_CAP):
        self._fetch = schedule_fetcher
        self._store = store
        self._now = now_fn or datetime.now
        self._cap = cap

    def track(self, date=None):
        """采集并落盘，返回本轮记录到的场次数。

        抓取失败返回 0 而不是抛错：采集是旁路，它挂掉不该影响任何请求。
        """
        try:
            matches = self._fetch(date)
        except Exception as exc:
            log.warning('篮球赔率追踪抓取失败: %s', exc)
            return 0
        if not matches:
            return 0

        now_iso = self._now().isoformat()
        history = self._store.load()
        count = sum(self._record(history, match, now_iso) for match in matches)

        self._store.save(history)
        log.info('篮球赔率追踪完成: %d 场', count)
        return count

    def _record(self, history, match, now_iso):
        match_key = match.get('id')
        if not match_key:
            return 0

        snapshot = {'ts': now_iso,
                    **{field: _normalize(field, match.get(field))
                       for field in SNAPSHOT_FIELDS}}
        if not any(snapshot[field] is not None for field in SNAPSHOT_FIELDS):
            return 0

        sequence = history.setdefault(match_key, [])
        if sequence and _same_odds(sequence[-1], snapshot):
            sequence[-1]['observed_ts'] = now_iso
        else:
            sequence.append(snapshot)
            if len(sequence) > self._cap:
                sequence[:] = sequence[-self._cap:]
        return 1


def _normalize(field, value):
    """按落库后的类型规范化，保证「写进去的」与「读出来的」可比。"""
    if value is None or field not in _TEXT_FIELDS:
        return value
    return str(value)


def _same_odds(previous, current):
    """只比盘口字段，不比时间——时间每轮都不同，带上它就永远判定为「变了」。"""
    return all(previous.get(field) == current.get(field)
               for field in SNAPSHOT_FIELDS)
