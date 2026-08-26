"""预测记录：存下每次给出的推荐，赛后回填比分，再反哺 Elo 与校准器。

这是模型能自我修正的唯一通路——没有它，校准器永远没有样本，Elo 永远停在
初始分。三条不变量：

- **刷新推荐不能抹掉已结算的结果**。数据源有时只返回半份赛程，若按当天
  重写整段记录，已经打完并结算的场次会连同赛果一起消失。
- **结算幂等**。同一场比赛重复结算不得重复更新 Elo、不得重复喂校准器，
  否则一场比赛会被算成好几场，评分与校准双双失真。
- **走盘不计入准确率**。让分正好打平、总分正好等于盘口，本就没有输赢，
  计入任何一侧都是在污染统计。
"""
import json
import logging
from datetime import datetime

from src.domain.sports.basketball.repository import PredictionRecordRepository

log = logging.getLogger('domain.basketball.records')

# 保留的记录条数上限。它同时限制了统计的回看窗口——线上一天几场，
# 500 条约合小半年。
MAX_RECORDS = 500

BET_TYPES = ('spf', 'rqspf', 'dx')
_JSON_FIELDS = BET_TYPES + ('result',)
_PLAIN_FIELDS = ('date', 'match_id', 'num', 'league', 'home', 'away',
                 'time', 'version', 'created_at')

# 每个玩法落库时保留的字段。刻意是白名单而不是整份透传：分析结果里还带着
# 一大堆中间量（各家赔率明细、快照序列），存下来既没人看又会把记录撑大。
_SPF_FIELDS = ('recommendation', 'pick_prob', 'skip_reason', 'home_prob',
               'away_prob', 'confidence', 'elo_home_prob', 'elo_trust',
               'market_home_prob')
_RQSPF_FIELDS = ('recommendation', 'pick_prob', 'skip_reason', 'handicap',
                 'home_prob', 'away_prob', 'confidence', 'elo_margin',
                 'elo_trust', 'market_home_prob', 'line_movement',
                 'water_inference')
_DX_FIELDS = ('recommendation', 'pick_prob', 'skip_reason', 'total_line',
              'over_prob', 'under_prob', 'confidence', 'elo_total', 'elo_trust',
              'market_over_prob', 'line_movement', 'water_inference')
_MOVEMENT_FLAGS = ('movement_led', 'sharp_confirmed')


class PredictionRecordStore:
    """{有序记录列表} 的存取。整体替换语义，与本领域其它仓储一致。"""

    def __init__(self, db):
        self.db = db
        self._repo = PredictionRecordRepository(db)

    def load(self):
        return [_row_to_record(row)
                for row in self._repo.find_all(order_by=['seq'])]

    def save(self, records):
        self._repo.delete_all()
        self._repo.insert_many(
            _record_to_row(seq, record) for seq, record in enumerate(records))


def _row_to_record(row):
    record = {field: row.get(field) for field in _PLAIN_FIELDS}
    for field in _JSON_FIELDS:
        record[field] = json.loads(row[field]) if row.get(field) else None
    return record


def _record_to_row(seq, record):
    row = {'seq': seq}
    row.update({field: record.get(field) for field in _PLAIN_FIELDS})
    for field in _JSON_FIELDS:
        value = record.get(field)
        row[field] = json.dumps(value, ensure_ascii=False) if value else None
    return row


def build_record(date, match, analysis, version, created_at):
    """把一场比赛的分析结果压成一条记录。"""
    return {
        'date': date,
        'match_id': match.get('id', ''),
        'num': match.get('num', ''),
        'league': match.get('league', ''),
        'home': match.get('home', ''),
        'away': match.get('away', ''),
        'time': match.get('time', ''),
        'version': version,
        'created_at': created_at,
        'spf': _bet_snapshot(analysis.get('spf'), _SPF_FIELDS, movement=False),
        'rqspf': _bet_snapshot(analysis.get('rqspf'), _RQSPF_FIELDS),
        'dx': _bet_snapshot(analysis.get('dx'), _DX_FIELDS),
        'result': None,
    }


def _bet_snapshot(bet, fields, movement=True):
    if not bet:
        return None
    snapshot = {
        'available': bet.get('available', False),
        'playable': bet.get('playable', True),
        # official 缺省时跟随 playable：早期版本没有这个字段，
        # 那时「可玩」就等于「计入准确率」。
        'official': bet.get('official', bet.get('playable', True)),
    }
    snapshot.update({field: bet.get(field) for field in fields})
    if movement:
        snapshot.update({flag: bet.get(flag, False) for flag in _MOVEMENT_FLAGS})
    return snapshot


class PredictionRecorder:
    """`PredictionService` 的 recorder 端口实现。"""

    def __init__(self, store, calibrator=None, elo=None, now_fn=None,
                 max_records=MAX_RECORDS):
        self._store = store
        self._calibrator = calibrator
        self._elo = elo
        self._now = now_fn or datetime.now
        self._max_records = max_records

    # ---------- 写入 ----------

    def save(self, date, results, version=''):
        """按 (date, match_id) 覆盖当天的记录，已结算的赛果保留。

        数据源有时只返回半份赛程，若按当天整段重写，已经打完并结算的场次
        会连同赛果一起消失——那是不可恢复的丢失，比赛已经结束了。
        """
        records = self._store.load()
        index = {(r.get('date'), r.get('match_id')): i
                 for i, r in enumerate(records) if r.get('match_id')}
        created_at = self._now().isoformat()

        for item in results:
            record = build_record(date, item.get('match', {}), item, version,
                                  created_at)
            position = index.get((date, record['match_id']))
            if position is None:
                index[(date, record['match_id'])] = len(records)
                records.append(record)
            else:
                record['result'] = records[position].get('result')
                records[position] = record

        self._store.save(records[-self._max_records:])
        log.info('篮球预测记录已保存: %s, %d 场', date, len(results))

    # ---------- 读取 ----------

    def get(self, date=None, limit=50):
        records = self._store.load()
        if date:
            records = [r for r in records if r.get('date') == date]
        return records[-limit:]

    def unsettled(self):
        return [r for r in self._store.load() if r.get('result') is None]

    def stats(self):
        return summarize(self._store.load())

    # ---------- 结算 ----------

    def settle(self, match_id, home_score, away_score, league='',
               status='finished'):
        """回填赛果、更新 Elo、喂校准器。对同一场比赛幂等。

        幂等靠 result 里的 elo_updated / calibration_fed 两个标记：重复结算
        会重复推高评分、重复投喂校准样本，等于把一场比赛算成好几场。
        """
        records = self._store.load()
        target = _find_by_match_id(records, match_id)
        if target is None:
            return {'ok': False, 'error': 'match_not_found', 'match_id': match_id}

        previous = target.get('result') if isinstance(target.get('result'), dict) else {}
        result = {
            'home_score': int(home_score),
            'away_score': int(away_score),
            'status': status,
            'settled_at': self._now().isoformat(),
            'elo_updated': bool(previous.get('elo_updated')),
            'calibration_fed': bool(previous.get('calibration_fed')),
            **evaluate_markets(target, int(home_score), int(away_score)),
        }

        if not result['elo_updated']:
            result['elo_updated'] = self._update_elo(
                target, home_score, away_score, league)

        target['result'] = result
        fed = 0
        if not result['calibration_fed']:
            fed = self._feed_one(target)
            result['calibration_fed'] = True
            target['result'] = result

        self._store.save(records)
        return {'ok': True, 'match_id': match_id, 'result': result,
                'calibration_samples': fed}

    def feed_calibration(self):
        """把已结算但没喂过校准器的记录补上。幂等。"""
        records = self._store.load()
        fed = 0
        dirty = False
        for record in records:
            result = record.get('result')
            if not result or result.get('calibration_fed'):
                continue
            fed += self._feed_one(record)
            result['calibration_fed'] = True
            record['result'] = result
            dirty = True

        if dirty:
            self._store.save(records)
        log.info('校准反馈完成: %d 条样本', fed)
        return fed

    def _update_elo(self, record, home_score, away_score, league):
        if self._elo is None:
            return False
        try:
            self._elo.update_ratings(
                record.get('home', ''), record.get('away', ''),
                int(home_score), int(away_score),
                league or record.get('league', '') or 'NBA')
            return True
        except Exception as exc:
            log.warning('结算 ELO 更新失败: %s', exc)
            return False

    def _feed_one(self, record):
        """把一条已结算记录喂给校准器。走盘与未评估的玩法一律跳过。"""
        if self._calibrator is None:
            return 0

        result = record.get('result') or {}
        hits = evaluate_markets(record, result.get('home_score', 0),
                                result.get('away_score', 0))
        league = record.get('league', '')
        fed = 0
        for bet_type in BET_TYPES:
            bet = record.get(bet_type) or {}
            if not _counts_towards_accuracy(bet, hits, bet_type):
                continue
            self._calibrator.record(
                bet_type, _picked_probability(bet, bet_type),
                bool(hits[f'{bet_type}_hit']), league,
                bet.get('confidence', 'medium'))
            fed += 1

        if fed:
            self._calibrator.save()
        return fed


def _find_by_match_id(records, match_id):
    for record in records:
        if record.get('match_id') == match_id:
            return record
    return None


_PICK_PROBABILITY_KEYS = {
    'spf': ('主胜', 'home_prob', 'away_prob'),
    'rqspf': ('让胜', 'home_prob', 'away_prob'),
    'dx': ('大分', 'over_prob', 'under_prob'),
}


def _picked_probability(bet, bet_type):
    """取被推荐那一侧的概率——校准要问的是「我说 62% 的时候，对了几成」。"""
    first_name, first_key, second_key = _PICK_PROBABILITY_KEYS[bet_type]
    key = first_key if bet.get('recommendation') == first_name else second_key
    return bet.get(key, 0.5)


def _counts_towards_accuracy(bet, hits, bet_type):
    return bool(bet.get('available') and bet.get('playable', True)
                and not hits[f'{bet_type}_void']
                and hits[f'{bet_type}_hit'] is not None)


def evaluate_markets(record, home_score, away_score):
    """逐玩法判定命中。走盘/平局记为 void——没有输赢，不该计入任何一侧。"""
    hits = {f'{bet_type}_hit': None for bet_type in BET_TYPES}
    hits.update({f'{bet_type}_void': False for bet_type in BET_TYPES})

    for bet_type, judge in (('spf', _judge_spf), ('rqspf', _judge_rqspf),
                            ('dx', _judge_total)):
        bet = record.get(bet_type) or {}
        if not (bet.get('available') and bet.get('playable', True)):
            continue
        actual = judge(bet, home_score, away_score)
        if actual is None:
            hits[f'{bet_type}_void'] = True
        else:
            hits[f'{bet_type}_hit'] = bet.get('recommendation') == actual
    return hits


def _judge_spf(bet, home_score, away_score):
    if home_score == away_score:
        return None
    return '主胜' if home_score > away_score else '客胜'


def _judge_rqspf(bet, home_score, away_score):
    """让分是加在主队得分上的有符号值，加完再比。"""
    adjusted = (home_score + _as_float(bet.get('handicap'))) - away_score
    if abs(adjusted) < 1e-9:
        return None
    return '让胜' if adjusted > 0 else '让负'


def _judge_total(bet, home_score, away_score):
    total = home_score + away_score
    line = _as_float(bet.get('total_line'))
    if abs(total - line) < 1e-9:
        return None
    return '大分' if total > line else '小分'


def _as_float(value, default=0.0):
    try:
        return default if value in (None, '') else float(value)
    except (TypeError, ValueError):
        return default


def summarize(records):
    """按玩法统计准确率。只看已结算、非走盘的记录。"""
    settled = [r for r in records if r.get('result')]
    stats = {
        'total_predictions': len(records),
        'settled_count': len(settled),
        'official_predictions': 0,
        **{bet_type: {'total': 0, 'correct': 0, 'void': 0, 'accuracy': 0.0}
           for bet_type in BET_TYPES},
        'water_inference': {
            bet_type: {'total': 0, 'correct': 0, 'accuracy': 0.0}
            for bet_type in ('rqspf', 'dx')},
    }

    for record in settled:
        result = record['result']
        hits = evaluate_markets(record, result.get('home_score', 0),
                                result.get('away_score', 0))
        for bet_type in BET_TYPES:
            _accumulate(stats, record, hits, bet_type)

    for bet_type in BET_TYPES:
        _finish_accuracy(stats[bet_type])
    for bet_type in ('rqspf', 'dx'):
        _finish_accuracy(stats['water_inference'][bet_type])
    return stats


def _accumulate(stats, record, hits, bet_type):
    bet = record.get(bet_type) or {}
    if not (bet.get('available') and bet.get('playable', True)):
        return

    stats['official_predictions'] += 1
    if hits[f'{bet_type}_void']:
        stats[bet_type]['void'] += 1
        return

    stats[bet_type]['total'] += 1
    hit = bool(hits[f'{bet_type}_hit'])
    stats[bet_type]['correct'] += int(hit)

    # 走势反推翻转过模型方向的那些单独统计——这套信号到底有没有用，
    # 只能靠它自己的命中率回答。
    if bet_type in stats['water_inference'] and bet.get('movement_led'):
        stats['water_inference'][bet_type]['total'] += 1
        stats['water_inference'][bet_type]['correct'] += int(hit)


def _finish_accuracy(item):
    if item['total'] > 0:
        item['accuracy'] = round(item['correct'] / item['total'], 4)
