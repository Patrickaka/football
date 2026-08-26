"""赔率快照的存取与追踪迁入领域层。

结构依据 2026-08-26 从线上 kv_store 实读（判据 4）：
`{match_key: [snapshot, ...]}`，match_key 形如 `2026-07-23_水星_火花`，
快照字段为 ts / observed_ts / 三类盘口的水位 / handicap（**字符串**）/ total_line。
线上共 32 场 171 条，单场最多 15 条，其中 119 条带 observed_ts。

**本批修一处静默丢数据**：`bb_odds_snapshot` 表没有 observed_ts 列，迁移脚本
按字段白名单拷贝，于是这个字段被无声丢弃。它记的是「这个盘口最后一次被确认
仍然没变的时刻」——丢了不会报错，只是往后再也说不清一条陈旧快照是刚被确认过
还是真的很久没人看了。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.domain.sports.basketball.odds_history import (
    HISTORY_CAP, OddsHistoryStore, OddsTracker,
)
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine
from tests.domain.golden import as_json, load

GOLDEN = load('odds_history')

NOW = datetime(2026, 8, 26, 12, 0, 0)
NOW_ISO = NOW.isoformat()

# 取自线上真实数据的一段
REAL_HISTORY = {
    '2026-07-23_水星_火花': [
        {'ts': '2026-07-22T11:38:12.250497', 'spf_home': 1.81, 'spf_away': 1.6,
         'rqspf_home': 1.7, 'rqspf_away': 1.7, 'dx_over': 1.66, 'dx_under': 1.74,
         'handicap': '-1.5', 'total_line': 177.5},
        {'ts': '2026-07-22T14:39:56.285547', 'spf_home': 1.86, 'spf_away': 1.56,
         'rqspf_home': 1.81, 'rqspf_away': 1.6, 'dx_over': 1.62, 'dx_under': 1.78,
         'handicap': '-1.5', 'total_line': 177.5,
         'observed_ts': '2026-07-22T16:10:01.000000'},
    ],
    '2026-07-23_天猫_风暴': [
        {'ts': '2026-07-22T11:38:12.250497', 'spf_home': 1.07, 'spf_away': 4.08,
         'rqspf_home': 1.7, 'rqspf_away': 1.7, 'dx_over': 1.62, 'dx_under': 1.78,
         'handicap': '+9.5', 'total_line': 177.5},
    ],
}

MATCHES = [
    {'id': '2026-08-27_甲_乙', 'spf_home': 1.80, 'spf_away': 2.00,
     'rqspf_home': 1.90, 'rqspf_away': 1.90, 'dx_over': 1.85, 'dx_under': 1.95,
     'handicap': '-3.5', 'total_line': 210.5},
    {'id': '2026-08-27_丙_丁', 'spf_home': None, 'spf_away': None,
     'rqspf_home': 1.70, 'rqspf_away': 1.70, 'dx_over': None, 'dx_under': None,
     'handicap': '+9.5', 'total_line': None},
    {'id': '2026-08-27_无赔率_对手', 'spf_home': None, 'spf_away': None,
     'rqspf_home': None, 'rqspf_away': None, 'dx_over': None, 'dx_under': None,
     'handicap': None, 'total_line': None},
    {'spf_home': 1.5, 'spf_away': 2.5},
]


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = OddsHistoryStore(self.db)


class StoreRoundTripTests(_Base):
    def test_load_from_empty_store(self):
        self.assertEqual(self.store.load(), {})

    def test_round_trip_preserves_every_field(self):
        self.store.save(REAL_HISTORY)
        self.assertEqual(self.store.load(), REAL_HISTORY)

    def test_observed_ts_survives_the_round_trip(self):
        """迁移脚本按字段白名单拷贝，observed_ts 曾被整列丢掉。"""
        self.store.save(REAL_HISTORY)
        loaded = self.store.load()
        self.assertEqual(loaded['2026-07-23_水星_火花'][1]['observed_ts'],
                         '2026-07-22T16:10:01.000000')
        self.assertNotIn('observed_ts', loaded['2026-07-23_天猫_风暴'][0],
                         '没有 observed_ts 的快照不该凭空多出这个键')

    def test_numeric_handicap_is_normalised_to_text(self):
        """让分列是字符串。上游偶尔给来数值，落库后读回就变了类型——
        两次采集因此永远判定「变了」，每轮都追加一条新快照。"""
        self.store.save({'m': [{'ts': '2026-08-26T09:00:00', 'handicap': -3.5,
                                'spf_home': 1.8, 'spf_away': 2.0}]})
        self.assertEqual(self.store.load()['m'][0]['handicap'], '-3.5')

    def test_handicap_stays_a_string(self):
        """线上 handicap 是 '+9.5' 这样的带符号字符串，转成数值会丢掉符号语义。"""
        self.store.save(REAL_HISTORY)
        value = self.store.load()['2026-07-23_天猫_风暴'][0]['handicap']
        self.assertIsInstance(value, str)
        self.assertEqual(value, '+9.5')

    def test_list_order_is_preserved_exactly(self):
        """存什么顺序就读什么顺序。按时间戳排序看着更"正确"，但那是在
        悄悄改写数据——顺序是源结构的一部分。"""
        reverse_chronological = {
            'm': list(reversed(REAL_HISTORY['2026-07-23_水星_火花']))}
        self.store.save(reverse_chronological)
        self.assertEqual(self.store.load(), reverse_chronological)

    def test_two_snapshots_sharing_a_timestamp_both_survive(self):
        """同一场、同一时刻的两条快照。真实数据里不会出现，但列表结构
        允许，主键若建在时间戳上就会静默少掉一条。"""
        same_ts = {'m': [
            _snap('2026-08-26T12:00:00', spf_home=1.8, spf_away=2.0),
            _snap('2026-08-26T12:00:00', spf_home=1.6, spf_away=2.2),
        ]}
        self.store.save(same_ts)
        self.assertEqual(self.store.load(), same_ts)

    def test_save_replaces_rather_than_accumulates(self):
        """整体替换语义：截断后的历史必须真的变短。"""
        self.store.save(REAL_HISTORY)
        trimmed = {'2026-07-23_水星_火花': REAL_HISTORY['2026-07-23_水星_火花'][1:]}
        self.store.save(trimmed)
        self.assertEqual(self.store.load(), trimmed)

    def test_rows_are_ordered_by_position_not_insertion(self):
        """按 seq 排序取回。不排序时 SQLite 恰好按插入顺序返回，正好掩盖
        这个问题——必须让插入顺序与 seq 顺序不一致才测得出来。"""
        from src.domain.sports.basketball.repository import OddsSnapshotRepository

        repo = OddsSnapshotRepository(self.db)
        for seq, home in ((2, 1.5), (0, 1.8), (1, 1.6)):
            repo.upsert({'match_key': 'm', 'seq': seq,
                         'captured_at': f'2026-08-26T1{seq}:00:00',
                         'spf_home': home}, key_cols=['match_key', 'seq'])
        self.assertEqual([s['spf_home'] for s in self.store.load()['m']],
                         [1.8, 1.6, 1.5])

    def test_history_for_one_match(self):
        self.store.save(REAL_HISTORY)
        self.assertEqual(len(self.store.history_for('2026-07-23_水星_火花')), 2)
        self.assertEqual(self.store.history_for('不存在'), [])


class TrackerGoldenTests(_Base):
    # (黄金键, 赛程, 已有历史)
    CASES = [
        ('first', MATCHES, {}),
        ('on_top_of_history', MATCHES, REAL_HISTORY),
        ('empty_schedule', [], REAL_HISTORY),
    ]

    def test_every_capture_scenario(self):
        for name, matches, history in self.CASES:
            with self.subTest(case=name):
                self.store.save(history)
                count = _tracker_for(self.store, matches).track('2026-08-27')
                self.assertEqual(
                    as_json({'count': count, 'history': self.store.load()}),
                    GOLDEN[name])

    def test_repeated_identical_capture(self):
        """连采两轮、盘口没动：第二轮只推进 observed_ts，不该追加快照。"""
        tracker = _tracker_for(self.store, MATCHES)
        tracker.track('2026-08-27')
        first = self.store.load()
        tracker.track('2026-08-27')
        second = self.store.load()
        self.assertEqual({k: len(v) for k, v in second.items()},
                         {k: len(v) for k, v in first.items()})


_FIELDS = ('spf_home', 'spf_away', 'rqspf_home', 'rqspf_away',
           'dx_over', 'dx_under', 'handicap', 'total_line')


def _memory_store():
    db = Database(make_engine('sqlite+pysqlite:///:memory:'))
    create_all(db)
    return OddsHistoryStore(db)


def _tracker_for(store, matches, fail=False):
    def fetch(date=None):
        if fail:
            raise IOError('源站挂了')
        return [dict(m) for m in matches]

    return OddsTracker(schedule_fetcher=fetch, store=store, now_fn=lambda: NOW)


def _snap(ts, **odds):
    """构造快照。八个盘口键一律齐全——线上数据就是这个形状。"""
    return {'ts': ts, **{field: odds.get(field) for field in _FIELDS}}


class TrackerBehaviourTests(_Base):
    def _tracker(self, matches, fail=False):
        def fetch(date=None):
            if fail:
                raise IOError('源站挂了')
            return [dict(m) for m in matches]

        return OddsTracker(schedule_fetcher=fetch, store=self.store,
                           now_fn=lambda: NOW)

    def test_counts_only_matches_worth_recording(self):
        """无 id 的行与三类盘口全空的场次都不计数——它们进不了历史。"""
        self.assertEqual(self._tracker(MATCHES).track('2026-08-27'), 2)

    def test_all_empty_snapshot_is_not_stored(self):
        self._tracker(MATCHES).track('2026-08-27')
        self.assertNotIn('2026-08-27_无赔率_对手', self.store.load())

    def test_numeric_handicap_does_not_defeat_deduplication(self):
        """入参给数值、库里存字符串时，去重仍要认得出「没变」。"""
        match = dict(MATCHES[0], handicap=-3.5)
        tracker = self._tracker([match])
        tracker.track('2026-08-27')
        tracker.track('2026-08-27')
        history = self.store.load()[match['id']]
        self.assertEqual(len(history), 1, '类型不一致导致每轮都追加了快照')
        self.assertEqual(history[0]['observed_ts'], NOW_ISO)

    def test_unchanged_odds_update_observed_ts_instead_of_appending(self):
        """盘口没动时不追加新快照，只把「最后确认时刻」往前推。

        否则每轮追踪都会造出一条新快照，把一个几小时没动过的旧信号
        伪装成刚刚发生的变化——走势的新鲜度判断会全线失真。
        """
        tracker = self._tracker(MATCHES)
        tracker.track('2026-08-27')
        before = self.store.load()['2026-08-27_甲_乙']
        tracker.track('2026-08-27')
        after = self.store.load()['2026-08-27_甲_乙']
        self.assertEqual(len(after), len(before), '盘口没变却追加了新快照')
        self.assertEqual(after[-1]['observed_ts'], NOW_ISO)
        self.assertEqual(after[-1]['ts'], before[-1]['ts'], '变化时刻被改写了')

    def test_changed_odds_append_a_new_snapshot(self):
        tracker = self._tracker(MATCHES)
        tracker.track('2026-08-27')
        moved = [dict(MATCHES[0], spf_home=1.60)]
        self._tracker(moved).track('2026-08-27')
        history = self.store.load()['2026-08-27_甲_乙']
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]['spf_home'], 1.60)

    def test_history_is_capped_and_keeps_the_newest(self):
        """上限用显式传入的值，不用被测模块的常量——拿常量当期望，把常量
        改坏时期望跟着变，这条测试就永远是绿的。

        留下的必须是**最新**的那些：走势只看首尾与最后一次变化，砍掉新的
        等于把刚发生的变盘丢了。
        """
        cap = 5
        old = {'2026-08-27_甲_乙': [
            _snap(f'2026-08-01T00:00:{i:02d}', spf_home=1.5 + i * 0.01, spf_away=2.0)
            for i in range(cap + 3)
        ]}
        self.store.save(old)
        tracker = OddsTracker(schedule_fetcher=lambda d=None: [dict(MATCHES[0])],
                              store=self.store, now_fn=lambda: NOW, cap=cap)
        tracker.track('2026-08-27')
        kept = self.store.load()['2026-08-27_甲_乙']
        self.assertEqual(len(kept), cap)
        self.assertEqual(kept[-1]['ts'], NOW_ISO, '本轮新快照没留下')
        self.assertEqual([s['ts'] for s in kept[:-1]],
                         [s['ts'] for s in old['2026-08-27_甲_乙'][-(cap - 1):]],
                         '截断砍掉的是新快照而不是旧的')

    def test_default_cap_is_240(self):
        """默认上限单独钉住，和上面那条一起：一条测行为，一条测取值。"""
        self.assertEqual(HISTORY_CAP, 240)

    def test_fetch_failure_returns_zero_and_keeps_history(self):
        self.store.save(REAL_HISTORY)
        self.assertEqual(self._tracker(MATCHES, fail=True).track('2026-08-27'), 0)
        self.assertEqual(self.store.load(), REAL_HISTORY)

    def test_empty_schedule_leaves_history_untouched(self):
        self.store.save(REAL_HISTORY)
        self.assertEqual(self._tracker([]).track('2026-08-27'), 0)
        self.assertEqual(self.store.load(), REAL_HISTORY)

    def test_empty_schedule_does_not_rewrite_the_table(self):
        """没有比赛就不该落盘。save 是整体替换（清空再写），白跑一次等于
        把整张表删掉重建——内容看不出差别，代价是实打实的。"""
        writes = []
        store = mock.Mock()
        store.load.return_value = {}
        store.save.side_effect = lambda h: writes.append(h)
        OddsTracker(schedule_fetcher=lambda d=None: [], store=store,
                    now_fn=lambda: NOW).track('2026-08-27')
        self.assertEqual(writes, [])


class NoLegacyImportTests(unittest.TestCase):
    def test_does_not_import_legacy_package(self):
        import ast
        import inspect

        from src.domain.sports.basketball import odds_history

        tree = ast.parse(inspect.getsource(odds_history))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(alias.name.startswith('src.basketball'))
            elif isinstance(node, ast.ImportFrom):
                module = ('.' * (node.level or 0)) + (node.module or '')
                self.assertFalse(module.startswith('src.basketball'), module)
                self.assertFalse(module.startswith('.'), f'不该有相对导入: {module}')


if __name__ == '__main__':
    unittest.main()
