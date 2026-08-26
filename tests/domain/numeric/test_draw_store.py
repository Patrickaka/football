"""开奖记录的持久化。

线上是 `data/kl8_history.json`：2048 期、878KB、每次追加都整文件重写。
入库后按期号增量写入，不再有「读 878KB → 改一条 → 写 878KB」。

结构依据实读（判据 4）：issue 唯一，numbers 恒 20 个，另带
date / source / fetched_at / checksum 四个溯源字段。
"""
import json
import unittest

from src.domain.numeric.draw import Draw, checksum_of
from src.domain.numeric.draw_store import DrawStore
from src.domain.numeric.repository import create_all
from src.foundation.store import Database, make_engine

VALID = [4, 9, 10, 12, 17, 18, 22, 28, 33, 38, 42, 44, 47, 48, 61, 63, 64, 67, 73, 74]
REAL = {
    'issue': '2026227', 'numbers': list(VALID), 'date': '2026-08-25',
    'source': 'api_huiniao', 'fetched_at': '2026-08-26T14:54:41',
    'checksum': 'ec0a8edd6cbd',
}


def _draw(issue, numbers=None, **overrides):
    return Draw.parse({**REAL, 'issue': issue,
                       'numbers': numbers or VALID, **overrides})


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = DrawStore(self.db, game='kl8')


class RoundTripTests(_Base):
    def test_empty_store(self):
        self.assertEqual(self.store.load(), [])

    def test_round_trip_preserves_every_field(self):
        self.store.save([_draw('2026227')])
        self.assertEqual(self.store.load(), [_draw('2026227')])

    def test_numbers_survive_as_integers(self):
        """号码存 JSON，读回必须还是整数列表——变成字符串的话所有集合
        运算都会静默失配，命中数恒为 0。"""
        self.store.save([_draw('2026227')])
        numbers = self.store.load()[0].numbers
        self.assertEqual(numbers, tuple(VALID))
        self.assertTrue(all(isinstance(n, int) for n in numbers))

    def test_returns_newest_first(self):
        """线上文件就是新→旧，下游多处直接取第一条当最新一期。"""
        self.store.save([_draw('2026100'), _draw('2026227'), _draw('2026150')])
        self.assertEqual([d.issue for d in self.store.load()],
                         ['2026227', '2026150', '2026100'])

    def test_latest_returns_the_newest_issue(self):
        self.store.save([_draw('2026100'), _draw('2026227')])
        self.assertEqual(self.store.latest().issue, '2026227')

    def test_latest_of_an_empty_store(self):
        self.assertIsNone(self.store.latest())

    def test_count(self):
        self.store.save([_draw('1'), _draw('2')])
        self.assertEqual(self.store.count(), 2)


class GameIsolationTests(_Base):
    """一张表装多种玩法的开奖，靠 game 列隔离。

    分表的话每加一种彩票就要建一张结构完全相同的表，而它们的字段确实相同
    ——不同的只是号码范围与个数，那是领域规则不是存储结构。
    """

    def test_games_do_not_see_each_other(self):
        other = DrawStore(self.db, game='dlt')
        self.store.save([_draw('2026227')])
        self.assertEqual(other.load(), [])
        self.assertEqual(len(self.store.load()), 1)

    def test_same_issue_in_two_games_both_survive(self):
        """不同彩种的期号会撞——它们各自编号，2026227 两边都可能有。"""
        other = DrawStore(self.db, game='dlt')
        self.store.save([_draw('2026227')])
        other.save([_draw('2026227')])
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(other.count(), 1)


class IncrementalSaveTests(_Base):
    """增量写入。整表替换会让 2048 期在每次追加时全部重写一遍。"""

    def test_save_adds_without_touching_the_rest(self):
        self.store.save([_draw('2026226'), _draw('2026227')])
        self.store.save([_draw('2026228')])
        self.assertEqual([d.issue for d in self.store.load()],
                         ['2026228', '2026227', '2026226'])

    def test_resaving_the_same_issue_is_idempotent(self):
        self.store.save([_draw('2026227')])
        self.store.save([_draw('2026227')])
        self.assertEqual(self.store.count(), 1)

    def test_conflicting_numbers_keep_the_stored_value(self):
        """已入库的开奖号码不允许被后来的抓取覆盖——冲突只能是某一边错了。"""
        self.store.save([_draw('2026227')])
        changed = VALID[:19] + [80]
        conflicts = self.store.save([_draw('2026227', numbers=changed)])
        self.assertEqual(self.store.load()[0].numbers, tuple(VALID))
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].issue, '2026227')

    def test_no_conflict_when_numbers_match(self):
        self.store.save([_draw('2026227')])
        self.assertEqual(self.store.save([_draw('2026227')]), [])

    def test_saving_nothing_is_a_noop(self):
        self.store.save([_draw('2026227')])
        self.assertEqual(self.store.save([]), [])
        self.assertEqual(self.store.count(), 1)

    def test_metadata_of_an_existing_issue_is_not_overwritten(self):
        """号码相同但来源不同时也不覆盖：先入库的那条记录的溯源信息
        才是真实抓取过程的记录。"""
        self.store.save([_draw('2026227')])
        self.store.save([_draw('2026227', source='api_other',
                               fetched_at='2027-01-01T00:00:00')])
        stored = self.store.load()[0]
        self.assertEqual(stored.source, 'api_huiniao')
        self.assertEqual(stored.fetched_at, '2026-08-26T14:54:41')


class ChecksumIntegrityTests(_Base):
    def test_stored_checksum_matches_the_numbers(self):
        self.store.save([_draw('2026227')])
        stored = self.store.load()[0]
        self.assertEqual(stored.checksum, checksum_of(stored.numbers))

    def _write_raw(self, issue, numbers, checksum):
        from src.domain.numeric.repository import DrawRepository

        DrawRepository(self.db).upsert(
            {'game': 'kl8', 'issue': issue, 'numbers': json.dumps(numbers),
             'date': '', 'source': '', 'fetched_at': '', 'checksum': checksum},
            key_cols=['game', 'issue'])

    def test_clean_store_has_no_issues(self):
        self.store.save([_draw('2026227')])
        self.assertEqual(self.store.find_corrupted(), [])

    def test_checksum_mismatch_is_reported(self):
        """校验码对不上说明号码在某个环节被改过。"""
        self._write_raw('2026227', VALID, 'deadbeefdead')
        issues = self.store.find_corrupted()
        self.assertEqual([(i.issue, i.reason) for i in issues],
                         [('2026227', 'checksum_mismatch')])
        self.assertEqual(issues[0].expected_checksum, 'ec0a8edd6cbd')

    def test_invalid_numbers_are_reported_not_silently_dropped(self):
        """号码不合规的行在 load() 里会被跳过，于是「损坏」看起来就成了
        「不存在」——比校验码不符更隐蔽，因为连痕迹都没有。"""
        self._write_raw('2026227', [1] * 20, 'ec0a8edd6cbd')
        self.assertEqual(self.store.load(), [], '不合规的行不该进入正常读取')
        issues = self.store.find_corrupted()
        self.assertEqual([(i.issue, i.reason) for i in issues],
                         [('2026227', 'invalid_numbers')])

    def test_unparsable_numbers_are_reported(self):
        from src.domain.numeric.repository import DrawRepository

        DrawRepository(self.db).upsert(
            {'game': 'kl8', 'issue': '2026227', 'numbers': '{坏的',
             'date': '', 'source': '', 'fetched_at': '', 'checksum': ''},
            key_cols=['game', 'issue'])
        self.assertEqual([i.reason for i in self.store.find_corrupted()],
                         ['invalid_numbers'])


if __name__ == '__main__':
    unittest.main()
