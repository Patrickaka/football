"""kl8 开奖历史的迁移。

样例取自线上真实文件的形状：顶层 `{'results': [...]}`，每条含
issue / numbers / date / source / fetched_at / checksum。
"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.migrate.kl8_history_to_store import migrate, parse_all, read_history, verify
from src.domain.numeric.draw_store import DrawStore
from src.domain.numeric.repository import create_all
from src.foundation.store import Database, make_engine

VALID = [4, 9, 10, 12, 17, 18, 22, 28, 33, 38, 42, 44, 47, 48, 61, 63, 64, 67, 73, 74]
OTHER = [1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27]


def _record(issue, numbers=None, **extra):
    return {'issue': issue, 'numbers': list(numbers or VALID), 'date': '2026-08-25',
            'source': 'api_huiniao', 'fetched_at': '2026-08-26T14:54:41',
            'checksum': 'ec0a8edd6cbd', **extra}


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = DrawStore(self.db, game='kl8')
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _file(self, payload):
        path = Path(self.dir.name) / 'kl8_history.json'
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        return str(path)


class ReadTests(_Base):
    """顶层形状历史上出现过三种，迁移脚本挑食会让人以为数据丢了。"""

    def test_reads_results_wrapper(self):
        path = self._file({'results': [_record('1')]})
        self.assertEqual(len(read_history(path)), 1)

    def test_reads_data_wrapper(self):
        path = self._file({'data': [_record('1')]})
        self.assertEqual(len(read_history(path)), 1)

    def test_reads_a_bare_list(self):
        path = self._file([_record('1')])
        self.assertEqual(len(read_history(path)), 1)

    def test_reads_an_empty_file(self):
        self.assertEqual(read_history(self._file({'results': []})), [])


class ParseTests(unittest.TestCase):
    def test_rejected_records_are_returned_not_dropped(self):
        """迁移时出现不合规记录，人得看到是哪几条——否则「2048 期只迁进来
        2040 期」无从解释。"""
        draws, rejected = parse_all([_record('1'), {'issue': '2', 'numbers': [1, 2]}])
        self.assertEqual(len(draws), 1)
        self.assertEqual(rejected, [{'issue': '2', 'numbers': [1, 2]}])


class MigrateTests(_Base):
    def test_migrates_and_verifies(self):
        path = self._file({'results': [_record('2026227'), _record('2026226', OTHER)]})
        stats = migrate(path, self.db)
        self.assertEqual(stats['read'], 2)
        self.assertEqual(stats['written'], 2)
        self.assertEqual(verify(path, self.db), [])

    def test_dry_run_writes_nothing(self):
        path = self._file({'results': [_record('2026227')]})
        stats = migrate(path, self.db, dry_run=True)
        self.assertEqual(stats['parsed'], 1)
        self.assertEqual(self.store.count(), 0)

    def test_rerun_is_idempotent(self):
        path = self._file({'results': [_record('2026227')]})
        migrate(path, self.db)
        stats = migrate(path, self.db)
        self.assertEqual(stats['written'], 0)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(verify(path, self.db), [])

    def test_invalid_records_are_counted_and_skipped(self):
        path = self._file({'results': [_record('1'), {'issue': '2', 'numbers': [1]}]})
        stats = migrate(path, self.db)
        self.assertEqual(stats['rejected'], 1)
        self.assertEqual(self.store.count(), 1)

    def test_conflicting_numbers_keep_the_stored_value(self):
        self.store.save(parse_all([_record('2026227')])[0])
        path = self._file({'results': [_record('2026227', OTHER)]})
        stats = migrate(path, self.db)
        self.assertEqual(stats['conflicts'], 1)
        self.assertEqual(self.store.load()[0].numbers, tuple(VALID))


class VerifyTests(_Base):
    def test_detects_missing_issues(self):
        path = self._file({'results': [_record('1'), _record('2', OTHER)]})
        migrate(path, self.db)
        path2 = self._file({'results': [_record('1'), _record('2', OTHER),
                                        _record('3', OTHER)]})
        self.assertTrue(verify(path2, self.db))

    def test_detects_changed_numbers(self):
        """只比条数会漏掉「条数对得上但号码被改过」——而号码正是这个领域里
        唯一不能失真的东西。"""
        path = self._file({'results': [_record('1')]})
        migrate(path, self.db)
        tampered = self._file({'results': [_record('1', OTHER)]})
        problems = verify(tampered, self.db)
        self.assertEqual(len(problems), 1)
        self.assertIn('号码不一致', problems[0])


if __name__ == '__main__':
    unittest.main()
