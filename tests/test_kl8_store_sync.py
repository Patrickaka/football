"""开奖数据同步进 foundation/store。

**这一批要解决的是一个分叉**：迁移之后，开奖数据有了两个写入方——线上抓取
仍然只写 `data/kl8_history.json`，而 `numeric_draw` 表只被迁移脚本写过一次。
不接上的话，库里的数据会越来越旧，**而且没有任何报错**。

做法是先双写：文件照旧（分析器还在读它），同时镜像进库，并提供一个对账入口。
读取路径的切换留到下一批——一次只动一头，出问题时才分得清是哪一头。
"""
import unittest
from unittest import mock

from src.domain.numeric.draw_store import DrawStore
from src.domain.numeric.repository import create_all
from src.foundation.store import Database, make_engine
from src.kl8 import store_sync

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
        patcher = mock.patch.object(store_sync, '_open_store', lambda: self.store)
        patcher.start()
        self.addCleanup(patcher.stop)


class MirrorTests(_Base):
    def test_writes_new_draws(self):
        stats = store_sync.mirror_to_store([_record('2026227'), _record('2026226', OTHER)])
        self.assertEqual(stats['written'], 2)
        self.assertEqual(self.store.count(), 2)

    def test_is_idempotent(self):
        store_sync.mirror_to_store([_record('2026227')])
        stats = store_sync.mirror_to_store([_record('2026227')])
        self.assertEqual(stats['written'], 0)
        self.assertEqual(self.store.count(), 1)

    def test_only_new_issues_are_written(self):
        """抓取每次都返回完整历史（2048 期），镜像时只该写新增的那几期。"""
        store_sync.mirror_to_store([_record(str(i)) for i in range(10)])
        stats = store_sync.mirror_to_store([_record(str(i)) for i in range(12)])
        self.assertEqual(stats['written'], 2)

    def test_invalid_records_are_counted_not_fatal(self):
        stats = store_sync.mirror_to_store([_record('1'), {'issue': '2', 'numbers': [1]}])
        self.assertEqual(stats['rejected'], 1)
        self.assertEqual(self.store.count(), 1)

    def test_conflicting_numbers_keep_the_stored_value(self):
        store_sync.mirror_to_store([_record('2026227')])
        stats = store_sync.mirror_to_store([_record('2026227', OTHER)])
        self.assertEqual(stats['conflicts'], 1)
        self.assertEqual(self.store.load()[0].numbers, tuple(VALID))

    def test_empty_input(self):
        self.assertEqual(store_sync.mirror_to_store([])['written'], 0)


class FailureIsolationTests(unittest.TestCase):
    """**镜像失败绝不能影响抓取**。

    抓取是主链路，落库是旁路。数据库抖一下就让开奖数据抓不下来，
    等于用一个次要设施的可用性绑架了主要业务。
    """

    def test_store_failure_is_swallowed(self):
        with mock.patch.object(store_sync, '_open_store',
                               side_effect=RuntimeError('连不上')):
            stats = store_sync.mirror_to_store([_record('1')])
        self.assertEqual(stats['error'], '连不上')
        self.assertEqual(stats['written'], 0)

    def test_write_failure_is_swallowed(self):
        broken = mock.Mock()
        broken.save.side_effect = RuntimeError('磁盘满了')
        with mock.patch.object(store_sync, '_open_store', lambda: broken):
            stats = store_sync.mirror_to_store([_record('1')])
        self.assertIn('error', stats)


class ReconcileTests(_Base):
    """对账入口。双写期间两头会不会分叉，只能靠主动比对回答。"""

    def test_reports_nothing_when_in_sync(self):
        records = [_record('2026227'), _record('2026226', OTHER)]
        store_sync.mirror_to_store(records)
        self.assertEqual(store_sync.reconcile(records), [])

    def test_reports_issues_missing_from_the_store(self):
        store_sync.mirror_to_store([_record('2026227')])
        problems = store_sync.reconcile([_record('2026227'), _record('2026226', OTHER)])
        self.assertEqual(len(problems), 1)
        self.assertIn('2026226', problems[0])

    def test_reports_number_divergence(self):
        """条数对得上但号码不同——这才是双写最怕的情形，只比条数看不出来。"""
        store_sync.mirror_to_store([_record('2026227')])
        problems = store_sync.reconcile([_record('2026227', OTHER)])
        self.assertEqual(len(problems), 1)
        self.assertIn('号码不一致', problems[0])

    def test_extra_rows_in_the_store_are_reported(self):
        """库里多出文件没有的期号，同样是分叉——方向相反而已。"""
        store_sync.mirror_to_store([_record('2026227'), _record('2026226', OTHER)])
        problems = store_sync.reconcile([_record('2026227')])
        self.assertTrue(any('2026226' in p for p in problems), problems)

    def test_unavailable_store_reports_a_problem_not_silence(self):
        with mock.patch.object(store_sync, '_open_store',
                               side_effect=RuntimeError('连不上')):
            problems = store_sync.reconcile([_record('1')])
        self.assertEqual(len(problems), 1)
        self.assertIn('连不上', problems[0])


if __name__ == '__main__':
    unittest.main()


class FetchPathWiringTests(unittest.TestCase):
    """抓取路径接线：接错了不会报错，只会让库安静地停在迁移那一刻。"""

    def test_save_kl8_data_mirrors_into_the_store(self):
        import json
        import tempfile
        from pathlib import Path

        from src.kl8 import fetch as kl8_fetch

        mirrored = []
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / 'kl8_history.json'
            history.write_text(json.dumps({'results': []}), encoding='utf-8')
            with mock.patch.object(kl8_fetch, 'KL8_HISTORY_FILE', str(history)), \
                 mock.patch.object(kl8_fetch, 'clear_cache', lambda: None), \
                 mock.patch('src.kl8.store_sync.mirror_to_store',
                            lambda records: mirrored.append(records) or
                            {'written': len(records)}):
                kl8_fetch.save_kl8_data([_record('2026227')])

        self.assertEqual(len(mirrored), 1, '抓取路径没有镜像进库')
        self.assertEqual([r['issue'] for r in mirrored[0]], ['2026227'])

    def test_mirror_failure_does_not_break_the_fetch(self):
        """镜像挂掉时抓取必须照常返回合并后的历史。"""
        import json
        import tempfile
        from pathlib import Path

        from src.kl8 import fetch as kl8_fetch

        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / 'kl8_history.json'
            history.write_text(json.dumps({'results': []}), encoding='utf-8')
            with mock.patch.object(kl8_fetch, 'KL8_HISTORY_FILE', str(history)), \
                 mock.patch.object(kl8_fetch, 'clear_cache', lambda: None), \
                 mock.patch('src.kl8.store_sync.mirror_to_store',
                            side_effect=RuntimeError('库挂了')):
                result = kl8_fetch.save_kl8_data([_record('2026227')])

        self.assertIsNotNone(result, '镜像失败把抓取一起拖垮了')
        self.assertEqual(len(result), 1)
