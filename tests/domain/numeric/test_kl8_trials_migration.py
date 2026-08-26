"""kl8 策略试验记录的迁移。"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.migrate.kl8_trials_to_store import migrate, read_trials, verify
from src.domain.numeric.repository import create_all
from src.domain.numeric.trial_store import TrialStore
from src.foundation.store import Database, make_engine
from tests.domain.numeric.test_trial_store import REAL


def _trial(**overrides):
    return {**REAL, **overrides}


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = TrialStore(self.db, game='kl8')
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _file(self, trials):
        path = Path(self.dir.name) / 'trials.json'
        path.write_text(json.dumps(trials, ensure_ascii=False), encoding='utf-8')
        return str(path)


class ReadTests(_Base):
    def test_reads_a_list(self):
        self.assertEqual(len(read_trials(self._file([_trial()]))), 1)

    def test_non_list_payload_reads_as_empty(self):
        self.assertEqual(read_trials(self._file({'trials': []})), [])


class MigrateTests(_Base):
    def test_migrates_and_verifies(self):
        path = self._file([_trial(tested_at=f'2026-06-26T14:07:{i:02d}')
                           for i in range(5)])
        stats = migrate(path, self.db)
        self.assertEqual(stats['read'], 5)
        self.assertEqual(stats['written'], 5)
        self.assertEqual(verify(path, self.db), [])

    def test_dry_run_writes_nothing(self):
        path = self._file([_trial()])
        self.assertEqual(migrate(path, self.db, dry_run=True)['read'], 1)
        self.assertEqual(self.store.count(), 0)

    def test_duplicate_keys_in_the_file_collapse(self):
        path = self._file([_trial(), _trial(raw_p_value=0.9)])
        stats = migrate(path, self.db)
        self.assertEqual(stats['unique'], 1)
        self.assertEqual(stats['written'], 1)
        self.assertEqual(verify(path, self.db), [])

    def test_rerun_is_idempotent(self):
        path = self._file([_trial()])
        migrate(path, self.db)
        self.assertEqual(migrate(path, self.db)['written'], 0)
        self.assertEqual(self.store.count(), 1)

    def test_batching_writes_everything(self):
        """整批一次 executemany 会把参数拼成极长的 SQL，MySQL 的
        max_allowed_packet 顶不住，所以要分批——分批不能丢数据。"""
        path = self._file([_trial(tested_at=f'2026-06-26T14:{i // 60:02d}:{i % 60:02d}')
                           for i in range(250)])
        stats = migrate(path, self.db, batch_size=37)
        self.assertEqual(stats['written'], 250)
        self.assertEqual(self.store.count(), 250)
        self.assertEqual(verify(path, self.db), [])


class VerifyTests(_Base):
    def test_detects_missing_records(self):
        path = self._file([_trial()])
        migrate(path, self.db)
        bigger = self._file([_trial(), _trial(tested_at='2026-06-26T14:07:26')])
        self.assertTrue(verify(bigger, self.db))

    def test_detects_changed_p_value(self):
        """只比条数会漏掉「条数对得上但 p 值被改过」——p 值是 FDR 校正的
        输入，错了会安静地把一个无效策略判成显著。"""
        path = self._file([_trial()])
        migrate(path, self.db)
        tampered = self._file([_trial(raw_p_value=0.999)])
        problems = verify(tampered, self.db)
        self.assertEqual(len(problems), 1)
        self.assertIn('raw_p_value', problems[0])

    def test_detects_changed_weights(self):
        path = self._file([_trial()])
        migrate(path, self.db)
        tampered = self._file([_trial(feature_weights={'frequency': 0.5})])
        self.assertTrue(any('feature_weights' in p for p in verify(tampered, self.db)))


if __name__ == '__main__':
    unittest.main()
