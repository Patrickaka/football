"""校准器迁至 foundation/store。

与 Elo 同构：存储调用原本只有 _load/save 两处，迁移是替换这两个方法，
概率校准算法逐字不动。

stats 的结构从代码读出（线上无数据）：
{bucket: {count, weighted_count, success, weighted_success,
          predicted_sum, weighted_predicted_sum}}
bucket 形如 'spf|NBA|medium'，三段式 bet_type|league|confidence，'*' 表通配。
"""
import unittest

from src.domain.sports.basketball.calibration import BasketballCalibrator
from src.domain.sports.basketball.calibration_store import CalibrationStore
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = CalibrationStore(self.db)

    def _calibrator(self):
        return BasketballCalibrator(store=self.store)


class StoreRoundTripTests(_Base):
    STATS = {
        'spf|NBA|medium': {
            'count': 10, 'weighted_count': 8.5, 'success': 6,
            'weighted_success': 5.1, 'predicted_sum': 5.8,
            'weighted_predicted_sum': 4.9,
        },
        'spf|*|*': {
            'count': 25, 'weighted_count': 20.0, 'success': 14,
            'weighted_success': 11.2, 'predicted_sum': 14.5,
            'weighted_predicted_sum': 11.6,
        },
    }

    def test_load_from_empty_store(self):
        self.assertEqual(self.store.load(), {})

    def test_round_trip_preserves_all_six_fields(self):
        self.store.save(self.STATS)
        loaded = self.store.load()
        self.assertEqual(set(loaded), set(self.STATS))
        for bucket, expected in self.STATS.items():
            for field, value in expected.items():
                self.assertAlmostEqual(loaded[bucket][field], value, places=6,
                                       msg=f'{bucket}.{field} 不一致')

    def test_bucket_with_wildcards_survives(self):
        """通配桶 'spf|*|*' 的键含特殊字符，不能被存储层改写。"""
        self.store.save(self.STATS)
        self.assertIn('spf|*|*', self.store.load())

    def test_save_replaces_rather_than_accumulates(self):
        self.store.save(self.STATS)
        self.store.save({'dx|*|*': {
            'count': 1, 'weighted_count': 1.0, 'success': 1,
            'weighted_success': 1.0, 'predicted_sum': 0.6,
            'weighted_predicted_sum': 0.6,
        }})
        loaded = self.store.load()
        self.assertEqual(set(loaded), {'dx|*|*'})


class CalibratorTests(_Base):
    def test_record_persists_across_reload(self):
        first = self._calibrator()
        first.record('spf', 0.6, True, league='NBA', confidence='medium')

        second = self._calibrator()
        self.assertEqual(second.stats.keys(), first.stats.keys())
        self.assertGreater(len(second.stats), 0)

    def test_record_creates_three_granularity_levels(self):
        """每条记录写入三个粒度：最细、通配联赛、全通配。"""
        calibrator = self._calibrator()
        calibrator.record('spf', 0.6, True, league='NBA', confidence='medium')
        self.assertEqual(len(calibrator.stats), 3)
        self.assertIn('spf|NBA|medium', calibrator.stats)
        self.assertIn('spf|*|*', calibrator.stats)

    def test_unknown_bet_type_is_ignored(self):
        calibrator = self._calibrator()
        calibrator.record('unknown', 0.6, True)
        self.assertEqual(calibrator.stats, {})

    def test_success_count_tracks_hits(self):
        calibrator = self._calibrator()
        calibrator.record('spf', 0.6, True, league='NBA', confidence='medium')
        calibrator.record('spf', 0.6, False, league='NBA', confidence='medium')
        bucket = calibrator.stats['spf|NBA|medium']
        self.assertEqual(bucket['count'], 2)
        self.assertEqual(bucket['success'], 1)

    def test_get_stats_filters_by_bet_type(self):
        calibrator = self._calibrator()
        calibrator.record('spf', 0.6, True, league='NBA', confidence='medium')
        calibrator.record('dx', 0.5, True, league='NBA', confidence='medium')
        self.assertTrue(all(k.startswith('spf')
                            for k in calibrator.get_stats(bet_type='spf')))


class NoStoreTests(unittest.TestCase):
    def test_without_store_it_degrades_to_memory_only(self):
        calibrator = BasketballCalibrator()
        calibrator.record('spf', 0.6, True, league='NBA', confidence='medium')
        self.assertEqual(len(calibrator.stats), 3)
        calibrator.save()  # 不抛异常


if __name__ == '__main__':
    unittest.main()
