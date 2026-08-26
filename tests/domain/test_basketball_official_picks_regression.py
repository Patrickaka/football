"""官方推荐与准确率统计的回归判据。

原本是 `tests/test_basketball_official_picks.py`，直接测 `src/basketball`。
核心区分是：**「模型的看法」与「计入准确率的官方推荐」是两件事**。
两者混在一起时，低置信度的场次会拖累统计，让人无从判断模型在真正有把握时
表现如何。

打桩方式随之改变：旧实现只能 patch `kv_store.load`，领域层直接给内存库。
"""
import unittest

from src.domain.sports.basketball.analysis import official_pick_status
from src.domain.sports.basketball.records import (
    PredictionRecorder, PredictionRecordStore, summarize,
)
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine


class OfficialPickStatusTests(unittest.TestCase):
    def test_low_confidence_pick_is_not_official(self):
        status = official_pick_status('spf', 0.58, 'low')
        self.assertFalse(status['playable'])
        self.assertFalse(status['official'])
        self.assertEqual(status['skip_reason'], 'low_confidence')

    def test_probability_below_threshold_is_not_official(self):
        status = official_pick_status('spf', 0.55, 'high')
        self.assertFalse(status['official'])
        self.assertEqual(status['skip_reason'], 'probability_below_threshold')

    def test_confident_and_strong_enough_is_official(self):
        status = official_pick_status('spf', 0.58, 'high')
        self.assertTrue(status['playable'])
        self.assertTrue(status['official'])
        self.assertIsNone(status['skip_reason'])


class StatsTests(unittest.TestCase):
    def test_stats_skip_non_playable_predictions(self):
        records = [{
            'result': {'home_score': 92, 'away_score': 88},
            'spf': {'available': True, 'playable': False,
                    'recommendation': '主胜'},
            'rqspf': None, 'dx': None,
        }]
        stats = summarize(records)
        self.assertEqual(stats['settled_count'], 1)
        self.assertEqual(stats['official_predictions'], 0)
        self.assertEqual(stats['spf']['total'], 0)
        self.assertEqual(stats['spf']['correct'], 0)

    def test_stats_count_playable_predictions(self):
        records = [{
            'result': {'home_score': 92, 'away_score': 88},
            'spf': {'available': True, 'playable': True,
                    'recommendation': '主胜'},
            'rqspf': None, 'dx': None,
        }]
        stats = summarize(records)
        self.assertEqual(stats['official_predictions'], 1)
        self.assertEqual(stats['spf']['total'], 1)
        self.assertEqual(stats['spf']['correct'], 1)
        self.assertEqual(stats['spf']['accuracy'], 1.0)

    def test_stats_report_water_led_accuracy_separately(self):
        """走势翻转过模型的那些单独记账——这套信号有没有用，只能靠它自己的
        命中率回答，混进总体准确率就看不出来了。"""
        records = [{
            'result': {'home_score': 100, 'away_score': 95},
            'spf': None,
            'rqspf': {'available': True, 'playable': True,
                      'recommendation': '让负', 'handicap': -8.5,
                      'movement_led': True},
            'dx': {'available': True, 'playable': True,
                   'recommendation': '小分', 'total_line': 210.5,
                   'movement_led': True},
        }]
        stats = summarize(records)
        self.assertEqual(stats['water_inference']['rqspf']['accuracy'], 1.0)
        self.assertEqual(stats['water_inference']['dx']['accuracy'], 1.0)


class SavedMetadataTests(unittest.TestCase):
    """落库时必须留下「为什么没出票」的痕迹，否则日后无法复盘。"""

    def setUp(self):
        db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(db)
        self.store = PredictionRecordStore(db)
        self.recorder = PredictionRecorder(self.store)

    def test_playable_metadata_is_persisted(self):
        self.recorder.save('2026-07-14', [{
            'match': {'id': 'm1', 'league': 'NBA', 'home': 'Home', 'away': 'Away'},
            'spf': {'available': True, 'recommendation': '主胜', 'pick_prob': 0.57,
                    'playable': False, 'official': False,
                    'skip_reason': 'low_confidence', 'home_prob': 0.57,
                    'away_prob': 0.43, 'confidence': 'low'},
        }], '')
        spf = self.store.load()[0]['spf']
        self.assertFalse(spf['playable'])
        self.assertFalse(spf['official'])
        self.assertEqual(spf['skip_reason'], 'low_confidence')
        self.assertEqual(spf['pick_prob'], 0.57)

    def test_water_inference_audit_fields_are_persisted(self):
        self.recorder.save('2026-08-20', [{
            'match': {'id': 'm2', 'league': 'NBA', 'home': 'H', 'away': 'A'},
            'rqspf': {'available': True, 'recommendation': '让负',
                      'pick_prob': .58, 'playable': True, 'official': True,
                      'movement_led': True,
                      'water_inference': {'actionable': True,
                                          'recommendation': '让负'}},
            'dx': {'available': True, 'recommendation': '小分', 'pick_prob': .57,
                   'playable': True, 'official': True, 'movement_led': True,
                   'water_inference': {'actionable': True,
                                       'recommendation': '小分'}},
        }], '')
        record = self.store.load()[0]
        self.assertTrue(record['rqspf']['movement_led'])
        self.assertEqual(record['rqspf']['water_inference']['recommendation'], '让负')
        self.assertTrue(record['dx']['movement_led'])


if __name__ == '__main__':
    unittest.main()
