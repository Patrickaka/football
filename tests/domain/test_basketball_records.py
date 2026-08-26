"""预测记录迁入领域层。

结构依据 2026-08-26 实读线上（判据 4）：`basketball_prediction_records` 是
一个有序列表，73 条，字段 date / match_id / num / league / home / away /
time / version / created_at / spf / rqspf / dx / result。
**match_id 不唯一**（73 条只有 43 个不同的 match_id）——同一场比赛在不同日期、
不同版本下各留一条；业务上的唯一键是 (date, match_id)。`result` 线上全为空，
即从来没有结算过，所以结算那条路只能靠构造用例覆盖。

主体仍是与旧实现的差分。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.basketball import records as legacy
from src.domain.sports.basketball.records import (
    MAX_RECORDS, PredictionRecorder, PredictionRecordStore, evaluate_markets,
    summarize,
)
from src.domain.sports.basketball.repository import create_all
from src.foundation.store import Database, make_engine

NOW = datetime(2026, 8, 26, 12, 0, 0)
NOW_ISO = NOW.isoformat()

# 取自线上真实记录
REAL_RECORD = {
    'date': '2026-07-13', 'match_id': '2026-07-14_火花_梦想', 'num': '周一301',
    'league': '美职女篮', 'home': '火花', 'away': '梦想', 'time': '07-14 07:00',
    'version': '2026-07-13-v2', 'created_at': '2026-07-13T15:07:08.280419',
    'spf': {'available': True, 'recommendation': '客胜', 'home_prob': 0.4075,
            'away_prob': 0.5925, 'confidence': 'low', 'elo_home_prob': 0.5925},
    'rqspf': {'available': True, 'recommendation': '让胜', 'handicap': '-7.5',
              'home_prob': 0.6507, 'away_prob': 0.3493, 'confidence': 'medium',
              'elo_margin': 4.2},
    'dx': {'available': True, 'recommendation': '小分', 'total_line': 181.5,
           'over_prob': 0.3714, 'under_prob': 0.6286, 'confidence': 'medium',
           'elo_total': 164.0},
    'result': None,
}

ANALYSIS = {
    'match': {'id': '2026-08-27_甲_乙', 'num': '周三301', 'league': 'NBA',
              'home': '甲', 'away': '乙', 'time': '08-27 07:00'},
    'spf': {'available': True, 'recommendation': '主胜', 'pick_prob': 0.62,
            'playable': True, 'official': True, 'skip_reason': None,
            'home_prob': 0.62, 'away_prob': 0.38, 'confidence': 'high',
            'elo_home_prob': 0.6, 'elo_trust': 1.0, 'market_home_prob': 0.58,
            'line_movement': {'side': 'home'}, 'sharp_confirmed': True},
    'rqspf': {'available': True, 'recommendation': '让胜', 'pick_prob': 0.57,
              'playable': True, 'official': True, 'skip_reason': None,
              'handicap': '-3.5', 'home_prob': 0.57, 'away_prob': 0.43,
              'confidence': 'medium', 'elo_margin': 4.0, 'elo_trust': 1.0,
              'market_home_prob': 0.55, 'line_movement': {'side': 'home'},
              'water_inference': {'actionable': True}, 'movement_led': True,
              'sharp_confirmed': False},
    'dx': {'available': True, 'recommendation': '大分', 'pick_prob': 0.56,
           'playable': True, 'official': True, 'skip_reason': None,
           'total_line': 210.5, 'over_prob': 0.56, 'under_prob': 0.44,
           'confidence': 'medium', 'elo_total': 215.0, 'elo_trust': 1.0,
           'market_over_prob': 0.54, 'line_movement': None,
           'water_inference': {'actionable': False}, 'movement_led': False,
           'sharp_confirmed': False},
}

UNAVAILABLE = {
    'match': {'id': '2026-08-27_丙_丁', 'num': '周三302', 'league': 'CBA',
              'home': '丙', 'away': '丁', 'time': '08-27 19:35'},
    'spf': {'available': False, 'reason': 'missing_odds'},
    'rqspf': None,
    'dx': {'available': True, 'recommendation': '小分', 'playable': False,
           'official': False, 'skip_reason': 'low_confidence',
           'total_line': 190.5, 'over_prob': 0.48, 'under_prob': 0.52,
           'confidence': 'low'},
}


class _FakeKv:
    def __init__(self, initial=None):
        self.data = {'basketball_prediction_records': list(initial or [])}

    def load(self, key, default=None):
        import copy

        return copy.deepcopy(self.data.get(key, default))

    def save(self, key, value):
        import copy

        self.data[key] = copy.deepcopy(value)


class _FakeCalibrator:
    def __init__(self):
        self.samples = []
        self.saves = 0

    def record(self, bet_type, prob, hit, league, confidence):
        self.samples.append((bet_type, round(prob, 6), hit, league, confidence))

    def save(self):
        self.saves += 1


class _FakeElo:
    def __init__(self, fail=False):
        self.updates = []
        self.fail = fail

    def update_ratings(self, home, away, home_score, away_score, league):
        if self.fail:
            raise RuntimeError('elo 挂了')
        self.updates.append((home, away, home_score, away_score, league))


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class _Base(unittest.TestCase):
    def setUp(self):
        self.db = Database(make_engine('sqlite+pysqlite:///:memory:'))
        create_all(self.db)
        self.store = PredictionRecordStore(self.db)
        self.calibrator = _FakeCalibrator()
        self.elo = _FakeElo()

    def _recorder(self, **kwargs):
        kwargs.setdefault('calibrator', self.calibrator)
        kwargs.setdefault('elo', self.elo)
        return PredictionRecorder(self.store, now_fn=lambda: NOW, **kwargs)


class StoreRoundTripTests(_Base):
    def test_empty_store(self):
        self.assertEqual(self.store.load(), [])

    def test_round_trip_preserves_the_real_record(self):
        self.store.save([REAL_RECORD])
        self.assertEqual(self.store.load(), [REAL_RECORD])

    def test_nested_market_dicts_survive_intact(self):
        """三个玩法是嵌套字典，字段随模型版本变化。存 JSON 原文，一个不丢。"""
        self.store.save([REAL_RECORD])
        loaded = self.store.load()[0]
        self.assertEqual(loaded['rqspf']['handicap'], '-7.5')
        self.assertEqual(loaded['dx']['elo_total'], 164.0)

    def test_settled_result_survives(self):
        settled = dict(REAL_RECORD, result={
            'home_score': 88, 'away_score': 90, 'status': 'finished',
            'settled_at': NOW_ISO, 'elo_updated': True, 'calibration_fed': True,
            'spf_hit': True, 'rqspf_hit': False, 'dx_hit': None,
            'spf_void': False, 'rqspf_void': False, 'dx_void': True})
        self.store.save([settled])
        self.assertEqual(self.store.load(), [settled])

    def test_order_is_preserved(self):
        records = [dict(REAL_RECORD, match_id=f'm{i}') for i in range(5)]
        self.store.save(records)
        self.assertEqual([r['match_id'] for r in self.store.load()],
                         [f'm{i}' for i in range(5)])

    def test_same_match_id_on_different_dates_both_survive(self):
        """线上 73 条只有 43 个不同的 match_id——唯一键是 (date, match_id)。"""
        records = [dict(REAL_RECORD, date='2026-07-13'),
                   dict(REAL_RECORD, date='2026-07-20')]
        self.store.save(records)
        self.assertEqual(len(self.store.load()), 2)

    def test_save_replaces_rather_than_accumulates(self):
        self.store.save([REAL_RECORD])
        self.store.save([])
        self.assertEqual(self.store.load(), [])


class SaveParityTests(_Base):
    """与旧 save_predictions 的差分。"""

    def _legacy(self, date, results, version, initial=None):
        kv = _FakeKv(initial)
        with mock.patch.object(legacy, 'kv_store', kv), \
             mock.patch.object(legacy, 'datetime', _FrozenDatetime):
            legacy.save_predictions(date, results, version)
        return kv.data['basketball_prediction_records']

    def _compare(self, date, results, version='v1', initial=None):
        self.store.save(list(initial or []))
        self._recorder().save(date, results, version)
        self.assertEqual(self.store.load(),
                         self._legacy(date, results, version, initial))

    def test_first_save_matches_legacy(self):
        self._compare('2026-08-27', [ANALYSIS, UNAVAILABLE])

    def test_second_save_overwrites_the_same_day(self):
        self.store.save([])
        self._recorder().save('2026-08-27', [ANALYSIS], 'v1')
        first = self.store.load()
        self._recorder().save('2026-08-27', [ANALYSIS], 'v2')
        self.assertEqual(len(self.store.load()), len(first))
        self.assertEqual(self.store.load()[0]['version'], 'v2')

    def test_overwrite_matches_legacy(self):
        initial = self._legacy('2026-08-27', [ANALYSIS], 'v1')
        self._compare('2026-08-27', [ANALYSIS], 'v2', initial=initial)

    def test_different_day_appends(self):
        initial = self._legacy('2026-08-26', [ANALYSIS], 'v1')
        self._compare('2026-08-27', [ANALYSIS], 'v1', initial=initial)

    def test_empty_results_matches_legacy(self):
        self._compare('2026-08-27', [], 'v1', initial=[REAL_RECORD])


class SettledResultPreservationTests(_Base):
    """刷新推荐不能抹掉已结算的赛果——比赛已经结束，丢了不可恢复。"""

    RESULT = {'home_score': 100, 'away_score': 95, 'status': 'finished',
              'settled_at': NOW_ISO, 'elo_updated': True,
              'calibration_fed': True}

    def test_result_survives_a_refresh(self):
        self._recorder().save('2026-08-27', [ANALYSIS], 'v1')
        records = self.store.load()
        records[0]['result'] = dict(self.RESULT)
        self.store.save(records)

        self._recorder().save('2026-08-27', [ANALYSIS], 'v2')
        refreshed = self.store.load()[0]
        self.assertEqual(refreshed['result'], self.RESULT)
        self.assertEqual(refreshed['version'], 'v2', '记录本身没被刷新')

    def test_partial_schedule_does_not_drop_other_matches(self):
        """数据源常常只返回半份赛程。整段重写会把没返回的那些一起抹掉。"""
        self._recorder().save('2026-08-27', [ANALYSIS, UNAVAILABLE], 'v1')
        self._recorder().save('2026-08-27', [ANALYSIS], 'v2')
        self.assertEqual({r['match_id'] for r in self.store.load()},
                         {'2026-08-27_甲_乙', '2026-08-27_丙_丁'})


class TruncationTests(_Base):
    def test_keeps_the_newest_records(self):
        old = [dict(REAL_RECORD, date=f'2026-01-{i:02d}', match_id=f'm{i}')
               for i in range(1, 6)]
        self.store.save(old)
        recorder = self._recorder(max_records=3)
        recorder.save('2026-08-27', [ANALYSIS], 'v1')
        kept = self.store.load()
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[-1]['match_id'], '2026-08-27_甲_乙')
        self.assertEqual([r['match_id'] for r in kept[:-1]], ['m4', 'm5'])

    def test_default_cap_is_500(self):
        self.assertEqual(MAX_RECORDS, 500)


class QueryTests(_Base):
    def setUp(self):
        super().setUp()
        self.store.save([
            dict(REAL_RECORD, date='2026-07-13', match_id='a'),
            dict(REAL_RECORD, date='2026-07-14', match_id='b'),
            dict(REAL_RECORD, date='2026-07-14', match_id='c',
                 result={'home_score': 90, 'away_score': 88}),
        ])

    def test_get_filters_by_date(self):
        self.assertEqual([r['match_id'] for r in self._recorder().get('2026-07-14')],
                         ['b', 'c'])

    def test_get_returns_the_newest_within_the_limit(self):
        self.assertEqual([r['match_id'] for r in self._recorder().get(limit=2)],
                         ['b', 'c'])

    def test_unsettled_excludes_settled(self):
        self.assertEqual([r['match_id'] for r in self._recorder().unsettled()],
                         ['a', 'b'])


class EvaluateMarketsParityTests(unittest.TestCase):
    RECORDS = [
        REAL_RECORD,
        {'spf': {'available': True, 'playable': True, 'recommendation': '主胜'},
         'rqspf': {'available': True, 'playable': True, 'recommendation': '让胜',
                   'handicap': '-3.5'},
         'dx': {'available': True, 'playable': True, 'recommendation': '大分',
                'total_line': 210.5}},
        {'spf': {'available': True, 'playable': True, 'recommendation': '客胜'},
         'rqspf': {'available': True, 'playable': True, 'recommendation': '让负',
                   'handicap': '+5'},
         'dx': {'available': True, 'playable': True, 'recommendation': '小分',
                'total_line': 200}},
        {'spf': {'available': False}, 'rqspf': None,
         'dx': {'available': True, 'playable': False, 'recommendation': '大分',
                'total_line': 200}},
        {'spf': {'available': True, 'playable': True, 'recommendation': '主胜'},
         'rqspf': {'available': True, 'playable': True, 'recommendation': '让胜',
                   'handicap': '不是数字'},
         'dx': {'available': True, 'playable': True, 'recommendation': '大分',
                'total_line': None}},
    ]
    SCORES = [(100, 95), (95, 100), (100, 100), (105, 95), (103, 97),
              (100, 100), (0, 0), (110, 90)]

    def test_matches_legacy(self):
        for record in self.RECORDS:
            for home, away in self.SCORES:
                with self.subTest(rec=record.get('match_id'), score=(home, away)):
                    self.assertEqual(evaluate_markets(record, home, away),
                                     legacy._evaluate_markets(record, home, away))

    def test_push_is_void_not_a_loss(self):
        """让分正好打平、总分正好等于盘口，本就没有输赢。

        让分要整数盘才可能打平：-3.5 这样的半球盘永远分得出胜负，
        用它构造 void 是构造不出来的。
        """
        self.assertTrue(evaluate_markets(self.RECORDS[2], 100, 100)['spf_void'])

        integer_line = {'rqspf': {'available': True, 'playable': True,
                                  'recommendation': '让胜', 'handicap': '-4'},
                        'dx': {'available': True, 'playable': True,
                               'recommendation': '大分', 'total_line': 200}}
        hits = evaluate_markets(integer_line, 102, 98)
        self.assertTrue(hits['rqspf_void'], '让 4 分后正好打平应记 void')
        self.assertTrue(hits['dx_void'], '总分正好等于盘口应记 void')

    def test_unplayable_market_is_not_judged(self):
        hits = evaluate_markets(self.RECORDS[3], 110, 95)
        self.assertIsNone(hits['dx_hit'])
        self.assertFalse(hits['dx_void'])


class SummarizeParityTests(unittest.TestCase):
    def _records(self):
        base = dict(ANALYSIS)
        record = {
            'date': '2026-08-27', 'match_id': 'm1', 'league': 'NBA',
            'home': '甲', 'away': '乙',
            'spf': dict(base['spf']), 'rqspf': dict(base['rqspf']),
            'dx': dict(base['dx']),
        }
        return [
            dict(record, match_id='hit', result={'home_score': 110, 'away_score': 90}),
            dict(record, match_id='miss', result={'home_score': 90, 'away_score': 110}),
            dict(record, match_id='void', result={'home_score': 105, 'away_score': 105}),
            dict(record, match_id='open', result=None),
        ]

    def test_matches_legacy(self):
        records = self._records()
        with mock.patch.object(legacy, 'kv_store', _FakeKv(records)):
            expected = legacy.get_prediction_stats()
        self.assertEqual(summarize(records), expected)

    def test_water_inference_is_tracked_separately(self):
        """走势反推翻转过模型方向的那些单独统计——这套信号有没有用，
        只能靠它自己的命中率回答。"""
        stats = summarize(self._records())
        # 三条已结算记录的 rqspf 都带 movement_led，且让分是 -3.5 半球盘、
        # 不可能打平，所以三条全部计入
        self.assertEqual(stats['water_inference']['rqspf']['total'], 3)
        self.assertEqual(stats['water_inference']['dx']['total'], 0,
                         'dx 那条没有 movement_led，不该计入')

    def test_void_does_not_count_towards_accuracy(self):
        stats = summarize(self._records())
        self.assertEqual(stats['spf']['void'], 1)
        self.assertEqual(stats['spf']['total'], 2)


class SettleTests(_Base):
    def setUp(self):
        super().setUp()
        self._recorder().save('2026-08-27', [ANALYSIS], 'v1')

    def test_unknown_match(self):
        result = self._recorder().settle('不存在', 100, 95)
        self.assertFalse(result['ok'])
        self.assertEqual(result['error'], 'match_not_found')

    def test_settle_writes_result_and_updates_elo(self):
        result = self._recorder().settle('2026-08-27_甲_乙', 110, 90, 'NBA')
        self.assertTrue(result['ok'])
        self.assertTrue(result['result']['elo_updated'])
        self.assertEqual(self.elo.updates,
                         [('甲', '乙', 110, 90, 'NBA')])
        self.assertEqual(self.store.load()[0]['result']['home_score'], 110)

    def test_settle_feeds_the_calibrator_once(self):
        recorder = self._recorder()
        first = recorder.settle('2026-08-27_甲_乙', 110, 90)
        self.assertEqual(first['calibration_samples'], 3)
        second = recorder.settle('2026-08-27_甲_乙', 110, 90)
        self.assertEqual(second['calibration_samples'], 0, '重复喂了校准器')
        self.assertEqual(len(self.calibrator.samples), 3)

    def test_settle_updates_elo_only_once(self):
        recorder = self._recorder()
        recorder.settle('2026-08-27_甲_乙', 110, 90)
        recorder.settle('2026-08-27_甲_乙', 110, 90)
        self.assertEqual(len(self.elo.updates), 1, '重复结算把一场算成了两场')

    def test_calibrator_gets_the_picked_side_probability(self):
        """校准要问的是「我说 62% 的时候对了几成」，喂的必须是被推荐那一侧。"""
        self._recorder().settle('2026-08-27_甲_乙', 110, 90)
        by_type = {s[0]: s for s in self.calibrator.samples}
        self.assertEqual(by_type['spf'][1], 0.62)
        self.assertEqual(by_type['rqspf'][1], 0.57)
        self.assertEqual(by_type['dx'][1], 0.56)
        # 110:90 → 主胜（spf 中）、让 3.5 后仍胜（rqspf 中）、总分 200 低于
        # 盘口 210.5（dx 推大分，错）。命中与否要如实喂给校准器
        self.assertEqual({s[0]: s[2] for s in self.calibrator.samples},
                         {'spf': True, 'rqspf': True, 'dx': False})

    def test_elo_failure_does_not_block_settlement(self):
        recorder = PredictionRecorder(self.store, calibrator=self.calibrator,
                                      elo=_FakeElo(fail=True), now_fn=lambda: NOW)
        result = recorder.settle('2026-08-27_甲_乙', 110, 90)
        self.assertTrue(result['ok'])
        self.assertFalse(result['result']['elo_updated'])
        self.assertEqual(result['calibration_samples'], 3)

    def test_void_markets_are_not_fed(self):
        self._recorder().settle('2026-08-27_甲_乙', 105, 105)
        self.assertNotIn('spf', {s[0] for s in self.calibrator.samples})


class FeedCalibrationTests(_Base):
    def test_feeds_only_the_unfed(self):
        recorder = self._recorder()
        recorder.save('2026-08-27', [ANALYSIS], 'v1')
        records = self.store.load()
        records[0]['result'] = {'home_score': 110, 'away_score': 90,
                                'calibration_fed': False}
        self.store.save(records)

        self.assertEqual(recorder.feed_calibration(), 3)
        self.assertEqual(recorder.feed_calibration(), 0, '重复喂了')
        self.assertTrue(self.store.load()[0]['result']['calibration_fed'])

    def test_unsettled_records_are_skipped(self):
        recorder = self._recorder()
        recorder.save('2026-08-27', [ANALYSIS], 'v1')
        self.assertEqual(recorder.feed_calibration(), 0)


class NoLegacyImportTests(unittest.TestCase):
    def test_does_not_import_legacy_package(self):
        import ast
        import inspect

        from src.domain.sports.basketball import records

        tree = ast.parse(inspect.getsource(records))
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
