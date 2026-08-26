"""预测流程迁入领域层并接上两层缓存。

本批的收益点。`/api/basketball` 系列端点此前**零缓存**（webapp/caching.py 里
没有 basketball 的任何条目），每个请求都要重跑一遍抓取加分析。

三类测试：
- **差分**：编排逻辑与旧的 generate_basketball_recommendations 逐字相等。
- **缓存**：并发冷启动只算一次（单飞），且 key 能区分参数。
- **契约**：payload 必须是纯 JSON（裁决 C）——否则它永远写不进 Redis，
  每次冷启动都要重算，缓存等于没接。
"""
import json
import threading
import time
import unittest
from unittest import mock

import src.basketball as legacy
from src.basketball import calibration as legacy_calibration
from src.basketball import elo as legacy_elo
from src.basketball import okooo as legacy_okooo
from src.basketball import records as legacy_records
from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.prediction import (
    PredictionService, find_value_bets,
)
from src.foundation.cache import Cache, MemoryBackend

DATE = '2026-08-27'

MATCHES = [
    {'id': 'a1', 'home': '湖人', 'away': '凯尔特人', 'league': 'NBA',
     'time': '09:00', 'status': 'not_started',
     'spf_home': 1.75, 'spf_away': 2.10, 'rqspf_home': 1.90, 'rqspf_away': 1.90,
     'handicap': -3.5, 'dx_over': 1.85, 'dx_under': 1.95, 'total_line': 221.5},
    {'id': 'a2', 'home': '广东', 'away': '辽宁', 'league': 'CBA',
     'time': '19:35', 'status': 'in_progress',
     'spf_home': 2.40, 'spf_away': 1.55, 'rqspf_home': 2.05, 'rqspf_away': 1.78,
     'handicap': 4.5, 'dx_over': 2.00, 'dx_under': 1.80, 'total_line': 185.5},
    {'id': 'a3', 'home': 'A 队', 'away': 'B 队', 'league': '没见过的联赛',
     'time': '02:00', 'status': 'not_started', 'spf_home': 1.90},
]

MOVEMENT_MAP = {
    'a1': {
        'spf': {'available': True, 'side': 'home', 'strength': 0.55, 'samples': 6,
                'stale': False, 'steam': False, 'signal_conflict': False,
                'water_side': 'home', 'line_side': 'flat', 'signal_agreement': False,
                'line_move': 0.0, 'home_move': -0.12, 'away_move': 0.06, 'kind': 'ml'},
        'rqspf': {'available': True, 'side': 'away', 'strength': 0.8, 'samples': 9,
                  'stale': False, 'steam': True, 'signal_conflict': False,
                  'water_side': 'away', 'line_side': 'away', 'signal_agreement': True,
                  'line_move': 1.5, 'home_move': 0.08, 'away_move': -0.15,
                  'kind': 'ah'},
        'dx': None,
    },
    'a2': {'spf': None, 'rqspf': None,
           'dx': {'available': True, 'side': 'over', 'strength': 0.5, 'samples': 6,
                  'stale': False, 'steam': False, 'signal_conflict': False,
                  'water_side': 'over', 'line_side': 'over', 'signal_agreement': True,
                  'line_move': 2.0, 'home_move': -0.1, 'away_move': 0.05,
                  'kind': 'ou', 'opening_line': 210.5, 'current_line': 212.5}},
}

VERSION = legacy.BASKETBALL_VERSION
HISTORY_STATS = {'total': 12, 'hit': 7}


class FakeElo:
    def predict_win_prob(self, home, away, league):
        return {'home_prob': 0.6, 'away_prob': 0.4,
                'home_games': 30, 'away_games': 25}

    def predict_margin(self, home, away, league):
        return {'expected_margin': 3.5}

    def predict_total_score(self, home, away, league):
        return {'expected_total': 218.0}


class FakeCalibrator:
    def calibrate(self, bet_type, predicted_prob, league, confidence):
        return predicted_prob * 1.03


class RecordingRecorder:
    def __init__(self):
        self.saved = []

    def save(self, date, results, version):
        self.saved.append((date, len(results), version))

    def stats(self):
        return dict(HISTORY_STATS)


ELO = FakeElo()
CALIBRATOR = FakeCalibrator()


def _analyzer():
    return BasketballAnalyzer(elo=ELO, calibrator=CALIBRATOR)


def _service(cache=None, matches=None, movement_map=None, recorder=None,
             okooo_matches=None, delay=0.0):
    def schedule_500(date):
        if delay:
            time.sleep(delay)
        return list(MATCHES if matches is None else matches)

    def schedule_okooo(date):
        if okooo_matches is None:
            raise RuntimeError('澳客不可用')
        return list(okooo_matches)

    return PredictionService(
        analyzer=_analyzer(),
        schedule_sources={'500': schedule_500, 'okooo': schedule_okooo},
        movement_provider=(lambda ms, src, d:
                           dict(MOVEMENT_MAP if movement_map is None else movement_map)),
        recorder=recorder,
        cache=cache,
        today_fn=lambda: DATE,
    )


class _LegacyPatch:
    """让旧实现吃到与新实现完全相同的外部输入。"""

    def __init__(self, matches=None, movement_map=None, okooo_matches=None):
        self.matches = MATCHES if matches is None else matches
        self.movement_map = MOVEMENT_MAP if movement_map is None else movement_map
        self.okooo_matches = okooo_matches
        self.saved = []

    def __enter__(self):
        def okooo_schedule(date):
            if self.okooo_matches is None:
                raise RuntimeError('澳客不可用')
            return list(self.okooo_matches)

        self._patches = [
            mock.patch.object(legacy, 'fetch_basketball_schedule',
                              lambda date=None: list(self.matches)),
            mock.patch.object(legacy, '_build_movement_map',
                              lambda ms, src, d: dict(self.movement_map)),
            mock.patch.object(legacy_okooo, 'fetch_okooo_basketball_schedule',
                              okooo_schedule),
            mock.patch.object(legacy_records, 'save_predictions',
                              lambda d, r, v: self.saved.append((d, len(r), v))),
            mock.patch.object(legacy_records, 'get_prediction_stats',
                              lambda: dict(HISTORY_STATS)),
            mock.patch.object(legacy.time, 'strftime', lambda fmt: DATE),
            # 旧实现在函数体里取 Elo/校准器的全局单例，那个单例会真的读库。
            # 不打桩的话两侧吃到的不是同一份输入，差分就无从谈起。
            mock.patch.object(legacy_elo, 'get_elo_system', lambda: ELO),
            mock.patch.object(legacy_calibration, 'get_calibrator',
                              lambda: CALIBRATOR),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()
        return False


class GenerateParityTests(unittest.TestCase):
    CASES = [
        {},
        {'date': DATE},
        {'bet_types': ['spf']},
        {'bet_types': ['rqspf', 'dx']},
        {'bet_types': []},
        {'use_movement': False},
        {'source': 'okooo'},
        {'source': '未知源'},
    ]

    def test_payload_matches_legacy(self):
        for case in self.CASES:
            with self.subTest(**case):
                recorder = RecordingRecorder()
                with _LegacyPatch() as patched:
                    expected = legacy.generate_basketball_recommendations(**case)
                actual = _service(recorder=recorder).generate(**case)
                self.assertEqual(actual, expected)
                self.assertEqual(recorder.saved, patched.saved)

    def test_okooo_source_used_when_available(self):
        okooo_matches = [dict(MATCHES[0], id='ok1', source='okooo')]
        with _LegacyPatch(okooo_matches=okooo_matches):
            expected = legacy.generate_basketball_recommendations(source='okooo')
        actual = _service(okooo_matches=okooo_matches,
                          recorder=RecordingRecorder()).generate(source='okooo')
        self.assertEqual(actual, expected)
        self.assertEqual(actual['results'][0]['match']['id'], 'ok1')

    def test_empty_schedule_matches_legacy(self):
        with _LegacyPatch(matches=[]) as patched:
            expected = legacy.generate_basketball_recommendations()
        recorder = RecordingRecorder()
        actual = _service(matches=[], recorder=recorder).generate()
        self.assertEqual(actual, expected)
        self.assertEqual(recorder.saved, patched.saved)
        self.assertEqual(recorder.saved, [], '无比赛时不该写入预测记录')

    def test_movement_provider_failure_falls_back_to_no_movement(self):
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: list(MATCHES)},
            movement_provider=mock.Mock(side_effect=RuntimeError('走势源挂了')),
            recorder=RecordingRecorder(), today_fn=lambda: DATE)
        payload = service.generate()
        self.assertEqual(payload['count'], len(MATCHES))
        self.assertEqual(payload['movement_stats']['with_movement'], 0)

    def test_recorder_failure_does_not_fail_the_request(self):
        broken = mock.Mock()
        broken.save.side_effect = RuntimeError('记录库挂了')
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: list(MATCHES)},
            recorder=broken, today_fn=lambda: DATE)
        payload = service.generate()
        self.assertEqual(payload['count'], len(MATCHES))
        self.assertNotIn('history_stats', payload)


class FindValueBetsParityTests(unittest.TestCase):
    def _results(self):
        with _LegacyPatch():
            return legacy.generate_basketball_recommendations()['results']

    def test_matches_legacy(self):
        results = self._results()
        for threshold in (-1.0, 0.0, 0.01, 0.05, 0.2, 0.9):
            with self.subTest(threshold=threshold):
                self.assertEqual(find_value_bets(results, threshold),
                                 legacy.find_value_bets(results, threshold))

    def test_empty_results(self):
        self.assertEqual(find_value_bets([]), [])


class JsonContractTests(unittest.TestCase):
    """裁决 C：进缓存的值必须是纯 JSON。

    不满足时 RedisBackend 的 json.dumps 会失败，而 Cache.set 吞掉 L2 的写入
    错误——表现不是报错，而是这个 key 永远进不了 L2、每次冷启动都要重算。
    缓存看着接上了，收益是零。
    """

    def test_payload_survives_json_round_trip(self):
        payload = _service(recorder=RecordingRecorder()).generate()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_value_bets_survive_json_round_trip(self):
        results = _service(recorder=RecordingRecorder()).generate()['results']
        bets = find_value_bets(results, threshold=-1.0)
        self.assertTrue(bets)
        self.assertEqual(json.loads(json.dumps(bets)), bets)


class CacheTests(unittest.TestCase):
    def _cache(self):
        return Cache(l1=MemoryBackend(), l2=MemoryBackend(), default_ttl=120)

    def test_second_call_does_not_recompute(self):
        calls = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: calls.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), cache=self._cache(), today_fn=lambda: DATE)
        first = service.generate()
        second = service.generate()
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_concurrent_cold_start_computes_once(self):
        """单飞的判据：并发冷启动的总墙钟 ≈ 一次计算耗时，而不是 N 倍。"""
        calls = []
        barrier = threading.Barrier(5)

        def schedule(date):
            calls.append(date)
            time.sleep(0.3)
            return list(MATCHES)

        service = PredictionService(
            analyzer=_analyzer(), schedule_sources={'500': schedule},
            recorder=RecordingRecorder(), cache=self._cache(), today_fn=lambda: DATE)

        payloads = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            payload = service.generate()
            with lock:
                payloads.append(payload)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.time() - start

        self.assertEqual(len(calls), 1, '并发冷启动重复计算了，单飞没生效')
        self.assertEqual(len(payloads), 5)
        self.assertTrue(all(p == payloads[0] for p in payloads))
        self.assertLess(elapsed, 0.9, f'总墙钟 {elapsed:.2f}s 接近串行，单飞没生效')

    def test_cache_key_distinguishes_every_parameter(self):
        seen = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: seen.append(date) or list(MATCHES),
                              'okooo': lambda date: seen.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), cache=self._cache(), today_fn=lambda: DATE)
        variants = [
            {}, {'date': '2026-09-01'}, {'source': 'okooo'},
            {'bet_types': ['spf']}, {'use_movement': False},
        ]
        for variant in variants:
            service.generate(**variant)
        self.assertEqual(len(seen), len(variants),
                         '不同参数命中了同一个缓存 key')

    def test_bet_type_order_does_not_split_the_cache(self):
        calls = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: calls.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), cache=self._cache(), today_fn=lambda: DATE)
        service.generate(bet_types=['spf', 'dx'])
        service.generate(bet_types=['dx', 'spf'])
        self.assertEqual(len(calls), 1, '同一组玩法的不同顺序不该算两次')

    def test_today_resolved_before_key_is_built(self):
        """date=None 必须先解析成具体日期，否则跨天时旧结果会一直被命中。"""
        today = ['2026-08-27']
        calls = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: calls.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), cache=self._cache(),
            today_fn=lambda: today[0])
        service.generate()
        today[0] = '2026-08-28'
        payload = service.generate()
        self.assertEqual(calls, ['2026-08-27', '2026-08-28'])
        self.assertEqual(payload['date'], '2026-08-28')

    def test_without_cache_every_call_computes(self):
        calls = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda date: calls.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), today_fn=lambda: DATE)
        service.generate()
        service.generate()
        self.assertEqual(len(calls), 2)


class NoLegacyImportTests(unittest.TestCase):
    def test_prediction_does_not_import_legacy_package(self):
        import ast
        import inspect

        from src.domain.sports.basketball import prediction

        tree = ast.parse(inspect.getsource(prediction))
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
