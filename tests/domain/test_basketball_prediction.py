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

from src.domain.sports.basketball.analysis import BasketballAnalyzer
from src.domain.sports.basketball.prediction import (
    PREDICTION_VERSION, PredictionService, find_value_bets,
)
from src.foundation.cache import Cache, MemoryBackend
from tests.domain.golden import as_json, load

GOLDEN = load('prediction')

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


def without_version(payload):
    """黄金比对不看 `version`——它由 `VersionIsReportedTests` 单独守。

    版本串逐字参与比对的话，`PREDICTION_VERSION` 一升就是 9 条一起红，
    而版本变化并不意味着预测变了。红得没有分辨力，人就会习惯性重新生成
    黄金，真回归也跟着被盖掉。
    """
    return {key: value for key, value in payload.items() if key != 'version'}


def _analyzer():
    return BasketballAnalyzer(elo=ELO, calibrator=CALIBRATOR)


def _service(cache=None, matches=None, movement_map=None, recorder=None,
             zgzcw_matches=None, delay=0.0, version=None):
    def schedule_500(date):
        if delay:
            time.sleep(delay)
        return list(MATCHES if matches is None else matches)

    def schedule_zgzcw(date):
        if zgzcw_matches is None:
            raise RuntimeError('中国足彩网不可用')
        return list(zgzcw_matches)

    return PredictionService(
        analyzer=_analyzer(),
        schedule_sources={'500': schedule_500, 'zgzcw': schedule_zgzcw},
        movement_provider=(lambda ms, src, d:
                           dict(MOVEMENT_MAP if movement_map is None else movement_map)),
        recorder=recorder,
        cache=cache,
        today_fn=lambda: DATE,
        **({'version': version} if version is not None else {}),
    )


class GenerateGoldenTests(unittest.TestCase):
    CASES = [
        {},
        {'date': DATE},
        {'bet_types': ['spf']},
        {'bet_types': ['rqspf', 'dx']},
        {'bet_types': []},
        {'use_movement': False},
        {'source': 'zgzcw'},
        {'source': '未知源'},
    ]

    def test_payload(self):
        for i, case in enumerate(self.CASES):
            with self.subTest(**case):
                actual = _service(recorder=RecordingRecorder()).generate(**case)
                self.assertEqual(without_version(as_json(actual)),
                                 GOLDEN[f'payload:{i}'])

    def test_zgzcw_source_used_when_available(self):
        zgzcw_matches = [dict(MATCHES[0], id='ok1', source='zgzcw')]
        actual = _service(zgzcw_matches=zgzcw_matches,
                          recorder=RecordingRecorder()).generate(source='zgzcw')
        self.assertEqual(actual['results'][0]['match']['id'], 'ok1')

    def test_empty_schedule(self):
        recorder = RecordingRecorder()
        actual = _service(matches=[], recorder=recorder).generate()
        self.assertEqual(without_version(as_json(actual)), GOLDEN['payload:empty'])
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


class VersionIsReportedTests(unittest.TestCase):
    """`version` 从黄金比对里摘出来之后，改由这里守。

    要守的是三件事：字段还在、它是构造参数透传而非某处写死的字面量、
    落库那条路径和 payload 用的是同一个版本。版本号本身写进断言就等于
    把上面那个坑搬个地方复现。
    """

    def test_every_case_reports_the_current_version(self):
        for case in GenerateGoldenTests.CASES:
            with self.subTest(**case):
                payload = _service(recorder=RecordingRecorder()).generate(**case)
                self.assertEqual(payload['version'], PREDICTION_VERSION)

    def test_the_version_is_threaded_through_rather_than_hardcoded(self):
        """换个版本进去，输出就得跟着换——否则说明某处写死了字面量。"""
        payload = _service(recorder=RecordingRecorder(),
                           version='vTEST-sentinel').generate()
        self.assertEqual(payload['version'], 'vTEST-sentinel')

    def test_the_recorded_version_matches_the_payload(self):
        """落库与接口返回必须是同一个版本。

        对不上的话，回头按版本筛历史预测会漏掉或错配——而两处各自看都是对的。
        """
        recorder = RecordingRecorder()
        payload = _service(recorder=recorder, version='vTEST-sentinel').generate()
        self.assertEqual([version for _, _, version in recorder.saved],
                         ['vTEST-sentinel'])
        self.assertEqual(recorder.saved[0][2], payload['version'])


class FetchScheduleTests(unittest.TestCase):
    """赛程列表端点：只取赛程、不做分析、不走缓存。"""

    def test_returns_the_raw_schedule(self):
        service = _service(recorder=RecordingRecorder())
        self.assertEqual(service.fetch_schedule(DATE), MATCHES)

    def test_defaults_to_today(self):
        seen = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda d: seen.append(d) or list(MATCHES)},
            today_fn=lambda: DATE)
        service.fetch_schedule()
        self.assertEqual(seen, [DATE])

    def test_does_not_use_the_cache(self):
        """赛程比整份 payload 便宜得多，缓存只会让刚开售的场次晚几分钟出现。"""
        calls = []
        service = PredictionService(
            analyzer=_analyzer(),
            schedule_sources={'500': lambda d: calls.append(d) or list(MATCHES)},
            cache=Cache(l1=MemoryBackend(), l2=MemoryBackend()),
            today_fn=lambda: DATE)
        service.fetch_schedule(DATE)
        service.fetch_schedule(DATE)
        self.assertEqual(len(calls), 2)

    def test_source_falls_back_like_generate(self):
        service = _service(recorder=RecordingRecorder())
        self.assertEqual(service.fetch_schedule(DATE, source='zgzcw'), MATCHES)


class FindValueBetsGoldenTests(unittest.TestCase):
    def _results(self):
        return _service(recorder=RecordingRecorder()).generate()['results']

    def test_every_threshold(self):
        results = self._results()
        for threshold in (-1.0, 0.0, 0.01, 0.05, 0.2, 0.9):
            with self.subTest(threshold=threshold):
                self.assertEqual(as_json(find_value_bets(results, threshold)),
                                 GOLDEN[f'value:{threshold}'])

    def test_empty_results(self):
        self.assertEqual(find_value_bets([]), [])

    def test_edge_exactly_at_threshold_is_excluded(self):
        """边际恰好等于阈值时不入选。真实数据几乎不会正好撞上，
        所以这条边界只能构造——不构造的话把 `<=` 写成 `<` 也测不出来。

        概率取 0.75 而不是 0.55：0.55-0.5 在二进制下是 0.05000000000000004，
        压根不等于阈值，构造出来的边界是假的。0.75-0.5 才是精确的 0.25。
        """
        results = [_value_bet_result('t1', spf_prob=0.75)]
        self.assertEqual(find_value_bets(results, 0.25), [])
        self.assertEqual(len(find_value_bets(results, 0.2499)), 1)

    def test_sharp_confirmation_outranks_a_larger_bare_edge(self):
        """聪明钱确认的增益只用于排序。差距小于增益时，被确认的那注要排前面
        ——排序键少了 movement_edge 就会退化成纯按边际排。"""
        results = [
            _value_bet_result('t2', spf_prob=0.60, sharp=False),
            _value_bet_result('t3', spf_prob=0.58, sharp=True),
        ]
        bets = find_value_bets(results, 0.05)
        self.assertEqual([b['match'] for b in bets],
                         ['t3 主 vs t3 客', 't2 主 vs t2 客'])


def _value_bet_result(tag, spf_prob, sharp=False, playable=True):
    """构造一条只含胜负盘的分析结果，概率与聪明钱标记可控。"""
    return {
        'match': {'home': f'{tag} 主', 'away': f'{tag} 客', 'league': 'NBA',
                  'handicap': -3.5, 'total_line': 220.5},
        'spf': {
            'available': True, 'playable': playable, 'official': playable,
            'home_prob': spf_prob, 'away_prob': round(1 - spf_prob, 4),
            'recommendation': '主胜', 'confidence': 'high',
            'sharp_confirmed': sharp, 'line_movement': None,
        },
        'rqspf': None,
        'dx': None,
    }


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
                              'zgzcw': lambda date: seen.append(date) or list(MATCHES)},
            recorder=RecordingRecorder(), cache=self._cache(), today_fn=lambda: DATE)
        variants = [
            {}, {'date': '2026-09-01'}, {'source': 'zgzcw'},
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
