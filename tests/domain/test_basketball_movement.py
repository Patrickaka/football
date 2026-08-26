"""赔率走势的纯计算部分迁入领域层。

主体原本是与旧实现的差分测试。旧实现删除后改为对**黄金文件**断言——
黄金值由当时已通过逐字差分的实现生成，记录的仍是迁移前的行为。
见 `tests/domain/golden.py`。

手写断言覆盖黄金文件之外的地方：注入的时钟（裁决 E）、边界值、
以及每条守卫存在的理由。
"""
import unittest
from datetime import datetime, timedelta
from unittest import mock

from src.domain.sports.basketball import movement as new
from tests.domain.golden import as_json, load

GOLDEN = load('movement')

NOW = datetime(2026, 8, 26, 12, 0, 0)


def _ts(minutes_ago):
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def _snapshot(minutes_ago, **fields):
    return {'ts': _ts(minutes_ago), **fields}


# 覆盖 flat/home/away、steam/非 steam、stale/新鲜、盘口与水位一致/冲突
SNAPSHOT_SEQS = [
    [],
    [_snapshot(60, spf_home=1.9, spf_away=1.9)],
    [_snapshot(60, spf_home=1.90, spf_away=1.90),
     _snapshot(10, spf_home=1.90, spf_away=1.90)],
    [_snapshot(60, spf_home=2.00, spf_away=1.80),
     _snapshot(5, spf_home=1.70, spf_away=2.10)],
    [_snapshot(600, spf_home=2.00, spf_away=1.80),
     _snapshot(500, spf_home=1.95, spf_away=1.85)],
    [_snapshot(90, spf_home=1.80, spf_away=2.00),
     _snapshot(45, spf_home=1.86, spf_away=1.94),
     _snapshot(3, spf_home=2.10, spf_away=1.70)],
    [_snapshot(120, spf_home=None, spf_away=1.9),
     _snapshot(30, spf_home=1.95, spf_away=1.85),
     _snapshot(2, spf_home=1.75, spf_away=2.05)],
    # 强度足够但变盘发生在很久以前：steam 要求「急」，只看强度不看新鲜度时
    # 这种陈年大幅位移会被误判成急单。未到 240 分钟，所以也不是 stale——
    # 必须靠 90 分钟这道窗口才能否掉它。
    [_snapshot(200, spf_home=2.20, spf_away=1.70),
     _snapshot(120, spf_home=1.60, spf_away=2.30)],
    # 窗口内侧的相邻值，把 90 这个边界钉死
    [_snapshot(150, spf_home=2.20, spf_away=1.70),
     _snapshot(89, spf_home=1.60, spf_away=2.30)],
]

HANDICAP_SEQS = [
    [_snapshot(120, rqspf_home=1.90, rqspf_away=1.90, handicap=-3.5),
     _snapshot(10, rqspf_home=1.80, rqspf_away=2.00, handicap=-5.5)],
    [_snapshot(120, rqspf_home=1.90, rqspf_away=1.90, handicap=-3.5),
     _snapshot(10, rqspf_home=2.05, rqspf_away=1.75, handicap=-5.5)],
    [_snapshot(300, rqspf_home=1.90, rqspf_away=1.90, handicap=2.5),
     _snapshot(290, rqspf_home=1.88, rqspf_away=1.92, handicap=2.5)],
]

TOTAL_SEQS = [
    [_snapshot(120, dx_over=1.90, dx_under=1.90, total_line=210.5),
     _snapshot(8, dx_over=1.78, dx_under=2.02, total_line=213.5)],
    [_snapshot(120, dx_over=1.90, dx_under=1.90, total_line=210.5),
     _snapshot(8, dx_over=2.02, dx_under=1.78, total_line=213.5)],
]

TRENDS = [
    None,
    {},
    {'direction': 'stable', 'strength': 0.0},
    {'direction': 'home_backing', 'strength': 0.12, 'home_move': -0.08,
     'away_move': 0.05, 'line_move': 0.0, 'samples': 4},
    {'direction': 'away_backing', 'strength': 0.26, 'home_move': 0.06,
     'away_move': -0.09, 'line_move': -1.0, 'samples': 6,
     'opening_line': -3.5, 'current_line': -4.5},
    {'direction': 'line_down', 'strength': 0.05, 'home_move': 0.0,
     'away_move': 0.0, 'line_move': -2.0, 'samples': 3,
     'opening_line': -3.5, 'current_line': -5.5},
    {'direction': 'line_up', 'strength': 0.3, 'home_move': -0.05,
     'away_move': -0.05, 'line_move': 1.5, 'samples': 5},
    {'direction': 'over_backing', 'strength': 0.4, 'home_move': -0.1,
     'away_move': 0.02, 'line_move': 2.0, 'samples': 9},
    {'direction': '未知方向', 'strength': 0.2, 'home_move': -0.05,
     'away_move': 0.03, 'line_move': 0.0, 'samples': 2},
]

KINDS = ('ml', 'ah', 'ou')


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class MovementFromSnapshotsGoldenTests(unittest.TestCase):
    """裁决 E：注入时钟是这些用例能复现的前提。"""

    def test_moneyline_sequences(self):
        for i, seq in enumerate(SNAPSHOT_SEQS):
            with self.subTest(n=len(seq)):
                self.assertEqual(
                    as_json(new.movement_from_snapshots(
                        seq, 'spf_home', 'spf_away', now_fn=lambda: NOW)),
                    GOLDEN[f'snap:ml:{i}'])

    def test_handicap_sequences(self):
        for i, seq in enumerate(HANDICAP_SEQS):
            with self.subTest(seq=seq[-1]['handicap']):
                self.assertEqual(
                    as_json(new.movement_from_snapshots(
                        seq, 'rqspf_home', 'rqspf_away', 'handicap', 'ah',
                        now_fn=lambda: NOW)),
                    GOLDEN[f'snap:ah:{i}'])

    def test_total_sequences(self):
        for i, seq in enumerate(TOTAL_SEQS):
            with self.subTest(over=seq[-1]['dx_over']):
                self.assertEqual(
                    as_json(new.movement_from_snapshots(
                        seq, 'dx_over', 'dx_under', 'total_line', 'ou',
                        now_fn=lambda: NOW)),
                    GOLDEN[f'snap:ou:{i}'])

    def test_now_fn_defaults_to_wall_clock(self):
        """默认不注入时，仍要按真实时间算——注入只是可选的测试接缝。"""
        fresh = [_snapshot(0, spf_home=2.0, spf_away=1.8),
                 _snapshot(0, spf_home=1.7, spf_away=2.1)]
        with mock.patch.object(new, 'datetime', _FrozenDatetime):
            result = new.movement_from_snapshots(fresh, 'spf_home', 'spf_away')
        self.assertIsNotNone(result)
        self.assertFalse(result['stale'])

    def test_window_min_tracks_injected_clock(self):
        """时钟真的被用上了：把 now 往后推，窗口宽度必须跟着变。"""
        seq = [_snapshot(30, spf_home=2.0, spf_away=1.8),
               _snapshot(10, spf_home=1.7, spf_away=2.1)]
        early = new.movement_from_snapshots(
            seq, 'spf_home', 'spf_away', now_fn=lambda: NOW)
        late = new.movement_from_snapshots(
            seq, 'spf_home', 'spf_away', now_fn=lambda: NOW + timedelta(minutes=60))
        self.assertAlmostEqual(late['window_min'] - early['window_min'], 60.0)
        self.assertAlmostEqual(
            late['last_move_age_min'] - early['last_move_age_min'], 60.0)


class NormalizeTrendGoldenTests(unittest.TestCase):
    def test_all_trend_shapes(self):
        for i, trend in enumerate(TRENDS):
            for kind in KINDS:
                with self.subTest(direction=(trend or {}).get('direction'), kind=kind):
                    self.assertEqual(as_json(new.normalize_okooo_trend(trend, kind)),
                                     GOLDEN[f'trend:{i}:{kind}'])


def _movements():
    out = [None, {}, {'available': False}]
    for trend in TRENDS:
        for kind in KINDS:
            movement = new.normalize_okooo_trend(trend, kind)
            if movement:
                out.append(movement)
    out.append({'available': True, 'side': 'home', 'strength': 0.9,
                'samples': 8, 'stale': True, 'steam': True,
                'signal_conflict': False, 'water_side': 'home',
                'line_side': 'home', 'signal_agreement': True,
                'line_move': -1.0, 'home_move': -0.1, 'away_move': 0.05,
                'opening_line': -3.5, 'current_line': -4.5})
    out.append({'available': True, 'side': 'over', 'strength': 0.7,
                'samples': 1, 'stale': False, 'steam': False,
                'signal_conflict': True, 'water_side': 'over',
                'line_side': 'under', 'signal_agreement': False,
                'line_move': 1.0, 'home_move': -0.1, 'away_move': 0.05})
    # 其余否决条件全部通过、只差强度——INFERENCE_MIN_STRENGTH 是唯一
    # 能否决它的门槛，缺了这一条那个阈值改成 0 都测不出来。
    for strength in (0.0, 0.1, 0.2499, 0.25):
        out.append({'available': True, 'side': 'home', 'strength': strength,
                    'samples': 5, 'stale': False, 'steam': False,
                    'signal_conflict': False, 'water_side': 'home',
                    'line_side': 'flat', 'signal_agreement': False,
                    'line_move': 0.0, 'home_move': -0.05, 'away_move': 0.02})
        out.append({'available': True, 'side': 'under', 'strength': strength,
                    'samples': 5, 'stale': False, 'steam': False,
                    'signal_conflict': False, 'water_side': 'under',
                    'line_side': 'flat', 'signal_agreement': False,
                    'line_move': 0.0, 'home_move': 0.02, 'away_move': -0.05})
    return out


class InferenceGoldenTests(unittest.TestCase):
    def test_infer(self):
        for i, movement in enumerate(_movements()):
            for market in ('rqspf', 'dx', 'spf'):
                with self.subTest(side=(movement or {}).get('side'), market=market):
                    self.assertEqual(
                        as_json(new.infer_market_from_movement(movement, market)),
                        GOLDEN[f'infer:{i}:{market}'])

    def test_apply_market_inference(self):
        for i, movement in enumerate(_movements()):
            for market in ('rqspf', 'dx'):
                for p in (0.35, 0.5, 0.58, 0.72):
                    with self.subTest(side=(movement or {}).get('side'),
                                      market=market, p=p):
                        self.assertEqual(
                            as_json(new.apply_market_inference(
                                p, 1 - p, movement, market)),
                            GOLDEN[f'apply_inf:{i}:{market}:{p}'])


class ApplyMovementGoldenTests(unittest.TestCase):
    def test_adjust_two_way_by_trend(self):
        for i, trend in enumerate(TRENDS):
            for kind in KINDS:
                t = dict(trend, kind=kind) if trend else trend
                for p in (0.4, 0.5, 0.65):
                    with self.subTest(direction=(trend or {}).get('direction'),
                                      kind=kind, p=p):
                        self.assertEqual(
                            as_json(new.adjust_two_way_by_trend(p, 1 - p, t)),
                            GOLDEN[f'adjust:{i}:{kind}:{p}'])

    def test_apply_movement(self):
        for i, movement in enumerate(_movements()):
            for p in (0.42, 0.5, 0.61):
                with self.subTest(side=(movement or {}).get('side'), p=p):
                    self.assertEqual(
                        as_json(new.apply_movement(p, 1 - p, movement)),
                        GOLDEN[f'apply_mv:{i}:{p}'])

    def test_movement_to_trend(self):
        for i, movement in enumerate(_movements()):
            with self.subTest(side=(movement or {}).get('side')):
                self.assertEqual(as_json(new.movement_to_trend(movement)),
                                 GOLDEN[f'to_trend:{i}'])


class SharpConfirmationGoldenTests(unittest.TestCase):
    RECOMMENDATIONS = ('主胜', '客胜', '让胜', '让负', '大分', '小分', '说不好', None)

    def test_all_recommendations(self):
        for i, movement in enumerate(_movements()):
            for rec in self.RECOMMENDATIONS:
                with self.subTest(side=(movement or {}).get('side'), rec=rec):
                    self.assertEqual(as_json(new.sharp_confirmation(movement, rec)),
                                     GOLDEN[f'sharp:{i}:{rec}'])


class BuildMovementForMatchGoldenTests(unittest.TestCase):
    HISTORY = {'m1': SNAPSHOT_SEQS[5], 'm2': HANDICAP_SEQS[0]}

    MATCHES = [
        {'id': 'm1', 'source': '500'},
        {'id': 'm2', 'source': '500'},
        {'id': 'm3', 'source': '500'},
        {'id': 'm1', 'source': 'okooo', 'rf_trend': TRENDS[4], 'dx_trend': TRENDS[7]},
        {'id': 'm9', 'source': 'okooo', 'rf_trend': None, 'dx_trend': None},
    ]

    BUNDLES = [
        None,
        {},
        {'ml': {'available': True, 'trend': TRENDS[3]},
         'ah': {'available': True, 'trend': TRENDS[5]},
         'ou': {'available': False}},
    ]

    def test_every_source_combination(self):
        for mi, match in enumerate(self.MATCHES):
            for bi, bundle in enumerate(self.BUNDLES):
                for hi, history in enumerate((None, {}, self.HISTORY)):
                    with self.subTest(mid=match['id'], src=match['source'],
                                      bundle=bool(bundle), hist=bool(history)):
                        self.assertEqual(
                            as_json(new.build_movement_for_match(
                                match, history=history, okooo_bundle=bundle,
                                now_fn=lambda: NOW)),
                            GOLDEN[f'build:{mi}:{bi}:{hi}'])


class DescribeMarketMovementGoldenTests(unittest.TestCase):
    @staticmethod
    def cases():
        movements = _movements()
        cases = [({}, {}), (None, None)]
        for movement in movements[3:9]:
            cases.append((
                {'spf': movement, 'rqspf': movements[4], 'dx': None},
                {'spf': {'line_movement': {'confirmed': True}},
                 'rqspf': {'line_movement': {'confirmed': False}}},
            ))
            cases.append(({'spf': movement}, {}))
            cases.append(({'dx': movement}, {'dx': {'line_movement': None}}))
        return cases

    def test_all_combinations(self):
        for i, (movements, bets) in enumerate(self.cases()):
            with self.subTest(keys=sorted((movements or {}))):
                self.assertEqual(
                    as_json(new.describe_market_movement(movements, bets)),
                    GOLDEN[f'describe:{i}'])


class NoLegacyImportTests(unittest.TestCase):
    """迁移的意义在于反向依赖被切断，这条断言是它的守卫。

    查 import 语句而不是查源码文本——文档里提到旧模块的名字是正常的，
    真正不该出现的是对它的依赖。
    """

    def _imported_modules(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(new))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(('.' * (node.level or 0)) + (node.module or ''))
        return modules

    def test_domain_movement_does_not_import_legacy_package(self):
        for module in self._imported_modules():
            self.assertFalse(module.startswith('src.basketball'), module)
            self.assertFalse(module.startswith('.'), f'领域层不该有相对导入: {module}')


if __name__ == '__main__':
    unittest.main()
