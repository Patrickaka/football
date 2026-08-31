"""为一天的比赛构建走势映射——2-5 里 `movement_provider` 注入点的真实实现。

它要在三个来源之间做取舍：中国足彩网赛程自带的 rf_trend / dx_trend、
源站详情页的各家共识、以及自己攒的 500 源快照序列。取舍规则本身是领域知识，采集与抓取
则一律注入。

500 源那条路上还夹着一次「按队名跨源匹配」：两边对同一场比赛用不同的主键，
只能靠队名对上，而队名写法不同（源站带 `[西2]` 这类排名标记）。
"""
import unittest
from datetime import datetime
from unittest import mock

from src.domain.sports.basketball.movement_map import MovementMapBuilder, normalize_team
from tests.domain.golden import as_json, load

GOLDEN = load('movement_map')

NOW = datetime(2026, 8, 26, 12, 0, 0)

FIVE_HUNDRED_MATCHES = [
    {'id': '2026-08-27_金州女武神_太阳', 'home': '金州女武神', 'away': '太阳',
     'league': 'WNBA', 'source': '500'},
    {'id': '2026-08-27_多伦多节奏_风暴', 'home': '多伦多节奏', 'away': '风暴',
     'league': 'WNBA', 'source': '500'},
    {'id': '2026-08-27_无对应_球队', 'home': '无对应', 'away': '球队',
     'league': 'CBA', 'source': '500'},
]

ZGZCW_MATCHES = [
    {'id': '5381400', 'home': '金州女武神[西2]', 'away': '太阳', 'source': 'zgzcw',
     'rf_trend': {'direction': 'home_backing', 'strength': 0.12,
                  'home_move': -0.08, 'away_move': 0.05, 'line_move': 0.0,
                  'samples': 4},
     'dx_trend': {'direction': 'over_backing', 'strength': 0.3,
                  'home_move': -0.06, 'away_move': 0.04, 'line_move': 1.0,
                  'samples': 5}},
    {'id': '5381881', 'home': '多伦多节奏', 'away': '风暴[东1]', 'source': 'zgzcw',
     'rf_trend': None, 'dx_trend': None},
]

BUNDLES = {
    '5381400': {'ml': {'available': True,
                       'trend': {'direction': 'away_backing', 'strength': 0.25,
                                 'home_move': 0.04, 'away_move': -0.07,
                                 'line_move': 0.0, 'samples': 6}},
                'ah': {'available': False}, 'ou': {'available': False}},
    '5381881': {'ml': {'available': False}, 'ah': {'available': False},
                'ou': {'available': False}},
}

SNAPSHOTS = {
    '2026-08-27_金州女武神_太阳': [
        {'ts': '2026-08-26T09:00:00', 'spf_home': 2.00, 'spf_away': 1.80,
         'rqspf_home': 1.90, 'rqspf_away': 1.90, 'dx_over': 1.85,
         'dx_under': 1.95, 'handicap': '-3.5', 'total_line': 210.5},
        {'ts': '2026-08-26T11:30:00', 'spf_home': 1.70, 'spf_away': 2.10,
         'rqspf_home': 1.80, 'rqspf_away': 2.00, 'dx_over': 1.95,
         'dx_under': 1.85, 'handicap': '-5.5', 'total_line': 213.5},
    ],
    '2026-08-27_无对应_球队': [
        {'ts': '2026-08-26T09:00:00', 'spf_home': 1.90, 'spf_away': 1.90,
         'rqspf_home': 1.90, 'rqspf_away': 1.90, 'dx_over': 1.90,
         'dx_under': 1.90, 'handicap': '+1.5', 'total_line': 190.5},
        {'ts': '2026-08-26T11:00:00', 'spf_home': 1.60, 'spf_away': 2.20,
         'rqspf_home': 1.70, 'rqspf_away': 2.10, 'dx_over': 1.80,
         'dx_under': 2.00, 'handicap': '+1.5', 'total_line': 190.5},
    ],
}


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class _Deps:
    """把新旧两侧要用到的外部依赖收在一处，保证两边喂的完全一样。"""

    def __init__(self, zgzcw_matches=None, bundles=None, snapshots=None,
                 zgzcw_raises=False, bundle_raises=False, track_raises=False):
        self.zgzcw_matches = ZGZCW_MATCHES if zgzcw_matches is None else zgzcw_matches
        self.bundles = BUNDLES if bundles is None else bundles
        self.snapshots = SNAPSHOTS if snapshots is None else snapshots
        self.zgzcw_raises = zgzcw_raises
        self.bundle_raises = bundle_raises
        self.track_raises = track_raises
        self.tracked = []

    def zgzcw_schedule(self, date=None):
        if self.zgzcw_raises:
            raise IOError('中国足彩网不可用')
        return [dict(m) for m in self.zgzcw_matches]

    def fetch_bundles(self, ids):
        if self.bundle_raises:
            raise IOError('详情页被 WAF 拦了')
        return {str(i): self.bundles.get(str(i), {}) for i in ids}

    def track(self, date=None):
        if self.track_raises:
            raise IOError('采集失败')
        self.tracked.append(date)
        return len(self.snapshots)

    def load_history(self):
        return {k: [dict(s) for s in v] for k, v in self.snapshots.items()}

    def builder(self):
        store = mock.Mock()
        store.load.side_effect = self.load_history
        return MovementMapBuilder(
            history_store=store,
            tracker=mock.Mock(track=self.track),
            zgzcw_schedule=self.zgzcw_schedule,
            bundle_fetcher=mock.Mock(fetch_many=self.fetch_bundles),
            now_fn=lambda: NOW)

class GoldenTests(unittest.TestCase):
    DATE = '2026-08-27'
    # (黄金键, 比赛列表, 数据源, 依赖配置)
    CASES = [
        ('zgzcw', ZGZCW_MATCHES, 'zgzcw', {}),
        ('500', FIVE_HUNDRED_MATCHES, '500', {}),
        ('500_no_zgzcw', FIVE_HUNDRED_MATCHES, '500', {'zgzcw_raises': True}),
        ('500_no_snapshots', FIVE_HUNDRED_MATCHES, '500', {'snapshots': {}}),
        ('zgzcw_blocked_details', ZGZCW_MATCHES, 'zgzcw', {'bundle_raises': True}),
        ('500_track_failed', FIVE_HUNDRED_MATCHES, '500', {'track_raises': True}),
        ('empty_500', [], '500', {}),
        ('empty_zgzcw', [], 'zgzcw', {}),
    ]

    def test_every_source_combination(self):
        for name, matches, source, kwargs in self.CASES:
            with self.subTest(case=name):
                actual = _Deps(**kwargs).builder()(matches, source, self.DATE)
                self.assertEqual(as_json(actual), GOLDEN[name])


class BehaviourTests(unittest.TestCase):
    DATE = '2026-08-27'

    def test_500_source_captures_a_fresh_snapshot_first(self):
        """500 源没有现成的走势，必须先采一轮再算——否则一整天里第一个
        请求永远看不到最新的那次变盘。"""
        deps = _Deps()
        deps.builder()(FIVE_HUNDRED_MATCHES, '500', self.DATE)
        self.assertEqual(deps.tracked, [self.DATE])

    def test_zgzcw_source_does_not_capture(self):
        """源站自带盘路历史，不需要我们自己攒。"""
        deps = _Deps()
        deps.builder()(ZGZCW_MATCHES, 'zgzcw', self.DATE)
        self.assertEqual(deps.tracked, [])

    def test_cross_source_matching_ignores_ranking_tags(self):
        """源站队名带 `[西2]` 这类排名标记，500 源没有。不剥掉就一场也对不上。"""
        self.assertEqual(normalize_team('金州女武神[西2]'), '金州女武神')
        self.assertEqual(normalize_team('  风暴 '), '风暴')
        self.assertEqual(normalize_team(None), '')

    def test_zgzcw_enrichment_reaches_the_500_match(self):
        result = _Deps().builder()(FIVE_HUNDRED_MATCHES, '500', self.DATE)
        enriched = result['2026-08-27_金州女武神_太阳']
        self.assertTrue(enriched['rqspf']['available'],
                        '按队名匹配到的源站走势没有落到 500 场次上')

    def test_unmatched_500_match_falls_back_to_snapshots(self):
        """源站没有的场次，退回自己攒的快照序列。"""
        result = _Deps().builder()(FIVE_HUNDRED_MATCHES, '500', self.DATE)
        fallback = result['2026-08-27_无对应_球队']
        self.assertTrue(fallback['spf']['available'])
        self.assertEqual(fallback['spf']['side'], 'home')

    def test_snapshots_fill_gaps_left_by_zgzcw(self):
        """澳客给不出胜负那一路时，用自己的快照补上。

        详情页被 WAF 拦是线上常态，所以这条补齐路径才是线上真正在走的那条
        ——用带 ml 的 bundle 去测，走的是「本来就有」而不是「补上了」。
        """
        blocked = {mid: {'ml': {'available': False}, 'ah': {'available': False},
                         'ou': {'available': False}} for mid in BUNDLES}
        result = _Deps(bundles=blocked).builder()(
            FIVE_HUNDRED_MATCHES, '500', self.DATE)
        matched = result['2026-08-27_金州女武神_太阳']
        self.assertTrue(matched['rqspf']['available'], '让分该来自澳客赛程页')
        self.assertTrue(matched['spf']['available'], 'spf 没有被快照补齐')
        self.assertEqual(matched['spf']['kind'], 'ml')

    def test_cross_source_matching_needs_both_teams(self):
        """只比主队会把同一支主队的两场比赛混为一谈——同一天主客双赛
        并不罕见（分区赛会、连续两日同一主场）。"""
        # 目标场次的对手是风暴。把它放在前面、另一场放后面：只比主队时
        # 两场会落到同一个键上，后写的那场覆盖前一场，于是拿到错的那份走势。
        zgzcw = [
            dict(ZGZCW_MATCHES[0], id='B', home='金州女武神[西2]', away='风暴',
                 rf_trend={'direction': 'away_backing', 'strength': 0.9,
                           'home_move': 0.2, 'away_move': -0.2,
                           'line_move': 0.0, 'samples': 9}),
            dict(ZGZCW_MATCHES[0], id='A', home='金州女武神[西2]', away='太阳'),
        ]
        matches = [dict(FIVE_HUNDRED_MATCHES[0], home='金州女武神', away='风暴',
                        id='2026-08-27_金州女武神_风暴')]
        result = _Deps(zgzcw_matches=zgzcw).builder()(matches, '500', self.DATE)
        movement = result['2026-08-27_金州女武神_风暴']
        self.assertEqual(movement['rqspf']['side'], 'away',
                         '匹配到了同一主队的另一场比赛')

    def test_detail_failure_keeps_the_schedule_trends(self):
        """**与旧实现刻意不同的一处。**

        旧实现把「取澳客赛程」和「取详情页」裹在同一个 try 里，详情页一旦
        失败就整段返回空，连赛程页自带的 rf_trend / dx_trend 也一起丢掉，
        退回粒度粗得多的快照序列。而详情页被 WAF 拦正是线上常态，赛程页
        的走势又是线上唯一活着的来源——为一个拿不到的东西丢掉唯一拿得到的，
        方向反了。这里改成分别处理：详情页失败只损失胜负那一路。

        真实的 bundle_fetcher 其实不会抛（逐页、逐场都吞了异常），所以这条
        差异只在注入了会抛的实现时才显现，线上行为不受影响。
        """
        result = _Deps(bundle_raises=True).builder()(
            FIVE_HUNDRED_MATCHES, '500', self.DATE)
        matched = result['2026-08-27_金州女武神_太阳']
        self.assertTrue(matched['rqspf']['available'],
                        '详情页失败把赛程页的让分走势也一起丢了')
        self.assertTrue(matched['dx']['available'])

    def test_missing_dependencies_degrade_to_empty(self):
        """三个依赖都不给时不该报错——走势是增强项。"""
        builder = MovementMapBuilder(history_store=None, tracker=None,
                                     zgzcw_schedule=None, bundle_fetcher=None,
                                     now_fn=lambda: NOW)
        result = builder(FIVE_HUNDRED_MATCHES, '500', self.DATE)
        self.assertEqual(set(result), {m['id'] for m in FIVE_HUNDRED_MATCHES})
        self.assertTrue(all(v == {'spf': None, 'rqspf': None, 'dx': None}
                            for v in result.values()))


class NoLegacyImportTests(unittest.TestCase):
    def test_does_not_import_legacy_package(self):
        import ast
        import inspect

        from src.domain.sports.basketball import movement_map

        tree = ast.parse(inspect.getsource(movement_map))
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
