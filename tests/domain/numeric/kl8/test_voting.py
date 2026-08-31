"""kl8 的多模型投票管道。

投票决定了最终选出哪些号码，而**它算错了不会报错，只会换一组号**。所以
参照物是从迁移前的实现生成的黄金文件（`tests/fixtures/golden/kl8_voting.json.gz`，
3 段历史 × 51 组参数），覆盖十一种最终选池模式、三种重号方向、三种频率模式、
六档重号上限，以及无信号、空池、超大池这些边界。

黄金比对走 `KL8Analyzer.multi_model_voting` 而不是直接调 `voting.vote`：
迁移要保证的是这个公开入口的输出一字不变，而权重缺省时回落到全局活跃权重
这件事也只在入口这层发生——绕过它就测不到。

另有一组手写用例守住领域层自己的语义：票数怎么从名次换算、
无信号的两个前提各自独立、未知模型权重非零时必须报错而不是静默忽略。
"""
import gzip
import json
import pathlib
import unittest
import unittest.mock

from src.domain.numeric.kl8 import voting
from src.kl8 import analyzer as analyzer_module
from src.kl8.analyzer import KL8Analyzer
from src.kl8.config import KL8_PREDICTOR_VERSION
from tests.domain.golden import as_json

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/kl8_voting.json.gz')
HISTORY = _load('numeric/kl8_history.json.gz')['results']

FW = {
    'freq_only': {'frequency': 1.0},
    'reference': {'frequency': 0.45, 'gap': 0.20, 'trend': 0.20,
                  'pair_cooccurrence': 0.10, 'position_residual': 0.05},
    'broad': {'frequency': 0.3, 'gap': 0.3, 'position_residual': 0.2,
              'position_residual_cross': 0.2, 'road_residual': 0.2, 'repeat': 0.3,
              'odd_even': 0.1, 'big_small': 0.1, 'trend': 0.3, 'adjacent': 0.2,
              'pair_cooccurrence': 0.2, 'next_transition': 0.3, 'seeded_random': 0.05},
    'all_zero': {'frequency': 0.0, 'gap': 0.0},
}

# `bayesian`/`markov` 两个 0 是历史形状：策略试验表里 23564 行都带着它们。
# 保留在语料里，是为了守住「权重为零的已删模型不该让预测失败」。
MW = {
    'rank_only': {'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0},
    'rank_half': {'rank': 0.5, 'bayesian': 0.0, 'markov': 0.0},
    'empty': {},
    'all_zero': {'rank': 0.0, 'bayesian': 0.0, 'markov': 0.0},
}

MODES = ('top_ranked', 'concentrated', 'high_tier_chase', 'balanced', 'diversified',
         'low_repeat', 'repeat_follow', 'zone_spread', 'prize_floor',
         'shape_balanced', 'best_variant')

BASE = dict(pick_n=5, top_n=20, repeat_direction='neutral',
            repeat_avoid_score=0.10, repeat_non_avoid_score=0.85,
            repeat_follow_score=0.90, repeat_non_follow_score=0.50,
            pool_diversify=True, pool_max_last_numbers=None,
            frequency_mode='mean_reversion', final_selection_mode='balanced')

CASES = {}


def _case(key, **over):
    CASES[key] = dict(BASE, **over)


for _name in FW:
    _case(f'fw:{_name}', feature_weights=FW[_name])
for _name in MW:
    _case(f'mw:{_name}', model_weights=MW[_name])
for _mode in MODES:
    _case(f'mode:{_mode}', final_selection_mode=_mode)
for _d in ('neutral', 'avoid', 'follow'):
    _case(f'dir:{_d}', repeat_direction=_d)
for _m in ('neutral', 'hot', 'mean_reversion'):
    _case(f'freqmode:{_m}', frequency_mode=_m)
for _cap in (None, 0, 1, 3, 20, 99):
    _case(f'cap:{_cap}', pool_max_last_numbers=_cap)
_case('diversify:off', pool_diversify=False)
_case('diversify:off_cap3', pool_diversify=False, pool_max_last_numbers=3)
for _pick, _top in ((1, 1), (5, 5), (7, 7), (6, 10), (20, 20), (0, 20), (5, 0), (25, 40)):
    _case(f'size:{_pick}x{_top}', pick_n=_pick, top_n=_top)
_case('scores:avoid', repeat_direction='avoid', repeat_avoid_score=0.05,
      repeat_non_avoid_score=0.95)
_case('scores:follow', repeat_direction='follow', repeat_follow_score=0.99,
      repeat_non_follow_score=0.11)
for _mode in ('balanced', 'best_variant', 'prize_floor', 'shape_balanced'):
    for _cap in (0, 2):
        _case(f'x:{_mode}:cap{_cap}', final_selection_mode=_mode,
              pool_max_last_numbers=_cap)

SLICES = {'full': HISTORY, 'recent60': HISTORY[:60], 'tiny5': HISTORY[:5]}


def build_analyzer(records):
    """按回测里的老办法造分析器：绕开文件加载，直接喂历史。"""
    analyzer = KL8Analyzer.__new__(KL8Analyzer)
    analyzer.history_data = sorted(records, key=lambda r: r['issue'], reverse=True)
    analyzer.using_simulated_data = False
    analyzer.history_file = ''
    analyzer._data_mtime = 0
    analyzer.update_statistics()
    return analyzer


def statistics_with(**overrides):
    """真实统计量，按需改几个键。

    评分要用到十几个统计量，手搓一个只填 `last_numbers` 的字典会在评分里
    直接 KeyError——测的就不是投票了。
    """
    stats = dict(build_analyzer(HISTORY[:30]).statistics)
    stats.update(overrides)
    return stats


def passthrough_shaper():
    """整形不做事，只把候选原样截断——让断言落在投票自己的行为上。"""
    return voting.PoolShaper(
        diversify=lambda cands, size, last, max_last_numbers=None: cands[:size],
        select_final=lambda cands, size, last, max_last_numbers=None,
        selection_mode='balanced': (cands[:size], selection_mode))


def run_case(analyzer, config):
    kwargs = dict(config)
    kwargs.setdefault('feature_weights', FW['reference'])
    kwargs.setdefault('model_weights', MW['rank_only'])
    return analyzer.multi_model_voting(**kwargs)


def without_version(payload):
    """黄金比对不看 `version`——它由 `VersionIsReportedTests` 单独守。

    版本串曾经在这份黄金里逐字参与比对，于是 `KL8_PREDICTOR_VERSION` 一升
    就是 153 条一起红，而版本变化**并不意味着选号变了**。红得没有分辨力，
    人就会习惯性重新生成黄金——那才是真正的风险：真回归也会被一起盖掉。
    """
    return {key: value for key, value in payload.items() if key != 'version'}


class VotingGoldenTests(unittest.TestCase):
    """迁移前后逐条比对。任何一条对不上，都意味着选号变了。"""

    def test_matches_golden(self):
        for slice_name, records in SLICES.items():
            analyzer = build_analyzer(records)
            for key, config in CASES.items():
                golden_key = f'vote:{slice_name}:{key}'
                with self.subTest(case=golden_key):
                    self.assertEqual(without_version(as_json(run_case(analyzer, config))),
                                     GOLDEN[golden_key])


class VersionIsReportedTests(unittest.TestCase):
    """`version` 从黄金比对里摘出来之后，改由这里守。

    要守的是两件事：字段还在，且它是**常量透传**而不是某处写死的字面量。
    版本号本身写进断言就等于把上面那个坑搬个地方复现。
    """

    def test_every_case_reports_the_current_version(self):
        for slice_name, records in SLICES.items():
            analyzer = build_analyzer(records)
            for key, config in CASES.items():
                with self.subTest(case=f'vote:{slice_name}:{key}'):
                    result = run_case(analyzer, config)
                    self.assertEqual(result.get('version'), KL8_PREDICTOR_VERSION)

    def test_the_version_is_threaded_through_rather_than_hardcoded(self):
        """把常量换掉，输出就得跟着换——否则说明某处写死了字面量。"""
        with unittest.mock.patch.object(analyzer_module, 'KL8_PREDICTOR_VERSION',
                                        'kl8-vTEST-sentinel'):
            result = run_case(build_analyzer(SLICES['full']), dict(BASE))
        self.assertEqual(result['version'], 'kl8-vTEST-sentinel')


class ModelWeightTests(unittest.TestCase):

    def setUp(self):
        self.statistics = statistics_with()
        self.shaper = passthrough_shaper()

    def _vote(self, model_weights, feature_weights=None):
        return voting.vote(
            self.statistics, feature_weights or {'frequency': 1.0},
            model_weights, self.shaper, version='test')

    def test_zero_weight_for_deleted_model_is_tolerated(self):
        """23564 条历史试验的权重字典都带着这两个 0，不能因此拒绝出号。"""
        result = self._vote({'rank': 1.0, 'bayesian': 0.0, 'markov': 0.0})
        self.assertNotIn('status', result)
        self.assertTrue(result['selected'])

    def test_nonzero_weight_for_deleted_model_raises(self):
        """静默忽略会按剩下的模型出一组号，而结果看上去毫无异常。"""
        for name in ('bayesian', 'markov', 'whatever'):
            with self.subTest(model=name):
                with self.assertRaises(ValueError) as caught:
                    self._vote({'rank': 1.0, name: 0.3})
                self.assertIn(name, str(caught.exception))

    def test_error_names_every_unknown_model_not_just_the_first(self):
        with self.assertRaises(ValueError) as caught:
            self._vote({'rank': 1.0, 'bayesian': 0.5, 'markov': 0.5})
        self.assertIn('bayesian', str(caught.exception))
        self.assertIn('markov', str(caught.exception))


class NoSignalTests(unittest.TestCase):
    """无信号的两个前提各自独立——只测一个的话，另一个接反了也发现不了。"""

    def setUp(self):
        self.statistics = statistics_with(last_numbers=set())
        self.shaper = passthrough_shaper()

    def _vote(self, feature_weights, model_weights):
        return voting.vote(self.statistics, feature_weights, model_weights,
                           self.shaper, version='v-test')

    def test_model_weight_alone_is_not_a_signal(self):
        """只有模型权重时，排名退化成「按号码大小排序」，那不是预测。"""
        result = self._vote({'frequency': 0.0}, {'rank': 1.0})
        self.assertEqual(result['status'], voting.NO_SIGNAL_STATUS)
        self.assertEqual(result['selected'], [])

    def test_feature_weight_alone_is_not_a_signal(self):
        result = self._vote({'frequency': 1.0}, {'rank': 0.0})
        self.assertEqual(result['status'], voting.NO_SIGNAL_STATUS)
        self.assertEqual(result['selected'], [])

    def test_both_weights_present_gives_a_signal(self):
        result = self._vote({'frequency': 1.0}, {'rank': 1.0})
        self.assertNotIn('status', result)

    def test_no_signal_result_carries_the_version(self):
        result = self._vote({'frequency': 0.0}, {'rank': 0.0})
        self.assertEqual(result['version'], 'v-test')


class TallyTests(unittest.TestCase):
    """名次换票数。这条曲线变了，候选顺序就变了。"""

    def test_first_place_gets_the_full_weight(self):
        self.assertAlmostEqual(voting._tally([7, 8, 9, 10], 1.0)[7], 1.0)

    def test_votes_decrease_linearly_with_rank(self):
        tally = voting._tally([1, 2, 3, 4], 1.0)
        self.assertEqual([tally[n] for n in (1, 2, 3, 4)], [1.0, 0.75, 0.5, 0.25])

    def test_last_place_still_gets_a_nonzero_vote(self):
        """末位归零的话，它与「根本没进排名」就分不出来了。"""
        tally = voting._tally([1, 2, 3, 4], 1.0)
        self.assertGreater(tally[4], 0)

    def test_weight_scales_every_vote(self):
        full = voting._tally([1, 2, 3, 4], 1.0)
        half = voting._tally([1, 2, 3, 4], 0.5)
        self.assertEqual({n: v / 2 for n, v in full.items()}, half)

    def test_empty_ranking_yields_no_votes(self):
        self.assertEqual(voting._tally([], 1.0), {})


class CandidateOrderTests(unittest.TestCase):
    """候选顺序。整条管道跑不出并列（单模型的票数两两不同），所以直接喂票数。"""

    def test_more_votes_comes_first(self):
        ordered = voting._order_candidates({7: 0.2, 3: 0.9, 5: 0.5})
        self.assertEqual([num for num, _ in ordered], [3, 5, 7])

    def test_ties_break_by_number_ascending(self):
        """并列的打破方式必须可复现，否则同一份输入两次运行会给出不同的号。"""
        ordered = voting._order_candidates({9: 0.5, 2: 0.5, 40: 0.5})
        self.assertEqual([num for num, _ in ordered], [2, 9, 40])

    def test_votes_outrank_numbers(self):
        """号码小但票少的不该插队——否则排名就白算了。"""
        ordered = voting._order_candidates({1: 0.1, 80: 0.9})
        self.assertEqual([num for num, _ in ordered], [80, 1])


class PoolShapingTests(unittest.TestCase):
    """整形是注入进来的，投票必须把参数原样递过去。"""

    def setUp(self):
        self.calls = []
        self.statistics = statistics_with(last_numbers={3, 5})

        def diversify(cands, size, last, max_last_numbers=None):
            self.calls.append(('diversify', size, last, max_last_numbers))
            return cands[:size]

        def select_final(cands, size, last, max_last_numbers=None,
                         selection_mode='balanced'):
            self.calls.append(('select_final', size, last, max_last_numbers,
                               selection_mode))
            return cands[:size], f'resolved:{selection_mode}'

        self.shaper = voting.PoolShaper(diversify, select_final)

    def _vote(self, **over):
        return voting.vote(self.statistics, {'frequency': 1.0}, {'rank': 1.0},
                           self.shaper, version='v', **over)

    def test_diversify_receives_last_numbers_and_cap(self):
        self._vote(top_n=12, pool_max_last_numbers=4)
        self.assertEqual(self.calls[0], ('diversify', 12, {3, 5}, 4))

    def test_pool_never_shrinks_below_the_minimum(self):
        """比 7 还小的池子没有整形余地，复式玩法也拿不到足够的号。"""
        self._vote(top_n=2)
        self.assertEqual(self.calls[0][1], voting.MIN_POOL_SIZE)

    def test_diversify_is_skipped_when_turned_off(self):
        result = self._vote(pool_diversify=False)
        self.assertEqual([kind for kind, *_ in self.calls], ['select_final'])
        self.assertFalse(result['diversified'])

    def test_reports_the_mode_the_shaper_actually_resolved(self):
        """best_variant 会挑一个具体模式，报回请求值等于隐去了实际用的那个。"""
        result = self._vote(final_selection_mode='best_variant')
        self.assertEqual(result['final_selection_mode'], 'resolved:best_variant')

    def test_select_final_receives_pick_n_not_pool_size(self):
        self._vote(pick_n=5, top_n=20)
        self.assertEqual(self.calls[1][1], 5)


class RankingWidthTests(unittest.TestCase):
    """投票用的排名比请求的池子宽，整形才有得挑。"""

    def setUp(self):
        self.shaper = passthrough_shaper()

    def _raw_count(self, top_n):
        result = voting.vote(statistics_with(), {'frequency': 1.0}, {'rank': 1.0},
                             self.shaper, version='v', top_n=top_n)
        return result['raw_candidate_count']

    def test_narrow_request_still_ranks_the_minimum_width(self):
        self.assertEqual(self._raw_count(5), voting.MIN_MODEL_TOP_N)

    def test_wide_request_widens_the_ranking(self):
        self.assertEqual(self._raw_count(60), 60)

    def test_ranking_never_exceeds_the_number_space(self):
        """要多宽都超不出 1..80——这条由号码空间兜着，投票不必再夹一层。"""
        from src.domain.numeric.kl8 import scoring
        self.assertEqual(self._raw_count(500), scoring.SPACE.size)


if __name__ == '__main__':
    unittest.main()
