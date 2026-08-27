"""福彩 3D 的评分与排名：多窗口集成、数字评分、直选打分与排序。

这一层直接决定推荐哪几注，**算错了不会报错，只会换一组号**。参照物是从
迁移前的实现生成的黄金文件（`tests/fixtures/golden/lottery3d_scoring.json.gz`，
1700 条），语料按六种长度的历史 × 三组窗口权重 × 四组胆码杀码 × 十个三元组
铺开，并覆盖 `rank_triplets` 的每一个开关组合（含线上关着的那几个）。

**黄金值里有一处是有意与迁移前不同的**：`triplet_weight_detail` 迁移前漏了
形态先验那一项，用户看到的得分拆解加起来对不上旁边的总分（组六差 4.32 分）。
迁移时把打分与分解合成同一份实现，这个缺口随之补上。选号结果一字未变——
`triplet_w` 与全部 `rank_*` 的黄金值都与迁移前逐条相同。
"""
import gzip
import json
import pathlib
import random
import unittest

from src.domain.numeric.lottery3d import (
    digit_scoring, ranking, weights as W, windows,
)
from src.domain.numeric.lottery3d.space import DIGIT_SPACE, POSITIONS
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_scoring.json.gz')
NUMBERS = [tuple(r['digits'])
           for r in _load('numeric/lottery3d_history.json.gz')['results']]

SERIES = {'full': NUMBERS, 'recent200': NUMBERS[-200:], 'recent30': NUMBERS[-30:],
          'recent5': NUMBERS[-5:], 'tiny2': NUMBERS[-2:], 'one': NUMBERS[-1:]}
WINDOWS = (5, 20, 50, 100)
# 'default' 用迁移当时的真实默认权重（`RECENT_WINDOWS` 均分），
# 另两组是刻意构造的：单窗口与两窗口，用来暴露「多窗口合成」本身的错。
WINDOW_WEIGHTS = {'default': {30: 0.25, 45: 0.25, 60: 0.25, 90: 0.25},
                  'flat': {30: 1.0},
                  'two': {10: 0.4, 100: 0.6}}
TRIPLES = [(0, 0, 0), (9, 9, 9), (1, 2, 3), (3, 2, 1), (0, 5, 9), (4, 4, 7),
           (7, 4, 4), (5, 6, 7), (9, 0, 1), (2, 2, 8)]
DAN_KILL = [([], []), ([1, 2], []), ([], [7, 8]), ([3], [4, 5])]


def _label(digits):
    return ''.join(map(str, digits))


# 手写用例直接调领域函数，参数写死为**迁移当时 `config.py` 里生效的值**。
# 不 import 配置：那样配置一改期望值会跟着挪（判据 12）。
DECAY = 0.96
BASELINES = W.Baselines(position_repeat=0.1, digit_reuse=0.2709999999999999)
DYNAMIC_BASE = {'position_repeat': 2.0, 'last_appear': 7.0, 'consecutive': 2.0}


def build_meta(series, window_weights):
    """直接借适配层搭 meta：手写用例关心的是领域函数的语义，
    不是「meta 怎么搭」——那一段已由黄金比对覆盖。"""
    return adapter.build_ranking_meta(series, window_weights)


def _context(meta, danma=(), kill=()):
    return adapter._triplet_context(list(danma), list(kill), meta)


# ── 语料通过适配层产出 ──
#
# 黄金比对走 `src/lottery3d/scoring` 而不是直接调领域函数：迁移要保证的是
# 这些公开入口的输出不变，而「哪个配置常量喂给哪个参数」只在适配层发生，
# 绕过去就测不到。领域函数自身的语义由下面的手写用例直接覆盖。
from src.lottery3d import scoring as adapter  # noqa: E402


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    yield from _tool_entries()
    yield from _window_entries()
    yield from _digit_entries()
    yield from _ranking_entries()
    yield from _pool_entries()


def _tool_entries():
    for value in (-5, 0, 0.5, 1, 7):
        for low, high in ((0, 1), (0.2, 1.6), (-1, 1)):
            yield f'clamp:{value}:{low}:{high}', adapter._clamp(value, low, high)
    for triple in TRIPLES:
        for last in ((1, 2, 3), (0, 0, 0), (3, 2, 1)):
            yield (f'pos_repeat:{_label(triple)}:{_label(last)}',
                   adapter.position_repeat_count(triple, last))
    yield 'empty_lag1', adapter._empty_lag1()


def _window_entries():
    for name, series in SERIES.items():
        sums = [sum(t) for t in series]
        spans = [max(t) - min(t) for t in series]
        for window in WINDOWS:
            yield f'lag1:{name}:{window}', adapter.analyze_lag1_dynamics(series, window)
            yield f'patterns:{name}:{window}', adapter.analyze_patterns(series, window)
            yield (f'sum_span:{name}:{window}',
                   adapter.analyze_sum_span(sums, spans, window))
        for wname, ww in WINDOW_WEIGHTS.items():
            yield f'ens_lag1:{name}:{wname}', adapter.ensemble_lag1_dynamics(series, ww)
            pattern = adapter.ensemble_patterns(series, ww)
            yield f'ens_patterns:{name}:{wname}', pattern
            raw = adapter.ensemble_sum_span(sums, spans, ww)
            yield f'ens_sum_span:{name}:{wname}', raw
            yield (f'dynamic:{name}:{wname}',
                   adapter.derive_dynamic_weights(
                       adapter.ensemble_lag1_dynamics(series, ww),
                       pattern['consec_rate']))
            yield f'meta_from_raw:{name}:{wname}', adapter._meta_from_raw(raw)


def _digit_entries():
    for name, series in SERIES.items():
        for wname, ww in WINDOW_WEIGHTS.items():
            lag1 = adapter.ensemble_lag1_dynamics(series, ww)
            pattern = adapter.ensemble_patterns(series, ww)
            dynamic = adapter.derive_dynamic_weights(lag1, pattern['consec_rate'])
            for dyn, tag in ((None, 'nodyn'), (dynamic, 'dyn')):
                for window in WINDOWS:
                    yield (f'digit_scores:{name}:{window}:{tag}',
                           adapter.digit_scores(series, window, dyn))
                    for position in range(POSITIONS):
                        yield (f'pos_digit_scores:{name}:{window}:{position}:{tag}',
                               adapter.position_digit_scores(series, position, window, dyn))
                yield (f'ens_digit_scores:{name}:{wname}:{tag}',
                       adapter.ensemble_digit_scores(series, ww, dyn))
                for position in range(POSITIONS):
                    yield (f'ens_pos_digit:{name}:{wname}:{position}:{tag}',
                           adapter.ensemble_position_digit_scores(series, position, ww, dyn))
            yield f'zu6_digit_scores:{name}:{wname}', adapter.zu6_digit_scores(series, ww)


def _ranking_entries():
    for name in ('full', 'recent200', 'recent30'):
        series = SERIES[name]
        for wname, ww in WINDOW_WEIGHTS.items():
            meta = adapter.build_ranking_meta(series, ww)
            yield (f'ranking_meta:{name}:{wname}',
                   {k: v for k, v in meta.items() if k != 'numbers'})
            score, _ = adapter.ensemble_digit_scores(series, ww,
                                                     dynamic=meta.get('dynamic'))
            yield f'blend_dan:{name}:{wname}', adapter._blend_dan_score(score, meta)
            for danma, kill in DAN_KILL:
                tag = f'{_label(danma) or "-"}:{_label(kill) or "-"}'
                for triple in TRIPLES:
                    key = _label(triple)
                    yield (f'triplet_w:{name}:{wname}:{tag}:{key}',
                           adapter.triplet_weight(*triple, score, danma, kill, meta))
                    yield (f'triplet_detail:{name}:{wname}:{tag}:{key}',
                           adapter.triplet_weight_detail(*triple, score, danma, kill, meta))
                yield (f'digit_base:{name}:{wname}:{tag}',
                       [adapter._triplet_digit_base(*t, score, meta) for t in TRIPLES])
                yield (f'pos_pool:{name}:{wname}:{tag}',
                       adapter._position_constrained_pool(score, danma, kill, meta)[:20])
                yield from _rank_entries(name, wname, tag, score, danma, kill, meta)
                yield (f'detail_list:{name}:{wname}:{tag}',
                       adapter.build_detail_list([(1.0, '123'), (0.9, '456')],
                                                 score, danma, kill, meta))
            yield from _random_entries(name, wname, score, meta)


def _rank_entries(name, wname, tag, score, danma, kill, meta):
    off = dict(enable_exploration=False, apply_noise=False,
               enable_cold_hot_balance=False, enable_diversity=False,
               enable_correlation=False, recent_recommendations=None)
    yield (f'rank_top3:{name}:{wname}:{tag}',
           adapter.rank_triplets(score, danma, kill, meta, top_n=3, **off))
    yield (f'rank_top30:{name}:{wname}:{tag}',
           adapter.rank_triplets(score, danma, kill, meta, top_n=30,
                                 **{**off, 'enable_diversity': True}))
    for knob in ('cold_hot_balance', 'correlation', 'both_off'):
        yield (f'rank_knob:{name}:{wname}:{tag}:{knob}',
               adapter.rank_triplets(
                   score, danma, kill, meta, top_n=20,
                   **{**off,
                      'enable_cold_hot_balance': knob == 'cold_hot_balance',
                      'enable_correlation': knob == 'correlation'}))
    yield (f'rank_recent:{name}:{wname}:{tag}',
           adapter.rank_triplets(
               score, danma, kill, meta, top_n=20,
               **{**off, 'recent_recommendations': [
                   {'recommendations': ['123', '456']},
                   {'recommendations': ['123']}]}))


def _random_entries(name, wname, score, meta):
    """带随机的两条路径，固定种子。线上两个开关都是关的，但它们是活代码。"""
    off = dict(enable_cold_hot_balance=False, enable_diversity=False,
               enable_correlation=False)
    for seed in (1, 7):
        random.seed(seed)
        yield (f'rank_noise:{name}:{wname}:{seed}',
               adapter.rank_triplets(score, [], [], meta, top_n=20,
                                     enable_exploration=False, apply_noise=True, **off))
        random.seed(seed)
        yield (f'rank_explore:{name}:{wname}:{seed}',
               adapter.rank_triplets(score, [], [], meta, top_n=20,
                                     enable_exploration=True, apply_noise=False, **off))
    rank = sorted(enumerate(score), key=lambda item: -item[1])
    for seed in (1, 7):
        random.seed(seed)
        yield (f'select_danma:{name}:{wname}:{seed}',
               sorted(adapter.select_danma(rank, enable_random=True)))
    yield (f'select_danma_fixed:{name}:{wname}',
           sorted(adapter.select_danma(rank, enable_random=False)))


def _pool_entries():
    pool = [(10.0 - i * 0.1, f'{i // 100}{(i // 10) % 10}{i % 10}') for i in range(200)]
    for top_n in (5, 20, 30):
        for use_diversity in (True, False):
            for use_correlation in (True, False):
                yield (f'diverse_pool:{top_n}:{use_diversity}:{use_correlation}',
                       adapter.select_diverse_pool(pool, top_n=top_n, candidate_size=100,
                                                   use_diversity=use_diversity,
                                                   use_correlation=use_correlation))
    yield ('merge_pools',
           adapter._merge_rank_pools([(3.0, '111'), (1.0, '222')],
                                     [(2.0, '222'), (2.5, '333')], top_n=3))


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。任何一条对不上，都意味着推荐的号变了。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


class DetailConsistencyTests(unittest.TestCase):
    """得分拆解必须加得出总分。

    迁移前 `triplet_weight_detail` 漏了形态先验，组六差 4.32 分——而总分
    本身是对的，所以谁也不会发现「解释」和「结论」不是一回事。
    """

    def setUp(self):
        self.meta = build_meta(SERIES['recent200'], WINDOW_WEIGHTS['flat'])
        self.score, _ = adapter.ensemble_digit_scores(
            SERIES['recent200'], WINDOW_WEIGHTS['flat'], dynamic=self.meta['dynamic'])

    def _detail(self, triple, danma=(), kill=()):
        return ranking.detail(triple, self.score, self.meta,
                              _context(self.meta, danma, kill))

    def test_terms_sum_to_the_total(self):
        for triple in TRIPLES:
            with self.subTest(triple=triple):
                found = self._detail(triple)
                terms = {k: v for k, v in found.items() if k != 'total'}
                self.assertAlmostEqual(sum(terms.values()), found['total'], places=9)

    def test_total_equals_the_weight_used_for_ranking(self):
        """分解的总分与实际排序用的分必须是同一个数。"""
        for triple in TRIPLES:
            for danma, kill in DAN_KILL:
                with self.subTest(triple=triple, danma=danma, kill=kill):
                    context = _context(self.meta, danma, kill)
                    self.assertAlmostEqual(
                        ranking.weight(triple, self.score, self.meta, context),
                        ranking.detail(triple, self.score, self.meta, context)['total'],
                        places=9)

    def test_form_prior_is_present_and_follows_the_theoretical_rates(self):
        """组六 > 组三 > 豹子，比例来自组合数，不是拟合值。"""
        zu6 = self._detail((1, 2, 3))['form_prior']
        zu3 = self._detail((1, 1, 3))['form_prior']
        baozi = self._detail((7, 7, 7))['form_prior']
        self.assertGreater(zu6, zu3)
        self.assertGreater(zu3, baozi)
        self.assertAlmostEqual(zu6, 6.0 * 0.72, places=9)


class ContextTests(unittest.TestCase):
    """只依赖历史的量建一次、用一千次。迁移前它们在逐注循环里各算一遍。"""

    def test_history_only_terms_are_computed_once(self):
        meta = build_meta(SERIES['recent200'], WINDOW_WEIGHTS['flat'])
        calls = []
        original = ranking._term_form_switch
        try:
            ranking._term_form_switch = lambda t, c: (calls.append(1)
                                                      or original(t, c))
            context = adapter._ranking.build_context(
                meta, adapter._TRIPLET_WEIGHTS, adapter.FEATURE_FLAGS,
                form_switch={'zu6': 1.0, 'zu3': 2.0})
            # 上下文里的值是现成的，不随候选重算
            self.assertEqual(context.form_switch, {'zu6': 1.0, 'zu3': 2.0})
        finally:
            ranking._term_form_switch = original

    def test_form_switch_reads_the_precomputed_value(self):
        meta = build_meta(SERIES['recent200'], WINDOW_WEIGHTS['flat'])
        flags = {**adapter.FEATURE_FLAGS, 'form_switch': True}
        context = adapter._ranking.build_context(
            meta, adapter._TRIPLET_WEIGHTS, flags,
            form_switch={'zu6': 1.0, 'zu3': 2.0})
        self.assertEqual(ranking._term_form_switch((1, 2, 3), context), 1.0)
        self.assertEqual(ranking._term_form_switch((1, 1, 3), context), 2.0)
        self.assertEqual(ranking._term_form_switch((7, 7, 7), context), 2.0)

    def test_disabled_flag_zeroes_the_term(self):
        meta = build_meta(SERIES['recent200'], WINDOW_WEIGHTS['flat'])
        context = adapter._ranking.build_context(
            meta, adapter._TRIPLET_WEIGHTS, adapter.FEATURE_FLAGS,
            form_switch={'zu6': 1.0, 'zu3': 2.0})
        self.assertEqual(ranking._term_form_switch((1, 2, 3), context), 0.0)


class DynamicWeightTests(unittest.TestCase):
    """自适应缩放的上下界是安全带：没有它，一段偶然的高复刻会让推荐照抄上期。"""

    def _lag1(self, position_rate, reuse_rate):
        return {**windows.empty_lag1(BASELINES),
                'avg_pos_repeat': position_rate,
                'pos_repeat_rate': [position_rate] * POSITIONS,
                'digit_reuse_rate': reuse_rate}

    def test_high_repeat_raises_the_weight(self):
        low = windows.derive_dynamic_weights(self._lag1(0.05, 0.27), 0.3,
                                             DYNAMIC_BASE, BASELINES)
        high = windows.derive_dynamic_weights(self._lag1(0.30, 0.27), 0.3,
                                              DYNAMIC_BASE, BASELINES)
        self.assertGreater(high['w_pos_repeat'], low['w_pos_repeat'])

    def test_scaling_is_clamped_at_both_ends(self):
        """界限写成字面量：引用被测常量的话，把界改坏断言会跟着挪（判据 12）。"""
        base = DYNAMIC_BASE['position_repeat']
        extreme_low = windows.derive_dynamic_weights(self._lag1(0.0, 0.0), 0.3,
                                                     DYNAMIC_BASE, BASELINES)
        extreme_high = windows.derive_dynamic_weights(self._lag1(10.0, 10.0), 0.3,
                                                      DYNAMIC_BASE, BASELINES)
        self.assertAlmostEqual(extreme_low['w_pos_repeat'], base * 0.2, places=9)
        self.assertAlmostEqual(extreme_high['w_pos_repeat'], base * 1.6, places=9)
        self.assertAlmostEqual(min(extreme_low['pos_mult']), 0.3, places=9)
        self.assertAlmostEqual(max(extreme_high['pos_mult']), 2.0, places=9)

    def test_rare_full_repeat_gets_the_heaviest_penalty(self):
        """全同号越少见，罚得越重——它出现在推荐里更像模型跑偏。"""
        rare = windows.derive_dynamic_weights(
            {**windows.empty_lag1(BASELINES), 'full_repeat_rate': 0.0}, 0.3,
            DYNAMIC_BASE, BASELINES)
        common = windows.derive_dynamic_weights(
            {**windows.empty_lag1(BASELINES), 'full_repeat_rate': 0.05}, 0.3,
            DYNAMIC_BASE, BASELINES)
        self.assertGreater(rare['w_full_repeat_penalty'],
                           common['w_full_repeat_penalty'])

    def test_zero_consecutive_rate_does_not_divide_by_zero(self):
        found = windows.derive_dynamic_weights(windows.empty_lag1(BASELINES), 0.0,
                                               DYNAMIC_BASE, BASELINES)
        self.assertGreater(found['w_consecutive'], 0)


class SumSpanTests(unittest.TestCase):

    def test_centres_are_rounded_to_integers(self):
        """和值与跨度都是整数量：用分数中心去框整数容差会少框一个取值。"""
        sums, spans = [13, 14, 13, 14], [5, 6, 5, 6]
        found = windows.ensemble_sum_span(sums, spans, {4: 1.0}, DECAY, 0.0)
        self.assertEqual(found['sum_center'], float(round(found['sum_center'])))
        self.assertEqual(found['span_center'], float(round(found['span_center'])))

    def test_recent_shift_is_off_by_default_in_this_corpus(self):
        sums = [0] * 20 + [27] * 5
        spans = [0] * 25
        without = windows.analyze_sum_span(sums, spans, 25, DECAY, 0.0)
        with_shift = windows.analyze_sum_span(sums, spans, 25, DECAY, 0.5)
        self.assertGreater(with_shift['sum_center'], without['sum_center'])


class DisabledFeatureTests(unittest.TestCase):
    """线上把 `miss` 开关关了，遗漏那一整段在黄金语料里走不到。

    开关随时会打开，代码是活的——所以补用例，而不是删代码。参数全部显式给，
    正好是领域层「不读全局配置」换来的好处。
    """

    WEIGHTS = W.DigitWeights(
        hot_global=0.0, hot_position=0.0, markov=0.0, markov2=0.0,
        markov_max=99.0, markov_alpha=1.0, miss_high=1.0, miss_mid=0.5,
        last_appear=0.0, neighbor=0.0, road_match=0.0, decay=1.0)
    FLAGS = {'hot': False, 'markov': False, 'miss': True,
             'neighbor': False, 'road': False}

    def _series(self, miss_for_seven):
        """造一段：数字 7 恰好遗漏 miss_for_seven 期。"""
        return [(7, 7, 7)] + [(1, 2, 3)] * miss_for_seven

    def _score(self, miss):
        scores, _ = digit_scoring.digit_scores(
            self._series(miss), 500, self.WEIGHTS, self.FLAGS)
        return scores[7]

    def test_below_the_mid_threshold_earns_nothing(self):
        """门槛写成字面量，不引用被测常量（判据 12）。"""
        self.assertEqual(self._score(11), 0.0)

    def test_at_the_mid_threshold_earns_the_mid_score(self):
        self.assertEqual(self._score(12), 0.5)

    def test_below_the_high_threshold_still_earns_only_the_mid_score(self):
        self.assertEqual(self._score(19), 0.5)

    def test_at_the_high_threshold_switches_to_the_scaled_score(self):
        # 20 期：1.0 * (1 + 20/20) = 2.0
        self.assertEqual(self._score(20), 2.0)

    def test_the_scaled_score_grows_with_the_miss_length(self):
        # 40 期：1.0 * (1 + 40/20) = 3.0
        self.assertEqual(self._score(40), 3.0)

    def test_markov_contribution_is_capped(self):
        """封顶：样本极少的转移经平滑后仍可能给出极高的分，那是分母太小。"""
        capped = W.DigitWeights(**{**self.WEIGHTS.__dict__,
                                   'markov': 100.0, 'markov_max': 2.4})
        uncapped = W.DigitWeights(**{**self.WEIGHTS.__dict__,
                                     'markov': 100.0, 'markov_max': 1e9})
        flags = {**self.FLAGS, 'miss': False, 'markov': True}
        series = [(3, 0, 0)] * 30 + [(3, 0, 0)]
        with_cap, _ = digit_scoring.digit_scores(series, 500, capped, flags)
        without, _ = digit_scoring.digit_scores(series, 500, uncapped, flags)
        self.assertLess(max(with_cap), max(without))
        # 三个位各贡献一阶+二阶，每份都被 2.4 顶住
        self.assertLessEqual(max(with_cap), 2.4 * POSITIONS * 2 + 1e-9)


class NoiseAndCandidateTests(unittest.TestCase):
    """线上 `RANDOM_NOISE` 是 0、`top_n` 不超过 30，几个常数在语料里走不到。"""

    def setUp(self):
        self.meta = build_meta(SERIES['recent200'], WINDOW_WEIGHTS['flat'])
        self.score, _ = adapter.ensemble_digit_scores(
            SERIES['recent200'], WINDOW_WEIGHTS['flat'], dynamic=self.meta['dynamic'])

    def test_noise_only_touches_the_head_of_the_pool(self):
        """只给前 50 注加扰动：后面的注本来就进不了推荐，加了纯属浪费。"""
        pool = [(100.0 - i, f'{i:03d}') for i in range(200)]
        noisy = ranking._add_noise(pool, 0.4, random.Random(3))
        changed = [i for i, (w, num) in enumerate(sorted(noisy, key=lambda x: x[1]))
                   if abs(w - (100.0 - int(num))) > 1e-12]
        self.assertTrue(changed)
        self.assertLessEqual(max(changed), 49)

    def test_candidate_set_widens_past_the_flat_minimum(self):
        """候选集是 max(注数*5, 150)：要到 top_n>30 才由倍数说了算。"""
        seen = {}
        original = ranking.select_diverse_pool

        def spy(pool, top_n, candidate_size, *a, **k):
            seen['size'] = candidate_size
            return original(pool, top_n, candidate_size, *a, **k)

        try:
            ranking.select_diverse_pool = spy
            context = _context(self.meta)
            ranking.rank_triplets(self.score, self.meta, context, 40,
                                  enable_diversity=True,
                                  diversity={'candidate_size': 150},
                                  position_top_k=5)
        finally:
            ranking.select_diverse_pool = original
        self.assertEqual(seen['size'], 200)


class HotColdBalanceTests(unittest.TestCase):
    """冷热平衡：黄金语料只比对最终 top20，档位构成的常数在那之下看不见。"""

    def _pool(self):
        # 号码全用三位互不相同的，方便按数字归档
        return [(100.0 - i, f'{(i // 100) % 10}{(i // 10) % 10}{i % 10}')
                for i in range(1000)]

    def _hot_cold(self, **over):
        base = {'hot': [1, 2, 3], 'warm': [4, 5, 6], 'cold': [7, 8, 9],
                'hot_share': 0.4, 'warm_share': 0.4, 'cold_share': 0.2}
        base.update(over)
        return base

    def test_each_bucket_keeps_its_share(self):
        """比例写成字面量：保留 100 注时是 40/40/20。"""
        balanced = ranking._balance_hot_cold(self._pool(), 20, self._hot_cold())
        self.assertEqual(len(balanced), 100)

    def test_keep_size_follows_the_larger_of_the_two_rules(self):
        """保留数是 max(注数*4, 100)：注数 30 以下由下限说了算。"""
        self.assertEqual(len(ranking._balance_hot_cold(self._pool(), 5, self._hot_cold())), 100)
        self.assertEqual(len(ranking._balance_hot_cold(self._pool(), 50, self._hot_cold())), 200)

    def test_a_cold_pick_needs_both_a_cold_and_a_warm_digit(self):
        """只要求冷号的话，这一档会被大量「冷+热」的注挤满。"""
        context = self._hot_cold()
        pool = [(9.0, '789'), (8.0, '147'), (7.0, '123')]
        balanced = ranking._balance_hot_cold(pool, 1, context)
        # 789 全冷、没有温号 → 不算冷注；147 冷+温 → 冷注
        self.assertEqual(sorted(n for _, n in balanced), ['123', '147', '789'])
        buckets = {'hot': [], 'warm': [], 'cold': []}
        for _, number in pool:
            digits = {int(c) for c in number}
            if len(digits & {1, 2, 3}) >= 2:
                buckets['hot'].append(number)
            elif digits & {7, 8, 9} and digits & {4, 5, 6}:
                buckets['cold'].append(number)
            else:
                buckets['warm'].append(number)
        self.assertEqual(buckets['cold'], ['147'])
        self.assertEqual(buckets['warm'], ['789'])


class RecentShiftTests(unittest.TestCase):
    """近期偏移的窗口大小，只有开着偏移时才看得见。线上是关的。"""

    def test_shift_uses_exactly_the_last_five_periods(self):
        # 前 20 期和值 0，最后 5 期和值 27；只取末 5 期时偏移目标是 27
        sums = [0] * 20 + [27] * 5
        spans = [0] * 25
        found = windows.analyze_sum_span(sums, spans, 25, 1.0, 1.0)
        self.assertAlmostEqual(found['sum_center'], 27.0, places=9)

    def test_a_wider_shift_window_would_pull_the_centre_lower(self):
        """窗口若取 10 期，末 10 期里有 5 期是 0，中心会被拉到 13.5。"""
        sums = [0] * 20 + [27] * 5
        self.assertEqual(sum(sums[-5:]) / 5, 27.0)
        self.assertEqual(sum(sums[-10:]) / 10, 13.5)


class DiversePoolTests(unittest.TestCase):

    def _pool(self):
        # 用三位互不相同的号：`111` 只有一个不同数字，重合永远到不了阈值 2，
        # 拿它测去相关等于什么也没测。
        return [(10.0, '123'), (9.9, '124'), (9.5, '567'), (9.4, '890')]

    def test_diversity_prefers_new_digits_over_a_marginally_higher_score(self):
        picked = ranking.select_diverse_pool(self._pool(), 2, 10, 1.5, 3.0, 2,
                                             use_diversity=True, use_correlation=False)
        self.assertEqual([number for _, number in picked], ['123', '567'])

    def test_without_diversity_it_is_plain_score_order(self):
        picked = ranking.select_diverse_pool(self._pool(), 2, 10, 1.5, 3.0, 2,
                                             use_diversity=False, use_correlation=False)
        self.assertEqual([number for _, number in picked], ['123', '124'])

    def test_correlation_penalises_overlapping_picks(self):
        picked = ranking.select_diverse_pool(self._pool(), 2, 10, 0.0, 3.0, 2,
                                             use_diversity=False, use_correlation=True)
        self.assertEqual([number for _, number in picked], ['123', '567'])

    def test_candidate_size_bounds_what_can_be_picked(self):
        picked = ranking.select_diverse_pool(self._pool(), 4, 2, 1.5, 3.0, 2)
        self.assertEqual(len(picked), 2)


class MergePoolTests(unittest.TestCase):

    def test_first_occurrence_wins_on_duplicates(self):
        merged = ranking.merge_pools([(3.0, '111')], [(9.0, '111')], top_n=2)
        self.assertEqual(merged, [(3.0, '111')])

    def test_result_is_sorted_by_score(self):
        merged = ranking.merge_pools([(1.0, '111')], [(5.0, '222')], top_n=2)
        self.assertEqual([number for _, number in merged], ['222', '111'])


if __name__ == '__main__':
    unittest.main()
