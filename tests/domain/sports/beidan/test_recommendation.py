"""北单四种玩法的推荐组装。

参照物是从迁移前的 `recommending.py` 生成的黄金文件
（`tests/fixtures/golden/beidan_recommendation.json.gz`，547 条），**逐条相同**。

两处外部依赖在黄金里用打桩替掉，**打在被测路径之外**：`fetch_ouzhi_odds`
是网络，`apply_beidan_history_calibration` 要读已结算历史。校准的桩给了两种
——一种什么也不做、一种真的改动概率。只喂恒等桩的话，「校准的结果在胜平负
那一路被后面覆盖掉了」这件事在黄金里根本看不出来。

**测试直接用适配层真实的 `_MODEL` / `_MARKET`**：它们两组一共十八个操作
全是纯计算（AST 查过，唯一读时钟的 `_beidan_market_snapshot` 不在里面），
所以不必造假，跑的就是线上那条管线。
"""
import ast
import gzip
import json
import pathlib
import unittest
from unittest import mock

from src.beidan import recommending as adapter
from src.domain.sports.beidan import recommendation
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'
GOLDEN = json.load(gzip.open(FIXTURES / 'golden/beidan_recommendation.json.gz',
                             'rt', encoding='utf-8'))

MODEL, MARKET = adapter._MODEL, adapter._MARKET

# 迁移当时生效的那组常量，写死不 import（判据 4、12）
BIFEN_WEIGHTS = (0.6, 0.4)
ZJQ_WEIGHTS = (0.55, 0.45)
FALLBACK_1X2 = {'胜': 0.33, '平': 0.33, '负': 0.34}
OVER_25_BUCKETS = ('3', '4', '5', '6', '7+')
TOP_KEPT = 3

MATCH = {'id': '1320957', 'num': '1', 'home': '安山小绿人', 'away': '大邱FC',
         'league': '英超', 'time': '18:30', 'handicap': '(-1)'}
OUZHI = {'home': 1.80, 'draw': 3.60, 'away': 4.20}
GOALS = {'history': [{'ts': '2026-08-28T09:00:00', 'line': '2.5',
                      'over_odds': 0.95, 'under_odds': 0.90},
                     {'ts': '2026-08-28T10:00:00', 'line': '2.5',
                      'over_odds': 0.80, 'under_odds': 1.05}]}
ASIAN = {'history': [{'ts': '2026-08-28T09:00:00', 'handicap': '-0.5',
                      'home_odds': 0.95, 'away_odds': 0.90},
                     {'ts': '2026-08-28T10:00:00', 'handicap': '-0.75',
                      'home_odds': 0.88, 'away_odds': 0.98}]}


def golden_entries():
    from scripts.gen_beidan_recommendation_golden import entries
    return entries()


def _tilt(probabilities, bet_type, league=None):
    """真的改动概率的校准桩：把首档抬高一成再归一。"""
    if not probabilities:
        return probabilities, {'applied': False, 'reason': 'empty'}
    first = next(iter(probabilities))
    adjusted = dict(probabilities)
    adjusted[first] = float(adjusted[first] or 0.0) * 1.10
    total = sum(adjusted.values())
    return ({key: value / total for key, value in adjusted.items()},
            {'applied': True, 'reason': 'stub'})


def spf(match=None, ouzhi=OUZHI, **kwargs):
    return recommendation.spf(match or MATCH, ouzhi, MODEL, MARKET, **kwargs)


def rqspf(match=None, ouzhi=OUZHI, handicap_value=-1.0, **kwargs):
    return recommendation.rqspf(match or MATCH, ouzhi, handicap_value,
                                MODEL, MARKET, **kwargs)


def bifen(match=None, ouzhi=OUZHI, **kwargs):
    return recommendation.bifen(match or MATCH, ouzhi, MODEL, MARKET, **kwargs)


def zjq(match=None, ouzhi=OUZHI, **kwargs):
    return recommendation.zjq(match or MATCH, ouzhi, MODEL, MARKET, **kwargs)


class GoldenTests(unittest.TestCase):

    def test_matches_golden(self):
        for key, value in golden_entries():
            with self.subTest(key=key):
                self.assertIn(key, GOLDEN)
                self.assertEqual(GOLDEN[key], as_comparable(value))


class EuroOddsSourceTests(unittest.TestCase):
    """欧赔的两个来源，与「三个都要在」那道门槛。"""

    FALLBACK = dict(MATCH, spf_sp=2.10, spf_s=3.30, spf_f=3.60)

    def test_fetched_odds_win_over_the_schedule(self):
        result = spf(self.FALLBACK)
        self.assertNotIn('odds_source', result)
        self.assertEqual(result['odds']['胜'], OUZHI['home'])

    def test_schedule_odds_are_used_when_nothing_was_fetched(self):
        result = spf(self.FALLBACK, ouzhi=None)
        self.assertEqual(result['odds_source'], 'okooo_main')
        self.assertEqual(result['odds']['胜'], 2.10)

    def test_a_partial_schedule_quote_is_refused_entirely(self):
        """**缺一个就整组不要**：补一个默认赔率会让去水后的概率悄悄偏向它。"""
        for missing in ('spf_sp', 'spf_s', 'spf_f'):
            with self.subTest(missing=missing):
                match = {k: v for k, v in self.FALLBACK.items() if k != missing}
                self.assertEqual(spf(match, ouzhi=None)['error'], '欧赔数据不可用')

    def test_every_bet_type_reports_its_own_error_text(self):
        """四条流水线的错误文案不同——混成一句就分不清是哪一路断的。"""
        self.assertEqual(spf(ouzhi=None)['error'], '欧赔数据不可用')
        self.assertEqual(rqspf(ouzhi=None)['error'],
                         '欧赔数据不可用，无法计算让球胜平负')
        self.assertEqual(bifen(ouzhi=None)['error'], '欧赔数据不可用，无法计算比分')
        self.assertEqual(zjq(ouzhi=None)['error'], '欧赔数据不可用，无法计算总进球')

    def test_dead_odds_leave_spf_without_probabilities(self):
        """去水后一档不剩时，胜平负**不写 error 也不写 odds**——
        只有表头。原样保留自迁移前（判据 17 的形状）。"""
        result = spf(ouzhi={'home': 0, 'draw': None, 'away': -1.0})
        self.assertNotIn('error', result)
        self.assertNotIn('odds', result)
        self.assertNotIn('probabilities', result)
        self.assertEqual(result['type'], 'spf')

    def test_dead_odds_make_rqspf_report_an_error(self):
        """同样的输入，让球那一路**是写 error 的**——两条路的处置不一样。"""
        self.assertEqual(rqspf(ouzhi={'home': 0, 'draw': None, 'away': -1.0})['error'],
                         '欧赔概率不可用，无法计算让球胜平负')


class MarginalConsistencyTests(unittest.TestCase):
    """胜平负、比分、总进球必须出自同一张表。"""

    def test_spf_probabilities_are_the_score_matrix_marginals(self):
        result = spf()
        matrix = {(h, a): p for h, a, p in result['score_probs']}
        self.assertAlmostEqual(
            result['probabilities']['胜'],
            sum(p for (h, a), p in matrix.items() if h > a), places=5)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=5)

    def test_prediction_is_the_argmax_of_those_marginals(self):
        result = spf()
        self.assertEqual(result['prediction'],
                         max(result['probabilities'], key=result['probabilities'].get))
        self.assertEqual(result['confidence'],
                         result['probabilities'][result['prediction']])

    def test_zjq_buckets_sum_to_one(self):
        result = zjq()
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_over25_counts_three_goals_and_up(self):
        result = zjq()
        expected = sum(result['probabilities'].get(bucket, 0)
                       for bucket in OVER_25_BUCKETS)
        self.assertAlmostEqual(result['over25_prob'], expected)
        self.assertAlmostEqual(result['under25_prob'], 1 - expected)

    def test_over_buckets_are_a_parameter(self):
        result = zjq(over_buckets=('0',))
        self.assertAlmostEqual(result['over25_prob'],
                               result['probabilities'].get('0', 0))


class CalibrationVisibilityTests(unittest.TestCase):
    """**同一个校准在两条路上的可见度不同**——这是实现使然，不是设计。"""

    def test_spf_overwrites_the_calibrated_probabilities(self):
        """胜平负的校准结果被矩阵边际覆盖掉了，只通过「改了模型的输入」
        间接起作用。所以最终概率不等于校准后的那一版。"""
        flat = spf(calibrate=recommendation.identity_calibration)
        tilted = spf(calibrate=_tilt)
        self.assertTrue(tilted['history_calibration']['applied'])
        # 校准确实起了作用（比分分布被换了一组输入）
        self.assertNotEqual(flat['probabilities'], tilted['probabilities'])
        # 但最终概率不是校准后的那一版——它是矩阵的边际
        matrix = {(h, a): p for h, a, p in tilted['score_probs']}
        self.assertAlmostEqual(
            tilted['probabilities']['胜'],
            sum(p for (h, a), p in matrix.items() if h > a), places=5)

    def test_rqspf_keeps_the_calibrated_probabilities(self):
        """让球那一路的校准结果直接留在输出里——与胜平负相反。"""
        flat = rqspf(calibrate=recommendation.identity_calibration)
        tilted = rqspf(calibrate=_tilt)
        self.assertNotEqual(flat['probabilities'], tilted['probabilities'])
        self.assertAlmostEqual(sum(tilted['probabilities'].values()), 1.0, places=6)

    def test_default_calibration_does_nothing_and_says_so(self):
        """判据 29：默认值本身是一条分支，而且它的 meta 要能与真校准区分。"""
        result = spf()
        self.assertEqual(result['history_calibration'],
                         {'applied': False, 'reason': 'no_calibrator'})

    def test_calibration_is_told_which_market_and_league(self):
        seen = []

        def recording(probabilities, bet_type, league=None):
            seen.append((bet_type, league))
            return probabilities, {'applied': False, 'reason': 'recorded'}

        spf(calibrate=recording)
        rqspf(calibrate=recording)
        bifen(calibrate=recording)
        zjq(calibrate=recording)
        self.assertEqual([bet for bet, _ in seen], ['spf', 'rqspf', 'bifen', 'zjq'])
        self.assertEqual({league for _, league in seen}, {'英超'})


class HandicapGateTests(unittest.TestCase):

    def test_missing_handicap_stops_before_anything_else(self):
        result = rqspf(handicap_value=None)
        self.assertEqual(result['error'], '让球值不可用，无法计算让球胜平负')
        self.assertNotIn('odds', result)

    def test_the_handicap_reaches_the_score_model(self):
        """让球值不同，比分分布就该不同——它不是只写进表头的装饰字段。"""
        level = rqspf(handicap_value=0.0)
        minus_one = rqspf(handicap_value=-1.0)
        self.assertNotEqual(level['probabilities'], minus_one['probabilities'])

    def test_handicap_is_echoed_in_the_header(self):
        self.assertEqual(rqspf()['handicap'], '(-1)')


class OfficialRqspfOddsTests(unittest.TestCase):

    def test_ready_made_odds_win(self):
        odds = {'让胜': 2.20, '让平': 3.40, '让负': 3.10}
        result = rqspf(dict(MATCH, rqspf_odds=odds))
        self.assertEqual(result['odds'], odds)
        self.assertTrue(result['official_odds_available'])

    def test_lottery_odds_are_the_second_source(self):
        odds = {'让胜': 2.05, '让平': 3.50, '让负': 3.30}
        self.assertEqual(rqspf(dict(MATCH, lottery_rqspf_odds=odds))['odds'], odds)

    def test_three_prices_are_assembled_in_order(self):
        result = rqspf(dict(MATCH, rqspf_sp=2.20, rqspf_s=3.40, rqspf_f=3.10))
        self.assertEqual(result['odds'],
                         {'让胜': 2.20, '让平': 3.40, '让负': 3.10})

    def test_a_price_at_or_below_one_rejects_the_whole_set(self):
        """赔率不可能不到 1——**整组都不认**，不是只丢那一个。"""
        for bad in (0.95, 1.0, 0):
            with self.subTest(bad=bad):
                result = rqspf(dict(MATCH, rqspf_sp=2.20, rqspf_s=bad, rqspf_f=3.10))
                self.assertEqual(result['odds'], {})
                self.assertFalse(result['official_odds_available'])

    def test_unparsable_prices_yield_no_odds(self):
        result = rqspf(dict(MATCH, rqspf_sp='x', rqspf_s=3.40, rqspf_f=3.10))
        self.assertEqual(result['odds'], {})

    def test_missing_prices_yield_no_odds(self):
        self.assertEqual(rqspf()['odds'], {})


class MarketBlendTests(unittest.TestCase):

    BIFEN_ODDS = {'1-0': 8.0, '1-1': 7.5, '2-1': 9.0, '0-1': 11.0}
    ZJQ_ODDS = {'0': 11.0, '1': 5.6, '2': 3.9, '3': 4.3,
                '4': 6.5, '5': 11.0, '6': 21.0, '7+': 26.0}

    def test_no_market_odds_means_model_only(self):
        result = bifen()
        self.assertFalse(result['market_adjusted'])
        self.assertNotIn('odds', result)

    def test_all_dead_market_odds_fall_back_to_the_model(self):
        result = bifen(market_odds={'1-0': 0, '1-1': None})
        self.assertFalse(result['market_adjusted'])

    def test_bifen_blend_leaves_two_key_types_side_by_side(self):
        """**钉住现状**：模型矩阵的键是 `(主, 客)` 元组，市场报价的键是
        `'1-0'` 字符串，融合取的是两者的并集——它们并列成两批档位，
        模型每格只留 60%、市场每格只留 40%，谁也没融合进谁。

        结果是 `top3` 被市场那几个报过价的比分占满，模型对榜首毫无贡献。
        改掉它会改变用户看到的比分推荐，是一次产品决策而不是迁移的副产品。
        """
        result = bifen(market_odds=self.BIFEN_ODDS)
        self.assertTrue(result['market_adjusted'])
        keys = list(result['probabilities'])
        tuples = [k for k in keys if isinstance(k, tuple)]
        strings = [k for k in keys if isinstance(k, str)]
        self.assertTrue(tuples and strings)
        self.assertEqual(len(strings), len(self.BIFEN_ODDS))
        # 榜首全是市场那几个字符串键
        self.assertTrue(all(isinstance(score, str) for score, _ in result['top3']))
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_zjq_blend_really_blends(self):
        """总进球两边都是字符串档位，所以档位数不会翻倍——与比分那一路的
        差别正在这里。"""
        model_only = zjq()
        blended = zjq(market_odds=self.ZJQ_ODDS)
        self.assertTrue(blended['market_adjusted'])
        self.assertEqual(set(blended['probabilities']),
                         set(model_only['probabilities']))
        self.assertNotEqual(blended['probabilities'], model_only['probabilities'])

    def test_zjq_weights_are_parameters(self):
        market_only = zjq(market_odds=self.ZJQ_ODDS,
                          model_weight=0.0, market_weight=1.0)
        expected = adapter.calculate_implied_probability(self.ZJQ_ODDS)
        for bucket, probability in expected.items():
            self.assertAlmostEqual(market_only['probabilities'][bucket],
                                   probability, places=6)

    def test_zjq_shipped_weights(self):
        """判据 29：出厂权重要有一条不传参数的用例守着。"""
        self.assertEqual(zjq(market_odds=self.ZJQ_ODDS)['probabilities'],
                         zjq(market_odds=self.ZJQ_ODDS,
                             model_weight=ZJQ_WEIGHTS[0],
                             market_weight=ZJQ_WEIGHTS[1])['probabilities'])

    def test_bifen_shipped_weights(self):
        self.assertEqual(bifen(market_odds=self.BIFEN_ODDS)['probabilities'],
                         bifen(market_odds=self.BIFEN_ODDS,
                               model_weight=BIFEN_WEIGHTS[0],
                               market_weight=BIFEN_WEIGHTS[1])['probabilities'])

    def test_partial_market_quotes_only_lift_the_quoted_buckets(self):
        blended = zjq(market_odds={'2': 3.9, '3': 4.3})
        model_only = zjq()
        self.assertGreater(blended['probabilities']['2'],
                           model_only['probabilities']['2'])


class TrendAttachmentTests(unittest.TestCase):
    """三份走势历史各自决定一个字段出不出现。"""

    def test_asian_history_attaches_the_trend_and_feeds_quality(self):
        """走势不只是多一个字段——它作为上下文喂进质量分档，
        改变了 `conflict` 的判定。只断言字段出现，把这条链断掉也发现不了。"""
        without = spf()
        self.assertNotIn('asian_trend', without)
        self.assertFalse(without['quality']['conflict'])
        with_asian = spf(asian_data=ASIAN)
        self.assertIn('asian_trend', with_asian)
        self.assertTrue(with_asian['quality']['conflict'])

    def test_empty_asian_history_attaches_nothing(self):
        self.assertNotIn('asian_trend', spf(asian_data={'history': []}))

    def test_goals_history_marks_the_adjustment_and_attaches_the_trend(self):
        without = zjq()
        self.assertNotIn('goals_adjusted', without)
        self.assertNotIn('goals_trend', without)
        with_goals = zjq(goals_data=GOALS)
        self.assertTrue(with_goals['goals_adjusted'])
        self.assertIn('goals_trend', with_goals)

    def test_correct_score_history_marks_both_flags(self):
        cs = {'history': [{'ts': '2026-08-28T09:00:00',
                           'scores': {'1-0': 8.0, '1-1': 7.5}},
                          {'ts': '2026-08-28T10:00:00',
                           'scores': {'1-0': 7.5, '1-1': 7.8}}]}
        without = spf()
        self.assertNotIn('cs_adjusted', without)
        self.assertNotIn('cs_trend', without)
        with_cs = spf(cs_data=cs)
        self.assertTrue(with_cs['cs_adjusted'])
        self.assertIn('cs_trend', with_cs)

    def test_goals_history_moves_the_total_line(self):
        """大小球历史要真的喂进目标总进球——不然它只是个装饰字段。"""
        self.assertNotEqual(spf()['target_total'],
                            spf(goals_data=GOALS)['target_total'])

    def test_joint_market_state_flag_follows_the_meta(self):
        result = spf(asian_data=ASIAN, goals_data=GOALS)
        self.assertEqual(result['asian_adjusted'],
                         bool(result['joint_market_state'].get('applied')))


class HeaderTests(unittest.TestCase):

    def test_every_bet_type_carries_the_same_header(self):
        for result in (spf(), rqspf(), bifen(), zjq()):
            with self.subTest(bet=result['type']):
                self.assertEqual(result['match_id'], MATCH['id'])
                self.assertEqual(result['num'], MATCH['num'])
                self.assertEqual(result['league'], MATCH['league'])

    def test_only_rqspf_echoes_the_handicap(self):
        self.assertIn('handicap', rqspf())
        for result in (spf(), bifen(), zjq()):
            with self.subTest(bet=result['type']):
                self.assertNotIn('handicap', result)

    def test_top3_keeps_three(self):
        for result in (bifen(), zjq()):
            with self.subTest(bet=result['type']):
                self.assertEqual(len(result['top3']), TOP_KEPT)
        self.assertEqual(len(spf()['scores']), TOP_KEPT)


class DeadOddsAsymmetryTests(unittest.TestCase):
    """**某一档赔率为 0 时，四条路的下场完全不同。** 钉住现状。

    `implied_probability` 会把非正赔率整档丢掉，而另外两处不会：

    - 胜平负先用过滤后的概率判「有没有」，紧接着算 `margin` 时又对**全部**
      赔率取倒数 —— 直接 `ZeroDivisionError`；
    - 比分与总进球根本不走过滤，`_normalised_1x2` 上来就取倒数 —— 同样崩；
    - 只有让球那一路真的走到了兜底概率：它拿过滤后的概率补齐三档。

    这是判据 17「一半严格一半放任」的实例，而且是**真的崩**，不是静默失败。
    迁移前后行为一致，修它会改变线上表现，应单独决策。
    """

    DEAD = {'home': 1.80, 'draw': 0, 'away': 4.20}

    def test_spf_raises_on_the_margin(self):
        with self.assertRaises(ZeroDivisionError):
            spf(ouzhi=self.DEAD)

    def test_bifen_and_zjq_raise_before_anything_else(self):
        for name, fn in (('bifen', bifen), ('zjq', zjq)):
            with self.subTest(bet=name):
                with self.assertRaises(ZeroDivisionError):
                    fn(ouzhi=self.DEAD)

    def test_only_rqspf_reaches_the_fallback(self):
        """让球那一路不崩，而且兜底值真的用上了。"""
        result = rqspf(ouzhi=self.DEAD)
        self.assertNotIn('平', result['raw_spf_probabilities'])
        self.assertEqual(sorted(result['probabilities']), ['让平', '让胜', '让负'])

    def test_the_three_fallbacks_sum_to_one(self):
        """三档兜底加起来正好是 1 —— 「三档全缺」等价于「毫无信息」。"""
        self.assertAlmostEqual(sum(FALLBACK_1X2.values()), 1.0)


class AdapterTests(unittest.TestCase):

    def test_rqspf_does_not_fetch_when_the_handicap_is_unusable(self):
        """**让球值先解析、解析不出来就不抓欧赔**——迁移前那道守卫排在
        抓取之前。顺序也是行为：没有盘口的场次不该白打一次网络。"""
        with mock.patch.object(adapter, 'fetch_ouzhi_odds') as fetched:
            result = adapter.analyze_rqspf(dict(MATCH, handicap='公司'))
        fetched.assert_not_called()
        self.assertEqual(result['error'], '让球值不可用，无法计算让球胜平负')

    def test_rqspf_fetches_when_the_handicap_parses(self):
        with mock.patch.object(adapter, 'fetch_ouzhi_odds',
                               return_value=OUZHI) as fetched:
            with mock.patch.object(adapter, 'apply_beidan_history_calibration',
                                   side_effect=recommendation.identity_calibration):
                adapter.analyze_rqspf(MATCH)
        fetched.assert_called_once_with(MATCH['id'])

    def _analyse(self, fn, *args):
        with mock.patch.object(adapter, 'fetch_ouzhi_odds', return_value=OUZHI):
            with mock.patch.object(
                    adapter, 'apply_beidan_history_calibration',
                    side_effect=recommendation.identity_calibration):
                return fn(MATCH, *args)

    def test_market_odds_are_picked_by_match_id(self):
        """报价表里只有别场的赛事时，这一场拿不到任何市场概率。

        **两条路的表现方式不同**：比分写 `market_adjusted: False`，
        总进球**根本不写这个键**（它没有 else 分支）。判据 17 的形状，
        原样保留——调用方靠 `.get` 还是 `[]` 取它，结果不一样。
        """
        other_match = {'999': {'1-0': 8.0, '2': 3.9}}
        self.assertFalse(self._analyse(adapter.analyze_bifen, other_match)
                         ['market_adjusted'])
        self.assertNotIn('market_adjusted',
                         self._analyse(adapter.analyze_zjq, other_match))

    def test_a_non_dict_odds_table_is_ignored(self):
        result = self._analyse(adapter.analyze_bifen, ['not', 'a', 'dict'])
        self.assertFalse(result['market_adjusted'])

    def test_the_calibrator_still_goes_through_the_module_global(self):
        """打桩要打得中：适配层引用的是模块全局，不是函数对象的快照。"""
        with mock.patch.object(adapter, 'fetch_ouzhi_odds', return_value=OUZHI):
            with mock.patch.object(adapter, 'apply_beidan_history_calibration',
                                   return_value=({'胜': 1.0}, {'applied': True,
                                                               'reason': 'patched'})):
                result = adapter.analyze_rqspf(MATCH)
        self.assertEqual(result['history_calibration']['reason'], 'patched')


class InjectedCollaboratorTests(unittest.TestCase):
    """两组协作者是鸭子类型，拼错一个属性要到运行时才炸——这道守卫盯着它。"""

    DOMAIN = 'src/domain/sports/beidan/recommendation.py'

    def _attributes_used(self, variable):
        tree = ast.parse(pathlib.Path(self.DOMAIN).read_text(encoding='utf-8'))
        return {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == variable}

    def test_the_adapter_provides_every_model_operation(self):
        self.assertLessEqual(self._attributes_used('model'), set(vars(MODEL)))

    def test_the_adapter_provides_every_market_operation(self):
        self.assertLessEqual(self._attributes_used('market'), set(vars(MARKET)))

    def test_nothing_injected_is_unused(self):
        """反过来也要对：注入了没人用的操作，和 §五·14 那处「引入一个
        没人用的常量」是同一个问题——读的人会以为改它就能生效。"""
        self.assertEqual(set(vars(MODEL)) - self._attributes_used('model'), set())
        self.assertEqual(set(vars(MARKET)) - self._attributes_used('market'), set())

    def test_the_guard_would_catch_a_missing_operation(self):
        """守卫本身要能被证伪。"""
        self.assertFalse(self._attributes_used('model') <= {'predict_scores'})


FORBIDDEN_IMPORTS = {'time', 'os', 'pathlib', 'requests', 'urllib.request',
                     'src.common.kv_store', 'src.common.repositories',
                     'src.foundation.store', 'src.beidan.fetching',
                     'src.beidan.recommending'}
FORBIDDEN_CALLS = {'now', 'today', 'utcnow'}


class NoSideEffectTests(unittest.TestCase):

    DOMAIN = 'src/domain/sports/beidan/recommendation.py'
    ADAPTER = 'src/beidan/recommending.py'

    def _tree(self, path):
        return ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))

    def _imports(self, path):
        found = set()
        for node in ast.walk(self._tree(path)):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
                found.update(f'{node.module}.{a.name}' for a in node.names)
        return found

    def _clock_calls(self, path):
        return {node.func.attr for node in ast.walk(self._tree(path))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in FORBIDDEN_CALLS}

    def test_domain_imports_nothing_stateful(self):
        self.assertEqual(self._imports(self.DOMAIN) & FORBIDDEN_IMPORTS, set())

    def test_domain_never_reads_the_clock(self):
        self.assertEqual(self._clock_calls(self.DOMAIN), set())

    def test_the_guards_would_catch_a_real_violation(self):
        self.assertNotEqual(self._imports(self.ADAPTER) & FORBIDDEN_IMPORTS, set())
        self.assertNotEqual(self._clock_calls(self.ADAPTER), set())


if __name__ == '__main__':
    unittest.main()
