"""福彩 3D 的融合与策略：规则×ML 融合、三套策略、模式/预算/注数、三路结算。

参照物是从迁移前的实现生成的黄金文件
（`tests/fixtures/golden/lottery3d_fusion.json.gz`，177 条），语料按八种
规则/ML 列表组合（重叠 / 不相交 / 完全相同 / 各自为空 / 都空 / 带拆解 / 极短）
× 四档 top_n × 四组权重铺开。

**这一批删掉了三个从没生效的参数**：`generate_strategy_recommendations` 的
`danma` 与 `kill`、`recommend_budget_level` 的 `stability`——函数体一个都
没读过，而 `recommend_budget_level` 的 docstring 还专门列着 `stability`。
删掉后输出一字未变，177 条黄金值全部逐条相同。
"""
import gzip
import json
import pathlib
import unittest

from src.domain.numeric.lottery3d import fusion
from src.lottery3d import fusion as adapter
from tests.domain.golden import as_comparable

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / 'fixtures'


def _load(name):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


GOLDEN = _load('golden/lottery3d_fusion.json.gz')


def _rule(nums, with_detail=False):
    return [{'num': num, 'score': 100.0 - index,
             **({'detail': {'base_digit': float(index)}} if with_detail else {})}
            for index, num in enumerate(nums)]


def _ml(nums):
    return [{'num': num, 'model_score': 50.0 - index}
            for index, num in enumerate(nums)]


A = [f'{i:03d}' for i in range(40)]
B = [f'{i:03d}' for i in range(20, 60)]
C = [f'{i:03d}' for i in range(100, 140)]

LISTS = {
    'overlap': (_rule(A), _ml(B)),
    'disjoint': (_rule(A), _ml(C)),
    'identical': (_rule(A), _ml(A)),
    'no_ml': (_rule(A), []),
    'no_rule': ([], _ml(B)),
    'both_empty': ([], []),
    'with_detail': (_rule(A, with_detail=True), _ml(B)),
    'tiny': (_rule(['001', '002']), _ml(['002', '003'])),
}
WEIGHTS = ((0.55, 0.45), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5))

MODE_CASES = [
    (0.5, 0.0, 0.03, 300), (0.5, -0.01, 0.03, 300), (0.9, 0.02, 0.02, 300),
    (0.9, 0.02, 0.04, 300), (0.5, 0.02, 0.03, 250), (0.5, 0.005, 0.03, 250),
    (0.5, 0.02, 0.03, 251), (0.81, 0.02, 0.029, 600), (0.0, 0.001, 0.0, 1000),
]
BUDGET_CASES = [
    (0.0, 0.5, 0.03), (-0.01, 0.5, 0.03), (0.016, 0.5, 0.03), (0.016, 0.5, 0.029),
    (0.015, 0.5, 0.03), (0.02, 0.9, 0.05), (0.001, 0.1, 0.0),
    (0.02, 0.0, 0.05), (0.02, 1.0, 0.05),
]
COUNT_CASES = [
    (0.0, 0.2, 0.03), (-0.01, 0.2, 0.03), (0.01, 0.18, 0.03), (0.01, 0.18, 0.029),
    (0.01, 0.12, 0.0), (0.01, 0.119, 0.0), (0.01, 0.0, 0.0), (0.05, 0.5, 0.1),
]

PERIODS = [f'20262{i:02d}' for i in range(20)]
NUMBERS = [(i % 10, (i + 1) % 10, (i + 2) % 10) for i in range(20)]


def _row(period, rule_nums, ml_nums, fused_nums, settled=False):
    return {'period': period, 'rule_only': list(rule_nums), 'ml_only': list(ml_nums),
            'fused': list(fused_nums), 'created_at': '2026-08-27 09:23:22',
            'settled': settled, 'revision': 1}


SETTLE_HISTORY = {
    'hit_top3': [_row(PERIODS[0], ['123', '234', '345'], [], ['123'])],
    'hit_top30': [_row(PERIODS[0], [f'{i:03d}' for i in range(30)], [], [])],
    'miss': [_row(PERIODS[0], ['999'], ['888'], ['777'])],
    'already_settled': [_row(PERIODS[0], ['123'], [], [], settled=True)],
    'unknown_period': [_row('9999999', ['123'], [], [])],
    'latest_period': [_row(PERIODS[-1], ['123'], [], [])],
    'empty_lists': [_row(PERIODS[0], [], [], [])],
}


def golden_entries():
    """按 (键, 值) 逐条产出全部语料，测试与重生成脚本共用。"""
    for name, (rule, ml) in LISTS.items():
        for top_n in (3, 10, 30, 100):
            for rule_weight, ml_weight in WEIGHTS:
                yield (f'fuse:{name}:{top_n}:{rule_weight}:{ml_weight}',
                       adapter.fuse_rule_ml(rule, ml, top_n, rule_weight, ml_weight))
    for name, (rule, ml) in LISTS.items():
        for tag in ('-:-', '12:78'):
            yield (f'strategies:{name}:{tag}',
                   adapter.generate_strategy_recommendations(rule, ml))
    for index, args in enumerate(MODE_CASES):
        yield f'mode:{index}', adapter.select_strategy_mode(*args)
    for index, args in enumerate(BUDGET_CASES):
        yield f'budget:{index}', adapter.recommend_budget_level(args[0], args[2])
    for index, args in enumerate(COUNT_CASES):
        yield f'count:{index}', adapter.auto_recommend_count(*args)
    yield from _settle_entries()


def _settle_entries():
    """走适配层：黄金记录的是**实际写回存储的内容**——没有改动时它什么都
    不写，那本身就是要守住的行为（别把没变的记录反复写一遍）。
    领域函数自己的语义由 `SettleHistoryTests` 直接覆盖。
    """
    original_load, original_save = adapter.kv_store.load, adapter.kv_store.save
    try:
        for name, history in SETTLE_HISTORY.items():
            saved = {}
            rows = [dict(row) for row in history]
            adapter.kv_store.load = (
                lambda key, default=None, r=rows:
                r if key == adapter.STRATEGY_RECORDS_KEY
                else (default if default is not None else []))
            adapter.kv_store.save = lambda key, value, s=saved: s.update({key: value})
            adapter.settle_strategy_records(PERIODS, NUMBERS)
            yield (f'settle_strategy:{name}',
                   saved.get(adapter.STRATEGY_RECORDS_KEY))
    finally:
        adapter.kv_store.load, adapter.kv_store.save = original_load, original_save


class GoldenTests(unittest.TestCase):
    """迁移前后逐条比对。"""

    def test_matches_golden(self):
        seen = set()
        for key, value in golden_entries():
            seen.add(key)
            with self.subTest(case=key):
                self.assertEqual(as_comparable(value), GOLDEN[key])
        self.assertEqual(sorted(set(GOLDEN) - seen), [])


class SignatureTests(unittest.TestCase):
    """删掉的三个参数：传了不生效比报错危险得多，现在传就是 TypeError。"""

    def test_strategy_recommendations_rejects_danma_and_kill(self):
        with self.assertRaises(TypeError):
            adapter.generate_strategy_recommendations([], [], [1, 2], [7, 8])

    def test_budget_level_rejects_stability(self):
        with self.assertRaises(TypeError):
            adapter.recommend_budget_level(0.02, 0.5, 0.03)


class FuseTests(unittest.TestCase):

    def _fuse(self, rule, ml, **over):
        args = {'top_n': 30, 'rule_weight': 0.55, 'ml_weight': 0.45}
        args.update(over)
        return fusion.fuse(rule, ml, args['top_n'], args['rule_weight'],
                           args['ml_weight'])

    def test_a_number_in_both_lists_gets_the_bonus(self):
        """两个独立模型撞到一起，是融合唯一真正新增的信息。"""
        both = self._fuse(_rule(['111']), _ml(['111']))[0]
        rule_only = self._fuse(_rule(['111']), _ml(['222']))
        only = next(item for item in rule_only if item['num'] == '111')
        self.assertAlmostEqual(both['fuse_score'] - only['fuse_score'],
                               20 + 100 * 0.45, places=6)

    def test_tags_name_where_the_number_came_from(self):
        result = {item['num']: item['tag']
                  for item in self._fuse(_rule(['111', '222']), _ml(['222', '333']))}
        self.assertEqual(result['111'], 'rule_preferred')
        self.assertEqual(result['222'], 'high_confidence')
        self.assertEqual(result['333'], 'exploration')

    def test_rank_score_bottoms_out_at_zero(self):
        """第一百名之后不倒扣：一个模型没推的号不该因此被罚。"""
        self.assertEqual(fusion._rank_score(100), 0)
        self.assertEqual(fusion._rank_score(999), 0)
        self.assertEqual(fusion._rank_score(0), 100)

    def test_weights_shift_the_order(self):
        rule, ml = _rule(['111', '222']), _ml(['222', '111'])
        rule_heavy = self._fuse(rule, ml, rule_weight=1.0, ml_weight=0.0)
        ml_heavy = self._fuse(rule, ml, rule_weight=0.0, ml_weight=1.0)
        self.assertEqual(rule_heavy[0]['num'], '111')
        self.assertEqual(ml_heavy[0]['num'], '222')

    def test_result_is_capped_at_top_n(self):
        self.assertEqual(len(self._fuse(_rule(A), _ml(B), top_n=5)), 5)

    def test_both_lists_empty_gives_nothing(self):
        self.assertEqual(self._fuse([], []), [])

    def test_ties_break_deterministically(self):
        """方向本身无所谓，可复现才是重点——同一份输入两次必须同一个顺序。"""
        rule, ml = _rule(['111', '222', '333']), []
        first = [item['num'] for item in self._fuse(rule, ml)]
        second = [item['num'] for item in self._fuse(rule, ml)]
        self.assertEqual(first, second)

    def test_rule_detail_is_carried_through(self):
        result = self._fuse(_rule(['111'], with_detail=True), [])
        self.assertEqual(result[0]['detail'], {'base_digit': 0.0})

    def test_ml_only_number_has_no_detail_without_a_builder(self):
        """规则模型没推过它，自然没有拆解。**不编一个。**"""
        result = self._fuse([], _ml(['111']))
        self.assertIsNone(result[0]['detail'])

    def test_a_builder_fills_in_the_missing_detail(self):
        result = fusion.fuse([], _ml(['111']), 30, 0.55, 0.45,
                             detail_for=lambda num: {'built': num})
        self.assertEqual(result[0]['detail'], {'built': '111'})

    def test_score_and_fuse_score_are_the_same_number(self):
        item = self._fuse(_rule(['111']), _ml(['111']))[0]
        self.assertEqual(item['score'], item['fuse_score'])


class StrategyTests(unittest.TestCase):

    def test_conservative_is_the_intersection(self):
        result = fusion.strategy_recommendations(_rule(['111', '222']), _ml(['222']))
        self.assertEqual([item['num'] for item in result['conservative']], ['222'])

    def test_balanced_is_rule_led_with_a_little_ml(self):
        result = fusion.strategy_recommendations(_rule(A), _ml(C))
        sources = [item['source'] for item in result['balanced']]
        self.assertEqual(sources.count('rule'), 20)
        self.assertEqual(sources.count('ml'), 0)   # 截到 20 之后 ML 的补不进来

    def test_explore_is_ml_only(self):
        result = fusion.strategy_recommendations(_rule(['111']), _ml(['222', '111']))
        self.assertEqual([item['num'] for item in result['explore']], ['222'])

    def test_each_lane_is_capped(self):
        result = fusion.strategy_recommendations(_rule(A), _ml(A))
        self.assertLessEqual(len(result['conservative']), 10)
        self.assertLessEqual(len(result['balanced']), 20)
        self.assertLessEqual(len(result['explore']), 10)

    def test_empty_inputs_give_three_empty_lanes(self):
        result = fusion.strategy_recommendations([], [])
        self.assertEqual(result, {'conservative': [], 'balanced': [], 'explore': []})


class ModeTests(unittest.TestCase):
    """模式与理由一起返回：模式只是三个词，人要看到为什么才知道该不该照做。"""

    def test_no_lift_always_explores(self):
        for rank in (100, 500, 1000):
            with self.subTest(rank=rank):
                self.assertEqual(fusion.select_mode(0.5, 0.0, 0.5, rank)[0], 'explore')

    def test_too_stable_and_missing_explores(self):
        self.assertEqual(fusion.select_mode(0.81, 0.02, 0.029, 300)[0], 'explore')

    def test_moderate_stability_is_not_yet_too_stable(self):
        """0.8 这道界要有紧贴两侧的样本：0.6 也判成过稳的话，正常波动会被
        当成「推荐不动了」而强行打散。门槛写字面量。"""
        self.assertNotEqual(fusion.select_mode(0.6, 0.02, 0.02, 300)[0], 'explore')
        self.assertEqual(fusion.select_mode(0.81, 0.02, 0.02, 300)[0], 'explore')

    def test_stable_but_hitting_does_not_explore(self):
        """两个条件是「与」：只满足一个就转探索的话，命中良好时也会被打散。"""
        self.assertNotEqual(fusion.select_mode(0.81, 0.02, 0.031, 300)[0], 'explore')

    def test_excellent_rank_goes_conservative(self):
        self.assertEqual(fusion.select_mode(0.5, 0.02, 0.03, 250)[0], 'conservative')

    def test_one_rank_worse_is_no_longer_excellent(self):
        """门槛写字面量：250 是随机期望 500 的一半。"""
        self.assertEqual(fusion.select_mode(0.5, 0.02, 0.03, 251)[0], 'balanced')

    def test_marginal_lift_is_not_enough_for_conservative(self):
        self.assertEqual(fusion.select_mode(0.5, 0.005, 0.03, 250)[0], 'balanced')

    def test_every_mode_comes_with_a_reason(self):
        for args in MODE_CASES:
            with self.subTest(args=args):
                self.assertTrue(fusion.select_mode(*args)[1])


class BudgetTests(unittest.TestCase):
    """默认是最低档。3D 是公平摇奖，多数时候「没有优势」就是实情。"""

    def test_no_lift_gives_the_lowest_level(self):
        self.assertEqual(fusion.budget_level(0.0, 0.5)['level'], '低')
        self.assertEqual(fusion.budget_level(-0.01, 0.5)['level'], '低')

    def test_strong_lift_with_decent_hits_raises_the_level(self):
        self.assertEqual(fusion.budget_level(0.016, 0.03)['level'], '中')

    def test_strong_lift_without_hits_stays_watchful(self):
        """两个条件是「与」：只有 lift 好看而线上没中，不该加注。"""
        self.assertEqual(fusion.budget_level(0.016, 0.029)['level'], '观察')

    def test_the_lift_threshold_is_strict(self):
        self.assertEqual(fusion.budget_level(0.015, 0.03)['level'], '观察')

    def test_every_level_comes_with_a_reason_and_a_count(self):
        for lift, rate in ((0.0, 0.0), (0.02, 0.05), (0.001, 0.0)):
            with self.subTest(lift=lift):
                result = fusion.budget_level(lift, rate)
                self.assertTrue(result['reason'])
                self.assertGreater(result['suggest_count'], 0)


class CountTests(unittest.TestCase):

    def test_no_lift_cuts_the_count(self):
        self.assertEqual(fusion.recommend_count(0.0, 0.5, 0.1)[0], 10)

    def test_good_coverage_and_hits_gives_the_most(self):
        self.assertEqual(fusion.recommend_count(0.01, 0.18, 0.03)[0], 30)

    def test_good_coverage_without_hits_gives_less(self):
        self.assertEqual(fusion.recommend_count(0.01, 0.18, 0.029)[0], 20)

    def test_coverage_between_the_two_thresholds_is_only_decent(self):
        """0.18 与 0.12 两道界要分得开：覆盖率 0.15 且线上命中达标时，
        仍然只是「尚可」而不是「良好」。"""
        self.assertEqual(fusion.recommend_count(0.01, 0.15, 0.03)[0], 20)
        self.assertEqual(fusion.recommend_count(0.01, 0.18, 0.03)[0], 30)

    def test_decent_coverage_gives_the_middle(self):
        self.assertEqual(fusion.recommend_count(0.01, 0.12, 0.0)[0], 20)

    def test_below_decent_coverage_gives_the_floor(self):
        """门槛写字面量：随机基准是 10%，0.12 只是「略好」。"""
        self.assertEqual(fusion.recommend_count(0.01, 0.119, 0.0)[0], 15)

    def test_count_never_drops_below_ten(self):
        for args in COUNT_CASES:
            with self.subTest(args=args):
                self.assertGreaterEqual(fusion.recommend_count(*args)[0], 10)


class SettleTests(unittest.TestCase):
    """三条赛道**各自**统计——混在一起就无从判断融合有没有比规则模型好。"""

    def _settle(self, rule_nums, ml_nums, fused_nums, actual='123'):
        row = _row('p', rule_nums, ml_nums, fused_nums)
        return fusion.settle_row(row, actual, 'p_next')

    def test_each_lane_is_scored_separately(self):
        row = self._settle(['123'], ['999'], ['123'])
        self.assertTrue(row['rule_only_hit_top3'])
        self.assertFalse(row['ml_only_hit_top3'])
        self.assertTrue(row['fused_hit_top3'])

    def test_top3_and_top30_are_not_interchangeable(self):
        """命中号排在第 6 位：只中 top30、不中 top3。两个切片写反了就会颠倒。"""
        nums = [f'{i:03d}' for i in range(30)]
        row = self._settle(nums, [], [], actual=nums[5])
        self.assertFalse(row['rule_only_hit_top3'])
        self.assertTrue(row['rule_only_hit_top30'])

    def test_tiers_are_nested(self):
        nums = [f'{i:03d}' for i in range(100)]
        row = self._settle(nums, [], [], actual=nums[50])
        self.assertFalse(row['rule_only_hit_top3'])
        self.assertFalse(row['rule_only_hit_top30'])
        self.assertTrue(row['rule_only_hit_top100'])

    def test_rank_is_one_based(self):
        row = self._settle(['999', '123'], [], [])
        self.assertEqual(row['rule_only_rank'], 2)

    def test_a_missing_number_ranks_beyond_the_pool(self):
        """记 1001 而不是 1000：要能与「碰巧排在最后一名」分开。"""
        self.assertEqual(self._settle(['999'], [], [])['rule_only_rank'], 1001)

    def test_settling_stamps_the_draw_period(self):
        row = self._settle(['123'], [], [])
        self.assertEqual(row['draw_period'], 'p_next')
        self.assertTrue(row['settled'])


class SettleHistoryTests(unittest.TestCase):

    def test_record_is_settled_against_the_following_draw(self):
        rows = [_row(PERIODS[0], ['123'], [], [])]
        self.assertTrue(fusion.settle_history(rows, PERIODS, NUMBERS))
        self.assertEqual(rows[0]['actual'], ''.join(map(str, NUMBERS[1])))

    def test_already_settled_rows_are_untouched(self):
        rows = [_row(PERIODS[0], ['123'], [], [], settled=True)]
        self.assertFalse(fusion.settle_history(rows, PERIODS, NUMBERS))
        self.assertNotIn('actual', rows[0])

    def test_the_latest_period_has_no_draw_yet(self):
        rows = [_row(PERIODS[-1], ['123'], [], [])]
        self.assertFalse(fusion.settle_history(rows, PERIODS, NUMBERS))

    def test_an_unknown_period_is_skipped(self):
        rows = [_row('9999999', ['123'], [], [])]
        self.assertFalse(fusion.settle_history(rows, PERIODS, NUMBERS))


class SaveStrategyRecordsTests(unittest.TestCase):
    """`save_strategy_records` 此前一条用例都没有——变异验证撞出来的。"""

    def setUp(self):
        self.saved = {}
        self.history = []
        self._load, self._save = adapter.kv_store.load, adapter.kv_store.save
        adapter.kv_store.load = (
            lambda key, default=None: self.history
            if key == adapter.STRATEGY_RECORDS_KEY
            else (default if default is not None else []))
        adapter.kv_store.save = lambda key, value: self.saved.update({key: value})

    def tearDown(self):
        adapter.kv_store.load, adapter.kv_store.save = self._load, self._save

    def _save_record(self, period):
        adapter.save_strategy_records(period, ['r'], ['m'], ['f'])
        return self.saved.get(adapter.STRATEGY_RECORDS_KEY)

    def test_a_new_period_is_written(self):
        written = self._save_record('p1')
        self.assertEqual([row['period'] for row in written], ['p1'])

    def test_an_existing_unsettled_record_is_not_overwritten(self):
        """首次发布的推荐就是当时真的发出去的那份，改了它等于篡改对比数据。"""
        self.history = [_row('p1', ['old'], [], [])]
        self.assertIsNone(self._save_record('p1'))
        self.assertEqual(self.history[0]['rule_only'], ['old'])

    def test_an_existing_settled_record_is_not_overwritten(self):
        self.history = [_row('p1', ['old'], [], [], settled=True)]
        self.assertIsNone(self._save_record('p1'))

    def test_history_is_trimmed_to_the_limit(self):
        """只留最近若干期：三路对比要的是趋势，不是全量档案。写字面量。"""
        self.history = [_row(f'p{i}', [], [], []) for i in range(200)]
        written = self._save_record('new')
        self.assertEqual(len(written), 200)
        self.assertEqual(written[-1]['period'], 'new')
        self.assertEqual(written[0]['period'], 'p1')


class DetailBuilderTests(unittest.TestCase):
    """补拆解要四个参数**都**给齐。缺一个就留 None，不编。

    打桩打在 `triplet_weight_detail` 上：这里要测的是「够不够条件去建」，
    不是「建出来的拆解对不对」——后者是 3-12 的黄金文件在管。
    """

    def setUp(self):
        self._original = adapter.triplet_weight_detail
        adapter.triplet_weight_detail = lambda *a, **k: {'built': True}

    def tearDown(self):
        adapter.triplet_weight_detail = self._original

    def _fuse(self, **over):
        args = {'score': [1.0] * 10, 'danma': [1], 'kill': [2], 'meta': {}}
        args.update(over)
        return adapter.fuse_rule_ml([], _ml(['111']), 30, 0.55, 0.45, **args)

    def test_all_four_present_builds_a_detail(self):
        self.assertEqual(self._fuse()[0]['detail'], {'built': True})

    def test_a_single_missing_argument_leaves_it_none(self):
        """四个里缺一个就不建。改成「任一存在即建」的话，剩下三个是 None，
        建的时候会直接炸在里面。"""
        for missing in ('score', 'danma', 'kill', 'meta'):
            with self.subTest(missing=missing):
                self.assertIsNone(self._fuse(**{missing: None})[0]['detail'])


class FirstPublishWinsTests(unittest.TestCase):
    """首次发布的记录不覆盖——那是当时真的发出去的推荐。"""

    def test_find_by_period_locates_an_existing_record(self):
        history = [_row('a', [], [], []), _row('b', [], [], [])]
        self.assertIs(fusion.find_by_period(history, 'b'), history[1])

    def test_find_by_period_returns_none_when_absent(self):
        self.assertIsNone(fusion.find_by_period([], 'a'))

    def test_a_new_record_starts_at_revision_one(self):
        record = fusion.new_strategy_record('p', ['1'], ['2'], ['3'], 'now')
        self.assertEqual(record['revision'], 1)
        self.assertFalse(record['settled'])


if __name__ == '__main__':
    unittest.main()
