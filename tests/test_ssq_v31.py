# -*- coding: utf-8 -*-
"""双色球 v3.1 蓝球去重覆盖 + 期号推算修复测试。"""
import unittest

from src import ssq


class SsqNextPeriodTests(unittest.TestCase):
    """_next_period 期号推算：前导零、跨年、特殊年份。"""

    def setUp(self):
        self.history = ssq.load_history()
        self.history = sorted(self.history, key=lambda x: str(x['period']))

    def test_plain_increment_keeps_five_digit_format(self):
        self.assertEqual(ssq._next_period('25099', self.history), '25100')
        self.assertEqual(ssq._next_period('25100', self.history), '25101')

    def test_leading_zero_is_preserved(self):
        # 旧实现 str(int('03001')+1)='3002' 会丢失前导零
        self.assertEqual(ssq._next_period('03001', self.history), '03002')
        self.assertEqual(ssq._next_period('03050', self.history), '03051')

    def test_year_rollover_uses_max_seq_of_year(self):
        # 2025 年 151 期完结（完整年份），第151期后跨年 → 26001
        self.assertEqual(ssq._next_period('25151', self.history), '26001')
        # 2020 年只有 134 期（特殊年份），第134期后跨年
        self.assertEqual(ssq._next_period('20134', self.history), '21001')
        # 2003 年只有 89 期
        self.assertEqual(ssq._next_period('03089', self.history), '04001')

    def test_incomplete_last_year_does_not_false_rollover(self):
        # 2026 年是数据最后一年且未完结（当前95期），26095 后应为 26096 而非 27001
        self.assertEqual(ssq._next_period('26095', self.history), '26096')
        self.assertEqual(ssq._next_period('26094', self.history), '26095')

    def test_no_history_fallback_uses_153(self):
        self.assertEqual(ssq._next_period('26153'), '27001')
        self.assertEqual(ssq._next_period('26152'), '26153')

    def test_next_period_matches_all_real_history_pairs(self):
        ok = 0
        for j in range(len(self.history) - 1):
            if ssq._next_period(str(self.history[j]['period']), self.history) == \
                    str(self.history[j + 1]['period']):
                ok += 1
        self.assertEqual(ok, len(self.history) - 1)


class SsqBlueDedupeTests(unittest.TestCase):
    """_predict_sets 蓝球去重覆盖。"""

    def setUp(self):
        self.history = ssq.load_history()
        self.history = sorted(self.history, key=lambda x: str(x['period']))
        self.analysis = ssq._analyze(self.history)

    def test_five_blues_are_all_distinct(self):
        sets = ssq._predict_sets(self.history, self.analysis, n=5, seed=26095)
        blues = [s['blue'] for s in sets]
        self.assertEqual(len(blues), 5)
        self.assertEqual(len(set(blues)), 5, f'蓝球应互不重复，实际 {blues}')
        for b in blues:
            self.assertIn(b, ssq.BLUE_RANGE)

    def test_seed_determinism(self):
        a = ssq._predict_sets(self.history, self.analysis, n=5, seed=26095)
        b = ssq._predict_sets(self.history, self.analysis, n=5, seed=26095)
        self.assertEqual(a, b)

    def test_blue_excludes_prev_blue(self):
        prev_blue = self.history[-1]['blue']
        sets = ssq._predict_sets(self.history, self.analysis, n=5,
                                 seed=int(self.history[-1]['period']))
        blues = [s['blue'] for s in sets]
        self.assertNotIn(prev_blue, blues)

    def test_blue_uniformity_not_broken_by_dedupe(self):
        # 去重覆盖后蓝球联合命中理论 = 5/16 = 31.25%，单注仍约 1/16
        # 这里只验证"单注均匀性不破坏"：500期单注命中率落在 3%~10% 之间
        hits = 0
        total = 0
        history = self.history
        for i in range(len(history) - 500, len(history)):
            train = history[:i]
            if len(train) < 200:
                continue
            analysis = ssq._analyze(train)
            actual_blue = history[i]['blue']
            sets = ssq._predict_sets(train, analysis, n=5,
                                     seed=int(history[i]['period']))
            hits += sum(1 for s in sets if s['blue'] == actual_blue)
            total += 5
        rate = hits / total
        self.assertGreater(rate, 0.03)
        self.assertLess(rate, 0.10)


class SsqRedCoverTests(unittest.TestCase):
    """v3.2 红球蛇形分组去重覆盖。"""

    def setUp(self):
        self.history = ssq.load_history()
        self.history = sorted(self.history, key=lambda x: str(x['period']))
        self.analysis = ssq._analyze(self.history)

    def test_five_reds_are_disjoint_union_30(self):
        sets = ssq._predict_sets(self.history, self.analysis, n=5, seed=26095)
        reds = [set(s['red']) for s in sets]
        union = set().union(*reds)
        self.assertEqual(len(union), 30, f'5注红球应覆盖30个不同号码，实际 union={len(union)}')
        for r in reds:
            self.assertEqual(len(r), 6)
            self.assertTrue(r <= set(ssq.RED_RANGE))

    def test_primary_red_passes_validity(self):
        # 主推注（第1组）必须满足合法性约束
        sets = ssq._predict_sets(self.history, self.analysis, n=5, seed=26095)
        self.assertTrue(ssq._is_valid_red(sets[0]['red']))

    def test_red_cover_ge2_above_theory(self):
        # 2000期验证：去重叠后任1注≥2码 ≈ 96.2%（蒙特卡洛理论95.9%），远高于重叠现状82%
        history = self.history
        ge2 = 0
        n = 0
        for i in range(len(history) - 2000, len(history)):
            train = history[:i]
            if len(train) < 200:
                continue
            analysis = ssq._analyze(train)
            actual_red = set(history[i]['red'])
            sets = ssq._predict_sets(train, analysis, n=5,
                                     seed=int(history[i]['period']))
            reds = [set(s['red']) for s in sets]
            n += 1
            if any(len(r & actual_red) >= 2 for r in reds):
                ge2 += 1
        rate = ge2 / n
        self.assertGreater(rate, 0.93, f'ge2联合应≈96%，实际 {rate:.1%}')
        self.assertLess(rate, 0.99)


class SsqRunPredictionNextPeriodTests(unittest.TestCase):
    def test_run_prediction_reports_correct_next_period(self):
        result = ssq.run_prediction(data=self.history_short())
        self.assertEqual(result['next_period'], '26001')
        self.assertEqual(result['version'], ssq.SSQ_PREDICTION_VERSION)

    @staticmethod
    def history_short():
        h = ssq.load_history()
        h = sorted(h, key=lambda x: str(x['period']))
        # 截断到 2025 年末，让下一期必然是跨年 26001（用历史推断）
        h = [r for r in h if str(r['period']) <= '25151']
        # 保留尾部足够数据做统计
        return h[-400:]


if __name__ == '__main__':
    unittest.main()
