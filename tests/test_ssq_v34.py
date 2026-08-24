"""SSQ v3.4 蓝球全覆盖保底方案测试。

16注蓝球互不相同覆盖全部16码：任意开奖蓝球必被其中1注命中（鸽笼原理）。
红球按5注推荐轮转复用，无额外预测假设。
"""
import unittest

from src.ssq import run_prediction, SSQ_PREDICTION_VERSION, load_history


class BlueFullCoverPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_prediction(data=load_history())

    def test_plan_exists_with_16_notes(self):
        plan = self.result['blue_full_cover_plan']
        self.assertEqual(plan['name'], '蓝球全覆盖保底（16注）')
        self.assertEqual(len(plan['notes']), 16)
        self.assertEqual(plan['cost_yuan'], 32)

    def test_blues_cover_all_16_exactly_once(self):
        notes = self.result['blue_full_cover_plan']['notes']
        blues = [note['blue'] for note in notes]
        self.assertEqual(sorted(blues), list(range(1, 17)))
        self.assertEqual(len(set(blues)), 16)

    def test_reds_cycle_through_recommended_sets(self):
        notes = self.result['blue_full_cover_plan']['notes']
        sets = self.result['prediction']['sets']
        for i, note in enumerate(notes):
            self.assertEqual(note['red'], sets[i % len(sets)]['red'])
            self.assertEqual(len(note['red']), 6)

    def test_version_bumped(self):
        self.assertEqual(SSQ_PREDICTION_VERSION, 'ssq-v3.4-blue-cover')
        self.assertEqual(self.result['version'], 'ssq-v3.4-blue-cover')

    def test_guarantee_is_pigeonhole_not_prediction(self):
        """诚实标注：保证文案声明鸽笼原理/组合覆盖，而非预测能力"""
        plan = self.result['blue_full_cover_plan']
        self.assertIn('鸽笼原理', plan['guarantee'])
        self.assertIn('非预测能力', plan['disclaimer'])
        self.assertIn('期望为负', plan['disclaimer'])


if __name__ == '__main__':
    unittest.main()
