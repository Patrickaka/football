import unittest

import src.lottery3d as lottery3d


class Lottery3DZu6Tests(unittest.TestCase):
    def test_presence_score_counts_repeat_only_once_per_draw(self):
        repeated = [(1, 1, 2)] * 25
        scores = lottery3d.zu6_digit_scores(repeated)

        self.assertAlmostEqual(scores[1], scores[2], places=5)
        self.assertGreater(scores[1], scores[0])

    def test_four_digit_pool_uses_recent_presence_not_positions(self):
        history = []
        for i in range(40):
            hot = (i % 4, (i + 1) % 4, (i + 2) % 4)
            history.append(hot if i % 2 else tuple(reversed(hot)))

        scores = lottery3d.zu6_digit_scores(history)
        pool = lottery3d.pick_zu6_four(scores, numbers=history)

        self.assertEqual(pool, [0, 1, 2, 3])

    def test_predictor_version_marks_form_dynamic_zu3(self):
        self.assertEqual(
            lottery3d.PREDICTOR_VERSION,
            "3d-v4.9-form-dynamic-zu3",
        )

    def test_primary_presence_model_uses_validated_25_period_window(self):
        self.assertEqual(lottery3d.ZU6_PRESENCE_WINDOWS, (25,))


class Lottery3DZu3Tests(unittest.TestCase):
    def test_zu3_presence_counts_each_draw_once(self):
        # 末段 12 期均为组三 (1,1,2)；窗口内仅这些组三样本 → 1、2 出现率 = 1.0
        history = [(3, 4, 5)] * 40 + [(1, 1, 2)] * 12
        presence = lottery3d.zu3_digit_presence(history)

        self.assertAlmostEqual(presence[1], 1.0, places=4)
        self.assertAlmostEqual(presence[2], 1.0, places=4)
        self.assertEqual(presence[3], 0.0)
        self.assertEqual(presence[4], 0.0)

    def test_zu3_presence_uniform_when_no_samples(self):
        # 无组三历史 → 无信息先验 0.2
        history = [(3, 4, 5)] * 60
        presence = lottery3d.zu3_digit_presence(history)
        self.assertEqual(presence, {d: 0.2 for d in range(10)})

    def test_zu3_pair_scores_normalize_to_one(self):
        scores = lottery3d.zu3_pair_scores({d: 0.2 for d in range(10)})
        self.assertEqual(len(scores), 45)  # C(10,2)
        self.assertAlmostEqual(sum(s for _, s in scores), 1.0, places=6)
        # 均匀先验下每个数对等概率 = 1/45
        self.assertAlmostEqual(scores[0][1], 1 / 45.0, places=6)

    def test_zu3_combos_from_pair_covers_six_notes(self):
        combos = lottery3d.zu3_combos_from_pair((1, 5))
        self.assertEqual(combos, ["115", "151", "155", "511", "515", "551"])
        # 任意对子恒为 6 注，成本 12 元
        self.assertEqual(len(lottery3d.zu3_combos_from_pair((0, 9))), 6)

    def test_pick_zu3_pairs_top_pair_and_cost(self):
        history = [(3, 4, 5)] * 40 + [(1, 1, 2)] * 12
        rec = lottery3d.pick_zu3_pairs(history)

        self.assertEqual(len(rec["pairs"]), lottery3d.ZU3_PAIRS_COUNT)  # 4 组
        self.assertEqual(rec["pairs"][0]["digits"], [1, 2])  # 1、2 是唯一高频对子
        for p in rec["pairs"]:
            self.assertEqual(len(p["combos"]), 6)
            self.assertEqual(p["cost"], 12)
            self.assertGreaterEqual(p["prob"], 0.0)
            self.assertLessEqual(p["prob"], 1.0)
        # 条件命中率 = 前4对子概率和
        self.assertAlmostEqual(
            rec["conditional_hit_rate"],
            sum(p["prob"] for p in rec["pairs"]),
            places=4,
        )
        self.assertEqual(rec["total_cost"], 4 * 12)
        # 随机基准：任取 4 组对子 = 4/45
        self.assertAlmostEqual(rec["random_conditional_hit_rate"], 4 / 45.0, places=4)

    def test_form_bet_primary_follows_blend_max(self):
        numbers = [(1, 2, 3)] * 20
        # blend 组三抬升 → 主推组三 + elevated 信号
        rec = lottery3d.recommend_form_bet(
            {"blend_p": {"zu6": 0.40, "zu3": 0.58, "baozi": 0.02}}, numbers
        )
        self.assertEqual(rec["primary"], "zu3")
        self.assertEqual(rec["zu3_signal"], "elevated")
        # 基准态 blend（组六 72%）→ 主推组六 + normal 信号
        rec2 = lottery3d.recommend_form_bet(
            {"blend_p": {"zu6": 0.72, "zu3": 0.27, "baozi": 0.01}}, numbers
        )
        self.assertEqual(rec2["primary"], "zu6")
        self.assertEqual(rec2["zu3_signal"], "normal")
        self.assertAlmostEqual(rec2["zu3_base_rate"], 0.27, places=4)
        # 三个形态概率相加 = 1
        blend = rec2["blend_p"]
        self.assertAlmostEqual(blend["zu6"] + blend["zu3"] + blend["baozi"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
