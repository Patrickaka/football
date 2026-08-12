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

    def test_predictor_version_marks_zu6_upgrade(self):
        self.assertEqual(lottery3d.PREDICTOR_VERSION, "3d-v4.7-zu6-presence")


if __name__ == "__main__":
    unittest.main()
