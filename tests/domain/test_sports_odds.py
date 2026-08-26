import json
import unittest

from src.domain.sports.odds import Odds, odds_to_prob


class OddsToProbTests(unittest.TestCase):
    def test_converts_decimal_odds(self):
        self.assertAlmostEqual(odds_to_prob(2.0), 0.5)
        self.assertAlmostEqual(odds_to_prob(4.0), 0.25)

    def test_rejects_non_positive_odds(self):
        for bad in (0, -1, -0.5):
            with self.assertRaises(ValueError):
                odds_to_prob(bad)


class OddsTests(unittest.TestCase):
    def test_implied_probs_are_normalised(self):
        probs = Odds(home=2.0, away=2.0).implied_probs()
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertAlmostEqual(probs[0], 0.5)

    def test_implied_probs_reflect_favourite(self):
        home_p, away_p = Odds(home=1.5, away=3.0).implied_probs()
        self.assertGreater(home_p, away_p)

    def test_to_dict_is_json_native(self):
        payload = Odds(home=1.5, away=3.0).to_dict()
        json.dumps(payload)
        self.assertEqual(payload, {'home': 1.5, 'away': 3.0})


if __name__ == '__main__':
    unittest.main()
