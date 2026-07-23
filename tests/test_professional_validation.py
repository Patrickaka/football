import unittest

from src.football.professional_validation import (
    evaluate_strategy,
    walk_forward_evaluate,
)


class ProfessionalValidationTests(unittest.TestCase):
    def test_roi_and_clv_are_settled_from_offered_prices(self):
        rows = [
            {'actual': 'H', 'probabilities': {'H': .60, 'D': .22, 'A': .18},
             'odds': {'H': 2.0, 'D': 3.5, 'A': 4.0},
             'closing_odds': {'H': 1.8, 'D': 3.6, 'A': 4.2}},
            {'actual': 'A', 'probabilities': {'H': .62, 'D': .20, 'A': .18},
             'odds': {'H': 2.0, 'D': 3.5, 'A': 4.0},
             'closing_odds': {'H': 1.9, 'D': 3.6, 'A': 4.2}},
        ]
        result = evaluate_strategy(rows, min_edge=.05)
        self.assertEqual(result['bets'], 2)
        self.assertAlmostEqual(result['roi'], 0.0)
        self.assertGreater(result['mean_clv'], 0)

    def test_walk_forward_never_uses_future_as_training(self):
        rows = []
        for index in range(18):
            rows.append({
                'date': f'2025-01-{index + 1:02d}',
                'match_id': str(index),
                'actual': 'H' if index % 2 == 0 else 'A',
                'probabilities': {'H': .60, 'D': .20, 'A': .20},
                'odds': {'H': 2.0, 'D': 3.5, 'A': 4.0},
            })
        report = walk_forward_evaluate(rows, initial_train=10, test_size=4, min_training_bets=1)
        self.assertEqual(report['out_of_sample_n'], 8)
        self.assertEqual(report['folds'][0]['train_n'], 10)
        self.assertEqual(report['folds'][1]['train_n'], 14)


if __name__ == '__main__':
    unittest.main()
