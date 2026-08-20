import unittest

from src.football.professional_validation import (
    blend_record_with_market,
    evaluate_rqspf_records,
    evaluate_strategy,
    select_market_residual_weight,
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

    def test_unproven_residual_falls_back_to_market_probability(self):
        rows = []
        for index in range(40):
            actual = 'H' if index % 2 == 0 else 'A'
            rows.append({
                'actual': actual,
                'probabilities': {'H': .10 if actual == 'H' else .80, 'D': .10,
                                  'A': .80 if actual == 'H' else .10},
                'odds': {'H': 2.0 if actual == 'H' else 5.0, 'D': 8.0,
                         'A': 2.0 if actual == 'A' else 5.0},
            })
        selected = select_market_residual_weight(rows)
        deployed = blend_record_with_market(rows[0], selected['weight'])

        self.assertEqual(selected['weight'], 0.0)
        self.assertEqual(selected['reason'], 'model_residual_not_proven')
        self.assertEqual(deployed['probabilities'], deployed['market_probabilities'])

    def test_market_only_probabilities_do_not_create_fake_zero_edge_bets(self):
        row = {
            'actual': 'H',
            'probabilities': {'H': .50, 'D': .25, 'A': .25},
            'odds': {'H': 2.0, 'D': 4.0, 'A': 4.0},
        }
        self.assertEqual(evaluate_strategy([row], min_edge=0.0)['bets'], 0)

    def test_rqspf_is_settled_against_its_own_handicap_and_official_odds(self):
        records = [{
            'actual_score': '2-1',
            'lottery_handicap': -1,
            'predicted_rqspf': {'让胜': .10, '让平': .70, '让负': .20},
            'odds_snapshot': {'lottery': {
                'rqspf_odds': {'让胜': 3.5, '让平': 3.2, '让负': 2.0},
            }},
        }]
        report = evaluate_rqspf_records(records, min_probability=.65)

        self.assertEqual(report['n'], 1)
        self.assertEqual(report['accuracy'], 1.0)
        self.assertEqual(report['strategy']['bets'], 1)
        self.assertFalse(report['production_ready'])


if __name__ == '__main__':
    unittest.main()
