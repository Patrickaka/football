import json
import os
import unittest

from src.football.ml_feature_schema import (
    FEATURE_VERSION,
    audit_feature_payload,
    get_feature_names,
    validate_features,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FootballFeatureContractTests(unittest.TestCase):
    def test_training_payload_matches_online_schema(self):
        path = os.path.join(ROOT, 'data', 'ml_training_data.jsonl')
        with open(path, encoding='utf-8') as handle:
            sample = json.loads(next(handle))

        audit = audit_feature_payload(sample['features'])
        self.assertTrue(audit['complete'], audit)
        self.assertEqual(FEATURE_VERSION, 'v2')

    def test_home_away_features_are_not_replaced_by_defaults(self):
        names = get_feature_names()
        self.assertIn('home_h_goals_for_5', names)
        payload = {name: 0 for name in names}
        payload['home_h_goals_for_5'] = 2.75
        validated = validate_features(payload)
        self.assertEqual(validated['home_h_goals_for_5'], 2.75)


if __name__ == '__main__':
    unittest.main()
