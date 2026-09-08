import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.football import ml_trainer
from src.football.ml_feature_schema import get_feature_names


class _FakeModel:
    n_features_in_ = 40
    classes_ = [0, 1, 2]

    def predict_proba(self, features):
        return [[0.4, 0.3, 0.3]]


class FootballModelArtifactTests(unittest.TestCase):
    def test_save_writes_portable_versioned_metadata(self):
        trainer = ml_trainer.MLModelTrainer()
        trainer.model = _FakeModel()
        trainer.feature_names = get_feature_names()
        trainer.metadata = {'train_count': 10, 'validation_count': 2, 'test_count': 3,
                            'train_end': '2025-01-01T00:00:00+00:00',
                            'validation_end': '2025-02-01T00:00:00+00:00',
                            'test_end': '2025-03-01T00:00:00+00:00'}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(ml_trainer, 'DATA_DIR', temp), \
                 patch.object(ml_trainer, 'MODEL_FILE', os.path.join(temp, 'model.pkl')), \
                 patch.object(ml_trainer, 'METADATA_FILE', os.path.join(temp, 'metadata.json')), \
                 patch.object(ml_trainer, 'TRAINING_DATA_FILE', os.path.join(temp, 'missing.jsonl')), \
                 patch.object(ml_trainer.kv_store, 'save'):
                trainer.save({'logloss': 1.0})
                with open(ml_trainer.METADATA_FILE, encoding='utf-8') as handle:
                    metadata = json.load(handle)
        self.assertEqual(metadata['feature_version'], 'v2')
        self.assertEqual(metadata['dataset']['split_method'], 'chronological-70-15-15')
        self.assertEqual(metadata['test_count'], 3)
        self.assertEqual(metadata['feature_builder_version'], 'v2-strict-2')
        self.assertEqual(len(metadata['model_sha256']), 64)
        self.assertGreater(metadata['training_cutoff_at'], metadata['split_dates']['test_end'])


if __name__ == '__main__':
    unittest.main()
