import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.football import ml_trainer


class _FakeModel:
    pass


class FootballModelArtifactTests(unittest.TestCase):
    def test_save_writes_portable_versioned_metadata(self):
        trainer = ml_trainer.MLModelTrainer()
        trainer.model = _FakeModel()
        trainer.feature_names = ['elo_home']
        trainer.metadata = {'train_count': 10, 'validation_count': 2, 'test_count': 3}
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


if __name__ == '__main__':
    unittest.main()
