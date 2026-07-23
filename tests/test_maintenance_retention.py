import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import maintenance


class MaintenanceRetentionTests(unittest.TestCase):
    def test_only_allowlisted_old_artifacts_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / 'cache'
            cache.mkdir()
            old_cache = cache / 'old.pkl'
            current_cache = cache / 'new.pkl'
            protected = cache / 'history.json'
            for path in (old_cache, current_cache, protected):
                path.write_text('x', encoding='utf-8')
            old_time = time.time() - 5 * 86400
            os.utime(old_cache, (old_time, old_time))
            os.utime(protected, (old_time, old_time))
            targets = ((cache, ('*.pkl',)),)
            with patch.object(maintenance, 'PROJECT_ROOT', root), \
                 patch.object(maintenance, 'REGENERABLE_TARGETS', targets):
                result = maintenance.cleanup_regenerable_artifacts(3)
            self.assertEqual(result['removed_count'], 1)
            self.assertFalse(old_cache.exists())
            self.assertTrue(current_cache.exists())
            self.assertTrue(protected.exists())

    def test_dry_run_never_deletes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_file = root / 'old.tmp'
            old_file.write_text('x', encoding='utf-8')
            old_time = time.time() - 5 * 86400
            os.utime(old_file, (old_time, old_time))
            with patch.object(maintenance, 'PROJECT_ROOT', root), \
                 patch.object(maintenance, 'REGENERABLE_TARGETS', ((root, ('*.tmp',)),)):
                result = maintenance.cleanup_regenerable_artifacts(3, dry_run=True)
            self.assertEqual(result['removed_count'], 1)
            self.assertTrue(old_file.exists())


if __name__ == '__main__':
    unittest.main()
