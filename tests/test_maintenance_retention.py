import os
import logging
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from src.common import maintenance
from src.common.logger import _SafeRotatingFileHandler


class MaintenanceRetentionTests(unittest.TestCase):
    def test_disk_status_has_warning_and_critical_levels(self):
        usage = namedtuple('usage', 'total used free')
        total = 20 * 1024 ** 3
        with patch.object(maintenance, 'DISK_WARNING_FREE_GB', 5), \
             patch.object(maintenance, 'DISK_WARNING_FREE_PERCENT', 15), \
             patch.object(maintenance, 'DISK_MIN_FREE_GB', 2), \
             patch.object(maintenance, 'DISK_MIN_FREE_PERCENT', 10), \
             patch.object(maintenance.shutil, 'disk_usage', return_value=usage(
                 total, total - 4 * 1024 ** 3, 4 * 1024 ** 3,
             )):
            warning = maintenance.disk_status()
        self.assertEqual(warning['pressure_level'], 'warning')
        self.assertTrue(warning['under_pressure'])
        self.assertFalse(warning['critical'])

        with patch.object(maintenance.shutil, 'disk_usage', return_value=usage(
            total, total - 1024 ** 3, 1024 ** 3,
        )):
            critical = maintenance.disk_status()
        self.assertEqual(critical['pressure_level'], 'critical')
        self.assertTrue(critical['critical'])

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

    def test_zero_day_retention_removes_recent_regenerable_files_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root / 'fresh.tmp'
            protected = root / 'history.json'
            generated.write_text('generated', encoding='utf-8')
            protected.write_text('business data', encoding='utf-8')
            with patch.object(maintenance, 'PROJECT_ROOT', root), \
                 patch.object(maintenance, 'REGENERABLE_TARGETS', ((root, ('*.tmp',)),)):
                result = maintenance.cleanup_regenerable_artifacts(0)
            self.assertEqual(result['removed_count'], 1)
            self.assertFalse(generated.exists())
            self.assertTrue(protected.exists())

    def test_disk_pressure_uses_emergency_retention(self):
        pressured = {
            'free_gb': 0.1, 'free_percent': 0.5, 'under_pressure': True,
            'total_bytes': 100, 'used_bytes': 99, 'free_bytes': 1,
        }
        recovered = {**pressured, 'free_gb': 5.0, 'free_percent': 20.0,
                     'under_pressure': False}
        with patch.object(maintenance, 'disk_status', side_effect=[pressured, recovered]), \
             patch.object(maintenance, 'purge_binlogs', return_value=True) as purge, \
             patch.object(maintenance, 'cleanup_rotated_logs', return_value=4) as logs, \
             patch.object(maintenance, 'cleanup_regenerable_artifacts', return_value={
                 'removed_count': 3, 'bytes_freed': 1024, 'errors': [],
             }) as artifacts:
            result = maintenance.run_maintenance()

        purge.assert_called_once_with(maintenance.EMERGENCY_BINLOG_RETENTION_DAYS)
        logs.assert_called_once_with(0)
        artifacts.assert_called_once_with(maintenance.EMERGENCY_ARTIFACT_RETENTION_DAYS)
        self.assertTrue(result['emergency'])
        self.assertEqual(result['disk_after'], recovered)

    def test_force_emergency_cleans_even_when_project_disk_looks_healthy(self):
        healthy = {
            'free_gb': 50.0, 'free_percent': 50.0, 'under_pressure': False,
            'total_bytes': 100, 'used_bytes': 50, 'free_bytes': 50,
        }
        with patch.object(maintenance, 'disk_status', side_effect=[healthy, healthy]), \
             patch.object(maintenance, 'purge_binlogs', return_value=True) as purge, \
             patch.object(maintenance, 'cleanup_rotated_logs', return_value=0) as logs, \
             patch.object(maintenance, 'cleanup_regenerable_artifacts', return_value={
                 'removed_count': 0, 'bytes_freed': 0, 'errors': [],
             }) as artifacts:
            result = maintenance.run_maintenance(force_emergency=True)

        purge.assert_called_once_with(maintenance.EMERGENCY_BINLOG_RETENTION_DAYS)
        logs.assert_called_once_with(0)
        artifacts.assert_called_once_with(maintenance.EMERGENCY_ARTIFACT_RETENTION_DAYS)
        self.assertTrue(result['emergency'])

    def test_warning_pressure_uses_staged_retention(self):
        warning = {
            'free_gb': 4.0, 'free_percent': 12.0, 'under_pressure': True,
            'critical': False, 'pressure_level': 'warning',
            'total_bytes': 100, 'used_bytes': 88, 'free_bytes': 12,
        }
        healthy = {**warning, 'free_gb': 20.0, 'free_percent': 20.0,
                   'under_pressure': False, 'pressure_level': 'healthy'}
        with patch.object(maintenance, 'disk_status', return_value=healthy), \
             patch.object(maintenance, 'purge_binlogs', return_value=True) as purge, \
             patch.object(maintenance, 'cleanup_rotated_logs', return_value=1) as logs, \
             patch.object(maintenance, 'cleanup_regenerable_artifacts', return_value={
                 'removed_count': 1, 'bytes_freed': 1024, 'errors': [],
             }) as artifacts:
            result = maintenance.run_maintenance(status=warning)

        purge.assert_called_once_with(maintenance.PRESSURE_BINLOG_RETENTION_DAYS)
        logs.assert_called_once_with(maintenance.PRESSURE_ARTIFACT_RETENTION_DAYS)
        artifacts.assert_called_once_with(maintenance.PRESSURE_ARTIFACT_RETENTION_DAYS)
        self.assertFalse(result['emergency'])
        self.assertEqual(result['pressure_level'], 'warning')

    def test_log_handler_has_bounded_backup_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'football.log'
            handler = _SafeRotatingFileHandler(
                path, maxBytes=128, backupCount=2, encoding='utf-8'
            )
            try:
                for index in range(30):
                    handler.emit(logging.LogRecord(
                        name='test', level=logging.INFO, pathname=__file__, lineno=1,
                        msg='line-%s-%s', args=(index, 'x' * 30), exc_info=None,
                    ))
            finally:
                handler.close()

            self.assertTrue(path.exists())
            self.assertLessEqual(len(list(Path(temp).glob('football.log.*'))), 2)

    def test_only_allowlisted_oversized_active_logs_are_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            football_log = root / 'football.log'
            redirected_log = root / 'server_run.log'
            protected_log = root / 'audit.log'
            for path in (football_log, redirected_log, protected_log):
                path.write_bytes(b'x' * 256)

            with patch.object(maintenance, 'LOG_DIR', root):
                result = maintenance.truncate_oversized_active_logs(128)

            self.assertEqual(result['truncated_count'], 2)
            self.assertEqual(football_log.stat().st_size, 0)
            self.assertEqual(redirected_log.stat().st_size, 0)
            self.assertEqual(protected_log.stat().st_size, 256)


if __name__ == '__main__':
    unittest.main()
