# -*- coding: utf-8 -*-
"""内存诊断端点。

这个端点存在的理由是区分两件会被混为一谈的事：常驻结构大，和堆碎片多。
所以测试盯住三点——深度遍历不能把整个解释器算进某条记录、单个探针挂掉
不能让整个端点 500、以及 unaccounted 的算术是 RSS 减去已追踪部分。
"""
import unittest
from contextlib import ExitStack
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.routers import diagnostics


def make_client():
    return TestClient(create_app(auth_settings=AuthSettings(credentials={})))


class DeepSize(unittest.TestCase):
    def test_counts_nested_content_not_just_the_container(self):
        payload = {'a': ['x' * 5000]}
        self.assertGreater(diagnostics._deep_size(payload), 5000)

    def test_survives_reference_cycles(self):
        node = {}
        node['self'] = node
        self.assertGreater(diagnostics._deep_size(node), 0)

    def test_counts_shared_objects_once(self):
        shared = ['y' * 5000]
        twice = diagnostics._deep_size([shared, shared])
        once = diagnostics._deep_size([shared])
        self.assertLess(twice - once, 1000)

    def test_does_not_traverse_into_modules_or_types(self):
        """跟进模块/类就会把整个解释器算成这条记录的体积。"""
        self.assertLess(diagnostics._deep_size([unittest, dict, len]), 1000)


class OwnSize(unittest.TestCase):
    """堆里有惰性模块代理（six.MovedModule），任意属性访问都会触发 import。

    诊断只是读一眼内存，不该有副作用，更不该因为某个对象的属性访问抛异常
    就让整个端点 500——这正是它第一次上线时踩的坑。
    """

    def test_does_not_touch_instance_attributes(self):
        class Landmine:
            def __getattr__(self, name):
                raise ModuleNotFoundError(f'属性访问触发了副作用: {name}')

        self.assertGreater(diagnostics._own_size(Landmine()), 0)

    def test_measures_numpy_arrays_by_nbytes(self):
        import numpy

        self.assertGreaterEqual(
            diagnostics._own_size(numpy.zeros(1000, dtype=numpy.float64)), 8000)

    def test_survives_a_broken_sizeof(self):
        class Broken:
            def __sizeof__(self):
                raise RuntimeError('boom')

        self.assertEqual(diagnostics._own_size(Broken()), 0)


class ProcessStatus(unittest.TestCase):
    def test_missing_proc_yields_nulls_instead_of_raising(self):
        with mock.patch('builtins.open', side_effect=OSError):
            status = diagnostics._process_status()
        self.assertIsNone(status['rss_bytes'])
        self.assertIn('note', status)

    def test_parses_kilobyte_fields_into_bytes(self):
        raw = 'VmRSS:\t  1024 kB\nVmSwap:\t     0 kB\nThreads:\t19\n'
        with mock.patch('builtins.open', mock.mock_open(read_data=raw)):
            status = diagnostics._process_status()
        self.assertEqual(status['rss_bytes'], 1024 * 1024)
        self.assertEqual(status['swap_bytes'], 0)
        self.assertEqual(status['threads'], 19)


class SectionIsolation(unittest.TestCase):
    def test_a_failing_probe_is_reported_not_raised(self):
        section = diagnostics._section(mock.Mock(side_effect=RuntimeError('boom')))
        self.assertFalse(section['available'])
        self.assertIn('boom', section['error'])



class Payload(unittest.TestCase):
    """组装逻辑不碰真实的预测历史——那会在 import 期连库加载整表。"""

    def _payload(self, rss, **sections):
        stubs = {
            '_prediction_history_section': {'bytes': 0},
            '_kl8_trials_section': {'bytes': 0},
            '_ml_model_section': {'bytes': 0},
            '_cache_l1_section': {'bytes': 0},
        }
        stubs.update(sections)
        status = {'rss_bytes': rss, 'swap_bytes': 0, 'threads': 1}
        with ExitStack() as stack:
            for name, value in stubs.items():
                stack.enter_context(
                    mock.patch.object(diagnostics, name, return_value=value))
            stack.enter_context(
                mock.patch.object(diagnostics, '_process_status', return_value=status))
            return diagnostics.memory_payload(state=None)

    def test_unaccounted_is_rss_minus_tracked(self):
        payload = self._payload(1000, _kl8_trials_section={'bytes': 400})
        self.assertEqual(payload['tracked_total_bytes'], 400)
        self.assertEqual(payload['unaccounted_bytes'], 600)

    def test_unaccounted_is_null_when_rss_is_unavailable(self):
        payload = self._payload(None, _kl8_trials_section={'bytes': 400})
        self.assertIsNone(payload['unaccounted_bytes'])


class Endpoint(unittest.TestCase):
    def _get(self, url):
        stub = {'bytes': 0}
        with mock.patch.object(diagnostics, '_prediction_history_section', return_value=stub), \
             mock.patch.object(diagnostics, '_kl8_trials_section', return_value=stub), \
             mock.patch.object(diagnostics, '_ml_model_section', return_value=stub):
            with make_client() as client:
                return client.get(url)

    def test_returns_the_expected_shape(self):
        body = self._get('/api/diagnostics/memory').json()
        self.assertIn('process', body)
        self.assertIn('prediction_history', body['tracked'])
        self.assertIn('unaccounted_bytes', body)
        self.assertTrue(body['notes'])

    def test_gc_histogram_is_off_by_default(self):
        self.assertIsNone(self._get('/api/diagnostics/memory').json()['gc_histogram'])

    def test_gc_histogram_is_opt_in(self):
        histogram = self._get('/api/diagnostics/memory?gc=1').json()['gc_histogram']
        self.assertTrue(histogram)
        self.assertIn('type', histogram[0])
