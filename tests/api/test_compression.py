# -*- coding: utf-8 -*-
"""响应压缩。

**切换入口时漏掉的能力**：旧入口一直在压（北单整页 332 KB → 45 KB，
一次批量预测 456 KB），新入口没有——那是手机端能直接感觉到的降级，
而且不会有任何报错。

阈值与压缩级别照抄旧入口：小于 1 KB 不压（压缩头本身有开销，
小响应压了反而更大），级别 6。
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.deps import Settings
from src.api.services import football as service

BIG = {'result': [{'league': '英超' * 8, 'index': i} for i in range(400)]}
SMALL = {'ok': 1}


def make_client(**settings):
    return TestClient(create_app(settings=Settings(**settings),
                                 auth_settings=AuthSettings(credentials={})))


class Compression(unittest.TestCase):

    def _fetch(self, payload, encoding='gzip', **settings):
        with mock.patch.object(service, 'sync_status_payload', return_value=payload):
            with make_client(**settings) as client:
                return client.get('/api/sync/status',
                                  headers={'Accept-Encoding': encoding})

    def test_a_large_response_is_compressed(self):
        self.assertEqual(self._fetch(BIG).headers.get('content-encoding'), 'gzip')

    def test_a_small_response_is_not(self):
        """**小响应压了反而不划算**——压缩头本身就有开销。"""
        self.assertIsNone(self._fetch(SMALL).headers.get('content-encoding'))

    def test_a_client_that_does_not_want_gzip_gets_plain_bytes(self):
        self.assertIsNone(self._fetch(BIG, encoding='identity').headers.get('content-encoding'))

    def test_the_content_survives_the_round_trip(self):
        """压缩不能改内容——httpx 会自动解压，比对解压后的 JSON。"""
        self.assertEqual(self._fetch(BIG).json(), BIG)

    def test_it_actually_saves_a_lot(self):
        """这类 JSON 高度重复，**压缩比应该到数倍以上**，不是几个百分点。

        `TestClient` 会自动解压，`len(response.content)` 拿到的是解压后的
        长度——要量压缩效果只能自己压一遍比。
        """
        import gzip
        import json

        plain = json.dumps(BIG, ensure_ascii=False).encode('utf-8')
        compressed = gzip.compress(plain, 6)
        self.assertLess(len(compressed) * 3, len(plain),
                        f'压缩后 {len(compressed)} 字节 vs 原始 {len(plain)}')
        self.assertEqual(self._fetch(BIG).headers.get('content-encoding'), 'gzip')

    def test_the_threshold_is_configurable(self):
        """把阈值抬高，同一个响应就不该再被压——证明配置真的接上了。"""
        self.assertIsNone(
            self._fetch(BIG, gzip_min_bytes=10_000_000).headers.get('content-encoding'))

    def test_settings_read_the_documented_environment_variables(self):
        settings = Settings.from_env({'JSON_GZIP_MIN_BYTES': '2048', 'JSON_GZIP_LEVEL': '9'})
        self.assertEqual(settings.gzip_min_bytes, 2048)
        self.assertEqual(settings.gzip_level, 9)

    def test_the_defaults_match_the_old_entry_point(self):
        """旧入口的默认值是 1024 / 6，换个数字就是悄悄改了线上行为。"""
        settings = Settings.from_env({})
        self.assertEqual((settings.gzip_min_bytes, settings.gzip_level), (1024, 6))


class MiddlewareOrder(unittest.TestCase):
    """**GZip 必须紧贴路由（最先注册 = 最内层）。**

    `@app.middleware('http')` 加的是 `BaseHTTPMiddleware`，它把响应转成流式、
    丢掉 `Content-Length`。GZip 排在它外面就拿不到长度，只能一律压缩——
    `minimum_size` 形同虚设，8 字节的 `{"ok":1}` 也会被压。
    第一版就是这么写的，小响应全被压了。
    """

    def test_gzip_is_registered_before_the_http_middlewares(self):
        import pathlib
        source = pathlib.Path('src/api/app.py').read_text(encoding='utf-8')
        self.assertLess(source.index('add_middleware(GZipMiddleware'),
                        source.index('install_auth(app)'),
                        'GZip 要最先注册，否则拿不到 Content-Length')

    def test_a_small_body_stays_uncompressed_with_all_middlewares_on(self):
        """端到端复现那个 bug：鉴权与限流都开着，小响应仍不该被压。"""
        with mock.patch.object(service, 'sync_status_payload', return_value=SMALL):
            client = TestClient(create_app(
                settings=Settings(rate_limit_per_sec=100),
                auth_settings=AuthSettings(credentials={'a': 'b'})))
            with client:
                client.post('/auth/login', json={'user': 'a', 'password': 'b'})
                response = client.get('/api/sync/status',
                                      headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get('content-encoding'))


if __name__ == '__main__':
    unittest.main()
