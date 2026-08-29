# -*- coding: utf-8 -*-
"""网页入口：单页应用本体与它的历史锚点入口。"""
import pathlib
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthSettings
from src.api.routers import pages


def make_client(credentials=None):
    return TestClient(create_app(
        auth_settings=AuthSettings(credentials=credentials or {})))


class IndexPage(unittest.TestCase):

    def test_it_serves_the_single_page_app(self):
        with make_client() as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('预测助手', response.text)

    def test_it_is_never_cached(self):
        """**缓存住了，用户会一直拿着旧版前端去打新版接口。**
        旧入口三个头一个不少，这里照抄。
        """
        with make_client() as client:
            headers = client.get('/').headers
            self.assertEqual(headers['cache-control'],
                             'no-cache, no-store, must-revalidate')
            self.assertEqual(headers['pragma'], 'no-cache')
            self.assertEqual(headers['expires'], '0')

    def test_a_missing_index_is_a_500_not_a_crash(self):
        # `PosixPath.read_text` 是只读属性，patch 不上——换掉整个路径对象。
        missing = pathlib.Path('/没有这个目录/index.html')
        with mock.patch.object(pages, 'INDEX_FILE', missing):
            with make_client() as client:
                response = client.get('/')
                self.assertEqual(response.status_code, 500)
                self.assertIn('缺失', response.text)

    def test_the_home_page_needs_a_session(self):
        """首页**不在豁免清单里**——豁免它等于整个界面裸奔。"""
        with make_client({'a': 'b'}) as client:
            response = client.get('/', headers={'accept': 'text/html'},
                                  follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers['location'], '/login')


class SsqRedirect(unittest.TestCase):

    def test_it_redirects_to_the_anchor(self):
        with make_client() as client:
            response = client.get('/ssq', follow_redirects=False)
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers['location'], './#ssq')

    def test_the_target_is_relative(self):
        """**线上反代把服务挂在 `/football/` 下并剥掉了前缀**，应用自己
        看不到这一段。写死 `/#ssq` 会把访问者甩到站点根目录。

        旧入口靠嗅探 `route.path` 是否以 `/football/` 开头来补前缀——
        那在剥了前缀的新入口下永远为假，等于没有。
        """
        with make_client() as client:
            location = client.get('/ssq', follow_redirects=False).headers['location']
            self.assertFalse(location.startswith('/'))


if __name__ == '__main__':
    unittest.main()
