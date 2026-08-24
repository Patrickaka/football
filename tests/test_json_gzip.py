"""JSON 响应压缩：阈值、协商、Vary 声明与失败回退"""

import gzip
import json
import unittest
from unittest.mock import patch

import server
import src.webapp.routing as routing


class _FakeHeaders(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


def _handler(accept_encoding='gzip, deflate'):
    handler = server.Handler.__new__(server.Handler)
    handler._log = server.log
    handler.headers = _FakeHeaders({'Accept-Encoding': accept_encoding} if accept_encoding else {})
    return handler


class GzipNegotiationTests(unittest.TestCase):
    def test_detects_gzip_support(self):
        self.assertTrue(_handler('gzip, deflate')._accepts_gzip())
        self.assertTrue(_handler('GZIP')._accepts_gzip())
        self.assertFalse(_handler('deflate, br')._accepts_gzip())
        self.assertFalse(_handler(None)._accepts_gzip())

    def test_large_body_is_compressed_and_round_trips(self):
        body = json.dumps({'rows': [{'i': i, 'name': '球队名称'} for i in range(500)]},
                          ensure_ascii=False).encode('utf-8')
        out, compressed = _handler()._maybe_gzip(body)
        self.assertTrue(compressed)
        self.assertLess(len(out), len(body))
        self.assertEqual(gzip.decompress(out), body, '解压后必须与原始字节一致')

    def test_small_body_is_left_alone(self):
        body = b'{"ok":true}'
        out, compressed = _handler()._maybe_gzip(body)
        self.assertFalse(compressed)
        self.assertIs(out, body)

    def test_client_without_gzip_gets_plain_body(self):
        body = b'x' * (routing.JSON_GZIP_MIN_BYTES + 100)
        out, compressed = _handler('identity')._maybe_gzip(body)
        self.assertFalse(compressed)
        self.assertIs(out, body)

    def test_compression_failure_falls_back_to_plain(self):
        """压缩出问题不能让整个请求失败，退回不压缩即可"""
        body = b'y' * (routing.JSON_GZIP_MIN_BYTES + 100)
        with patch.object(routing.gzip, 'compress', side_effect=RuntimeError('boom')):
            out, compressed = _handler()._maybe_gzip(body)
        self.assertFalse(compressed)
        self.assertIs(out, body)


class ServeJsonHeaderTests(unittest.TestCase):
    def _capture(self, payload, accept_encoding='gzip'):
        handler = _handler(accept_encoding)
        # 按发送顺序记成列表而不是字典：重复的 Content-Length 是协议错误，
        # 用字典收集会被后写的那个盖掉，测不出来。
        emitted = []
        sent = {'emitted': emitted, 'body': None, 'status': None}
        handler.send_response = lambda code: sent.__setitem__('status', code)
        handler.send_header = lambda k, v: emitted.append((k, v))
        handler.end_headers = lambda: None
        handler.wfile = type('W', (), {'write': lambda _self, b: sent.__setitem__('body', b)})()
        handler._serve_json(payload)
        sent['headers'] = dict(emitted)
        return sent

    @staticmethod
    def _header_count(sent, name):
        return sum(1 for k, _ in sent['emitted'] if k.lower() == name.lower())

    def test_compressed_response_declares_encoding_and_vary(self):
        sent = self._capture({'rows': [{'i': i, 'name': '球队'} for i in range(500)]})
        self.assertEqual(sent['headers']['Content-Encoding'], 'gzip')
        self.assertEqual(sent['headers']['Vary'], 'Accept-Encoding')

    def test_content_length_matches_the_bytes_actually_written(self):
        """压缩后必须按压缩后的长度报 Content-Length，否则客户端会挂住"""
        sent = self._capture({'rows': [{'i': i, 'name': '球队'} for i in range(500)]})
        self.assertEqual(int(sent['headers']['Content-Length']), len(sent['body']))
        self.assertEqual(self._header_count(sent, 'Content-Length'), 1,
                         'Content-Length 只能发一次，发两个是协议错误')

    def test_plain_response_declares_no_encoding(self):
        sent = self._capture({'ok': True}, accept_encoding=None)
        self.assertNotIn('Content-Encoding', sent['headers'])
        self.assertEqual(int(sent['headers']['Content-Length']), len(sent['body']))

    def test_payload_survives_the_round_trip(self):
        payload = {'result': {'recommendations': [{'match_id': str(i)} for i in range(300)]}}
        sent = self._capture(payload)
        self.assertEqual(json.loads(gzip.decompress(sent['body']).decode('utf-8')), payload)


if __name__ == '__main__':
    unittest.main()
