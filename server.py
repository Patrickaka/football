"""
足球比分预测 - 网页服务
========================
标准库 http.server 实现，零第三方依赖。

运行：python3 server.py
然后浏览器打开 http://localhost:8000
"""

import sys
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import football

sys.stdout.reconfigure(encoding='utf-8')

HOST = '127.0.0.1'
PORT = 8000
INDEX_FILE = Path(__file__).parent / 'index.html'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path)
        if route.path == '/':
            self._serve_index()
        elif route.path == '/api/matches':
            self._serve_json(self._matches_payload())
        elif route.path == '/api/predict':
            params = parse_qs(route.query)
            self._serve_json(self._predict_payload(params))
        else:
            self._send(404, 'text/plain; charset=utf-8', b'Not Found')

    def _serve_index(self):
        try:
            body = INDEX_FILE.read_bytes()
        except OSError:
            self._send(500, 'text/plain; charset=utf-8', 'index.html 缺失'.encode('utf-8'))
            return
        self._send(200, 'text/html; charset=utf-8', body)

    def _serve_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self._send(200, 'application/json; charset=utf-8', body)

    def _matches_payload(self):
        try:
            return {'matches': football.fetch_match_list()}
        except Exception as e:
            return {'error': f'获取比赛列表失败: {e}'}

    def _predict_payload(self, params):
        match_id = params.get('match_id', [''])[0]
        if not match_id:
            return {'error': '缺少 match_id 参数'}
        match = {
            'match_id': match_id,
            'home': params.get('home', [''])[0],
            'away': params.get('away', [''])[0],
            'league': params.get('league', [''])[0],
            'time': params.get('time', [''])[0],
        }
        try:
            return {'result': football.analyze_match(match)}
        except Exception as e:
            return {'error': f'赔率分析失败: {e}'}

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} - {fmt % args}\n")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f'http://localhost:{PORT}'
    print("=" * 50)
    print("  足球比分预测 - 网页服务已启动")
    print(f"  访问地址: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止。")
        server.shutdown()


if __name__ == '__main__':
    main()
