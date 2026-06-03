"""
足球比分预测 - 网页服务
========================
标准库 http.server 实现，零第三方依赖。

运行：python3 server.py
然后浏览器打开 http://localhost:8000
"""

import os
import sys
import json
import hmac
import base64
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import football

sys.stdout.reconfigure(encoding='utf-8')

HOST = '0.0.0.0'  # 监听所有网卡，局域网/公网（经端口转发或隧道）可访问
PORT = int(os.environ.get('FOOTBALL_PORT', '9000'))
INDEX_FILE = Path(__file__).parent / 'index.html'

# 公网暴露时务必设置：导出 FOOTBALL_USER / FOOTBALL_PASS 启用 HTTP Basic 鉴权
AUTH_USER = os.environ.get('FOOTBALL_USER', '')
AUTH_PASS = os.environ.get('FOOTBALL_PASS', '')
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self._authorized():
            return
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

    def _authorized(self):
        """启用鉴权时校验 HTTP Basic 凭据；未启用则放行"""
        if not AUTH_ENABLED:
            return True
        header = self.headers.get('Authorization', '')
        if header.startswith('Basic '):
            try:
                user, _, pwd = base64.b64decode(header[6:]).decode('utf-8').partition(':')
                ok = hmac.compare_digest(user, AUTH_USER) & hmac.compare_digest(pwd, AUTH_PASS)
                if ok:
                    return True
            except (ValueError, UnicodeDecodeError):
                pass
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="football"')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return False

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


def _is_private_lan(ip):
    """是否为常见家庭/办公局域网段（排除代理/VPN 虚拟段如 198.18.x）"""
    if ip.startswith('192.168.') or ip.startswith('10.'):
        return True
    parts = ip.split('.')
    return len(parts) == 4 and parts[0] == '172' and parts[1].isdigit() and 16 <= int(parts[1]) <= 31


def _candidate_ips():
    """收集本机所有非回环 IPv4，私有局域网段排在前面"""
    ips = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(('10.255.255.255', 1))
            ips.add(s.getsockname()[0])
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips.discard('127.0.0.1')
    return sorted(ips, key=lambda ip: (not _is_private_lan(ip), ip))


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    local_url = f'http://localhost:{PORT}'
    candidates = _candidate_ips()
    print("=" * 50)
    print("  足球比分预测 - 网页服务已启动")
    print(f"  本机访问:   {local_url}")
    if candidates:
        print(f"  局域网访问: http://{candidates[0]}:{PORT}  （手机/其它设备用这个）")
        if len(candidates) > 1:
            others = '  '.join(f'http://{ip}:{PORT}' for ip in candidates[1:])
            print(f"  其它候选地址（若上面连不上，依次试）: {others}")
    if AUTH_ENABLED:
        print(f"  鉴权: 已启用 HTTP Basic（用户名 {AUTH_USER}）")
    else:
        print("  鉴权: 未启用 ⚠ 公网暴露前请设置 FOOTBALL_USER / FOOTBALL_PASS")
    print("  公网访问: 用隧道（cloudflared/ngrok）或路由器端口转发，详见 README")
    print("  按 Ctrl+C 停止")
    print("=" * 50)
    try:
        webbrowser.open(local_url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止。")
        server.shutdown()


if __name__ == '__main__':
    main()
