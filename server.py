"""
预测服务 - 网页服务
========================
标准库 http.server 实现，零第三方依赖。
集成：足球比分预测 + 福彩3D预测

运行：python3 server.py
然后浏览器打开 http://localhost:9000
"""

import os
import sys
import json
import hmac
import base64
import socket
import webbrowser
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import football
import lottery3d
import lottery3d_ml

_ROOT = Path(__file__).parent
INDEX_FILE = _ROOT / 'index.html'

sys.stdout.reconfigure(encoding='utf-8')

HOST = '0.0.0.0'  # 监听所有网卡，局域网/公网（经端口转发或隧道）可访问
PORT = int(os.environ.get('FOOTBALL_PORT', '9004'))

# 公网暴露时务必设置鉴权。两种方式（可并用）：
#   多用户: FOOTBALL_USERS="alice:pass1,bob:pass2"
#   单用户: FOOTBALL_USER=alice FOOTBALL_PASS=pass1
def _load_credentials():
    """解析鉴权凭据为 {用户名: 密码}；无任何配置则返回空（不启用鉴权）"""
    creds = {}
    for pair in os.environ.get('FOOTBALL_USERS', '').split(','):
        user, sep, pwd = pair.strip().partition(':')
        if sep and user.strip() and pwd.strip():
            creds[user.strip()] = pwd.strip()
    single_user = os.environ.get('FOOTBALL_USER', '').strip()
    single_pass = os.environ.get('FOOTBALL_PASS', '').strip()
    if single_user and single_pass:
        creds.setdefault(single_user, single_pass)
    return creds


CREDENTIALS = _load_credentials()
AUTH_ENABLED = bool(CREDENTIALS)


def _json_default(obj):
    """json.dumps 兜底：numpy 标量等转为原生 Python 类型"""
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


class Handler(BaseHTTPRequestHandler):
    @staticmethod
    def _normalize_path(path):
        """兼容子路径部署（如反代到 /football/）与本地根路径访问"""
        p = path.split('?', 1)[0].rstrip('/') or '/'
        if p == '/football':
            return '/'
        if p.startswith('/football/'):
            return p[len('/football'):] or '/'
        return p

    def do_GET(self):
        if not self._authorized():
            return
        route = urlparse(self.path)
        path = self._normalize_path(route.path)
        if path == '/':
            self._serve_index()
        elif path == '/api/matches':
            self._serve_json(self._matches_payload())
        elif path == '/api/predict':
            params = parse_qs(route.query)
            self._serve_json(self._predict_payload(params))
        elif path == '/api/3d':
            self._serve_json(self._lottery_3d_payload())
        elif path == '/api/3d-ml':
            self._serve_json(self._lottery_3d_ml_payload())
        else:
            self._send_json_error(404, f'Not Found: {route.path}')

    def _authorized(self):
        """启用鉴权时校验 HTTP Basic 凭据；未启用则放行"""
        if not AUTH_ENABLED:
            return True
        header = self.headers.get('Authorization', '')
        if header.startswith('Basic '):
            try:
                user, _, pwd = base64.b64decode(header[6:]).decode('utf-8').partition(':')
                expected = CREDENTIALS.get(user)
                if expected is not None and hmac.compare_digest(pwd, expected):
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
        try:
            body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode('utf-8')
        except (TypeError, ValueError) as e:
            self._send_json_error(500, f'JSON 序列化失败: {e}')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, status, message):
        body = json.dumps({'error': message}, ensure_ascii=False).encode('utf-8')
        self._send(status, 'application/json; charset=utf-8', body)

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

    def _lottery_3d_payload(self):
        try:
            importlib.reload(lottery3d)
            return {'result': lottery3d.run_prediction()}
        except Exception as e:
            return {'error': f'3D 预测失败: {e}'}

    def _lottery_3d_ml_payload(self):
        try:
            importlib.reload(lottery3d_ml)
            data = lottery3d_ml.fetch_data()
            numbers = [x[2] for x in data] if data else []
            # 使用自动模型选择（优先 LightGBM）
            result = lottery3d_ml.predict_current(numbers, model_type="auto")
            
            # 格式化结果以匹配前端期望
            formatted = {
                'model_type': result.get('model_type', 'unknown'),
                'model_info': result.get('model_info', '未知模型'),
                'n_trees': lottery3d_ml.N_TREES if result.get('model_type') == 'random_forest' else 50,
                'total_samples': int(result.get('total_samples', 0)),
                'pos_samples': int(result.get('pos_samples', 0)),
                'neg_samples': int(result.get('neg_samples', 0)),
                'recommendations': [
                    {'num': r['num'], 'probability': float(r['probability'])}
                    for r in result.get('recommendations', [])
                ],
                'top3': [
                    {'num': r['num'], 'probability': float(r['probability'])}
                    for r in result.get('top3', [])
                ],
                'feature_importance': result.get('feature_importance', []),
            }
            return {'result': formatted}
        except Exception as e:
            import traceback
            return {'error': f'ML 3D 预测失败: {e}\n{traceback.format_exc()}'}

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
    print("  预测服务 - 网页服务已启动")
    print(f"  本机访问:   {local_url}  （Tab 切换：足球 / 福彩3D）")
    if candidates:
        print(f"  局域网访问: http://{candidates[0]}:{PORT}  （手机/其它设备用这个）")
        if len(candidates) > 1:
            others = '  '.join(f'http://{ip}:{PORT}' for ip in candidates[1:])
            print(f"  其它候选地址（若上面连不上，依次试）: {others}")
    if AUTH_ENABLED:
        print(f"  鉴权: 已启用 HTTP Basic（用户: {', '.join(sorted(CREDENTIALS))}）")
    else:
        print("  鉴权: 未启用 ⚠ 公网暴露前请设置 FOOTBALL_USERS 或 FOOTBALL_USER/PASS")
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
