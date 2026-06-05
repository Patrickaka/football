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
import time
import webbrowser
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.football import fetch_match_list, analyze_match
from src.lottery3d import run_prediction
from src.lottery3d.ml import fetch_data, predict_current, N_TREES
from src.pailie5 import get_pailie5_analyzer
from src.lottery_lottery import get_lottery_analyzer
from src.logger import setup_logger

# 回测模块（延迟导入以加速启动）
backtest = None
hyperopt = None
dynamic_threshold = None

def _import_backtest_modules():
    """延迟导入回测相关模块"""
    global backtest, hyperopt, dynamic_threshold
    if backtest is None:
        import backtest as bt
        backtest = bt
    if hyperopt is None:
        import hyperopt as ho
        hyperopt = ho
    if dynamic_threshold is None:
        import dynamic_threshold as dt
        dynamic_threshold = dt

log = setup_logger('server')

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
    _log = log

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
        start = time.perf_counter()
        if not self._authorized():
            self._log_request(401, start)
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
        elif path == '/api/pailie5':
            self._serve_json(self._pailie5_payload())
        elif path == '/api/pailie5/recommend':
            params = parse_qs(route.query)
            self._serve_json(self._pailie5_recommend_payload(params))
        elif path == '/api/pailie5/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._pailie5_backtest_payload(params))
        elif path == '/api/pailie5/fetch':
            self._serve_json(self._pailie5_fetch_payload())
        elif path == '/api/pailie5/optimize':
            self._serve_json(self._pailie5_optimize_payload())
        elif path == '/api/pailie5/markov':
            self._serve_json(self._pailie5_markov_payload())
        elif path == '/api/pailie5/filter':
            params = parse_qs(route.query)
            self._serve_json(self._pailie5_filter_payload(params))
        elif path == '/api/pailie5/rank':
            params = parse_qs(route.query)
            self._serve_json(self._pailie5_rank_payload(params))
        elif path == '/api/pailie5/ensemble':
            params = parse_qs(route.query)
            self._serve_json(self._pailie5_ensemble_payload(params))
        elif path == '/api/pailie5/cycles':
            self._serve_json(self._pailie5_cycles_payload())
        elif path == '/api/pailie5/contribution':
            self._serve_json(self._pailie5_contribution_payload())
        elif path == '/api/lottery':
            self._serve_json(self._lottery_payload())
        elif path == '/api/lottery/recommend':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_recommend_payload(params))
        elif path == '/api/lottery/rank':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_rank_payload(params))
        elif path == '/api/lottery/ensemble':
            self._serve_json(self._lottery_ensemble_payload())
        elif path == '/api/lottery/cycles':
            self._serve_json(self._lottery_cycles_payload())
        elif path == '/api/lottery/contribution':
            self._serve_json(self._lottery_contribution_payload())
        elif path == '/api/lottery/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_backtest_payload(params))
        elif path == '/api/lottery/fetch':
            self._serve_json(self._lottery_fetch_payload())
        elif path == '/api/calibrate':
            params = parse_qs(route.query)
            self._serve_json(self._calibrate_payload(params))
        elif path == '/api/calibrate/list':
            self._serve_json(self._calibrate_list_payload())
        elif path == '/api/calibrate/clear':
            self._serve_json(self._calibrate_clear_payload())
        elif path == '/api/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._backtest_payload(params))
        elif path == '/api/backtest/threshold':
            self._serve_json(self._threshold_payload())
        else:
            self._send_json_error(404, f'Not Found: {route.path}')
        self._log_request(200, start)

    def _log_request(self, status, start):
        elapsed = time.perf_counter() - start
        self._log.info('%s %s %d %.3fs',
                       self.command, self.path, status, elapsed)

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
        self._log.warning('鉴权失败 %s', self.address_string())
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
            return {'matches': fetch_match_list()}
        except Exception:
            self._log.error('获取比赛列表失败', exc_info=True)
            return {'error': '获取比赛列表失败'}

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
            return {'result': analyze_match(match)}
        except Exception:
            self._log.error('赔率分析失败 match_id=%s', match_id, exc_info=True)
            return {'error': '赔率分析失败'}

    def _lottery_3d_payload(self):
        try:
            return {'result': run_prediction()}
        except Exception:
            self._log.error('3D 预测失败', exc_info=True)
            return {'error': '3D 预测失败'}

    def _lottery_3d_ml_payload(self):
        try:
            data = fetch_data()
            numbers = [x[2] for x in data] if data else []
            result = predict_current(numbers, model_type="auto")

            formatted = {
                'model_type': result.get('model_type', 'unknown'),
                'model_info': result.get('model_info', '未知模型'),
                'n_trees': N_TREES if result.get('model_type') == 'random_forest' else 50,
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
        except Exception:
            self._log.error('ML 3D 预测失败', exc_info=True)
            return {'error': 'ML 3D 预测失败'}

    def _pailie5_payload(self):
        """获取排列五统计分析"""
        try:
            analyzer = get_pailie5_analyzer()
            stats = analyzer.get_statistics()
            recent = analyzer.get_recent_results(10)
            
            # 延迟加载回测数据（不在首次请求时计算，改为前端按需加载）
            # backtest = analyzer.rolling_backtest(trials=30)
            
            return {
                'result': {
                    'statistics': stats,
                    'recent_results': recent,
                    # 'backtest': backtest,  # 改为按需加载
                }
            }
        except Exception:
            self._log.error('排列五分析失败', exc_info=True)
            return {'error': '排列五分析失败'}

    def _pailie5_recommend_payload(self, params):
        """获取排列五推荐号码"""
        try:
            analyzer = get_pailie5_analyzer()
            method = params.get('method', ['balanced'])[0]
            
            recommendations = []
            for _ in range(5):
                nums = analyzer.generate_recommendation(method)
                recommendations.append(nums)
            
            return {
                'result': {
                    'method': method,
                    'recommendations': recommendations,
                    'hot_numbers': analyzer.get_hot_numbers(5),
                    'cold_numbers': analyzer.get_cold_numbers(5),
                }
            }
        except Exception:
            self._log.error('排列五推荐失败', exc_info=True)
            return {'error': '排列五推荐失败'}

    def _pailie5_fetch_payload(self):
        """动态抓取排列五最新开奖号码"""
        try:
            analyzer = get_pailie5_analyzer()
            
            result = analyzer.fetch_latest_results()
            
            return {'result': result}
        except Exception:
            self._log.error('排列五抓取失败', exc_info=True)
            return {'error': '排列五抓取失败'}

    def _pailie5_backtest_payload(self, params):
        """排列五历史回测"""
        try:
            analyzer = get_pailie5_analyzer()
            method = params.get('method', ['bayesian'])[0]
            periods = int(params.get('periods', [30])[0])
            
            result = analyzer.backtest(method=method, test_periods=periods)
            
            return {'result': result}
        except Exception:
            self._log.error('排列五回测失败', exc_info=True)
            return {'error': '排列五回测失败'}

    def _pailie5_optimize_payload(self):
        """排列五自动权重优化"""
        try:
            analyzer = get_pailie5_analyzer()
            weights = analyzer.optimize_weights()
            
            return {'result': weights}
        except Exception:
            self._log.error('排列五权重优化失败', exc_info=True)
            return {'error': '排列五权重优化失败'}

    def _pailie5_markov_payload(self):
        """排列五马尔可夫链预测"""
        try:
            analyzer = get_pailie5_analyzer()
            recent = analyzer.get_recent_results(2)
            recent_numbers = [r['numbers'] for r in recent]
            prediction = analyzer.predict_with_markov(recent_numbers)
            
            return {
                'result': {
                    'recent_results': recent_numbers,
                    'prediction': prediction
                }
            }
        except Exception:
            self._log.error('排列五马尔可夫预测失败', exc_info=True)
            return {'error': '排列五马尔可夫预测失败'}

    def _pailie5_filter_payload(self, params):
        """排列五多条件缩水"""
        try:
            analyzer = get_pailie5_analyzer()
            
            # 生成候选号码
            candidates = []
            for _ in range(100):
                candidates.append(analyzer.generate_recommendation('balanced'))
            
            # 解析条件参数
            conditions = {}
            
            if 'sum_min' in params and 'sum_max' in params:
                conditions['sum_range'] = (int(params['sum_min'][0]), int(params['sum_max'][0]))
            
            if 'span_min' in params and 'span_max' in params:
                conditions['span_range'] = (int(params['span_min'][0]), int(params['span_max'][0]))
            
            if 'odd_min' in params and 'odd_max' in params:
                conditions['odd_count'] = (int(params['odd_min'][0]), int(params['odd_max'][0]))
            
            if 'big_min' in params and 'big_max' in params:
                conditions['big_count'] = (int(params['big_min'][0]), int(params['big_max'][0]))
            
            # 应用筛选
            filtered = analyzer.multi_condition_filter(candidates, conditions)
            
            return {
                'result': {
                    'conditions': conditions,
                    'total_candidates': len(candidates),
                    'filtered_count': len(filtered),
                    'filtered': filtered[:20]  # 返回前20个
                }
            }
        except Exception:
            self._log.error('排列五多条件缩水失败', exc_info=True)
            return {'error': '排列五多条件缩水失败'}

    def _pailie5_rank_payload(self, params):
        """排列五排名模型 - Top-N排序"""
        try:
            analyzer = get_pailie5_analyzer()
            top_n = int(params.get('top_n', [5])[0])
            
            # 获取排名结果
            ranked = analyzer.rank_model(top_n=top_n)
            
            return {
                'result': {
                    'top_n': top_n,
                    'ranked_numbers': [{'number': n, 'score': s} for n, s in ranked]
                }
            }
        except Exception:
            self._log.error('排列五排名模型失败', exc_info=True)
            return {'error': '排列五排名模型失败'}

    def _pailie5_ensemble_payload(self, params):
        """排列五多模型集成投票"""
        try:
            analyzer = get_pailie5_analyzer()
            method = params.get('method', ['voting'])[0]
            
            result = analyzer.ensemble_predict(method=method)
            
            return {'result': result}
        except Exception:
            self._log.error('排列五集成预测失败', exc_info=True)
            return {'error': '排列五集成预测失败'}

    def _pailie5_cycles_payload(self):
        """排列五周期与状态识别"""
        try:
            analyzer = get_pailie5_analyzer()
            
            cycles = analyzer.identify_cycles()
            
            return {'result': cycles}
        except Exception:
            self._log.error('排列五周期识别失败', exc_info=True)
            return {'error': '排列五周期识别失败'}

    def _pailie5_contribution_payload(self):
        """排列五特征贡献度分析"""
        try:
            analyzer = get_pailie5_analyzer()
            
            contributions = analyzer.feature_contribution()
            
            return {'result': contributions}
        except Exception:
            self._log.error('排列五特征贡献度分析失败', exc_info=True)
            return {'error': '排列五特征贡献度分析失败'}

    def _calibrate_payload(self, params):
        """手动触发联赛重新校准"""
        league = params.get('league', [''])[0]
        if not league:
            return {'error': '缺少 league 参数'}
        recent_matches = int(params.get('matches', ['10'])[0])
        
        try:
            from src.football import recalibrate_league
            result = recalibrate_league(league, recent_matches=recent_matches)
            return {'result': result}
        except Exception as e:
            self._log.error('校准失败 league=%s', league, exc_info=True)
            return {'error': f'校准失败: {str(e)}'}

    def _calibrate_list_payload(self):
        """列出所有已校准的联赛"""
        try:
            from src.football import list_calibrated_leagues
            leagues = list_calibrated_leagues()
            return {'result': {'leagues': leagues, 'count': len(leagues)}}
        except Exception as e:
            self._log.error('获取校准列表失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _calibrate_clear_payload(self):
        """清空校准缓存"""
        try:
            from src.football import clear_calibration_cache
            result = clear_calibration_cache()
            return {'result': result}
        except Exception as e:
            self._log.error('清空校准缓存失败', exc_info=True)
            return {'error': f'清空失败: {str(e)}'}

    def _backtest_payload(self, params):
        """执行回测"""
        try:
            _import_backtest_modules()
            
            league = params.get('league', ['英超'])[0]
            start_date = params.get('start', ['2024-01-01'])[0]
            end_date = params.get('end', ['2024-06-30'])[0]
            
            result = backtest.run_backtest(league, start_date, end_date)
            return {'result': result}
        except Exception as e:
            self._log.error('回测失败', exc_info=True)
            return {'error': f'回测失败: {str(e)}'}

    def _threshold_payload(self):
        """获取动态阈值状态"""
        try:
            _import_backtest_modules()
            
            manager = dynamic_threshold.get_threshold_manager()
            stats = manager.get_statistics()
            thresholds = manager.get_thresholds()
            
            return {
                'result': {
                    'statistics': stats,
                    'thresholds': thresholds
                }
            }
        except Exception as e:
            self._log.error('获取阈值状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _lottery_payload(self):
        """获取大乐透统计分析"""
        try:
            analyzer = get_lottery_analyzer()
            stats = analyzer.get_statistics()
            recent = analyzer.get_recent_results(10)
            
            # 执行滚动回测
            backtest = analyzer.rolling_backtest(trials=50)
            
            return {
                'result': {
                    'statistics': stats,
                    'recent_results': recent,
                    'backtest': backtest,
                }
            }
        except Exception:
            self._log.error('大乐透分析失败', exc_info=True)
            return {'error': '大乐透分析失败'}

    def _lottery_recommend_payload(self, params):
        """获取大乐透推荐号码 - 返回3组概率最高的推荐"""
        try:
            analyzer = get_lottery_analyzer()
            method = params.get('method', ['balanced'])[0]
            
            # 生成3组推荐
            recommendations = []
            for _ in range(3):
                rec = analyzer.generate_recommendation(method)
                recommendations.append(rec)
            
            return {
                'result': {
                    'method': method,
                    'recommendations': recommendations,
                    'count': len(recommendations)
                }
            }
        except Exception:
            self._log.error('大乐透推荐失败', exc_info=True)
            return {'error': '大乐透推荐失败'}

    def _lottery_rank_payload(self, params):
        """大乐透排名模型 - Top-N排序"""
        try:
            analyzer = get_lottery_analyzer()
            top_n = int(params.get('top_n', [10])[0])
            
            front_ranked, back_ranked = analyzer.rank_model(top_n=top_n)
            
            return {
                'result': {
                    'top_n': top_n,
                    'front_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in front_ranked],
                    'back_ranked': [{'number': n, 'score': s, 'features': f} for n, s, f in back_ranked],
                }
            }
        except Exception:
            self._log.error('大乐透排名模型失败', exc_info=True)
            return {'error': '大乐透排名模型失败'}

    def _lottery_ensemble_payload(self):
        """大乐透多模型集成投票"""
        try:
            analyzer = get_lottery_analyzer()
            
            result = analyzer.multi_model_voting()
            
            return {'result': result}
        except Exception:
            self._log.error('大乐透集成预测失败', exc_info=True)
            return {'error': '大乐透集成预测失败'}

    def _lottery_cycles_payload(self):
        """大乐透周期与状态识别"""
        try:
            analyzer = get_lottery_analyzer()
            
            cycles = analyzer.identify_cycles()
            
            return {'result': cycles}
        except Exception:
            self._log.error('大乐透周期识别失败', exc_info=True)
            return {'error': '大乐透周期识别失败'}

    def _lottery_contribution_payload(self):
        """大乐透特征贡献度分析"""
        try:
            analyzer = get_lottery_analyzer()
            
            contributions = analyzer.feature_contribution()
            
            return {'result': contributions}
        except Exception:
            self._log.error('大乐透特征贡献度分析失败', exc_info=True)
            return {'error': '大乐透特征贡献度分析失败'}

    def _lottery_backtest_payload(self, params):
        """大乐透历史回测"""
        try:
            analyzer = get_lottery_analyzer()
            method = params.get('method', ['balanced'])[0]
            periods = int(params.get('periods', [30])[0])
            
            result = analyzer.backtest(method=method, test_periods=periods)
            
            return {'result': result}
        except Exception:
            self._log.error('大乐透回测失败', exc_info=True)
            return {'error': '大乐透回测失败'}

    def _lottery_fetch_payload(self):
        """动态抓取最新开奖号码"""
        try:
            analyzer = get_lottery_analyzer()
            
            result = analyzer.fetch_latest_results()
            
            return {'result': result}
        except Exception:
            self._log.error('大乐透抓取失败', exc_info=True)
            return {'error': '大乐透抓取失败'}

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        self._log.debug('%s - %s', self.address_string(), fmt % args)


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
    log.info('=' * 50)
    log.info('预测服务启动 端口=%s', PORT)
    if candidates:
        log.info('候选地址: %s %s', local_url,
                 ' '.join(f'http://{ip}:{PORT}' for ip in candidates))
    if AUTH_ENABLED:
        log.info('鉴权: 已启用 (用户: %s)', ', '.join(sorted(CREDENTIALS)))
    else:
        log.warning('鉴权: 未启用 — 公网暴露前请设置 FOOTBALL_USERS')
    log.info('=' * 50)
    try:
        webbrowser.open(local_url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info('服务已停止')
        server.shutdown()


if __name__ == '__main__':
    main()
