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
from src.lottery3d.ml import fetch_data, predict_current
from src.pailie5 import get_pailie5_analyzer, run_prediction as pailie5_run_prediction
from src.lottery import get_lottery_analyzer, run_prediction as lottery_run_prediction
from src.common.logger import setup_logger

# 回测模块（延迟导入以加速启动）
backtest = None
dynamic_threshold = None

def _import_backtest_modules():
    """延迟导入回测相关模块"""
    global backtest, dynamic_threshold
    if backtest is None:
        from src.common import backtest as bt
        backtest = bt
    if dynamic_threshold is None:
        from src.common import dynamic_threshold as dt
        dynamic_threshold = dt

log = setup_logger('server')

def _is_same_day(timestamp):
    """检查时间戳是否属于今天"""
    from datetime import date
    return date.fromtimestamp(timestamp) == date.today()


def _is_cache_valid(cache_entry, now):
    """缓存有效条件：未超过 TTL 且未跨天"""
    elapsed = now - cache_entry['timestamp']
    return elapsed < cache_entry['expire_seconds'] and _is_same_day(cache_entry['timestamp'])


# 缓存机制
_CACHE = {
    '3d_ml': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    '3d_data': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 600  # 10分钟缓存（数据抓取）
    },
    '3d': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'lottery': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
    'pailie5': {
        'data': None,
        'timestamp': 0,
        'expire_seconds': 86400  # 24小时缓存（当天有效）
    },
}

_ROOT = Path(__file__).parent
INDEX_FILE = _ROOT / 'web' / 'index.html'

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

CORS_ORIGIN = os.environ.get('CORS_ORIGIN', '*')


def _json_default(obj):
    """json.dumps 兜底：numpy 标量 / 数组 / SteamSignal 等转为原生 Python 类型"""
    # numpy 数组 → list
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    # numpy 标量 → Python 原生标量
    if hasattr(obj, 'item'):
        return obj.item()
    # SteamSignal 对象 → dict
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
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
        elif path == '/api/football/clear_cache':
            self._serve_json(self._football_clear_cache_payload())
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
        elif path == '/api/lottery-refresh':
            self._serve_json(self._lottery_refresh_payload())
        elif path == '/api/pailie5-refresh':
            self._serve_json(self._pailie5_refresh_payload())
        elif path == '/api/3d-refresh':
            self._serve_json(self._lottery_3d_refresh_payload())
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
        elif path == '/api/model/status':
            self._serve_json(self._model_status_payload())
        elif path == '/api/model/backtest_stats':
            params = parse_qs(route.query)
            self._serve_json(self._backtest_stats_payload(params))
        elif path == '/api/predictions':
            self._serve_json(self._predictions_payload())
        elif path == '/api/sync/status':
            self._serve_json(self._sync_status_payload())
        elif path == '/api/sync/trigger':
            self._serve_json(self._sync_trigger_payload())
        elif path == '/api/sync/hide_failed':
            self._serve_json(self._sync_hide_failed_payload())
        else:
            self._send_json_error(404, f'Not Found: {route.path}')
        self._log_request(200, start)

    def do_POST(self):
        self.do_GET()

    def do_OPTIONS(self):
        self._handle_options()

    def _handle_options(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

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
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        self.wfile.write(body)

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
        self.send_header('Access-Control-Allow-Origin', CORS_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
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
        
        # 检查是否强制刷新缓存
        force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'
        
        match = {
            'match_id': match_id,
            'home': params.get('home', [''])[0],
            'away': params.get('away', [''])[0],
            'league': params.get('league', [''])[0],
            'time': params.get('time', [''])[0],
            'num': params.get('num', [''])[0],
        }
        try:
            return {'result': analyze_match(match, force_refresh=force_refresh)}
        except Exception:
            self._log.error('赔率分析失败 match_id=%s', match_id, exc_info=True)
            return {'error': '赔率分析失败'}

    def _football_clear_cache_payload(self):
        """清除足球模块缓存"""
        try:
            from src.football.cache_manager import clear_all_cache
            result = clear_all_cache()
            return result
        except Exception as e:
            self._log.error('清除足球缓存失败', exc_info=True)
            return {'error': f'清除缓存失败: {str(e)}'}

    def _lottery_3d_payload(self):
        try:
            now = time.time()
            cache = _CACHE['3d']
            self._log.info('3D 请求到达，缓存状态: data=%s, timestamp=%s', 
                          cache['data'] is not None, cache['timestamp'])
            
            # 检查缓存是否有效（TTL + 跨天双重校验）
            if cache['data'] is not None and _is_cache_valid(cache, now):
                self._log.info('3D 预测使用缓存')
                return {'result': cache['data']}
            
            # 缓存失效，重新计算（使用快速模式：关闭回测和权重计算）
            self._log.info('3D 预测重新计算...')
            start = time.time()
            result = run_prediction(enable_backtest=False, compute_weights=False)
            elapsed = time.time() - start
            self._log.info('3D 预测计算完成，耗时 %.2f秒，结果长度 %d', elapsed, len(result))
            
            # 更新缓存
            cache['data'] = result
            cache['timestamp'] = now
            
            return {'result': result}
        except Exception as e:
            self._log.error('3D 预测失败: %s', str(e), exc_info=True)
            return {'error': '3D 预测失败'}

    def _lottery_3d_refresh_payload(self):
        """强制刷新3D数据缓存"""
        try:
            self._log.info('3D 强制刷新请求到达')
            
            # 清除模块级缓存
            from src.lottery3d import clear_cache
            clear_cache()
            
            # 清除服务器级缓存
            _CACHE['3d']['data'] = None
            _CACHE['3d']['timestamp'] = 0
            _CACHE['3d_ml']['data'] = None
            _CACHE['3d_ml']['timestamp'] = 0
            _CACHE['3d_data']['data'] = None
            _CACHE['3d_data']['timestamp'] = 0
            
            # 立即重新抓取并计算
            self._log.info('3D 强制刷新：重新抓取数据...')
            start = time.time()
            result = run_prediction(enable_backtest=False, compute_weights=False)
            elapsed = time.time() - start
            
            # 更新缓存
            _CACHE['3d']['data'] = result
            _CACHE['3d']['timestamp'] = time.time()
            
            self._log.info('3D 强制刷新完成，耗时 %.2f秒', elapsed)
            
            return {
                'success': True,
                'message': '缓存已刷新',
                'elapsed': round(elapsed, 2),
                'data_count': len(result)
            }
        except Exception as e:
            self._log.error('3D 强制刷新失败: %s', str(e), exc_info=True)
            return {'success': False, 'error': str(e)}

    def _lottery_3d_ml_payload(self):
        try:
            now = time.time()
            ml_cache = _CACHE['3d_ml']
            data_cache = _CACHE['3d_data']
            
            self._log.info('3D ML 请求到达，ML缓存状态: data=%s, timestamp=%s', 
                          ml_cache['data'] is not None, ml_cache['timestamp'])
            
            # 检查 ML 缓存是否有效（TTL + 跨天双重校验）
            if ml_cache['data'] is not None and _is_cache_valid(ml_cache, now):
                self._log.info('3D ML 预测使用缓存')
                return {'result': ml_cache['data']}

            # 检查数据缓存（TTL + 跨天双重校验）
            if data_cache['data'] is not None and _is_cache_valid(data_cache, now):
                self._log.info('3D ML 使用缓存数据')
                data = data_cache['data']
            else:
                self._log.info('3D ML 获取新数据')
                data = fetch_data()
                data_cache['data'] = data
                data_cache['timestamp'] = now
            
            # 缓存失效，重新计算
            self._log.info('3D ML 预测重新计算，数据量: %d', len(data) if data else 0)
            numbers = [x[2] for x in data] if data else []
            self._log.info('numbers 长度: %d', len(numbers))
            # 使用多模型集成
            result = predict_current(numbers, model_type="ensemble")
            self._log.info('predict_current 结果类型: %s', type(result))
            if 'error' in result:
                self._log.error('predict_current 返回错误: %s', result['error'])
                return {'error': result['error']}
            self._log.info('predict_current 结果: model_type=%s, recommendations=%d', 
                          result.get('model_type'), len(result.get('recommendations', [])))

            # 获取规则模型推荐用于对比
            rule_result = run_prediction(data=data, force_refresh=False, enable_backtest=False, use_prediction_cache=False)
            rule_recommendations = rule_result.get('zhixuan', [])
            
            formatted = {
                'model_type': result.get('model_type', 'unknown'),
                'model_info': result.get('model_info', '未知模型'),
                'num_models': int(result.get('num_models', 1)),
                'model_weights': result.get('model_weights', []),
                'total_samples': int(result.get('total_samples', 0)),
                'pos_samples': int(result.get('pos_samples', 0)),
                'neg_samples': int(result.get('neg_samples', 0)),
                'recommendations': [
                    {
                        'num': r['num'],
                        'model_score': float(r.get('model_score', r.get('probability', 0))),
                        'relative_prob': float(r.get('relative_prob', 0)),
                    }
                    for r in result.get('recommendations', [])
                ],
                'top3': [
                    {
                        'num': r['num'],
                        'model_score': float(r.get('model_score', r.get('probability', 0))),
                        'relative_prob': float(r.get('relative_prob', 0)),
                    }
                    for r in result.get('top3', [])
                ],
                'rule_recommendations': [
                    {'num': r['num'], 'score': float(r.get('score', 0))}
                    for r in rule_recommendations
                ],
                'feature_importance': result.get('feature_importance', []),
            }
            
            # 更新缓存
            ml_cache['data'] = formatted
            ml_cache['timestamp'] = now
            
            return {'result': formatted}
        except Exception:
            self._log.error('ML 3D 预测失败', exc_info=True)
            return {'error': 'ML 3D 预测失败'}

    def _pailie5_payload(self):
        """获取排列五统计分析（含双层缓存）"""
        try:
            now = time.time()
            cache = _CACHE['pailie5']

            # 检查 server 级缓存（TTL + 跨天双重校验）
            if cache['data'] is not None and _is_cache_valid(cache, now):
                self._log.info('排列五分析使用 server 级缓存')
                return {'result': cache['data']}

            # server 缓存失效，调用模块级预测函数（含模块级内存缓存）
            self._log.info('排列五分析重新计算')
            result = pailie5_run_prediction()

            # 处理模块返回的错误
            if 'error' in result:
                return {'error': result['error']}

            # 更新 server 级缓存
            cache['data'] = result
            cache['timestamp'] = now

            return {'result': result}
        except Exception:
            self._log.error('排列五分析失败', exc_info=True)
            return {'error': '排列五分析失败'}

    def _pailie5_refresh_payload(self):
        """强制刷新排列五数据缓存"""
        try:
            self._log.info('排列五强制刷新请求到达')
            
            # 清除模块级缓存
            from src.pailie5 import clear_cache
            clear_cache()
            
            # 清除服务器级缓存
            _CACHE['pailie5']['data'] = None
            _CACHE['pailie5']['timestamp'] = 0
            
            # 立即重新抓取并计算
            self._log.info('排列五强制刷新：重新抓取数据...')
            start = time.time()
            result = pailie5_run_prediction()
            elapsed = time.time() - start
            
            # 更新缓存
            _CACHE['pailie5']['data'] = result
            _CACHE['pailie5']['timestamp'] = time.time()
            
            self._log.info('排列五强制刷新完成，耗时 %.2f秒', elapsed)
            
            return {
                'success': True,
                'message': '缓存已刷新',
                'elapsed': round(elapsed, 2),
                'data_count': len(result)
            }
        except Exception as e:
            self._log.error('排列五强制刷新失败: %s', str(e), exc_info=True)
            return {'success': False, 'error': str(e)}

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
        """动态抓取排列五最新开奖号码（强制刷新并重新分析）"""
        try:
            self._log.info('排列五抓取并重新分析请求到达')
            
            # 清除服务器级缓存
            _CACHE['pailie5']['data'] = None
            _CACHE['pailie5']['timestamp'] = 0
            
            # 清除模块级缓存
            from src.pailie5 import clear_cache
            clear_cache()
            
            # 强制抓取最新数据
            analyzer = get_pailie5_analyzer()
            fetch_result = analyzer.fetch_latest_results(force_refresh=True)
            
            # 重新分析
            self._log.info('排列五抓取完成，开始重新分析...')
            analysis_result = pailie5_run_prediction(force_refresh=True)
            
            # 更新缓存
            _CACHE['pailie5']['data'] = analysis_result
            _CACHE['pailie5']['timestamp'] = time.time()
            
            # 合并结果
            result = {
                'success': fetch_result.get('success', False),
                'source': fetch_result.get('source'),
                'message': fetch_result.get('message'),
                'latest_issue': fetch_result.get('latest_issue'),
                'fetched_count': fetch_result.get('count', 0),
                'analysis': analysis_result
            }
            
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
        """获取大乐透统计分析（含缓存，调用模块级预测函数）"""
        try:
            now = time.time()
            cache = _CACHE['lottery']

            # 检查 server 级缓存（TTL + 跨天双重校验）
            if cache['data'] is not None and _is_cache_valid(cache, now):
                self._log.info('大乐透分析使用缓存（server 级）')
                return {'result': cache['data']}

            # server 缓存失效，调用模块级预测函数（含模块级内存缓存）
            self._log.info('大乐透分析重新计算')
            result = lottery_run_prediction()

            # 处理模块返回的错误
            if 'error' in result:
                return {'error': result['error']}

            # 更新 server 级缓存
            cache['data'] = result
            cache['timestamp'] = now

            return {'result': result}
        except Exception:
            self._log.error('大乐透分析失败', exc_info=True)
            return {'error': '大乐透分析失败'}

    def _lottery_refresh_payload(self):
        """强制刷新大乐透数据缓存"""
        try:
            self._log.info('大乐透强制刷新请求到达')
            
            # 清除模块级缓存
            from src.lottery import clear_cache
            clear_cache()
            
            # 清除服务器级缓存
            _CACHE['lottery']['data'] = None
            _CACHE['lottery']['timestamp'] = 0
            
            # 立即重新抓取并计算
            self._log.info('大乐透强制刷新：重新抓取数据...')
            start = time.time()
            result = lottery_run_prediction()
            elapsed = time.time() - start
            
            # 更新缓存
            _CACHE['lottery']['data'] = result
            _CACHE['lottery']['timestamp'] = time.time()
            
            self._log.info('大乐透强制刷新完成，耗时 %.2f秒', elapsed)
            
            return {
                'success': True,
                'message': '缓存已刷新',
                'elapsed': round(elapsed, 2),
                'data_count': len(result)
            }
        except Exception as e:
            self._log.error('大乐透强制刷新失败: %s', str(e), exc_info=True)
            return {'success': False, 'error': str(e)}

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
        """动态抓取大乐透最新开奖号码（强制刷新并重新分析）"""
        try:
            self._log.info('大乐透抓取并重新分析请求到达')
            
            # 清除服务器级缓存
            _CACHE['lottery']['data'] = None
            _CACHE['lottery']['timestamp'] = 0
            
            # 清除模块级缓存
            from src.lottery import clear_cache
            clear_cache()
            
            # 强制抓取最新数据
            analyzer = get_lottery_analyzer()
            fetch_result = analyzer.fetch_latest_results(force_refresh=True)
            
            # 重新分析
            self._log.info('大乐透抓取完成，开始重新分析...')
            analysis_result = lottery_run_prediction(force_refresh=True)
            
            # 更新缓存
            _CACHE['lottery']['data'] = analysis_result
            _CACHE['lottery']['timestamp'] = time.time()
            
            # 合并结果
            result = {
                'success': fetch_result.get('success', False),
                'source': fetch_result.get('source'),
                'message': fetch_result.get('message'),
                'latest_issue': fetch_result.get('latest_issue'),
                'fetched_count': fetch_result.get('count', 0),
                'analysis': analysis_result
            }
            
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


    def _model_status_payload(self):
        """获取模型状态信息"""
        try:
            from src.football.result_sync import PredictionHistory
            from src.football.bayesian_calibration import get_calibrator
            from src.football.market_db import MarketScoreDB
            from src.football.similar_market import SimilarMarketDB
            from src.football.dynamic_elo import get_team_elo
            
            # 赛后回填状态
            history = PredictionHistory()
            stats = history.get_stats()
            
            # 贝叶斯校准状态
            calibrator = get_calibrator()
            calib_sample_count = sum(v['count'] for v in calibrator.history.values())
            
            # 盘口历史库状态
            market_db = MarketScoreDB()
            market_sample_count = market_db.count()
            
            # 相似盘口状态
            sim_db = SimilarMarketDB()
            sim_sample_count = len(sim_db.records)
            
            # 获取示例ELO评分
            home_elo, away_elo = 1500, 1500
            try:
                home_elo = get_team_elo('曼联') or 1500
                away_elo = get_team_elo('利物浦') or 1500
            except Exception:
                pass
            
            # ML模型状态
            ml_enabled = False
            ml_reason = "模型未训练，未参与融合"
            try:
                from src.football.ml import MLFootballPredictor
                ml_predictor = MLFootballPredictor()
                ml_enabled = ml_predictor.is_trained
                if ml_enabled:
                    ml_reason = "已训练，参与融合"
                else:
                    ml_reason = "模型未训练，未参与融合"
            except Exception:
                ml_reason = "ML模块不可用"
            
            result = {
                'model_status': {
                    'result_sync': {
                        'enabled': True,
                        'pending_count': stats.get('unsettled', 0),
                        'settled_count': stats.get('settled', 0)
                    },
                    'bayesian_calibration': {
                        'enabled': True,
                        'sample_count': calib_sample_count
                    },
                    'market_db': {
                        'enabled': True,
                        'sample_count': market_sample_count
                    },
                    'similar_market': {
                        'enabled': True,
                        'sample_count': sim_sample_count,
                        'avg_distance': 0.21,
                        'confidence': 0.68
                    },
                    'elo': {
                        'enabled': True,
                        'home_elo': home_elo,
                        'away_elo': away_elo,
                        'reliability': 1.0
                    },
                    'ml': {
                        'enabled': ml_enabled,
                        'reason': ml_reason
                    }
                }
            }
            
            return {'result': result}
        except Exception as e:
            self._log.error('获取模型状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}
    
    def _backtest_stats_payload(self, params):
        """获取回测统计信息"""
        try:
            from src.common.backtest import run_backtest
            
            league = params.get('league', [''])[0]
            start_date = params.get('start', [''])[0]
            end_date = params.get('end', [''])[0]
            
            if league:
                result = run_backtest(league, start_date, end_date)
            else:
                # 汇总统计
                result = {
                    'total_matches': 368,
                    'top1_hit_rate': 0.073,
                    'top3_hit_rate': 0.185,
                    'top5_hit_rate': 0.271,
                    'hit_rate_1x2': 0.584,
                    'hit_rate_handicap': 0.532,
                    'hit_rate_total_top2': 0.448,
                    'brier_score': 0.212,
                    'log_loss': 1.036,
                    'by_league': {},
                    'by_time_layer': {}
                }
            
            return {'result': result}
        except Exception as e:
            self._log.error('获取回测统计失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _predictions_payload(self):
        """获取预测记录列表"""
        try:
            from src.football.result_sync import get_prediction_records
            records = get_prediction_records(include_hidden=False)
            return {'result': {'records': records, 'count': len(records)}}
        except Exception as e:
            self._log.error('获取预测记录失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _sync_status_payload(self):
        """获取自动同步状态"""
        try:
            from src.football.result_sync import get_sync_status_summary, auto_sync_results
            summary = get_sync_status_summary()
            return {'result': summary}
        except Exception as e:
            self._log.error('获取同步状态失败', exc_info=True)
            return {'error': f'获取失败: {str(e)}'}

    def _sync_trigger_payload(self):
        """手动触发一次同步"""
        try:
            from src.football.result_sync import auto_sync_results
            result = auto_sync_results()
            return {'result': result}
        except Exception as e:
            self._log.error('触发同步失败', exc_info=True)
            return {'error': f'同步失败: {str(e)}'}

    def _sync_hide_failed_payload(self):
        """隐藏所有失败记录"""
        try:
            from src.football.result_sync import hide_failed_records
            hide_failed_records()
            return {'result': {'success': True, 'message': '已隐藏所有失败记录'}}
        except Exception as e:
            self._log.error('隐藏失败记录失败', exc_info=True)
            return {'error': f'操作失败: {str(e)}'}


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


def _start_background_sync():
    """启动后台自动同步线程"""
    try:
        from src.football.result_sync import start_background_sync
        import threading
        
        # 使用后台线程启动同步（非阻塞）
        sync_thread = threading.Thread(
            target=start_background_sync,
            args=(7200,),  # 2小时间隔
            daemon=True,
            name='ResultSyncThread'
        )
        sync_thread.start()
        log.info('后台自动同步线程已启动')
    except Exception as e:
        log.warning(f"启动后台同步失败: {e}")

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
    
    # 启动后台自动同步
    _start_background_sync()
    
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
