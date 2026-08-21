# -*- coding: utf-8 -*-
"""HTTP 请求处理与路由分发"""

import os
import sys
import json
import math
import hmac
import base64
import socket
import time
import re
from datetime import datetime, timedelta
import importlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from src.common.logger import setup_logger
from src.common.paths import data_path

log = setup_logger('server')

from .settings import (
    AUTH_ENABLED, CORS_ORIGIN, CREDENTIALS, INDEX_FILE,
)
from .http_util import (
    _json_default, _sanitize_json,
)
from .football_api import FootballApiMixin
from .lottery_api import LotteryApiMixin
from .kl8_api import KL8ApiMixin
from .beidan_api import BeidanApiMixin
from .basketball_api import BasketballApiMixin

class Handler(FootballApiMixin, LotteryApiMixin, KL8ApiMixin,
              BeidanApiMixin, BasketballApiMixin, BaseHTTPRequestHandler):
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
        elif path.startswith('/reports/'):
            self._serve_report_file(path)
        elif path == '/api/matches':
            self._serve_json(self._matches_payload())
        elif path == '/api/predict':
            params = parse_qs(route.query)
            self._serve_json(self._predict_payload(params))
        elif path == '/api/football/clear_cache':
            self._serve_json(self._football_clear_cache_payload())
        elif path == '/api/football/prepare_ml_data':
            self._serve_json(self._prepare_ml_history_data_payload())
        elif path == '/api/football/diagnostics':
            params = parse_qs(route.query)
            self._serve_json(self._football_diagnostics_payload(params))
        elif path == '/api/football/review':
            params = parse_qs(route.query)
            self._serve_json(self._football_review_payload(params))
        elif path == '/api/football/professional-status':
            self._serve_json(self._football_professional_status_payload())
        elif path == '/api/3d':
            self._serve_json(self._lottery_3d_payload())
        elif path == '/api/3d-ml':
            self._serve_json(self._lottery_3d_ml_payload())
        elif path == '/api/beidan':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_payload(params))
        elif path == '/api/beidan/matches':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_matches_payload(params))
        elif path == '/api/beidan/value':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_value_payload(params))
        elif path == '/api/beidan/history':
            params = parse_qs(route.query)
            self._serve_json(self._beidan_history_payload(params))
        elif path == '/api/basketball':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_payload(params))
        elif path == '/api/basketball/matches':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_matches_payload(params))
        elif path == '/api/basketball/value':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_value_payload(params))
        elif path == '/api/basketball/track':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_track_payload(params))
        elif path == '/api/basketball/movement':
            params = parse_qs(route.query)
            self._serve_json(self._basketball_movement_payload(params))
        elif path == '/api/lottery':
            self._serve_json(self._lottery_payload())
        elif path == '/api/lottery-refresh':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_refresh_payload(params))
        elif path == '/api/lottery/task-status':
            self._serve_json(self._lottery_task_status_payload())
        elif path == '/api/3d-refresh':
            params = parse_qs(route.query)
            self._serve_json(self._lottery_3d_refresh_payload(params))
        elif path == '/ssq':
            prefix = '/football' if route.path.startswith('/football/') else ''
            self.send_response(302)
            self.send_header('Location', f'{prefix}/#ssq')
            self.end_headers()
        elif path == '/api/ssq':
            self._serve_json(self._ssq_payload())
        elif path == '/api/ssq-refresh':
            self._serve_json(self._ssq_refresh_payload())
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
        elif path == '/api/lottery/ml':
            self._serve_json(self._lottery_ml_payload())
        elif path == '/api/lottery/ml-refresh':
            self._serve_json(self._lottery_ml_refresh_payload())
        elif path == '/api/kl8':
            self._serve_json(self._kl8_payload())
        elif path == '/api/kl8-refresh':
            self._serve_json(self._kl8_refresh_payload())
        elif path == '/api/kl8/fetch':
            self._serve_json(self._kl8_fetch_payload())
        elif path == '/api/kl8/exclude-recalculate':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_exclude_recalculate_payload(params))
        elif path == '/api/kl8/snapshots':
            self._serve_json(self._kl8_snapshots_payload())
        elif path == '/api/kl8/records':
            self._serve_json(self._kl8_records_payload())
        elif path == '/api/kl8/settle':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_settle_payload(params))
        elif path == '/api/kl8/backtest':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_backtest_payload(params))
        elif path == '/api/kl8/parameter-search':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_payload(params))
        elif path == '/api/kl8/parameter-search/start':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_start_payload(params))
        elif path == '/api/kl8/parameter-search/status':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_parameter_search_status_payload(params))
        elif path == '/api/kl8/integrity':
            self._serve_json(self._kl8_integrity_payload())
        elif path == '/api/kl8/conflicts':
            self._serve_json(self._kl8_conflicts_payload())
        elif path == '/api/kl8/activate':
            params = parse_qs(route.query)
            self._serve_json(self._kl8_activate_payload(params))
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
        elif path == '/api/predictions/export':
            self._serve_json(self._predictions_export_payload())
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
            body = json.dumps(_sanitize_json(payload), ensure_ascii=False,
                              allow_nan=False, default=_json_default).encode('utf-8')
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

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        self._log.debug('%s - %s', self.address_string(), fmt % args)

