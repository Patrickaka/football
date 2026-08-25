# -*- coding: utf-8 -*-
"""服务配置：根路径、监听地址、鉴权凭据、CORS"""

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



_ROOT = Path(__file__).resolve().parents[2]


INDEX_FILE = _ROOT / 'web' / 'index.html'


sys.stdout.reconfigure(encoding='utf-8')


HOST = os.environ.get('FOOTBALL_HOST', '0.0.0.0')  # 默认监听所有网卡；线上经反代时设为 127.0.0.1 收窄暴露面


PORT = int(os.environ.get('FOOTBALL_PORT', '9004'))


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

