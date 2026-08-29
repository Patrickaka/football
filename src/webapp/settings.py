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
from src.foundation.auth import parse_credentials
from src.common.paths import data_path

log = setup_logger('server')



_ROOT = Path(__file__).resolve().parents[2]


INDEX_FILE = _ROOT / 'web' / 'index.html'


sys.stdout.reconfigure(encoding='utf-8')


HOST = os.environ.get('FOOTBALL_HOST', '0.0.0.0')  # 默认监听所有网卡；线上经反代时设为 127.0.0.1 收窄暴露面


PORT = int(os.environ.get('FOOTBALL_PORT', '9004'))


# 解析逻辑住在底座，新旧两个入口共用同一份——各写一份必然会漂（判据 11）
CREDENTIALS = parse_credentials(os.environ)


AUTH_ENABLED = bool(CREDENTIALS)


CORS_ORIGIN = os.environ.get('CORS_ORIGIN', '*')

