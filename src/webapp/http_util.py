# -*- coding: utf-8 -*-
"""JSON 序列化工具"""

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


def _sanitize_json(obj):
    """递归把非有限浮点（inf/nan，含 numpy 标量/数组）替换为 None。

    json.dumps 默认 allow_nan=True 会输出 Infinity/NaN 字面量，浏览器 JSON.parse
    无法解析（distance=inf 等业务哨兵值即属此类），故序列化前统一清洗为 null。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes, bytearray)):
        return _sanitize_json(obj.tolist())
    return obj

