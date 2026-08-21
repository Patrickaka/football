# -*- coding: utf-8 -*-
"""业务模块访问层：足球/3D/报告模块懒加载与各彩种入口导入"""

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

from src.ssq import run_prediction as ssq_run_prediction, clear_cache as ssq_clear_cache
from src.lottery import get_lottery_analyzer, run_prediction as lottery_run_prediction
from src.lottery.ml import predict_with_ml, clear_ml_cache
from src.kl8 import (
    get_kl8_analyzer, run_prediction as kl8_run_prediction,
    clear_cache as kl8_clear_cache, list_prediction_snapshots as kl8_list_snapshots,
    list_exclude_recalculations as kl8_list_recalculations,
    has_active_signal, is_prediction_ready as kl8_is_prediction_ready,
    KL8RollingBacktest, load_prize_table as kl8_load_prize_table,
    check_data_integrity as kl8_check_data_integrity,
    list_conflict_queue as kl8_list_conflict_queue,
    ACTIVE_STRATEGIES, REFERENCE_STRATEGY, KL8_PREDICTOR_VERSION,
    benjamini_hochberg_fdr, bonferroni_correction,
    validate_and_activate_strategy, resolve_play_strategy,
)
from src.common.logger import setup_logger
from src.common.paths import data_path



_FOOTBALL_MODULE = None


_BAYES_REPORT_MODULE = None


_LOTTERY3D_MODULE = None


_LOTTERY3D_ML_MODULE = None


_FOOTBALL_IMPORT_LOCK = threading.Lock()


_LOTTERY3D_IMPORT_LOCK = threading.Lock()


_BAYES_REPORT_AVAILABLE = True


def _get_football_module():
    global _FOOTBALL_MODULE
    if _FOOTBALL_MODULE is None:
        with _FOOTBALL_IMPORT_LOCK:
            if _FOOTBALL_MODULE is None:
                _FOOTBALL_MODULE = importlib.import_module('src.football')
    return _FOOTBALL_MODULE


def fetch_match_list(*args, **kwargs):
    return _get_football_module().fetch_match_list(*args, **kwargs)


def get_match_list_status():
    return _get_football_module().get_match_list_status()


def analyze_match(*args, **kwargs):
    return _get_football_module().analyze_match(*args, **kwargs)


def _get_bayes_report_module():
    global _BAYES_REPORT_MODULE, _BAYES_REPORT_AVAILABLE
    if _BAYES_REPORT_MODULE is None:
        try:
            _BAYES_REPORT_MODULE = importlib.import_module('src.football.bayes_report')
        except Exception:
            _BAYES_REPORT_AVAILABLE = False
            raise
    return _BAYES_REPORT_MODULE


def ensure_football_report(*args, **kwargs):
    return _get_bayes_report_module().ensure_football_report(*args, **kwargs)


def ensure_beidan_report(*args, **kwargs):
    return _get_bayes_report_module().ensure_beidan_report(*args, **kwargs)


def football_reportable_ids(*args, **kwargs):
    return _get_bayes_report_module().football_reportable_ids(*args, **kwargs)


def persist_beidan_recs(*args, **kwargs):
    return _get_bayes_report_module().persist_beidan_recs(*args, **kwargs)


def sync_football_reports(*args, **kwargs):
    return _get_bayes_report_module().sync_football_reports(*args, **kwargs)


def sync_beidan_reports(*args, **kwargs):
    return _get_bayes_report_module().sync_beidan_reports(*args, **kwargs)


def refresh_football_cache_index(*args, **kwargs):
    return _get_bayes_report_module().refresh_football_cache_index(*args, **kwargs)


def _get_lottery3d_module():
    global _LOTTERY3D_MODULE
    if _LOTTERY3D_MODULE is None:
        with _LOTTERY3D_IMPORT_LOCK:
            if _LOTTERY3D_MODULE is None:
                _LOTTERY3D_MODULE = importlib.import_module('src.lottery3d')
    return _LOTTERY3D_MODULE


def _get_lottery3d_ml_module():
    global _LOTTERY3D_ML_MODULE
    if _LOTTERY3D_ML_MODULE is None:
        with _LOTTERY3D_IMPORT_LOCK:
            if _LOTTERY3D_ML_MODULE is None:
                _LOTTERY3D_ML_MODULE = importlib.import_module('src.lottery3d.ml')
    return _LOTTERY3D_ML_MODULE


def run_prediction(*args, **kwargs):
    return _get_lottery3d_module().run_prediction(*args, **kwargs)


def fetch_data(*args, **kwargs):
    return _get_lottery3d_ml_module().fetch_data(*args, **kwargs)


def predict_current(*args, **kwargs):
    return _get_lottery3d_ml_module().predict_current(*args, **kwargs)


def _load_beidan_helpers():
    try:
        from src.beidan import (
            generate_beidan_recommendations,
            find_value_bets,
            summarize_beidan_history,
        )
        return generate_beidan_recommendations, find_value_bets, summarize_beidan_history
    except ModuleNotFoundError as exc:
        if exc.name == 'requests':
            raise RuntimeError('北单模块需要安装 requests；其他页面可正常使用') from exc
        raise


def _load_basketball_helpers():
    try:
        from src.basketball import (
            generate_basketball_recommendations,
            find_value_bets,
            summarize_basketball_history,
        )
        return generate_basketball_recommendations, find_value_bets, summarize_basketball_history
    except Exception as exc:
        log.error(f"加载篮球模块失败: {exc}")
        raise


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

