# -*- coding: utf-8 -*-
"""北单接口 handler（mixin）"""

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

from .lazy_modules import (
    _BAYES_REPORT_AVAILABLE, _load_beidan_helpers, persist_beidan_recs,
)
from .jobs import (
    _attach_bayes_report_url, _trigger_beidan_report_sync,
)
from .beidan_cache import (
    beidan_cache_key, read_beidan_cache, refresh_beidan_async, write_beidan_cache,
)

class BeidanApiMixin:
    def _beidan_payload(self, params):
        """获取北单推荐预测"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            bet_types = params.get('types', ['spf,rqspf,zjq'])[0].split(',')
            
            force_refresh = params.get('force_refresh', ['false'])[0].lower() == 'true'

            self._log.info(f'北单推荐请求: date={date}, source={source}, types={bet_types}')

            cache_key = beidan_cache_key(date, source, bet_types)
            generate_beidan_recommendations, _, _ = _load_beidan_helpers()

            def _compute():
                return generate_beidan_recommendations(
                    date=date, bet_types=bet_types, source=source)

            cached, fresh = read_beidan_cache(cache_key)
            if cached is None:
                # 从来没算过，只能同步算这一次；之后都走缓存 + 后台刷新
                result = _compute()
                if 'error' not in result:
                    write_beidan_cache(cache_key, result)
            else:
                # 有缓存就立刻返回。整页重算在线上要 160 秒，远超网关超时，
                # 让用户等于必然 504，所以过期只触发后台刷新、不阻塞这次请求。
                result = cached
                if force_refresh or not fresh:
                    started = refresh_beidan_async(cache_key, _compute)
                    # 无论本次是否真的起了新线程，都有一轮刷新在跑（未起说明已有同键在刷），
                    # 都要告诉前端「正在刷新」，否则它不会回来取更新后的数据。
                    result = dict(result)
                    result['refreshing'] = True
                    self._log.info('北单返回缓存并%s后台刷新: %s',
                                   '触发' if started else '复用进行中的', cache_key)
                else:
                    self._log.info('北单推荐命中缓存: %s', cache_key)

            if 'error' in result:
                return result

            # 为北单推荐持久化 rec（供按需生成报告）并附加深度报告 URL
            recs = result.get('recommendations')
            if isinstance(recs, list):
                if _BAYES_REPORT_AVAILABLE:
                    persisted = set(persist_beidan_recs(recs))
                    for rec in recs:
                        mid = str(rec.get('match_id') or '')
                        if mid and mid in persisted:
                            rec['bayes_report_url'] = f"/reports/beidan_bayes_{mid}.html"
                else:
                    _attach_bayes_report_url(recs, kind='beidan')
                # 后台预生成深度报告：无报告则生成、变盘则重生成
                _trigger_beidan_report_sync(recs)

            return {'result': result}
        except Exception as e:
            self._log.error('北单推荐失败', exc_info=True)
            return {'error': f'北单推荐失败: {str(e)}'}


    def _beidan_matches_payload(self, params):
        """获取北单比赛列表"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            
            if source == 'okooo':
                from src.beidan import fetch_okooo_schedule
                matches = fetch_okooo_schedule(date=date)
            else:
                from src.beidan import fetch_beidan_schedule
                matches = fetch_beidan_schedule(date=date, source=source)
            
            return {'matches': matches}
        except Exception as e:
            self._log.error('北单比赛列表获取失败', exc_info=True)
            return {'error': f'获取比赛列表失败: {str(e)}'}


    def _beidan_value_payload(self, params):
        """获取北单价值投注推荐"""
        try:
            date = params.get('date', [None])[0]
            source = params.get('source', ['okooo'])[0]
            threshold = float(params.get('threshold', [0.05])[0])
            
            _, find_value_bets, _ = _load_beidan_helpers()
            result = find_value_bets(date=date, threshold=threshold, source=source)
            
            if 'error' in result:
                return {'error': result['error']}
            
            return {'result': result}
        except Exception as e:
            self._log.error('北单价值投注失败', exc_info=True)
            return {'error': f'价值投注分析失败: {str(e)}'}


    def _beidan_history_payload(self, params):
        """获取北单预测记录摘要"""
        try:
            limit = int(params.get('limit', ['200'])[0])
            _, _, summarize_beidan_history = _load_beidan_helpers()
            return {'result': summarize_beidan_history(limit=limit)}
        except Exception as e:
            self._log.error('北单预测记录获取失败', exc_info=True)
            return {'error': f'北单预测记录获取失败: {str(e)}'}

