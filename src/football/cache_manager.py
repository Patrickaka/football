#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球模块缓存管理器
==================

功能：
1. 支持动态TTL缓存（基于比赛时间）
2. 支持赔率数据、球队数据、预测结果的缓存
3. 支持时间分层缓存（T-24h/T-6h/T-1h/T-15min/final）
4. 提供清除缓存和强制刷新功能

缓存策略：
- 距离开赛 >6小时：标准缓存（按天）
- 距离开赛 1-6小时：TTL 10分钟
- 距离开赛 <1小时：TTL 2分钟
"""

import os
import json
import pickle
import hashlib
import logging
import threading
from collections import OrderedDict

_log = logging.getLogger('football')

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any


class FootballCacheManager:
    """足球模块缓存管理器"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'cache')
        self.memory_limit = max(0, int(os.getenv('FOOTBALL_MEMORY_CACHE_ITEMS', '256')))
        self._memory = OrderedDict()
        self._memory_lock = threading.RLock()
        self._ensure_cache_dir()

    def _memory_get(self, file_path: str, ttl_minutes: int = None):
        if self.memory_limit <= 0:
            return None
        with self._memory_lock:
            entry = self._memory.get(file_path)
            if not entry:
                return None
            cached_at, value = entry
            max_age = ttl_minutes * 60 if ttl_minutes is not None else 86400
            if datetime.now().timestamp() - cached_at >= max_age:
                self._memory.pop(file_path, None)
                return None
            self._memory.move_to_end(file_path)
            return value

    def _memory_set(self, file_path: str, value: Any):
        if self.memory_limit <= 0:
            return
        with self._memory_lock:
            self._memory[file_path] = (datetime.now().timestamp(), value)
            self._memory.move_to_end(file_path)
            while len(self._memory) > self.memory_limit:
                self._memory.popitem(last=False)
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在。

        `exist_ok=True` 不是保险起见：模块级的单例会在每个进程 import 时构造，
        并发跑测试或多进程启动时，先检查再创建之间必然有人插队。
        """
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_today_str(self) -> str:
        """获取今天的日期字符串（YYYY-MM-DD）"""
        return date.today().strftime('%Y-%m-%d')
    
    def _get_cache_file_path(self, cache_type: str, key: str, time_layer: str = None) -> str:
        """生成缓存文件路径"""
        today_str = self._get_today_str()
        # 使用MD5哈希key以避免文件名问题
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:16]
        
        if time_layer:
            return os.path.join(self.cache_dir, f"{today_str}_{cache_type}_{key_hash}_{time_layer}.pkl")
        return os.path.join(self.cache_dir, f"{today_str}_{cache_type}_{key_hash}.pkl")
    
    def _is_cache_valid(self, file_path: str, ttl_minutes: int = None) -> bool:
        """
        检查缓存是否有效
        
        参数：
            file_path: 缓存文件路径
            ttl_minutes: TTL（分钟），如果为None则按天检查
        """
        if not os.path.exists(file_path):
            return False
        
        if ttl_minutes is None:
            # 按天检查
            file_date = date.fromtimestamp(os.path.getctime(file_path))
            return file_date == date.today()
        else:
            # 按TTL检查
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            return datetime.now() - file_mtime < timedelta(minutes=ttl_minutes)
    
    def _get_time_to_match(self, match_time_str: str) -> Optional[timedelta]:
        """
        计算距离开赛的时间
        
        参数：
            match_time_str: 比赛时间字符串
        
        返回：
            距离开赛的时间差，如果解析失败返回None
        """
        try:
            # 尝试多种格式解析
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m-%d %H:%M', '%H:%M']:
                try:
                    if fmt == '%H:%M':
                        # 只有时间，假设是今天
                        match_time = datetime.strptime(match_time_str, fmt)
                        match_datetime = datetime.now().replace(hour=match_time.hour, minute=match_time.minute, second=0)
                    elif fmt == '%m-%d %H:%M':
                        match_datetime = datetime.strptime(f"{datetime.now().year}-{match_time_str}", '%Y-%m-%d %H:%M')
                    else:
                        match_datetime = datetime.strptime(match_time_str, fmt)
                    return match_datetime - datetime.now()
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _get_ttl_for_match(self, match_time_str: str) -> Optional[int]:
        """
        根据比赛时间获取TTL（分钟）
        
        返回：
            TTL分钟数，如果无法确定返回None（使用默认按天缓存）
        """
        time_to_match = self._get_time_to_match(match_time_str)
        
        if time_to_match is None:
            return None
        
        if time_to_match < timedelta(minutes=15):
            return 2  # 15分钟内，TTL 2分钟
        elif time_to_match < timedelta(hours=1):
            return 10  # 1小时内，TTL 10分钟
        elif time_to_match < timedelta(hours=6):
            return 60  # 6小时内，TTL 1小时
        else:
            return None  # 超过6小时，使用按天缓存
    
    def _get_time_layer(self, match_time_str: str) -> str:
        """
        获取时间分层标识
        
        返回：
            时间分层标识: 'T24h', 'T6h', 'T1h', 'T15min', 'final'
        """
        time_to_match = self._get_time_to_match(match_time_str)
        
        if time_to_match is None:
            return 'unknown'
        
        if time_to_match < timedelta(minutes=15):
            return 'T15min'
        elif time_to_match < timedelta(hours=1):
            return 'T1h'
        elif time_to_match < timedelta(hours=6):
            return 'T6h'
        elif time_to_match < timedelta(hours=24):
            return 'T24h'
        else:
            return 'early'
    
    def get(self, cache_type: str, key: str, match_time_str: str = None) -> Optional[Any]:
        """
        获取缓存数据
        
        参数：
            cache_type: 缓存类型
            key: 缓存键
            match_time_str: 比赛时间（可选），用于动态TTL判断
        
        返回：
            缓存数据，如果无效返回None
        """
        file_path = self._get_cache_file_path(cache_type, key)
        
        # 根据比赛时间决定TTL
        ttl_minutes = self._get_ttl_for_match(match_time_str) if match_time_str else None

        memory_value = self._memory_get(file_path, ttl_minutes)
        if memory_value is not None:
            return memory_value
        
        if not self._is_cache_valid(file_path, ttl_minutes):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                value = pickle.load(f)
            self._memory_set(file_path, value)
            return value
        except Exception as e:
            _log.debug(f"读取缓存失败: {e}")
            return None
    
    def set(self, cache_type: str, key: str, data: Any, match_time_str: str = None):
        """
        设置缓存数据
        
        参数：
            cache_type: 缓存类型
            key: 缓存键
            data: 缓存数据
            match_time_str: 比赛时间（可选），用于保存时间分层
        """
        # 保存标准缓存
        file_path = self._get_cache_file_path(cache_type, key)
        
        try:
            temp_path = file_path + f'.{os.getpid()}.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            os.replace(temp_path, file_path)
            self._memory_set(file_path, data)
        except Exception as e:
            _log.debug(f"写入缓存失败: {e}")
        
        # 如果提供了比赛时间，同时保存时间分层缓存
        if match_time_str:
            time_layer = self._get_time_layer(match_time_str)
            layer_file_path = self._get_cache_file_path(cache_type, key, time_layer)
            
            try:
                temp_layer_path = layer_file_path + f'.{os.getpid()}.tmp'
                with open(temp_layer_path, 'wb') as f:
                    pickle.dump(data, f)
                os.replace(temp_layer_path, layer_file_path)
                
                _log.debug(f"保存时间分层缓存: {time_layer}")
            except Exception as e:
                _log.debug(f"写入时间分层缓存失败: {e}")
    
    def get_time_layer_cache(self, cache_type: str, key: str, time_layer: str) -> Optional[Any]:
        """
        获取指定时间分层的缓存
        
        参数：
            cache_type: 缓存类型
            key: 缓存键
            time_layer: 时间分层标识
        
        返回：
            缓存数据，如果不存在返回None
        """
        file_path = self._get_cache_file_path(cache_type, key, time_layer)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            _log.debug(f"读取时间分层缓存失败: {e}")
            return None
    
    def invalidate(self, cache_type: str = None, key: str = None):
        """
        失效缓存
        
        参数：
            cache_type: 缓存类型（可选），如果为None则失效所有类型
            key: 缓存键（可选），如果为None则失效该类型下所有缓存
        """
        today_str = self._get_today_str()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.startswith(today_str):
                continue
            
            if cache_type and not filename.startswith(f"{today_str}_{cache_type}"):
                continue
            
            if key:
                key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:16]
                if key_hash not in filename:
                    continue
            
            file_path = os.path.join(self.cache_dir, filename)
            try:
                os.remove(file_path)
                with self._memory_lock:
                    self._memory.pop(file_path, None)
            except Exception as e:
                _log.debug(f"删除缓存文件失败: {e}")
    
    def clear_all(self):
        """清除所有缓存"""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                _log.debug(f"删除缓存文件失败: {e}")
        with self._memory_lock:
            self._memory.clear()
    
    def clear_expired(self):
        """清除过期缓存（昨天及更早的）"""
        today_str = self._get_today_str()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.startswith(today_str):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    _log.debug(f"删除过期缓存失败: {e}")
        with self._memory_lock:
            for file_path in list(self._memory):
                if not os.path.basename(file_path).startswith(today_str):
                    self._memory.pop(file_path, None)


# 全局缓存管理器实例
_global_cache_manager = FootballCacheManager()


# ==================== 便捷函数 ====================

def get_cache(cache_type: str, key: str, match_time: str = None) -> Optional[Any]:
    """
    获取缓存
    
    参数：
        cache_type: 缓存类型
        key: 缓存键
        match_time: 比赛时间（可选），用于动态TTL判断
    """
    return _global_cache_manager.get(cache_type, key, match_time)


def set_cache(cache_type: str, key: str, data: Any, match_time: str = None):
    """
    设置缓存
    
    参数：
        cache_type: 缓存类型
        key: 缓存键
        data: 缓存数据
        match_time: 比赛时间（可选），用于保存时间分层
    """
    _global_cache_manager.set(cache_type, key, data, match_time)


def get_time_layer_cache(cache_type: str, key: str, time_layer: str) -> Optional[Any]:
    """
    获取指定时间分层的缓存
    
    参数：
        cache_type: 缓存类型
        key: 缓存键
        time_layer: 时间分层标识（如 'T24h', 'T6h', 'T1h', 'T15min'）
    """
    return _global_cache_manager.get_time_layer_cache(cache_type, key, time_layer)


def invalidate_cache(cache_type: str = None, key: str = None):
    """失效指定缓存"""
    _global_cache_manager.invalidate(cache_type, key)


def clear_all_cache():
    """清除所有缓存"""
    _global_cache_manager.clear_all()
    _log.info("已清除所有足球模块缓存")
    return {'status': 'success', 'message': '所有缓存已清空'}


def clear_expired_cache():
    """清除过期缓存"""
    _global_cache_manager.clear_expired()


# ==================== 装饰器 ====================

def cached(cache_type: str, ttl_days: int = 1):
    """
    缓存装饰器
    
    参数：
        cache_type: 缓存类型标识
        ttl_days: 缓存有效期（天数），默认为1天（自然天）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
            cache_key = f"{func.__name__}_{'_'.join(key_parts)}"
            
            # 尝试获取缓存
            cached_data = get_cache(cache_type, cache_key)
            if cached_data is not None:
                _log.debug(f"使用缓存: {cache_type} - {func.__name__}")
                return cached_data
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 设置缓存
            set_cache(cache_type, cache_key, result)
            
            return result
        return wrapper
    return decorator
