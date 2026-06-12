#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球模块缓存管理器
==================

功能：
1. 按自然天管理缓存（每天自动失效）
2. 支持赔率数据、球队数据、预测结果的缓存
3. 提供清除缓存和强制刷新功能
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, date
from typing import Optional, Dict, Any


class FootballCacheManager:
    """足球模块缓存管理器"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), 'cache')
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_today_str(self) -> str:
        """获取今天的日期字符串（YYYY-MM-DD）"""
        return date.today().strftime('%Y-%m-%d')
    
    def _get_cache_file_path(self, cache_type: str, key: str) -> str:
        """生成缓存文件路径"""
        today_str = self._get_today_str()
        # 使用MD5哈希key以避免文件名问题
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{today_str}_{cache_type}_{key_hash}.pkl")
    
    def _is_cache_valid(self, file_path: str) -> bool:
        """检查缓存是否有效（是否为今天创建）"""
        if not os.path.exists(file_path):
            return False
        
        # 检查文件创建日期是否为今天
        file_date = date.fromtimestamp(os.path.getctime(file_path))
        return file_date == date.today()
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """获取缓存数据"""
        file_path = self._get_cache_file_path(cache_type, key)
        
        if not self._is_cache_valid(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            import logging
            log = logging.getLogger('football')
            log.debug(f"读取缓存失败: {e}")
            return None
    
    def set(self, cache_type: str, key: str, data: Any):
        """设置缓存数据"""
        file_path = self._get_cache_file_path(cache_type, key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            import logging
            log = logging.getLogger('football')
            log.debug(f"写入缓存失败: {e}")
    
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
            except Exception as e:
                import logging
                log = logging.getLogger('football')
                log.debug(f"删除缓存文件失败: {e}")
    
    def clear_all(self):
        """清除所有缓存"""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                import logging
                log = logging.getLogger('football')
                log.debug(f"删除缓存文件失败: {e}")
    
    def clear_expired(self):
        """清除过期缓存（昨天及更早的）"""
        today_str = self._get_today_str()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.startswith(today_str):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    import logging
                    log = logging.getLogger('football')
                    log.debug(f"删除过期缓存失败: {e}")


# 全局缓存管理器实例
_global_cache_manager = FootballCacheManager()


# ==================== 便捷函数 ====================

def get_cache(cache_type: str, key: str) -> Optional[Any]:
    """获取缓存"""
    return _global_cache_manager.get(cache_type, key)


def set_cache(cache_type: str, key: str, data: Any):
    """设置缓存"""
    _global_cache_manager.set(cache_type, key, data)


def invalidate_cache(cache_type: str = None, key: str = None):
    """失效指定缓存"""
    _global_cache_manager.invalidate(cache_type, key)


def clear_all_cache():
    """清除所有缓存"""
    _global_cache_manager.clear_all()
    import logging
    log = logging.getLogger('football')
    log.info("已清除所有足球模块缓存")
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
                import logging
                log = logging.getLogger('football')
                log.debug(f"使用缓存: {cache_type} - {func.__name__}")
                return cached_data
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 设置缓存
            set_cache(cache_type, cache_key, result)
            
            return result
        return wrapper
    return decorator
