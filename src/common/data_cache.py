"""
数据缓存工具 - 每天只抓取一次数据

用于所有彩票模块的历史开奖数据缓存
"""

import os
import json
import time
from datetime import datetime

def get_cache_path(module_name):
    """获取缓存文件路径"""
    cache_dir = os.path.join(os.path.dirname(__file__), '../../data')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{module_name}_cache.json')

def get_today_str():
    """获取今天的日期字符串 YYYY-MM-DD"""
    return datetime.now().strftime('%Y-%m-%d')

def load_cached_data(module_name):
    """加载缓存数据"""
    cache_path = get_cache_path(module_name)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 检查是否是今天的数据
                if data.get('date') == get_today_str():
                    return data.get('data', None)
        except Exception:
            pass
    return None

def save_cached_data(module_name, data):
    """保存缓存数据"""
    cache_path = get_cache_path(module_name)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'date': get_today_str(),
                'timestamp': time.time(),
                'data': data
            }, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def cached_fetch(module_name, fetch_func, force_refresh=False):
    """
    带缓存的数据抓取
    
    参数：
        module_name: 模块名称（用于缓存文件名）
        fetch_func: 数据抓取函数
        force_refresh: 是否强制刷新
    
    返回：
        数据（可能是缓存的或新抓取的）
    """
    # 如果不是强制刷新，先尝试加载缓存
    if not force_refresh:
        cached = load_cached_data(module_name)
        if cached is not None:
            return cached
    
    # 缓存不存在或需要刷新，调用抓取函数
    data = fetch_func()
    
    # 保存到缓存
    if data is not None:
        save_cached_data(module_name, data)
    
    return data

def is_cache_valid(module_name):
    """检查缓存是否有效（今天的数据）"""
    return load_cached_data(module_name) is not None

def clear_cache(module_name=None):
    """
    清除缓存
    
    参数：
        module_name: 模块名称，如果为None则清除所有缓存
    """
    cache_dir = os.path.join(os.path.dirname(__file__), '../../data')
    if not os.path.exists(cache_dir):
        return
    
    if module_name:
        cache_path = get_cache_path(module_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
    else:
        for filename in os.listdir(cache_dir):
            if filename.endswith('_cache.json'):
                os.remove(os.path.join(cache_dir, filename))