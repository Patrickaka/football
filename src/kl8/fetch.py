"""
快乐8数据抓取模块
=================
从网上抓取快乐8开奖历史数据并保存到本地。
使用免费API: http://api.huiniao.top (type=klb)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from src.common.paths import data_path
from src.common.logger import setup_logger

log = setup_logger('kl8_fetch')

KL8_HISTORY_FILE = data_path('kl8_history.json')
KL8_CACHE_FILE = data_path('kl8_cache.json')

# API数据源中的20个号码字段名
NUM_FIELDS = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'
]


def fetch_kl8_data(pages: int = 10, per_page: int = 50) -> List[Dict]:
    """从免费API抓取快乐8开奖数据

    API: http://api.huiniao.top/interface/home/lotteryHistory?type=klb
    返回格式: [{'issue': '2026165', 'numbers': [1,9,10,...], 'date': '2026-06-24'}]
    """
    import urllib.request

    all_results = []

    for page in range(1, pages + 1):
        url = f'http://api.huiniao.top/interface/home/lotteryHistory?type=klb&page={page}&limit={per_page}'
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode('utf-8'))

            if raw.get('code') != 1:
                log.warning(f'API返回错误: code={raw.get("code")}, info={raw.get("info")}')
                continue

            data_obj = raw.get('data', {})
            list_data = data_obj.get('data', {}).get('list', [])
            if not list_data:
                # 可能是最后一页没有数据了
                break

            for item in list_data:
                issue = item.get('code', '')
                day = item.get('day', '')

                # 提取20个号码
                nums = []
                for field in NUM_FIELDS:
                    val = item.get(field)
                    if val is not None:
                        # 可能是字符串或整数
                        try:
                            n = int(str(val).strip())
                            if 1 <= n <= 80:
                                nums.append(n)
                        except (ValueError, TypeError):
                            continue

                if len(nums) == 20:
                    nums.sort()
                    all_results.append({
                        'issue': issue,
                        'numbers': nums,
                        'date': day,
                    })
                else:
                    log.warning(f'期号{issue}号码数量异常: {len(nums)}')

            total_pages = data_obj.get('data', {}).get('totalPage', 0)
            if page >= total_pages:
                break

        except Exception as e:
            log.warning(f'API第{page}页抓取失败: {e}')
            continue

    if all_results:
        # 按期号降序排列(最新在前)
        all_results.sort(key=lambda x: x['issue'], reverse=True)
        # 去重
        seen = set()
        unique = []
        for r in all_results:
            if r['issue'] not in seen:
                seen.add(r['issue'])
                unique.append(r)
        log.info(f'成功抓取{len(unique)}期快乐8数据')
        return unique

    log.error('所有数据源均抓取失败')
    return []


def save_kl8_data(data: List[Dict]):
    """保存快乐8数据到本地"""
    path = Path(KL8_HISTORY_FILE)
    path.write_text(
        json.dumps({'results': data}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    log.info(f'快乐8数据保存到 {path}, 共{len(data)}期')


def load_cached_data() -> Optional[List[Dict]]:
    """从缓存文件加载快乐8数据"""
    path = Path(KL8_CACHE_FILE)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        data = raw.get('data', [])
        # 检查缓存是否过期(超过24小时)
        ts = raw.get('timestamp', 0)
        if time.time() - ts > 86400:
            return None
        return data
    except Exception:
        return None


def save_cached_data(data: List[Dict]):
    """保存到缓存文件"""
    path = Path(KL8_CACHE_FILE)
    cache = {
        'data': data,
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d'),
    }
    path.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding='utf-8'
    )
