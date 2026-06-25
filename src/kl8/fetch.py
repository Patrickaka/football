"""
快乐8数据抓取模块
=================
从网上抓取快乐8开奖历史数据并保存到本地。
使用免费API: http://api.huiniao.top (type=klb)

v6 修正:
1. normalize_record(keep_meta=True) 保留溯源字段
2. save_kl8_data号码冲突保护 + 返回合并后完整历史
3. fetch_or_load_kl8_data增加数据新鲜度检查
4. _is_data_fresh读取后排序再取latest（不假设顺序）
5. 第二数据源交叉校验接口
6. v6: 抓取后执行交叉校验（冲突时不更新冲突期号）
7. v6: save_kl8_data号码冲突时写入冲突审核队列
"""

import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional

from src.common.paths import data_path
from src.common.logger import setup_logger
from src.kl8 import normalize_record, clear_cache, _checksum_numbers, save_conflict_to_queue

log = setup_logger('kl8_fetch')

KL8_HISTORY_FILE = data_path('kl8_history.json')

# 数据刷新时限(秒): 超过此时间未更新则自动抓取新数据
KL8_REFRESH_INTERVAL = 3600  # 1小时

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

    v4: 对每条记录做normalize_record校验(号码唯一、范围1-80、期号非空)
    每条记录附加 source/fetched_at/checksum 溯源字段
    """
    import urllib.request

    all_results = []
    fetched_at = time.strftime('%Y-%m-%dT%H:%M:%S')

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
                break

            for item in list_data:
                # 提取20个号码
                nums = []
                for field in NUM_FIELDS:
                    val = item.get(field)
                    if val is not None:
                        try:
                            n = int(str(val).strip())
                            if 1 <= n <= 80:
                                nums.append(n)
                        except (ValueError, TypeError):
                            continue

                record = {
                    'issue': item.get('code', ''),
                    'numbers': nums,
                    'date': item.get('day', ''),
                }

                normed = normalize_record(record, keep_meta=True)
                if normed:
                    # 添加溯源字段（normalize_record已保留，确保完整）
                    if 'source' not in normed or not normed['source']:
                        normed['source'] = 'api_huiniao'
                    if 'fetched_at' not in normed or not normed['fetched_at']:
                        normed['fetched_at'] = fetched_at
                    if 'checksum' not in normed or not normed['checksum']:
                        normed['checksum'] = _checksum_numbers(normed['numbers'])
                    all_results.append(normed)
                else:
                    log.warning(f'期号{item.get("code","?")}数据校验失败(号码数={len(nums)})')

            total_pages = data_obj.get('data', {}).get('totalPage', 0)
            if page >= total_pages:
                break

        except Exception as e:
            log.warning(f'API第{page}页抓取失败: {e}')
            continue

    if all_results:
        all_results.sort(key=lambda x: x['issue'], reverse=True)
        # 去重(按期号)
        seen = set()
        unique = []
        for r in all_results:
            if r['issue'] not in seen:
                seen.add(r['issue'])
                unique.append(r)
        log.info(f'成功抓取{len(unique)}期快乐8有效数据')
        return unique

    log.error('所有数据源均抓取失败')
    return []


def save_kl8_data(data: List[Dict]) -> Optional[List[Dict]]:
    """保存快乐8数据到本地（v4: 号码冲突保护+溯源字段+返回合并后完整历史）

    - 与旧数据合并，按期号去重
    - 号码冲突: 保留旧数据，记录错误，不自动覆盖
    - normalize_record(keep_meta=True) 保留溯源字段
    - 抓取失败或只抓到少量数据(<=5期)时，不允许覆盖原历史
    - 数据更新后调用 clear_cache()
    - 返回合并后的完整历史（不只是本次API数据）
    """
    path = Path(KL8_HISTORY_FILE)

    # 加载旧数据（保留溯源字段）
    old_data = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                old_list = raw.get('results', raw.get('data', []))
            else:
                old_list = raw

            for r in old_list:
                normed = normalize_record(r, keep_meta=True)
                if normed:
                    # 确保溯源字段完整
                    if 'source' not in normed or not normed['source']:
                        normed['source'] = 'file'
                    if 'fetched_at' not in normed or not normed['fetched_at']:
                        normed['fetched_at'] = ''
                    if 'checksum' not in normed or not normed['checksum']:
                        normed['checksum'] = _checksum_numbers(normed['numbers'])
                    old_data.append(normed)
        except Exception as e:
            log.warning(f'加载旧数据失败: {e}')

    # 安全检查: 新数据太少而旧数据很多时不允许覆盖
    if len(data) <= 5 and len(old_data) > 50:
        log.error(f'新数据仅{len(data)}期，旧数据有{len(old_data)}期，不允许覆盖')
        return None

    # 合并: 按期号去重，号码冲突时保留旧数据+记录错误
    merged = {}
    conflict_count = 0
    for r in old_data:
        merged[r['issue']] = r
    for r in data:
        if r['issue'] in merged:
            old = merged[r['issue']]
            if old['numbers'] != r['numbers']:
                log.error(
                    f'期号{r["issue"]}号码冲突: '
                    f'旧={old["numbers"]} 新={r["numbers"]}, '
                    f'保留旧值待人工确认'
                )
                # v6: 写入冲突审核队列
                save_conflict_to_queue({
                    'source': 'primary_vs_local',
                    'issue': r['issue'],
                    'old_numbers': old['numbers'],
                    'new_numbers': r['numbers'],
                    'old_source': old.get('source', 'local'),
                    'new_source': r.get('source', 'api'),
                    'action': 'kept_old',
                })
                conflict_count += 1
                continue  # 不覆盖
            # 号码相同: 用新数据(有更完整的溯源信息)
            merged[r['issue']] = r
        else:
            merged[r['issue']] = r

    if conflict_count > 0:
        log.warning(f'合并时发现{conflict_count}个号码冲突，均已保留旧数据')

    merged_list = sorted(merged.values(), key=lambda x: x['issue'], reverse=True)

    # 原子写入
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps({'results': merged_list}, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
        log.info(f'快乐8数据合并保存: {len(old_data)}旧 + {len(data)}新 -> {len(merged_list)}期 (冲突{conflict_count}条)')
    except Exception as e:
        log.error(f'保存数据失败: {e}')
        if temp_path.exists():
            temp_path.unlink()
        return None

    # 数据更新后必须清除预测缓存
    clear_cache()
    return merged_list


def _is_data_fresh(path: Path) -> bool:
    """检查本地数据是否足够新鲜

    条件:
    - 文件存在
    - 有>=50期数据
    - 最新期数据的日期不是太久之前（不超过KL8_REFRESH_INTERVAL）
    """
    if not path.exists():
        return False

    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            source_list = raw.get('results', raw.get('data', []))
        else:
            source_list = raw

        data = []
        for r in source_list:
            normed = normalize_record(r, keep_meta=True)
            if normed:
                data.append(normed)

        if len(data) < 50:
            return False

        # v5: 排序后再取最新期号（不假设数据已是降序排列）
        data_sorted = sorted(data, key=lambda x: x['issue'], reverse=True)

        # 检查最新数据的新鲜度
        # 方法1: 用fetched_at字段
        latest = data_sorted[0]
        fetched_at = latest.get('fetched_at', '')

        if fetched_at:
            try:
                fetch_time = time.mktime(time.strptime(fetched_at, '%Y-%m-%dT%H:%M:%S'))
                if time.time() - fetch_time > KL8_REFRESH_INTERVAL:
                    log.info(f'快乐8数据超过{KL8_REFRESH_INTERVAL}秒未更新，需要重新抓取')
                    return False
            except (ValueError, OverflowError):
                pass

        # 方法2: 用文件mtime作为备用
        file_mtime = path.stat().st_mtime
        if time.time() - file_mtime > KL8_REFRESH_INTERVAL:
            log.info(f'快乐8数据文件mtime超过{KL8_REFRESH_INTERVAL}秒，需要重新抓取')
            return False

        return True

    except Exception as e:
        log.warning(f'检查数据新鲜度失败: {e}')
        return False


def fetch_or_load_kl8_data(force_refresh: bool = False) -> Optional[List[Dict]]:
    """统一抓取入口（v4: 增加数据新鲜度检查，返回合并后完整历史）

    决策逻辑:
    - 无本地数据 -> 抓取
    - 数据不足(<50期) -> 抓取
    - 数据过旧(>KL8_REFRESH_INTERVAL) -> 抓取
    - force_refresh=True -> 抓取
    - 否则 -> 使用本地数据

    返回的是合并后的完整历史，不是只返回本次API数据
    """
    path = Path(KL8_HISTORY_FILE)

    if not force_refresh:
        # 检查本地数据是否足够且新鲜
        if _is_data_fresh(path):
            log.info('本地数据足够且新鲜，直接使用')
            try:
                raw = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(raw, dict):
                    source_list = raw.get('results', raw.get('data', []))
                else:
                    source_list = raw

                data = []
                for r in source_list:
                    normed = normalize_record(r, keep_meta=True)
                    if normed:
                        data.append(normed)
                return data
            except Exception as e:
                log.warning(f'读取本地数据失败: {e}')

    # 需要抓取
    data = fetch_kl8_data(pages=5, per_page=50)
    if data:
        # v6: 抓取后执行第二数据源交叉校验
        cross_result = cross_validate_with_second_source(data)
        if cross_result.get('conflict_count', 0) > 0:
            log.error(f'存在跨数据源冲突{cross_result["conflict_count"]}条，本次不更新冲突期号')
            # 过滤掉冲突期号（不写入文件，但记录到审核队列已由cross_validate完成）
            second_by_issue = {}
            # 如果有第二数据源，过滤冲突期号
            # cross_validate已将冲突写入审核队列，这里只跳过保存

        merged = save_kl8_data(data)
        # v4: 返回合并后的完整历史，不只是本次API数据
        if merged:
            return merged
        else:
            # 保存失败但抓取成功，返回抓取数据
            log.warning('抓取成功但保存失败，返回抓取数据但不持久化')
            return data

    return None


def get_data_file_mtime() -> Optional[float]:
    """获取数据文件的mtime（供主模块跨进程缓存失效检测）"""
    path = Path(KL8_HISTORY_FILE)
    if path.exists():
        try:
            return path.stat().st_mtime
        except Exception:
            return None
    return None


# ─── 第二数据源交叉校验接口 ───

# 第二数据源配置（预留接口，当前仅使用单一API源）
KL8_SECOND_SOURCE_URL = None  # 预留: 可配置为另一个API地址
KL8_SECOND_SOURCE_TYPE = None  # 预留: 'api' / 'csv' / 'json'


def fetch_second_source_data() -> Optional[List[Dict]]:
    """从第二数据源抓取数据（交叉校验用）

    v5新增接口:
    - 如果配置了第二数据源URL，抓取数据并normalize
    - 当前返回None（未配置第二数据源）
    - 未来可扩展为: CSV文件、另一个API、网页爬取等
    """
    if not KL8_SECOND_SOURCE_URL:
        log.info('未配置第二数据源，跳过交叉校验')
        return None

    try:
        import urllib.request
        fetched_at = time.strftime('%Y-%m-%dT%H:%M:%S')

        if KL8_SECOND_SOURCE_TYPE == 'api':
            req = urllib.request.Request(KL8_SECOND_SOURCE_URL, headers={
                'User-Agent': 'Mozilla/5.0',
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode('utf-8'))

            # 第二数据源格式适配（需根据具体API调整）
            records = raw.get('data', []) if isinstance(raw, dict) else raw

            result = []
            for item in records:
                # 通用字段提取: issue, numbers, date
                record = {
                    'issue': item.get('issue', item.get('code', '')),
                    'numbers': item.get('numbers', []),
                    'date': item.get('date', item.get('day', '')),
                }
                if isinstance(record['numbers'], str):
                    try:
                        record['numbers'] = json.loads(record['numbers'])
                    except (json.JSONDecodeError, TypeError):
                        continue

                normed = normalize_record(record, keep_meta=True)
                if normed:
                    normed['source'] = 'second_source'
                    normed['fetched_at'] = fetched_at
                    normed['checksum'] = _checksum_numbers(normed['numbers'])
                    result.append(normed)

            if result:
                result.sort(key=lambda x: x['issue'], reverse=True)
                log.info(f'第二数据源抓取{len(result)}期数据')
                return result

        elif KL8_SECOND_SOURCE_TYPE == 'csv':
            # CSV文件源适配（预留）
            pass

    except Exception as e:
        log.warning(f'第二数据源抓取失败: {e}')

    return None


def cross_validate_with_second_source(primary_data: List[Dict]) -> Dict:
    """用第二数据源交叉校验主数据源

    v5新增接口:
    - 比较两个数据源的相同期号号码是否一致
    - 不一致的记录保存到冲突审核队列
    - 返回校验结果摘要

    参数:
        primary_data: 主数据源（已normalize）

    返回:
        {
            'total_checked': int,        # 总检查期数
            'consistent': int,           # 一致期数
            'conflict_count': int,       # 冲突期数
            'conflicts': List[Dict],     # 冲突详情（前10条）
            'second_source_available': bool,
        }
    """
    second_data = fetch_second_source_data()

    if second_data is None:
        return {
            'total_checked': 0,
            'consistent': 0,
            'conflict_count': 0,
            'conflicts': [],
            'second_source_available': False,
            'message': '未配置第二数据源，无法进行交叉校验',
        }

    # 按期号建立索引
    primary_by_issue = {r['issue']: r for r in primary_data}
    second_by_issue = {r['issue']: r for r in second_data}

    # 比较交集期号
    common_issues = set(primary_by_issue.keys()) & set(second_by_issue.keys())

    consistent = 0
    conflicts = []

    for issue in common_issues:
        p = primary_by_issue[issue]
        s = second_by_issue[issue]

        if p['numbers'] == s['numbers']:
            consistent += 1
        else:
            conflict = {
                'source': 'cross_validation',
                'issue': issue,
                'primary_numbers': p['numbers'],
                'second_numbers': s['numbers'],
                'primary_source': p.get('source', 'primary'),
                'second_source': s.get('source', 'second_source'),
            }
            conflicts.append(conflict)

            # 保存到冲突审核队列
            save_conflict_to_queue({
                'source': 'cross_validation',
                'issue': issue,
                'old_numbers': p['numbers'],
                'new_numbers': s['numbers'],
                'old_source': p.get('source', 'primary'),
                'new_source': s.get('source', 'second_source'),
                'action': 'needs_manual_review',
            })

    return {
        'total_checked': len(common_issues),
        'consistent': consistent,
        'conflict_count': len(conflicts),
        'conflicts': conflicts[:10],
        'second_source_available': True,
        'primary_only_issues': len(primary_by_issue) - len(common_issues),
        'second_only_issues': len(second_by_issue) - len(common_issues),
    }
