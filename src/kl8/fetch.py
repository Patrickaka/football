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


def fetch_kl8_data(
    pages: int = 10,
    per_page: int = 50,
    start_page: int = 1,
    page_delay: float = 0.0,
) -> List[Dict]:
    """从免费API抓取快乐8开奖数据

    v9.2改动:
    - 新增 start_page 参数，支持分批抓取（不再永远从第1页开始）
    - 新增 page_delay 参数，每页间隔秒数（不再只在pages>10时才延时）
    - 每页失败自动重试2次（不再只在401时才重试1次）
    - 重试之间等待递增（1秒→3秒）

    API: http://api.huiniao.top/interface/home/lotteryHistory?type=klb
    返回格式: [{'issue': '2026165', 'numbers': [1,9,10,...], 'date': '2026-06-24'}]
    """
    import urllib.request

    all_results = []
    fetched_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    max_retries = 2  # 每页最多重试2次（总共最多3次请求）

    for page in range(start_page, start_page + pages):
        url = (
            'http://api.huiniao.top/interface/home/lotteryHistory'
            f'?type=klb&page={page}&limit={per_page}'
        )

        # v9.2: 每页间隔延时（不再只在补数时才延时）
        if page_delay > 0 and page > start_page:
            time.sleep(page_delay)

        success = False
        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = json.loads(resp.read().decode('utf-8'))

                if raw.get('code') != 1:
                    log.warning(
                        f'API返回错误: code={raw.get("code")}, '
                        f'info={raw.get("info")}, 第{page}页, 第{attempt+1}次尝试'
                    )
                    # 401限流时等待更长
                    if raw.get('code') == 401:
                        wait_time = 5 * (attempt + 1)
                        log.info(f'API限流(401)，等待{wait_time}秒后重试')
                        time.sleep(wait_time)
                        continue  # 重试
                    else:
                        if attempt < max_retries:
                            time.sleep(1)
                            continue
                        break  # 非限流错误，不重试

                # 请求成功
                success = True
                break

            except Exception as e:
                log.warning(
                    f'API第{page}页抓取失败(第{attempt+1}次): {e}'
                )
                if attempt < max_retries:
                    wait_time = 1 + 2 * attempt  # 1秒→3秒递增
                    time.sleep(wait_time)
                    continue
                break

        if not success:
            log.warning(f'API第{page}页最终失败（已重试{max_retries}次）')
            continue

        # 解析数据（success=True时raw已有合法数据）
        data_obj = raw.get('data', {})
        list_data = data_obj.get('data', {}).get('list', [])
        if not list_data:
            # 空页 = 已到末尾
            break

        for item in list_data:
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

    if all_results:
        all_results.sort(key=lambda x: x['issue'], reverse=True)
        seen = set()
        unique = []
        for r in all_results:
            if r['issue'] not in seen:
                seen.add(r['issue'])
                unique.append(r)
        log.info(f'成功抓取{len(unique)}期快乐8有效数据(start_page={start_page}, pages={pages})')
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

    # 镜像进 foundation/store。迁移之后开奖数据有两个写入方，不接上的话
    # 库里的数据会越来越旧且没有任何报错。失败只告警——抓取是主链路，
    # 落库是旁路，不该让后者的可用性绑架前者。
    _mirror_to_store(merged_list)

    # 数据更新后必须清除预测缓存
    clear_cache()
    return merged_list


def _mirror_to_store(merged_list):
    try:
        from .store_sync import mirror_to_store

        stats = mirror_to_store(merged_list)
        if stats.get('written'):
            log.info('开奖数据已镜像入库: 新增 %d 期', stats['written'])
    except Exception as e:
        log.warning(f'开奖数据镜像入库失败（不影响抓取）: {e}')


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
    # v8: 日常只抓最近1-2页；历史补数用 fetch_kl8_history_backfill
    data = fetch_kl8_data(pages=2, per_page=50)
    if data:
        # v6: 抓取后执行第二数据源交叉校验
        cross_result = cross_validate_with_second_source(data)
        if cross_result.get('conflict_count', 0) > 0:
            log.error(f'存在跨数据源冲突{cross_result["conflict_count"]}条，过滤冲突期号')
            # v9: 使用完整冲突期号列表（不再只取前10条的issues）
            conflict_issues = set(cross_result.get('conflict_issues', []))
            if not conflict_issues:
                # 回退：从 conflicts_preview 中提取
                conflict_issues = {item['issue'] for item in cross_result.get('conflicts_preview', [])}
            data = [row for row in data if row['issue'] not in conflict_issues]
            log.info(f'过滤{len(conflict_issues)}个冲突期号后，剩余{len(data)}期数据')

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
        'conflict_issues': [x['issue'] for x in conflicts],  # v9: 完整冲突期号列表
        'conflicts_preview': conflicts[:10],  # v9: 详情预览只取前10条
        'second_source_available': True,
        'primary_only_issues': len(primary_by_issue) - len(common_issues),
        'second_only_issues': len(second_by_issue) - len(common_issues),
    }


# ─── 历史数据补数（v8新增, v9.2改为分批）───

KL8_BACKFILL_MIN_PERIODS = 800   # v9.2: 最低目标改为800期（验证所需）
KL8_BACKFILL_RECOMMENDED_PERIODS = 800  # v9.2: 推荐目标也800期
KL8_BACKFILL_PAGES = 40  # 全量补数用(人工一次性抓40页)
KL8_BACKFILL_BATCH_PAGES = 5  # v9.2: 分批补数每批5页
KL8_BACKFILL_OVERLAP_PAGES = 1  # v9.2: 每批与上一批重叠1页，避免漏期
KL8_BACKFILL_STATE_FILE = data_path('kl8_backfill_state.json')


def check_need_backfill() -> Dict:
    """检查是否需要历史补数

    v9.2改动: 目标改为800期（验证所需）
    """
    path = Path(KL8_HISTORY_FILE)
    if not path.exists():
        return {
            'need_backfill': True,
            'current_periods': 0,
            'min_target': KL8_BACKFILL_MIN_PERIODS,
            'recommended_target': KL8_BACKFILL_RECOMMENDED_PERIODS,
        }

    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            source_list = raw.get('results', raw.get('data', []))
        else:
            source_list = raw

        valid_count = 0
        for r in source_list:
            normed = normalize_record(r)
            if normed:
                valid_count += 1

        need = valid_count < KL8_BACKFILL_MIN_PERIODS

        return {
            'need_backfill': need,
            'current_periods': valid_count,
            'min_target': KL8_BACKFILL_MIN_PERIODS,
            'recommended_target': KL8_BACKFILL_RECOMMENDED_PERIODS,
        }
    except Exception as e:
        log.warning(f'检查补数需求失败: {e}')
        return {
            'need_backfill': True,
            'current_periods': 0,
            'min_target': KL8_BACKFILL_MIN_PERIODS,
            'recommended_target': KL8_BACKFILL_RECOMMENDED_PERIODS,
        }


def count_valid_history_periods() -> int:
    """统计本地历史数据的有效期数"""
    path = Path(KL8_HISTORY_FILE)
    if not path.exists():
        return 0

    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            source_list = raw.get('results', raw.get('data', []))
        else:
            source_list = raw

        count = 0
        for r in source_list:
            if normalize_record(r):
                count += 1
        return count
    except Exception:
        return 0


def _load_json_or_default(path: Path, default) -> any:
    """安全加载JSON文件，失败返回default"""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _save_json_atomic(path: Path, data: any):
    """原子写入JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.json.tmp')
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        temp_path.replace(path)
    except Exception as e:
        log.warning(f'原子写入JSON失败: {e}')
        if temp_path.exists():
            temp_path.unlink()


def fetch_kl8_history_backfill_batch(
    target_periods: int = KL8_BACKFILL_RECOMMENDED_PERIODS,
    pages_per_batch: int = KL8_BACKFILL_BATCH_PAGES,
) -> Dict:
    """分批补数 — 每次只抓5页，持久化游标，下次从游标继续

    v9.2新增:
    - 不再一次抓40页，改为每批5页，每10分钟跑一次
    - 持久化游标(next_page)，下次从游标位置继续
    - 每批与上一批重叠1页，避免新开奖插入后分页偏移导致漏期
    - 达到目标期数后删除游标文件，标记完成

    返回:
        {
            'success': bool,
            'completed': bool,  # 是否已达到目标
            'start_page': int,
            'fetched_count': int,
            'periods_before': int,
            'periods_after': int,
            'error': str (if failed),
        }
    """
    import math

    state_path = Path(KL8_BACKFILL_STATE_FILE)
    current_count = count_valid_history_periods()

    if current_count >= target_periods:
        # 已达到目标，删除游标文件
        if state_path.exists():
            state_path.unlink()
        return {
            'success': True,
            'completed': True,
            'periods_after': current_count,
            'message': '历史数据已达到验证目标',
        }

    # 加载游标状态
    state = _load_json_or_default(state_path, {})

    # 首次执行：根据当前已有数据估算从第几页继续
    if not state:
        estimated_pages = math.ceil(current_count / 50) if current_count > 0 else 1
        # 从估算位置开始（但至少第1页）
        next_page = max(1, estimated_pages)
        # 首次执行时记录started_at
        state = {
            'next_page': next_page,
            'target_periods': target_periods,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    else:
        next_page = max(1, int(state.get('next_page', 1)))

    # 与上一批重叠1页，避免最新开奖插入后分页偏移造成漏期
    start_page = max(1, next_page - KL8_BACKFILL_OVERLAP_PAGES)

    # 分批抓取（每页间隔1秒）
    data = fetch_kl8_data(
        pages=pages_per_batch,
        per_page=50,
        start_page=start_page,
        page_delay=1.0,
    )

    if not data:
        return {
            'success': False,
            'completed': False,
            'error': '本批历史抓取失败',
            'periods_before': current_count,
        }

    # 保存并合并
    merged = save_kl8_data(data)
    periods_after = len(merged or []) if merged else current_count

    completed = periods_after >= target_periods

    if completed:
        # 已完成，删除游标文件
        if state_path.exists():
            state_path.unlink()
    else:
        # 更新游标：下次从本批末页之后继续
        state['next_page'] = start_page + pages_per_batch
        state['last_batch_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        _save_json_atomic(state_path, state)

    return {
        'success': True,
        'completed': completed,
        'start_page': start_page,
        'fetched_count': len(data),
        'periods_before': current_count,
        'periods_after': periods_after,
    }


def fetch_kl8_history_backfill(target_periods: int = KL8_BACKFILL_RECOMMENDED_PERIODS) -> Dict:
    """一次性历史补数 — 仅供人工全量补数，定时任务不再使用

    v9.2改动:
    - 定时任务改用 fetch_kl8_history_backfill_batch() 分批补数
    - 此方法保留给人工需要一次性全量补数时使用
    - 不再由定时任务直接调用

    参数:
        target_periods: 目标期数，默认800

    返回:
        {
            'success': bool,
            'periods_before': int,
            'periods_after': int,
            'fetched_count': int,
            'error': str (if failed),
        }
    """
    # 先检查当前有多少期
    path = Path(KL8_HISTORY_FILE)
    periods_before = 0
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(raw, dict):
                source_list = raw.get('results', raw.get('data', []))
            else:
                source_list = raw
            for r in source_list:
                if normalize_record(r):
                    periods_before += 1
        except Exception:
            pass

    if periods_before >= target_periods:
        log.info(f'快乐8历史数据已足够({periods_before}期 >= {target_periods}期)，无需补数')
        return {
            'success': True,
            'periods_before': periods_before,
            'periods_after': periods_before,
            'fetched_count': 0,
            'message': f'历史数据已满足目标({periods_before}期)',
        }

    # 计算需要抓取的页数
    pages_needed = max(KL8_BACKFILL_PAGES, (target_periods - periods_before) // 50 + 5)
    log.info(f'快乐8开始历史补数: 当前{periods_before}期，目标{target_periods}期，将抓取{pages_needed}页')

    # 抓取大量数据
    data = fetch_kl8_data(pages=pages_needed, per_page=50)
    if not data:
        return {
            'success': False,
            'periods_before': periods_before,
            'periods_after': periods_before,
            'fetched_count': 0,
            'error': '补数抓取失败',
        }

    # 保存并合并
    merged = save_kl8_data(data)
    periods_after = len(merged) if merged else periods_before

    log.info(f'快乐8补数完成: {periods_before}期 -> {periods_after}期 (新增{len(data)}期抓取)')

    return {
        'success': True,
        'periods_before': periods_before,
        'periods_after': periods_after,
        'fetched_count': len(data),
    }
