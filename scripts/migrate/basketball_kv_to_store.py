"""把 basketball 的数据从 common/kv_store 迁到 foundation/store。

kv_store 是「MySQL + JSON 文件降级」，foundation/store 声明「MySQL 唯一真相源、
不做 JSON 兜底」——两套哲学对立，迁移后 basketball 只读写后者，kv_store 里的
对应数据归档保留、不再读写。

kv_loader 做成可注入参数而非直接 import kv_store：否则测试必须连真实 MySQL，
违反「测试禁止依赖外部服务」的约束。

本脚本迁 elo（ratings / history / recent_form）与 odds_history。prediction_records 的表结构依赖尚不确定的
查询模式，留到迁移 records.py 时一并处理；calibration_db / match_results /
prediction_history 三个 key 在线上不存在。

用法：
    python3 -m scripts.migrate.basketball_kv_to_store --dry-run
    python3 -m scripts.migrate.basketball_kv_to_store
    python3 -m scripts.migrate.basketball_kv_to_store --verify-only
"""
import logging

from src.domain.sports.basketball.repository import (
    EloHistoryRepository, EloRatingRepository, EloRecentFormRepository,
    OddsSnapshotRepository, create_all,
)

log = logging.getLogger('migrate.basketball')

ELO_KEY = 'basketball_elo_ratings'
ODDS_KEY = 'basketball_odds_history'

_SNAPSHOT_FIELDS = (
    'spf_home', 'spf_away', 'rqspf_home', 'rqspf_away',
    'dx_over', 'dx_under', 'handicap', 'total_line',
)


def _elo_rating_rows(kv_loader):
    """{'ratings': {队名: 评分}, 'updated_at': ...} → 每队一行。

    ratings 里只有评分数值，没有各队自己的时间戳，故统一取顶层 updated_at。
    """
    blob = kv_loader(ELO_KEY, None) or {}
    ratings = blob.get('ratings') or {}
    updated_at = blob.get('updated_at') or ''
    return [
        {'team': team, 'rating': float(rating), 'updated_at': updated_at}
        for team, rating in ratings.items()
    ]


def _elo_history_rows(kv_loader):
    """{'history': {队名: [{rating, date, event}]}} → 摊平成每条一行。"""
    blob = kv_loader(ELO_KEY, None) or {}
    history = blob.get('history') or {}
    rows = []
    for team, entries in history.items():
        for entry in entries or []:
            rows.append({
                'team': team,
                'recorded_at': entry.get('date') or '',
                'rating': float(entry.get('rating') or 0),
                'event': entry.get('event') or '',
            })
    return rows


def _elo_recent_form_rows(kv_loader):
    """{'recent_form': {队名: [胜负数值]}} → 每场一行，用位置索引保序。

    线上当前全为空列表（所有球队只有 initialized 事件、尚无真实比赛），
    但这是活字段——update_ratings 每场都会追加、_form_factor 读近 5 场
    算胜率并影响预测概率，所以必须迁，不能因为当前为空就跳过。
    """
    blob = kv_loader(ELO_KEY, None) or {}
    recent_form = blob.get('recent_form') or {}
    rows = []
    for team, results in recent_form.items():
        for seq, result in enumerate(results or []):
            rows.append({'team': team, 'seq': seq, 'result': float(result)})
    return rows


def _odds_snapshot_rows(kv_loader):
    """{match_key: [快照]} → 摊平成每条快照一行。

    快照的 ts 字段作为 captured_at；三类盘口字段并非每次都齐全，缺的留空。
    """
    blob = kv_loader(ODDS_KEY, None) or {}
    rows = []
    for match_key, snapshots in blob.items():
        for snap in snapshots or []:
            row = {'match_key': match_key, 'captured_at': snap.get('ts') or ''}
            for field in _SNAPSHOT_FIELDS:
                if field in snap:
                    row[field] = snap[field]
            rows.append(row)
    return rows


_PLAN = (
    ('bb_elo_rating', EloRatingRepository, _elo_rating_rows, ['team']),
    ('bb_elo_history', EloHistoryRepository, _elo_history_rows, ['team', 'recorded_at']),
    ('bb_elo_recent_form', EloRecentFormRepository, _elo_recent_form_rows,
     ['team', 'seq']),
    ('bb_odds_snapshot', OddsSnapshotRepository, _odds_snapshot_rows,
     ['match_key', 'captured_at']),
)


def migrate(kv_loader, db, dry_run=False):
    """迁移并返回 {表名: {'migrated': n, 'skipped': n}}。

    用 upsert 而非 insert_many，保证可重复执行——迁移脚本中途失败后重跑
    是常态，不该因为主键冲突而卡住。
    """
    stats = {}
    for table_name, repo_cls, extract, key_cols in _PLAN:
        rows = extract(kv_loader)
        stats[table_name] = {'migrated': len(rows), 'skipped': 0}
        if dry_run:
            continue
        repo = repo_cls(db)
        for row in rows:
            repo.upsert(row, key_cols=key_cols)
    return stats


def verify(kv_loader, db):
    """比对源与库，返回不一致项的描述；空列表表示一致。

    先比行数，再逐行比对内容——只比行数会漏掉「行数对得上但内容被改过」。
    """
    problems = []
    for table_name, repo_cls, extract, key_cols in _PLAN:
        expected = extract(kv_loader)
        repo = repo_cls(db)
        actual = repo.find_all()
        if len(expected) != len(actual):
            problems.append(
                f'{table_name}: 行数不一致，源 {len(expected)} 库 {len(actual)}')
            continue
        actual_by_key = {
            tuple(row[col] for col in key_cols): row for row in actual
        }
        for row in expected:
            key = tuple(row[col] for col in key_cols)
            found = actual_by_key.get(key)
            if found is None:
                problems.append(f'{table_name}: 库中缺少 {key}')
                continue
            for col, value in row.items():
                if _differs(found.get(col), value):
                    problems.append(
                        f'{table_name}: {key} 的 {col} 不一致，'
                        f'源 {value!r} 库 {found.get(col)!r}')
    return problems


def _differs(left, right):
    """浮点用容差比较，其余直接比。"""
    if isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            return left is not right
        return abs(float(left) - float(right)) > 1e-9
    return left != right


def main():
    import argparse

    parser = argparse.ArgumentParser(description='basketball 数据迁移')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写入')
    parser.add_argument('--verify-only', action='store_true', help='只校验不迁移')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    from src.common import kv_store
    from src.foundation.store import Database, make_engine, mysql_url_from_env

    db = Database(make_engine(mysql_url_from_env()))
    create_all(db)

    if args.verify_only:
        problems = verify(kv_store.load, db)
        if problems:
            for item in problems:
                log.error(item)
            raise SystemExit(1)
        log.info('校验通过，源与库一致')
        return

    stats = migrate(kv_store.load, db, dry_run=args.dry_run)
    for table_name, item in stats.items():
        log.info('%s: %d 行%s', table_name, item['migrated'],
                 '（dry-run，未写入）' if args.dry_run else '')

    if not args.dry_run:
        problems = verify(kv_store.load, db)
        if problems:
            for item in problems:
                log.error(item)
            raise SystemExit(1)
        log.info('迁移完成且校验通过')


if __name__ == '__main__':
    main()
