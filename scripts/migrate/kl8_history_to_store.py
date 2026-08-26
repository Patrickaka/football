"""把 kl8 的开奖历史从 JSON 文件迁进 foundation/store。

线上 `data/kl8_history.json` 是 878KB、2048 期，每次追加新一期都要整文件
重写。入库后按期号增量写入。

用法：
    python3 -m scripts.migrate.kl8_history_to_store --dry-run   # 先看统计
    python3 -m scripts.migrate.kl8_history_to_store             # 执行并自动校验
    python3 -m scripts.migrate.kl8_history_to_store --verify-only

**原 JSON 文件不删。** 代码可以推倒，数据不可丢——确认线上稳定运行一段
时间后再由人工清理。
"""
import argparse
import json
import logging
from pathlib import Path

from src.domain.numeric.draw import Draw
from src.domain.numeric.draw_store import DrawStore
from src.domain.numeric.repository import create_all

log = logging.getLogger('migrate.kl8_history')

GAME = 'kl8'


def read_history(path):
    """读出原始记录列表。

    文件顶层历史上出现过三种形状：{'results': [...]}、{'data': [...]}、
    以及裸列表。三种都认——迁移脚本挑食会让人以为数据丢了。
    """
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        return raw.get('results') or raw.get('data') or []
    return raw if isinstance(raw, list) else []


def parse_all(records):
    """返回 (可用记录, 被拒绝的原始记录)。

    被拒绝的要**原样带出来**而不是丢掉：迁移时出现不合规记录，人得看到
    是哪几条、坏在哪，否则「2048 期只迁进来 2040 期」无从解释。
    """
    draws, rejected = [], []
    for record in records:
        draw = Draw.parse(record)
        (draws.append(draw) if draw is not None else rejected.append(record))
    return draws, rejected


def migrate(path, db, dry_run=False):
    records = read_history(path)
    draws, rejected = parse_all(records)
    stats = {'read': len(records), 'parsed': len(draws),
             'rejected': len(rejected), 'conflicts': 0, 'written': 0}

    for record in rejected:
        log.error('记录不合规，未迁移: %s', json.dumps(record, ensure_ascii=False)[:200])

    if dry_run:
        return stats

    store = DrawStore(db, game=GAME)
    before = store.count()
    conflicts = store.save(draws)
    stats['conflicts'] = len(conflicts)
    stats['written'] = store.count() - before
    for conflict in conflicts:
        log.error('期号 %s 号码冲突：库内 %s 文件内 %s，保留库内值',
                  conflict.issue, conflict.kept.numbers, conflict.rejected.numbers)
    return stats


def verify(path, db):
    """逐期比对文件与库，返回不一致项的描述；空列表表示一致。

    只比条数会漏掉「条数对得上但号码被改过」——号码正是这个领域里唯一
    不能失真的东西。
    """
    draws, _ = parse_all(read_history(path))
    stored = {d.issue: d for d in DrawStore(db, game=GAME).load()}

    problems = []
    if len(draws) != len(stored):
        problems.append(f'期数不一致：文件 {len(draws)} 库 {len(stored)}')
    for draw in draws:
        found = stored.get(draw.issue)
        if found is None:
            problems.append(f'库中缺少期号 {draw.issue}')
        elif found.numbers != draw.numbers:
            problems.append(
                f'期号 {draw.issue} 号码不一致：文件 {draw.numbers} 库 {found.numbers}')
    return problems


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='kl8 开奖历史迁移')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写入')
    parser.add_argument('--verify-only', action='store_true', help='只校验')
    parser.add_argument('--file', default=None, help='历史文件路径')
    args = parser.parse_args()

    from src.common.paths import data_path
    from src.foundation.store import Database, make_engine, mysql_url_from_env

    path = args.file or data_path('kl8_history.json')
    db = Database(make_engine(mysql_url_from_env()))
    create_all(db)

    if args.verify_only:
        _report(verify(path, db))
        return

    stats = migrate(path, db, dry_run=args.dry_run)
    log.info('读取 %d 条，可用 %d 条，不合规 %d 条，写入 %d 条，冲突 %d 处%s',
             stats['read'], stats['parsed'], stats['rejected'], stats['written'],
             stats['conflicts'], '（dry-run，未写入）' if args.dry_run else '')
    if not args.dry_run:
        _report(verify(path, db))


def _report(problems):
    if problems:
        for item in problems:
            log.error(item)
        raise SystemExit(1)
    log.info('校验通过，文件与库一致')


if __name__ == '__main__':
    main()
