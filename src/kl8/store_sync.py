"""把开奖数据镜像进 foundation/store。

迁移之后开奖数据有了两个写入方：抓取路径写 `data/kl8_history.json`，
而 `numeric_draw` 表只被迁移脚本写过一次。不接上的话库里的数据会越来越旧，
**而且没有任何报错**——这类分叉正是重构期最容易留下的暗伤。

现在先双写：文件照旧（分析器还在读它），同时镜像进库。读取路径的切换留到
下一批——一次只动一头，出问题时才分得清是哪一头。`reconcile()` 用来在
双写期间主动比对两头。

本模块是 kl8（旧代码）对领域层的适配。方向是**旧依赖新**，反过来不行。
"""
import logging

from src.domain.numeric.draw import Draw
from src.domain.numeric.draw_store import DrawStore

log = logging.getLogger('kl8.store_sync')

GAME = 'kl8'


def _open_store():
    """按需建库连接。抓取路径不常跑，没必要为它常驻一个连接池。"""
    from src.foundation.store import Database, make_engine, mysql_url_from_env

    db = Database(make_engine(mysql_url_from_env()))
    return DrawStore(db, game=GAME)


def _parse(records):
    draws, rejected = [], []
    for record in records or []:
        draw = Draw.parse(record)
        (draws.append(draw) if draw is not None else rejected.append(record))
    return draws, rejected


def mirror_to_store(records):
    """把开奖记录镜像进库，返回统计。

    **任何失败都只记录不外抛。** 抓取是主链路，落库是旁路；数据库抖一下就让
    开奖数据抓不下来，等于用一个次要设施的可用性绑架了主要业务。
    """
    draws, rejected = _parse(records)
    stats = {'received': len(records or []), 'rejected': len(rejected),
             'written': 0, 'conflicts': 0}
    if rejected:
        log.warning('镜像时跳过 %d 条不合规记录', len(rejected))

    try:
        store = _open_store()
        before = store.count()
        conflicts = store.save(draws)
        stats['written'] = store.count() - before
        stats['conflicts'] = len(conflicts)
        for conflict in conflicts:
            log.error('期号 %s 号码冲突：库内 %s 抓到 %s，保留库内值',
                      conflict.issue, conflict.kept.numbers,
                      conflict.rejected.numbers)
    except Exception as exc:
        log.warning('开奖数据镜像入库失败（不影响抓取）: %s', exc)
        stats['error'] = str(exc)
    return stats


def load_from_store():
    """从库里读回开奖记录，转成旧代码用的字典形状。

    库不可用时返回空列表而不是抛错——分析器是多源合并，少一个来源应当
    自动退回其余来源，而不是让整个预测链路停摆。这条降级路径在迁移期尤其
    重要：新来源刚接上，出问题的概率比老来源高。
    """
    try:
        draws = _open_store().load()
    except Exception as exc:
        log.warning('从库读取开奖数据失败，本次退回其它来源: %s', exc)
        return []
    log.info('快乐8: 库中加载了 %d 期有效数据', len(draws))
    return [draw.to_dict() for draw in draws]


def reconcile(records):
    """比对文件与库，返回不一致项的描述；空列表表示一致。

    双写期间两头会不会分叉，只能靠主动比对回答。三种分叉都要报：库里缺、
    库里多、以及**条数对得上但号码不同**——最后一种只比条数看不出来，
    而它恰恰是最危险的。
    """
    draws, _ = _parse(records)
    expected = {d.issue: d for d in draws}

    try:
        stored = {d.issue: d for d in _open_store().load()}
    except Exception as exc:
        return [f'无法读取库中开奖数据: {exc}']

    problems = []
    for issue, draw in sorted(expected.items()):
        found = stored.get(issue)
        if found is None:
            problems.append(f'库中缺少期号 {issue}')
        elif found.numbers != draw.numbers:
            problems.append(
                f'期号 {issue} 号码不一致：文件 {draw.numbers} 库 {found.numbers}')
    for issue in sorted(set(stored) - set(expected)):
        problems.append(f'库中多出期号 {issue}（文件中没有）')
    return problems
