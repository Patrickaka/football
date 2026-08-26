from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=['health'])


class HealthResponse(BaseModel):
    status: str
    components: dict


@router.get('/healthz', response_model=HealthResponse)
async def healthz(request: Request):
    components = {
        'cache': _probe_cache(request.app.state.cache),
        'database': _probe_database(request.app.state.db),
        'tasks': _probe_tasks(request.app.state.tasks),
    }
    healthy = all(v == 'ok' for v in components.values())
    return HealthResponse(status='ok' if healthy else 'degraded', components=components)


def _probe_cache(cache):
    try:
        cache.set('__health__', 1, ttl=10)
        return 'ok' if cache.get('__health__', lambda: 0) == 1 else 'degraded'
    except Exception:
        return 'error'


def _probe_database(db):
    from sqlalchemy import text

    try:
        with db.connect() as conn:
            conn.execute(text('select 1'))
        return 'ok'
    except Exception:
        return 'error'


def _probe_tasks(scheduler):
    """探测调度器是否**真的在跑**，而不只是对象存在。

    首版只判断 `scheduler is not None`，那个判断永远为真——它不是信号，
    是会让人误以为一切正常的噪声。第一个真实后台任务（篮球赔率采样）
    上线时一并升级。

    「没有任务要跑」与「有任务却没在跑」要分开：前者是正常的空闲，后者才是
    故障。不分开的话，一个还没接后台任务的服务会天天报 degraded，
    久而久之这个信号就没人看了。
    """
    try:
        if scheduler is None:
            return 'error'
        if scheduler.task_count() == 0:
            return 'ok'
        return 'ok' if scheduler.is_running() else 'degraded'
    except Exception:
        return 'error'
