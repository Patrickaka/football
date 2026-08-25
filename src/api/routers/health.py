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
    try:
        return 'ok' if scheduler is not None else 'error'
    except Exception:
        return 'error'
