import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.deps import Settings, build_cache, build_database, get_executor, shutdown_executor
from src.api.routers import health
from src.foundation.tasks import TaskScheduler

log = logging.getLogger('api.app')


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.cache = build_cache(settings)
        app.state.db = build_database(settings)
        # 本阶段不提交任何实际任务（业务预热任务属于后续阶段），仅完成装配：
        # 调度器创建后挂到 app.state，供健康检查观测；不调用 start()——
        # 没有待跑任务时启动线程池纯属空转，且会提前关闭 submit() 窗口
        # （TaskScheduler.submit 在 start() 之后一律 RuntimeError）。
        # 后续阶段在此处补充 submit(...) 调用后再决定何时 start()。
        app.state.tasks = TaskScheduler(max_workers=settings.max_task_workers)
        # get_executor 是模块级全局单例，只有首次调用的 workers 参数生效；
        # 必须在任何 run_blocking(...) 之前、在此显式用 settings.executor_workers
        # 完成首次初始化，否则该字段会因为“从未被首次调用消费”而形同虚设。
        get_executor(settings.executor_workers)
        log.info('API 启动完成')
        yield
        app.state.tasks.shutdown(wait=True)
        app.state.db.dispose()
        shutdown_executor()
        log.info('API 已停止')

    app = FastAPI(title='Football 预测服务', version='2.0.0', lifespan=lifespan)
    app.include_router(health.router)
    return app
