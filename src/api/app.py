import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.deps import Settings, build_cache, build_database, shutdown_executor
from src.api.routers import health
from src.foundation.tasks import TaskScheduler

log = logging.getLogger('api.app')


def create_app(settings=None):
    settings = settings or Settings.from_env()

    # 组件在此处（应用构造期）而非 lifespan 内装配：Cache/Database/TaskScheduler
    # 的构造本身不做真正的网络握手（引擎连接、Redis ping 已在 build_cache /
    # build_database 内部各自兜底降级），是同步且廉价的。放在 lifespan 里则
    # TestClient 在不使用 `with` 语句时不会触发 lifespan 事件，app.state 上
    # 就取不到这些组件——ASGI 的 lifespan 协议本就是可选的，生产环境
    # （uvicorn）会在真正 accept 连接前触发它，但测试与部分嵌入式场景不会。
    # 因此装配放构造期，lifespan 只负责启动日志与优雅停机时的资源释放。
    app_state = {
        'settings': settings,
        'cache': build_cache(settings),
        'db': build_database(settings),
        # 本阶段不提交任何实际任务（业务预热任务属于后续阶段），仅完成装配：
        # 调度器创建后挂到 app.state，供健康检查观测；不调用 start()——
        # 没有待跑任务时启动线程池纯属空转，且会提前关闭 submit() 窗口
        # （TaskScheduler.submit 在 start() 之后一律 RuntimeError）。
        # 后续阶段在此处补充 submit(...) 调用后再决定何时 start()。
        'tasks': TaskScheduler(max_workers=settings.max_task_workers),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for key, value in app_state.items():
            setattr(app.state, key, value)
        log.info('API 启动完成')
        yield
        app.state.tasks.shutdown(wait=True)
        app.state.db.dispose()
        shutdown_executor()
        log.info('API 已停止')

    app = FastAPI(title='Football 预测服务', version='2.0.0', lifespan=lifespan)
    for key, value in app_state.items():
        setattr(app.state, key, value)
    app.include_router(health.router)
    return app
