import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.auth import AuthSettings, build_session_manager, install_auth
from src.api.deps import Settings, build_cache, build_database, get_executor, shutdown_executor
from src.api.routers import auth as auth_routes
from src.api.routers import health
from src.foundation.tasks import TaskScheduler

log = logging.getLogger('api.app')


LOGIN_PAGE = pathlib.Path(__file__).resolve().parents[2] / 'web' / 'login.html'


def create_app(settings=None, auth_settings=None):
    settings = settings or Settings.from_env()
    auth_settings = auth_settings or AuthSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.cache = build_cache(settings)
        # 会话存 L2：生产态是 Redis，跨进程重启保留，撤销也是真的撤销。
        # 降级为内存时重启即全员登出——是预期行为，不是故障。
        app.state.auth = auth_settings
        app.state.sessions = build_session_manager(app.state.cache.l2, auth_settings)
        if not auth_settings.enabled:
            log.warning('鉴权未启用（未配置 FOOTBALL_USERS）——所有接口对外开放')
        else:
            log.info('鉴权已启用，用户: %s', ', '.join(sorted(auth_settings.credentials)))
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
        # 关闭顺序：先排空 SWR 后台刷新，再停消费者，最后释放消费者依赖的资源。
        # 1. cache.wait_for_refreshes：SWR 刷新线程是 daemon，进程退出即被杀，
        #    `finally: self.l2.unlock(key)` 得不到执行，会在 Redis 残留一把
        #    TTL 最长 lock_timeout 秒的锁——重启后第一个请求撞上这把锁，
        #    复现 P1 惊群。必须在关闭序列的最前面排空。
        # 2. tasks.shutdown → shutdown_executor：先停消费者（任务调度器与
        #    线程池），再释放它们依赖的资源（db），顺序不能反——旧写法先
        #    dispose db 再关 executor，executor 里仍在跑的任务可能这期间
        #    还在用 db。
        app.state.cache.wait_for_refreshes(timeout=app.state.cache.lock_timeout)
        app.state.tasks.shutdown(wait=True)
        shutdown_executor()
        app.state.db.dispose()
        log.info('API 已停止')

    app = FastAPI(title='Football 预测服务', version='2.0.0', lifespan=lifespan)
    # 登录页在建 app 时读一次。读不到不该让服务起不来——鉴权本身照常工作，
    # 只是登录页显示一句提示（`/auth/login` 仍可直接调用）。
    try:
        app.state.login_page = LOGIN_PAGE.read_text(encoding='utf-8')
    except OSError as exc:
        log.error('登录页读取失败（%s）：%s', LOGIN_PAGE, exc)
        app.state.login_page = '<!doctype html><meta charset="utf-8">登录页缺失'
    app.state.auth = auth_settings
    app.include_router(health.router)
    app.include_router(auth_routes.router)
    install_auth(app)
    return app
