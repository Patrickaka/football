import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.auth import AuthSettings, build_session_manager, install_auth
from src.api.deps import Settings, build_cache, build_database, get_executor, shutdown_executor
from src.api.rate_limit import ClientRateLimiters, install_rate_limit
from src.api.routers import auth as auth_routes
from src.api.routers import basketball, beidan, football, health, kl8, lottery
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
    install_validation_error_shape(app)
    app.include_router(health.router)
    app.include_router(auth_routes.router)
    app.include_router(basketball.router)
    app.include_router(beidan.router)
    app.include_router(lottery.router)
    app.include_router(kl8.router)
    app.include_router(football.router)

    # 中间件按**注册的逆序**执行：后注册的先跑。限流要排在鉴权前面，
    # 否则未登录的洪水请求会先去查一遍会话（打 Redis）再被 401 挡下——
    # 那正好是最不该在被攻击时做的事。
    install_auth(app)
    install_rate_limit(app, build_rate_limiters(settings))
    return app


def install_validation_error_shape(app):
    """参数校验失败时，在 FastAPI 的 `detail` 之外补一个 `error` 字段。

    状态码保持 422——那是正确的语义，旧入口把参数错误伪装成 `200` 里的
    一句 error 字符串，不该照搬。但**网页整套是按 `data.error` 判错的**
    （`web/index.html` 里 `if (data.error)`），只给 `detail` 的话前端会
    以为请求成功、然后拿不到数据，页面空白且没有任何提示。

    两边都给：机器读 `detail`，网页读 `error`。
    """

    @app.exception_handler(RequestValidationError)
    async def _shape(request: Request, exc: RequestValidationError):
        first = (exc.errors() or [{}])[0]
        field = '.'.join(str(part) for part in first.get('loc', ())[1:]) or '参数'
        return JSONResponse(
            {'error': f"参数 {field} 无效: {first.get('msg', '校验失败')}",
             'detail': exc.errors()},
            status_code=422,
        )

    return app


def build_rate_limiters(settings):
    """限流器；未配置速率则返回 None（不限流）。"""
    if settings.rate_limit_per_sec <= 0:
        log.info('入站限流未启用（RATE_LIMIT_PER_SEC 未配置）')
        return None
    log.info('入站限流已启用：%.1f 次/秒，突发 %d',
             settings.rate_limit_per_sec, settings.rate_limit_burst)
    return ClientRateLimiters(
        rate_per_sec=settings.rate_limit_per_sec,
        burst=settings.rate_limit_burst,
        maxsize=settings.rate_limit_clients,
    )
