# -*- coding: utf-8 -*-
"""新入口的 JSON 序列化。

**和旧入口用同一套清洗**（判据 11）——`_sanitize_json` + `_json_default`
住在 `src/webapp/http_util.py`，两边共用一份。

不用 FastAPI 默认的 `jsonable_encoder` 是因为**它序列化不了这个项目的
返回值**：payload 里带着 numpy 标量与数组（模型算出来的概率）、
`SteamSignal` 这类带 `to_dict` 的对象，`jsonable_encoder` 走到
`vars(obj)` 就抛 `TypeError: vars() argument must have __dict__ attribute`,
整个接口 500。切换入口当天 `/api/predict/batch` 就是这么全挂的。

`allow_nan=False` 同样是照抄旧入口：默认的 `allow_nan=True` 会输出
`Infinity` / `NaN` 字面量，**浏览器的 `JSON.parse` 解析不了**
（`distance=inf` 这种业务哨兵值属于此类），所以序列化前先把非有限浮点
换成 `null`。
"""

import json

from fastapi.responses import JSONResponse

from src.webapp.http_util import _json_default, _sanitize_json


class SanitizedJSONResponse(JSONResponse):
    """按旧入口的口径序列化：先清洗非有限浮点，再用 numpy 兜底转换。"""

    def render(self, content) -> bytes:
        return json.dumps(
            _sanitize_json(content),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode('utf-8')
