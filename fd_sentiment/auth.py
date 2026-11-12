"""HTTP 层共享 bearer 门——本线自有的常命钥匙（change converge-line-identity-and-keys 8.6）。

门只在 ``MCP_BEARER_TOKEN`` 非空时**存在**：未配置时 ``main()`` 仍然走
``mcp.run(transport="http")``，一个中间件都不挂（与接入前逐字节一致）；配置后
每个 http 请求必须带 ``Authorization: Bearer <token>``，恒时比较，否则在进入
MCP 处理之前就 401。stdio 不经过这里。

形状与 ``fd-open-data-mcp`` / ``fd-cn-report`` / ``fd-info`` 的门逐行同族，运维只需
记一个变量名（``MCP_BEARER_TOKEN``）。它给的是「摘掉壹座后本线仍能被合法消费」这条
能力，而**不是**客户级身份：一把钥匙全客户共用，谁在用由外层（网关/registry）记录。
每客户一把钥匙需要钥匙库 + 吊销面，属 8.x 的后续工作，不在这里假装做到。
"""

from __future__ import annotations

import hmac
import os

ENV_TOKEN = "MCP_BEARER_TOKEN"

# 健康面豁免：探针永远不带钥匙，且只回状态，不是数据面。本线今天把 /health
# 挂在另一个监听上，所以这条豁免在这里不改行为；写进来是为了**同族中立**——
# 抄这份文件的人如果沿用同一 app 挂 /healthz，不会把探针打死。
EXEMPT_PATHS = ("/health", "/healthz")


def configured_token() -> str:
    """每请求重读：改旗/轮换不需要重启进程。"""
    return os.environ.get(ENV_TOKEN, "").strip()


class BearerAuthMiddleware:
    """ASGI 中间件：配了 token 就要求匹配的 bearer，未配则不设门。

    只拦 ``http`` scope——lifespan 必须放过去，否则应用起不来；健康面按
    ``EXEMPT_PATHS`` 豁免。
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    def __call__(self, scope, receive, send):
        token = configured_token()
        if scope["type"] != "http" or not token:
            return self.asgi_app(scope, receive, send)
        if (scope.get("path") or "") in EXEMPT_PATHS:
            return self.asgi_app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization", b"")
        expected = ("Bearer " + token).encode("utf-8", "surrogateescape")
        if hmac.compare_digest(provided, expected):
            return self.asgi_app(scope, receive, send)

        async def _reject(receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b'{"error": "unauthorized"}'})

        return _reject(receive, send)
