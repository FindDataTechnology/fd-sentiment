"""共享 bearer 门的判据（change converge-line-identity-and-keys 8.6）。

两层：
1. **门本身**用真 ASGI scope 直测——不装 fastmcp 也能跑，故 CI 永远验得到。
2. **接线**（``serve_http`` 走哪条分支）用假 mcp + 假 uvicorn 观测调用，证明
   「没配 token ⇒ 老路径 mcp.run」「配了 ⇒ http_app 带中间件」，而不是证明
   "中间件内部空转"。
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from fd_sentiment.auth import ENV_TOKEN, EXEMPT_PATHS, BearerAuthMiddleware, configured_token


def _run(mw, headers, *, path="/mcp", scope_type="http"):
    """喂一个 scope 进去，返回 (status, body, reached_downstream)。"""
    reached = []
    sent = []

    async def downstream(scope, receive, send):
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": scope_type,
        "path": path,
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
    }
    asyncio.run(mw(downstream)(scope, receive, send))

    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, bool(reached)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_TOKEN, raising=False)
    yield


def test_no_token_means_no_gate():
    status, _, reached = _run(BearerAuthMiddleware, {})
    assert (status, reached) == (200, True)
    assert configured_token() == ""


def test_matching_bearer_passes(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, body, reached = _run(BearerAuthMiddleware, {"authorization": "Bearer s3cret"})
    assert (status, reached, body) == (200, True, b"ok")


def test_missing_header_rejected(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, body, reached = _run(BearerAuthMiddleware, {})
    assert status == 401
    assert reached is False, "门必须在进 MCP 处理之前拦住"
    assert b"unauthorized" in body


def test_wrong_token_rejected(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, _, reached = _run(BearerAuthMiddleware, {"authorization": "Bearer wrong"})
    assert (status, reached) == (401, False)


def test_wrong_scheme_rejected(monkeypatch):
    """裸 token（无 Bearer 前缀）不算——判据是整串恒时比较，不做宽容解析。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for value in ("s3cret", "Basic s3cret", "Bearer  s3cret"):
        status, _, reached = _run(BearerAuthMiddleware, {"authorization": value})
        assert (status, reached) == (401, False), value


def test_lifespan_passes_through(monkeypatch):
    """lifespan 不被拦，否则应用根本起不来。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    status, _, reached = _run(BearerAuthMiddleware, {}, scope_type="lifespan")
    assert (status, reached) == (200, True)


def test_token_reread_per_request(monkeypatch):
    """改旗/轮换不需要重启：同一中间件类，改 env 后立刻生效。"""
    monkeypatch.setenv(ENV_TOKEN, "one")
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer one"})[0] == 200
    monkeypatch.setenv(ENV_TOKEN, "two")
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer one"})[0] == 401
    assert _run(BearerAuthMiddleware, {"authorization": "Bearer two"})[0] == 200


# --- 接线：走哪条分支 ------------------------------------------------

def _fakes(monkeypatch):
    from fd_sentiment import server as server_mod

    calls = {}

    class _Middleware:
        def __init__(self, cls, *args, **kwargs):
            self.cls = cls

    fake_mw = types.ModuleType("starlette.middleware")
    fake_mw.Middleware = _Middleware
    fake_starlette = types.ModuleType("starlette")
    fake_starlette.middleware = fake_mw
    monkeypatch.setitem(sys.modules, "starlette", fake_starlette)
    monkeypatch.setitem(sys.modules, "starlette.middleware", fake_mw)

    class FakeMcp:
        def run(self, **kwargs):
            calls["run"] = kwargs

        def http_app(self, **kwargs):
            calls["http_app"] = kwargs
            return "ASGI-APP"

    calls["_fake_app"] = FakeMcp()
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: calls.setdefault("uvicorn", {"app": app, **kwargs})
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    return server_mod, calls


def test_serve_http_without_token_keeps_old_path(monkeypatch):
    server_mod, calls = _fakes(monkeypatch)
    server_mod.serve_http(calls["_fake_app"], 9410)
    assert "run" in calls and calls["run"]["transport"] == "http"
    assert calls["run"]["port"] == 9410
    assert "http_app" not in calls and "uvicorn" not in calls


def test_serve_http_with_token_wraps_middleware(monkeypatch):
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    server_mod, calls = _fakes(monkeypatch)
    server_mod.serve_http(calls["_fake_app"], 9411)
    assert "http_app" in calls, "配了 token 就必须走带中间件的 ASGI 面"
    mw = calls["http_app"]["middleware"]
    assert len(mw) == 1 and mw[0].cls is BearerAuthMiddleware
    assert calls["uvicorn"]["app"] == "ASGI-APP"
    assert calls["uvicorn"]["port"] == 9411
    assert "run" not in calls

def test_health_path_exempt(monkeypatch):
    """探针永远不带钥匙：豁免路径必须放行（本族有把 /healthz 与 /mcp 挂同一 app 的线）。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for path in EXEMPT_PATHS:
        status, _, reached = _run(BearerAuthMiddleware, {}, path=path)
        assert (status, reached) == (200, True), path


def test_non_health_paths_not_exempt(monkeypatch):
    """豁免只认精确路径：带前缀/后缀的不是健康面。"""
    monkeypatch.setenv(ENV_TOKEN, "s3cret")
    for path in ("/healthz/x", "/api/v1/healthz", "/mcp", "/"):
        status, _, reached = _run(BearerAuthMiddleware, {}, path=path)
        assert (status, reached) == (401, False), path
