"""领域异常 → 封闭错误码（mcp-error-semantics 组 4）。

**为什么需要这一层**：`LLMError` / `ScoreError` / `ValueError` 都是普通异常，
fastmcp 捕获后包成一段文本、`structuredContent` 为空——调用方只能读字符串。
包一层后错误码走结构化通道，`isError` 也同时到达。

映射规则（每类的处置都不同，这正是要分码的原因）：

| 领域异常 | 码 | 调用方该做什么 |
|---|---|---|
| `LLMError` | `upstream_unavailable` | 可重试（网关 503 / 超时） |
| `ScoreError`（拒答/证据不成立） | `invalid_input` | 换文档或补证据，重试无用 |
| `ValueError`（金标集不存在等） | `invalid_input` | 改参数 |
| `sqlite3.OperationalError` / `OSError` | `upstream_unavailable` | 本地库不可用，可重试 |

**为什么拒答算 invalid_input 而不是 upstream**：模型明确说了"这份文本里没有
可用的情绪证据"，重试同一份文本不会得到不同答案。把它归到可重试类会让调用方
反复烧 token。

**未知异常照常抛**：真正的 bug 要带堆栈出现在日志里，不能伪装成结构化错误。
"""
from __future__ import annotations

import functools
import sqlite3
from typing import Any, Callable

from .errors import INVALID_INPUT, UPSTREAM_UNAVAILABLE, error_result


def _classify(exc: BaseException) -> tuple[str, str, dict[str, Any]] | None:
    from .llm import LLMError
    from .models import ScoreError

    if isinstance(exc, LLMError):
        return UPSTREAM_UNAVAILABLE, str(exc), {"upstream": "llm_gateway"}
    if isinstance(exc, ScoreError):
        return INVALID_INPUT, str(exc), {"reason": "refused_or_unevidenced"}
    if isinstance(exc, ValueError):
        return INVALID_INPUT, str(exc), {}
    if isinstance(exc, (sqlite3.OperationalError, OSError)):
        return UPSTREAM_UNAVAILABLE, f"本地存储不可用：{type(exc).__name__}: {exc}", {
            "upstream": "local_sqlite_store",
        }
    return None


def map_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """包一层：已知领域异常 → 结构化错误码；未知异常照常抛。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — 分类后未知异常会重新抛出
            classified = _classify(exc)
            if classified is None:
                raise
            code, message, extra = classified
            return error_result(code, message, **extra)

    return wrapper
