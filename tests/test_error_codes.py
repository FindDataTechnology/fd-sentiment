"""错误码契约（mcp-error-semantics 组 4）。

本服务的四类错误里，两类的区分最关键：

- **`upstream_unavailable`（可重试）**：LLM 网关不可达 / 503 / 超时。重试
  有意义，因为下一次可能就通了。
- **`invalid_input`（不可重试）**：模型拒答（"这份文本没有可用的情绪证据"）
  或证据不成立。重试同一份文本不会得到不同答案，把它归到可重试类会让调用方
  反复烧 token。

这条区分此前不存在：两者都返回一段中文文本，调用方只能靠关键词猜。
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from fd_sentiment import errors as E
from fd_sentiment.error_mapping import map_errors
from fd_sentiment.errors import CODES, RETRYABLE, error_body
from fd_sentiment.llm import LLMError
from fd_sentiment.models import ScoreError

REPO = pathlib.Path(__file__).resolve().parents[1]


# ── 码表 ───────────────────────────────────────────────────────

def test_codes_cover_the_four_classes():
    for required in (E.INVALID_INPUT, E.NOT_FOUND, E.NOT_WIRED, E.UPSTREAM_UNAVAILABLE):
        assert required in CODES


def test_only_upstream_is_retryable():
    assert RETRYABLE[E.UPSTREAM_UNAVAILABLE] is True
    for code in (E.INVALID_INPUT, E.NOT_FOUND, E.NOT_WIRED, E.INTERNAL_ERROR):
        assert RETRYABLE[code] is False


def test_unregistered_code_rejected():
    with pytest.raises(ValueError):
        error_body("made_up", "x")


# ── 领域异常映射 ───────────────────────────────────────────────

def test_llm_error_maps_to_upstream_unavailable():
    @map_errors
    def tool():
        raise LLMError("网关返回 503")

    err = tool().structured_content["error"]
    assert err["code"] == E.UPSTREAM_UNAVAILABLE
    assert err["upstream"] == "llm_gateway"
    assert err["retryable"] is True


def test_score_refusal_maps_to_invalid_input_not_retryable():
    """拒答不是故障：重试同一份文本不会得到不同答案。"""
    @map_errors
    def tool():
        raise ScoreError("模型拒绝打分（无可用情绪证据）：文本为纯事实陈述")

    err = tool().structured_content["error"]
    assert err["code"] == E.INVALID_INPUT
    assert err["retryable"] is False
    assert "拒绝打分" in err["message"]


def test_local_store_failure_maps_to_upstream():
    @map_errors
    def tool():
        raise OSError(13, "Permission denied")

    err = tool().structured_content["error"]
    assert err["code"] == E.UPSTREAM_UNAVAILABLE
    assert err["upstream"] == "local_sqlite_store"


def test_unknown_exception_re_raised():
    @map_errors
    def tool():
        raise KeyError("real bug")

    with pytest.raises(KeyError):
        tool()


# ── 真实 MCP 往返 ─────────────────────────────────────────────

def _server(tmp_path):
    import importlib
    import os

    os.environ["FD_SENTIMENT_DB"] = str(tmp_path / "s.db")
    import fd_sentiment.server as srv

    importlib.reload(srv)
    return srv.build_server()


def test_llm_failure_over_the_wire_is_upstream_unavailable(tmp_path, monkeypatch):
    """没有网关凭据时，打分工具必须报 upstream_unavailable 而非空结果。

    "取不到"与"真的没有"必须可区分——这条是 spec 里"不可静默降级为成功"
    的直接落地。
    """
    import asyncio
    import os

    for k in ("FD_SENTIMENT_LLM_BASE_URL", "FD_SENTIMENT_LLM_API_KEY",
              "FD_SENTIMENT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    os.environ["FD_SENTIMENT_DB"] = str(tmp_path / "s.db")

    import importlib

    import fd_sentiment.server as srv

    importlib.reload(srv)
    from fastmcp import Client

    async def go():
        async with Client(srv.build_server()) as c:
            return await c.call_tool(
                "score_document",
                {"doc_id": "d1", "text": "公司业绩预增，净利润同比增长 50%。",
                 "as_of": "2025-01-02"},
                raise_on_error=False)

    r = asyncio.run(go())
    assert r.is_error is True, "网关不可用不能被当成成功返回"
    err = r.structured_content["error"]
    assert err["code"] == E.UPSTREAM_UNAVAILABLE
    assert err["upstream"] == "llm_gateway"


def test_read_only_query_still_succeeds_and_is_not_an_error(tmp_path):
    import asyncio
    import os

    os.environ["FD_SENTIMENT_DB"] = str(tmp_path / "s.db")
    import importlib

    import fd_sentiment.server as srv

    importlib.reload(srv)
    from fastmcp import Client

    async def go():
        async with Client(srv.build_server()) as c:
            return await c.call_tool("list_scores", {}, raise_on_error=False)

    r = asyncio.run(go())
    assert r.is_error is False
    assert r.data.get("count") == 0


# ── 镜像一致性 ────────────────────────────────────────────────

def test_errors_module_mirrors_the_quant_copy():
    """错误码表与 fd-quant 的同名模块逐字一致（代码部分）。

    码表分叉意味着同一个码在两个服务里含义可能不同——那比"没有码"更坏，
    因为调用方会以为自己已经处理了。
    """
    src = REPO.parent / "fd-quant" / "fd_quant" / "errors.py"
    if not src.exists():
        pytest.skip("fd-quant 不在本工作区")

    def code_part(path):
        text = path.read_text(encoding="utf-8")
        end = text.find('"""', 3)
        return text[end + 3:].lstrip("\n")

    assert code_part(src) == code_part(REPO / "fd_sentiment" / "errors.py"), (
        "errors.py 与 fd-quant 的副本已分叉"
    )
