"""工具行为属性契约（mcp-tool-annotations / mcp-surface-contract）。

三件事必须成立，缺一条客户端就会误判自动批准策略：

1. **齐备**：8 个工具在 `tools/list` 里都带四项提示——缺一项客户端就得靠猜。
2. **与实际副作用一致**：写库的工具不得标只读。这里不靠读代码判，而是把
   「哪个工具写库」写成显式期望表，再与运行时声明逐项比对；实现若改了副作用
   而没改声明，本文件会红。
3. **同类一致**：查询/聚合/元信息/对比评测同属只读族，属性必须相同。

另有一条针对 `score_batch` 的特殊断言：它的幂等键由内容决定（重复提交不会
重复调模型、库里也不多行），但**返回会变**（第二次 ok→skipped）。MCP 的幂等
提示回答的是"能否安全重放且结果相同"，故它必须标非幂等——标成幂等会让客户端
把 skipped 误读成"这批没打过分"。
"""
from __future__ import annotations

import asyncio

import pytest

from fd_sentiment.server import build_server
from fd_sentiment.surface import TOOLS

HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")

#: 写持久状态的工具（分数落 SQLite / 评测报告落库 / 改写任务表）。
#: 这是**期望**而非实现派生值——正是它让"实现改了而声明没改"能被检出。
WRITES_STATE = {"score_document", "score_batch", "run_evaluation", "retry_failed"}

#: 调用期会出网（LLM 网关）。get_task_status 只读本地任务表，不出网。
TOUCHES_NETWORK = {"score_document", "score_batch", "run_evaluation", "retry_failed"}


@pytest.fixture(scope="module")
def runtime_annotations() -> dict[str, dict]:
    app = build_server()
    out: dict[str, dict] = {}
    for t in asyncio.run(app._list_tools()):
        a = t.annotations
        assert a is not None, f"{t.name} 未声明任何行为属性"
        out[t.name] = {k: getattr(a, k) for k in HINTS}
    return out


def test_every_tool_declares_all_four_hints(runtime_annotations):
    names = {t.name for t in TOOLS}
    assert set(runtime_annotations) == names
    for name, a in runtime_annotations.items():
        for k in HINTS:
            assert a[k] is not None, f"{name}.{k} 未声明"


def test_tools_that_write_state_are_not_marked_readonly(runtime_annotations):
    for name, a in runtime_annotations.items():
        if name in WRITES_STATE:
            assert a["readOnlyHint"] is False, (
                f"{name} 写持久状态却标为只读——客户端会自动批准它"
            )


def test_query_tools_are_readonly_and_idempotent(runtime_annotations):
    readers = {t.name for t in TOOLS} - WRITES_STATE
    assert readers, "只读族为空，说明 WRITES_STATE 覆盖过头"
    for name in readers:
        a = runtime_annotations[name]
        assert a["readOnlyHint"] is True, f"{name} 只读却未标只读"
        assert a["idempotentHint"] is True, f"{name} 只读却不幂等"


def test_open_world_matches_network_reach(runtime_annotations):
    for name, a in runtime_annotations.items():
        expected = name in TOUCHES_NETWORK
        assert a["openWorldHint"] is expected, (
            f"{name}: openWorldHint={a['openWorldHint']}，期望 {expected}"
        )


def test_score_batch_is_not_idempotent(runtime_annotations):
    """幂等键在内容上，但重复提交的**返回**不同（ok → skipped）。"""
    assert runtime_annotations["score_batch"]["idempotentHint"] is False


def test_score_document_is_idempotent(runtime_annotations):
    """同 (doc_id, content_hash, prompt_version, model) 返回既有记录且 created=False。"""
    assert runtime_annotations["score_document"]["idempotentHint"] is True


def test_no_tool_is_marked_destructive(runtime_annotations):
    """全部是追加式写入：没有工具会删除或不可恢复地改写既有行。"""
    for name, a in runtime_annotations.items():
        assert a["destructiveHint"] is False, f"{name} 标了破坏性，与追加式写入不符"


def test_same_family_shares_identical_attributes(runtime_annotations):
    """只读查询族（get_score / list_scores / aggregate_scores / compare_evaluations /
    current_prompt_version / get_task_status）同属性。"""
    family = ["get_score", "list_scores", "aggregate_scores", "compare_evaluations",
              "current_prompt_version", "get_task_status"]
    first = runtime_annotations[family[0]]
    for name in family[1:]:
        assert runtime_annotations[name] == first, f"{name} 与同族属性不一致"


def test_annotations_are_copies_not_shared(runtime_annotations):
    """改一个工具的属性不得污染其它工具（共享 dict 会让一次误改扩散到全族）。"""
    from fd_sentiment.surface import annotations_for

    a = annotations_for("get_score")
    a["readOnlyHint"] = False
    assert annotations_for("list_scores")["readOnlyHint"] is True
