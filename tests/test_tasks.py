"""长任务原语（mcp-long-running-operations 组 5）。

五条要求各有一组断言，其中两条是这条契约的全部价值所在：

- **5.3/5.4 任务句柄与分项重试**：一批 60 条里 8 条失败后，重试**只**重跑
  这 8 条，其余 52 条不再调用模型。sentiment v3 实测过不这么做的代价——
  一次网关 503 让 38 条样本作废，只能整轮重来（重复计费 + 两批分数不可比）。
- **5.5 重启后状态可查**：任务状态落 SQLite。重启后查询返回 `interrupted`
  （并说明已完成的条目结果仍有效），**不**返回"未知任务"——后者会让调用方
  分不清是任务丢了还是自己记错了 id。

另有一条"进度只增不减"的断言：processed 由条目状态重算而非自增，故并发下
不会倒退，而"长时间不变"就是判断卡住的可靠信号。
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from fd_sentiment import tasks as T


@pytest.fixture()
def store():
    return T.TaskStore(str(Path(tempfile.mkdtemp()) / "t.db"))


# ── 5.1/5.2 逐条结果与进度 ─────────────────────────────────────

def test_every_item_carries_status_and_failure_reason(store):
    tid = store.create("score_batch", [f"d{i}" for i in range(3)])
    store.mark_running(tid)
    store.record_item(tid, "d0", "ok", result={"score": 0.5})
    store.record_item(tid, "d1", "failed", error="网关 503")
    store.record_item(tid, "d2", "ok")
    items = store.items(tid)
    assert {i["item_key"] for i in items} == {"d0", "d1", "d2"}
    by_key = {i["item_key"]: i for i in items}
    assert by_key["d1"]["status"] == "failed"
    assert "503" in by_key["d1"]["error"], "失败项必须带原因"


def test_progress_never_regresses(store):
    """processed 由条目状态重算，不是自增——并发下不会倒退。"""
    tid = store.create("score_batch", ["a", "b", "c"])
    assert store.status(tid)["processed"] == 0
    store.record_item(tid, "a", "ok")
    store.record_item(tid, "b", "failed", error="x")
    assert store.status(tid)["processed"] == 2
    # 重复记录同一条不会让计数虚高
    store.record_item(tid, "a", "ok")
    assert store.status(tid)["processed"] == 2
    assert store.status(tid)["progress"] == "2/3"


def test_progress_shows_counts_so_a_stall_is_detectable(store):
    tid = store.create("score_batch", ["a", "b"])
    store.record_item(tid, "a", "ok")
    info = store.status(tid)
    assert info["counts"] == {"pending": 1, "ok": 1, "failed": 0}
    assert info["progress"] == "1/2"


def test_unknown_task_returns_none_not_a_fake_status(store):
    """从未提交过的 id 返回 None → 工具报 not_found。

    这与"重启后仍能查到"是同一枚硬币的两面：未知只用于**真的没提交过**，
    不能用来掩盖"曾经提交过但状态丢了"。
    """
    assert store.status("task-never-submitted") is None


# ── 5.3 任务句柄 ───────────────────────────────────────────────

def test_submit_returns_a_handle_and_status_is_queryable(store):
    tid = store.create("score_batch", ["a", "b"], meta={"model": "m"})
    assert tid.startswith("task-")
    info = store.status(tid)
    assert info["task_id"] == tid
    assert info["kind"] == "score_batch"
    assert info["status"] == T.QUEUED
    assert info["total"] == 2
    assert info["meta"] == {"model": "m"}


def test_finish_sets_a_terminal_state(store):
    tid = store.create("score_batch", ["a"])
    store.record_item(tid, "a", "ok")
    store.finish(tid, T.COMPLETED)
    info = store.status(tid)
    assert info["status"] == T.COMPLETED
    assert info["finished_at"]


def test_finish_rejects_non_terminal_states(store):
    tid = store.create("score_batch", ["a"])
    with pytest.raises(ValueError):
        store.finish(tid, T.RUNNING)


# ── 5.4 分项重试：只重跑失败项 ────────────────────────────────

def test_failed_keys_excludes_successful_items(store):
    """60 条里 8 条失败 → 重试列表只有这 8 条。"""
    keys = [f"d{i}" for i in range(60)]
    failed = {f"d{i}" for i in range(52, 60)}  # 恰好 8 条
    tid = store.create("score_batch", keys)
    for k in keys:
        if k in failed:
            store.record_item(tid, k, "failed", error="网关 503")
        else:
            store.record_item(tid, k, "ok", result={"score": 0.1})

    retry = store.failed_keys(tid)
    assert set(retry) == failed
    assert len(retry) == 8, "只有 8 条该被重跑"
    # 成功条目不在列表里 = 不会被重新处理、不会重复计费
    assert not (set(keys) - failed) & set(retry)


def test_pending_items_are_retried_after_an_interruption(store):
    """被中断的任务里，pending 条目也算待重试。

    只挑 failed 会漏掉一半工作：中断时那些还没轮到的条目状态是 pending，
    它们不是"成功"。
    """
    tid = store.create("score_batch", ["a", "b", "c"])
    store.record_item(tid, "a", "ok")
    store.record_item(tid, "b", "failed", error="503")
    assert set(store.failed_keys(tid)) == {"b", "c"}


# ── 5.5 重启后状态可查 ────────────────────────────────────────

def test_restart_marks_inflight_tasks_interrupted_not_unknown(store):
    """重启后查询返回 interrupted，不是"未知任务"。"""
    tid = store.create("score_batch", ["a", "b"])
    store.mark_running(tid)
    store.record_item(tid, "a", "ok")  # 已完成的部分仍然有效

    # 模拟重启：同一个 db 文件上的新 store 实例
    restarted = T.TaskStore(store.db_path)
    n = restarted.mark_interrupted_stale()
    assert n == 1

    info = restarted.status(tid)
    assert info["status"] == T.INTERRUPTED
    assert info["progress"] == "1/2", "已完成的条目结果仍然有效并被计入"
    assert "重启" in (info.get("error") or "")
    # 关键：这不是"未知"——调用方能看出任务曾经提交过、停在哪
    assert info is not None


def test_completed_tasks_survive_a_restart(store):
    """已完成任务在重启后仍是 completed，不被误标 interrupted。"""
    tid = store.create("score_batch", ["a"])
    store.record_item(tid, "a", "ok")
    store.finish(tid, T.COMPLETED)
    restarted = T.TaskStore(store.db_path)
    restarted.mark_interrupted_stale()
    assert restarted.status(tid)["status"] == T.COMPLETED


def test_interrupted_task_still_lists_its_failed_keys(store):
    tid = store.create("score_batch", ["a", "b"])
    store.mark_running(tid)
    store.record_item(tid, "a", "ok")
    restarted = T.TaskStore(store.db_path)
    restarted.mark_interrupted_stale()
    assert set(restarted.failed_keys(tid)) == {"b"}


# ── 与 score_batch 的接线 ─────────────────────────────────────

def test_score_batch_records_a_task_when_a_store_is_passed(monkeypatch, tmp_path):
    """接上任务表后，score_batch 返回 task_id 且逐条结果落库。"""
    from fd_sentiment import batch
    from fd_sentiment.llm import LLMConfig
    from fd_sentiment.scoring import ScoreRequest

    class FakeClient:
        cfg = LLMConfig(base_url="http://f", api_key="k", model="fake")
        calls: list[str] = []

        def chat(self, messages, *, max_tokens=1500, doc_id=None):
            FakeClient.calls.append(doc_id)
            if doc_id == "bad":
                raise RuntimeError("网关 503")
            return '{"score":0.5,"band":"bullish","confidence":0.8,' \
                   '"evidence_spans":[{"quote":"业绩预增","char_start":0,"char_end":4}]}'

    class FakeStore:
        def get_score(self, *a, **k):
            return None

        def insert_score(self, rec):
            return dict(rec), True

    texts = {"ok1": "公司业绩预增，净利润同比增长 50%。",
             "ok2": "公司业绩预增，净利润同比增长 30%。",
             "bad": "公司业绩预增，净利润同比增长 20%。"}
    reqs = [ScoreRequest(doc_id=k, text=v, as_of="2025-01-02") for k, v in texts.items()]
    tstore = T.TaskStore(str(tmp_path / "t.db"))
    out = batch.score_batch(FakeClient(), reqs, store=FakeStore(),
                            max_concurrency=1, skip_existing=False, task_store=tstore)
    assert out["task_id"].startswith("task-")
    assert out["ok"] >= 1 and out["failed"] >= 1

    info = tstore.status(out["task_id"])
    assert info["total"] == 3
    assert info["processed"] == 3
    assert info["counts"]["failed"] == 1
    assert set(tstore.failed_keys(out["task_id"])) == {"bad"}


def test_retry_via_batch_uses_the_same_task_id(monkeypatch, tmp_path):
    """retry_failed 传回原 task_id：重试结果并入原任务，不产生第二个任务。"""
    from fd_sentiment import batch
    from fd_sentiment.llm import LLMConfig
    from fd_sentiment.scoring import ScoreRequest

    class FakeClient:
        cfg = LLMConfig(base_url="http://f", api_key="k", model="fake")

        def chat(self, messages, *, max_tokens=1500, doc_id=None):
            return '{"score":0.5,"band":"bullish","confidence":0.8,' \
                   '"evidence_spans":[{"quote":"业绩预增","char_start":0,"char_end":4}]}'

    class FakeStore:
        def get_score(self, *a, **k):
            return None

        def insert_score(self, rec):
            return dict(rec), True

    tstore = T.TaskStore(str(tmp_path / "t.db"))
    tid = tstore.create("score_batch", ["a", "b"])
    tstore.record_item(tid, "a", "ok")
    tstore.record_item(tid, "b", "failed", error="网关 503")

    out = batch.score_batch(
        FakeClient(),
        [ScoreRequest(doc_id="b", text="公司业绩预增，净利润同比增长 20%。",
                      as_of="2025-01-02")],
        store=FakeStore(), max_concurrency=1, skip_existing=False,
        task_store=tstore, task_id=tid)
    assert out["task_id"] == tid
    # a 仍是 ok（没被重跑），b 变 ok
    items = {i["item_key"]: i["status"] for i in tstore.items(tid)}
    assert items == {"a": "ok", "b": "ok"}
    assert tstore.status(tid)["status"] == T.COMPLETED
