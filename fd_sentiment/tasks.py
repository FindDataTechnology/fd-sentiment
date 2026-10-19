"""长任务句柄与持久化状态（mcp-long-running-operations）。

**为什么需要这一层**：`score_batch` 一次要调几十次模型，耗时以分钟计。调用方
在一次 MCP 调用里干等会撞上自己的超时预算；而"整轮重跑"在网关抖动时会把
已经成功的几十条重做一遍（重复计费、结果不可比）。sentiment v3 评测实测过
这个代价：一次网关 503 让 38 条样本作废，只能整轮重来。

**状态落 SQLite 而不是内存**：进程重启后内存里的任务表就没了，调用方拿到的
是"未知任务"——那与"从未提交"无法区分。spec 明确要求重启后返回真实状态
（含"已中断"），所以状态必须与结果一起持久化。

**状态机的三个终态**：`completed` / `failed` / `interrupted`。
`interrupted` 不是"失败"：任务是被重启打断的，结果可能部分有效，重试应只
补未完成的条目。区分它是为了让调用方知道"这次没跑完，但已跑的部分是真的"。

**逐条结果与重试的关系**：每个条目独立记 `status` 与 `error`，重试只挑
`status == "failed"` 的条目——成功条目不会因重试被重新处理（spec 的
"已成功条目 SHALL NOT 被重复处理或重复计费"）。
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
import uuid
from typing import Any, Iterable, Optional

#: 任务状态。前三个是进行中，后三个是终态。
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
INTERRUPTED = "interrupted"

IN_PROGRESS = (QUEUED, RUNNING)
TERMINAL = (COMPLETED, FAILED, INTERRUPTED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    total         INTEGER NOT NULL DEFAULT 0,
    processed     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    finished_at   TEXT,
    -- 任务级摘要（如 prompt_version / model），供查询时回显而不必翻条目
    meta_json     TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS task_items (
    task_id       TEXT NOT NULL,
    item_key      TEXT NOT NULL,
    status        TEXT NOT NULL,          -- pending | ok | failed
    error         TEXT,
    result_json   TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (task_id, item_key)
);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class TaskStore:
    """任务状态与逐条结果的持久化存储（SQLite，与分数库同一文件）。"""

    def __init__(self, db_path: str | "Any"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── 生命周期 ────────────────────────────────────────────────

    def create(self, kind: str, item_keys: Iterable[str], meta: Optional[dict] = None) -> str:
        """登记一个新任务：所有条目以 pending 起算。

        `total` 在创建时就定下来（而不是随进度增长），这样"已处理/总数"
        从第一秒起就是个有意义的比值。
        """
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        keys = list(item_keys)
        now = _now()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO tasks (task_id, kind, status, total, processed,"
                " created_at, updated_at, meta_json) VALUES (?,?,?,?,?,?,?,?)",
                (task_id, kind, QUEUED, len(keys), 0, now, now,
                 json.dumps(meta or {}, ensure_ascii=False)),
            )
            c.executemany(
                "INSERT INTO task_items (task_id, item_key, status, updated_at)"
                " VALUES (?,?,'pending',?)",
                [(task_id, k, now) for k in keys],
            )
        return task_id

    def mark_running(self, task_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                      (RUNNING, _now(), task_id))

    def record_item(self, task_id: str, item_key: str, status: str,
                    *, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        """记一条结果并推进 processed 计数。

        `processed` 只统计**终态条目**（ok / failed），所以"已处理数"不会因为
        一条还在跑而虚高。进度只增不减由 SQL 保证（processed 由子查询重算，
        不是自增），并发下不会出现倒退。
        """
        now = _now()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO task_items (task_id, item_key, status, error,"
                " result_json, updated_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT (task_id, item_key) DO UPDATE SET"
                " status=excluded.status, error=excluded.error,"
                " result_json=excluded.result_json, updated_at=excluded.updated_at",
                (task_id, item_key, status, error,
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 now),
            )
            c.execute(
                "UPDATE tasks SET processed = ("
                "  SELECT COUNT(*) FROM task_items"
                "  WHERE task_id=? AND status IN ('ok','failed')"
                "), updated_at=? WHERE task_id=?",
                (task_id, now, task_id),
            )

    def finish(self, task_id: str, status: str, error: Optional[str] = None) -> None:
        if status not in TERMINAL:
            raise ValueError(f"finish 只接受终态，收到 {status!r}")
        now = _now()
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE tasks SET status=?, updated_at=?, finished_at=?, error=?"
                " WHERE task_id=?",
                (status, now, now, error, task_id),
            )

    def mark_interrupted_stale(self) -> int:
        """进程启动时调用：把上次进程留下的"进行中"任务标为 interrupted。

        这是"重启后不返回未知任务"的实现点——任务表在盘上，所以能看出"有个
        任务停在 running 而进程已经换了一茬"，据此如实报告 interrupted，
        而不是让调用方看到一个永远不会推进的 running。
        """
        now = _now()
        with self._lock, self._conn() as c:
            cur = c.execute(
                "UPDATE tasks SET status=?, updated_at=?, error=?"
                " WHERE status IN (?,?)",
                (INTERRUPTED, now,
                 "服务重启：任务在运行中被中断，已完成的条目结果仍然有效；"
                 "用 retry_failed 只重跑未完成的条目。",
                 QUEUED, RUNNING),
            )
            return cur.rowcount

    # ── 查询 ────────────────────────────────────────────────────

    def status(self, task_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                return None
            counts = {r["status"]: r["n"] for r in c.execute(
                "SELECT status, COUNT(*) n FROM task_items WHERE task_id=?"
                " GROUP BY status", (task_id,)).fetchall()}
        out = {
            "task_id": row["task_id"],
            "kind": row["kind"],
            "status": row["status"],
            "total": row["total"],
            "processed": row["processed"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "finished_at": row["finished_at"],
            "counts": {
                "pending": counts.get("pending", 0),
                "ok": counts.get("ok", 0),
                "failed": counts.get("failed", 0),
            },
            "meta": json.loads(row["meta_json"] or "{}"),
        }
        if row["error"]:
            out["error"] = row["error"]
        # 进度可观测：进行中且 processed 长时间不变，调用方据此判断卡住。
        out["progress"] = (f"{row['processed']}/{row['total']}")
        return out

    def items(self, task_id: str, status: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM task_items WHERE task_id=?"
        params: list[Any] = [task_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY item_key"
        with self._conn() as c:
            rows = c.execute(sql, tuple(params)).fetchall()
        return [{
            "item_key": r["item_key"],
            "status": r["status"],
            "error": r["error"],
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
        } for r in rows]

    def failed_keys(self, task_id: str) -> list[str]:
        """待重试的条目：失败项 + 重启后仍未处理的 pending 项。

        pending 也要算进来——一个被中断的任务里，未处理的条目不是"成功"，
        重试时必须补上，否则"只重跑失败项"会漏掉一半工作。
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT item_key FROM task_items WHERE task_id=?"
                " AND status IN ('failed','pending') ORDER BY item_key",
                (task_id,)).fetchall()
        return [r["item_key"] for r in rows]
