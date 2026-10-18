"""SQLite 存储：scores / gold_set / eval_runs / llm_calls。

幂等键唯一约束放在库层而不是应用层——应用层的"先查再插"在并发下会漏，
唯一索引是唯一的真保证。同键重复插入捕获 IntegrityError 后回读既有行。

`llm_calls` 是调用计数的账：3.1 的幂等验收要求"模型不被重复调用——以调用
计数验证"，没有这张表就只能信日志。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    model           TEXT NOT NULL,
    score           REAL NOT NULL,
    band            TEXT NOT NULL,
    confidence      REAL NOT NULL,
    evidence_json   TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    scored_at       TEXT NOT NULL,
    chunk_index     INTEGER,
    chunk_total     INTEGER,
    doc_type        TEXT,
    tier_check_json TEXT,
    UNIQUE (doc_id, content_hash, prompt_version, model)
);
CREATE INDEX IF NOT EXISTS idx_scores_doc     ON scores (doc_id, as_of);
CREATE INDEX IF NOT EXISTS idx_scores_asof    ON scores (as_of);
CREATE INDEX IF NOT EXISTS idx_scores_version ON scores (prompt_version, model);

CREATE TABLE IF NOT EXISTS gold_set (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    set_version    TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    as_of          TEXT NOT NULL,
    expected_band  TEXT NOT NULL,
    text           TEXT NOT NULL,
    note           TEXT,
    UNIQUE (set_version, doc_id)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL UNIQUE,
    prompt_version  TEXT NOT NULL,
    model           TEXT NOT NULL,
    set_version     TEXT NOT NULL,
    samples         INTEGER NOT NULL,
    hits            INTEGER NOT NULL,
    hit_rate        REAL NOT NULL,
    calibration_ok  INTEGER NOT NULL,
    report_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        TEXT,
    model         TEXT NOT NULL,
    purpose       TEXT NOT NULL,
    ok            INTEGER NOT NULL,
    latency_ms    INTEGER,
    error         TEXT,
    created_at    TEXT NOT NULL
);
"""


class Store:
    """线程安全的最小存储层。批量打分用线程池，故每次操作独立连接。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            # 老库迁移：strong_tier_check 是 v3 引入的列，已存在的库 ALTER 补上
            # （executescript 的 CREATE TABLE IF NOT EXISTS 不会改已建的表）
            cols = {r[1] for r in c.execute("PRAGMA table_info(scores)")}
            if "tier_check_json" not in cols:
                c.execute("ALTER TABLE scores ADD COLUMN tier_check_json TEXT")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ---- scores -------------------------------------------------------
    def get_score(self, doc_id: str, content_hash: str, prompt_version: str, model: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM scores WHERE doc_id=? AND content_hash=? AND prompt_version=? AND model=?",
                (doc_id, content_hash, prompt_version, model),
            ).fetchone()
            return _row_to_score(row) if row else None

    def insert_score(self, rec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """返回 (记录, created)。同键已存在时返回既有行且 created=False。"""
        with self._lock, self._conn() as c:
            try:
                cur = c.execute(
                    """INSERT INTO scores
                       (doc_id, content_hash, prompt_version, model, score, band, confidence,
                        evidence_json, as_of, scored_at, chunk_index, chunk_total, doc_type, tier_check_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec["doc_id"], rec["content_hash"], rec["prompt_version"], rec["model"],
                        rec["score"], rec["band"], rec["confidence"],
                        json.dumps(rec["evidence_spans"], ensure_ascii=False),
                        rec["as_of"], rec["scored_at"],
                        rec.get("chunk_index"), rec.get("chunk_total"), rec.get("doc_type"),
                        json.dumps(rec["strong_tier_check"], ensure_ascii=False) if rec.get("strong_tier_check") else None,
                    ),
                )
                row = c.execute("SELECT * FROM scores WHERE id=?", (cur.lastrowid,)).fetchone()
                return _row_to_score(row), True
            except sqlite3.IntegrityError:
                row = c.execute(
                    "SELECT * FROM scores WHERE doc_id=? AND content_hash=? AND prompt_version=? AND model=?",
                    (rec["doc_id"], rec["content_hash"], rec["prompt_version"], rec["model"]),
                ).fetchone()
                return _row_to_score(row), False

    def list_scores(self, *, doc_id: str | None = None, prompt_version: str | None = None,
                    since: str | None = None, until: str | None = None,
                    limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM scores WHERE 1=1"
        args: list[Any] = []
        if doc_id:
            sql += " AND doc_id=?"; args.append(doc_id)
        if prompt_version:
            sql += " AND prompt_version=?"; args.append(prompt_version)
        if since:
            sql += " AND as_of>=?"; args.append(since)
        if until:
            sql += " AND as_of<=?"; args.append(until)
        sql += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            return [_row_to_score(r) for r in c.execute(sql, args).fetchall()]

    def has_scores(self, doc_id: str, prompt_version: str, model: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM scores WHERE doc_id=? AND prompt_version=? AND model=? LIMIT 1",
                (doc_id, prompt_version, model),
            ).fetchone()
            return row is not None

    def score_count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM scores").fetchone()[0])

    # ---- llm_calls ----------------------------------------------------
    def log_llm_call(self, *, doc_id: str | None, model: str, purpose: str,
                     ok: bool, latency_ms: int | None, error: str | None, created_at: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO llm_calls (doc_id, model, purpose, ok, latency_ms, error, created_at) VALUES (?,?,?,?,?,?,?)",
                (doc_id, model, purpose, 1 if ok else 0, latency_ms, error, created_at),
            )

    def llm_call_count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0])

    # ---- gold set -----------------------------------------------------
    def upsert_gold(self, *, set_version: str, doc_id: str, as_of: str,
                    expected_band: str, text: str, note: str | None = None) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO gold_set (set_version, doc_id, as_of, expected_band, text, note)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT (set_version, doc_id) DO UPDATE SET
                     as_of=excluded.as_of, expected_band=excluded.expected_band,
                     text=excluded.text, note=excluded.note""",
                (set_version, doc_id, as_of, expected_band, text, note),
            )

    def list_gold(self, set_version: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM gold_set WHERE set_version=? ORDER BY doc_id", (set_version,)
            ).fetchall()
            return [dict(r) for r in rows]

    def gold_versions(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT DISTINCT set_version FROM gold_set ORDER BY set_version")]

    # ---- eval runs ----------------------------------------------------
    def insert_eval_run(self, rec: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO eval_runs
                   (run_id, prompt_version, model, set_version, samples, hits, hit_rate,
                    calibration_ok, report_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (rec["run_id"], rec["prompt_version"], rec["model"], rec["set_version"],
                 rec["samples"], rec["hits"], rec["hit_rate"], 1 if rec["calibration_ok"] else 0,
                 json.dumps(rec["report"], ensure_ascii=False), rec["created_at"]),
            )

    def list_eval_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["calibration_ok"] = bool(d["calibration_ok"])
                d["report"] = json.loads(d["report_json"])
                out.append(d)
            return out

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM eval_runs WHERE run_id=?", (run_id,)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["calibration_ok"] = bool(d["calibration_ok"])
            d["report"] = json.loads(d["report_json"])
            return d


def _row_to_score(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["evidence_spans"] = json.loads(d.pop("evidence_json"))
    tj = d.pop("tier_check_json", None)
    d["strong_tier_check"] = json.loads(tj) if tj else None
    return d
