"""批量与增量：并发上限、单条超时、逐条报错、跳过已打分。

纪律（spec）：
  - 单条失败不中断整批：每条独立捕获，结果里逐条带原因。
  - 增量：同版本已打分文档跳过并计数（不调模型）。
  - 超时按失败记录，不挂起。
"""
from __future__ import annotations

import concurrent.futures as _cf
import dataclasses
from typing import Any, Iterable

from .llm import LLMClient
from .models import ScoreError
from .prompt import PromptArtifact, default_prompt
from .scoring import ScoreRequest, score_document


def score_batch(
    client: LLMClient,
    docs: Iterable[ScoreRequest],
    *,
    store,
    prompt: PromptArtifact | None = None,
    max_concurrency: int = 4,
    per_doc_timeout_s: float = 180.0,
    skip_existing: bool = True,
    task_store=None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """批量打分。返回 {results, skipped, failed, ok, total, task_id}。

    results 逐条含 doc_id / status(ok|skipped|failed) / 记录或 error。

    传入 `task_store` 时，每条结果同步落到任务表（`tasks.py`），使进度可被
    查询、失败项可单独重试、重启后状态仍可读（mcp-long-running-operations）。
    不传则保持纯内存行为，离线测试与嵌入式调用不受影响。
    """
    # 不传 prompt 时用服务缺省工件（与单篇入口同源），**不回退 v1**。
    # 兜底到某个历史版本会让批量与单篇落在不同幂等键上，两批分数不可比——
    # 生产库里的 12 条 v1 分数就是这么来的。
    prompt = prompt or default_prompt()
    docs = list(docs)
    if task_store is not None and task_id is None:
        task_id = task_store.create(
            "score_batch", [d.doc_id for d in docs],
            meta={"prompt_version": prompt.version, "model": client.cfg.model})
    if task_store is not None and task_id is not None:
        task_store.mark_running(task_id)
    results: list[dict[str, Any]] = []
    skipped = failed = ok = 0

    def work(req: ScoreRequest) -> dict[str, Any]:
        if skip_existing and store is not None:
            from .scoring import content_hash

            chash = content_hash(req.text)
            if store.get_score(req.doc_id, chash, prompt.version, client.cfg.model):
                return {"doc_id": req.doc_id, "status": "skipped"}
        try:
            rec = score_document(client, req, prompt=prompt, store=store)
            return {"doc_id": req.doc_id, "status": "ok", "record": rec}
        except ScoreError as exc:
            return {"doc_id": req.doc_id, "status": "failed", "error": f"拒绝：{exc}"}
        except Exception as exc:  # 模型/网络错误也逐条记，不中断
            return {"doc_id": req.doc_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    with _cf.ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
        futures = {pool.submit(work, d): d for d in docs}
        for fut, req in futures.items():
            try:
                r = fut.result(timeout=per_doc_timeout_s)
            except _cf.TimeoutError:
                r = {"doc_id": req.doc_id, "status": "failed", "error": f"超时（>{per_doc_timeout_s}s）"}
            except Exception as exc:
                r = {"doc_id": req.doc_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if r["status"] == "ok":
                ok += 1
            elif r["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
            results.append(r)
            if task_store is not None and task_id is not None:
                # skipped 记为 ok：它确实是被处理过的（幂等命中），重试时不该
                # 再跑一次——"已成功条目不得因重试被重复处理"包含这种情形。
                task_store.record_item(
                    task_id, r["doc_id"],
                    "ok" if r["status"] in ("ok", "skipped") else "failed",
                    result=r.get("record"), error=r.get("error"))

    out = {
        "total": len(docs),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "results": results,
        "prompt_version": prompt.version,
        "model": client.cfg.model,
    }
    if task_store is not None and task_id is not None:
        task_store.finish(task_id, "failed" if failed else "completed",
                          error=f"{failed} 条失败" if failed else None)
        out["task_id"] = task_id
    return out


def requests_from_rows(rows: Iterable[dict[str, Any]]) -> list[ScoreRequest]:
    """把外部文档行（doc_id/text/as_of/doc_type）转成打分请求。字段缺失即报错点名。"""
    out: list[ScoreRequest] = []
    for i, r in enumerate(rows):
        missing = [k for k in ("doc_id", "text", "as_of") if not r.get(k)]
        if missing:
            raise ValueError(f"第 {i} 行缺少字段：{', '.join(missing)}")
        out.append(ScoreRequest(doc_id=r["doc_id"], text=r["text"], as_of=r["as_of"],
                                doc_type=r.get("doc_type")))
    return out
