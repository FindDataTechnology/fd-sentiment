#!/usr/bin/env python3
"""端到端：真实公告 → 批量打分 → evidence 抽查 → 聚合（task 7.1）。

数据来源：fd-cn-report 的 cninfo 通道（真实公告 PDF）。本脚本只读公告，
不取市场数据——打分输入只有公告正文（point-in-time 由接口形状保证）。

用法：
    python3 scripts/e2e_real_announcements.py --limit 20
输出：reports/e2e-real-announcements.md + 控制台摘要
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fd-cn-report"))

from fd_sentiment.aggregate import aggregate
from fd_sentiment.batch import score_batch
from fd_sentiment.llm import LLMClient, LLMConfig
from fd_sentiment.models import locate_quote
from fd_sentiment.scoring import ScoreRequest
from fd_sentiment.storage import Store

COMPANIES = ["600519", "000333", "000001", "600036", "000858"]
CATEGORIES = ["业绩预告", "业绩快报"]


def fetch_filings(limit: int) -> list[dict]:
    import cninfo_client as c

    out: list[dict] = []
    for ticker in COMPANIES:
        co = c.lookup_company(ticker)
        if not co:
            continue
        for cat in CATEGORIES:
            try:
                rows = c.query_announcements(co["stock_code"], co["org_id"], category=cat, limit=limit)
            except Exception as exc:
                print(f"  ! {ticker}/{cat} 列表失败：{exc}")
                continue
            for r in rows:
                if r.get("pdf_url"):
                    out.append(r)
            if len(out) >= limit * 2:
                break
        if len(out) >= limit * 2:
            break
    return out[:limit]


def fetch_pdf_text(url: str) -> str:
    import httpx

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    # 复用 fd-cn-report 的 PDF 文本抽取（同一套解析，避免两套口径）
    import cnreport_tools as t

    return t.extract_pdf_text(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--db", default=str(ROOT / "e2e.db"))
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cfg = LLMConfig.from_env()
    if not cfg.base_url or not cfg.api_key or not cfg.model:
        print("✗ LLM 配置缺失（FD_SENTIMENT_LLM_BASE_URL / _API_KEY / _MODEL）", file=sys.stderr)
        return 2
    store = Store(args.db)
    print(f"model={cfg.model} db={args.db}")

    print("① 取真实公告列表…")
    filings = fetch_filings(args.limit)
    print(f"   {len(filings)} 条")

    print("② 下载并抽取正文…")
    docs: list[ScoreRequest] = []
    skipped: list[dict] = []
    for f in filings:
        try:
            text = fetch_pdf_text(f["pdf_url"])
        except Exception as exc:
            skipped.append({"announcement_id": f["announcement_id"], "error": f"PDF 失败：{exc}"})
            continue
        if len(text.strip()) < 100:
            skipped.append({"announcement_id": f["announcement_id"], "error": "正文过短（<100 字）"})
            continue
        docs.append(ScoreRequest(
            doc_id=str(f["announcement_id"]),
            text=text,
            as_of=f["published"][:10],
            doc_type=f["title"][:40],
        ))
    print(f"   可打分 {len(docs)} 篇，跳过 {len(skipped)} 篇")

    print("③ 批量打分…")
    out = score_batch(LLMClient(cfg, store=store), docs, store=store, max_concurrency=args.concurrency)
    print(f"   ok={out['ok']} skipped={out['skipped']} failed={out['failed']}")

    print("④ evidence 抽查（逐条回原文定位）…")
    text_by_id = {d.doc_id: d.text for d in docs}
    checks: list[dict] = []
    for r in out["results"]:
        if r["status"] != "ok":
            continue
        rec = r["record"]
        text = text_by_id[rec["doc_id"]]
        for span in rec["evidence_spans"]:
            loc = locate_quote(span["quote"], text)
            sliced = text[span["char_start"]:span["char_end"]] if loc else ""
            checks.append({
                "doc_id": rec["doc_id"],
                "quote": span["quote"][:60],
                "located": loc is not None,
                "slice_matches": sliced == span["quote"],
                "char_start": span["char_start"],
                "char_end": span["char_end"],
            })
    loc_ok = sum(1 for c in checks if c["located"])
    slice_ok = sum(1 for c in checks if c["slice_matches"])
    print(f"   evidence 片段 {len(checks)} 条：可定位 {loc_ok}，切片逐字一致 {slice_ok}")

    print("⑤ 聚合视图…")
    agg = aggregate(store, universe_total=len(filings))

    report = [
        "# 端到端：真实公告批量打分（task 7.1）",
        "",
        f"- 运行时间：{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
        f"- 模型：`{cfg.model}`｜prompt：`{out['prompt_version']}`｜库：`{args.db}`",
        f"- 公告来源：fd-cn-report / cninfo（真实公告 PDF）；公司池 {'/'.join(COMPANIES)}",
        "",
        "## 批量结果",
        "",
        f"- 取回公告 {len(filings)} 条；可打分 {len(docs)} 篇；跳过 {len(skipped)} 篇",
        f"- 打分 ok={out['ok']} / skipped={out['skipped']} / failed={out['failed']}",
        "",
        "## evidence 抽查（每条都在原文定位）",
        "",
        f"- 片段总数 {len(checks)}；可定位 {loc_ok}；切片与 quote 逐字一致 {slice_ok}",
        "",
        "| doc_id | quote（截断） | 可定位 | 切片一致 | 区间 |",
        "|---|---|---|---|---|",
    ]
    for c in checks[:20]:
        report.append(
            f"| {c['doc_id']} | {c['quote']} | {'✓' if c['located'] else '✗'} | "
            f"{'✓' if c['slice_matches'] else '✗'} | {c['char_start']}..{c['char_end']} |"
        )
    report += [
        "",
        "## 聚合视图（分布统计，非情绪指数）",
        "",
        "```json",
        json.dumps(agg, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 失败/跳过明细",
        "",
    ]
    for r in out["results"]:
        if r["status"] == "failed":
            report.append(f"- 失败 {r['doc_id']}：{r['error']}")
    for s in skipped:
        report.append(f"- 跳过 {s['announcement_id']}：{s['error']}")
    if not any(r["status"] == "failed" for r in out["results"]) and not skipped:
        report.append("- 无")

    rp = ROOT / "reports" / "e2e-real-announcements.md"
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"报告：{rp}")
    return 0 if (loc_ok == len(checks) and slice_ok == len(checks)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
