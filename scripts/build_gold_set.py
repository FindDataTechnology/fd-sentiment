#!/usr/bin/env python3
"""金标集构建（task 5.1）：≥50 条已标注公告，三档以上均衡。

标注原则（写在这里，避免"标注者口径随心情变"）：
  - 以**公告对相关主体经营/财务的影响方向**为准，不以市场反应为准。
  - 业绩预增/快报净利大幅增长且措辞强（"大幅上升""显著增长"）→ strong_bullish
  - 温和增长、略超预期 → bullish
  - 纯事务性（会议通知、章程修订、分红实施公告）或正负相抵 → neutral
  - 温和下滑/略低于预期 → bearish
  - 大幅下滑/亏损/重大风险 → strong_bearish
  - **标注只看标题与正文方向词**，不看数字大小猜（数字大小归打分器判断）。

用法：
    python3 scripts/build_gold_set.py --set-version gold-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fd-cn-report"))

# 标注规则表：按 (类别, 标题关键词) → band。这是**人工标注的显式规则**，
# 不是模型输出——金标的独立性靠这条保证（打分器不能自己给自己出题）。
RULES: list[tuple[str, tuple[str, ...], str]] = [
    ("业绩预告", ("预增", "预盈", "扭亏", "大幅上升", "同向上升", "增长"), "bullish"),
    ("业绩预告", ("预减", "预亏", "首亏", "续亏", "大幅下降", "同向下降"), "bearish"),
    # 风险提示类：几乎全是负面（诉讼/处罚/退市风险/资金占用），用来补 bearish 档
    ("风险提示", ("退市", "暂停上市", "重大违法", "资金占用", "违规担保", "立案"), "strong_bearish"),
    ("风险提示", ("诉讼", "处罚", "警示", "问询"), "bearish"),
    # "股票交易异常波动"本身只是披露义务触发，方向取决于同期消息面 →
    # 归中性（不能把披露义务当利空，这是标注口径问题，不是模型问题）。
    ("风险提示", ("异常波动", "风险提示"), "neutral"),
    # 权益分派：**只有"方案"公告才带方向信息**（首次宣布提高/调整分红是回报政策
    # 的信号）；"实施公告"是事务性执行通知（按既定方案发钱），不构成新利好 →
    # 归中性。v2 评测实证：把实施公告标成 bullish 是标注错误（11 条全被模型正确
    # 判成 neutral），模型对、金标错。
    ("权益分派", ("方案", "调整"), "bullish"),
    ("权益分派", ("实施",), "neutral"),
    # 纯事务性公告 → 中性（补 neutral 档）
    ("股东大会", (), "neutral"),
    ("董事会", (), "neutral"),
    ("监事会", (), "neutral"),
    ("公司治理", (), "neutral"),
]

STRONG_WORDS = ("大幅", "显著", "重大", "翻倍", "超过", "激增")
NEUTRAL_TITLE = ("通知", "章程", "制度", "议事规则", "管理办法", "股东大会", "董事会决议")


def label(title: str, category: str) -> str | None:
    for kw in NEUTRAL_TITLE:
        if kw in title:
            return "neutral"
    for cat, kws, band in RULES:
        if cat != category:
            continue
        if not kws:  # 空关键词 = 整类按该档
            return band
        for kw in kws:
            if kw in title:
                if band in ("bullish", "bearish") and any(s in title for s in STRONG_WORDS):
                    return "strong_" + band
                return band
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-version", default="gold-v1")
    ap.add_argument("--target", type=int, default=54)
    ap.add_argument("--db", default=str(ROOT / "e2e.db"))
    args = ap.parse_args()

    from fd_sentiment.sanitize import flatten
    from fd_sentiment.storage import Store
    import cninfo_client as c
    import cnreport_tools as t
    import httpx

    store = Store(args.db)
    # 公司池：前六个是蓝筹（业绩预告天然偏正面，用于 bullish/neutral）；
    # 后六个是当期真实的预亏/预减公司（2026Q1 由 akshare 业绩预告快照筛出，
    # 137 家负面前景里取前几个）——三档均衡必须靠真实负面样本，不能靠把
    # 中性公告标成负面来凑。
    tickers = [
        "600519", "000333", "000001", "600036", "000858", "601318",
        "001393", "688106", "001248", "002027", "300024", "600029",
    ]
    counts: dict[str, int] = {}
    added = 0
    seen: set[str] = set()

    # 轮转采样：先每家公司取 1 条，再取第 2 条……避免前几家公司（蓝筹）
    # 把配额吃满而负面公司池永远轮不到——三档均衡靠的是覆盖面，不是排序。
    pending: dict[str, list[tuple[str, dict]]] = {}
    for ticker in tickers:
        co = c.lookup_company(ticker)
        if not co:
            continue
        bucket: list[tuple[str, dict]] = []
        for category in ("业绩预告", "风险提示", "权益分派", "股东大会", "董事会", "公司治理"):
            try:
                rows = c.query_announcements(co["stock_code"], co["org_id"], category=category, limit=15)
            except Exception as exc:
                print(f"  ! {ticker}/{category}: {exc}")
                continue
            for r in rows:
                if r.get("pdf_url"):
                    bucket.append((category, r))
        pending[ticker] = bucket

    round_idx = 0
    while added < args.target and any(round_idx < len(v) for v in pending.values()):
        for ticker in tickers:
            if added >= args.target:
                break
            bucket = pending.get(ticker, [])
            if round_idx >= len(bucket):
                continue
            category, r = bucket[round_idx]
            aid = r["announcement_id"]
            if aid in seen:
                continue
            band = label(r["title"], category)
            if band is None:
                continue
            try:
                with httpx.Client(timeout=60, follow_redirects=True) as cl:
                    text = flatten(t.extract_pdf_text(cl.get(r["pdf_url"]).content))
            except Exception as exc:
                print(f"  ! {aid} PDF: {exc}")
                continue
            if len(text) < 100:
                continue
            seen.add(aid)
            store.upsert_gold(
                set_version=args.set_version, doc_id=aid, as_of=r["published"][:10],
                expected_band=band, text=text,
                note=f"{ticker}｜{category}｜{r['title'][:40]}",
            )
            counts[band] = counts.get(band, 0) + 1
            added += 1
            print(f"  + {aid} {band:15s} {r['title'][:34]}")
        round_idx += 1

    if False:
        for ticker in tickers:
            if added >= args.target:
                break
            co = c.lookup_company(ticker)
            if not co:
                continue
            for category in ("x",):
                pass
            try:
                rows = c.query_announcements(co["stock_code"], co["org_id"], category=category, limit=15)
            except Exception as exc:
                print(f"  ! {ticker}/{category}: {exc}")
                continue
            for r in rows:
                if added >= args.target:
                    break
                aid = r["announcement_id"]
                if aid in seen or not r.get("pdf_url"):
                    continue
                band = label(r["title"], category)
                if band is None:
                    continue
                try:
                    with httpx.Client(timeout=60, follow_redirects=True) as cl:
                        text = flatten(t.extract_pdf_text(cl.get(r["pdf_url"]).content))
                except Exception as exc:
                    print(f"  ! {aid} PDF: {exc}")
                    continue
                if len(text) < 100:
                    continue
                seen.add(aid)
                store.upsert_gold(
                    set_version=args.set_version, doc_id=aid, as_of=r["published"][:10],
                    expected_band=band, text=text,
                    note=f"{ticker}｜{category}｜{r['title'][:40]}",
                )
                counts[band] = counts.get(band, 0) + 1
                added += 1
                print(f"  + {aid} {band:15s} {r['title'][:34]}")

    print(f"\n金标集 {args.set_version}：{added} 条，分布 {counts}")
    bands = [b for b in counts if counts[b] > 0]
    if len(bands) < 3:
        print("✗ 三档均衡未达标（需 ≥3 档）", file=sys.stderr)
        return 1
    if added < 50:
        print(f"✗ 样本量不足 50（实为 {added}）", file=sys.stderr)
        return 1
    print("✓ 满足 ≥50 条且 ≥3 档")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
