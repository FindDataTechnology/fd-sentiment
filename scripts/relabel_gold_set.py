#!/usr/bin/env python3
"""重标一个既有金标集：样本不动，只按当前标注规则重算档位。

用法：
    python3 scripts/relabel_gold_set.py --db /tmp/goldv9.db --set-version gold-v9 --dry-run
    python3 scripts/relabel_gold_set.py --db /tmp/goldv9.db --set-version gold-v9

为什么单独一个脚本：`build_gold_set.py` 会重新采样、重新下载 PDF，做不到
「只换标签不换样本」。而规则修正（差额式、万元口径、上年同期小节识别）
必须能在同一批样本上重算，才能把「尺子变了」与「题变了」两个变量分开——
这是 gold-v8 立下的方法论。

标题必须重新取：`gold_set.note` 里存的是标题（旧版本截断到 40 字），长标题
会把关键词截掉（实测「…摊薄即期回报的风险提示公告」只剩「…摊薄即期」），
拿截断标题重标会让 5 条风险提示变成「无规则命中」。故按 note 里的 ticker
重新查 CNINFO，用 announcement_id 换回完整标题；查不到的样本保留原档并
显式列出，不静默跳过。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "fd-cn-report"))

import build_gold_set as g  # noqa: E402

ALL_BANDS = ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish")
CATEGORIES = ("业绩预告", "风险提示", "权益分派", "股东大会", "董事会", "公司治理")


def parse_note(note: str | None) -> tuple[str, str, str] | None:
    """note 形如 `ticker｜category｜title`。"""
    if not note:
        return None
    parts = note.split("｜", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def fetch_titles(tickers: list[str]) -> dict[str, str]:
    """按公司拉六类公告，建 doc_id → 完整标题 的映射。

    查询参数与 build_gold_set 保持一致（每类 limit=25）：金标样本就是从这
    几批结果里选的，用同一窗口才能覆盖到同一条公告。
    """
    import cninfo_client as c

    titles: dict[str, str] = {}
    for ticker in tickers:
        co = c.lookup_company(ticker)
        if not co:
            print(f"  ! {ticker}: 无法解析公司", file=sys.stderr)
            continue
        for category in CATEGORIES:
            try:
                rows = c.query_announcements(
                    co["stock_code"], co["org_id"], category=category, limit=25
                )
            except Exception as exc:  # 网络/上游异常：跳过并记录，不中断整轮
                print(f"  ! {ticker}/{category}: {exc}", file=sys.stderr)
                continue
            for r in rows:
                titles[r["announcement_id"]] = r["title"]
    return titles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--set-version", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只打印变更，不写库")
    args = ap.parse_args()

    from fd_sentiment.storage import Store

    store = Store(args.db)
    rows = store.list_gold(args.set_version)
    if not rows:
        print(f"error: {args.set_version} 在 {args.db} 里没有样本", file=sys.stderr)
        return 1

    parsed = {r["doc_id"]: parse_note(r.get("note")) for r in rows}
    tickers = sorted({p[0] for p in parsed.values() if p})
    print(f"样本 {len(rows)} 条，涉及 {len(tickers)} 家公司；回查完整标题……")
    titles = fetch_titles(tickers)

    missing_title = [d for d in parsed if d not in titles]
    if missing_title:
        print(f"⚠ {len(missing_title)} 条查不到完整标题，将沿用 note 里的标题："
              f"{'、'.join(missing_title)}", file=sys.stderr)

    before = Counter(r["expected_band"] for r in rows)
    after: Counter[str] = Counter()
    changes: list[tuple[str, str, str, str, str]] = []  # doc_id, ticker, old, new, title
    unresolved: list[tuple[str, str, str]] = []        # doc_id, ticker, title

    for r in rows:
        doc_id = r["doc_id"]
        note = parsed[doc_id]
        if note is None:
            after[r["expected_band"]] += 1
            unresolved.append((doc_id, "?", r.get("note") or ""))
            continue
        ticker, category, title_hint = note
        title = titles.get(doc_id, title_hint)
        band = g.label(title, category, r["text"])
        if band is None:
            # 规则表没命中：保留原档，显式列出——这通常意味着标题关键词被
            # 截断或公告类型不在规则表覆盖范围内，需要人工看，不能猜。
            after[r["expected_band"]] += 1
            unresolved.append((doc_id, ticker, title))
            continue
        after[band] += 1
        if band != r["expected_band"]:
            changes.append((doc_id, ticker, r["expected_band"], band, title))
        if not args.dry_run:
            store.upsert_gold(
                set_version=args.set_version, doc_id=doc_id, as_of=r["as_of"],
                expected_band=band, text=r["text"],
                note=f"{ticker}｜{category}｜{title}",
            )

    print(f"\n档位分布 前 → 后：")
    for b in ALL_BANDS:
        mark = "  " if before[b] == after[b] else "→ "
        print(f"  {mark}{b:15s} {before[b]:3d} → {after[b]:3d}")
    print(f"  合计 {sum(before.values())} → {sum(after.values())}")

    print(f"\n变更 {len(changes)} 条：")
    for doc_id, ticker, old, new, title in changes:
        print(f"  {doc_id} {ticker}  {old:15s} → {new:15s}  {title[:50]}")

    if unresolved:
        print(f"\n未判定（保留原档）{len(unresolved)} 条：", file=sys.stderr)
        for doc_id, ticker, title in unresolved:
            print(f"  {doc_id} {ticker}  {title[:60]}", file=sys.stderr)

    missing_bands = [b for b in ALL_BANDS if after[b] == 0]
    if missing_bands:
        print(f"\n✗ 重标后缺档：{'、'.join(missing_bands)}", file=sys.stderr)
        return 1
    thin = [b for b in ALL_BANDS if after[b] < g.WEAK_BAND_MIN]
    if thin:
        print(f"\n⚠ 低于 {g.WEAK_BAND_MIN} 条的档位（结论不具统计意义）："
              f"{'、'.join(thin)}", file=sys.stderr)

    if args.dry_run:
        print("\n(dry-run：未写库)")
    else:
        print(f"\n✓ 已写回 {args.db} 的 {args.set_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
