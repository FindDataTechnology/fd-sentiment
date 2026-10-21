#!/usr/bin/env python3
"""以既有金标集为基座，重标 + 追加新类别样本（gold-v10）。

用法：
    python3 scripts/extend_gold_set.py --base-db /tmp/goldv9.db --base gold-v9 \
        --target-db /tmp/goldv10.db --set-version gold-v10 --dry-run

与 `build_gold_set.py` 的分工：
- `build_gold_set.py` 从头采样（重新轮转六类公告），样本集合会变；
- `relabel_gold_set.py` 只换尺子不换题；
- **本脚本两者都做**：105 条基座原样保留（只按当前规则重算档位），再追加
  新类别样本。追加而非重采样，是为了让「gold-v9 → gold-v10」的分数差异
  只来自「新增了哪些题」+「尺子怎么改」，而不是「题全换了」。

为什么要追加新类别：`bearish` 档在 gold-v9 只有 1 条，原因是取样池只覆盖
业绩预告/风险提示/权益分派/股东大会/董事会/公司治理六类，而 A 股真实的
温和负面公告大量落在股东减持（股权变动）与业绩预告更正/涉及诉讼（补充更正）。
这不是模型问题，是取样偏差——补不齐这一档，它的准确率永远不可用。
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
BASE_CATEGORIES = ("业绩预告", "风险提示", "权益分派", "股东大会", "董事会", "公司治理")
# 新增取样类别：温和负面公告的真实来源
EXTRA_CATEGORIES = ("股权变动", "补充更正")

# 追加样本的公司池：在 gold-v9 的 28 家基础上补一批大中市值公司。
# 补的理由是**覆盖面**不是"找负面"：减持计划几乎每家大公司都有，但单家
# 只有 1–4 条，池子太窄会让 bearish 档被一两家公司垄断。
EXTRA_TICKERS = (
    "000651", "600276", "601166", "000538", "600887", "601012",
    "000725", "600690", "002304", "600104", "601668", "600585",
    "000100", "600900", "601857", "600050", "000039", "600309",
    "002027", "600346",
    # 600606 绿地：gold-v9 的公司池里没有它（业绩预告配额轮转没轮到），
    # 但它有两条「业绩预告更正」（亏损扩大、金额重大 → strong_bearish）
    # 和一条「涉及诉讼的补充公告」（→ bearish）。strong_bearish 原本只有 9 条，
    # 补它同时把两档都抬到 10 条以上。
    "600606",
)


def parse_note(note: str | None) -> tuple[str, str, str] | None:
    if not note:
        return None
    parts = note.split("｜", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def fetch_titles(tickers: list[str], categories: tuple[str, ...]) -> dict[str, str]:
    import cninfo_client as c

    titles: dict[str, str] = {}
    for ticker in tickers:
        co = c.lookup_company(ticker)
        if not co:
            print(f"  ! {ticker}: 无法解析公司", file=sys.stderr)
            continue
        for category in categories:
            try:
                rows = c.query_announcements(
                    co["stock_code"], co["org_id"], category=category, limit=25
                )
            except Exception as exc:
                print(f"  ! {ticker}/{category}: {exc}", file=sys.stderr)
                continue
            for r in rows:
                titles[r["announcement_id"]] = r["title"]
    return titles


def _is_wanted(category: str, title: str) -> bool:
    """类别内的类型闸：只取本 change 明确要覆盖的三类公告。

    不做这道闸会捞进大量无信息样本（实测 `补充更正` 里返回过「董事会决议
    公告」，`股权变动` 里有股权激励、增持、回购授权）——它们判中性不假，
    但对「负面档可用性」零贡献，只是把金标撑大。
    """
    if category == "股权变动":
        return "减持" in title
    if category == "补充更正":
        return "业绩预告" in title or "涉及诉讼" in title or "涉及仲裁" in title
    return False


def collect_new(
    tickers: list[str], existing: set[str], per_band_cap: int, bearish_target: int,
    bearish_have: int, neutral_cap: int = 5,
) -> list[tuple[str, str, dict, str, str]]:
    """拉新类别公告并按规则判档，返回 (ticker, category, row, text, band)。

    判档只用标题 + 正文 + 规则表，**不看模型输出**（金标独立性）。

    配额必须在**取正文重判之后**才生效：业绩预告更正公告按标题只能判到
    neutral（正文没到、数字读不到），若先按标题档位卡 neutral 配额，这些
    公告会在取正文前被挡掉——实测绿地/海航的更正公告就这样全丢了。
    """
    import httpx

    from fd_sentiment.sanitize import flatten

    import cninfo_client as c
    import cnreport_tools as t

    picked: list[tuple[str, str, dict, str, str]] = []
    band_counts: Counter[str] = Counter()
    bearish_so_far = bearish_have
    seen: set[str] = set(existing)

    for ticker in tickers:
        co = c.lookup_company(ticker)
        if not co:
            continue
        for category in EXTRA_CATEGORIES:
            try:
                rows = c.query_announcements(
                    co["stock_code"], co["org_id"], category=category, limit=25
                )
            except Exception as exc:
                print(f"  ! {ticker}/{category}: {exc}", file=sys.stderr)
                continue
            for r in rows:
                aid = r["announcement_id"]
                title = r["title"]
                if aid in seen or not r.get("pdf_url"):
                    continue
                if not _is_wanted(category, title):
                    continue
                try:
                    with httpx.Client(timeout=60, follow_redirects=True) as cl:
                        text = flatten(t.extract_pdf_text(cl.get(r["pdf_url"]).content))
                except Exception as exc:
                    print(f"  ! {aid} PDF: {exc}", file=sys.stderr)
                    continue
                if len(text) < 100:
                    continue
                band = g.label(title, category, text)  # 正文到位后判（趋势类需要数字）
                if band is None:
                    continue
                if band == "bearish" and bearish_so_far >= bearish_target:
                    continue
                cap = neutral_cap if band == "neutral" else per_band_cap
                if band_counts[band] >= cap:
                    continue
                seen.add(aid)
                picked.append((ticker, category, r, text, band))
                band_counts[band] += 1
                if band == "bearish":
                    bearish_so_far += 1
                print(f"  + {aid} {band:15s} {ticker} {title[:52]}")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-db", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--target-db", required=True)
    ap.add_argument("--set-version", required=True)
    ap.add_argument("--bearish-target", type=int, default=12)
    ap.add_argument("--per-band-cap", type=int, default=6)
    # 中性对照的独立配额：减持执行回执/更正无数字这类样本只是用来验证
    # 「计划 vs 回执」的区分，5 条足够；放开会把 neutral 撑到 50+ 并稀释信号类。
    ap.add_argument("--neutral-cap", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from fd_sentiment.storage import Store

    base = Store(args.base_db)
    rows = base.list_gold(args.base)
    if not rows:
        print(f"error: {args.base} 在 {args.base_db} 里没有样本", file=sys.stderr)
        return 1

    parsed = {r["doc_id"]: parse_note(r.get("note")) for r in rows}
    base_tickers = sorted({p[0] for p in parsed.values() if p})
    all_tickers = list(dict.fromkeys(base_tickers + list(EXTRA_TICKERS)))
    print(f"基座 {len(rows)} 条 / {len(base_tickers)} 家公司；追加池 {len(all_tickers)} 家")

    titles = fetch_titles(all_tickers, BASE_CATEGORIES + EXTRA_CATEGORIES)

    target = Store(args.target_db)
    before = Counter(r["expected_band"] for r in rows)
    after: Counter[str] = Counter()
    changes: list[tuple[str, str, str, str]] = []
    unresolved: list[tuple[str, str]] = []

    # 1) 基座重标（样本不动）
    for r in rows:
        doc_id = r["doc_id"]
        note = parsed[doc_id]
        if note is None:
            after[r["expected_band"]] += 1
            unresolved.append((doc_id, r.get("note") or ""))
            continue
        ticker, category, title_hint = note
        title = titles.get(doc_id, title_hint)
        band = g.label(title, category, r["text"])
        if band is None:
            after[r["expected_band"]] += 1
            unresolved.append((doc_id, title))
            band = r["expected_band"]
        after[band] += 1
        if band != r["expected_band"]:
            changes.append((doc_id, ticker, r["expected_band"], band, title))
        if not args.dry_run:
            target.upsert_gold(
                set_version=args.set_version, doc_id=doc_id, as_of=r["as_of"],
                expected_band=band, text=r["text"],
                note=f"{ticker}｜{category}｜{title}",
            )

    print(f"\n基座重标变更 {len(changes)} 条：")
    for doc_id, ticker, old, new, title in changes:
        print(f"  {doc_id} {ticker}  {old:15s} → {new:15s}  {title[:50]}")

    # 2) 追加新类别样本
    print("\n追加新类别样本：")
    existing = {r["doc_id"] for r in rows}
    new = collect_new(
        all_tickers, existing, args.per_band_cap, args.bearish_target, after["bearish"],
        neutral_cap=args.neutral_cap,
    )
    for ticker, category, r, text, band in new:
        after[band] += 1
        if not args.dry_run:
            target.upsert_gold(
                set_version=args.set_version, doc_id=r["announcement_id"],
                as_of=r["published"][:10], expected_band=band, text=text,
                note=f"{ticker}｜{category}｜{r['title']}",
            )

    print(f"\n档位分布 前 → 后：")
    for b in ALL_BANDS:
        print(f"  {b:15s} {before[b]:3d} → {after[b]:3d}")
    print(f"  合计 {sum(before.values())} → {sum(after.values())}")

    if unresolved:
        print(f"\n未判定（保留原档）{len(unresolved)} 条：", file=sys.stderr)
        for doc_id, title in unresolved:
            print(f"  {doc_id} {title[:60]}", file=sys.stderr)

    missing = [b for b in ALL_BANDS if after[b] == 0]
    if missing:
        print(f"\n✗ 缺档：{'、'.join(missing)}", file=sys.stderr)
        return 1
    thin = [b for b in ALL_BANDS if after[b] < g.WEAK_BAND_MIN]
    if thin:
        print(f"\n⚠ 低于 {g.WEAK_BAND_MIN} 条：{'、'.join(thin)}", file=sys.stderr)

    if args.dry_run:
        print("\n(dry-run：未写库)")
    else:
        print(f"\n✓ 已写入 {args.target_db} 的 {args.set_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
