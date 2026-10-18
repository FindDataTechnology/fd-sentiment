#!/usr/bin/env python3
"""金标集构建：≥50 条已标注公告，五档齐全。

标注原则（写在这里，避免"标注者口径随心情变"）：
  - 以**公告对相关主体经营/财务的影响方向**为准，不以市场反应为准。
  - 强度档按**正文量化锚点**判定，与打分 prompt 的强档门槛使用同一套判据：
    正文载有净利/营收同比 ≥50% 的明确数字，或出现「大幅上升/大幅增长/
    显著增长/大幅下降/大幅亏损」等强度词，且理由指向经营实质
    （销量/价格/毛利率/订单/产能/行业地位）→ 强档。
  - 一次性损益（政府补贴、资产处置、公允价值变动、税收返还）驱动的增长
    **不进强档**，最高到偏档。
  - 纯事务性（会议通知、章程修订、分红**实施**公告）或正负相抵 → neutral。
  - **不看标题判强度**：标题是否有"大幅"字样不构成强度依据。实测
    gold-v6 的 13 条误判全部源于此——标题无强度词但正文写着 +50%~+500%，
    金标按标题判 bullish，打分器按正文判 strong_bullish，两边各按自己的
    规则做对了，却在同一张评测表上对撞。

用法：
    python3 scripts/build_gold_set.py --set-version gold-v1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fd-cn-report"))

# 标注规则表：按 (类别, 标题关键词) → band。这是**人工标注的显式规则**，
# 不是模型输出——金标的独立性靠这条保证（打分器不能自己给自己出题）。
#
# 每条规则后标注其依据 prompt 的哪一条（sentiment-gold-coverage：标注依据
# 与判据一致性）。档位判定不得没有出处：
#   [强档门槛] bands.md 第一档：方向明确 + 量化锚点（≥50% 或强度词）+ 经营实质
#   [中性定义] bands.md neutral 条：纯事务性 / 正负相抵
#   [拒答边界] system.md「何时拒答」：只对读不出内容者拒答（不属于标注范畴）
RULES: list[tuple[str, tuple[str, ...], str]] = [
    # [强档门槛] 标题命中"预增/预盈"等只是方向；强度再交给 _strengthen 按正文判
    ("业绩预告", ("预增", "预盈", "扭亏", "大幅上升", "同向上升", "增长"), "bullish"),
    ("业绩预告", ("预减", "预亏", "首亏", "续亏", "大幅下降", "同向下降"), "bearish"),
    # 风险提示类：几乎全是负面（诉讼/处罚/退市风险/资金占用），用来补负面档
    ("风险提示", ("退市", "暂停上市", "重大违法", "资金占用", "违规担保", "立案"), "strong_bearish"),
    ("风险提示", ("诉讼", "处罚", "警示", "问询"), "bearish"),
    # [中性定义] "股票交易异常波动"本身只是披露义务触发，方向取决于同期消息面 →
    # 归中性（不能把披露义务当利空，这是标注口径问题，不是模型问题）。
    ("风险提示", ("异常波动", "风险提示"), "neutral"),
    # [中性定义] 权益分派：**方案公告带方向信息**（首次宣布或上调分红金额是
    # 回报政策信号），**实施公告是事务性执行通知**（按既定方案发钱）。
    # 2026-10-19 裁决：口径分歧已由用户拍板——方案判 bullish，并已同步
    # prompt v5 的 bands.md（v4 及以前把两者都归 neutral，导致 8 条对不上）。
    # v2 评测实证补充：把"实施公告"标成 bullish 是标注错误（11 条全被模型
    # 正确判成 neutral），模型对、金标错。
    ("权益分派", ("方案", "调整"), "bullish"),
    ("权益分派", ("实施",), "neutral"),
    # [中性定义] 纯事务性公告 → 中性（补 neutral 档）
    ("股东大会", (), "neutral"),
    ("董事会", (), "neutral"),
    ("监事会", (), "neutral"),
    ("公司治理", (), "neutral"),
]


def rule_sources() -> list[tuple[str, str, str]]:
    """规则 → 依据的对照表，供审计与测试核对（每条规则都有出处）。"""
    src = "[强档门槛]"
    neu = "[中性定义]"
    return [
        ("业绩预告/预增类", "bullish→_strengthen", src),
        ("业绩预告/预减类", "bearish→_strengthen", src),
        ("风险提示/退市类", "strong_bearish", src),
        ("风险提示/诉讼类", "bearish", neu),
        ("风险提示/异常波动", "neutral", neu),
        ("权益分派/方案", "bullish", neu),
        ("权益分派/实施", "neutral", neu),
        ("股东大会", "neutral", neu),
        ("董事会", "neutral", neu),
        ("监事会", "neutral", neu),
        ("公司治理", "neutral", neu),
    ]

STRONG_WORDS = ("大幅", "显著", "重大", "翻倍", "超过", "激增")
NEUTRAL_TITLE = ("通知", "章程", "制度", "议事规则", "管理办法", "股东大会", "董事会决议")

# 一次性损益：由这些来源驱动的增长不构成"经营实质改善"，不进强档。
# 与 prompt v5 的强档第 ③ 条一致（理由须指向经营实质）。
ONE_OFF_SOURCES = ("政府补助", "政府补贴", "资产处置", "处置收益", "公允价值变动",
                   "税收返还", "债务重组", "投资收益", "非经常性损益")
# 经营实质来源：出现这些词说明增长来自主业。
OPERATIONAL_SOURCES = ("销量", "销售", "价格", "毛利率", "订单", "产能", "产量",
                       "营业收入", "主营业务", "市场份额", "行业地位", "成本")

# 同比增幅的量化门槛，与 prompt v5 bands.md 的 50% 锚点一致。
STRONG_PCT_THRESHOLD = 50.0


def _max_yoy_pct(text: str) -> float | None:
    """从正文里取最大同比增幅（百分数）。用于判定是否达到强档的量化锚点。

    只认形如「增长 62%」「增加 110% 到 130%」「下降 84.88%」的明确数字；
    不认无方向修饰的裸百分比（那可能是持股比例、分红率等）。
    """
    if not text:
        return None
    pat = re.compile(
        r"(?:增长|增加|上升|提高|下滑|下降|减少|亏损)[^0-9%]{0,12}"
        r"(\d{1,3}(?:\.\d+)?)\s*%"
    )
    vals = []
    for m in pat.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        # 500% 以上多半是笔误或非年度口径，不采信为强档依据
        if 0 < v <= 500:
            vals.append(v)
    return max(vals) if vals else None


def _has_operational_basis(text: str) -> bool:
    """强档第 ③ 条：理由指向经营实质，且不能只来自一次性损益。"""
    if not text:
        return False
    has_op = any(s in text for s in OPERATIONAL_SOURCES)
    one_off_hits = sum(1 for s in ONE_OFF_SOURCES if s in text)
    # 只出现一次性损益来源、没有经营实质词 → 不满足 ③
    return has_op or one_off_hits == 0


def label(title: str, category: str, text: str = "") -> str | None:
    """按标题 + 正文共同判定档位。

    text 为空时退化为标题判据（向后兼容旧调用），但**正常构建必须传正文**：
    强度档靠正文的量化锚点判，标题只用于识别公告类型与排除事务性公告。
    """
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
                if band not in ("bullish", "bearish"):
                    return band
                return _strengthen(band, title, text)
    return None


def _strengthen(band: str, title: str, text: str) -> str:
    """偏档 → 强档的三条件判定（与 prompt v5 的强档门槛同源）。

    ① 方向明确：由调用方已保证（进了 bullish/bearish 分支）
    ② 命中量化锚点：正文明示同比 ≥50%，或标题/正文出现强度词
    ③ 理由指向经营实质：非一次性损益驱动
    """
    pct = _max_yoy_pct(text)
    anchor_hit = (pct is not None and pct >= STRONG_PCT_THRESHOLD) or any(
        w in title or w in text[:400] for w in STRONG_WORDS
    )
    if not anchor_hit:
        return band
    if not _has_operational_basis(text):
        return band
    return "strong_" + band


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
    # 扩充到 20 家：五档齐全需要足够的强档样本，而单家公司的预增/预亏公告
    # 有限（gold-v6 的 12 家只凑出三档）。前 8 家蓝筹给正面/中性，后 12 家
    # 含周期股与已知预亏公司给负面与强负面。
    tickers = [
        "600519", "000333", "000001", "600036", "000858", "601318", "600030", "601899",
        "601111", "600029", "000002", "600048", "601088", "600188", "000898", "600019",
        "601398", "600028", "000063", "002415",
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
                rows = c.query_announcements(co["stock_code"], co["org_id"], category=category, limit=25)
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
            # 正文必须先取：强度档靠正文的量化锚点判定（sentiment-gold-coverage:
            # 标注依据与判据一致性）。先判档再取正文会让标注意义上"看不见数字"，
            # 这正是 gold-v6 那 13 条误判的成因。
            try:
                with httpx.Client(timeout=60, follow_redirects=True) as cl:
                    text = flatten(t.extract_pdf_text(cl.get(r["pdf_url"]).content))
            except Exception as exc:
                print(f"  ! {aid} PDF: {exc}")
                continue
            if len(text) < 100:
                continue
            band = label(r["title"], category, text)
            if band is None:
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

    # 覆盖度检查：五档必须齐全，缺档就失败（sentiment-gold-coverage）。
    # 金标缺档会让模型输出该档时被记成"错误"，从而系统性低估打分器——
    # gold-v6 缺 strong_* 整整两档，导致 exact 被压低 23 个百分点。
    ALL_BANDS = ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish")
    missing = [b for b in ALL_BANDS if counts.get(b, 0) == 0]

    print(f"\n金标集 {args.set_version}：{added} 条，分布 {counts}")
    print("五档覆盖：" + " ｜ ".join(f"{b}={counts.get(b, 0)}" for b in ALL_BANDS))
    fail = False
    if missing:
        print(f"✗ 档位覆盖不全，缺：{'、'.join(missing)}", file=sys.stderr)
        print("  缺档会让模型输出该档时被记成错误（gold-v6 即因此把 exact 压低 23 个百分点）",
              file=sys.stderr)
        fail = True
    if added < 50:
        print(f"✗ 样本量不足 50（实为 {added}）", file=sys.stderr)
        fail = True
    if fail:
        return 1
    print("✓ 满足 ≥50 条且五档齐全")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
