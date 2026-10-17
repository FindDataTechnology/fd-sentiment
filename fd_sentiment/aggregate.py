"""聚合视图：分布统计，明确不是情绪指数。

设计 D5：情绪指数需要定义（加权/样本/中性基线）与验证（跟什么相关），
未经定义就给"情绪指数=0.32"是用精确形式包装未验证的构造。这里只给事实：
各 band 计数、均值分、均值置信度，外加**覆盖说明**（已打分 vs 范围内总数）。
字段名与描述里不出现"指数"类用法（4.2 的验收点）。
"""
from __future__ import annotations

from typing import Any

from .storage import Store

VIEW_TITLE = "已打分文档的情绪分布统计（非市场情绪指数、非行情指标）"


def aggregate(
    store: Store,
    *,
    since: str | None = None,
    until: str | None = None,
    doc_type: str | None = None,
    prompt_version: str | None = None,
    universe_total: int | None = None,
) -> dict[str, Any]:
    """按时间/类型聚合。`universe_total` 给出时回报覆盖率（已打分/全量）。"""
    rows = store.list_scores(since=since, until=until, prompt_version=prompt_version, limit=100_000)
    if doc_type:
        rows = [r for r in rows if r.get("doc_type") == doc_type]

    bands = ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]
    band_counts = {b: 0 for b in bands}
    for r in rows:
        band_counts[r["band"]] = band_counts.get(r["band"], 0) + 1

    n = len(rows)
    mean_score = round(sum(r["score"] for r in rows) / n, 6) if n else None
    mean_conf = round(sum(r["confidence"] for r in rows) / n, 6) if n else None

    out: dict[str, Any] = {
        "title": VIEW_TITLE,
        "window": {"since": since, "until": until},
        "doc_type": doc_type,
        "prompt_version": prompt_version,
        "scored_count": n,
        "band_counts": band_counts,
        "mean_score": mean_score,
        "mean_confidence": mean_conf,
        "coverage": {
            "scored": n,
            "universe_total": universe_total,
            "ratio": round(n / universe_total, 4) if universe_total else None,
            "note": "覆盖说明：已打分数与统计范围内文档总数；universe_total 未知时为 null，"
                    "不推断、不补算。",
        },
        "caveat": "本视图是已打分文档的分布统计，不是情绪指数，不用于行情或交易判断。",
    }
    return out
