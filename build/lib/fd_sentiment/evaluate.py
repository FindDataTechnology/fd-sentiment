"""金标评测与校准。

纪律（spec D6）：只有准确率时，"全猜中性"在均衡金标上也有 ~33%；更糟的是
高置信度若不可信，下游 pack 会按置信度加权。所以评测必须同时给：
  ① 命中率（band 一致率）；
  ② 按置信度分桶的准确率 —— 高置信桶低于低置信桶即**校准失败**，显式标注。

金标集按 set_version 记录；新增标注不破坏既有报告可比性（旧报告仍指向旧版本）。
"""
from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any

from .llm import LLMClient
from .models import ScoreError
from .prompt import PromptArtifact, load_prompt
from .scoring import ScoreRequest, score_document
from .storage import Store

CONF_BUCKETS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0001))

# 五档的序数位置与符号，用于 ±1 档与方向两个宽口径。
_BAND_ORDINAL = {
    "strong_bearish": -2, "bearish": -1, "neutral": 0, "bullish": 1, "strong_bullish": 2,
}
_BAND_SIGN = {
    "strong_bearish": -1, "bearish": -1, "neutral": 0, "bullish": 1, "strong_bullish": 1,
}


def _calipers(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    """三个口径并列：exact / 方向 / ±1 档。

    单一口径会误导下游——实测 gold-v6（prompt v3，54 条）上 exact 40.7%、
    ±1 档 98.1%、方向 79.6%。exact 低估（偏离几乎全在相邻档）、±1 档高估
    （它把"强度判错"也算对）。所以报告必须三个一起给，让读者自己选口径。
    """
    n = len(per_item)
    if not n:
        return {"samples": 0, "exact": 0.0, "direction": 0.0, "adjacent": 0.0}
    exact = sum(1 for i in per_item if i["hit"])
    direction = sum(
        1 for i in per_item
        if _BAND_SIGN.get(i["predicted_band"]) == _BAND_SIGN.get(i["expected_band"])
    )
    adjacent = sum(
        1 for i in per_item
        if i["predicted_band"] in _BAND_ORDINAL and i["expected_band"] in _BAND_ORDINAL
        and abs(_BAND_ORDINAL[i["predicted_band"]] - _BAND_ORDINAL[i["expected_band"]]) <= 1
    )
    return {
        "samples": n,
        "exact": round(exact / n, 4),
        "direction": round(direction / n, 4),
        "adjacent": round(adjacent / n, 4),
    }


def _bucket(conf: float) -> str:
    for lo, hi in CONF_BUCKETS:
        if lo <= conf < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "unknown"


def run_eval(
    client: LLMClient,
    store: Store,
    *,
    set_version: str,
    prompt: PromptArtifact | None = None,
) -> dict[str, Any]:
    """对金标集跑一次评测并落库。返回报告 dict。"""
    prompt = prompt or load_prompt("v1")
    gold = store.list_gold(set_version)
    if not gold:
        raise ValueError(f"金标集 {set_version!r} 为空或不存在")

    per_item: list[dict[str, Any]] = []
    hits = 0
    scored = 0
    failures: list[dict[str, str]] = []

    for g in gold:
        try:
            rec = score_document(
                client,
                ScoreRequest(doc_id=g["doc_id"], text=g["text"], as_of=g["as_of"]),
                prompt=prompt,
                store=store,
            )
        except ScoreError as exc:
            failures.append({"doc_id": g["doc_id"], "error": f"拒绝：{exc}"})
            continue
        except Exception as exc:  # 模型错误也记下，不让一次失败炸掉整轮评测
            failures.append({"doc_id": g["doc_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        scored += 1
        hit = rec["band"] == g["expected_band"]
        hits += 1 if hit else 0
        per_item.append({
            "doc_id": g["doc_id"],
            "expected_band": g["expected_band"],
            "predicted_band": rec["band"],
            "score": rec["score"],
            "confidence": rec["confidence"],
            "hit": hit,
            "bucket": _bucket(rec["confidence"]),
        })

    hit_rate = round(hits / scored, 4) if scored else 0.0

    # 校准表：每个置信桶的样本数与准确率
    buckets: dict[str, dict[str, Any]] = {}
    for it in per_item:
        b = buckets.setdefault(it["bucket"], {"samples": 0, "hits": 0})
        b["samples"] += 1
        b["hits"] += 1 if it["hit"] else 0
    calib_rows = []
    for b in sorted(buckets):
        v = buckets[b]
        calib_rows.append({
            "bucket": b,
            "samples": v["samples"],
            "hits": v["hits"],
            "accuracy": round(v["hits"] / v["samples"], 4) if v["samples"] else None,
        })

    # 校准失败判定：最高有样本桶的准确率 < 最低有样本桶的准确率
    populated = [r for r in calib_rows if r["samples"] >= 3 and r["accuracy"] is not None]
    calibration_ok = True
    calibration_note = "样本不足以判定校准（每个置信桶需 ≥3 条）"
    if len(populated) >= 2:
        lowest = populated[0]["accuracy"]
        highest = populated[-1]["accuracy"]
        calibration_ok = highest >= lowest
        calibration_note = (
            f"低置信桶 {populated[0]['bucket']} 准确率 {lowest}，高置信桶 "
            f"{populated[-1]['bucket']} 准确率 {highest} → "
            + ("校准正常（高置信更准）" if calibration_ok else "**校准失败**（高置信桶不如低置信桶）")
        )

    report = {
        "prompt_version": prompt.version,
        "prompt_digest": prompt.digest,
        "model": client.cfg.model,
        "set_version": set_version,
        "samples_expected": len(gold),
        "samples_scored": scored,
        "hits": hits,
        "hit_rate": hit_rate,
        "calipers": _calipers(per_item),
        "calibration": {"ok": calibration_ok, "note": calibration_note, "buckets": calib_rows},
        "failures": failures,
        "per_item": per_item,
    }

    run_id = "eval-" + uuid.uuid4().hex[:12]
    store.insert_eval_run({
        "run_id": run_id,
        "prompt_version": prompt.version,
        "model": client.cfg.model,
        "set_version": set_version,
        "samples": scored,
        "hits": hits,
        "hit_rate": hit_rate,
        "calibration_ok": calibration_ok,
        "report": report,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    })
    report["run_id"] = run_id
    return report


def compare_runs(store: Store, run_ids: list[str]) -> dict[str, Any]:
    """两个（或更多）评测结果并列对比 —— 版本对比的验收点（5.4）。"""
    rows = []
    for rid in run_ids:
        r = store.get_eval_run(rid)
        if not r:
            rows.append({"run_id": rid, "error": "不存在"})
            continue
        rows.append({
            "run_id": rid,
            "prompt_version": r["prompt_version"],
            "model": r["model"],
            "set_version": r["set_version"],
            "samples": r["samples"],
            "hit_rate": r["hit_rate"],
            "calibration_ok": r["calibration_ok"],
            "calibration_note": r["report"]["calibration"]["note"],
        })
    return {"runs": rows}
