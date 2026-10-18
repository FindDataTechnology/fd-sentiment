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
from concurrent.futures import ThreadPoolExecutor
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


def _score_one(client: LLMClient, store: Store, g: dict[str, Any], prompt: PromptArtifact) -> dict[str, Any]:
    """打一条金标。异常在此收敛成 error 字段，不让单条失败炸掉整轮。"""
    try:
        rec = score_document(
            client,
            ScoreRequest(doc_id=g["doc_id"], text=g["text"], as_of=g["as_of"]),
            prompt=prompt,
            store=store,
        )
    except ScoreError as exc:
        return {"doc_id": g["doc_id"], "expected_band": g["expected_band"], "rec": None,
                "hit": False, "error": f"拒绝：{exc}"}
    except Exception as exc:
        return {"doc_id": g["doc_id"], "expected_band": g["expected_band"], "rec": None,
                "hit": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"doc_id": g["doc_id"], "expected_band": g["expected_band"], "rec": rec,
            "hit": rec["band"] == g["expected_band"], "error": None}


def _score_gold(
    client: LLMClient, store: Store, gold: list[dict[str, Any]], prompt: PromptArtifact, concurrency: int
) -> list[dict[str, Any]]:
    """按 concurrency 打分。结果按 gold 原序返回，报告与串行版逐字段可比。

    Store 每次操作独立连接、写入加锁，LLMClient 无可变共享状态，故多线程安全。
    """
    if concurrency <= 1 or len(gold) <= 1:
        return [_score_one(client, store, g, prompt) for g in gold]
    workers = min(concurrency, len(gold))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda g: _score_one(client, store, g, prompt), gold))


def run_eval(
    client: LLMClient,
    store: Store,
    *,
    set_version: str,
    prompt: PromptArtifact | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """对金标集跑一次评测并落库。返回报告 dict。

    `concurrency` 默认 1（串行），因为并发会改变失败率的构成：并发下网关
    更容易超时，而超时样本被计入 failures 而非分母——失败样本的多少会
    间接影响命中率。故并发是**显式选择**而非默认：想要快就明说，并知道
    自己拿到的是一份可能与串行不同的报告。

    实测（gold-v6 60 条，prompt v3，deepseek-v4.1-flash）：串行 920s；
    并发 4 路时约 250-350s，且失败样本集合与串行不完全相同（超时增多，
    但 reasoning 泄漏不增多——后者由 max_tokens 决定）。
    """
    prompt = prompt or load_prompt("v1")
    gold = store.list_gold(set_version)
    if not gold:
        raise ValueError(f"金标集 {set_version!r} 为空或不存在")

    scored_results = _score_gold(client, store, gold, prompt, concurrency)

    per_item: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for res in scored_results:
        if res["error"] is not None:
            failures.append({"doc_id": res["doc_id"], "error": res["error"]})
            continue
        per_item.append({
            "doc_id": res["doc_id"],
            "expected_band": res["expected_band"],
            "predicted_band": res["rec"]["band"],
            "score": res["rec"]["score"],
            "confidence": res["rec"]["confidence"],
            "hit": res["hit"],
            "bucket": _bucket(res["rec"]["confidence"]),
        })

    scored = len(per_item)
    hits = sum(1 for i in per_item if i["hit"])

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
        "concurrency": concurrency,
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
