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

# 弱档最小可信样本量：低于此值的档位在报告中标为「样本不足」。
WEAK_BAND_MIN = 10

# 置信桶最小样本量：低于此值的桶不参与校准判定，但单列不丢弃。
WEAK_BUCKET_MIN = 3

# 高置信阈值：成因分类时，只看 >= 此值的样本（它们才是"自信却判错"的候选）。
CALIB_HIGH_CONF = 0.85

# 五档的序数位置与符号，用于 ±1 档与方向两个宽口径。
_BAND_ORDINAL = {
    "strong_bearish": -2, "bearish": -1, "neutral": 0, "bullish": 1, "strong_bullish": 2,
}
_BAND_SIGN = {
    "strong_bearish": -1, "bearish": -1, "neutral": 0, "bullish": 1, "strong_bullish": 1,
}


def _calipers(per_item: list[dict[str, Any]], uncovered: list[str] | None = None) -> dict[str, Any]:
    """三个口径并列：exact / 方向 / ±1 档，并附各口径的失真方向说明。

    单一口径会误导下游。三个口径各有系统性失真：

    - `exact` 在偏离集中于相邻档时**偏低**（强度判错就算全错）。
    - `±1 档` 在相邻档误判时**偏高**；更严重的是，当金标存在未覆盖档位时，
      它会把「模型判未覆盖档、金标判相邻档」计为命中——而 exact 那边已经把这些
      样本单列了。两个口径对同一批样本的处理不一致，±1 档会系统性偏高，
      所以它不能单独作为质量证据。
    - 方向口径受档位覆盖缺口影响最小（三分类与金标对齐），是当前首选。
    """
    n = len(per_item)
    if not n:
        return {"samples": 0, "exact": 0.0, "direction": 0.0, "adjacent": 0.0,
                "exact_biased_direction": "low",
                "adjacent_biased_direction": "high" if uncovered else "neutral",
                "exact_note": "无样本",
                "adjacent_note": "无样本",
                "direction_note": "无样本"}
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
    has_gap = bool(uncovered)
    adjacent_note = (
        f"受覆盖缺口影响（金标缺 {'、'.join(uncovered)}）：模型判这些档位的样本会被"
        "算作相邻命中，该口径系统性偏高，不可单独作为质量证据"
        if has_gap
        else "相邻档误判计为命中、跨两级才算错，故对「强度判过头」偏宽松"
    )
    return {
        "samples": n,
        "exact": round(exact / n, 4),
        "direction": round(direction / n, 4),
        "adjacent": round(adjacent / n, 4),
        "exact_biased_direction": "low",
        "adjacent_biased_direction": "high" if has_gap else "neutral",
        "exact_note": "强度档判错即算全错，故对「方向对、强度偏」的情形偏低",
        "adjacent_note": adjacent_note,
        "direction_note": "三分类与金标对齐，受档位覆盖缺口影响最小，当前首选口径",
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

    # 金标覆盖度：缺档会让模型输出该档时被记成「预测错误」——那是评测表的洞，
    # 不是模型的错。gold-v6 缺 strong_* 两档，实测把 exact 压低约 23 个百分点。
    coverage = store.gold_coverage(set_version)
    uncovered = [b for b, n in coverage.items() if n == 0]

    per_item: list[dict[str, Any]] = []
    coverage_gap_items: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for res in scored_results:
        if res["error"] is not None:
            failures.append({"doc_id": res["doc_id"], "error": res["error"]})
            continue
        item = {
            "doc_id": res["doc_id"],
            "expected_band": res["expected_band"],
            "predicted_band": res["rec"]["band"],
            "score": res["rec"]["score"],
            "confidence": res["rec"]["confidence"],
            "hit": res["hit"],
            "bucket": _bucket(res["rec"]["confidence"]),
        }
        # 模型判了一个金标里没有的档 -> 单列，不算「预测错误」
        if res["rec"]["band"] in uncovered:
            item["uncovered_band"] = True
            coverage_gap_items.append(item)
        else:
            per_item.append(item)

    scored = len(per_item)
    hits = sum(1 for i in per_item if i["hit"])

    hit_rate = round(hits / scored, 4) if scored else 0.0

    # 分层表：按「金标期望档位」分组给准确率。
    # 金标按真实公告类型分布构建（不人为均衡），故某些档位样本天然稀少；
    # 单看总命中率会被最大的一组主导（gold-v7 里 strong_bullish 占 53%），
    # 必须分层才能看出哪些档真的测到了、哪些档的结论无统计意义。
    by_band: dict[str, dict[str, Any]] = {}
    for it in per_item:
        b = by_band.setdefault(it["expected_band"], {"samples": 0, "hits": 0})
        b["samples"] += 1
        b["hits"] += 1 if it["hit"] else 0
    band_rows = []
    for b in sorted(by_band, key=lambda x: _BAND_ORDINAL.get(x, 99)):
        v = by_band[b]
        band_rows.append({
            "expected_band": b,
            "samples": v["samples"],
            "hits": v["hits"],
            "accuracy": round(v["hits"] / v["samples"], 4) if v["samples"] else None,
            "enough_samples": v["samples"] >= WEAK_BAND_MIN,
        })
    thin_bands = [r["expected_band"] for r in band_rows if not r["enough_samples"]]

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

    # 校准判定：桶数 >=3 时检查**整体形态**，而非只比两个端点。
    # 端点比较在非单调形状上会给出误导性结论——实测 v4 的三桶是
    # 0.70 / 0.41 / 0.47，最低点在中间，但端点比较只说"高置信不如低置信"。
    populated = [r for r in calib_rows if r["samples"] >= WEAK_BUCKET_MIN and r["accuracy"] is not None]
    excluded_buckets = [
        {"bucket": r["bucket"], "samples": r["samples"], "accuracy": r["accuracy"]}
        for r in calib_rows
        if r["samples"] < WEAK_BUCKET_MIN and r["samples"] > 0
    ]
    calibration_ok = True
    shape = "insufficient"
    lowest_bucket = None
    calibration_note = f"样本不足以判定校准（每个置信桶需 ≥{WEAK_BUCKET_MIN} 条）"
    if len(populated) >= 2:
        accs = [r["accuracy"] for r in populated]
        if len(populated) == 2:
            # 两桶时端点比较即完整信息
            calibration_ok = accs[-1] >= accs[0]
            shape = "monotonic_increasing" if calibration_ok else "monotonic_decreasing"
        else:
            up = all(b >= a for a, b in zip(accs, accs[1:]))
            down = all(b <= a for a, b in zip(accs, accs[1:]))
            if up:
                calibration_ok, shape = True, "monotonic_increasing"
            elif down:
                calibration_ok, shape = False, "monotonic_decreasing"
                lowest_bucket = populated[-1]["bucket"]
            else:
                # 非单调：指出最低点在哪桶，不能笼统说"高置信不如低置信"
                calibration_ok = accs[-1] >= accs[0]
                shape = "non_monotonic"
                lowest_bucket = min(populated, key=lambda r: r["accuracy"])["bucket"]

        detail = "、".join(f"{r['bucket']}={r['accuracy']}" for r in populated)
        if shape == "monotonic_increasing":
            calibration_note = f"校准正常（准确率随置信度单调递增）：{detail}"
        elif shape == "monotonic_decreasing":
            calibration_note = f"**校准失败**（准确率随置信度单调递减）：{detail}"
        else:
            calibration_note = (
                f"**非单调**：{detail}；最低点在 {lowest_bucket} 桶。"
                + ("端点上看高置信仍优于低置信，但中间桶塌陷需单独审视。"
                   if calibration_ok else "且端点比较亦为高置信不如低置信。")
            )

    # 成因分类：校准失败的高置信样本，究竟是"落在金标未覆盖档"（评测表缺档）
    # 还是"在两个都有覆盖的档之间判错"（模型标定失准）。两者处置完全不同：
    # 前者要补金标，后者才要改 prompt 的 confidence 语义。
    # gold-v6 缺 strong_* 两档，模型判强档全被记为错、而模型对强档最自信，
    # 于是"校准失败"其实是覆盖缺口的伪装。
    cause: dict[str, Any] = {"kind": "unknown", "coverage_gap": 0, "miscalibration": 0,
                             "note": "金标无覆盖信息，无法区分成因"}
    all_scored = per_item + coverage_gap_items
    if coverage:
        high_conf = [i for i in all_scored if i["confidence"] >= CALIB_HIGH_CONF]
        gap = sum(1 for i in high_conf if i.get("uncovered_band"))
        mis = sum(1 for i in high_conf if not i["hit"] and not i.get("uncovered_band"))
        if not high_conf:
            cause = {"kind": "no_high_conf", "coverage_gap": 0, "miscalibration": 0,
                     "note": "没有高置信样本，无从归因"}
        elif mis == 0 and gap > 0:
            cause = {"kind": "coverage_gap", "coverage_gap": gap, "miscalibration": 0,
                     "note": f"高置信误判 {gap} 条全部落在金标未覆盖档位上——"
                             f"这是评测表缺档，不是模型标定失准。应补金标而非调 prompt"}
        elif gap == 0 and mis > 0:
            cause = {"kind": "miscalibration", "coverage_gap": 0, "miscalibration": mis,
                     "note": f"高置信误判 {mis} 条全部落在金标已覆盖档位之间——"
                             f"这是模型对自身确定度的估计失准"}
        elif gap > 0 and mis > 0:
            total = gap + mis
            cause = {"kind": "both", "coverage_gap": gap, "miscalibration": mis,
                     "note": f"高置信误判 {total} 条：{gap} 条来自覆盖缺口"
                             f"（{gap/total:.0%}）、{mis} 条来自标定失准（{mis/total:.0%}）"}
        else:
            cause = {"kind": "none", "coverage_gap": 0, "miscalibration": 0,
                     "note": "高置信样本无判错，无需归因"}

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
        "coverage": {
            "bands": coverage,
            "uncovered": uncovered,
            "note": (
                "金标五档齐全"
                if not uncovered
                else f"金标未覆盖 {'、'.join(uncovered)}：模型输出这些档位的样本已单列，"
                     "不计入 exact 分母（这些不是预测错误，而是评测表缺档）"
            ),
            "excluded_from_exact": len(coverage_gap_items),
        },
        "coverage_gap_items": coverage_gap_items,
        "by_band": {
            "rows": band_rows,
            "insufficient": thin_bands,
            "note": (
                f"以下档位样本 <{WEAK_BAND_MIN} 条，其准确率不具统计意义：{'、'.join(thin_bands)}"
                if thin_bands else "各档样本量均达到最低可信门槛"
            ),
        },
        "calipers": _calipers(per_item, uncovered),
        "calibration": {
            "ok": calibration_ok,
            "shape": shape,
            "lowest_bucket": lowest_bucket,
            "note": calibration_note,
            "buckets": calib_rows,
            "excluded_buckets": excluded_buckets,
            "cause": cause,
        },
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
