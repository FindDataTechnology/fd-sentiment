"""单文档打分：LLM 调用 + evidence_spans 强制校验 + 无证据拒绝 + 分块合并。

关键纪律（spec D2/D4/D7）：
  - evidence_spans 非空是合法分数的必要条件；模型拒答或证据不成立 → 抛 ScoreError，
    **不返回无证据的分数**。
  - quote 必须是原文逐字子串：先按模型给的偏移切片比对，对不上则在原文里定位
    （空白归一化）并**回写修正偏移**；两路都失败 → 判定编造引用，拒绝。
  - 打分入口不接受市场数据/额外交互参数（point-in-time 由签名保证）。
  - 内容先 flatten + 包裹（防注入）。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, LLMError, extract_json_block
from .models import (
    BANDS, EvidenceSpan, SentimentScore, ScoreError, band_for, locate_quote, validate_score,
)
from .prompt import PromptArtifact, load_prompt
from .sanitize import flatten, wrap_document

# 分块阈值（字符）：推理型模型的输出预算与**输入长度**间接相关（推理量随文本
# 变长而膨胀），实测 24k 字符块在 16k max_tokens 下常被截断。8k 字符块是
# "足够完整表达一份公告、又留得下推理与 JSON"的平衡点，可由环境变量覆盖。
CHUNK_CHAR_LIMIT = int(os.getenv("FD_SENTIMENT_CHUNK_CHARS", "8000"))
# 单块最小字符：太碎的尾块不单独打（并进前一块），避免无意义调用
MIN_CHUNK_CHARS = 800


@dataclass
class ScoreRequest:
    doc_id: str
    text: str
    as_of: str
    doc_type: str | None = None


def content_hash(text: str) -> str:
    """归一化后再哈希：空白差异（换行/多空格）不应触发重打。

    注意与 `flatten` 的区别：flatten 保留段落结构（分块要用），哈希则把所有
    空白压成单空格——"内容变了"应指词句变了，不是排版变了。
    """
    normalized = re.sub(r"\s+", " ", flatten(text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _chunk_text(text: str, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """按段落边界切块；单段超限时按句子再切。保证每块 <= limit（尽量）。"""
    if len(text) <= limit:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > limit:
            # 超长段按句号切
            sentences = re.split(r"(?<=[。！？；.!?;])", p)
            for s in sentences:
                if len(buf) + len(s) + 2 > limit and buf:
                    chunks.append(buf.strip())
                    buf = ""
                buf += s
                if len(buf) >= limit:
                    chunks.append(buf.strip())
                    buf = ""
            continue
        if len(buf) + len(p) + 2 > limit and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += p + "\n\n"
    if buf.strip():
        chunks.append(buf.strip())
    # 尾块过小则并入前一块
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()
    return chunks


def _score_chunk(
    client: LLMClient, prompt: PromptArtifact, chunk: str, *, doc_id: str, as_of: str,
    doc_type: str | None, chunk_index: int, chunk_total: int,
    original_text: str | None = None,
) -> SentimentScore:
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user_template.format(wrapped=wrap_document(chunk))},
    ]
    raw = client.chat(messages, doc_id=doc_id)
    data = extract_json_block(raw)
    # 模型实际看到的材料文本（含前置说明与分隔符）——它引用的 quote 与
    # 偏移都基于这份文本。校验必须对齐"模型看到的"，否则合法引文会被
    # 误判成编造（真实公告实测：模型引对了，我们查错了基准）。
    seen_text = wrap_document(chunk)

    if data.get("refuse"):
        raise ScoreError(f"模型拒绝打分（无可用情绪证据）：{data.get('reason', '未说明')}")

    try:
        score_val = float(data["score"])
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreError(f"模型输出的 score/confidence 不合法：{data!r}") from exc

    # 强档自洽性核查（v3 引入）：模型在 strong_tier_check 里逐条申报三个条件，
    # 服务据此核对分数。若它自称"未达强档"却给出 |score| >= 0.6，等于自相矛盾——
    # 这是 v2 的失败模式（口头认同取温和档、手上打 0.8 分被映射回强档）。
    # 处置：**拒绝该次打分并要求重打**，不静默把分数压回去（压回去等于替模型
    # 做判断，且掩盖了 prompt 未被执行的事实）。
    stc = data.get("strong_tier_check")
    if isinstance(stc, dict):
        tier_ok = all(
            bool(stc.get(k)) for k in ("direction_clear", "quant_anchor_hit", "reason_operational")
        )
        claimed = str(stc.get("summary", "")).strip()
        if not tier_ok and abs(score_val) >= 0.6:
            raise ScoreError(
                "强档自洽性核查失败：模型申报未满足强档三条件"
                f"（summary: {claimed[:80]!r}）却给出 score={score_val}（|score|>=0.6 即宣称进强档）。"
                "按 v3 契约要求重打：要么补足量化锚点，要么把分数落到 (-0.6, 0.6)。"
            )
        if tier_ok and abs(score_val) < 0.6 and claimed:
            # 自称够强档但分数在偏档区间：同样自相矛盾（虽然后果轻得多）
            raise ScoreError(
                "强档自洽性核查失败：模型申报满足强档三条件"
                f"（summary: {claimed[:80]!r}）却给出 score={score_val}（偏档区间）。"
                "分数与门槛申报必须一致。"
            )

    band = data.get("band")
    derived = band_for(score_val)
    if band not in BANDS:
        # 模型给的 band 不合法时按 score 推导（并在 evidence 之外不做静默修正：
        # band 与 score 的自洽由 validate_score 再查一次）
        band = derived
    elif band != derived:
        # 自洽性由分数决定：band 与 score 冲突时以 score 为准（分数是连续量，
        # band 是它的离散化，冲突说明模型抄错了档位名）
        band = derived

    spans_raw = data.get("evidence_spans") or []
    if not isinstance(spans_raw, list) or not spans_raw:
        raise ScoreError("模型未给出 evidence_spans：无原文证据的分数不被接受（spec D2）")

    spans: list[EvidenceSpan] = []
    for i, sp in enumerate(spans_raw):
        if not isinstance(sp, dict) or not isinstance(sp.get("quote"), str):
            raise ScoreError(f"evidence_spans[{i}] 形状不对：{sp!r}")
        quote = sp["quote"].strip()
        if not quote:
            raise ScoreError(f"evidence_spans[{i}].quote 为空")
        cs = sp.get("char_start")
        ce = sp.get("char_end")
        if not isinstance(cs, int) or not isinstance(ce, int):
            cs = ce = None
        span = EvidenceSpan(quote=quote, char_start=cs if cs is not None else 0,
                            char_end=ce if ce is not None else 0)
        # 校验并修正偏移：模型给的偏移常不准，但引文必须真实存在。
        # 定位顺序：① 模型实际看到的材料文本（wrap 后）→ ② 全篇 flatten（引文
        # 可能跨分块边界）→ ③ 原文。三处都找不到才判编造。
        # 教训：只在"块原文"里找会把合法引文误判成编造——模型看到的是包裹文本。
        quote_text = None
        loc = locate_quote(quote, seen_text)
        if loc is not None:
            quote_text = seen_text[loc[0]:loc[1]]
        elif original_text is not None:
            full = flatten(original_text)
            loc = locate_quote(quote, full)
            if loc is not None:
                quote_text = full[loc[0]:loc[1]]
        if quote_text is None:
            raise ScoreError(
                f"evidence_spans[{i}].quote 不是原文逐字子串（疑似编造引用）：{quote[:60]!r}"
            )
        if original_text is not None:
            orig_loc = locate_quote(quote_text, original_text)
            if orig_loc is not None:
                span = EvidenceSpan(quote=quote_text, char_start=orig_loc[0], char_end=orig_loc[1])
            else:
                # 原文与 flatten 文本在该片段上不一致（如控制字符差异）：
                # 偏移指向 flatten 文本口径，quote 本身仍是原文可核内容
                span = EvidenceSpan(quote=quote_text, char_start=0, char_end=0)
        else:
            span = EvidenceSpan(quote=quote_text, char_start=loc[0], char_end=loc[1])
        spans.append(span)

    stc_out = data.get("strong_tier_check")
    if isinstance(stc_out, dict):
        stc_out = {k: stc_out.get(k) for k in ("direction_clear", "quant_anchor_hit", "anchor_text", "reason_operational", "summary")}

    s = SentimentScore(
        doc_id=doc_id,
        score=score_val,
        band=band,
        confidence=confidence,
        evidence_spans=spans,
        model=client.cfg.model,
        prompt_version=prompt.version,
        as_of=as_of,
        scored_at=_now(),
        chunk_index=chunk_index if chunk_total > 1 else None,
        chunk_total=chunk_total if chunk_total > 1 else None,
    )
    # 分块内也要过一遍：quote 必须在**模型看到的材料文本**里（全文档级由合并后再查）
    validate_score(s, text=seen_text)
    return s, stc_out


def _merge(scores: list[SentimentScore], weights: list[int]) -> SentimentScore:
    """按 prompt 工件里的合并规则：分数按块字符数加权，confidence 取最小。"""
    total_w = sum(weights) or 1
    merged = sum(s.score * w for s, w in zip(scores, weights)) / total_w
    conf = min(s.confidence for s in scores)
    # 证据取 |score| 最大的块的前 3 条
    ranked = sorted(zip(scores, weights), key=lambda sw: abs(sw[0].score), reverse=True)
    spans: list[EvidenceSpan] = []
    for s, _ in ranked:
        for sp in s.evidence_spans:
            if len(spans) >= 3:
                break
            spans.append(sp)
        if len(spans) >= 3:
            break
    first = scores[0]
    return SentimentScore(
        doc_id=first.doc_id,
        score=round(merged, 6),
        band=band_for(merged),
        confidence=round(conf, 6),
        evidence_spans=spans,
        model=first.model,
        prompt_version=first.prompt_version,
        as_of=first.as_of,
        scored_at=_now(),
        chunk_index=None,
        chunk_total=len(scores),
    )


def score_document(
    client: LLMClient,
    req: ScoreRequest,
    *,
    prompt: PromptArtifact | None = None,
    store=None,
) -> dict[str, Any]:
    """对一篇文档打分并落库。返回记录 dict（含 created 标志与 content_hash）。

    **签名里没有市场数据/上下文参数**——point-in-time 由接口形状保证（spec D7）。
    幂等：同 (doc_id, content_hash, prompt_version, model) 直接返回既有记录，
    不调模型（调用计数可验证）。
    """
    prompt = prompt or load_prompt("v1")
    flat = flatten(req.text)
    if not flat:
        raise ScoreError("文档正文为空（flatten 后）")
    chash = content_hash(req.text)

    if store is not None:
        existing = store.get_score(req.doc_id, chash, prompt.version, client.cfg.model)
        if existing:
            existing["created"] = False
            existing["content_hash"] = chash
            return existing

    chunks = _chunk_text(flat)
    part_scores: list[SentimentScore] = []
    stc_checks: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        s_i, stc_i = _score_chunk(
            client, prompt, chunk, doc_id=req.doc_id, as_of=req.as_of,
            doc_type=req.doc_type, chunk_index=i, chunk_total=len(chunks),
            original_text=req.text,
        )
        part_scores.append(s_i)
        if stc_i:
            stc_checks.append(stc_i)

    final = part_scores[0] if len(part_scores) == 1 else _merge(part_scores, [len(c) for c in chunks])
    # 全文档级复核：证据须能在原文定位；偏移优先对齐原文，退化为 flatten 文本口径
    validate_score(final, text=req.text)

    rec = final.to_dict()
    rec["content_hash"] = chash
    rec["doc_type"] = req.doc_type
    # strong_tier_check 落库：门槛申报是可审计性的载体，只用于服务端核查而不
    # 落库等于"可审计"落空（v3 实测时该字段为 null 才发现）
    if stc_checks:
        rec["strong_tier_check"] = stc_checks[0] if len(stc_checks) == 1 else {"chunks": stc_checks}
    if store is not None:
        stored, created = store.insert_score(rec)
        stored["created"] = created
        stored["content_hash"] = chash
        return stored
    rec["created"] = True
    return rec


def check_no_market_params(fn) -> bool:
    """入口签名核查（1.4 验收用）：不得出现市场数据类参数。"""
    import inspect

    forbidden = {"price", "prices", "market", "market_data", "returns", "ohlc", "volume", "quote_price"}
    params = set(inspect.signature(fn).parameters)
    return not (params & forbidden)
