"""SentimentScore 与 EvidenceSpan —— 与 spec 逐字段对齐的数据形状。

spec 钉死的字段一个不能少、不能改名：
  doc_id / score(-1..+1) / band(五档之一) / confidence(0..1) /
  evidence_spans(非空，每项 quote + char_start + char_end) /
  model / prompt_version / as_of(ISO 日期) / scored_at

校验放在这里而不是打分流程里：任何写入路径（单条、批量、评测导入）
都必须过 `validate_score`，避免"某个入口绕过了校验"。
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any

BANDS = ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish")
BAND_ZH = {
    "strong_bullish": "强正面",
    "bullish": "偏正面",
    "neutral": "中性",
    "bearish": "偏负面",
    "strong_bearish": "强负面",
}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class NotFoundError(ValueError):
    """任务或记录不存在：查询合法但无此标识。

    独立类型而不是 ScoreError 的子类：两者语义不同——ScoreError 是"这份文本
    打不出分"，NotFoundError 是"这个 id 从未提交过"，调用方处置完全不同
    （前者改输入，后者改 id 或确认是否记错）。
    """


class ScoreError(ValueError):
    """分数不合法（形状/取值/证据）。调用方据此拒绝该次打分。"""


@dataclass(frozen=True)
class EvidenceSpan:
    quote: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SentimentScore:
    doc_id: str
    score: float
    band: str
    confidence: float
    evidence_spans: list[EvidenceSpan]
    model: str
    prompt_version: str
    as_of: str
    scored_at: str = ""
    # 分块信息（长文档才有）：块序号与总块数，输出可见，不隐藏截断
    chunk_index: int | None = None
    chunk_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_spans"] = [s.to_dict() if isinstance(s, EvidenceSpan) else s for s in self.evidence_spans]
        return d


def band_for(score: float) -> str:
    """连续分 → 五档。边界取闭区间外侧，保证 -1 与 +1 落强档。"""
    if score >= 0.6:
        return "strong_bullish"
    if score >= 0.2:
        return "bullish"
    if score > -0.2:
        return "neutral"
    if score > -0.6:
        return "bearish"
    return "strong_bearish"


def _check_iso_date(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        raise ScoreError(f"{field_name} 必须是 ISO 日期（YYYY-MM-DD），收到 {value!r}")
    try:
        _dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ScoreError(f"{field_name} 不是合法日期：{value!r}") from exc


def validate_score(s: SentimentScore, *, text: str | None = None, require_evidence: bool = True) -> None:
    """按 spec 校验一条分数。`text` 给出时同时校验 quote 是原文逐字子串。"""
    if not isinstance(s.doc_id, str) or not s.doc_id.strip():
        raise ScoreError("doc_id 不能为空")
    if not isinstance(s.score, (int, float)) or isinstance(s.score, bool):
        raise ScoreError("score 必须是数值")
    if not -1.0 <= float(s.score) <= 1.0:
        raise ScoreError(f"score 必须在 -1..+1，收到 {s.score}")
    if s.band not in BANDS:
        raise ScoreError(f"band 必须是五档之一，收到 {s.band!r}")
    if not isinstance(s.confidence, (int, float)) or isinstance(s.confidence, bool):
        raise ScoreError("confidence 必须是数值")
    if not 0.0 <= float(s.confidence) <= 1.0:
        raise ScoreError(f"confidence 必须在 0..1，收到 {s.confidence}")
    if not isinstance(s.model, str) or not s.model.strip():
        raise ScoreError("model 不能为空")
    if not isinstance(s.prompt_version, str) or not s.prompt_version.strip():
        raise ScoreError("prompt_version 不能为空")
    _check_iso_date(s.as_of, "as_of")

    if not isinstance(s.evidence_spans, list):
        raise ScoreError("evidence_spans 必须是数组")
    if require_evidence and not s.evidence_spans:
        raise ScoreError("evidence_spans 不能为空：无原文证据的分数不被接受（spec D2）")

    for i, span in enumerate(s.evidence_spans):
        if isinstance(span, dict):
            span = EvidenceSpan(**span)
        if not isinstance(span, EvidenceSpan):
            raise ScoreError(f"evidence_spans[{i}] 形状不对")
        if not span.quote or not span.quote.strip():
            raise ScoreError(f"evidence_spans[{i}].quote 不能为空")
        if not isinstance(span.char_start, int) or not isinstance(span.char_end, int):
            raise ScoreError(f"evidence_spans[{i}] 的 char_start/char_end 必须是整数")
        if span.char_start < 0 or span.char_end <= span.char_start:
            raise ScoreError(f"evidence_spans[{i}] 区间非法：{span.char_start}..{span.char_end}")
        if text is not None:
            verify_quote(span, text, index=i)


_WS = re.compile(r"\s+")
# 破折号 / 波浪号 / 连接号 / 全角减号 / "至" 统一成 ASCII 连字符（仅用于定位，
# 不改动原文与 quote 本身）
_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d~\uff5e]")


def _normalize_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def verify_quote(span: EvidenceSpan, text: str, *, index: int = 0) -> None:
    """校验 quote 是原文的逐字子串（允许空白/全半角归一化，不允许改写）。

    三条路径，任一通过即算合法：
      ① 按 char_start/char_end 切片，与 quote 归一化后相等；
      ② 切片对不上（模型给的偏移不准但引文真实）→ 走 locate_quote 的归一化定位；
      ③ 兜底：把 quote 与 text 都按"去空白 + 全半角 + 破折号归一"压平后做子串查找。

    **为什么要复用 locate_quote**：本函数曾被写成"只查整串原文"，而打分链路里
    定位用的却是 locate_quote 的三级归一化——两套判据不一致，导致块内定位通过、
    全文档复核却把同一条合法引文判成编造（v2 冒烟实测撞到）。判据必须只有一套。
    """
    sliced = text[span.char_start : span.char_end]
    if _normalize_ws(sliced) == _normalize_ws(span.quote):
        return
    if locate_quote(span.quote, text) is not None:
        return
    raise ScoreError(
        f"evidence_spans[{index}].quote 不是原文逐字子串（疑似编造引用）：{span.quote[:60]!r}"
    )


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """在原文中定位 quote，返回 (start, end)；找不到返回 None。

    三级匹配（逐级放宽，但**词序与用词始终不许改**）：
      ① 精确子串；
      ② 空白归一化：quote 按空白切词，词间允许任意空白（原文换行/多空格）；
      ③ 空白 + 全半角归一化：NFKC 后再按 ② 匹配（中文公告里 "97 亿元" vs
         "97亿元"、全角括号与半角括号这类差异很常见，模型引用时常按视觉习惯
         归一，不应因此判定"编造"）。
    命中后返回的区间对齐**原文**（③ 的区间由原文词位推回）。
    """
    idx = text.find(quote)
    if idx >= 0:
        return idx, idx + len(quote)

    words = [w for w in _WS.split(quote) if w]
    if not words:
        return None
    # 空白**可有可无**：模型常按视觉习惯在中文数字/单位间插空格，
    # 原文未必有；要求 \s+ 会把这类真实引文误判成编造。
    pattern = r"\s*".join(re.escape(w) for w in words)
    m = re.search(pattern, text)
    if m:
        return m.start(), m.end()

    # ③ 全半角归一化 + 去空白：在归一化文本上定位，再把区间映射回原文。
    # 中文公告里 "97 亿元" vs "97亿元"、全角括号 vs 半角括号的差异极常见；
    # 模型引用时常按视觉习惯归一。这里把两边的空白全去掉再比，仍要求词序与用词一致。
    norm_text = unicodedata.normalize("NFKC", text)
    norm_quote = unicodedata.normalize("NFKC", quote)
    # 破折号/波浪号/连接号变体归一（– — － ~ ～ 至 均视作 -）：
    # 中文公告写 "60% – 70%"，模型引用常写成 "60%-70%" 或 "60%至70%"。
    squeezed_text = _DASHES.sub("-", _WS.sub("", norm_text))
    squeezed_quote = _DASHES.sub("-", _WS.sub("", norm_quote))
    if not squeezed_quote:
        return None
    pos = squeezed_text.find(squeezed_quote)
    if pos < 0:
        return None
    # 映射回原文：squeezed 是"原文去空白 + 破折号归一"，逐字符重建索引
    orig = _squeezed_to_original_span(text, pos, pos + len(squeezed_quote))
    if orig is None:
        return None
    nm_start, nm_end = orig
    # NFKC 通常保长度（除个别兼容字符），逐字符映射：用归一化后的前缀长度
    # 反查原文位置——按"归一化文本前 N 个字符对应原文第几个字符"重建索引。
    return _map_norm_span(text, nm_start, nm_end)


def _squeezed_to_original_span(text: str, sq_start: int, sq_end: int) -> tuple[int, int] | None:
    """把"去空白 + 破折号归一"文本上的 [sq_start, sq_end) 映射回原文区间。"""
    count = 0
    orig_start = orig_end = None
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        if orig_start is None and count >= sq_start:
            orig_start = i
        count += 1
        if count >= sq_end:
            orig_end = i + 1
            break
    if orig_start is None:
        return None
    if orig_end is None:
        orig_end = len(text)
    return orig_start, orig_end


def _map_norm_span(text: str, nstart: int, nend: int) -> tuple[int, int] | None:
    """把归一化文本的 [nstart, nend) 映射回原文区间。"""
    import unicodedata as _u

    # 逐个字符归一化并累计，找到覆盖 nstart / nend 的原文下标
    def norm_len(ch: str) -> int:
        return len(_u.normalize("NFKC", ch))

    pos = 0
    orig_start = orig_end = None
    for i, ch in enumerate(text):
        ln = norm_len(ch)
        if orig_start is None and pos + ln > nstart:
            orig_start = i
        if pos + ln >= nend:
            orig_end = i + 1
            break
        pos += ln
    if orig_start is None:
        return None
    if orig_end is None:
        orig_end = len(text)
    return orig_start, orig_end
