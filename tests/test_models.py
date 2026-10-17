"""形状校验与证据校验：spec 的硬约束逐条测。"""
import pytest

from fd_sentiment.models import (
    EvidenceSpan, SentimentScore, ScoreError, band_for, locate_quote, validate_score, verify_quote,
)

TEXT = "公司发布业绩预告：预计上半年净利润同比增长 120% 至 150%，主要因主营业务收入大幅增长。"


def _score(**kw):
    base = dict(
        doc_id="d1", score=0.7, band="strong_bullish", confidence=0.9,
        evidence_spans=[EvidenceSpan("预计上半年净利润同比增长 120% 至 150%", 9, 33)],
        model="m", prompt_version="v1", as_of="2026-10-17", scored_at="2026-10-17T00:00:00Z",
    )
    base.update(kw)
    return SentimentScore(**base)


def test_band_boundaries():
    assert band_for(0.6) == "strong_bullish"
    assert band_for(0.59) == "bullish"
    assert band_for(0.2) == "bullish"
    assert band_for(0.19) == "neutral"
    assert band_for(-0.19) == "neutral"
    assert band_for(-0.2) == "bearish"
    assert band_for(-0.6) == "strong_bearish"
    assert band_for(-0.61) == "strong_bearish"


def test_valid_score_passes():
    validate_score(_score(), text=TEXT)


def test_empty_evidence_rejected():
    with pytest.raises(ScoreError, match="evidence_spans 不能为空"):
        validate_score(_score(evidence_spans=[]))


def test_score_out_of_range_rejected():
    with pytest.raises(ScoreError, match="-1..\\+1"):
        validate_score(_score(score=1.5))


def test_bad_band_rejected():
    with pytest.raises(ScoreError, match="五档"):
        validate_score(_score(band="positive"))


def test_confidence_out_of_range_rejected():
    with pytest.raises(ScoreError, match="0..1"):
        validate_score(_score(confidence=1.2))


def test_as_of_must_be_iso_date():
    with pytest.raises(ScoreError, match="ISO 日期"):
        validate_score(_score(as_of="2026/10/17"))


def test_fabricated_quote_rejected():
    span = EvidenceSpan("净利润增长 500%", 0, 10)
    with pytest.raises(ScoreError, match="逐字子串"):
        verify_quote(span, TEXT)


def test_whitespace_normalized_quote_accepted():
    span = EvidenceSpan("预计上半年净利润同比增长\n120% 至 150%", 0, 0)
    # 原文里是单空格；归一化后应匹配（偏移不准但引文真实）
    verify_quote(span, TEXT)


def test_locate_quote_exact_and_fuzzy():
    q = "预计上半年净利润同比增长 120% 至 150%"
    loc = locate_quote(q, TEXT)
    assert loc is not None
    assert TEXT[loc[0]:loc[1]] == q
    assert locate_quote("不存在的句子", TEXT) is None


def test_evidence_span_interval_validation():
    with pytest.raises(ScoreError, match="区间非法"):
        validate_score(_score(evidence_spans=[EvidenceSpan("x", 10, 10)]))


def test_locate_quote_tolerates_space_and_fullwidth():
    """中文引用的常见差异：模型加空格、全半角括号——都应能定位（不许判成编造）。"""
    t1 = "预计净利润计增加97亿元左右,同比增加58%左右。"
    q1 = "计增加97 亿元左右,同比增加58%左右"
    loc1 = locate_quote(q1, t1)
    assert loc1 is not None and t1[loc1[0]:loc1[1]] == "计增加97亿元左右,同比增加58%左右"

    t2 = "公司营业收入（人民币）为 100 万元"
    q2 = "公司营业收入(人民币)为100万元"
    loc2 = locate_quote(q2, t2)
    assert loc2 is not None and t2[loc2[0]:loc2[1]] == t2

    # 词序变了仍必须失败（不能宽到失去意义）
    assert locate_quote("58%左右计增加97亿元", t1) is None


def test_verify_quote_shares_locate_criteria():
    """判据必须只有一套：块内定位用 locate_quote，全文档复核用 verify_quote。
    曾被写成两套（verify 只查整串），合法引文被全文档复核判成编造。"""
    text = "预计归属于上市公司股东的净利润为人民币 58 亿元至 62 亿元，同比增长 45% 至 55%。"
    # 模型把全角括号/空格/破折号按视觉习惯归一 —— locate_quote 能定位，
    # verify_quote 就必须同样放行
    quote = "预计归属于上市公司股东的净利润为人民币 58 亿元至 62 亿元,同比增长 45% 至 55%"
    span = EvidenceSpan(quote=quote, char_start=0, char_end=0)
    verify_quote(span, text)  # 不抛即通过
    assert locate_quote(quote, text) is not None
