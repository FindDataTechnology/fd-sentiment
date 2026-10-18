"""存储 + 打分 + 幂等 + 批量 + 聚合 + 评测的离线测试（假 LLM）。"""
import pytest

from fd_sentiment.aggregate import aggregate
from fd_sentiment.batch import score_batch
from fd_sentiment.evaluate import run_eval, compare_runs
from fd_sentiment.models import BANDS, ScoreError
from fd_sentiment.scoring import ScoreRequest, content_hash, score_document
from fd_sentiment.storage import Store

from fake_llm import FakeLLM, good_response

TEXT = "公司发布业绩预告：预计上半年净利润同比增长 120% 至 150%，主要因主营业务收入大幅增长。"
QUOTE = "预计上半年净利润同比增长 120% 至 150%"


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "t.db")


def test_insert_and_idempotent(store):
    llm = FakeLLM([good_response(QUOTE)])
    req = ScoreRequest(doc_id="d1", text=TEXT, as_of="2026-10-17")
    r1 = score_document(llm, req, store=store)
    assert r1["created"] is True
    assert len(llm.calls) == 1
    # 同键再打：不调模型、不新增记录
    r2 = score_document(llm, req, store=store)
    assert r2["created"] is False
    assert len(llm.calls) == 1
    assert store.score_count() == 1
    assert r2["evidence_spans"] == r1["evidence_spans"]


def test_content_change_triggers_new_record(store):
    llm = FakeLLM([good_response(QUOTE), good_response("主营业务收入大幅增长", score=-0.5, band="bearish")])
    score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    score_document(llm, ScoreRequest("d1", TEXT + "补充：另有诉讼风险。", "2026-10-17"), store=store)
    assert store.score_count() == 2
    rows = store.list_scores(doc_id="d1")
    assert len(rows) == 2  # 旧记录保留


def test_fabricated_quote_rejected(store):
    llm = FakeLLM([good_response("这句话原文里根本没有出现过")]
    )
    with pytest.raises(ScoreError, match="逐字子串"):
        score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert store.score_count() == 0


def test_refusal_path(store):
    llm = FakeLLM([{"refuse": True, "reason": "纯数字表格，无情绪语句"}])
    with pytest.raises(ScoreError, match="拒绝打分"):
        score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert store.score_count() == 0


def test_band_derived_from_score_when_mismatched(store):
    # 模型说 band=bearish 但 score=0.7 → 以 score 为准
    llm = FakeLLM([good_response(QUOTE, score=0.7, band="bearish")])
    rec = score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert rec["band"] == "strong_bullish"


def test_batch_mixed_success_and_failure(store):
    docs = [
        ScoreRequest("ok1", TEXT, "2026-10-17"),
        ScoreRequest("bad", TEXT, "2026-10-17"),
        ScoreRequest("ok2", TEXT, "2026-10-17"),
    ]
    llm = FakeLLM([
        good_response(QUOTE),
        good_response("不存在的引文"),
        good_response(QUOTE, score=-0.3, band="bearish"),
    ])
    out = score_batch(llm, docs, store=store, max_concurrency=1)
    assert out["total"] == 3
    assert out["ok"] == 2
    assert out["failed"] == 1
    failed = [r for r in out["results"] if r["status"] == "failed"]
    assert failed and "逐字子串" in failed[0]["error"]


def test_batch_skips_existing(store):
    llm = FakeLLM([good_response(QUOTE)])
    score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    out = score_batch(llm, [ScoreRequest("d1", TEXT, "2026-10-17")], store=store)
    assert out["skipped"] == 1
    assert out["ok"] == 0
    assert len(llm.calls) == 1  # 跳过时不调模型


def test_aggregate_distribution(store):
    llm = FakeLLM([
        good_response(QUOTE, score=0.7, band="strong_bullish", confidence=0.9),
        good_response(QUOTE, score=-0.4, band="bearish", confidence=0.6),
    ])
    score_document(llm, ScoreRequest("a", TEXT, "2026-10-17"), store=store)
    score_document(llm, ScoreRequest("b", TEXT, "2026-10-17"), store=store)
    agg = aggregate(store, universe_total=10)
    assert agg["scored_count"] == 2
    assert agg["band_counts"]["strong_bullish"] == 1
    assert agg["band_counts"]["bearish"] == 1
    assert agg["mean_score"] == pytest.approx(0.15)
    assert agg["coverage"]["ratio"] == pytest.approx(0.2)
    # 明确非指数
    assert "非市场情绪指数" in agg["title"]
    assert "指数" not in str(agg["band_counts"])


def test_aggregate_unknown_universe_is_null(store):
    agg = aggregate(store)
    assert agg["coverage"]["universe_total"] is None
    assert agg["coverage"]["ratio"] is None


def test_eval_and_calibration(store):
    # 金标：三条，模型全对，置信 0.9 → 高置信桶 100%
    store.upsert_gold(set_version="g1", doc_id="x1", as_of="2026-10-17",
                      expected_band="strong_bullish", text=TEXT)
    store.upsert_gold(set_version="g1", doc_id="x2", as_of="2026-10-17",
                      expected_band="strong_bullish", text=TEXT)
    store.upsert_gold(set_version="g1", doc_id="x3", as_of="2026-10-17",
                      expected_band="strong_bullish", text=TEXT)
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(3)])
    rep = run_eval(llm, store, set_version="g1")
    assert rep["samples_scored"] == 3
    assert rep["hit_rate"] == 1.0
    assert rep["calibration"]["ok"] is True
    assert rep["run_id"].startswith("eval-")
    # 对比
    cmp = compare_runs(store, [rep["run_id"]])
    assert cmp["runs"][0]["hit_rate"] == 1.0


def test_eval_detects_calibration_failure(store):
    # 低置信桶全对、高置信桶全错 → 校准失败必须被标注
    for i in range(3):
        store.upsert_gold(set_version="g2", doc_id=f"lo{i}", as_of="2026-10-17",
                          expected_band="strong_bullish", text=TEXT)
    for i in range(3):
        store.upsert_gold(set_version="g2", doc_id=f"hi{i}", as_of="2026-10-17",
                          expected_band="bearish", text=TEXT)
    # 金标按 doc_id 排序遍历（hi0..hi2 在 lo0..lo2 之前），响应顺序必须对齐
    responses = [good_response(QUOTE, 0.7, "strong_bullish", 0.95) for _ in range(3)]  # 高置信但全错
    responses += [good_response(QUOTE, 0.7, "strong_bullish", 0.4) for _ in range(3)]  # 低置信但全对
    llm = FakeLLM(responses)
    rep = run_eval(llm, store, set_version="g2")
    assert rep["calibration"]["ok"] is False
    assert "校准失败" in rep["calibration"]["note"]


def test_prompt_version_separation(store):
    from fd_sentiment.prompt import PromptArtifact, load_prompt

    v1 = load_prompt("v1")
    v2 = PromptArtifact(version="v2", system=v1.system + "\n（v2）",
                        user_template=v1.user_template, bands_spec=v1.bands_spec, merge_rule=v1.merge_rule)
    llm = FakeLLM([good_response(QUOTE), good_response(QUOTE)])
    score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), prompt=v1, store=store)
    score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), prompt=v2, store=store)
    rows = store.list_scores(doc_id="d1")
    assert {r["prompt_version"] for r in rows} == {"v1", "v2"}
    assert len(rows) == 2


def test_llm_call_logging_counts(store):
    llm = FakeLLM([good_response(QUOTE)])
    score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    # FakeLLM 不写 llm_calls；真实客户端的记账由 test_llm.py 覆盖
    assert store.score_count() == 1


def test_content_hash_normalizes_whitespace():
    assert content_hash("a  b\n\nc") == content_hash("a b c")
    assert content_hash("a b") != content_hash("a c")


def test_strong_tier_consistency_rejects_contradiction(store):
    """v3 契约：模型申报未达强档却给 |score|>=0.6 → 拒绝（v2 的失败模式）。"""
    llm = FakeLLM([{
        "score": 0.8,
        "band": "strong_bullish",
        "confidence": 0.9,
        "strong_tier_check": {"direction_clear": True, "quant_anchor_hit": False,
                              "anchor_text": "none", "reason_operational": True,
                              "summary": "增幅未达50%，未命中量化锚点"},
        "evidence_spans": [{"quote": "预计净利润增长", "char_start": 0, "char_end": 8}],
    }])
    with pytest.raises(ScoreError, match="强档自洽性核查失败"):
        score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert store.score_count() == 0


def test_strong_tier_consistency_accepts_coherent(store):
    """申报与分数一致时正常通过。"""
    llm = FakeLLM([{
        "score": 0.5, "band": "bullish", "confidence": 0.6,
        "strong_tier_check": {"direction_clear": True, "quant_anchor_hit": False,
                              "anchor_text": "none", "reason_operational": True,
                              "summary": "增幅15%-20%未达50%"},
        "evidence_spans": [{"quote": QUOTE, "char_start": 0, "char_end": len(QUOTE)}],
    }])
    rec = score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert rec["score"] == 0.5 and rec["band"] == "bullish"


def test_strong_tier_absent_still_works(store):
    """v1/v2 的输出没有该字段——不能因此崩（向后兼容）。"""
    llm = FakeLLM([good_response(QUOTE)])
    rec = score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), prompt=None, store=store)
    assert rec["band"] in BANDS


def test_strong_tier_check_persisted(store):
    """门槛申报必须落库——只用于服务端核查而不落库等于"可审计"落空。"""
    llm = FakeLLM([{
        "score": 0.68, "band": "strong_bullish", "confidence": 0.85,
        "strong_tier_check": {"direction_clear": True, "quant_anchor_hit": True,
                              "anchor_text": "同比增长 65% 至 80%", "reason_operational": True,
                              "summary": "增幅超50%且理由指向经营实质"},
        "evidence_spans": [{"quote": QUOTE, "char_start": 0, "char_end": len(QUOTE)}],
    }])
    rec = score_document(llm, ScoreRequest("d1", TEXT, "2026-10-17"), store=store)
    assert rec["strong_tier_check"]["quant_anchor_hit"] is True
    assert rec["strong_tier_check"]["anchor_text"] == "同比增长 65% 至 80%"
    # 落库后仍可读回
    again = store.get_score("d1", rec["content_hash"], rec["prompt_version"], rec["model"])
    assert again["strong_tier_check"]["summary"]


def test_current_prompt_version_reports_default_and_env_override(monkeypatch):
    """current_prompt_version 不得引用未定义变量（线上 sha-b38f482 曾报
    `name 'prompt_version' is not defined`，工具整体不可用）。"""
    import asyncio

    from fastmcp import Client

    from fd_sentiment.server import build_server

    monkeypatch.setenv("FD_SENTIMENT_PROMPT_VERSION", "v2")

    async def run():
        app = build_server()
        async with Client(app) as c:
            return await c.call_tool("current_prompt_version", {})

    res = asyncio.run(run())
    payload = res.structured_content or {}
    assert payload.get("prompt_version") == "v2"
    assert payload.get("env_override") == "v2"
    assert "v1" in (payload.get("available_versions") or [])
    assert payload.get("prompt_digest")


def test_calipers_three_calibers_coexist():
    """三口径必须并列：exact 低估、±1 档高估，只用其中一个会误导下游。

    夹具按 2026-10-18 gold-v6 满样本实测的**真实配对分布**构造（54 条），
    见 reports/v3-gold-v6-evaluation-2026-10-18.md。期望值即当时的实测值。
    """
    from fd_sentiment.evaluate import _calipers

    # (expected_band, predicted_band) -> 条数，与实测混淆矩阵逐格一致
    PAIRS = {
        ("bearish", "bearish"): 4,
        ("bearish", "strong_bearish"): 8,
        ("bullish", "bullish"): 2,
        ("bullish", "neutral"): 8,
        ("bullish", "strong_bullish"): 13,
        ("neutral", "bearish"): 2,
        ("neutral", "neutral"): 16,
        ("neutral", "strong_bearish"): 1,
    }
    items = [
        {"expected_band": e, "predicted_band": p, "hit": (e == p)}
        for (e, p), n in PAIRS.items()
        for _ in range(n)
    ]

    got = _calipers(items)
    assert got["samples"] == 54
    # exact：4 + 2 + 16 = 22
    assert got["exact"] == round(22 / 54, 4) == 0.4074
    # 方向（正/中/负）：只有 bullish->neutral(8) 与 neutral->bearish(2) 及
    # neutral->strong_bearish(1) 方向错，共 11 条错 → 43/54
    assert got["direction"] == round(43 / 54, 4) == 0.7963
    # ±1 档：只有 neutral->strong_bearish(1) 跨两级 → 53/54
    assert got["adjacent"] == round(53 / 54, 4) == 0.9815


def test_calipers_empty_is_safe():
    """空清单不炸（评测全失败时报告仍要能落库）。"""
    from fd_sentiment.evaluate import _calipers

    got = _calipers([])
    assert got["samples"] == 0
    assert got["exact"] == 0.0 and got["direction"] == 0.0 and got["adjacent"] == 0.0
