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


def test_eval_concurrent_matches_serial_shape(store):
    """并发评测的结果形状、分母、逐项顺序都必须与串行一致。

    并发只是提速手段，不能改变报告语义：命中数、失败数、per_item 的
    doc_id 顺序（按金标原序）都要一样。若哪天并发实现打乱了顺序，
    compare_evaluations 的并列对比就会错位。
    """
    for i in range(6):
        store.upsert_gold(set_version="gc", doc_id=f"c{i}", as_of="2026-10-17",
                          expected_band="strong_bullish", text=TEXT)
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(6)])
    rep = run_eval(llm, store, set_version="gc", concurrency=3)
    assert rep["concurrency"] == 3
    assert rep["samples_expected"] == 6
    assert rep["samples_scored"] == 6
    assert rep["hit_rate"] == 1.0
    assert rep["calipers"]["samples"] == 6
    # 顺序必须是金标原序（Doc_id 升序），不是完成先后
    assert [i["doc_id"] for i in rep["per_item"]] == sorted(f"c{i}" for i in range(6))


def test_eval_concurrent_failures_not_counted_in_denominator(store):
    """并发下失败的样本同样计入 failures、不进分母——与串行同一口径。"""
    for i in range(4):
        store.upsert_gold(set_version="gd", doc_id=f"d{i}", as_of="2026-10-17",
                          expected_band="strong_bullish", text=TEXT)
    from fd_sentiment.models import ScoreError

    # 前两条成功，后两条拒答；并发执行时响应顺序不确定，故用可重入的假客户端
    class RefusingLLM(FakeLLM):
        def chat(self, messages, *, max_tokens=1500, doc_id=None):
            if doc_id and doc_id >= "d2":
                raise ScoreError("无可用情绪证据")
            return super().chat(messages, max_tokens=max_tokens, doc_id=doc_id)

    llm = RefusingLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(2)])
    rep = run_eval(llm, store, set_version="gd", concurrency=2)
    assert rep["samples_expected"] == 4
    assert rep["samples_scored"] == 2
    assert len(rep["failures"]) == 2
    assert all("拒绝" in f["error"] for f in rep["failures"])
    # 分母只有成功的两条
    assert rep["hit_rate"] == 1.0


def test_eval_serial_is_default(store):
    """默认必须是串行——并发是显式选择，不能悄悄改变失败率构成。"""
    store.upsert_gold(set_version="ge", doc_id="e1", as_of="2026-10-17",
                      expected_band="strong_bullish", text=TEXT)
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9)])
    rep = run_eval(llm, store, set_version="ge")
    assert rep["concurrency"] == 1


def test_prompts_shipped_inside_package():
    """prompts 必须在包内（随 wheel 分发），不能靠仓库根目录的相对路径。

    曾经的缺陷：prompts 放仓库根，pip install 不打包，从非 /app 目录调用
    就报「prompt 版本 'v3' 不存在（site-packages/prompts/v3）」。
    """
    from fd_sentiment.prompt import PROMPT_DIR, load_prompt

    assert PROMPT_DIR.name == "prompts"
    assert PROMPT_DIR.parent.name == "fd_sentiment", "prompts 必须是包的子目录"
    assert (PROMPT_DIR / "v3" / "system.md").exists()
    # 与 cwd 无关
    import os

    old = os.getcwd()
    try:
        os.chdir("/")
        assert load_prompt("v3").digest
    finally:
        os.chdir(old)


def test_coverage_reports_all_five_bands(store):
    """覆盖度必须报出全部五档（缺档补 0），不能只报有样本的档。"""
    store.upsert_gold(set_version="cv1", doc_id="a", as_of="2026-10-18",
                      expected_band="bullish", text=TEXT)
    cov = store.gold_coverage("cv1")
    assert set(cov) == {"strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"}
    assert cov["bullish"] == 1
    assert cov["strong_bullish"] == 0


def test_eval_excludes_uncovered_band_from_exact(store):
    """模型判了金标没有的档 -> 单列，不计入 exact 分母。

    这是 gold-v6 的核心缺陷：金标缺 strong_*，模型判强档必然记错。
    正确行为是承认「评测表缺档」，而不是把模型记为错。
    """
    # 金标只有 bullish 与 neutral，没有 strong_bullish
    store.upsert_gold(set_version="cv2", doc_id="m1", as_of="2026-10-18",
                      expected_band="bullish", text=TEXT)
    store.upsert_gold(set_version="cv2", doc_id="m2", as_of="2026-10-18",
                      expected_band="neutral", text=TEXT)
    # 模型：m1 判 strong_bullish（未覆盖档）、m2 判 neutral（命中）
    # 金标按 doc_id 升序遍历，故响应顺序须对齐（m1 先于 m2）
    llm = FakeLLM([
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),
        good_response(QUOTE, 0.0, "neutral", 0.8),
    ])
    rep = run_eval(llm, store, set_version="cv2")
    # 金标只有 bullish/neutral，故其余三档都是未覆盖
    assert set(rep["coverage"]["uncovered"]) == {"strong_bullish", "bearish", "strong_bearish"}
    assert rep["coverage"]["excluded_from_exact"] == 1
    assert len(rep["coverage_gap_items"]) == 1
    assert rep["coverage_gap_items"][0]["predicted_band"] == "strong_bullish"
    # 分母只剩 1 条（m2），且命中
    assert rep["samples_scored"] == 1
    assert rep["hit_rate"] == 1.0
    assert "未覆盖" in rep["coverage"]["note"]


def test_eval_full_coverage_counts_normally(store):
    """五档齐全时不做任何排除，行为与原先一致。"""
    for band, score in (("strong_bullish", 0.7), ("bullish", 0.3), ("neutral", 0.0),
                        ("bearish", -0.3), ("strong_bearish", -0.7)):
        store.upsert_gold(set_version="cv3", doc_id=f"f_{band}", as_of="2026-10-18",
                          expected_band=band, text=TEXT)
    # doc_id 升序：bearish < bullish < neutral < strong_bearish < strong_bullish
    # （字符串序，'bearish' < 'bullish' < 'neutral' < 'strong_...' 且 be < bu < st）
    llm = FakeLLM([
        good_response(QUOTE, -0.3, "bearish", 0.8),
        good_response(QUOTE, 0.3, "bullish", 0.8),
        good_response(QUOTE, 0.0, "neutral", 0.85),
        good_response(QUOTE, -0.7, "strong_bearish", 0.9),
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),
    ])
    rep = run_eval(llm, store, set_version="cv3")
    assert rep["coverage"]["uncovered"] == []
    assert rep["coverage"]["excluded_from_exact"] == 0
    assert rep["coverage_gap_items"] == []
    assert rep["samples_scored"] == 5
    assert rep["hit_rate"] == 1.0
    assert "五档齐全" in rep["coverage"]["note"]


def test_eval_reports_per_band_breakdown(store):
    """分层表必须按期望档位给准确率，并标出样本不足的档。

    金标按真实公告分布构建（不人为均衡），单看总命中率会被最大的一组主导，
    故必须分层。样本不足的档要明确标出「结论无统计意义」，不能混在总命中率里。
    """
    # strong_bullish 12 条（足够），bearish 2 条（不足）
    for i in range(12):
        store.upsert_gold(set_version="bd1", doc_id=f"a{i:02d}", as_of="2026-10-18",
                          expected_band="strong_bullish", text=TEXT)
    for i in range(2):
        store.upsert_gold(set_version="bd1", doc_id=f"z{i:02d}", as_of="2026-10-18",
                          expected_band="bearish", text=TEXT)
    llm = FakeLLM(
        [good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(12)]
        + [good_response(QUOTE, -0.3, "bearish", 0.7) for _ in range(2)]
    )
    rep = run_eval(llm, store, set_version="bd1")
    rows = {r["expected_band"]: r for r in rep["by_band"]["rows"]}
    assert rows["strong_bullish"]["samples"] == 12
    assert rows["strong_bullish"]["enough_samples"] is True
    assert rows["strong_bullish"]["accuracy"] == 1.0
    assert rows["bearish"]["samples"] == 2
    assert rows["bearish"]["enough_samples"] is False
    assert rep["by_band"]["insufficient"] == ["bearish"]
    assert "不具统计意义" in rep["by_band"]["note"]


def _make_calib(store, set_version, spec):
    """spec: [(expected_band, confidence, predicted_band), ...] 按 doc_id 序。"""
    for i, (exp, conf, pred) in enumerate(spec):
        store.upsert_gold(set_version=set_version, doc_id=f"k{i:03d}", as_of="2026-10-18",
                          expected_band=exp, text=TEXT)
    return FakeLLM([
        good_response(QUOTE, _score_of(pred), pred, conf) for _, conf, pred in spec
    ])


def _score_of(band):
    return {"strong_bullish": 0.7, "bullish": 0.3, "neutral": 0.0,
            "bearish": -0.3, "strong_bearish": -0.7}[band]


def test_calibration_detects_non_monotonic_shape(store):
    """三桶非单调时必须指出形状与最低桶，不能只说「高置信不如低置信」。

    这是实际踩到的：v4 三桶 0.70/0.41/0.47，最低在中间桶。
    """
    # 全判 strong_bullish、金标全 strong_bullish，只靠 confidence 分桶
    # 低桶(0.6) 全对、中桶(0.75) 全错、高桶(0.9) 全对 -> 非单调
    spec = (
        [("strong_bullish", 0.60, "strong_bullish") for _ in range(3)]      # 低桶 100%
        + [("bullish", 0.75, "strong_bullish") for _ in range(4)]           # 中桶 0%
        + [("strong_bullish", 0.90, "strong_bullish") for _ in range(4)]    # 高桶 100%
    )
    llm = _make_calib(store, "nm1", spec)
    rep = run_eval(llm, store, set_version="nm1")
    cal = rep["calibration"]
    assert cal["shape"] == "non_monotonic"
    assert cal["lowest_bucket"] == "0.70-0.85"
    assert "非单调" in cal["note"]
    assert "最低点在" in cal["note"]


def test_calibration_monotonic_increasing_passes(store):
    """单调递增才算校准正常。"""
    spec = (
        [("bearish", 0.55, "strong_bullish") for _ in range(3)]              # 低桶 0%
        + [("strong_bullish", 0.75, "strong_bullish") for _ in range(3)]     # 中桶 100%
        + [("strong_bullish", 0.90, "strong_bullish") for _ in range(3)]     # 高桶 100%
    )
    llm = _make_calib(store, "mi1", spec)
    rep = run_eval(llm, store, set_version="mi1")
    assert rep["calibration"]["shape"] == "monotonic_increasing"
    assert rep["calibration"]["ok"] is True
    assert "单调递增" in rep["calibration"]["note"]


def test_calibration_monotonic_decreasing_fails(store):
    """单调递减是明确的校准失败。"""
    spec = (
        [("strong_bullish", 0.55, "strong_bullish") for _ in range(3)]       # 低桶 100%
        + [("bearish", 0.75, "strong_bullish") for _ in range(3)]            # 中桶 0%
        + [("bearish", 0.90, "strong_bullish") for _ in range(3)]            # 高桶 0%
    )
    llm = _make_calib(store, "md1", spec)
    rep = run_eval(llm, store, set_version="md1")
    assert rep["calibration"]["shape"] == "monotonic_decreasing"
    assert rep["calibration"]["ok"] is False
    assert "单调递减" in rep["calibration"]["note"]


def test_calibration_excluded_bucket_reported(store):
    """样本不足的桶从判定中排除，但要单列，不能静默丢弃。"""
    spec = (
        [("strong_bullish", 0.60, "strong_bullish") for _ in range(3)]       # 低桶 3 条
        + [("strong_bullish", 0.90, "strong_bullish") for _ in range(3)]     # 高桶 3 条
        + [("bearish", 0.75, "strong_bullish") for _ in range(1)]            # 中桶仅 1 条
    )
    llm = _make_calib(store, "eb1", spec)
    rep = run_eval(llm, store, set_version="eb1")
    cal = rep["calibration"]
    assert cal["excluded_buckets"], "样本不足的桶必须被列出"
    assert cal["excluded_buckets"][0]["bucket"] == "0.70-0.85"
    assert cal["excluded_buckets"][0]["samples"] == 1
    # 判定只用两个满桶（都在 1.0），形态为递增
    assert cal["shape"] == "monotonic_increasing"


def test_calibration_cause_is_coverage_gap(store):
    """高置信误判全落在金标未覆盖档 -> 归因「覆盖缺口」，不是模型标定失准。

    这是 gold-v6 的真实情形：金标缺 strong_*，模型判强档全被记为错，
    而模型对强档最自信，于是高置信桶塌陷。
    """
    # 金标只有 bullish，无 strong_bullish
    for i in range(6):
        store.upsert_gold(set_version="cg1", doc_id=f"g{i}", as_of="2026-10-18",
                          expected_band="bullish", text=TEXT)
    # 全判 strong_bullish（未覆盖档），高置信
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(6)])
    rep = run_eval(llm, store, set_version="cg1")
    cause = rep["calibration"]["cause"]
    assert cause["kind"] == "coverage_gap"
    assert cause["coverage_gap"] == 6
    assert cause["miscalibration"] == 0
    assert "评测表缺档" in cause["note"]


def test_calibration_cause_is_miscalibration(store):
    """金标五档齐全、高置信仍判错 -> 归因「标定失准」。"""
    for band in ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"):
        store.upsert_gold(set_version="ms1", doc_id=f"m_{band}", as_of="2026-10-18",
                          expected_band=band, text=TEXT)
    # 全部高置信地判成 strong_bullish：strong_bullish 那条对，其余四条错
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(5)])
    rep = run_eval(llm, store, set_version="ms1")
    cause = rep["calibration"]["cause"]
    assert cause["kind"] == "miscalibration"
    assert cause["coverage_gap"] == 0
    assert cause["miscalibration"] == 4
    assert "失准" in cause["note"]


def test_calibration_cause_both(store):
    """覆盖缺口与标定失准并存时给出各自占比。"""
    # 金标：bullish / bearish（缺 strong_*）
    for i in range(3):
        store.upsert_gold(set_version="bt1", doc_id=f"b{i}", as_of="2026-10-18",
                          expected_band="bullish", text=TEXT)
    for i in range(3):
        store.upsert_gold(set_version="bt1", doc_id=f"c{i}", as_of="2026-10-18",
                          expected_band="bearish", text=TEXT)
    # 前三条判 strong_bullish（未覆盖档），后三条判 bullish（覆盖档但错）
    llm = FakeLLM(
        [good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(3)]
        + [good_response(QUOTE, 0.3, "bullish", 0.9) for _ in range(3)]
    )
    rep = run_eval(llm, store, set_version="bt1")
    cause = rep["calibration"]["cause"]
    assert cause["kind"] == "both"
    assert cause["coverage_gap"] == 3
    assert cause["miscalibration"] == 3
    assert "50%" in cause["note"]


def test_calipers_flags_adjacent_bias_when_coverage_gap(store):
    """金标有未覆盖档时，±1 档口径必须标明「系统性偏高、不可单独作证据」。"""
    # 混合：前三条判 bullish（覆盖档、命中），后三条判 strong_bullish（未覆盖档）
    for i in range(6):
        store.upsert_gold(set_version="cb1", doc_id=f"q{i}", as_of="2026-10-18",
                          expected_band="bullish", text=TEXT)
    llm = FakeLLM(
        [good_response(QUOTE, 0.3, "bullish", 0.9) for _ in range(3)]
        + [good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(3)]
    )
    rep = run_eval(llm, store, set_version="cb1")
    cal = rep["calipers"]
    assert cal["adjacent_biased_direction"] == "high"
    assert "系统性偏高" in cal["adjacent_note"]
    assert "不可单独作为质量证据" in cal["adjacent_note"]
    assert cal["exact_biased_direction"] == "low"
    assert "首选口径" in cal["direction_note"]


def test_calipers_no_gap_uses_general_note(store):
    """五档齐全时 ±1 档的提示改为一般性宽松说明，不再说「系统性偏高」。"""
    for band in ("strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"):
        store.upsert_gold(set_version="cb2", doc_id=f"r_{band}", as_of="2026-10-18",
                          expected_band=band, text=TEXT)
    llm = FakeLLM([
        good_response(QUOTE, -0.7, "strong_bearish", 0.9),   # r_bearish
        good_response(QUOTE, -0.3, "bearish", 0.9),          # r_bullish  -> 错
        good_response(QUOTE, 0.0, "neutral", 0.9),           # r_neutral
        good_response(QUOTE, -0.7, "strong_bearish", 0.9),   # r_strong_bearish
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),    # r_strong_bullish
    ])
    rep = run_eval(llm, store, set_version="cb2")
    cal = rep["calipers"]
    assert cal["adjacent_biased_direction"] == "neutral"
    assert "系统性偏高" not in cal["adjacent_note"]
    assert "偏宽松" in cal["adjacent_note"]


def test_batch_and_document_use_same_default_prompt(store):
    """批量入口与单篇入口在不传 prompt 时必须用同一版本。

    实测缺陷：score_batch 没透传 prompt，batch.py 兜底 load_prompt("v1")，
    于是批量分数全以 v1 落库（生产库 12 条），与单篇入口版本不同。
    落库的 prompt_version 是幂等键要素，版本分叉等于两批分数不可比。
    """
    from fd_sentiment.prompt import default_version

    llm = FakeLLM([good_response(QUOTE), good_response(QUOTE)])
    r1 = score_document(llm, ScoreRequest("doc-single", TEXT, "2026-10-18"), store=store)
    out = score_batch(llm, [ScoreRequest("doc-batch", TEXT, "2026-10-18")],
                      store=store, max_concurrency=1)
    r2 = out["results"][0]["record"]

    assert r1["prompt_version"] == default_version()
    assert r2["prompt_version"] == r1["prompt_version"]
    assert out["prompt_version"] == r1["prompt_version"]
    assert r1["prompt_version"] != "v1", "缺省不得静默回退到 v1"


def test_default_prompt_is_not_historical_version():
    """服务缺省版本必须等于 DEFAULT_VERSION，不得是硬编码的历史版本。

    实测缺陷：server.py 用 os.getenv(..., "v4") 就地兜底，未设环境变量时
    对外自称 v4，而 spec 声明的评测口径是 v5。
    2026-10-22 起缺省为 v6（重标 gold-v9 上 exact 91.26% / 方向 99.03%，
    优于 v5 的 87.38% / 93.20%，见 reports/gold-v9-trend-direction-2026-10-22.md）。
    """
    from fd_sentiment.prompt import DEFAULT_VERSION, default_prompt, default_version

    assert default_version() != "v1"
    assert default_prompt().version == DEFAULT_VERSION
    assert DEFAULT_VERSION == "v6", "缺省版本须与 spec 声明的评测口径一致"


def test_current_prompt_version_reports_consistency(monkeypatch):
    """current_prompt_version 必须报出「声明版本 == 实际使用版本」。"""
    import asyncio

    from fastmcp import Client

    from fd_sentiment.server import build_server

    monkeypatch.delenv("FD_SENTIMENT_PROMPT_VERSION", raising=False)

    async def run():
        app = build_server()
        async with Client(app) as c:
            return await c.call_tool("current_prompt_version", {})

    payload = (asyncio.run(run())).structured_content or {}
    assert payload.get("declared_matches_actual") is True
    assert payload.get("prompt_version") == payload.get("default_version")
    assert payload.get("env_override") is None


def test_eval_without_prompt_uses_default_not_v1(store):
    """评测不传 prompt 时也用服务缺省版本，不得评 v1。"""
    from fd_sentiment.prompt import default_version

    for band, doc in (("strong_bullish", "e1"), ("neutral", "e2")):
        store.upsert_gold(set_version="ev1", doc_id=doc, as_of="2026-10-18",
                          expected_band=band, text=TEXT)
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9),
                   good_response(QUOTE, 0.0, "neutral", 0.9)])
    rep = run_eval(llm, store, set_version="ev1")
    assert rep["prompt_version"] == default_version()
    assert rep["prompt_version"] != "v1"


def test_hit_rate_note_names_the_dominant_band(store):
    """总命中率是加权平均，报告必须指名哪一档在牵引它。

    spec `sentiment-calibration-verdict` 要求总命中率不得单独对外引用：
    一档样本占多数时，它的准确率就是总值的主要成分，单看总值会掩盖
    其余档的真实表现。
    """
    # 3 条 strong_bullish（全对）+ 1 条 neutral（全错）→ 主导档占 75%
    for i in range(3):
        store.upsert_gold(set_version="w1", doc_id=f"b{i}", as_of="2026-10-18",
                          expected_band="strong_bullish", text=TEXT)
    store.upsert_gold(set_version="w1", doc_id="n0", as_of="2026-10-18",
                      expected_band="neutral", text=TEXT)
    llm = FakeLLM([
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),
        good_response(QUOTE, 0.7, "strong_bullish", 0.9),
        good_response(QUOTE, 0.0, "strong_bullish", 0.9),
    ])
    rep = run_eval(llm, store, set_version="w1")
    note = rep["hit_rate_note"]
    assert "strong_bullish" in note
    assert "75%" in note
    assert "不得只引用总命中率" in note


def test_hit_rate_note_flags_thin_bands(store):
    """样本不足的档必须在总命中率附注里被点名。"""
    for i in range(12):
        store.upsert_gold(set_version="w2", doc_id=f"m{i}", as_of="2026-10-18",
                          expected_band="strong_bullish", text=TEXT)
    store.upsert_gold(set_version="w2", doc_id="thin", as_of="2026-10-18",
                      expected_band="bearish", text=TEXT)
    llm = FakeLLM([good_response(QUOTE, 0.7, "strong_bullish", 0.9) for _ in range(13)])
    rep = run_eval(llm, store, set_version="w2")
    assert "样本不足" in rep["hit_rate_note"]
    assert rep["by_band"]["insufficient"] == ["bearish"]
