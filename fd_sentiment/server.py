"""fd-sentiment MCP 服务入口（stdio + HTTP 双传输）。

启动：
    python -m fd_sentiment.server              # stdio（MCP 默认）
    python -m fd_sentiment.server --http 8080  # HTTP（streamable-http）

工具命名：动词_名词。每个工具描述自足（evidence 必填、无证据拒绝、幂等键、as_of）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from . import __version__
from .aggregate import aggregate as _aggregate
from .batch import score_batch as _score_batch_impl, requests_from_rows
from .evaluate import compare_runs as _compare_runs, run_eval as _run_eval
from .llm import LLMClient, LLMConfig
from .models import ScoreError
from .prompt import available_versions, load_prompt
from .scoring import ScoreRequest, score_document as _score_document_impl
from .storage import Store

SERVER_NAME = "fd-sentiment"

DEFAULT_DB = os.getenv("FD_SENTIMENT_DB", str(Path.cwd() / "fd_sentiment.db"))


def _store() -> Store:
    return Store(DEFAULT_DB)


def _client(store: Store | None = None) -> LLMClient:
    return LLMClient(LLMConfig.from_env(), store=store)


def build_server() -> FastMCP:
    app = FastMCP(
        name=SERVER_NAME,
        instructions=(
            "fd-sentiment 文档情绪打分服务：对公告/新闻文本产出可审计的结构化分数。"
            "分数带连续值 score（-1..+1）、五档 band、confidence 与 evidence_spans"
            "（原文逐字引用 + 字符区间）。无原文证据时拒绝打分（不返回无证据分数）。"
            "幂等键 (doc_id, content_hash, prompt_version, model)：重复请求返回既有分数、"
            "不重复调用模型。as_of 为文档发布日期——服务只依据文档本身，不接收市场数据。"
            "聚合视图是已打分文档的分布统计，不是情绪指数。"
        ),
    )

    DEFAULT_PROMPT = os.getenv("FD_SENTIMENT_PROMPT_VERSION", "v4")

    @app.tool
    def score_document(doc_id: str, text: str, as_of: str, doc_type: str | None = None) -> dict:
        """对单篇文档打分。

        doc_id 稳定标识（如公告 ID）；text 文档正文（服务内部会 flatten 并包裹防注入）；
        as_of 文档发布日期 ISO（YYYY-MM-DD）；doc_type 可选分类（用于聚合视图）。
        返回 SentimentScore：score/band/confidence/evidence_spans/model/prompt_version/as_of/scored_at。
        evidence_spans 非空且 quote 为原文逐字子串；文本无情绪证据时本工具报错拒绝，不返回分数。
        幂等：同 (doc_id, content_hash, prompt_version, model) 返回既有记录（created=false）。
        """
        store = _store()
        try:
            rec = _score_document_impl(_client(store), ScoreRequest(doc_id=doc_id, text=text, as_of=as_of, doc_type=doc_type), store=store, prompt=load_prompt(DEFAULT_PROMPT))
            return rec
        except ScoreError as exc:
            return {"error": "refused", "reason": str(exc)}

    @app.tool
    def score_batch(docs: list[dict], max_concurrency: int = 4, per_doc_timeout_s: float = 180.0, skip_existing: bool = True) -> dict:
        """批量打分。docs 为 [{doc_id, text, as_of, doc_type?}] 列表。

        单条失败不中断：结果逐条带 status(ok|skipped|failed) 与原因；
        skip_existing=True 时同版本已打分文档跳过并计入 skipped（不调用模型）——
        幂等键 (doc_id, content_hash, prompt_version, model)，与单条入口同一把钥匙。
        每条成功记录同样含 as_of（文档发布日期）与 evidence_spans（非空、quote 为
        原文逐字子串）；无证据的文档计入 failed，不返回空证据分数。
        返回 {total, ok, skipped, failed, results, prompt_version, model}。
        """
        store = _store()
        reqs = requests_from_rows(docs)
        return _score_batch_impl(_client(store), reqs, store=store, max_concurrency=max_concurrency,
                            per_doc_timeout_s=per_doc_timeout_s, skip_existing=skip_existing)

    @app.tool
    def get_score(doc_id: str, prompt_version: str | None = None) -> dict:
        """按 doc_id 取最近一条分数（可按 prompt_version 过滤）。

取的是**已落库**的既有分数——幂等键 (doc_id, content_hash, prompt_version, model)。
不调用模型。返回的记录含 evidence_spans（非空，quote 为原文逐字子串）与 as_of
（文档发布日期）。无记录时返回 {found: false}，不返回空分数字段。"""
        store = _store()
        rows = store.list_scores(doc_id=doc_id, prompt_version=prompt_version, limit=1)
        return rows[0] if rows else {"found": False, "doc_id": doc_id}

    @app.tool
    def list_scores(doc_id: str | None = None, prompt_version: str | None = None,
                    since: str | None = None, until: str | None = None, limit: int = 200) -> dict:
        """列分数记录（可按 doc_id / prompt_version / since / until 过滤）。返回 {count, items}。

每条含 as_of（文档发布日期）与 evidence_spans。同一文档在不同 prompt_version /
model 下的分数**并存不覆盖**（幂等键含版本与模型，故换版是新增而非改写）。
只读，不调用模型。"""
        store = _store()
        rows = store.list_scores(doc_id=doc_id, prompt_version=prompt_version,
                                 since=since, until=until, limit=limit)
        return {"count": len(rows), "items": rows}

    @app.tool
    def aggregate_scores(since: str | None = None, until: str | None = None,
                         doc_type: str | None = None, prompt_version: str | None = None,
                         universe_total: int | None = None) -> dict:
        """区间聚合：各 band 计数、平均分、平均置信度 + 覆盖说明。

        这是**已打分文档的分布统计，不是情绪指数**（不用于行情或交易判断）。
        universe_total 给出时回报覆盖率（已打分/范围内总数）；未知时为 null，不推断。
        统计基线是已落库分数（幂等键去重后的记录），按 as_of 落在 [since, until] 区间。
        每篇入统的分数都带 evidence_spans（quote 为原文逐字子串）——分布来自可复核的
        判断，不是印象。不含市场数据维度（本服务不取行情）。
        """
        return _aggregate(_store(), since=since, until=until, doc_type=doc_type,
                          prompt_version=prompt_version, universe_total=universe_total)

    @app.tool
    def current_prompt_version() -> dict:
        """查询当前 prompt 版本与模型配置（与分数记录中的 prompt_version / model 字段一致）。

        返回版本标识、工件摘要（prompt 换版即变）、可用版本列表、模型名与温度。
        prompt_version 与 model 是幂等键四要素之二——换版或换模型都会产生新分数记录
        而非覆盖旧记录（分数记录里的 as_of 是文档发布日期，与本服务版本无关）。
        本工具只报版本与配置，不产出分数与 evidence_spans（那是打分入口的事，
        evidence 必填与"无证据拒绝"两条纪律见 score_document / score_batch）。"""
        cfg = LLMConfig.from_env()
        art = load_prompt(DEFAULT_PROMPT)
        return {"prompt_version": art.version, "prompt_digest": art.digest,
                "available_versions": available_versions(), "model": cfg.model,
                "temperature": cfg.temperature, "env_override":
                    os.getenv("FD_SENTIMENT_PROMPT_VERSION")}

    @app.tool
    def run_evaluation(
        set_version: str, prompt_version: str | None = None, concurrency: int = 1
    ) -> dict:
        """对金标集跑评测：命中率 + 按置信度分桶的准确率（校准）。

        高置信桶准确率低于低置信桶即**校准失败**，报告显式标注。
        结果持久化（含 prompt_version 与金标集 set_version），返回报告含 run_id。
        评测走同一条打分链路：evidence_spans 必填（quote 为原文逐字子串）、无原文
        证据的样本计入 failures 而非命中率分母；样本按 as_of（文档发布日期）组织，
        换 prompt 版本即换一把幂等键，故不同版本的评测结果互不覆盖、可直接对比。
        样本不足时如实标注"不足以判定校准"，不伪造结论。
        prompt_version 省略时用服务当前默认版本（与 score_document 同一把），
        避免"评的是 v1、跑的是 v3"这种版本错配。

        concurrency 默认 1（串行）。并发会把 60 条从约 15 分钟压到约 5 分钟，
        但并发下网关超时更多，而超时样本计入 failures 不进分母——两份报告的
        命中率不能直接当同一次测量比较。报告里的 concurrency 字段记录该轮用的是
        哪个值，compare_evaluations 会一并呈现。
        """
        store = _store()
        art = load_prompt(prompt_version or DEFAULT_PROMPT)
        return _run_eval(
            _client(store), store, set_version=set_version, prompt=art,
            concurrency=max(1, int(concurrency)),
        )

    @app.tool
    def compare_evaluations(run_ids: list[str]) -> dict:
        """并列对比多个评测结果（prompt 版本 / 模型对比用）。返回 {runs: [...]}。

        每个 run 带 prompt_version、model（与幂等键同一组维度，故换版对比才有意义）、
        set_version、samples、hit_rate 与 calibration_ok。samples 为 0 的 run 一并
        列出（如实呈现，不隐藏失败轮次）；评测样本按 as_of（文档发布日期）组织。
        本工具只读评测结果，不产出 evidence_spans——证据纪律由被对比的评测轮
        各自遵守（见 run_evaluation）。"""
        return _compare_runs(_store(), run_ids)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fd-sentiment MCP server")
    parser.add_argument("--http", type=int, default=None, help="HTTP 端口（默认 stdio）")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    app = build_server()
    if args.http:
        _start_health_server(app, args.http)
        app.run(transport="http", host="0.0.0.0", port=args.http)
    else:
        app.run()
    return 0


def _start_health_server(app: FastMCP, port: int) -> None:
    """起一个 /health 端点（k8s 探针用）。

    **不要拿 /mcp 当探针**：它只接 POST，GET 的 400/405 是协议正常形态，
    会把健康读成故障（fd-quant 的教训）。/health 只回答"进程活着且工具面
    已装配"——工具数为 0 时返回 503，让"工具面没装上"暴露成探针失败。
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            import asyncio

            tools = asyncio.run(app._list_tools())
            body = json.dumps({"status": "ok", "tools": len(tools)}, ensure_ascii=False)
            code = 200 if tools else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body.encode())))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):  # 静音访问日志（探针每 10 秒一条会淹没什么都没说的 stdout）
            pass

    def serve():
        srv = HTTPServer(("0.0.0.0", port + 1), Handler)
        srv.serve_forever()

    threading.Thread(target=serve, daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
