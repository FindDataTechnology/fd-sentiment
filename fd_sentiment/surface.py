"""工具面真值汇总点：manifest 与运行时注册共用这一份清单。

为什么单独一层：selfcheck 必须问"服务真的注册了什么"，而不是"我们打算注册
什么"。这里 `all_surface_tools()` 与 `server.build_server()` 用的是同一份工具
定义（都从 TOOLS 元数据生成），故"能列出"与"能服务"必然是同一份清单。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str
    level: str  # commercial | internal
    summary: str


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("score_document", "scoring", "commercial",
             "单文档打分：SentimentScore 全字段；evidence_spans 非空且 quote 为原文逐字子串；无证据拒绝"),
    ToolSpec("score_batch", "scoring", "commercial",
             "批量打分：单条失败不中断、逐条报错、并发上限、单条超时；增量跳过已打分"),
    ToolSpec("get_score", "query", "commercial",
             "按 doc_id 取最近一条分数（可按 prompt_version 过滤）"),
    ToolSpec("list_scores", "query", "commercial",
             "列分数记录（doc_id / prompt_version / 日期范围过滤）"),
    ToolSpec("aggregate_scores", "aggregate", "commercial",
             "区间聚合：band 计数/均值分/均值置信度 + 覆盖说明；明确非情绪指数"),
    ToolSpec("current_prompt_version", "meta", "commercial",
             "当前 prompt 版本、摘要与模型配置"),
    ToolSpec("run_evaluation", "eval", "commercial",
             "金标评测：命中率 + 按置信度分桶的校准表；校准失败显式标注"),
    ToolSpec("compare_evaluations", "eval", "commercial",
             "并列对比多个评测结果（版本对比）"),
)


def all_surface_tools() -> list[str]:
    return [t.name for t in TOOLS]


def tools_by_group() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in TOOLS:
        out.setdefault(t.group, []).append(t.name)
    return out
