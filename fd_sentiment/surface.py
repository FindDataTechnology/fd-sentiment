"""工具面真值汇总点：manifest 与运行时注册共用这一份清单。

为什么单独一层：selfcheck 必须问"服务真的注册了什么"，而不是"我们打算注册
什么"。这里 `all_surface_tools()` 与 `server.build_server()` 用的是同一份工具
定义（都从 TOOLS 元数据生成），故"能列出"与"能服务"必然是同一份清单。

行为属性（只读 / 破坏性 / 幂等 / 是否触达外部世界）也在这里声明一次：
运行时注册与 tool-surface.json 都从这里取，避免"协议层说只读、manifest 说
有状态"这种两处口径。判据是**副作用**，不是名字像不像查询。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 只读：不改变任何持久状态。四个 flag 一次给全——缺一个客户端就得靠猜。
READ_ONLY: dict[str, Any] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

# 写库但**状态幂等**：同入参重复调用后库里仍是同一行（幂等键去重）。
# openWorldHint=True 是因为这条路真的会出网调 LLM 网关。
WRITE_IDEMPOTENT: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}

# 写库且**非幂等**：重复调用会追加新行（评测轮次）或返回不同的逐条状态。
WRITE_NOT_IDEMPOTENT: dict[str, Any] = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": True,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    group: str
    level: str  # commercial | internal
    summary: str
    annotations: dict[str, Any] = field(default_factory=lambda: dict(READ_ONLY))


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("score_document", "scoring", "commercial",
             "单文档打分：SentimentScore 全字段；evidence_spans 非空且 quote 为原文逐字子串；无证据拒绝",
             annotations=WRITE_IDEMPOTENT),
    # score_batch 不标幂等：幂等键 (doc_id, content_hash, prompt_version, model) 由内容
    # 决定，故重复提交**不会重复调用模型**、库里也不会多出行——但**返回不同**：
    # 第二次调用同一批文档时逐条状态由 ok 变成 skipped（skip_existing 缺省为真），
    # 且超时项在调用返回后才可能落库。客户端若把重试当无操作，就会把 skipped
    # 误读成"这批没打过分"。MCP 的幂等提示是给"能不能安全重放"用的，这里答案是
    # "重放安全但结果不同"，故必须标非幂等。
    ToolSpec("score_batch", "scoring", "commercial",
             "批量打分：单条失败不中断、逐条报错、并发上限、单条超时；增量跳过已打分",
             annotations=WRITE_NOT_IDEMPOTENT),
    ToolSpec("get_score", "query", "commercial",
             "按 doc_id 取最近一条分数（可按 prompt_version 过滤）"),
    ToolSpec("list_scores", "query", "commercial",
             "列分数记录（doc_id / prompt_version / 日期范围过滤）"),
    ToolSpec("aggregate_scores", "aggregate", "commercial",
             "区间聚合：band 计数/均值分/均值置信度 + 覆盖说明；明确非情绪指数"),
    ToolSpec("current_prompt_version", "meta", "commercial",
             "当前 prompt 版本、摘要与模型配置"),
    ToolSpec("run_evaluation", "eval", "commercial",
             "金标评测：命中率 + 按置信度分桶的校准表；校准失败显式标注",
             annotations=WRITE_NOT_IDEMPOTENT),
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


def annotations_for(name: str) -> dict[str, Any]:
    """取某工具的行为属性（返回副本：调用方改了不会污染全族）。"""
    for t in TOOLS:
        if t.name == name:
            return dict(t.annotations)
    raise KeyError(f"未登记的工具：{name!r}（surface.TOOLS 是唯一真值）")


def all_annotations() -> dict[str, dict[str, Any]]:
    """manifest 生成用：{工具名: 属性}。"""
    return {t.name: dict(t.annotations) for t in TOOLS}
