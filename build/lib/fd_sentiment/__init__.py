"""fd-sentiment：可审计的文档情绪打分服务。

核心模块：
  models      —— SentimentScore / EvidenceSpan 数据形状（与 spec 逐字段对齐）
  storage     —— SQLite 存储（scores 表幂等键唯一约束 / 金标表 / 评测结果表）
  prompt      —— 版本化 prompt 工件（含五档定义与分块合并规则）
  llm         —— OpenAI 兼容网关客户端（低温、重试上限、明确错误）
  scoring     —— 单文档打分 + evidence_spans 强制校验 + 分块合并
  batch       —— 批量/增量（并发上限、单条超时、逐条报错）
  aggregate   —— 分布统计视图（明确非指数）
  evaluate    —— 金标评测 + 校准报告
  sanitize    —— 第三方文本 flatten + 包裹（防提示注入）
"""

__version__ = "0.1.0"
