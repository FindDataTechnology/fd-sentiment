# fd-sentiment — 柏讯文档情绪打分服务

对公告/新闻文本产出**可审计**的结构化情绪分数。三件事是这个服务的全部：

1. **可验证**：每个分数必须带 `evidence_spans`（原文逐字引用 + 字符区间）。无证据 → 拒绝打分，不返回空证据的分数。
2. **可复现**：幂等键 `(doc_id, content_hash, prompt_version, model)`。同键重复请求返回既有分数、不重复调模型；换 prompt 版本或换模型产生**新记录**而非覆盖。
3. **可评测**：金标集 + 命中率 + **按置信度分桶的校准**。高置信桶准确率低于低置信桶即判"校准失败"并显式标注——因为下游会按 confidence 加权，不可信的 confidence 比没有更糟。

## 快速开始

```bash
pip install -e ".[server,test]"
pytest -q                                    # 38 项全离线（假 LLM）

export FD_SENTIMENT_LLM_BASE_URL=https://token.finddatatech.cloud/v1
export FD_SENTIMENT_LLM_API_KEY=sk-...
export FD_SENTIMENT_MODEL=deepseek-v4.1-flash
export FD_SENTIMENT_DB=/tmp/fd_sentiment.db

python -m fd_sentiment.server            # stdio
python -m fd_sentiment.server --http 8080  # HTTP
```

## 工具面（8 工具 / 5 组，全 commercial）

| 工具 | 作用 |
|---|---|
| `score_document` | 单文档打分（SentimentScore 全字段） |
| `score_batch` | 批量：单条失败不中断、逐条报错、增量跳过 |
| `get_score` / `list_scores` | 取/列分数（版本与模型并存不覆盖） |
| `aggregate_scores` | 分布统计（**明确非情绪指数**） |
| `current_prompt_version` | 当前 prompt 版本与模型 |
| `run_evaluation` / `compare_evaluations` | 金标评测（含校准）与版本对比 |

`python scripts/selfcheck_surface.py` 断言 `tool-surface.json` ↔ 运行时真实注册表一致，并做反向自检（摘条目必须报错）。

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `FD_SENTIMENT_LLM_BASE_URL` | — | OpenAI 兼容网关（必填） |
| `FD_SENTIMENT_LLM_API_KEY` | — | 网关令牌（必填） |
| `FD_SENTIMENT_MODEL` | — | 模型名（必填） |
| `FD_SENTIMENT_TEMPERATURE` | `0` | 低温：分数要可复现 |
| `FD_SENTIMENT_MAX_TOKENS` | `4000` | 推理型模型需调大（推理会吃预算） |
| `FD_SENTIMENT_CHUNK_CHARS` | `8000` | 分块字符阈值 |
| `FD_SENTIMENT_DB` | `./fd_sentiment.db` | 库路径（生产必须指持久卷） |

## 当前状态（2026-10-17）

- 服务实现完成，**39 项测试全绿**，真实网关/幂等/断网/真实公告端到端均实测。
- 金标集 `gold-v5`（60 条，bullish 24 / neutral 24 / bearish 12），标注规则是显式人工规则表。
- prompt 迭代（gold-v5 同尺对比）：

  | | v1 | v2 | v3 |
  |---|---|---|---|
  | 命中率 | 0.412 | 0.556 | **0.591** |
  | 高置信桶准确率 | 0.154 | 0.333 | **0.556** |
  | 强档误判合计 | 21 | 19 | **4** |
  | 样本数 | 51 | 54 | **22** |
  | 校准 | 失败 | 失败 | 样本不足，无法判定 |

  v3 把量化门槛**绑到分数区间**（未达强档则 score 必须落在 (-0.6, 0.6)），并新增
  `strong_tier_check` 字段让门槛可审计——服务端会核查"申报未达强档却给 |score|>=0.6"
  的自相矛盾并拒绝该次打分。效果明确：强档误判 19→4。
  但 v3 样本仅 22/60（38 条失败**全部是网关 503**，非 prompt 问题），
  每置信桶只有个位数样本，**校准判定统计上不成立**。

  **结论：v1/v2 不合格；v3 方向正确但样本不足待验。三代均不可作为下游 pack 的输入。**
  下一步：网关稳定后用幂等补跑失败条目，拿到 ≥50 样本再判定 v3 的校准。
- 上牌：surface 与 selfcheck 已完成，网关登记待部署（镜像与 CI 已就位）。
