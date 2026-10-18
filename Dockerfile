# fd-sentiment image — 柏讯(wire) 文档情绪打分 MCP。
#
# 发布通道：GitHub Actions -> TCR（.github/workflows/image.yml）。
# 本服务与计算面服务的最大区别：**运行时依赖 LLM 网关**与**持久化库**。
#   - LLM：FD_SENTIMENT_LLM_BASE_URL / _API_KEY / _MODEL（部署期注入，不进镜像）
#   - 库：FD_SENTIMENT_DB 必须指向持久卷（幂等缓存、金标、评测结果都在里面，
#         跟 pod 生命周期绑在一起会让"同键不重打"的承诺失效）
FROM python:3.12-slim

ARG PIP_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FD_SENTIMENT_DB=/data/fd_sentiment.db

WORKDIR /app

COPY pyproject.toml ./
COPY fd_sentiment ./fd_sentiment

RUN if [ -n "$PIP_INDEX_URL" ]; then pip install --no-cache-dir -i "$PIP_INDEX_URL" ".[server]"; else pip install --no-cache-dir ".[server]"; fi \
 && useradd -m -u 1000 appuser \
 && mkdir -p /data && chown appuser:appuser /data

USER appuser

EXPOSE 8080

# HTTP 传输（MCP streamable-http）；stdio 由客户端以 `python -m fd_sentiment.server` 拉起
CMD ["python", "-m", "fd_sentiment.server", "--http", "8080"]
