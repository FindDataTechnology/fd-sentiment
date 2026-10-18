"""LLM 客户端：OpenAI 兼容网关，低温、重试上限、明确错误。

设计要点：
  - 走既有 token 网关（不新接入 provider）；base_url / api_key / model 全部可配。
  - temperature 固定低温（0.0）：分数要可复现，采样波动是噪声不是多样性。
  - 重试上限显式（默认 2 次），指数退避；断网/超时给出明确错误而不是挂起——
    批量场景里"挂住"比"失败"更糟（占着并发位）。
  - 每次调用都记进 llm_calls（幂等验收要用调用计数）。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


class LLMError(RuntimeError):
    """LLM 调用失败（网络/超时/非 200/响应不可解析）。"""


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    # 推理型模型（deepseek-v4.1-flash 等）先输出 reasoning_content，预算太小会把
    # 最终 JSON 截断——默认给足；换非推理模型可调小。
    max_tokens: int = 32000
    timeout_s: float = 120.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("FD_SENTIMENT_LLM_BASE_URL", os.getenv("LLM_GATEWAY_BASE_URL", "")),
            api_key=os.getenv("FD_SENTIMENT_LLM_API_KEY", os.getenv("LLM_GATEWAY_API_KEY", "")),
            model=os.getenv("FD_SENTIMENT_MODEL", ""),
            temperature=float(os.getenv("FD_SENTIMENT_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("FD_SENTIMENT_MAX_TOKENS", "32000")),
            timeout_s=float(os.getenv("FD_SENTIMENT_TIMEOUT_S", "120")),
            max_retries=int(os.getenv("FD_SENTIMENT_MAX_RETRIES", "2")),
        )

    def require_ready(self) -> None:
        missing = [k for k in ("base_url", "api_key", "model") if not getattr(self, k)]
        if missing:
            raise LLMError(
                "LLM 配置不全：" + ", ".join(missing)
                + "（设置 FD_SENTIMENT_LLM_BASE_URL / FD_SENTIMENT_LLM_API_KEY / FD_SENTIMENT_MODEL）"
            )


class LLMClient:
    def __init__(self, cfg: LLMConfig, store=None):
        self.cfg = cfg
        self.store = store

    def _post(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]:
        self.cfg.require_ready()
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                with httpx.Client(timeout=self.cfg.timeout_s) as client:
                    resp = client.post(url, json=payload, headers=headers)
                latency = int((time.monotonic() - started) * 1000)
                if resp.status_code != 200:
                    raise LLMError(f"网关返回 {resp.status_code}：{resp.text[:300]}")
                body = resp.json()
                self._log(ok=True, latency_ms=latency)
                return body
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                latency = int((time.monotonic() - started) * 1000)
                last_err = exc
                self._log(ok=False, latency_ms=latency, error=f"{type(exc).__name__}: {exc}")
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise LLMError(f"LLM 网关不可达（已重试 {self.cfg.max_retries} 次）：{exc}") from exc
            except LLMError as exc:
                latency = int((time.monotonic() - started) * 1000)
                last_err = exc
                self._log(ok=False, latency_ms=latency, error=str(exc))
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise
        raise LLMError(f"LLM 调用失败：{last_err}")

    def _log(self, *, ok: bool, latency_ms: int, error: str | None = None, doc_id: str | None = None, purpose: str = "score") -> None:
        if not self.store:
            return
        try:
            self.store.log_llm_call(
                doc_id=doc_id, model=self.cfg.model, purpose=purpose, ok=ok,
                latency_ms=latency_ms, error=error,
                created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception:  # 记账失败不该掩盖主流程
            pass

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None, doc_id: str | None = None) -> str:
        """返回 assistant 文本。调用记账挂在这里，保证每次真实调用都进 llm_calls。

        兼容推理型模型（deepseek-v4.1-flash 等）：这类模型先输出 reasoning_content，
        若 max_tokens 被推理吃光，content 会是空串或 None。此时按以下顺序取：
          ① message.content 非空 → 用它；
          ② 否则取 message.reasoning_content（推理里往往已含最终 JSON）；
          ③ 两者皆空 → 明确报错（不返回空串让上层误判成"模型没话说"）。
        """
        body = self._post(messages, max_tokens=max_tokens)
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"网关响应缺少 choices[0].message：{json.dumps(body)[:300]}") from exc
        if not isinstance(msg, dict):
            raise LLMError("网关响应的 message 不是对象")
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            # 推理内容里含最终 JSON 时可直接用；不含则由上层 extract_json_block 报错
            return reasoning
        finish = body["choices"][0].get("finish_reason")
        raise LLMError(
            f"模型未返回可用文本（content 与 reasoning_content 均为空；finish_reason={finish!r}）。"
            "推理型模型可能被 max_tokens 截断——调大 max_tokens 或换非推理模型。"
        )


def extract_json_block(text: str) -> dict[str, Any]:
    """从模型输出里取 JSON 对象（容忍 ```json 围栏、前后说明、推理片段）。

    推理型模型的输出里可能含**多个**花括号片段（中间步骤 + 最终答案）。
    取"首个 { 到末个 }"会切出非法串，因此按括号深度扫描出所有顶层对象，
    **从后往前**尝试解析——最终答案通常在最后。返回第一个解析成功且是
    dict 的对象。
    """
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
            if t.lower().startswith("json"):
                t = t[4:]

    candidates: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(t[start : i + 1])
                    start = -1

    parsed: list[dict[str, Any]] = []
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)

    # 优先取"打分结果"形状的对象（含 score 或 refuse 键）——推理文本里常有
    # 内层 evidence span 对象，从后往前取会误取到它。多个候选时取最后一个
    # 带 score/refuse 的（最终答案通常在末尾）。
    for obj in reversed(parsed):
        if "score" in obj or obj.get("refuse") is True:
            return obj
    if parsed:
        return parsed[-1]
    raise LLMError(f"模型输出中没有可解析的 JSON 对象：{text[:200]!r}")
