"""假 LLM：不打网关，按预设脚本返回，用于存储/幂等/批量/校验的离线测试。"""
from __future__ import annotations

import json

from fd_sentiment.llm import LLMConfig


class FakeLLM:
    """实现 chat() 的最小替身。可脚本化返回，并记录调用次数。"""

    def __init__(self, responses=None, model="fake-model"):
        self.cfg = LLMConfig(base_url="http://fake", api_key="k", model=model)
        self.responses = list(responses or [])
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, max_tokens=1500, doc_id=None):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM 无预设响应")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, dict):
            return json.dumps(r, ensure_ascii=False)
        return str(r)


def good_response(quote: str, score=0.7, band="strong_bullish", confidence=0.9):
    return {
        "score": score,
        "band": band,
        "confidence": confidence,
        "evidence_spans": [{"quote": quote, "char_start": 0, "char_end": len(quote)}],
    }
