from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import request

from ..config import ModelConfig


@dataclass
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLM(Protocol):
    last_usage: LLMUsage | None

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class OllamaLLM:
    config: ModelConfig
    last_usage: LLMUsage | None = None

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.request_timeout) as resp:
            body = resp.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            self.last_usage = None
            return body
        prompt_tokens = _to_int(parsed.get("prompt_eval_count"))
        completion_tokens = _to_int(parsed.get("eval_count"))
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        self.last_usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return parsed.get("response", "")


@dataclass
class DeepSeekLLM:
    config: ModelConfig
    last_usage: LLMUsage | None = None

    def generate(self, prompt: str) -> str:
        # DeepSeek uses an OpenAI-compatible Chat Completions API.
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.config.api_base_url.rstrip("/") + "/chat/completions"
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.request_timeout) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        self.last_usage = LLMUsage(
            prompt_tokens=_to_int(usage.get("prompt_tokens")),
            completion_tokens=_to_int(usage.get("completion_tokens")),
            total_tokens=_to_int(usage.get("total_tokens")),
        )
        return parsed["choices"][0]["message"]["content"]


@dataclass
class DummyLLM:
    """Mock LLM used for smoke tests (no real model call)."""
    response: str = "OK"
    last_usage: LLMUsage | None = None

    def generate(self, prompt: str) -> str:
        completion_tokens = len(self.response.split())
        self.last_usage = LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=completion_tokens,
            total_tokens=len(prompt.split()) + completion_tokens,
        )
        return self.response


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
