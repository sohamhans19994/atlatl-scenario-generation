"""Shared LLM configuration for OpenAI and OpenAI-compatible local servers."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal

Provider = Literal["openai", "llama-cpp"]
ThinkingMode = Literal["off", "high", "max"]


@dataclass(frozen=True)
class LLMConfig:
    provider: Provider
    model: str
    base_url: str | None = None
    context_window: int = 32_768
    max_output_tokens: int | None = None
    temperature: float = 0.1
    timeout: float = 500
    thinking: ThinkingMode = "off"

    def for_structured_output(
        self, *, max_output_tokens: int = 4_096
    ) -> LLMConfig:
        """Return a non-reasoning configuration for strict JSON tasks."""
        if max_output_tokens <= 0:
            raise ValueError("Structured output token limit must be positive")
        return replace(
            self,
            max_output_tokens=max_output_tokens,
            thinking="off",
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata without credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "thinking": self.thinking,
        }


@dataclass(frozen=True)
class CompletionResult:
    """Minimal response shape consumed by ``safe_llm_complete``."""

    text: str

    def __str__(self) -> str:
        return self.text


class OpenAICompatibleLLM:
    """Text-only client for llama.cpp's OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        if not config.base_url:
            raise RuntimeError(
                "--llm-base-url is required when --llm-provider=llama-cpp."
            )
        self.model = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.thinking = config.thinking
        self.client = OpenAI(
            base_url=config.base_url.rstrip("/"),
            api_key=os.getenv("LLAMA_CPP_API_KEY", "no-key"),
            timeout=config.timeout,
            max_retries=3,
        )

    def complete(self, prompt: str) -> CompletionResult:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_output_tokens is not None:
            request["max_tokens"] = self.max_output_tokens
        template_kwargs: dict[str, Any] = {
            "enable_thinking": self.thinking != "off"
        }
        if self.thinking != "off":
            template_kwargs["reasoning_effort"] = self.thinking
        request["extra_body"] = {"chat_template_kwargs": template_kwargs}
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        return CompletionResult(text=content or "")


def create_llm(config: LLMConfig) -> Any:
    """Create the configured LlamaIndex completion client."""
    common = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "timeout": config.timeout,
    }

    if config.provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when --llm-provider=openai."
            )
        return OpenAI(api_key=api_key, **common)

    if config.provider == "llama-cpp":
        return OpenAICompatibleLLM(config)

    raise ValueError(f"Unsupported LLM provider: {config.provider}")
