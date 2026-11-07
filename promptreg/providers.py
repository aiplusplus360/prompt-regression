"""Model providers.

Two are shipped:

  EchoProvider              deterministic, no network. Used by the test suite
                            and by `--provider echo` so the CLI is demoable
                            with no key and no cost.
  OpenAICompatibleProvider  any endpoint speaking the OpenAI chat-completions
                            shape: OpenAI, DeepSeek, Groq, Mistral, Ollama,
                            OpenRouter, vLLM, LM Studio.

Kept on `urllib` rather than a vendor SDK on purpose — a prompt-testing tool
that pulls in a 40MB client library for one POST is a bad trade, and pinning
to the OpenAI SDK makes every other provider second class.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str: ...


class ProviderError(RuntimeError):
    """A provider failed. Retried by `complete_with_retry` unless permanent."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        # 429 and 5xx are transient. Everything else (auth, bad request) will
        # fail identically on the next attempt, so retrying only wastes time.
        if self.status is None:
            return True
        return self.status == 429 or 500 <= self.status < 600


@dataclass
class EchoProvider:
    """Returns the prompt back, prefixed. Deterministic by construction."""

    name: str = "echo"

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        body = prompt.strip()
        return f"{body}\n\n(echo)" if body else "(echo)"


@dataclass
class OpenAICompatibleProvider:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0
    host_header: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = f"openai-compatible:{self.model}"
        self.base_url = self.base_url.rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                **({"Host": self.host_header} if self.host_header else {}),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # noqa: PERF203 - one call site
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(
                f"{self.name} returned HTTP {exc.code}: {detail}", status=exc.code
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.name} unreachable: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.name} returned an unexpected body: {str(body)[:300]}"
            ) from exc


def get_provider(
    kind: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Provider:
    """Build a provider from CLI flags."""
    if kind == "echo":
        return EchoProvider()
    if kind in {"openai", "compatible"}:
        if not model:
            raise SystemExit("--model is required for a live provider")
        return OpenAICompatibleProvider(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=temperature,
        )
    raise SystemExit(f"unknown provider: {kind!r} (expected 'echo' or 'openai')")


def complete_with_retry(
    provider: Provider,
    prompt: str,
    *,
    system: Optional[str] = None,
    attempts: int = 3,
    backoff: float = 1.5,
) -> str:
    """Retry transient failures with exponential backoff.

    Permanent failures raise immediately: retrying an auth error three times
    turns a clear "your key is wrong" into a slow "your key is wrong".
    """
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return provider.complete(prompt, system=system)
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == attempts:
                raise
            time.sleep(backoff ** attempt)
    raise ProviderError(str(last) if last else "provider failed")
