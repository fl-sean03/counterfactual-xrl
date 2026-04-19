"""LLM client wrappers (Anthropic and OpenAI) with caching, retries, cost tracking.

- ``AnthropicClient``, Claude via Anthropic SDK, with prompt caching.
- ``OpenAIClient``, GPT models via OpenAI SDK.
- ``MockClient``, deterministic offline stub for tests and no-key dev.
- ``make_client``, picks a real client based on env vars, mock otherwise.

The key is read from ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``; we never
read from or write to disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CallResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    raw: Any = None


# Rough USD pricing per 1M tokens. Update as providers change rates.
PRICING = {
    # Anthropic
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    "claude-opus-4-7": {"in": 15.00, "out": 75.00, "cache_read": 1.50, "cache_write": 18.75},
    # OpenAI
    "gpt-4o": {"in": 2.50, "out": 10.00, "cache_read": 1.25, "cache_write": 0.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60, "cache_read": 0.075, "cache_write": 0.0},
    "gpt-4.1": {"in": 2.00, "out": 8.00, "cache_read": 0.50, "cache_write": 0.0},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60, "cache_read": 0.10, "cache_write": 0.0},
    "gpt-4.1-nano": {"in": 0.10, "out": 0.40, "cache_read": 0.025, "cache_write": 0.0},
}


def estimate_cost(model: str, result: CallResult) -> float:
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (
        result.input_tokens * p["in"] / 1e6
        + result.output_tokens * p["out"] / 1e6
        + result.cache_read_tokens * p["cache_read"] / 1e6
        + result.cache_creation_tokens * p["cache_write"] / 1e6
    )


class _BaseRealClient:
    def __init__(
        self,
        model: str,
        max_tokens: int = 1024,
        cost_cap_usd: float = 10.0,
        log_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.cost_cap_usd = cost_cap_usd
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.total_cost: float = 0.0
        self.calls: int = 0

    def _log(self, system: str, user: str, result: CallResult) -> None:
        if not self.log_dir:
            return
        with open(self.log_dir / f"call_{self.calls:04d}.json", "w") as f:
            json.dump(
                {
                    "model": result.model,
                    "system": system,
                    "user": user,
                    "text": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_read_tokens": result.cache_read_tokens,
                    "cache_creation_tokens": result.cache_creation_tokens,
                    "cost_usd": result.cost_usd,
                },
                f,
                indent=2,
            )


class AnthropicClient(_BaseRealClient):
    """Claude via Anthropic SDK with prompt caching on the system prompt."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs: Any) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("anthropic SDK not installed") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        super().__init__(model=model, **kwargs)
        self._client = Anthropic()

    def call(self, system: str, user: str, cache_system: bool = True) -> CallResult:
        if self.total_cost >= self.cost_cap_usd:
            raise RuntimeError(f"Cost cap ${self.cost_cap_usd:.2f} reached")
        system_blocks = [{"type": "text", "text": system}]
        if cache_system:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user}],
                )
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Anthropic call failed: {last_err}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = getattr(resp, "usage", None)
        result = CallResult(
            text=text,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            model=self.model,
            raw=resp,
        )
        result.cost_usd = estimate_cost(self.model, result)
        self.total_cost += result.cost_usd
        self.calls += 1
        self._log(system, user, result)
        return result


class OpenAIClient(_BaseRealClient):
    """OpenAI Chat Completions client.

    OpenAI auto-caches on the backend for prompts ≥1024 tokens with stable
    prefixes; ``cache_system`` is a no-op here (kept for API parity).
    Usage fields expose ``prompt_tokens_details.cached_tokens`` which we
    record.
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("openai SDK not installed") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        super().__init__(model=model, **kwargs)
        self._client = OpenAI()

    def call(self, system: str, user: str, cache_system: bool = True) -> CallResult:
        if self.total_cost >= self.cost_cap_usd:
            raise RuntimeError(f"Cost cap ${self.cost_cap_usd:.2f} reached")
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=self.max_tokens,
                )
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"OpenAI call failed: {last_err}")
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = getattr(resp, "usage", None)
        cached = 0
        try:
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
        except Exception:
            pass
        result = CallResult(
            text=text,
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cache_read_tokens=cached,
            cache_creation_tokens=0,
            model=self.model,
            raw=resp,
        )
        result.cost_usd = estimate_cost(self.model, result)
        self.total_cost += result.cost_usd
        self.calls += 1
        self._log(system, user, result)
        return result


class MockClient:
    """Deterministic offline client for tests and no-API-key local dev."""

    def __init__(
        self, model: str = "mock", cost_cap_usd: float = 10.0, log_dir=None, **_: Any
    ) -> None:
        self.model = model
        self.cost_cap_usd = cost_cap_usd
        self.total_cost = 0.0
        self.calls = 0
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def call(self, system: str, user: str, cache_system: bool = True) -> CallResult:
        self.calls += 1
        payload = json.loads(user) if user.strip().startswith("{") else {}
        stats = payload.get("per_action_stats", [])
        chosen = payload.get("chosen_action", 0)
        chosen_stats = next(
            (s for s in stats if s.get("action") == chosen), stats[0] if stats else {}
        )
        alts = [s for s in stats if s.get("action") != chosen]
        best_alt = max(alts, key=lambda s: s.get("mean_return", -1e9)) if alts else {}
        sr = chosen_stats.get("success_rate", 0.0)
        cr = chosen_stats.get("collision_rate", 0.0)
        response = {
            "rationale": (
                f"Action {chosen} was chosen because its counterfactual "
                f"success rate of {sr:.2f} exceeds the alternatives "
                f"(best alternative: {best_alt.get('action', 'n/a')} at "
                f"{best_alt.get('success_rate', 0.0):.2f})."
            ),
            "counterfactual": (
                f"If the agent had taken action {best_alt.get('action', 'n/a')} "
                f"instead, the collision rate would have been "
                f"{best_alt.get('collision_rate', 0.0):.2f}."
            ),
            "confidence": min(1.0, max(0.0, sr)),
            "claims": [
                {
                    "text": f"action {chosen} success rate = {sr:.2f}",
                    "type": "rate",
                    "action": chosen,
                    "metric": "success_rate",
                    "value": sr,
                },
                {
                    "text": f"action {chosen} collision rate = {cr:.2f}",
                    "type": "rate",
                    "action": chosen,
                    "metric": "collision_rate",
                    "value": cr,
                },
            ],
        }
        text = json.dumps(response)
        result = CallResult(text=text, model=self.model, cost_usd=0.0)
        if self.log_dir:
            with open(self.log_dir / f"call_{self.calls:04d}.json", "w") as f:
                json.dump({"system": system, "user": user, "text": text, "mock": True}, f, indent=2)
        return result


def make_client(
    mock: bool | None = None,
    model: str | None = None,
    cost_cap_usd: float = 10.0,
    log_dir: str | Path | None = None,
    provider: str | None = None,
) -> AnthropicClient | OpenAIClient | MockClient:
    """Auto-select a client based on env + explicit provider.

    ``provider`` ∈ {"anthropic", "openai", None}. If None:
      1. If ANTHROPIC_API_KEY set → Anthropic.
      2. Elif OPENAI_API_KEY set → OpenAI.
      3. Else mock.
    """
    if mock is None:
        has_anth = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_oai = bool(os.environ.get("OPENAI_API_KEY"))
        mock = not (has_anth or has_oai)

    if mock:
        return MockClient(
            model=f"mock-{model or 'default'}", cost_cap_usd=cost_cap_usd, log_dir=log_dir
        )

    if provider is None:
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"

    if provider == "anthropic":
        default = "claude-sonnet-4-6"
        return AnthropicClient(model=model or default, cost_cap_usd=cost_cap_usd, log_dir=log_dir)
    if provider == "openai":
        default = "gpt-4o-mini"
        return OpenAIClient(model=model or default, cost_cap_usd=cost_cap_usd, log_dir=log_dir)
    raise ValueError(f"Unknown provider: {provider}")
