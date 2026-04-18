"""Anthropic client wrapper with caching, retries, and cost tracking.

Designed to be swappable with a ``MockClient`` for tests and for
offline development when ``ANTHROPIC_API_KEY`` is not set.
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


# Rough pricing (per 1M tokens) — update before real runs.
PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "cache_read": 0.10, "cache_write": 1.25},
    "claude-opus-4-7": {"in": 15.00, "out": 75.00, "cache_read": 1.50, "cache_write": 18.75},
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


class AnthropicClient:
    """Real Anthropic SDK wrapper. Uses prompt caching on system prompt."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        cost_cap_usd: float = 10.0,
        log_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("anthropic SDK not installed") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Either `export ANTHROPIC_API_KEY=...` "
                "or use MockClient for offline runs."
            )
        self._client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.cost_cap_usd = cost_cap_usd
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.total_cost: float = 0.0
        self.calls: int = 0

    def call(
        self,
        system: str,
        user: str,
        cache_system: bool = True,
    ) -> CallResult:
        if self.total_cost >= self.cost_cap_usd:
            raise RuntimeError(
                f"Cost cap ${self.cost_cap_usd:.2f} reached; aborting further API calls."
            )

        system_blocks = [{"type": "text", "text": system}]
        if cache_system:
            # cache_control only takes effect if the cached block is >= min
            # cacheable tokens (1024 for sonnet). If your system prompt is
            # shorter, caching is a no-op.
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
            raise RuntimeError(
                f"Anthropic call failed after {self.max_retries} retries: {last_err}"
            )

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        result = CallResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=self.model,
            raw=resp,
        )
        result.cost_usd = estimate_cost(self.model, result)
        self.total_cost += result.cost_usd
        self.calls += 1

        if self.log_dir:
            with open(self.log_dir / f"call_{self.calls:04d}.json", "w") as f:
                json.dump(
                    {
                        "model": result.model,
                        "system": system,
                        "user": user,
                        "text": text,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cache_read_tokens": result.cache_read_tokens,
                        "cache_creation_tokens": result.cache_creation_tokens,
                        "cost_usd": result.cost_usd,
                    },
                    f,
                    indent=2,
                )

        return result


class MockClient:
    """Deterministic offline client for tests and no-API-key local dev.

    Returns a canned JSON explanation built from the provided evidence.
    Lets the full pipeline, schema validation, and metric computation
    run end-to-end without hitting the network.
    """

    def __init__(
        self, model: str = "mock-sonnet", cost_cap_usd: float = 10.0, log_dir=None, **_: Any
    ) -> None:
        self.model = model
        self.cost_cap_usd = cost_cap_usd
        self.total_cost = 0.0
        self.calls = 0
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._canned_log: list[dict] = []

    def call(self, system: str, user: str, cache_system: bool = True) -> CallResult:
        self.calls += 1
        payload = json.loads(user) if user.strip().startswith("{") else {}
        stats = payload.get("per_action_stats", [])
        chosen = payload.get("chosen_action", 0)
        # Pick the chosen action's stats and the best alternative.
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
    model: str = "claude-sonnet-4-6",
    cost_cap_usd: float = 10.0,
    log_dir: str | Path | None = None,
) -> AnthropicClient | MockClient:
    """Auto-select a real vs mock client based on env."""
    if mock is None:
        mock = not os.environ.get("ANTHROPIC_API_KEY")
    if mock:
        return MockClient(model=f"mock-{model}", cost_cap_usd=cost_cap_usd, log_dir=log_dir)
    return AnthropicClient(model=model, cost_cap_usd=cost_cap_usd, log_dir=log_dir)
