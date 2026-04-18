"""Explainer pipeline: DecisionRecord → Explanation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrl.analysis.records import DecisionRecord
from xrl.explainer.client import AnthropicClient, MockClient
from xrl.explainer.prompts import PROMPT_VERSION, build_system_prompt, build_user_prompt


@dataclass
class Explanation:
    state_id: str
    agent_id: str
    source: str
    chosen_action: int
    rationale: str
    counterfactual: str
    confidence: float
    claims: list[dict[str, Any]]
    prompt_version: int
    model: str
    raw_text: str
    cost_usd: float = 0.0


def _record_hash(rec: DecisionRecord) -> str:
    payload = json.dumps(rec.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_explanation_json(text: str) -> dict:
    """Best-effort JSON parsing — Claude sometimes wraps in fences."""
    t = text.strip()
    if t.startswith("```"):
        # strip ``` or ```json fence
        t = t.split("```", 2)
        if len(t) >= 2:
            body = t[1]
            if body.lstrip().lower().startswith("json"):
                body = body.split("\n", 1)[1] if "\n" in body else body
            t = body
        else:  # pragma: no cover
            t = "".join(t)
    return json.loads(t)


def explain(
    record: DecisionRecord,
    client: AnthropicClient | MockClient,
    cache_dir: Path | None = None,
) -> Explanation:
    """Generate one explanation for one DecisionRecord.

    If ``cache_dir`` is set, a hash of the record is used as a cache key
    and already-processed records are skipped. This makes re-runs cheap
    and deterministic.
    """
    rhash = _record_hash(record)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{rhash}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                d = json.load(f)
            return Explanation(**d)

    system = build_system_prompt(record.source)
    user = build_user_prompt(record.to_dict())
    result = client.call(system=system, user=user, cache_system=True)

    parsed = parse_explanation_json(result.text)
    exp = Explanation(
        state_id=record.state_id,
        agent_id=record.agent_id,
        source=record.source,
        chosen_action=record.chosen_action,
        rationale=str(parsed.get("rationale", "")),
        counterfactual=str(parsed.get("counterfactual", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        claims=list(parsed.get("claims", [])),
        prompt_version=PROMPT_VERSION,
        model=result.model,
        raw_text=result.text,
        cost_usd=result.cost_usd,
    )

    if cache_dir:
        with open(cache_path, "w") as f:
            json.dump(asdict(exp), f, indent=2)

    return exp
