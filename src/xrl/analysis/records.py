"""DecisionRecord schema: the structured evidence we hand to the LLM.

One DecisionRecord per (episode_seed, step). Both the learned-policy
(counterfactual rollout stats) and MCTS (tree stats) pipelines produce
records in this same shape, so the downstream explainer prompt is
identical across agents.

Reviewer-driven schema change: ``ActionStats`` now exposes the agent's
actual optimization target, the ``discounted_return``, instead of
``success_rate`` / ``collision_rate``. MDP agents do not directly
optimize success rate; surfacing it as evidence invited type-incorrect
rationales of the form "the agent preferred A because of its higher
success rate." See report Section "Response to Reviewers" for context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActionStats:
    """Per-action statistics at a single decision point.

    All numbers are aggregated over ``n_rollouts`` Monte Carlo samples
    (PPO: stochastic policy rollouts; MCTS: simulations through the
    corresponding root child). The primary metric is
    ``discounted_return`` (γ=0.99), since that is what an MDP agent
    actually maximizes. ``mean_steps_to_end`` is reported as an
    auxiliary diagnostic; it is *not* claimable as the optimization
    target.
    """

    action: int
    discounted_return: float
    std_return: float
    mean_steps_to_end: float
    return_ci: tuple[float, float]
    n_rollouts: int


@dataclass
class DecisionRecord:
    """Everything the LLM sees for one decision.

    Fields:
        source: ``"policy_rollout"``, ``"mcts_tree"``, which evidence pipeline
                produced the stats. ``policy_rollout`` covers any learned
                policy (currently PPO; the legacy DQN ablation uses the
                same tag).
        agent_id: free-form identifier (e.g., ``"ppo_tuned"``).
        state_id: ``{episode_seed}:{step}``.
        step: decision index within the episode.
        agent_pos / agent_dir / obstacle_positions: human-readable state
        chosen_action: the action the agent actually took.
        per_action_stats: list[ActionStats], one entry per legal action.
        agent_metadata: agent-native extras (PPO action probabilities,
                        MCTS visit counts at root, MCTS tree
                        diagnostics, budget, etc.).
    """

    source: str
    agent_id: str
    state_id: str
    step: int
    agent_pos: tuple[int, int]
    agent_dir: int
    obstacle_positions: list[tuple[int, int]]
    chosen_action: int
    per_action_stats: list[ActionStats]
    agent_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import math

        d = asdict(self)
        # jsonschema array validation requires Python lists, not tuples.
        d["agent_pos"] = list(d["agent_pos"])
        d["obstacle_positions"] = [list(p) for p in d["obstacle_positions"]]
        for s in d["per_action_stats"]:
            s["return_ci"] = list(s["return_ci"])
            # MCTS does not track per-rollout step counts, so this field
            # may be NaN. Serialize as null for strict-JSON compatibility
            # with downstream consumers (LLM, jsonschema validators).
            v = s.get("mean_steps_to_end")
            if isinstance(v, float) and math.isnan(v):
                s["mean_steps_to_end"] = None
        return d


DECISION_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DecisionRecord",
    "type": "object",
    "required": [
        "source",
        "agent_id",
        "state_id",
        "step",
        "agent_pos",
        "agent_dir",
        "obstacle_positions",
        "chosen_action",
        "per_action_stats",
    ],
    "properties": {
        "source": {"enum": ["policy_rollout", "mcts_tree", "dqn_rollout"]},
        "agent_id": {"type": "string"},
        "state_id": {"type": "string"},
        "step": {"type": "integer", "minimum": 0},
        "agent_pos": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "agent_dir": {"type": "integer", "minimum": 0, "maximum": 3},
        "obstacle_positions": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        "chosen_action": {"type": "integer", "minimum": 0, "maximum": 2},
        "per_action_stats": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "action",
                    "discounted_return",
                    "std_return",
                    "mean_steps_to_end",
                    "return_ci",
                    "n_rollouts",
                ],
                "properties": {
                    "action": {"type": "integer", "minimum": 0, "maximum": 2},
                    "discounted_return": {"type": "number"},
                    "std_return": {"type": "number"},
                    "mean_steps_to_end": {
                        "type": ["number", "null"],
                        "minimum": 0,
                    },
                    "return_ci": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "n_rollouts": {"type": "integer", "minimum": 1},
                },
            },
        },
        "agent_metadata": {"type": "object"},
    },
}


def save_record(record: DecisionRecord, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)


def load_record(path: str | Path) -> DecisionRecord:
    with open(path) as f:
        d = json.load(f)
    stats = []
    for s in d["per_action_stats"]:
        # Round-trip null → NaN for the MCTS branch.
        if s.get("mean_steps_to_end") is None:
            s["mean_steps_to_end"] = float("nan")
        stats.append(ActionStats(**s))
    return DecisionRecord(
        source=d["source"],
        agent_id=d["agent_id"],
        state_id=d["state_id"],
        step=d["step"],
        agent_pos=tuple(d["agent_pos"]),
        agent_dir=d["agent_dir"],
        obstacle_positions=[tuple(p) for p in d["obstacle_positions"]],
        chosen_action=d["chosen_action"],
        per_action_stats=stats,
        agent_metadata=d.get("agent_metadata", {}),
    )


def validate_record_dict(d: dict) -> None:
    """Validate a DecisionRecord dict against the JSON schema."""
    import jsonschema

    jsonschema.validate(d, DECISION_RECORD_SCHEMA)
