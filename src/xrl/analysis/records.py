"""DecisionRecord schema: the structured evidence we hand to the LLM.

One DecisionRecord per (episode_seed, step). Both the DQN (rollout stats)
and MCTS (tree stats) pipelines produce records in this same shape, so the
downstream explainer prompt is identical across agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActionStats:
    """Per-action statistics at a single decision point."""

    action: int
    mean_return: float
    std_return: float
    success_rate: float
    collision_rate: float
    mean_steps_to_end: float
    success_ci: tuple[float, float]
    collision_ci: tuple[float, float]
    n_rollouts: int


@dataclass
class DecisionRecord:
    """Everything the LLM sees for one decision.

    Fields:
        source: ``"dqn_rollout"``, ``"mcts_tree"``, which evidence pipeline
                produced the stats.
        agent_id: free-form identifier (e.g., ``"dqn_baseline_seed0"``).
        state_id: ``{episode_seed}:{step}``.
        step: decision index within the episode.
        agent_pos / agent_dir / obstacle_positions: human-readable state
        chosen_action: the action the agent actually took.
        per_action_stats: list[ActionStats], one entry per legal action.
        agent_metadata: agent-native extras (DQN Q-values, MCTS visit
                        counts at root, budget, etc.).
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
        d = asdict(self)
        # jsonschema array validation requires Python lists, not tuples.
        d["agent_pos"] = list(d["agent_pos"])
        d["obstacle_positions"] = [list(p) for p in d["obstacle_positions"]]
        for s in d["per_action_stats"]:
            s["success_ci"] = list(s["success_ci"])
            s["collision_ci"] = list(s["collision_ci"])
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
        "source": {"enum": ["dqn_rollout", "mcts_tree"]},
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
                    "mean_return",
                    "std_return",
                    "success_rate",
                    "collision_rate",
                    "mean_steps_to_end",
                    "success_ci",
                    "collision_ci",
                    "n_rollouts",
                ],
                "properties": {
                    "action": {"type": "integer", "minimum": 0, "maximum": 2},
                    "mean_return": {"type": "number"},
                    "std_return": {"type": "number"},
                    "success_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "collision_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "mean_steps_to_end": {"type": "number", "minimum": 0},
                    "success_ci": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "collision_ci": {
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
    stats = [ActionStats(**s) for s in d["per_action_stats"]]
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
