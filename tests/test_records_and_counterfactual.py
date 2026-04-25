"""Tests for DecisionRecord schema and counterfactual rollouts."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")
pytest.importorskip("jsonschema")

from xrl.agents.base import RandomAgent  # noqa: E402
from xrl.agents.mcts import MCTS, MCTSConfig  # noqa: E402
from xrl.analysis.counterfactual import counterfactual_rollouts  # noqa: E402
from xrl.analysis.records import (  # noqa: E402
    ActionStats,
    DecisionRecord,
    validate_record_dict,
)
from xrl.analysis.tree_stats import mcts_root_to_action_stats  # noqa: E402
from xrl.envs.simulator import Simulator  # noqa: E402


def _dummy_obs_fn(sim):
    return None


def test_counterfactual_rollouts_produces_3_stats() -> None:
    sim = Simulator.from_seed(seed=42)
    agent = RandomAgent(n_actions=3, rng_seed=0)
    stats = counterfactual_rollouts(
        sim, lambda obs: agent.predict(obs), _dummy_obs_fn, n_per_action=10, seed=0
    )
    assert len(stats) == 3
    for s in stats:
        assert 0 <= s.success_rate <= 1
        assert 0 <= s.collision_rate <= 1
        assert s.n_rollouts == 10
    sim.close()


def test_mcts_stats_round_trip_through_schema() -> None:
    sim = Simulator.from_seed(seed=7)
    mcts = MCTS(MCTSConfig(sims_per_decision=50, rollout_policy="random"))
    action, root = mcts.plan(sim)
    stats = mcts_root_to_action_stats(root, legal_actions=[0, 1, 2])
    rec = DecisionRecord(
        source="mcts_tree",
        agent_id="test",
        state_id="7:0",
        step=0,
        agent_pos=(1, 1),
        agent_dir=0,
        obstacle_positions=[(2, 3), (4, 5), (5, 2), (3, 6)],
        chosen_action=int(action),
        per_action_stats=stats,
    )
    validate_record_dict(rec.to_dict())
    sim.close()


def test_action_stats_is_serialisable() -> None:
    s = ActionStats(
        action=0,
        mean_return=0.5,
        std_return=0.2,
        success_rate=0.7,
        collision_rate=0.2,
        mean_steps_to_end=30.0,
        success_ci=(0.6, 0.8),
        collision_ci=(0.1, 0.3),
        n_rollouts=100,
    )
    rec = DecisionRecord(
        source="policy_rollout",
        agent_id="test",
        state_id="0:0",
        step=0,
        agent_pos=(1, 1),
        agent_dir=0,
        obstacle_positions=[(2, 2)],
        chosen_action=0,
        per_action_stats=[s, s, s],
    )
    validate_record_dict(rec.to_dict())
