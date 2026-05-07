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
from xrl.analysis.tree_stats import (  # noqa: E402
    mcts_root_to_action_stats,
    mcts_tree_diagnostics,
)
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
        # Discounted return is bounded by [-1, +1] on this env (goal
        # reward at most ~1, collision -1, no other rewards).
        assert -1.05 <= s.discounted_return <= 1.05
        assert s.n_rollouts == 10
        lo, hi = s.return_ci
        assert lo <= s.discounted_return <= hi or abs(hi - lo) < 1e-9
    sim.close()


def test_mcts_stats_round_trip_through_schema() -> None:
    sim = Simulator.from_seed(seed=7)
    mcts = MCTS(MCTSConfig(sims_per_decision=50, rollout_policy="random"))
    action, root = mcts.plan(sim)
    stats = mcts_root_to_action_stats(root, legal_actions=[0, 1, 2])
    diag = mcts_tree_diagnostics(root, legal_actions=[0, 1, 2])
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
        agent_metadata={"tree_diagnostics": diag},
    )
    validate_record_dict(rec.to_dict())
    sim.close()


def test_mcts_tree_diagnostics_shape() -> None:
    sim = Simulator.from_seed(seed=11)
    mcts = MCTS(MCTSConfig(sims_per_decision=80, rollout_policy="greedy"))
    _, root = mcts.plan(sim)
    diag = mcts_tree_diagnostics(root, legal_actions=[0, 1, 2])
    assert "principal_variations" in diag
    assert "depth2_visit_distribution" in diag
    assert "value_variance_by_action" in diag
    pvs = diag["principal_variations"]
    assert 1 <= len(pvs) <= 3
    for entry in pvs:
        assert "first_action" in entry
        assert "principal_variation" in entry
        assert "root_child_visits" in entry
        assert isinstance(entry["principal_variation"], list)
    sim.close()


def test_action_stats_is_serialisable() -> None:
    s = ActionStats(
        action=0,
        discounted_return=0.5,
        std_return=0.2,
        mean_steps_to_end=30.0,
        return_ci=(0.45, 0.55),
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
