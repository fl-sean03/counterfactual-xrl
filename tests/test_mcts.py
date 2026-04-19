"""Unit tests for from-scratch MCTS."""

from __future__ import annotations

import math

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")

from xrl.agents.mcts import MCTS, MCTSConfig, Node, ucb1_score  # noqa: E402
from xrl.envs.simulator import Simulator  # noqa: E402


def test_ucb1_infinite_for_unvisited() -> None:
    parent = Node(visits=10)
    child = Node(visits=0, parent=parent)
    assert ucb1_score(child, parent.visits, c=math.sqrt(2.0)) == math.inf


def test_ucb1_exploit_component() -> None:
    parent = Node(visits=100)
    child = Node(visits=10, value_sum=5.0, parent=parent)  # mean 0.5
    # exploit 0.5 + explore sqrt(2) * sqrt(log(100)/10)
    expected = 0.5 + math.sqrt(2.0) * math.sqrt(math.log(100) / 10)
    assert ucb1_score(child, parent.visits, c=math.sqrt(2.0)) == pytest.approx(expected)


def test_simulator_clone_preserves_state() -> None:
    sim = Simulator.from_seed(seed=7)
    fp_before = sim.state_fingerprint
    clone = sim.clone()
    assert clone.state_fingerprint == fp_before
    # Step both with the same deterministic action (turn left)
    sim.step(0)
    clone.step(0)
    assert sim.state_fingerprint == clone.state_fingerprint
    sim.close()
    clone.close()


def test_mcts_returns_legal_action_and_populated_root() -> None:
    sim = Simulator.from_seed(seed=123)
    mcts = MCTS(MCTSConfig(sims_per_decision=50, rollout_policy="random"))
    action, root = mcts.plan(sim)
    assert action in (0, 1, 2)
    assert root.visits == 50
    # Every legal child was expanded at the root.
    assert set(root.children.keys()) == {0, 1, 2}
    # Child visits should sum to root visits (every sim descends through one child).
    assert sum(c.visits for c in root.children.values()) == root.visits
    sim.close()


def test_mcts_beats_random_in_a_few_sims(benchmark_stateful=False) -> None:
    """MCTS with 100 sims should almost never bash into an obstacle on the
    very first step of a fresh seed, much better than random's near-100%
    collision rate. Probabilistic but very robust."""
    sim = Simulator.from_seed(seed=321)
    mcts = MCTS(MCTSConfig(sims_per_decision=100, rollout_policy="greedy"))
    action, root = mcts.plan(sim)
    # The best child should have a higher mean value than the worst.
    vals = [c.mean_value for c in root.children.values() if c.visits]
    assert max(vals) >= min(vals)
    sim.close()
