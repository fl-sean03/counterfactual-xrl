"""From-scratch MCTS (UCT) for MiniGrid-Dynamic-Obstacles.

Implements standard UCT with:
- UCB1 selection
- Full-width expansion (|A| = 3 is small, every child is expanded on
  first visit, no progressive widening needed)
- Rollout to terminal or depth cap with either a uniform-random or
  greedy-toward-goal policy
- Mean-value backup
- Sampled-transition handling for stochastic obstacles: each simulation
  re-samples the obstacle transition at every step (the env.step() in
  the simulator already does this). Visit counts at the root converge to
  the expectation over futures.

**From scratch**, no RL library is used for MCTS itself. The only
external library touched is the ``Simulator`` around MiniGrid.

Tree-dump schema (root): see docs/tree_schema.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from xrl.envs.simulator import Simulator


@dataclass
class Node:
    parent: Node | None = None
    action_from_parent: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)
    # Per-action counters aggregated across all simulations through this
    # node, used for explanation evidence (not for UCB itself).
    success_count: int = 0  # rollouts through this node that reached the goal
    collision_count: int = 0  # rollouts that ended in collision
    terminal: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def ucb1_score(child: Node, parent_visits: int, c: float) -> float:
    if child.visits == 0:
        return math.inf
    exploit = child.mean_value
    explore = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploit + explore


def _greedy_toward_goal(sim: Simulator) -> int:
    """Heuristic rollout policy: turn/step toward (6, 6) using Manhattan logic."""
    u = sim.env.unwrapped
    ax, ay = u.agent_pos
    goal = (u.width - 2, u.height - 2)  # (6,6) for 8x8 grid
    dx, dy = goal[0] - ax, goal[1] - ay
    # MiniGrid dir: 0=east(+x), 1=south(+y), 2=west(-x), 3=north(-y)
    if abs(dx) >= abs(dy):
        desired = 0 if dx > 0 else 2
    else:
        desired = 1 if dy > 0 else 3
    d = int(u.agent_dir)
    if d == desired:
        return 2  # forward
    # Choose rotation that gets us there fastest.
    diff = (desired - d) % 4
    return 0 if diff == 3 else 1  # 0=left, 1=right; diff==1 → right, diff==3 → left


def _random_rollout_policy(sim: Simulator) -> int:
    return int(np.random.randint(0, 3))


ROLLOUT_POLICIES: dict[str, Callable[[Simulator], int]] = {
    "random": _random_rollout_policy,
    "greedy": _greedy_toward_goal,
}


@dataclass
class MCTSConfig:
    sims_per_decision: int = 500
    c_ucb: float = math.sqrt(2.0)
    max_rollout_depth: int = 50
    rollout_policy: str = "greedy"
    gamma: float = 1.0
    seed: int = 0


class MCTS:
    """UCT planner. Use ``plan(sim)`` to pick an action at a live state."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.cfg = config or MCTSConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._last_root: Node | None = None

    def _rollout(self, sim: Simulator) -> tuple[float, bool, bool]:
        """Random-policy Monte Carlo rollout. Returns (return, success, collision)."""
        policy = ROLLOUT_POLICIES[self.cfg.rollout_policy]
        total = 0.0
        discount = 1.0
        success = False
        collision = False
        for _ in range(self.cfg.max_rollout_depth):
            action = policy(sim)
            r = sim.step(action)
            total += discount * r.reward
            discount *= self.cfg.gamma
            if r.terminated or r.truncated:
                success = r.terminated and r.reward > 0
                collision = r.terminated and r.reward < 0
                break
        return total, success, collision

    def _simulate_once(self, root: Node, root_sim: Simulator) -> None:
        """One full MCTS iteration: select→expand→simulate→backup."""
        node = root
        sim = root_sim.clone()
        sim.reseed_dynamics(int(self._rng.integers(0, 2**31 - 1)))
        path: list[Node] = [root]
        trajectory_return = 0.0
        discount = 1.0
        terminated = False
        truncated = False
        reward = 0.0

        # --- Selection: descend while all children are visited. ---
        while node.children and not node.terminal:
            # All legal actions are assumed present as children once expanded.
            best_action = max(
                node.children.keys(),
                key=lambda a: ucb1_score(node.children[a], node.visits, self.cfg.c_ucb),
            )
            r = sim.step(best_action)
            reward = r.reward
            trajectory_return += discount * reward
            discount *= self.cfg.gamma
            node = node.children[best_action]
            path.append(node)
            if r.terminated or r.truncated:
                terminated = r.terminated
                truncated = r.truncated
                node.terminal = True
                break

        # --- Expansion: if non-terminal, add all legal children. ---
        if not node.terminal and not (terminated or truncated):
            for a in sim.legal_actions():
                node.children.setdefault(a, Node(parent=node, action_from_parent=a))
            # Pick one child to roll out through.
            chosen_action = int(self._rng.choice(list(node.children.keys())))
            r = sim.step(chosen_action)
            reward = r.reward
            trajectory_return += discount * reward
            discount *= self.cfg.gamma
            child = node.children[chosen_action]
            path.append(child)
            if r.terminated or r.truncated:
                terminated = r.terminated
                truncated = r.truncated
                child.terminal = True

        # --- Simulation: random rollout from leaf to terminal/depth cap. ---
        rollout_return = 0.0
        success = False
        collision = False
        if not (terminated or truncated):
            rollout_return, success, collision = self._rollout(sim)
        else:
            # Already terminal at leaf, attribute success/collision by last reward.
            success = terminated and reward > 0
            collision = terminated and reward < 0

        total_return = trajectory_return + discount * rollout_return

        # --- Backup ---
        for n in path:
            n.visits += 1
            n.value_sum += total_return
            if success:
                n.success_count += 1
            if collision:
                n.collision_count += 1

        sim.close()

    def plan(self, sim: Simulator) -> tuple[int, Node]:
        """Run ``sims_per_decision`` simulations; return (best_action, root_node)."""
        root = Node()
        # Populate root children upfront so visit counts are comparable.
        for a in sim.legal_actions():
            root.children[a] = Node(parent=root, action_from_parent=a)
        for _ in range(self.cfg.sims_per_decision):
            self._simulate_once(root, sim)
        # Robust-child: pick the action with the most visits.
        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        self._last_root = root
        return best_action, root

    @property
    def last_root(self) -> Node | None:
        return self._last_root


def root_to_dict(root: Node, legal_actions: list[int]) -> dict:
    """Serialize a root node and its immediate children for an explanation."""
    return {
        "root_visits": root.visits,
        "root_value": root.mean_value,
        "children": [
            {
                "action": a,
                "visits": root.children[a].visits,
                "mean_value": root.children[a].mean_value,
                "success_count": root.children[a].success_count,
                "collision_count": root.children[a].collision_count,
                "success_rate": (
                    root.children[a].success_count / root.children[a].visits
                    if root.children[a].visits
                    else 0.0
                ),
                "collision_rate": (
                    root.children[a].collision_count / root.children[a].visits
                    if root.children[a].visits
                    else 0.0
                ),
            }
            for a in legal_actions
            if a in root.children
        ],
    }
