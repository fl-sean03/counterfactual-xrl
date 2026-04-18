"""Convert an MCTS root node into ``ActionStats`` for the unified DecisionRecord.

MCTS tree gives us, per child (action), the number of visits, their mean
value, and the aggregate success/collision counts of rollouts that passed
through that child. We convert those into the same ``ActionStats`` schema
we use for DQN rollouts so the explainer sees the same format either way.
"""

from __future__ import annotations

import math

from xrl.agents.mcts import Node
from xrl.analysis.records import ActionStats


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — good for small n and proportions near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcts_root_to_action_stats(root: Node, legal_actions: list[int]) -> list[ActionStats]:
    """Walk the root's children and produce per-action stats.

    Note: MCTS's per-child ``mean_value`` is the total-discounted-return
    over rollouts that passed through that child (success: roughly
    1 − 0.9·step/256; collision: −1). We use that as ``mean_return``.
    ``std_return`` is not tracked during MCTS; we synthesize it from
    success/collision counts assuming a bi-modal distribution (return ≈
    1 with prob p_success, −1 with prob p_collision). For strict
    accuracy this should be measured alongside; adequate for explainer
    evidence.
    """
    out: list[ActionStats] = []
    for a in legal_actions:
        child = root.children.get(a)
        if child is None or child.visits == 0:
            out.append(
                ActionStats(
                    action=a,
                    mean_return=0.0,
                    std_return=0.0,
                    success_rate=0.0,
                    collision_rate=0.0,
                    mean_steps_to_end=0.0,
                    success_ci=(0.0, 0.0),
                    collision_ci=(0.0, 0.0),
                    n_rollouts=0,
                )
            )
            continue
        p_succ = child.success_count / child.visits
        p_coll = child.collision_count / child.visits
        # Bi-modal approximation for std.
        mu = child.mean_value
        var = p_succ * (1.0 - mu) ** 2 + p_coll * (-1.0 - mu) ** 2
        std = math.sqrt(max(var, 0.0))
        out.append(
            ActionStats(
                action=a,
                mean_return=float(child.mean_value),
                std_return=float(std),
                success_rate=float(p_succ),
                collision_rate=float(p_coll),
                mean_steps_to_end=float("nan"),  # not tracked during MCTS
                success_ci=_wilson_ci(child.success_count, child.visits),
                collision_ci=_wilson_ci(child.collision_count, child.visits),
                n_rollouts=int(child.visits),
            )
        )
    return out
