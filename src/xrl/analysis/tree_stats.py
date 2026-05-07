"""Convert an MCTS root node into structured evidence for the explainer.

Two outputs:

1. ``mcts_root_to_action_stats``: per-root-child ``ActionStats`` with the
   discounted return aggregated during search. Same schema as the PPO
   counterfactual-rollout pipeline produces, so the explainer prompt is
   uniform across agents.

2. ``mcts_tree_diagnostics``: deeper tree information that PPO has no
   analogue for: top-3 principal variations, depth-2 visit
   distributions, and per-child variance. This was added in response to
   reviewer feedback that limiting MCTS evidence to root-level
   statistics deliberately equalized the bandwidth between the two
   substrates and therefore could not surface the deliberation-tree
   advantage MCTS is supposed to confer.
"""

from __future__ import annotations

import math
from typing import Any

from xrl.agents.mcts import Node
from xrl.analysis.records import ActionStats


def _wald_ci(mean: float, std: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation CI on the mean. Adequate for n >= 30, which
    holds for any non-trivially-visited root child at our 500-sim budget.
    """
    if n == 0:
        return (0.0, 0.0)
    se = std / math.sqrt(n)
    return (mean - z * se, mean + z * se)


def mcts_root_to_action_stats(root: Node, legal_actions: list[int]) -> list[ActionStats]:
    """Walk the root's children and produce per-action stats.

    The MCTS per-child ``mean_value`` is the empirical mean of
    discounted returns over rollouts that passed through that child;
    that is the agent's optimization target. ``std_return`` is computed
    from the running ``value_sum_sq`` we now track on each Node.
    ``return_ci`` is a normal-approximation CI on the mean.
    ``mean_steps_to_end`` is left as NaN since MCTS does not track
    rollout lengths separately during search.
    """
    out: list[ActionStats] = []
    for a in legal_actions:
        child = root.children.get(a)
        if child is None or child.visits == 0:
            out.append(
                ActionStats(
                    action=a,
                    discounted_return=0.0,
                    std_return=0.0,
                    mean_steps_to_end=0.0,
                    return_ci=(0.0, 0.0),
                    n_rollouts=0,
                )
            )
            continue
        mean = float(child.mean_value)
        std = math.sqrt(child.value_variance)
        out.append(
            ActionStats(
                action=a,
                discounted_return=mean,
                std_return=float(std),
                mean_steps_to_end=float("nan"),
                return_ci=_wald_ci(mean, std, child.visits),
                n_rollouts=int(child.visits),
            )
        )
    return out


def _principal_variation(child: Node, max_depth: int = 5) -> list[int]:
    """Greedy descent by argmax visits from ``child`` to depth ``max_depth``.

    Returns the action sequence of grandchildren chosen along the way
    (does not include the action that led into ``child``).
    """
    actions: list[int] = []
    node = child
    for _ in range(max_depth):
        if not node.children:
            break
        # Argmax by visits among visited children.
        visited = {a: c for a, c in node.children.items() if c.visits > 0}
        if not visited:
            break
        best = max(visited.keys(), key=lambda a: visited[a].visits)
        actions.append(int(best))
        node = visited[best]
    return actions


def _depth2_visit_distribution(child: Node) -> dict[str, int]:
    """For a root child, return {grandchild_action: visits}."""
    return {
        str(a): int(g.visits)
        for a, g in child.children.items()
        if g.visits > 0
    }


def mcts_tree_diagnostics(
    root: Node,
    legal_actions: list[int],
    pv_depth: int = 5,
    top_k: int = 3,
) -> dict[str, Any]:
    """Build the deeper tree summary handed to the explainer.

    Output keys:
      principal_variations: top-K most-visited root children, each with
          its argmax-visit principal variation (action sequence) of
          length up to ``pv_depth``, the visit count of the root child,
          and that child's mean discounted return.
      depth2_visit_distribution: per root child, the visit count of each
          grandchild action. Reveals what the search "expected" to do
          next, conditional on this first action.
      value_variance_by_action: per root child, variance of discounted
          returns over rollouts that passed through that child. Higher
          variance indicates a noisier value estimate.
    """
    visited_children = [
        (a, root.children[a])
        for a in legal_actions
        if a in root.children and root.children[a].visits > 0
    ]
    visited_children.sort(key=lambda ac: ac[1].visits, reverse=True)
    top = visited_children[:top_k]

    principal_variations = [
        {
            "first_action": int(a),
            "principal_variation": _principal_variation(c, max_depth=pv_depth),
            "root_child_visits": int(c.visits),
            "root_child_discounted_return": float(c.mean_value),
        }
        for a, c in top
    ]
    depth2 = {
        str(a): _depth2_visit_distribution(c)
        for a, c in visited_children
    }
    value_variance = {
        str(a): float(c.value_variance)
        for a, c in visited_children
    }
    return {
        "principal_variations": principal_variations,
        "depth2_visit_distribution": depth2,
        "value_variance_by_action": value_variance,
    }
