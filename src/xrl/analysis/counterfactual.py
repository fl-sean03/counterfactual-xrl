"""Counterfactual Monte-Carlo rollouts from a decision state.

Given a live env (or simulator snapshot), a trained policy, and the set of
legal actions, this module produces per-action rollout statistics: for each
candidate action, we force it at step 0 and then follow the policy until
termination. Repeating that N times gives us the distribution of outcomes
conditional on the first action.

The resulting ``ActionStats`` are what the explainer reads as evidence.
The primary metric is ``discounted_return`` (gamma-discounted), since
that is what an MDP agent actually optimizes. Reviewer-driven change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from xrl.analysis.records import ActionStats
from xrl.envs.simulator import Simulator


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, seed: int = 0
) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def rollout_from(
    sim: Simulator,
    policy_predict: Callable[[Any], int],
    obs_fn: Callable[[Simulator], Any],
    max_steps: int = 256,
    gamma: float = 1.0,
    discount_start: float = 1.0,
    step_penalty: float = 0.0,
) -> tuple[float, float, bool, bool, int]:
    """Run one rollout through the given simulator with the given policy.

    Returns ``(undiscounted_total, discounted_total, success, collision, steps)``.

    ``discount_start`` is the discount factor that applies to the *first*
    reward in this rollout segment. Callers that have already taken steps
    (e.g. forced the first counterfactual action) should pass
    ``gamma**1 = gamma`` for ``discount_start`` so the second-step reward
    is multiplied by gamma**1. ``step_penalty`` is the same flat per-
    non-terminal-step cost applied during PPO training and MCTS planning.
    """
    undisc = 0.0
    disc = 0.0
    discount = discount_start
    for step in range(max_steps):
        obs = obs_fn(sim)
        action = policy_predict(obs)
        r = sim.step(int(action))
        terminal = r.terminated or r.truncated
        shaped = r.reward - (step_penalty if not terminal else 0.0)
        undisc += shaped
        disc += discount * shaped
        discount *= gamma
        if terminal:
            success = r.terminated and r.reward > 0
            collision = r.terminated and r.reward < 0
            return undisc, disc, success, collision, step + 1
    return undisc, disc, False, False, max_steps


def counterfactual_rollouts(
    root_sim: Simulator,
    policy_predict: Callable[[Any], int],
    obs_fn: Callable[[Simulator], Any],
    n_per_action: int = 100,
    seed: int = 0,
    max_steps: int = 256,
    gamma: float = 0.99,
    step_penalty: float = 0.01,
) -> list[ActionStats]:
    """For each legal action, force it from ``root_sim`` and roll out N times.

    The policy is used for steps >= 1. Rollouts through the stochastic
    obstacle transitions are seeded so re-runs are reproducible. Returns
    are gamma-discounted and include the same per-step penalty PPO
    trained against and MCTS plans against, so all three sides see the
    identical MDP.
    """
    rng = np.random.default_rng(seed)
    results: list[ActionStats] = []
    for a in root_sim.legal_actions():
        disc_returns = np.zeros(n_per_action)
        steps = np.zeros(n_per_action)

        for i in range(n_per_action):
            sim = root_sim.clone()
            r0 = sim.step(a)
            terminal0 = r0.terminated or r0.truncated
            shaped0 = r0.reward - (step_penalty if not terminal0 else 0.0)
            # Forced first action: discount=1 for the first reward.
            disc_total = 1.0 * shaped0
            if terminal0:
                disc_returns[i] = disc_total
                steps[i] = 1
                sim.close()
                continue
            # Remaining rollout under policy. Second reward gets gamma**1.
            _, sub_disc, _, _, n_steps = rollout_from(
                sim,
                policy_predict,
                obs_fn,
                max_steps=max_steps - 1,
                gamma=gamma,
                discount_start=gamma,
                step_penalty=step_penalty,
            )
            disc_returns[i] = disc_total + sub_disc
            steps[i] = n_steps + 1
            sim.close()

        stats = ActionStats(
            action=a,
            discounted_return=float(disc_returns.mean()),
            std_return=float(disc_returns.std()),
            mean_steps_to_end=float(steps.mean()),
            return_ci=_bootstrap_ci(disc_returns, seed=int(rng.integers(0, 2**31 - 1))),
            n_rollouts=n_per_action,
        )
        results.append(stats)
    return results
