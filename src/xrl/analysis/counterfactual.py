"""Counterfactual Monte-Carlo rollouts from a decision state.

Given a live env (or simulator snapshot), a trained policy, and the set of
legal actions, this module produces per-action rollout statistics: for each
candidate action, we force it at step 0 and then follow the policy until
termination. Repeating that N times gives us the distribution of outcomes
conditional on the first action.

The resulting ``ActionStats`` are what the explainer reads as evidence.
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
) -> tuple[float, bool, bool, int]:
    """Run one rollout through the given simulator with the given policy.

    Returns ``(total_return, success, collision, steps)``.
    """
    total = 0.0
    for step in range(max_steps):
        obs = obs_fn(sim)
        action = policy_predict(obs)
        r = sim.step(int(action))
        total += r.reward
        if r.terminated or r.truncated:
            success = r.terminated and r.reward > 0
            collision = r.terminated and r.reward < 0
            return total, success, collision, step + 1
    return total, False, False, max_steps


def counterfactual_rollouts(
    root_sim: Simulator,
    policy_predict: Callable[[Any], int],
    obs_fn: Callable[[Simulator], Any],
    n_per_action: int = 100,
    seed: int = 0,
    max_steps: int = 256,
) -> list[ActionStats]:
    """For each legal action, force it from ``root_sim`` and roll out N times.

    The policy is used for steps ≥ 1. Rollouts through the stochastic
    obstacle transitions are seeded so re-runs are reproducible.
    """
    rng = np.random.default_rng(seed)
    results: list[ActionStats] = []
    for a in root_sim.legal_actions():
        returns = np.zeros(n_per_action)
        successes = np.zeros(n_per_action)
        collisions = np.zeros(n_per_action)
        steps = np.zeros(n_per_action)

        for i in range(n_per_action):
            # Fresh branch per rollout.
            sim = root_sim.clone()
            sim.reseed_dynamics(int(rng.integers(0, 2**31 - 1)))
            # Force first action from this counterfactual branch.
            r0 = sim.step(a)
            total = r0.reward
            if r0.terminated or r0.truncated:
                returns[i] = total
                successes[i] = float(r0.terminated and r0.reward > 0)
                collisions[i] = float(r0.terminated and r0.reward < 0)
                steps[i] = 1
                sim.close()
                continue
            # Remaining rollout under policy.
            sub_ret, succ, coll, n_steps = rollout_from(
                sim, policy_predict, obs_fn, max_steps=max_steps - 1
            )
            returns[i] = total + sub_ret
            successes[i] = float(succ)
            collisions[i] = float(coll)
            steps[i] = n_steps + 1
            sim.close()
        # Reseed per action so CIs are reproducible per action.
        stats = ActionStats(
            action=a,
            mean_return=float(returns.mean()),
            std_return=float(returns.std()),
            success_rate=float(successes.mean()),
            collision_rate=float(collisions.mean()),
            mean_steps_to_end=float(steps.mean()),
            success_ci=_bootstrap_ci(successes, seed=int(rng.integers(0, 2**31 - 1))),
            collision_ci=_bootstrap_ci(collisions, seed=int(rng.integers(0, 2**31 - 1))),
            n_rollouts=n_per_action,
        )
        results.append(stats)
    return results
