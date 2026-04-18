# MCTS Design

## Algorithm

Standard UCT (Kocsis & Szepesvári 2006) with the following choices:

| Decision | Choice | Why |
|---|---|---|
| Selection | UCB1, `c = √2` | Textbook default; tunable. |
| Expansion | Full-width on first visit | `|A| = 3` is small — no PW needed. |
| Rollout policy | Greedy-toward-goal OR random | Greedy is a domain heuristic (rotate to face goal, step forward). Random is the control. Config-switchable. |
| Rollout depth cap | 60 steps | `max_episode_steps = 256`; 60 is enough to usually hit a terminal from depth-5 frontiers, bounded to keep sims cheap. |
| Backup | Mean value | Standard MCTS. |
| Discount γ | 1.0 | Matches MDP; step-count pressure is in the reward formula. |
| Stochastic transitions | Sampled-transition UCT | Each sim re-samples the obstacle transition via the simulator. Root visit counts converge in expectation. |

## Tree dump schema

At each real decision step, the root node plus its immediate children are
serialized (see `src/xrl/agents/mcts.py::root_to_dict`). Schema:

```json
{
  "root_visits": 500,
  "root_value": 0.4,
  "children": [
    {"action": 0, "visits": 85,  "mean_value": 0.12, "success_count": 10, "collision_count": 40, "success_rate": 0.12, "collision_rate": 0.47},
    {"action": 1, "visits": 110, "mean_value": 0.18, "success_count": 14, "collision_count": 34, "success_rate": 0.13, "collision_rate": 0.31},
    {"action": 2, "visits": 305, "mean_value": 0.65, "success_count": 220, "collision_count": 15, "success_rate": 0.72, "collision_rate": 0.05}
  ]
}
```

`success_count` / `collision_count` are aggregated *through* each child
across every simulation that touched it (selection + rollout). These are
the raw counts the explainer uses as "evidence that this action was
better"; `success_rate` / `collision_rate` are derived.

## Fair-comparison notes

- **Budget parity:** budget is time + simulations per decision. To match
  MCTS against DQN at a fixed per-decision wall-clock budget, document
  `sims_per_decision` and measured average ms-per-decision.
- **Stochastic variance:** with `sims_per_decision = 500`, empirical
  top-action agreement across re-runs on the same state should be ≥ 0.9.
  Measure this during Phase 3 exit-gate checks; bump budget if needed.
- **Information access:** MCTS operates on full state via the simulator.
  DQN sees only the 7×7 egocentric image. This is addressed by the
  symbolic-DQN control variant (docs/observation_vs_state.md).
