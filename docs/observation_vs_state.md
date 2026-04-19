# Observation vs. State, Fairness Note

MiniGrid-Dynamic-Obstacles-8x8 is technically a POMDP: the default
observation is a 7×7 egocentric partial view plus a direction scalar. We
formulate it as an MDP because:

1. The full state is small (positions of agent + 4 obstacles + direction
   + step count) and available from the environment via
   `env.unwrapped`, so MCTS can plan on it.
2. The proposal committed to an MDP formulation, and the explainability
   question does not hinge on partial observability.

## The Asymmetry

| Agent | Sees | Plans with |
|---|---|---|
| DQN | 7×7×3 partial image + direction | same (via neural policy) |
| MCTS | full state (via simulator copy) | full state |

This is not a fair head-to-head on task performance, MCTS has strictly
more information. Two consequences:

1. **Task-performance comparison is confounded.** If MCTS outperforms
   DQN, it may be information access rather than planning ability. The
   Phase 2/3 matched-performance gate (±0.05 success rate) tries to
   neutralize this: tune both agents until they perform the same, then
   compare explanations at equal performance.
2. **Explanation fairness.** The explainer sees the same
   `DecisionRecord` format for both agents; it does not see raw
   observations. So the explanation-quality comparison is not directly
   contaminated by the observation-access gap, but the *evidence
   content* is. MCTS's tree records outcomes of simulated futures
   computed from full state; DQN's counterfactual rollouts are also
   computed from full state (because rollouts go through the simulator,
   not through the DQN's partial view). So the evidence is comparable in
   terms of information content; only the *origin* differs (tree search
   vs post-hoc rollout).

## Mitigation: Symbolic-Obs DQN Variant

To control for observability, Phase 2 includes a `FlatSymbolicObsWrapper`
DQN variant that learns from full-state symbolic input. Comparing MCTS
against *both* image-DQN and symbolic-DQN lets us decompose the
explainability gap (if any) into observability vs training-paradigm
components.

## Report Treatment

The report's §Results must include:

- Success rates for all three agents (image-DQN, symbolic-DQN, MCTS).
- Explanation quality metrics for all three.
- A "Threats to Validity" paragraph naming observability as a confounder
  and describing the symbolic-DQN control.
