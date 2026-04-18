# Contributions

_Paste the below into the report's Contributions section after Sean+Andrew
agree on the actual split._

## Sean Florez

- Problem formulation (docs/problem_formulation.md; report §III).
- From-scratch MCTS implementation (src/xrl/agents/mcts.py,
  src/xrl/envs/simulator.py) with tree-dump schema and unit tests.
- Counterfactual rollout framework and DecisionRecord schema
  (src/xrl/analysis/).
- LLM explanation pipeline including prompt templates, Anthropic client
  with caching and cost cap, and cache-aware pipeline
  (src/xrl/explainer/).
- Evaluation metrics: fidelity, soundness, post-hoc inferability
  (src/xrl/eval/metrics.py).
- Final report drafting and figures.

## Andrew Wernersbach

- DQN baseline: SB3 wrapper, observation wrappers (image and symbolic),
  training and eval scripts, hyperparameter tuning
  (src/xrl/agents/dqn.py, src/xrl/envs/wrappers.py, configs/dqn_*.yaml).
- Reward-shaping wrapper and DQN re-training.
- Experimental runs and bookkeeping (results/dqn/*).
- Report review and co-authorship.

## Shared

- Execution plan and engineering log.
- Test suite.
- Proposal and prof-feedback response.
- Final report writing and submission.

---

_Andrew: please edit the Sean/Andrew split above to match what actually
happened, then copy the final list into `report/main.tex`._
