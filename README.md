# Counterfactual Explainability: Deep RL vs. Online Planning

ASEN 5264 Final Project, CU Boulder, Spring 2026

**Authors:** Sean Florez, Andrew Wernersbach
**Repo:** github.com/fl-sean03/counterfactual-xrl

## Research Question

Is suboptimal online planning more explainable than suboptimal deep RL,
specifically because the search tree records the agent's deliberation?

## Approach

Train DQN and MCTS agents on `MiniGrid-Dynamic-Obstacles-8x8-v0`, extract
counterfactual statistics from each (rollouts for DQN, tree stats for MCTS),
feed structured evidence to Claude, and compare generated explanations on
fidelity, soundness, and post-hoc inferability.

See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the full phased plan.

## Layout

```
src/xrl/
  agents/         # DQN (SB3 wrapper) + MCTS (from scratch)
  envs/           # Wrappers + deep-copy simulator
  analysis/       # Counterfactual rollouts + tree stats
  explainer/      # Anthropic client + prompts + pipeline
  eval/           # Metrics + eval runner
  utils/          # Seeding, IO
tests/            # pytest suite
configs/          # YAML configs per experiment
scripts/          # Entry points: train, eval, explain, smoke
results/          # Run artifacts (gitignored)
report/           # IEEE tex sources
docs/             # Design notes + problem formulation
Proposal/         # Original proposal + prof feedback
```

## Setup

```bash
make install        # creates/updates conda env `xrl`, installs package
make smoke          # sanity-check env loads and GPU is visible
make test           # run pytest
make lint           # ruff + black checks
```

## License / Release

Release status TBD at report submission (see EXECUTION_PLAN.md §5 Phase 8).

## References

See `Proposal/proposal.md` for the citation list and
`EXECUTION_PLAN.md` for the full research plan.
