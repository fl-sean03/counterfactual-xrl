# Counterfactual Explainability: Deep RL vs. Online Planning

ASEN 5264 Final Project, CU Boulder, Spring 2026

**Authors:** Sean Florez, Andrew Wernersbach
**Repo:** github.com/fl-sean03/counterfactual-xrl

## Research Question

Does online-planning evidence support different natural-language
explanations than post-hoc counterfactual evidence from a learned RL
policy?

## Approach

Train PPO and MCTS agents on `MiniGrid-Dynamic-Obstacles-8x8-v0`, extract
counterfactual rollout statistics for PPO and tree statistics for MCTS,
feed structured evidence to a GPT-4o explainer, and compare generated
explanations on fidelity, soundness, and post-hoc inferability. DQN
variants are included as diagnostic ablations because they collapse to
stall or near-stall policies on this sparse-reward environment.

See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the full phased plan.

## Layout

```
src/xrl/
  agents/         # DQN/PPO (SB3 wrappers) + MCTS (from scratch)
  envs/           # Wrappers + deep-copy simulator
  analysis/       # Counterfactual rollouts + tree stats
  explainer/      # OpenAI/Anthropic clients + prompts + pipeline
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

The authors grant permission for the final report to be posted publicly.

## References

See `Proposal/proposal.md` for the citation list and
`EXECUTION_PLAN.md` for the full research plan.
