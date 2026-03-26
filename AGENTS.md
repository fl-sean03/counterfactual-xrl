# Counterfactual XRL Project

## Quick Reference

- **Course:** ASEN 5264 (Decision Making under Uncertainty)
- **Project:** Counterfactual Explainability — Deep RL vs. Online Planning
- **Environment:** MiniGrid-Dynamic-Obstacles-8x8-v0
- **Language:** Python (MiniGrid/SB3 ecosystem)
- **Student:** Sean Florez (sean.florez@colorado.edu)

## Status

- [x] Project ideas submitted (3 ideas, 10 pts)
- [x] Proposal written
- [ ] Proposal submitted to Gradescope
- [ ] DQN agent trained
- [ ] MCTS agent implemented
- [ ] Counterfactual rollout framework
- [ ] LLM explanation pipeline
- [ ] MCTS vs DQN comparison
- [ ] Final report (4-8 pages, IEEE format)

## Architecture

```
src/
├── agents/
│   ├── dqn_agent.py      # DQN training + inference (SB3)
│   └── mcts_agent.py     # MCTS planner for MiniGrid
├── analysis/
│   ├── rollout.py         # MC rollout framework
│   ├── tree_extractor.py  # Extract stats from MCTS tree
│   └── stats.py           # Statistical comparison utilities
└── explainer/
    ├── pipeline.py        # Log -> LLM -> explanation
    ├── prompts.py         # Prompt templates
    └── chatbot.py         # Interactive interface (stretch)
```

## Key Decisions

- Python over Julia for this project (MiniGrid + SB3 are Python-native)
- Stable-Baselines3 for DQN (simple, well-documented)
- Claude API for LLM explanations
- MCTS from scratch (adapted from HW3 concepts, ported to Python)

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train DQN
python src/agents/dqn_agent.py

# Run experiments
python experiments/run_comparison.py
```

## Professor's Feedback (Key Direction)

> "Is suboptimal online planning more explainable than suboptimal reinforcement learning?
> (because the 'training data' is in the tree)"

This is the central question driving the comparison. The hypothesis: MCTS explanations
are higher fidelity because the search tree provides direct evidence for decisions,
while DQN requires post-hoc rollout analysis to reconstruct similar evidence.
