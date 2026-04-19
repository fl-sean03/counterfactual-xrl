# Counterfactual XRL Project

## Quick Reference

- **Course:** ASEN 5264 (Decision Making under Uncertainty)
- **Project:** Counterfactual Explainability, Deep RL vs. Online Planning
- **Environment:** MiniGrid-Dynamic-Obstacles-8x8-v0
- **Language:** Python (MiniGrid/SB3 ecosystem)
- **Student:** Sean Florez (sean.florez@colorado.edu)
- **Collaborator:** andrewwer416 (GitHub)

## Status

- [x] Project ideas submitted (3 ideas, 10 pts)
- [x] Proposal written
- [ ] Proposal submitted to Gradescope
- [ ] Implementation (not started)
- [ ] Final report (4-8 pages, IEEE format)

## Planned Architecture

```
src/
├── agents/          # DQN (SB3) + MCTS (from scratch)
├── analysis/        # MC rollout framework, tree extraction, stats
└── explainer/       # LLM explanation pipeline, chatbot (stretch)
```

## Key Decisions

- Python over Julia for this project (MiniGrid + SB3 are Python-native)
- Stable-Baselines3 for DQN (simple, well-documented)
- Claude API for LLM explanations
- MCTS from scratch (adapted from HW3 concepts, ported to Python)

## Professor's Feedback (Key Direction)

> "Is suboptimal online planning more explainable than suboptimal reinforcement learning?
> (because the 'training data' is in the tree)"

This is the central question driving the comparison. The hypothesis: MCTS explanations
are higher fidelity because the search tree provides direct evidence for decisions,
while DQN requires post-hoc rollout analysis to reconstruct similar evidence.

## Post-Proposal Feedback

Prof. Sunberg approved the proposal and suggested a **pilot human study** as a depth
extension, have participants unfamiliar with the environment judge whether the
generated explanations make sense. Full feedback: `Proposal/feedback.md`.
