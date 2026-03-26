# Counterfactual Explainability: Deep RL vs. Online Planning

ASEN 5264 Final Project — CU Boulder, Spring 2026

## Overview

Comparing the explainability of deep RL (DQN) and online planning (MCTS) on the MiniGrid-Dynamic-Obstacles-8x8 environment. The core question: is online planning inherently more explainable because the search tree records the agent's deliberation?

**Approach:**
1. Train a DQN agent and implement an MCTS planner on the same environment
2. Build a counterfactual rollout framework that simulates alternative actions at each decision point
3. Use an LLM to translate rollout/tree statistics into natural language explanations
4. Compare explanation quality between the two paradigms

## Project Structure

```
├── proposal/           # Project proposal
├── src/
│   ├── agents/         # DQN and MCTS agent implementations
│   ├── analysis/       # MC rollout framework, logging, statistics
│   └── explainer/      # LLM explanation pipeline
├── experiments/        # Experiment configs, scripts, results
├── notebooks/          # Exploration and visualization notebooks
├── report/             # Final report (IEEE format)
└── requirements.txt    # Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment

- **MiniGrid-Dynamic-Obstacles-8x8-v0**: 8x8 gridworld with moving obstacles
- State: agent position/orientation, obstacle positions, goal location
- Actions: turn left, turn right, move forward
- Reward: +1 (scaled by speed) for goal, -1 for collision

## Key References

- Milani et al. (2024) — Explainable RL survey
- Amitai et al. (2024) — COViz: counterfactual action outcome visualization
- Baier & Kaisers (2021) — Towards Explainable MCTS
- Belouadah et al. (2025) — LLMs for explainable deep RL
- Gajcin & Dusparic (2024) — Counterfactual explanations for RL
