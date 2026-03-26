# Counterfactual Explainability: Deep RL vs. Online Planning

**Sean Florez** (sean.florez@colorado.edu)

## Problem Statement

When a DQN agent picks an action, we know what it chose but not why. The Q-values behind the decision live inside neural network weights, so the reasoning is opaque. Online planners like MCTS are different: the search tree stores simulated outcomes for every action the agent considered, so the reasoning process is recorded and inspectable. This project investigates a question that comes up naturally: is suboptimal online planning more explainable than suboptimal deep RL, specifically because the "training data" is in the tree?

We will work in MiniGrid-Dynamic-Obstacles-8x8, which is a gridworld where an agent navigates around randomly moving obstacles to reach a goal. This is formulated as an MDP. The state includes agent position and orientation, obstacle positions, and goal location. The action space is {turn left, turn right, move forward}. Reaching the goal gives a positive reward scaled by speed, while hitting an obstacle gives -1 and ends the episode. The dynamic obstacles and partial observability (the agent sees a 7x7 egocentric view) make this environment a good testbed because near-collision states create decisions that genuinely need explaining.

Our approach: train both a DQN agent and an MCTS planner on this environment, then build a system that generates counterfactual explanations for each agent's decisions using Monte Carlo rollouts and translates those into natural language with an LLM.

## Levels of Success

**1. Minimum Working Example.** Train a DQN agent on MiniGrid-Dynamic-Obstacles-8x8 using Stable-Baselines3. Build a counterfactual rollout framework that, at each step of the trained policy, runs 100+ MC rollouts per available action and logs mean return, collision rate, success rate, and steps to termination. Produce statistical summaries showing why the chosen action was preferred over alternatives. This baseline gives us a working explanation system for DQN decisions grounded in simulated outcomes.

**2. Main Approach.** Implement an MCTS agent for the same environment and extract equivalent counterfactual statistics from the search tree: visit counts, child node values, and simulated collision/success rates per action. Then build an LLM pipeline that converts the structured rollout logs (from DQN) and tree logs (from MCTS) into natural language explanations. Compare the two on explanation quality using fidelity (does the explanation match what actually happened?) and soundness (is the stated reasoning correct?), following metrics from recent XRL evaluation work (Belouadah et al., 2025). This directly tests whether tree-based planning produces better explanations than post-hoc analysis of a learned policy.

**3. Stretch Goal.** Build an interactive chatbot where a user steps through an agent's trajectory and asks questions like "why did you turn left here?" or "what would have happened if you went forward?" The chatbot answers from the rollout data and supports follow-up questions, so a user can explore the agent's behavior conversationally.

## Implementation Plan

**From scratch:** counterfactual MC rollout framework and logging, MCTS agent adapted for MiniGrid (building on our HW3 MCTS work), LLM explanation pipeline and prompt design, explanation evaluation metrics, chatbot interface (stretch).

**Off-the-shelf:** MiniGrid environment (Farama Foundation), DQN training (Stable-Baselines3 or CleanRL), LLM API (Claude), standard Python stack (numpy, matplotlib, pandas).
