# ASEN 5264 Final Project — Context & Reference

## Timeline

- **Proposal due:** Now (Q1 — 10 pts, Gradescope submission: markdown or PDF)
- **Final report due:** End of semester (TBD)
- **Group size:** Up to 3 (single proposal per group)

---

## Proposal Requirements (< 1 page, outline OK)

1. **Problem Statement** — formulation as MDP, POMDP, optimization, or game
2. **Three Levels of Success:**
   - **Minimum working example** — baseline/first attempt (grade penalty if not achieved)
   - **Main approach** — confident implementation, expected in final report
   - **Stretch goal** — ambitious extension if time allows
3. **Implementation plan** — from-scratch vs. off-the-shelf components

---

## Final Report Requirements (4-8 pages, IEEE format)

- **Audience:** Class peers (assume familiarity with course concepts)
- **Outline:** Introduction, Background/Related Work, Problem Formulation (S,A,T,R), Solution Approach, Results (with uncertainty), Conclusion/Future Work, Contributions & Release
- **Min 5 citations** of related published work
- **Contributions** per team member
- **Release statement** (public or private)
- **Clearly indicate** from-scratch vs. off-the-shelf algorithms

---

## Course Topics Covered (Available for Project)

| Unit | Topic | Key Methods |
|------|-------|-------------|
| 1 | Probability & Bayesian Networks | Bayes rule, d-separation, BN factorization |
| 2 | MDPs & Dynamic Programming | Value iteration, policy iteration |
| 3 | Online MDPs | Monte Carlo rollouts, UCB1, MCTS |
| 4 | Tabular RL | Q-learning, SARSA, epsilon-greedy, Thompson sampling, policy gradient, neural nets |
| 5 | Deep RL | DQN, advanced policy gradient (upcoming) |
| 6 | POMDPs | Particle filters, QMDP, SARSOP (upcoming) |

---

## Tool Stack

- **Julia** primary language (Python allowed)
- **POMDPs.jl** — standard MDP/POMDP interface
- **CommonRLInterface.jl** — RL environment interaction
- **SARSOP.jl** — near-optimal POMDP solver
- **Flux.jl** — deep learning
- **DMUStudent.jl** — course evaluation package
- **D3Trees.jl** — tree visualization
- **StaticArrays.jl** — efficient state representations
- **Plots.jl** — visualization

---

## Project Ideas from Course

### Straightforward
1. **Apply class methods to own research** — MDP/POMDP formulation of research problem
2. **Billiards shot engine** — long-horizon planning (existing codebase: github.com/donceykong/billiards-rl)
3. **Skiing robot** — (PO)MDP descent strategy
4. **RL paper reproduction** — recreate published results + additional experiment
5. **DESPOT solver variant** — non-determinized vs. determinized comparison
6. **POMDPs.jl solver contribution** — e.g., neural fitted value iteration
7. **Laser Tag Belief MDP** — game-theoretic planning (contact: Himanshu Gupta)
8. **Nash equilibria package** — lrsNash for bimatrix games
9. **Satellite RL** — BSK-RL framework (github.com/AVSLab/bsk_rl)
10. **Human-in-the-loop Atari** — imitation learning / multiplayer RL
11. **POMDPs.jl domain contribution** — new problem domains
12. **Decisions.jl contribution** — multi-agent extensions (contact: mel.krusniak@colorado.edu)

### Research-Level (2025)
1. **Parallel rollouts for ESP navigation** — replace HJB/PDE rollouts with parallel motion planning in crowded environments (contact: himanshu.gupta@colorado.edu)
2. **Safety-guided MCTS** — offline stochastic safety barriers to prune/bias MCTS-DPW (contact: himanshu.gupta@colorado.edu)
3. **Value-gradient MCTS** — gradient-informed tree search for continuous spaces, possible POMDP extension (contact: himanshu.gupta@colorado.edu)

### Previous student projects
- Available at: github.com/zsunberg/CU-DMU-Materials/tree/master/previous/projects

---

## Sean's Completed Work (Skill Baseline)

| HW | Topic | Implementation |
|----|-------|---------------|
| HW1 | Probabilistic Models | Elementwise max of matrix-vector products |
| HW2 | MDPs | Value iteration, UnresponsiveACASMDP |
| HW3 | Online MDPs | MCTS with heuristic rollouts |
| HW4 | Tabular RL | Q-Learning, SARSA, Flux.jl neural nets, bandits |

**Strengths:** Solid RL fundamentals, clean Julia code, empirical evaluation with learning curves, neural network training with Flux.jl, proper epsilon-greedy exploration.

---

## Relevant Background (Sean's Research)

- Graduate student in chemical engineering (CHEN 5838 concurrent)
- Works with molecular dynamics (LAMMPS), DFT (Quantum ESPRESSO)
- Builds agent orchestration platforms (AgentGate, MatterStack)
- Experience with GPU computing (RTX 5080, CUDA)
- Python/Julia/Rust multilingual
- Could leverage research domain (materials science, molecular simulation) for "apply to research" project idea
