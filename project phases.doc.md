# PhD Research Project: 16-Phase Execution Plan

## Overview
This document formalizes the 16-phase development and research lifecycle for the PhD project:
*"Optimizing Deep Reinforcement Learning for Board Games: A Comparative Study of DQN, PPO, and Advanced DRL Architectures on Ludo with Structured State Encoding"*.

---

### Phase 1 — Project Foundation (Current Phase)
- **Goal:** Clean, modular, reproducible project foundation without implementing RL algorithms yet.
- **Tasks:**
  - Create standard modular directory layout.
  - Setup requirements and dependency verification.
  - Implement configuration management (YAML/dataclasses).
  - Implement structured logging infrastructure.
  - Implement experiment identification support (`run_id`, metadata hashing).
  - Implement random seed registry across Python, NumPy, PyTorch.
  - Establish basic project validation tests.

### Phase 2 — Complete Ludo Game Rules
- **Goal:** Authoritative, bug-free implementation of the 2-player Ludo game mechanics.
- **Tasks:**
  - Board representation (52 track squares, 5-square home column per player).
  - Token positions and lifecycle states (`YARD`, `TRACK`, `HOME`, `FINISHED`).
  - Safe squares and immunity semantics.
  - Blockade mechanics (impassable barrier formed by 2+ friendly tokens).
  - Capture mechanics (sending single opponent token back to `YARD`).
  - Dice roll logic (1..6) and bonus roll on rolling 6.
  - Legal move generation and exact-roll finish criteria.

### Phase 3 — Environment Lifecycle & Gymnasium Interface
- **Goal:** Full MDP environment with deterministic seed control and action masking.
- **Tasks:**
  - Gymnasium-compliant `reset()` and `step()` functions.
  - Binary legal action mask generation ($M \in \{0, 1\}^4$).
  - Illegal action rejection without state corruption.
  - Episode termination and winner declaration.
  - Comprehensive environment lifecycle test suite.

### Phase 4 — State Representation & Encoders
- **Goal:** Pure, deterministic feature extraction into standardized 128-D latent vectors.
- **Tasks:**
  - Structured 4-Channel Spatial CNN Encoder (preserves 2D board geometry).
  - Flat Encoder (30/240-D concatenation into 2 FC layers $\to 128$-D).
  - One-Hot Categorical Index Encoder ($\to 128$-D).
  - Feature normalization layers and dimension contracts.

### Phase 5 — Centralized Reward Engine
- **Goal:** Exact replication of domain-grounded reward values from published study.
- **Tasks:**
  - Token Release: $+0.25$
  - Defensive Move: $+0.20$
  - Capture Opponent: $+0.15$
  - Advance Toward Home: $+0.10$
  - Form Blockade: $+0.05$
  - Win Game: $+1.00$
  - Knocked by Opponent: $-0.25$
  - Lose Game: $-1.00$
  - Reward-state transition consistency validation.

### Phase 6 — Standard Baseline Opponents
- **Goal:** Implement non-learning baselines for training and evaluation.
- **Tasks:**
  - Random Opponent (uniform sampling over masked legal actions).
  - Heuristic Expert Opponent (Alhajry 2012 priority: Release on 6 > Defend > Capture > Advance).
  - Validation of expert win rate against random (~66–70%).

### Phase 7 — Model 1: Double Dueling DQN with PER
- **Goal:** Reproduce published DQN results.
- **Tasks:**
  - Value stream ($256$ units) & Advantage stream ($256$ units).
  - Prioritized Experience Replay (50k capacity, $\alpha=0.6, \beta=0.4 \to 1.0$).
  - Action masking on Q-values.
  - Target network soft update ($\tau=0.005$).
  - $\epsilon$-greedy exploration ($1.0 \to 0.05$ over 100k steps).

### Phase 8 — Model 2: PPO Actor-Critic with GAE
- **Goal:** Reproduce published PPO results.
- **Tasks:**
  - Shared encoder with policy and value heads.
  - Action-masked softmax policy with $-\infty$ logit substitution.
  - 8 parallel environment rollouts (128 steps/env).
  - Generalized Advantage Estimation ($\lambda=0.95, \gamma=0.99$).
  - Clipped surrogate loss ($\epsilon=0.2$) and mini-batch optimization (256 batch size, 4 epochs).

### Phase 9 — Standardized Training Pipeline
- **Goal:** Scientifically rigorous training orchestrator.
- **Tasks:**
  - Unified experiment configuration loader.
  - Multi-seed training coordinator (5 seeds per model).
  - Checkpoint manager and evaluation callbacks.
  - Metric loggers (JSON, TensorBoard-compatible summaries).

### Phase 10 — Standardized 5,000-Game Evaluation Protocol
- **Goal:** Strict evaluation mirroring Alhajry et al. (2012) and published benchmark.
- **Tasks:**
  - 5,000 independent games vs Random and vs Expert.
  - Win rate, loss rate, average reward, episode length, token finish rates, capture counts.
  - Variance and confidence interval reporting across random seeds.

### Phase 11 — Model 3 Implementation (Pending Selection)
- **Goal:** Implement the 3rd RL algorithm after explicit researcher approval.
- **Tasks:**
  - Implement algorithm adhering to `BaseAgent` and $128$-D shared state input.
  - Smoke tests and training validation.

### Phase 12 — Model 4 Implementation (Pending Selection)
- **Goal:** Implement the 4th RL algorithm after explicit researcher approval.
- **Tasks:**
  - Implement algorithm adhering to `BaseAgent` and $128$-D shared state input.
  - Smoke tests and training validation.

### Phase 13 — 4-Model Comparative Experiments
- **Goal:** Head-to-head empirical comparison across DQN, PPO, Model 3, and Model 4.
- **Tasks:**
  - Run multi-seed experiments across identical 500k-step budgets.
  - Track sample efficiency, convergence velocity, and asymptotic win rates.

### Phase 14 — Ablation Studies & Stochasticity Analysis
- **Goal:** Isolate representation benefits vs algorithmic benefits.
- **Tasks:**
  - Encoder ablation: Flat vs Structured CNN across all algorithms.
  - Reward ablation: Sparse vs Shaped vs Dense.
  - Environmental stochasticity analysis: Perturbing dice variance from $1.0\times$ to $1.4\times$.

### Phase 15 — Publication-Ready Visualization & Tables
- **Goal:** Generate publication figures and LaTeX-ready summary tables.
- **Tasks:**
  - Smoothed training curves with shaded confidence bands.
  - Head-to-head win rate comparison bar charts.
  - LaTeX tables with mean $\pm$ standard error.

### Phase 16 — Research Integration, Verification & Defense Deliverables
- **Goal:** Final packaging of PhD codebase and replication package.
- **Tasks:**
  - Freeze models, checkpoints, logs, and artifacts.
  - Full automated regression test pass.
  - Replication README and complete research report generation.
