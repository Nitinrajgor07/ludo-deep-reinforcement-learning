# PhD Research System Architecture: DRL for Ludo

## 1. High-Level Architectural Overview

The platform is engineered around strict modularity, mathematical rigor, reproducibility, and unbiased scientific comparison across four Deep Reinforcement Learning algorithms on the stochastic multi-agent board game of Ludo.

```text
+-----------------------------------------------------------------------------------+
|                                Research Pipeline                                  |
+-----------------------------------------------------------------------------------+
|  [ Configuration Manager ] <---> [ Experiment Identifier & Random Seed Registry ]  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
                +---------------------------------------------------+
                |             Ludo MDP Environment (2P & 4P)        |
                |   - 52 Common Track Squares + Home Columns        |
                |   - Safe Squares, Blockades, Knock/Yard Reset     |
                |   - Stochastic Dice (1..6) & Bonus Turn on 6      |
                |   - Legal Action Mask Generator (0..3)            |
                +-------------------------+-------------------------+
                                          |
                                          v
                +---------------------------------------------------+
                |            State Preprocessing Encoders           |
                |  - Frozen 4-Channel Spatial CNN Encoder (4x52)    |
                |  - 12-D Frozen Context Feature Vector             |
                |  - Standardized Fixed Latent Projection: 128-D    |
                +-------------------------+-------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Modular Agent Framework                               |
+-----------------------------------------------------------------------------------+
|  Model 1: Double Dueling DQN + PER (Value & Advantage Streams, tau=0.005)         |
|  Model 2: Actor-Critic PPO + GAE (Masked Softmax Policy Head + Critic Head)       |
|  Model 3: Discrete Soft Actor-Critic (SAC) (Twin Q-Networks, Auto Entropy Alpha)  |
|  Model 4: Categorical C51 (Distributional RL, 51-Atom Return Distribution)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            Standardized Opponents                                 |
+-----------------------------------------------------------------------------------+
|  - Uniform Random Opponent (Uniform legal sampling)                              |
|  - Heuristic Expert Opponent (Alhajry Priority: Release > Defend > Capture > Race)|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Training & Evaluation Protocols                            |
+-----------------------------------------------------------------------------------+
|  - Vectorized Multi-Env Training Manager with Independent Seeds                   |
|  - Metric Logger & Model Checkpointing (.pt)                                      |
|  - Controlled Evaluation Benchmark (Balanced 50/50 Seated Fairness)              |
|  - Inferential Statistical Pipeline (Welch's t, Mann-Whitney U, Cohen's d)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Markov Decision Process (MDP) Formalization

The two-player competitive Ludo environment is modeled as a formal discrete-time MDP tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

1. **State Space $\mathcal{S}$:**
   - Full board token coordinates: 4 tokens for Player 0, 4 tokens for Player 1.
   - States: `YARD` ($-1$), `TRACK` ($0 \dots 51$), `HOME_COLUMN` ($100 \dots 104$), `GOAL` ($999$).
   - Current turn player: $p \in \{0, 1\}$.
   - Current dice value: $d \in \{1, 2, 3, 4, 5, 6\}$.
   - Consecutive six count: $c \in \{0, 1, 2\}$.

2. **Action Space $\mathcal{A}$:**
   - Discrete 4-action space: $\mathcal{A} = \{0, 1, 2, 3\}$, commanding token $i \in \{0, 1, 2, 3\}$ of the active player to execute the valid move for roll $d$.
   - **Action Masking $M(s) \in \{0, 1\}^4$:** $M(s)_i = 1$ if token $i$ can legally move under dice $d$, and $0$ otherwise. Invalid actions are suppressed by setting logits to $-10^9$.
   - When $\sum_i M(s)_i = 0$ (no legal move available), the turn automatically advances without agent penalty.

3. **Transition Dynamics $\mathcal{P}(s' \mid s, a)$:**
   - Deterministic token update given $(s, a, d)$, followed by stochastic environment transition sampling next dice roll $d' \sim \text{DiscreteUniform}(1, 6)$.
   - Bonus roll granted on $d = 6$ or upon successful opponent capture.

4. **Reward Function $\mathcal{R}(s, a, s')$:**
   - Symmetrical shaped reward formulation grounded in published baselines:
     $$r_t = +0.25 \cdot \mathbf{1}_{\text{release}} + 0.20 \cdot \mathbf{1}_{\text{defend}} + 0.15 \cdot \mathbf{1}_{\text{capture}} + 0.10 \cdot \mathbf{1}_{\text{advance}} + 0.05 \cdot \mathbf{1}_{\text{blockade}} + 1.00 \cdot \mathbf{1}_{\text{win}} - 0.25 \cdot \mathbf{1}_{\text{knocked}} - 1.00 \cdot \mathbf{1}_{\text{loss}}$$

5. **Discount Factor:** $\gamma = 0.99$.

---

## 3. State Representation Encoders

All RL algorithms interface with a standardized latent representation $\mathbf{z} \in \mathbb{R}^{128}$ extracted by the frozen encoding pipeline:

### 3.1 Structured Spatial CNN Encoder (`StructuredCNNEncoder`)
- **Input Tensor:** Shape $(B, 4, 52)$ representing spatial token planes:
  - Channel 0: Active player token occupancy.
  - Channel 1: Opponent token occupancy.
  - Channel 2: Safe square indicators ($1.0$ at 8 safe squares, $0.0$ elsewhere).
  - Channel 3: Blockade indicators ($1.0$ at tiles with $\ge 2$ friendly tokens, $0.0$ elsewhere).
- **Context Vector:** Shape $(B, 12)$ containing invariant normalized features:
  - Active yard count, opponent yard count, active home count, opponent home count, active goal count, opponent goal count, normalized dice roll ($d/6.0$), active player P0 one-hot, active player P1 one-hot, turn fraction, bonus roll flag, capture bonus flag.
- **Layers:**
  1. `Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1, padding_mode='circular')` + ReLU
  2. `Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular')` + ReLU
  3. `AdaptiveAvgPool1d(1)` $\to$ Flatten ($64$ features)
  4. Concatenate with 12-D context vector ($64 + 12 = 76$) $\to$ `Linear(76, 128)` + ReLU $\to 128$-D latent state.

---

## 4. Deep RL Algorithm Architectures

The framework evaluates four distinct mathematical paradigms under identical frozen environment and state contracts:

1. **Model 1: Double Dueling DQN + PER (`phd_ludo_drl/agents/dqn/`)**
   - Value-based Q-learning with decoupled action selection and evaluation.
   - Dueling architecture: Value stream $V(s) \in \mathbb{R}$ and Advantage stream $A(s, a) \in \mathbb{R}^4$.
   - Prioritized Experience Replay (PER) with proportional priority sampling ($\alpha = 0.6$) and importance sampling weights ($\beta = 0.4 \to 1.0$).
   - Polyak soft target updates ($\tau = 0.005$).

2. **Model 2: Actor-Critic PPO (`phd_ludo_drl/agents/ppo/`)**
   - On-policy policy gradient with clipped surrogate objective ($\epsilon = 0.2$).
   - Shared 128-D representation encoder with dual heads: masked categorical policy head $\pi_\theta(a \mid s)$ and state-value critic head $V_\phi(s)$.
   - Generalized Advantage Estimation (GAE, $\lambda = 0.95$).

3. **Model 3: Discrete SAC (`phd_ludo_drl/agents/model3/`)**
   - Maximum-entropy actor-critic adapted for discrete action spaces.
   - Twin Q-networks $Q_{\theta_1}, Q_{\theta_2}$ to mitigate overestimation bias.
   - Exact expectation target computation over all 4 discrete actions.
   - Automatic entropy temperature $\alpha$ adjustment.

4. **Model 4: Categorical C51 (`phd_ludo_drl/agents/model4/`)**
   - Distributional reinforcement learning modeling value distributions over $N = 51$ discrete atoms ($V_{\min} = -2.0, V_{\max} = +2.0$).
   - Categorical Bellman projection onto the support grid.
   - Cross-entropy loss minimization with action masking.

---

## 5. Repository Layout

```text
PROJECT/
├── phd_ludo_drl/             # Core research library
│   ├── environment/          # Gymnasium 2P/4P Ludo MDP, rules, tokens, action masking
│   ├── encoding/             # State encoders (Structured CNN, Flat, Protocol Validation)
│   ├── agents/               # 4 DRL architectures (DQN, PPO, Discrete SAC, C51)
│   ├── opponents/            # Standardized baselines (Uniform Random, Heuristic Expert)
│   ├── training/             # Unified training runner, checkpointing, factory
│   ├── evaluation/           # 200,000-game tournament pipeline, seated fairness
│   ├── analysis/             # Inferential statistics (Welch's t, Mann-Whitney U, Cohen's d)
│   ├── visualization/        # Publication-grade figure plotting
│   └── config/               # Configuration management and seed registry
├── configs/                  # Experiment configuration YAMLs
├── experiments/              # Raw experiment runs, configs, and pilot logs
├── results/                  # Generated figures, tables, ablations, and robustness data
├── scripts/                  # Standalone execution CLI scripts across research phases
├── tests/                    # Comprehensive unit and integration test suite (268 tests)
├── web/                      # FastAPI + Vanilla JS Light Glassmorphic Research Platform
├── docs/                     # Consolidated research reports and documentation
├── pyproject.toml            # Build specification & dependencies
├── requirements.txt          # Python runtime requirements
├── README.md                 # Primary project overview
└── PROJECT_ARCHITECTURE.md   # This document
```
