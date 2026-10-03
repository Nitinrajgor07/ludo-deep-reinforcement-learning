# PhD Research System Architecture: DRL for Ludo

## 1. High-Level Architectural Overview

The platform is designed around strict separation of concerns, ensuring modularity, reproducibility, and rigorous scientific comparison across four Reinforcement Learning algorithms on the stochastic board game of Ludo.

```text
+-----------------------------------------------------------------------------------+
|                                Research Pipeline                                  |
+-----------------------------------------------------------------------------------+
|  [ Configuration Manager ] <---> [ Experiment Identifier & Random Seed Registry ]  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
                +---------------------------------------------------+
                |             Ludo MDP Environment (2P)             |
                |   - 52 Common Track Squares + 2x5 Home Columns   |
                |   - Safe Squares, Blockades, Knock/Yard Reset     |
                |   - Stochastic Dice (1..6) & Bonus Turn on 6      |
                |   - Legal Action Mask Generator (0..3)            |
                +-------------------------+-------------------------+
                                          |
                                          v
                +---------------------------------------------------+
                |            State Preprocessing Encoders           |
                |  - Structured 4-Channel Spatial CNN Encoder       |
                |  - Flat 30/240-D Baseline Vector Encoder          |
                |  - One-Hot Categorical Index Encoder              |
                |  --> Standardized Fixed Latent Output: 128-D      |
                +-------------------------+-------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                             Modular Agent Framework                               |
+-----------------------------------------------------------------------------------+
|  Model 1: Double Dueling DQN + PER (Value & Advantage Streams, tau=0.005)         |
|  Model 2: PPO Actor-Critic + GAE (Masked Softmax Policy Head + Value Head)        |
|  Model 3: [Pending Final Selection & Approval]                                    |
|  Model 4: [Pending Final Selection & Approval]                                    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            Standardized Opponents                                 |
+-----------------------------------------------------------------------------------+
|  - Random Opponent (Uniform legal sampling)                                       |
|  - Expert Opponent (Alhajry 2012 priority: Release > Defend > Capture > Advance)  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Training & Evaluation Protocols                            |
+-----------------------------------------------------------------------------------+
|  - Vectorized Multi-Env Training Manager                                          |
|  - Metric Logger & Model Checkpointing                                            |
|  - 5,000-Game Evaluation Benchmark against Baselines                              |
|  - Statistical Significance & Ablation Analysis Engine                            |
+-----------------------------------------------------------------------------------+
```

## 2. Directory Structure

```text
phd_ludo_drl/
├── environment/      # Pure Ludo game logic, board state, token lifecycle, Gymnasium API
├── encoding/         # State representation encoders (Structured CNN, Flat, One-Hot -> 128-D)
├── agents/           # RL algorithm implementations adhering to BaseAgent interface
│   ├── base_agent.py
│   ├── dqn/          # Model 1: Double Dueling DQN + Prioritized Experience Replay
│   ├── ppo/          # Model 2: PPO Actor-Critic + GAE + Action Masking
│   ├── model3/       # Model 3: Reserved for chosen algorithm (unimplemented)
│   └── model4/       # Model 4: Reserved for chosen algorithm (unimplemented)
├── opponents/        # Standardized benchmark opponents (Random, Alhajry Heuristic Expert)
├── training/         # Standardized training loops, parallel runners, replay management
├── evaluation/       # 5,000-game evaluation protocol, head-to-head tournament runner
├── analysis/         # Ablation studies, dice stochasticity variance analysis
├── visualization/    # Publication-ready figure generation (win rate, reward, convergence)
├── config/           # Centralized configuration dataclasses, YAML loader, seed manager
├── experiments/      # Experiment manifest, run ID generator, execution orchestrator
├── checkpoints/      # Model weight storage (.pt) with metadata
├── logs/             # Structured JSON & console loggers
├── results/          # Tabular evaluation metrics, statistical records
└── docs/             # Technical specifications, phase docs, memory logs
```

## 3. Core Component Interfaces

1. **`LudoEnvironment`**: Implements standard `reset() -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`. Emits `action_mask` of shape `(4,)`.
2. **`StateEncoder`**: Transforms game state snapshot into a fixed-size latent representation $\mathbf{z} \in \mathbb{R}^{128}$.
3. **`BaseAgent`**: Implements `select_action(state, action_mask, deterministic=False) -> int`, `train_step(...)`, `save_checkpoint(path)`, and `load_checkpoint(path)`.
4. **`BaseOpponent`**: Provides reproducible baseline opponent policies.
