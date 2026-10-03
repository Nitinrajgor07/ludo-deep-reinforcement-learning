# PhD Research: Deep Reinforcement Learning for Ludo

Continuation of published IEEE research: *"Optimizing Deep Reinforcement Learning for Board Games: A Comparative Study of DQN and PPO on Ludo with Structured State Encoding"*.

## Project Objective
Extend the comparative evaluation from two algorithms (Double Dueling DQN and PPO) to four modern Deep Reinforcement Learning architectures under rigorous empirical control:
1. **DQN** (Double Dueling DQN + PER)
2. **PPO** (Actor-Critic + GAE + Masked Policy)
3. **Model 3** (To be finalized by researcher)
4. **Model 4** (To be finalized by researcher)

## Project Layout
```text
phd_ludo_drl/
├── environment/      # Ludo 2-player MDP, rules, Gymnasium interface
├── encoding/         # 4-channel CNN, Flat, One-Hot encoders (128-D)
├── agents/           # BaseAgent, DQN, PPO, Model3, Model4
├── opponents/        # Random & Alhajry Heuristic Expert opponents
├── training/         # Multi-env vectorized training manager
├── evaluation/       # 5,000-game standardized benchmark
├── analysis/         # Ablation and stochasticity analysis
├── visualization/    # Publication figures and charts
├── config/           # Dataclass config manager, seeds, YAML
├── experiments/      # Run ID, experiment orchestrator
├── checkpoints/      # Model weight snapshots
├── logs/             # Structured JSON & console logger
├── results/          # Evaluation metrics & summary tables
└── docs/             # Technical specifications & project rules
```
