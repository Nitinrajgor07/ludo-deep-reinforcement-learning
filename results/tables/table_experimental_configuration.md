### Table 1: Experimental Architecture & Protocol Configuration

| Dimension             | Specification                                                                           | Module                               |
|:----------------------|:----------------------------------------------------------------------------------------|:-------------------------------------|
| Environment           | Two-Player Stochastic Ludo MDP (v0.1.0)                                                 | phd_ludo_drl.environment.ludo_env    |
| Board Geometry        | 52-Tile Circular Outer Track + 8 Safe Squares + 5-Square Private Home Column            | Environment Core                     |
| State Representation  | Structured Spatial Tensor (4x52) + 12 Context Features = 128-D Latent State             | phd_ludo_drl.encoding.cnn_encoder    |
| Encoder Architecture  | Conv1d (Circular Padding) [32 -> 64 filters] + AdaptiveAvgPool1d + Linear(76, 128)      | StructuredCNNEncoder                 |
| Action Space          | Discrete(4) — Token Selection strictly constrained by Action Mask                       | phd_ludo_drl.environment.action_mask |
| Illegal Action Rate   | Strictly 0.00% across all algorithms (-10^9 logit suppression)                          | Action Mask Contract                 |
| Model 1: DQN          | Double Dueling DQN + Prioritized Replay (50k, alpha=0.6, beta=0.4->1.0), soft tau=0.005 | DQNAgent                             |
| Model 2: PPO          | Actor-Critic Network, GAE lambda=0.95, epsilon-clip=0.2, entropy bonus 0.01             | PPOAgent                             |
| Model 3: Discrete SAC | Twin Q-critics, discrete expectation targets, auto-entropy alpha tuning, tau=0.005      | DiscreteSACAgent                     |
| Model 4: C51 DQN      | 51-Atom Categorical Return Distribution [-20, +30], delta_z=1.0, cross-entropy          | CategoricalDQNAgent                  |
| Master Seed Sequence  | S = {42, 101, 202, 303, 404}; Eval Base Seed = S + 1000 = 1042                          | FINAL_EXPERIMENT_PROTOCOL.md         |
| Evaluation Tournament | 250 Games per Matchup; Strict 50% P0 / 50% P1 alternating seat schedule                 | StandardizedEvaluator                |

*Generated automatically from verified experimental data adhering to Final Experiment Protocol.*
