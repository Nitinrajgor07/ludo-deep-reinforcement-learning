# FINAL EXPERIMENT PROTOCOL
## Comparative Study of Four Deep Reinforcement Learning Algorithms on Ludo
### PhD Research Benchmark Specification — Version 1.0 (FROZEN)

---

## Document Control & Governance
- **Document Version:** 1.0.0 (FROZEN)
- **Status:** APPROVED & LOCKED
- **Effective Date:** October 3, 2026
- **Research Foundation:** Continuation of IEEE Paper: *"Optimizing Deep Reinforcement Learning for Board Games: A Comparative Study of DQN and PPO on Ludo with Structured State Encoding"* (Adithya Tej BM, Abhishek Sharma, Addwin Antony Stephen, Deekshith Gowda R, Nitin Rajgor).
- **Core Directive:** No hyperparameter, reward value, architectural parameter, seed sequence, or evaluation protocol documented in this specification may be modified silently. Any future alteration requires a formal version increment (e.g., v1.1.0) with an accompanying justification entry in Section 22.

---

## 1. Environment Specification & Version
- **Module:** `phd_ludo_drl.environment.ludo_env.LudoEnvironment`
- **Environment Version:** `v0.1.0`
- **MDP Type:** Two-player turn-based, stochastic, imperfect-information (from dice stochasticity), discrete Markov Decision Process.
- **Board Topology:**
  - Standard circular outer track of 52 tiles numbered `0` to `51`.
  - 8 Fixed Safe Squares (invulnerable to opponent capture): Indices `[0, 8, 13, 21, 26, 34, 39, 47]`.
  - Player 0 Track Entry: Tile `0` | Home Column Entry: Tile `50` $\to$ Home Column `[0..4]` $\to$ Goal.
  - Player 1 Track Entry: Tile `26` | Home Column Entry: Tile `24` $\to$ Home Column `[0..4]` $\to$ Goal.
- **Mechanics:**
  - Token release from yard requires an exact roll of `6`.
  - Capturing an opponent's token sends it back to the yard and awards a bonus roll.
  - Rolling a `6` awards a bonus roll (consecutive bonus turns are supported).
  - Blockades: Two or more tokens of the same color on a non-safe square form a blockade; opponents can neither land on nor pass through a blockade.
  - Home Column Entry: Tokens travel through 5 private home column tiles; landing in the Goal requires an **exact roll**.
  - Turn Advancement: When the active player has zero legal moves, the turn automatically advances (`auto_advance_no_moves=True`) to maintain Markovian state transitions.
  - Max Game Horizon: $1,000$ turns cutoff to terminate potential perpetual blockades.

---

## 2. Player Model & Turn Management
- **Total Players:** 2 (Player 0 and Player 1).
- **Tokens per Player:** 4 tokens per player ($4 \times 2 = 8$ tokens total).
- **Token States:** `YARD`, `TRACK`, `HOME_COLUMN`, `GOAL`.
- **Winning Condition:** A player wins immediately when all 4 of their tokens reach `GOAL`.
- **First-Mover Control:** In all head-to-head evaluation tournaments, seat positions alternate equally (50% Player 0, 50% Player 1) to eliminate first-mover advantage bias.

---

## 3. State Representation Architecture
Every model consumes an identical structured state representation:
- **Spatial Board Tensor $S \in \mathbb{R}^{4 \times 52}$:**
  - Channel 0: Active player token count per track tile ($t \in [0..51]$).
  - Channel 1: Opponent token count per track tile ($t \in [0..51]$).
  - Channel 2: Safe square spatial indicators ($1.0$ at safe tiles, $0.0$ elsewhere).
  - Channel 3: Blockade indicators ($1.0$ at tiles occupied by $\ge 2$ tokens of the same color).
- **Context Vector $C \in \mathbb{R}^{12}$:**
  - Index 0: Active player tokens in yard $[0..4]$ normalized by 4.
  - Index 1: Opponent tokens in yard $[0..4]$ normalized by 4.
  - Index 2: Active player tokens in home column $[0..4]$ normalized by 4.
  - Index 3: Opponent tokens in home column $[0..4]$ normalized by 4.
  - Index 4: Active player tokens in goal $[0..4]$ normalized by 4.
  - Index 5: Opponent tokens in goal $[0..4]$ normalized by 4.
  - Index 6: Current dice roll $[1..6]$ normalized by 6.
  - Index 7–8: Active player ID one-hot encoding (`[1, 0]` for P0, `[0, 1]` for P1).
  - Index 9: Turn fraction $(\text{turn} / 1000)$.
  - Index 10: Bonus roll flag ($1.0$ if rolling after 6, else $0.0$).
  - Index 11: Capture bonus flag ($1.0$ if rolling after capture, else $0.0$).
- **Ablation Baseline Representation:**
  - Flat 60-dimensional vector: 8 token positions (yard, track $0..51$, home $0..4$, goal) + dice + turn metrics (reserved for Phase 14 representation ablation).

---

## 4. Shared Encoder Specification
- **Module:** `phd_ludo_drl.encoding.cnn_encoder.StructuredCNNEncoder`
- **Circular Padding Geometry:**
  To reflect the closed circular loop of the 52-tile track without boundary distortion, Conv1d layers utilize `padding_mode='circular'`.
- **Layer Dimensions:**
  1. `Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1, padding_mode='circular')` $\to$ `ReLU()`
  2. `Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular')` $\to$ `ReLU()`
  3. `AdaptiveAvgPool1d(1)` $\to$ spatial feature vector of shape $(B, 64)$
  4. Context concatenation: $(B, 64 + 12) = (B, 76)$
  5. Dense projection: `Linear(76, 128)` $\to$ `ReLU()`
- **Output:** Standardized $128$-dimensional latent state feature vector $h(s) \in \mathbb{R}^{128}$.
- **Invariant:** All 4 RL models must interface with this exact encoder architecture.

---

## 5. Action Space Contract
- **Space:** Discrete action space of size 4: $\mathcal{A} = \{0, 1, 2, 3\}$.
- **Semantic Meaning:** Action index $i \in \{0, 1, 2, 3\}$ corresponds to selecting token $i$ of the active player to move with the current dice roll.
- **Pass Action:** There is NO explicit "pass" action. Turns without legal actions advance automatically.

---

## 6. Action Masking Layer
- **Module:** `phd_ludo_drl.environment.action_mask`
- **Mask Vector:** Binary vector $M(s) \in \{0, 1\}^4$ where $M_i = 1$ if moving token $i$ is legal under current game rules, else $0$.
- **Value-Based Action Masking (DQN, C51):**
  $$Q_{\text{masked}}(s, a) = Q(s, a) + (1 - M_a) \times (-10^9)$$
  $$\pi(s) = \operatorname{argmax}_a Q_{\text{masked}}(s, a)$$
- **Policy-Based Action Masking (PPO, Discrete SAC):**
  $$\operatorname{logits}_{\text{masked}}(s, a) = \operatorname{logits}(s, a) + (1 - M_a) \times (-10^9)$$
  $$\pi(a|s) = \operatorname{softmax}(\operatorname{logits}_{\text{masked}}(s, a))$$
- **Invariant:** Zero probability mass is assigned to illegal actions. Agent decision making is strictly constrained to $\mathcal{A}_{\text{legal}}(s)$.

---

## 7. Standardized Reward Engine
- **Module:** `phd_ludo_drl.environment.reward_calculator.RewardCalculator`
- **Published Research Baseline Reward Schedule:**

| Game Event | Reward Value | Rationale |
| :--- | :---: | :--- |
| **Token Release** | $+0.25$ | Incentivizes bringing tokens from yard onto track on roll of 6 |
| **Defensive Move** | $+0.20$ | Entering a safe square or home column to avoid vulnerability |
| **Capture Opponent Token** | $+0.15$ | High-value tactical action sending opponent token to yard |
| **Track Progress** | $+0.10$ | Advancing forward along the 52-tile track toward home column |
| **Blockade Formation** | $+0.05$ | Creating a two-token fortress obstructing opponent advance |
| **Win Game** | $+1.00$ | Terminal success reward (all 4 tokens reached goal) |
| **Token Captured (Victim)** | $-0.25$ | Penalty when an opponent knocks the player's token to yard |
| **Loss Game** | $-1.00$ | Terminal penalty when the opponent wins |
| **Illegal Move Attempt** | $-0.50$ | Safety fallback (strictly prevented by action mask) |

- **Empirical Control Rule:** The reward function is completely frozen. No reward shaping may be altered between models.

---

## 8. Model 1: Double Dueling DQN Configuration
- **Package:** `phd_ludo_drl.agents.dqn`
- **Architecture:** Double Dueling Q-Network with Prioritized Experience Replay (PER)
- **Value/Advantage Streams:**
  - Shared CNN encoder $\to 128$-D latent state
  - Value Stream: `Linear(128, 256) -> ReLU -> Linear(256, 1)`
  - Advantage Stream: `Linear(128, 256) -> ReLU -> Linear(256, 4)`
  - Dueling Aggregation: $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$
- **Hyperparameters:**
  - Learning Rate: $1 \times 10^{-4}$ (Adam, $\epsilon=10^{-8}$)
  - Discount Factor $\gamma$: $0.99$
  - Soft Target Update $\tau$: $0.005$ (Polyak averaging per step)
  - Replay Buffer Capacity: $50,000$ transitions
  - Batch Size: 64
  - PER Exponent $\alpha$: $0.6$
  - PER Importance Sampling $\beta$: $0.4 \to 1.0$ (linear annealing)
  - Exploration Schedule: $\epsilon$-greedy from $1.0 \to 0.05$ over $100,000$ steps
  - Gradient Clipping: Max norm $10.0$

---

## 9. Model 2: PPO Baseline Configuration
- **Package:** `phd_ludo_drl.agents.ppo`
- **Architecture:** Shared Actor-Critic Network with Action Masking
- **Heads:**
  - Policy Head $\pi(a|s)$: `Linear(128, 128) -> ReLU -> Linear(128, 4)`
  - Value Head $V(s)$: `Linear(128, 128) -> ReLU -> Linear(128, 1)`
- **Hyperparameters:**
  - Learning Rate: $3 \times 10^{-4}$ (Adam, $\epsilon=10^{-5}$)
  - Discount Factor $\gamma$: $0.99$
  - GAE Parameter $\lambda$: $0.95$
  - PPO Clip Epsilon $\epsilon_{\text{clip}}$: $0.20$
  - Value Loss Coefficient $c_1$: $0.50$
  - Entropy Bonus Coefficient $c_2$: $0.01$
  - Rollout Steps per Iteration: $128$ steps
  - Optimization Epochs: $4$ epochs per rollout
  - Mini-batch Size: $256$ (or buffer capacity if smaller)
  - Gradient Clipping: Max norm $0.50$

---

## 10. Model 3: Discrete SAC Configuration
- **Package:** `phd_ludo_drl.agents.model3`
- **Architecture:** Discrete Soft Actor-Critic (Christodoulou, 2019)
- **Networks:**
  - Twin Critics $Q_1(s, a), Q_2(s, a)$: Each with structured CNN encoder $\to$ `Linear(128, 128) -> ReLU -> Linear(128, 4)`
  - Twin Target Critics $Q_{\text{targ}, 1}, Q_{\text{targ}, 2}$: Soft Polyak updated
  - Actor Policy Network $\pi(a|s)$: Structured CNN encoder $\to$ `Linear(128, 128) -> ReLU -> Linear(128, 4)`
- **Target Value Computation:** Exact discrete expectation
  $$V(s') = \sum_{a \in \mathcal{A}_{\text{legal}}(s')} \pi(a|s') \cdot \left[ \min_{j=1,2} Q_{\text{targ}, j}(s', a) - \alpha \log \pi(a|s') \right]$$
- **Hyperparameters:**
  - Critic Learning Rate: $3 \times 10^{-4}$ (Adam)
  - Actor Learning Rate: $3 \times 10^{-4}$ (Adam)
  - Temperature $\alpha$ Learning Rate: $3 \times 10^{-4}$ (Adam)
  - Discount Factor $\gamma$: $0.99$
  - Soft Target Update $\tau$: $0.005$
  - Replay Buffer Capacity: $50,000$ transitions
  - Batch Size: 64
  - Initial Temperature $\alpha$: $0.20$
  - Target Entropy $\bar{\mathcal{H}}$: $\log(4) \times 0.98 \approx 1.3586$ (automatic dual gradient tuning)
  - Gradient Clipping: Max norm $10.0$

---

## 11. Model 4: Categorical DQN (C51) Configuration
- **Package:** `phd_ludo_drl.agents.model4`
- **Architecture:** Distributional Categorical DQN over 51 Atoms (Bellemare et al., 2017)
- **Networks:**
  - Distributional Head: `Linear(128, 128) -> ReLU -> Linear(128, 4 * 51)`
  - Reshaped to $(B, 4, 51)$ with `softmax(dim=-1)`
- **Support Parameters:**
  - Number of Atoms $N$: $51$
  - Return Range: $[V_{\min}, V_{\max}] = [-20.0, 30.0]$
  - Atom Spacing $\Delta z$: $(30.0 - (-20.0)) / (51 - 1) = 1.0$
- **Hyperparameters:**
  - Learning Rate: $1 \times 10^{-4}$ (Adam, $\epsilon=10^{-8}$)
  - Discount Factor $\gamma$: $0.99$
  - Soft Target Update $\tau$: $0.005$
  - Replay Buffer Capacity: $50,000$ transitions
  - Batch Size: 64
  - Exploration: $\epsilon$-greedy from $1.0 \to 0.05$ over $100,000$ steps
  - Gradient Clipping: Max norm $10.0$

---

## 12. Random Seed Methodology
- **Master Research Seeds:** 5 standardized seeds:
  $$\mathcal{S} = \{42, 101, 202, 303, 404\}$$
- **Deterministic Derivation Rule:**
  - Environment Seed: $S_{\text{env}} = S$
  - Agent / Neural Network Seed: $S_{\text{agent}} = S$
  - Opponent Stochasticity Seed: $S_{\text{opp}} = S + 10$
  - Evaluation Tournament Seed: $S_{\text{eval}} = S + 1000$
- **Reproducibility Guarantee:** `set_seed(S)` controls Python `random`, NumPy RNG, PyTorch CPU RNG, and disables PyTorch non-deterministic convolution algorithms.

---

## 13. Number of Training Runs
- **Runs per Algorithm:** 5 independent training runs (one per master seed).
- **Total Primary Runs:** $4 \text{ models} \times 5 \text{ seeds} = 20\text{ full training runs}$.

---

## 14. Training Budget
- **Interaction Steps per Run:** $500,000$ steps.
- **Total Training Budget:**
  $$20 \text{ runs} \times 500,000 \text{ steps} = 10,000,000 \text{ interaction steps}$$
- **Intermediate Evaluation Interval:** Every $10,000$ steps ($100$ evaluation games vs benchmark opponents).
- **Checkpoint Snapshot Interval:** Every $25,000$ steps (saved to `checkpoints/`).

---

## 15. Evaluation Game Count
- **Published Research Protocol:**
  - Each trained model is evaluated over **5,000 games** against `RandomOpponent`.
  - Each trained model is evaluated over **5,000 games** against `ExpertOpponent`.
- **Total Games per Seed:** $10,000$ tournament games.
- **Total Evaluation Games per Model:** $50,000$ games across 5 seeds.
- **Grand Total Evaluation Games:** $200,000$ tournament games across all 4 models.
- **Seat Assignment:** Exactly 50% Player 0 and 50% Player 1 (seat alternating).

---

## 16. Opponent Definitions
1. **Random Opponent (`RandomOpponent`):**
   - Uniform random selection strictly from the set of legal actions: $a \sim \mathcal{U}(\mathcal{A}_{\text{legal}}(s))$.
   - Guaranteed 0% illegal action attempts.
2. **Expert Opponent (`ExpertOpponent`):**
   - Deterministic rule-based priority hierarchy matching published Alhajry (2012) strategy:
     - **Priority 1 (Release):** If roll is 6 and tokens remain in yard, release token to tile 0/26.
     - **Priority 2 (Defense):** If a token is in danger of being captured by an opponent within roll range, move to a safe square or home column.
     - **Priority 3 (Capture):** If a legal move captures an opponent token, execute capture.
     - **Priority 4 (Fast Progress):** Otherwise, move the token furthest along the track toward the home column.

---

## 17. Evaluation Metrics
For every evaluation tournament, the following metrics are tracked:
1. **Win Rate ($W\%$):** Ratio of games won by the RL agent to total completed games.
2. **Average Turns ($\bar{T}$):** Average number of turns required to complete a game.
3. **Average Episode Reward ($\bar{R}$):** Cumulative undiscounted reward earned per game.
4. **Token Capture Rate ($\bar{C}$):** Mean number of opponent tokens knocked to yard per game.
5. **Tokens Home Rate ($\bar{G}$):** Mean number of tokens successfully reaching goal per game $[0..4]$.
6. **Illegal Action Rate:** Must be strictly **0.00%** (validated continuously).

---

## 18. Robustness Experiments Protocol
Conducted in Phase 14:
1. **Stochasticity Perturbation:**
   - Evaluates performance under skewed dice distributions (e.g., loaded dice with higher probability of rolling 1 or 6).
2. **Opponent Strategy Mixing:**
   - Evaluates agents against an $\epsilon$-mixed opponent choosing Random with probability $\epsilon$ and Expert with probability $1 - \epsilon$.
3. **Seat Permutation Test:**
   - Tests hypothesis $H_0: W(P_0) = W(P_1)$ to guarantee fairness across player seats.

---

## 19. Ablation Experiments Protocol
Conducted in Phase 14:
1. **Representation Ablation:**
   - Compares Structured CNN Representation (4-channel spatial) against Flat 60-D observation baseline.
2. **Action Masking Ablation:**
   - Compares Action Masking ($-10^9$ logit suppression) against Unmasked learning with $-0.50$ illegal penalty.
3. **Reward Shaping Ablation:**
   - Compares Full 8-Event Dense Reward against Sparse Terminal Reward ($\pm 1.0$ only).

---

## 20. Statistical Analysis Methodology
1. **Aggregations:** Mean $\mu$ and standard deviation $\sigma$ across the 5 independent seeds.
2. **Confidence Intervals:** 95% Bootstrap Confidence Intervals (10,000 resamples) for all win rates and episode rewards.
3. **Hypothesis Testing:**
   - Welch's two-sample $t$-test (parametric) and Wilcoxon rank-sum test (non-parametric) for pairwise algorithm comparisons.
   - Significance threshold: $\alpha = 0.05$ (two-tailed).
4. **Effect Size:** Cohen's $d$ calculated for all pairwise performance contrasts.

---

## 21. Hardware & Software Specification
- **Operating System:** Windows 11 (64-bit)
- **Python Version:** 3.11.9
- **PyTorch Build:** `2.13.0+cpu` (CPU vectorized execution)
- **Supporting Libraries:** Gymnasium 1.0.0, NumPy 2.2.3, Matplotlib 3.11.0, PyYAML 6.0.2, Pytest 9.1.1, psutil 7.0.0
- **Hardware Architecture:** Multi-core x86_64 CPU, 16+ GB RAM, no GPU required.
- **Baseline Git Commit:** Tagged commit for Protocol v1.0.0.

---

## 22. Protocol Revision History & Amendment Log
| Version | Date | Author / Role | Section Affected | Nature of Amendment | Justification |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **1.0.0** | 2026-10-03 | Lead Researcher | All | Initial Protocol Freeze | Established frozen baseline specifications for 4-model PhD research benchmark. |

*Any subsequent amendment requires formal recording in this table prior to executing experimental runs.*
