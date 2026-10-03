# Model 4: Categorical DQN (C51 / Distributional RL)

## Overview
Model 4 is implemented as **Categorical DQN (C51)** (Bellemare, Dabney, Munos, 2017), modeling the complete probability distribution of returns over 51 discrete atoms $\{z_i\}_{i=0}^{50}$ instead of only estimating the scalar expected value.

## Primary Invariants (Strict Empirical Control)
To guarantee rigorous empirical comparison against Model 1 (Double Dueling DQN), Model 2 (PPO), and Model 3 (Discrete SAC):
- **Environment:** `LudoEnvironment` (2 players, circular 52-tile track, 4 tokens per player).
- **State Representation:** Standard 4-channel spatial board tensor `(4, 52)` + context vector `(12,)`.
- **Feature Encoder:** Identical `StructuredCNNEncoder` producing standard 128-D latent state representation.
- **Action Space:** Discrete token selection $A \in \{0, 1, 2, 3\}$.
- **Action Masking:** Q-values masked to $-10^9$ for illegal actions; next action in Bellman projection chosen from legal moves.
- **Reward Engine:** Standardized 8-event baseline reward framework.
- **Opponent Benchmarks:** Evaluated against `RandomOpponent` and `ExpertOpponent` via `SelfPlayArena`.

## Mathematical Formulation
1. **Support Atoms:**
   $$z_i = V_{\min} + i \cdot \Delta z, \quad i \in \{0, \dots, N-1\}, \quad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}$$
   with $N = 51, V_{\min} = -20.0, V_{\max} = 30.0 \implies \Delta z = 1.0$.

2. **Expected Q-Values:**
   $$Q(s, a) = \sum_{i=0}^{N-1} z_i \cdot p_i(s, a)$$

3. **Bellman Categorical Projection:**
   $$T z_j = \text{clamp}(r + \gamma (1 - d) z_j, V_{\min}, V_{\max})$$
   $$b_j = \frac{T z_j - V_{\min}}{\Delta z}, \quad l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil$$
   $$m_l \leftarrow m_l + p_j^*(s', a^*) \cdot (u - b_j)$$
   $$m_u \leftarrow m_u + p_j^*(s', a^*) \cdot (b_j - l)$$

4. **Cross-Entropy Loss:**
   $$\mathcal{L} = -\sum_{i=0}^{N-1} m_i \cdot \log p_i(s, a)$$
