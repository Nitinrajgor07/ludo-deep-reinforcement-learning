# Model 3: Discrete Soft Actor-Critic (SAC)

## Overview
Model 3 is formulated as **Discrete Soft Actor-Critic (SAC)** (Christodoulou, 2019; Haarnoja et al., 2018), extending maximum-entropy off-policy actor-critic reinforcement learning to discrete action spaces with strict action masking.

## Primary Invariants (Strict Empirical Control)
To ensure unbiased comparison with Model 1 (Double Dueling DQN) and Model 2 (PPO):
- **Environment:** `LudoEnvironment` (2 players, circular 52-tile track, 4 tokens per player).
- **State Representation:** Common 4-channel spatial board tensor `(4, 52)` + context vector `(12,)`.
- **Feature Encoder:** Identical `StructuredCNNEncoder` producing standard 128-D latent state representation.
- **Action Space:** Discrete token selection $A \in \{0, 1, 2, 3\}$.
- **Action Masking:** Logits masked to $-10^9$ before softmax; illegal actions excluded from exact value expectations.
- **Reward Engine:** Standardized 8-event baseline reward framework.
- **Opponent Benchmarks:** Evaluated against `RandomOpponent` and `ExpertOpponent` via `SelfPlayArena`.

## Mathematical Formulation
1. **Target Value (Exact Expectation):**
   $$V(s') = \sum_{a \in \mathcal{A}_{\text{legal}}(s')} \pi(a|s') \left[ \min_{j=1,2} Q_{\text{targ}, j}(s', a) - \alpha \log \pi(a|s') \right]$$
   $$y = r + \gamma (1 - d) V(s')$$

2. **Twin Critic Loss:**
   $$\mathcal{L}_{Q} = \frac{1}{2} \mathbb{E}\left[ (Q_1(s, a) - y)^2 + (Q_2(s, a) - y)^2 \right]$$

3. **Actor Loss:**
   $$\mathcal{L}_{\pi} = \mathbb{E}_{s}\left[ \sum_{a \in \mathcal{A}_{\text{legal}}(s)} \pi(a|s) \left( \alpha \log \pi(a|s) - \min_{j=1,2} Q_j(s, a) \right) \right]$$

4. **Temperature ($\alpha$) Tuning:**
   $$\mathcal{L}_{\alpha} = -\log \alpha \cdot \sum_{a} \pi(a|s) \left( \log \pi(a|s) + \bar{\mathcal{H}} \right)$$
   where target entropy $\bar{\mathcal{H}} = \log(4) \times 0.98 \approx 1.3586$.
