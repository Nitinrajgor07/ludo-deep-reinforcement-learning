# PhD Research Project Design Document

## 1. Problem Formulation & Markov Decision Process (MDP)

Ludo is formulated as a two-player stochastic game modeled as an MDP tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

1. **State Space $\mathcal{S}$:**
   - Full board token positions: 4 tokens for Player 0, 4 tokens for Player 1.
   - Token coordinate spaces:
     - `YARD`: Index $-1$
     - `TRACK`: Indices $0 \dots 51$
     - `HOME_COLUMN`: Indices $100 \dots 104$ (or local offset $0 \dots 4$)
     - `FINISHED`: Index $999$ (or completed flag)
   - Current turn player: $p \in \{0, 1\}$.
   - Current dice roll: $d \in \{1, 2, 3, 4, 5, 6\}$.
   - Consecutive six count: $c \in \{0, 1, 2\}$.

2. **Action Space $\mathcal{A}$:**
   - Discrete 4-action space: $\mathcal{A} = \{0, 1, 2, 3\}$, where choosing action $i$ commands token $i$ of the active player to execute the valid move for dice roll $d$.
   - **Action Masking $M(s) \in \{0, 1\}^4$:**
     - $M(s)_i = 1$ if token $i$ can legally move under dice $d$, and $0$ otherwise.
     - When $\sum_i M(s)_i = 0$ (no legal move possible), the turn passes automatically without agent penalty.

3. **Transition Dynamics $\mathcal{P}(s' | s, a)$:**
   - Deterministic token movement given $(s, a, d)$, followed by stochastic environment transition sampling next dice roll $d' \sim \text{DiscreteUniform}(1, 6)$.
   - Bonus roll granted if $d = 6$ or upon successful opponent capture.

4. **Reward Function $\mathcal{R}(s, a, s')$:**
   - Shaped reward formulation:
     $$r_t = 0.25 \cdot \mathbf{1}_{\text{release}} + 0.20 \cdot \mathbf{1}_{\text{defend}} + 0.15 \cdot \mathbf{1}_{\text{capture}} + 0.10 \cdot \mathbf{1}_{\text{advance}} + 0.05 \cdot \mathbf{1}_{\text{blockade}} + 1.00 \cdot \mathbf{1}_{\text{win}} - 0.25 \cdot \mathbf{1}_{\text{knocked}} - 1.00 \cdot \mathbf{1}_{\text{loss}}$$

5. **Discount Factor:** $\gamma = 0.99$.

---

## 2. State Encoding Architecture

All RL models interface with a standardized latent representation $\mathbf{z} \in \mathbb{R}^{128}$.

### Structured 4-Channel Spatial CNN Encoder (Primary)
- **Input Tensor:** Shape $(4, 15, 15)$ or $(4, 52)$ representing spatial token planes:
  - Channel 0: Active player token occupancy.
  - Channel 1: Opponent token occupancy.
  - Channel 2: Safe square indicators and blockade locations.
  - Channel 3: Home column and goal progression status.
- **Layers:**
  1. `Conv2D(in_channels=4, out_channels=32, kernel_size=3, padding=1)` + ReLU
  2. `Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)` + ReLU
  3. Flatten + `Linear(in_features=..., out_features=128)` + ReLU
  4. Concatenate with 6-D one-hot dice vector $\to$ Linear projection $\to 128$-D latent state.

### Flat Encoder (Baseline)
- Input: Concatenation of token positions, status flags, dice, and player indicators.
- 2 Fully Connected layers: `Linear(input_dim, 256)` + ReLU $\to$ `Linear(256, 128)` + ReLU.

---

## 3. RL Algorithm Architectures

### Model 1: Double Dueling DQN + PER
- **Shared Latent:** $\mathbf{z} \in \mathbb{R}^{128}$
- **Value Stream:** $\text{Linear}(128, 256) \to \text{ReLU} \to \text{Linear}(256, 1) \to V(s)$
- **Advantage Stream:** $\text{Linear}(128, 256) \to \text{ReLU} \to \text{Linear}(256, 4) \to A(s, a)$
- **Dueling Combination:** $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{4}\sum_{a'} A(s, a')\right)$
- **Masked Q-selection:** $Q_{\text{masked}}(s, a) = Q(s, a) + (1 - M(s)_a) \cdot (-1\times 10^9)$
- **Replay Buffer:** Capacity 50,000, $\alpha=0.6, \beta=0.4 \to 1.0$.

### Model 2: PPO Actor-Critic + GAE
- **Shared Latent:** $\mathbf{z} \in \mathbb{R}^{128}$
- **Policy Head:** $\text{Linear}(128, 128) \to \text{ReLU} \to \text{Linear}(128, 4) \to \mathbf{l}(s)$
  - Masked logits: $\hat{l}_i(s) = l_i(s) \text{ if } M(s)_i = 1 \text{ else } -\infty$
  - Policy distribution: $\pi(a | s) = \text{Categorical}(\text{softmax}(\hat{\mathbf{l}}(s)))$
- **Value Head:** $\text{Linear}(128, 128) \to \text{ReLU} \to \text{Linear}(128, 1) \to V(s)$
- **GAE:** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$.

### Models 3 & 4
- Placeholders in the agent registry. Will consume the exact same $\mathbf{z} \in \mathbb{R}^{128}$ latent and action mask $M(s)$.
