# Research Design: Future Four-Player Multi-Agent Extension for Stochastic Board Games (Ludo)

**Document Classification:** Future Research Design & Theoretical Framework (Phase 22 Deliverable)  
**Status:** Independent Proposal & Architectural Blueprint (NOT implemented in current two-player codebase)  
**Target Domain:** Four-Player Stochastic Non-Cooperative Ludo Markov Game  
**Author:** Adithya Tej et al. (PhD Research Program)

---

## 1. Executive Summary & Research Boundary

This document formulates an advanced research design for extending the deep reinforcement learning framework from the completed two-player benchmark (`phd_ludo_drl`) to a **four-player competitive multi-agent environment**.

### Explicit Separation from Current Two-Player Study
* **Completed Study (Phases 1–21):** Evaluated four single-agent DRL architectures (Double Dueling DQN, PPO, Discrete SAC, and C51) on a strictly controlled two-player zero-sum formulation ($\text{Seat } 0 \text{ vs. } \text{Seat } 1$) where the opponent's strategy is treated as part of the transition probability $\mathcal{P}(s' \mid s, a)$ or simulated via mirrored self-play.
* **Future Four-Player Extension (Phase 22):** Formalizes a non-zero-sum / general-sum, partially competitive, non-transitive **four-player Markov Game** (Stochastic Game). This extension introduces emergent multi-agent phenomena, non-stationarity, cyclic dominance, multi-agent credit assignment, and implicit coalition dynamics that are theoretically distinct from two-player zero-sum MDPs.
* **Implementation Invariant:** **No modifications** will be made to the current two-player code, model weights, or experimental records until this research design is formally approved for subsequent experimental phases.

---

## 2. Four-Player Environment Formalism

### 2.1 Markov Game (Stochastic Game) Formulation
In a four-player setting, the environment transitions from a single-agent MDP to a partially competitive, turn-based **four-player Markov Game** defined by the tuple:
$$\mathcal{M}_4 = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, \mathcal{P}, \{\mathcal{R}^i\}_{i \in \mathcal{N}}, \gamma, \{\Omega^i\}_{i \in \mathcal{N}} \rangle$$
where:
* **Player Set $\mathcal{N} = \{0, 1, 2, 3\}$:** Four agents positioned symmetrically around the board at $90^\circ$ relative offsets (Red: Player 0, Green: Player 1, Yellow: Player 2, Blue: Player 3).
* **Joint State Space $\mathcal{S}$:**
  $$s_t = \left( \mathbf{T}^{(0)}, \mathbf{T}^{(1)}, \mathbf{T}^{(2)}, \mathbf{T}^{(3)}, d_t, p_t \right)$$
  where $\mathbf{T}^{(i)} \in \{-1, 0, \dots, 51, 100, \dots, 105\}^4$ tracks the coordinates of Player $i$'s four tokens, $d_t \in \{1, \dots, 6\}$ is the current die roll, and $p_t \in \mathcal{N}$ designates the active turn.
* **Action Space $\mathcal{A}^i = \{0, 1, 2, 3\}$:** Active player $p_t$ selects one of four tokens to advance. Inactive players execute a null action ($a^j = \emptyset$ for $j \ne p_t$).
* **Action Mask Contract $\Omega^i(s_t, d_t) \subseteq \mathcal{A}^i$:** Token movement legality enforced per player:
  1. Entry from Yard onto starting tile requires $d_t = 6$. Player starting tiles are distributed at intervals of 13 tiles:
     * Player 0 (Red): Tile index $0$
     * Player 1 (Green): Tile index $13$
     * Player 2 (Yellow): Tile index $26$
     * Player 3 (Blue): Tile index $39$
  2. Movement along the circular 52-tile track with wrap-around ($51 \to 0$).
  3. Turn-in to dedicated, non-overlapping 5-square home columns entering at $(Start_i - 1 + 52) \bmod 52$.
  4. Immunity on the 8 universal safe squares: $\{0, 8, 13, 21, 26, 34, 39, 47\}$.
* **Transition Function $\mathcal{P}(s_{t+1} \mid s_t, a_t^{p_t})$:** Stochastic transition conditioned on active agent action, game rules (capturing any single opponent token on a shared unsafe track tile), bonus rolls ($d_t = 6$ or successful capture), and subsequent uniform die sampling $d_{t+1} \sim \mathcal{U}\{1, 6\}$.
* **Individual Reward Functions $\mathcal{R}^i(s_t, a_t^{p_t})$:** General-sum payoffs reflecting final rank ordering (1st place, 2nd place, 3rd place, 4th place) and intermediate tactical events.

### 2.2 Symmetrical Egocentric State Projection
To permit parameter sharing across all four seats without spatial bias, the global board state must be mapped into an **egocentric perspective** for player $i$:
$$\phi_i(s_t) = \text{Rotate}\left(s_t, -13 \times i\right)$$
Under this transformation:
* Player $i$'s tokens are always mapped to Channel 0 (relative start at tile 0).
* Opponent 1 (left neighbor, $(i+1) \bmod 4$) is mapped to Channel 1 (start at tile 13).
* Opponent 2 (opposite player, $(i+2) \bmod 4$) is mapped to Channel 2 (start at tile 26).
* Opponent 3 (right neighbor, $(i+3) \bmod 4$) is mapped to Channel 3 (start at tile 39).
* Safe squares and relative home columns rotate accordingly.

This mathematical symmetry ensures that a single neural policy $\pi_\theta(a \mid \phi_i(s))$ can execute identically from any of the four physical board seats.

---

## 3. Multi-Agent Interaction Dynamics

### 3.1 Non-Zero-Sum Payoffs and Finish Rankings
In two-player Ludo, the game is strictly zero-sum: $R^{(0)} + R^{(1)} = 0$ at termination. In four-player Ludo, terminating upon the first player's victory creates a non-zero-sum game. If play continues to establish full rankings ($1^{\text{st}}, 2^{\text{nd}}, 3^{\text{rd}}, 4^{\text{th}}$), payoffs follow an ordinal structure:
$$\mathbf{r}_{\text{terminal}} = \begin{cases} +10.0 & \text{for } 1^{\text{st}} \text{ Place (First to home 4 tokens)} \\ +3.0 & \text{for } 2^{\text{nd}} \text{ Place} \\ -2.0 & \text{for } 3^{\text{rd}} \text{ Place} \\ -10.0 & \text{for } 4^{\text{th}} \text{ Place} \end{cases}$$
This ordinal structure alters optimal risk preferences: a player lagging in 4th place must adopt high-variance tactical gambits (hunting captures), whereas a player safely in 2nd place may prioritize defensive avoidance to secure positive terminal reward.

### 3.2 Implicit and Emergent Coalitions (The "Kingmaker" Dilemma)
In a four-player game, an agent often faces choices where action $a_1$ harms Player $A$ while action $a_2$ harms Player $B$. When Player $A$ is leading, rational uncoordinated agents will naturally direct captures toward Player $A$ to prolong the game and maximize their own win probability—an emergent self-correcting dynamic known as **leader suppression**. Conversely, late-game states introduce the *Kingmaker dilemma*, where a lagging agent with no mathematical chance of victory determines which of two leading opponents wins via a discretionary capture.

### 3.3 Spatial Collision Density and Capture Cascades
With 16 active tokens inhabiting a 52-tile track:
* Mean token density increases from $0.154 \text{ tokens/tile}$ (two-player) to **$0.308 \text{ tokens/tile}$** (four-player).
* Unsafe squares feature substantially higher collision probabilities.
* Single-token captures trigger cascading tempo shifts, where sending an opponent token back to the Yard alters the turn order and threat field for all three opposing players simultaneously.

---

## 4. Self-Play Architecture for Four Players

Single-agent self-play (e.g., fictitious play or naive parameter sharing) in multi-agent environments is vulnerable to **non-transitive cycles (rock-paper-scissors dynamics)** and **policy collapse**, where agents overfit to idiosyncrasies of their own clones.

```
+-----------------------------------------------------------------------------------+
|                        League Self-Play Training Architecture                     |
+-----------------------------------------------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
     [Main Learning Agents]                              [Historical Checkpoint Pool]
    (Latest Policy Updates)                               (Past Snapshots: t - k*dt)
                |                                                   |
                +-------------------------+-------------------------+
                                          |
                                          v
                         +---------------------------------+
                         |     Matchmaking Sampler Engine  |
                         +---------------------------------+
                                          |
      +-----------------------------------+-----------------------------------+
      | (50% Matches)                     | (30% Matches)                     | (20% Matches)
      v                                   v                                   v
[Homogeneous Self-Play]          [Heterogeneous League]            [Exploiter / Heuristic]
 Seat 0: Current Policy           Seat 0: Current Policy            Seat 0: Current Policy
 Seat 1: Current Policy           Seat 1: Historical Snapshot A     Seat 1: Rule-Based Expert
 Seat 2: Current Policy           Seat 2: Historical Snapshot B     Seat 2: Aggressive Capturer
 Seat 3: Current Policy           Seat 3: Prioritized Opponent C    Seat 3: Defensive Loiterer
```

### 4.1 League-Based Training Pool
To prevent cyclic stagnation, we propose a **Four-Player League Training Framework** adapted from AlphaStar (Vinyals et al., 2019):
1. **Main Agent:** Actively trains via gradient descent, participating in matches to improve general playing strength.
2. **Main Exploiter Agents:** Specialized policies conditioned to detect and exploit tactical vulnerabilities in the Main Agent (e.g., hyper-defensive loitering or neglect of yard-opening).
3. **League Checkpoint Pool:** An append-only repository of frozen policy snapshots preserved at regular training intervals (every $50,000$ steps).
4. **Stochastic Matchmaker:** For each four-player match, seats are sampled according to a distribution:
   * $50\%$ probability: Pure homogeneous self-play ($\pi_{\theta}, \pi_{\theta}, \pi_{\theta}, \pi_{\theta}$).
   * $30\%$ probability: Heterogeneous league play ($\pi_{\theta}$ vs. three distinct historical checkpoints).
   * $20\%$ probability: Mixed benchmark play ($\pi_{\theta}$ vs. rule-based heuristic agents and exploiters).

### 4.2 Empirical Elo and Glicko-2 Multi-Player Rating
To evaluate progress across training iterations, traditional two-player Elo rating is replaced with a **Multi-Player Glicko-2 Rating Engine** that updates skill ratings based on the full 4-player finishing order.

---

## 5. Opponent Modeling

In a game with three independent adversaries, treating opponent behavior as stationary noise leads to severe policy miscalibration. We propose two complementary opponent modeling mechanisms:

```
                                [Raw Board State (s_t)]
                                           |
                                           v
                             +---------------------------+
                             |   Spatial CNN Backbone    |
                             +---------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  Main Actor-Critic    |                     |   Opponent Modeling   |
        |      Policy Head      |                     |      Auxiliary Head   |
        +-----------------------+                     +-----------------------+
                    |                                             |
                    |                                             v
                    |                             Predicted Next Action Distribution:
                    |                             P(a^{j} | s_t) for j in {1, 2, 3}
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                             [Conditioned Value & Policy]
                             Q(s_t, a_t | \hat{p}^1, \hat{p}^2, \hat{p}^3)
```

### 5.1 Explicit Auxiliary Action Prediction
The shared spatial encoder is augmented with an auxiliary supervised classification head trained to predict the next action distribution of each visible opponent:
$$\mathcal{L}_{\text{OM}}(\psi) = -\sum_{j \ne i} \sum_{k=0}^3 \mathbb{I}(a^j = k) \log \hat{P}_\psi\left(a^j = k \mid \phi_j(s_t)\right)$$
This auxiliary objective forces the latent representation to encode opponent token affordances, vulnerability vectors, and intention metrics without manual feature engineering.

### 5.2 Style and Policy Latent Embeddings (Policy Recognition)
Following He et al. (2016), each opponent $j \in \{1, 2, 3\}$ is assigned an inferred style latent vector $\mathbf{z}^j \in \mathbb{R}^8$ generated by a recurrent encoder observing the opponent's historical action choices over the past $\tau = 20$ turns:
$$\mathbf{z}_t^j = \text{GRU}\left(\mathbf{z}_{t-1}^j, (s_{t-1}, a_{t-1}^j, d_{t-1})\right)$$
The agent's policy and value heads are conditioned on the concatenated style representations:
$$\pi(a \mid s_t, \mathbf{z}_t^1, \mathbf{z}_t^2, \mathbf{z}_t^3)$$
This enables the agent to dynamically differentiate between an overly aggressive opponent (who chases captures regardless of danger) and a passive opponent (who loiters on safe squares).

---

## 6. Multi-Agent Reinforcement Learning (MARL) Paradigms

For four-player Ludo, we formulate three candidate MARL architectures based on the **Centralized Training with Decentralized Execution (CTDE)** paradigm:

```
Centralized Training with Decentralized Execution (CTDE) in 4-Player Ludo:

[Training Phase (Centralized Critic)]
Global State S = (T^0, T^1, T^2, T^3, d_t, p_t) ---> Centralized Value Critic V_central(S; phi)
                                                      (Evaluates joint board state,
                                                       counterfactual baselines)
                                                                 |
                                                                 v
                                                     [Advantage Computation A^i]
                                                                 |
                                                                 v
[Execution Phase (Decentralized Actor)]
Egocentric State phi_i(s_t) ------------------------> Policy Actor pi^i(a^i | phi_i(s_t); theta)
Action Mask Omega^i(s_t, d_t) -----------------------> Strictly 0% illegal moves
```

### 6.1 Multi-Agent PPO with Centralized Critic (MAPPO)
* **Decentralized Actor:** Each player selects actions using strictly local egocentric observations $\phi_i(s_t)$ and local action masks: $\pi_\theta(a^i \mid \phi_i(s_t))$.
* **Centralized Critic:** During training, the value network has access to the full unmasked global board state $\mathcal{S}_t$:
  $$V_\phi(\mathcal{S}_t) \approx \mathbb{E}\left[ \sum_{k=0}^\infty \gamma^k r_{t+k}^i \mid \mathcal{S}_t \right]$$
* **Generalized Advantage Estimation:** GAE is computed using the centralized value function, drastically reducing gradient variance caused by unseen opponent trajectories.

### 6.2 Counterfactual Multi-Agent Policy Gradients (COMA)
In four-player interactions, measuring whether Player $i$'s specific action contributed to success or whether success resulted from Opponent $A$ neutralizing Opponent $B$ is challenging. COMA (Foerster et al., 2018) computes a counterfactual advantage baseline:
$$A^i(s, \mathbf{a}) = Q(s, \mathbf{a}) - \sum_{a'^i} \pi^i(a'^i \mid \phi_i(s)) Q(s, (a'^i, \mathbf{a}^{-i}))$$
By marginalizing out Player $i$'s action while holding all opponent actions $\mathbf{a}^{-i}$ fixed, the network isolates the exact marginal credit attributable to Player $i$.

### 6.3 Value Factorization (QMIX Adaptation for Competitive Games)
While classical QMIX (Rashid et al., 2018) is designed for cooperative teams, competitive four-player games can be decomposed into an **ego vs. environment (all opponents)** zero-sum surrogate:
$$Q_{\text{total}}(s, \mathbf{a}) = f_{\text{mix}}\left(Q_0, Q_1, Q_2, Q_3; s\right)$$
where the mixing network enforces monotonicity constraints on individual payoff bounds.

---

## 7. Credit Assignment Across Multi-Agent Horizons

Credit assignment in four-player Ludo is compounded by three structural properties:
1. **Turn Latency:** Between Player $0$'s turn at step $t$ and Player $0$'s subsequent turn at step $t+4$, three intermediate actions from opponents take place. A capture or positional advantage created at step $t$ may be erased before Player $0$ acts again.
2. **Long Episode Trajectories:** Four-player Ludo games regularly exceed $250$ to $350$ game turns, magnifying gradient vanishing/exploding problems in standard temporal difference updates.
3. **Credit Smearing:** If Player $1$ captures Player $2$, Player $0$ benefits indirectly without taking any action.

### Proposed Solutions for Multi-Agent Credit Assignment:
* **Turn-Skipping Multi-Step TD Targets:**
  Rather than standard $1$-step transitions $(s_t, a_t, r_{t+1}, s_{t+1})$ where $s_{t+1}$ belongs to Player 1, construct an **own-turn transition tuple**:
  $$\left( \phi_0(s_t), a_t^0, \sum_{k=1}^4 \gamma^{k-1} r_{t+k}^0, \phi_0(s_{t+4}) \right)$$
  This collapses the intermediate turns of opponents into an environmental transition operator, enabling direct Bellman updates between an agent's consecutive decision points.
* **Hindsight Credit Assignment:**
  Implement Temporal Value Decomposition (TVD) to weight intermediate token movements by whether the specific moved token ultimately reached the Home goal or was captured before scoring.

---

## 8. Competitive Dynamics & Game-Theoretic Equilibria

### 8.1 Non-Transitivity and Cyclic Dominance
In four-player games, strategy spaces frequently exhibit cyclic dominance:
$$\text{Strategy } A \succ \text{Strategy } B \succ \text{Strategy } C \succ \text{Strategy } A$$
For instance:
* *Aggressive Hunting* defeats *Fast Single-Token Runner* (by repeatedly capturing isolated forward tokens).
* *Safe-Square Loitering* defeats *Aggressive Hunting* (by denying capture opportunities and forcing attackers into overextension).
* *Fast Single-Token Runner* defeats *Safe-Square Loitering* (by racing to the home goal while loiterers waste tempo).

Standard gradient ascent without population diversity will cycle indefinitely through these meta-strategies without converging.

### 8.2 Empirical Game-Theoretic Analysis (EGTA) & $\alpha$-Rank
To quantify equilibrium convergence in the future four-player benchmark, we specify:
1. **Payoff Matrix Construction:** An empirical meta-game payoff tensor $\mathbf{M} \in \mathbb{R}^{K \times K \times K \times K}$ where entry $M(i, j, k, l)$ records the mean tournament payoff when Player 0 plays policy $i$, Player 1 plays policy $j$, Player 2 plays policy $k$, and Player 3 plays policy $l$ across $N = 100$ games.
2. **$\alpha$-Rank Computation:** Rather than computing standard Nash equilibria (which are computationally intractable and often non-unique in multi-player general-sum games), we will apply **$\alpha$-Rank** (Omidshafiei et al., 2019), a Markov chain evaluation over the response graph that formally characterizes evolutionary stability and non-transitive loops.

---

## 9. Phased Research Roadmap (Phase 22 Extension)

When formally scheduled and approved, the four-player extension should proceed through four controlled stages:

```
Stage 1: Environment & Engine Formalization
- Implement four-player board geometry, 4 starting yards, 4 private home columns.
- Implement egocentric state projection (phi_i) and invariant 4-player action masking.
- Verify 100% test coverage and invariant 0.00% illegal move rate.

Stage 2: Opponent & Heuristic Baseline Suite
- Implement 4-player Random Opponent, Greedy Capturer, Defensive Loiterer, and Multi-Criteria Expert.
- Establish baseline tournament matrix across rule-based heuristic agents.

Stage 3: Multi-Agent Training Pipeline (MAPPO / CTDE)
- Implement Centralized Critic architecture with shared egocentric actor.
- Implement 4-player League Self-Play matchmaker and historical checkpoint pool.
- Execute multi-seed training runs across master seeds S = {42, 101, 202, 303, 404}.

Stage 4: Evaluation, Statistical Analysis & Meta-Game Game Theory
- Run 1000-game seat-balanced tournament evaluations.
- Compute multi-player Glicko-2 ratings and alpha-Rank evolutionary stability.
- Publish comparative monograph on 2-Player vs. 4-Player stochastic DRL dynamics.
```

---

## 10. Summary & Methodological Integrity

1. **Isolation Guarantee:** This document serves strictly as a prospective design blueprint. No changes have been made to existing production files in `phd_ludo_drl/`, `results/`, or the validated two-player research pipeline.
2. **Scientific Merit:** This proposal extends the core innovations established in the two-player study—specifically circular convolutional topologies (`StructuredCNNEncoder`) and invariant action masking—into the richer, more complex domain of multi-agent stochastic game theory.

---

STOP.
