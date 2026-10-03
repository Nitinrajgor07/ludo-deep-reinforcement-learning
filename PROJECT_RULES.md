# PhD Research Project Rules & Scientific Constraints

## Project Identity
- **Project Title:** Optimizing Deep Reinforcement Learning for Board Games: A Comparative Study of DQN, PPO, and Advanced DRL Architectures on Ludo with Structured State Encoding
- **Level:** PhD-Level Empirical Reinforcement Learning Research
- **Context:** Continuation of published IEEE conference paper by Adithya Tej BM, Abhishek Sharma, Addwin Antony Stephen, Deekshith Gowda R, Nitin Rajgor (Department of CS & IT, JAIN University).

---

## Strict Research Rules & Behavioral Constraints

### 1. Scientific Integrity & Data Truth
- **NEVER Fabricate Results:** All reported win rates, loss curves, reward trajectories, and ablation statistics must originate directly from verified training and evaluation logs.
- **NEVER Alter Published Baselines:** The foundational baseline figures established in the published paper must remain immutable:
  - Alhajry et al. (2012) TD(λ): 66.0% vs Random, 30.0% vs Expert
  - Alhajry et al. (2012) Q-Learning: 63.0% vs Random, 27.0% vs Expert
  - DQN + Flat Encoding: 69.2% vs Random, 28.4% vs Expert
  - DQN + Structured CNN: 78.4% vs Random, 36.1% vs Expert
  - PPO + Flat Encoding: 74.8% vs Random, 33.7% vs Expert
  - PPO + Structured CNN: 83.7% vs Random, 42.3% vs Expert

### 2. Algorithmic Scope & Extension Boundaries
- The research objective is to extend the comparison to **four RL algorithms**:
  1. **DQN** (Double Dueling DQN + Prioritized Experience Replay) — *Established Baseline*
  2. **PPO** (Actor-Critic with GAE, Action Masking, Multi-Env Rollout) — *Established Baseline*
  3. **Model 3** — *Pending Supervisor Selection; DO NOT IMPLEMENT PREMATURELY*
  4. **Model 4** — *Pending Supervisor Selection; DO NOT IMPLEMENT PREMATURELY*
- **Model Selection Protocol:** Neither Model 3 nor Model 4 shall be selected, implemented, or hardcoded until explicitly approved by the researcher.

### 3. Architectural Parity & Controlled Comparison
- To guarantee fair comparative analysis across the 4 RL algorithms:
  - **Shared State Encoding:** All algorithms must consume the identical $128\text{-D}$ latent state vector derived from the authoritative encoders (or raw structured board tensors processed identically).
  - **Shared Action Masking:** Illegal actions must be masked identically across all algorithms prior to action selection/sampling.
  - **Shared Reward Formulation:** The shaped reward engine derived from domain knowledge and published literature must be held constant across all 4 algorithm benchmarks.
  - **Shared Evaluation Benchmark:** Performance must be measured over exactly **5,000 games** against the standardized Random Opponent and Expert Opponent.

### 4. Codebase Preservation & Non-Destructive Development
- Existing working research components, tests, and configurations must never be silently deleted or corrupted.
- Modular component interfaces (Environment, Encoder, Agent, Opponent, Trainer, Evaluator) must be strictly maintained.

### 5. Reproducibility & Determinism
- Every experiment must record:
  - Random seeds across Python `random`, `numpy`, and `torch`.
  - Complete configuration snapshot (hyperparameters, architecture details).
  - Environment version and git commit hash.
  - Evaluation logs with per-game win/loss outcomes, steps, and captures.
