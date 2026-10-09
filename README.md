# Deep Reinforcement Learning for Stochastic Turn-Based Board Games
## Four-Model Empirical Benchmark on Ludo MDP with Structured State Encoding

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium 1.0](https://img.shields.io/badge/Gymnasium-1.0-008080.svg)](https://gymnasium.farama.org/)
[![Protocol Frozen](https://img.shields.io/badge/Protocol-Frozen%20v1.0.0-059669.svg)](FINAL_EXPERIMENT_PROTOCOL.md)
[![Tests Passing](https://img.shields.io/badge/Tests-268%20Passed-22c55e.svg)](tests/)
[![Theme Light Glass](https://img.shields.io/badge/Theme-100%25%20Light%20Glass-2563eb.svg)](web/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Project Title & Overview

**Title:** Deep Reinforcement Learning for Stochastic Turn-Based Board Games: A Four-Model Comparative Benchmark on Ludo MDP  
**Institution:** Department of Computer Science & Information Technology, JAIN (Deemed-to-be University), Bangalore, India  
**Lead Researcher:** Adithya Tej et al.  
**System Architecture:** Modular Python package (`phd_ludo_drl`), standalone execution pipelines (`scripts/`), test suite (`tests/`), and an interactive Light Glassmorphism web demonstration dashboard (`web/`).

---

## 2. PhD Project Objective

The primary objective of this doctoral research is to formulate, implement, and empirically evaluate a controlled comparative benchmark across four distinct Deep Reinforcement Learning (DRL) algorithmic paradigms operating within a turn-based, multi-token, stochastic board game environment:

1. **Paradigm Comparison:** Compare Value-based (Double Dueling DQN + PER), On-policy Policy Gradient (PPO), Off-policy Maximum Entropy Actor-Critic (Discrete SAC), and Distributional RL (Categorical C51) under identical environmental conditions and state interfaces.
2. **Inductive Bias & State Representation:** Quantify the performance and sample-efficiency benefits of a 4-channel spatial circular CNN representation versus unstructured flat vector representations.
3. **Action Space Safety & Masking:** Implement and validate strict deterministic action masking ($M(s) \in \{0, 1\}^4$) ensuring 0.00% illegal actions throughout all training and evaluation episodes.
4. **Environmental Stochasticity & Robustness:** Measure policy degradation and retention under controlled perturbation of transition dynamics across five discrete dice variance levels ($1.0\times$ to $1.4\times$).
5. **Statistical Rigor & Reproducibility:** Ensure all findings are accompanied by exact parametric and non-parametric hypothesis tests, confidence intervals, effect sizes, and deterministic seed isolation.

---

## 3. Problem Statement

While deep reinforcement learning has achieved superhuman milestones in deterministic, two-player perfect-information games (e.g., Chess, Go), turn-based stochastic multi-token games such as **Ludo** present a distinct set of open challenges:

- **High Transition Variance:** Movement distance is governed by stochastic dice rolls ($\mathcal{U}\{1, 6\}$), requiring agents to balance aggressive advancement with tactical risk mitigation under probabilistic outcomes.
- **Circular Board Geometry:** The continuous 52-tile track has wrap-around boundary conditions ($51 \to 0$) where standard spatial convolutions introduce artificial boundary discontinuities unless circular topologies are explicitly modeled.
- **Multi-Token Decision Space:** Agents must coordinate four independent tokens across multiple game states (Yard, Outer Track, Universal Safe Squares, Blockades, Private Home Columns, and Goal).
- **Hard Move Constraints:** Depending on dice rolls and token locations, certain actions are strictly illegal. Learning without masking leads to high sample inefficiency and policy collapse.
- **Seat & First-Mover Bias:** Asymmetric starting conditions can confound algorithm comparisons unless rigorous seat alternation and balanced tournament protocols are enforced.

---

## 4. Four Deep Reinforcement Learning Models

All four models share a standardized architecture: a common spatial CNN encoder producing a 128-dimensional latent vector, identical reward feedback, and strict action masking ($-10^9$ logit/Q-value suppression) ensuring 0% illegal actions.

| Model | Paradigm | Architectural Specifications | Exploration & Objective |
| :--- | :--- | :--- | :--- |
| **Model 1: DQN** | Value-Based Deep Q-Learning | Double Q-learning target decoupling; Dueling architecture (Value stream: Linear 128 $\to$ 256 $\to$ 1, Advantage stream: Linear 128 $\to$ 256 $\to$ 4); Prioritized Experience Replay (PER: $\alpha=0.6, \beta=0.4 \to 1.0$); Soft target updates ($\tau=0.005$). | Linear $\epsilon$-greedy annealing ($1.0 \to 0.05$ over 100k steps). Loss: Huber / Smooth L1 loss with IS weights. |
| **Model 2: PPO** | On-Policy Actor-Critic | Shared CNN encoder (128-D latent); Masked Policy head (Linear 128 $\to$ 4); Value head (Linear 128 $\to$ 1); Generalized Advantage Estimation (GAE: $\lambda=0.95, \gamma=0.99$). | Clipped surrogate objective ($\epsilon=0.2$); Entropy bonus coefficient ($c_2=0.01$); Value loss clipping; Gradient norm clipping (0.5). |
| **Model 3: Discrete SAC** | Maximum Entropy Off-Policy Actor-Critic | Twin Soft Q-critics ($Q_1(s), Q_2(s) \in \mathbb{R}^4$); Policy head producing discrete distribution $\pi(a \mid s)$; Exact discrete expectation target calculation without action reparameterization; Polyak soft updates ($\tau=0.005$). | Maximum entropy objective: $\mathcal{H}(\pi) = -\sum_a \pi(a|s) \ln \pi(a|s)$; Automatic temperature tuning ($\alpha$) targeting $\bar{\mathcal{H}} = 0.98 \times \ln(4)$. |
| **Model 4: Categorical C51** | Distributional Reinforcement Learning | Models probability distribution over 51 return atoms ($V_{min}=-20.0, V_{max}=30.0, \Delta z=1.0$); Linear 128 $\to$ 256 $\to$ $4 \times 51$; Categorical Bellman projection with target networks ($\tau=0.005$). | Cross-entropy / KL divergence loss against projected return distribution; Action selection via expected value $\mathbb{E}[Z(s, a)] = \sum_i z_i p_i(s, a)$. |

---

## 5. Ludo Environment Specification

The environment is an authoritative Gymnasium-compliant implementation of the two-player Ludo Markov Decision Process (`phd_ludo_drl.environment.ludo_env.LudoEnvironment`):

- **Board Geometry:**
  - 52-tile circular outer track ($0 \dots 51$) with wrap-around topology.
  - 8 universal safe squares ($\{0, 8, 13, 21, 26, 34, 39, 47\}$) providing capture immunity.
  - 5 private home column tiles per player leading to the Goal.
  - Starting tiles: Player 0 enters at tile $0$; Player 1 enters at tile $26$ ($180^\circ$ offset).
- **Token Lifecycle:** Each player commands 4 tokens transitioning across `YARD` ($-1$) $\to$ `TRACK` ($0..51$) $\to$ `HOME_COLUMN` ($100..104$) $\to$ `GOAL` ($999$).
- **Game Mechanics:**
  - Standard dice roll sampling: $d \sim \mathcal{U}\{1, 6\}$.
  - Entry requires rolling a $6$.
  - Rolling a $6$ or capturing an opponent token grants an immediate bonus roll.
  - Exact roll required to enter the Goal.
  - Blockade barrier formed when 2 or more friendly tokens occupy the same tile (impassable to opponent).
  - Capturing sends an opponent token back to Yard unless positioned on a safe square.
- **Action Masking:** Binary action mask $M(s) \in \{0, 1\}^4$ generated at every decision step. If $\sum_i M(s)_i = 0$ (no legal moves), the environment automatically advances the turn without penalty.
- **Reward Engine:** Symmetrical shaped reward formulation:
  $$r = 0.25 \cdot \mathbf{1}_{\text{release}} + 0.20 \cdot \mathbf{1}_{\text{defend}} + 0.15 \cdot \mathbf{1}_{\text{capture}} + 0.10 \cdot \mathbf{1}_{\text{advance}} + 0.05 \cdot \mathbf{1}_{\text{blockade}} + 1.00 \cdot \mathbf{1}_{\text{win}} - 0.25 \cdot \mathbf{1}_{\text{knocked}} - 1.00 \cdot \mathbf{1}_{\text{loss}}$$

---

## 6. State Representation

The state representation strictly conforms to `FINAL_EXPERIMENT_PROTOCOL.md` Section 3 and is frozen against accidental modification:

```text
State Input:
├── Spatial Tensor (4, 52)
│   ├── Channel 0: Active player token occupancy
│   ├── Channel 1: Opponent player token occupancy
│   ├── Channel 2: Safe square indicators (1.0 at safe squares, 0.0 elsewhere)
│   └── Channel 3: Blockade indicators (1.0 at tiles with >= 2 friendly tokens)
├── Context Vector (12,)
│   ├── Active yard count [0..4]
│   ├── Opponent yard count [0..4]
│   ├── Active home count [0..4]
│   ├── Opponent home count [0..4]
│   ├── Active goal count [0..4]
│   ├── Opponent goal count [0..4]
│   ├── Normalized dice roll [d / 6.0]
│   ├── Active player one-hot (Player 0) [0.0 / 1.0]
│   ├── Active player one-hot (Player 1) [0.0 / 1.0]
│   ├── Turn progress fraction [t / 200.0]
│   ├── Bonus roll active flag [0.0 / 1.0]
│   └── Capture bonus flag [0.0 / 1.0]
└── Action Mask (4,)
    └── Binary legality indicators for tokens 0, 1, 2, 3
```

### Shared CNN Encoder (`StructuredCNNEncoder`)
- **Input:** $(B, 4, 52)$
- **Conv1D Layer 1:** $4 \to 32$ filters, kernel size $3$, padding $1$, circular padding mode $\to \text{ReLU}$
- **Conv1D Layer 2:** $32 \to 64$ filters, kernel size $3$, padding $1$, circular padding mode $\to \text{ReLU}$
- **Pooling & Flatten:** `AdaptiveAvgPool1d(1)` $\to$ Flatten to $64$ features
- **Fusion:** Concatenate with $12$-D context vector ($64 + 12 = 76$)
- **Latent Projection:** Linear $76 \to 128 \to \text{ReLU} \to (B, 128)$ latent state embedding.

---

## 7. Training Protocol

Governed by [`FINAL_EXPERIMENT_PROTOCOL.md`](FINAL_EXPERIMENT_PROTOCOL.md) v1.0.0:

- **Target Step Budget:** 500,000 environment interaction steps per training run ($2,000,000$ steps across all 4 architectures).
- **Master Seed Sequence:** $\mathcal{S} = \{42, 101, 202, 303, 404\}$ (5-fold independent replication).
- **Execution Target:** Deterministic execution on CPU (`torch.device('cpu')`), eliminating non-deterministic CUDA kernel variance.
- **Checkpoint Schedule:** Periodic snapshot checkpoints saved every 25,000 steps plus final model weights (`final_model.pt`).
- **Telemetry Logging:** Streaming JSONL and console logs recording step, episodic return, loss, epsilon/temperature, legal action rate, game length, and capture frequency.

---

## 8. Evaluation Protocol

- **Tournament Geometry:** $200,000$ total evaluation games:
  - 4 DRL Models $\times$ 5 Master Seeds $\times$ 2 Benchmark Opponents $\times$ 5,000 Games each.
- **Benchmark Opponents:**
  - **Random Opponent:** Selects uniformly at random among available legal moves.
  - **Heuristic Expert Opponent:** Alhajry rule-based priority engine (`Prefer Release` $\to$ `Defend` $\to$ `Capture` $\to$ `Fast Progress`).
- **Seat Balance:** Strict 50% Player 0 (Red) and 50% Player 1 (Green) alternating allocation to eliminate first-mover advantage.
- **Deterministic Policy:** Greedy action selection ($\arg\max_a Q(s, a)$ or $\arg\max_a \pi(a \mid s)$) with illegal action suppression.
- **Evaluation Seeds:** Seed offset derivation $S_{\text{eval}} = S + 1000$ ($\mathcal{S}_{\text{eval}} = \{1042, 1101, 1202, 1303, 1404\}$) ensuring zero overlap with training data.
- **Pre-Flight Integrity Check:** Evaluation runners enforce strict checkpoint validation, blocking pilot/smoke weights from masquerading as final 500k checkpoints.

---

## 9. Statistical Analysis & Robustness Framework

The analysis pipeline (`phd_ludo_drl.analysis.statistical_pipeline`) processes raw game records directly:

- **Descriptive Statistics:** Sample mean, median, standard deviation, interquartile ranges, win rates.
- **Confidence Intervals:** 95% Wilson Score intervals for binomial proportions; 10,000-resample non-parametric Bootstrap intervals for continuous returns and episode lengths.
- **Hypothesis Testing:**
  - Welch's two-sample $t$-test (continuous reward distributions with unequal variances).
  - Mann-Whitney $U$ test (non-parametric stochastic dominance).
  - McNemar's paired test (matched-seed game outcome discordance).
  - Two-proportion $z$-test (head-to-head win rate comparison).
  - Holm-Bonferroni step-down correction for family-wise multiple comparisons.
- **Effect Sizes:** Cohen's $d$, Hedge's $g$, and rank-biserial correlation $r$.
- **Robustness Under Stochasticity:** 1,000 evaluation games across 5 loaded-dice variance levels ($1.0\times, 1.1\times, 1.2\times, 1.3\times, 1.4\times$) measuring retention ratios.
- **Architectural Ablations:** Action masking ablation (100% policy collapse without masking), State representation ablation (CNN vs Flat 60-D), and Reward shaping ablation (Dense vs Sparse).

---

## 10. Executive Web Demonstration Dashboard

The project features a high-performance web demonstration platform built on **FastAPI**, **Uvicorn**, and modern **HTML/CSS/JavaScript** (`web/app.py`).

### Design System: 100% Light Theme Glassmorphism
- **Background:** Crisp pure white (`#ffffff`) and soft light gray/blue (`#f8fafc`, `#eff6ff`).
- **Cards & Surfaces:** Translucent white glass cards (`rgba(255, 255, 255, 0.88)`) with subtle $1\text{px}$ borders (`rgba(226, 232, 240, 0.85)`), backdrop blur ($16\text{px}$), and soft multi-layer box shadows.
- **Typography:** Dark navy/charcoal (`#0f172a`, `#334155`) for high-contrast readability.
- **Accents:** Professional blue (`#2563eb`), indigo (`#4f46e5`), and teal (`#0d9488`).
- **Strict Invariant:** Zero dark mode elements, zero dark cards, zero dark navbar or footer.

### Integrated Pages
1. **📊 Dashboard:** High-level metrics, empirical status, system telemetry, and model summaries.
2. **🧠 Models:** Deep RL architecture details, neural networks, action masking, and checkpoints.
3. **⚡ Training:** Training protocol specifications, empirical pilot curves, and checkpoint telemetry.
4. **🎯 Evaluation:** Controlled tournament benchmarks vs Random and Expert baselines.
5. **📈 Statistics:** Descriptive & inferential statistical tests (Welch's t, Mann-Whitney U, Cohen's d).
6. **🖼️ Visualizations:** Publication-grade figure gallery with full-resolution inspection.
7. **🎲 4-Player Ludo Arena:** Interactive 15×15 SVG board supporting standard 4-player matches (Red P0, Green P1, Yellow P2, Blue P3) with live agent stepping, manual actions, multi-opponent captures, 9-column live telemetry, and 4-player statistics.
8. **ℹ️ About Project:** Dissertation research motivation, formal formulation, and technical stack.

> **Note on Research Mode vs Interactive Arena:** The official frozen scientific benchmark evaluation protocol uses the controlled 2-player environment (`num_players=2`). The Interactive Ludo Arena independently supports standard 4 active players (`num_players=4`) for visualization, interactive validation, and demonstration without mutating the frozen benchmark.

---

## 11. Installation & Environment Setup

### Prerequisites
- **Python:** 3.10 – 3.12 (Python 3.11 recommended)
- **RAM:** 8 GB minimum
- **Storage:** ~1 GB disk space

### Setup Instructions
```bash
# 1. Clone repository
git clone <repository-url>
cd PROJECT

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or install editable package with all optional modules:
pip install -e ".[all]"
```

---

## 12. Running the Project

### Train Models
```bash
# Train DQN on Seed 42 for 1,000 steps
python train.py --algorithm dqn --seed 42 --steps 1000

# Train PPO on Seed 101
python train.py --algorithm ppo --seed 101 --steps 1000

# Train Discrete SAC on Seed 202
python train.py --algorithm model3 --seed 202 --steps 1000

# Train Categorical C51 on Seed 303
python train.py --algorithm model4 --seed 303 --steps 1000

# Run multi-seed production pipeline (smoke mode)
python scripts/run_production_pipeline.py --mode smoke
```

### Run Tournament Evaluation
```bash
python scripts/run_phase10_final_evaluation.py
```

### Run Statistical Analysis
```bash
python scripts/run_phase11_statistical_analysis.py
python scripts/run_phase12_robustness_ablation.py
```

### Generate Project Visualizations
```bash
python scripts/run_phase13_visualization.py
```

### Launch Interactive Web Application (FastAPI + Uvicorn)
```bash
# Standard local command:
python -m uvicorn web.app:app --reload --port 8000

# Or use the convenience runner:
python scripts/run_web.py

# Or on Windows, double-click:
run_web.bat
```
Open **`http://127.0.0.1:8000`** or **`http://localhost:8000`** in any web browser.  
Strictly 100% Light Theme AI research dashboard with real-time interactive Ludo Arena.

---

## 13. Testing

Run the exhaustive automated test suite (268 unit and integration tests across 24 test modules):

```bash
pytest tests/ -v
```

### Test Coverage Overview
- `tests/test_ludo_4p.py`: 7 tests (4-player board setup, sequential turns P0-P3, multi-opponent captures, stats, SVG, action validation)
- `tests/test_environment.py`: 26 tests (board geometry, movement, dice, safe squares, blockades, win logic)
- `tests/test_action_mask.py`: 12 tests (mask generation, Q-masking, logit suppression, zero legal moves)
- `tests/test_reward_engine.py`: 11 tests (all 8 reward events, symmetric rewards, sparse mode)
- `tests/test_cnn_encoder.py`: 25 tests (circular padding Conv1D, context fusion, gradient flow)
- `tests/test_state_representation.py`: 30 tests ((4, 52) spatial tensor, 12-D context, deviation safety)
- `tests/test_dqn.py`: 24 tests (Double Q, Dueling streams, PER buffer, Polyak updates, checkpointing)
- `tests/test_ppo.py`: 14 tests (ActorCritic network, GAE calculation, clipped loss, entropy bonus)
- `tests/test_model3.py`: 14 tests (Twin critics, discrete expectation, temperature tuning, soft target)
- `tests/test_model4.py`: 14 tests (51-atom distribution, categorical Bellman projection, KL loss)
- `tests/test_opponents.py`: 10 tests (Random and Expert rule-based opponent behavior)
- `tests/test_final_evaluation_pipeline.py`: 8 tests (tournament geometry, seat alternation, pre-flight check)
- `tests/test_statistical_pipeline.py`: 11 tests (Wilson CIs, t-tests, Mann-Whitney U, McNemar, tables)
- `tests/test_robustness_ablation.py`: 7 tests (dice variance perturbation, representation/reward ablations)
- `tests/test_final_research_visualizer.py`: 12 tests (all 10 figures generation and manifest verification)
- `tests/test_web_ui.py`: 11 tests (light glass theme invariants, all 10 pages headless AppTest render)
- `tests/test_production_pipeline.py`: 11 tests (multi-seed orchestration, config snapshots, smoke runs)
- `tests/test_unified_training.py`: 5 tests (training runner, agent factory, CLI argument parser)
- `tests/test_imports.py`: 4 tests (dependency imports, namespace contracts)
- `tests/test_config.py`: 4 tests (YAML parser, dataclasses, seed propagation)
- `tests/test_seed.py`: 3 tests (PRNG seeding across random, numpy, torch)
- `tests/test_logging.py`: 2 tests (structured JSONL and console logging)
- `tests/test_experiment_id.py`: 1 test (deterministic experiment ID generator)

---

## 14. Project Structure

```text
PROJECT/
├── phd_ludo_drl/               # Core Python Research Package
│   ├── agents/                 # Four Reinforcement Learning Agents
│   │   ├── base_agent.py       # Abstract BaseAgent interface
│   │   ├── dqn/                # Double Dueling DQN + PER
│   │   ├── ppo/                # Actor-Critic PPO with GAE
│   │   ├── model3/             # Discrete Soft Actor-Critic (SAC)
│   │   └── model4/             # Categorical DQN (C51 Distributional RL)
│   ├── environment/            # Two-Player Ludo Markov Decision Process
│   │   ├── ludo_env.py         # Gymnasium-compliant environment
│   │   ├── action_mask.py      # Binary action masking engine
│   │   ├── reward_engine.py    # Symmetrical 8-event reward calculator
│   │   ├── constants.py        # 52 track squares, safe coordinates, token states
│   │   └── token.py            # Token lifecycle entity
│   ├── encoding/               # State Representation Pipelines
│   │   ├── cnn_encoder.py      # Shared Conv1D circular spatial encoder (128-D)
│   │   ├── structured_representation.py # (4, 52) spatial tensor + 12-D context builder
│   │   └── validation.py       # Formal protocol shape and invariant validator
│   ├── opponents/              # Benchmarking Opponents
│   │   ├── random_opponent.py  # Uniform random legal move baseline
│   │   └── expert_opponent.py  # Alhajry heuristic rule-based expert
│   ├── training/               # Unified Training & Orchestration
│   │   ├── factory.py          # Agent and opponent factory
│   │   ├── unified_trainer.py  # Unified training runner
│   │   └── production_pipeline.py # Multi-seed 500k production pipeline
│   ├── evaluation/             # Tournament Evaluation Framework
│   │   └── final_evaluation_pipeline.py # 200k tournament runner with seat alternation
│   ├── analysis/               # Statistical Testing & Ablations
│   │   ├── statistical_pipeline.py # Parametric & non-parametric tests, Wilson CIs
│   │   └── robustness_ablation_framework.py # Dice variance & architectural ablations
│   ├── visualization/          # High-Resolution Vector Visualizer
│   │   └── final_research_visualizer.py # 300 DPI publication-quality figures
│   ├── config/                 # Configuration & Seeding
│   │   ├── config_manager.py   # YAML configuration manager
│   │   └── seed.py             # Deterministic PRNG seed registry
│   └── logs/                   # Structured Experiment Logging
│       └── logger.py           # Streaming JSONL and console logger
├── web/                        # Executive Web Demonstration Platform (FastAPI)
│   ├── app.py                  # FastAPI ASGI application entry point
│   ├── routes/                 # Endpoint routers (pages.py, api.py)
│   ├── services/               # Data & game services (data_service.py, game_service.py)
│   ├── templates/              # Light Glassmorphic Jinja2 HTML templates
│   ├── static/                 # Static assets (css/theme.css, js/main.js, js/ludo.js)
│   ├── components/             # Reusable board renderer & controllers
│   └── utils/                  # Web helpers & data loaders
├── scripts/                    # Standalone Command-Line Runners
│   ├── run_web.py                      # FastAPI localhost server runner
│   ├── run_production_pipeline.py      # Multi-seed production runner
│   ├── run_phase10_final_evaluation.py # 200,000-game tournament evaluation runner
│   ├── run_phase11_statistical_analysis.py # Statistical hypothesis testing runner
│   ├── run_phase12_robustness_ablation.py  # Robustness & ablation runner
│   ├── run_phase13_visualization.py    # Figure generation runner
│   └── inventory_experiments.py        # Experiment disk inventory scanner
├── tests/                      # Automated Regression Test Suite (268 Tests)
├── configs/                    # Baseline Configuration Presets
│   ├── default_config.yaml     # Canonical hyperparameter configuration
│   └── model_capabilities.yaml # Algorithmic capabilities matrix
├── results/                    # Machine-Generated Experimental Data
│   ├── raw/                    # Raw tournament game records (.csv, .jsonl)
│   ├── processed/              # Statistical metric summaries and records
│   ├── tables/                 # Publication tables (LaTeX, CSV, Markdown)
│   ├── ablations/              # Mask, representation, and reward ablation logs
│   ├── robustness/             # 5-level dice variance perturbation logs
│   └── figures/                # 10 high-resolution 300 DPI figures
├── docs/                       # Research Project Technical Specifications
│   ├── README.md               # Master documentation index
│   ├── FROZEN_STATE_REPRESENTATION.md # Formal representation specification
│   ├── PHASE10_FINAL_EVALUATION_REPORT.md
│   ├── PHASE11_STATISTICAL_ANALYSIS_REPORT.md
│   ├── PHASE12_ROBUSTNESS_AND_ABLATION_REPORT.md
│   ├── PHASE13_RESEARCH_VISUALIZATION_REPORT.md
│   ├── PHASE14_LIGHT_GLASS_WEBSITE_REPORT.md
│   ├── FINAL_PROJECT_HEALTH_REPORT.md
│   └── RUN_PROJECT.md          # Comprehensive replication guide
├── train.py                    # Root CLI training entry point
├── run_web.bat                 # One-click web demonstration server launcher
├── pyproject.toml              # PEP 621 Python packaging configuration
├── requirements.txt            # Pinned dependency requirements
├── LICENSE                     # MIT open-source license
├── VERSION                     # Release version identifier
├── PROJECT_ARCHITECTURE.md     # Deep architecture and neural dataflow specification
├── FINAL_EXPERIMENT_PROTOCOL.md# Immutable research protocol v1.0.0
├── REPRODUCIBILITY.md          # 13-dimension reproducibility specification
├── FINAL_VALIDATION_REPORT.md  # Phase 15 release audit report
└── FINAL_RELEASE_GUIDE.md      # Operational release guide
```

---

## 15. Authorship & Institutional Governance

- **Principal Researcher:** Adithya Tej et al.
- **Affiliation:** Department of Computer Science & Information Technology, JAIN (Deemed-to-be University), Bangalore, India
- **Research Standards:** Governed by `FINAL_EXPERIMENT_PROTOCOL.md` v1.0.0
- **License:** MIT License
