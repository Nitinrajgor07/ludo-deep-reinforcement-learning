# Independent Reproducibility Guide & Replication Package
## Comparative Study of Deep Reinforcement Learning for Board Games on Ludo

**Project Title:** Optimizing Deep Reinforcement Learning for Board Games: A Comparative Study of DQN, PPO, and Advanced DRL Architectures on Ludo with Structured State Encoding  
**Research Specification:** `FINAL_EXPERIMENT_PROTOCOL.md` Version 1.0.0 (FROZEN)  
**Package Version:** v1.0.0  
**Date:** October 4, 2026  
**License:** MIT / Academic Replication License  

---

## 1. System Requirements & Environment Specifications

### 1.1 Python Version
- **Target Python Version:** `Python 3.11.9`
- **Supported Python Range:** `Python >= 3.10, <= 3.12`

### 1.2 Operating System & Architecture
- **Verified Systems:**
  - Microsoft Windows 10/11 (64-bit)
  - Ubuntu Linux 22.04 LTS / 24.04 LTS (`x86_64`)
  - macOS 12+ (Apple Silicon or Intel `x86_64`)
- **GPU Requirements:** None. The environment and neural networks are optimized for vectorized CPU tensor execution via PyTorch CPU backend.

### 1.3 Hardware Requirements
- **Processor:** Modern multi-core x86_64 CPU (4+ physical cores recommended).
- **RAM:** 8 GB minimum; 16 GB recommended. Peak memory during training is $\le 335$ MB per process.
- **Disk Storage:** $\ge 500$ MB free disk space for raw logs, neural checkpoints, publication figures, and processed tables.

---

## 2. Dependencies & Clean Environment Installation

### 2.1 Dependencies Specification
All dependencies are frozen in `requirements.txt` and `pyproject.toml`:

```text
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
gymnasium>=0.29.0
pyyaml>=6.0
pytest>=8.0.0
tabulate>=0.9.0
```

*Verified runtime library versions:*
- PyTorch: `2.13.0+cpu`
- NumPy: `2.4.6`
- SciPy: `1.17.1`
- Pandas: `3.0.3`
- Matplotlib: `3.11.0`
- Gymnasium: `1.0.0`
- PyYAML: `6.0.2`
- Pytest: `9.1.1`

### 2.2 Step-by-Step Installation Procedure

```bash
# 1. Clone repository and navigate to root
git clone <repository_url>
cd PROJECT

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS (bash/zsh):
source .venv/bin/activate

# 4. Upgrade pip and install package dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Install package in editable development mode
pip install -e .
```

### 2.3 Verification of Clean Environment Installation
Execute the complete test suite to verify algorithmic components, environment rules, action masking, encoders, and reward engines:

```bash
pytest tests/ -v
```

*Expected output:* `109 passed in ~16.5s` (100% test pass rate).

---

## 3. Configuration Management & Reproducibility Invariants

### 3.1 Deterministic Random Seed Derivation Rule
To guarantee independent reproduction across machines, seeds are strictly derived from the Master Seed Sequence defined in Protocol Section 12:

$$\mathcal{S} = \{42, 101, 202, 303, 404\}$$

For any master seed $S \in \mathcal{S}$:
- **Environment Seed:** $S_{\text{env}} = S$
- **Agent / Neural Weights Seed:** $S_{\text{agent}} = S$
- **Opponent Seed:** $S_{\text{opp}} = S + 10$
- **Evaluation Tournament Seed Base:** $S_{\text{eval}} = S + 1000$ (e.g. $1042 \dots 1291$ for $N=250$)
- **Ablation Tournament Seed Base:** $S_{\text{ablation}} = S + 3000$ (e.g. $3042 \dots 3091$ for $N=50$)

Seed control is enforced programmatically via:
```python
from phd_ludo_drl.config.seed import set_seed
set_seed(42)  # Synchronously locks random, np.random, torch.manual_seed
```

### 3.2 Action Space & Action Masking Invariant
The action space consists of 4 discrete token indices $\mathcal{A} = \{0, 1, 2, 3\}$. Action masking is mandatory:
- Value-based (DQN, C51): $Q_{\text{masked}}(s, a) = Q(s, a) + (1 - M_a) \times (-10^9)$
- Policy-based (PPO, SAC): $\operatorname{logits}_{\text{masked}}(s, a) = \operatorname{logits}(s, a) + (1 - M_a) \times (-10^9)$
- **Invariant:** Zero illegal actions ($0.00\%$) across all runs.

---

## 4. Execution Commands: Training Pipeline

### 4.1 Pilot Training Verification Runs (500 Steps)
To rapidly verify model convergence and check for memory leaks or non-finite losses:

```bash
# Model 1: Double Dueling DQN
python scripts/run_dqn_pilot.py

# Model 2: PPO Baseline
python scripts/run_ppo_pilot.py

# Model 3: Discrete SAC
python scripts/run_model3_pilot.py

# Model 4: Categorical DQN (C51)
python scripts/run_model4_pilot.py

# Execute all 4 pilots simultaneously and produce diagnostic curves:
python scripts/run_phase11_pilots.py
```

### 4.2 Unified Training Runner
Train any algorithm with custom parameters using the centralized CLI:

```bash
# Train DQN on Seed 42 for 1,000 steps
python train.py --algo dqn --steps 1000 --seed 42

# Train PPO on Seed 101
python train.py --algo ppo --steps 1000 --seed 101

# Train Discrete SAC on Seed 42
python train.py --algo model3 --steps 1000 --seed 42

# Train Categorical DQN (C51) on Seed 42
python train.py --algo model4 --steps 1000 --seed 42
```

### 4.3 Multi-Seed Production Training Campaign (500k Steps per Model)
To launch large-scale training across all 5 seeds ($500,000$ steps each):

```bash
# Single model multi-seed
python scripts/run_phase13_final_training.py --algo dqn --seeds 42,101,202,303,404

# Full 20-run benchmark campaign (4 models x 5 seeds)
python scripts/run_phase13_final_training.py --all-models --all-seeds
```

---

## 5. Execution Commands: Evaluation & Benchmarking

### 5.1 Standardized 250-Game Matchup Evaluation
Evaluates checkpoints against Random Opponent, Expert Opponent (Alhajry 2012 priority strategy), and Self-Play with strict 50% Seat 0 / 50% Seat 1 alternating assignments:

```bash
python scripts/run_phase14_evaluation.py
```
*Outputs stored directly to `results/raw/`.*

### 5.2 Environmental Stochasticity & Robustness Analysis (Loaded Dice)
Evaluates policy retention across 5 predefined dice variance levels ($1.0\times, 1.1\times, 1.2\times, 1.3\times, 1.4\times$ variance):

```bash
python scripts/run_phase15_robustness.py
```
*Outputs stored directly to `results/robustness/`.*

### 5.3 Architectural & Reward Shaping Ablation Studies
Executes controlled ablations isolating the Structured CNN vs Flat 60-D baseline, Dense 8-event vs Sparse terminal reward, and Action Masking vs Unmasked penalties:

```bash
python scripts/run_phase16_ablations.py
```
*Outputs stored directly to `results/ablations/raw/`.*

---

## 6. Execution Commands: Statistical Analysis & Visualization

### 6.1 Phase 17 — Statistical Analysis Pipeline
Computes mean, variance, SEM, 95% Wilson Score CIs, 10,000-resample Bootstrap CIs, paired tests (McNemar, paired t), independent tests (Welch's t, Mann-Whitney U, two-proportion z), seat fairness tests, and effect sizes (Cohen's d, Hedges' g, Odds Ratio):

```bash
# Analyze primary results/raw/
python scripts/run_phase17_statistical_analysis.py

# Include all stored raw experimental sets (ablations + robustness)
python scripts/run_phase17_statistical_analysis.py --include-all-stored
```
*Outputs written to `results/processed/` and `results/tables/`.*

### 6.2 Phase 18 — Publication-Quality Visualization & Tables
Generates 10 high-resolution figures (300 DPI) and 5 publication tables (CSV, Markdown, LaTeX) without hardcoded values:

```bash
python scripts/run_phase18_visualization.py
```
*Figures written to `results/figures/`; tables written to `results/tables/`.*

---

## 7. Artifact Locations & File Lineage

### 7.1 Checkpoint Registry
| Model / Experiment | Checkpoint Path | Description |
| :--- | :--- | :--- |
| **DQN Pilot** | `experiments/phase11_pilots/dqn/checkpoints/pilot_checkpoint.pt` | Model 1 trained weights |
| **PPO Pilot** | `experiments/phase11_pilots/ppo/checkpoints/pilot_checkpoint.pt` | Model 2 trained weights |
| **Discrete SAC Pilot** | `experiments/phase11_pilots/model3/checkpoints/pilot_checkpoint.pt` | Model 3 trained weights |
| **Categorical DQN Pilot** | `experiments/phase11_pilots/model4/checkpoints/pilot_checkpoint.pt` | Model 4 trained weights |
| **DQN Production** | `experiments/EXP_DQN_structured_cnn_42_20261003_214829/checkpoints/final_model.pt` | Multi-seed final checkpoint |
| **Flat Encoder Ablation** | `results/ablations/EXP_ABLATION_REPR_FLAT_42/checkpoints/checkpoint.pt` | Flat 60-D baseline weights |
| **Sparse Reward Ablation**| `results/ablations/EXP_ABLATION_REWARD_SPARSE_42/checkpoints/checkpoint.pt` | Sparse terminal reward weights |

### 7.2 Stored Raw Experimental Results (`results/raw/`)
- `dqn_vs_random_games.csv` / `.json`: 250 individual games with turns, rewards, captures, goal finishes, seat indicators.
- `dqn_vs_expert_games.csv` / `.json`: 250 individual games vs Alhajry (2012) expert.
- `dqn_vs_self_play_games.csv` / `.json`: 250 individual games vs cloned self.
- `ppo_vs_random_games.csv` / `.json`: 250 individual games with matched evaluation seeds.
- `evaluation_protocol_frozen.json`: Frozen protocol parameters and tournament invariants.

### 7.3 Processed Statistical Ledgers (`results/processed/`)
- `consolidated_game_records.csv`: Game-level master ledger ($N=1,000$).
- `descriptive_statistics.csv` / `.json`: Complete aggregations, Wilson 95% CIs, and Bootstrap 10k CIs.
- `hypothesis_tests.csv` / `.json`: Complete paired and independent hypothesis tests.
- `effect_sizes.csv` / `.json`: Cohen's d, Hedges' g, Odds Ratios, Risk Differences.
- `distributional_normality_tests.csv`: Shapiro-Wilk and D'Agostino omnibus diagnostics.
- `seed_level_statistics.csv`: Evaluation seed block tracking.

### 7.4 Publication Figures (`results/figures/`, 300 DPI)
- `training_reward_curves.png`: Training reward progression across models.
- `loss_curves.png`: 4-panel step-level loss decay trajectories.
- `random_opponent_win_rates.png`: Comparative win rates vs Random Opponent with Wilson CIs.
- `expert_opponent_win_rates.png`: Performance comparison against Heuristic Expert.
- `self_play_results.png`: Self-play outcome parity and turn distribution.
- `convergence_curves.png`: Relative loss reduction percentage across models.
- `sample_efficiency.png`: Step efficiency and wall-clock training velocity.
- `stability_variance_profiles.png`: Return and duration dispersion boxplots.
- `stochasticity_robustness.png`: Performance retention across dice variance perturbations ($1.0\times \dots 1.4\times$).
- `ablation_results.png`: Representation and reward shaping ablation comparisons.

### 7.5 Publication Tables (`results/tables/`, CSV, Markdown, LaTeX)
- `table_experimental_configuration`: Architecture, hyperparameters, state dimensions.
- `table_training_results`: Steps, runtime, memory, initial/final loss, loss reduction %.
- `table_evaluation_results`: Win rates, Wilson CIs, turns, rewards, captures, seat bias.
- `table_statistical_results`: Hypothesis tests, test stats, exact p-values, conclusions.
- `table_ablation_results`: Representation and reward shaping impacts.

---

## 8. Clean Environment Reproduction Checklist

To certify full replication from a pristine system:
1. [x] Clone repository to a clean directory.
2. [x] Create clean virtual environment (`python -m venv .venv`).
3. [x] Install dependencies (`pip install -r requirements.txt`).
4. [x] Run test suite (`pytest tests/`) $\to$ Confirm 109/109 tests pass.
5. [x] Execute statistical analysis (`python scripts/run_phase17_statistical_analysis.py`) $\to$ Verify outputs in `results/processed/` and `results/tables/`.
6. [x] Execute visualization suite (`python scripts/run_phase18_visualization.py`) $\to$ Verify 10 figures in `results/figures/`.
7. [x] Confirm no numerical values were hardcoded in analysis or visualization scripts.

---
*For questions, replication issues, or formal verification contact: Lead PhD Researcher.*
