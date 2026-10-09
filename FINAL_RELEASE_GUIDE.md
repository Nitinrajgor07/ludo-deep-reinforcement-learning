# PhD Research Platform: Final Release & Replication Guide
## Ludo Deep Reinforcement Learning — Four-Model Comparative Study

**Release Version:** `v1.0.0` (Production & Academic Release)  
**Research Specification:** `FINAL_EXPERIMENT_PROTOCOL.md` Version 1.0.0 (FROZEN)  
**Lead Researcher:** Adithya Tej  
**Institution:** Department of Computer Science & Information Technology, JAIN (Deemed-to-be University), Bangalore, India  
**Date:** October 2026  
**License:** MIT License / Open Academic Replication Package  

---

## 1. Executive Summary & Purpose

This document serves as the formal release guide and comprehensive operational manual for the PhD research replication package: **Ludo Deep Reinforcement Learning — Four-Model Comparative Study**. 

The platform implements an empirical comparison of four distinct reinforcement learning paradigms operating on the 2-player stochastic Ludo Markov Decision Process:
1. **DQN:** Double Dueling Deep Q-Network with Prioritized Experience Replay (PER)
2. **PPO:** On-Policy Proximal Policy Optimization with Generalized Advantage Estimation (GAE)
3. **Model 3 (Discrete SAC):** Soft Actor-Critic with Twin Q-Networks and Target Entropy
4. **Model 4 (C51):** Distributional Categorical Deep Q-Network with 51-atom return support

Every model operates on an identical 4-channel spatial circular CNN representation (128-dimensional latent vector), identical shaped reward functions, and identical strict action masking rules.

---

## 2. System Architecture & Environment Specifications

### 2.1 Python Runtime
- **Target Python Version:** `Python 3.11.9` (64-bit)
- **Supported Python Range:** `Python >= 3.10, <= 3.12`
- **Compiler:** MSC v.1938 64 bit (AMD64) / GCC 11+ on Linux

### 2.2 Verified Hardware Platforms
- **Operating Systems:**
  - Microsoft Windows 10/11 (64-bit)
  - Ubuntu Linux 22.04 LTS / 24.04 LTS (`x86_64`)
  - macOS 12+ (Apple Silicon or Intel `x86_64`)
- **Execution Backend:** CPU Vectorized Execution (`torch.device('cpu')`). The research protocol intentionally locks execution to CPU vectorized tensors to eliminate non-deterministic GPU kernel float accumulation variances across different hardware targets.
- **Hardware Profile:**
  - CPU: 4+ physical cores (Intel i5/i7/i9, AMD Ryzen 5/7/9, or Apple M-series)
  - RAM: 8 GB minimum; 16 GB recommended (measured peak process memory is $\le 335$ MB per run)
  - Storage: $\ge 500$ MB free storage for experiments, checkpoints, and visualization figures.

---

## 3. Dependency Inventory & Packaging Separation

Dependencies are strictly categorized across runtime, research analysis, web demonstration, and automated testing in [`pyproject.toml`](pyproject.toml) and [`requirements.txt`](requirements.txt):

| Category | Package | Minimum Version | Verified Version | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Core Runtime** | `torch` | `>=2.0.0` | `2.13.0+cpu` | Neural network architectures, autograd, tensor ops |
| **Core Runtime** | `numpy` | `>=1.24.0` | `2.4.6` | Vectorized matrix operations, seed generation |
| **Core Runtime** | `gymnasium` | `>=0.29.0` | `1.3.0` | Standardized MDP environment interfaces |
| **Core Runtime** | `pyyaml` | `>=6.0` | `6.0.3` | Hierarchical configuration file management |
| **Research Analysis** | `pandas` | `>=2.0.0` | `3.0.3` | Tabular metrics, benchmark datasets, run aggregation |
| **Research Analysis** | `scipy` | `>=1.10.0` | `1.17.1` | Hypothesis testing (Welch's t-test, Mann-Whitney, Wilcoxon) |
| **Research Analysis** | `matplotlib` | `>=3.7.0` | `3.11.0` | 300 DPI publication vector figures, convergence charts |
| **Research Analysis** | `tabulate` | `>=0.9.0` | `0.9.0` | Exporting formatted Markdown and LaTeX tables for papers |
| **Research Analysis** | `psutil` | `>=5.9.0` | `7.2.2` | System CPU, memory, and throughput profiling |
| **Web Interface** | `fastapi` | `>=0.100.0` | `0.115.0` | High-performance ASGI REST & web application |
| **Web Interface** | `uvicorn` | `>=0.22.0` | `0.32.0` | Production ASGI web server |
| **Web Interface** | `jinja2` | `>=3.1.0` | `3.1.4` | Light glassmorphic HTML templating engine |
| **Testing** | `pytest` | `>=8.0.0` | `9.1.1` | Comprehensive automated regression test suite |

---

## 4. Clean Step-by-Step Installation

### Step 1: Clone Repository
```bash
git clone <repository_url>
cd PROJECT
```

### Step 2: Initialize Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

# Or install editable package with all optional modules:
pip install -e ".[all]"
```

### Step 4: Verify Installation
```bash
python -c "import torch, numpy, gymnasium, pandas, scipy, matplotlib, fastapi, uvicorn, pytest; print('Environment Verified Successfully')"
```

---

## 5. Automated Verification & Quality Assurance

### 5.1 Python Compilation Check
Verify that all source modules, tests, scripts, and web components compile cleanly without syntax errors:
```bash
python -m compileall -q phd_ludo_drl web tests scripts train.py
```
*(Expected: exit code 0, zero errors)*

### 5.2 Test Discovery Check
Verify test collection across the test suite:
```bash
python -m pytest --collect-only
```
*(Expected: 118 tests collected across 17 modules)*

### 5.3 Automated Regression Test Execution
Execute the full regression test suite:
```bash
python -m pytest
```
*(Expected: 118 passed in ~22 seconds)*

---

## 6. Authoritative Research Protocol & Replication

The research replication protocol is strictly governed by `FINAL_EXPERIMENT_PROTOCOL.md` v1.0.0:

### 6.1 Protocol Invariants
- **Training Budget:** 500,000 interaction steps per run.
- **Master Seeds:** $\mathcal{S} = \{42, 101, 202, 303, 404\}$ (five-fold replication).
- **Evaluation Seed Derivation:** $S_{eval} = S + 1000$ ($\mathcal{S}_{eval} = \{1042, 1101, 1202, 1303, 1404\}$).
- **Evaluation Protocol:**
  - 5,000 games vs Random baseline per model/seed.
  - 5,000 games vs Expert heuristic baseline per model/seed.
- **Seat Balancing:** Exactly 50% of games played as Player 0 (Red) and 50% as Player 1 (Green) to eliminate first-mover advantage.
- **Action Masking:** $-10^9$ logit suppression strictly preventing 100% of illegal actions.

### 6.2 CLI Training Replication Commands
```bash
# 1. Double Dueling DQN + PER
python train.py --algorithm dqn --seed 42 --steps 10000 --eval-games 20 --opponent random

# 2. On-Policy PPO
python train.py --algorithm ppo --seed 101 --steps 10000 --eval-games 20 --opponent random

# 3. Discrete SAC
python train.py --algorithm model3 --seed 202 --steps 10000 --eval-games 20 --opponent random

# 4. Categorical C51 DQN
python train.py --algorithm model4 --seed 303 --steps 10000 --eval-games 20 --opponent random
```

### 6.3 Data Storage & Experimental Integrity
- **Raw Experimental Data:** Stored in isolated directories under `experiments/EXP_<ALGORITHM>_<ENCODING>_<SEED>_<TIMESTAMP>/`.
- **Processed Summary Tables:** Stored in `results/tables/`.
- **Publication Figures:** Stored in `results/figures/` (300 DPI vector charts).
- **Integrity Rule:** Raw records are **never overwritten**. Processed tables derive deterministically from raw records without manual data modification.

---

## 7. Executive Web Demonstration Platform

### 7.1 Starting the Web Platform
Launch the FastAPI executive demonstration platform via Uvicorn:
```bash
# Standard command:
python -m uvicorn web.app --reload

# Or via runner:
python scripts/run_web.py

# Or on Windows:
run_web.bat
```
Navigate to: **`http://127.0.0.1:8000`** or **`http://localhost:8000`**

### 7.2 Design System & Aesthetics
- **100% Light Glassmorphism:** Off-white background (`#f8fafc`), translucent white cards (`rgba(255, 255, 255, 0.88)`), subtle borders (`#e2e8f0`), and soft shadows. Zero dark theme elements.
- **Academic Research Standards:** Real empirical data displayed, no simulated final curves, honest "Final training pending" badges where applicable.
- **Pages:**
  1. `📊 Dashboard`: Research overview, four-model architectural specifications, telemetry status.
  2. `🧠 Models`: Algorithmic formulation, neural architectures, action masking, and checkpoints.
  3. `⚡ Training`: Training protocol specifications, empirical pilot curves, and checkpoint telemetry.
  4. `🎯 Evaluation`: Controlled tournament benchmarks vs Random and Expert baselines.
  5. `📈 Statistics`: Descriptive and inferential statistical analysis tables.
  6. `🖼️ Visualizations`: Publication-grade figure gallery with full-resolution inspection.
  7. `🎲 Ludo Arena`: Interactive 15×15 SVG board with real-time AI and opponent execution.
  8. `ℹ️ About Project`: Dissertation research motivation, formal formulation, and technical stack.

---

## 8. Release Sign-Off Checklist

- [x] **Project Structure Clean:** Package boundaries verified (`phd_ludo_drl`, `web`, `scripts`, `tests`).
- [x] **Dependencies Synchronized:** `requirements.txt` and `pyproject.toml` completely synchronized and categorized.
- [x] **Python Compilation:** `python -m compileall` passed on 100% of project files.
- [x] **Test Discovery:** `pytest --collect-only` discovered 118 test cases.
- [x] **Regression Test Suite:** 118/118 tests passed (100% pass rate).
- [x] **Deterministic Seeds:** Seeds $\mathcal{S}=\{42, 101, 202, 303, 404\}$ verified.
- [x] **Web Application Light Theme:** Pure Light Glassmorphism UI verified in real browser without defects.
- [x] **Documentation Complete:** `README.md`, `pyproject.toml`, `requirements.txt`, and `FINAL_RELEASE_GUIDE.md` fully up to date.
