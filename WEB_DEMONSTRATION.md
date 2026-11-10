# Web Demonstration Interface Guide: PhD Deep Reinforcement Learning for Ludo

**Document Identification:** `docs/WEB_DEMONSTRATION.md`  
**System Title:** Interactive Browser Demonstration & Research Visualization Platform  
**Target URL:** `http://127.0.0.1:8000`  
**Framework:** FastAPI + Uvicorn (Python 3.11)  
**Associated Specification:** `FINAL_EXPERIMENT_PROTOCOL.md` (v1.0.0 FROZEN)

---

## 1. Overview of the Web Demonstration Platform

The Web Demonstration platform is an interactive, browser-based graphical user interface designed for showcasing the PhD research project to academic supervisors, examiners, and research collaborators. 

### Core Capabilities
1. **Interactive Ludo Demonstration:**
   - Visual 15x15 Ludo board rendering in scalable vector graphics (SVG) with drop shadows, pulsing legal action highlights, safe square star badges, and goal triangle sectors.
   - Live turn-by-turn stepping and real-time execution of neural model inference.
   - Symmetrical support for Player 0 (Red - Primary RL Agent) and Player 1 (Green - Opponent or Self-Play Clone).
2. **Four-Model Selection & Checkpoint Loading:**
   - Dropdown selection across all four benchmarked architectures:
     * **Model 1:** Double Dueling DQN + Prioritized Experience Replay (PER)
     * **Model 2:** PPO Actor-Critic with Generalized Advantage Estimation (GAE)
     * **Model 3:** Discrete Soft Actor-Critic (SAC) with automatic entropy temperature tuning
     * **Model 4:** Categorical DQN (C51) with 51-atom distributional projection
   - Automatic scanner discovering all existing `.pt` model weights on disk with metadata inspection (step, seed, file size).
3. **Opponent Selection:**
   - Uniform Random Opponent ($0.00\%$ illegal moves guaranteed)
   - Heuristic Expert Opponent (Alhajry 2012 priority strategy: Prefer Release > Defend > Capture > Fast Progress)
   - Self-Play Clone (symmetric parameter matching)
4. **Research Results Dashboard:**
   - Side-by-side comparison tables reading directly from machine-generated summary files (`phase11_pilot_summary.json`, `results/tables/`).
   - Standardized tournament results ($N=250$, 95% Wilson Score CIs).
   - Statistical hypothesis test logs (Two-proportion $z$-test, McNemar, Welch's $t$, Mann-Whitney $U$, Cohen's $d$).
   - Stochastic noise robustness retention curves ($1.0\times \to 1.4\times$ loaded dice).
   - Representation and reward shaping ablation tables ($N=400$).
5. **Publication Visualizations:**
   - High-resolution (300 DPI) publication figures covering reward curves, loss trajectories, sample efficiency, and stability profiles.
6. **Experiment Directory Explorer:**
   - Local artifact inspector allowing the user to select any experiment directory, inspect configuration metadata, view step curves, and list checkpoint weights.

---

## 2. Quick Start Guide

### 2.1 Starting the Web Server
Open a terminal in the project root directory (`c:\Users\asus\Desktop\TDPCL\PROJECT`) and run:

```bash
# Recommended command:
python -m uvicorn web.app:app --reload --port 8000

# Alternative runner script:
python scripts/run_web.py

# Windows batch file:
run_web.bat
```

### 2.2 Accessing the Interface
Open any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox) and navigate to:
```text
http://127.0.0.1:8000
```

### 2.3 Stopping the Web Server
In the terminal window running Uvicorn:
- Press `Ctrl + C` to stop the server process.

---

## 3. How to Use the Interface

### 3.1 Navigating Sections
Use the left sidebar navigation radio buttons:
* **🏠 1. Project Home:** High-level executive overview, IEEE paper continuation context, and algorithmic summaries.
* **🎲 2. Interactive Match Demonstration:** The live playable Ludo board where AI models play against opponents.
* **📊 3. Research Results Dashboard:** Verified tabular comparison across training, evaluation, statistics, robustness, and ablations.
* **📈 4. Training Visualizations:** High-resolution 300 DPI publication plots.
* **📂 5. Experiment Explorer:** Deep dive into raw experiment folders and JSON metrics.
* **📜 6. Protocol & Reproducibility:** Reference documentation on deterministic seed derivation and CLI commands.

### 3.2 Running an Interactive Match Demonstration
1. Go to **"🎲 2. Interactive Match Demonstration"**.
2. Expand the **"⚙️ Match Configuration & Trained Checkpoint Selection"** accordion.
3. Select the **Primary AI Agent** (DQN, PPO, Model 3 SAC, or Model 4 C51).
4. Select the **Opponent** (Random, Expert, or Self-Play Clone).
5. Choose a specific trained checkpoint (`.pt` file) from the dropdown list, or choose default initial weights.
6. Click **"Apply Configuration & Restart Game"**.
7. Interactive Game Controls:
   - Click **"▶️ AI Move / Step"** to execute the current player's action. The board updates immediately, and the last moved token is highlighted with a gold border.
   - Click **"⏩ Auto-Play (10 Turns)"** to step forward 10 moves automatically.
   - Click **"🔄 Reset Match"** to re-initialize the board state.
8. Monitor the live telemetry cards:
   - **P0 Reward:** Undiscounted cumulative reward accumulated by Player 0.
   - **Game Turn:** Current turn counter (max 1000).
   - **Red in Goal / Green in Goal:** Number of finished tokens ($0/4 \dots 4/4$).
   - **Move History Table:** Real-time log showing player, roll, selected action, step reward, and total reward.

### 3.3 Inspecting Experiment Results & Tables
1. Go to **"📊 3. Research Results Dashboard"**.
2. Select between the 5 research tabs:
   - *Training Convergence:* Throughput speedup ($6.4\times$ PPO advantage), loss reduction %.
   - *Tournament Evaluations:* $N=250$ head-to-head win rates against Random and Expert.
   - *Statistical Analysis:* Filter by comparison category to view exact test statistics and $p$-values.
   - *Stochasticity Robustness:* View retention ratios under perturbed dice noise ($1.22$ SAC, $1.11$ C51).
   - *Ablation Studies:* View Representation and Reward shaping contrasts.

---

## 4. Research Safety & Integrity Principles

The web interface enforces strict scientific boundaries:
1. **Zero Fabrication:** The dashboard reads strictly from machine-generated JSON and CSV files on disk. If a table or experiment file is absent, a warning is displayed rather than inventing default numbers.
2. **Prominent Classification Badges:**
   - 🟢 **Demonstration Match:** Interactive browser play session (illustrative).
   - 🟡 **Pilot Results:** Small-scale exploratory runs (Phase 11 / Phase 14).
   - 🔵 **Published Baseline:** Historical 500,000-step figures from the precursor IEEE conference paper.
   - 🟣 **Tournament Benchmarks:** Rigorous $N=250$ paired-seed evaluations with 95% Wilson Score confidence intervals.
3. **Architectural Isolation:** The web interface resides cleanly in `web/` and only interfaces with the research core via public API methods (`create_agent`, `create_opponent`, `LudoEnvironment`). The core research models and training loops remain completely untouched.

---

## 5. Automated Verification

The web layer includes an automated unit test suite in `tests/test_web_ui.py`:
- `test_project_root`: Locates root directory and protocol files.
- `test_scan_available_checkpoints`: Verifies discovery of valid `.pt` checkpoints.
- `test_load_pilot_summary`: Confirms JSON parsing of multi-model pilot logs.
- `test_load_table_csv`: Validates CSV loading into DataFrames.
- `test_render_ludo_board_svg`: Checks SVG generation, safe square stars, and yard tags.
- `test_game_session_initialization_and_step`: Confirms valid environment stepping and action mask enforcement.
- `test_game_session_different_algorithms`: Verifies seamless execution across DQN, PPO, Model 3 SAC, and Model 4 C51.

Run all tests:
```bash
pytest tests/ -v
```
*Current test suite status:* **116 passed in 18.68s** (100% passing).

---

STOP.
