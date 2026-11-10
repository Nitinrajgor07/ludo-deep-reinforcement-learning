# Running and Demonstrating the PhD Ludo DRL Project

This guide provides the exact commands to execute unit tests, train or evaluate all four reinforcement learning models, and launch the interactive browser demonstration for supervisor meetings and presentations.

---

## 1. Run Tests

To execute the complete regression test suite across the environment, models, opponents, statistical pipeline, and web UI components:

```bash
python -m pytest
```

*Expected output: `117 passed in ~18s`.*

---

## 2. Run DQN (Model 1: Double Dueling DQN + PER)

To train or run an end-to-end experiment with Double Dueling DQN:

```bash
python train.py --algorithm dqn --seed 42 --steps 1000 --eval-games 10 --opponent random --description "DQN Evaluation Run"
```

---

## 3. Run PPO (Model 2: On-Policy Actor-Critic)

To train or run an end-to-end experiment with PPO:

```bash
python train.py --algorithm ppo --seed 42 --steps 1000 --eval-games 10 --opponent random --description "PPO Evaluation Run"
```

---

## 4. Run Model 3 (Model 3: Discrete Soft Actor-Critic)

To train or run an end-to-end experiment with Discrete SAC:

```bash
python train.py --algorithm model3 --seed 42 --steps 1000 --eval-games 10 --opponent random --description "Discrete SAC Evaluation Run"
```

---

## 5. Run Model 4 (Model 4: Categorical Distributional DQN / C51)

To train or run an end-to-end experiment with C51 DQN:

```bash
python train.py --algorithm model4 --seed 42 --steps 1000 --eval-games 10 --opponent random --description "C51 Distributional DQN Run"
```

---

## 6. Start Browser Demo

To launch the interactive supervisor research demonstration web UI:

```bash
python -m uvicorn web.app:app --reload --port 8000
```

*Or run via helper script:*
```bash
python scripts/run_web.py
```

*Or on Windows:*
```bash
run_web.bat
```

---

## 7. Browser URL

Open any modern web browser and navigate to:

```
http://127.0.0.1:8000
```
or
```
http://localhost:8000
```

The application is configured for local research inspection with 100% Light Glassmorphism design and zero authentication friction.

---

## 8. How to Demonstrate the Project to Your Supervisor

Follow these sequential steps during your PhD supervision meeting:

### Step 1: Start the Web Application
Run the start command in your terminal:
```bash
python -m uvicorn web.app --reload
```

### Step 2: Open Browser
1. Navigate to `http://127.0.0.1:8000` (or `http://localhost:8000`).
2. The browser automatically lands on the **Executive Dashboard**.

### Step 3: Walk Through Dashboard (`📊 Dashboard`)
1. Show the **Operational Status** banner and research program overview.
2. Present the **Four Model Architecture Cards**:
   - Model 1: Double Dueling DQN + Prioritized Replay (PER).
   - Model 2: Proximal Policy Optimization (PPO) with Action Masking.
   - Model 3: Discrete Soft Actor-Critic (SAC) with twin critics and automatic entropy tuning.
   - Model 4: Categorical DQN (C51) with 51-atom distribution.
3. Review the **Published Baselines vs. Advanced 4-Model Framework Table**.
4. Review the **Latest Experiments** table drawn from real experiment summaries.

### Step 4: Run a Live Interactive Match (`🎮 Live Ludo`)
1. In the top configuration card:
   - Select agent: **DQN**, **PPO**, **Model 3**, or **Model 4**.
   - Select an available trained checkpoint from the dropdown.
   - Select the opponent: **Random** or **Expert**.
2. Demonstrate real neural network inference:
   - Click **"AI MOVE"** to execute a single turn. Point out the real-time dice roll, legal action pills, selected neural network action, and reward engine feedback.
   - Click **"AUTO PLAY"** to run 10 autonomous turns on the vector SVG board.
   - Click **"RESET MATCH"** to cleanly reset board state and telemetry.

### Step 5: Show Stored Experimental Results (`📊 Research Results`)
1. Navigate to **"📊 Research Results"** in the sidebar.
2. Review the 4 algorithm summary cards with scientific status badges (Final Research Result vs Published Baseline).
3. Click through the 5 empirical evidence tabs:
   - **🏋️ Training Convergence:** Show PPO's $6.4\times$ computational throughput speedup over DQN and loss trajectories.
   - **🏆 Tournament Evaluations:** Show win rates against Random ($N=250$) and Expert ($N=250$) baselines.
   - **📐 Statistical Analysis:** Present Wilson score 95% confidence intervals and McNemar significance tests with dynamic category filtering.
   - **🌪️ Stochasticity Robustness:** Show retention ratios under loaded dice and noisy dice conditions (Model 3 Discrete SAC achieves $1.22\times$ retention).
   - **🔬 Ablation Studies:** Show performance drops when removing circular CNN padding or action masking.

### Step 6: Present Publication Visualizations (`📈 Training Analytics`)
Navigate to **"📈 Training Analytics"** to present the high-resolution publication figures generated directly from experimental data:
- Figure 1: Training Reward Curves
- Figure 2: Loss Convergence Trajectories
- Figure 7: Sample Efficiency Radar Chart
- Figure 9: Stochasticity Robustness Comparison

### Step 7: Inspect Raw Checkpoints and Experiments (`🧪 Experiment Explorer`)
Navigate to **"🧪 Experiment Explorer"** to browse the master experiment catalog, filter by algorithm or seed, and inspect raw `summary.json`, `config_frozen.yaml`, and `.pt` model weights.

### Step 8: Review Experimental Protocol (`📋 Protocol & Reproducibility`)
Review the master seeds $\mathcal{S} = \{42, 101, 202, 303, 404\}$, environment specifications, and 1-click reproduction commands.

### Step 9: Sign Out
Click the **"Logout"** button at the bottom of the sidebar to securely clear session state and return to the login screen.
