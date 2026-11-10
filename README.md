# Documentation & Research Index

Welcome to the comprehensive technical and research documentation for **Deep Reinforcement Learning for Stochastic Turn-Based Board Games: Four-Model Empirical Benchmark on Ludo MDP**.

---

## 📚 Core Architecture & Protocols

These documents define the governing scientific protocols, state abstractions, and system architecture:

| Document | Description |
| :--- | :--- |
| **[`../FINAL_EXPERIMENT_PROTOCOL.md`](../FINAL_EXPERIMENT_PROTOCOL.md)** | **Frozen Benchmark Protocol (v1.0.0):** 22 research sections governing seeds, algorithms, hyperparameters, and evaluation rules. |
| **[`../PROJECT_ARCHITECTURE.md`](../PROJECT_ARCHITECTURE.md)** | **System Architecture:** Detailed system blueprint, modular design, circular CNN encoding, and algorithm structures. |
| **[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)** | **13-Dimension Reproducibility Standard:** Verification checklists, seeds, hardware specifications, and audit trails. |
| **[`FROZEN_STATE_REPRESENTATION.md`](FROZEN_STATE_REPRESENTATION.md)** | **State Representation Specification:** Strict definitions of the $(4, 52)$ spatial tensor, $12$-D context vector, and action masking. |
| **[`future_four_player_research.md`](future_four_player_research.md)** | **4-Player Theoretical Extensions:** Multi-agent extensions, coalition dynamics, and empirical design for 4 active players. |

---

## 🚀 Execution & Demonstration Guides

Step-by-step guides for training, evaluating, and running the interactive web demonstration platform:

| Guide | Description |
| :--- | :--- |
| **[`RUN_PROJECT.md`](RUN_PROJECT.md)** | **Comprehensive Run Guide:** Quick-start commands for smoke tests, training pipelines, evaluations, and tests. |
| **[`WEB_DEMONSTRATION.md`](WEB_DEMONSTRATION.md)** | **Web Interface Guide:** Architecture of the FastAPI + Light Glassmorphism interactive dashboard & 4-player arena. |
| **[`../FINAL_RELEASE_GUIDE.md`](../FINAL_RELEASE_GUIDE.md)** | **Deployment & Release Guide:** Verification checklists, packaging instructions, and deployment validation. |

---

## 📊 Empirical Analysis & Experimental Reports

Scientific reports covering tournament evaluations, hypothesis tests, robustness, and ablations:

| Report | Focus Area |
| :--- | :--- |
| **[`research_analysis.md`](research_analysis.md)** | **Consolidated Research Findings:** Cross-paradigm statistical analysis, effect sizes, and doctoral conclusions. |
| **[`PHASE10_FINAL_EVALUATION_REPORT.md`](PHASE10_FINAL_EVALUATION_REPORT.md)** | **Tournament Evaluation:** 200,000-game benchmark against Random and Heuristic Expert opponents. |
| **[`PHASE11_STATISTICAL_ANALYSIS_REPORT.md`](PHASE11_STATISTICAL_ANALYSIS_REPORT.md)** | **Statistical Tests:** Welch's $t$-test, Mann-Whitney $U$, McNemar's test, and Holm-Bonferroni corrections. |
| **[`PHASE12_ROBUSTNESS_AND_ABLATION_REPORT.md`](PHASE12_ROBUSTNESS_AND_ABLATION_REPORT.md)** | **Robustness & Ablations:** Dice variance perturbation ($1.0\times$ to $1.4\times$) and masking ablation. |
| **[`PHASE13_RESEARCH_VISUALIZATION_REPORT.md`](PHASE13_RESEARCH_VISUALIZATION_REPORT.md)** | **Figure Catalog:** Publication-grade visualizations and 95% confidence intervals. |
| **[`PHASE14_LIGHT_GLASS_WEBSITE_REPORT.md`](PHASE14_LIGHT_GLASS_WEBSITE_REPORT.md)** | **Web Platform Report:** Implementation details for the real-time simulation frontend. |

---

## 🔬 Model Development Reports

Incremental validation and development milestones for each RL architecture:

| Report | Description |
| :--- | :--- |
| **[`PHASE6_PPO_VALIDATION_REPORT.md`](PHASE6_PPO_VALIDATION_REPORT.md)** | On-policy Proximal Policy Optimization (PPO) agent implementation and loss convergence. |
| **[`PHASE7_DISCRETE_SAC_VALIDATION_REPORT.md`](PHASE7_DISCRETE_SAC_VALIDATION_REPORT.md)** | Maximum Entropy Discrete Soft Actor-Critic (SAC) implementation and temperature tuning. |
| **[`PHASE8_C51_VALIDATION_REPORT.md`](PHASE8_C51_VALIDATION_REPORT.md)** | Categorical DQN (C51) distributional reinforcement learning with 51 return atoms. |
| **[`PHASE9_PRODUCTION_TRAINING_PIPELINE_REPORT.md`](PHASE9_PRODUCTION_TRAINING_PIPELINE_REPORT.md)** | Unified training orchestrator with deterministic seed control and checkpoint tracking. |

---

## 🛡️ Health & Quality Audits

Quality assurance reports ensuring zero discrepancies and 100% test passing rates:

| Audit Document | Description |
| :--- | :--- |
| **[`../FINAL_VALIDATION_REPORT.md`](../FINAL_VALIDATION_REPORT.md)** | **System Release Audit:** 268/268 unit tests passed, invariant validation, and zero test failures. |
| **[`FINAL_PROJECT_AUDIT.md`](FINAL_PROJECT_AUDIT.md)** | Multi-level audit of code architecture, file structure, and experimental integrity. |
| **[`FINAL_PROJECT_HEALTH_REPORT.md`](FINAL_PROJECT_HEALTH_REPORT.md)** | Package health check, dependency isolation, and environment verification. |
