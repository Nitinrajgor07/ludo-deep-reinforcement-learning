# Second PhD Research Paper: Outline & Evidence Mapping

**Working Title:** Comparative Analysis of Four Deep Reinforcement Learning Architectures on Turn-Based Stochastic Board Games with Circular Topologies and Action Masking  
**Target Venue:** IEEE Transactions on Games / IEEE Conference on Games (CoG)  
**Authorship / Affiliation:** Department of CS & IT, JAIN (Deemed-to-be University)  
**Document Status:** Complete Paper Outline & Evidence Mapping (Phase 21)  

---

## 1. Traceability & Evidence Mapping Matrix

This matrix establishes the 1-to-1 correspondence between each paper section, the underlying research question, and verified empirical artifacts.

| Paper Section | Topic & Scope | Underlying Research Question | Source Artifacts & Verified Files | Key Empirical Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **1. Title** | Precise, non-overclaiming scholarly title | — | `FINAL_EXPERIMENT_PROTOCOL.md` | Formal definition of 4 DRL models on Ludo |
| **2. Abstract** | Structured summary of benchmark, methods, findings | RQ1–RQ6 | `results/processed/`, `results/tables/` | $N=250$ tournament, $N=1000$ robustness, $p$-values |
| **3. Introduction** | Problem definition, board games as DRL testbeds | Background | IEEE baseline paper (Adithya Tej et al.) | Ground-truth published baseline context |
| **4. Related Work** | Game AI, TD-learning on Ludo, DRL board games | Prior literature | Alhajry (2012), Mnih (2015), Schulman (2017) | Published benchmarks: TD($\lambda$) 66%, Q-learning 63% |
| **5. Research Gap** | Value vs Policy vs MaxEnt vs Distributional comparison | Gap analysis | `PROJECT_RULES.md`, `project design.md` | Absence of 4-family controlled comparison on circular MDPs |
| **6. Research Questions**| 6 formal research questions (RQ1–RQ6) | RQ1–RQ6 | `docs/research_analysis.md` | Formalization of hypotheses $H_0$ and $H_1$ |
| **7. Methodology** | MDP tuple, rules, action masking contract | Formalism | `phd_ludo_drl/environment/ludo_env.py` | 52-tile track, 8 safe squares, $\mathcal{A}=\{0,1,2,3\}$ |
| **8. Proposed Architecture** | Spatial CNN with circular padding, context fusion | RQ2 | `phd_ludo_drl/encoding/cnn_encoder.py` | Conv1d 32/64 circular padding $\to$ 128-D latent |
| **9. Experimental Setup** | Master seed sequence, alternating seats, horizon | Method control | `FINAL_EXPERIMENT_PROTOCOL.md`, Section 12 | $\mathcal{S}=\{42, 101, 202, 303, 404\}$, 50% P0 / 50% P1 |
| **10. Four-Algorithm Framework**| Double Dueling DQN, PPO, Discrete SAC, C51 | Architecture | `phd_ludo_drl/agents/`, `train.py` | Implementation invariants across all 4 paradigms |
| **11. Results** | Playing performance, sample efficiency, runtime | RQ1, RQ5 | `results/tables/table_evaluation_results.csv` | DQN 49.20%, PPO 44.00% vs Random; 15.6% vs Expert |
| **12. Statistical Analysis** | Parametric & non-parametric tests, effect sizes | RQ1, RQ6 | `results/processed/hypothesis_tests.csv` | Two-prop $z=1.17, p=0.24$; McNemar $p=0.26$; $d=0.035$ |
| **13. Ablation Study** | Representation (CNN vs Flat), Reward (Dense vs Sparse)| RQ2, RQ3 | `results/tables/table_ablation_results.csv` | Flat 52% vs CNN 42%; Sparse 52% vs Dense 42% |
| **14. Robustness Analysis**| Perturbed dice variance ($1.0\times \dots 1.4\times$) | RQ4 | `results/robustness/master_robustness_summary.json` | Retention ratios: SAC 1.22, C51 1.11, PPO 1.10, DQN 1.00 |
| **15. Discussion** | Interpretation of trade-offs, expert barrier, speed | Synthesis | `docs/research_analysis.md` | PPO $6.4\times$ speedup; tactical barrier vs heuristics |
| **16. Limitations** | Training budget limits, single-game domain | Scientific rigor | `FINAL_EXPERIMENT_PROTOCOL.md` | 500-step pilots vs 500k budget; zero-shot expert |
| **17. Research Contribution** | 5 discrete methodological & empirical contributions | Impact | `PROJECT_ARCHITECTURE.md` | First 4-algorithm benchmark, circular padding, dataset |
| **18. Conclusion** | Summary of findings and future research roadmap | Synthesis | `memory.md` | Summary of supported vs open research questions |
| **19. References** | Complete scholarly citations (no fabricated refs) | Academic record| Actual published papers | 18 verified academic references |

---

## 2. Distinction Between Prior Baseline and Current Study

To ensure strict academic honesty, the paper enforces an explicit boundary between prior literature and new experimental contributions:

### Prior Literature & Published Baseline (Adithya Tej et al., IEEE Paper)
- Explored only 2 algorithms: standard DQN and baseline PPO.
- Evaluated on a preliminary two-player Ludo prototype.
- Reported win rates after 500k interaction steps:
  - Alhajry et al. (2012) TD($\lambda$): 66.0% vs Random, 30.0% vs Expert
  - Alhajry et al. (2012) Q-Learning: 63.0% vs Random, 27.0% vs Expert
  - DQN + Flat Encoding: 69.2% vs Random, 28.4% vs Expert
  - DQN + Structured CNN: 78.4% vs Random, 36.1% vs Expert
  - PPO + Flat Encoding: 74.8% vs Random, 33.7% vs Expert
  - PPO + Structured CNN: 83.7% vs Random, 42.3% vs Expert

### Current Study's New Experimental Evidence (Phase 1–20)
- Expansion to **Four Distinct Algorithmic Paradigms**:
  1. Value-based off-policy with prioritized experience replay (Double Dueling DQN + PER).
  2. On-policy actor-critic with generalized advantage estimation (PPO).
  3. Maximum-entropy off-policy actor-critic with exact discrete expectation (Discrete SAC).
  4. Distributional reinforcement learning with categorical atom projection (C51 DQN).
- **Architectural Enhancements:** Circular convolution padding (`padding_mode='circular'`) preserving closed track topology without boundary distortion.
- **Statistical Rigor:** 250-game seat-balanced tournament evaluations with matched pseudorandom seeds, reporting Wilson Score 95% CIs, 10,000-resample Bootstrap CIs, parametric Welch/paired $t$-tests, non-parametric Mann-Whitney $U$ and McNemar tests, Cohen's $d$, and Hedges' $g$.
- **Robustness Framework:** 1,000-game systematic dice variance perturbation ($1.0\times$ to $1.4\times$ loaded dice).
- **Controlled Ablation Suite:** Disentangling representation (Structured CNN vs Flat 60-D) and reward shaping (Dense 8-event vs Sparse terminal).

---

## 3. Detailed Structural Plan for the Full Paper Draft

```text
Section 1: Title & Author Metadata
Section 2: Abstract (Structured: Context, Problem, Methods, Findings, Significance)
Section 3: Introduction (Motivation, Board Game Complexity, Research Objective)
Section 4: Related Work (RL in Board Games, Ludo Heuristics, Representation Learning)
Section 5: Research Gap (Absence of Multi-Family Benchmark, Topological Invariance)
Section 6: Formal Research Questions (RQ1 to RQ6 with explicit hypotheses)
Section 7: Mathematical Formulation & MDP (State, Actions, Transition, Rewards)
Section 8: Proposed State Representation & Circular CNN Architecture
Section 9: Experimental Setup & Seed Derivation Methodology
Section 10: Four-Algorithm Implementation Framework (DQN, PPO, Discrete SAC, C51)
Section 11: Empirical Playing Performance Results (Tables 1, 2, 3)
Section 12: Statistical Analysis & Hypothesis Testing (Table 4)
Section 13: Controlled Ablation Studies (Table 5)
Section 14: Stochasticity & Policy Robustness Analysis
Section 15: Discussion & Theoretical Trade-Offs (Sample vs Wall-Clock Efficiency)
Section 16: Limitations & Threats to Validity
Section 17: Research Contributions
Section 18: Conclusion & Future Roadmap
Section 19: References
```

---
*Created as part of Phase 21 documentation in preparation for full manuscript drafting.*
