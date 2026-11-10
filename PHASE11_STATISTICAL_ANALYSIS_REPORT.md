# Phase 11 — Statistical Analysis Pipeline Specification & Audit Report

**Document Version:** 1.0.0 (FROZEN)  
**Date:** October 6, 2026  
**Status:** Pipeline Prepared & Verified | Analysis Reported as INCOMPLETE (Raw Final Data Missing)  
**Governing Protocol:** [FINAL_EXPERIMENT_PROTOCOL.md](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/FINAL_EXPERIMENT_PROTOCOL.md) Section 20  
**Implementation Modules:**
- [statistical_pipeline.py](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/phd_ludo_drl/analysis/statistical_pipeline.py)
- [run_phase11_statistical_analysis.py](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/scripts/run_phase11_statistical_analysis.py)
- [test_statistical_pipeline.py](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/tests/test_statistical_pipeline.py) (11/11 Passed)

---

## Executive Summary

Phase 11 formalizes and constructs the statistical analysis pipeline to evaluate experimental data using **ONLY** raw game-level records, strictly prohibiting manually typed numbers or fabricated substitute values.

### Protocol Verification Verdict: **INCOMPLETE (DATA SAFETY DIRECTIVE)**
- **Required Final Raw Data Budget:** 200,000 tournament games ($4\text{ models} \times 5\text{ master seeds} \times 2\text{ opponents} \times 5,000\text{ games}$ from verified $500\text{k}$ checkpoints).
- **Final Raw Evaluation Games Found:** **0 / 200,000** ($0.0\%$).
- **Preliminary / Pilot Games Discovered:** 1,000 games ($\le 500\text{k}$ checkpoints, $\le 250$ games per condition).
- **Mandatory Directives Enforced:**
  > *"Every result must be reproducible from raw data. Clearly distinguish PRELIMINARY/PILOT from FINAL EXPERIMENT. If final raw data is missing, report the analysis as incomplete rather than generating substitute values."*
- **Action Taken:**
  1. The final statistical analysis is officially reported as **INCOMPLETE (PENDING FINAL EVALUATION DATA)**.
  2. No substitute, synthetic, or interpolated values have been generated.
  3. The pipeline architecture has been fully constructed, tested, and validated, standing ready to process final raw records the moment Phase 10 evaluation games are generated.

---

## 1. Raw Data Audit & Data Tier Classification

The pipeline audits raw directories and classifies datasets into explicit data tiers:

```
[2026-10-06 18:44:25] Auditing Raw Evaluation Data for Statistical Pipeline...
========================================================================================
PHASE 11 RAW EVALUATION DATA AUDIT REPORT
Protocol Version: 1.0.0 (FROZEN)
Audit Timestamp: 2026-10-06T13:14:25.392080+00:00
Total Games Discovered: 0 (results/final_evaluation/raw/) / 1,000 (results/raw/)
Final Experiment Games: 0 / 200,000
Preliminary Pilot Games: 1,000
Data Tier: PRELIMINARY_PILOT
----------------------------------------------------------------------------------------
STATUS: INCOMPLETE (Final raw evaluation data is missing or partial).
Required: 200,000 games across 4 models x 5 master seeds from 500k checkpoints.
Protocol Directive: Final statistical analysis is marked INCOMPLETE.
Under NO circumstances will substitute or fabricated values be generated.
========================================================================================
```

### Data Tier Criteria:
| Tier Designation | Eligibility Criteria | Status in Repo | Quarantining & Transparency |
| :--- | :--- | :---: | :--- |
| **`FINAL_EXPERIMENT`** | Verified $\ge 500,000$ steps checkpoint; 5,000 games per matchup; 200,000 total games across 4 models $\times$ 5 seeds | **0 Games** (Pending) | Required for publication claims; reported as INCOMPLETE until generated. |
| **`PRELIMINARY_PILOT`** | Step count $< 500,000$ (e.g. 50/460/500/1000 steps); $\le 250$ games per matchup | **1,000 Games** | Stored in `results/raw/`; labelled with explicit `PRELIMINARY_PILOT` tag; never conflated with final benchmark. |

---

## 2. Statistical Pipeline Architecture & Capabilities

The pipeline in [statistical_pipeline.py](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/phd_ludo_drl/analysis/statistical_pipeline.py) computes all required statistical dimensions purely from raw data:

### A. Descriptive Statistics & Rigorous Confidence Intervals
- **Point Estimates:** Sample size $N$, Mean ($\mu$), Median, Standard Deviation ($\sigma$, $ddof=1$), Variance ($\sigma^2$), Interquartile Range ($\text{IQR} = Q_3 - Q_1$), Minimum, Maximum, and Standard Error of the Mean ($\text{SEM}$).
- **Proportion CIs (Win Rates):**
  - **95% Wilson Score Interval:** Asymmetric binomial proportion interval without zero-margin boundary collapse.
  - **95% Non-Parametric Bootstrap Interval:** 10,000 Monte Carlo resamples per condition.
- **Continuous CIs (Rewards, Turns, Captures):**
  - **95% Student's $t$-Distribution Interval:** Parametric interval based on sample SEM and $t_{n-1}$.
  - **95% Non-Parametric Bootstrap Interval:** 10,000 resamples for mean reward, game length, and captures.

### B. Pairwise Model Comparisons & Multiple-Testing Corrections
- **All Pairwise Contrasts:** Evaluates all $\binom{K}{2}$ pairs across models present (DQN vs PPO, DQN vs Discrete SAC, DQN vs C51, PPO vs Discrete SAC, PPO vs C51, Discrete SAC vs C51).
- **Hypothesis Testing Suite:**
  - **Win Rates:** Two-Proportion $z$-test (independent), Fisher's Exact test (contingency table), and McNemar test with continuity correction (matched seeds).
  - **Continuous Rewards & Turns:** Welch's two-sample heteroscedastic $t$-test (parametric) and Mann-Whitney $U$ test (non-parametric rank sum).
- **Multiple-Comparison Corrections:**
  - **Bonferroni Adjustment:** FWER control $p_{\text{adj}} = \min(1.0, p \times m)$.
  - **Holm-Bonferroni Step-Down:** More powerful FWER control sorting $p$-values: $p_{(i)} \times (m - i + 1)$.
  - **Benjamini-Hochberg (1995) FDR:** False Discovery Rate control: $p_{(i)} \times m / i$.
  - Reports both unadjusted and adjusted $p$-values to prevent inflated Type I error across model families.

### C. Standardized Effect Sizes
- **Continuous Metrics:**
  - **Cohen's $d$:** Standardized mean difference pooled by sample size and variance.
  - **Hedges' $g$:** Unbiased small-sample corrected effect size $g = d \times J(df)$.
  - **95% Confidence Intervals for Cohen's $d$:** Derived via Hedges & Olkin asymptotic standard error.
  - **Standard Qualitative Interpretation:** Negligible ($<0.2$), Small ($0.2-0.5$), Medium ($0.5-0.8$), Large ($\ge 0.8$).
- **Proportion Metrics:**
  - **Risk Difference ($p_1 - p_2$):** With 95% Wald interval.
  - **Odds Ratio (OR):** With Haldane-Anscombe zero-cell correction and Woolf's logit 95% interval.
  - **Wilcoxon $r$:** Non-parametric effect size $r = |Z| / \sqrt{N}$.

### D. Cross-Seed Stability & Invariance Analysis
- Stratifies performance seed-by-seed across all master seeds ($\mathcal{S} = \{42, 101, 202, 303, 404\}$).
- Computes inter-seed mean $\bar{\mu}$, inter-seed standard deviation $s_{\text{seeds}}$, and Coefficient of Variation ($CV = s_{\text{seeds}} / \bar{\mu}$).
- Conducts **Kruskal-Wallis $H$-test** across seeds to test the null hypothesis $H_0: \text{Seed invariance}$.

### E. Opponent Sensitivity Analysis
- Stratifies performance against Random Opponent vs Expert Opponent.
- Computes performance drops ($\Delta W = W_{\text{rnd}} - W_{\text{exp}}$, $\Delta \bar{R}$, $\Delta \bar{T}$).
- Conducts Two-Proportion $z$-tests and Welch's $t$-tests to evaluate adaptation degradation.

### F. Seat & First-Mover Fairness Evaluation
- Compares Seat 0 (Player 0) vs Seat 1 (Player 1) outcomes for all models and opponents.
- Computes Seat Bias $\Delta W_{\text{seat}} = W_{P0} - W_{P1}$.
- Tests hypothesis $H_0: W_{P0} = W_{P1}$ via Fisher's Exact test and Two-Proportion $z$-test.
- Tests seat reward neutrality via Welch's $t$-test.
- Flags conditions as "Seat Neutral" ($p \ge 0.05$) vs "Seat Biased" ($p < 0.05$).

---

## 3. Publication Tables Generated Directly from Raw Data

Every table is generated dynamically without manual entry into CSV, GitHub Markdown, and publication LaTeX formats:

| Table Identifier | File Stem | Contents & Scope |
| :--- | :--- | :--- |
| **Table 1** | `table1_descriptive_summary` | Sample size $N$, Mean, StdDev, Variance, Median (IQR), 95% CI |
| **Table 2** | `table2_win_rate_and_confidence_intervals` | Win rates with 95% Wilson Score CI vs 10,000-resample Bootstrap CI |
| **Table 3** | `table3_comparative_hypothesis_tests` | Pairwise model hypothesis tests with raw, Bonferroni, Holm, and FDR $p$-values |
| **Table 4** | `table4_effect_sizes` | Cohen's $d$, Hedges' $g$, Odds Ratio, Risk Difference with 95% CIs |
| **Table 5** | `table5_seat_fairness_analysis` | Player 0 vs Player 1 win rates, seat bias, and seat neutrality $p$-values |
| **Table 6** | `table6_distributional_normality_tests` | Skewness, kurtosis, Shapiro-Wilk $W$, D'Agostino tests |
| **Table 7** | `table7_cross_seed_analysis` | Individual seed metrics, inter-seed variance, CV, Kruskal-Wallis test |
| **Table 8** | `table8_opponent_sensitivity_analysis` | Random vs Expert contrast, performance drops, sensitivity effect sizes |

---

## 4. Verification & Unit Test Evidence

Verified with 11 automated unit tests in [test_statistical_pipeline.py](file:///c:/Users/asus/Desktop/TDPCL/PROJECT/tests/test_statistical_pipeline.py):

```powershell
============================= test session starts =============================
tests/test_statistical_pipeline.py::test_wilson_score_interval PASSED    [  9%]
tests/test_statistical_pipeline.py::test_bootstrap_ci PASSED             [ 18%]
tests/test_statistical_pipeline.py::test_students_t_ci PASSED            [ 27%]
tests/test_statistical_pipeline.py::test_cohens_d PASSED                 [ 36%]
tests/test_statistical_pipeline.py::test_proportion_effects PASSED       [ 45%]
tests/test_statistical_pipeline.py::test_mcnemar_test PASSED             [ 54%]
tests/test_statistical_pipeline.py::test_two_proportion_z_test PASSED    [ 63%]
tests/test_statistical_pipeline.py::test_multiple_comparison_corrections PASSED [ 72%]
tests/test_statistical_pipeline.py::test_data_tier_audit_reports_incomplete_on_missing_final_data PASSED [ 81%]
tests/test_statistical_pipeline.py::test_data_tier_classification_pilot_vs_final PASSED [ 90%]
tests/test_statistical_pipeline.py::test_full_pipeline_execution_and_table_generation PASSED [100%]
======================= 11 passed, 6 warnings in 12.68s =======================
```

---

## 5. Usage Commands

```bash
# 1. Audit raw evaluation data completeness:
python scripts/run_phase11_statistical_analysis.py --audit

# 2. Run final statistical analysis (strictly enforces 200k raw data presence; reports incomplete if missing):
python scripts/run_phase11_statistical_analysis.py --mode final

# 3. Analyze preliminary pilot data (with explicit pilot tier branding):
python scripts/run_phase11_statistical_analysis.py --raw-dir results/raw --mode preliminary
```
