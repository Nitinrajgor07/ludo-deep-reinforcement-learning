"""Statistical Analysis Pipeline for PhD Ludo DRL Research.

Adheres strictly to FINAL_EXPERIMENT_PROTOCOL.md Section 20:
- Reads stored experimental results directly from results/raw/ (and optional raw stores).
- Computes mean, variance, standard deviation, interquartile range (IQR), and SEM.
- Calculates parametric and non-parametric confidence intervals (Wilson Score, Bootstrap 10k).
- Conducts appropriate hypothesis tests matching the experimental design:
  * Paired tests on matched seeds (McNemar, Paired t-test, Wilcoxon Signed-Rank).
  * Independent tests (Welch's t-test, Mann-Whitney U, Two-Proportion z-test, Fisher's Exact).
  * Seat fairness tests (Seat 0 vs Seat 1 balance).
- Evaluates effect sizes (Cohen's d, Hedges' g, Odds Ratio, Risk Difference, Wilcoxon r).
- Generates structured files in results/processed/ and publication tables in results/tables/.
- Adheres to the directive: Do NOT make an automatic "best model" claim; objectively report
  measured differences and statistical evidence.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


# ==============================================================================
# CONFIDENCE INTERVAL CALCULATORS
# ==============================================================================

def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
    continuity_correction: bool = False,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.
    
    References:
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
    Journal of the American Statistical Association, 22(158), 209-212.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    n = float(total)

    if not continuity_correction:
        if successes == 0:
            denom = 1.0 + z2 / n
            upper = min(1.0, (z2 / n) / denom)
            return (0.0, float(upper))
        if successes == total:
            denom = 1.0 + z2 / n
            lower = max(0.0, 1.0 - (z2 / n) / denom)
            return (float(lower), 1.0)
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * (n**2))))
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (float(lower), float(upper))
    else:
        # Wilson score interval with continuity correction (Newcombe, 1998)
        term1 = 2.0 * n * p + z2
        term2_lower = z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1.0 - p) + (4.0 * p - 2.0)) + 1.0
        term2_upper = z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1.0 - p) - (4.0 * p - 2.0)) + 1.0
        denom = 2.0 * (n + z2)
        lower = max(0.0, (term1 - term2_lower) / denom) if p > 0 else 0.0
        upper = min(1.0, (term1 + term2_upper) / denom) if p < 1 else 1.0
        return (float(lower), float(upper))


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic_fn=np.mean,
    confidence: float = 0.95,
    num_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap percentile confidence interval with 10,000 resamples."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        val = float(statistic_fn(arr))
        return (val, val)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_resamples, n))
    bootstrap_samples = arr[indices]
    boot_stats = np.apply_along_axis(statistic_fn, 1, bootstrap_samples)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower = float(np.percentile(boot_stats, lower_pct))
    upper = float(np.percentile(boot_stats, upper_pct))
    return (lower, upper)


def compute_students_t_ci(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute parametric Student's t confidence interval for mean."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n < 2:
        val = float(arr[0]) if n == 1 else 0.0
        return (val, val)
    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    if sem == 0.0:
        return (mean, mean)
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_crit * sem
    return (mean - margin, mean + margin)


# ==============================================================================
# EFFECT SIZE CALCULATORS
# ==============================================================================

def compute_cohens_d(x: List[float], y: List[float]) -> Dict[str, float]:
    """Compute Cohen's d and Hedges' g with pooled variance and 95% confidence intervals."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "d_ci_lower": 0.0, "d_ci_upper": 0.0, "se_d": 0.0}

    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0.0:
        return {"cohens_d": 0.0, "hedges_g": 0.0, "d_ci_lower": 0.0, "d_ci_upper": 0.0, "se_d": 0.0}

    diff = float(np.mean(x) - np.mean(y))
    d = diff / pooled_sd

    # Hedges' g correction factor J(df)
    df = nx + ny - 2
    j_factor = 1.0 - (3.0 / (4.0 * df - 1.0))
    g = d * j_factor

    # Standard error of Cohen's d (Hedges & Olkin, 1985)
    se_d = math.sqrt(((nx + ny) / (nx * ny)) + ((d**2) / (2.0 * (nx + ny))))
    z_crit = float(stats.norm.ppf(0.975))
    d_ci_lower = d - z_crit * se_d
    d_ci_upper = d + z_crit * se_d

    return {
        "cohens_d": float(d),
        "hedges_g": float(g),
        "d_ci_lower": float(d_ci_lower),
        "d_ci_upper": float(d_ci_upper),
        "se_d": float(se_d),
    }


def compute_proportion_effect_sizes(
    w1: int, n1: int, w2: int, n2: int
) -> Dict[str, Any]:
    """Compute Risk Difference, Relative Risk, and Odds Ratio with 95% CIs."""
    if n1 == 0 or n2 == 0:
        return {}
    p1 = w1 / n1
    p2 = w2 / n2
    risk_diff = p1 - p2

    # Standard error for risk difference (Wald)
    se_rd = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    z = 1.95996
    rd_lower = risk_diff - z * se_rd
    rd_upper = risk_diff + z * se_rd

    # Relative risk (Risk Ratio)
    relative_risk = (p1 / p2) if p2 > 0 else (float("nan") if p1 == 0 else float("inf"))

    # Odds Ratio with Woolf's logit method
    # 2x2 table: [[w1, n1-w1], [w2, n2-w2]] = [[a, b], [c, d]]
    a, b = float(w1), float(n1 - w1)
    c, d = float(w2), float(n2 - w2)

    # Haldane-Anscombe correction if any cell is 0
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    or_val = (a * d) / (b * c)
    se_ln_or = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    or_lower = math.exp(math.log(or_val) - z * se_ln_or)
    or_upper = math.exp(math.log(or_val) + z * se_ln_or)

    return {
        "risk_difference": float(risk_diff),
        "risk_diff_ci_lower": float(rd_lower),
        "risk_diff_ci_upper": float(rd_upper),
        "relative_risk": float(relative_risk) if not math.isnan(relative_risk) else None,
        "odds_ratio": float(or_val),
        "odds_ratio_ci_lower": float(or_lower),
        "odds_ratio_ci_upper": float(or_upper),
    }


def compute_wilcoxon_r(z_score: float, total_n: int) -> float:
    """Compute non-parametric Wilcoxon effect size r = |Z| / sqrt(N)."""
    if total_n <= 0:
        return 0.0
    return float(abs(z_score) / math.sqrt(total_n))


def interpret_cohens_d(d: float) -> str:
    """Standard Cohen (1988) heuristic interpretation."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ==============================================================================
# STATISTICAL HYPOTHESIS TESTS
# ==============================================================================

def run_mcnemar_test(
    paired_outcomes_1: List[bool],
    paired_outcomes_2: List[bool],
) -> Dict[str, Any]:
    """Execute McNemar test for paired binary outcomes on matched seeds."""
    if len(paired_outcomes_1) != len(paired_outcomes_2):
        raise ValueError("Paired outcomes must have identical length.")

    # Contingency table:
    # a: both True, b: 1 True & 2 False, c: 1 False & 2 True, d: both False
    b = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if x and not y)
    c = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if not x and y)
    both_won = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if x and y)
    both_lost = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if not x and not y)
    n = len(paired_outcomes_1)

    discordant = b + c
    if discordant == 0:
        return {
            "test_name": "McNemar Test",
            "statistic": 0.0,
            "p_value": 1.0,
            "exact_p_value": 1.0,
            "b": b,
            "c": c,
            "both_won": both_won,
            "both_lost": both_lost,
            "df": 1,
            "method": "exact",
        }

    # Chi-square with Edwards continuity correction
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / discordant
    p_chi2 = float(stats.chi2.sf(chi2_stat, df=1))

    # Exact two-sided binomial test on discordant pairs
    exact_res = stats.binomtest(b, discordant, p=0.5, alternative="two-sided")
    p_exact = float(exact_res.pvalue)

    # Use exact p-value if discordant < 25 per statistical standard
    p_val = p_exact if discordant < 25 else p_chi2

    return {
        "test_name": "McNemar Test",
        "statistic": float(chi2_stat),
        "p_value": float(p_val),
        "exact_p_value": float(p_exact),
        "chi2_p_value": float(p_chi2),
        "b": b,
        "c": c,
        "both_won": both_won,
        "both_lost": both_lost,
        "df": 1,
        "method": "exact_binomial" if discordant < 25 else "chi2_continuity_corrected",
    }


def run_two_proportion_z_test(w1: int, n1: int, w2: int, n2: int) -> Tuple[float, float]:
    """Execute two-proportion z-test (independent groups)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1 = w1 / n1
    p2 = w2 / n2
    p_pooled = (w1 + w2) / (n1 + n2)
    if p_pooled == 0.0 or p_pooled == 1.0:
        return (0.0, 1.0)
    se_pooled = math.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2))
    z = (p1 - p2) / se_pooled
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return (float(z), float(p_val))


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

class StatisticalAnalysisPipeline:
    """End-to-end statistical analysis pipeline for PhD Ludo DRL results."""

    def __init__(
        self,
        raw_dirs: Optional[List[Union[str, Path]]] = None,
        processed_dir: Union[str, Path] = "results/processed",
        tables_dir: Union[str, Path] = "results/tables",
        master_seed: int = 42,
    ):
        self.project_root = Path.cwd()
        if raw_dirs is None:
            self.raw_dirs = [self.project_root / "results" / "raw"]
        else:
            self.raw_dirs = [Path(d) if Path(d).is_absolute() else self.project_root / d for d in raw_dirs]

        self.processed_dir = Path(processed_dir) if Path(processed_dir).is_absolute() else self.project_root / processed_dir
        self.tables_dir = Path(tables_dir) if Path(tables_dir).is_absolute() else self.project_root / tables_dir
        self.master_seed = master_seed

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.raw_game_records: List[Dict[str, Any]] = []
        self.df_games: Optional[pd.DataFrame] = None

    def ingest_stored_results(self) -> pd.DataFrame:
        """Read experimental results directly from results/raw/ without manual values."""
        all_rows: List[Dict[str, Any]] = []

        for raw_dir in self.raw_dirs:
            if not raw_dir.exists():
                continue

            csv_files = sorted(list(raw_dir.glob("*.csv")))
            for csv_file in csv_files:
                # Infer experiment type from directory or filename
                stem = csv_file.stem
                if "robustness" in str(raw_dir):
                    exp_name = "robustness_analysis"
                elif "ablation" in str(raw_dir):
                    exp_name = "ablation_study"
                else:
                    exp_name = "benchmark_evaluation"

                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"[Warning] Failed to read {csv_file}: {e}")
                    continue

                required_cols = {"game_idx", "seed", "algorithm", "opponent_type", "agent_won", "agent_reward", "turns"}
                if not required_cols.issubset(df.columns):
                    continue

                for _, row in df.iterrows():
                    rec = {
                        "experiment": exp_name,
                        "file_source": csv_file.name,
                        "game_idx": int(row["game_idx"]),
                        "seed": int(row["seed"]),
                        "algorithm": str(row["algorithm"]).lower(),
                        "opponent": str(row["opponent_type"]).lower(),
                        "agent_seat": int(row.get("agent_seat", 0)),
                        "winner": int(row.get("winner", 0)),
                        "agent_won": bool(row["agent_won"]),
                        "opp_won": bool(row.get("opp_won", False)),
                        "is_draw": bool(row.get("is_draw", False)),
                        "turns": float(row["turns"]),
                        "agent_reward": float(row["agent_reward"]),
                        "opp_reward": float(row.get("opp_reward", 0.0)),
                        "agent_captures": float(row.get("agent_captures", 0.0)),
                        "opp_captures": float(row.get("opp_captures", 0.0)),
                        "agent_tokens_home": float(row.get("agent_tokens_home", 0.0)),
                        "opp_tokens_home": float(row.get("opp_tokens_home", 0.0)),
                        "dice_variance_multiplier": float(row.get("dice_variance_multiplier", 1.0)),
                    }
                    all_rows.append(rec)

        self.df_games = pd.DataFrame(all_rows)
        return self.df_games

    def compute_normality_diagnostics(self) -> pd.DataFrame:
        """Assess distributional properties (normality, skewness, kurtosis) to justify test choices."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        metrics = ["agent_reward", "turns", "agent_captures", "agent_tokens_home"]

        grouped = self.df_games.groupby(["experiment", "algorithm", "opponent"])
        for (exp, algo, opp), group in grouped:
            seed_label = f"{group['seed'].min()}-{group['seed'].max()} (N={len(group)})"
            for m in metrics:
                vals = group[m].dropna().values
                n = len(vals)
                if n < 3:
                    continue

                # Shapiro-Wilk test (valid for n up to 5000)
                shapiro_stat, shapiro_p = stats.shapiro(vals)
                # D'Agostino-Pearson omnibus test
                dagostino_stat, dagostino_p = (
                    stats.normaltest(vals) if n >= 20 else (np.nan, np.nan)
                )
                skew_val = float(stats.skew(vals))
                kurt_val = float(stats.kurtosis(vals))

                rows.append({
                    "experiment": exp,
                    "algorithm": algo,
                    "seed": seed_label,
                    "opponent": opp,
                    "metric": m,
                    "n": n,
                    "skewness": round(skew_val, 4),
                    "kurtosis": round(kurt_val, 4),
                    "shapiro_wilk_stat": round(float(shapiro_stat), 4),
                    "shapiro_wilk_pvalue": float(shapiro_p),
                    "dagostino_stat": round(float(dagostino_stat), 4) if not np.isnan(dagostino_stat) else None,
                    "dagostino_pvalue": float(dagostino_p) if not np.isnan(dagostino_p) else None,
                    "is_normal_alpha_05": bool(shapiro_p >= 0.05),
                    "distribution_shape": "symmetric" if abs(skew_val) < 0.5 else ("right_skewed" if skew_val > 0 else "left_skewed"),
                    "statistical_justification": "Parametric & non-parametric tests both reported" if shapiro_p < 0.05 else "Normal distribution assumptions satisfied",
                })

        return pd.DataFrame(rows)

    def compute_descriptive_statistics(self) -> pd.DataFrame:
        """Compute mean, variation, SEM, and 95% confidence intervals across all conditions."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        grouped = self.df_games.groupby(["experiment", "algorithm", "opponent"])

        for (exp, algo, opp), group in grouped:
            n = len(group)
            seed_label = f"{group['seed'].min()}-{group['seed'].max()} (N={n})"

            # 1. Win Rate (Binary proportion)
            wins = int(group["agent_won"].sum())
            wr = wins / n
            wr_var = wr * (1.0 - wr)  # Bernoulli variance
            wr_sem = math.sqrt(wr_var / n)
            wr_wilson_lower, wr_wilson_upper = compute_wilson_score_interval(wins, n, confidence=0.95)
            wr_boot_lower, wr_boot_upper = compute_bootstrap_ci(
                group["agent_won"].astype(float).values,
                statistic_fn=np.mean,
                confidence=0.95,
                num_resamples=10000,
                seed=self.master_seed,
            )

            rows.append({
                "experiment": exp,
                "algorithm": algo,
                "seed": seed_label,
                "opponent": opp,
                "metric": "win_rate",
                "n": n,
                "mean": round(wr, 4),
                "std": round(math.sqrt(wr_var), 4),
                "variance": round(wr_var, 4),
                "sem": round(wr_sem, 4),
                "median": round(float(np.median(group["agent_won"].astype(float))), 4),
                "iqr": round(float(np.percentile(group["agent_won"].astype(float), 75) - np.percentile(group["agent_won"].astype(float), 25)), 4),
                "min": 0.0,
                "max": 1.0,
                "ci_95_lower": round(wr_wilson_lower, 4),
                "ci_95_upper": round(wr_wilson_upper, 4),
                "ci_method": "Wilson Score Interval (95%)",
                "ci_bootstrap_lower": round(wr_boot_lower, 4),
                "ci_bootstrap_upper": round(wr_boot_upper, 4),
            })

            # Seat 0 and Seat 1 Win Rates
            s0_df = group[group["agent_seat"] == 0]
            s1_df = group[group["agent_seat"] == 1]
            if not s0_df.empty:
                w0, n0 = int(s0_df["agent_won"].sum()), len(s0_df)
                p0 = w0 / n0
                w0_low, w0_high = compute_wilson_score_interval(w0, n0)
                rows.append({
                    "experiment": exp,
                    "algorithm": algo,
                    "seed": seed_label,
                    "opponent": opp,
                    "metric": "seat_0_win_rate",
                    "n": n0,
                    "mean": round(p0, 4),
                    "std": round(math.sqrt(p0 * (1 - p0)), 4),
                    "variance": round(p0 * (1 - p0), 4),
                    "sem": round(math.sqrt(p0 * (1 - p0) / n0), 4),
                    "median": round(float(np.median(s0_df["agent_won"].astype(float))), 4),
                    "iqr": round(float(np.percentile(s0_df["agent_won"].astype(float), 75) - np.percentile(s0_df["agent_won"].astype(float), 25)), 4),
                    "min": 0.0,
                    "max": 1.0,
                    "ci_95_lower": round(w0_low, 4),
                    "ci_95_upper": round(w0_high, 4),
                    "ci_method": "Wilson Score Interval (95%)",
                    "ci_bootstrap_lower": None,
                    "ci_bootstrap_upper": None,
                })
            if not s1_df.empty:
                w1, n1 = int(s1_df["agent_won"].sum()), len(s1_df)
                p1 = w1 / n1
                w1_low, w1_high = compute_wilson_score_interval(w1, n1)
                rows.append({
                    "experiment": exp,
                    "algorithm": algo,
                    "seed": seed_label,
                    "opponent": opp,
                    "metric": "seat_1_win_rate",
                    "n": n1,
                    "mean": round(p1, 4),
                    "std": round(math.sqrt(p1 * (1 - p1)), 4),
                    "variance": round(p1 * (1 - p1), 4),
                    "sem": round(math.sqrt(p1 * (1 - p1) / n1), 4),
                    "median": round(float(np.median(s1_df["agent_won"].astype(float))), 4),
                    "iqr": round(float(np.percentile(s1_df["agent_won"].astype(float), 75) - np.percentile(s1_df["agent_won"].astype(float), 25)), 4),
                    "min": 0.0,
                    "max": 1.0,
                    "ci_95_lower": round(w1_low, 4),
                    "ci_95_upper": round(w1_high, 4),
                    "ci_method": "Wilson Score Interval (95%)",
                    "ci_bootstrap_lower": None,
                    "ci_bootstrap_upper": None,
                })

            # 2. Continuous metrics
            continuous_metrics = [
                ("agent_reward", "cumulative episodic reward"),
                ("opp_reward", "opponent episodic reward"),
                ("turns", "game duration (turns)"),
                ("agent_captures", "opponent tokens knocked home"),
                ("agent_tokens_home", "friendly tokens reached goal [0..4]"),
            ]

            for col_name, _ in continuous_metrics:
                vals = group[col_name].dropna().values
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1))
                var_val = float(np.var(vals, ddof=1))
                sem_val = float(stats.sem(vals))
                med_val = float(np.median(vals))
                iqr_val = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))

                t_low, t_high = compute_students_t_ci(vals, confidence=0.95)
                boot_low, boot_high = compute_bootstrap_ci(
                    vals,
                    statistic_fn=np.mean,
                    confidence=0.95,
                    num_resamples=10000,
                    seed=self.master_seed,
                )

                rows.append({
                    "experiment": exp,
                    "algorithm": algo,
                    "seed": seed_label,
                    "opponent": opp,
                    "metric": col_name,
                    "n": n,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "variance": round(var_val, 4),
                    "sem": round(sem_val, 4),
                    "median": round(med_val, 4),
                    "iqr": round(iqr_val, 4),
                    "min": round(min_val, 4),
                    "max": round(max_val, 4),
                    "ci_95_lower": round(t_low, 4),
                    "ci_95_upper": round(t_high, 4),
                    "ci_method": "Student's t-Distribution (95%)",
                    "ci_bootstrap_lower": round(boot_low, 4),
                    "ci_bootstrap_upper": round(boot_high, 4),
                })

        return pd.DataFrame(rows)

    def compute_seed_level_statistics(self) -> pd.DataFrame:
        """Compute statistics grouped by seed blocks to verify invariance across random seeds."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        # Group by 50-game blocks representing evaluation seed tranches
        df_copy = self.df_games.copy()
        df_copy["seed_block"] = (df_copy["game_idx"] // 50).map(lambda b: f"seed_block_{b+1} (games {b*50}-{(b+1)*50-1})")

        grouped = df_copy.groupby(["experiment", "algorithm", "opponent", "seed_block"])
        for (exp, algo, opp, block), group in grouped:
            n = len(group)
            seed_range = f"{group['seed'].min()}..{group['seed'].max()}"

            # Win rate
            wins = int(group["agent_won"].sum())
            wr = wins / n
            rows.append({
                "experiment": exp,
                "algorithm": algo,
                "opponent": opp,
                "seed": seed_range,
                "seed_block": block,
                "metric": "win_rate",
                "count": n,
                "mean": round(wr, 4),
                "variance": round(wr * (1 - wr), 4),
            })

            # Continuous metrics
            for m in ["agent_reward", "turns", "agent_captures", "agent_tokens_home"]:
                vals = group[m].values
                rows.append({
                    "experiment": exp,
                    "algorithm": algo,
                    "opponent": opp,
                    "seed": seed_range,
                    "seed_block": block,
                    "metric": m,
                    "count": n,
                    "mean": round(float(np.mean(vals)), 4),
                    "variance": round(float(np.var(vals, ddof=1)), 4),
                })

        return pd.DataFrame(rows)

    def compute_hypothesis_tests(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Conduct parametric and non-parametric hypothesis tests and effect sizes matching experimental design."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame(), pd.DataFrame()

        test_rows = []
        effect_rows = []

        # ----------------------------------------------------------------------
        # 1. Algorithm Comparison: DQN vs PPO on Random Opponent (Matched Seeds)
        # ----------------------------------------------------------------------
        dqn_rnd = self.df_games[(self.df_games["algorithm"] == "dqn") & (self.df_games["opponent"] == "random")].sort_values("seed")
        ppo_rnd = self.df_games[(self.df_games["algorithm"] == "ppo") & (self.df_games["opponent"] == "random")].sort_values("seed")

        if not dqn_rnd.empty and not ppo_rnd.empty:
            matched_seeds = bool(len(dqn_rnd) == len(ppo_rnd) and (dqn_rnd["seed"].values == ppo_rnd["seed"].values).all())
            seed_label = f"{dqn_rnd['seed'].min()}..{dqn_rnd['seed'].max()}"
            n1, n2 = len(dqn_rnd), len(ppo_rnd)
            w1, w2 = int(dqn_rnd["agent_won"].sum()), int(ppo_rnd["agent_won"].sum())
            p1, p2 = w1 / n1, w2 / n2

            # Win Rate Tests
            # 1a. Two-proportion z-test (independent)
            z_stat, p_z = run_two_proportion_z_test(w1, n1, w2, n2)
            test_rows.append({
                "experiment": "benchmark_evaluation",
                "seed": seed_label,
                "comparison_category": "algorithm_contrast",
                "group1_algorithm": "dqn",
                "group1_opponent": "random",
                "group2_algorithm": "ppo",
                "group2_opponent": "random",
                "metric": "win_rate",
                "test_name": "Two-Proportion z-Test",
                "test_type": "independent",
                "h0_statement": "Win rate of DQN vs Random equals win rate of PPO vs Random",
                "h1_statement": "Win rate of DQN vs Random differs from PPO vs Random",
                "group1_value": round(p1, 4),
                "group2_value": round(p2, 4),
                "measured_difference": round(p1 - p2, 4),
                "test_statistic": round(z_stat, 4),
                "p_value": float(p_z),
                "df": None,
                "alpha": 0.05,
                "significant_alpha_05": bool(p_z < 0.05),
                "significant_alpha_01": bool(p_z < 0.01),
                "statistical_conclusion": "Reject H0 (statistically significant difference)" if p_z < 0.05 else "Fail to reject H0 (no statistically significant difference detected at alpha=0.05)",
            })

            # 1b. Fisher's Exact Test
            contingency = [[w1, n1 - w1], [w2, n2 - w2]]
            odds_r, p_fisher = stats.fisher_exact(contingency)
            test_rows.append({
                "experiment": "benchmark_evaluation",
                "seed": seed_label,
                "comparison_category": "algorithm_contrast",
                "group1_algorithm": "dqn",
                "group1_opponent": "random",
                "group2_algorithm": "ppo",
                "group2_opponent": "random",
                "metric": "win_rate",
                "test_name": "Fisher's Exact Test",
                "test_type": "independent",
                "h0_statement": "Odds of winning are independent of algorithm (DQN vs PPO)",
                "h1_statement": "Odds of winning depend on algorithm (DQN vs PPO)",
                "group1_value": round(p1, 4),
                "group2_value": round(p2, 4),
                "measured_difference": round(p1 - p2, 4),
                "test_statistic": round(float(odds_r), 4),
                "p_value": float(p_fisher),
                "df": None,
                "alpha": 0.05,
                "significant_alpha_05": bool(p_fisher < 0.05),
                "significant_alpha_01": bool(p_fisher < 0.01),
                "statistical_conclusion": "Reject H0" if p_fisher < 0.05 else "Fail to reject H0 (no association detected)",
            })

            # 1c. McNemar Paired Test (controlling for identical game seeds)
            if matched_seeds:
                mcnemar_res = run_mcnemar_test(dqn_rnd["agent_won"].tolist(), ppo_rnd["agent_won"].tolist())
                test_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": "win_rate",
                    "test_name": f"McNemar Test ({mcnemar_res['method']})",
                    "test_type": "paired_matched_seeds",
                    "h0_statement": "Marginal probabilities of winning on identical seeds are equal (DQN = PPO)",
                    "h1_statement": "Marginal probabilities of winning on identical seeds differ",
                    "group1_value": round(p1, 4),
                    "group2_value": round(p2, 4),
                    "measured_difference": round(p1 - p2, 4),
                    "test_statistic": round(mcnemar_res["statistic"], 4),
                    "p_value": float(mcnemar_res["p_value"]),
                    "df": 1,
                    "alpha": 0.05,
                    "significant_alpha_05": bool(mcnemar_res["p_value"] < 0.05),
                    "significant_alpha_01": bool(mcnemar_res["p_value"] < 0.01),
                    "statistical_conclusion": "Reject H0" if mcnemar_res["p_value"] < 0.05 else "Fail to reject H0 (no paired divergence detected)",
                })

            # Win Rate Effect Sizes
            prop_effects = compute_proportion_effect_sizes(w1, n1, w2, n2)
            for eff_name, eff_val, low_ci, high_ci in [
                ("Risk Difference (Win Rate Diff)", prop_effects["risk_difference"], prop_effects["risk_diff_ci_lower"], prop_effects["risk_diff_ci_upper"]),
                ("Odds Ratio", prop_effects["odds_ratio"], prop_effects["odds_ratio_ci_lower"], prop_effects["odds_ratio_ci_upper"]),
            ]:
                effect_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": "win_rate",
                    "effect_measure": eff_name,
                    "point_estimate": round(eff_val, 4),
                    "ci_95_lower": round(low_ci, 4),
                    "ci_95_upper": round(high_ci, 4),
                    "interpretation": "contains 0 (not statistically distinguishable from 0)" if eff_name.startswith("Risk") and low_ci <= 0 <= high_ci else ("contains 1 (not distinguishable from 1.0)" if eff_name.startswith("Odds") and low_ci <= 1.0 <= high_ci else "statistically distinguishable effect"),
                })

            # Continuous metrics comparison: agent_reward, turns, agent_captures, agent_tokens_home
            for metric in ["agent_reward", "turns", "agent_captures", "agent_tokens_home"]:
                x = dqn_rnd[metric].values
                y = ppo_rnd[metric].values
                mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
                diff = mean_x - mean_y

                # Welch's t-test (independent, unequal variance)
                t_welch, p_welch = stats.ttest_ind(x, y, equal_var=False)
                test_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": metric,
                    "test_name": "Welch's Two-Sample t-Test",
                    "test_type": "independent",
                    "h0_statement": f"Mean {metric} is identical between DQN and PPO",
                    "h1_statement": f"Mean {metric} differs between DQN and PPO",
                    "group1_value": round(mean_x, 4),
                    "group2_value": round(mean_y, 4),
                    "measured_difference": round(diff, 4),
                    "test_statistic": round(float(t_welch), 4),
                    "p_value": float(p_welch),
                    "df": round(float(((np.var(x)/n1 + np.var(y)/n2)**2) / ((np.var(x)/n1)**2/(n1-1) + (np.var(y)/n2)**2/(n2-1))), 2),
                    "alpha": 0.05,
                    "significant_alpha_05": bool(p_welch < 0.05),
                    "significant_alpha_01": bool(p_welch < 0.01),
                    "statistical_conclusion": "Reject H0" if p_welch < 0.05 else "Fail to reject H0",
                })

                # Paired t-test across matched seeds
                if matched_seeds:
                    t_paired, p_paired = stats.ttest_rel(x, y)
                    test_rows.append({
                        "experiment": "benchmark_evaluation",
                        "seed": seed_label,
                        "comparison_category": "algorithm_contrast",
                        "group1_algorithm": "dqn",
                        "group1_opponent": "random",
                        "group2_algorithm": "ppo",
                        "group2_opponent": "random",
                        "metric": metric,
                        "test_name": "Paired Student's t-Test",
                        "test_type": "paired_matched_seeds",
                        "h0_statement": f"Mean difference of {metric} across identical seeds is zero",
                        "h1_statement": f"Mean difference of {metric} across identical seeds is non-zero",
                        "group1_value": round(mean_x, 4),
                        "group2_value": round(mean_y, 4),
                        "measured_difference": round(diff, 4),
                        "test_statistic": round(float(t_paired), 4),
                        "p_value": float(p_paired),
                        "df": n1 - 1,
                        "alpha": 0.05,
                        "significant_alpha_05": bool(p_paired < 0.05),
                        "significant_alpha_01": bool(p_paired < 0.01),
                        "statistical_conclusion": "Reject H0" if p_paired < 0.05 else "Fail to reject H0",
                    })

                # Mann-Whitney U test (non-parametric independent)
                u_stat, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
                test_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": metric,
                    "test_name": "Mann-Whitney U Test",
                    "test_type": "independent_rank_sum",
                    "h0_statement": f"Distribution of {metric} is identical between DQN and PPO",
                    "h1_statement": f"Distribution of {metric} differs between DQN and PPO",
                    "group1_value": round(mean_x, 4),
                    "group2_value": round(mean_y, 4),
                    "measured_difference": round(diff, 4),
                    "test_statistic": round(float(u_stat), 4),
                    "p_value": float(p_u),
                    "df": None,
                    "alpha": 0.05,
                    "significant_alpha_05": bool(p_u < 0.05),
                    "significant_alpha_01": bool(p_u < 0.01),
                    "statistical_conclusion": "Reject H0" if p_u < 0.05 else "Fail to reject H0",
                })

                # Wilcoxon signed-rank test (non-parametric paired)
                if matched_seeds and not np.all(x == y):
                    diffs = x - y
                    if np.any(diffs != 0):
                        w_stat, p_w = stats.wilcoxon(diffs)
                        test_rows.append({
                            "experiment": "benchmark_evaluation",
                            "seed": seed_label,
                            "comparison_category": "algorithm_contrast",
                            "group1_algorithm": "dqn",
                            "group1_opponent": "random",
                            "group2_algorithm": "ppo",
                            "group2_opponent": "random",
                            "metric": metric,
                            "test_name": "Wilcoxon Signed-Rank Test",
                            "test_type": "paired_rank_sum",
                            "h0_statement": f"Median difference in {metric} on identical seeds is zero",
                            "h1_statement": f"Median difference in {metric} on identical seeds is non-zero",
                            "group1_value": round(mean_x, 4),
                            "group2_value": round(mean_y, 4),
                            "measured_difference": round(diff, 4),
                            "test_statistic": round(float(w_stat), 4),
                            "p_value": float(p_w),
                            "df": None,
                            "alpha": 0.05,
                            "significant_alpha_05": bool(p_w < 0.05),
                            "significant_alpha_01": bool(p_w < 0.01),
                            "statistical_conclusion": "Reject H0" if p_w < 0.05 else "Fail to reject H0",
                        })

                # Continuous Effect Sizes: Cohen's d and Hedges' g
                d_res = compute_cohens_d(x.tolist(), y.tolist())
                effect_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": metric,
                    "effect_measure": "Cohen's d",
                    "point_estimate": round(d_res["cohens_d"], 4),
                    "ci_95_lower": round(d_res["d_ci_lower"], 4),
                    "ci_95_upper": round(d_res["d_ci_upper"], 4),
                    "interpretation": interpret_cohens_d(d_res["cohens_d"]),
                })
                effect_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "algorithm_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": "random",
                    "group2_algorithm": "ppo",
                    "group2_opponent": "random",
                    "metric": metric,
                    "effect_measure": "Hedges' g",
                    "point_estimate": round(d_res["hedges_g"], 4),
                    "ci_95_lower": round(d_res["d_ci_lower"] * (d_res["hedges_g"] / d_res["cohens_d"]) if d_res["cohens_d"] != 0 else 0.0, 4),
                    "ci_95_upper": round(d_res["d_ci_upper"] * (d_res["hedges_g"] / d_res["cohens_d"]) if d_res["cohens_d"] != 0 else 0.0, 4),
                    "interpretation": interpret_cohens_d(d_res["hedges_g"]),
                })

        # ----------------------------------------------------------------------
        # 2. Opponent Comparisons for DQN: Random vs Expert vs Self-Play
        # ----------------------------------------------------------------------
        dqn_exp = self.df_games[(self.df_games["algorithm"] == "dqn") & (self.df_games["opponent"] == "expert")].sort_values("seed")
        dqn_slf = self.df_games[(self.df_games["algorithm"] == "dqn") & (self.df_games["opponent"] == "self_play")].sort_values("seed")

        dqn_opp_pairs = [
            ("random", "expert", dqn_rnd, dqn_exp),
            ("random", "self_play", dqn_rnd, dqn_slf),
            ("expert", "self_play", dqn_exp, dqn_slf),
        ]

        for opp1, opp2, df1, df2 in dqn_opp_pairs:
            if df1.empty or df2.empty:
                continue

            matched = bool(len(df1) == len(df2) and (df1["seed"].values == df2["seed"].values).all())
            seed_label = f"{df1['seed'].min()}..{df1['seed'].max()}"
            w1, w2 = int(df1["agent_won"].sum()), int(df2["agent_won"].sum())
            n1, n2 = len(df1), len(df2)
            p1, p2 = w1 / n1, w2 / n2

            # Win rate comparison
            z_stat, p_z = run_two_proportion_z_test(w1, n1, w2, n2)
            test_rows.append({
                "experiment": "benchmark_evaluation",
                "seed": seed_label,
                "comparison_category": "opponent_contrast",
                "group1_algorithm": "dqn",
                "group1_opponent": opp1,
                "group2_algorithm": "dqn",
                "group2_opponent": opp2,
                "metric": "win_rate",
                "test_name": "Two-Proportion z-Test",
                "test_type": "independent",
                "h0_statement": f"DQN win rate vs {opp1} equals DQN win rate vs {opp2}",
                "h1_statement": f"DQN win rate vs {opp1} differs from DQN win rate vs {opp2}",
                "group1_value": round(p1, 4),
                "group2_value": round(p2, 4),
                "measured_difference": round(p1 - p2, 4),
                "test_statistic": round(z_stat, 4),
                "p_value": float(p_z),
                "df": None,
                "alpha": 0.05,
                "significant_alpha_05": bool(p_z < 0.05),
                "significant_alpha_01": bool(p_z < 0.01),
                "statistical_conclusion": "Reject H0 (significant divergence across opponent difficulty)" if p_z < 0.05 else "Fail to reject H0",
            })

            if matched:
                mcn = run_mcnemar_test(df1["agent_won"].tolist(), df2["agent_won"].tolist())
                test_rows.append({
                    "experiment": "benchmark_evaluation",
                    "seed": seed_label,
                    "comparison_category": "opponent_contrast",
                    "group1_algorithm": "dqn",
                    "group1_opponent": opp1,
                    "group2_algorithm": "dqn",
                    "group2_opponent": opp2,
                    "metric": "win_rate",
                    "test_name": f"McNemar Test ({mcn['method']})",
                    "test_type": "paired_matched_seeds",
                    "h0_statement": f"Marginal win probabilities on identical seeds are equal between {opp1} and {opp2}",
                    "h1_statement": f"Marginal win probabilities differ between {opp1} and {opp2}",
                    "group1_value": round(p1, 4),
                    "group2_value": round(p2, 4),
                    "measured_difference": round(p1 - p2, 4),
                    "test_statistic": round(mcn["statistic"], 4),
                    "p_value": float(mcn["p_value"]),
                    "df": 1,
                    "alpha": 0.05,
                    "significant_alpha_05": bool(mcn["p_value"] < 0.05),
                    "significant_alpha_01": bool(mcn["p_value"] < 0.01),
                    "statistical_conclusion": "Reject H0" if mcn["p_value"] < 0.05 else "Fail to reject H0",
                })

            # Effect size for opponent contrast
            d_res = compute_cohens_d(df1["agent_reward"].values.tolist(), df2["agent_reward"].values.tolist())
            effect_rows.append({
                "experiment": "benchmark_evaluation",
                "seed": seed_label,
                "comparison_category": "opponent_contrast",
                "group1_algorithm": "dqn",
                "group1_opponent": opp1,
                "group2_algorithm": "dqn",
                "group2_opponent": opp2,
                "metric": "agent_reward",
                "effect_measure": "Cohen's d",
                "point_estimate": round(d_res["cohens_d"], 4),
                "ci_95_lower": round(d_res["d_ci_lower"], 4),
                "ci_95_upper": round(d_res["d_ci_upper"], 4),
                "interpretation": interpret_cohens_d(d_res["cohens_d"]),
            })

            # Turns comparison
            d_turns = compute_cohens_d(df1["turns"].values.tolist(), df2["turns"].values.tolist())
            effect_rows.append({
                "experiment": "benchmark_evaluation",
                "seed": seed_label,
                "comparison_category": "opponent_contrast",
                "group1_algorithm": "dqn",
                "group1_opponent": opp1,
                "group2_algorithm": "dqn",
                "group2_opponent": opp2,
                "metric": "turns",
                "effect_measure": "Cohen's d",
                "point_estimate": round(d_turns["cohens_d"], 4),
                "ci_95_lower": round(d_turns["d_ci_lower"], 4),
                "ci_95_upper": round(d_turns["d_ci_upper"], 4),
                "interpretation": interpret_cohens_d(d_turns["cohens_d"]),
            })

        # ----------------------------------------------------------------------
        # 3. Seat Fairness Tests (Player 0 vs Player 1)
        # ----------------------------------------------------------------------
        grouped_conds = self.df_games.groupby(["experiment", "algorithm", "opponent"])
        for (exp, algo, opp), grp in grouped_conds:
            s0 = grp[grp["agent_seat"] == 0]
            s1 = grp[grp["agent_seat"] == 1]
            if s0.empty or s1.empty:
                continue

            seed_label = f"{grp['seed'].min()}..{grp['seed'].max()}"
            w0, n0 = int(s0["agent_won"].sum()), len(s0)
            w1, n1 = int(s1["agent_won"].sum()), len(s1)
            p0, p1 = w0 / n0, w1 / n1

            # Fisher's Exact test on Seat 0 vs Seat 1
            tbl = [[w0, n0 - w0], [w1, n1 - w1]]
            odds_r, p_fisher = stats.fisher_exact(tbl)

            # Two-proportion z-test on Seat 0 vs Seat 1
            z_seat, p_z_seat = run_two_proportion_z_test(w0, n0, w1, n1)

            test_rows.append({
                "experiment": exp,
                "seed": seed_label,
                "comparison_category": "seat_fairness_test",
                "group1_algorithm": f"{algo}_seat0",
                "group1_opponent": opp,
                "group2_algorithm": f"{algo}_seat1",
                "group2_opponent": opp,
                "metric": "win_rate",
                "test_name": "Fisher's Exact Test",
                "test_type": "seat_permutation",
                "h0_statement": f"Seat 0 win rate equals Seat 1 win rate for {algo} vs {opp}",
                "h1_statement": f"Seat 0 win rate differs from Seat 1 win rate for {algo} vs {opp}",
                "group1_value": round(p0, 4),
                "group2_value": round(p1, 4),
                "measured_difference": round(p0 - p1, 4),
                "test_statistic": round(float(odds_r), 4),
                "p_value": float(p_fisher),
                "df": None,
                "alpha": 0.05,
                "significant_alpha_05": bool(p_fisher < 0.05),
                "significant_alpha_01": bool(p_fisher < 0.01),
                "statistical_conclusion": "First-mover advantage detected (p < 0.05)" if p_fisher < 0.05 else "Seat neutrality verified (fail to reject H0; no significant seat advantage)",
            })

            # Continuous reward seat difference (Welch t-test)
            r0 = s0["agent_reward"].values
            r1 = s1["agent_reward"].values
            t_seat_r, p_seat_r = stats.ttest_ind(r0, r1, equal_var=False)

            test_rows.append({
                "experiment": exp,
                "seed": seed_label,
                "comparison_category": "seat_fairness_test",
                "group1_algorithm": f"{algo}_seat0",
                "group1_opponent": opp,
                "group2_algorithm": f"{algo}_seat1",
                "group2_opponent": opp,
                "metric": "agent_reward",
                "test_name": "Welch's t-Test (Seat Reward)",
                "test_type": "seat_permutation",
                "h0_statement": f"Mean episodic reward in Seat 0 equals Seat 1 for {algo} vs {opp}",
                "h1_statement": f"Mean episodic reward differs across seats for {algo} vs {opp}",
                "group1_value": round(float(np.mean(r0)), 4),
                "group2_value": round(float(np.mean(r1)), 4),
                "measured_difference": round(float(np.mean(r0) - np.mean(r1)), 4),
                "test_statistic": round(float(t_seat_r), 4),
                "p_value": float(p_seat_r),
                "df": round(float(len(r0) + len(r1) - 2), 1),
                "alpha": 0.05,
                "significant_alpha_05": bool(p_seat_r < 0.05),
                "significant_alpha_01": bool(p_seat_r < 0.01),
                "statistical_conclusion": "Significant seat reward difference" if p_seat_r < 0.05 else "Fair seat reward balance (p >= 0.05)",
            })

        return pd.DataFrame(test_rows), pd.DataFrame(effect_rows)

    def generate_publication_tables(
        self,
        df_desc: pd.DataFrame,
        df_norm: pd.DataFrame,
        df_tests: pd.DataFrame,
        df_effects: pd.DataFrame,
        df_seed: pd.DataFrame,
    ) -> Dict[str, str]:
        """Format publication-ready tables in CSV, Markdown, and LaTeX."""
        saved_files: Dict[str, str] = {}

        # ----------------------------------------------------------------------
        # Table 1: Master Descriptive Summary
        # ----------------------------------------------------------------------
        t1_rows = []
        for _, r in df_desc[df_desc["metric"].isin(["win_rate", "agent_reward", "turns", "agent_captures", "agent_tokens_home"])].iterrows():
            t1_rows.append({
                "Algorithm": r["algorithm"].upper(),
                "Opponent": r["opponent"].upper(),
                "Metric": r["metric"],
                "N": r["n"],
                "Mean": f"{r['mean']:.4f}",
                "StdDev": f"{r['std']:.4f}",
                "Variance": f"{r['variance']:.4f}",
                "Median (IQR)": f"{r['median']:.2f} ({r['iqr']:.2f})",
                "95% CI": f"[{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}]",
                "CI Method": r["ci_method"],
            })
        df_t1 = pd.DataFrame(t1_rows)
        self._write_table_formats(df_t1, "table1_descriptive_summary", "Table 1: Master Descriptive Statistics across Stored Evaluations", saved_files)

        # ----------------------------------------------------------------------
        # Table 2: Win Rates and 95% Confidence Intervals
        # ----------------------------------------------------------------------
        t2_rows = []
        for _, r in df_desc[df_desc["metric"] == "win_rate"].iterrows():
            t2_rows.append({
                "Algorithm": r["algorithm"].upper(),
                "Opponent": r["opponent"].upper(),
                "Games (N)": r["n"],
                "Win Rate (%)": f"{r['mean'] * 100.0:.2f}%",
                "Bernoulli Variance": f"{r['variance']:.4f}",
                "SEM (%)": f"{r['sem'] * 100.0:.2f}%",
                "95% Wilson Score CI": f"[{r['ci_95_lower'] * 100.0:.2f}%, {r['ci_95_upper'] * 100.0:.2f}%]",
                "95% Bootstrap CI": f"[{r['ci_bootstrap_lower'] * 100.0:.2f}%, {r['ci_bootstrap_upper'] * 100.0:.2f}%]",
            })
        df_t2 = pd.DataFrame(t2_rows)
        self._write_table_formats(df_t2, "table2_win_rate_and_confidence_intervals", "Table 2: Win Rates and Rigorous Confidence Intervals (Wilson vs Bootstrap 10k)", saved_files)

        # ----------------------------------------------------------------------
        # Table 3: Comparative Hypothesis Testing Suite
        # ----------------------------------------------------------------------
        t3_rows = []
        for _, r in df_tests.iterrows():
            p_val = r["p_value"]
            p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
            t3_rows.append({
                "Comparison Category": r["comparison_category"],
                "Group 1": f"{r['group1_algorithm'].upper()} ({r['group1_opponent']})",
                "Group 2": f"{r['group2_algorithm'].upper()} ({r['group2_opponent']})",
                "Metric": r["metric"],
                "Test Name": r["test_name"],
                "Design Type": r["test_type"],
                "Observed Diff": f"{r['measured_difference']:+.4f}",
                "Test Stat": f"{r['test_statistic']:.4f}",
                "p-Value": p_str,
                "Sig (alpha=0.05)": "Yes (*)" if r["significant_alpha_05"] else "No (ns)",
                "Conclusion": r["statistical_conclusion"],
            })
        df_t3 = pd.DataFrame(t3_rows)
        self._write_table_formats(df_t3, "table3_comparative_hypothesis_tests", "Table 3: Parametric & Non-Parametric Hypothesis Tests", saved_files)

        # ----------------------------------------------------------------------
        # Table 4: Effect Sizes and Point Estimates
        # ----------------------------------------------------------------------
        t4_rows = []
        for _, r in df_effects.iterrows():
            t4_rows.append({
                "Comparison Category": r["comparison_category"],
                "Group 1": f"{r['group1_algorithm'].upper()} ({r['group1_opponent']})",
                "Group 2": f"{r['group2_algorithm'].upper()} ({r['group2_opponent']})",
                "Metric": r["metric"],
                "Effect Measure": r["effect_measure"],
                "Point Estimate": f"{r['point_estimate']:+.4f}",
                "95% CI Lower": f"{r['ci_95_lower']:+.4f}",
                "95% CI Upper": f"{r['ci_95_upper']:+.4f}",
                "Interpretation": r["interpretation"],
            })
        df_t4 = pd.DataFrame(t4_rows)
        self._write_table_formats(df_t4, "table4_effect_sizes", "Table 4: Measured Effect Sizes and Standardized Magnitudes", saved_files)

        # ----------------------------------------------------------------------
        # Table 5: Seat Fairness Analysis
        # ----------------------------------------------------------------------
        t5_rows = []
        for _, r in df_tests[df_tests["comparison_category"] == "seat_fairness_test"].iterrows():
            p_val = r["p_value"]
            p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
            t5_rows.append({
                "Algorithm": r["group1_algorithm"].replace("_seat0", "").upper(),
                "Opponent": r["group1_opponent"].upper(),
                "Metric": r["metric"],
                "Seat 0 Value": f"{r['group1_value']:.4f}",
                "Seat 1 Value": f"{r['group2_value']:.4f}",
                "Seat Bias (P0 - P1)": f"{r['measured_difference']:+.4f}",
                "Test Statistic": f"{r['test_statistic']:.4f}",
                "p-Value": p_str,
                "Fair at alpha=0.05": "Yes (Neutral)" if not r["significant_alpha_05"] else "No (Biased)",
            })
        df_t5 = pd.DataFrame(t5_rows)
        self._write_table_formats(df_t5, "table5_seat_fairness_analysis", "Table 5: First-Mover / Seat Permutation Fairness Evaluation", saved_files)

        # ----------------------------------------------------------------------
        # Table 6: Distributional Diagnostics & Normality Tests
        # ----------------------------------------------------------------------
        t6_rows = []
        for _, r in df_norm.iterrows():
            p_str = f"{r['shapiro_wilk_pvalue']:.4e}" if r['shapiro_wilk_pvalue'] < 0.0001 else f"{r['shapiro_wilk_pvalue']:.4f}"
            t6_rows.append({
                "Algorithm": r["algorithm"].upper(),
                "Opponent": r["opponent"].upper(),
                "Metric": r["metric"],
                "N": r["n"],
                "Skewness": f"{r['skewness']:+.4f}",
                "Kurtosis": f"{r['kurtosis']:+.4f}",
                "Shapiro-Wilk W": f"{r['shapiro_wilk_stat']:.4f}",
                "Shapiro-Wilk p": p_str,
                "Normal (alpha=0.05)": "Yes" if r["is_normal_alpha_05"] else "No (Non-normal)",
                "Method Justification": r["statistical_justification"],
            })
        df_t6 = pd.DataFrame(t6_rows)
        self._write_table_formats(df_t6, "table6_distributional_normality_tests", "Table 6: Distributional Characteristics & Normality Testing", saved_files)

        # ----------------------------------------------------------------------
        # Table 7: Seed Block Robustness Summary
        # ----------------------------------------------------------------------
        if not df_seed.empty:
            t7_rows = []
            for _, r in df_seed[df_seed["metric"].isin(["win_rate", "agent_reward"])].iterrows():
                t7_rows.append({
                    "Algorithm": r["algorithm"].upper(),
                    "Opponent": r["opponent"].upper(),
                    "Seed Range": r["seed"],
                    "Seed Block": r["seed_block"],
                    "Metric": r["metric"],
                    "N": r["count"],
                    "Mean": f"{r['mean']:.4f}",
                    "Variance": f"{r['variance']:.4f}",
                })
            df_t7 = pd.DataFrame(t7_rows)
            self._write_table_formats(df_t7, "table7_seed_block_summary", "Table 7: Empirical Seed Block Invariance Breakdown", saved_files)

        return saved_files

    def _write_table_formats(
        self,
        df: pd.DataFrame,
        base_name: str,
        caption: str,
        saved_registry: Dict[str, str],
    ) -> None:
        """Write DataFrame as CSV, GitHub Markdown, and publication LaTeX."""
        if df.empty:
            return

        # 1. CSV
        csv_path = self.tables_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        saved_registry[f"{base_name}_csv"] = str(csv_path)

        # 2. Markdown
        md_path = self.tables_dir / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"### {caption}\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n*Source: Experimental results stored in results/raw/ adhering to Final Experiment Protocol Section 20.*\n")
        saved_registry[f"{base_name}_md"] = str(md_path)

        # 3. LaTeX table
        tex_path = self.tables_dir / f"{base_name}.tex"
        latex_str = df.to_latex(
            index=False,
            caption=caption,
            label=f"tab:{base_name}",
            escape=True,
            column_format="l" * len(df.columns),
        )
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        saved_registry[f"{base_name}_tex"] = str(tex_path)

    def run(self) -> Dict[str, Any]:
        """Execute full statistical analysis pipeline."""
        print("=" * 80)
        print("PHASE 17: STATISTICAL ANALYSIS PIPELINE")
        print(f"Reading directly from: {[str(d) for d in self.raw_dirs]}")
        print(f"Destination directories: {self.processed_dir} and {self.tables_dir}")
        print("=" * 80)

        # 1. Ingest raw game data
        df_games = self.ingest_stored_results()
        if df_games.empty:
            raise RuntimeError(f"No valid game records found in {[str(d) for d in self.raw_dirs]}")

        print(f"[1/5] Ingested {len(df_games)} stored game records across algorithms: {df_games['algorithm'].unique().tolist()}")

        # Save consolidated games CSV to results/processed/
        games_out = self.processed_dir / "consolidated_game_records.csv"
        df_games.to_csv(games_out, index=False, encoding="utf-8")
        print(f"      Saved: {games_out.name}")

        # 2. Normality tests
        print("[2/5] Running normality and distributional diagnostics...")
        df_norm = self.compute_normality_diagnostics()
        norm_out = self.processed_dir / "distributional_normality_tests.csv"
        df_norm.to_csv(norm_out, index=False, encoding="utf-8")

        # 3. Descriptive statistics
        print("[3/5] Computing descriptive statistics and confidence intervals...")
        df_desc = self.compute_descriptive_statistics()
        desc_csv = self.processed_dir / "descriptive_statistics.csv"
        desc_json = self.processed_dir / "descriptive_statistics.json"
        df_desc.to_csv(desc_csv, index=False, encoding="utf-8")
        with open(desc_json, "w", encoding="utf-8") as f:
            json.dump(df_desc.to_dict(orient="records"), f, indent=2)

        # Seed level statistics
        df_seed = self.compute_seed_level_statistics()
        seed_out = self.processed_dir / "seed_level_statistics.csv"
        df_seed.to_csv(seed_out, index=False, encoding="utf-8")

        # 4. Hypothesis tests and effect sizes
        print("[4/5] Executing hypothesis tests and effect size calculations...")
        df_tests, df_effects = self.compute_hypothesis_tests()
        tests_csv = self.processed_dir / "hypothesis_tests.csv"
        tests_json = self.processed_dir / "hypothesis_tests.json"
        df_tests.to_csv(tests_csv, index=False, encoding="utf-8")
        with open(tests_json, "w", encoding="utf-8") as f:
            json.dump(df_tests.to_dict(orient="records"), f, indent=2)

        effects_csv = self.processed_dir / "effect_sizes.csv"
        effects_json = self.processed_dir / "effect_sizes.json"
        df_effects.to_csv(effects_csv, index=False, encoding="utf-8")
        with open(effects_json, "w", encoding="utf-8") as f:
            json.dump(df_effects.to_dict(orient="records"), f, indent=2)

        # 5. Publication Tables
        print("[5/5] Generating publication-ready tables in results/tables/...")
        table_files = self.generate_publication_tables(df_desc, df_norm, df_tests, df_effects, df_seed)

        print("=" * 80)
        print("PHASE 17 STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"Generated {len(df_games)} processed records, {len(df_tests)} hypothesis tests, {len(df_effects)} effect sizes.")
        print(f"Outputs saved to: {self.processed_dir} and {self.tables_dir}")
        print("=" * 80)

        return {
            "num_games": len(df_games),
            "num_tests": len(df_tests),
            "num_effects": len(df_effects),
            "table_files": table_files,
        }
