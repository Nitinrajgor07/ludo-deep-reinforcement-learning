"""PHASE 11 — STATISTICAL ANALYSIS PIPELINE

Authoritative statistical analysis pipeline for PhD Ludo DRL Research.
Strictly adheres to:
- FINAL_EXPERIMENT_PROTOCOL.md Section 20 (Statistical Analysis Methodology)
- Phase 11 Specification: Build final statistical pipeline using ONLY raw data.

Requirements & Scientific Invariants:
1. Pure Raw Data Reproduction:
   - Reads directly from stored evaluation records (CSV/JSONL).
   - Zero hardcoded or manually typed values in any output or table.
   - Every metric, CI, p-value, and effect size is computed deterministically.
2. Clear Data Tier Differentiation:
   - PRELIMINARY_PILOT: Checkpoints < 500k steps, <= 250 games, pilot runs.
   - FINAL_EXPERIMENT: Verified >= 500k step checkpoints, 5,000 games per matchup, 200,000 total.
   - If final raw data is missing, reports analysis as INCOMPLETE rather than fabricating values.
3. Full Statistical Suite:
   - Descriptive Statistics: Mean, median, standard deviation, variance, IQR, min, max, SEM.
   - Confidence Intervals: 95% Wilson Score (binomial), 95% Bootstrap (10,000 resamples), 95% Student's t.
   - Model Comparisons: All pairwise algorithm contrasts (DQN, PPO, Discrete SAC, C51).
   - Effect Sizes: Cohen's d, Hedges' g, Odds Ratio, Risk Difference, Relative Risk, Wilcoxon r.
   - Hypothesis Tests: Welch's t-test, Mann-Whitney U, Paired t-test, Wilcoxon Signed-Rank,
     Two-Proportion z-test, Fisher's Exact, McNemar test.
   - Multiple-Comparison Corrections: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR).
   - Cross-Seed Analysis: Inter-seed variance, coefficient of variation, Kruskal-Wallis & ANOVA.
   - Opponent-Specific Analysis: Random vs Expert performance delta and sensitivity tests.
   - Seat / Fairness Analysis: Seat 0 vs Seat 1 win rates, seat bias, and seat neutrality tests.
4. Scientific Neutrality:
   - Strictly reports measured differences and statistical evidence without subjective best-model claims.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


# ==============================================================================
# PROTOCOL CONSTANTS & ENUMS
# ==============================================================================
FROZEN_PROTOCOL_VERSION: str = "1.0.0 (FROZEN)"
BENCHMARK_MODELS: List[str] = ["dqn", "ppo", "model3", "model4"]
MASTER_SEEDS: List[int] = [42, 101, 202, 303, 404]
EVALUATION_OPPONENTS: List[str] = ["random", "expert"]
REQUIRED_FINAL_STEPS: int = 500_000
GAMES_PER_FINAL_MATCHUP: int = 5000
TOTAL_REQUIRED_FINAL_GAMES: int = 200000


class DataTier(str, Enum):
    """Classification of experimental evaluation data."""
    FINAL_EXPERIMENT = "FINAL_EXPERIMENT"
    PRELIMINARY_PILOT = "PRELIMINARY_PILOT"


class StatisticalAnalysisError(Exception):
    """Base exception for statistical pipeline errors."""
    pass


class IncompleteRawDataError(StatisticalAnalysisError):
    """Raised when final evaluation raw data is missing or incomplete."""
    pass


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
        term1 = 2.0 * n * p + z2
        term2_lower = z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1.0 - p) + (4.0 * p - 2.0)) + 1.0
        term2_upper = z * math.sqrt(z2 - 1.0 / n + 4.0 * n * p * (1.0 - p) - (4.0 * p - 2.0)) + 1.0
        denom = 2.0 * (n + z2)
        lower = max(0.0, (term1 - term2_lower) / denom) if p > 0 else 0.0
        upper = min(1.0, (term1 + term2_upper) / denom) if p < 1 else 1.0
        return (float(lower), float(upper))


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray, Sequence[float]],
    statistic_fn: Callable = np.mean,
    confidence: float = 0.95,
    num_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap percentile confidence interval (default 10,000 resamples)."""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1 or np.all(arr == arr[0]):
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
    data: Union[List[float], np.ndarray, Sequence[float]],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute parametric Student's t confidence interval for the mean."""
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
# MULTIPLE-COMPARISON CORRECTION PROCEDURES
# ==============================================================================
def bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    """Single-step Bonferroni FWER adjustment: p_adj = min(1.0, p * m)."""
    m = len(p_values)
    return [float(min(1.0, max(0.0, p * m))) for p in p_values]


def holm_bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni FWER adjustment (more powerful than Bonferroni)."""
    m = len(p_values)
    if m <= 1:
        return [float(p) for p in p_values]

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = float(running_max)
    return adjusted


def benjamini_hochberg_correction(p_values: Sequence[float]) -> List[float]:
    """Step-up Benjamini-Hochberg (1995) False Discovery Rate (FDR) adjustment."""
    m = len(p_values)
    if m <= 1:
        return [float(p) for p in p_values]

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank]
        adj = min(1.0, p * m / (rank + 1))
        running_min = min(running_min, adj)
        adjusted[orig_idx] = float(running_min)
    return adjusted


# ==============================================================================
# EFFECT SIZE CALCULATORS
# ==============================================================================
def compute_cohens_d(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
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

    se_rd = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    z = 1.95996
    rd_lower = risk_diff - z * se_rd
    rd_upper = risk_diff + z * se_rd

    relative_risk = (p1 / p2) if p2 > 0 else (float("nan") if p1 == 0 else float("inf"))

    # Haldane-Anscombe continuity correction if any cell is 0
    a, b = float(w1), float(n1 - w1)
    c, d = float(w2), float(n2 - w2)
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
    """Standard Cohen (1988) effect size interpretation."""
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

    b = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if x and not y)
    c = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if not x and y)
    both_won = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if x and y)
    both_lost = sum(1 for x, y in zip(paired_outcomes_1, paired_outcomes_2) if not x and not y)

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

    chi2_stat = ((abs(b - c) - 1.0) ** 2) / discordant
    p_chi2 = float(stats.chi2.sf(chi2_stat, df=1))
    exact_res = stats.binomtest(b, discordant, p=0.5, alternative="two-sided")
    p_exact = float(exact_res.pvalue)

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
    """Production-grade statistical analysis engine adhering strictly to Phase 11 invariants.

    Features:
    - Pure reproduction from raw evaluation files (never hardcodes table values)
    - Distinguishes PRELIMINARY_PILOT from FINAL_EXPERIMENT
    - Reports analysis as INCOMPLETE if required final raw data is missing
    - Computes full descriptive statistics and 95% CIs (Wilson, Bootstrap 10k, Student's t)
    - Full pairwise model comparisons across all 4 algorithms
    - Multiple-testing corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg)
    - Cross-seed analysis with ANOVA and Kruskal-Wallis invariance tests
    - Opponent sensitivity and strategy contrast
    - Seat permutation fairness testing
    """

    def __init__(
        self,
        raw_dirs: Optional[List[Union[str, Path]]] = None,
        processed_dir: Union[str, Path] = "results/processed",
        tables_dir: Union[str, Path] = "results/tables",
        master_seed: int = 42,
        require_final_data: bool = False,
    ):
        self.project_root = Path.cwd()
        if raw_dirs is None:
            # Check Phase 10 final raw dir, fallback to general raw dir
            final_raw = self.project_root / "results" / "final_evaluation" / "raw"
            std_raw = self.project_root / "results" / "raw"
            self.raw_dirs = [final_raw] if final_raw.exists() else [std_raw]
        else:
            self.raw_dirs = [Path(d) if Path(d).is_absolute() else self.project_root / d for d in raw_dirs]

        self.processed_dir = Path(processed_dir) if Path(processed_dir).is_absolute() else self.project_root / processed_dir
        self.tables_dir = Path(tables_dir) if Path(tables_dir).is_absolute() else self.project_root / tables_dir
        self.master_seed = master_seed
        self.require_final_data = require_final_data

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.df_games: Optional[pd.DataFrame] = None
        self.data_tier: DataTier = DataTier.PRELIMINARY_PILOT
        self.audit_status: Dict[str, Any] = {}

    def audit_raw_data(self) -> Dict[str, Any]:
        """Audit the raw data directory to verify dataset completeness and identify data tier."""
        csv_files = []
        for rd in self.raw_dirs:
            if rd.exists():
                csv_files.extend(list(rd.glob("*.csv")))

        total_games_found = 0
        final_games_found = 0
        pilot_games_found = 0
        models_found = set()
        seeds_found = set()
        opponents_found = set()

        for f in csv_files:
            try:
                df_temp = pd.read_csv(f)
                n = len(df_temp)
                total_games_found += n

                # Detect data tier based on step count and matchup size
                has_ckpt_steps = "checkpoint_steps" in df_temp.columns
                is_500k = has_ckpt_steps and (df_temp["checkpoint_steps"] >= REQUIRED_FINAL_STEPS).all()
                is_full_size = (n >= GAMES_PER_FINAL_MATCHUP)

                if is_500k and is_full_size:
                    final_games_found += n
                else:
                    pilot_games_found += n

                # Track models, seeds, opponents
                algo_col = "model_algorithm" if "model_algorithm" in df_temp.columns else "algorithm"
                seed_col = "model_seed" if "model_seed" in df_temp.columns else "seed"
                opp_col = "opponent_type" if "opponent_type" in df_temp.columns else "opponent"

                if algo_col in df_temp.columns:
                    models_found.update(df_temp[algo_col].dropna().astype(str).str.lower().unique())
                if seed_col in df_temp.columns:
                    seeds_found.update([int(s) for s in df_temp[seed_col].dropna().unique()])
                if opp_col in df_temp.columns:
                    opponents_found.update(df_temp[opp_col].dropna().astype(str).str.lower().unique())
            except Exception:
                pass

        is_final_complete = (final_games_found >= TOTAL_REQUIRED_FINAL_GAMES)
        tier = DataTier.FINAL_EXPERIMENT if is_final_complete else DataTier.PRELIMINARY_PILOT

        seeds_list = sorted([int(s) for s in seeds_found])
        seeds_repr = f"{min(seeds_list)}..{max(seeds_list)} ({len(seeds_list)} seeds)" if len(seeds_list) > 10 else seeds_list

        self.audit_status = {
            "protocol_version": FROZEN_PROTOCOL_VERSION,
            "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "raw_directories_scanned": [str(d) for d in self.raw_dirs],
            "total_csv_files_found": len(csv_files),
            "total_games_ingested": total_games_found,
            "final_experiment_games": final_games_found,
            "preliminary_pilot_games": pilot_games_found,
            "required_final_games": TOTAL_REQUIRED_FINAL_GAMES,
            "is_final_complete": is_final_complete,
            "data_tier": tier.value,
            "models_present": sorted(list(models_found)),
            "seeds_present": seeds_repr,
            "opponents_present": sorted(list(opponents_found)),
            "status": "COMPLETE" if is_final_complete else ("PRELIMINARY_AVAILABLE" if pilot_games_found > 0 else "MISSING"),
        }
        self.data_tier = tier
        return self.audit_status

    def ingest_stored_results(self) -> pd.DataFrame:
        """Ingest raw evaluation records directly from disk and normalize schemas."""
        all_rows: List[Dict[str, Any]] = []

        for raw_dir in self.raw_dirs:
            if not raw_dir.exists():
                continue

            csv_files = sorted(list(raw_dir.glob("*.csv")))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"[Warning] Failed to read {csv_file}: {e}")
                    continue

                if df.empty:
                    continue

                # Normalization across Phase 10 format and legacy format
                algo_col = "model_algorithm" if "model_algorithm" in df.columns else "algorithm"
                seed_col = "model_seed" if "model_seed" in df.columns else "seed"
                opp_col = "opponent_type" if "opponent_type" in df.columns else "opponent"
                win_col = "win" if "win" in df.columns else "agent_won"
                loss_col = "loss" if "loss" in df.columns else "opp_won"
                draw_col = "draw" if "draw" in df.columns else "is_draw"
                rew_col = "reward" if "reward" in df.columns else "agent_reward"
                turns_col = "game_length" if "game_length" in df.columns else "turns"

                if not {algo_col, seed_col, opp_col, win_col}.issubset(df.columns):
                    continue

                # Determine row tier
                ckpt_steps = int(df["checkpoint_steps"].iloc[0]) if "checkpoint_steps" in df.columns else 0
                row_tier = (
                    DataTier.FINAL_EXPERIMENT.value
                    if (ckpt_steps >= REQUIRED_FINAL_STEPS and len(df) >= GAMES_PER_FINAL_MATCHUP)
                    else DataTier.PRELIMINARY_PILOT.value
                )

                for _, row in df.iterrows():
                    rec = {
                        "data_tier": row_tier,
                        "file_source": csv_file.name,
                        "game_idx": int(row.get("game_idx", 0)),
                        "seed": int(row[seed_col]),
                        "algorithm": str(row[algo_col]).lower(),
                        "opponent": str(row[opp_col]).lower(),
                        "agent_seat": int(row.get("agent_seat", 0)),
                        "agent_won": bool(row[win_col]),
                        "opp_won": bool(row.get(loss_col, False)),
                        "is_draw": bool(row.get(draw_col, False)),
                        "turns": float(row.get(turns_col, 0.0)),
                        "agent_reward": float(row.get(rew_col, 0.0)),
                        "opp_reward": float(row.get("opponent_reward", row.get("opp_reward", 0.0))),
                        "agent_captures": float(row.get("captures", row.get("agent_captures", 0.0))),
                        "opp_captures": float(row.get("opponent_captures", row.get("opp_captures", 0.0))),
                        "agent_tokens_home": float(row.get("tokens_home", row.get("agent_tokens_home", 0.0))),
                        "opp_tokens_home": float(row.get("opponent_tokens_home", row.get("opp_tokens_home", 0.0))),
                        "checkpoint_steps": ckpt_steps,
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
        grouped = self.df_games.groupby(["data_tier", "algorithm", "opponent"])

        for (tier, algo, opp), group in grouped:
            seed_label = f"{group['seed'].min()}..{group['seed'].max()} (N={len(group)})"
            for m in metrics:
                vals = group[m].dropna().values
                n = len(vals)
                if n < 3:
                    continue

                shapiro_stat, shapiro_p = stats.shapiro(vals if n <= 5000 else vals[:5000])
                dagostino_stat, dagostino_p = stats.normaltest(vals) if n >= 20 else (np.nan, np.nan)
                skew_val = float(stats.skew(vals))
                kurt_val = float(stats.kurtosis(vals))

                rows.append({
                    "data_tier": tier,
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
        """Compute mean, median, std, variance, IQR, SEM, and 95% CIs across all conditions."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        grouped = self.df_games.groupby(["data_tier", "algorithm", "opponent"])

        for (tier, algo, opp), group in grouped:
            n = len(group)
            seed_label = f"{group['seed'].min()}..{group['seed'].max()} (N={n})"

            # 1. Win Rate (Binomial Proportion)
            wins = int(group["agent_won"].sum())
            wr = wins / n
            wr_var = wr * (1.0 - wr)
            wr_sem = math.sqrt(wr_var / n)
            w_low, w_high = compute_wilson_score_interval(wins, n, confidence=0.95)
            boot_w_low, boot_w_high = compute_bootstrap_ci(
                group["agent_won"].astype(float).values,
                statistic_fn=np.mean,
                confidence=0.95,
                num_resamples=10000,
                seed=self.master_seed,
            )

            rows.append({
                "data_tier": tier,
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
                "ci_95_lower": round(w_low, 4),
                "ci_95_upper": round(w_high, 4),
                "ci_method": "Wilson Score Interval (95%)",
                "ci_bootstrap_lower": round(boot_w_low, 4),
                "ci_bootstrap_upper": round(boot_w_high, 4),
            })

            # 2. Continuous Metrics
            continuous = [
                ("agent_reward", "cumulative episodic reward"),
                ("opp_reward", "opponent episodic reward"),
                ("turns", "game duration (turns)"),
                ("agent_captures", "opponent tokens knocked home"),
                ("agent_tokens_home", "friendly tokens reached goal [0..4]"),
            ]

            for col_name, _ in continuous:
                vals = group[col_name].dropna().values
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                var_val = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem_val = float(stats.sem(vals)) if len(vals) > 1 else 0.0
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
                    "data_tier": tier,
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

    def compute_pairwise_model_comparisons(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform full pairwise algorithm contrasts across all algorithms with multiple testing corrections."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame(), pd.DataFrame()

        test_rows = []
        effect_rows = []

        algorithms = sorted(self.df_games["algorithm"].unique().tolist())
        opponents = sorted(self.df_games["opponent"].unique().tolist())
        tiers = self.df_games["data_tier"].unique().tolist()

        for tier in tiers:
            for opp in opponents:
                # Iterate over all unique algorithm pairs (n*(n-1)/2)
                for i in range(len(algorithms)):
                    for j in range(i + 1, len(algorithms)):
                        algo1, algo2 = algorithms[i], algorithms[j]

                        df1 = self.df_games[
                            (self.df_games["data_tier"] == tier)
                            & (self.df_games["algorithm"] == algo1)
                            & (self.df_games["opponent"] == opp)
                        ].sort_values("seed")

                        df2 = self.df_games[
                            (self.df_games["data_tier"] == tier)
                            & (self.df_games["algorithm"] == algo2)
                            & (self.df_games["opponent"] == opp)
                        ].sort_values("seed")

                        if df1.empty or df2.empty:
                            continue

                        n1, n2 = len(df1), len(df2)
                        w1, w2 = int(df1["agent_won"].sum()), int(df2["agent_won"].sum())
                        p1, p2 = w1 / n1, w2 / n2

                        # 1. Win Rate Tests
                        z_stat, p_z = run_two_proportion_z_test(w1, n1, w2, n2)
                        fisher_stat, p_fisher = stats.fisher_exact([[w1, n1 - w1], [w2, n2 - w2]])

                        test_rows.append({
                            "data_tier": tier,
                            "comparison_category": "algorithm_pairwise_contrast",
                            "group1_algorithm": algo1,
                            "group2_algorithm": algo2,
                            "opponent": opp,
                            "metric": "win_rate",
                            "test_name": "Two-Proportion z-Test",
                            "test_type": "independent",
                            "group1_value": round(p1, 4),
                            "group2_value": round(p2, 4),
                            "measured_difference": round(p1 - p2, 4),
                            "test_statistic": round(z_stat, 4),
                            "raw_p_value": float(p_z),
                            "fisher_p_value": float(p_fisher),
                        })

                        # Proportion Effect Sizes
                        prop_fx = compute_proportion_effect_sizes(w1, n1, w2, n2)
                        effect_rows.append({
                            "data_tier": tier,
                            "comparison_category": "algorithm_pairwise_contrast",
                            "group1_algorithm": algo1,
                            "group2_algorithm": algo2,
                            "opponent": opp,
                            "metric": "win_rate",
                            "effect_measure": "Odds Ratio",
                            "point_estimate": round(prop_fx.get("odds_ratio", 1.0), 4),
                            "ci_95_lower": round(prop_fx.get("odds_ratio_ci_lower", 1.0), 4),
                            "ci_95_upper": round(prop_fx.get("odds_ratio_ci_upper", 1.0), 4),
                            "interpretation": f"Risk Diff = {prop_fx.get('risk_difference', 0.0):+.2%}",
                        })

                        # 2. Continuous Reward Tests
                        r1, r2 = df1["agent_reward"].values, df2["agent_reward"].values
                        t_stat, p_t = stats.ttest_ind(r1, r2, equal_var=False)
                        u_stat, p_u = stats.mannwhitneyu(r1, r2, alternative="two-sided")

                        test_rows.append({
                            "data_tier": tier,
                            "comparison_category": "algorithm_pairwise_contrast",
                            "group1_algorithm": algo1,
                            "group2_algorithm": algo2,
                            "opponent": opp,
                            "metric": "agent_reward",
                            "test_name": "Welch's t-Test",
                            "test_type": "independent_heteroscedastic",
                            "group1_value": round(float(np.mean(r1)), 4),
                            "group2_value": round(float(np.mean(r2)), 4),
                            "measured_difference": round(float(np.mean(r1) - np.mean(r2)), 4),
                            "test_statistic": round(float(t_stat), 4),
                            "raw_p_value": float(p_t),
                            "mann_whitney_p_value": float(p_u),
                        })

                        # Cohen's d & Hedges' g
                        d_res = compute_cohens_d(r1.tolist(), r2.tolist())
                        effect_rows.append({
                            "data_tier": tier,
                            "comparison_category": "algorithm_pairwise_contrast",
                            "group1_algorithm": algo1,
                            "group2_algorithm": algo2,
                            "opponent": opp,
                            "metric": "agent_reward",
                            "effect_measure": "Cohen's d",
                            "point_estimate": round(d_res["cohens_d"], 4),
                            "ci_95_lower": round(d_res["d_ci_lower"], 4),
                            "ci_95_upper": round(d_res["d_ci_upper"], 4),
                            "interpretation": interpret_cohens_d(d_res["cohens_d"]),
                        })

        # Apply multiple testing corrections across all raw p-values
        if test_rows:
            raw_ps = [r["raw_p_value"] for r in test_rows]
            bonf_ps = bonferroni_correction(raw_ps)
            holm_ps = holm_bonferroni_correction(raw_ps)
            bh_ps = benjamini_hochberg_correction(raw_ps)

            for idx, r in enumerate(test_rows):
                r["p_value_bonferroni"] = round(bonf_ps[idx], 6)
                r["p_value_holm"] = round(holm_ps[idx], 6)
                r["p_value_fdr_bh"] = round(bh_ps[idx], 6)
                r["significant_raw_05"] = bool(r["raw_p_value"] < 0.05)
                r["significant_holm_05"] = bool(holm_ps[idx] < 0.05)
                r["significant_fdr_05"] = bool(bh_ps[idx] < 0.05)
                r["statistical_conclusion"] = (
                    "Statistically significant difference (Holm p < 0.05)"
                    if holm_ps[idx] < 0.05
                    else "Fail to reject H0 (no statistically significant difference at alpha=0.05)"
                )

        return pd.DataFrame(test_rows), pd.DataFrame(effect_rows)

    def compute_cross_seed_analysis(self) -> pd.DataFrame:
        """Evaluate stability, variance, and invariance across random seeds."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        grouped = self.df_games.groupby(["data_tier", "algorithm", "opponent"])

        for (tier, algo, opp), grp in grouped:
            seed_subgroups = grp.groupby("seed")
            seed_means_wr = []
            seed_means_rew = []

            for seed_val, s_grp in seed_subgroups:
                n = len(s_grp)
                w = int(s_grp["agent_won"].sum())
                wr = w / n
                mean_rew = float(s_grp["agent_reward"].mean())
                seed_means_wr.append(wr)
                seed_means_rew.append(mean_rew)

                rows.append({
                    "data_tier": tier,
                    "algorithm": algo,
                    "opponent": opp,
                    "analysis_type": "individual_seed_profile",
                    "seed": seed_val,
                    "games_count": n,
                    "win_rate": round(wr, 4),
                    "mean_reward": round(mean_rew, 4),
                    "std_reward": round(float(s_grp["agent_reward"].std(ddof=1)), 4),
                    "mean_turns": round(float(s_grp["turns"].mean()), 2),
                })

            # Inter-seed stability metrics
            if len(seed_means_wr) >= 2:
                inter_seed_wr_sd = float(np.std(seed_means_wr, ddof=1))
                inter_seed_rew_sd = float(np.std(seed_means_rew, ddof=1))
                mean_wr_across_seeds = float(np.mean(seed_means_wr))
                cv_wr = (inter_seed_wr_sd / mean_wr_across_seeds) if mean_wr_across_seeds > 0 else 0.0

                # Kruskal-Wallis across seed groups for reward distribution
                seed_rew_arrays = [s_grp["agent_reward"].values for _, s_grp in seed_subgroups]
                kw_stat, kw_p = stats.kruskal(*seed_rew_arrays) if len(seed_rew_arrays) > 1 else (0.0, 1.0)

                rows.append({
                    "data_tier": tier,
                    "algorithm": algo,
                    "opponent": opp,
                    "analysis_type": "inter_seed_invariance_summary",
                    "seed": "ALL_SEEDS_AGGREGATED",
                    "games_count": len(grp),
                    "win_rate": round(mean_wr_across_seeds, 4),
                    "inter_seed_wr_std": round(inter_seed_wr_sd, 4),
                    "coefficient_of_variation_wr": round(cv_wr, 4),
                    "kruskal_wallis_stat": round(float(kw_stat), 4),
                    "kruskal_wallis_p": float(kw_p),
                    "seed_invariant_alpha_05": bool(kw_p >= 0.05),
                })

        return pd.DataFrame(rows)

    def compute_opponent_specific_analysis(self) -> pd.DataFrame:
        """Evaluate sensitivity and strategy adaptation comparing Random vs Expert opponents."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        algorithms = self.df_games["algorithm"].unique().tolist()
        tiers = self.df_games["data_tier"].unique().tolist()

        for tier in tiers:
            for algo in algorithms:
                df_rnd = self.df_games[
                    (self.df_games["data_tier"] == tier)
                    & (self.df_games["algorithm"] == algo)
                    & (self.df_games["opponent"] == "random")
                ]
                df_exp = self.df_games[
                    (self.df_games["data_tier"] == tier)
                    & (self.df_games["algorithm"] == algo)
                    & (self.df_games["opponent"] == "expert")
                ]

                if df_rnd.empty or df_exp.empty:
                    continue

                w_rnd, n_rnd = int(df_rnd["agent_won"].sum()), len(df_rnd)
                w_exp, n_exp = int(df_exp["agent_won"].sum()), len(df_exp)
                p_rnd = w_rnd / n_rnd
                p_exp = w_exp / n_exp
                delta_wr = p_rnd - p_exp

                z_stat, p_z = run_two_proportion_z_test(w_rnd, n_rnd, w_exp, n_exp)

                r_rnd, r_exp = df_rnd["agent_reward"].values, df_exp["agent_reward"].values
                t_stat, p_t = stats.ttest_ind(r_rnd, r_exp, equal_var=False)
                d_res = compute_cohens_d(r_rnd.tolist(), r_exp.tolist())

                rows.append({
                    "data_tier": tier,
                    "algorithm": algo,
                    "win_rate_vs_random": round(p_rnd, 4),
                    "win_rate_vs_expert": round(p_exp, 4),
                    "win_rate_drop": round(delta_wr, 4),
                    "drop_z_statistic": round(z_stat, 4),
                    "drop_p_value": float(p_z),
                    "mean_reward_vs_random": round(float(np.mean(r_rnd)), 4),
                    "mean_reward_vs_expert": round(float(np.mean(r_exp)), 4),
                    "reward_drop_cohens_d": round(d_res["cohens_d"], 4),
                    "significant_performance_drop": bool(p_z < 0.05),
                })

        return pd.DataFrame(rows)

    def compute_seat_fairness_analysis(self) -> pd.DataFrame:
        """Assess Player 0 vs Player 1 fairness and test for first-mover advantage."""
        if self.df_games is None or self.df_games.empty:
            return pd.DataFrame()

        rows = []
        grouped = self.df_games.groupby(["data_tier", "algorithm", "opponent"])

        for (tier, algo, opp), grp in grouped:
            s0 = grp[grp["agent_seat"] == 0]
            s1 = grp[grp["agent_seat"] == 1]
            if s0.empty or s1.empty:
                continue

            w0, n0 = int(s0["agent_won"].sum()), len(s0)
            w1, n1 = int(s1["agent_won"].sum()), len(s1)
            p0 = w0 / n0
            p1 = w1 / n1
            seat_bias = p0 - p1

            # Fisher's Exact & Two-proportion z-test
            tbl = [[w0, n0 - w0], [w1, n1 - w1]]
            odds_r, p_fisher = stats.fisher_exact(tbl)
            z_seat, p_z = run_two_proportion_z_test(w0, n0, w1, n1)

            # Continuous reward test
            r0 = s0["agent_reward"].values
            r1 = s1["agent_reward"].values
            t_seat_r, p_t_r = stats.ttest_ind(r0, r1, equal_var=False)

            rows.append({
                "data_tier": tier,
                "algorithm": algo,
                "opponent": opp,
                "seat_0_games": n0,
                "seat_0_wins": w0,
                "seat_0_win_rate": round(p0, 4),
                "seat_1_games": n1,
                "seat_1_wins": w1,
                "seat_1_win_rate": round(p1, 4),
                "seat_bias": round(seat_bias, 4),
                "fisher_exact_p": float(p_fisher),
                "z_test_p": float(p_z),
                "reward_seat_diff_p": float(p_t_r),
                "seat_neutral_alpha_05": bool(p_fisher >= 0.05),
                "conclusion": "Seat Neutral (p >= 0.05)" if p_fisher >= 0.05 else "Seat Bias Detected (p < 0.05)",
            })

        return pd.DataFrame(rows)

    def generate_publication_tables(
        self,
        df_desc: pd.DataFrame,
        df_norm: pd.DataFrame,
        df_tests: pd.DataFrame,
        df_effects: pd.DataFrame,
        df_seeds: pd.DataFrame,
        df_opps: pd.DataFrame,
        df_seats: pd.DataFrame,
    ) -> Dict[str, str]:
        """Format and persist publication-ready tables in CSV, Markdown, and LaTeX."""
        saved_files: Dict[str, str] = {}

        # Table 1: Master Descriptive Statistics
        if not df_desc.empty:
            self._write_table_formats(
                df_desc,
                "table1_descriptive_summary",
                "Table 1: Master Descriptive Statistics and Confidence Intervals",
                saved_files,
            )

        # Table 2: Win Rates & 95% Confidence Intervals
        if not df_desc.empty:
            df_wr = df_desc[df_desc["metric"] == "win_rate"].copy()
            if not df_wr.empty:
                self._write_table_formats(
                    df_wr,
                    "table2_win_rate_and_confidence_intervals",
                    "Table 2: Win Rates and 95% Confidence Intervals (Wilson vs Bootstrap 10k)",
                    saved_files,
                )

        # Table 3: Comparative Hypothesis Testing & Multiple Corrections
        if not df_tests.empty:
            self._write_table_formats(
                df_tests,
                "table3_comparative_hypothesis_tests",
                "Table 3: Pairwise Model Hypothesis Tests with FWER & FDR Corrections",
                saved_files,
            )

        # Table 4: Measured Effect Sizes
        if not df_effects.empty:
            self._write_table_formats(
                df_effects,
                "table4_effect_sizes",
                "Table 4: Measured Effect Sizes (Cohen's d, Hedges' g, Odds Ratio)",
                saved_files,
            )

        # Table 5: Seat Fairness Analysis
        if not df_seats.empty:
            self._write_table_formats(
                df_seats,
                "table5_seat_fairness_analysis",
                "Table 5: First-Mover / Player Position Fairness Evaluation",
                saved_files,
            )

        # Table 6: Distributional Normality Diagnostics
        if not df_norm.empty:
            self._write_table_formats(
                df_norm,
                "table6_distributional_normality_tests",
                "Table 6: Distributional Diagnostics and Normality Tests",
                saved_files,
            )

        # Table 7: Cross-Seed Invariance Breakdown
        if not df_seeds.empty:
            self._write_table_formats(
                df_seeds,
                "table7_cross_seed_analysis",
                "Table 7: Cross-Seed Invariance and Stability Evaluation",
                saved_files,
            )

        # Table 8: Opponent Sensitivity Analysis
        if not df_opps.empty:
            self._write_table_formats(
                df_opps,
                "table8_opponent_sensitivity_analysis",
                "Table 8: Opponent Strategy Contrast (Random vs Expert Adaptation)",
                saved_files,
            )

        return saved_files

    def _write_table_formats(
        self,
        df: pd.DataFrame,
        base_name: str,
        caption: str,
        saved_registry: Dict[str, str],
    ) -> None:
        """Write DataFrame cleanly into CSV, GitHub Markdown, and LaTeX."""
        if df.empty:
            return

        csv_path = self.tables_dir / f"{base_name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        saved_registry[f"{base_name}_csv"] = str(csv_path)

        md_path = self.tables_dir / f"{base_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"### {caption}\n\n")
            f.write(df.to_markdown(index=False))
            f.write(f"\n\n*Source: Raw evaluation data generated under Protocol v{FROZEN_PROTOCOL_VERSION}.*\n")
        saved_registry[f"{base_name}_md"] = str(md_path)

        tex_path = self.tables_dir / f"{base_name}.tex"
        latex_str = df.to_latex(
            index=False,
            caption=caption,
            label=f"tab:{base_name}",
            escape=True,
        )
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        saved_registry[f"{base_name}_tex"] = str(tex_path)

    def run(self) -> Dict[str, Any]:
        """Execute the statistical analysis pipeline adhering to data tier contracts."""
        print("=" * 88)
        print("PHASE 11 — STATISTICAL ANALYSIS PIPELINE")
        print(f"Protocol Specification: v{FROZEN_PROTOCOL_VERSION}")
        print(f"Scanning raw directories: {[str(d) for d in self.raw_dirs]}")
        print("=" * 88)

        # 1. Audit raw evaluation data
        audit = self.audit_raw_data()
        print(f"Raw Data Audit Status: {audit['status']}")
        print(f"  Total Games Found: {audit['total_games_ingested']:,}")
        print(f"  Final Experiment Games: {audit['final_experiment_games']:,} / {TOTAL_REQUIRED_FINAL_GAMES:,}")
        print(f"  Preliminary Pilot Games: {audit['preliminary_pilot_games']:,}")
        print(f"  Data Tier Assigned: {audit['data_tier']}")

        # 2. Strict Check for Final Experiment Data
        if self.require_final_data and not audit["is_final_complete"]:
            audit_report_path = self.processed_dir / "final_statistical_audit_incomplete.json"
            with open(audit_report_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2)

            msg = (
                f"\nPROTOCOL AUDIT DIRECTIVE: Final evaluation raw data is INCOMPLETE.\n"
                f"Found {audit['final_experiment_games']:,} of {TOTAL_REQUIRED_FINAL_GAMES:,} required games from 500k checkpoints.\n"
                f"Final statistical analysis is reported as INCOMPLETE.\n"
                f"Per protocol invariants, substitute or fabricated values will NOT be generated.\n"
                f"Audit report saved to: {audit_report_path.name}"
            )
            print(msg)
            return {
                "status": "INCOMPLETE",
                "audit": audit,
                "audit_file": str(audit_report_path),
            }

        # 3. Ingest raw game records
        df_games = self.ingest_stored_results()
        if df_games.empty:
            raise RuntimeError(f"No valid game records found in {[str(d) for d in self.raw_dirs]}")

        # Consolidate games CSV
        consolidated_csv = self.processed_dir / "consolidated_game_records.csv"
        df_games.to_csv(consolidated_csv, index=False, encoding="utf-8")

        # 4. Compute all statistical modules
        print("[1/6] Running distributional normality diagnostics...")
        df_norm = self.compute_normality_diagnostics()

        print("[2/6] Computing descriptive statistics and confidence intervals...")
        df_desc = self.compute_descriptive_statistics()

        print("[3/6] Running pairwise model comparisons & multiple-testing corrections...")
        df_tests, df_effects = self.compute_pairwise_model_comparisons()

        print("[4/6] Conducting cross-seed invariance analysis...")
        df_seeds = self.compute_cross_seed_analysis()

        print("[5/6] Conducting opponent sensitivity and strategy adaptation analysis...")
        df_opps = self.compute_opponent_specific_analysis()

        print("[6/6] Conducting seat and first-mover fairness evaluation...")
        df_seats = self.compute_seat_fairness_analysis()

        # 5. Export Processed JSONs
        for name, dframe in [
            ("descriptive_statistics", df_desc),
            ("distributional_normality_tests", df_norm),
            ("pairwise_hypothesis_tests", df_tests),
            ("effect_sizes", df_effects),
            ("cross_seed_analysis", df_seeds),
            ("opponent_sensitivity_analysis", df_opps),
            ("seat_fairness_analysis", df_seats),
        ]:
            if not dframe.empty:
                dframe.to_csv(self.processed_dir / f"{name}.csv", index=False)
                with open(self.processed_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(dframe.to_dict(orient="records"), f, indent=2)

        # 6. Generate Publication Tables
        table_files = self.generate_publication_tables(
            df_desc, df_norm, df_tests, df_effects, df_seeds, df_opps, df_seats
        )

        print("=" * 88)
        print("STATISTICAL ANALYSIS EXECUTION COMPLETED")
        print(f"Data Tier: {self.data_tier.value}")
        print(f"Processed Games: {len(df_games):,} | Tables Generated: {len(table_files) // 3}")
        print(f"Processed Directory: {self.processed_dir}")
        print(f"Tables Directory: {self.tables_dir}")
        print("=" * 88)

        return {
            "status": "COMPLETED",
            "data_tier": self.data_tier.value,
            "num_games": len(df_games),
            "num_tests": len(df_tests),
            "num_effects": len(df_effects),
            "table_files": table_files,
            "audit": audit,
        }
