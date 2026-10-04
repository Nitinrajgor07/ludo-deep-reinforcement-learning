"""Stochasticity and Robustness Analysis Module for PhD Ludo DRL.

Adheres strictly to FINAL_EXPERIMENT_PROTOCOL.md (Section 18) and project phases.doc.md:
- Predefined environmental stochasticity levels: Dice variance perturbation from 1.0x to 1.4x.
- Skewed dice distributions: Loaded dice with higher probability of rolling 1 or 6.
- Invariant across all 4 algorithms: DQN, PPO, Discrete SAC, Categorical DQN (C51).
- Metrics measured: Win rate, Reward, Convergence, Performance Variation, Robustness Degradation.
"""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# ==============================================================================
# PREDEFINED STOCHASTICITY CONDITIONS (FROZEN PROTOCOL SPECIFICATION)
# ==============================================================================
BASE_DICE_VARIANCE = 35.0 / 12.0  # 2.9166666666666665
VARIANCE_MULTIPLIERS = [1.0, 1.1, 1.2, 1.3, 1.4]


def compute_perturbed_dice_distribution(multiplier: float) -> np.ndarray:
    """Compute exact symmetric loaded dice probability distribution for variance multiplier k.

    Under Section 18 ("loaded dice with higher probability of rolling 1 or 6") and
    project phases ("perturbing dice variance from 1.0x to 1.4x"):
    Linear perturbation around uniform distribution preserves mean mu = 3.5:
        p_i = 1/6 + (5/64) * (k - 1) * ((i - 3.5)^2 - 35/12)
    """
    if not (1.0 <= multiplier <= 1.5):
        raise ValueError(f"Predefined multiplier must be between 1.0 and 1.5, got {multiplier}")

    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    dev2 = (vals - 3.5) ** 2
    c = (5.0 / 64.0) * (multiplier - 1.0)
    p = (1.0 / 6.0) + c * (dev2 - BASE_DICE_VARIANCE)

    # Sanity checks
    assert np.all(p > 0), f"Negative probabilities generated: {p}"
    assert np.isclose(np.sum(p), 1.0, atol=1e-6), f"Probabilities do not sum to 1.0: {np.sum(p)}"
    actual_var = np.sum(p * dev2)
    expected_var = multiplier * BASE_DICE_VARIANCE
    assert np.isclose(actual_var, expected_var, atol=1e-5), f"Variance mismatch: {actual_var} vs {expected_var}"

    return p


PREDEFINED_STOCHASTICITY_LEVELS: Dict[float, Dict[str, Any]] = {}
for m in VARIANCE_MULTIPLIERS:
    dist = compute_perturbed_dice_distribution(m)
    PREDEFINED_STOCHASTICITY_LEVELS[m] = {
        "multiplier": m,
        "name": f"{m:.1f}x Variance",
        "description": "Baseline Fair Die" if m == 1.0 else f"Skewed Loaded Die (P(1)=P(6)={dist[0]:.3f})",
        "target_variance": round(m * BASE_DICE_VARIANCE, 4),
        "probabilities": [round(float(x), 6) for x in dist],
        "p_roll_1_or_6": round(float(dist[0] + dist[5]), 4),
        "p_roll_2_or_5": round(float(dist[1] + dist[4]), 4),
        "p_roll_3_or_4": round(float(dist[2] + dist[3]), 4),
    }


def compute_robustness_metrics(
    eval_results_by_multiplier: Dict[float, Dict[str, Any]],
    convergence_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive robustness degradation and variation metrics for an algorithm.

    Args:
        eval_results_by_multiplier: Dictionary mapping variance multiplier (e.g. 1.0..1.4)
                                    to aggregate evaluation summary dictionary.
        convergence_data: Optional training convergence metrics for the algorithm.

    Returns:
        Structured robustness analysis dictionary.
    """
    if 1.0 not in eval_results_by_multiplier:
        raise KeyError("Baseline 1.0x variance results required to compute degradation.")

    baseline = eval_results_by_multiplier[1.0]
    base_wr = baseline["win_rate"]
    base_reward = baseline["average_reward"]

    degradation_by_level: Dict[str, Dict[str, Any]] = {}
    multipliers = sorted(eval_results_by_multiplier.keys())
    wr_list: List[float] = []
    rew_list: List[float] = []

    for m in multipliers:
        res = eval_results_by_multiplier[m]
        wr = res["win_rate"]
        rew = res["average_reward"]
        wr_list.append(wr)
        rew_list.append(rew)

        delta_wr = round(wr - base_wr, 4)
        pct_wr = round(((wr - base_wr) / base_wr * 100.0), 2) if base_wr > 0 else 0.0
        delta_rew = round(rew - base_reward, 4)
        pct_rew = round(((rew - base_reward) / abs(base_reward) * 100.0), 2) if abs(base_reward) > 1e-6 else 0.0

        degradation_by_level[f"{m:.1f}x"] = {
            "multiplier": m,
            "win_rate": wr,
            "win_rate_ci_95": res.get("win_rate_ci_95_wilson", [0.0, 0.0]),
            "delta_win_rate": delta_wr,
            "pct_win_rate_degradation": pct_wr,
            "average_reward": rew,
            "reward_ci_95": res.get("reward_ci_95", [0.0, 0.0]),
            "delta_reward": delta_rew,
            "pct_reward_degradation": pct_rew,
            "bernoulli_win_variance": res.get("bernoulli_win_variance", wr * (1 - wr)),
            "reward_variance": res.get("reward_variance", 0.0),
            "reward_std": res.get("reward_std", 0.0),
            "turns_variance": res.get("turns_variance", 0.0),
            "turns_std": res.get("turns_std", 0.0),
            "average_turns": res.get("average_turns", 0.0),
            "coefficient_of_variation_reward": round(res.get("reward_std", 0.0) / abs(rew), 4) if abs(rew) > 1e-4 else 0.0,
        }

    # Linear sensitivity slopes (dW/dk and dR/dk)
    m_arr = np.array(multipliers)
    wr_arr = np.array(wr_list)
    rew_arr = np.array(rew_list)

    if len(multipliers) > 1:
        slope_wr, intercept_wr, r_val_wr, p_val_wr, _ = stats.linregress(m_arr, wr_arr)
        slope_rew, intercept_rew, r_val_rew, p_val_rew, _ = stats.linregress(m_arr, rew_arr)
    else:
        slope_wr, intercept_wr, r_val_wr, p_val_wr = 0.0, 0.0, 0.0, 1.0
        slope_rew, intercept_rew, r_val_rew, p_val_rew = 0.0, 0.0, 0.0, 1.0

    # Summary robustness indices
    extreme_res = eval_results_by_multiplier.get(1.4, baseline)
    extreme_wr = extreme_res["win_rate"]
    extreme_rew = extreme_res["average_reward"]

    max_wr_drop = max(0.0, base_wr - min(wr_list))
    total_pct_degradation = round(((extreme_wr - base_wr) / base_wr * 100.0), 2) if base_wr > 0 else 0.0
    retention_index = round(extreme_wr / base_wr, 4) if base_wr > 0 else 0.0

    return {
        "baseline_1_0x": {
            "win_rate": base_wr,
            "average_reward": base_reward,
            "bernoulli_variance": baseline.get("bernoulli_win_variance", base_wr * (1 - base_wr)),
            "reward_variance": baseline.get("reward_variance", 0.0),
        },
        "extreme_1_4x": {
            "win_rate": extreme_wr,
            "average_reward": extreme_rew,
            "bernoulli_variance": extreme_res.get("bernoulli_win_variance", extreme_wr * (1 - extreme_wr)),
            "reward_variance": extreme_res.get("reward_variance", 0.0),
        },
        "robustness_summary": {
            "max_win_rate_drop": round(max_wr_drop, 4),
            "total_pct_win_rate_degradation": total_pct_degradation,
            "retention_index_at_1_4x": retention_index,
            "win_rate_sensitivity_slope": round(float(slope_wr), 4),
            "reward_sensitivity_slope": round(float(slope_rew), 4),
            "win_rate_r_squared": round(float(r_val_wr**2), 4),
            "win_rate_spread": round(float(np.max(wr_arr) - np.min(wr_arr)), 4),
            "reward_spread": round(float(np.max(rew_arr) - np.min(rew_arr)), 4),
        },
        "per_condition_metrics": degradation_by_level,
        "convergence": convergence_data or {},
    }


def generate_robustness_figures(
    master_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Generate high-resolution publication-quality comparison plots from actual experiment data.

    Args:
        master_results: Dictionary containing 'algorithms', with each algorithm's
                        robustness analysis data.
        output_dir: Directory where PNG plots are saved.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # Palette and style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    algo_colors = {
        "dqn": "#1f77b4",       # Solid blue
        "ppo": "#ff7f0e",       # Vivid orange
        "model3": "#2ca02c",    # Vibrant green (Discrete SAC)
        "model4": "#d62728",    # Crisp red (C51 Categorical DQN)
    }
    algo_labels = {
        "dqn": "Model 1: Double Dueling DQN",
        "ppo": "Model 2: PPO Baseline",
        "model3": "Model 3: Discrete SAC",
        "model4": "Model 4: Categorical DQN (C51)",
    }
    multipliers = VARIANCE_MULTIPLIERS

    # --------------------------------------------------------------------------
    # Plot 1: Win Rate vs Dice Variance Multiplier (with Wilson 95% CI bands)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for algo_key, color in algo_colors.items():
        if algo_key not in master_results:
            continue
        data = master_results[algo_key]["per_condition_metrics"]
        wrs = [data[f"{m:.1f}x"]["win_rate"] * 100.0 for m in multipliers]
        ci_low = [data[f"{m:.1f}x"]["win_rate_ci_95"][0] * 100.0 for m in multipliers]
        ci_high = [data[f"{m:.1f}x"]["win_rate_ci_95"][1] * 100.0 for m in multipliers]

        ax.plot(multipliers, wrs, marker="o", linewidth=2.4, markersize=7, color=color, label=algo_labels[algo_key])
        ax.fill_between(multipliers, ci_low, ci_high, color=color, alpha=0.15)

    ax.set_title("Policy Robustness: Win Rate vs. Dice Variance Perturbation", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Environmental Dice Variance Multiplier (Fair 1.0× to Loaded 1.4×)", fontsize=11, labelpad=8)
    ax.set_ylabel("Win Rate vs Expert Opponent (%)", fontsize=11, labelpad=8)
    ax.set_xticks(multipliers)
    ax.set_xticklabels([f"{m:.1f}×" for m in multipliers], fontsize=10)
    ax.set_ylim(-2, 102)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = output_dir / "robustness_win_rate_curves.png"
    plt.savefig(p1)
    plt.close()
    generated_files.append(p1)

    # --------------------------------------------------------------------------
    # Plot 2: Average Episode Reward vs Dice Variance Multiplier (with 95% Bootstrap CI)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for algo_key, color in algo_colors.items():
        if algo_key not in master_results:
            continue
        data = master_results[algo_key]["per_condition_metrics"]
        rews = [data[f"{m:.1f}x"]["average_reward"] for m in multipliers]
        ci_low = [data[f"{m:.1f}x"]["reward_ci_95"][0] for m in multipliers]
        ci_high = [data[f"{m:.1f}x"]["reward_ci_95"][1] for m in multipliers]

        ax.plot(multipliers, rews, marker="s", linewidth=2.4, markersize=7, color=color, label=algo_labels[algo_key])
        ax.fill_between(multipliers, ci_low, ci_high, color=color, alpha=0.15)

    ax.set_title("Episode Reward Trajectory Across Stochasticity Perturbation Levels", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Environmental Dice Variance Multiplier (1.0× to 1.4×)", fontsize=11, labelpad=8)
    ax.set_ylabel("Mean Cumulative Episode Reward", fontsize=11, labelpad=8)
    ax.set_xticks(multipliers)
    ax.set_xticklabels([f"{m:.1f}×" for m in multipliers], fontsize=10)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = output_dir / "robustness_reward_curves.png"
    plt.savefig(p2)
    plt.close()
    generated_files.append(p2)

    # --------------------------------------------------------------------------
    # Plot 3: Robustness Degradation Bar Chart at Extreme Stochasticity (1.4x)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    algos_present = [a for a in ["dqn", "ppo", "model3", "model4"] if a in master_results]
    x_pos = np.arange(len(algos_present))

    delta_wrs = [master_results[a]["per_condition_metrics"]["1.4x"]["delta_win_rate"] * 100.0 for a in algos_present]
    pct_wrs = [master_results[a]["per_condition_metrics"]["1.4x"]["pct_win_rate_degradation"] for a in algos_present]
    bar_colors = [algo_colors[a] for a in algos_present]

    # Subplot 1: Absolute Win Rate Change (percentage points)
    bars1 = ax1.bar(x_pos, delta_wrs, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax1.set_title("Absolute Win Rate Shift at 1.4× Variance", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Win Rate Shift (Percentage Points)", fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([algo_labels[a].split(":")[0] for a in algos_present], fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + (0.5 if yval >= 0 else -0.5), f"{yval:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    # Subplot 2: Relative Degradation (%)
    bars2 = ax2.bar(x_pos, pct_wrs, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_title("Relative Win Rate Degradation (%) at 1.4× Variance", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Relative Degradation (% vs 1.0× Baseline)", fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([algo_labels[a].split(":")[0] for a in algos_present], fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + (0.8 if yval >= 0 else -0.8), f"{yval:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    plt.suptitle("Comparative Robustness Degradation Analysis Across Algorithms", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p3 = output_dir / "robustness_degradation_comparison.png"
    plt.savefig(p3)
    plt.close()
    generated_files.append(p3)

    # --------------------------------------------------------------------------
    # Plot 4: Performance Variance Profiles (Reward Variance vs Variance Multiplier)
    # --------------------------------------------------------------------------
    fig, (ax_var1, ax_var2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    for algo_key, color in algo_colors.items():
        if algo_key not in master_results:
            continue
        data = master_results[algo_key]["per_condition_metrics"]
        rew_vars = [data[f"{m:.1f}x"]["reward_variance"] for m in multipliers]
        turn_vars = [data[f"{m:.1f}x"]["turns_variance"] for m in multipliers]

        ax_var1.plot(multipliers, rew_vars, marker="^", linewidth=2.0, markersize=6, color=color, label=algo_labels[algo_key])
        ax_var2.plot(multipliers, turn_vars, marker="d", linewidth=2.0, markersize=6, color=color, label=algo_labels[algo_key])

    ax_var1.set_title("Reward Variance Across Perturbation Levels", fontsize=12, fontweight="bold")
    ax_var1.set_xlabel("Dice Variance Multiplier", fontsize=10)
    ax_var1.set_ylabel("Empirical Reward Variance (σ²)", fontsize=10)
    ax_var1.set_xticks(multipliers)
    ax_var1.grid(True, linestyle="--", alpha=0.6)
    ax_var1.legend(loc="best", fontsize=9)

    ax_var2.set_title("Episode Length (Turns) Variance Across Perturbation Levels", fontsize=12, fontweight="bold")
    ax_var2.set_xlabel("Dice Variance Multiplier", fontsize=10)
    ax_var2.set_ylabel("Turns Variance (σ²)", fontsize=10)
    ax_var2.set_xticks(multipliers)
    ax_var2.grid(True, linestyle="--", alpha=0.6)
    ax_var2.legend(loc="best", fontsize=9)

    plt.suptitle("Empirical Performance Variation Under Environmental Stochasticity", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p4 = output_dir / "robustness_variance_profiles.png"
    plt.savefig(p4)
    plt.close()
    generated_files.append(p4)

    # --------------------------------------------------------------------------
    # Plot 5: Predefined Dice Distributions (1.0x to 1.4x)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    dice_values = np.array([1, 2, 3, 4, 5, 6])
    n_levels = len(multipliers)
    bar_width = 0.15
    cmap = plt.get_cmap("viridis")

    for idx, m in enumerate(multipliers):
        probs = np.array(PREDEFINED_STOCHASTICITY_LEVELS[m]["probabilities"])
        offsets = dice_values - 0.35 + idx * bar_width
        ax.bar(offsets, probs * 100.0, width=bar_width, color=cmap(idx / max(1, n_levels - 1)), label=f"{m:.1f}× Var (P(1,6)={probs[0]:.1%})", edgecolor="black", linewidth=0.5)

    ax.set_title("Predefined Stochasticity Levels: Skewed Loaded Dice Distributions", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Dice Face Value (1..6)", fontsize=11, labelpad=8)
    ax.set_ylabel("Roll Probability (%)", fontsize=11, labelpad=8)
    ax.set_xticks(dice_values)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    p5 = output_dir / "predefined_dice_distributions.png"
    plt.savefig(p5)
    plt.close()
    generated_files.append(p5)

    return generated_files
