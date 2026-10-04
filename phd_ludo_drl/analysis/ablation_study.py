"""Ablation Studies Analysis and Visualization Module for PhD Ludo DRL.

Adheres strictly to FINAL_EXPERIMENT_PROTOCOL.md (Section 19):
1. Representation Ablation: Structured CNN (4-channel spatial) vs Flat 60-D Observation Baseline.
2. Reward Shaping Ablation: Full 8-Event Dense Reward vs Sparse Terminal Reward (+/- 1.0 only).
3. Action Masking Ablation: Action Masking (-10^9 suppression) vs Unmasked learning with -0.50 penalty.

Key Invariant:
Change ONLY the component being studied.
Keep all other hyperparameters, seeds, and training budgets identical.
"""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def compute_ablation_impact(
    baseline_summary: Dict[str, Any],
    ablation_summary: Dict[str, Any],
    ablation_name: str,
    component_studied: str,
) -> Dict[str, Any]:
    """Compute comparative impact metrics between baseline and an ablated condition."""
    base_wr = baseline_summary.get("win_rate", 0.0)
    ab_wr = ablation_summary.get("win_rate", 0.0)
    base_rew = baseline_summary.get("average_reward", 0.0)
    ab_rew = ablation_summary.get("average_reward", 0.0)

    delta_wr = round(ab_wr - base_wr, 4)
    pct_wr_degradation = round(((ab_wr - base_wr) / base_wr * 100.0), 2) if base_wr > 0 else 0.0
    delta_rew = round(ab_rew - base_rew, 4)
    pct_rew_degradation = round(((ab_rew - base_rew) / abs(base_rew) * 100.0), 2) if abs(base_rew) > 1e-5 else 0.0

    # Two proportion z-test on win rates
    w1, n1 = int(round(base_wr * baseline_summary.get("total_games", 50))), baseline_summary.get("total_games", 50)
    w2, n2 = int(round(ab_wr * ablation_summary.get("total_games", 50))), ablation_summary.get("total_games", 50)
    p_pool = (w1 + w2) / (n1 + n2) if (n1 + n2) > 0 else 0.0

    if p_pool in (0.0, 1.0) or n1 == 0 or n2 == 0:
        z_stat, z_pval = 0.0, 1.0
    else:
        se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_stat = (ab_wr - base_wr) / se_pool
        z_pval = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {
        "ablation_name": ablation_name,
        "component_studied": component_studied,
        "baseline_win_rate": base_wr,
        "ablation_win_rate": ab_wr,
        "delta_win_rate": delta_wr,
        "pct_win_rate_impact": pct_wr_degradation,
        "baseline_reward": base_rew,
        "ablation_reward": ab_rew,
        "delta_reward": delta_rew,
        "pct_reward_impact": pct_rew_degradation,
        "baseline_illegal_rate": baseline_summary.get("illegal_action_rate", 0.0),
        "ablation_illegal_rate": ablation_summary.get("illegal_action_rate", 0.0),
        "z_statistic": round(float(z_stat), 4),
        "z_pvalue": float(z_pval),
        "significant_alpha_005": bool(z_pval < 0.05),
    }


def generate_ablation_figures(
    ablation_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Generate publication-quality comparison plots from actual ablation experiment data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    conditions = ["Baseline (Full)", "Flat Encoder", "Sparse Reward", "Unmasked"]
    keys = ["baseline", "ablation_repr_flat", "ablation_reward_sparse", "ablation_mask_unmasked"]
    colors = ["#1f77b4", "#9467bd", "#e377c2", "#d62728"]

    # Filter present conditions
    active_pairs = [(cond, k, col) for cond, k, col in zip(conditions, keys, colors) if k in ablation_data]
    if not active_pairs:
        return []

    cond_labels = [p[0] for p in active_pairs]
    cond_keys = [p[1] for p in active_pairs]
    bar_cols = [p[2] for p in active_pairs]
    x_pos = np.arange(len(cond_labels))

    # --------------------------------------------------------------------------
    # Plot 1: Win Rate Comparison vs Expert and Random (with Wilson 95% CIs)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    width = 0.35

    expert_wrs = [ablation_data[k]["eval_expert"]["win_rate"] * 100.0 for k in cond_keys]
    random_wrs = [ablation_data[k]["eval_random"]["win_rate"] * 100.0 for k in cond_keys]

    expert_err_low = [expert_wrs[i] - ablation_data[k]["eval_expert"]["win_rate_ci_95_wilson"][0] * 100.0 for i, k in enumerate(cond_keys)]
    expert_err_high = [ablation_data[k]["eval_expert"]["win_rate_ci_95_wilson"][1] * 100.0 - expert_wrs[i] for i, k in enumerate(cond_keys)]
    expert_yerr = [expert_err_low, expert_err_high]

    random_err_low = [random_wrs[i] - ablation_data[k]["eval_random"]["win_rate_ci_95_wilson"][0] * 100.0 for i, k in enumerate(cond_keys)]
    random_err_high = [ablation_data[k]["eval_random"]["win_rate_ci_95_wilson"][1] * 100.0 - random_wrs[i] for i, k in enumerate(cond_keys)]
    random_yerr = [random_err_low, random_err_high]

    bars1 = ax.bar(x_pos - width / 2, expert_wrs, width, yerr=expert_yerr, capsize=4, label="vs Expert Opponent", color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x_pos + width / 2, random_wrs, width, yerr=random_yerr, capsize=4, label="vs Random Opponent", color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.7)

    ax.set_title("Ablation Study: Empirical Win Rate Comparison Across Architectures", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Win Rate (%)", fontsize=11, labelpad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars1:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 2.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 2.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    p1 = output_dir / "ablation_win_rate_comparison.png"
    plt.savefig(p1)
    plt.close()
    generated_files.append(p1)

    # --------------------------------------------------------------------------
    # Plot 2: Average Cumulative Episode Reward Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    rewards_exp = [ablation_data[k]["eval_expert"]["average_reward"] for k in cond_keys]
    rew_err_low = [rewards_exp[i] - ablation_data[k]["eval_expert"]["reward_ci_95"][0] for i, k in enumerate(cond_keys)]
    rew_err_high = [ablation_data[k]["eval_expert"]["reward_ci_95"][1] - rewards_exp[i] for i, k in enumerate(cond_keys)]
    rew_yerr = [rew_err_low, rew_err_high]

    bars = ax.bar(x_pos, rewards_exp, width=0.5, yerr=rew_yerr, capsize=5, color=bar_cols, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_title("Ablation Study: Mean Cumulative Episode Reward vs Expert", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Cumulative Episode Reward", fontsize=11, labelpad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y + (0.25 if y >= 0 else -0.25), f"{y:.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    plt.tight_layout()
    p2 = output_dir / "ablation_reward_comparison.png"
    plt.savefig(p2)
    plt.close()
    generated_files.append(p2)

    # --------------------------------------------------------------------------
    # Plot 3: Training Loss Convergence Trajectories
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    for cond, k, col in active_pairs:
        loss_hist = ablation_data[k]["training"].get("loss_history", [])
        if loss_hist:
            x_steps = np.arange(1, len(loss_hist) + 1)
            # Apply rolling window smoothing for clear curves
            if len(loss_hist) > 5:
                window = min(5, len(loss_hist) // 3)
                smoothed = np.convolve(loss_hist, np.ones(window) / window, mode="valid")
                ax.plot(x_steps[window - 1:], smoothed, label=cond, color=col, linewidth=2.0)
            else:
                ax.plot(x_steps, loss_hist, label=cond, color=col, linewidth=2.0, marker="o")

    ax.set_title("Training Loss Convergence Across Ablation Variants", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Updates", fontsize=11, labelpad=8)
    ax.set_ylabel("Loss (Huber / Smooth L1)", fontsize=11, labelpad=8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = output_dir / "ablation_training_loss_curves.png"
    plt.savefig(p3)
    plt.close()
    generated_files.append(p3)

    # --------------------------------------------------------------------------
    # Plot 4: Illegal Action Attempt Rates (Safety Analysis)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    illegal_rates = [ablation_data[k]["eval_expert"].get("illegal_action_rate", 0.0) * 100.0 for k in cond_keys]

    bars = ax.bar(x_pos, illegal_rates, width=0.48, color=["#2ca02c" if r == 0.0 else "#d62728" for r in illegal_rates], edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_title("Action Legality Invariant: Illegal Action Attempt Rates", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Illegal Move Attempts (%)", fontsize=10, labelpad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cond_labels, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.8, f"{y:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p4 = output_dir / "ablation_illegal_action_rates.png"
    plt.savefig(p4)
    plt.close()
    generated_files.append(p4)

    # --------------------------------------------------------------------------
    # Plot 5: Net Win Rate Impact Waterfall vs Baseline (Percentage Points)
    # --------------------------------------------------------------------------
    abl_keys = [k for k in cond_keys if k != "baseline"]
    if abl_keys:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
        abl_labels = [cond_labels[cond_keys.index(k)] for k in abl_keys]
        base_wr_val = ablation_data["baseline"]["eval_expert"]["win_rate"] * 100.0
        delta_wrs = [(ablation_data[k]["eval_expert"]["win_rate"] * 100.0 - base_wr_val) for k in abl_keys]
        impact_cols = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_wrs]

        x_abl = np.arange(len(abl_labels))
        bars = ax.bar(x_abl, delta_wrs, width=0.45, color=impact_cols, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.9)
        ax.set_title("Win Rate Impact vs Full Baseline (Δ Percentage Points vs Expert)", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Δ Win Rate (Percentage Points)", fontsize=10, labelpad=8)
        ax.set_xticks(x_abl)
        ax.set_xticklabels(abl_labels, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        for bar in bars:
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y + (0.5 if y >= 0 else -0.5), f"{y:+.1f}%", ha="center", va=va, fontsize=9.5, fontweight="bold")

        plt.tight_layout()
        p5 = output_dir / "ablation_degradation_waterfall.png"
        plt.savefig(p5)
        plt.close()
        generated_files.append(p5)

    return generated_files
