"""PHASE 14 — FINAL EVALUATION RUNNER

Evaluates all four trained algorithms using the frozen evaluation protocol:
- Model 1: Double Dueling DQN
- Model 2: PPO
- Model 3: Discrete SAC
- Model 4: Categorical DQN (C51)

For every algorithm evaluates against:
1. Random opponent
2. Expert opponent (Alhajry 2012 priority strategy)
3. Self-play (Agent vs Cloned Self)

Collects:
- win rate (overall, P0, P1, seat bias)
- average reward (and variance, 95% bootstrap CI)
- episode length (mean turns, std, min, max, median)
- performance variance (Bernoulli variance, reward variance, turn variance)
- convergence information (losses, updates, learning curves, final params)
- sample efficiency metrics (win rate / 1k steps, reward / 1k steps)

Stores all machine-readable files into:
results/raw/
"""

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.evaluation.evaluator import StandardizedEvaluator, MatchupGameRecord

# ==============================================================================
# FROZEN EVALUATION PROTOCOL SPECIFICATION (PHASE 14)
# ==============================================================================
PROTOCOL_VERSION = "1.0.0 (FROZEN)"
ALGORITHMS = ["dqn", "ppo", "model3", "model4"]
OPPONENTS = ["random", "expert", "self_play"]
GAMES_PER_MATCHUP = 250  # 125 as P0, 125 as P1 (seat-alternating)
MASTER_SEED = 42
EVAL_BASE_SEED = MASTER_SEED + 1000  # 1042 per Protocol Section 12
MAX_TURNS = 1000

CHECKPOINT_PATHS = {
    "dqn": PROJECT_ROOT / "experiments" / "phase11_pilots" / "dqn" / "checkpoints" / "pilot_checkpoint.pt",
    "ppo": PROJECT_ROOT / "experiments" / "phase11_pilots" / "ppo" / "checkpoints" / "pilot_checkpoint.pt",
    "model3": PROJECT_ROOT / "experiments" / "phase11_pilots" / "model3" / "checkpoints" / "pilot_checkpoint.pt",
    "model4": PROJECT_ROOT / "experiments" / "phase11_pilots" / "model4" / "checkpoints" / "pilot_checkpoint.pt",
}

PILOT_RESULT_PATHS = {
    "dqn": PROJECT_ROOT / "experiments" / "phase11_pilots" / "dqn" / "pilot_results.json",
    "ppo": PROJECT_ROOT / "experiments" / "phase11_pilots" / "ppo" / "pilot_results.json",
    "model3": PROJECT_ROOT / "experiments" / "phase11_pilots" / "model3" / "pilot_results.json",
    "model4": PROJECT_ROOT / "experiments" / "phase11_pilots" / "model4" / "pilot_results.json",
}

RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"
DIVIDER = "=" * 88


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compute_two_proportion_z_test(w1: int, n1: int, w2: int, n2: int) -> Tuple[float, float]:
    """Perform two-proportion z-test comparing win rates."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1 = w1 / n1
    p2 = w2 / n2
    p_pooled = (w1 + w2) / (n1 + n2)
    if p_pooled == 0.0 or p_pooled == 1.0:
        return (0.0, 1.0)
    se_pooled = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pooled
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return (float(z), float(p_val))


def compute_cohens_d(x: List[float], y: List[float]) -> float:
    """Compute Cohen's d effect size for two independent samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0.0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def extract_convergence_metrics(algo: str) -> Dict[str, Any]:
    """Extract convergence metrics directly from training pilot logs."""
    path = PILOT_RESULT_PATHS.get(algo)
    if not path or not path.exists():
        return {"error": f"Pilot results missing for {algo}"}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_metrics = data.get("training_metrics", {})
    loss_hist = data.get("loss_history", [])

    init_loss = float(loss_hist[0]) if loss_hist else None
    final_loss = float(train_metrics.get("final_loss", loss_hist[-1] if loss_hist else 0.0))
    loss_reduction_pct = (
        round(((init_loss - final_loss) / abs(init_loss)) * 100.0, 2)
        if init_loss and abs(init_loss) > 1e-8
        else 0.0
    )

    conv_info: Dict[str, Any] = {
        "algorithm": algo,
        "total_training_steps": train_metrics.get("total_steps", 0),
        "total_training_episodes": train_metrics.get("episodes", 0),
        "training_win_rate": train_metrics.get("win_rate", 0.0),
        "training_avg_reward": train_metrics.get("avg_reward", 0.0),
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "loss_history_length": len(loss_hist),
        "training_runtime_seconds": data.get("runtime_seconds", 0.0),
        "peak_memory_mb": data.get("peak_memory_mb", 0.0),
        "updates": train_metrics.get("updates", len(loss_hist)),
    }

    # Algorithm-specific metadata
    if algo == "dqn":
        conv_info["final_epsilon"] = train_metrics.get("final_epsilon", None)
    elif algo == "ppo":
        conv_info["final_policy_loss"] = train_metrics.get("final_policy_loss", None)
        conv_info["final_value_loss"] = train_metrics.get("final_value_loss", None)
        conv_info["final_entropy"] = train_metrics.get("final_entropy", None)
    elif algo == "model3":
        conv_info["final_critic_loss"] = train_metrics.get("final_critic_loss", None)
        conv_info["final_actor_loss"] = train_metrics.get("final_actor_loss", None)
        conv_info["final_alpha"] = train_metrics.get("final_alpha", None)
    elif algo == "model4":
        conv_info["final_epsilon"] = train_metrics.get("final_epsilon", None)
        conv_info["num_atoms"] = 51

    return conv_info


def save_records_to_csv(records: List[MatchupGameRecord], filepath: Path) -> None:
    """Save raw game records to CSV."""
    if not records:
        return
    fieldnames = list(records[0].to_dict().keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def save_records_to_json(records: List[MatchupGameRecord], filepath: Path) -> None:
    """Save raw game records to JSON."""
    data = [r.to_dict() for r in records]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_final_evaluation():
    """Main evaluation orchestrator adhering strictly to Phase 14 directives."""
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ConfigManager.load_config()
    evaluator = StandardizedEvaluator(config=cfg, base_seed=EVAL_BASE_SEED, max_turns=MAX_TURNS)

    print(DIVIDER)
    print("PHASE 14: FINAL EVALUATION CAMPAIGN")
    print(f"Protocol: FINAL_EXPERIMENT_PROTOCOL.md Version {PROTOCOL_VERSION}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Opponents: {OPPONENTS}")
    print(f"Games per Matchup: {GAMES_PER_MATCHUP} (50% P0 / 50% P1 alternating)")
    print(f"Total Matchups: {len(ALGORITHMS) * len(OPPONENTS)}  Total Games: {len(ALGORITHMS) * len(OPPONENTS) * GAMES_PER_MATCHUP}")
    print(f"Evaluation Seed Base: {EVAL_BASE_SEED} (Deterministic Range: {EVAL_BASE_SEED}..{EVAL_BASE_SEED + GAMES_PER_MATCHUP - 1})")
    print(f"Storage Directory: {RESULTS_RAW_DIR}")
    print(DIVIDER)

    # 1. Freeze and save evaluation protocol metadata
    protocol_meta = {
        "phase": "Phase 14 Final Evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithms": ALGORITHMS,
        "opponents": OPPONENTS,
        "games_per_matchup": GAMES_PER_MATCHUP,
        "master_seed": MASTER_SEED,
        "eval_base_seed": EVAL_BASE_SEED,
        "max_turns": MAX_TURNS,
        "seat_balance": "50% Player 0, 50% Player 1 (strict alternating)",
        "deterministic_action_selection": True,
        "illegal_action_handling": "Masked to -inf (0.00% illegal actions)",
        "metrics_collected": [
            "win_rate", "average_reward", "episode_length", "performance_variance",
            "confidence_intervals_95", "convergence_metrics", "sample_efficiency_metrics"
        ],
        "checkpoints_evaluated": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in CHECKPOINT_PATHS.items()},
    }
    with open(RESULTS_RAW_DIR / "evaluation_protocol_frozen.json", "w", encoding="utf-8") as f:
        json.dump(protocol_meta, f, indent=2)
    log("Frozen evaluation protocol metadata saved: evaluation_protocol_frozen.json")

    # Store all results in memory for master aggregation
    all_summaries: Dict[str, Dict[str, Any]] = {algo: {} for algo in ALGORITHMS}
    all_game_records: Dict[str, Dict[str, List[MatchupGameRecord]]] = {algo: {} for algo in ALGORITHMS}
    convergence_info: Dict[str, Dict[str, Any]] = {}
    sample_efficiency_info: Dict[str, Dict[str, Any]] = {}

    matchup_idx = 0
    total_matchups = len(ALGORITHMS) * len(OPPONENTS)
    eval_start_time = time.time()

    # 2. Iterate through each algorithm and opponent
    for algo in ALGORITHMS:
        ckpt_path = CHECKPOINT_PATHS[algo]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Authoritative checkpoint missing for {algo}: {ckpt_path}")

        # Extract convergence info
        conv_data = extract_convergence_metrics(algo)
        convergence_info[algo] = conv_data

        print("")
        log(f">>> Evaluating Algorithm: {algo.upper()} (Checkpoint: {ckpt_path.name})")

        for opp in OPPONENTS:
            matchup_idx += 1
            games_json_path = RESULTS_RAW_DIR / f"{algo}_vs_{opp}_games.json"
            games_csv_path = RESULTS_RAW_DIR / f"{algo}_vs_{opp}_games.csv"

            # Check if valid existing evaluation records exist
            loaded_existing = False
            if games_json_path.exists() and games_csv_path.exists():
                try:
                    with open(games_json_path, "r", encoding="utf-8") as f:
                        cached_raw = json.load(f)
                    if len(cached_raw) == GAMES_PER_MATCHUP:
                        records = [MatchupGameRecord(**r) for r in cached_raw]
                        summary = evaluator._compute_aggregate_metrics(records, algo, opp, GAMES_PER_MATCHUP)
                        all_game_records[algo][opp] = records
                        all_summaries[algo][opp] = summary
                        loaded_existing = True
                        wr = summary["win_rate"]
                        avg_r = summary["average_reward"]
                        avg_t = summary["average_turns"]
                        bias = summary["seat_bias"]
                        w_ci = summary["win_rate_ci_95_wilson"]
                        log(f"  [{matchup_idx}/{total_matchups}] Tournament: {algo.upper()} vs {opp.upper()} -> Loaded existing {len(records)} games | WinRate: {wr:.2%} | AvgReward: {avg_r:.2f} | AvgTurns: {avg_t:.1f}")
                except Exception as e:
                    log(f"  Existing records invalid for {algo} vs {opp}: {e}, re-running...")

            if not loaded_existing:
                log(f"  [{matchup_idx}/{total_matchups}] Tournament: {algo.upper()} vs {opp.upper()} ({GAMES_PER_MATCHUP} games)...")
                t0 = time.time()

                records, summary = evaluator.evaluate_matchup(
                    algorithm=algo,
                    checkpoint_path=ckpt_path,
                    opponent_type=opp,
                    num_games=GAMES_PER_MATCHUP,
                    base_seed=EVAL_BASE_SEED,
                )
                elapsed = time.time() - t0

                # Store records and summary
                all_game_records[algo][opp] = records
                all_summaries[algo][opp] = summary

                # Save raw game records (JSON and CSV)
                save_records_to_json(records, games_json_path)
                save_records_to_csv(records, games_csv_path)

                wr = summary["win_rate"]
                avg_r = summary["average_reward"]
                avg_t = summary["average_turns"]
                bias = summary["seat_bias"]
                w_ci = summary["win_rate_ci_95_wilson"]
                log(f"    Completed in {elapsed:.2f}s | WinRate: {wr:.2%} 95%CI:[{w_ci[0]:.2%}, {w_ci[1]:.2%}] | AvgReward: {avg_r:.2f} | AvgTurns: {avg_t:.1f} | SeatBias: {bias:+.2%}")

        # Compute sample efficiency metrics for this algorithm
        train_steps = conv_data.get("total_training_steps", 500)
        random_wr = all_summaries[algo]["random"]["win_rate"]
        random_reward = all_summaries[algo]["random"]["average_reward"]
        expert_wr = all_summaries[algo]["expert"]["win_rate"]

        sample_eff = {
            "algorithm": algo,
            "training_steps": train_steps,
            "training_episodes": conv_data.get("total_training_episodes", 6),
            "training_runtime_seconds": conv_data.get("training_runtime_seconds", 0.0),
            "eval_win_rate_vs_random": random_wr,
            "eval_win_rate_vs_expert": expert_wr,
            "eval_avg_reward_vs_random": random_reward,
            "win_rate_per_1k_steps": round(random_wr / (train_steps / 1000.0), 4) if train_steps > 0 else 0.0,
            "reward_per_1k_steps": round(random_reward / (train_steps / 1000.0), 4) if train_steps > 0 else 0.0,
            "step_sample_efficiency": round(random_wr / train_steps, 6) if train_steps > 0 else 0.0,
            "wallclock_sample_efficiency": round(random_wr / conv_data.get("training_runtime_seconds", 1.0), 4) if conv_data.get("training_runtime_seconds", 0.0) > 0 else 0.0,
        }
        sample_efficiency_info[algo] = sample_eff

        # Save per-algorithm summary JSON
        algo_summary = {
            "algorithm": algo,
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "protocol_version": PROTOCOL_VERSION,
            "matchups": all_summaries[algo],
            "convergence": conv_data,
            "sample_efficiency": sample_eff,
        }
        with open(RESULTS_RAW_DIR / f"{algo}_evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(algo_summary, f, indent=2)
        log(f"Saved algorithm summary: {algo}_evaluation_summary.json")

    total_eval_time = time.time() - eval_start_time

    # 3. Perform Pairwise Statistical Hypothesis Testing across all 4 models
    log("Computing pairwise statistical hypothesis tests...")
    statistical_tests: Dict[str, Any] = {"vs_random": {}, "vs_expert": {}}

    for opp_key in ["random", "expert"]:
        for i in range(len(ALGORITHMS)):
            for j in range(i + 1, len(ALGORITHMS)):
                a1 = ALGORITHMS[i]
                a2 = ALGORITHMS[j]
                pair_name = f"{a1.upper()}_vs_{a2.upper()}"

                rec1 = all_game_records[a1][opp_key]
                rec2 = all_game_records[a2][opp_key]

                rew1 = [r.agent_reward for r in rec1]
                rew2 = [r.agent_reward for r in rec2]
                w1 = sum(1 for r in rec1 if r.agent_won)
                w2 = sum(1 for r in rec2 if r.agent_won)
                n1, n2 = len(rec1), len(rec2)

                # Welch's t-test (rewards)
                t_stat, t_pval = stats.ttest_ind(rew1, rew2, equal_var=False)

                # Wilcoxon rank-sum / Mann-Whitney U test (rewards)
                u_stat, u_pval = stats.mannwhitneyu(rew1, rew2, alternative="two-sided")

                # Cohen's d effect size (rewards)
                cohen_d = compute_cohens_d(rew1, rew2)

                # Two-proportion z-test (win rates)
                z_stat, z_pval = compute_two_proportion_z_test(w1, n1, w2, n2)

                statistical_tests[f"vs_{opp_key}"][pair_name] = {
                    "algorithm_1": a1,
                    "algorithm_2": a2,
                    "opponent": opp_key,
                    "n1": n1,
                    "n2": n2,
                    "win_rate_1": round(w1 / n1, 4),
                    "win_rate_2": round(w2 / n2, 4),
                    "reward_mean_1": round(float(np.mean(rew1)), 4),
                    "reward_mean_2": round(float(np.mean(rew2)), 4),
                    "welch_t_stat": round(float(t_stat), 4),
                    "welch_t_pvalue": float(t_pval),
                    "mann_whitney_u_stat": round(float(u_stat), 4),
                    "mann_whitney_u_pvalue": float(u_pval),
                    "cohens_d": round(cohen_d, 4),
                    "prop_z_stat": round(z_stat, 4),
                    "prop_z_pvalue": z_pval,
                    "significant_alpha_005": bool(t_pval < 0.05),
                }

    with open(RESULTS_RAW_DIR / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(statistical_tests, f, indent=2)
    log("Saved statistical hypothesis tests: statistical_tests.json")

    # 4. Generate master comparative CSV table
    comparative_csv_path = RESULTS_RAW_DIR / "comparative_metrics.csv"
    csv_rows = []
    for algo in ALGORITHMS:
        for opp in OPPONENTS:
            s = all_summaries[algo][opp]
            csv_rows.append({
                "algorithm": algo.upper(),
                "opponent": opp.upper(),
                "total_games": s["total_games"],
                "agent_wins": s["agent_wins"],
                "opponent_wins": s["opponent_wins"],
                "draws": s["draws"],
                "win_rate": s["win_rate"],
                "win_rate_ci_lower": s["win_rate_ci_95_wilson"][0],
                "win_rate_ci_upper": s["win_rate_ci_95_wilson"][1],
                "average_reward": s["average_reward"],
                "reward_std": s["reward_std"],
                "reward_variance": s["reward_variance"],
                "reward_ci_lower": s["reward_ci_95"][0],
                "reward_ci_upper": s["reward_ci_95"][1],
                "average_turns": s["average_turns"],
                "turns_std": s["turns_std"],
                "turns_variance": s["turns_variance"],
                "p0_win_rate": s["p0_win_rate"],
                "p1_win_rate": s["p1_win_rate"],
                "seat_bias": s["seat_bias"],
                "average_captures": s["average_captures"],
                "average_tokens_home": s["average_tokens_home"],
                "bernoulli_win_variance": s["bernoulli_win_variance"],
                "illegal_action_rate": s["illegal_action_rate"],
            })

    fieldnames = list(csv_rows[0].keys())
    with open(comparative_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log("Saved comparative CSV table: comparative_metrics.csv")

    # 5. Save master comprehensive JSON
    master_summary = {
        "phase": "Phase 14 Final Evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(total_eval_time, 2),
        "protocol_metadata": protocol_meta,
        "benchmarks": all_summaries,
        "convergence": convergence_info,
        "sample_efficiency": sample_efficiency_info,
        "statistical_tests": statistical_tests,
    }
    with open(RESULTS_RAW_DIR / "master_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    log("Saved master evaluation summary: master_evaluation_summary.json")

    # 6. Final Terminal Presentation
    print("")
    print(DIVIDER)
    print("PHASE 14 FINAL BENCHMARK SUMMARY")
    print(DIVIDER)
    print(f"{'Algorithm':<10} | {'Opponent':<10} | {'Win Rate':<10} | {'95% CI (Wilson)':<18} | {'Avg Reward':<12} | {'Avg Turns':<10} | {'Bias (P0-P1)':<12}")
    print("-" * 88)
    for row in csv_rows:
        ci_str = f"[{row['win_rate_ci_lower']:.1%}, {row['win_rate_ci_upper']:.1%}]"
        print(f"{row['algorithm']:<10} | {row['opponent']:<10} | {row['win_rate']:<10.2%} | {ci_str:<18} | {row['average_reward']:<12.2f} | {row['average_turns']:<10.1f} | {row['seat_bias']:<+12.2%}")
    print(DIVIDER)
    print(f"Total evaluation completed in {total_eval_time:.1f}s ({total_eval_time / 60:.2f} minutes).")
    print(f"Machine-readable results written to {RESULTS_RAW_DIR}")
    print(DIVIDER)
    return master_summary


if __name__ == "__main__":
    run_final_evaluation()
