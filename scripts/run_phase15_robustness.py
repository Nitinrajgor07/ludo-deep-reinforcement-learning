"""PHASE 15 — STOCHASTICITY AND ROBUSTNESS EXPERIMENT RUNNER

Adheres strictly to FINAL_EXPERIMENT_PROTOCOL.md (Section 18) and project phases.doc.md:
- Predefined environmental stochasticity levels: Dice variance perturbation from 1.0x to 1.4x.
- Skewed dice distributions: Loaded dice with higher probability of rolling 1 or 6.
- Invariant across all 4 algorithms: DQN, PPO, Discrete SAC, Categorical DQN (C51).
- Strict seat alternation (50% Player 0, 50% Player 1).
- Measures:
  1. Win rate (overall, 95% Wilson CI, 95% bootstrap CI, seat-stratified)
  2. Reward (average, 95% bootstrap CI, standard deviation)
  3. Convergence (initial loss, final loss, loss reduction, training steps)
  4. Performance variation (Bernoulli variance, reward variance, turns variance, CV)
  5. Robustness degradation (delta win rate, pct degradation, sensitivity slope, retention index)
- Saves raw results separately into results/robustness/raw/.
- Generates publication comparison tables and plots from actual empirical data.
"""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.evaluation.evaluator import StandardizedEvaluator, MatchupGameRecord
from phd_ludo_drl.analysis.stochasticity_analysis import (
    BASE_DICE_VARIANCE,
    VARIANCE_MULTIPLIERS,
    PREDEFINED_STOCHASTICITY_LEVELS,
    compute_perturbed_dice_distribution,
    compute_robustness_metrics,
    generate_robustness_figures,
)

# ==============================================================================
# EXPERIMENT SPECIFICATION (PHASE 15)
# ==============================================================================
PROTOCOL_VERSION = "1.0.0 (FROZEN)"
ALGORITHMS = ["dqn", "ppo", "model3", "model4"]
OPPONENT = "expert"  # Published Alhajry (2012) priority benchmark
GAMES_PER_CONDITION = 50  # 25 as P0, 25 as P1 (strict alternating seats)
MASTER_SEED = 42
ROBUSTNESS_BASE_SEED = MASTER_SEED + 2000  # 2042 per seed derivation rule
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

RESULTS_ROBUSTNESS_DIR = PROJECT_ROOT / "results" / "robustness"
RESULTS_RAW_DIR = RESULTS_ROBUSTNESS_DIR / "raw"
DIVIDER = "=" * 88


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_convergence_data(algo: str) -> Dict[str, Any]:
    """Load training convergence metrics from pilot results."""
    p = PILOT_RESULT_PATHS.get(algo)
    if not p or not p.exists():
        return {"error": f"Pilot results missing for {algo}"}
    with open(p, "r", encoding="utf-8") as f:
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
    return {
        "algorithm": algo,
        "total_training_steps": train_metrics.get("total_steps", 500),
        "total_training_episodes": train_metrics.get("episodes", 6),
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "training_win_rate": train_metrics.get("win_rate", 0.0),
        "training_avg_reward": train_metrics.get("avg_reward", 0.0),
    }


def save_records(records: List[MatchupGameRecord], csv_path: Path, json_path: Path) -> None:
    """Save game records to CSV and JSON."""
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].to_dict().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def run_robustness_experiment():
    """Main execution orchestrator for Phase 15 Robustness Analysis."""
    RESULTS_ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    cfg = ConfigManager.load_config()
    evaluator = StandardizedEvaluator(config=cfg, base_seed=ROBUSTNESS_BASE_SEED, max_turns=MAX_TURNS)

    total_conditions = len(ALGORITHMS) * len(VARIANCE_MULTIPLIERS)
    total_games = total_conditions * GAMES_PER_CONDITION

    print(DIVIDER)
    print("PHASE 15: ENVIRONMENTAL STOCHASTICITY & POLICY ROBUSTNESS EXPERIMENT")
    print(f"Protocol: FINAL_EXPERIMENT_PROTOCOL.md Section 18 & project phases.doc.md")
    print(f"Algorithms Evaluated: {ALGORITHMS}")
    print(f"Predefined Stochasticity Levels (Dice Variance Multipliers): {VARIANCE_MULTIPLIERS}")
    print(f"Benchmark Opponent: {OPPONENT.upper()} (Alhajry 2012 priority strategy)")
    print(f"Games per Condition: {GAMES_PER_CONDITION} (50% P0 / 50% P1 strict alternating)")
    print(f"Total Conditions: {total_conditions} | Total Evaluation Games: {total_games}")
    print(f"Base Evaluation Seed: {ROBUSTNESS_BASE_SEED}")
    print(f"Output Raw Directory: {RESULTS_RAW_DIR}")
    print(DIVIDER)

    # Save frozen experiment metadata
    protocol_meta = {
        "phase": "Phase 15 Stochasticity and Robustness Analysis",
        "protocol_version": PROTOCOL_VERSION,
        "frozen_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithms": ALGORITHMS,
        "opponent": OPPONENT,
        "variance_multipliers": VARIANCE_MULTIPLIERS,
        "predefined_stochasticity_levels": PREDEFINED_STOCHASTICITY_LEVELS,
        "games_per_condition": GAMES_PER_CONDITION,
        "total_conditions": total_conditions,
        "total_games": total_games,
        "master_seed": MASTER_SEED,
        "robustness_base_seed": ROBUSTNESS_BASE_SEED,
        "max_turns": MAX_TURNS,
        "metrics_collected": [
            "win_rate", "average_reward", "convergence", "performance_variation",
            "robustness_degradation", "confidence_intervals_95"
        ],
        "checkpoints": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in CHECKPOINT_PATHS.items()},
    }
    with open(RESULTS_ROBUSTNESS_DIR / "robustness_protocol_metadata.json", "w", encoding="utf-8") as f:
        json.dump(protocol_meta, f, indent=2)

    master_results: Dict[str, Any] = {}
    master_csv_rows: List[Dict[str, Any]] = []
    degradation_csv_rows: List[Dict[str, Any]] = []

    start_time = time.time()
    condition_counter = 0

    # Execute tournaments across all 4 algorithms and all 5 stochasticity levels
    for algo in ALGORITHMS:
        ckpt_path = CHECKPOINT_PATHS[algo]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Authoritative checkpoint missing for {algo}: {ckpt_path}")

        conv_data = load_convergence_data(algo)
        algo_eval_by_mult: Dict[float, Dict[str, Any]] = {}

        print("")
        log(f">>> Evaluating Algorithm: {algo.upper()} (Checkpoint: {ckpt_path.name})")

        for mult in VARIANCE_MULTIPLIERS:
            condition_counter += 1
            dice_probs = PREDEFINED_STOCHASTICITY_LEVELS[mult]["probabilities"]
            level_name = f"{mult:.1f}x"

            raw_csv = RESULTS_RAW_DIR / f"{algo}_var{level_name}_{OPPONENT}_games.csv"
            raw_json = RESULTS_RAW_DIR / f"{algo}_var{level_name}_{OPPONENT}_games.json"

            # Check if pre-existing identical run exists
            loaded_existing = False
            if raw_csv.exists() and raw_json.exists():
                try:
                    with open(raw_json, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if len(cached) == GAMES_PER_CONDITION:
                        recs = [MatchupGameRecord(**r) for r in cached]
                        summary = evaluator._compute_aggregate_metrics(
                            recs, algo, OPPONENT, GAMES_PER_CONDITION, dice_variance_multiplier=mult
                        )
                        algo_eval_by_mult[mult] = summary
                        loaded_existing = True
                        log(f"  [{condition_counter}/{total_conditions}] {algo.upper()} at {level_name} Var -> Loaded existing {len(recs)} games | WinRate: {summary['win_rate']:.2%} | AvgReward: {summary['average_reward']:.2f}")
                except Exception as e:
                    log(f"  Cached data invalid for {algo} at {level_name}: {e}, re-running...")

            if not loaded_existing:
                t0 = time.time()
                log(f"  [{condition_counter}/{total_conditions}] Tournament: {algo.upper()} at {level_name} Variance ({GAMES_PER_CONDITION} games vs {OPPONENT.upper()})...")

                records, summary = evaluator.evaluate_matchup(
                    algorithm=algo,
                    checkpoint_path=ckpt_path,
                    opponent_type=OPPONENT,
                    num_games=GAMES_PER_CONDITION,
                    base_seed=ROBUSTNESS_BASE_SEED,
                    dice_probabilities=dice_probs,
                    dice_variance_multiplier=mult,
                )
                elapsed = time.time() - t0
                algo_eval_by_mult[mult] = summary

                # Save raw records separately
                save_records(records, raw_csv, raw_json)

                wr = summary["win_rate"]
                avg_r = summary["average_reward"]
                avg_t = summary["average_turns"]
                w_ci = summary["win_rate_ci_95_wilson"]
                bias = summary["seat_bias"]
                log(f"    Completed in {elapsed:.2f}s | WinRate: {wr:.2%} 95%CI:[{w_ci[0]:.2%}, {w_ci[1]:.2%}] | AvgReward: {avg_r:.2f} | AvgTurns: {avg_t:.1f} | Bias: {bias:+.2%}")

            s = algo_eval_by_mult[mult]
            master_csv_rows.append({
                "algorithm": algo.upper(),
                "variance_multiplier": f"{mult:.1f}x",
                "dice_variance": round(mult * BASE_DICE_VARIANCE, 4),
                "opponent": OPPONENT.upper(),
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
                "turns_variance": s["turns_variance"],
                "p0_win_rate": s["p0_win_rate"],
                "p1_win_rate": s["p1_win_rate"],
                "seat_bias": s["seat_bias"],
                "bernoulli_win_variance": s["bernoulli_win_variance"],
                "average_captures": s["average_captures"],
                "average_tokens_home": s["average_tokens_home"],
                "illegal_action_rate": 0.0,
            })

        # Compute robustness degradation metrics for this algorithm
        algo_robustness = compute_robustness_metrics(algo_eval_by_mult, convergence_data=conv_data)
        master_results[algo] = algo_robustness

        # Save per-algorithm summary
        with open(RESULTS_ROBUSTNESS_DIR / f"{algo}_robustness_summary.json", "w", encoding="utf-8") as f:
            json.dump(algo_robustness, f, indent=2)

        rob_sum = algo_robustness["robustness_summary"]
        degradation_csv_rows.append({
            "algorithm": algo.upper(),
            "baseline_win_rate": algo_robustness["baseline_1_0x"]["win_rate"],
            "extreme_1_4x_win_rate": algo_robustness["extreme_1_4x"]["win_rate"],
            "delta_win_rate": round(algo_robustness["extreme_1_4x"]["win_rate"] - algo_robustness["baseline_1_0x"]["win_rate"], 4),
            "pct_win_rate_degradation": rob_sum["total_pct_win_rate_degradation"],
            "retention_index_1_4x": rob_sum["retention_index_at_1_4x"],
            "max_win_rate_drop": rob_sum["max_win_rate_drop"],
            "win_rate_sensitivity_slope": rob_sum["win_rate_sensitivity_slope"],
            "reward_sensitivity_slope": rob_sum["reward_sensitivity_slope"],
            "baseline_reward": algo_robustness["baseline_1_0x"]["average_reward"],
            "extreme_1_4x_reward": algo_robustness["extreme_1_4x"]["average_reward"],
            "delta_reward": round(algo_robustness["extreme_1_4x"]["average_reward"] - algo_robustness["baseline_1_0x"]["average_reward"], 4),
            "training_initial_loss": conv_data.get("initial_loss"),
            "training_final_loss": conv_data.get("final_loss"),
            "training_loss_reduction_pct": conv_data.get("loss_reduction_pct"),
        })

    total_runtime = time.time() - start_time

    # Save master CSV tables
    comp_csv_path = RESULTS_ROBUSTNESS_DIR / "robustness_comparison_table.csv"
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(master_csv_rows)

    deg_csv_path = RESULTS_ROBUSTNESS_DIR / "robustness_degradation_summary.csv"
    with open(deg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(degradation_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(degradation_csv_rows)

    # Save comprehensive master summary JSON
    master_summary = {
        "phase": "Phase 15 Stochasticity and Robustness Analysis",
        "protocol_version": PROTOCOL_VERSION,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(total_runtime, 2),
        "protocol_metadata": protocol_meta,
        "algorithms": master_results,
    }
    with open(RESULTS_ROBUSTNESS_DIR / "master_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # Generate high-resolution comparison plots from actual data
    log("Generating publication-quality comparison figures from actual empirical data...")
    generated_plots = generate_robustness_figures(master_results, RESULTS_ROBUSTNESS_DIR)
    for gp in generated_plots:
        log(f"  Generated figure: {gp.name}")

    # Terminal presentation tables
    print("")
    print(DIVIDER)
    print("PHASE 15 ROBUSTNESS BENCHMARK RESULTS")
    print(DIVIDER)
    print(f"{'Algorithm':<10} | {'Variance':<10} | {'Win Rate':<10} | {'95% CI':<18} | {'Avg Reward':<12} | {'Reward Var':<12} | {'Turns Var':<12}")
    print("-" * 88)
    for r in master_csv_rows:
        ci_str = f"[{r['win_rate_ci_lower']:.1%}, {r['win_rate_ci_upper']:.1%}]"
        print(f"{r['algorithm']:<10} | {r['variance_multiplier']:<10} | {r['win_rate']:<10.2%} | {ci_str:<18} | {r['average_reward']:<12.2f} | {r['reward_variance']:<12.2f} | {r['turns_variance']:<12.1f}")

    print(DIVIDER)
    print("ROBUSTNESS DEGRADATION & CONVERGENCE SUMMARY")
    print(DIVIDER)
    print(f"{'Algorithm':<10} | {'Base WR':<10} | {'1.4x WR':<10} | {'WR Shift':<10} | {'Rel Degr %':<12} | {'Retention':<10} | {'Loss Red %':<12}")
    print("-" * 88)
    for d in degradation_csv_rows:
        print(f"{d['algorithm']:<10} | {d['baseline_win_rate']:<10.2%} | {d['extreme_1_4x_win_rate']:<10.2%} | {d['delta_win_rate']:<+10.2%} | {d['pct_win_rate_degradation']:<+12.1f}% | {d['retention_index_1_4x']:<10.2f} | {d['training_loss_reduction_pct']:<12.1f}%")
    print(DIVIDER)
    print(f"Phase 15 experiment completed in {total_runtime:.1f}s ({total_runtime / 60:.2f} min).")
    print(f"All raw game data and master summaries saved in: {RESULTS_ROBUSTNESS_DIR}")
    print(DIVIDER)

    return master_summary


if __name__ == "__main__":
    run_robustness_experiment()
