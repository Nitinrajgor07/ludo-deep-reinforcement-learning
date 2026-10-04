"""PHASE 13 — LARGE-SCALE FINAL TRAINING RUNNER

Adheres strictly to FINAL_EXPERIMENT_PROTOCOL.md (Version 1.0.0 FROZEN).

Executes primary benchmark training runs across all 4 DRL models:
1. DQN (Double Dueling + Prioritized Experience Replay)
2. PPO (Actor-Critic with Action Masking & GAE)
3. Model 3 (Discrete Soft Actor-Critic)
4. Model 4 (Categorical DQN / C51 Distributional RL)

Across 5 independent master random seeds:
S = [42, 101, 202, 303, 404]
Total primary runs: 4 models x 5 seeds = 20 runs.
Training budget per run: 500,000 steps.

For every run records:
- experiment ID
- algorithm
- seed
- configuration
- training steps
- episode count
- reward
- loss where applicable
- runtime
- checkpoint
- training logs
"""

import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.training.unified_trainer import UnifiedTrainingRunner

ALGORITHMS = ["dqn", "ppo", "model3", "model4"]
SEEDS = [42, 101, 202, 303, 404]
TOTAL_STEPS = 500_000
EVAL_INTERVAL = 10_000
EVAL_GAMES_INTERVAL = 100
OPPONENT_TRAINING = "random"
DESCRIPTION_TEMPLATE = "Phase 13 Final Training: {algo} Seed {seed}"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUMMARY_FILE = EXPERIMENTS_DIR / "phase13_training_summary.json"
DIVIDER = "=" * 80


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def is_run_complete(experiments_dir: Path, algo: str, seed: int, target_steps: int = TOTAL_STEPS):
    """Check if a run for given algo and seed with at least target_steps already exists."""
    algo_tag = algo.upper()
    matching = sorted(experiments_dir.glob(f"EXP_{algo_tag}_*_{seed}_*"))
    for run_dir in matching:
        summary_path = run_dir / "results" / "summary.json"
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    "training_metrics" in data
                    and "evaluation_metrics" in data
                    and data["training_metrics"].get("total_steps", 0) >= target_steps
                ):
                    return True, run_dir.name, data
            except Exception:
                pass
    return False, None, None


def save_summary_file(summary_data: dict) -> None:
    """Safely persist training summary to disk."""
    temp_path = SUMMARY_FILE.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    temp_path.replace(SUMMARY_FILE)


def run_training_campaign():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ConfigManager.load_config()
    runner = UnifiedTrainingRunner(base_dir=PROJECT_ROOT)
    total_runs = len(ALGORITHMS) * len(SEEDS)
    completed_runs = []
    skipped_runs = []
    failed_runs = []
    campaign_start = time.time()
    campaign_start_utc = datetime.now(timezone.utc).isoformat()
    run_index = 0

    print(DIVIDER)
    print("PHASE 13: LARGE-SCALE FINAL TRAINING CAMPAIGN")
    print("Specification: FINAL_EXPERIMENT_PROTOCOL.md (Version 1.0.0 FROZEN)")
    print(f"Algorithms: {[a.upper() for a in ALGORITHMS]}")
    print(f"Master Seeds: {SEEDS}")
    print(f"Steps per Run: {TOTAL_STEPS:,} | Eval Interval: {EVAL_INTERVAL:,} | Total Planned Runs: {total_runs}")
    print(DIVIDER)

    for algo in ALGORITHMS:
        for seed in SEEDS:
            run_index += 1
            run_label = f"[{run_index}/{total_runs}] {algo.upper()} | Seed {seed}"
            print("")
            log(f"Inspecting run: {run_label}")

            complete, existing_id, existing_data = is_run_complete(EXPERIMENTS_DIR, algo, seed, target_steps=TOTAL_STEPS)
            if complete:
                log(f"  SKIP: Run already fully completed -> {existing_id}")
                skipped_runs.append({
                    "experiment_id": existing_id,
                    "algorithm": algo,
                    "seed": seed,
                    "status": "skipped_already_completed",
                    "training_steps": existing_data["training_metrics"]["total_steps"],
                    "episode_count": existing_data["training_metrics"]["episodes"],
                    "reward": existing_data["evaluation_metrics"]["avg_p0_reward"],
                    "checkpoint": existing_data.get("artifacts", {}).get("checkpoint"),
                    "training_logs": existing_data.get("artifacts", {}).get("log_dir"),
                })
                continue

            description = DESCRIPTION_TEMPLATE.format(algo=algo.upper(), seed=seed)
            log(f"  STARTING TRAINING: {run_label} ({TOTAL_STEPS:,} steps)")
            run_start = time.time()

            try:
                result = runner.run_experiment(
                    algorithm=algo,
                    seed=seed,
                    total_steps=TOTAL_STEPS,
                    opponent_type=OPPONENT_TRAINING,
                    eval_interval=EVAL_INTERVAL,
                    eval_games=EVAL_GAMES_INTERVAL,
                    config=cfg,
                    description=description,
                    device=None,
                )
                elapsed = time.time() - run_start
                exp_id = result["metadata"]["experiment_id"]
                win_rate = result["evaluation_metrics"]["p0_win_rate"]
                steps = result["training_metrics"]["total_steps"]
                episodes = result["training_metrics"]["episodes"]
                avg_eval_reward = result["evaluation_metrics"]["avg_p0_reward"]
                avg_train_reward = result["training_metrics"].get("avg_reward", 0.0)
                final_loss = result["training_metrics"].get("final_loss", None)

                log(
                    f"  COMPLETED: {exp_id} | Steps={steps:,} | Episodes={episodes} | "
                    f"Eval Win Rate={win_rate:.2%} | Avg Reward={avg_eval_reward:.2f} | Time={elapsed:.1f}s"
                )

                run_record = {
                    "experiment_id": exp_id,
                    "algorithm": algo,
                    "seed": seed,
                    "configuration": cfg.to_dict(),
                    "training_steps": steps,
                    "episode_count": episodes,
                    "reward": avg_eval_reward,
                    "training_avg_reward": avg_train_reward,
                    "loss": final_loss,
                    "runtime_seconds": round(elapsed, 2),
                    "checkpoint": result["artifacts"]["checkpoint"],
                    "training_logs": result["artifacts"]["log_dir"],
                    "status": "completed",
                    "eval_win_rate_vs_random": win_rate,
                    "avg_turns": result["evaluation_metrics"]["avg_turns"],
                }
                completed_runs.append(run_record)

            except Exception as exc:
                elapsed = time.time() - run_start
                log(f"  FAILED after {elapsed:.1f}s: {exc}")
                traceback.print_exc()
                failed_runs.append({
                    "experiment_id": f"FAILED_{algo.upper()}_{seed}_{int(run_start)}",
                    "algorithm": algo,
                    "seed": seed,
                    "configuration": cfg.to_dict(),
                    "training_steps": 0,
                    "episode_count": 0,
                    "reward": None,
                    "loss": None,
                    "runtime_seconds": round(elapsed, 2),
                    "checkpoint": None,
                    "training_logs": None,
                    "status": "failed",
                    "reason_for_failure": str(exc),
                })

            # Incrementally persist campaign summary
            current_total_elapsed = time.time() - campaign_start
            interim_summary = {
                "phase": "Phase 13 Large-Scale Final Training",
                "protocol_version": "1.0.0 (FROZEN)",
                "campaign_start_utc": campaign_start_utc,
                "latest_update_utc": datetime.now(timezone.utc).isoformat(),
                "total_runtime_seconds": round(current_total_elapsed, 2),
                "planned_runs": total_runs,
                "completed_count": len(completed_runs),
                "skipped_count": len(skipped_runs),
                "failed_count": len(failed_runs),
                "campaign_status": "IN_PROGRESS",
                "completed_runs": completed_runs,
                "skipped_runs": skipped_runs,
                "failed_runs": failed_runs,
            }
            save_summary_file(interim_summary)

    total_elapsed = time.time() - campaign_start

    # Compile final list of all completed runs
    all_complete = []
    for algo in ALGORITHMS:
        for seed in SEEDS:
            complete, exp_id, data = is_run_complete(EXPERIMENTS_DIR, algo, seed, target_steps=TOTAL_STEPS)
            if complete:
                all_complete.append({
                    "experiment_id": exp_id,
                    "algorithm": algo,
                    "seed": seed,
                    "configuration": data.get("config", {}),
                    "training_steps": data["training_metrics"]["total_steps"],
                    "episode_count": data["training_metrics"]["episodes"],
                    "reward": data["evaluation_metrics"]["avg_p0_reward"],
                    "training_avg_reward": data["training_metrics"].get("avg_reward", 0.0),
                    "loss": data["training_metrics"].get("final_loss", None),
                    "runtime_seconds": data.get("runtime_seconds", None),
                    "checkpoint": data.get("artifacts", {}).get("checkpoint"),
                    "training_logs": data.get("artifacts", {}).get("log_dir"),
                    "eval_win_rate_vs_random": data["evaluation_metrics"]["p0_win_rate"],
                    "avg_turns": data["evaluation_metrics"]["avg_turns"],
                })

    status = "COMPLETE" if len(all_complete) == total_runs else ("FAILED" if failed_runs and not completed_runs else "PARTIAL")

    final_summary = {
        "phase": "Phase 13 Large-Scale Final Training",
        "protocol_version": "1.0.0 (FROZEN)",
        "campaign_start_utc": campaign_start_utc,
        "campaign_end_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(total_elapsed, 2),
        "planned_runs": total_runs,
        "total_complete_runs": len(all_complete),
        "completed_this_session": len(completed_runs),
        "skipped_already_done": len(skipped_runs),
        "failed_this_session": len(failed_runs),
        "campaign_status": status,
        "all_completed_runs": all_complete,
        "this_session_completed": completed_runs,
        "skipped_runs": skipped_runs,
        "failed_runs": failed_runs,
    }

    save_summary_file(final_summary)

    print("")
    print(DIVIDER)
    print("PHASE 13 CAMPAIGN SUMMARY")
    print(DIVIDER)
    print(f"Status: {status}")
    print(f"Completed: {len(all_complete)}/{total_runs} (This session: {len(completed_runs)} done, {len(skipped_runs)} skipped, {len(failed_runs)} failed)")
    print(f"Total Runtime: {total_elapsed / 60:.2f} minutes")
    print("Per-Algorithm Progress:")
    for algo in ALGORITHMS:
        runs = [r for r in all_complete if r["algorithm"] == algo]
        if runs:
            avg_wr = sum(r["eval_win_rate_vs_random"] for r in runs) / len(runs)
            avg_ep = sum(r["episode_count"] for r in runs) / len(runs)
            print(f"  {algo.upper():8s}: {len(runs)}/5 runs | AvgWinRate={avg_wr:.2%} | AvgEp={avg_ep:.0f}")
        else:
            print(f"  {algo.upper():8s}: 0/5 runs")

    if failed_runs:
        print("FAILED RUNS:")
        for fr in failed_runs:
            print(f"  {fr['algorithm'].upper()} seed={fr['seed']}: {fr['reason_for_failure']}")

    print(f"Summary saved to: {SUMMARY_FILE}")
    print(DIVIDER)
    return final_summary


if __name__ == "__main__":
    summary = run_training_campaign()
    sys.exit(0 if summary["campaign_status"] == "COMPLETE" else 1)
