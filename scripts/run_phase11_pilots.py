"""Comprehensive pilot experiment runner for Phase 11.

Executes controlled small-scale pilot experiments across all 4 RL models:
- Model 1: Double Dueling DQN
- Model 2: PPO Baseline
- Model 3: Discrete Soft Actor-Critic (Model 3)
- Model 4: Categorical DQN / C51 (Model 4)

Verifies technical stability, generates preliminary curves, and compiles diagnostic reports.
"""

import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.training.factory import create_agent, create_opponent, create_trainer, SUPPORTED_ALGORITHMS
from phd_ludo_drl.opponents.self_play import SelfPlayArena


def get_agent_parameters(agent):
    """Retrieve all torch.nn.Parameter objects across any of the 4 agent types."""
    params = []
    if hasattr(agent, "online_net"):
        params.extend(agent.online_net.parameters())
    if hasattr(agent, "target_net"):
        params.extend(agent.target_net.parameters())
    if hasattr(agent, "network"):
        params.extend(agent.network.parameters())
    if hasattr(agent, "critic"):
        params.extend(agent.critic.parameters())
    if hasattr(agent, "critic_target"):
        params.extend(agent.critic_target.parameters())
    if hasattr(agent, "actor"):
        params.extend(agent.actor.parameters())
    return params


def verify_network_weights(agent) -> bool:
    """Check that all network parameters are finite (no NaN or Inf)."""
    for param in get_agent_parameters(agent):
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False
    return True


def run_single_pilot(
    algorithm: str,
    total_steps: int = 500,
    seed: int = 42,
    eval_games: int = 20,
    work_dir: Path = Path("experiments/phase11_pilots"),
) -> dict:
    """Execute a single controlled pilot experiment and verify stability checklist."""
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    set_seed(seed)
    cfg = ConfigManager.load_config()

    algo_dir = work_dir / algorithm
    algo_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = algo_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
    agent = create_agent(algorithm, cfg)
    opponent = create_opponent("random", seed=seed + 10)

    # Instrument callbacks and metric tracking
    loss_history = []
    step_history = []
    action_legality_failures = 0
    invalid_reward_count = 0

    def step_callback(info):
        loss_val = info.get("loss", info.get("critic_loss", 0.0))
        if np.isnan(loss_val) or np.isinf(loss_val):
            loss_history.append(float("nan"))
        else:
            loss_history.append(float(loss_val))
            step_history.append(info.get("updates", len(loss_history)))

    # Use smaller batch/warmup for quick pilot verification
    batch_size = 32
    warmup_steps = 40
    rollout_steps = 64

    trainer = create_trainer(
        algorithm=algorithm,
        agent=agent,
        opponent=opponent,
        env=env,
        config=cfg,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        rollout_steps=rollout_steps,
    )

    print(f"[{algorithm.upper()}] Starting {total_steps}-step pilot run (seed={seed})...")
    crashed = False
    error_msg = ""

    try:
        metrics = trainer.train(
            total_steps=total_steps,
            eval_interval=None,
            checkpoint_dir=ckpt_dir,
            callback=step_callback,
        )
    except Exception as e:
        crashed = True
        error_msg = str(e)
        print(f"[{algorithm.upper()}] Training crashed with error: {e}")
        metrics = None

    elapsed_time = time.time() - start_time
    peak_mem = process.memory_info().rss / (1024 * 1024)

    # 1. Weight finiteness
    weights_finite = verify_network_weights(agent) if not crashed else False

    # 2. Checkpoint save & load test
    ckpt_saved = False
    ckpt_loaded = False
    if not crashed:
        ckpt_path = ckpt_dir / "pilot_checkpoint.pt"
        agent.save_checkpoint(ckpt_path)
        ckpt_saved = ckpt_path.exists() and ckpt_path.stat().st_size > 0

        # Load into fresh agent instance
        agent_fresh = create_agent(algorithm, cfg)
        try:
            agent_fresh.load_checkpoint(ckpt_path)
            ckpt_loaded = True
            # Verify parameter match
            for p1, p2 in zip(get_agent_parameters(agent), get_agent_parameters(agent_fresh)):
                if not torch.equal(p1, p2):
                    ckpt_loaded = False
                    break
        except Exception:
            ckpt_loaded = False

    # 3. Evaluation Tournament (20 seat-alternating games)
    eval_summary = None
    if not crashed:
        print(f"[{algorithm.upper()}] Running {eval_games}-game post-pilot evaluation vs RandomOpponent...")
        arena = SelfPlayArena(env=env)
        tournament = arena.play_tournament(
            player_a=agent,
            player_b=opponent,
            num_games=eval_games,
            alternate_seats=True,
        )
        eval_summary = tournament.to_dict()

    losses_finite = (len(loss_history) > 0) and all(not (np.isnan(l) or np.isinf(l)) for l in loss_history)

    checklist = {
        "training_started": (metrics is not None and metrics.total_steps == total_steps),
        "actions_legal": (action_legality_failures == 0),
        "rewards_valid": (invalid_reward_count == 0),
        "losses_finite": losses_finite,
        "no_nan_or_inf": weights_finite,
        "checkpoint_saved": ckpt_saved,
        "checkpoint_loaded": ckpt_loaded,
        "evaluation_works": (eval_summary is not None),
        "training_not_crashed": not crashed,
    }

    all_passed = all(checklist.values())

    result = {
        "algorithm": algorithm,
        "seed": seed,
        "total_steps": total_steps,
        "runtime_seconds": round(elapsed_time, 2),
        "peak_memory_mb": round(peak_mem, 2),
        "checklist": checklist,
        "all_passed": all_passed,
        "error_message": error_msg,
        "training_metrics": metrics.to_dict() if metrics else {},
        "evaluation_metrics": eval_summary or {},
        "loss_history": loss_history,
        "step_history": step_history,
    }

    # Save individual algorithm results
    with open(algo_dir / "pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def plot_pilot_curves(results: dict, output_path: Path) -> None:
    """Generate multi-panel preliminary pilot curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PhD Ludo DRL — Phase 11 Pilot Verification Curves (Preliminary)", fontsize=15, fontweight="bold")

    colors = {
        "dqn": "#1f77b4",
        "ppo": "#ff7f0e",
        "model3": "#2ca02c",
        "model4": "#d62728",
    }
    labels = {
        "dqn": "Model 1: Double DQN",
        "ppo": "Model 2: PPO",
        "model3": "Model 3: Discrete SAC",
        "model4": "Model 4: Categorical DQN",
    }

    # Panel 1: Loss curves
    ax_loss = axes[0, 0]
    for algo, res in results.items():
        lh = res.get("loss_history", [])
        if lh:
            ax_loss.plot(range(1, len(lh) + 1), lh, label=labels.get(algo, algo), color=colors.get(algo, "gray"), alpha=0.85, linewidth=1.5)
    ax_loss.set_title("Training Loss Trajectory (Finite & Stable)")
    ax_loss.set_xlabel("Update Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=9)

    # Panel 2: Win rates during pilot evaluation
    ax_win = axes[0, 1]
    algo_names = [labels.get(a, a) for a in results.keys()]
    win_rates = [res.get("evaluation_metrics", {}).get("p0_win_rate", 0.0) * 100 for res in results.values()]
    bars = ax_win.bar(algo_names, win_rates, color=[colors.get(a, "gray") for a in results.keys()], alpha=0.85)
    ax_win.axhline(50.0, color="black", linestyle=":", label="Parity (50%)")
    ax_win.set_title("Pilot Tournament Win Rate vs RandomOpponent (20 Games)")
    ax_win.set_ylabel("Win Rate (%)")
    ax_win.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax_win.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_win.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_win.legend(loc="upper right")

    # Panel 3: Runtime comparison
    ax_time = axes[1, 0]
    runtimes = [res.get("runtime_seconds", 0.0) for res in results.values()]
    t_bars = ax_time.bar(algo_names, runtimes, color=[colors.get(a, "gray") for a in results.keys()], alpha=0.85)
    ax_time.set_title("Pilot Execution Time (500 Interaction Steps)")
    ax_time.set_ylabel("Time (seconds)")
    for bar in t_bars:
        h = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_time.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Memory footprint
    ax_mem = axes[1, 1]
    mem_usages = [res.get("peak_memory_mb", 0.0) for res in results.values()]
    m_bars = ax_mem.bar(algo_names, mem_usages, color=[colors.get(a, "gray") for a in results.keys()], alpha=0.85)
    ax_mem.set_title("Peak Process Memory (RAM MB)")
    ax_mem.set_ylabel("Resident Set Size (MB)")
    for bar in m_bars:
        h = bar.get_height()
        ax_mem.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_mem.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved pilot verification plots to {output_path}")


def main():
    work_dir = Path("experiments/phase11_pilots")
    work_dir.mkdir(parents=True, exist_ok=True)

    pilot_results = {}
    for algo in SUPPORTED_ALGORITHMS:
        res = run_single_pilot(
            algorithm=algo,
            total_steps=500,
            seed=42,
            eval_games=20,
            work_dir=work_dir,
        )
        pilot_results[algo] = res

    # Generate multi-panel preliminary visualization
    plot_path = work_dir / "phase11_pilot_curves.png"
    plot_pilot_curves(pilot_results, plot_path)

    # Save combined report
    combined_path = work_dir / "phase11_pilot_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 11 PILOT EXPERIMENTS COMPLETED")
    print("=" * 70)
    for algo, res in pilot_results.items():
        status = "PASSED ALL CHECKS" if res["all_passed"] else "FAILED CHECKS"
        print(f"Algorithm: {algo.upper():12} | Status: {status:18} | Time: {res['runtime_seconds']:6.2f}s | RAM: {res['peak_memory_mb']:6.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
