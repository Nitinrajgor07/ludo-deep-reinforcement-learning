"""PHASE 16 — ABLATION STUDIES RUNNER

Executes the three predefined ablation experiments from FINAL_EXPERIMENT_PROTOCOL.md (Section 19):
1. Representation Ablation: Structured CNN (4-channel spatial) vs Flat 60-D Observation Baseline.
2. Reward Shaping Ablation: Full 8-Event Dense Reward vs Sparse Terminal Reward (+/- 1.0 only).
3. Action Masking Ablation: Action Masking (-10^9 suppression) vs Unmasked learning with -0.50 penalty.

Strict Invariants:
- Change ONLY the component being studied.
- Keep other conditions fixed as much as possible.
- Every ablation has a unique experiment ID.
- Save raw data separately.
- Generate comparison tables and figures automatically from actual data.
"""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager, DQNConfig, RewardConfig
from phd_ludo_drl.experiments.experiment_id import ExperimentIdentifier, ExperimentMetadata
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.reward_calculator import LudoRewardCalculator
from phd_ludo_drl.environment.reward_engine import RewardEngine
from phd_ludo_drl.encoding.structured_representation import build_structured_representation
from phd_ludo_drl.encoding.raw_observation import extract_raw_observation
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent
from phd_ludo_drl.evaluation.evaluator import (
    StandardizedEvaluator,
    MatchupGameRecord,
    calculate_wilson_ci,
    calculate_bootstrap_ci,
)
from phd_ludo_drl.analysis.ablation_study import (
    compute_ablation_impact,
    generate_ablation_figures,
)

PROTOCOL_VERSION = "1.0.0 (FROZEN)"
MASTER_SEED = 42
TRAIN_STEPS = 500
EVAL_GAMES_PER_OPPONENT = 50  # 25 as P0, 25 as P1 (seat-alternating)
EVAL_BASE_SEED = MASTER_SEED + 3000  # 3042

RESULTS_ABLATION_DIR = PROJECT_ROOT / "results" / "ablations"
RESULTS_RAW_DIR = RESULTS_ABLATION_DIR / "raw"
DIVIDER = "=" * 88


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def train_ablation_agent(
    exp_id: str,
    encoder_type: str = "structured_cnn",
    reward_mode: str = "shaped",
    use_action_masking: bool = True,
    seed: int = 42,
    total_steps: int = 500,
    work_dir: Path = RESULTS_ABLATION_DIR,
) -> Tuple[DQNAgent, Path, Dict[str, Any]]:
    """Train a DQN agent under an isolated ablation condition."""
    set_seed(seed)
    cfg = ConfigManager.load_config()

    ckpt_dir = work_dir / exp_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"

    # 1. Instantiate environment with specified reward mode
    reward_calc = LudoRewardCalculator(config=cfg.reward, mode=reward_mode)
    env = LudoEnvironment(
        seed=seed,
        auto_advance_no_moves=True,
        reward_calculator=reward_calc,
    )

    # 2. Instantiate agent with specified encoder and masking settings
    agent = DQNAgent(
        config=cfg.dqn,
        latent_dim=cfg.encoding.latent_dim,
        encoder_type=encoder_type,
        use_action_masking=use_action_masking,
    )

    opponent = RandomOpponent(seed=seed + 10)

    # Training state
    batch_size = 32
    warmup_steps = 40
    step = 0
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    wins = 0
    losses = 0
    draws = 0
    illegal_attempts = 0
    loss_history: List[float] = []

    t0 = time.time()
    while step < total_steps:
        curr_p = env.current_player
        legal_actions = env.get_legal_actions()

        if not legal_actions:
            obs, info = env.reset()
            continue

        if curr_p == 0:
            # Agent's turn
            mask = info["action_mask"]
            if encoder_type == "flat":
                raw_obs = extract_raw_observation(obs)
                state_rep = raw_obs.to_flat_vector()
                context_rep = np.zeros(12, dtype=np.float32)
            else:
                struct_state = build_structured_representation(obs)
                state_rep = struct_state.spatial_tensor
                context_rep = struct_state.context_vector

            action = agent.select_action(state_rep, mask, deterministic=False)

            # Check if illegal action was chosen (under unmasked learning)
            if action not in legal_actions:
                illegal_attempts += 1

            next_obs, reward, term, trunc, step_info = env.step(action)

            if encoder_type == "flat":
                next_raw = extract_raw_observation(next_obs)
                next_state_rep = next_raw.to_flat_vector()
                next_context_rep = np.zeros(12, dtype=np.float32)
            else:
                next_struct = build_structured_representation(next_obs)
                next_state_rep = next_struct.spatial_tensor
                next_context_rep = next_struct.context_vector

            next_mask = step_info["action_mask"]
            done = term or trunc

            # Store in Prioritized Replay
            agent.buffer.push(
                state=state_rep,
                context=context_rep,
                action=action,
                reward=reward,
                next_state=next_state_rep,
                next_context=next_context_rep,
                done=done,
                action_mask=mask,
                next_action_mask=next_mask,
            )

            # Optimization step
            if step >= warmup_steps and len(agent.buffer) >= batch_size:
                t_info = agent.train_step(batch_size)
                if t_info and "loss" in t_info:
                    loss_history.append(float(t_info["loss"]))

            episode_reward += reward
            step += 1
            obs = next_obs
            info = step_info

            if done:
                if env.winner == 0:
                    wins += 1
                elif env.winner is not None:
                    losses += 1
                else:
                    draws += 1
                episode_count += 1
                episode_reward = 0.0
                obs, info = env.reset()
        else:
            # Opponent turn
            opp_act = opponent.select_action(obs, legal_actions)
            next_obs, _, term, trunc, step_info = env.step(opp_act)
            obs = next_obs
            info = step_info

            if term or trunc:
                if env.winner == 0:
                    wins += 1
                elif env.winner is not None:
                    losses += 1
                else:
                    draws += 1
                episode_count += 1
                episode_reward = 0.0
                obs, info = env.reset()

    train_runtime = time.time() - t0
    agent.save_checkpoint(ckpt_path)

    init_loss = loss_history[0] if loss_history else 0.0
    final_loss = loss_history[-1] if loss_history else 0.0
    loss_red = round(((init_loss - final_loss) / abs(init_loss)) * 100.0, 2) if abs(init_loss) > 1e-7 else 0.0

    train_summary = {
        "experiment_id": exp_id,
        "encoder_type": encoder_type,
        "reward_mode": reward_mode,
        "use_action_masking": use_action_masking,
        "total_steps": step,
        "episodes": episode_count,
        "agent_wins": wins,
        "opponent_wins": losses,
        "draws": draws,
        "training_win_rate": round(wins / max(1, episode_count), 4),
        "illegal_attempts_during_train": illegal_attempts,
        "illegal_attempt_rate": round(illegal_attempts / max(1, step), 4),
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_red,
        "runtime_seconds": round(train_runtime, 2),
        "loss_history": loss_history,
    }

    return agent, ckpt_path, train_summary


def evaluate_ablation_model(
    agent: DQNAgent,
    opponent_type: str,
    exp_id: str,
    num_games: int = 50,
    base_seed: int = 3042,
    reward_mode: str = "shaped",
) -> Tuple[List[MatchupGameRecord], Dict[str, Any]]:
    """Evaluate an ablation agent in a seat-balanced tournament against an opponent."""
    cfg = ConfigManager.load_config()
    reward_calc = LudoRewardCalculator(config=cfg.reward, mode=reward_mode)
    env = LudoEnvironment(seed=base_seed, auto_advance_no_moves=True, reward_calculator=reward_calc)

    opponent = ExpertOpponent(seed=base_seed + 10) if opponent_type == "expert" else RandomOpponent(seed=base_seed + 10)
    records: List[MatchupGameRecord] = []
    illegal_attempts_eval = 0

    for g in range(num_games):
        game_seed = base_seed + g
        agent_seat = 1 if (g % 2 == 1) else 0

        obs, info = env.reset(seed=game_seed)
        p0_r = 0.0
        p1_r = 0.0
        p0_c = 0
        p1_c = 0

        while not env.is_terminal and env.turn_count < 1000:
            curr_p = env.current_player
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                obs, info = env.step(0) if not env.is_terminal else (obs, {})
                continue

            mask = info["action_mask"]
            if curr_p == agent_seat:
                # Query agent
                act = agent.select_action(obs, mask, deterministic=True)
                if act not in legal_actions:
                    illegal_attempts_eval += 1
            else:
                # Query opponent
                act = opponent.select_action(obs, legal_actions)

            next_obs, rew, term, trunc, step_info = env.step(act)
            if curr_p == 0:
                p0_r += rew
                if "captured" in step_info and step_info["captured"]:
                    p0_c += len(step_info["captured"])
            else:
                p1_r += rew
                if "captured" in step_info and step_info["captured"]:
                    p1_c += len(step_info["captured"])

            obs = next_obs
            info = step_info

        # Record game results
        agent_won = (env.winner == agent_seat)
        opp_won = (env.winner is not None and env.winner != agent_seat)
        is_draw = (env.winner is None)

        p0_home = sum(1 for t in env.tokens if t.player_id == 0 and t.is_finished)
        p1_home = sum(1 for t in env.tokens if t.player_id == 1 and t.is_finished)

        agent_rew = p0_r if agent_seat == 0 else p1_r
        opp_rew = p1_r if agent_seat == 0 else p0_r
        agent_caps = p0_c if agent_seat == 0 else p1_c
        opp_caps = p1_c if agent_seat == 0 else p0_c
        agent_home = p0_home if agent_seat == 0 else p1_home
        opp_home = p1_home if agent_seat == 0 else p0_home

        records.append(
            MatchupGameRecord(
                game_idx=g,
                seed=game_seed,
                algorithm="dqn",
                opponent_type=opponent_type,
                agent_seat=agent_seat,
                winner=env.winner,
                agent_won=agent_won,
                opp_won=opp_won,
                is_draw=is_draw,
                turns=env.turn_count,
                agent_reward=round(agent_rew, 4),
                opp_reward=round(opp_rew, 4),
                agent_captures=agent_caps,
                opp_captures=opp_caps,
                agent_tokens_home=agent_home,
                opp_tokens_home=opp_home,
                terminated=env.is_terminal,
                truncated=(env.turn_count >= 1000),
                dice_variance_multiplier=1.0,
            )
        )

    # Compute aggregate metrics
    wins = sum(1 for r in records if r.agent_won)
    p0_recs = [r for r in records if r.agent_seat == 0]
    p1_recs = [r for r in records if r.agent_seat == 1]
    p0_wins = sum(1 for r in p0_recs if r.agent_won)
    p1_wins = sum(1 for r in p1_recs if r.agent_won)

    wr = wins / num_games
    p0_wr = p0_wins / len(p0_recs) if p0_recs else 0.0
    p1_wr = p1_wins / len(p1_recs) if p1_recs else 0.0
    bias = p0_wr - p1_wr

    rewards = [r.agent_reward for r in records]
    turns = [r.turns for r in records]

    wilson_ci = calculate_wilson_ci(wins, num_games, confidence=0.95)
    rew_ci = calculate_bootstrap_ci(rewards, num_resamples=10000, confidence=0.95)

    summary = {
        "experiment_id": exp_id,
        "opponent": opponent_type,
        "total_games": num_games,
        "agent_wins": wins,
        "opponent_wins": sum(1 for r in records if r.opp_won),
        "draws": sum(1 for r in records if r.is_draw),
        "win_rate": round(wr, 4),
        "win_rate_ci_95_wilson": [round(wilson_ci[0], 4), round(wilson_ci[1], 4)],
        "average_reward": round(float(np.mean(rewards)), 4),
        "reward_std": round(float(np.std(rewards, ddof=1)), 4),
        "reward_variance": round(float(np.var(rewards, ddof=1)), 4),
        "reward_ci_95": [round(rew_ci[0], 4), round(rew_ci[1], 4)],
        "average_turns": round(float(np.mean(turns)), 2),
        "turns_variance": round(float(np.var(turns, ddof=1)), 2),
        "p0_win_rate": round(p0_wr, 4),
        "p1_win_rate": round(p1_wr, 4),
        "seat_bias": round(bias, 4),
        "illegal_action_rate": round(illegal_attempts_eval / max(1, sum(turns)), 4),
        "total_illegal_attempts": illegal_attempts_eval,
    }

    return records, summary


def save_records(records: List[MatchupGameRecord], csv_path: Path, json_path: Path) -> None:
    """Save evaluation records to CSV and JSON."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].to_dict().keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def run_phase16_ablations():
    """Main orchestrator executing all predefined ablation experiments."""
    RESULTS_ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(DIVIDER)
    print("PHASE 16: PREDEFINED ABLATION STUDIES")
    print(f"Protocol: FINAL_EXPERIMENT_PROTOCOL.md Section 19")
    print("Ablation 1: Representation (Structured CNN vs Flat 60-D Observation)")
    print("Ablation 2: Reward Shaping (Dense 8-Event vs Sparse Terminal +/- 1.0)")
    print("Ablation 3: Action Masking (-10^9 Logit Masking vs Unmasked Learning with -0.50 Penalty)")
    print(f"Control Base Seed: {MASTER_SEED} | Evaluation Base Seed: {EVAL_BASE_SEED}")
    print(f"Evaluation Horizon: {EVAL_GAMES_PER_OPPONENT} games vs Expert + {EVAL_GAMES_PER_OPPONENT} games vs Random")
    print(f"Output Directory: {RESULTS_ABLATION_DIR}")
    print(DIVIDER)

    now_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the 4 experimental conditions with deterministic IDs
    configurations = [
        {
            "key": "baseline",
            "name": "Baseline (Structured CNN + Shaped + Masked)",
            "exp_id": "EXP_DQN_structured_cnn_42",
            "encoder_type": "structured_cnn",
            "reward_mode": "shaped",
            "use_action_masking": True,
            "description": "Full baseline architecture with all components active",
            "component_studied": "None (Full Model Control)",
        },
        {
            "key": "ablation_repr_flat",
            "name": "Ablation 1: Flat Encoder",
            "exp_id": "EXP_ABLATION_REPR_FLAT_42",
            "encoder_type": "flat",
            "reward_mode": "shaped",
            "use_action_masking": True,
            "description": "Flat 60-D observation baseline replacing 4-channel circular CNN",
            "component_studied": "State Representation (Flat 60-D vs Structured CNN)",
        },
        {
            "key": "ablation_reward_sparse",
            "name": "Ablation 2: Sparse Reward",
            "exp_id": "EXP_ABLATION_REWARD_SPARSE_42",
            "encoder_type": "structured_cnn",
            "reward_mode": "sparse",
            "use_action_masking": True,
            "description": "Sparse terminal reward (+/- 1.0 only) replacing dense 8-event reward",
            "component_studied": "Reward Shaping (Sparse +/- 1.0 vs Dense Shaped)",
        },
        {
            "key": "ablation_mask_unmasked",
            "name": "Ablation 3: Unmasked Action",
            "exp_id": "EXP_ABLATION_MASK_UNMASKED_42",
            "encoder_type": "structured_cnn",
            "reward_mode": "shaped",
            "use_action_masking": False,
            "description": "Unmasked action selection with -0.50 illegal move penalty",
            "component_studied": "Action Masking (Unmasked with -0.50 vs Masked)",
        },
    ]

    master_ablation_results: Dict[str, Any] = {}
    master_table_rows: List[Dict[str, Any]] = []

    start_total = time.time()

    for config in configurations:
        key = config["key"]
        name = config["name"]
        exp_id = config["exp_id"]
        enc = config["encoder_type"]
        rew_m = config["reward_mode"]
        mask_flag = config["use_action_masking"]

        print("")
        log(f">>> Running Experiment: {name.upper()} (ID: {exp_id})")

        exp_expert_csv = RESULTS_RAW_DIR / f"{exp_id}_expert_games.csv"
        exp_expert_json = RESULTS_RAW_DIR / f"{exp_id}_expert_games.json"
        exp_random_csv = RESULTS_RAW_DIR / f"{exp_id}_random_games.csv"
        exp_random_json = RESULTS_RAW_DIR / f"{exp_id}_random_games.json"

        # 1. Train agent under condition
        log(f"  Training {TRAIN_STEPS} steps under condition (Encoder={enc}, Reward={rew_m}, Masking={mask_flag})...")
        agent, ckpt_path, train_data = train_ablation_agent(
            exp_id=exp_id,
            encoder_type=enc,
            reward_mode=rew_m,
            use_action_masking=mask_flag,
            seed=MASTER_SEED,
            total_steps=TRAIN_STEPS,
            work_dir=RESULTS_ABLATION_DIR,
        )
        log(f"  Training completed in {train_data['runtime_seconds']}s | Final Loss: {train_data['final_loss']:.4f} | Loss Reduction: {train_data['loss_reduction_pct']}% | Illegal Attempts: {train_data['illegal_attempts_during_train']}")

        # 2. Evaluate agent vs Expert Opponent
        log(f"  Evaluating {EVAL_GAMES_PER_OPPONENT} games vs EXPERT Opponent (seed={EVAL_BASE_SEED})...")
        expert_recs, expert_sum = evaluate_ablation_model(
            agent=agent,
            opponent_type="expert",
            exp_id=exp_id,
            num_games=EVAL_GAMES_PER_OPPONENT,
            base_seed=EVAL_BASE_SEED,
            reward_mode=rew_m,
        )
        save_records(expert_recs, exp_expert_csv, exp_expert_json)
        log(f"    vs EXPERT -> WinRate: {expert_sum['win_rate']:.2%} | AvgReward: {expert_sum['average_reward']:.2f} | IllegalRate: {expert_sum['illegal_action_rate']:.2%}")

        # 3. Evaluate agent vs Random Opponent
        log(f"  Evaluating {EVAL_GAMES_PER_OPPONENT} games vs RANDOM Opponent (seed={EVAL_BASE_SEED})...")
        random_recs, random_sum = evaluate_ablation_model(
            agent=agent,
            opponent_type="random",
            exp_id=exp_id,
            num_games=EVAL_GAMES_PER_OPPONENT,
            base_seed=EVAL_BASE_SEED,
            reward_mode=rew_m,
        )
        save_records(random_recs, exp_random_csv, exp_random_json)
        log(f"    vs RANDOM -> WinRate: {random_sum['win_rate']:.2%} | AvgReward: {random_sum['average_reward']:.2f} | IllegalRate: {random_sum['illegal_action_rate']:.2%}")

        master_ablation_results[key] = {
            "config": config,
            "training": train_data,
            "eval_expert": expert_sum,
            "eval_random": random_sum,
        }

        # Store rows for comparative table
        for opp_label, s in [("EXPERT", expert_sum), ("RANDOM", random_sum)]:
            master_table_rows.append({
                "condition": name,
                "experiment_id": exp_id,
                "opponent": opp_label,
                "encoder": enc,
                "reward_mode": rew_m,
                "action_masking": mask_flag,
                "total_games": s["total_games"],
                "win_rate": s["win_rate"],
                "win_rate_ci_lower": s["win_rate_ci_95_wilson"][0],
                "win_rate_ci_upper": s["win_rate_ci_95_wilson"][1],
                "average_reward": s["average_reward"],
                "reward_variance": s["reward_variance"],
                "reward_ci_lower": s["reward_ci_95"][0],
                "reward_ci_upper": s["reward_ci_95"][1],
                "average_turns": s["average_turns"],
                "illegal_action_rate": s["illegal_action_rate"],
                "training_final_loss": train_data["final_loss"],
                "training_loss_reduction": train_data["loss_reduction_pct"],
            })

    total_elapsed = time.time() - start_total

    # 4. Compute comparative impact against Baseline
    baseline_expert = master_ablation_results["baseline"]["eval_expert"]
    impact_rows: List[Dict[str, Any]] = []

    for cfg_item in configurations[1:]:
        k = cfg_item["key"]
        comp = cfg_item["component_studied"]
        abl_expert = master_ablation_results[k]["eval_expert"]
        impact = compute_ablation_impact(baseline_expert, abl_expert, cfg_item["name"], comp)
        impact_rows.append(impact)

    # 5. Save master CSV and JSON files
    comp_csv_path = RESULTS_ABLATION_DIR / "ablation_comparison_table.csv"
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(master_table_rows)
    log(f"Saved master ablation comparison table: {comp_csv_path.name}")

    impact_csv_path = RESULTS_ABLATION_DIR / "ablation_impact_summary.csv"
    with open(impact_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(impact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(impact_rows)
    log(f"Saved ablation impact summary: {impact_csv_path.name}")

    master_summary = {
        "phase": "Phase 16 Ablation Studies",
        "protocol_version": PROTOCOL_VERSION,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(total_elapsed, 2),
        "master_seed": MASTER_SEED,
        "eval_base_seed": EVAL_BASE_SEED,
        "ablations": master_ablation_results,
        "impact_analysis": impact_rows,
    }
    with open(RESULTS_ABLATION_DIR / "master_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    log("Saved master ablation JSON summary: master_ablation_summary.json")

    # 6. Generate publication comparison figures
    log("Generating publication-quality comparison figures from actual empirical data...")
    generated_plots = generate_ablation_figures(master_ablation_results, RESULTS_ABLATION_DIR)
    for gp in generated_plots:
        log(f"  Generated figure: {gp.name}")

    # 7. Print summary presentation in terminal
    print("")
    print(DIVIDER)
    print("PHASE 16 ABLATION TOURNAMENT RESULTS (ACTUAL DATA)")
    print(DIVIDER)
    print(f"{'Condition':<30} | {'Opponent':<8} | {'Win Rate':<10} | {'95% CI':<18} | {'Avg Reward':<12} | {'Illegal Rate':<12}")
    print("-" * 105)
    for r in master_table_rows:
        ci_str = f"[{r['win_rate_ci_lower']:.1%}, {r['win_rate_ci_upper']:.1%}]"
        print(f"{r['condition']:<30} | {r['opponent']:<8} | {r['win_rate']:<10.2%} | {ci_str:<18} | {r['average_reward']:<12.2f} | {r['illegal_action_rate']:<12.2%}")

    print(DIVIDER)
    print("ABLATION IMPACT & DEGRADATION ANALYSIS (vs EXPERT)")
    print(DIVIDER)
    print(f"{'Ablation':<28} | {'Base WR':<10} | {'Abl WR':<10} | {'Δ WR':<10} | {'Impact %':<12} | {'Δ Reward':<10} | {'Sig (p<0.05)':<12}")
    print("-" * 105)
    for imp in impact_rows:
        print(f"{imp['ablation_name']:<28} | {imp['baseline_win_rate']:<10.2%} | {imp['ablation_win_rate']:<10.2%} | {imp['delta_win_rate']:<+10.2%} | {imp['pct_win_rate_impact']:<+12.1f}% | {imp['delta_reward']:<+10.2f} | {str(imp['significant_alpha_005']):<12}")
    print(DIVIDER)
    print(f"Phase 16 completed in {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min).")
    print(f"All raw game files and plots saved in: {RESULTS_ABLATION_DIR}")
    print(DIVIDER)

    return master_summary


if __name__ == "__main__":
    run_phase16_ablations()
