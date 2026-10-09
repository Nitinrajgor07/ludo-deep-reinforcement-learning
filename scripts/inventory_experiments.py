"""Inventory scanner for experiments directory."""
import json
from pathlib import Path

def main():
    exp_dir = Path("experiments")
    rows = []
    for p in sorted(exp_dir.iterdir()):
        if not p.is_dir() or p.name == "phase11_pilots":
            continue
        exp_id = p.name
        summary_file = p / "results" / "summary.json"
        ckpt_file = p / "checkpoints" / "final_model.pt"
        
        algo = "UNKNOWN"
        seed = "-"
        steps = "-"
        eval_win_rate = "N/A"
        ckpt_status = "Present" if ckpt_file.exists() else "Missing"
        
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                algo = data.get("metadata", {}).get("algorithm", "UNKNOWN").upper()
                seed = data.get("metadata", {}).get("seed", "-")
                steps = data.get("training_metrics", {}).get("total_steps", "-")
                eval_wr = data.get("evaluation_metrics", {}).get("p0_win_rate")
                if eval_wr is not None:
                    eval_win_rate = f"{eval_wr * 100:.1f}%"
                status = "Complete"
            except Exception as e:
                status = f"Corrupt: {e}"
        else:
            status = "No summary"
            
        rows.append((algo, exp_id, str(seed), str(steps), eval_win_rate, ckpt_status, status))

    print("| Model | Experiment ID | Seed | Steps | Evaluation | Checkpoint | Status |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")

if __name__ == "__main__":
    main()
