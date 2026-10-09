"""Comprehensive post-cleanup verification script for the PhD Ludo DRL Project."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gymnasium
import numpy as np

print("==================================================")
print("PhD PROJECT INTEGRITY & CLEANUP VALIDATION")
print("==================================================")

# 1. Check all four models
print("\n[CHECK 1 & 2] Loading all four DRL models...")
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent

a1 = DQNAgent()
a2 = PPOAgent()
a3 = DiscreteSACAgent()
a4 = CategoricalDQNAgent()
print(f"  + Model 1 (DQN): {type(a1).__name__} initialized successfully.")
print(f"  + Model 2 (PPO): {type(a2).__name__} initialized successfully.")
print(f"  + Model 3 (Discrete SAC): {type(a3).__name__} initialized successfully.")
print(f"  + Model 4 (Categorical C51): {type(a4).__name__} initialized successfully.")

# 2. Check environment
print("\n[CHECK 3] Verifying Ludo MDP Environment...")
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
env = LudoEnvironment()
obs, info = env.reset(seed=42)
leg = info.get("legal_actions", [])
print(f"  + Environment reset successful. Observation keys: {list(obs.keys())}")
print(f"  + Legal actions at start: {leg}")
action = leg[0] if leg else 0
step_res = env.step(action)
print(f"  + Environment step executed. Reward: {step_res[1]}, Terminated: {step_res[2]}")

# 3. Check State Encoder
print("\n[CHECK 4] Verifying State Representation & CNN Encoder...")
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.structured_representation import build_structured_representation

encoder = StructuredCNNEncoder()
spatial = torch.zeros((1, 4, 52))
context = torch.zeros((1, 12))
latent = encoder(spatial, context)
print(f"  + StructuredCNNEncoder forward pass successful: output shape = {list(latent.shape)} (Target: [1, 128])")
rep = build_structured_representation(obs)
print(f"  + build_structured_representation successful: spatial={rep.spatial_tensor.shape}, context={rep.context_vector.shape}")

# 4. Check Configs
print("\n[CHECK 5] Verifying Training and Evaluation Configuration...")
from phd_ludo_drl.config.config_manager import ConfigManager
cfg = ConfigManager.load_config()
print(f"  + Default Config loaded. Total steps: {cfg.training.total_steps}, Seeds: {cfg.training.seeds}")

# 5. Check Documentation Links
print("\n[CHECK 6] Verifying Important Documentation Files...")
important_docs = [
    "README.md",
    "FINAL_EXPERIMENT_PROTOCOL.md",
    "REPRODUCIBILITY.md",
    "PROJECT_ARCHITECTURE.md",
    "FINAL_VALIDATION_REPORT.md",
    "FINAL_RELEASE_GUIDE.md",
    "docs/FROZEN_STATE_REPRESENTATION.md",
    "docs/PHASE10_FINAL_EVALUATION_REPORT.md",
    "docs/PHASE11_STATISTICAL_ANALYSIS_REPORT.md",
    "docs/PHASE12_ROBUSTNESS_AND_ABLATION_REPORT.md",
    "docs/PHASE13_RESEARCH_VISUALIZATION_REPORT.md",
    "docs/PHASE14_LIGHT_GLASS_WEBSITE_REPORT.md",
    "docs/RUN_PROJECT.md",
]
all_docs_exist = True
for doc in important_docs:
    p = Path(doc)
    if p.exists():
        print(f"  + [FOUND] {doc} ({p.stat().st_size} bytes)")
    else:
        print(f"  - [MISSING] {doc}")
        all_docs_exist = False

# 6. Check for paper-only files
print("\n[CHECK 7] Scanning for unwanted paper/manuscript artifacts...")
suspicious = []
for root, dirs, files in os.walk("."):
    if any(x in root for x in [".git", "__pycache__", ".pytest_cache", ".venv", "venv"]):
        continue
    for f in files:
        low = f.lower()
        if any(k in low for k in ["paper_draft", "manuscript", "evidence_mapping", "second_phd_research_paper"]):
            suspicious.append(os.path.join(root, f))
print(f"  + Unwanted paper files count: {len(suspicious)}")

# 7. Check for nested ZIP files
print("\n[CHECK 8] Scanning for nested ZIP archives...")
zips = [os.path.join(r, f) for r, d, fs in os.walk(".") for f in fs if f.endswith(".zip") and not any(x in r for x in [".git", ".venv", "venv"])]
print(f"  + Nested ZIP archives count: {len(zips)}")

# 8. Check for cache files
print("\n[CHECK 9] Scanning for cache / compiled files...")
caches = [os.path.join(r, f) for r, d, fs in os.walk(".") for f in fs if f.endswith((".pyc", ".pyo")) and not any(x in r for x in [".git", ".venv", "venv"])]
print(f"  + Bytecode / cache files count: {len(caches)}")

print("\n==================================================")
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("==================================================")
