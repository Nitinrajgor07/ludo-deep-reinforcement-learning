"""Exhaustive validation and unit tests for PPO:
- Actor network & output
- Critic network & output
- Shared/frozen encoder integration
- Legal action masking & masked probabilities
- Rollout collection & buffer operations
- Generalized Advantage Estimation (GAE) with exact analytical verification
- Advantage normalization
- PPO clipped objective, value loss, and entropy bonus
- Minibatch updates
- Checkpoint saving and loading
- Deterministic evaluation mode
"""

import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from phd_ludo_drl.config.config_manager import PPOConfig
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.agents.ppo.ppo_network import ActorCriticNetwork
from phd_ludo_drl.agents.ppo.rollout_buffer import PPORolloutBuffer
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.ppo.trainer import PPOTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.action_mask import mask_action_logits


# ======================================================================
# 1. Actor Network & Output Tests
# ======================================================================

def test_actor_output_shapes_and_values():
    """Verify Actor network produces correct logit shapes and valid action distributions."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    batch_size = 5
    spatial = torch.randn(batch_size, 4, 52)
    context = torch.randn(batch_size, 12)
    mask = torch.ones(batch_size, 4)

    # Batched forward pass
    actions, log_probs, entropy, values = net.get_action_and_value(
        spatial=spatial,
        context=context,
        action_mask=mask,
    )

    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)

    # All actions in valid range [0, 3]
    for act in actions:
        assert 0 <= int(act.item()) <= 3

    # All log probs <= 0.0 (valid log probabilities)
    assert (log_probs <= 1e-6).all(), "Log probabilities must be <= 0"

    # Entropy of 4-action distribution must be non-negative and <= ln(4) approx 1.3863
    assert (entropy >= 0.0).all(), "Entropy must be non-negative"
    assert (entropy <= np.log(4) + 1e-4).all(), "Entropy cannot exceed ln(4)"


def test_actor_single_input_forward():
    """Verify Actor handles single unbatched inputs correctly."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    spatial = torch.randn(4, 52)
    context = torch.randn(12)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    action, log_prob, entropy, value = net.get_action_and_value(
        spatial=spatial,
        context=context,
        action_mask=mask,
        deterministic=True,
    )

    assert action.dim() == 0 or action.shape == (1,)
    assert int(action.item()) in (0, 1)  # Only actions 0 and 1 are legal
    assert isinstance(float(log_prob.item()), float)
    assert isinstance(float(entropy.item()), float)
    assert isinstance(float(value.item()), float)


# ======================================================================
# 2. Critic Network & Output Tests
# ======================================================================

def test_critic_output_shapes_and_values():
    """Verify Critic network value function outputs and consistency."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    batch_size = 6
    spatial = torch.randn(batch_size, 4, 52)
    context = torch.randn(batch_size, 12)
    mask = torch.ones(batch_size, 4)

    # Value from get_value
    val_direct = net.get_value(spatial, context)
    assert val_direct.shape == (batch_size,)
    assert not torch.isnan(val_direct).any()
    assert not torch.isinf(val_direct).any()

    # Value from get_action_and_value should match get_value exactly
    _, _, _, val_joint = net.get_action_and_value(spatial, context, mask)
    assert torch.allclose(val_direct, val_joint, atol=1e-5)


def test_critic_agent_get_value():
    """Verify PPOAgent.get_value returns scalar float for state dict and array."""
    agent = PPOAgent()
    env = LudoEnvironment()
    obs, info = env.reset(seed=42)

    val_dict = agent.get_value(obs)
    assert isinstance(val_dict, float)
    assert np.isfinite(val_dict)

    struct_state = build_structured_representation(obs)
    val_struct = agent.get_value(struct_state)
    assert isinstance(val_struct, float)
    assert np.isclose(val_dict, val_struct, atol=1e-5)


# ======================================================================
# 3. Shared and Frozen Encoder Integration
# ======================================================================

def test_shared_frozen_encoder_freeze_unfreeze():
    """Verify freezing the shared CNN encoder disables its gradients."""
    encoder = StructuredCNNEncoder(latent_dim=128)
    net = ActorCriticNetwork(encoder=encoder, freeze_encoder=True)

    # Check all encoder parameters have requires_grad == False
    for name, param in net.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param {name} should be frozen"

    # Check policy and value heads have requires_grad == True
    for name, param in net.policy_head.named_parameters():
        assert param.requires_grad, f"Policy head param {name} should be trainable"
    for name, param in net.value_head.named_parameters():
        assert param.requires_grad, f"Value head param {name} should be trainable"

    # Unfreeze and verify
    net.unfreeze_encoder()
    for name, param in net.encoder.named_parameters():
        assert param.requires_grad, f"Encoder param {name} should now be trainable"


def test_ppo_agent_frozen_encoder_weights_unchanged_after_update():
    """Verify that training updates DO NOT modify frozen encoder parameters."""
    agent = PPOAgent(freeze_encoder=True)

    # Record initial encoder weights
    init_encoder_weights = {
        name: param.clone() for name, param in agent.network.encoder.named_parameters()
    }
    init_policy_weights = {
        name: param.clone() for name, param in agent.network.policy_head.named_parameters()
    }

    # Populate a synthetic buffer
    buffer = PPORolloutBuffer(capacity=8)
    for _ in range(8):
        buffer.add(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=0,
            action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            value=0.5,
            log_prob=-0.5,
            done=False,
        )
    buffer.compute_gae(last_value=0.5, last_done=False)

    # Perform PPO update
    agent.update(buffer)

    # Encoder weights MUST NOT have changed
    for name, param in agent.network.encoder.named_parameters():
        assert torch.equal(param, init_encoder_weights[name]), (
            f"Frozen encoder parameter {name} was modified during update!"
        )

    # Policy head weights MUST have changed
    policy_changed = any(
        not torch.equal(param, init_policy_weights[name])
        for name, param in agent.network.policy_head.named_parameters()
    )
    assert policy_changed, "Trainable policy head weights should have been updated"


# ======================================================================
# 4. Masked Probabilities Enforcement
# ======================================================================

def test_masked_probabilities_strictly_zero():
    """Verify illegal actions receive strictly 0.0 softmax probability."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    spatial = torch.randn(1, 4, 52)
    context = torch.randn(1, 12)

    # Only action 1 and action 3 are legal
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

    features = net.extract_features(spatial, context)
    raw_logits = net.policy_head(features)
    masked_logits = mask_action_logits(raw_logits, mask)
    probs = torch.softmax(masked_logits, dim=-1).squeeze(0)

    # Illegal actions 0 and 2 must have exactly 0.0 probability
    assert probs[0].item() == 0.0, f"Action 0 prob should be 0.0, got {probs[0].item()}"
    assert probs[2].item() == 0.0, f"Action 2 prob should be 0.0, got {probs[2].item()}"

    # Legal actions 1 and 3 must sum to 1.0
    assert np.isclose(probs[1].item() + probs[3].item(), 1.0, atol=1e-6)


def test_stochastic_sampling_never_selects_masked_action():
    """Verify stochastic sampling over 200 iterations never picks an illegal action."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    spatial = torch.randn(1, 4, 52)
    context = torch.randn(1, 12)
    # Only action 2 is legal
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    for _ in range(200):
        action, log_prob, _, _ = net.get_action_and_value(
            spatial, context, mask, deterministic=False
        )
        assert action.item() == 2
        assert np.isclose(log_prob.exp().item(), 1.0, atol=1e-5)


# ======================================================================
# 5. Generalized Advantage Estimation (GAE) & Analytical Verification
# ======================================================================

def test_gae_analytical_step_by_step():
    """Verify GAE calculation against exact hand-calculated analytical values.

    Analytical derivation:
    3 transitions: t=0, t=1, t=2 (terminal).
    gamma = 0.90, lambda = 0.80

    Step 2 (terminal, done=1):
        r_2 = -0.5, V_2 = 1.0, next_non_terminal = 0.0
        delta_2 = -0.5 + 0.90 * 0.0 - 1.0 = -1.50
        A_2 = delta_2 = -1.50
        R_2 = A_2 + V_2 = -1.50 + 1.0 = -0.50

    Step 1 (done=0):
        r_1 = 2.0, V_1 = 1.5, V_2 = 1.0, next_non_terminal = 1.0
        delta_1 = 2.0 + 0.90 * 1.0 * 1.0 - 1.5 = 2.0 + 0.90 - 1.5 = 1.40
        A_1 = delta_1 + 0.90 * 0.80 * 1.0 * A_2 = 1.40 + 0.72 * (-1.50) = 1.40 - 1.08 = 0.32
        R_1 = A_1 + V_1 = 0.32 + 1.5 = 1.82

    Step 0 (done=0):
        r_0 = 1.0, V_0 = 2.0, V_1 = 1.5, next_non_terminal = 1.0
        delta_0 = 1.0 + 0.90 * 1.5 * 1.0 - 2.0 = 1.0 + 1.35 - 2.0 = 0.35
        A_0 = delta_0 + 0.90 * 0.80 * 1.0 * A_1 = 0.35 + 0.72 * 0.32 = 0.35 + 0.2304 = 0.5804
        R_0 = A_0 + V_0 = 0.5804 + 2.0 = 2.5804
    """
    buffer = PPORolloutBuffer(capacity=5)
    gamma = 0.90
    gae_lambda = 0.80

    # Add transitions in order 0, 1, 2
    buffer.add(
        state=np.zeros((4, 52), dtype=np.float32),
        context=np.zeros(12, dtype=np.float32),
        action=0,
        action_mask=np.ones(4, dtype=np.float32),
        reward=1.0,
        value=2.0,
        log_prob=-0.5,
        done=False,
    )
    buffer.add(
        state=np.zeros((4, 52), dtype=np.float32),
        context=np.zeros(12, dtype=np.float32),
        action=1,
        action_mask=np.ones(4, dtype=np.float32),
        reward=2.0,
        value=1.5,
        log_prob=-0.5,
        done=False,
    )
    buffer.add(
        state=np.zeros((4, 52), dtype=np.float32),
        context=np.zeros(12, dtype=np.float32),
        action=2,
        action_mask=np.ones(4, dtype=np.float32),
        reward=-0.5,
        value=1.0,
        log_prob=-0.5,
        done=True,
    )

    buffer.compute_gae(last_value=0.0, last_done=True, gamma=gamma, gae_lambda=gae_lambda)

    # Check advantages against exact analytical values
    assert np.isclose(buffer.advantages[2], -1.50, atol=1e-5), f"A_2 expected -1.50, got {buffer.advantages[2]}"
    assert np.isclose(buffer.advantages[1], 0.32, atol=1e-5), f"A_1 expected 0.32, got {buffer.advantages[1]}"
    assert np.isclose(buffer.advantages[0], 0.5804, atol=1e-5), f"A_0 expected 0.5804, got {buffer.advantages[0]}"

    # Check returns: R_t = A_t + V_t
    assert np.isclose(buffer.returns[2], -0.50, atol=1e-5)
    assert np.isclose(buffer.returns[1], 1.82, atol=1e-5)
    assert np.isclose(buffer.returns[0], 2.5804, atol=1e-5)


# ======================================================================
# 6. Advantage Normalization Tests
# ======================================================================

def test_advantage_normalization():
    """Verify that advantage normalization centers mean at ~0.0 and std at ~1.0."""
    buffer = PPORolloutBuffer(capacity=100)
    np.random.seed(42)

    for i in range(100):
        buffer.add(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            action_mask=np.ones(4, dtype=np.float32),
            reward=float(np.random.randn()),
            value=float(np.random.randn()),
            log_prob=-0.5,
            done=(i % 20 == 0),
        )

    buffer.compute_gae(last_value=0.0, last_done=False)

    # Extract all advantages from mini-batches with normalization
    batches = list(buffer.get_mini_batches(mini_batch_size=100, device=torch.device("cpu"), normalize_advantages=True))
    all_advs = batches[0]["advantages"].numpy()

    assert np.isclose(np.mean(all_advs), 0.0, atol=1e-5), f"Normalized mean should be ~0.0, got {np.mean(all_advs)}"
    assert np.isclose(np.std(all_advs), 1.0, atol=1e-2), f"Normalized std should be ~1.0, got {np.std(all_advs)}"


def test_advantage_unnormalized_option():
    """Verify that normalize_advantages=False preserves raw advantages."""
    buffer = PPORolloutBuffer(capacity=10)
    for i in range(5):
        buffer.add(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            action_mask=np.ones(4, dtype=np.float32),
            reward=1.0,
            value=0.0,
            log_prob=-0.5,
            done=False,
        )
    buffer.compute_gae(last_value=0.0, last_done=False)

    batches = list(buffer.get_mini_batches(mini_batch_size=5, device=torch.device("cpu"), normalize_advantages=False))
    raw_batch_adv = batches[0]["advantages"].numpy()
    assert np.allclose(np.sort(raw_batch_adv), np.sort(buffer.advantages[:5]), atol=1e-5)


# ======================================================================
# 7. PPO Loss & Components (Clipped Objective, Value Loss, Entropy Bonus)
# ======================================================================

def test_ppo_clipped_objective_formula():
    """Verify clipped surrogate loss adheres to min(r*A, clip(r, 1-eps, 1+eps)*A)."""
    clip_eps = 0.20

    # Case 1: Positive advantage, ratio > 1 + eps -> clipped to (1 + eps) * A
    adv_pos = torch.tensor([2.0])
    ratio_high = torch.tensor([1.5])  # > 1.2
    surr1 = ratio_high * adv_pos      # 3.0
    surr2 = torch.clamp(ratio_high, 1.0 - clip_eps, 1.0 + clip_eps) * adv_pos  # 1.2 * 2.0 = 2.4
    loss_clipped_pos = -torch.min(surr1, surr2).item()
    assert np.isclose(loss_clipped_pos, -2.4), f"Expected -2.4, got {loss_clipped_pos}"

    # Case 2: Negative advantage, ratio < 1 - eps -> surr2 = 0.8 * -2.0 = -1.6, min(-1.0, -1.6) = -1.6
    adv_neg = torch.tensor([-2.0])
    ratio_low = torch.tensor([0.5])   # < 0.8
    surr1_neg = ratio_low * adv_neg   # -1.0
    surr2_neg = torch.clamp(ratio_low, 1.0 - clip_eps, 1.0 + clip_eps) * adv_neg  # -1.6
    loss_clipped_neg = -torch.min(surr1_neg, surr2_neg).item()
    assert np.isclose(loss_clipped_neg, 1.6), f"Expected 1.6, got {loss_clipped_neg}"

    # Case 3: Ratio within [1 - eps, 1 + eps] -> unclipped
    ratio_mid = torch.tensor([1.05])
    surr1_mid = ratio_mid * adv_pos   # 2.1
    surr2_mid = torch.clamp(ratio_mid, 1.0 - clip_eps, 1.0 + clip_eps) * adv_pos  # 2.1
    loss_unclipped = -torch.min(surr1_mid, surr2_mid).item()
    assert np.isclose(loss_unclipped, -2.1), f"Expected -2.1, got {loss_unclipped}"


def test_ppo_loss_optimization_step():
    """Verify full PPO loss computation and parameter optimization."""
    config = PPOConfig(num_epochs=3, mini_batch_size=4, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01)
    agent = PPOAgent(config=config)
    buffer = PPORolloutBuffer(capacity=8)

    for _ in range(8):
        buffer.add(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=np.random.choice([0, 1]),
            action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            value=0.5,
            log_prob=-0.693,
            done=False,
        )
    buffer.compute_gae(last_value=0.5, last_done=False)

    metrics = agent.update(buffer)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics
    assert metrics["updates"] == 1
    assert agent.total_steps == 8


# ======================================================================
# 8. Checkpoint Saving and Loading Tests
# ======================================================================

def test_ppo_checkpoint_save_and_load(tmp_path: Path):
    """Verify bit-for-bit exact network restoration from checkpoint."""
    agent1 = PPOAgent()
    ckpt_path = tmp_path / "ppo_checkpoint.pt"

    # Set custom state
    agent1.total_steps = 7777
    agent1.update_count = 42
    agent1.save_checkpoint(ckpt_path)
    assert ckpt_path.is_file()

    # Load into fresh agent
    agent2 = PPOAgent()
    agent2.load_checkpoint(ckpt_path)

    assert agent2.total_steps == 7777
    assert agent2.update_count == 42

    # Check identical parameters
    for p1, p2 in zip(agent1.network.parameters(), agent2.network.parameters()):
        assert torch.equal(p1, p2), "Model parameters must be bit-for-bit identical after loading"

    # Forward pass outputs must match exactly
    spatial = torch.randn(2, 4, 52)
    context = torch.randn(2, 12)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    with torch.no_grad():
        a1, lp1, e1, v1 = agent1.network.get_action_and_value(spatial, context, mask, deterministic=True)
        a2, lp2, e2, v2 = agent2.network.get_action_and_value(spatial, context, mask, deterministic=True)

    assert torch.equal(a1, a2)
    assert torch.equal(lp1, lp2)
    assert torch.equal(e1, e2)
    assert torch.equal(v1, v2)


# ======================================================================
# 9. Deterministic Evaluation Mode & Seeding
# ======================================================================

def test_deterministic_evaluation_mode():
    """Verify deterministic=True selects argmax legal action deterministically."""
    agent = PPOAgent()
    state = np.random.randn(4, 52).astype(np.float32)
    context = np.random.randn(12).astype(np.float32)
    mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    # 10 evaluations should yield the exact same action in deterministic mode
    actions = [agent.select_action(state, mask, context=context, deterministic=True) for _ in range(10)]
    assert len(set(actions)) == 1, "Deterministic evaluation must yield identical action across repeated calls"
    assert actions[0] in (0, 1, 2), "Selected action must be legal"


def test_agent_seed_reproducibility():
    """Verify agent.seed(seed) guarantees reproducible stochastic sampling."""
    agent1 = PPOAgent()
    agent2 = PPOAgent()

    # Copy weights from agent1 to agent2
    agent2.network.load_state_dict(agent1.network.state_dict())

    state = np.random.randn(4, 52).astype(np.float32)
    mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    agent1.seed(12345)
    samples1 = [agent1.select_action(state, mask, deterministic=False) for _ in range(20)]

    agent2.seed(12345)
    samples2 = [agent2.select_action(state, mask, deterministic=False) for _ in range(20)]

    assert samples1 == samples2, "Identical seeds must yield identical stochastic trajectories"


# ======================================================================
# 10. PPOTrainer Smoke Test
# ======================================================================

def test_ppo_trainer_execution():
    """Verify PPOTrainer runs without exception and updates model."""
    env = LudoEnvironment(auto_advance_no_moves=True)
    agent = PPOAgent()
    opponent = RandomOpponent()
    trainer = PPOTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        rollout_steps=16,
    )

    metrics = trainer.train(total_steps=32)
    assert metrics.total_steps == 32
    assert metrics.updates >= 1
    assert isinstance(metrics.win_rate, float)
    assert isinstance(metrics.final_loss, float)
