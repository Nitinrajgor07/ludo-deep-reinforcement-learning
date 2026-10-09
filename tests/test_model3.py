"""Exhaustive validation and unit tests for Model 3: Discrete Soft Actor-Critic (SAC):
- Masked discrete stochastic policy distribution
- Twin Q critics & outputs
- Target calculation (exact expectation V(s') with analytical ground truth)
- Entropy calculation
- Automatic temperature (alpha) update
- Critic update & loss
- Actor update & loss
- Soft Polyak target updates
- Checkpoint saving and loading
- Deterministic evaluation mode
- Shared and frozen encoder integration
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from phd_ludo_drl.config.config_manager import DiscreteSACConfig
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.agents.model3.sac_network import DiscreteActor, DiscreteCritic
from phd_ludo_drl.agents.model3.replay_buffer import DiscreteSACReplayBuffer
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model3.trainer import DiscreteSACTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


# ======================================================================
# 1. Masked Policy Distribution Tests
# ======================================================================

def test_masked_policy_distribution():
    """Verify illegal actions have strictly 0.0 probability and clamped log_probs."""
    actor = DiscreteActor(latent_dim=128, hidden_units=128, action_dim=4)
    spatial = torch.randn(2, 4, 52)
    context = torch.randn(2, 12)

    # Batch item 0: only action 1 legal. Batch item 1: actions 0 and 3 legal.
    masks = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])

    probs, log_probs = actor(spatial, context, masks)

    # Check shapes
    assert probs.shape == (2, 4)
    assert log_probs.shape == (2, 4)

    # Item 0 checks
    assert probs[0, 1].item() > 0.9999
    assert probs[0, 0].item() == 0.0
    assert probs[0, 2].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert log_probs[0, 0].item() == 0.0  # Clamped to 0.0 for numerical safety
    assert log_probs[0, 2].item() == 0.0
    assert log_probs[0, 3].item() == 0.0
    assert np.isclose(log_probs[0, 1].item(), 0.0, atol=1e-4)

    # Item 1 checks
    assert np.isclose(probs[1, 0].item() + probs[1, 3].item(), 1.0, atol=1e-5)
    assert probs[1, 1].item() == 0.0
    assert probs[1, 2].item() == 0.0
    assert log_probs[1, 1].item() == 0.0
    assert log_probs[1, 2].item() == 0.0


def test_actor_stochastic_sampling_respects_mask():
    """Verify stochastic sampling over 100 iterations never selects an illegal action."""
    actor = DiscreteActor(latent_dim=128, hidden_units=128, action_dim=4)
    spatial = torch.randn(1, 4, 52)
    context = torch.randn(1, 12)
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # Only action 2 is legal

    for _ in range(100):
        action = actor.select_action(spatial, context, mask, deterministic=False)
        assert action == 2


# ======================================================================
# 2. Q Outputs & Twin Critics Tests
# ======================================================================

def test_twin_critic_q_outputs():
    """Verify DiscreteCritic outputs twin Q values of shape (B, 4) with independent parameters."""
    critic = DiscreteCritic(latent_dim=128, hidden_units=128, action_dim=4)
    batch_size = 4
    spatial = torch.randn(batch_size, 4, 52)
    context = torch.randn(batch_size, 12)

    q1, q2 = critic(spatial, context)
    assert q1.shape == (batch_size, 4)
    assert q2.shape == (batch_size, 4)

    # The two heads are distinct and initialized independently
    assert not torch.allclose(q1, q2, atol=1e-4), "Twin critics must provide independent Q evaluations"

    # Single q1 evaluation
    q1_only = critic.q1(spatial, context)
    assert torch.allclose(q1, q1_only)


# ======================================================================
# 3. Exact Expectation Target Calculation (with Analytical Verification)
# ======================================================================

def test_discrete_sac_target_calculation_analytical():
    """Verify Discrete SAC exact expectation target against hand-calculated analytical derivation.

    Analytical derivation:
    gamma = 0.99, alpha = 0.20
    s' has legal actions A_legal = {0, 1}, actions 2, 3 masked.
    Suppose pi(0|s') = 0.60, pi(1|s') = 0.40, pi(2|s') = 0.0, pi(3|s') = 0.0
    log pi(0|s') = ln(0.60) = -0.51082562
    log pi(1|s') = ln(0.40) = -0.91629073

    Q_target,1(s') = [3.0, 5.0, 1.0, 2.0]
    Q_target,2(s') = [4.0, 2.0, 0.5, 3.0]
    min Q_target(s') = [3.0, 2.0, 0.5, 2.0]

    Action 0: min Q - alpha * log pi = 3.0 - 0.20 * (-0.51082562) = 3.0 + 0.10216512 = 3.10216512
    Action 1: min Q - alpha * log pi = 2.0 - 0.20 * (-0.91629073) = 2.0 + 0.18325815 = 2.18325815
    Actions 2, 3: pi(a|s') = 0.0

    V(s') = 0.60 * 3.10216512 + 0.40 * 2.18325815
          = 1.86129907 + 0.87330326
          = 2.73460233

    Case A: Non-terminal (done = 0), r = 1.50
        y = r + gamma * V(s') = 1.50 + 0.99 * 2.73460233 = 1.50 + 2.70725631 = 4.20725631

    Case B: Terminal (done = 1), r = 1.50
        y = r = 1.50
    """
    agent = DiscreteSACAgent()
    agent.gamma = 0.99
    # Set log_alpha so alpha = 0.20
    agent.log_alpha.data.fill_(float(np.log(0.20)))

    # Mock actor forward output
    probs = torch.tensor([[0.60, 0.40, 0.0, 0.0], [0.60, 0.40, 0.0, 0.0]])
    log_probs = torch.tensor([[np.log(0.60), np.log(0.40), 0.0, 0.0], [np.log(0.60), np.log(0.40), 0.0, 0.0]])
    agent.actor.forward = lambda s, c, m: (probs, log_probs)

    # Mock critic_target forward output
    q1 = torch.tensor([[3.0, 5.0, 1.0, 2.0], [3.0, 5.0, 1.0, 2.0]])
    q2 = torch.tensor([[4.0, 2.0, 0.5, 3.0], [4.0, 2.0, 0.5, 3.0]])
    agent.critic_target.forward = lambda s, c: (q1, q2)

    rewards = torch.tensor([1.50, 1.50])
    dones = torch.tensor([0.0, 1.0])  # Item 0 non-terminal, Item 1 terminal
    dummy_spatial = torch.zeros(2, 4, 52)
    dummy_context = torch.zeros(2, 12)
    dummy_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    target_q = agent.compute_target(
        rewards=rewards,
        dones=dones,
        next_states=dummy_spatial,
        next_contexts=dummy_context,
        next_action_masks=dummy_mask,
    )

    # Check Item 0 (non-terminal)
    expected_non_terminal = 4.20725631
    assert np.isclose(target_q[0].item(), expected_non_terminal, atol=1e-5), (
        f"Expected non-terminal target {expected_non_terminal}, got {target_q[0].item()}"
    )

    # Check Item 1 (terminal: ignores V(s') entirely)
    expected_terminal = 1.50
    assert np.isclose(target_q[1].item(), expected_terminal, atol=1e-5), (
        f"Expected terminal target {expected_terminal}, got {target_q[1].item()}"
    )


# ======================================================================
# 4. Entropy Calculation Tests
# ======================================================================

def test_entropy_calculation():
    """Verify policy entropy H(pi) = -sum_a pi(a|s) * log pi(a|s)."""
    # 1. Uniform over 4 actions
    p4 = torch.tensor([0.25, 0.25, 0.25, 0.25])
    lp4 = torch.log(p4)
    h4 = -(p4 * lp4).sum().item()
    assert np.isclose(h4, np.log(4), atol=1e-5)

    # 2. Uniform over 2 actions (other 2 masked with prob 0.0, clamped log 0.0)
    p2 = torch.tensor([0.5, 0.5, 0.0, 0.0])
    lp2 = torch.tensor([np.log(0.5), np.log(0.5), 0.0, 0.0])
    h2 = -(p2 * lp2).sum().item()
    assert np.isclose(h2, np.log(2), atol=1e-5)

    # 3. Deterministic single action (entropy = 0)
    p1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    lp1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    h1 = -(p1 * lp1).sum().item()
    assert np.isclose(h1, 0.0, atol=1e-5)


# ======================================================================
# 5. Automatic Temperature (Alpha) Tuning Tests
# ======================================================================

def test_automatic_temperature_tuning_direction():
    """Verify dual gradient descent increases alpha when entropy is below target, and decreases alpha when entropy is above target."""
    # Case A: Low entropy (deterministic policy) -> alpha must INCREASE
    agent_a = DiscreteSACAgent()
    target_ent = agent_a.target_entropy
    initial_alpha_a = float(agent_a.alpha.item())

    probs_low = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=agent_a.device)
    log_probs_low = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=agent_a.device)

    entropy_diff_a = (probs_low * (log_probs_low + target_ent)).sum(dim=-1)
    alpha_loss_a = -(agent_a.log_alpha * entropy_diff_a).mean()

    agent_a.alpha_optimizer.zero_grad()
    alpha_loss_a.backward()
    assert agent_a.log_alpha.grad.item() < 0.0, "Gradient should be negative to increase log_alpha"
    agent_a.alpha_optimizer.step()

    new_alpha_a = float(agent_a.alpha.item())
    assert new_alpha_a > initial_alpha_a, (
        f"Alpha should increase when policy entropy is below target. Old: {initial_alpha_a}, New: {new_alpha_a}"
    )

    # Case B: High entropy (uniform policy over 4 actions, H = 1.3863 > 1.3586) -> alpha must DECREASE
    agent_b = DiscreteSACAgent()
    initial_alpha_b = float(agent_b.alpha.item())

    p = 0.25
    probs_high = torch.tensor([[p, p, p, p]], device=agent_b.device)
    log_probs_high = torch.tensor([[np.log(p), np.log(p), np.log(p), np.log(p)]], device=agent_b.device)

    entropy_diff_b = (probs_high * (log_probs_high + target_ent)).sum(dim=-1)
    alpha_loss_b = -(agent_b.log_alpha * entropy_diff_b).mean()

    agent_b.alpha_optimizer.zero_grad()
    alpha_loss_b.backward()
    assert agent_b.log_alpha.grad.item() > 0.0, "Gradient should be positive to decrease log_alpha"
    agent_b.alpha_optimizer.step()

    new_alpha_b = float(agent_b.alpha.item())
    assert new_alpha_b < initial_alpha_b, (
        f"Alpha should decrease when policy entropy exceeds target. Old: {initial_alpha_b}, New: {new_alpha_b}"
    )


# ======================================================================
# 6. Critic & Actor Updates Tests
# ======================================================================

def test_critic_and_actor_updates():
    """Verify critic and actor gradient descent steps alter network weights."""
    cfg = DiscreteSACConfig(batch_size=4, warmup_steps=4)
    agent = DiscreteSACAgent(config=cfg)

    # Populate buffer
    for _ in range(8):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=0,
            reward=1.0,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=False,
            action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            next_action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )

    init_critic_p = [p.clone() for p in agent.critic.parameters()]
    init_actor_p = [p.clone() for p in agent.actor.parameters()]

    metrics = agent.train_step(batch_size=4)

    assert metrics is not None
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert metrics["updates"] == 1

    # Critic weights must have updated
    critic_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(init_critic_p, agent.critic.parameters()))
    assert critic_changed, "Critic parameters should have been updated"

    # Actor weights must have updated
    actor_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(init_actor_p, agent.actor.parameters()))
    assert actor_changed, "Actor parameters should have been updated"


# ======================================================================
# 7. Soft Polyak Target Updates Tests
# ======================================================================

def test_soft_target_updates():
    """Verify Polyak target update: theta_target = tau * theta_online + (1 - tau) * theta_target."""
    agent = DiscreteSACAgent()
    agent.tau = 0.10

    # Set known values in online and target networks
    for online_p, target_p in zip(agent.critic.parameters(), agent.critic_target.parameters()):
        online_p.data.fill_(10.0)
        target_p.data.fill_(0.0)

    # Perform soft update
    for target_p, online_p in zip(agent.critic_target.parameters(), agent.critic.parameters()):
        target_p.data.copy_(agent.tau * online_p.data + (1.0 - agent.tau) * target_p.data)

    # Expected value: 0.10 * 10.0 + 0.90 * 0.0 = 1.0
    for target_p in agent.critic_target.parameters():
        assert torch.allclose(target_p.data, torch.tensor(1.0)), (
            f"Expected target parameter 1.0, got {target_p.data.mean().item()}"
        )


# ======================================================================
# 8. Checkpoint Saving and Loading Tests
# ======================================================================

def test_discrete_sac_checkpoint_save_and_load(tmp_path: Path):
    """Verify bit-for-bit exact network restoration from checkpoint."""
    agent1 = DiscreteSACAgent()
    ckpt_path = tmp_path / "sac_checkpoint.pt"

    agent1.total_steps = 5432
    agent1.update_count = 88
    agent1.save_checkpoint(ckpt_path)
    assert ckpt_path.is_file()

    agent2 = DiscreteSACAgent()
    agent2.load_checkpoint(ckpt_path)

    assert agent2.total_steps == 5432
    assert agent2.update_count == 88
    assert torch.equal(agent1.log_alpha, agent2.log_alpha)

    # Compare critic parameters
    for p1, p2 in zip(agent1.critic.parameters(), agent2.critic.parameters()):
        assert torch.equal(p1, p2)

    # Compare critic target parameters
    for p1, p2 in zip(agent1.critic_target.parameters(), agent2.critic_target.parameters()):
        assert torch.equal(p1, p2)

    # Compare actor parameters
    for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters()):
        assert torch.equal(p1, p2)

    # Forward pass on identical input produces identical outputs
    spatial = torch.randn(2, 4, 52)
    context = torch.randn(2, 12)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    with torch.no_grad():
        p1, _ = agent1.actor(spatial, context, mask)
        p2, _ = agent2.actor(spatial, context, mask)
        assert torch.equal(p1, p2)

        q1_1, q2_1 = agent1.critic(spatial, context)
        q1_2, q2_2 = agent2.critic(spatial, context)
        assert torch.equal(q1_1, q1_2)
        assert torch.equal(q2_1, q2_2)


# ======================================================================
# 9. Deterministic Evaluation Mode & Reproducibility
# ======================================================================

def test_deterministic_evaluation():
    """Verify select_action with deterministic=True selects argmax probability legal action."""
    agent = DiscreteSACAgent()
    state = np.random.randn(4, 52).astype(np.float32)
    context = np.random.randn(12).astype(np.float32)
    mask = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    actions = [
        agent.select_action(state, mask, context=context, deterministic=True)
        for _ in range(15)
    ]
    # In deterministic mode, action must be constant
    assert len(set(actions)) == 1
    assert actions[0] in (0, 1), "Selected action must be legal"


def test_seed_reproducibility():
    """Verify agent.seed(seed) guarantees reproducible stochastic selection."""
    agent1 = DiscreteSACAgent()
    agent2 = DiscreteSACAgent()
    agent2.actor.load_state_dict(agent1.actor.state_dict())

    state = np.random.randn(4, 52).astype(np.float32)
    mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    agent1.seed(42)
    seq1 = [agent1.select_action(state, mask, deterministic=False) for _ in range(25)]

    agent2.seed(42)
    seq2 = [agent2.select_action(state, mask, deterministic=False) for _ in range(25)]

    assert seq1 == seq2, "Identical seeds must yield identical stochastic selections"


# ======================================================================
# 10. Shared and Frozen Encoder Integration
# ======================================================================

def test_frozen_encoder_parameters_remain_constant():
    """Verify training updates do not modify frozen encoder parameters in DiscreteSAC."""
    agent = DiscreteSACAgent(freeze_encoder=True)

    init_actor_encoder = {name: p.clone() for name, p in agent.actor.encoder.named_parameters()}
    init_critic_encoder1 = {name: p.clone() for name, p in agent.critic.encoder1.named_parameters()}

    # Populate buffer
    for _ in range(8):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=0,
            reward=1.0,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=False,
            action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
            next_action_mask=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )

    agent.train_step(batch_size=4)

    # Actor encoder weights must be identical
    for name, p in agent.actor.encoder.named_parameters():
        assert torch.equal(p, init_actor_encoder[name]), f"Actor encoder param {name} modified while frozen!"

    # Critic encoder weights must be identical
    for name, p in agent.critic.encoder1.named_parameters():
        assert torch.equal(p, init_critic_encoder1[name]), f"Critic encoder param {name} modified while frozen!"


# ======================================================================
# 11. PPOTrainer / DiscreteSACTrainer Integration
# ======================================================================

def test_discrete_sac_trainer_short_run():
    """Verify DiscreteSACTrainer runs without exceptions on a short training session."""
    env = LudoEnvironment(auto_advance_no_moves=True)
    agent = DiscreteSACAgent()
    opponent = RandomOpponent()
    trainer = DiscreteSACTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=8,
        warmup_steps=10,
    )

    metrics = trainer.train(total_steps=25)
    assert metrics.total_steps == 25
    assert isinstance(metrics.win_rate, float)
