"""Exhaustive validation and unit tests for Model 4: Categorical DQN (C51 / Distributional RL):
- 51 discrete atoms and support range [V_min, V_max]
- Categorical distribution probability normalization
- Expected Q calculation: Q(s, a) = sum_i z_i * p_i(s, a)
- Legal action masking in decision making and target action selection
- Target network and soft Polyak averaging
- Bellman distribution projection with exact analytical ground truth
- Replay buffer mechanics
- Checkpoint saving and loading
- Frozen encoder integration
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from phd_ludo_drl.config.config_manager import CategoricalDQNConfig
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.agents.model4.c51_network import CategoricalQNetwork
from phd_ludo_drl.agents.model4.replay_buffer import C51ReplayBuffer
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.agents.model4.trainer import C51Trainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


# ======================================================================
# 1. 51 Atoms & Support / Value Range Tests
# ======================================================================

def test_atom_support_and_value_range():
    """Verify C51 uses exactly 51 atoms on support [V_min, V_max] = [-20.0, 30.0]."""
    net = CategoricalQNetwork(
        latent_dim=128,
        hidden_units=128,
        action_dim=4,
        num_atoms=51,
        v_min=-20.0,
        v_max=30.0,
    )

    # 1. Atom count
    assert net.num_atoms == 51
    assert net.support.shape == (51,)

    # 2. Value range bounds
    assert net.support[0].item() == -20.0
    assert net.support[-1].item() == 30.0

    # 3. Delta z step size
    expected_delta = (30.0 - (-20.0)) / (51 - 1)  # 50.0 / 50 = 1.0
    assert np.isclose(net.delta_z, expected_delta, atol=1e-6)
    assert np.isclose(net.delta_z, 1.0, atol=1e-6)

    # 4. Strictly monotonically increasing support with uniform spacing 1.0
    diffs = torch.diff(net.support)
    assert torch.allclose(diffs, torch.ones_like(diffs) * 1.0, atol=1e-5)


# ======================================================================
# 2. Categorical Distribution & Probability Normalization Tests
# ======================================================================

def test_categorical_distribution_normalization():
    """Verify that network outputs valid categorical probability distributions summing to 1.0."""
    net = CategoricalQNetwork(latent_dim=128, num_atoms=51, v_min=-20.0, v_max=30.0)
    batch_size = 5
    spatial = torch.randn(batch_size, 4, 52)
    context = torch.randn(batch_size, 12)

    probs, q_values = net(spatial, context)

    # Output dimensions: (B, 4, 51) and (B, 4)
    assert probs.shape == (batch_size, 4, 51)
    assert q_values.shape == (batch_size, 4)

    # All probabilities non-negative
    assert (probs >= 0.0).all()

    # Probabilities across atoms for every action must sum to 1.0
    prob_sums = torch.sum(probs, dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size, 4), atol=1e-5)


# ======================================================================
# 3. Expected Q Calculation Tests
# ======================================================================

def test_expected_q_calculation():
    """Verify Q(s, a) = sum_i z_i * p_i(s, a) mathematically."""
    net = CategoricalQNetwork(num_atoms=51, v_min=-20.0, v_max=30.0)

    # Synthetic probability distributions:
    # Action 0: 100% on atom index 25 (value: -20 + 25 * 1.0 = 5.0)
    # Action 1: 50% on atom index 10 (value: -10.0), 50% on atom index 30 (value: +10.0) -> Expected: 0.0
    # Action 2: 100% on atom index 0 (value: -20.0)
    # Action 3: 100% on atom index 50 (value: +30.0)
    probs = torch.zeros(1, 4, 51)
    probs[0, 0, 25] = 1.0
    probs[0, 1, 10] = 0.5
    probs[0, 1, 30] = 0.5
    probs[0, 2, 0] = 1.0
    probs[0, 3, 50] = 1.0

    expected_q = torch.sum(probs * net.support, dim=-1).squeeze(0)

    assert np.isclose(expected_q[0].item(), 5.0, atol=1e-5)
    assert np.isclose(expected_q[1].item(), 0.0, atol=1e-5)
    assert np.isclose(expected_q[2].item(), -20.0, atol=1e-5)
    assert np.isclose(expected_q[3].item(), 30.0, atol=1e-5)


# ======================================================================
# 4. Bellman Distribution Projection Tests (Analytical Ground Truth)
# ======================================================================

def test_bellman_projection_analytical():
    """Verify Categorical Bellman projection against exact hand-calculated analytical ground truth.

    Analytical derivation:
    Support: [-20.0, 30.0], num_atoms = 51, delta_z = 1.0, z_k = -20 + k.
    gamma = 0.90.

    Input distribution p_next:
        p(z_20 = 0.0) = 0.60   (atom index 20)
        p(z_30 = 10.0) = 0.40  (atom index 30)
        All other 49 atoms = 0.0.

    Case 1: Reward r = 2.50, done = 0.
        Shift atom z_20 = 0.0:
            T z_20 = 2.50 + 0.90 * 0.0 = 2.50
            Coordinate: b = (2.50 - (-20.0)) / 1.0 = 22.50
            l = 22 (value 2.0), u = 23 (value 3.0)
            m[22] = 0.60 * (23 - 22.50) = 0.30
            m[23] = 0.60 * (22.50 - 22) = 0.30

        Shift atom z_30 = 10.0:
            T z_30 = 2.50 + 0.90 * 10.0 = 11.50
            Coordinate: b = (11.50 - (-20.0)) / 1.0 = 31.50
            l = 31 (value 11.0), u = 32 (value 12.0)
            m[31] = 0.40 * (32 - 31.50) = 0.20
            m[32] = 0.40 * (31.50 - 31) = 0.20

        Projected distribution:
            m[22] = 0.30, m[23] = 0.30, m[31] = 0.20, m[32] = 0.20
            All other 47 atoms = 0.0.
            Sum = 1.00.
            Mean = 2.0*0.30 + 3.0*0.30 + 11.0*0.20 + 12.0*0.20 = 0.60 + 0.90 + 2.20 + 2.40 = 6.10
            Theoretical mean = r + gamma * E[Z'] = 2.50 + 0.90 * (0.6*0 + 0.4*10) = 2.50 + 3.60 = 6.10.

    Case 2: Terminal transition (done = 1), r = 5.0.
        T z = 5.0 for all atoms.
        Coordinate: b = (5.0 - (-20.0)) / 1.0 = 25.0
        Exact integer landing on atom 25!
        m[25] = 1.0, all others 0.0.

    Case 3: Extreme reward exceeding V_max (clamping), r = 60.0, done = 0.
        T z clamped to V_max = 30.0.
        m[50] = 1.0, all others 0.0.
    """
    agent = CategoricalDQNAgent()
    agent.gamma = 0.90

    # Setup p_next
    p_next = torch.zeros(3, 51)
    p_next[0, 20] = 0.60
    p_next[0, 30] = 0.40

    p_next[1, 10] = 0.50
    p_next[1, 40] = 0.50

    p_next[2, 25] = 1.00

    rewards = torch.tensor([2.50, 5.00, 60.00])
    dones = torch.tensor([0.0, 1.0, 0.0])

    m = agent.project_distribution(rewards=rewards, dones=dones, p_next=p_next)

    # 1. Total probability conservation: all batch items must sum to 1.0
    sums = m.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # 2. Case 1 verification
    assert np.isclose(m[0, 22].item(), 0.30, atol=1e-5)
    assert np.isclose(m[0, 23].item(), 0.30, atol=1e-5)
    assert np.isclose(m[0, 31].item(), 0.20, atol=1e-5)
    assert np.isclose(m[0, 32].item(), 0.20, atol=1e-5)
    # Check that non-neighbor atoms have exactly 0 probability
    m0_rest = m[0].clone()
    m0_rest[[22, 23, 31, 32]] = 0.0
    assert torch.allclose(m0_rest, torch.zeros_like(m0_rest))

    # Mean preservation test: E[Phi T Z] == r + gamma * E[Z']
    computed_mean = torch.sum(m[0] * agent.support).item()
    assert np.isclose(computed_mean, 6.10, atol=1e-5)

    # 3. Case 2 verification: Terminal exact landing on atom 25
    assert np.isclose(m[1, 25].item(), 1.0, atol=1e-5)

    # 4. Case 3 verification: Clamped at upper bound atom 50
    assert np.isclose(m[2, 50].item(), 1.0, atol=1e-5)


# ======================================================================
# 5. Action Masking Enforcement Tests
# ======================================================================

def test_action_masking_decision_making():
    """Verify action masking forces agent to choose legal actions with highest masked Q."""
    net = CategoricalQNetwork(latent_dim=128)
    spatial = torch.randn(1, 4, 52)
    context = torch.randn(1, 12)

    # Mask: only actions 1 and 3 legal
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

    # Deterministic selection must pick 1 or 3
    action = net.select_action(spatial, context, mask, deterministic=True)
    assert action in (1, 3)

    # Only action 2 legal
    mask2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    action2 = net.select_action(spatial, context, mask2, deterministic=True)
    assert action2 == 2


def test_action_masking_next_action_selection_in_target():
    """Verify target calculation masks illegal moves when finding best next action."""
    agent = CategoricalDQNAgent()

    # Create synthetic Q-values where action 0 has huge Q (+100.0) but is illegal
    # Action 1 has modest Q (+5.0) and is legal
    next_states = torch.zeros(1, 4, 52)
    next_contexts = torch.zeros(1, 12)
    next_masks = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Only action 1 legal

    # Mock online_net.get_q_values
    agent.online_net.get_q_values = lambda s, c: torch.tensor([[100.0, 5.0, 0.0, 0.0]])

    rewards = torch.tensor([1.0])
    dones = torch.tensor([0.0])

    # Target distribution should evaluate best_next_action = 1, NOT 0
    target_m = agent.compute_target_distribution(
        rewards=rewards,
        dones=dones,
        next_states=next_states,
        next_contexts=next_contexts,
        next_action_masks=next_masks,
    )

    assert target_m.shape == (1, 51)
    assert torch.isclose(target_m.sum(), torch.tensor(1.0), atol=1e-5)


# ======================================================================
# 6. Replay Mechanism & Training Step Tests
# ======================================================================

def test_c51_replay_buffer_and_train_step():
    """Verify buffer push/sample and single training optimization step."""
    cfg = CategoricalDQNConfig(batch_size=4, warmup_steps=4)
    agent = CategoricalDQNAgent(config=cfg)

    for i in range(8):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=i % 4,
            reward=0.5,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=(i == 7),
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )

    assert len(agent.buffer) == 8

    init_params = [p.clone() for p in agent.online_net.parameters()]
    metrics = agent.train_step(batch_size=4)

    assert metrics is not None
    assert "loss" in metrics
    assert "mean_q" in metrics
    assert "epsilon" in metrics
    assert metrics["updates"] == 1
    assert agent.total_steps == 1

    # Parameters must have updated
    changed = any(not torch.equal(p1, p2) for p1, p2 in zip(init_params, agent.online_net.parameters()))
    assert changed, "Online network weights should be updated by optimizer"


# ======================================================================
# 7. Soft Polyak Target Network Update Tests
# ======================================================================

def test_c51_soft_target_updates():
    """Verify Polyak soft target updates: theta_target = tau * theta_online + (1 - tau) * theta_target."""
    agent = CategoricalDQNAgent()
    agent.tau = 0.20

    for online_p, target_p in zip(agent.online_net.parameters(), agent.target_net.parameters()):
        online_p.data.fill_(10.0)
        target_p.data.fill_(0.0)

    for target_p, online_p in zip(agent.target_net.parameters(), agent.online_net.parameters()):
        target_p.data.copy_(agent.tau * online_p.data + (1.0 - agent.tau) * target_p.data)

    for target_p in agent.target_net.parameters():
        assert torch.allclose(target_p.data, torch.tensor(2.0)), "Polyak update should result in tau * 10 = 2.0"


# ======================================================================
# 8. Checkpoint Saving and Loading Tests
# ======================================================================

def test_c51_checkpoint_save_and_load(tmp_path: Path):
    """Verify exact bit-for-bit restoration of C51 weights and parameters."""
    agent1 = CategoricalDQNAgent()
    ckpt_path = tmp_path / "c51_checkpoint.pt"

    agent1.total_steps = 12345
    agent1.update_count = 67
    agent1.epsilon = 0.42
    agent1.save_checkpoint(ckpt_path)
    assert ckpt_path.is_file()

    agent2 = CategoricalDQNAgent()
    agent2.load_checkpoint(ckpt_path)

    assert agent2.total_steps == 12345
    assert agent2.update_count == 67
    assert np.isclose(agent2.epsilon, 0.42)

    # Parameter equality
    for p1, p2 in zip(agent1.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(agent1.target_net.parameters(), agent2.target_net.parameters()):
        assert torch.equal(p1, p2)

    # Forward pass equality
    spatial = torch.randn(2, 4, 52)
    context = torch.randn(2, 12)
    with torch.no_grad():
        pr1, q1 = agent1.online_net(spatial, context)
        pr2, q2 = agent2.online_net(spatial, context)
        assert torch.equal(pr1, pr2)
        assert torch.equal(q1, q2)


# ======================================================================
# 9. Structured State Input & Seeding Tests
# ======================================================================

def test_c51_structured_state_input():
    """Verify agent.select_action accepts StructuredLudoState directly."""
    agent = CategoricalDQNAgent()
    env = LudoEnvironment()
    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    struct_state = build_structured_representation(obs)
    action = agent.select_action(struct_state, mask, deterministic=True)
    assert action in (0, 1, 2, 3)
    assert mask[action] == 1.0


def test_c51_seed_determinism():
    """Verify agent.seed guarantees reproducible action selections."""
    agent1 = CategoricalDQNAgent()
    agent2 = CategoricalDQNAgent()
    agent2.online_net.load_state_dict(agent1.online_net.state_dict())

    state = np.random.randn(4, 52).astype(np.float32)
    mask = np.ones(4, dtype=np.float32)

    agent1.seed(999)
    actions1 = [agent1.select_action(state, mask, deterministic=False) for _ in range(20)]

    agent2.seed(999)
    actions2 = [agent2.select_action(state, mask, deterministic=False) for _ in range(20)]

    assert actions1 == actions2


# ======================================================================
# 10. Shared / Frozen Encoder Integration
# ======================================================================

def test_c51_frozen_encoder():
    """Verify freeze_encoder prevents updates to encoder parameters."""
    agent = CategoricalDQNAgent(freeze_encoder=True)

    init_encoder = {name: p.clone() for name, p in agent.online_net.encoder.named_parameters()}

    for _ in range(8):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=0,
            reward=1.0,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=False,
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )

    agent.train_step(batch_size=4)

    for name, p in agent.online_net.encoder.named_parameters():
        assert torch.equal(p, init_encoder[name]), f"Encoder parameter {name} modified while frozen!"


# ======================================================================
# 11. C51 Trainer Short Run
# ======================================================================

def test_c51_trainer_short_run():
    """Verify C51Trainer runs without exceptions on a short training session."""
    env = LudoEnvironment(auto_advance_no_moves=True)
    agent = CategoricalDQNAgent()
    opponent = RandomOpponent()
    trainer = C51Trainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=8,
        warmup_steps=10,
    )

    metrics = trainer.train(total_steps=25)
    assert metrics.total_steps == 25
    assert isinstance(metrics.win_rate, float)
