"""Unit and Integration Tests for Double Dueling DQN Baseline (Phase 6).

Verifies:
- Dueling Q-network architecture and stream shapes
- Prioritized Experience Replay buffer operations
- Double DQN update logic and Bellman loss computation
- Masked action selection (both greedy and exploration)
- Soft target network update
- Checkpoint saving and loading
- Pilot training loop execution (small verification run)
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from phd_ludo_drl.config.config_manager import DQNConfig
from phd_ludo_drl.environment.constants import TokenState
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.dqn.dqn_network import DuelingQNetwork
from phd_ludo_drl.agents.dqn.replay_buffer import PrioritizedReplayBuffer
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.dqn.trainer import DQNTrainer, DQNTrainingMetrics
from phd_ludo_drl.opponents.random_opponent import RandomOpponent


# =====================================================================
# 1. NETWORK ARCHITECTURE TESTS
# =====================================================================

def test_dueling_network_forward_shape():
    """Verify DuelingQNetwork maps spatial board tensor to 4 Q-values."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4)

    # Single spatial state
    single_x = torch.randn(4, 52)
    single_q = net(single_x)
    assert single_q.shape == (1, 4)

    # Batched spatial state (B=8)
    batch_x = torch.randn(8, 4, 52)
    batch_context = torch.randn(8, 12)
    batch_q = net(batch_x, batch_context)
    assert batch_q.shape == (8, 4)


def test_dueling_aggregation_math():
    """Verify Q-values equal V(s) + (A(s, a) - mean(A))."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4, use_internal_encoder=False)
    features = torch.randn(5, 128)

    q_vals = net(features)
    val = net.value_stream(features)
    adv = net.advantage_stream(features)
    expected_q = val + (adv - adv.mean(dim=-1, keepdim=True))

    assert torch.allclose(q_vals, expected_q, atol=1e-5)


# =====================================================================
# 2. PRIORITIZED EXPERIENCE REPLAY BUFFER TESTS
# =====================================================================

def test_replay_buffer_push_and_sample():
    """Verify transitions are stored and sampled correctly with IS weights."""
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)

    # Push 10 transitions
    for i in range(10):
        buffer.push(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=i % 4,
            reward=float(i),
            next_state=np.zeros((4, 52), dtype=np.float32),
            next_context=np.zeros(12, dtype=np.float32),
            done=(i == 9),
            action_mask=np.array([1, 1, 0, 0], dtype=np.float32),
            next_action_mask=np.array([0, 1, 1, 0], dtype=np.float32),
        )

    assert len(buffer) == 10

    batch, indices, weights = buffer.sample(batch_size=4)
    assert batch["states"].shape == (4, 4, 52)
    assert batch["rewards"].shape == (4,)
    assert len(indices) == 4
    assert weights.shape == (4,)
    assert weights.max().item() <= 1.0 + 1e-5  # normalized max <= 1.0


def test_replay_buffer_priority_update():
    """Verify TD errors update transition priorities."""
    buffer = PrioritizedReplayBuffer(capacity=10)
    for _ in range(5):
        buffer.push(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            reward=1.0,
            next_state=np.zeros((4, 52), dtype=np.float32),
            next_context=np.zeros(12, dtype=np.float32),
            done=False,
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )

    td_errors = np.array([5.0, 1.0, 0.1, 10.0])
    indices = np.array([0, 1, 2, 3])
    buffer.update_priorities(indices, td_errors)

    assert buffer.priorities[3] > buffer.priorities[0] > buffer.priorities[1]


# =====================================================================
# 3. DQN AGENT TESTS
# =====================================================================

def test_dqn_action_selection_respects_mask():
    """Verify DQNAgent never selects an illegal action in greedy or exploration mode."""
    agent = DQNAgent()
    dummy_spatial = np.random.randn(4, 52).astype(np.float32)
    mask = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # Only action 1 legal

    # Greedy test (deterministic=True)
    for _ in range(20):
        act = agent.select_action(dummy_spatial, mask, deterministic=True)
        assert act == 1

    # Exploration test (deterministic=False with epsilon=1.0)
    agent.epsilon = 1.0
    for _ in range(20):
        act = agent.select_action(dummy_spatial, mask, deterministic=False)
        assert act == 1


def test_dqn_double_q_train_step():
    """Verify Double DQN training step executes loss computation and backward pass."""
    agent = DQNAgent()

    # Populate buffer with mock transitions
    for i in range(100):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=i % 4,
            reward=0.25,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=False,
            action_mask=np.array([1, 1, 0, 0], dtype=np.float32),
            next_action_mask=np.array([0, 1, 1, 0], dtype=np.float32),
        )

    info = agent.train_step(batch_size=32)
    assert info is not None
    assert "loss" in info
    assert "mean_q" in info
    assert info["loss"] >= 0.0


def test_dqn_soft_target_update():
    """Verify target network weights smoothly track online network."""
    agent = DQNAgent()
    # Mutate online net parameter
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.add_(1.0)

    # Initial target param is different
    target_p_before = next(agent.target_net.parameters()).clone()
    online_p = next(agent.online_net.parameters()).clone()
    assert not torch.allclose(target_p_before, online_p)

    agent._soft_update(tau=0.5)
    target_p_after = next(agent.target_net.parameters()).clone()
    # Should move toward online param
    expected = 0.5 * online_p + 0.5 * target_p_before
    assert torch.allclose(target_p_after, expected, atol=1e-5)


def test_dqn_checkpoint_save_and_load(tmp_path: Path):
    """Verify DQNAgent checkpoint serialization and state restoration."""
    agent = DQNAgent()
    agent.total_steps = 1500
    agent.epsilon = 0.75

    ckpt_path = tmp_path / "test_dqn.pt"
    agent.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    new_agent = DQNAgent()
    new_agent.load_checkpoint(ckpt_path)
    assert new_agent.total_steps == 1500
    assert new_agent.epsilon == pytest.approx(0.75)


# =====================================================================
# 4. SMALL PILOT TRAINING VERIFICATION
# =====================================================================

def test_dqn_pilot_training_loop():
    """Run a small pilot training loop (50 steps) to verify end-to-end integration."""
    env = LudoEnvironment(seed=42, auto_advance_no_moves=True)
    agent = DQNAgent()
    opponent = RandomOpponent(seed=101)

    trainer = DQNTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=16,
        warmup_steps=20,
    )

    metrics = trainer.train(total_steps=50)

    assert isinstance(metrics, DQNTrainingMetrics)
    assert metrics.total_steps == 50
    assert metrics.episodes >= 0
    assert len(agent.buffer) >= 20
