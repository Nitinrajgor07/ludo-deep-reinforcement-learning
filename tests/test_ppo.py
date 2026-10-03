"""Unit tests for PPO baseline: ActorCriticNetwork, PPORolloutBuffer, PPOAgent, and PPOTrainer."""

import numpy as np
import pytest
import torch

from phd_ludo_drl.config.config_manager import PPOConfig
from phd_ludo_drl.agents.ppo.ppo_network import ActorCriticNetwork
from phd_ludo_drl.agents.ppo.rollout_buffer import PPORolloutBuffer
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.ppo.trainer import PPOTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


def test_actor_critic_forward_shapes():
    """Verify ActorCriticNetwork output dimensions and types."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    batch_size = 4
    spatial = torch.zeros(batch_size, 4, 52)
    context = torch.zeros(batch_size, 12)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * batch_size)

    actions, log_probs, entropy, values = net.get_action_and_value(spatial, context, mask)

    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)

    # Value-only evaluation
    val = net.get_value(spatial, context)
    assert val.shape == (batch_size,)


def test_action_masking_enforcement():
    """Verify that illegal actions have zero probability and are never sampled."""
    net = ActorCriticNetwork(latent_dim=128, hidden_units=128)
    spatial = torch.zeros(1, 4, 52)
    context = torch.zeros(1, 12)
    # Only action 2 is legal
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    for _ in range(50):
        action, _, _, _ = net.get_action_and_value(spatial, context, mask, deterministic=False)
        assert action.item() == 2


def test_rollout_buffer_gae():
    """Verify GAE and returns computation in PPORolloutBuffer."""
    buffer = PPORolloutBuffer(capacity=10)
    gamma = 0.99
    gae_lambda = 0.95

    # Push 5 synthetic transitions
    for i in range(5):
        buffer.add(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=i % 4,
            action_mask=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            value=0.5,
            log_prob=-0.5,
            done=(i == 4),
        )

    assert len(buffer) == 5
    buffer.compute_gae(last_value=0.0, last_done=True, gamma=gamma, gae_lambda=gae_lambda)

    # Returns should equal advantages + values
    for i in range(5):
        assert np.isclose(buffer.returns[i], buffer.advantages[i] + buffer.values[i], atol=1e-5)

    # Mini-batch generator
    batches = list(buffer.get_mini_batches(mini_batch_size=2, device=torch.device("cpu")))
    assert len(batches) == 3  # 5 items split into chunks of 2, 2, 1
    total_samples = sum(len(b["actions"]) for b in batches)
    assert total_samples == 5


def test_ppo_agent_step_and_select_action():
    """Verify PPOAgent action selection and step method."""
    agent = PPOAgent()
    state = np.zeros((4, 52), dtype=np.float32)
    mask = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    action = agent.select_action(state, mask, deterministic=True)
    assert action == 1

    act, log_prob, val = agent.step(state, mask)
    assert act == 1
    assert isinstance(log_prob, float)
    assert isinstance(val, float)


def test_ppo_agent_update():
    """Verify PPO clipped loss optimization step."""
    cfg = PPOConfig(num_epochs=2, mini_batch_size=4)
    agent = PPOAgent(config=cfg)
    buffer = PPORolloutBuffer(capacity=8)

    for i in range(8):
        buffer.add(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            action_mask=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=0.1,
            value=0.0,
            log_prob=-0.1,
            done=False,
        )

    buffer.compute_gae(last_value=0.0, last_done=False)
    metrics = agent.update(buffer)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert metrics["updates"] == 1


def test_ppo_checkpointing(tmp_path):
    """Verify saving and loading of PPO checkpoints."""
    agent = PPOAgent()
    ckpt_path = tmp_path / "ppo_test.pt"

    agent.total_steps = 1234
    agent.update_count = 12
    agent.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    agent2 = PPOAgent()
    agent2.load_checkpoint(ckpt_path)
    assert agent2.total_steps == 1234
    assert agent2.update_count == 12

    # Verify identical weights
    for p1, p2 in zip(agent.network.parameters(), agent2.network.parameters()):
        assert torch.equal(p1, p2)


def test_ppo_trainer_short_run():
    """Verify PPOTrainer runs without exceptions on a short training session."""
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
