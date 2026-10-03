"""Unit tests for Model 4: Categorical DQN (C51 / Distributional RL)."""

import numpy as np
import pytest
import torch

from phd_ludo_drl.config.config_manager import CategoricalDQNConfig
from phd_ludo_drl.agents.model4.c51_network import CategoricalQNetwork
from phd_ludo_drl.agents.model4.replay_buffer import C51ReplayBuffer
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.agents.model4.trainer import C51Trainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


def test_categorical_q_network_shapes():
    """Verify CategoricalQNetwork output dimensions and probability normalization."""
    net = CategoricalQNetwork(
        latent_dim=128,
        hidden_units=128,
        action_dim=4,
        num_atoms=51,
        v_min=-20.0,
        v_max=30.0,
    )
    spatial = torch.zeros(2, 4, 52)
    context = torch.zeros(2, 12)

    probs, q_values = net(spatial, context)
    assert probs.shape == (2, 4, 51)
    assert q_values.shape == (2, 4)

    # Probabilities across atoms for each action must sum to 1.0
    prob_sums = torch.sum(probs, dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_c51_action_selection_masking():
    """Verify that action masking correctly influences Q-argmax selection."""
    net = CategoricalQNetwork(
        latent_dim=128,
        hidden_units=128,
        action_dim=4,
        num_atoms=51,
        v_min=-20.0,
        v_max=30.0,
    )
    spatial = torch.zeros(1, 4, 52)
    context = torch.zeros(1, 12)
    # Only action index 2 is legal
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    action = net.select_action(spatial, context, mask, deterministic=True)
    assert action == 2


def test_c51_replay_buffer():
    """Verify C51ReplayBuffer push and sample mechanics."""
    buf = C51ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(
            state=np.zeros((4, 52)),
            context=np.zeros(12),
            action=i % 4,
            reward=0.25,
            next_state=np.zeros((4, 52)),
            next_context=np.zeros(12),
            done=False,
            action_mask=np.array([1.0, 0.0, 0.0, 0.0]),
            next_action_mask=np.array([0.0, 1.0, 0.0, 0.0]),
        )

    assert len(buf) == 5
    batch = buf.sample(batch_size=3, device=torch.device("cpu"))
    assert batch["states"].shape == (3, 4, 52)
    assert batch["actions"].shape == (3,)
    assert batch["rewards"].shape == (3,)
    assert batch["action_masks"].shape == (3, 4)


def test_c51_agent_train_step():
    """Verify C51Agent Bellman projection and cross-entropy loss computation."""
    cfg = CategoricalDQNConfig(batch_size=4, warmup_steps=4)
    agent = CategoricalDQNAgent(config=cfg)

    for i in range(6):
        agent.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            reward=0.1,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.zeros(12, dtype=np.float32),
            done=False,
            action_mask=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            next_action_mask=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    metrics = agent.train_step(batch_size=4)
    assert metrics is not None
    assert "loss" in metrics
    assert "mean_q" in metrics
    assert metrics["updates"] == 1


def test_c51_checkpointing(tmp_path):
    """Verify serialization and restoration of CategoricalDQNAgent."""
    agent = CategoricalDQNAgent()
    ckpt_path = tmp_path / "c51_test.pt"

    agent.total_steps = 777
    agent.update_count = 33
    agent.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    agent2 = CategoricalDQNAgent()
    agent2.load_checkpoint(ckpt_path)
    assert agent2.total_steps == 777
    assert agent2.update_count == 33

    for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)


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
