"""Unit tests for Model 3: Discrete Soft Actor-Critic (SAC)."""

import numpy as np
import pytest
import torch

from phd_ludo_drl.config.config_manager import DiscreteSACConfig
from phd_ludo_drl.agents.model3.sac_network import DiscreteActor, DiscreteCritic
from phd_ludo_drl.agents.model3.replay_buffer import DiscreteSACReplayBuffer
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model3.trainer import DiscreteSACTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


def test_discrete_critic_shapes():
    """Verify DiscreteCritic twin Q-value outputs."""
    critic = DiscreteCritic(latent_dim=128, hidden_units=128, action_dim=4)
    spatial = torch.zeros(2, 4, 52)
    context = torch.zeros(2, 12)

    q1, q2 = critic(spatial, context)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)

    # Q1 single evaluation
    q1_only = critic.q1(spatial, context)
    assert q1_only.shape == (2, 4)


def test_discrete_actor_masking():
    """Verify that illegal actions have strictly zero probability and are never selected."""
    actor = DiscreteActor(latent_dim=128, hidden_units=128, action_dim=4)
    spatial = torch.zeros(1, 4, 52)
    context = torch.zeros(1, 12)
    # Only action index 1 is legal
    mask = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    probs, log_probs = actor(spatial, context, mask)
    assert np.isclose(probs[0, 1].item(), 1.0, atol=1e-5)
    assert probs[0, 0].item() == 0.0
    assert probs[0, 2].item() == 0.0
    assert probs[0, 3].item() == 0.0

    # Ensure select_action chooses action 1 every time
    for _ in range(30):
        action = actor.select_action(spatial, context, mask, deterministic=False)
        assert action == 1


def test_discrete_replay_buffer():
    """Verify DiscreteSACReplayBuffer push and sample mechanics."""
    buf = DiscreteSACReplayBuffer(capacity=10)
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


def test_discrete_sac_train_step():
    """Verify DiscreteSACAgent optimization step (critic, actor, and alpha updates)."""
    cfg = DiscreteSACConfig(batch_size=4, warmup_steps=4)
    agent = DiscreteSACAgent(config=cfg)

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
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert metrics["updates"] == 1


def test_discrete_sac_checkpointing(tmp_path):
    """Verify serialization and restoration of DiscreteSACAgent."""
    agent = DiscreteSACAgent()
    ckpt_path = tmp_path / "sac_test.pt"

    agent.total_steps = 987
    agent.update_count = 45
    agent.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    agent2 = DiscreteSACAgent()
    agent2.load_checkpoint(ckpt_path)
    assert agent2.total_steps == 987
    assert agent2.update_count == 45

    # Check identical parameters
    for p1, p2 in zip(agent.critic.parameters(), agent2.critic.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
        assert torch.equal(p1, p2)


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
