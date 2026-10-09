"""Comprehensive Unit and Integration Tests for Double Dueling DQN + PER (Phase 5).

Verifies strict compliance with FINAL_EXPERIMENT_PROTOCOL.md Section 8:
- Double DQN algorithm formulation (decoupled action selection & evaluation)
- Dueling Q-network architecture (Value and Advantage streams)
- Prioritized Experience Replay (proportional priorities & importance sampling)
- Target network and Polyak soft updates (tau=0.005) & hard updates
- Epsilon-greedy exploration & linear epsilon decay
- Legal action masking (-10^9 suppression and strictly legal exploration)
- Replay sampling and priority updates from TD errors
- Checkpoint saving and loading (restoration of weights, optimizer, step counters, epsilon, beta)
- Deterministic seed behavior and reproducibility
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import torch

from phd_ludo_drl.config.config_manager import DQNConfig
from phd_ludo_drl.environment.constants import TokenState, ACTION_SPACE_SIZE
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.agents.dqn.dqn_network import DuelingQNetwork
from phd_ludo_drl.agents.dqn.replay_buffer import PrioritizedReplayBuffer
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.dqn.trainer import DQNTrainer, DQNTrainingMetrics
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.environment.action_mask import mask_q_values


@pytest.fixture
def env():
    e = LudoEnvironment(seed=42, auto_advance_no_moves=False)
    e.reset()
    return e


# =====================================================================
# 1. FORWARD PASS TESTS
# =====================================================================

def test_dueling_network_single_forward():
    """Verify DuelingQNetwork forward pass on single spatial and context tensors."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4)
    spatial = torch.randn(4, 52)
    context = torch.randn(12)

    q_vals = net(spatial, context)
    assert q_vals.shape == (1, 4)
    assert q_vals.dtype == torch.float32


def test_dueling_network_batch_forward():
    """Verify DuelingQNetwork forward pass on batched spatial tensors."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4)
    for b in [1, 4, 16, 64]:
        spatial = torch.randn(b, 4, 52)
        context = torch.randn(b, 12)
        q_vals = net(spatial, context)
        assert q_vals.shape == (b, 4)


def test_dueling_network_without_context():
    """Verify DuelingQNetwork operates gracefully when context is omitted."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4)
    spatial = torch.randn(2, 4, 52)
    q_vals = net(spatial)
    assert q_vals.shape == (2, 4)


def test_dueling_aggregation_math():
    """Verify Q(s, a) strictly equals V(s) + (A(s, a) - mean_a'(A(s, a')))."""
    net = DuelingQNetwork(latent_dim=128, hidden_units=256, action_dim=4, use_internal_encoder=False)
    features = torch.randn(8, 128)

    q_vals = net(features)
    val = net.value_stream(features)           # (8, 1)
    adv = net.advantage_stream(features)       # (8, 4)

    expected_q = val + (adv - adv.mean(dim=-1, keepdim=True))
    assert torch.allclose(q_vals, expected_q, atol=1e-5)
    # The mean of (A - mean(A)) across actions must be identically 0
    assert torch.allclose((adv - adv.mean(dim=-1, keepdim=True)).mean(dim=-1), torch.zeros(8), atol=1e-6)


def test_agent_select_action_with_structured_state():
    """Verify DQNAgent.select_action consumes StructuredLudoState directly."""
    env = LudoEnvironment(seed=42, auto_advance_no_moves=True)
    obs, info = env.reset()
    agent = DQNAgent()
    struct_state = build_structured_representation(obs)

    action = agent.select_action(struct_state, info["action_mask"], deterministic=True)
    assert isinstance(action, int)
    assert 0 <= action < ACTION_SPACE_SIZE
    assert info["action_mask"][action] == 1.0


# =====================================================================
# 2. ACTION MASKING TESTS
# =====================================================================

def test_action_masking_suppression_value():
    """Verify illegal actions are suppressed with -1e9 in Q-values."""
    q_vals = torch.tensor([1.5, 2.5, 3.5, 4.5])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    masked_q = mask_q_values(q_vals, mask)
    assert masked_q[0] == pytest.approx(1.5)
    assert masked_q[1] <= -1e8
    assert masked_q[2] == pytest.approx(3.5)
    assert masked_q[3] <= -1e8
    assert torch.argmax(masked_q).item() == 2  # Action 2 (val 3.5) beats action 3 (-1e9)


@pytest.mark.parametrize("legal_action", [0, 1, 2, 3])
def test_greedy_action_selection_strictly_respects_single_legal_action(legal_action):
    """Verify greedy selection always picks the legal action, even if other actions have higher unmasked Q."""
    agent = DQNAgent()
    mask = np.zeros(4, dtype=np.float32)
    mask[legal_action] = 1.0

    dummy_spatial = np.random.randn(4, 52).astype(np.float32)
    dummy_context = np.random.randn(12).astype(np.float32)

    for _ in range(10):
        act = agent.select_action(
            dummy_spatial,
            mask,
            context=dummy_context,
            deterministic=True,
        )
        assert act == legal_action, f"Expected action {legal_action}, got {act}"


def test_exploration_strictly_samples_from_legal_actions():
    """Verify exploration mode (epsilon=1.0) never selects an illegal action."""
    agent = DQNAgent()
    agent.epsilon = 1.0
    mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)  # Only actions 1 and 3 are legal

    dummy_spatial = np.zeros((4, 52), dtype=np.float32)

    selected_actions = set()
    for _ in range(50):
        act = agent.select_action(dummy_spatial, mask, deterministic=False)
        assert act in (1, 3), f"Exploration selected illegal action {act}"
        selected_actions.add(act)

    # In 50 trials, both legal actions should be explored
    assert selected_actions == {1, 3}


def test_empty_legal_actions_raises_value_error():
    """Verify selecting action with empty mask raises ValueError."""
    agent = DQNAgent()
    empty_mask = np.zeros(4, dtype=np.float32)
    dummy_spatial = np.zeros((4, 52), dtype=np.float32)

    with pytest.raises(ValueError, match="legal_actions is empty"):
        agent.select_action(dummy_spatial, empty_mask)


# =====================================================================
# 3. REPLAY BUFFER & PRIORITIZED SAMPLING TESTS
# =====================================================================

def test_replay_buffer_fifo_overwrite():
    """Verify circular buffer correctly overwrites oldest samples at capacity."""
    capacity = 5
    buffer = PrioritizedReplayBuffer(capacity=capacity)

    # Push 8 items (capacity is 5)
    for i in range(8):
        buffer.push(
            state=np.full((4, 52), i, dtype=np.float32),
            context=np.full(12, i, dtype=np.float32),
            action=i % 4,
            reward=float(i),
            next_state=np.full((4, 52), i + 1, dtype=np.float32),
            next_context=np.full(12, i + 1, dtype=np.float32),
            done=False,
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )

    assert len(buffer) == 5
    # The buffer should now contain states 3, 4, 5, 6, 7
    stored_rewards = sorted([buffer.rewards[i] for i in range(5)])
    assert stored_rewards == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_replay_buffer_sample_batch_contracts():
    """Verify sampled batch contains all required tensors with correct shapes and types."""
    buffer = PrioritizedReplayBuffer(capacity=50, alpha=0.6, beta_start=0.4)

    for i in range(20):
        buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=i % 4,
            reward=float(i) * 0.1,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=(i % 5 == 0),
            action_mask=np.array([1, 1, 0, 0], dtype=np.float32),
            next_action_mask=np.array([0, 1, 1, 0], dtype=np.float32),
        )

    batch, indices, weights = buffer.sample(batch_size=8)

    assert batch["states"].shape == (8, 4, 52)
    assert batch["contexts"].shape == (8, 12)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["next_states"].shape == (8, 4, 52)
    assert batch["next_contexts"].shape == (8, 12)
    assert batch["dones"].shape == (8,)
    assert batch["action_masks"].shape == (8, 4)
    assert batch["next_action_masks"].shape == (8, 4)

    assert len(indices) == 8
    assert weights.shape == (8,)
    assert weights.max().item() <= 1.0 + 1e-5
    assert torch.all(weights >= 0.0)


def test_beta_annealing():
    """Verify importance-sampling exponent beta linearly anneals from beta_start to beta_end."""
    buffer = PrioritizedReplayBuffer(capacity=100, beta_start=0.4, beta_end=1.0)
    total_steps = 1000

    b0 = buffer.anneal_beta(0, total_steps)
    assert b0 == pytest.approx(0.4)

    b500 = buffer.anneal_beta(500, total_steps)
    assert b500 == pytest.approx(0.7)

    b1000 = buffer.anneal_beta(1000, total_steps)
    assert b1000 == pytest.approx(1.0)

    # Beyond total steps stays clamped at 1.0
    b1500 = buffer.anneal_beta(1500, total_steps)
    assert b1500 == pytest.approx(1.0)


# =====================================================================
# 4. PRIORITY UPDATES TESTS
# =====================================================================

def test_priority_updates_from_td_errors():
    """Verify TD errors update transition priorities according to (|delta| + eps)^alpha."""
    buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6, epsilon=1e-5)

    for i in range(4):
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

    # Set distinct TD errors: 10.0, 2.0, 0.5, 0.01
    td_errors = np.array([10.0, 2.0, 0.5, 0.01])
    indices = np.array([0, 1, 2, 3])
    buffer.update_priorities(indices, td_errors)

    assert buffer.priorities[0] > buffer.priorities[1] > buffer.priorities[2] > buffer.priorities[3]
    assert buffer.max_priority == pytest.approx(10.0, rel=1e-4)


def test_high_priority_sampled_more_frequently():
    """Verify transition with massive TD error is sampled with much higher frequency."""
    buffer = PrioritizedReplayBuffer(capacity=10, alpha=1.0)

    for _ in range(5):
        buffer.push(
            state=np.zeros((4, 52), dtype=np.float32),
            context=np.zeros(12, dtype=np.float32),
            action=0,
            reward=0.0,
            next_state=np.zeros((4, 52), dtype=np.float32),
            next_context=np.zeros(12, dtype=np.float32),
            done=False,
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )

    # Give index 0 an enormous priority, others tiny
    buffer.update_priorities(np.array([0]), np.array([1000.0]))
    buffer.update_priorities(np.array([1, 2, 3, 4]), np.array([0.001, 0.001, 0.001, 0.001]))

    # Sample repeatedly (size 1) and count index 0 appearances
    samples = [buffer.sample(1)[1][0] for _ in range(100)]
    assert samples.count(0) > 90, "High priority sample must dominate sampling distribution"


# =====================================================================
# 5. DOUBLE DQN TARGET CALCULATION TESTS
# =====================================================================

def test_double_dqn_target_calculation_math():
    """Verify target calculation: y = r + (1 - done) * gamma * Q_target(s', argmax Q_online(s'))."""
    agent = DQNAgent()
    agent.gamma = 0.99

    batch_size = 2
    rewards = torch.tensor([[1.0], [0.5]])
    dones = torch.tensor([[0.0], [1.0]])  # Item 1 is terminal!
    next_states = torch.zeros(batch_size, 4, 52)
    next_contexts = torch.zeros(batch_size, 12)
    next_masks = torch.ones(batch_size, 4)

    # Mock Q_online next predictions to prefer action 2
    def mock_online(s, c):
        return torch.tensor([[1.0, 2.0, 5.0, 3.0], [4.0, 3.0, 2.0, 1.0]])

    # Mock Q_target predictions
    def mock_target(s, c):
        return torch.tensor([[10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]])

    agent.online_net = mock_online
    agent.target_net = mock_target

    targets = agent.compute_targets(rewards, next_states, next_contexts, dones, next_masks)

    # Item 0:
    # argmax Q_online([1, 2, 5, 3]) is action 2
    # Q_target for action 2 is 30.0
    # y_0 = 1.0 + (1 - 0) * 0.99 * 30.0 = 1.0 + 29.7 = 30.7
    assert targets[0].item() == pytest.approx(1.0 + 0.99 * 30.0, rel=1e-5)

    # Item 1: done = 1.0 -> y_1 = r = 0.5 (future value zeroed out)
    assert targets[1].item() == pytest.approx(0.5, rel=1e-5)


def test_target_calculation_respects_next_action_mask():
    """Verify Double DQN target selection does NOT pick an illegal next action even if its Q is highest."""
    agent = DQNAgent()
    agent.gamma = 0.99

    rewards = torch.tensor([[0.0]])
    dones = torch.tensor([[0.0]])
    next_states = torch.zeros(1, 4, 52)
    next_contexts = torch.zeros(1, 12)
    # Only action 1 is legal in next state!
    next_masks = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    # Unmasked online prefers action 3 (Q=10.0), but action 3 is ILLEGAL!
    agent.online_net = lambda s, c: torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    agent.target_net = lambda s, c: torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    targets = agent.compute_targets(rewards, next_states, next_contexts, dones, next_masks)

    # Masked online forces selection of action 1 (Q=2.0)
    # Target value for action 1 is 20.0
    # Target = 0.0 + 0.99 * 20.0 = 19.8
    assert targets[0].item() == pytest.approx(0.99 * 20.0, rel=1e-5)


def test_soft_target_update_and_hard_update():
    """Verify target network soft update and hard update mechanisms."""
    agent = DQNAgent()

    # Alter online parameters
    with torch.no_grad():
        for p in agent.online_net.parameters():
            p.fill_(5.0)
        for p in agent.target_net.parameters():
            p.fill_(1.0)

    # 1. Soft update with tau = 0.1
    agent._soft_update(tau=0.1)
    # Expected param: 0.1 * 5.0 + 0.9 * 1.0 = 0.5 + 0.9 = 1.4
    for p in agent.target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.4), atol=1e-5)

    # 2. Hard update
    agent.hard_update()
    for p in agent.target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 5.0), atol=1e-5)


def test_epsilon_linear_decay():
    """Verify exploration rate epsilon decays linearly over training steps."""
    agent = DQNAgent()
    agent.epsilon_start = 1.0
    agent.epsilon_end = 0.05
    agent.epsilon_decay_steps = 1000

    agent.total_steps = 0
    agent._decay_epsilon()
    assert agent.epsilon == pytest.approx(1.0)

    agent.total_steps = 500
    agent._decay_epsilon()
    assert agent.epsilon == pytest.approx(0.525)

    agent.total_steps = 1000
    agent._decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)

    # Stays at 0.05 beyond decay steps
    agent.total_steps = 2000
    agent._decay_epsilon()
    assert agent.epsilon == pytest.approx(0.05)


# =====================================================================
# 6. CHECKPOINT SAVE & LOAD TESTS
# =====================================================================

def test_checkpoint_save_and_load(tmp_path: Path):
    """Verify complete checkpoint serialization and exact restoration of state and Q-values."""
    agent1 = DQNAgent()
    agent1.total_steps = 42000
    agent1.epsilon = 0.35
    agent1.buffer.beta = 0.88

    # Run a train step with dummy data to modify weights and optimizer
    for i in range(10):
        agent1.buffer.push(
            state=np.random.randn(4, 52).astype(np.float32),
            context=np.random.randn(12).astype(np.float32),
            action=i % 4,
            reward=0.1,
            next_state=np.random.randn(4, 52).astype(np.float32),
            next_context=np.random.randn(12).astype(np.float32),
            done=False,
            action_mask=np.ones(4, dtype=np.float32),
            next_action_mask=np.ones(4, dtype=np.float32),
        )
    agent1.train_step(batch_size=8)

    expected_steps = agent1.total_steps
    expected_epsilon = agent1.epsilon
    expected_beta = agent1.buffer.beta

    ckpt_file = tmp_path / "dqn_checkpoint.pt"
    agent1.save_checkpoint(ckpt_file)
    assert ckpt_file.exists()

    # Create new fresh agent
    agent2 = DQNAgent()
    agent2.load_checkpoint(ckpt_file)

    assert agent2.total_steps == expected_steps
    assert agent2.epsilon == pytest.approx(expected_epsilon)
    assert agent2.buffer.beta == pytest.approx(expected_beta)

    # Verify online net parameters match exactly
    for p1, p2 in zip(agent1.online_net.parameters(), agent2.online_net.parameters()):
        assert torch.equal(p1, p2)

    # Verify target net parameters match exactly
    for p1, p2 in zip(agent1.target_net.parameters(), agent2.target_net.parameters()):
        assert torch.equal(p1, p2)

    # Verify forward pass outputs are identical
    test_spatial = torch.randn(4, 4, 52)
    test_context = torch.randn(4, 12)
    q1 = agent1.online_net(test_spatial, test_context)
    q2 = agent2.online_net(test_spatial, test_context)
    assert torch.equal(q1, q2)


# =====================================================================
# 7. DETERMINISTIC SEED BEHAVIOR TESTS
# =====================================================================

def test_deterministic_seed_initialization():
    """Verify identical random seeds produce identical initial weights."""
    torch.manual_seed(42)
    np.random.seed(42)
    agent_a = DQNAgent()

    torch.manual_seed(42)
    np.random.seed(42)
    agent_b = DQNAgent()

    for p_a, p_b in zip(agent_a.online_net.parameters(), agent_b.online_net.parameters()):
        assert torch.equal(p_a, p_b), "Initial network parameters must be identical under same seed"


def test_deterministic_action_selection_under_seed():
    """Verify stochastic action selection is reproducible when seeded."""
    mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    spatial = np.random.randn(4, 52).astype(np.float32)
    context = np.random.randn(12).astype(np.float32)

    agent_a = DQNAgent()
    agent_b = DQNAgent()

    # Copy weights so online networks are identical
    agent_b.online_net.load_state_dict(agent_a.online_net.state_dict())

    agent_a.seed(999)
    agent_a.epsilon = 0.5
    actions_a = [agent_a.select_action(spatial, mask, context=context, deterministic=False) for _ in range(20)]

    agent_b.seed(999)
    agent_b.epsilon = 0.5
    actions_b = [agent_b.select_action(spatial, mask, context=context, deterministic=False) for _ in range(20)]

    assert actions_a == actions_b, "Action sequences under identical seeds must match bitwise"
