"""Comprehensive Unit Tests for Shared Structured CNN Encoder (Phase 4).

Verifies strict compliance with FINAL_EXPERIMENT_PROTOCOL.md Section 4:
- Input spatial tensor: 4 x 52
- CNN:
  - Conv1D: 4 -> 32 (kernel=3, padding=1, padding_mode='circular')
  - Conv1D: 32 -> 64 (kernel=3, padding=1, padding_mode='circular')
  - AdaptiveAvgPool1D: output length = 1
  - Flatten: 64 features
  - Concatenate with 12-dimensional context: 64 + 12 = 76
  - Fully connected: 76 -> 128
- Unit tests for:
  - batch input
  - single input
  - invalid shape
  - output shape
  - deterministic forward pass
  - gradient flow
- Programmatic dimension verification
- Integration across all 4 algorithm architectures
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from phd_ludo_drl.encoding.cnn_encoder import (
    StructuredCNNEncoder,
    SPATIAL_IN_CHANNELS,
    SPATIAL_LENGTH,
    CONV1_OUT_CHANNELS,
    CONV2_OUT_CHANNELS,
    POOL_OUTPUT_SIZE,
    SPATIAL_FEATURE_DIM,
    CONTEXT_INPUT_DIM,
    FUSED_INPUT_DIM,
    LATENT_OUTPUT_DIM,
)
from phd_ludo_drl.agents.dqn.dqn_network import DuelingQNetwork
from phd_ludo_drl.agents.ppo.ppo_network import ActorCriticNetwork
from phd_ludo_drl.agents.model3.sac_network import DiscreteCritic, DiscreteActor
from phd_ludo_drl.agents.model4.c51_network import CategoricalQNetwork


# =====================================================================
# 1. PROGRAMMATIC DIMENSION VERIFICATION
# =====================================================================

def test_programmatic_dimension_verification():
    """Verify all frozen layer dimensions programmatically."""
    spec = StructuredCNNEncoder.verify_dimensions()

    assert spec["spatial_input_shape"] == (4, 52)
    assert spec["conv1_in_channels"] == 4
    assert spec["conv1_out_channels"] == 32
    assert spec["conv1_kernel_size"] == 3
    assert spec["conv1_padding_mode"] == "circular"
    assert spec["conv2_in_channels"] == 32
    assert spec["conv2_out_channels"] == 64
    assert spec["conv2_kernel_size"] == 3
    assert spec["conv2_padding_mode"] == "circular"
    assert spec["pool_output_size"] == 1
    assert spec["spatial_flatten_features"] == 64
    assert spec["context_dimension"] == 12
    assert spec["total_fused_features"] == 76
    assert spec["fc_in_features"] == 76
    assert spec["fc_out_features"] == 128
    assert spec["latent_output_dimension"] == 128


# =====================================================================
# 2. SINGLE INPUT TESTS
# =====================================================================

def test_single_input_with_context():
    """Verify single unbatched input: (4, 52) spatial + (12,) context -> (128,)."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(4, 52)
    context = torch.randn(12)

    out = encoder(spatial, context)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (128,)
    assert out.dtype == torch.float32


def test_single_input_numpy_array():
    """Verify single numpy array input produces torch.Tensor of shape (128,)."""
    encoder = StructuredCNNEncoder()
    spatial_np = np.random.randn(4, 52).astype(np.float32)
    context_np = np.random.randn(12).astype(np.float32)

    out = encoder(spatial_np, context_np)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (128,)


def test_single_input_without_context():
    """Verify single input when context is omitted defaults to zero context."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(4, 52)

    out = encoder(spatial)
    assert out.shape == (128,)

    # Verify identical output when passing explicit zeros
    zeros_context = torch.zeros(12)
    out_zeros = encoder(spatial, zeros_context)
    assert torch.allclose(out, out_zeros, atol=1e-6)


# =====================================================================
# 3. BATCH INPUT TESTS
# =====================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_batch_input_various_sizes(batch_size):
    """Verify batched inputs for various standard RL batch sizes."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(batch_size, 4, 52)
    context = torch.randn(batch_size, 12)

    out = encoder(spatial, context)
    assert out.shape == (batch_size, 128)
    assert out.dtype == torch.float32


def test_batch_input_without_context():
    """Verify batched forward pass when context is omitted."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(16, 4, 52)

    out = encoder(spatial)
    assert out.shape == (16, 128)


# =====================================================================
# 4. INVALID SHAPE REJECTION TESTS
# =====================================================================

def test_invalid_spatial_rank():
    """Verify error raised on invalid spatial tensor ranks (rank 1 or rank 4)."""
    encoder = StructuredCNNEncoder()

    # Rank 1
    with pytest.raises(ValueError, match="Invalid spatial tensor rank"):
        encoder(torch.randn(52))

    # Rank 4
    with pytest.raises(ValueError, match="Invalid spatial tensor rank"):
        encoder(torch.randn(2, 4, 4, 52))


def test_invalid_spatial_channels():
    """Verify error raised on wrong number of spatial channels."""
    encoder = StructuredCNNEncoder()

    # 3 channels instead of 4
    with pytest.raises(ValueError, match="Expected"):
        encoder(torch.randn(3, 52))

    # Batched 5 channels
    with pytest.raises(ValueError, match="Expected"):
        encoder(torch.randn(8, 5, 52))


def test_invalid_spatial_length():
    """Verify error raised on wrong track length."""
    encoder = StructuredCNNEncoder()

    # 50 tiles instead of 52
    with pytest.raises(ValueError, match="Expected"):
        encoder(torch.randn(4, 50))

    # Batched 54 tiles
    with pytest.raises(ValueError, match="Expected"):
        encoder(torch.randn(8, 4, 54))


def test_invalid_context_dimension():
    """Verify error raised on wrong context dimension."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(4, 52)

    # 11 features instead of 12
    with pytest.raises(ValueError, match="context"):
        encoder(spatial, torch.randn(11))

    # 13 features instead of 12
    with pytest.raises(ValueError, match="context"):
        encoder(spatial, torch.randn(13))


def test_batch_size_mismatch():
    """Verify error raised when spatial and context batch sizes do not match."""
    encoder = StructuredCNNEncoder()
    spatial = torch.randn(8, 4, 52)
    context = torch.randn(4, 12)  # mismatched batch size: 4 != 8

    with pytest.raises(ValueError, match="Batch size mismatch"):
        encoder(spatial, context)


def test_invalid_input_type():
    """Verify error raised when non-tensor and non-numpy input is passed."""
    encoder = StructuredCNNEncoder()
    with pytest.raises(TypeError):
        encoder([[1.0, 2.0]])


# =====================================================================
# 5. OUTPUT SHAPE CONTRACT TESTS
# =====================================================================

def test_output_shape_and_non_negativity_from_relu():
    """Verify output shape and non-negativity guarantee from final ReLU activation."""
    encoder = StructuredCNNEncoder()
    encoder.eval()

    spatial = torch.randn(10, 4, 52)
    context = torch.randn(10, 12)

    out = encoder(spatial, context)
    assert out.shape == (10, 128)
    assert torch.all(out >= 0.0), "Output of final ReLU projection must be non-negative"


# =====================================================================
# 6. DETERMINISTIC FORWARD PASS
# =====================================================================

def test_deterministic_forward_pass():
    """Verify identical inputs yield identical outputs in eval mode."""
    encoder = StructuredCNNEncoder()
    encoder.eval()

    spatial = torch.randn(8, 4, 52)
    context = torch.randn(8, 12)

    out1 = encoder(spatial, context)
    out2 = encoder(spatial, context)

    assert torch.equal(out1, out2), "Forward pass must be strictly deterministic"


# =====================================================================
# 7. GRADIENT FLOW TESTS
# =====================================================================

def test_gradient_flow_to_all_layers():
    """Verify gradients propagate to all encoder layers during backpropagation."""
    encoder = StructuredCNNEncoder()
    encoder.train()

    spatial = torch.randn(4, 4, 52, requires_grad=True)
    context = torch.randn(4, 12, requires_grad=True)

    out = encoder(spatial, context)
    loss = out.sum()
    loss.backward()

    # 1. Gradients in Conv1D Layer 1
    assert encoder.conv1.weight.grad is not None
    assert encoder.conv1.weight.grad.abs().sum() > 0
    assert encoder.conv1.bias.grad is not None

    # 2. Gradients in Conv1D Layer 2
    assert encoder.conv2.weight.grad is not None
    assert encoder.conv2.weight.grad.abs().sum() > 0
    assert encoder.conv2.bias.grad is not None

    # 3. Gradients in Fully Connected Projection
    assert encoder.fc.weight.grad is not None
    assert encoder.fc.weight.grad.abs().sum() > 0
    assert encoder.fc.bias.grad is not None

    # 4. Gradients flow back to spatial input tensor
    assert spatial.grad is not None
    assert spatial.grad.abs().sum() > 0

    # 5. Gradients flow back to context input vector
    assert context.grad is not None
    assert context.grad.abs().sum() > 0


# =====================================================================
# 8. CIRCULAR PADDING WRAP-AROUND BEHAVIOR
# =====================================================================

def test_circular_padding_wrap_around():
    """Verify circular padding connects tile 0 and tile 51 seamlessly."""
    encoder = StructuredCNNEncoder()
    encoder.eval()

    # Base input with zeros
    spatial1 = torch.zeros(1, 4, 52)
    spatial2 = torch.zeros(1, 4, 52)

    # In spatial2, perturb tile 51 (the last tile)
    spatial2[0, 0, 51] = 1.0

    # Conv1 with circular padding kernel size 3 will affect tile 0 from tile 51
    conv_out1 = encoder.relu1(encoder.conv1(spatial1))
    conv_out2 = encoder.relu1(encoder.conv1(spatial2))

    # Tile 0 in conv_out2 must differ from conv_out1 due to circular neighbor at 51
    diff_at_tile_0 = torch.abs(conv_out2[0, :, 0] - conv_out1[0, :, 0]).sum()
    assert diff_at_tile_0 > 0, "Circular padding must propagate information from tile 51 to tile 0"


# =====================================================================
# 9. INTEGRATION ACROSS ALL 4 ALGORITHM NETWORKS
# =====================================================================

def test_all_four_models_use_identical_encoder_architecture():
    """Verify DQN, PPO, SAC, and C51 all embed the exact frozen encoder."""
    # 1. DQN
    dqn = DuelingQNetwork(latent_dim=128)
    assert isinstance(dqn.encoder, StructuredCNNEncoder)
    assert dqn.encoder.fc.in_features == 76
    assert dqn.encoder.fc.out_features == 128

    # 2. PPO
    ppo = ActorCriticNetwork(latent_dim=128)
    assert isinstance(ppo.encoder, StructuredCNNEncoder)
    assert ppo.encoder.fc.in_features == 76
    assert ppo.encoder.fc.out_features == 128

    # 3. Discrete SAC (Critic twin encoders + Actor encoder)
    sac_critic = DiscreteCritic(latent_dim=128)
    assert isinstance(sac_critic.encoder1, StructuredCNNEncoder)
    assert isinstance(sac_critic.encoder2, StructuredCNNEncoder)
    assert sac_critic.encoder1.fc.in_features == 76
    assert sac_critic.encoder2.fc.in_features == 76

    sac_actor = DiscreteActor(latent_dim=128)
    assert isinstance(sac_actor.encoder, StructuredCNNEncoder)
    assert sac_actor.encoder.fc.in_features == 76

    # 4. C51 (Categorical DQN)
    c51 = CategoricalQNetwork(latent_dim=128)
    assert isinstance(c51.encoder, StructuredCNNEncoder)
    assert c51.encoder.fc.in_features == 76
    assert c51.encoder.fc.out_features == 128


def test_all_four_models_forward_pass_with_shared_encoder():
    """Verify forward passes of all 4 models work seamlessly with the frozen encoder."""
    spatial = torch.randn(2, 4, 52)
    context = torch.randn(2, 12)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

    # 1. DQN
    dqn = DuelingQNetwork(latent_dim=128)
    q_vals = dqn(spatial, context)
    assert q_vals.shape == (2, 4)

    # 2. PPO
    ppo = ActorCriticNetwork(latent_dim=128)
    actions, log_probs, entropy, values = ppo.get_action_and_value(spatial, context, mask)
    assert actions.shape == (2,)
    assert values.shape == (2,)

    # 3. SAC Critic & Actor
    sac_critic = DiscreteCritic(latent_dim=128)
    q1, q2 = sac_critic(spatial, context)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)

    sac_actor = DiscreteActor(latent_dim=128)
    probs, log_probs = sac_actor(spatial, context, mask)
    assert probs.shape == (2, 4)

    # 4. C51
    c51 = CategoricalQNetwork(latent_dim=128)
    probs, q_vals = c51(spatial, context)
    assert probs.shape == (2, 4, 51)
    assert q_vals.shape == (2, 4)
