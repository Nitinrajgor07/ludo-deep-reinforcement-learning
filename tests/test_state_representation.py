"""Comprehensive Tests for State Representation & CNN Encoding (Phase 3).

Verifies:
- Raw Ludo observation extraction
- Observation validation (types, bounds, one-hot sums)
- Flat baseline representation & encoder (60-D -> 128-D)
- Structured 4-channel spatial representation (4, 52) + (12,) context
- Structured CNN encoder (Conv1d 32/64 filters -> 128-D)
- Batch processing (B=1, B=8, B=16)
- Output dimension contract (strictly 128-D)
- CPU device compatibility
- Full StateRepresentationPipeline integration
"""

import numpy as np
import pytest
import torch

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.constants import TokenState
from phd_ludo_drl.encoding.raw_observation import (
    RAW_STATE_DIMENSION,
    RawLudoObservation,
    extract_raw_observation,
)
from phd_ludo_drl.encoding.validation import (
    StateValidationError,
    StateDimensionError,
    validate_raw_observation,
    validate_flat_vector,
    validate_structured_tensor,
)
from phd_ludo_drl.encoding.flat_encoder import FlatEncoder
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.encoder_pipeline import (
    STANDARD_FEATURE_DIMENSION,
    StateRepresentationPipeline,
)


@pytest.fixture
def env():
    e = LudoEnvironment(seed=42, auto_advance_no_moves=False)
    e.reset()
    return e


# =====================================================================
# 1. RAW OBSERVATION EXTRACTION & DIMENSIONS
# =====================================================================

def test_raw_observation_extraction(env):
    """Verify raw observation dimensions and flat vector length."""
    state_dict = env.get_state_dict()
    raw_obs = extract_raw_observation(state_dict)

    assert isinstance(raw_obs, RawLudoObservation)
    assert raw_obs.current_player_vec.shape == (2,)
    assert raw_obs.dice_vec.shape == (6,)
    assert raw_obs.tokens_vec.shape == (48,)
    assert raw_obs.winner_vec.shape == (3,)
    assert raw_obs.terminal_vec.shape == (1,)
    assert raw_obs.action_mask.shape == (4,)

    flat_vec = raw_obs.to_flat_vector()
    assert flat_vec.shape == (RAW_STATE_DIMENSION,)
    assert flat_vec.shape == (60,)
    assert flat_vec.dtype == np.float32


# =====================================================================
# 2. OBSERVATION VALIDATION
# =====================================================================

def test_observation_validation_valid(env):
    """Verify clean state passes validation."""
    state_dict = env.get_state_dict()
    raw_obs = extract_raw_observation(state_dict)
    assert validate_raw_observation(raw_obs) is True
    assert validate_flat_vector(raw_obs.to_flat_vector()) is True


def test_observation_validation_corrupted_player(env):
    """Verify corrupted current player one-hot raises StateValidationError."""
    state_dict = env.get_state_dict()
    raw_obs = extract_raw_observation(state_dict)
    raw_obs.current_player_vec = np.array([0.5, 0.0], dtype=np.float32)  # Invalid sum

    with pytest.raises(StateValidationError):
        validate_raw_observation(raw_obs)


def test_observation_validation_dimension_mismatch():
    """Verify invalid flat vector dimension raises StateDimensionError."""
    corrupted_vec = np.zeros(59, dtype=np.float32)
    with pytest.raises(StateDimensionError):
        validate_flat_vector(corrupted_vec)


# =====================================================================
# 3. FLAT ENCODER (ABLATION BASELINE)
# =====================================================================

def test_flat_encoder_forward():
    """Verify FlatEncoder processes 60-D vector to 128-D latent."""
    encoder = FlatEncoder(input_dim=60, hidden_dim=256, latent_dim=128)

    # Single vector
    single_x = torch.randn(60)
    out_single = encoder(single_x)
    assert out_single.shape == (128,)

    # Batch of vectors (B=8)
    batch_x = torch.randn(8, 60)
    out_batch = encoder(batch_x)
    assert out_batch.shape == (8, 128)


# =====================================================================
# 4. STRUCTURED REPRESENTATION (4 CHANNELS)
# =====================================================================

def test_structured_representation_shapes(env):
    """Verify structured spatial board tensor has shape (4, 52) and context (12,)."""
    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    assert isinstance(struct_state, StructuredLudoState)
    assert struct_state.spatial_tensor.shape == (4, 52)
    assert struct_state.context_vector.shape == (12,)
    assert struct_state.action_mask.shape == (4,)
    assert validate_structured_tensor(struct_state.spatial_tensor) is True


def test_structured_representation_safe_squares_and_blockades(env):
    """Verify Channel 2 reflects safe squares and blockade formations."""
    # Place friendly blockade at square 10
    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t1 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 1)
    for t in [t0, t1]:
        t.state = TokenState.TRACK
        t.step_count = 10

    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    # Square 10 should have friendly blockade value (+1.0)
    assert struct_state.spatial_tensor[2, 10] == 1.0
    # Safe square 8 should have safe indicator (+0.5)
    assert struct_state.spatial_tensor[2, 8] == 0.5


# =====================================================================
# 5. STRUCTURED CNN ENCODER
# =====================================================================

def test_structured_cnn_encoder_forward():
    """Verify StructuredCNNEncoder produces exactly 128-D output."""
    encoder = StructuredCNNEncoder(in_channels=4, context_dim=12, latent_dim=128)

    # Single state
    spatial = torch.randn(4, 52)
    context = torch.randn(12)
    out_single = encoder(spatial, context)
    assert out_single.shape == (128,)

    # Batched forward (B=16)
    batch_spatial = torch.randn(16, 4, 52)
    batch_context = torch.randn(16, 12)
    out_batch = encoder(batch_spatial, batch_context)
    assert out_batch.shape == (16, 128)


def test_cnn_encoder_without_explicit_context():
    """Verify CNN encoder works when context is omitted (defaults to zeros)."""
    encoder = StructuredCNNEncoder(in_channels=4, context_dim=12, latent_dim=128)
    spatial = torch.randn(4, 52)
    out = encoder(spatial)
    assert out.shape == (128,)


# =====================================================================
# 6. END-TO-END PIPELINE & OUTPUT DIMENSION VALIDATION
# =====================================================================

def test_state_representation_pipeline_single_and_batch(env):
    """Verify complete pipeline produces valid 128-D EncodedState."""
    pipeline = StateRepresentationPipeline(encoder_type="structured_cnn")

    # 1. Single state encoding
    state_dict = env.get_state_dict()
    encoded = pipeline.encode_state(state_dict)

    assert encoded.feature_vector.shape == (STANDARD_FEATURE_DIMENSION,)
    assert encoded.feature_vector.shape == (128,)
    assert encoded.action_mask.shape == (4,)

    # 2. Batched encoding (B=5)
    batch_dicts = [env.get_state_dict() for _ in range(5)]
    batch_encoded = pipeline.encode_batch(batch_dicts)

    assert batch_encoded.feature_vector.shape == (5, 128)
    assert batch_encoded.action_mask.shape == (5, 4)


def test_pipeline_flat_mode(env):
    """Verify pipeline functions correctly in flat ablation mode."""
    pipeline = StateRepresentationPipeline(encoder_type="flat")
    state_dict = env.get_state_dict()
    encoded = pipeline.encode_state(state_dict)

    assert encoded.feature_vector.shape == (128,)
    assert encoded.action_mask.shape == (4,)


def test_device_compatibility(env):
    """Verify pipeline works seamlessly on designated torch device (CPU)."""
    device = torch.device("cpu")
    pipeline = StateRepresentationPipeline(encoder_type="structured_cnn", device=device)
    state_dict = env.get_state_dict()
    encoded = pipeline.encode_state(state_dict)

    assert encoded.feature_vector.device.type == "cpu"
    assert encoded.action_mask.device.type == "cpu"
