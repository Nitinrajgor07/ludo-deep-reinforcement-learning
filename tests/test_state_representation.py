"""Comprehensive Tests for State Representation & CNN Encoding (Phase 3).

Verifies strict compliance with FINAL_EXPERIMENT_PROTOCOL.md Section 3:
- Spatial board tensor shape (4, 52) and channel semantics:
  Channel 0: Active-player token occupancy (0..51)
  Channel 1: Opponent token occupancy (0..51)
  Channel 2: Safe square indicators (1.0 at safe squares, 0.0 elsewhere)
  Channel 3: Blockade indicators (1.0 at tiles with >= 2 tokens of same color)
- Context vector exactly 12 dimensions in documented order:
  0: Active yard count / 4.0
  1: Opponent yard count / 4.0
  2: Active home column count / 4.0
  3: Opponent home column count / 4.0
  4: Active goal count / 4.0
  5: Opponent goal count / 4.0
  6: Dice roll / 6.0
  7: Active player P0 one-hot
  8: Active player P1 one-hot
  9: Turn fraction (turn / 1000.0)
  10: Bonus roll after 6 flag
  11: Capture bonus flag
- Automated tests that FAIL if representation deviates from the frozen protocol
- Exact CNN input/output dimensions
"""

from copy import deepcopy
import numpy as np
import pytest
import torch

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    SAFE_SQUARES,
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    ACTION_SPACE_SIZE,
    TokenState,
)
from phd_ludo_drl.encoding.raw_observation import (
    RAW_STATE_DIMENSION,
    RawLudoObservation,
    extract_raw_observation,
)
from phd_ludo_drl.encoding.validation import (
    StateValidationError,
    StateDimensionError,
    ProtocolDeviationError,
    validate_raw_observation,
    validate_flat_vector,
    validate_structured_tensor,
    validate_spatial_tensor,
    validate_context_vector,
    validate_frozen_state,
    verify_representation_against_protocol,
)
from phd_ludo_drl.encoding.flat_encoder import FlatEncoder
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
    SPATIAL_SHAPE,
    CONTEXT_DIMENSION,
    ACTION_MASK_DIMENSION,
    CHANNEL_ACTIVE_OCCUPANCY,
    CHANNEL_OPPONENT_OCCUPANCY,
    CHANNEL_SAFE_SQUARES,
    CHANNEL_BLOCKADES,
    CHANNEL_NAMES,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_IDX_ACTIVE_YARD,
    CONTEXT_IDX_OPPONENT_YARD,
    CONTEXT_IDX_ACTIVE_HOME,
    CONTEXT_IDX_OPPONENT_HOME,
    CONTEXT_IDX_ACTIVE_GOAL,
    CONTEXT_IDX_OPPONENT_GOAL,
    CONTEXT_IDX_DICE,
    CONTEXT_IDX_PLAYER_0,
    CONTEXT_IDX_PLAYER_1,
    CONTEXT_IDX_TURN_FRACTION,
    CONTEXT_IDX_BONUS_SIX,
    CONTEXT_IDX_BONUS_CAPTURE,
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

    single_x = torch.randn(60)
    out_single = encoder(single_x)
    assert out_single.shape == (128,)

    batch_x = torch.randn(8, 60)
    out_batch = encoder(batch_x)
    assert out_batch.shape == (8, 128)


# =====================================================================
# 4. SPATIAL TENSOR: SHAPE AND CHANNEL SEMANTICS
# =====================================================================

def test_spatial_tensor_shape_and_type(env):
    """Verify spatial tensor shape is exactly (4, 52) and float32."""
    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    assert isinstance(struct_state, StructuredLudoState)
    assert struct_state.spatial_tensor.shape == (4, 52)
    assert struct_state.spatial_tensor.dtype == np.float32
    assert validate_spatial_tensor(struct_state.spatial_tensor) is True


def test_spatial_channel_0_active_token_occupancy(env):
    """Verify Channel 0 captures active-player token occupancy only."""
    # Place active player (P0) tokens on track at square 5 and square 12
    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t1 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 1)
    t0.state = TokenState.TRACK
    t0.step_count = 5   # P0 start is 0 -> square 5
    t1.state = TokenState.TRACK
    t1.step_count = 12  # square 12

    # Place opponent token on track at square 26
    opp_t = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    opp_t.state = TokenState.TRACK
    opp_t.step_count = 0  # P1 start is 26 -> square 26

    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    # Active player tokens on Channel 0
    assert struct_state.spatial_tensor[0, 5] == 1.0
    assert struct_state.spatial_tensor[0, 12] == 1.0
    assert struct_state.spatial_tensor[0, 26] == 0.0  # opponent must NOT appear in Channel 0
    assert struct_state.spatial_tensor[0].sum() == 2.0


def test_spatial_channel_1_opponent_token_occupancy(env):
    """Verify Channel 1 captures opponent token occupancy only."""
    # Opponent (P1) tokens on track
    opp0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    opp1 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 1)
    opp0.state = TokenState.TRACK
    opp0.step_count = 0  # square 26
    opp1.state = TokenState.TRACK
    opp1.step_count = 5  # square 31

    # Active token on track
    act0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    act0.state = TokenState.TRACK
    act0.step_count = 4  # square 4

    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    assert struct_state.spatial_tensor[1, 26] == 1.0
    assert struct_state.spatial_tensor[1, 31] == 1.0
    assert struct_state.spatial_tensor[1, 4] == 0.0   # active player must NOT appear in Channel 1
    assert struct_state.spatial_tensor[1].sum() == 2.0


def test_spatial_channel_2_safe_squares(env):
    """Verify Channel 2 contains exactly 1.0 at safe squares and 0.0 elsewhere."""
    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    expected_safe_squares = {0, 8, 13, 21, 26, 34, 39, 47}
    for sq in range(TRACK_LENGTH):
        if sq in expected_safe_squares:
            assert struct_state.spatial_tensor[2, sq] == 1.0, f"Square {sq} should be marked safe (1.0)"
        else:
            assert struct_state.spatial_tensor[2, sq] == 0.0, f"Square {sq} should be non-safe (0.0)"

    assert struct_state.spatial_tensor[2].sum() == 8.0


def test_spatial_channel_3_blockades(env):
    """Verify Channel 3 marks squares with >= 2 tokens of the same color."""
    # Place friendly blockade (2 active tokens) at square 10
    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t1 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 1)
    t0.state = TokenState.TRACK
    t0.step_count = 10
    t1.state = TokenState.TRACK
    t1.step_count = 10

    # Place opponent blockade (2 opponent tokens) at square 30
    opp0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    opp1 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 1)
    opp0.state = TokenState.TRACK
    opp0.step_count = 4   # 26 + 4 = 30
    opp1.state = TokenState.TRACK
    opp1.step_count = 4   # 26 + 4 = 30

    # Place a single token at square 20 (no blockade)
    t2 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 2)
    t2.state = TokenState.TRACK
    t2.step_count = 20

    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    # Square 10 has friendly blockade
    assert struct_state.spatial_tensor[3, 10] == 1.0
    # Square 30 has opponent blockade
    assert struct_state.spatial_tensor[3, 30] == 1.0
    # Square 20 has single token: NOT a blockade
    assert struct_state.spatial_tensor[3, 20] == 0.0
    # Exactly 2 blockades total
    assert struct_state.spatial_tensor[3].sum() == 2.0


# =====================================================================
# 5. CONTEXT VECTOR: EXACT 12 DIMENSIONS & ORDERED FEATURES
# =====================================================================

def test_context_vector_exact_shape():
    """Verify context vector dimension is exactly 12."""
    assert CONTEXT_DIMENSION == 12
    assert len(CONTEXT_FEATURE_NAMES) == 12


def test_context_feature_ordering_and_semantics(env):
    """Verify exact feature ordering: yard, home, goal, dice, player, turn, bonus flags."""
    # Custom configuration:
    # Active (P0): 2 in yard, 1 in home column, 1 finished in goal
    # Opponent (P1): 1 in yard, 2 in home column, 1 on track (0 in goal)
    p0_toks = [t for t in env.tokens if t.player_id == 0]
    p1_toks = [t for t in env.tokens if t.player_id == 1]

    # P0: T0, T1 yard; T2 home column; T3 finished
    p0_toks[0].state = TokenState.YARD
    p0_toks[1].state = TokenState.YARD
    p0_toks[2].state = TokenState.HOME
    p0_toks[2].step_count = 53
    p0_toks[3].state = TokenState.FINISHED
    p0_toks[3].step_count = 56

    # P1: T0 yard; T1, T2 home column; T3 track
    p1_toks[0].state = TokenState.YARD
    p1_toks[1].state = TokenState.HOME
    p1_toks[1].step_count = 52
    p1_toks[2].state = TokenState.HOME
    p1_toks[2].step_count = 54
    p1_toks[3].state = TokenState.TRACK
    p1_toks[3].step_count = 10

    state_dict = env.get_state_dict()
    state_dict["current_player"] = 0
    state_dict["dice_value"] = 6
    state_dict["turn_count"] = 250
    state_dict["bonus_six"] = True
    state_dict["bonus_capture"] = False

    struct_state = build_structured_representation(state_dict)
    c = struct_state.context_vector

    # Index 0: Active yard count / 4.0 = 2 / 4 = 0.5
    assert c[0] == 0.5, "Index 0 must be active yard normalized"
    # Index 1: Opponent yard count / 4.0 = 1 / 4 = 0.25
    assert c[1] == 0.25, "Index 1 must be opponent yard normalized"
    # Index 2: Active home column count / 4.0 = 1 / 4 = 0.25
    assert c[2] == 0.25, "Index 2 must be active home column normalized"
    # Index 3: Opponent home column count / 4.0 = 2 / 4 = 0.5
    assert c[3] == 0.5, "Index 3 must be opponent home column normalized"
    # Index 4: Active goal count / 4.0 = 1 / 4 = 0.25
    assert c[4] == 0.25, "Index 4 must be active goal count normalized"
    # Index 5: Opponent goal count / 4.0 = 0 / 4 = 0.0
    assert c[5] == 0.0, "Index 5 must be opponent goal count normalized"
    # Index 6: Current dice roll / 6.0 = 6 / 6 = 1.0
    assert c[6] == 1.0, "Index 6 must be current dice roll / 6.0"
    # Index 7–8: Active player ID one-hot [1.0, 0.0] for P0
    assert c[7] == 1.0, "Index 7 must be 1.0 for Player 0"
    assert c[8] == 0.0, "Index 8 must be 0.0 for Player 0"
    # Index 9: Turn fraction (250 / 1000) = 0.25
    assert c[9] == 0.25, "Index 9 must be turn fraction turn/1000"
    # Index 10: Bonus roll after 6 = 1.0
    assert c[10] == 1.0, "Index 10 must be bonus roll after 6 flag"
    # Index 11: Capture bonus = 0.0
    assert c[11] == 0.0, "Index 11 must be capture bonus flag"

    assert validate_context_vector(c) is True
    assert verify_representation_against_protocol(struct_state, state_dict) is True


def test_player_1_active_context_encoding(env):
    """Verify player encoding when Player 1 is active."""
    state_dict = env.get_state_dict()
    state_dict["current_player"] = 1
    state_dict["dice_value"] = 3
    state_dict["turn_count"] = 400
    state_dict["bonus_six"] = False
    state_dict["bonus_capture"] = True

    struct_state = build_structured_representation(state_dict)
    c = struct_state.context_vector

    # Player 1 active one-hot: [0.0, 1.0]
    assert c[7] == 0.0
    assert c[8] == 1.0
    # Dice roll: 3 / 6 = 0.5
    assert c[6] == 0.5
    # Turn fraction: 400 / 1000 = 0.4
    assert c[9] == 0.4
    # Bonus flags: no six, capture bonus
    assert c[10] == 0.0
    assert c[11] == 1.0

    assert validate_context_vector(c) is True
    assert verify_representation_against_protocol(struct_state, state_dict) is True


def test_dice_normalization_range(env):
    """Verify all 6 dice values normalize strictly to [1..6] / 6.0."""
    state_dict = env.get_state_dict()
    for d in range(1, 7):
        state_dict["dice_value"] = d
        struct_state = build_structured_representation(state_dict)
        assert np.isclose(struct_state.context_vector[6], d / 6.0)


def test_turn_fraction_clamping(env):
    """Verify turn fraction is correctly clamped in [0.0, 1.0]."""
    state_dict = env.get_state_dict()
    state_dict["turn_count"] = 1500
    struct_state = build_structured_representation(state_dict)
    assert struct_state.context_vector[9] == 1.0


# =====================================================================
# 6. PROTOCOL DEVIATION AUTOMATED FAILURE TESTS
# =====================================================================

def test_deviation_fails_on_wrong_spatial_shape():
    """Verify protocol verification fails if spatial tensor shape deviates."""
    # Wrong channels: 3 instead of 4
    corrupted_spatial_3ch = np.zeros((3, 52), dtype=np.float32)
    with pytest.raises(StateDimensionError):
        validate_spatial_tensor(corrupted_spatial_3ch)

    # Wrong length: 50 instead of 52
    corrupted_spatial_50len = np.zeros((4, 50), dtype=np.float32)
    with pytest.raises(StateDimensionError):
        validate_spatial_tensor(corrupted_spatial_50len)


def test_deviation_fails_on_wrong_context_dimension():
    """Verify protocol verification fails if context dimension deviates from 12."""
    # 11 dimensions (missing feature)
    with pytest.raises(StateDimensionError):
        validate_context_vector(np.zeros(11, dtype=np.float32))

    # 13 dimensions (silently added feature)
    with pytest.raises(StateDimensionError):
        validate_context_vector(np.zeros(13, dtype=np.float32))


def test_deviation_fails_on_corrupted_safe_squares(env):
    """Verify protocol verification fails if safe squares in Channel 2 deviate."""
    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    corrupted_spatial = struct_state.spatial_tensor.copy()
    corrupted_spatial[2, 0] = 0.0  # safe square 0 corrupted to non-safe
    with pytest.raises(ProtocolDeviationError):
        validate_spatial_tensor(corrupted_spatial)


def test_deviation_fails_on_missed_blockade(env):
    """Verify protocol verification fails if Channel 3 misses a blockade."""
    state_dict = env.get_state_dict()
    struct_state = build_structured_representation(state_dict)

    corrupted_spatial = struct_state.spatial_tensor.copy()
    corrupted_spatial[0, 10] = 2.0  # 2 active tokens at square 10
    corrupted_spatial[3, 10] = 0.0  # failed to mark blockade
    with pytest.raises(ProtocolDeviationError):
        validate_spatial_tensor(corrupted_spatial)


def test_deviation_fails_on_corrupted_player_onehot():
    """Verify protocol verification fails if player one-hot does not sum to 1.0."""
    context = np.zeros(12, dtype=np.float32)
    context[0] = 1.0   # 4 in yard
    context[1] = 1.0   # 4 in yard
    context[6] = 0.5   # dice
    context[7] = 0.5   # corrupted one-hot sum
    context[8] = 0.5

    with pytest.raises(ProtocolDeviationError):
        validate_context_vector(context)


def test_deviation_fails_on_unnormalized_token_counts():
    """Verify protocol verification fails if token counts are not multiples of 0.25."""
    context = np.zeros(12, dtype=np.float32)
    context[0] = 0.333  # invalid fraction (not / 4.0)
    context[1] = 1.0
    context[7] = 1.0

    with pytest.raises(ProtocolDeviationError):
        validate_context_vector(context)


def test_deviation_fails_on_token_conservation_violation():
    """Verify protocol verification fails if yard + home + goal > 1.0."""
    context = np.zeros(12, dtype=np.float32)
    context[0] = 0.75  # 3 in yard
    context[2] = 0.50  # 2 in home -> sum = 1.25 > 1.0 (5 tokens!)
    context[1] = 1.0
    context[7] = 1.0

    with pytest.raises(ProtocolDeviationError):
        validate_context_vector(context)


def test_deviation_fails_on_non_binary_bonus_flag():
    """Verify protocol verification fails if bonus flags are not binary."""
    context = np.zeros(12, dtype=np.float32)
    context[0] = 1.0
    context[1] = 1.0
    context[7] = 1.0
    context[10] = 0.5  # non-binary bonus flag

    with pytest.raises(ProtocolDeviationError):
        validate_context_vector(context)


def test_verify_representation_against_ground_truth_detects_feature_swaps(env):
    """Verify verify_representation_against_protocol catches swapped context indices."""
    state_dict = env.get_state_dict()
    state_dict["current_player"] = 0
    state_dict["dice_value"] = 5
    state_dict["turn_count"] = 100

    struct_state = build_structured_representation(state_dict)

    # Swap index 0 (active yard) and index 1 (opp yard)
    corrupted_context = struct_state.context_vector.copy()
    corrupted_context[0] = 0.25
    corrupted_context[1] = 1.0

    corrupted_state = StructuredLudoState(
        spatial_tensor=struct_state.spatial_tensor,
        context_vector=corrupted_context,
        action_mask=struct_state.action_mask,
    )

    with pytest.raises(ProtocolDeviationError):
        verify_representation_against_protocol(corrupted_state, state_dict)


# =====================================================================
# 7. STRUCTURED CNN ENCODER EXACT INPUT / OUTPUT DIMENSIONS
# =====================================================================

def test_structured_cnn_encoder_exact_dimensions():
    """Verify StructuredCNNEncoder layer dimensions and input/output contracts.

    Inputs:
    - Spatial: (batch, 4, 52)
    - Context: (batch, 12)
    Outputs:
    - Standardized 128-D latent state feature vector.
    """
    encoder = StructuredCNNEncoder(in_channels=4, context_dim=12, latent_dim=128)

    # Verify layer specifications
    assert encoder.conv1.in_channels == 4
    assert encoder.conv1.out_channels == 32
    assert encoder.conv1.kernel_size == (3,)
    assert encoder.conv1.padding_mode == "circular"

    assert encoder.conv2.in_channels == 32
    assert encoder.conv2.out_channels == 64
    assert encoder.conv2.kernel_size == (3,)
    assert encoder.conv2.padding_mode == "circular"

    assert encoder.pool.output_size == 1
    assert encoder.spatial_feature_dim == 64
    assert encoder.fused_dim == 76  # 64 spatial + 12 context
    assert encoder.fc.in_features == 76
    assert encoder.fc.out_features == 128

    # Single state
    spatial = torch.randn(4, 52)
    context = torch.randn(12)
    out_single = encoder(spatial, context)
    assert out_single.shape == (128,)

    # Batched forward passes: B=1, 8, 16, 64
    for b in [1, 8, 16, 64]:
        b_spatial = torch.randn(b, 4, 52)
        b_context = torch.randn(b, 12)
        b_out = encoder(b_spatial, b_context)
        assert b_out.shape == (b, 128)


def test_cnn_encoder_without_explicit_context():
    """Verify CNN encoder works when context is omitted (defaults to zeros)."""
    encoder = StructuredCNNEncoder(in_channels=4, context_dim=12, latent_dim=128)
    spatial = torch.randn(4, 52)
    out = encoder(spatial)
    assert out.shape == (128,)


# =====================================================================
# 8. END-TO-END PIPELINE & PROTOCOL INTEGRATION
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


def test_environment_gameplay_states_satisfy_protocol(env):
    """Run simulated game steps and verify every state strictly satisfies frozen protocol."""
    obs, info = env.reset(seed=123)

    for _ in range(50):
        if env.is_terminal:
            break
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        act = np.random.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(act)

        struct_state = build_structured_representation(obs)
        # Authoritative protocol verification on every live game state
        assert verify_representation_against_protocol(struct_state, obs) is True
