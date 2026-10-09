"""Encoding package for PhD Ludo DRL."""

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
)
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.encoder_pipeline import (
    STANDARD_FEATURE_DIMENSION,
    EncodedState,
    StateRepresentationPipeline,
)

__all__ = [
    "RAW_STATE_DIMENSION",
    "RawLudoObservation",
    "extract_raw_observation",
    "StateValidationError",
    "StateDimensionError",
    "ProtocolDeviationError",
    "validate_raw_observation",
    "validate_flat_vector",
    "validate_structured_tensor",
    "validate_spatial_tensor",
    "validate_context_vector",
    "validate_frozen_state",
    "verify_representation_against_protocol",
    "FlatEncoder",
    "StructuredLudoState",
    "build_structured_representation",
    "SPATIAL_SHAPE",
    "CONTEXT_DIMENSION",
    "ACTION_MASK_DIMENSION",
    "CHANNEL_ACTIVE_OCCUPANCY",
    "CHANNEL_OPPONENT_OCCUPANCY",
    "CHANNEL_SAFE_SQUARES",
    "CHANNEL_BLOCKADES",
    "CHANNEL_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "StructuredCNNEncoder",
    "STANDARD_FEATURE_DIMENSION",
    "EncodedState",
    "StateRepresentationPipeline",
]
