"""Encoding package for PhD Ludo DRL."""

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
    EncodedState,
    StateRepresentationPipeline,
)

__all__ = [
    "RAW_STATE_DIMENSION",
    "RawLudoObservation",
    "extract_raw_observation",
    "StateValidationError",
    "StateDimensionError",
    "validate_raw_observation",
    "validate_flat_vector",
    "validate_structured_tensor",
    "FlatEncoder",
    "StructuredLudoState",
    "build_structured_representation",
    "StructuredCNNEncoder",
    "STANDARD_FEATURE_DIMENSION",
    "EncodedState",
    "StateRepresentationPipeline",
]
