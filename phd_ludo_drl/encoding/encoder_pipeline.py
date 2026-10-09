"""End-to-end state representation pipeline connecting environment states to 128-D features."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from phd_ludo_drl.encoding.raw_observation import (
    RAW_STATE_DIMENSION,
    RawLudoObservation,
    extract_raw_observation,
)
from phd_ludo_drl.encoding.validation import (
    validate_raw_observation,
    validate_structured_tensor,
    validate_frozen_state,
    StateDimensionError,
)
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.encoding.flat_encoder import FlatEncoder

STANDARD_FEATURE_DIMENSION = 128


@dataclass
class EncodedState:
    """Standardized model input package."""
    feature_vector: torch.Tensor  # shape (128,) or (batch, 128)
    action_mask: torch.Tensor     # shape (4,) or (batch, 4) companion tensor


class StateRepresentationPipeline:
    """Authoritative State Representation Pipeline.

    Execution Flow:
    Ludo State Dict
    -> State Validation
    -> Structured Representation (4-channel + context)
    -> CNN Encoder
    -> Feature Representation (strictly 128-D) + Action Mask (4-D)
    """

    def __init__(
        self,
        encoder_type: str = "structured_cnn",
        device: Optional[torch.device] = None,
    ):
        self.encoder_type = encoder_type.lower()
        self.device = device or torch.device("cpu")

        if self.encoder_type == "structured_cnn":
            self.encoder: nn.Module = StructuredCNNEncoder().to(self.device)
        elif self.encoder_type == "flat":
            self.encoder = FlatEncoder().to(self.device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}. Must be 'structured_cnn' or 'flat'.")

    def encode_state(self, state_dict: Dict[str, Any]) -> EncodedState:
        """Process a single environment state dictionary through the pipeline.

        Args:
            state_dict: Environment state dictionary from env.get_state_dict().

        Returns:
            EncodedState containing 128-D feature_vector and 4-D action_mask.
        """
        # Step 1 & 2: Extract & Validate raw observation
        raw_obs = extract_raw_observation(state_dict)
        validate_raw_observation(raw_obs)

        # Step 3 & 4: Construct representation & encode
        if self.encoder_type == "structured_cnn":
            struct_state = build_structured_representation(state_dict)
            validate_frozen_state(struct_state)

            with torch.no_grad():
                features = self.encoder(
                    struct_state.spatial_tensor,
                    struct_state.context_vector,
                )
            mask_tensor = torch.from_numpy(struct_state.action_mask).float().to(self.device)
        else:
            flat_vec = raw_obs.to_flat_vector()
            with torch.no_grad():
                features = self.encoder(flat_vec)
            mask_tensor = torch.from_numpy(raw_obs.action_mask).float().to(self.device)

        # Step 5: Output dimension validation
        if features.shape[-1] != STANDARD_FEATURE_DIMENSION:
            raise StateDimensionError(
                f"Encoder output dimension mismatch: expected {STANDARD_FEATURE_DIMENSION}, got {features.shape[-1]}"
            )

        return EncodedState(
            feature_vector=features,
            action_mask=mask_tensor,
        )

    def encode_batch(self, state_dicts: List[Dict[str, Any]]) -> EncodedState:
        """Process a batch of environment state dictionaries through the pipeline."""
        if not state_dicts:
            raise ValueError("Cannot encode an empty batch of states.")

        spatials = []
        contexts = []
        masks = []
        flats = []

        for sd in state_dicts:
            raw_obs = extract_raw_observation(sd)
            validate_raw_observation(raw_obs)

            if self.encoder_type == "structured_cnn":
                st = build_structured_representation(sd)
                validate_frozen_state(st)
                spatials.append(st.spatial_tensor)
                contexts.append(st.context_vector)
                masks.append(st.action_mask)
            else:
                flats.append(raw_obs.to_flat_vector())
                masks.append(raw_obs.action_mask)

        with torch.no_grad():
            if self.encoder_type == "structured_cnn":
                spatials_arr = np.stack(spatials, axis=0)  # (B, 4, 52)
                contexts_arr = np.stack(contexts, axis=0)  # (B, 12)
                features = self.encoder(spatials_arr, contexts_arr)
            else:
                flats_arr = np.stack(flats, axis=0)        # (B, 60)
                features = self.encoder(flats_arr)

        masks_t = torch.from_numpy(np.stack(masks, axis=0)).float().to(self.device)

        # Validate batch output dimension
        if features.shape[-1] != STANDARD_FEATURE_DIMENSION or features.shape[0] != len(state_dicts):
            raise StateDimensionError(
                f"Batch encoder output mismatch: expected shape ({len(state_dicts)}, {STANDARD_FEATURE_DIMENSION}), got {features.shape}"
            )

        return EncodedState(
            feature_vector=features,
            action_mask=masks_t,
        )
