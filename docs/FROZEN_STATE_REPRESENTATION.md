# Frozen State Representation Specification

**Status:** APPROVED & LOCKED  
**Document Reference:** `FINAL_EXPERIMENT_PROTOCOL.md` Section 3 (Version 1.0.0 FROZEN)  
**Implementation Modules:**  
- `phd_ludo_drl.encoding.structured_representation`  
- `phd_ludo_drl.encoding.cnn_encoder`  
- `phd_ludo_drl.encoding.validation`  
- `phd_ludo_drl.encoding.encoder_pipeline`  

---

## 1. Spatial Board Tensor ($S \in \mathbb{R}^{4 \times 52}$)

The spatial representation directly encodes the circular 52-tile geometry of the Ludo track across 4 specialized planar feature channels:

| Channel | Identifier | Semantic Definition | Values / Encoding |
| :---: | :--- | :--- | :--- |
| **0** | `active_token_occupancy` | Active player token count per track tile ($t \in [0..51]$) | Non-negative integer count $[0.0, 4.0]$ |
| **1** | `opponent_token_occupancy` | Opponent token count per track tile ($t \in [0..51]$) | Non-negative integer count $[0.0, 4.0]$ |
| **2** | `safe_square_indicators` | Safe square spatial indicators | Binary: $1.0$ at safe squares $\{0, 8, 13, 21, 26, 34, 39, 47\}$, $0.0$ elsewhere |
| **3** | `blockade_indicators` | Blockade indicators | Binary: $1.0$ at tiles occupied by $\ge 2$ tokens of the same color, $0.0$ elsewhere |

- **Shape:** `(4, 52)` (or `(B, 4, 52)` when batched)
- **Data Type:** `numpy.float32` / `torch.float32`
- **Track Length:** Exactly 52 squares ($0$ to $51$).

---

## 2. Context Vector ($C \in \mathbb{R}^{12}$)

The scalar context vector provides essential global state and non-spatial progress metrics. All 12 features are strictly ordered and normalized within $[0.0, 1.0]$:

| Index | Feature Name | Normalization / Encoding | Value Range |
| :---: | :--- | :--- | :--- |
| **0** | `active_yard_tokens` | Active player tokens in yard $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **1** | `opponent_yard_tokens` | Opponent tokens in yard $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **2** | `active_home_tokens` | Active player tokens in home column $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **3** | `opponent_home_tokens` | Opponent tokens in home column $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **4** | `active_goal_tokens` | Active player tokens in goal $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **5** | `opponent_goal_tokens` | Opponent tokens in goal $[0..4]$ normalized by 4 | $\{0.0, 0.25, 0.50, 0.75, 1.0\}$ |
| **6** | `dice_roll_normalized` | Current dice roll $[1..6]$ normalized by 6 | $[1/6, 1.0]$ ($0.0$ if unrolled) |
| **7** | `active_player_p0_onehot` | One-hot index 0 for active player ID | $1.0$ if Player 0 active, else $0.0$ |
| **8** | `active_player_p1_onehot` | One-hot index 1 for active player ID | $1.0$ if Player 1 active, else $0.0$ |
| **9** | `turn_fraction` | Turn counter normalized by maximum horizon ($\text{turn} / 1000.0$) | $[0.0, 1.0]$ |
| **10** | `bonus_roll_six_flag` | Flag indicating active roll was awarded for rolling a 6 | $\{0.0, 1.0\}$ |
| **11** | `bonus_roll_capture_flag` | Flag indicating active roll was awarded for capturing an opponent | $\{0.0, 1.0\}$ |

---

## 3. Companion Action Mask ($M \in \{0, 1\}^4$)

- Binary 4-dimensional vector indicating legal actions for tokens $0, 1, 2, 3$ belonging to the active player.
- Prevents invalid moves by suppressing illegal action logits by $-10^9$.

---

## 4. Shared CNN Encoder Specification (`StructuredCNNEncoder`)

The encoder maps the $(4, 52)$ spatial board tensor and $(12,)$ context vector into a unified $128$-dimensional latent state vector $h(s) \in \mathbb{R}^{128}$:

```
Spatial Input (B, 4, 52)
       │
       ▼
Conv1d(4 -> 32, kernel=3, padding=1, padding_mode='circular') + ReLU   --> (B, 32, 52)
       │
       ▼
Conv1d(32 -> 64, kernel=3, padding=1, padding_mode='circular') + ReLU  --> (B, 64, 52)
       │
       ▼
AdaptiveAvgPool1d(1)                                                   --> (B, 64, 1)
       │
       ▼
Flatten (64 features)                                                  --> (B, 64)
       │
       ├───────────────────────────────────────── Context Vector (B, 12)
       ▼                                                 │
Concatenation (64 + 12) ─────────────────────────────────┘             --> (B, 76)
       │
       ▼
Linear(76 -> 128) + ReLU                                              --> (B, 128)
       │
       ▼
Latent State Representation h(s)                                       --> (B, 128)
```

- **Exact Input Dimensions:** Spatial tensor `(B, 4, 52)` and Context vector `(B, 12)`
- **Exact Output Dimension:** Latent representation `(B, 128)`

---

## 5. Automated Protocol Verification Suite

The repository contains rigorous automated tests enforcing protocol invariants:
1. `validate_spatial_tensor`: Validates shape `(4, 52)`, non-negativity, exact safe-square indices, and blockade indicator rules.
2. `validate_context_vector`: Validates shape `(12,)`, token conservation (yard + home + goal $\le 1.0$), player one-hot sum $= 1.0$, dice normalization, and bonus binary flags.
3. `verify_representation_against_protocol`: Ground-truth verification comparing structured representations directly against raw simulation states.
4. `test_deviation_fails_*`: Automated negative tests verifying that any deviation in shape, feature ordering, normalization, or semantics raises `ProtocolDeviationError` or `StateDimensionError`.
