"""Two-player Ludo MDP environment complying with research benchmark specifications."""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from phd_ludo_drl.environment.constants import (
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    ACTION_SPACE_SIZE,
    TokenState,
)
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.dice import Dice
from phd_ludo_drl.environment.board import LudoBoard
from phd_ludo_drl.environment.rules import LudoRules, MoveResult
from phd_ludo_drl.environment.reward_calculator import LudoRewardCalculator


class LudoEnvironment:
    """Authoritative Two-Player Ludo Environment.

    Adheres strictly to Alhajry et al. (2012) and IEEE published research foundation:
    - 2 players (Player 0 and Player 1)
    - 4 tokens per player
    - 52 common circular track squares
    - 5 dedicated home column squares per player + 1 goal square
    - Safe squares where tokens cannot be captured
    - Blockades where 2+ same-player tokens block all opponent movement
    - Exact roll required to reach goal
    - Discrete 4-action space (token selection 0..3)
    - Bonus roll on rolling a 6 or capturing an opponent token
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_turns: int = 1000,
        reward_calculator: Optional[LudoRewardCalculator] = None,
        auto_advance_no_moves: bool = True,
        dice_probabilities: Optional[Any] = None,
    ):
        self.max_turns = max_turns
        self.auto_advance_no_moves = auto_advance_no_moves
        self.dice = Dice(seed=seed, probabilities=dice_probabilities)
        self.board = LudoBoard()
        self.rules = LudoRules(self.board)
        self.reward_calculator = reward_calculator or LudoRewardCalculator()

        # Initialize tokens: 4 tokens for Player 0, 4 tokens for Player 1
        self.tokens: List[Token] = [
            Token(player_id=p, token_id=t)
            for p in range(NUM_PLAYERS)
            for t in range(TOKENS_PER_PLAYER)
        ]

        self.current_player: int = 0
        self.turn_count: int = 0
        self.is_terminal: bool = False
        self.winner: Optional[int] = None
        self.consecutive_sixes: int = 0

    def seed(self, seed: Optional[int] = None) -> None:
        """Reseed the environment's dice RNG."""
        self.dice.seed(seed)

    def set_dice_distribution(self, probabilities: Optional[Any]) -> None:
        """Set or clear custom dice probability distribution for stochasticity experiments."""
        self.dice.set_distribution(probabilities)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            observation (state_dict) and info dict.
        """
        if seed is not None:
            self.seed(seed)

        # Reset tokens to YARD
        for token in self.tokens:
            token.reset()

        self.current_player = 0
        self.turn_count = 0
        self.is_terminal = False
        self.winner = None
        self.consecutive_sixes = 0

        # Roll initial die
        self.dice.roll()

        # If initial player has no moves, advance if auto_advance is enabled
        if self.auto_advance_no_moves:
            self._advance_until_legal_or_terminal()

        obs = self.get_state_dict()
        info = {
            "current_player": self.current_player,
            "dice_value": self.dice.value,
            "legal_actions": self.get_legal_actions(),
            "action_mask": self.get_action_mask(),
            "is_terminal": self.is_terminal,
            "winner": self.winner,
        }
        return obs, info

    def get_legal_actions(self) -> List[int]:
        """Return list of legal token actions [0..3] for the current player."""
        if self.is_terminal:
            return []
        return self.rules.get_legal_actions(self.tokens, self.current_player, self.dice.value)

    def get_action_mask(self) -> np.ndarray:
        """Return binary action mask of shape (4,) for current player."""
        if self.is_terminal:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        return self.rules.get_action_mask(self.tokens, self.current_player, self.dice.value)

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute a token-selection action (0..3).

        Args:
            action: Token index (0, 1, 2, or 3) belonging to the active player.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.is_terminal:
            return self.get_state_dict(), 0.0, True, False, {"error": "Game is already terminated"}

        legal_actions = self.get_legal_actions()

        # Handle illegal action rejection
        if action not in legal_actions:
            penalty = self.reward_calculator.get_illegal_action_penalty()
            info = {
                "error": "Illegal action attempted",
                "attempted_action": action,
                "legal_actions": legal_actions,
                "action_mask": self.get_action_mask(),
                "success": False,
            }
            return self.get_state_dict(), penalty, False, False, info

        # Locate the specific token belonging to active player
        acting_token = next(
            t for t in self.tokens
            if t.player_id == self.current_player and t.token_id == action
        )

        # Apply move
        dice_used = self.dice.value
        move_result = self.rules.apply_move(self.tokens, acting_token, dice_used)
        self.turn_count += 1

        # Check for game completion
        winner = self.rules.check_winner(self.tokens)
        if winner is not None:
            self.is_terminal = True
            self.winner = winner

        # Check truncation limit
        truncated = False
        if not self.is_terminal and self.turn_count >= self.max_turns:
            truncated = True
            self.is_terminal = True

        # Compute shaped reward
        reward = self.reward_calculator.compute_move_reward(
            move_result,
            is_terminal=self.is_terminal,
            winner=self.winner,
        )

        # Handle turn progression and bonus rolls if game continues
        if not self.is_terminal:
            if move_result.bonus_roll:
                # Same player rolls again
                if dice_used == 6:
                    self.consecutive_sixes += 1
                self.dice.roll()
            else:
                # Turn transitions to opponent
                self.consecutive_sixes = 0
                self.current_player = 1 - self.current_player
                self.dice.roll()

            if self.auto_advance_no_moves:
                self._advance_until_legal_or_terminal()

        obs = self.get_state_dict()
        info = {
            "move_result": move_result,
            "current_player": self.current_player,
            "dice_value": self.dice.value,
            "legal_actions": self.get_legal_actions(),
            "action_mask": self.get_action_mask(),
            "is_terminal": self.is_terminal,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "success": True,
        }
        return obs, reward, self.is_terminal, truncated, info

    def _advance_until_legal_or_terminal(self) -> None:
        """Advance turns if the active player has no legal moves."""
        pass_count = 0
        max_consecutive_passes = 200  # safeguard against deadlock

        while not self.is_terminal and len(self.get_legal_actions()) == 0:
            self.consecutive_sixes = 0
            self.current_player = 1 - self.current_player
            self.dice.roll()
            self.turn_count += 1
            pass_count += 1

            if self.turn_count >= self.max_turns or pass_count >= max_consecutive_passes:
                self.is_terminal = True
                break

    def get_state_dict(self) -> Dict[str, Any]:
        """Return a detached snapshot of authoritative state."""
        return {
            "current_player": self.current_player,
            "dice_value": self.dice.value,
            "tokens": [t.copy() for t in self.tokens],
            "action_mask": self.get_action_mask(),
            "is_terminal": self.is_terminal,
            "winner": self.winner,
            "turn_count": self.turn_count,
            "consecutive_sixes": self.consecutive_sixes,
        }

    def render(self, mode: str = "text") -> str:
        """Render a readable summary of current board state."""
        lines = [
            f"--- Ludo Game State (Turn {self.turn_count}) ---",
            f"Current Player: P{self.current_player} | Dice Roll: {self.dice.value}",
            f"Legal Actions: {self.get_legal_actions()} | Mask: {self.get_action_mask().tolist()}",
            f"Terminal: {self.is_terminal} | Winner: {self.winner}",
        ]
        for p in range(NUM_PLAYERS):
            p_tokens = [t for t in self.tokens if t.player_id == p]
            token_desc = [
                f"T{t.token_id}: {t.state.name} (step={t.step_count}, track={t.track_square}, home={t.home_index})"
                for t in p_tokens
            ]
            lines.append(f"Player {p} Tokens: " + ", ".join(token_desc))
        out = "\n".join(lines)
        if mode == "human":
            print(out)
        return out
