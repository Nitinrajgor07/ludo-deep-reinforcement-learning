"""Game rules, legal action generation, and move execution engine for Ludo."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from phd_ludo_drl.environment.constants import (
    TokenState,
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    TRACK_STEPS,
    GOAL_STEP,
    PLAYER_START_SQUARES,
)
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.board import LudoBoard


@dataclass
class MoveResult:
    """Outcome of executing a token move."""
    token_id: int
    player_id: int
    previous_step: int
    new_step: int
    released: bool = False
    captured_tokens: List[Token] = field(default_factory=list)
    formed_blockade: bool = False
    entered_home: bool = False
    finished: bool = False
    moved_to_safe: bool = False
    bonus_roll: bool = False


class LudoRules:
    """Authoritative rule verification and move execution with optional team awareness."""

    def __init__(self, board: Optional[LudoBoard] = None, teams: Optional[Dict[int, int]] = None):
        self.board = board or LudoBoard()
        self.teams = teams

    def is_move_legal(
        self,
        tokens: List[Token],
        token: Token,
        dice: int,
        teams: Optional[Dict[int, int]] = None,
    ) -> bool:
        """Determine if a token can legally move given the dice roll."""
        if token.is_finished:
            return False

        active_teams = teams if teams is not None else self.teams

        if token.is_in_yard:
            if dice != 6:
                return False
            # Check if start square is blocked by an opponent blockade
            return not self.board.is_path_blocked_by_opponent(tokens, token, dice, teams=active_teams)

        new_step = token.step_count + dice

        # Exact finish requirement: cannot overshoot the goal
        if new_step > GOAL_STEP:
            return False

        # Check for opponent blockades along track path
        if self.board.is_path_blocked_by_opponent(tokens, token, dice, teams=active_teams):
            return False

        return True

    def get_legal_actions(
        self,
        tokens: List[Token],
        player_id: int,
        dice: int,
        teams: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        """Return list of token IDs [0..3] that can legally move."""
        active_teams = teams if teams is not None else self.teams
        player_tokens = [t for t in tokens if t.player_id == player_id]
        return [
            t.token_id for t in player_tokens
            if self.is_move_legal(tokens, t, dice, teams=active_teams)
        ]

    def get_action_mask(
        self,
        tokens: List[Token],
        player_id: int,
        dice: int,
        teams: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Return a binary mask of shape (4,) where 1=legal, 0=illegal."""
        mask = np.zeros(TOKENS_PER_PLAYER, dtype=np.int8)
        legal_actions = self.get_legal_actions(tokens, player_id, dice, teams=teams)
        for act in legal_actions:
            mask[act] = 1
        return mask

    def apply_move(
        self,
        tokens: List[Token],
        token: Token,
        dice: int,
        teams: Optional[Dict[int, int]] = None,
    ) -> MoveResult:
        """Execute a validated legal move and update game state."""
        active_teams = teams if teams is not None else self.teams
        prev_step = token.step_count
        captured: List[Token] = []
        released = False
        entered_home = False
        finished = False
        moved_to_safe = False

        if token.is_in_yard:
            # Release from yard to start square
            released = True
            token.state = TokenState.TRACK
            token.step_count = 0
        else:
            token.step_count += dice
            if token.step_count == GOAL_STEP:
                token.state = TokenState.FINISHED
                finished = True
            elif token.step_count >= TRACK_STEPS:
                if prev_step < TRACK_STEPS:
                    entered_home = True
                token.state = TokenState.HOME
            else:
                token.state = TokenState.TRACK

        # Check safe square & captures if currently on track
        if token.is_on_track:
            sq = token.track_square
            if sq is not None:
                if self.board.is_safe_square(sq):
                    moved_to_safe = True
                else:
                    # Capture opponent tokens on this square (safe squares prevent capture)
                    for other in tokens:
                        if active_teams is None:
                            is_opponent = (other.player_id != token.player_id)
                        else:
                            is_opponent = (active_teams.get(other.player_id) != active_teams.get(token.player_id))

                        if (
                            is_opponent
                            and other.is_on_track
                            and other.track_square == sq
                        ):
                            other.reset()
                            captured.append(other)

        # Check blockade formation
        formed_blockade = False
        if token.is_on_track and token.track_square is not None:
            if active_teams is not None and token.player_id in active_teams:
                formed_blockade = self.board.has_team_blockade(
                    tokens, token.track_square, active_teams[token.player_id], active_teams
                )
            else:
                formed_blockade = self.board.has_blockade(tokens, token.track_square, token.player_id)

        # Bonus roll granted if dice was 6 or if an opponent was captured
        bonus_roll = (dice == 6) or (len(captured) > 0)

        return MoveResult(
            token_id=token.token_id,
            player_id=token.player_id,
            previous_step=prev_step,
            new_step=token.step_count,
            released=released,
            captured_tokens=captured,
            formed_blockade=formed_blockade,
            entered_home=entered_home,
            finished=finished,
            moved_to_safe=moved_to_safe,
            bonus_roll=bonus_roll,
        )

    @staticmethod
    def check_winner(tokens: List[Token], teams: Optional[Dict[int, int]] = None) -> Optional[int]:
        """Check if any player or team has moved all required tokens to the goal.

        When teams is None (Individual mode):
            Returns the winning player_id when all 4 tokens of that player are finished.
        When teams is provided (Team mode):
            Returns the winning team_id when all 8 tokens of that team are finished.
        """
        if teams is None:
            player_ids = sorted(list(set(t.player_id for t in tokens)))
            for p in player_ids:
                finished_count = sum(
                    1 for t in tokens if t.player_id == p and t.is_finished
                )
                if finished_count == TOKENS_PER_PLAYER:
                    return p
            return None
        else:
            team_ids = sorted(list(set(teams.values())))
            for tid in team_ids:
                team_tokens = [t for t in tokens if teams.get(t.player_id) == tid]
                if len(team_tokens) > 0 and all(t.is_finished for t in team_tokens):
                    return tid
            return None

    @staticmethod
    def check_team_winner(tokens: List[Token], teams: Dict[int, int]) -> Optional[int]:
        """Explicitly check if a team has achieved victory with all 8 team tokens finished."""
        return LudoRules.check_winner(tokens, teams=teams)
