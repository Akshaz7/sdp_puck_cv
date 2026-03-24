import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GameMode(Enum):
    FREE = "FREE"           # Free play, no win condition
    FIRST5 = "FIRST5"       # First to 5 goals
    FIRST7 = "FIRST7"       # First to 7 goals
    TIMED3 = "TIMED3"       # 3 minute match
    TIMED5 = "TIMED5"       # 5 minute match


class GameState(Enum):
    WAITING = "WAITING"     # Waiting for mode selection / start
    PLAYING = "PLAYING"     # Game in progress
    FINISHED = "FINISHED"   # Game over, showing result


class GameManager:
    """Manages game state, scoring, and win conditions."""

    def __init__(self) -> None:
        self._mode: GameMode = GameMode.FREE
        self._state: GameState = GameState.WAITING
        self._human_score: int = 0
        self._robot_score: int = 0
        self._start_time: float = 0.0
        self._time_limit: float = 0.0  # seconds, 0 = no limit

    def set_mode(self, mode: GameMode) -> None:
        """Set the game mode. Only works in WAITING state."""
        if self._state != GameState.WAITING:
            return
        self._mode = mode
        logger.info(f"Game mode set to {mode.value}")

    def start(self) -> None:
        """Start the game. Resets scores and timer."""
        self._human_score = 0
        self._robot_score = 0
        self._start_time = time.time()
        self._state = GameState.PLAYING

        if self._mode == GameMode.TIMED3:
            self._time_limit = 180.0
        elif self._mode == GameMode.TIMED5:
            self._time_limit = 300.0
        else:
            self._time_limit = 0.0

        logger.info(f"Game started: {self._mode.value}")

    def record_goal(self, scorer: str) -> None:
        """Record a goal. scorer is 'human' or 'robot'."""
        if self._state != GameState.PLAYING:
            return

        if scorer == "human":
            self._human_score += 1
            logger.info(f"Human scores! {self._human_score}-{self._robot_score}")
        elif scorer == "robot":
            self._robot_score += 1
            logger.info(f"Robot scores! {self._human_score}-{self._robot_score}")

        self._check_win_condition()

    def update(self) -> None:
        """Called each frame to check time-based win conditions."""
        if self._state != GameState.PLAYING:
            return
        if self._time_limit > 0 and self.elapsed_seconds >= self._time_limit:
            self._state = GameState.FINISHED
            logger.info(
                f"Time's up! Final: {self._human_score}-{self._robot_score}"
            )

    def reset(self) -> None:
        """Reset to waiting state."""
        self._state = GameState.WAITING
        self._human_score = 0
        self._robot_score = 0
        self._start_time = 0.0
        logger.info("Game reset")

    def _check_win_condition(self) -> None:
        target = 0
        if self._mode == GameMode.FIRST5:
            target = 5
        elif self._mode == GameMode.FIRST7:
            target = 7

        if target > 0:
            if self._human_score >= target or self._robot_score >= target:
                self._state = GameState.FINISHED
                winner = (
                    "Human" if self._human_score >= target else "Robot"
                )
                logger.info(f"Game over! {winner} wins!")

    @property
    def mode(self) -> GameMode:
        return self._mode

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def human_score(self) -> int:
        return self._human_score

    @property
    def robot_score(self) -> int:
        return self._robot_score

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining for timed modes. Returns 0 if untimed."""
        if self._time_limit <= 0:
            return 0.0
        remaining = self._time_limit - self.elapsed_seconds
        return max(0.0, remaining)

    @property
    def winner(self) -> Optional[str]:
        """Return 'human', 'robot', or 'draw' if finished. None if not finished."""
        if self._state != GameState.FINISHED:
            return None
        if self._human_score > self._robot_score:
            return "human"
        elif self._robot_score > self._human_score:
            return "robot"
        return "draw"
