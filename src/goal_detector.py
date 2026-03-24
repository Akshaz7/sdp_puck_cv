import logging
import time
from typing import Optional
from src.models import Position

logger = logging.getLogger(__name__)


class GoalDetector:
    """Detects when the puck crosses a goal line using CV position data.

    A goal is triggered when:
        - The puck's Y coordinate crosses y=0 (robot scores on human)
        - The puck's Y coordinate crosses y=table_height (human scores on robot)

    Uses a cooldown period to prevent duplicate detections from the same event.
    Requires the puck to have been visible recently (not a phantom detection).
    """

    def __init__(
        self,
        table_height: float = 1220.0,
        goal_margin: float = 30.0,
        cooldown_seconds: float = 2.0,
    ):
        """Initialise the goal detector.

        Args:
            table_height: Table height in mm.
            goal_margin: How close to the edge (in mm) counts as a goal.
            cooldown_seconds: Minimum time between goal detections.
        """
        self._table_height = table_height
        self._goal_margin = goal_margin
        self._cooldown = cooldown_seconds
        self._last_goal_time: float = 0.0
        self._prev_position: Optional[Position] = None

    def check(self, position: Optional[Position]) -> Optional[str]:
        """Check if a goal has been scored this frame.

        Args:
            position: Current puck position in mm, or None if not detected.

        Returns:
            "human" if human scored (puck at robot's goal, y >= table_height - margin),
            "robot" if robot scored (puck at human's goal, y <= margin),
            None if no goal.
        """
        if position is None:
            self._prev_position = None
            return None

        now = time.time()
        if now - self._last_goal_time < self._cooldown:
            self._prev_position = position
            return None

        # Need a previous position to confirm the puck was moving into the goal
        # (not just appearing near the edge)
        result = None
        if self._prev_position is not None:
            if position.y >= self._table_height - self._goal_margin:
                # Puck reached robot's goal line — human scores
                result = "human"
            elif position.y <= self._goal_margin:
                # Puck reached human's goal line — robot scores
                result = "robot"

        if result is not None:
            self._last_goal_time = now
            logger.info(f"Goal detected: {result} scored")

        self._prev_position = position
        return result

    def reset(self) -> None:
        """Reset detector state (e.g. after game reset)."""
        self._last_goal_time = 0.0
        self._prev_position = None
