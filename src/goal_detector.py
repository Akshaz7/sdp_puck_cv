import logging
import time
from typing import Optional
from src.models import Position, Velocity

logger = logging.getLogger(__name__)


class GoalDetector:
    """Detects goals by tracking puck disappearance near goal slits.

    The goal slits are on the LEFT and RIGHT edges of the table (short sides).
    From the camera's perspective the table is landscape.

    A goal is scored when:
        1. Puck was last seen near a goal slit (correct edge + within slit Y range)
        2. Puck was moving toward that edge
        3. Puck disappears for N consecutive frames
    """

    def __init__(
        self,
        table_width: float = 610.0,
        table_height: float = 1220.0,
        left_slit_y_min: Optional[float] = None,
        left_slit_y_max: Optional[float] = None,
        right_slit_y_min: Optional[float] = None,
        right_slit_y_max: Optional[float] = None,
        edge_margin: float = 80.0,
        disappear_frames: int = 10,
        cooldown_seconds: float = 3.0,
    ):
        """Initialise the goal detector.

        Args:
            table_width: Table width in mm (long axis as seen by camera).
            table_height: Table height in mm (short axis as seen by camera).
            left_slit_y_min: Top edge of left goal slit in mm.
            left_slit_y_max: Bottom edge of left goal slit in mm.
            right_slit_y_min: Top edge of right goal slit in mm.
            right_slit_y_max: Bottom edge of right goal slit in mm.
            edge_margin: How close to the left/right edge (in mm) the puck
                         must be before disappearing to count as near the goal.
            disappear_frames: How many consecutive frames the puck must be
                              missing before we call it a goal.
            cooldown_seconds: Minimum time between goal detections.
        """
        self._table_width = table_width
        self._table_height = table_height
        self._edge_margin = edge_margin
        self._disappear_frames = disappear_frames
        self._cooldown = cooldown_seconds
        self._last_goal_time: float = 0.0

        # Left slit Y bounds (default to center 40% of height)
        if left_slit_y_min is not None and left_slit_y_max is not None:
            self._left_slit_y_min = left_slit_y_min
            self._left_slit_y_max = left_slit_y_max
        else:
            self._left_slit_y_min = table_height * 0.3
            self._left_slit_y_max = table_height * 0.7

        # Right slit Y bounds
        if right_slit_y_min is not None and right_slit_y_max is not None:
            self._right_slit_y_min = right_slit_y_min
            self._right_slit_y_max = right_slit_y_max
        else:
            self._right_slit_y_min = table_height * 0.3
            self._right_slit_y_max = table_height * 0.7

        # Tracking state
        self._last_position: Optional[Position] = None
        self._last_velocity: Optional[Velocity] = None
        self._frames_missing: int = 0
        self._goal_candidate: Optional[str] = None

    def update(
        self, position: Optional[Position], velocity: Optional[Velocity] = None
    ) -> Optional[str]:
        """Update with current frame's detection result.

        Args:
            position: Current puck position in mm, or None if not detected.
            velocity: Current puck velocity in mm/s, or None.

        Returns:
            "human" if human scored (puck went into right goal, x=table_width),
            "robot" if robot scored (puck went into left goal, x=0),
            None if no goal this frame.
        """
        now = time.time()

        if now - self._last_goal_time < self._cooldown:
            if position is not None:
                self._last_position = position
                self._last_velocity = velocity
                self._frames_missing = 0
                self._goal_candidate = None
            return None

        if position is not None:
            self._last_position = position
            self._last_velocity = velocity
            self._frames_missing = 0
            self._goal_candidate = None
            return None

        # Puck is NOT visible
        self._frames_missing += 1

        if self._frames_missing == 1 and self._last_position is not None:
            self._goal_candidate = self._check_goal_candidate()

        if (
            self._goal_candidate is not None
            and self._frames_missing >= self._disappear_frames
        ):
            result = self._goal_candidate
            self._last_goal_time = now
            self._goal_candidate = None
            self._last_position = None
            self._last_velocity = None
            self._frames_missing = 0
            logger.info(
                "GOAL detected: %s scored (puck disappeared into %s goal)"
                % (result, "right" if result == "human" else "left")
            )
            return result

        return None

    def _check_goal_candidate(self) -> Optional[str]:
        """Check if the last known position + velocity qualifies as a goal."""
        pos = self._last_position
        vel = self._last_velocity

        if pos is None:
            return None

        # Check LEFT edge (x near 0) — human's goal
        near_left = pos.x <= self._edge_margin
        in_left_slit = self._left_slit_y_min <= pos.y <= self._left_slit_y_max

        if near_left and in_left_slit:
            moving_toward = vel is None or vel.vx <= 0
            if moving_toward:
                return "robot"  # robot scores (puck entered human's goal on left)

        # Check RIGHT edge (x near table_width) — robot's goal
        near_right = pos.x >= self._table_width - self._edge_margin
        in_right_slit = self._right_slit_y_min <= pos.y <= self._right_slit_y_max

        if near_right and in_right_slit:
            moving_toward = vel is None or vel.vx >= 0
            if moving_toward:
                return "human"  # human scores (puck entered robot's goal on right)

        return None

    def reset(self) -> None:
        """Reset detector state."""
        self._last_goal_time = 0.0
        self._last_position = None
        self._last_velocity = None
        self._frames_missing = 0
        self._goal_candidate = None
