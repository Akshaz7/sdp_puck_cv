import logging
import numpy as np
from src.models import Position, Velocity, Prediction

logger = logging.getLogger(__name__)


class TrajectoryPredictor:
    """Predicts puck trajectory with wall bounce reflections.

    Given a position and velocity, simulates the puck's path forward in time.
    The puck bounces off the left wall (x=0) and right wall (x=table_width).
    Prediction ends when the puck crosses the defence_y line (robot's side)
    or when max_bounces is reached.

    Algorithm:
        1. Start at current position with current velocity.
        2. Compute time to hit each boundary:
           - Left wall (x=0): t = -x / vx (only if vx < 0)
           - Right wall (x=table_width): t = (table_width - x) / vx (only if vx > 0)
           - Defence line (y=defence_y): t = (defence_y - y) / vy (only if vy > 0)
           - Top wall (y=0): t = -y / vy (only if vy < 0)
        3. Take the smallest positive t.
        4. If the nearest boundary is the defence line, record interception point
           and stop.
        5. If the nearest boundary is a side wall, move to that point, reflect vx
           (multiply by -1), continue.
        6. If the nearest boundary is the top wall, move to that point, reflect vy
           (multiply by -1), continue.
        7. Record each waypoint for visualisation.
        8. Repeat until defence line is crossed or max_bounces is reached.

    If the puck is moving away from the robot (vy <= 0), set is_approaching
    to False and return a prediction with the current x position as a default
    interception.
    """

    def __init__(
        self,
        table_width: float = 610.0,
        table_height: float = 1220.0,
        defence_y: float = 1100.0,
        max_bounces: int = 10,
    ):
        """Initialise the predictor.

        Args:
            table_width: Table width in mm.
            table_height: Table height in mm.
            defence_y: Y coordinate of the robot's defence line in mm.
            max_bounces: Maximum number of wall bounces to simulate before
                         giving up.
        """
        self._table_width = table_width
        self._table_height = table_height
        self._defence_y = defence_y
        self._max_bounces = max_bounces

    def predict(self, position: Position, velocity: Velocity) -> Prediction:
        """Predict the puck's trajectory from current state.

        Args:
            position: Current puck position in mm coordinates.
            velocity: Current puck velocity in mm/second.

        Returns:
            Prediction with interception point and trajectory waypoints.
            If puck is not approaching, returns is_approaching=False with
            interception_x set to current x position.
        """
        if velocity.vy <= 0:
            return Prediction(
                interception_x=self._clamp_x(position.x),
                interception_y=self._defence_y,
                time_to_intercept=0.0,
                trajectory_points=[],
                is_approaching=False,
            )

        x = position.x
        y = position.y
        vx = velocity.vx
        vy = velocity.vy
        total_time = 0.0
        trajectory: list[Position] = [
            Position(x=x, y=y, timestamp=position.timestamp)
        ]

        for _ in range(self._max_bounces + 1):
            candidates: list[tuple[float, str]] = []

            # Defence line
            if vy > 0:
                t_defence = (self._defence_y - y) / vy
                if t_defence > 1e-9:
                    candidates.append((t_defence, "defence"))

            # Top wall
            if vy < 0:
                t_top = -y / vy
                if t_top > 1e-9:
                    candidates.append((t_top, "top"))

            # Left wall
            if vx < 0:
                t_left = -x / vx
                if t_left > 1e-9:
                    candidates.append((t_left, "left"))

            # Right wall
            if vx > 0:
                t_right = (self._table_width - x) / vx
                if t_right > 1e-9:
                    candidates.append((t_right, "right"))

            if not candidates:
                break

            candidates.sort(key=lambda c: c[0])
            t_min, boundary = candidates[0]

            x = x + vx * t_min
            y = y + vy * t_min
            total_time += t_min

            x = self._clamp_x(x)
            y = max(0.0, min(self._table_height, y))

            trajectory.append(
                Position(
                    x=x,
                    y=y,
                    timestamp=position.timestamp + total_time,
                )
            )

            if boundary == "defence":
                return Prediction(
                    interception_x=self._clamp_x(x),
                    interception_y=y,
                    time_to_intercept=total_time,
                    trajectory_points=trajectory,
                    is_approaching=True,
                )
            elif boundary in ("left", "right"):
                vx = -vx
            elif boundary == "top":
                vy = -vy

        return Prediction(
            interception_x=self._clamp_x(x),
            interception_y=y,
            time_to_intercept=total_time,
            trajectory_points=trajectory,
            is_approaching=True,
        )

    def _clamp_x(self, x: float) -> float:
        """Clamp x to table bounds [0, table_width]."""
        return max(0.0, min(self._table_width, x))
