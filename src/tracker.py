import logging
import numpy as np
from src.models import Position, Velocity, TrackingState

logger = logging.getLogger(__name__)


class PuckTracker:
    """Tracks puck position over time and estimates velocity.

    Maintains a rolling history of the last N positions.
    Estimates velocity using linear regression over the smoothing window.
    Handles missed detections by incrementing frames_since_detection.
    After max_missed_frames consecutive misses, resets tracking state.
    """

    def __init__(
        self,
        history_length: int = 10,
        smoothing_window: int = 5,
        max_missed_frames: int = 15,
    ):
        """Initialise the tracker.

        Args:
            history_length: Maximum number of positions to store in history.
            smoothing_window: Number of recent positions used for velocity
                              estimation. Must be >= 2. Uses the last N
                              positions from history.
            max_missed_frames: After this many consecutive frames with no
                               detection, reset the tracking state entirely.
        """
        self._history_length = history_length
        self._smoothing_window = smoothing_window
        self._max_missed_frames = max_missed_frames
        self._positions: list[Position] = []
        self._current_position: Position | None = None
        self._velocity: Velocity | None = None
        self._frames_since_detection: int = 0

    def update(self, position: Position | None) -> TrackingState:
        """Update the tracker with a new position (or None if not detected).

        If position is not None:
            - Append to history (drop oldest if at capacity)
            - Reset frames_since_detection to 0
            - Estimate velocity if enough history exists

        If position is None:
            - Increment frames_since_detection
            - If exceeded max_missed_frames, call reset()
            - Velocity remains as last known estimate

        Velocity estimation method:
            Take the last smoothing_window positions from history.
            For each consecutive pair, compute dx/dt and dy/dt where dt is
            the timestamp difference. Average all dx/dt and dy/dt values
            to get the velocity vector. Units: mm/second.

        Returns:
            Current TrackingState.
        """
        if position is not None:
            self._positions.append(position)
            if len(self._positions) > self._history_length:
                self._positions = self._positions[-self._history_length:]
            self._current_position = position
            self._frames_since_detection = 0
            self._estimate_velocity()
        else:
            self._frames_since_detection += 1
            if self._frames_since_detection > self._max_missed_frames:
                self.reset()

        return self.state

    def _estimate_velocity(self) -> None:
        """Estimate velocity from recent position history."""
        if len(self._positions) < 2:
            self._velocity = None
            return

        window = self._positions[-self._smoothing_window:]
        if len(window) < 2:
            self._velocity = None
            return

        vx_values: list[float] = []
        vy_values: list[float] = []

        for i in range(1, len(window)):
            dt = window[i].timestamp - window[i - 1].timestamp
            if dt <= 0:
                continue
            dx = window[i].x - window[i - 1].x
            dy = window[i].y - window[i - 1].y
            vx_values.append(dx / dt)
            vy_values.append(dy / dt)

        if vx_values:
            avg_vx = sum(vx_values) / len(vx_values)
            avg_vy = sum(vy_values) / len(vy_values)
            self._velocity = Velocity(vx=avg_vx, vy=avg_vy)
        else:
            self._velocity = None

    def reset(self) -> None:
        """Clear all history and reset state."""
        self._positions.clear()
        self._current_position = None
        self._velocity = None
        self._frames_since_detection = 0

    @property
    def state(self) -> TrackingState:
        """Return the current tracking state."""
        return TrackingState(
            current_position=self._current_position,
            velocity=self._velocity,
            positions=list(self._positions),
            frames_since_detection=self._frames_since_detection,
        )

    @property
    def has_velocity(self) -> bool:
        """Return True if a velocity estimate is available (need >= 2 positions)."""
        return self._velocity is not None
