from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Position:
    """A 2D position with timestamp."""
    x: float
    y: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Velocity:
    """Velocity vector in units per second."""
    vx: float
    vy: float


@dataclass
class Detection:
    """Result of puck detection on a single frame."""
    found: bool
    position_px: Optional[Position] = None  # pixel coordinates
    contour_area: float = 0.0
    bounding_rect: Optional[tuple[int, int, int, int]] = None  # x, y, w, h


@dataclass
class TrackingState:
    """Current tracking state with position history."""
    current_position: Optional[Position] = None  # real-world mm coordinates
    velocity: Optional[Velocity] = None
    positions: list[Position] = field(default_factory=list)
    frames_since_detection: int = 0


@dataclass
class Prediction:
    """Predicted trajectory result."""
    interception_x: float          # mm, where puck will cross robot's defence line
    interception_y: float          # mm, should be close to robot's y position
    time_to_intercept: float       # seconds until puck reaches defence line
    trajectory_points: list[Position] = field(default_factory=list)  # for visualisation
    is_approaching: bool = False   # True if puck is moving towards robot's goal


@dataclass
class HSVRange:
    """HSV colour range for thresholding."""
    lower: tuple[int, int, int] = (35, 100, 100)
    upper: tuple[int, int, int] = (85, 255, 255)
