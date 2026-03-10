import cv2
import logging
import numpy as np
from typing import Optional
from src.models import Detection, TrackingState, Prediction, Position

logger = logging.getLogger(__name__)


class Visualiser:
    """Draws debug overlays on camera frames.

    All methods modify the frame in-place and return it for chaining.
    """

    def __init__(
        self,
        show_fps: bool = True,
        show_trajectory: bool = True,
        show_table_boundary: bool = True,
        show_bounce_markers: bool = True,
        show_prediction_info: bool = True,
        trajectory_persistence_frames: int = 15,
        table_width_mm: float = 610.0,
    ):
        """Initialise the visualiser.

        Args:
            show_fps: Whether to draw FPS counter.
            show_trajectory: Whether to draw predicted trajectory lines.
            show_table_boundary: Whether to draw table boundary overlay.
            show_bounce_markers: Whether to draw diamond markers at bounce points.
            show_prediction_info: Whether to draw prediction info text overlay.
            trajectory_persistence_frames: Number of frames to keep showing the
                last prediction after detection is lost.
            table_width_mm: Table width in mm, used for defence line drawing.
        """
        self._show_fps = show_fps
        self._show_trajectory = show_trajectory
        self._show_table_boundary = show_table_boundary
        self._show_bounce_markers = show_bounce_markers
        self._show_prediction_info = show_prediction_info
        self._persistence_frames = trajectory_persistence_frames
        self._table_width_mm = table_width_mm
        self._cached_prediction: Optional[Prediction] = None
        self._frames_since_prediction: int = 0
        self._stability_threshold_mm: float = 20.0

    def update_prediction_cache(
        self, prediction: Optional[Prediction]
    ) -> Optional[Prediction]:
        """Cache the latest prediction and return the best available one.

        Only updates the cached prediction if the interception point has
        moved by more than the stability threshold, preventing jitter.

        If None is provided, increments the counter and returns the cached
        prediction until persistence_frames is exceeded.

        Args:
            prediction: The current frame's prediction, or None if unavailable.

        Returns:
            The prediction to use for drawing (may be cached), or None.
        """
        if prediction is not None:
            if self._cached_prediction is not None:
                dx = abs(
                    prediction.interception_x
                    - self._cached_prediction.interception_x
                )
                dy = abs(
                    prediction.interception_y
                    - self._cached_prediction.interception_y
                )
                approaching_changed = (
                    prediction.is_approaching
                    != self._cached_prediction.is_approaching
                )
                if (
                    dx < self._stability_threshold_mm
                    and dy < self._stability_threshold_mm
                    and not approaching_changed
                ):
                    self._frames_since_prediction = 0
                    return self._cached_prediction
            self._cached_prediction = prediction
            self._frames_since_prediction = 0
            return self._cached_prediction

        self._frames_since_prediction += 1
        if (
            self._cached_prediction is not None
            and self._frames_since_prediction <= self._persistence_frames
        ):
            return self._cached_prediction

        self._cached_prediction = None
        return None

    def draw_detection(
        self, frame: np.ndarray, detection: Detection
    ) -> np.ndarray:
        """Draw detection result on frame.

        If detected:
            - Green circle (radius 10) at centroid
            - Green bounding rectangle around contour
            - White text "PUCK (x, y)" above bounding rect

        If not detected:
            - Red text "NO PUCK" in top-left corner

        Args:
            frame: BGR image to draw on. Modified in-place.
            detection: Detection result.

        Returns:
            The same frame (modified in-place).
        """
        if detection.found and detection.position_px is not None:
            cx = int(detection.position_px.x)
            cy = int(detection.position_px.y)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            if detection.bounding_rect is not None:
                x, y, w, h = detection.bounding_rect
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                text = f"PUCK ({cx}, {cy})"
                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        else:
            cv2.putText(
                frame,
                "NO PUCK",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame

    def draw_table_boundary(
        self,
        frame: np.ndarray,
        corner_pixels: np.ndarray,
        defence_line_px: Optional[list[list[int]]] = None,
    ) -> np.ndarray:
        """Draw table boundary overlay on frame.

        Draws a cyan polygon outline of the table edges and a red dashed
        defence line defined by two pixel coordinates.

        Args:
            frame: BGR image.
            corner_pixels: Array of shape (4, 2) with table corner pixel coords.
            defence_line_px: Two pixel points [[x1,y1],[x2,y2]] defining the
                defence line. If None, defence line is not drawn.

        Returns:
            The same frame.
        """
        if not self._show_table_boundary:
            return frame

        # Draw cyan table boundary polygon
        pts = corner_pixels.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2
        )

        # Draw red dashed defence line from pixel coordinates
        if defence_line_px is not None and len(defence_line_px) == 2:
            p1 = (int(defence_line_px[0][0]), int(defence_line_px[0][1]))
            p2 = (int(defence_line_px[1][0]), int(defence_line_px[1][1]))

            self._draw_dashed_line(
                frame, p1, p2, colour=(0, 0, 255), thickness=2,
                dash_len=15, gap_len=10,
            )

            # "ROBOT ZONE" label offset from the defence line midpoint
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            # Offset label to the right/below the line
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Perpendicular offset (towards robot side)
            perp_x = -dy
            perp_y = dx
            length = max(1, int(np.sqrt(perp_x ** 2 + perp_y ** 2)))
            offset = 25
            label_x = mid_x + int(perp_x * offset / length) - 55
            label_y = mid_y + int(perp_y * offset / length) + 5
            cv2.putText(
                frame,
                "ROBOT ZONE",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return frame

    @staticmethod
    def _draw_dashed_line(
        frame: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        colour: tuple[int, int, int],
        thickness: int = 1,
        dash_len: int = 10,
        gap_len: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        line_len = max(1, int(np.sqrt(dx * dx + dy * dy)))
        for i in range(0, line_len, dash_len + gap_len):
            start_frac = i / line_len
            end_frac = min((i + dash_len) / line_len, 1.0)
            s = (
                int(p1[0] + dx * start_frac),
                int(p1[1] + dy * start_frac),
            )
            e = (
                int(p1[0] + dx * end_frac),
                int(p1[1] + dy * end_frac),
            )
            cv2.line(frame, s, e, colour, thickness)

    def draw_trajectory(
        self,
        frame: np.ndarray,
        prediction: Prediction,
        mm_to_pixel_func: callable,
    ) -> np.ndarray:
        """Draw predicted trajectory on frame.

        First segment (before first bounce): bold green solid line.
        Subsequent segments (after first bounce): faint yellow dashed line.
        Draws a diamond marker at the first bounce point.
        Draws a red circle at the interception point.

        Args:
            frame: BGR image.
            prediction: Trajectory prediction.
            mm_to_pixel_func: Function that converts Position (mm) to
                              Position (px).

        Returns:
            The same frame.
        """
        if not self._show_trajectory:
            return frame

        if not prediction.trajectory_points:
            return frame

        pixel_points: list[tuple[int, int]] = []
        for pt in prediction.trajectory_points:
            px = mm_to_pixel_func(pt)
            pixel_points.append((int(px.x), int(px.y)))

        # First segment (start to first bounce): bold bright green solid
        if len(pixel_points) >= 2:
            cv2.line(
                frame, pixel_points[0], pixel_points[1],
                (0, 255, 0), 5,
            )

        # Subsequent segments: faint yellow dashed
        if len(pixel_points) > 2:
            for i in range(2, len(pixel_points)):
                self._draw_dashed_line(
                    frame, pixel_points[i - 1], pixel_points[i],
                    colour=(0, 200, 255), thickness=2,
                )

        # Draw bounce marker at first bounce point (index 1, if there are
        # more points after it meaning it's a bounce not the final interception)
        if self._show_bounce_markers and len(pixel_points) > 2:
            px, py = pixel_points[1]
            diamond_size = 7
            diamond_pts = np.array([
                [px, py - diamond_size],
                [px + diamond_size, py],
                [px, py + diamond_size],
                [px - diamond_size, py],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [diamond_pts], (255, 0, 255))
            cv2.polylines(
                frame, [diamond_pts], isClosed=True,
                color=(255, 255, 255), thickness=1,
            )

        # Draw red circle at interception point
        intercept_mm = Position(
            x=prediction.interception_x, y=prediction.interception_y
        )
        intercept_px = mm_to_pixel_func(intercept_mm)
        cv2.circle(
            frame,
            (int(intercept_px.x), int(intercept_px.y)),
            12,
            (0, 0, 255),
            -1,
        )

        text = f"T: {prediction.time_to_intercept:.2f}s"
        cv2.putText(
            frame,
            text,
            (int(intercept_px.x) + 10, int(intercept_px.y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        return frame

    def draw_prediction_info(
        self, frame: np.ndarray, prediction: Prediction
    ) -> np.ndarray:
        """Draw prediction info overlay in the bottom-left corner.

        Shows bounce count, time to intercept, and approach status.

        Args:
            frame: BGR image.
            prediction: Trajectory prediction.

        Returns:
            The same frame.
        """
        if not self._show_prediction_info:
            return frame

        h = frame.shape[0]
        x0 = 10
        y0 = h - 70

        # Background rectangle
        cv2.rectangle(frame, (x0 - 5, y0 - 15), (x0 + 230, y0 + 55), (0, 0, 0), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1

        if prediction.is_approaching:
            status_text = "APPROACHING"
            status_colour = (0, 255, 0)
        else:
            status_text = "MOVING AWAY"
            status_colour = (0, 165, 255)

        cv2.putText(
            frame, f"Status: {status_text}",
            (x0, y0), font, scale, status_colour, thickness,
        )
        cv2.putText(
            frame, f"Bounces: {prediction.bounce_count}",
            (x0, y0 + 18), font, scale, (255, 255, 255), thickness,
        )
        cv2.putText(
            frame, f"Intercept T: {prediction.time_to_intercept:.3f}s",
            (x0, y0 + 36), font, scale, (255, 255, 255), thickness,
        )

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter in top-right corner.

        White text on black background rectangle for readability.
        Format: "FPS: {fps:.1f}"

        Args:
            frame: BGR image.
            fps: Current frames per second.

        Returns:
            The same frame.
        """
        if not self._show_fps:
            return frame

        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )

        h, w = frame.shape[:2]
        x = w - text_w - 10
        y = 25

        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame, text, (x, y), font, scale, (255, 255, 255), thickness
        )

        return frame

    def draw_tracking_state(
        self,
        frame: np.ndarray,
        state: TrackingState,
        mm_to_pixel_func: callable,
    ) -> np.ndarray:
        """Draw position history as fading dots.

        Draws small circles at each historical position. Older positions
        are more transparent (use decreasing brightness from white to grey).

        If velocity is available, draw a short blue arrow from current
        position in the velocity direction.

        Args:
            frame: BGR image.
            state: Current tracking state.
            mm_to_pixel_func: Coordinate conversion function.

        Returns:
            The same frame.
        """
        if not state.positions:
            return frame

        n = len(state.positions)
        for i, pos in enumerate(state.positions):
            px = mm_to_pixel_func(pos)
            brightness = int(100 + (155 * i / max(n - 1, 1)))
            colour = (brightness, brightness, brightness)
            cv2.circle(frame, (int(px.x), int(px.y)), 4, colour, -1)

        if (
            state.velocity is not None
            and state.current_position is not None
        ):
            current_px = mm_to_pixel_func(state.current_position)
            arrow_scale = 0.05
            end_mm = Position(
                x=state.current_position.x
                + state.velocity.vx * arrow_scale,
                y=state.current_position.y
                + state.velocity.vy * arrow_scale,
            )
            end_px = mm_to_pixel_func(end_mm)
            cv2.arrowedLine(
                frame,
                (int(current_px.x), int(current_px.y)),
                (int(end_px.x), int(end_px.y)),
                (255, 0, 0),
                2,
                tipLength=0.3,
            )

        return frame
