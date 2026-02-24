import cv2
import numpy as np
import pytest
from src.models import HSVRange
from src.detector import PuckDetector


@pytest.fixture
def detector() -> PuckDetector:
    """Create a PuckDetector with default green HSV range."""
    return PuckDetector(hsv_range=HSVRange(), min_contour_area=100, blur_kernel=5)


def _make_black_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestPuckDetector:
    def test_detect_green_circle(self, detector: PuckDetector) -> None:
        """Detect a green circle drawn on a black background."""
        frame = _make_black_frame()
        cv2.circle(frame, (320, 240), 25, (0, 255, 0), -1)
        result = detector.detect(frame)
        assert result.found is True
        assert result.position_px is not None
        assert abs(result.position_px.x - 320) <= 2
        assert abs(result.position_px.y - 240) <= 2
        assert result.contour_area > 100

    def test_detect_no_puck(self, detector: PuckDetector) -> None:
        """No detection on a plain black frame."""
        frame = _make_black_frame()
        result = detector.detect(frame)
        assert result.found is False
        assert result.position_px is None

    def test_detect_ignores_small_noise(self) -> None:
        """Tiny green dot should be filtered out by min_contour_area."""
        detector = PuckDetector(
            hsv_range=HSVRange(), min_contour_area=100, blur_kernel=5
        )
        frame = _make_black_frame()
        cv2.circle(frame, (100, 100), 2, (0, 255, 0), -1)
        result = detector.detect(frame)
        assert result.found is False

    def test_detect_selects_largest(self, detector: PuckDetector) -> None:
        """When two green circles exist, detect the larger one."""
        frame = _make_black_frame()
        cv2.circle(frame, (100, 100), 10, (0, 255, 0), -1)
        cv2.circle(frame, (400, 300), 30, (0, 255, 0), -1)
        result = detector.detect(frame)
        assert result.found is True
        assert result.position_px is not None
        dist_to_large = (
            (result.position_px.x - 400) ** 2
            + (result.position_px.y - 300) ** 2
        ) ** 0.5
        dist_to_small = (
            (result.position_px.x - 100) ** 2
            + (result.position_px.y - 100) ** 2
        ) ** 0.5
        assert dist_to_large < dist_to_small

    def test_get_mask_returns_binary(self, detector: PuckDetector) -> None:
        """Mask should be single-channel uint8 with only 0 or 255 values."""
        frame = _make_black_frame()
        cv2.circle(frame, (320, 240), 25, (0, 255, 0), -1)
        mask = detector.get_mask(frame)
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, 255})
