import cv2
import logging
import numpy as np
from src.models import Detection, Position, HSVRange

logger = logging.getLogger(__name__)


class PuckDetector:
    """Detects a coloured puck in a BGR frame using HSV thresholding.

    Pipeline:
        1. Apply Gaussian blur (kernel size from config, must be odd)
        2. Convert BGR to HSV
        3. Apply inRange threshold using HSV bounds
        4. Apply morphological opening (3x3 kernel) to remove noise
        5. Apply morphological closing (3x3 kernel) to fill gaps
        6. Find contours
        7. Filter contours by minimum area
        8. Select the largest valid contour
        9. Compute centroid using cv2.moments
        10. Compute bounding rect using cv2.boundingRect
    """

    def __init__(
        self,
        hsv_range: HSVRange,
        min_contour_area: int = 100,
        blur_kernel: int = 5,
    ):
        """Initialise detector.

        Args:
            hsv_range: HSV colour range for the puck.
            min_contour_area: Minimum contour area in pixels to count as a detection.
            blur_kernel: Gaussian blur kernel size. Must be odd.
        """
        self._hsv_range = hsv_range
        self._min_contour_area = min_contour_area
        self._blur_kernel = blur_kernel
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )

    def detect(self, frame: np.ndarray) -> Detection:
        """Detect the puck in a single frame.

        Args:
            frame: BGR image as numpy array, shape (H, W, 3), dtype uint8.

        Returns:
            Detection result with pixel coordinates if found.
        """
        mask = self.get_mask(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) >= self._min_contour_area
        ]

        if not valid_contours:
            return Detection(found=False)

        largest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return Detection(found=False)

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        x, y, w, h = cv2.boundingRect(largest)

        return Detection(
            found=True,
            position_px=Position(x=cx, y=cy),
            contour_area=area,
            bounding_rect=(x, y, w, h),
        )

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the binary threshold mask for a frame. Useful for debugging.

        Args:
            frame: BGR image.

        Returns:
            Binary mask as numpy array, shape (H, W), dtype uint8, values 0 or 255.
        """
        blurred = cv2.GaussianBlur(
            frame, (self._blur_kernel, self._blur_kernel), 0
        )
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower = np.array(self._hsv_range.lower, dtype=np.uint8)
        upper = np.array(self._hsv_range.upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        return mask

    def update_hsv_range(self, hsv_range: HSVRange) -> None:
        """Update the HSV range at runtime (used by calibration tool)."""
        self._hsv_range = hsv_range
