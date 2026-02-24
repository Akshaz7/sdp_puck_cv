import numpy as np
import pytest
from src.models import Position
from src.homography import HomographyMapper


@pytest.fixture
def identity_mapper() -> HomographyMapper:
    """Create a mapper where pixel coords equal mm coords (identity)."""
    corners = np.array(
        [[0, 0], [610, 0], [610, 1220], [0, 1220]], dtype=np.float32
    )
    return HomographyMapper(
        pixel_corners=corners,
        table_width_mm=610.0,
        table_height_mm=1220.0,
    )


class TestHomographyMapper:
    def test_identity_mapping(self, identity_mapper: HomographyMapper) -> None:
        """With identity corners, pixel (305, 610) should map to mm (305, 610)."""
        pos_px = Position(x=305.0, y=610.0, timestamp=1.0)
        pos_mm = identity_mapper.pixel_to_mm(pos_px)
        assert abs(pos_mm.x - 305.0) < 1.0
        assert abs(pos_mm.y - 610.0) < 1.0

    def test_corner_mapping(self, identity_mapper: HomographyMapper) -> None:
        """Each pixel corner should map to corresponding real-world corner."""
        test_cases = [
            (Position(x=0.0, y=0.0), (0.0, 0.0)),
            (Position(x=610.0, y=0.0), (610.0, 0.0)),
            (Position(x=610.0, y=1220.0), (610.0, 1220.0)),
            (Position(x=0.0, y=1220.0), (0.0, 1220.0)),
        ]
        for pos_px, (expected_x, expected_y) in test_cases:
            pos_mm = identity_mapper.pixel_to_mm(pos_px)
            assert abs(pos_mm.x - expected_x) < 1.0
            assert abs(pos_mm.y - expected_y) < 1.0

    def test_round_trip(self, identity_mapper: HomographyMapper) -> None:
        """Pixel -> mm -> pixel should return original within 1px."""
        pos_px = Position(x=200.0, y=400.0, timestamp=1.0)
        pos_mm = identity_mapper.pixel_to_mm(pos_px)
        pos_back = identity_mapper.mm_to_pixel(pos_mm)
        assert abs(pos_back.x - pos_px.x) < 1.0
        assert abs(pos_back.y - pos_px.y) < 1.0
