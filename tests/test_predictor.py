import random
import pytest
from src.models import Position, Velocity
from src.predictor import TrajectoryPredictor


@pytest.fixture
def predictor() -> TrajectoryPredictor:
    """Create a TrajectoryPredictor with default settings."""
    return TrajectoryPredictor(
        table_width=610.0,
        table_height=1220.0,
        defence_y=1100.0,
        max_bounces=10,
    )


class TestTrajectoryPredictor:
    def test_straight_down(self, predictor: TrajectoryPredictor) -> None:
        """Puck moving straight down should intercept at same x."""
        pos = Position(x=305.0, y=0.0, timestamp=0.0)
        vel = Velocity(vx=0.0, vy=1000.0)
        result = predictor.predict(pos, vel)
        assert result.is_approaching is True
        assert abs(result.interception_x - 305.0) < 1.0
        assert abs(result.time_to_intercept - 1.1) < 0.05

    def test_moving_away(self, predictor: TrajectoryPredictor) -> None:
        """Puck moving upward should not be approaching."""
        pos = Position(x=305.0, y=600.0, timestamp=0.0)
        vel = Velocity(vx=0.0, vy=-500.0)
        result = predictor.predict(pos, vel)
        assert result.is_approaching is False

    def test_single_wall_bounce(self, predictor: TrajectoryPredictor) -> None:
        """Puck near right wall with rightward velocity should bounce."""
        pos = Position(x=500.0, y=0.0, timestamp=0.0)
        vel = Velocity(vx=500.0, vy=1000.0)
        result = predictor.predict(pos, vel)
        assert result.is_approaching is True
        assert 0 <= result.interception_x <= 610.0

    def test_stationary_puck(self, predictor: TrajectoryPredictor) -> None:
        """Stationary puck should not be approaching."""
        pos = Position(x=305.0, y=600.0, timestamp=0.0)
        vel = Velocity(vx=0.0, vy=0.0)
        result = predictor.predict(pos, vel)
        assert result.is_approaching is False

    def test_interception_within_bounds(
        self, predictor: TrajectoryPredictor
    ) -> None:
        """Random trajectories should always intercept within table bounds."""
        random.seed(42)
        for _ in range(100):
            x = random.uniform(0, 610)
            y = random.uniform(0, 600)
            vx = random.uniform(-1000, 1000)
            vy = random.uniform(100, 2000)
            pos = Position(x=x, y=y, timestamp=0.0)
            vel = Velocity(vx=vx, vy=vy)
            result = predictor.predict(pos, vel)
            if result.is_approaching:
                assert 0 <= result.interception_x <= 610.0
