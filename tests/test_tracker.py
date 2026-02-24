import pytest
from src.models import Position, Velocity
from src.tracker import PuckTracker


class TestPuckTracker:
    def test_update_single_position(self) -> None:
        """Single update should set current_position, no velocity yet."""
        tracker = PuckTracker()
        pos = Position(x=100.0, y=200.0, timestamp=1.0)
        state = tracker.update(pos)
        assert state.current_position is not None
        assert state.current_position.x == 100.0
        assert state.current_position.y == 200.0
        assert state.velocity is None
        assert state.frames_since_detection == 0

    def test_velocity_estimation(self) -> None:
        """Constant rightward motion should give ~100 mm/s vx, ~0 vy."""
        tracker = PuckTracker(history_length=10, smoothing_window=5)
        positions = [
            Position(x=0.0, y=0.0, timestamp=0.0),
            Position(x=10.0, y=0.0, timestamp=0.1),
            Position(x=20.0, y=0.0, timestamp=0.2),
            Position(x=30.0, y=0.0, timestamp=0.3),
            Position(x=40.0, y=0.0, timestamp=0.4),
        ]
        for pos in positions:
            state = tracker.update(pos)
        assert state.velocity is not None
        assert abs(state.velocity.vx - 100.0) < 5.0
        assert abs(state.velocity.vy) < 1.0

    def test_missed_frames_increment(self) -> None:
        """Three consecutive misses should increment frames_since_detection to 3."""
        tracker = PuckTracker()
        tracker.update(Position(x=100.0, y=200.0, timestamp=1.0))
        tracker.update(None)
        tracker.update(None)
        state = tracker.update(None)
        assert state.frames_since_detection == 3

    def test_reset_after_max_missed(self) -> None:
        """After exceeding max_missed_frames, tracker should reset."""
        tracker = PuckTracker(max_missed_frames=3)
        tracker.update(Position(x=100.0, y=200.0, timestamp=1.0))
        for _ in range(4):
            state = tracker.update(None)
        assert state.current_position is None
        assert len(state.positions) == 0

    def test_history_length_limit(self) -> None:
        """History should not exceed history_length."""
        tracker = PuckTracker(history_length=5)
        for i in range(10):
            state = tracker.update(
                Position(x=float(i), y=0.0, timestamp=float(i) * 0.1)
            )
        assert len(state.positions) == 5
