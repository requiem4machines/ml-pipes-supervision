import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import TrackingTimer


def tracked_detections(*tracker_ids: int) -> sv.Detections:
    count = len(tracker_ids)
    return sv.Detections(
        xyxy=np.zeros((count, 4), dtype=np.float32),
        tracker_id=np.asarray(tracker_ids, dtype=np.int32),
    )


def test_tracking_timer_tracks_continuous_dwell_time() -> None:
    timer = TrackingTimer(fps=10)

    first = timer(tracked_detections(7))
    second = timer(tracked_detections(7))
    timer(tracked_detections())
    reentered = timer(tracked_detections(7))

    np.testing.assert_allclose(first.data["tracking_time"], [0.0])
    np.testing.assert_allclose(second.data["tracking_time"], [0.1])
    np.testing.assert_allclose(reentered.data["tracking_time"], [0.0])


def test_tracking_timer_requires_tracking_ids() -> None:
    timer = TrackingTimer(fps=30)
    detections = sv.Detections(xyxy=np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="tracker_id"):
        timer(detections)


def test_tracking_timer_can_retain_time_across_missing_frames() -> None:
    timer = TrackingTimer(fps=10, reset_missing_tracks=False)

    timer(tracked_detections(7))
    timer(tracked_detections())
    reentered = timer(tracked_detections(7))

    np.testing.assert_allclose(reentered.data["tracking_time"], [0.2])


def test_tracking_timer_assigns_zero_to_unconfirmed_tracks() -> None:
    timer = TrackingTimer(fps=10)

    unconfirmed = timer(tracked_detections(-1))
    confirmed = timer(tracked_detections(7))

    np.testing.assert_allclose(unconfirmed.data["tracking_time"], [0.0])
    np.testing.assert_allclose(confirmed.data["tracking_time"], [0.0])


@pytest.mark.parametrize("fps", [0, -1])
def test_tracking_timer_requires_positive_frame_rate(fps: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        TrackingTimer(fps=fps)
