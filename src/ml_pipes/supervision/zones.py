from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from ml_pipes.operator import Operator


@Operator
class TriggerZone:
    def __init__(self, zone: sv.PolygonZone) -> None:
        self.zone = zone

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return detections[self.zone.trigger(detections)]


@Operator
class TrackingTimer:
    """Attach elapsed presence time to tracked detections.

    The input stream defines membership. For example, place it after a
    :class:`TriggerZone` to measure a track's time in that zone, or after any
    other filtering or merging stages to measure time in the resulting stream.

    When ``reset_missing_tracks`` is true, a track starts again at zero after
    it is absent from a frame. When false, a reappearing track retains its
    original entry time. Unconfirmed tracker IDs (negative values) always
    receive a duration of zero and do not retain timer state.
    """

    def __init__(
        self,
        fps: float,
        field: str = "tracking_time",
        reset_missing_tracks: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if not field:
            raise ValueError("field must not be empty.")

        self.fps = fps
        self.field = field
        self.reset_missing_tracks = reset_missing_tracks
        self.frame_id = 0
        self._entered_at: dict[int, int] = {}

    def reset(self) -> None:
        """Clear accumulated timing state for a new video stream."""
        self.frame_id = 0
        self._entered_at.clear()

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        """Return tracked detections with their elapsed presence time in seconds."""
        if detections.tracker_id is None:
            raise ValueError(
                "TrackingTimer requires detections with tracker_id values."
            )

        self.frame_id += 1
        tracker_ids = np.asarray(detections.tracker_id, dtype=np.int64)
        if self.reset_missing_tracks:
            active_ids = {
                int(tracker_id) for tracker_id in tracker_ids if tracker_id >= 0
            }
            self._entered_at = {
                tracker_id: entered_at
                for tracker_id, entered_at in self._entered_at.items()
                if tracker_id in active_ids
            }
        durations = np.asarray(
            [
                0.0
                if tracker_id < 0
                else (
                    self.frame_id
                    - self._entered_at.setdefault(int(tracker_id), self.frame_id)
                )
                / self.fps
                for tracker_id in tracker_ids
            ],
            dtype=np.float32,
        )
        data: dict[str, Any] = dict(detections.data)
        data[self.field] = durations
        return sv.Detections(
            xyxy=detections.xyxy.copy(),
            mask=detections.mask,
            confidence=None
            if detections.confidence is None
            else detections.confidence.copy(),
            class_id=None
            if detections.class_id is None
            else detections.class_id.copy(),
            tracker_id=detections.tracker_id.copy(),
            data=data,
            metadata=dict(detections.metadata),
        )


@Operator
class TriggerLineZone:
    def __init__(self, line_zone: sv.LineZone) -> None:
        self.line_zone = line_zone
        self.crossed_in = np.zeros((0,), dtype=bool)
        self.crossed_out = np.zeros((0,), dtype=bool)

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        self.crossed_in, self.crossed_out = self.line_zone.trigger(detections)
        return detections
