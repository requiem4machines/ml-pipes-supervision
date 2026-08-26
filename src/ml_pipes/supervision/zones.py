from __future__ import annotations

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
class TriggerLineZone:
    def __init__(self, line_zone: sv.LineZone) -> None:
        self.line_zone = line_zone
        self.crossed_in = np.zeros((0,), dtype=bool)
        self.crossed_out = np.zeros((0,), dtype=bool)

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        self.crossed_in, self.crossed_out = self.line_zone.trigger(detections)
        return detections
