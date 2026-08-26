"""ml-pipes operators for the external :mod:`trackers` package."""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import supervision as sv

with warnings.catch_warnings():
    # The trackers package emits this warning during import.
    warnings.filterwarnings(
        "ignore",
        message="target=None is deprecated since `v0.8`; use `TargetMode.NOTIFY` instead. Will be removed in `v1.0`.",
        category=FutureWarning,
    )
    from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker, SORTTracker
    from trackers.core.base import BaseTracker
    from trackers.utils.state_representations import (
        BaseStateEstimator,
        XCYCWHStateEstimator,
        XCYCSRStateEstimator,
        XYXYStateEstimator,
    )

from ml_pipes.operator import Operator

__all__ = [
    "BoTSORT",
    "ByteTrack",
    "OCSORT",
    "ReadTrackedObjects",
    "SORT",
    "UpdateTrackedObjects",
]

CmcMethod = Literal["orb", "sift", "sparseOptFlow", "ecc"]


@Operator
class UpdateTrackedObjects:
    def __init__(self, tracker: BaseTracker) -> None:
        self.tracker = tracker

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update(detections)

    def reset(self) -> None:
        self.tracker.reset()

    @property
    def tracked_objects(self) -> sv.Detections:
        return self.tracker.tracked_objects

    def __call__(
        self,
        payload: sv.Detections | tuple[sv.Detections, npt.NDArray[np.uint8]],
    ) -> sv.Detections:
        detections = payload[0] if isinstance(payload, tuple) else payload
        return self.update(detections)


@Operator
class ByteTrack(UpdateTrackedObjects):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
    ) -> None:
        super().__init__(
            ByteTrackTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=track_activation_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                high_conf_det_threshold=high_conf_det_threshold,
                state_estimator_class=state_estimator_class,
            )
        )


@Operator
class BoTSORT(UpdateTrackedObjects):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        high_conf_det_threshold: float = 0.6,
        enable_cmc: bool = True,
        cmc_method: CmcMethod = "sparseOptFlow",
        cmc_downscale: int = 2,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        self.enable_cmc = enable_cmc
        super().__init__(
            BoTSORTTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=track_activation_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
                minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
                minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
                high_conf_det_threshold=high_conf_det_threshold,
                enable_cmc=enable_cmc,
                cmc_method=cmc_method,
                cmc_downscale=cmc_downscale,
                instant_first_frame_activation=instant_first_frame_activation,
                state_estimator_class=state_estimator_class,
            )
        )

    def update(
        self,
        detections: sv.Detections,
        frame: npt.NDArray[np.uint8] | None = None,
    ) -> sv.Detections:
        if self.enable_cmc and frame is None:
            raise ValueError("BoTSORT requires the current frame when enable_cmc=True.")
        return self.tracker.update(detections, frame)

    def __call__(
        self,
        payload: sv.Detections | tuple[sv.Detections, npt.NDArray[np.uint8]],
    ) -> sv.Detections:
        if isinstance(payload, tuple):
            detections, frame = payload
            return self.update(detections, frame)
        return self.update(payload)

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del validation_error_type
        if self.enable_cmc:
            return (tuple[sv.Detections, npt.NDArray[np.uint8]],), sv.Detections
        return (upstream_annotation,), sv.Detections


@Operator
class OCSORT(UpdateTrackedObjects):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        direction_consistency_weight: float = 0.2,
        high_conf_det_threshold: float = 0.6,
        delta_t: int = 3,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
    ) -> None:
        super().__init__(
            OCSORTTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                direction_consistency_weight=direction_consistency_weight,
                high_conf_det_threshold=high_conf_det_threshold,
                delta_t=delta_t,
                state_estimator_class=state_estimator_class,
            )
        )


@Operator
class SORT(UpdateTrackedObjects):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
    ) -> None:
        super().__init__(
            SORTTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=track_activation_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                state_estimator_class=state_estimator_class,
            )
        )


@Operator
class ReadTrackedObjects:
    def __init__(self, tracker: BaseTracker) -> None:
        self.tracker = tracker

    def __call__(self) -> sv.Detections:
        return self.tracker.tracked_objects
