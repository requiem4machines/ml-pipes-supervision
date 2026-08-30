from __future__ import annotations

from typing import Any

import numpy as np

from ml_pipes.inspection import (
    PipelineInspector,
    TextBlock,
    ndarray_image_formatter,
    register_value_formatter,
)

from .annotators import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    ComparisonAnnotator,
    CropAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    FPSAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    IconAnnotator,
    LabelAnnotator,
    LineZoneAnnotator,
    MaskAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    PixelateAnnotator,
    PolygonAnnotator,
    PolygonZoneAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
)
from .core import (
    DetectionsFromInference,
    DetectionsFromTensorRegistry,
    DetectionsFromUltralytics,
    DetectionsFilter,
    DetectionsNMS,
    DetectionsNMM,
    DetectionsSmoother,
    DetectionsStitch,
    ImageToArray,
)
from .views import FPSMonitor, ImageWindow, PlotImage
from .zones import TriggerLineZone, TriggerZone

# Supervision/OpenCV image arrays use BGR channel order.  Replace ml-pipes'
# RGB ndarray default so every inspector renders raw arrays correctly once this
# integration package has been imported.
# PipelineInspector registers its defaults lazily, so initialize them before
# replacing the ndarray formatter.
PipelineInspector()
register_value_formatter(
    np.ndarray,
    ndarray_image_formatter(default_color_space="BGR"),
    allow_override=True,
)


def _format_ultralytics_result(value: Any) -> list[TextBlock]:
    """Render an Ultralytics result without implying it produced an image."""
    prediction_parts = (value.boxes, value.masks, value.probs, value.keypoints, value.obb)
    summary = value.verbose().strip() if any(part is not None for part in prediction_parts) else "no predictions"
    return [TextBlock("Ultralytics Results", [
        ("path", str(value.path)),
        ("original shape", str(value.orig_shape)),
        ("detections", str(0 if value.boxes is None else len(value.boxes))),
        ("masks", str(0 if value.masks is None else len(value.masks))),
        ("keypoints", str(0 if value.keypoints is None else len(value.keypoints))),
        ("oriented boxes", str(0 if value.obb is None else len(value.obb))),
        ("classification", "yes" if value.probs is not None else "no"),
        ("summary", summary),
    ])]


try:
    from ultralytics.engine.results import Results as _UltralyticsResults
except ImportError:
    pass
else:
    register_value_formatter(_UltralyticsResults, _format_ultralytics_result)


class Detections:
    """Discovery namespace for operators that create or transform detections."""

    FromInference = DetectionsFromInference
    FromTensorRegistry = DetectionsFromTensorRegistry
    FromUltralytics = DetectionsFromUltralytics
    Filter = DetectionsFilter
    NMS = DetectionsNMS
    NMM = DetectionsNMM
    Stitch = DetectionsStitch


__all__ = [
    "BackgroundOverlayAnnotator",
    "BlurAnnotator",
    "BoxAnnotator",
    "BoxCornerAnnotator",
    "CircleAnnotator",
    "ColorAnnotator",
    "ComparisonAnnotator",
    "CropAnnotator",
    "Detections",
    "DetectionsSmoother",
    "DotAnnotator",
    "EllipseAnnotator",
    "FPSAnnotator",
    "FPSMonitor",
    "HaloAnnotator",
    "HeatMapAnnotator",
    "IconAnnotator",
    "ImageToArray",
    "ImageWindow",
    "LabelAnnotator",
    "LineZoneAnnotator",
    "MaskAnnotator",
    "OrientedBoxAnnotator",
    "PercentageBarAnnotator",
    "PixelateAnnotator",
    "PlotImage",
    "PolygonAnnotator",
    "PolygonZoneAnnotator",
    "RichLabelAnnotator",
    "RoundBoxAnnotator",
    "TraceAnnotator",
    "TriangleAnnotator",
    "TriggerLineZone",
    "TriggerZone",
]
