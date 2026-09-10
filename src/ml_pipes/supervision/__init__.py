from __future__ import annotations

import numpy as np

from ml_pipes.inspection import (
    ndarray_image_formatter,
)
from ml_pipes.inspection._global_registry import register_value_formatter

from .annotators import (
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    ComparisonAnnotator,
    CustomLabelAnnotator,
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
    Detection,
    ImageToArray,
)
from .views import FPSMonitor, ImageWindow, PlotImage
from .zones import TrackingTimer, TriggerLineZone, TriggerZone

# Supervision/OpenCV image arrays use BGR channel order.  Replace ml-pipes'
# RGB ndarray default so every inspector renders raw arrays correctly once this
# integration package has been imported.
register_value_formatter(
    np.ndarray,
    ndarray_image_formatter(default_color_space="BGR"),
    override=True,
)


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
    "CustomLabelAnnotator",
    "Detection",
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
    "TrackingTimer",
    "TriangleAnnotator",
    "TriggerLineZone",
    "TriggerZone",
]
