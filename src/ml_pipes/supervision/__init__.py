from __future__ import annotations

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
