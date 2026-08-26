"""
Oriented bounding-box detection through Ultralytics and Supervision.

Port of Supervision's "Oriented Bounding Boxes" notebook. This example uses
YOLO11-OBB, trained on DOTA aerial classes, to detect ships in the marina image.
It requires Ultralytics:

    python -m pip install ultralytics

Run from the repo root:
    python examples/run_oriented_bounding_boxes.py
    python examples/run_oriented_bounding_boxes.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv

from common import ASSETS_DIR, resolve_input_path
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Recall, Store
from ml_pipes.supervision import (
    Detections,
    ImageToArray,
    OrientedBoxAnnotator,
    PlotImage,
)
from ml_pipes.vision import Decode, LoadFile

DEFAULT_MODEL_ID = "yolo11n-obb.pt"
DEFAULT_IMAGE_NAME = "boats.jpg"
DEFAULT_IMAGE_URL = "https://ultralytics.com/images/boats.jpg"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_CLASS_NAME = "ship"


@Operator
class UltralyticsInference:
    def __init__(
        self,
        model_id: str,
        image_size: int,
        *,
        nms_iou_threshold: float | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            print(
                "Ultralytics is required for oriented bounding-box detection. "
                "Install it with: python -m pip install ultralytics",
                file=sys.stderr,
            )
            raise SystemExit(1) from None

        self.model_id = model_id
        self.image_size = image_size
        self.nms_iou_threshold = nms_iou_threshold
        self.model = YOLO(model_id)

    def __call__(self, image: npt.NDArray[np.uint8]) -> Any:
        kwargs: dict[str, Any] = {"imgsz": self.image_size, "verbose": False}
        if self.nms_iou_threshold is not None:
            kwargs["iou"] = self.nms_iou_threshold
        return self.model(image, **kwargs)[0]


def build_pipeline(
    model_id: str,
    image_size: int,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    """Build the final OBB pipeline with OBB-aware NMS and annotation."""
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            UltralyticsInference(model_id=model_id, image_size=image_size, nms_iou_threshold=0.9),
            Detections.FromUltralytics(),
            Detections.Filter(
                lambda detections: detections.data["class_name"] == DEFAULT_CLASS_NAME
            ),
            Detections.NMS(threshold=0.3),
            Recall("source_image", prepend=True),
            OrientedBoxAnnotator(color=sv.Color.GREEN, thickness=2),
            PlotImage(at=0),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Ultralytics OBB model id. Defaults to yolo11n-obb.pt.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path. Defaults to the notebook's boats.jpg source.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Ultralytics inference image size. Defaults to 1024.",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args.model_id, args.imgsz)
    input_path = resolve_input_path(
        args.input,
        ASSETS_DIR / DEFAULT_IMAGE_NAME,
        DEFAULT_IMAGE_URL,
    )
    pipeline.validate()
    pipeline.describe()
    pipeline(input_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
