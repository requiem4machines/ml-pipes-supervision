"""Blur faces in MediaPipe's sample image with local MediaPipe detection.

Run from the repository root:
    python examples/run_blur_faces.py
    python examples/run_blur_faces.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import supervision as sv
from ml_pipes.inspection import PipelineInspector

from common import ASSETS_DIR
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Recall, Store
from ml_pipes.supervision import BoxAnnotator, BlurAnnotator, ImageToArray, PlotImage
from ml_pipes.vision import Decode, LoadFile

DEFAULT_IMAGE_NAME = "mediapipe-faces.jpg"
DEFAULT_IMAGE_URL = "https://i.imgur.com/Vu2Nqwb.jpeg"
FACE_CLASS_ID = 0
FACE_CLASS_NAME = "face"


def download_image_if_missing(url: str, destination: Path) -> Path:
    """Download the sample image with curl, avoiding Imgur's urllib rate limit."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    try:
        subprocess.run(
            ["curl", "-L", "-sS", "--fail", url, "-o", str(temporary_path)],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Could not download sample image from {url}.") from error
    temporary_path.replace(destination)
    return destination


@Operator
class MediaPipeFaceDetection:
    """Detect faces in a BGR image with MediaPipe's local face detector."""

    def __init__(
        self,
        model_selection: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        if model_selection not in (0, 1):
            raise ValueError("model_selection must be 0 (short range) or 1 (full range).")
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1.")

        self._detector: Any = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def __call__(self, frame: npt.NDArray[np.uint8]) -> sv.Detections:
        """Return face detections for a BGR HWC uint8 image."""
        result = self._detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = frame.shape[:2]
        boxes: list[list[float]] = []
        confidences: list[float] = []
        for detection in result.detections or []:
            box = detection.location_data.relative_bounding_box
            boxes.append(
                [
                    max(0, box.xmin * width),
                    max(0, box.ymin * height),
                    min(width, (box.xmin + box.width) * width),
                    min(height, (box.ymin + box.height) * height),
                ]
            )
            confidences.append(detection.score[0])
        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(confidences, dtype=np.float32),
            class_id=np.full(len(boxes), FACE_CLASS_ID, dtype=np.int32),
            data={
                "class_name": np.full(len(boxes), FACE_CLASS_NAME, dtype=str),
            },
        )

    def close(self) -> None:
        """Release MediaPipe resources when the operator is no longer needed."""
        self._detector.close()


def build_pipeline() -> Pipeline:
    """Build a file-to-image pipeline that detects and blurs faces locally."""
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_frame"),
            MediaPipeFaceDetection(model_selection=1),
            Recall("source_frame", prepend=True),
            BoxAnnotator(),
            BlurAnnotator(kernel_size=100),
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
        "--input",
        type=Path,
        default=None,
        help="Input image path. Defaults to MediaPipe's sample image.",
    )
    args = parser.parse_args()

    image_path = args.input or download_image_if_missing(
        DEFAULT_IMAGE_URL,
        ASSETS_DIR / DEFAULT_IMAGE_NAME,
    )

    pipeline = build_pipeline()
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
