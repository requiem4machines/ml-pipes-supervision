"""
RF-DETR instance segmentation annotated through Roboflow Inference and Supervision.

Run from the repo root:
    python examples/run_segment_and_annotate.py
    python examples/run_segment_and_annotate.py --input path/to/photo.jpg
    python examples/run_segment_and_annotate.py --model-id rfdetr-seg-medium
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import numpy.typing as npt
import supervision as sv
from supervision.assets import ImageAssets, download_assets

from common import ASSETS_DIR
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision import (
    Detections,
    ImageToArray,
    LabelAnnotator,
    MaskAnnotator,
    PlotImage,
)

from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store
from ml_pipes.vision import Decode, LoadFile

DEFAULT_MODEL_ID = "rfdetr-seg-small"


def build_pipeline(
    model_id: str,
    api_key: str | None,
) -> Pipeline[str | Path, tuple[npt.NDArray[np.uint8], sv.Detections]]:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            ImageToArray(),
            Store("source_image"),
            RoboflowInference(model_id=model_id, api_key=api_key),
            Select(0),
            Detections.FromInference(),
            Recall("source_image", prepend=True),
            MaskAnnotator(),
            LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS),
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
        help="Roboflow Inference model id. Defaults to the RF-DETR segmentation small pretrained alias.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Only needed for private or account-scoped model ids.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image path. Defaults to Supervision's PEOPLE_WALKING image asset.",
    )
    args = parser.parse_args()

    image_path = args.input or Path(
        download_assets(ImageAssets.PEOPLE_WALKING, directory=ASSETS_DIR)
    )

    pipeline = build_pipeline(args.model_id, args.api_key)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
