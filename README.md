![python-version](https://img.shields.io/pypi/pyversions/supervision)
[![Supervision Coverage](https://img.shields.io/badge/supervision-0.30.0-purple)](./docs/coverage.md)

# ml-pipes-supervision

## Hello

[Supervision](https://github.com/roboflow/supervision) is your essential toolkit for computer vision. From data loading to real-time zone counting, it provides the building blocks so you can focus on building applications around your models.
`ml-pipes-supervision` provides `Supervision`
capabilities as composable operators in [ml-pipes](https://github.com/trained-by-humans/ml-pipes).

## Coverage

| Task                        | Status      |
|-----------------------------|-------------|
| Classification              | Not covered |
| Detection                   | Covered     |
| Segmentation                | Covered     |
| Keypoints                   | Not covered |
| Tracking                    | Covered     |
| Tools (Zones, Slicer, etc.) | Covered     |
| Dataset                     | Not covered |
| Evaluation                  | Not covered |
| Vision-language models      | Not covered |

See [coverage](./docs/coverage.md) for role definitions and the detailed API
compatibility matrix.

## Install

Install `ml-pipes-supervision` in a [Python >=3.10](https://www.python.org/)
environment:

```bash
pip install ml-pipes-supervision
```

This also installs the required `ml-pipes` packages, including
`ml-pipes-core` and `ml-pipes-vision`, plus the Supervision and Roboflow
Inference and tracker runtime dependencies.

The public operators are available from `ml_pipes.supervision`. Roboflow
Inference and external tracker boundaries are available from
`ml_pipes.supervision.inference` and `ml_pipes.supervision.trackers`.

## Quickstart

Build the usual detection-and-annotation flow as one pipeline. The operators
below are the same thin boundaries used by the runnable examples.

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Select, Store
from ml_pipes.vision import Decode, LoadFile
from ml_pipes.supervision import BoxAnnotator, Detections, ImageToArray, LabelAnnotator, PlotImage
from ml_pipes.supervision.inference import RoboflowInference

pipeline = Pipeline(
    [
        LoadFile(),
        Decode(),
        ImageToArray(),
        Store("source_image"),
        RoboflowInference(model_id="rfdetr-small"),
        Select(0),
        Detections.FromInference(),
        Recall("source_image", prepend=True),
        BoxAnnotator(),
        LabelAnnotator(),
        PlotImage(),
    ]
)
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

For model integrations that produce an `ml_pipes.tensor.TensorRegistry`, use
`Detections.FromTensorRegistry()` before Supervision annotators, trackers,
zones, or sinks. This is the explicit boundary from tensor post-processing to
Supervision data.

## Tutorials

Want to learn how to use Supervision with `ml-pipes`? Explore our
[how-to guides](https://requiem4machines.github.io/ml-pipes-supervision/tutorials/detect_and_annotate/)
and [end-to-end examples](./examples/)!

## Built with Supervision x ml-pipes

| Example | Upstream Source | Section | Note |
|---|---|---|---|
| [`run_detect_and_annotate.py`](./examples/run_detect_and_annotate.py) | [`Detect and Annotate` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_and_annotate/) | `Run Detection`, `Annotate Image with Detections`, `Display Custom Labels` | Runs object detection, then draws bounding boxes and available class labels on the image. |
| [`run_filter_detections.py`](./examples/run_filter_detections.py) | [`Filter Detections` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/filter_detections/) | `Filter Detections` | Keeps detections by class, confidence, and relative bounding-box area before annotation. |
| [`run_segment_and_annotate.py`](./examples/run_segment_and_annotate.py) | [`Detect and Annotate` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_and_annotate/) | `Run Detection`, `Annotate Image with Segmentations` | Runs instance segmentation and draws masks and labels on the image. |
| [`run_detection_video.py`](./examples/run_detection_video.py) | [`Annotate Video with Detections`](https://supervision.roboflow.com/0.30.0/notebooks/annotate-video-with-detections/) | `Run Detection` | Detects and annotates objects on each video frame, with an FPS overlay. |
| [`run_save_detections.py`](./examples/run_save_detections.py) | [`Save Detections`](https://supervision.roboflow.com/latest/how_to/save_detections/) | `Save Detections` | Runs detection on each video frame and writes the results to CSV. |
| [`run_track_objects.py`](./examples/run_track_objects.py) | [`Track Objects` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/track_objects/) | `Track Objects`, `Annotate Tracking IDs`, `Annotate Traces`, `Smooth Tracked Detections` | Assigns persistent IDs, smooths tracked boxes, and draws IDs, classes, and motion paths. |
| [`run_count_in_zone.py`](./examples/run_count_in_zone.py) | [`Count Objects in Zone` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/count_in_zone/) | `Count Objects in Zone` | Counts and annotates detections inside each configured polygon zone. |
| [`run_count_objects_crossing_line.py`](./examples/run_count_objects_crossing_line.py) | [`Count Objects Crossing the Line`](https://supervision.roboflow.com/latest/notebooks/count-objects-crossing-the-line/#process-video) | `Process Video` | Tracks objects and counts their crossings in each direction over a line. |
| [`run_detect_small_objects.py`](./examples/run_detect_small_objects.py) | [`Detect Small Objects` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/how_to/detect_small_objects/) | `Use InferenceSlicer` | Splits an image into overlapping tiles, detects objects per tile, and merges the results. |
| [`run_zero_shot_object_detection.py`](./examples/run_zero_shot_object_detection.py) | [`Zero-Shot Object Detection with YOLO-World` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/notebooks/zero-shot-object-detection-with-yolo-world/) | `Process Video` | Detects objects matching a supplied text prompt and filters duplicate or oversized predictions. |
| [`run_oriented_bounding_boxes.py`](./examples/run_oriented_bounding_boxes.py) | [`Oriented Bounding Boxes` (`0.30.0`)](https://supervision.roboflow.com/0.30.0/notebooks/oriented-bounding-boxes/) | `Oriented Box Annotation` | Detects ships and draws their rotated bounding boxes. |

## Documentation

Visit our [documentation](https://requiem4machines.github.io/ml-pipes-supervision/)
to learn how `ml-pipes-supervision` composes Supervision capabilities into
validated pipelines.

## Dependency boundary
The current dependency surface includes the following ml-pipes bridges. A
later refactor can move them out of the top-level import surface and reduce
the base dependency to `ml-pipes-core`:

- `ImagePayload` to `ndarray` conversion
- `TensorRegistry` to `sv.Detections` conversion
- Vision `TileRect` detection stitching
- `ImagePayload` convenience support in inference and image-window operators
