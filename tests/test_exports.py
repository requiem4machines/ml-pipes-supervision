import pytest

from ml_pipes import supervision
from ml_pipes.core import Pipeline
from ml_pipes.tensor import TensorRegistry
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision.trackers import ByteTrack


def test_public_operator_exports() -> None:
    assert supervision.BoxAnnotator
    assert supervision.Detections
    assert supervision.TriggerZone
    assert RoboflowInference
    assert ByteTrack


def test_tensor_registry_detection_adapter() -> None:
    import numpy as np

    detections = supervision.Detections.FromTensorRegistry()(
        TensorRegistry(
            {
                "boxes": np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
                "classes": np.asarray([1], dtype=np.int32),
            }
        )
    )

    assert detections.xyxy.tolist() == [[0.0, 0.0, 10.0, 10.0]]
    assert detections.class_id.tolist() == [1]


def test_tensor_registry_detection_adapter_accepts_custom_mask_fields() -> None:
    import numpy as np

    detections = supervision.Detections.FromTensorRegistry(
        boxes="xyxy", scores="confidence", classes="class_id", masks="mask"
    )(
        TensorRegistry(
            {
                "xyxy": np.asarray([[0, 0, 2, 2]], dtype=np.float32),
                "confidence": np.asarray([0.75], dtype=np.float32),
                "class_id": np.asarray([3], dtype=np.int32),
                "mask": np.ones((1, 2, 2), dtype=bool),
            }
        )
    )

    assert detections.mask is not None
    assert detections.mask.shape == (1, 2, 2)


def test_detections_filter_accepts_unannotated_lambda() -> None:
    import numpy as np

    pipeline = Pipeline(
        [
            supervision.Detections.Filter(
                lambda detections: detections.class_id == 1
            )
        ],
        auto_validate=True,
    )
    detections = supervision.Detections.FromTensorRegistry()(
        TensorRegistry(
            {
                "boxes": np.asarray([[0, 0, 10, 10], [1, 1, 2, 2]], dtype=np.float32),
                "scores": np.asarray([0.9, 0.8], dtype=np.float32),
                "classes": np.asarray([1, 2], dtype=np.int32),
            }
        )
    )

    result = pipeline(detections)

    assert result.class_id.tolist() == [1]


def test_detections_filter_requires_detections_result() -> None:
    import numpy as np

    detections = supervision.Detections.FromTensorRegistry()(
        TensorRegistry(
            {
                "boxes": np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
                "classes": np.asarray([1], dtype=np.int32),
            }
        )
    )

    with pytest.raises(TypeError, match="must return supervision.Detections or a boolean"):
        supervision.Detections.Filter(lambda _: None)(detections)
