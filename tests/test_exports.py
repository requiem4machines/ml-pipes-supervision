import pytest

from ml_pipes import supervision
from ml_pipes.core import Pipeline
from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock
from ml_pipes.tensor import TensorRegistry
from ml_pipes.supervision.inference import RoboflowInference
from ml_pipes.supervision.trackers import ByteTrack


def test_public_operator_exports() -> None:
    assert supervision.BoxAnnotator
    assert supervision.Detections
    assert supervision.TriggerZone
    assert RoboflowInference
    assert ByteTrack


def test_raw_ndarray_inspection_uses_supervision_bgr_convention() -> None:
    import numpy as np

    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)

    blocks = PipelineInspector()._value_to_blocks(bgr)

    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ndarray  1×1  BGR"
    np.testing.assert_array_equal(blocks[0].array, np.array([[[255, 0, 0]]], dtype=np.uint8))


def test_ultralytics_results_inspection_renders_result_summary() -> None:
    import numpy as np
    from ultralytics.engine.results import Results

    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
    result = Results(orig_img=bgr, path="frame.jpg", names={})

    blocks = PipelineInspector()._value_to_blocks(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "Ultralytics Results"
    assert ("original shape", "(1, 1)") in blocks[0].rows
    assert ("detections", "0") in blocks[0].rows


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
