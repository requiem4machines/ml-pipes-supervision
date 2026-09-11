import numpy as np
import pytest
import supervision as sv

from ml_pipes.supervision import (
    BoxAnnotator,
    Detection,
    LabelAnnotator,
    RichLabelAnnotator,
)


def test_box_annotator_preserves_source_scene() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    source = scene.copy()
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
    )

    annotated, returned_detections = BoxAnnotator()(scene, detections)

    np.testing.assert_array_equal(scene, source)
    assert annotated is not scene
    assert not np.array_equal(annotated, source)
    assert returned_detections is detections


def test_label_annotator_passes_each_detection_to_callback() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int32),
        tracker_id=np.array([12], dtype=np.int32),
        data={"time_in_zone": np.array([1.5], dtype=np.float32)},
    )
    received: list[Detection] = []

    annotated, returned_detections = LabelAnnotator(
        label_formatter=lambda detection: received.append(detection) or "01:30"
    )(scene, detections)

    assert annotated is not scene
    assert not np.array_equal(annotated, scene)
    assert returned_detections is detections
    assert received[0].tracker_id == 12
    assert received[0].data["time_in_zone"] == 1.5


def test_label_annotator_requires_string_label() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32))

    with pytest.raises(TypeError, match="must return str"):
        LabelAnnotator(label_formatter=lambda _: 1)(scene, detections)


def test_label_annotator_rejects_callback_with_standard_fields() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        LabelAnnotator(label_formatter=lambda _: "label", show_class=True)


def test_rich_label_annotator_passes_each_detection_to_callback() -> None:
    scene = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[4, 4, 28, 28]], dtype=np.float32),
        class_id=np.array([3], dtype=np.int32),
        tracker_id=np.array([12], dtype=np.int32),
    )
    received: list[Detection] = []

    annotated, returned_detections = RichLabelAnnotator(
        label_formatter=lambda detection: received.append(detection) or "車両 #12"
    )(scene, detections)

    assert annotated is not scene
    assert not np.array_equal(annotated, scene)
    assert returned_detections is detections
    assert received[0].tracker_id == 12


def test_rich_label_annotator_rejects_callback_with_standard_fields() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        RichLabelAnnotator(label_formatter=lambda _: "label", show_class=True)
