import numpy as np
import supervision as sv

from ml_pipes.supervision import BoxAnnotator


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
