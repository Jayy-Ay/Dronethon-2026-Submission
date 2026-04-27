import unittest

import numpy as np

from src.stages.yolo_detector import YoloDetector


class TestYoloDetectorPostprocess(unittest.TestCase):
    def _build_detector(self) -> YoloDetector:
        detector = YoloDetector.__new__(YoloDetector)
        detector._labels = ["person", "bicycle", "car"]
        detector._conf_thresh = 0.35
        detector._nms_thresh = 0.45
        detector._input_size = 640
        detector._model_family = "auto"
        return detector

    def test_postprocess_yolov5_output_uses_objectness_times_class_score(self):
        detector = self._build_detector()
        detector._model_family = "yolov5"
        output = np.array(
            [
                [
                    [320.0, 320.0, 100.0, 120.0, 0.90, 0.10, 0.80, 0.20],
                ]
            ],
            dtype=np.float32,
        )

        detections = detector._postprocess(output, (640, 640, 3), 1.0, 0, 0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "bicycle")
        self.assertAlmostEqual(detections[0].confidence, 0.72, places=5)

    def test_postprocess_yolov8_output_uses_class_score_directly(self):
        detector = self._build_detector()
        detector._model_family = "yolov8"
        output = np.array(
            [
                [
                    [320.0, 320.0, 100.0, 120.0, 0.10, 0.85, 0.20],
                ]
            ],
            dtype=np.float32,
        )

        detections = detector._postprocess(output, (640, 640, 3), 1.0, 0, 0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "bicycle")
        self.assertAlmostEqual(detections[0].confidence, 0.85, places=5)

    def test_extract_result_masks_resizes_and_binarizes_masks(self):
        class _FakeTensor:
            def __init__(self, array):
                self._array = array

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._array

        class _FakeMasks:
            def __init__(self, array):
                self.data = _FakeTensor(array)

        class _FakeResult:
            def __init__(self, array):
                self.masks = _FakeMasks(array)

        result = _FakeResult(
            np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 1.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            )
        )

        masks = YoloDetector._extract_result_masks(result, (4, 6), expected_count=2)

        self.assertIsNotNone(masks)
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].shape, (4, 6))
        self.assertEqual(masks[0].dtype, np.bool_)
        self.assertTrue(masks[0].any())
        self.assertTrue(masks[1].any())


if __name__ == "__main__":
    unittest.main()
