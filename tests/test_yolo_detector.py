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


if __name__ == "__main__":
    unittest.main()
