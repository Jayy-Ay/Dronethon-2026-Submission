import unittest

import numpy as np

from src.stages.yolo_detector import YoloDetection
from src.vision.object_geolocator import (
    TelemetryPoseNed,
    detection_reference_pixel,
    estimate_object_ground_position,
)


DOWNWARD_BODY_TO_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class TestObjectGeolocator(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.pose = TelemetryPoseNed(
            north_m=10.0,
            east_m=20.0,
            down_m=-5.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
        )

    def test_detection_reference_pixel_uses_bbox_center_without_mask(self):
        detection = YoloDetection(
            class_id=0,
            label="person",
            confidence=0.9,
            x1=10,
            y1=30,
            x2=30,
            y2=50,
        )

        u_px, v_px = detection_reference_pixel(detection)

        self.assertEqual((u_px, v_px), (20.0, 40.0))

    def test_estimate_object_ground_position_projects_center_pixel_below_drone(self):
        detection = YoloDetection(
            class_id=0,
            label="person",
            confidence=0.9,
            x1=40,
            y1=40,
            x2=60,
            y2=60,
        )

        estimate = estimate_object_ground_position(
            detection=detection,
            pose=self.pose,
            camera_matrix=self.camera_matrix,
            body_to_camera_rotation=DOWNWARD_BODY_TO_CAMERA,
            dist_coeffs=np.zeros(5, dtype=np.float64),
            ground_down_m=0.0,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.north_m, 10.0, places=5)
        self.assertAlmostEqual(estimate.east_m, 20.0, places=5)
        self.assertAlmostEqual(estimate.down_m, 0.0, places=5)
        self.assertAlmostEqual(estimate.slant_range_m, 5.0, places=5)

    def test_estimate_object_ground_position_shifts_with_pixel_offset(self):
        detection = YoloDetection(
            class_id=0,
            label="person",
            confidence=0.9,
            x1=60,
            y1=40,
            x2=80,
            y2=60,
        )

        estimate = estimate_object_ground_position(
            detection=detection,
            pose=self.pose,
            camera_matrix=self.camera_matrix,
            body_to_camera_rotation=DOWNWARD_BODY_TO_CAMERA,
            dist_coeffs=np.zeros(5, dtype=np.float64),
            ground_down_m=0.0,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.north_m, 10.0, places=5)
        self.assertAlmostEqual(estimate.east_m, 21.0, places=5)
        self.assertAlmostEqual(estimate.down_m, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
