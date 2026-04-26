"""ArUco/AprilTag detector running on PC-side frames."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


FAMILY_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


@dataclass(frozen=True)
class ArucoDetection:
    """A single marker detection for telemetry and overlays."""

    tag_id: int
    center_x: float
    center_y: float
    area_px: float
    corners: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    distance_m: Optional[float] = None
    tvec_m: Optional[Tuple[float, float, float]] = None
    rvec: Optional[Tuple[float, float, float]] = None
    camera_position_m: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, float | int]:
        payload: Dict[str, float | int] = {
            "tag_id": self.tag_id,
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "area_px": round(self.area_px, 2),
        }
        if self.distance_m is not None:
            payload["distance_m"] = round(self.distance_m, 3)
        if self.tvec_m is not None:
            payload["x_m"] = round(self.tvec_m[0], 3)
            payload["y_m"] = round(self.tvec_m[1], 3)
            payload["z_m"] = round(self.tvec_m[2], 3)
        if self.camera_position_m is not None:
            payload["cam_x_m"] = round(self.camera_position_m[0], 3)
            payload["cam_y_m"] = round(self.camera_position_m[1], 3)
            payload["cam_z_m"] = round(self.camera_position_m[2], 3)
        return payload


def _default_camera_matrix(width: int, height: int) -> np.ndarray:
    """Build a rough pinhole camera model when calibration is unavailable."""
    fx = fy = 1.2 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class ArucoDetector:
    """AprilTag detector wrapper for per-frame detections."""

    def __init__(
        self,
        family: str = "tag36h11",
        marker_length_m: Optional[float] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> None:
        if family not in FAMILY_MAP:
            choices = ", ".join(sorted(FAMILY_MAP))
            raise ValueError(f"Unsupported family '{family}'. Choose from: {choices}")

        aruco_dict = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[family])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 5.0
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self._marker_length_m = marker_length_m if marker_length_m and marker_length_m > 0 else None
        self._camera_matrix = camera_matrix.astype(np.float32) if camera_matrix is not None else None
        self._dist_coeffs = dist_coeffs.astype(np.float32) if dist_coeffs is not None else np.zeros(5, dtype=np.float32)
        self._obj_points = None
        if self._marker_length_m is not None:
            half = self._marker_length_m / 2.0
            self._obj_points = np.array(
                [
                    [-half, half, 0.0],
                    [half, half, 0.0],
                    [half, -half, 0.0],
                    [-half, -half, 0.0],
                ],
                dtype=np.float32,
            )

    def detect(self, frame: np.ndarray) -> List[ArucoDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return []

        detections: List[ArucoDetection] = []
        for tag_corners, tag_id in zip(corners, ids.flatten()):
            pts = tag_corners.reshape(4, 2)
            center = pts.mean(axis=0)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            distance_m = None
            tvec_m = None
            rvec_tuple = None
            camera_position_m = None

            if self._obj_points is not None:
                camera_matrix = self._camera_matrix
                if camera_matrix is None:
                    camera_matrix = _default_camera_matrix(frame.shape[1], frame.shape[0])

                success, rvec, tvec = cv2.solvePnP(
                    self._obj_points,
                    pts.astype(np.float32),
                    camera_matrix,
                    self._dist_coeffs,
                )
                if success:
                    tvec = tvec.reshape(3)
                    rvec = rvec.reshape(3)
                    distance_m = float(np.linalg.norm(tvec))
                    tvec_m = (float(tvec[0]), float(tvec[1]), float(tvec[2]))
                    rvec_tuple = (float(rvec[0]), float(rvec[1]), float(rvec[2]))

                    rot_mtx, _ = cv2.Rodrigues(rvec)
                    camera_pos = (-rot_mtx.T @ tvec.reshape(3, 1)).reshape(3)
                    camera_position_m = (
                        float(camera_pos[0]),
                        float(camera_pos[1]),
                        float(camera_pos[2]),
                    )

            detections.append(
                ArucoDetection(
                    tag_id=int(tag_id),
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    area_px=area,
                    corners=tuple((float(x), float(y)) for x, y in pts),
                    distance_m=distance_m,
                    tvec_m=tvec_m,
                    rvec=rvec_tuple,
                    camera_position_m=camera_position_m,
                )
            )

        return detections
