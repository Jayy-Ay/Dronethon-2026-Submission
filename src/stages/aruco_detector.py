"""ArUco/AprilTag detector running on PC-side frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "tag_id": self.tag_id,
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "area_px": round(self.area_px, 2),
        }


class ArucoDetector:
    """AprilTag detector wrapper for per-frame detections."""

    def __init__(self, family: str = "tag36h11") -> None:
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
            detections.append(
                ArucoDetection(
                    tag_id=int(tag_id),
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    area_px=area,
                    corners=tuple((float(x), float(y)) for x, y in pts),
                )
            )

        return detections
