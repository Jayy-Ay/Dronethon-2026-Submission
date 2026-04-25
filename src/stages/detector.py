"""Backward-compatible detector aliases.

Use `src.stages.aruco_detector` and `src.stages.yolo_detector` directly for new code.
"""

from src.stages.aruco_detector import ArucoDetection as Detection
from src.stages.aruco_detector import ArucoDetector as TagDetector
from src.stages.aruco_detector import FAMILY_MAP

__all__ = ["Detection", "TagDetector", "FAMILY_MAP"]
