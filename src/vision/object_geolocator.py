"""Estimate ground-plane object positions from image detections and MAVLink pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

from src.stages.yolo_detector import YoloDetection


@dataclass(frozen=True)
class TelemetryPoseNed:
    """Drone body pose expressed in a local NED frame."""

    north_m: float
    east_m: float
    down_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(frozen=True)
class ObjectPositionEstimate:
    """Ground-plane estimate for one detected object."""

    label: str
    confidence: float
    image_u_px: float
    image_v_px: float
    north_m: float
    east_m: float
    down_m: float
    slant_range_m: float


def rotation_matrix_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a ZYX rotation matrix from Euler angles in degrees."""
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    sr, cr = np.sin(roll_rad), np.cos(roll_rad)
    sp, cp = np.sin(pitch_rad), np.cos(pitch_rad)
    sy, cy = np.sin(yaw_rad), np.cos(yaw_rad)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def detection_reference_pixel(det: YoloDetection) -> tuple[float, float]:
    """Return the image-space point used for geolocation."""
    if det.mask is not None and det.mask.any():
        # Segmentation centroid is usually more stable than box center on irregular shapes.
        ys, xs = np.nonzero(det.mask)
        return float(np.mean(xs)), float(np.mean(ys))

    return float(det.x1 + det.x2) * 0.5, float(det.y1 + det.y2) * 0.5


def pixel_to_camera_ray(
    image_u_px: float,
    image_v_px: float,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert an image pixel into a unit ray in the OpenCV camera frame."""
    if dist_coeffs is not None:
        undistorted = cv2.undistortPoints(
            np.array([[[image_u_px, image_v_px]]], dtype=np.float64),
            camera_matrix.astype(np.float64),
            dist_coeffs.astype(np.float64),
        )
        x_norm = float(undistorted[0, 0, 0])
        y_norm = float(undistorted[0, 0, 1])
        ray = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    else:
        inv_k = np.linalg.inv(camera_matrix.astype(np.float64))
        ray = inv_k @ np.array([image_u_px, image_v_px, 1.0], dtype=np.float64)

    norm = float(np.linalg.norm(ray))
    if norm <= 1e-9:
        raise ValueError("Invalid camera ray with near-zero norm")
    return ray / norm


def estimate_object_ground_position(
    detection: YoloDetection,
    pose: TelemetryPoseNed,
    camera_matrix: np.ndarray,
    body_to_camera_rotation: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    camera_offset_body_m: Iterable[float] = (0.0, 0.0, 0.0),
    ground_down_m: float = 0.0,
    max_range_m: Optional[float] = None,
) -> Optional[ObjectPositionEstimate]:
    """Project a detection ray onto the horizontal ground plane in local NED."""
    # 1) Build the camera-frame observation ray from the detection pixel.
    image_u_px, image_v_px = detection_reference_pixel(detection)
    ray_camera = pixel_to_camera_ray(
        image_u_px,
        image_v_px,
        camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    rot_body_to_world = rotation_matrix_from_euler_deg(
        pose.roll_deg,
        pose.pitch_deg,
        pose.yaw_deg,
    )
    rot_camera_to_body = body_to_camera_rotation.astype(np.float64).T
    ray_body = rot_camera_to_body @ ray_camera
    ray_world = rot_body_to_world @ ray_body

    camera_offset_body = np.array(list(camera_offset_body_m), dtype=np.float64)
    body_position_world = np.array([pose.north_m, pose.east_m, pose.down_m], dtype=np.float64)
    camera_position_world = body_position_world + (rot_body_to_world @ camera_offset_body)

    ray_down_component = float(ray_world[2])
    if abs(ray_down_component) <= 1e-9:
        # Ray is nearly parallel to the ground plane.
        return None

    distance_along_ray = (float(ground_down_m) - float(camera_position_world[2])) / ray_down_component
    if distance_along_ray <= 0.0:
        # Intersection is behind the camera.
        return None

    object_position_world = camera_position_world + distance_along_ray * ray_world
    slant_range_m = float(np.linalg.norm(object_position_world - camera_position_world))
    if max_range_m is not None and slant_range_m > max_range_m:
        return None

    return ObjectPositionEstimate(
        label=detection.label,
        confidence=detection.confidence,
        image_u_px=image_u_px,
        image_v_px=image_v_px,
        north_m=float(object_position_world[0]),
        east_m=float(object_position_world[1]),
        down_m=float(object_position_world[2]),
        slant_range_m=slant_range_m,
    )
