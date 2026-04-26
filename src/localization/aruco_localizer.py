"""Estimate camera/drone world pose from one or more floor ArUco markers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import cv2
import numpy as np
from src.stages.aruco_detector import ArucoDetection


def _rvec_tvec_to_camera_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV world-to-camera pose into camera rotation/position in world."""
    rot_world_to_camera, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    rot_camera_to_world = rot_world_to_camera.T
    camera_position_world = (-rot_camera_to_world @ tvec.reshape(3, 1)).reshape(3)
    return rot_camera_to_world, camera_position_world


def _rotation_matrix_to_rvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into OpenCV Rodrigues-vector form."""
    rvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
    return rvec.reshape(3)


@dataclass(frozen=True)
class MarkerPose:
    """Known fixed pose of one ArUco marker in the world frame."""

    marker_id: int
    center_m: tuple[float, float, float]
    yaw_rad: float = 0.0

    def corner_points(self, marker_length_m: float) -> np.ndarray:
        """Return marker corners in world coordinates in OpenCV corner order."""
        half = marker_length_m / 2.0
        local = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        cos_yaw = float(np.cos(self.yaw_rad))
        sin_yaw = float(np.sin(self.yaw_rad))
        rot_z = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = np.array(self.center_m, dtype=np.float32)
        return (local @ rot_z.T) + center


@dataclass(frozen=True)
class MarkerMap:
    """World-frame geometry for all known floor markers."""

    marker_length_m: float
    markers: Dict[int, MarkerPose]

    @staticmethod
    def rectangular_floor_map(
        marker_length_m: float,
        area_width_m: float,
        area_height_m: float,
        origin_at_top_left: bool = True,
    ) -> "MarkerMap":
        """Create the 4-marker layout described in the project notes."""
        if origin_at_top_left:
            markers = {
                0: MarkerPose(0, (0.0, 0.0, 0.0)),
                1: MarkerPose(1, (area_width_m, 0.0, 0.0)),
                2: MarkerPose(2, (0.0, area_height_m, 0.0)),
                3: MarkerPose(3, (area_width_m, area_height_m, 0.0)),
            }
        else:
            half_w = area_width_m / 2.0
            half_h = area_height_m / 2.0
            markers = {
                0: MarkerPose(0, (-half_w, half_h, 0.0)),
                1: MarkerPose(1, (half_w, half_h, 0.0)),
                2: MarkerPose(2, (-half_w, -half_h, 0.0)),
                3: MarkerPose(3, (half_w, -half_h, 0.0)),
            }
        return MarkerMap(marker_length_m=marker_length_m, markers=markers)


@dataclass(frozen=True)
class PoseEstimate:
    """Pose estimate derived from currently visible markers."""

    success: bool
    marker_ids: tuple[int, ...]
    image_point_count: int
    camera_position_world_m: tuple[float, float, float]
    camera_rvec_world: tuple[float, float, float]
    body_position_world_m: Optional[tuple[float, float, float]] = None
    body_rvec_world: Optional[tuple[float, float, float]] = None
    reprojection_error_px: Optional[float] = None


class ArucoWorldLocalizer:
    """Estimate world pose by solving PnP against all visible mapped markers."""

    def __init__(
        self,
        marker_map: MarkerMap,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        body_to_camera_rotation: Optional[np.ndarray] = None,
        camera_offset_body_m: Optional[Iterable[float]] = None,
        smoothing_alpha: float = 0.0,
    ) -> None:
        self._marker_map = marker_map
        self._camera_matrix = camera_matrix.astype(np.float32)
        self._dist_coeffs = dist_coeffs.astype(np.float32)
        self._rot_body_to_camera = (
            body_to_camera_rotation.astype(np.float64)
            if body_to_camera_rotation is not None
            else np.eye(3, dtype=np.float64)
        )
        self._camera_offset_body_m = np.array(
            list(camera_offset_body_m) if camera_offset_body_m is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 0.95))
        self._last_camera_position_world: Optional[np.ndarray] = None
        self._last_rot_camera_to_world: Optional[np.ndarray] = None

    def estimate_pose(self, detections: Iterable[ArucoDetection]) -> PoseEstimate:
        """Fuse all currently visible mapped markers into one camera/body pose."""
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        used_marker_ids: list[int] = []

        for detection in detections:
            marker_pose = self._marker_map.markers.get(detection.tag_id)
            if marker_pose is None:
                continue

            object_points.append(marker_pose.corner_points(self._marker_map.marker_length_m))
            image_points.append(np.array(detection.corners, dtype=np.float32))
            used_marker_ids.append(detection.tag_id)

        if not object_points:
            return PoseEstimate(
                success=False,
                marker_ids=(),
                image_point_count=0,
                camera_position_world_m=(0.0, 0.0, 0.0),
                camera_rvec_world=(0.0, 0.0, 0.0),
            )

        object_points_np = np.vstack(object_points).astype(np.float32)
        image_points_np = np.vstack(image_points).astype(np.float32)

        pnp_flag = cv2.SOLVEPNP_IPPE if len(used_marker_ids) == 1 else cv2.SOLVEPNP_ITERATIVE
        success, rvec_world_to_camera, tvec_world_to_camera = cv2.solvePnP(
            object_points_np,
            image_points_np,
            self._camera_matrix,
            self._dist_coeffs,
            flags=pnp_flag,
        )
        if not success:
            return PoseEstimate(
                success=False,
                marker_ids=tuple(used_marker_ids),
                image_point_count=int(len(image_points_np)),
                camera_position_world_m=(0.0, 0.0, 0.0),
                camera_rvec_world=(0.0, 0.0, 0.0),
            )

        rot_camera_to_world, camera_position_world = _rvec_tvec_to_camera_pose(
            rvec_world_to_camera,
            tvec_world_to_camera,
        )
        rot_camera_to_world, camera_position_world = self._smooth_pose(
            rot_camera_to_world,
            camera_position_world,
        )

        reprojection_error_px = self._reprojection_error(
            object_points_np,
            image_points_np,
            rot_camera_to_world,
            camera_position_world,
        )

        camera_rvec_world = _rotation_matrix_to_rvec(rot_camera_to_world)

        body_position_world, body_rvec_world = self._body_pose_from_camera_pose(
            rot_camera_to_world,
            camera_position_world,
        )

        return PoseEstimate(
            success=True,
            marker_ids=tuple(sorted(set(used_marker_ids))),
            image_point_count=int(len(image_points_np)),
            camera_position_world_m=tuple(float(x) for x in camera_position_world),
            camera_rvec_world=tuple(float(x) for x in camera_rvec_world),
            body_position_world_m=tuple(float(x) for x in body_position_world),
            body_rvec_world=tuple(float(x) for x in body_rvec_world),
            reprojection_error_px=reprojection_error_px,
        )

    def _body_pose_from_camera_pose(
        self,
        rot_camera_to_world: np.ndarray,
        camera_position_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert camera-world pose into drone-body world pose."""
        rot_camera_from_body = self._rot_body_to_camera.T
        rot_body_to_world = rot_camera_to_world @ rot_camera_from_body
        body_position_world = camera_position_world - (rot_body_to_world @ self._camera_offset_body_m)
        body_rvec_world = _rotation_matrix_to_rvec(rot_body_to_world)
        return body_position_world, body_rvec_world

    def _smooth_pose(
        self,
        rot_camera_to_world: np.ndarray,
        camera_position_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply optional first-order smoothing for prototype stability."""
        alpha = self._smoothing_alpha
        if alpha <= 0.0:
            self._last_camera_position_world = camera_position_world.copy()
            self._last_rot_camera_to_world = rot_camera_to_world.copy()
            return rot_camera_to_world, camera_position_world

        if self._last_camera_position_world is None or self._last_rot_camera_to_world is None:
            self._last_camera_position_world = camera_position_world.copy()
            self._last_rot_camera_to_world = rot_camera_to_world.copy()
            return rot_camera_to_world, camera_position_world

        smoothed_position = (
            alpha * self._last_camera_position_world
            + (1.0 - alpha) * camera_position_world
        )
        blended_rotation = (
            alpha * self._last_rot_camera_to_world
            + (1.0 - alpha) * rot_camera_to_world
        )
        u, _, vt = np.linalg.svd(blended_rotation)
        smoothed_rotation = u @ vt

        self._last_camera_position_world = smoothed_position.copy()
        self._last_rot_camera_to_world = smoothed_rotation.copy()
        return smoothed_rotation, smoothed_position

    def _reprojection_error(
        self,
        object_points_world: np.ndarray,
        image_points_px: np.ndarray,
        rot_camera_to_world: np.ndarray,
        camera_position_world: np.ndarray,
    ) -> float:
        """Compute mean reprojection error as a simple quality signal."""
        rot_world_to_camera = rot_camera_to_world.T
        tvec_world_to_camera = (-rot_world_to_camera @ camera_position_world.reshape(3, 1)).reshape(3)
        rvec_world_to_camera = _rotation_matrix_to_rvec(rot_world_to_camera)
        reprojected_points, _ = cv2.projectPoints(
            object_points_world,
            rvec_world_to_camera.reshape(3, 1),
            tvec_world_to_camera.reshape(3, 1),
            self._camera_matrix,
            self._dist_coeffs,
        )
        reprojected_points = reprojected_points.reshape(-1, 2)
        error_px = np.linalg.norm(reprojected_points - image_points_px, axis=1)
        return float(np.mean(error_px))
