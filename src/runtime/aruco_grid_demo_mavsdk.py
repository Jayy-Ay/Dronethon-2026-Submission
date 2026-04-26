"""MAVSDK + ArUco floor-marker grid traversal demo for a Pixhawk 4 companion computer.

This script is designed for an early proof-of-concept:
- The Raspberry Pi 5 estimates the drone pose from floor ArUco markers.
- MAVSDK sends high-level movement commands to the Pixhawk 4.
- The Pixhawk 4 handles stabilization and motor control.

Important limitations:
- ArUco localisation quality depends strongly on calibration, lighting, blur,
  marker visibility, and marker placement accuracy.
- PX4 Offboard mode generally expects a valid position source. If PX4 is not
  already configured with a local position estimate, additional vision-pose
  integration may be required before autonomous movement is accepted safely.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.localization import ArucoWorldLocalizer, MarkerMap, PoseEstimate
from src.stages.aruco_detector import ArucoDetection, ArucoDetector

try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
except ImportError as exc:  # pragma: no cover - dependency is optional in this environment
    raise SystemExit(
        "MAVSDK is required for this demo. Install it with `pip install mavsdk`."
    ) from exc


@dataclass(frozen=True)
class SearchArea:
    """Rectangular search area in the ArUco world frame."""

    width_m: float
    length_m: float
    row_spacing_m: float
    scan_altitude_m: float


@dataclass(frozen=True)
class Waypoint:
    """World-frame target waypoint for the snake scan."""

    x_m: float
    y_m: float
    z_m: float
    tolerance_xy_m: float
    tolerance_z_m: float
    max_speed_m_s: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ArUco grid traversal demo."""
    parser = argparse.ArgumentParser(description="MAVSDK ArUco-based snake grid traversal demo")
    parser.add_argument("--system-address", default="serial:///dev/ttyAMA0:921600", help="MAVSDK system address")
    parser.add_argument("--camera-source", default="0", help="OpenCV camera source index, device path, video file, or stream URL")
    parser.add_argument("--calibration-file", required=True, help="NPZ file containing camera_matrix and dist_coeffs")
    parser.add_argument("--family", default="6x6_250", help="ArUco family name")
    parser.add_argument("--marker-length-m", type=float, required=True, help="Printed marker edge length in meters")
    parser.add_argument("--area-width-m", type=float, required=True, help="Center distance from marker 0 to marker 1")
    parser.add_argument("--area-height-m", type=float, required=True, help="Center distance from marker 0 to marker 2")
    parser.add_argument("--row-spacing-m", type=float, required=True, help="Snake scan row spacing in meters")
    parser.add_argument("--scan-altitude-m", type=float, default=1.5, help="Target scan altitude in meters")
    parser.add_argument("--waypoint-tolerance-m", type=float, default=0.20, help="Horizontal waypoint completion tolerance")
    parser.add_argument("--altitude-tolerance-m", type=float, default=0.25, help="Vertical waypoint completion tolerance")
    parser.add_argument("--max-speed-m-s", type=float, default=0.25, help="Maximum XY speed in meters per second")
    parser.add_argument("--vertical-speed-m-s", type=float, default=0.20, help="Maximum vertical speed in meters per second")
    parser.add_argument("--kp-horizontal", type=float, default=0.8, help="Horizontal proportional gain")
    parser.add_argument("--kp-vertical", type=float, default=0.6, help="Vertical proportional gain")
    parser.add_argument("--takeoff-wait-s", type=float, default=8.0, help="Seconds to wait after takeoff command")
    parser.add_argument("--hold-time-s", type=float, default=1.0, help="Hold duration at each waypoint")
    parser.add_argument("--marker-loss-timeout-s", type=float, default=2.0, help="Maximum continuous no-marker time before abort")
    parser.add_argument("--max-reprojection-error-px", type=float, default=4.0, help="Reject pose if reprojection error exceeds this")
    parser.add_argument("--min-visible-markers", type=int, default=1, help="Minimum visible mapped markers required to move")
    parser.add_argument("--max-mission-time-s", type=float, default=180.0, help="Abort mission after this many seconds")
    parser.add_argument("--battery-min-fraction", type=float, default=0.30, help="Minimum battery fraction before arming")
    parser.add_argument("--smoothing-alpha", type=float, default=0.15, help="Pose smoothing factor for the ArUco localizer")
    parser.add_argument("--camera-roll-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: roll")
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: pitch")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: yaw")
    parser.add_argument("--camera-offset-x-m", type=float, default=0.0, help="Camera offset from drone body origin in body X")
    parser.add_argument("--camera-offset-y-m", type=float, default=0.0, help="Camera offset from drone body origin in body Y")
    parser.add_argument("--camera-offset-z-m", type=float, default=0.0, help="Camera offset from drone body origin in body Z")
    parser.add_argument("--show", action="store_true", help="Show annotated local preview window")
    return parser.parse_args()


def load_camera_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics and distortion coefficients from an NPZ file."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    data = np.load(calibration_path)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError("Calibration file must contain 'camera_matrix' and 'dist_coeffs'")
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def generate_snake_waypoints(area: SearchArea, xy_tolerance_m: float, z_tolerance_m: float, max_speed_m_s: float) -> list[Waypoint]:
    """Create a top-left-origin snake path over the rectangular marker area."""
    waypoints: list[Waypoint] = [
        Waypoint(
            x_m=0.0,
            y_m=0.0,
            z_m=area.scan_altitude_m,
            tolerance_xy_m=xy_tolerance_m,
            tolerance_z_m=z_tolerance_m,
            max_speed_m_s=max_speed_m_s,
        )
    ]

    row_index = 0
    current_y_m = 0.0
    while True:
        x_target_m = area.width_m if row_index % 2 == 0 else 0.0
        waypoints.append(
            Waypoint(
                x_m=x_target_m,
                y_m=current_y_m,
                z_m=area.scan_altitude_m,
                tolerance_xy_m=xy_tolerance_m,
                tolerance_z_m=z_tolerance_m,
                max_speed_m_s=max_speed_m_s,
            )
        )

        if current_y_m >= area.length_m:
            break

        next_y_m = min(current_y_m + area.row_spacing_m, area.length_m)
        waypoints.append(
            Waypoint(
                x_m=x_target_m,
                y_m=next_y_m,
                z_m=area.scan_altitude_m,
                tolerance_xy_m=xy_tolerance_m,
                tolerance_z_m=z_tolerance_m,
                max_speed_m_s=min(max_speed_m_s, 0.20),
            )
        )

        current_y_m = next_y_m
        row_index += 1

    return _dedupe_consecutive_waypoints(waypoints)


def _dedupe_consecutive_waypoints(waypoints: list[Waypoint]) -> list[Waypoint]:
    """Drop repeated consecutive waypoints created by edge clamping."""
    deduped: list[Waypoint] = []
    for waypoint in waypoints:
        if not deduped:
            deduped.append(waypoint)
            continue
        prev = deduped[-1]
        if (
            abs(prev.x_m - waypoint.x_m) < 1e-6
            and abs(prev.y_m - waypoint.y_m) < 1e-6
            and abs(prev.z_m - waypoint.z_m) < 1e-6
        ):
            continue
        deduped.append(waypoint)
    return deduped


def _rotation_matrix_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a ZYX rotation matrix from Euler angles in degrees."""
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)

    sr, cr = math.sin(roll_rad), math.cos(roll_rad)
    sp, cp = math.sin(pitch_rad), math.cos(pitch_rad)
    sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _pose_rvec_to_yaw_rad(body_rvec_world: tuple[float, float, float] | None) -> float:
    """Extract approximate world yaw from the body pose rotation vector."""
    if body_rvec_world is None:
        return 0.0
    rot_body_to_world, _ = cv2.Rodrigues(np.array(body_rvec_world, dtype=np.float64).reshape(3, 1))
    return math.atan2(rot_body_to_world[1, 0], rot_body_to_world[0, 0])


class ArucoGridMission:
    """Full ArUco-localised grid traversal mission using MAVSDK offboard control."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._drone = System()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_battery_fraction: Optional[float] = None
        self._telemetry_tasks: list[asyncio.Task[None]] = []
        self._offboard_started = False

        camera_matrix, dist_coeffs = load_camera_calibration(args.calibration_file)
        marker_map = MarkerMap.rectangular_floor_map(
            marker_length_m=args.marker_length_m,
            area_width_m=args.area_width_m,
            area_height_m=args.area_height_m,
        )

        body_to_camera_rotation = _rotation_matrix_from_euler_deg(
            args.camera_roll_deg,
            args.camera_pitch_deg,
            args.camera_yaw_deg,
        )
        camera_offset_body_m = [args.camera_offset_x_m, args.camera_offset_y_m, args.camera_offset_z_m]

        self._detector = ArucoDetector(
            family=args.family,
            marker_length_m=args.marker_length_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self._localizer = ArucoWorldLocalizer(
            marker_map=marker_map,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            body_to_camera_rotation=body_to_camera_rotation,
            camera_offset_body_m=camera_offset_body_m,
            smoothing_alpha=args.smoothing_alpha,
        )
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

        self._search_area = SearchArea(
            width_m=args.area_width_m,
            length_m=args.area_height_m,
            row_spacing_m=args.row_spacing_m,
            scan_altitude_m=args.scan_altitude_m,
        )
        self._waypoints = generate_snake_waypoints(
            self._search_area,
            xy_tolerance_m=args.waypoint_tolerance_m,
            z_tolerance_m=args.altitude_tolerance_m,
            max_speed_m_s=args.max_speed_m_s,
        )

    async def connect_to_pixhawk(self) -> None:
        """Connect MAVSDK to the Pixhawk 4 and start telemetry subscriptions."""
        await self._drone.connect(system_address=self._args.system_address)
        print(f"Connecting to Pixhawk 4 via {self._args.system_address}")
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                print("Connected to Pixhawk 4")
                break

        self._telemetry_tasks.append(asyncio.create_task(self._stream_battery()))

    def open_camera_stream(self) -> None:
        """Open the configured camera source via OpenCV."""
        source: int | str
        if self._args.camera_source.isdigit():
            source = int(self._args.camera_source)
        else:
            source = self._args.camera_source

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._args.camera_source}")
        print(f"Opened camera source {self._args.camera_source}")

    async def run_preflight_checks(self) -> None:
        """Run conservative checks before arming and autonomous motion."""
        if self._args.area_width_m <= 0.0 or self._args.area_height_m <= 0.0:
            raise ValueError("Search area dimensions must be positive")
        if self._args.row_spacing_m <= 0.0:
            raise ValueError("Row spacing must be positive")
        if self._args.row_spacing_m > self._args.area_height_m:
            raise ValueError("Row spacing should not exceed the search area height")
        if self._args.max_speed_m_s <= 0.0 or self._args.max_speed_m_s > 0.5:
            raise ValueError("Max horizontal speed must be within (0.0, 0.5] m/s for this prototype")
        if self._args.min_visible_markers < 1 or self._args.min_visible_markers > 4:
            raise ValueError("Minimum visible markers must be between 1 and 4")

        battery_wait_started = time.monotonic()
        while self._latest_battery_fraction is None and time.monotonic() - battery_wait_started < 5.0:
            await asyncio.sleep(0.1)
        if self._latest_battery_fraction is None:
            raise RuntimeError("Battery telemetry unavailable")
        if self._latest_battery_fraction < self._args.battery_min_fraction:
            raise RuntimeError(
                f"Battery too low for mission: {self._latest_battery_fraction:.2f} < {self._args.battery_min_fraction:.2f}"
            )

        frame = self.get_camera_frame()
        detections = self.detect_aruco_markers(frame)
        visible_ids = sorted(det.tag_id for det in detections)
        if not detections:
            raise RuntimeError("No ArUco markers visible during preflight check")
        print(f"Preflight visible markers: {visible_ids}")
        print(f"Planned waypoint count: {len(self._waypoints)}")

    async def arm_drone(self) -> None:
        """Arm the vehicle."""
        print("Arming drone")
        await self._drone.action.arm()

    async def takeoff_to_safe_height(self) -> None:
        """Take off to the configured scan altitude."""
        print(f"Setting takeoff altitude to {self._args.scan_altitude_m:.2f}m")
        await self._drone.action.set_takeoff_altitude(self._args.scan_altitude_m)
        print("Taking off")
        await self._drone.action.takeoff()
        await asyncio.sleep(self._args.takeoff_wait_s)

    async def start_offboard_mode(self) -> None:
        """Start PX4 Offboard mode after pushing an initial hold setpoint."""
        await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.2)
        try:
            await self._drone.offboard.start()
        except OffboardError as error:
            raise RuntimeError(f"Failed to start Offboard mode: {error}") from error
        self._offboard_started = True
        print("Offboard mode started")

    def get_camera_frame(self) -> np.ndarray:
        """Read and return the next camera frame."""
        if self._cap is None:
            raise RuntimeError("Camera stream is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera stream")
        return frame

    def detect_aruco_markers(self, frame: np.ndarray) -> list[ArucoDetection]:
        """Detect ArUco markers in one frame."""
        return self._detector.detect(frame)

    def estimate_drone_pose_from_markers(self, detections: list[ArucoDetection]) -> PoseEstimate:
        """Estimate the drone pose in world coordinates from visible markers."""
        return self._localizer.estimate_pose(detections)

    def get_next_waypoint(self, waypoint_index: int) -> Waypoint:
        """Return the next target waypoint."""
        return self._waypoints[waypoint_index]

    async def send_mavsdk_movement_command(self, pose: PoseEstimate, waypoint: Waypoint) -> None:
        """Compute and send a conservative body-frame velocity command."""
        if pose.body_position_world_m is None:
            raise RuntimeError("Body position unavailable from ArUco localizer")

        current_position = np.array(pose.body_position_world_m, dtype=np.float64)
        target_position = np.array([waypoint.x_m, waypoint.y_m, waypoint.z_m], dtype=np.float64)
        error_world = target_position - current_position

        desired_world_xy = self._args.kp_horizontal * error_world[:2]
        desired_world_speed = float(np.linalg.norm(desired_world_xy))
        if desired_world_speed > waypoint.max_speed_m_s and desired_world_speed > 1e-6:
            desired_world_xy *= waypoint.max_speed_m_s / desired_world_speed

        yaw_rad = _pose_rvec_to_yaw_rad(pose.body_rvec_world)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        rot_body_from_world = np.array(
            [
                [cos_yaw, sin_yaw],
                [-sin_yaw, cos_yaw],
            ],
            dtype=np.float64,
        )
        desired_body_xy = rot_body_from_world @ desired_world_xy

        desired_up_m_s = float(np.clip(self._args.kp_vertical * error_world[2], -self._args.vertical_speed_m_s, self._args.vertical_speed_m_s))
        forward_m_s = float(desired_body_xy[0])
        right_m_s = float(desired_body_xy[1])
        down_m_s = -desired_up_m_s

        await self._drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s=forward_m_s,
                right_m_s=right_m_s,
                down_m_s=down_m_s,
                yawspeed_deg_s=0.0,
            )
        )

    def waypoint_reached(self, pose: PoseEstimate, waypoint: Waypoint) -> bool:
        """Return whether the current estimated drone position is within waypoint tolerance."""
        if pose.body_position_world_m is None:
            return False

        dx = waypoint.x_m - pose.body_position_world_m[0]
        dy = waypoint.y_m - pose.body_position_world_m[1]
        dz = waypoint.z_m - pose.body_position_world_m[2]
        return math.hypot(dx, dy) <= waypoint.tolerance_xy_m and abs(dz) <= waypoint.tolerance_z_m

    async def hold_position(self, hold_time_s: float) -> None:
        """Send zero velocity commands for a short hold period."""
        deadline = time.monotonic() + hold_time_s
        hold_command = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        while time.monotonic() < deadline:
            await self._drone.offboard.set_velocity_body(hold_command)
            await asyncio.sleep(0.05)

    async def follow_waypoints(self) -> None:
        """Run the main detect-localize-move loop until the path is complete."""
        if len(self._waypoints) <= 1:
            print("No movement required; waypoint list contains only the start point")
            return

        mission_started = time.monotonic()
        marker_loss_started: Optional[float] = None
        waypoint_index = 1

        while waypoint_index < len(self._waypoints):
            if time.monotonic() - mission_started > self._args.max_mission_time_s:
                raise RuntimeError("Mission timeout exceeded")

            frame = self.get_camera_frame()
            detections = self.detect_aruco_markers(frame)

            mapped_detections = [det for det in detections if det.tag_id in {0, 1, 2, 3}]
            if len(mapped_detections) < self._args.min_visible_markers:
                await self.hold_position(0.1)
                if marker_loss_started is None:
                    marker_loss_started = time.monotonic()
                if time.monotonic() - marker_loss_started > self._args.marker_loss_timeout_s:
                    raise RuntimeError("Lost sufficient ArUco marker visibility for too long")
                self._maybe_show_frame(frame, detections, None, self.get_next_waypoint(waypoint_index))
                continue

            pose = self.estimate_drone_pose_from_markers(mapped_detections)
            marker_loss_started = None

            if not pose.success:
                await self.hold_position(0.1)
                self._maybe_show_frame(frame, detections, None, self.get_next_waypoint(waypoint_index))
                continue

            if pose.reprojection_error_px is not None and pose.reprojection_error_px > self._args.max_reprojection_error_px:
                await self.hold_position(0.1)
                self._maybe_show_frame(frame, detections, pose, self.get_next_waypoint(waypoint_index))
                continue

            target_waypoint = self.get_next_waypoint(waypoint_index)
            await self.send_mavsdk_movement_command(pose, target_waypoint)

            if self.waypoint_reached(pose, target_waypoint):
                reproj_text = (
                    f"{pose.reprojection_error_px:.2f}"
                    if pose.reprojection_error_px is not None
                    else "nan"
                )
                print(
                    f"Reached waypoint {waypoint_index + 1}/{len(self._waypoints)} "
                    f"target=({target_waypoint.x_m:.2f}, {target_waypoint.y_m:.2f}, {target_waypoint.z_m:.2f}) "
                    f"markers={pose.marker_ids} reproj={reproj_text}"
                )
                await self.hold_position(self._args.hold_time_s)
                waypoint_index += 1

            self._maybe_show_frame(frame, detections, pose, target_waypoint)
            await asyncio.sleep(0.05)

        print("Mission path complete")
        await self.hold_position(self._args.hold_time_s)

    async def stop_drone_safely(self) -> None:
        """Stop active movement without assuming mission success."""
        try:
            await self._drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        except Exception:
            pass

    async def land_or_disarm_safely(self) -> None:
        """Hold, stop offboard, and land or disarm safely."""
        await self.stop_drone_safely()

        if self._offboard_started:
            try:
                await self._drone.offboard.stop()
                print("Offboard mode stopped")
            except OffboardError as error:
                print(f"Offboard stop error: {error}")
            except Exception as error:
                print(f"Unexpected offboard stop error: {error}")

        try:
            print("Landing")
            await self._drone.action.land()
            await asyncio.sleep(8.0)
        except Exception as error:
            print(f"Landing failed: {error}")
            try:
                print("Attempting disarm")
                await self._drone.action.disarm()
            except Exception as disarm_error:
                print(f"Disarm failed: {disarm_error}")

    def close_camera_stream(self) -> None:
        """Release the camera stream and close preview windows."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()

    async def close_mavlink_connection(self) -> None:
        """Cancel background telemetry tasks."""
        for task in self._telemetry_tasks:
            task.cancel()
        if self._telemetry_tasks:
            await asyncio.gather(*self._telemetry_tasks, return_exceptions=True)
        self._telemetry_tasks.clear()

    async def _stream_battery(self) -> None:
        """Cache battery telemetry for preflight checks."""
        try:
            async for battery in self._drone.telemetry.battery():
                self._latest_battery_fraction = float(getattr(battery, "remaining_percent", 0.0))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Battery telemetry stream stopped: {error}")

    def _maybe_show_frame(
        self,
        frame: np.ndarray,
        detections: list[ArucoDetection],
        pose: Optional[PoseEstimate],
        target_waypoint: Waypoint,
    ) -> None:
        """Optionally display the current frame with basic overlays."""
        if not self._args.show:
            return

        vis = frame.copy()
        for det in detections:
            pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"ID {det.tag_id}",
                (int(det.center_x) + 6, int(det.center_y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        lines = [
            f"target xyz: {target_waypoint.x_m:.2f}, {target_waypoint.y_m:.2f}, {target_waypoint.z_m:.2f} m",
            f"visible markers: {[det.tag_id for det in detections]}",
        ]
        if pose is not None and pose.success and pose.body_position_world_m is not None:
            bx, by, bz = pose.body_position_world_m
            lines.append(f"drone xyz: {bx:.2f}, {by:.2f}, {bz:.2f} m")
            if pose.reprojection_error_px is not None:
                lines.append(f"reprojection: {pose.reprojection_error_px:.2f} px")
        else:
            lines.append("pose unavailable")

        y_px = 24
        for line in lines:
            cv2.putText(vis, line, (12, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            y_px += 24

        cv2.imshow("ArUco Grid Demo", vis)
        cv2.waitKey(1)


async def run_demo(args: argparse.Namespace) -> None:
    """Run the full mission with safe try/except/finally semantics."""
    mission = ArucoGridMission(args)

    try:
        await mission.connect_to_pixhawk()
        mission.open_camera_stream()
        await mission.run_preflight_checks()
        await mission.arm_drone()
        await mission.takeoff_to_safe_height()
        await mission.start_offboard_mode()
        await mission.follow_waypoints()

    except KeyboardInterrupt:
        await mission.stop_drone_safely()
        print("Mission manually interrupted.")

    except Exception as error:
        await mission.stop_drone_safely()
        print(f"Mission failed: {error}")

    finally:
        await mission.land_or_disarm_safely()
        mission.close_camera_stream()
        await mission.close_mavlink_connection()
        print("Mission terminated safely.")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
