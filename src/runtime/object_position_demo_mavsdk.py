"""MAVSDK + YOLO demo that estimates object floor positions with ArUco fallback."""
from __future__ import annotations
import argparse
import asyncio
import time
from typing import Optional
import cv2
import numpy as np
from src.localization import ArucoWorldLocalizer, MarkerMap, PoseEstimate
from src.stages.aruco_detector import ArucoDetection, ArucoDetector
from src.stages.yolo_detector import YoloDetection, YoloDetector
from src.vision.object_geolocator import (
    ObjectPositionEstimate,
    TelemetryPoseNed,
    pixel_to_camera_ray,
)
try:
    from mavsdk import System
except ImportError as exc:  # pragma: no cover - dependency is optional in this environment
    raise SystemExit(
        "MAVSDK is required for this demo. Install it with `pip install mavsdk`."
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse runtime options for the object-position demo."""
    parser = argparse.ArgumentParser(description="MAVSDK YOLO object-position demo")
    parser.add_argument("--system-address", default="serial:///dev/ttyAMA0:921600", help="MAVSDK system address")
    parser.add_argument("--camera-source", default="0", help="OpenCV camera source index, device path, video file, or stream URL")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="Path to YOLO model")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class labels file")
    parser.add_argument("--yolo-input", type=int, default=640, help="YOLO square input size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold")
    parser.add_argument("--target-label", action="append", default=[], help="Optional object label to keep; may be passed multiple times")
    parser.add_argument("--aruco-family", default="6x6_250", help="ArUco family for floor markers")
    parser.add_argument("--marker-length-m", type=float, default=0.267, help="Printed marker edge length in meters")
    parser.add_argument("--area-width-m", type=float, default=0.60, help="Distance from marker 0 center to marker 1 center")
    parser.add_argument("--area-height-m", type=float, default=0.60, help="Distance from marker 0 center to marker 2 center")
    parser.add_argument("--camera-fx", type=float, default=1421.1369082868994, help="Camera focal length fx in pixels")
    parser.add_argument("--camera-fy", type=float, default=1417.6988685113936, help="Camera focal length fy in pixels")
    parser.add_argument("--camera-cx", type=float, default=614.0247919076297, help="Camera principal point cx in pixels")
    parser.add_argument("--camera-cy", type=float, default=341.55448642330805, help="Camera principal point cy in pixels")
    parser.add_argument("--smoothing-alpha", type=float, default=0.2, help="ArUco pose smoothing; 0 disables smoothing")
    parser.add_argument("--show", action="store_true", help="Show the annotated preview window")
    parser.add_argument("--log-interval-s", type=float, default=1.0, help="Minimum seconds between terminal reports")
    return parser.parse_args()


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Generate a stable pseudo-random color for a class id."""
    return (
        (37 * class_id + 80) % 255,
        (17 * class_id + 160) % 255,
        (29 * class_id + 220) % 255,
    )


def _object_floor_reference_pixel(det: YoloDetection) -> tuple[float, float]:
    """Use the object's bottom point because floor projection targets ground contact."""
    if det.mask is not None and det.mask.any():
        ys, xs = np.nonzero(det.mask)
        bottom_y = float(np.max(ys))
        bottom_xs = xs[ys == int(bottom_y)]
        return float(np.mean(bottom_xs)), bottom_y

    return float(det.x1 + det.x2) * 0.5, float(det.y2)


def _rotation_matrix_from_rvec(rvec: tuple[float, float, float]) -> np.ndarray:
    """Convert a Rodrigues vector tuple back into a rotation matrix."""
    rotation, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
    return rotation


class ObjectPositionDemo:
    """Detect objects and prefer ArUco floor projection over Pixhawk pose fallback."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._drone = System()
        self._cap: Optional[cv2.VideoCapture] = None
        self._telemetry_tasks: list[asyncio.Task[None]] = []
        self._latest_pose: Optional[TelemetryPoseNed] = None
        self._latest_battery_fraction: Optional[float] = None
        self._last_log_s = 0.0

        self._camera_matrix = np.array(
            [
                [args.camera_fx, 0.0, args.camera_cx],
                [0.0, args.camera_fy, args.camera_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._dist_coeffs = np.zeros(5, dtype=np.float32)
        self._detector = YoloDetector(
            model_path=args.yolo_model,
            classes_path=args.yolo_classes,
            input_size=args.yolo_input,
            conf_thresh=args.yolo_conf,
            nms_thresh=args.yolo_nms,
        )
        self._aruco_detector = ArucoDetector(
            family=args.aruco_family,
            marker_length_m=args.marker_length_m,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
        )
        marker_map = MarkerMap.rectangular_floor_map(
            marker_length_m=args.marker_length_m,
            area_width_m=args.area_width_m,
            area_height_m=args.area_height_m,
        )
        self._aruco_localizer = ArucoWorldLocalizer(
            marker_map=marker_map,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            smoothing_alpha=args.smoothing_alpha,
        )
        # Optional label whitelist so the demo can focus on mission-relevant classes.
        self._target_labels = {label.strip() for label in args.target_label if label.strip()}

    async def connect_to_pixhawk(self) -> None:
        """Connect to MAVSDK and start telemetry subscriptions."""
        await self._drone.connect(system_address=self._args.system_address)
        print(f"Connecting to Pixhawk via {self._args.system_address}")
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                print("Connected to Pixhawk")
                break

        self._telemetry_tasks.append(asyncio.create_task(self._stream_position_velocity_ned()))
        self._telemetry_tasks.append(asyncio.create_task(self._stream_attitude_euler()))
        self._telemetry_tasks.append(asyncio.create_task(self._stream_battery()))

    def open_camera_stream(self) -> None:
        """Open the configured camera source."""
        source: int | str
        if self._args.camera_source.isdigit():
            source = int(self._args.camera_source)
        else:
            source = self._args.camera_source

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._args.camera_source}")
        print(
            f"Opened camera source {self._args.camera_source} "
            f"with model={self._args.yolo_model} backend={self._detector.backend} device={self._detector.device}"
        )

    async def run_preflight_checks(self) -> None:
        """Wait for enough telemetry to tag detections with drone pose."""
        # Ensure both position and attitude streams have produced at least one sample.
        started = time.monotonic()
        while self._latest_pose is None and time.monotonic() - started < 5.0:
            await asyncio.sleep(0.1)
        if self._latest_pose is None:
            raise RuntimeError("Timed out waiting for MAVLink local position + attitude")

        battery_started = time.monotonic()
        while self._latest_battery_fraction is None and time.monotonic() - battery_started < 5.0:
            await asyncio.sleep(0.1)
        if self._latest_battery_fraction is None:
            print("Battery telemetry unavailable; continuing without battery status")

        frame = self.get_camera_frame()
        _ = self._detector.detect(frame)
        print("Preflight checks complete")

    def get_camera_frame(self) -> np.ndarray:
        """Read and return the next camera frame."""
        if self._cap is None:
            raise RuntimeError("Camera stream is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera stream")
        return frame

    def detect_objects(self, frame: np.ndarray) -> list[YoloDetection]:
        """Run YOLO object detection on one frame."""
        detections = self._detector.detect(frame)
        if not self._target_labels:
            return detections
        # Keep only configured labels when a filter is provided.
        return [det for det in detections if det.label in self._target_labels]

    def detect_aruco_markers(self, frame: np.ndarray) -> list[ArucoDetection]:
        """Detect floor markers used as the metric reference for object positions."""
        return self._aruco_detector.detect(frame)

    def estimate_aruco_pose(self, markers: list[ArucoDetection]) -> PoseEstimate:
        """Estimate camera pose in the marker-defined floor frame."""
        return self._aruco_localizer.estimate_pose(markers)

    def estimate_positions(self, detections: list[YoloDetection], aruco_pose: Optional[PoseEstimate]) -> list[ObjectPositionEstimate]:
        """Estimate object floor positions from ArUco pose, or fall back to Pixhawk pose."""
        estimates = self._estimate_positions_from_aruco(detections, aruco_pose)
        if estimates:
            return estimates

        pose = self._latest_pose
        if pose is None:
            return []

        estimates = []
        for detection in detections:
            # If no floor-marker pose is available, keep the older behavior:
            # associate each detection with the drone's current Pixhawk NED pose.
            estimates.append(
                ObjectPositionEstimate(
                    label=detection.label,
                    confidence=detection.confidence,
                    image_u_px=0.0,
                    image_v_px=0.0,
                    north_m=pose.north_m,
                    east_m=pose.east_m,
                    down_m=pose.down_m,
                    slant_range_m=0.0,
                )
            )
        return estimates

    def _estimate_positions_from_aruco(self, detections: list[YoloDetection], aruco_pose: Optional[PoseEstimate]) -> list[ObjectPositionEstimate]:
        """Project YOLO detection bottom points onto the floor plane using ArUco pose."""
        if aruco_pose is None or not aruco_pose.success:
            return []

        camera_position_world = np.array(aruco_pose.camera_position_world_m, dtype=np.float64)
        rot_camera_to_world = _rotation_matrix_from_rvec(aruco_pose.camera_rvec_world)
        estimates: list[ObjectPositionEstimate] = []

        for detection in detections:
            image_u_px, image_v_px = _object_floor_reference_pixel(detection)
            ray_camera = pixel_to_camera_ray(
                image_u_px,
                image_v_px,
                self._camera_matrix,
                dist_coeffs=self._dist_coeffs,
            )
            ray_world = rot_camera_to_world @ ray_camera
            z_component = float(ray_world[2])
            if abs(z_component) <= 1e-9:
                continue

            distance_along_ray = -float(camera_position_world[2]) / z_component
            if distance_along_ray <= 0.0:
                continue

            object_position_world = camera_position_world + distance_along_ray * ray_world
            slant_range_m = float(np.linalg.norm(object_position_world - camera_position_world))
            estimates.append(
                ObjectPositionEstimate(
                    label=detection.label,
                    confidence=detection.confidence,
                    image_u_px=image_u_px,
                    image_v_px=image_v_px,
                    north_m=float(object_position_world[0]),
                    east_m=float(object_position_world[1]),
                    down_m=float(-object_position_world[2]),
                    slant_range_m=slant_range_m,
                )
            )

        return estimates

    def draw_overlay(self, frame: np.ndarray, detections: list[YoloDetection], estimates: list[ObjectPositionEstimate], markers: list[ArucoDetection], aruco_pose: Optional[PoseEstimate]) -> np.ndarray:
        """Render detections plus the drone pose associated with each object."""
        vis = frame.copy()

        for marker in markers:
            pts = np.array(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"ID {marker.tag_id}",
                (int(marker.center_x) + 6, int(marker.center_y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        for index, det in enumerate(detections):
            color = _color_for_class(det.class_id)
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            # Estimates are kept in the same order as detections, so we can
            # pair them by list index without any pixel-based matching.
            estimate = estimates[index] if index < len(estimates) else None
            label = f"{det.label} {det.confidence:.2f}"
            if estimate is not None:
                label += f" X={estimate.north_m:.1f} Y={estimate.east_m:.1f}"
                if estimate.slant_range_m > 0.0:
                    cv2.circle(vis, (int(estimate.image_u_px), int(estimate.image_v_px)), 5, color, -1)
            cv2.putText(
                vis,
                label,
                (det.x1, max(24, det.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        pose = self._latest_pose
        lines = [f"markers: {[marker.tag_id for marker in markers]}"]
        if aruco_pose is not None and aruco_pose.success:
            cam_x, cam_y, cam_z = aruco_pose.camera_position_world_m
            lines.append(f"aruco camera xyz: {cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}")
        elif pose is not None:
            lines.append(f"fallback NED: {pose.north_m:.2f}, {pose.east_m:.2f}, {pose.down_m:.2f}")
        lines.append(f"objects: {len(estimates)}/{len(detections)}")
        for index, text in enumerate(lines):
            cv2.putText(
                vis,
                text,
                (20, 30 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        return vis

    async def run(self) -> None:
        """Run the main detection and object-position loop."""
        print("Starting object-position demo")
        while True:
            frame = self.get_camera_frame()
            markers = self.detect_aruco_markers(frame)
            aruco_pose = self.estimate_aruco_pose(markers)
            detections = self.detect_objects(frame)
            estimates = self.estimate_positions(detections, aruco_pose)

            now = time.monotonic()
            if now - self._last_log_s >= self._args.log_interval_s:
                self._log_status(detections, estimates, markers, aruco_pose)
                self._last_log_s = now

            if self._args.show:
                vis = self.draw_overlay(frame, detections, estimates, markers, aruco_pose)
                cv2.imshow("MAVSDK Object Position Demo", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            await asyncio.sleep(0.01)

    def _log_status(self, detections: list[YoloDetection], estimates: list[ObjectPositionEstimate], markers: list[ArucoDetection], aruco_pose: Optional[PoseEstimate]) -> None:
        """Print a short textual status update."""
        pose = self._latest_pose
        if pose is None and not (aruco_pose is not None and aruco_pose.success):
            print("telemetry=waiting")
            return

        if not detections:
            marker_ids = [marker.tag_id for marker in markers]
            print(f"markers={marker_ids} detections=0")
            return

        source = "aruco_floor" if aruco_pose is not None and aruco_pose.success and any(estimate.slant_range_m > 0.0 for estimate in estimates) else "drone_pose"

        estimate_strings = [
            (
                f"{estimate.label}:{estimate.confidence:.2f} "
                f"x={estimate.north_m:.2f} y={estimate.east_m:.2f} "
                f"z={-estimate.down_m:.2f} range={estimate.slant_range_m:.2f} source={source}"
            )
            for estimate in estimates[:8]
        ]
        if estimate_strings:
            print("object_positions " + " | ".join(estimate_strings))
            return

        summary = ", ".join(f"{det.label}:{det.confidence:.2f}" for det in detections[:8])
        print(f"detections_without_pose {summary}")

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

    async def _stream_position_velocity_ned(self) -> None:
        """Background task: cache local NED position."""
        try:
            async for sample in self._drone.telemetry.position_velocity_ned():
                position = getattr(sample, "position", None)
                if position is None:
                    continue
                current = self._latest_pose
                # MAVSDK publishes position and attitude on separate streams.
                # Keep the most recent attitude values while refreshing only
                # the N/E/D fields from this position update.
                self._latest_pose = TelemetryPoseNed(
                    north_m=float(getattr(position, "north_m", 0.0)),
                    east_m=float(getattr(position, "east_m", 0.0)),
                    down_m=float(getattr(position, "down_m", 0.0)),
                    roll_deg=current.roll_deg if current is not None else 0.0,
                    pitch_deg=current.pitch_deg if current is not None else 0.0,
                    yaw_deg=current.yaw_deg if current is not None else 0.0,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"NED telemetry stream stopped: {error}")

    async def _stream_attitude_euler(self) -> None:
        """Background task: cache attitude angles."""
        try:
            async for attitude in self._drone.telemetry.attitude_euler():
                current = self._latest_pose
                # Merge this attitude sample into the latest cached pose
                # without overwriting the most recent N/E/D position sample.
                self._latest_pose = TelemetryPoseNed(
                    north_m=current.north_m if current is not None else 0.0,
                    east_m=current.east_m if current is not None else 0.0,
                    down_m=current.down_m if current is not None else 0.0,
                    roll_deg=float(getattr(attitude, "roll_deg", 0.0)),
                    pitch_deg=float(getattr(attitude, "pitch_deg", 0.0)),
                    yaw_deg=float(getattr(attitude, "yaw_deg", 0.0)),
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Attitude telemetry stream stopped: {error}")

    async def _stream_battery(self) -> None:
        """Background task: cache battery state."""
        try:
            async for battery in self._drone.telemetry.battery():
                self._latest_battery_fraction = float(getattr(battery, "remaining_percent", 0.0))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"Battery telemetry stream stopped: {error}")


async def _async_main() -> None:
    """Create, run, and tear down the demo cleanly."""
    args = parse_args()
    demo = ObjectPositionDemo(args)
    try:
        await demo.connect_to_pixhawk()
        demo.open_camera_stream()
        await demo.run_preflight_checks()
        await demo.run()
    finally:
        demo.close_camera_stream()
        await demo.close_mavlink_connection()


def main() -> None:
    """Entrypoint for `python -m` execution."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\nStopping object-position demo")


if __name__ == "__main__":
    main()
