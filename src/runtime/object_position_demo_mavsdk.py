"""MAVSDK + YOLO demo that estimates detected object positions on the ground plane."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.stages.yolo_detector import YoloDetection, YoloDetector
from src.vision.object_geolocator import (
    ObjectPositionEstimate,
    TelemetryPoseNed,
    detection_reference_pixel,
    estimate_object_ground_position,
    rotation_matrix_from_euler_deg,
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
    parser.add_argument("--calibration-file", required=True, help="NPZ file containing camera_matrix and dist_coeffs")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="Path to YOLO model")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class labels file")
    parser.add_argument("--yolo-input", type=int, default=640, help="YOLO square input size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold")
    parser.add_argument("--target-label", action="append", default=[], help="Optional object label to keep; may be passed multiple times")
    parser.add_argument("--ground-down-m", type=float, default=0.0, help="Ground plane expressed in local NED down meters")
    parser.add_argument("--max-range-m", type=float, default=25.0, help="Ignore estimated hits farther than this slant range")
    parser.add_argument("--camera-roll-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: roll")
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: pitch")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0, help="Rotation from drone body frame to camera frame: yaw")
    parser.add_argument("--camera-offset-x-m", type=float, default=0.0, help="Camera offset from drone body origin in body forward")
    parser.add_argument("--camera-offset-y-m", type=float, default=0.0, help="Camera offset from drone body origin in body right")
    parser.add_argument("--camera-offset-z-m", type=float, default=0.0, help="Camera offset from drone body origin in body down")
    parser.add_argument("--show", action="store_true", help="Show the annotated preview window")
    parser.add_argument("--log-interval-s", type=float, default=1.0, help="Minimum seconds between terminal reports")
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


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Generate a stable pseudo-random color for a class id."""
    return (
        (37 * class_id + 80) % 255,
        (17 * class_id + 160) % 255,
        (29 * class_id + 220) % 255,
    )


class ObjectPositionDemo:
    """Detect objects and estimate their ground positions from MAVLink telemetry."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._drone = System()
        self._cap: Optional[cv2.VideoCapture] = None
        self._telemetry_tasks: list[asyncio.Task[None]] = []
        self._latest_pose: Optional[TelemetryPoseNed] = None
        self._latest_battery_fraction: Optional[float] = None
        self._last_log_s = 0.0

        self._camera_matrix, self._dist_coeffs = load_camera_calibration(args.calibration_file)
        self._detector = YoloDetector(
            model_path=args.yolo_model,
            classes_path=args.yolo_classes,
            input_size=args.yolo_input,
            conf_thresh=args.yolo_conf,
            nms_thresh=args.yolo_nms,
        )
        self._body_to_camera_rotation = rotation_matrix_from_euler_deg(
            args.camera_roll_deg,
            args.camera_pitch_deg,
            args.camera_yaw_deg,
        )
        self._camera_offset_body_m = np.array(
            [args.camera_offset_x_m, args.camera_offset_y_m, args.camera_offset_z_m],
            dtype=np.float64,
        )
        # Optional label whitelist so geolocation can focus on mission-relevant classes.
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
        """Wait for enough telemetry to produce geolocation estimates."""
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

    def estimate_positions(self, detections: list[YoloDetection]) -> list[ObjectPositionEstimate]:
        """Estimate ground positions for the currently detected objects."""
        pose = self._latest_pose
        if pose is None:
            return []

        estimates: list[ObjectPositionEstimate] = []
        for detection in detections:
            estimate = estimate_object_ground_position(
                detection=detection,
                pose=pose,
                camera_matrix=self._camera_matrix,
                body_to_camera_rotation=self._body_to_camera_rotation,
                dist_coeffs=self._dist_coeffs,
                camera_offset_body_m=self._camera_offset_body_m,
                ground_down_m=self._args.ground_down_m,
                max_range_m=self._args.max_range_m,
            )
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def draw_overlay(
        self,
        frame: np.ndarray,
        detections: list[YoloDetection],
        estimates: list[ObjectPositionEstimate],
    ) -> np.ndarray:
        """Render detections plus estimated ground positions."""
        vis = frame.copy()
        estimates_by_label_and_pixel = {
            (estimate.label, round(estimate.image_u_px, 1), round(estimate.image_v_px, 1)): estimate
            for estimate in estimates
        }

        for det in detections:
            color = _color_for_class(det.class_id)
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            ref_u_px, ref_v_px = detection_reference_pixel(det)
            estimate = estimates_by_label_and_pixel.get((det.label, round(ref_u_px, 1), round(ref_v_px, 1)))
            label = f"{det.label} {det.confidence:.2f}"
            if estimate is not None:
                label += f" N={estimate.north_m:.1f} E={estimate.east_m:.1f}"
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
        if pose is not None:
            lines = [
                f"NED: {pose.north_m:.2f}, {pose.east_m:.2f}, {pose.down_m:.2f}",
                f"RPY: {pose.roll_deg:.1f}, {pose.pitch_deg:.1f}, {pose.yaw_deg:.1f}",
                f"objects: {len(estimates)}/{len(detections)}",
            ]
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
            detections = self.detect_objects(frame)
            estimates = self.estimate_positions(detections)

            now = time.monotonic()
            if now - self._last_log_s >= self._args.log_interval_s:
                self._log_status(detections, estimates)
                self._last_log_s = now

            if self._args.show:
                vis = self.draw_overlay(frame, detections, estimates)
                cv2.imshow("MAVSDK Object Position Demo", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            await asyncio.sleep(0.01)

    def _log_status(
        self,
        detections: list[YoloDetection],
        estimates: list[ObjectPositionEstimate],
    ) -> None:
        """Print a short textual status update."""
        pose = self._latest_pose
        if pose is None:
            print("telemetry=waiting")
            return

        if not detections:
            print(
                f"telemetry_ned=({pose.north_m:.2f},{pose.east_m:.2f},{pose.down_m:.2f}) "
                "detections=0"
            )
            return

        estimate_strings = [
            (
                f"{estimate.label}:{estimate.confidence:.2f} "
                f"north={estimate.north_m:.2f} east={estimate.east_m:.2f} "
                f"down={estimate.down_m:.2f} range={estimate.slant_range_m:.2f}"
            )
            for estimate in estimates[:8]
        ]
        if estimate_strings:
            print("object_positions " + " | ".join(estimate_strings))
            return

        summary = ", ".join(f"{det.label}:{det.confidence:.2f}" for det in detections[:8])
        print(f"detections_without_ground_intersection {summary}")

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
                # Preserve most recent attitude fields while replacing position.
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
                # Preserve most recent position fields while replacing attitude.
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
