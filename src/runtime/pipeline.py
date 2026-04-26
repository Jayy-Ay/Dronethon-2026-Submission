"""PC runtime pipeline: receive Pi frames and run ArUco + YOLO in separate threads."""

from __future__ import annotations
import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, List, Optional, TypeVar
import cv2
import numpy as np
from src.stages.aruco_detector import ArucoDetection, ArucoDetector
from src.stages.yolo_detector import YoloDetection, YoloDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


T = TypeVar("T")


@dataclass(frozen=True)
class FrameTask:
    """A frame tagged with a monotonically increasing ID."""

    frame_id: int
    frame: np.ndarray


class DetectorWorker(Generic[T]):
    """Runs a detector continuously in its own thread on latest submitted frames."""

    def __init__(
        self,
        name: str,
        detect_fn: Callable[[np.ndarray], List[T]],
        rate_hz: float,
        on_result: Optional[Callable[[int, np.ndarray, List[T]], None]] = None,
    ) -> None:
        self._name = name
        self._detect_fn = detect_fn
        self._on_result = on_result
        self._rate_hz = max(rate_hz, 0.1)

        self._queue: "queue.Queue[FrameTask]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"detector-{name}")

        self._latest: List[T] = []
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Start the detector worker thread."""
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown and wait briefly for the worker to exit."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def submit(self, frame_id: int, frame: np.ndarray) -> None:
        """Queue the newest frame, replacing any stale pending task."""
        task = FrameTask(frame_id=frame_id, frame=frame)
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(task)

    def latest(self) -> List[T]:
        """Return a snapshot of the most recent successful detector output."""
        with self._lock:
            return list(self._latest)

    def last_error(self) -> Optional[str]:
        """Return the latest detector exception string, if any."""
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        """Consume queued frames, run detection, and rate-limit the loop."""
        interval = 1.0 / self._rate_hz
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            start = time.time()
            try:
                result = self._detect_fn(task.frame)
                with self._lock:
                    self._latest = result
                    self._last_error = None
                if self._on_result is not None:
                    self._on_result(task.frame_id, task.frame, result)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{self._name}: {exc}"

            elapsed = time.time() - start
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)


@dataclass
class FrameCache:
    """Thread-safe cache of the latest frame plus freshest detector results."""
    frame: Optional[np.ndarray] = None
    aruco_dets: List[ArucoDetection] = field(default_factory=list)
    yolo_dets: List[YoloDetection] = field(default_factory=list)
    latest_frame_id: int = -1
    _version: int = 0
    _closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _updated: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        """Create the condition variable after the lock dataclass field exists."""
        self._updated = threading.Condition(self._lock)

    def publish_aruco(self, frame_id: int, frame: np.ndarray, aruco_dets: List[ArucoDetection]) -> None:
        """Store ArUco results and publish the freshest available combined view."""
        self._publish_result(frame_id, frame, "aruco", aruco_dets)

    def publish_yolo(self, frame_id: int, frame: np.ndarray, yolo_dets: List[YoloDetection]) -> None:
        """Store YOLO results and publish the freshest available combined view."""
        self._publish_result(frame_id, frame, "yolo", yolo_dets)

    def _publish_result(self, frame_id: int, frame: np.ndarray, detector: str, detections: List[object]) -> None:
        """Publish the newest frame immediately while keeping the latest results per detector."""
        with self._updated:
            if frame_id <= self.latest_frame_id:
                return

            if detector == "aruco":
                self.aruco_dets = list(detections) if detections else []
            elif detector == "yolo":
                self.yolo_dets = list(detections) if detections else []
            else:
                raise ValueError(f"Unsupported detector '{detector}'")

            self.frame = frame
            self.latest_frame_id = frame_id
            self._version += 1
            self._updated.notify_all()

    def get_latest(self) -> tuple[Optional[np.ndarray], List[ArucoDetection], List[YoloDetection]]:
        """Get current cached frame and detections."""
        with self._lock:
            return self.frame, list(self.aruco_dets), list(self.yolo_dets)

    def wait_for_update(
        self,
        last_version: int,
        timeout: float = 0.1,
    ) -> tuple[int, Optional[np.ndarray], List[ArucoDetection], List[YoloDetection]]:
        """Wait for a new processed frame to be published."""
        deadline = time.time() + timeout
        with self._updated:
            while self._version <= last_version and not self._closed:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._updated.wait(timeout=remaining)

            return self._version, self.frame, list(self.aruco_dets), list(self.yolo_dets)

    def close(self) -> None:
        with self._updated:
            self._closed = True
            self._updated.notify_all()


class DisplayWorker:
    """Runs display in its own thread, pulling latest frame and detections from cache."""

    def __init__(self, cache: FrameCache) -> None:
        self._cache = cache
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="display")

    def start(self) -> None:
        """Start the display thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the display loop and wake any thread blocked on frame updates."""
        self._stop.set()
        self._cache.close()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        """Display loop: render immediately when a newly processed frame arrives."""
        last_version = 0
        while not self._stop.is_set():
            version, frame, aruco_dets, yolo_dets = self._cache.wait_for_update(
                last_version,
                timeout=0.05,
            )
            if version == last_version:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    self._stop.set()
                    break
                continue

            last_version = version
            if frame is None:
                continue

            vis = draw_overlays(frame, aruco_dets, yolo_dets)
            cv2.imshow("PC AI Pipeline", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                self._stop.set()
                break


@dataclass(frozen=True)
class Args:
    bind_ip: str
    video_port: int
    rtsp_url: Optional[str]
    rtsp_width: int
    rtsp_height: int
    frame_timeout: float
    max_no_frame_seconds: float
    family: str
    marker_length_m: float
    camera_fx: float
    camera_fy: float
    camera_cx: float
    camera_cy: float
    show: bool
    aruco_rate: float
    yolo_rate: float
    yolo_model: str
    yolo_classes: str
    yolo_input: int
    yolo_conf: float
    yolo_nms: float


def parse_args() -> Args:
    """Parse CLI flags into the strongly typed Args dataclass."""
    parser = argparse.ArgumentParser(description="Receive Pi stream and run threaded ArUco + YOLO on PC")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="PC IP to bind for incoming video")
    parser.add_argument("--video-port", type=int, default=5600, help="UDP port to receive video")
    parser.add_argument("--rtsp-url", default=None, help="Optional RTSP camera URL from the Pi/go2rtc feed")
    parser.add_argument("--rtsp-width", type=int, default=1280, help="Width to request from ffmpeg RTSP stream")
    parser.add_argument("--rtsp-height", type=int, default=720, help="Height to request from ffmpeg RTSP stream")
    parser.add_argument("--frame-timeout", type=float, default=3.0, help="Seconds to wait for a frame before retrying")
    parser.add_argument(
        "--max-no-frame-seconds",
        type=float,
        default=30.0,
        help="Exit if no frame is received for this many seconds",
    )
    parser.add_argument("--family", default="tag36h11", help="ArUco/AprilTag family")
    parser.add_argument("--marker-length-m", type=float, default=0.1, help="Physical ArUco marker edge length in meters")
    parser.add_argument("--camera-fx", type=float, default=0.0, help="Camera focal length fx in pixels; 0 uses an approximate model")
    parser.add_argument("--camera-fy", type=float, default=0.0, help="Camera focal length fy in pixels; 0 uses an approximate model")
    parser.add_argument("--camera-cx", type=float, default=0.0, help="Camera principal point cx in pixels; 0 uses image center")
    parser.add_argument("--camera-cy", type=float, default=0.0, help="Camera principal point cy in pixels; 0 uses image center")
    parser.add_argument("--show", action="store_true", help="Show annotated local preview")

    parser.add_argument("--aruco-rate", type=float, default=30.0, help="ArUco detection rate (Hz)")
    parser.add_argument("--yolo-rate", type=float, default=20.0, help="YOLO detection rate (Hz)")

    parser.add_argument("--yolo-model", default="yolov8s.pt", help="Path to YOLO model on PC (.pt for CUDA via Ultralytics or .onnx for OpenCV DNN)")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class names file on PC")
    parser.add_argument("--yolo-input", type=int, default=416, help="YOLO square input size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold")

    ns = parser.parse_args()
    return Args(
        bind_ip=ns.bind_ip,
        video_port=ns.video_port,
        rtsp_url=ns.rtsp_url,
        rtsp_width=ns.rtsp_width,
        rtsp_height=ns.rtsp_height,
        frame_timeout=ns.frame_timeout,
        max_no_frame_seconds=ns.max_no_frame_seconds,
        family=ns.family,
        marker_length_m=ns.marker_length_m,
        camera_fx=ns.camera_fx,
        camera_fy=ns.camera_fy,
        camera_cx=ns.camera_cx,
        camera_cy=ns.camera_cy,
        show=ns.show,
        aruco_rate=ns.aruco_rate,
        yolo_rate=ns.yolo_rate,
        yolo_model=ns.yolo_model,
        yolo_classes=ns.yolo_classes,
        yolo_input=ns.yolo_input,
        yolo_conf=ns.yolo_conf,
        yolo_nms=ns.yolo_nms,
    )


def draw_overlays(frame: np.ndarray, aruco_dets: List[ArucoDetection], yolo_dets: List[YoloDetection]) -> np.ndarray:
    """Draw ArUco poses and YOLO boxes onto a copy of the current frame."""
    vis = frame.copy()

    for det in aruco_dets:
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        x = int(det.center_x)
        y = int(det.center_y)
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        label = f"ARUCO {det.tag_id} area={det.area_px:.0f}"
        if det.distance_m is not None:
            label += f" dist={det.distance_m:.2f}m"
        cv2.putText(
            vis,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        if det.tvec_m is not None:
            tx, ty, tz = det.tvec_m
            cv2.putText(
                vis,
                f"x={tx:.2f} y={ty:.2f} z={tz:.2f} m",
                (x + 8, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
            )

    for det in yolo_dets:
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (255, 200, 0), 2)
        cv2.putText(
            vis,
            f"YOLO {det.label} {det.confidence:.2f}",
            (det.x1, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
        )

    return vis



def main() -> None:
    """Run the threaded PC-side detection pipeline until interrupted."""
    args = parse_args()

    # Stage 1: Choose the upstream frame source, either RTSP or the UDP stream from the Pi.
    if args.rtsp_url:
        provider = RtspFrameProvider(args.rtsp_url, width=args.rtsp_width, height=args.rtsp_height)
        print(f"Receiving RTSP frames from {args.rtsp_url}")
    else:
        provider = StreamFrameProvider(ip=args.bind_ip, port=args.video_port)
        print(f"Listening for UDP video stream on {args.bind_ip}:{args.video_port}")

    # Stage 2: Build camera intrinsics for pose estimation when calibration values are available.
    camera_matrix = None
    if args.camera_fx > 0 and args.camera_fy > 0:
        cx = args.camera_cx if args.camera_cx > 0 else args.rtsp_width / 2.0
        cy = args.camera_cy if args.camera_cy > 0 else args.rtsp_height / 2.0
        camera_matrix = np.array(
            [[args.camera_fx, 0.0, cx], [0.0, args.camera_fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    # Stage 3: Construct the ArUco detector, which is always enabled in this pipeline.
    aruco_detector = ArucoDetector(
        family=args.family,
        marker_length_m=args.marker_length_m,
        camera_matrix=camera_matrix,
    )

    # Stage 4: Optionally enable YOLO if the model assets are present on disk.
    yolo_detector = None
    yolo_model_path = Path(args.yolo_model)
    yolo_classes_path = Path(args.yolo_classes)
    if yolo_model_path.exists() and yolo_classes_path.exists():
        yolo_detector = YoloDetector(
            model_path=str(yolo_model_path),
            classes_path=str(yolo_classes_path),
            input_size=args.yolo_input,
            conf_thresh=args.yolo_conf,
            nms_thresh=args.yolo_nms,
        )
        print(
            f"YOLO inference backend: {yolo_detector.backend} "
            f"device: {yolo_detector.device}"
        )
    else:
        print(
            "YOLO disabled: missing model/classes file. "
            f"model={yolo_model_path} classes={yolo_classes_path}"
        )

    # Stage 5: Cache the freshest frame plus the most recent outputs from each detector.
    cache = FrameCache()

    # Stage 6: Start the dedicated detector threads so inference runs off the main ingest loop.
    aruco_worker = DetectorWorker(
        "aruco",
        aruco_detector.detect,
        rate_hz=args.aruco_rate,
        on_result=cache.publish_aruco if args.show else None,
    )
    aruco_worker.start()

    yolo_worker: Optional[DetectorWorker[YoloDetection]] = None
    if yolo_detector is not None:
        yolo_worker = DetectorWorker(
            "yolo",
            yolo_detector.detect,
            rate_hz=args.yolo_rate,
            on_result=cache.publish_yolo if args.show else None,
        )
        yolo_worker.start()

    # Stage 7: Optionally start a display thread that waits for synchronized detector outputs.
    print("Running ArUco and YOLO in separate detector threads on PC")
    if args.show:
        print("Running display in separate thread")

    display_worker = None
    if args.show:
        display_worker = DisplayWorker(cache)
        display_worker.start()

    last_log = 0.0
    last_frame_time = time.time()
    last_no_frame_log = 0.0
    frame_id = 0
    try:
        # Stage 8: Main ingest loop. Pull frames, hand them to detector workers, and monitor health.
        while True:
            frame = provider.get_frame_with_timeout(timeout=args.frame_timeout)
            if frame is None:
                now = time.time()
                gap = now - last_frame_time
                # Stage 8a: Handle stalls in the upstream stream and stop after a prolonged outage.
                if now - last_no_frame_log >= 2.0:
                    print(
                        "Waiting for Pi stream frame... "
                        f"no-frame-for={gap:.1f}s "
                        f"(timeout={args.frame_timeout:.1f}s)"
                    )
                    last_no_frame_log = now
                if gap >= args.max_no_frame_seconds:
                    print(
                        "No frame received from Pi stream for too long; "
                        f"stopping after {gap:.1f}s"
                    )
                    break
                continue

            last_frame_time = time.time()

            # Stage 8b: Fan out the newest frame to each detector thread using the shared frame ID.
            aruco_worker.submit(frame_id, frame)
            if yolo_worker is not None:
                yolo_worker.submit(frame_id, frame)
            frame_id += 1

            # Stage 8c: Read the latest completed detector outputs for logging and quick visibility.
            aruco_dets = aruco_worker.latest()
            yolo_dets = yolo_worker.latest() if yolo_worker is not None else []

            now = time.time()
            if now - last_log >= 1.0:
                # Stage 8d: Emit a lightweight heartbeat so we can see detections and worker failures.
                print(f"aruco={len(aruco_dets)} yolo={len(yolo_dets)}")
                if aruco_dets:
                    ids = ",".join(str(det.tag_id) for det in aruco_dets)
                    print(f"aruco_ids={ids}")
                    for det in aruco_dets:
                        if det.tvec_m is None:
                            continue
                        tx, ty, tz = det.tvec_m
                        print(
                            f"aruco_pose id={det.tag_id} "
                            f"dist={det.distance_m:.3f}m "
                            f"x={tx:.3f} y={ty:.3f} z={tz:.3f}"
                        )
                if aruco_worker.last_error() is not None:
                    print(aruco_worker.last_error())
                if yolo_worker is not None and yolo_worker.last_error() is not None:
                    print(yolo_worker.last_error())
                last_log = now
    except KeyboardInterrupt:
        print("\nStopping pipeline")
    finally:
        # Stage 9: Tear everything down in dependency order so threads exit cleanly.
        provider.close()
        aruco_worker.stop()
        if yolo_worker is not None:
            yolo_worker.stop()
        if display_worker is not None:
            display_worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
