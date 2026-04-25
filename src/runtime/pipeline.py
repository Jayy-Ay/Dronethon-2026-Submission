"""PC runtime pipeline: receive Pi frames and run ArUco + YOLO in separate threads."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, List, Optional, TypeVar

import cv2
import numpy as np

from src.stages.aruco_detector import ArucoDetection, ArucoDetector
from src.stages.yolo_detector import YoloDetection, YoloDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


T = TypeVar("T")


class DetectorWorker(Generic[T]):
    """Runs a detector continuously in its own thread on latest submitted frames."""

    def __init__(self, name: str, detect_fn: Callable[[np.ndarray], List[T]], rate_hz: float) -> None:
        self._name = name
        self._detect_fn = detect_fn
        self._rate_hz = max(rate_hz, 0.1)

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"detector-{name}")

        self._latest: List[T] = []
        self._last_error: Optional[str] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)

    def latest(self) -> List[T]:
        with self._lock:
            return list(self._latest)

    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        interval = 1.0 / self._rate_hz
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            start = time.time()
            try:
                result = self._detect_fn(frame)
                with self._lock:
                    self._latest = result
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{self._name}: {exc}"

            elapsed = time.time() - start
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)


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
    show: bool
    aruco_rate: float
    yolo_rate: float
    yolo_model: str
    yolo_classes: str
    yolo_input: int
    yolo_conf: float
    yolo_nms: float


def parse_args() -> Args:
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
    parser.add_argument("--show", action="store_true", help="Show annotated local preview")

    parser.add_argument("--aruco-rate", type=float, default=12.0, help="ArUco detection rate (Hz)")
    parser.add_argument("--yolo-rate", type=float, default=8.0, help="YOLO detection rate (Hz)")

    parser.add_argument("--yolo-model", default="yolov8s.onnx", help="Path to YOLO ONNX model on PC")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class names file on PC")
    parser.add_argument("--yolo-input", type=int, default=640, help="YOLO square input size")
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
    vis = frame.copy()

    for det in aruco_dets:
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        x = int(det.center_x)
        y = int(det.center_y)
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"ARUCO {det.tag_id} area={det.area_px:.0f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
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
    args = parse_args()

    if args.rtsp_url:
        provider = RtspFrameProvider(args.rtsp_url, width=args.rtsp_width, height=args.rtsp_height)
        print(f"Receiving RTSP frames from {args.rtsp_url}")
    else:
        provider = StreamFrameProvider(ip=args.bind_ip, port=args.video_port)
        print(f"Listening for UDP video stream on {args.bind_ip}:{args.video_port}")

    aruco_detector = ArucoDetector(family=args.family)

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
    else:
        print(
            "YOLO disabled: missing model/classes file. "
            f"model={yolo_model_path} classes={yolo_classes_path}"
        )

    aruco_worker = DetectorWorker("aruco", aruco_detector.detect, rate_hz=args.aruco_rate)
    aruco_worker.start()

    yolo_worker: Optional[DetectorWorker[YoloDetection]] = None
    if yolo_detector is not None:
        yolo_worker = DetectorWorker("yolo", yolo_detector.detect, rate_hz=args.yolo_rate)
        yolo_worker.start()

    print("Running ArUco and YOLO in separate detector threads on PC")

    last_log = 0.0
    last_frame_time = time.time()
    last_no_frame_log = 0.0
    try:
        while True:
            frame = provider.get_frame_with_timeout(timeout=args.frame_timeout)
            if frame is None:
                now = time.time()
                gap = now - last_frame_time
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

            aruco_worker.submit(frame)
            if yolo_worker is not None:
                yolo_worker.submit(frame)

            aruco_dets = aruco_worker.latest()
            yolo_dets = yolo_worker.latest() if yolo_worker is not None else []

            now = time.time()
            if now - last_log >= 1.0:
                print(f"aruco={len(aruco_dets)} yolo={len(yolo_dets)}")
                if aruco_dets:
                    ids = ",".join(str(det.tag_id) for det in aruco_dets)
                    print(f"aruco_ids={ids}")
                if aruco_worker.last_error() is not None:
                    print(aruco_worker.last_error())
                if yolo_worker is not None and yolo_worker.last_error() is not None:
                    print(yolo_worker.last_error())
                last_log = now

            if args.show:
                vis = draw_overlays(frame, aruco_dets, yolo_dets)
                cv2.imshow("PC AI Pipeline", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\nStopping pipeline")
    finally:
        provider.close()
        aruco_worker.stop()
        if yolo_worker is not None:
            yolo_worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
