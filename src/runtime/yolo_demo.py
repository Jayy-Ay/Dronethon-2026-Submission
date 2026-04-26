"""YOLO-only demo runtime for live preview from Pi RTSP or UDP stream."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.stages.yolo_detector import YoloDetection, YoloDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the standalone YOLO preview demo."""
    parser = argparse.ArgumentParser(description="Run YOLO-only object detection demo")
    parser.add_argument("--rtsp-url", default="rtsp://dronetastic.local:8554/cam1", help="Optional RTSP URL")
    parser.add_argument("--rtsp-width", type=int, default=1280, help="RTSP decode width")
    parser.add_argument("--rtsp-height", type=int, default=720, help="RTSP decode height")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="UDP bind IP when --rtsp-url is empty")
    parser.add_argument("--video-port", type=int, default=5600, help="UDP bind port when --rtsp-url is empty")
    parser.add_argument("--yolo-model", default="yolov8s.onnx", help="Path to YOLO ONNX model")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class labels file")
    parser.add_argument("--yolo-input", type=int, default=416, help="YOLO square input size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold")
    parser.add_argument("--show", action="store_true", help="Show the annotated preview window")
    parser.add_argument("--frame-timeout", type=float, default=3.0, help="Seconds to wait per frame")
    parser.add_argument("--max-no-frame-seconds", type=float, default=60.0, help="Exit after this no-frame gap")
    return parser.parse_args()


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Generate a stable pseudo-random color for a class id."""
    return (
        (37 * class_id + 80) % 255,
        (17 * class_id + 160) % 255,
        (29 * class_id + 220) % 255,
    )


def draw_yolo_overlays(frame: np.ndarray, detections: list[YoloDetection]) -> np.ndarray:
    """Render YOLO boxes, labels, and confidences on a frame."""
    vis = frame.copy()
    for det in detections:
        color = _color_for_class(det.class_id)
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        label_y = max(24, det.y1 - 10)
        cv2.putText(
            vis,
            label,
            (det.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return vis


def main() -> None:
    """Run the YOLO-only demo loop against RTSP or UDP video input."""
    args = parse_args()

    provider: RtspFrameProvider | StreamFrameProvider
    if args.rtsp_url:
        provider = RtspFrameProvider(args.rtsp_url, width=args.rtsp_width, height=args.rtsp_height)
        print(f"Receiving RTSP frames from {args.rtsp_url}")
    else:
        provider = StreamFrameProvider(ip=args.bind_ip, port=args.video_port)
        print(f"Listening for UDP video stream on {args.bind_ip}:{args.video_port}")

    detector = YoloDetector(
        model_path=args.yolo_model,
        classes_path=args.yolo_classes,
        input_size=args.yolo_input,
        conf_thresh=args.yolo_conf,
        nms_thresh=args.yolo_nms,
    )

    print(f"YOLO demo running with model={args.yolo_model} device={detector.device}")

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
                        "Waiting for frame... "
                        f"no-frame-for={gap:.1f}s "
                        f"(timeout={args.frame_timeout:.1f}s)"
                    )
                    last_no_frame_log = now
                if gap >= args.max_no_frame_seconds:
                    print(f"No frame for {gap:.1f}s, exiting")
                    break
                continue

            last_frame_time = time.time()
            detections = detector.detect(frame)
            vis = draw_yolo_overlays(frame, detections)

            if args.show:
                cv2.imshow("YOLO Only Demo", vis)

            now = time.time()
            if now - last_log >= 1.0:
                if detections:
                    summary = ", ".join(f"{det.label}:{det.confidence:.2f}" for det in detections[:8])
                    print(f"yolo={len(detections)} {summary}")
                else:
                    print("yolo=0")
                last_log = now

            if args.show and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        print("\nStopping YOLO demo")
    finally:
        provider.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
