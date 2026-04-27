"""YOLO segmentation demo runtime for live preview from Pi RTSP or UDP stream."""

from __future__ import annotations

import argparse
import time

import cv2

from src.runtime.yolo_demo import draw_yolo_overlays
from src.stages.yolo_detector import YoloDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the standalone YOLO segmentation preview demo."""
    parser = argparse.ArgumentParser(description="Run YOLO segmentation demo")
    parser.add_argument("--rtsp-url", default="rtsp://dronetastic.local:8554/cam1", help="Optional RTSP URL")
    parser.add_argument("--rtsp-width", type=int, default=1280, help="RTSP decode width")
    parser.add_argument("--rtsp-height", type=int, default=720, help="RTSP decode height")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="UDP bind IP when --rtsp-url is empty")
    parser.add_argument("--video-port", type=int, default=5600, help="UDP bind port when --rtsp-url is empty")
    parser.add_argument("--yolo-model", default="yolov8n-seg.pt", help="Path to YOLO segmentation model")
    parser.add_argument("--yolo-classes", default="coco.names", help="Path to class labels file")
    parser.add_argument("--yolo-input", type=int, default=640, help="YOLO square input size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold")
    parser.add_argument("--show", action="store_true", help="Show the annotated preview window")
    parser.add_argument("--frame-timeout", type=float, default=3.0, help="Seconds to wait per frame")
    parser.add_argument("--max-no-frame-seconds", type=float, default=60.0, help="Exit after this no-frame gap")
    return parser.parse_args()


def main() -> None:
    """Run the YOLO segmentation demo loop."""
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

    print(
        f"YOLO segmentation demo running with model={args.yolo_model} "
        f"backend={detector.backend} device={detector.device}"
    )

    last_log = 0.0
    last_frame_time = time.time()
    last_no_frame_log = 0.0
    warned_missing_masks = False

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

            has_masks = any(det.mask is not None for det in detections)
            if detections and not has_masks and not warned_missing_masks:
                print(
                    "Detections are coming through, but no segmentation masks were returned. "
                    "Use a segmentation-capable model such as yolov8n-seg.pt."
                )
                warned_missing_masks = True

            if args.show:
                cv2.imshow("YOLO Segmentation Demo", vis)

            now = time.time()
            if now - last_log >= 1.0:
                if detections:
                    mask_count = sum(det.mask is not None for det in detections)
                    summary = ", ".join(f"{det.label}:{det.confidence:.2f}" for det in detections[:8])
                    print(f"segments={mask_count}/{len(detections)} {summary}")
                else:
                    print("segments=0")
                last_log = now

            if args.show and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        print("\nStopping YOLO segmentation demo")
    finally:
        provider.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
