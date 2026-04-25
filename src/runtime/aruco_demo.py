"""ArUco-only demo runtime for live preview from Pi RTSP or UDP stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.stages.aruco_detector import ArucoDetection, ArucoDetector
from src.vision.frame_provider import RtspFrameProvider, StreamFrameProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ArUco-only detection demo")
    parser.add_argument("--rtsp-url", default="rtsp://dronetastic.local:8554/cam1", help="Optional RTSP URL")
    parser.add_argument("--rtsp-width", type=int, default=1280, help="RTSP decode width")
    parser.add_argument("--rtsp-height", type=int, default=720, help="RTSP decode height")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="UDP bind IP when --rtsp-url is empty")
    parser.add_argument("--video-port", type=int, default=5600, help="UDP bind port when --rtsp-url is empty")
    parser.add_argument("--family", default="6x6_250", help="ArUco/AprilTag family")
    parser.add_argument("--frame-timeout", type=float, default=3.0, help="Seconds to wait per frame")
    parser.add_argument("--max-no-frame-seconds", type=float, default=60.0, help="Exit after this no-frame gap")
    return parser.parse_args()


def draw_aruco_overlays(frame: np.ndarray, detections: list[ArucoDetection]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
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
    return vis


def main() -> None:
    args = parse_args()

    provider: RtspFrameProvider | StreamFrameProvider
    if args.rtsp_url:
        provider = RtspFrameProvider(args.rtsp_url, width=args.rtsp_width, height=args.rtsp_height)
        print(f"Receiving RTSP frames from {args.rtsp_url}")
    else:
        provider = StreamFrameProvider(ip=args.bind_ip, port=args.video_port)
        print(f"Listening for UDP video stream on {args.bind_ip}:{args.video_port}")

    detector = ArucoDetector(family=args.family)

    print(f"ArUco demo running with family={args.family}")

    frame_index = 0
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

            vis = draw_aruco_overlays(frame, detections)
            cv2.imshow("ArUco Only Demo", vis)

            now = time.time()
            if now - last_log >= 1.0:
                if detections:
                    ids = ",".join(str(det.tag_id) for det in detections)
                    print(f"aruco={len(detections)} ids={ids}")
                else:
                    print("aruco=0")
                last_log = now

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            frame_index += 1
    except KeyboardInterrupt:
        print("\nStopping ArUco demo")
    finally:
        provider.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
